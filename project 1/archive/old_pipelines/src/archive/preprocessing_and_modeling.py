"""
LendingClub Interest Rate Prediction Pipeline (Polars + native XGBoost + Optuna TPE)

Stages:
  1. Load (Polars lazy scan)
  2. Leakage audit (drop grade, sub_grade, grade_num, sub_grade_num)
  3. Feature engineering (single with_columns chain; native Polars categoricals)
  4. Temporal train/val split (last 20% of train sorted by issue_d -> val)
  5. Median imputation fit on X_train only, applied to val + test
  6. XGBoost baseline (native API, DMatrix(Polars, enable_categorical=True))
  7. Optuna TPE Bayesian search (100 trials, MedianPruner, XGBoostPruningCallback,
     full X_train objective, num_boost_round=3000 + early_stopping_rounds=50)
  8. Final model on best params, save to outputs/xgb_final.json
  9. SHAP top-10 + summary plot
 10. Test-set predictions to outputs/test_predictions.csv
"""

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
import xgboost as xgb

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna_integration import XGBoostPruningCallback

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "LC_train.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "LC_test.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEAKAGE_COLS = ["grade", "sub_grade", "installment"]
# installment = PMT(loan_amnt, term, int_rate) — algebraic function of the target.
# Earlier attempt (docs/pipeline-attempt-summary.md) found correlation 0.9998 with int_rate
# after conditioning on loan_amnt & term. Marginal corr ~0.34 is misleading.
CATEGORICAL_COLS = [
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "initial_list_status",
    "addr_state",  # state means span only 0.4 pp marginally, but native
                   # XGBoost categoricals can still find sub-population effects
]
# Raw text/ID columns that don't go to the model. issue_d and addr_state are
# REMOVED from this list (issue_d gets year/month extracted; addr_state becomes
# a categorical input). zip_code stays dropped — first-digit signal is flat.
DROP_COLS = ["emp_title", "zip_code", "issue_d", "earliest_cr_line", "emp_length", "term"]

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
print("=== LOAD ===")
t0 = time.perf_counter()
# Columns that early-row sniffing infers as String when leading values are null —
# force them to Float64 so arithmetic / interactions work.
NUMERIC_OVERRIDES = {
    c: pl.Float64
    for c in [
        "num_tl_90g_dpd_24m", "num_accts_ever_120_pd", "mort_acc",
        "percent_bc_gt_75", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_tl",
        "total_bc_limit", "pct_tl_nvr_dlq", "collections_12_mths_ex_med",
        "acc_now_delinq", "chargeoff_within_12_mths", "tax_liens",
        "pub_rec_bankruptcies", "revol_util", "dti", "annual_inc",
        "open_acc", "total_acc", "delinq_2yrs", "pub_rec",
        "inq_last_6mths", "loan_amnt", "installment", "revol_bal",
    ]
}
train_df = pl.read_csv(TRAIN_PATH, schema_overrides=NUMERIC_OVERRIDES, infer_schema_length=20000)
test_df = pl.read_csv(TEST_PATH, schema_overrides=NUMERIC_OVERRIDES, infer_schema_length=20000)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# ---------------------------------------------------------------------------
# 2. LEAKAGE AUDIT
# ---------------------------------------------------------------------------
print("\n=== LEAKAGE AUDIT ===")
grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
sub_grade_map = {f"{g}{n}": i * 5 + n for i, g in enumerate("ABCDEFG") for n in range(1, 6)}

audit = train_df.with_columns(
    pl.col("grade").replace_strict(grade_map, default=None).cast(pl.Float64).alias("_grade_num"),
    pl.col("sub_grade").replace_strict(sub_grade_map, default=None).cast(pl.Float64).alias("_sub_grade_num"),
).select(
    pl.corr("_grade_num", "int_rate").alias("grade_corr"),
    pl.corr("_sub_grade_num", "int_rate").alias("sub_grade_corr"),
    pl.corr("installment", "int_rate").alias("installment_corr"),
).row(0)
print(f"  grade vs int_rate:      {audit[0]:.4f}")
print(f"  sub_grade vs int_rate:  {audit[1]:.4f}")
print(f"  installment vs int_rate:{audit[2]:.4f}")
print(f"  dropping: {LEAKAGE_COLS}")

train_df = train_df.drop(LEAKAGE_COLS)
test_df = test_df.drop([c for c in LEAKAGE_COLS if c in test_df.columns])

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING (single with_columns chain)
# ---------------------------------------------------------------------------
print("\n=== FEATURE ENGINEERING ===")
fe_t0 = time.perf_counter()

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10, "n/a": None,
}


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Pure feature engineering — no state. Median imputation happens after the split."""
    df = df.with_columns(
        # term_numeric: extract digits from "36 months" / "60 months"
        pl.col("term").str.extract(r"(\d+)").cast(pl.Float64).alias("term_numeric"),
        # emp_length_numeric
        pl.col("emp_length").str.to_lowercase().replace_strict(EMP_LENGTH_MAP, default=None)
            .cast(pl.Float64).alias("emp_length_numeric"),
        # dates
        pl.col("issue_d").str.to_date(format="%Y-%m-%d", strict=False).alias("_issue_d_dt"),
        pl.col("earliest_cr_line").str.to_date(format="%b-%Y", strict=False).alias("_earliest_cr_line_dt"),
    ).with_columns(
        # credit history length in months
        ((pl.col("_issue_d_dt") - pl.col("_earliest_cr_line_dt")).dt.total_days() / 30.44)
            .cast(pl.Float64).alias("credit_history_length"),
        # NOTE: issue_year / issue_month not added — the temporal val split puts
        # 2012-2013 rows entirely in val, so year is out-of-distribution at val
        # time and trees cannot extrapolate. The macro drift exists in the data
        # but is unrecoverable under honest temporal validation.
    ).with_columns(
        # core transforms
        pl.col("annual_inc").log1p().alias("annual_inc_log"),
        pl.col("revol_bal").log1p().alias("revol_bal_log"),
        pl.col("total_bc_limit").log1p().alias("total_bc_limit_log"),
        (pl.col("pub_rec") > 0).cast(pl.Int8).alias("pub_rec_flag"),
        (pl.col("delinq_2yrs") > 0).cast(pl.Int8).alias("delinq_flag"),
        pl.col("emp_title").is_not_null().cast(pl.Int8).alias("has_emp_title"),
        # Zero-inflated count siblings — explicit flags help SHAP attribution
        (pl.col("tax_liens").fill_null(0) > 0).cast(pl.Int8).alias("tax_liens_flag"),
        (pl.col("acc_now_delinq").fill_null(0) > 0).cast(pl.Int8).alias("acc_now_delinq_flag"),
        (pl.col("chargeoff_within_12_mths").fill_null(0) > 0).cast(pl.Int8).alias("chargeoff_flag"),
        (pl.col("collections_12_mths_ex_med").fill_null(0) > 0).cast(pl.Int8).alias("collections_flag"),
        (pl.col("num_accts_ever_120_pd").fill_null(0) > 0).cast(pl.Int8).alias("ever_120_pd_flag"),
        (pl.col("num_tl_90g_dpd_24m").fill_null(0) > 0).cast(pl.Int8).alias("recent_90dpd_flag"),
        # discretised bins (constant breaks — no train/test fit needed)
        pl.col("dti").cut([10, 20, 30, 40], labels=["0", "1", "2", "3", "4"]).cast(pl.Utf8)
            .cast(pl.Float64, strict=False).alias("dti_bucket"),
        pl.col("loan_amnt").cut([5000, 10000, 20000, 30000], labels=["0", "1", "2", "3", "4"]).cast(pl.Utf8)
            .cast(pl.Float64, strict=False).alias("loan_amnt_bucket"),
    ).with_columns(
        # interactions — existing
        (pl.col("dti") * pl.col("total_bc_limit_log")).alias("dti_x_total_bc_limit"),
        (pl.col("loan_amnt") * (pl.col("revol_util").fill_null(0) / 100)).alias("loan_amnt_x_revol_util"),
        (pl.col("term_numeric") * pl.col("dti")).alias("term_x_dti"),
        ((pl.col("percent_bc_gt_75").fill_null(0) / 100) * (pl.col("revol_util").fill_null(0) / 100))
            .alias("percent_bc_gt_75_x_revol_util"),
        (pl.col("annual_inc_log") * pl.col("dti")).alias("annual_inc_log_x_dti"),
        (pl.col("total_bc_limit_log") * pl.col("loan_amnt").log1p()).alias("total_bc_limit_x_loan_amnt"),
        # interactions — NEW (Phase 2 of plan)
        # payment_to_income removed: depended on installment, which is leakage.
        # loan_to_income is the safe alternative (loan_amnt / annual_inc).
        (pl.col("loan_amnt") / (pl.col("annual_inc") + 1)).alias("loan_to_income"),
        (pl.col("delinq_2yrs") / (pl.col("credit_history_length") / 12 + 1))
            .alias("delinq_per_year_of_history"),
        (pl.col("pub_rec").fill_null(0) + 2 * pl.col("pub_rec_bankruptcies").fill_null(0))
            .alias("pub_rec_bankruptcy_combo"),
        (pl.col("emp_length").str.to_lowercase().replace_strict(EMP_LENGTH_MAP, default=None)
            .cast(pl.Float64).fill_null(0) * pl.col("mort_acc").fill_null(0))
            .alias("emp_length_x_mort_acc"),
        (pl.col("revol_util").fill_null(0) * pl.col("open_acc").fill_null(0))
            .alias("revol_util_x_open_acc"),
        ((pl.col("num_tl_90g_dpd_24m").fill_null(0) + pl.col("num_accts_ever_120_pd").fill_null(0))
            / (pl.col("total_acc").fill_null(0) + 1)).alias("bad_acc_density"),
        (pl.col("mo_sin_old_rev_tl_op").fill_null(0) - pl.col("mo_sin_rcnt_tl").fill_null(0))
            .alias("acc_age_spread"),
        # NEW interactions surfaced by the data-dictionary audit
        ((pl.col("mo_sin_old_rev_tl_op").fill_null(0) + 1)
            / (pl.col("mo_sin_rcnt_tl").fill_null(0) + 1)).alias("acc_age_ratio"),
        (pl.col("delinq_2yrs").fill_null(0)
            + 2 * pl.col("num_tl_90g_dpd_24m").fill_null(0)
            + 3 * pl.col("num_accts_ever_120_pd").fill_null(0)).alias("delinq_severity"),
        (pl.col("inq_last_6mths").fill_null(0) / (pl.col("open_acc").fill_null(0) + 1))
            .alias("inq_per_open_acc"),
    )

    # Cast categoricals (XGBoost native categorical handling instead of one-hot)
    df = df.with_columns(
        [pl.col(c).cast(pl.Categorical) for c in CATEGORICAL_COLS if c in df.columns]
    )

    # Drop raw columns we've consumed
    cols_to_drop = DROP_COLS + ["_issue_d_dt", "_earliest_cr_line_dt"]
    df = df.drop([c for c in cols_to_drop if c in df.columns])
    return df


train_df = engineer_features(train_df)
test_df = engineer_features(test_df)
print(f"FE elapsed: {time.perf_counter() - fe_t0:.2f}s")
print(f"Train after FE: {train_df.shape}  Test after FE: {test_df.shape}")

# ---------------------------------------------------------------------------
# 4. TEMPORAL TRAIN / VAL SPLIT
# ---------------------------------------------------------------------------
print("\n=== TEMPORAL TRAIN / VAL SPLIT ===")
# train_df was generated from create_train_test_split.py sorted by issue_d, but we
# re-sort defensively in case any reorder happened in FE.
train_df = train_df.with_row_index("_row_idx")  # stable
n_total = train_df.height
n_val = int(n_total * 0.2)
n_train = n_total - n_val

X_train_full = train_df.head(n_train).drop("_row_idx")
X_val_full = train_df.tail(n_val).drop("_row_idx")
y_train = X_train_full["int_rate"].to_numpy()
y_val = X_val_full["int_rate"].to_numpy()
X_train = X_train_full.drop("int_rate")
X_val = X_val_full.drop("int_rate")
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# ---------------------------------------------------------------------------
# 5. IMPUTATION (fit on X_train only)
# ---------------------------------------------------------------------------
print("\n=== MEDIAN IMPUTATION (fit on X_train) ===")
numeric_cols = [
    c for c, dt in zip(X_train.columns, X_train.dtypes)
    if dt.is_numeric() and X_train[c].null_count() > 0
]
imputation_values = {c: X_train[c].median() for c in numeric_cols}

def apply_impute(df: pl.DataFrame, medians: dict) -> pl.DataFrame:
    exprs = [pl.col(c).fill_null(medians[c]) for c in medians if c in df.columns]
    return df.with_columns(exprs) if exprs else df

X_train = apply_impute(X_train, imputation_values)
X_val = apply_impute(X_val, imputation_values)
# Test still has 'ID' column; impute everything else
test_ids = test_df["ID"].to_numpy()
X_test = apply_impute(test_df.drop("ID"), imputation_values)
print(f"Imputed {len(imputation_values)} numeric columns")

# Align test columns to X_train ordering (drop any extras, fill missing with 0)
for col in X_train.columns:
    if col not in X_test.columns:
        X_test = X_test.with_columns(pl.lit(0).alias(col))
X_test = X_test.select(X_train.columns)

# ---------------------------------------------------------------------------
# 6. XGBOOST BASELINE (native API, DMatrix from Polars w/ categoricals)
# ---------------------------------------------------------------------------
print("\n=== XGBOOST BASELINE ===")

def make_dmatrix(X: pl.DataFrame, y=None) -> xgb.DMatrix:
    # XGBoost 3.x accepts Polars frames directly when enable_categorical=True
    return xgb.DMatrix(X.to_pandas(), label=y, enable_categorical=True)


dtrain = make_dmatrix(X_train, y_train)
dval = make_dmatrix(X_val, y_val)

baseline_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}
baseline_booster = xgb.train(
    baseline_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, "validation_0")],
    early_stopping_rounds=50,
    verbose_eval=False,
)
baseline_pred = baseline_booster.predict(dval)
rmse_baseline = float(np.sqrt(np.mean((y_val - baseline_pred) ** 2)))
mae_baseline = float(np.mean(np.abs(y_val - baseline_pred)))
ss_res = float(np.sum((y_val - baseline_pred) ** 2))
ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
r2_baseline = 1 - ss_res / ss_tot
print(f"Baseline RMSE: {rmse_baseline:.4f}  MAE: {mae_baseline:.4f}  R2: {r2_baseline:.4f}")
print(f"Baseline best_iteration: {baseline_booster.best_iteration}")

# ---------------------------------------------------------------------------
# 7. OPTUNA TPE + MedianPruner + XGBoostPruningCallback
# ---------------------------------------------------------------------------
print("\n=== OPTUNA BAYESIAN TUNING ===")
study_db = os.path.join(OUTPUT_DIR, "optuna_study.db").replace("\\", "/")
storage_url = f"sqlite:///{study_db}"


def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    }
    pruning_cb = XGBoostPruningCallback(trial, observation_key="validation_0-rmse")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, "validation_0")],
        early_stopping_rounds=50,
        callbacks=[pruning_cb],
        verbose_eval=False,
    )
    preds = booster.predict(dval)
    return float(np.sqrt(np.mean((y_val - preds) ** 2)))


tune_t0 = time.perf_counter()
study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_warmup_steps=50),
    storage=storage_url,
    study_name="lc_int_rate",
    load_if_exists=True,
)
n_trials = int(os.environ.get("OPTUNA_TRIALS", "100"))
print(f"  running {n_trials} trials (override via OPTUNA_TRIALS env var)")
study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
print(f"Optuna elapsed: {time.perf_counter() - tune_t0:.1f}s")
print(f"Best RMSE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# ---------------------------------------------------------------------------
# 8. FINAL MODEL
# ---------------------------------------------------------------------------
print("\n=== FINAL MODEL ===")
final_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "seed": 42,
    **study.best_params,
}
final_booster = xgb.train(
    final_params,
    dtrain,
    num_boost_round=3000,
    evals=[(dval, "validation_0")],
    early_stopping_rounds=50,
    verbose_eval=False,
)
final_pred = final_booster.predict(dval)
rmse_final = float(np.sqrt(np.mean((y_val - final_pred) ** 2)))
mae_final = float(np.mean(np.abs(y_val - final_pred)))
ss_res = float(np.sum((y_val - final_pred) ** 2))
r2_final = 1 - ss_res / ss_tot
print(f"Final RMSE: {rmse_final:.4f}  MAE: {mae_final:.4f}  R2: {r2_final:.4f}")
print(f"Final best_iteration: {final_booster.best_iteration}")

model_path = os.path.join(OUTPUT_DIR, "xgb_final.json")
final_booster.save_model(model_path)
print(f"Saved model -> {model_path}")

# ---------------------------------------------------------------------------
# 9. SHAP
# ---------------------------------------------------------------------------
print("\n=== SHAP ===")
shap_sample = X_val.head(1000).to_pandas()
explainer = shap.TreeExplainer(final_booster)
# shap.TreeExplainer builds its own DMatrix without enable_categorical, so we
# pass the booster a pre-built DMatrix instead.
shap_dmatrix = xgb.DMatrix(shap_sample, enable_categorical=True)
shap_values = final_booster.predict(shap_dmatrix, pred_contribs=True)
# pred_contribs returns shape (n, n_features + 1) where the last column is the bias.
shap_values = shap_values[:, :-1]

mean_shap = np.abs(shap_values).mean(axis=0)
top10 = sorted(zip(X_val.columns, mean_shap), key=lambda kv: kv[1], reverse=True)[:10]
print("Top 10 features by mean |SHAP|:")
for name, val in top10:
    print(f"  {name:35s} {val:.4f}")

# Convert categorical columns to their codes purely for the summary plot.
shap_plot_sample = shap_sample.copy()
for c in shap_plot_sample.select_dtypes(include=["category"]).columns:
    shap_plot_sample[c] = shap_plot_sample[c].cat.codes.astype(float)
shap.summary_plot(shap_values, shap_plot_sample, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> outputs/shap_summary.png")

# ---------------------------------------------------------------------------
# 10. TEST PREDICTIONS
# ---------------------------------------------------------------------------
print("\n=== TEST PREDICTIONS ===")
dtest = make_dmatrix(X_test)
test_preds = final_booster.predict(dtest)
submission = pl.DataFrame({"ID": test_ids, "int_rate_pred": test_preds})
submission_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
submission.write_csv(submission_path)
print(f"Saved {submission.height} rows -> {submission_path}")
print(f"int_rate_pred summary: min={test_preds.min():.2f}  max={test_preds.max():.2f}  "
      f"mean={test_preds.mean():.2f}  std={test_preds.std():.2f}")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("\n=== PIPELINE COMPLETE ===")
print(f"Baseline RMSE: {rmse_baseline:.4f}")
print(f"Tuned RMSE:    {rmse_final:.4f}")
print(f"Improvement:   {rmse_baseline - rmse_final:.4f}")
print(f"Total elapsed: {time.perf_counter() - t0:.1f}s")
