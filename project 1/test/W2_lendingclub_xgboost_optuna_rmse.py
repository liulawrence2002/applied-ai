"""
LendingClub Interest Rate Prediction: XGBoost + Optuna (RMSE)
=============================================================

Dataset: true data/LC_train.csv (100k x 39) and true data/LC_test.csv (10k x 39, no target).
Target:  int_rate (annual interest rate, %).
Feature definitions: true data/LCDataDictionary.xlsx.

Workflow mirrors the W1 demo: leak-free Pipeline, baseline, then hyperparameter
tuning with cross-validation. Grid/Random Search are replaced by Optuna's TPE
sampler for principled Bayesian search. The held-out validation set is used only
at the end. The model is XGBoost, and the optimization objective is RMSE
(lower is better).

Run from the project root or from the test/ folder:
    python "test/W2_lendingclub_xgboost_optuna_rmse.py"
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing + model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Cross-validation and tuning
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Bayesian hyperparameter search
import optuna
from optuna.samplers import TPESampler

# XGBoost
from xgboost import XGBRegressor


# =============================================================================
# Step 0: Setup
# =============================================================================
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Fix the seed so every run produces the same split, CV folds, and XGBoost sampling
RANDOM_STATE = 6604

# Optuna budget. Higher = better tuning, slower runtime.
N_TRIALS = 40


# =============================================================================
# Step 1: Load data
# =============================================================================
# Canonical files are in true data/. Eight numeric columns arrive as strings
# with literal "NA"; cast at load. loan_status is post-origination leakage and
# is dropped immediately. int_rate is the target in train; test contains an
# ID column for submission alignment.

# Resolve true data/ regardless of whether the script is launched from the
# project root or from this test/ folder.
CWD = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = CWD if (CWD / "true data").exists() else CWD.parent
DATA_DIR = PROJECT_ROOT / "true data"
assert DATA_DIR.exists(), f"Could not locate true data/ from {CWD}"

# Columns that arrive as strings with literal "NA" (per EDA). Force numeric.
STRING_NUMERIC_COLS = [
    "dti", "revol_util", "all_util",
    "mo_sin_old_il_acct",
    "mths_since_last_record", "mths_since_rcnt_il",
    "mths_since_recent_bc", "mths_since_recent_inq",
    "tot_cur_bal",
]

dtype_overrides = {c: "float64" for c in STRING_NUMERIC_COLS}

print("=" * 70)
print("Step 1: Load data")
print("=" * 70)

train_raw = pd.read_csv(DATA_DIR / "LC_train.csv", na_values=["NA"], dtype=dtype_overrides)
test_raw  = pd.read_csv(DATA_DIR / "LC_test.csv",  na_values=["NA"], dtype=dtype_overrides)

# Drop post-origination leakage unconditionally
LEAKAGE_COLS = ["loan_status"]
train_raw = train_raw.drop(columns=LEAKAGE_COLS)
test_raw  = test_raw.drop(columns=LEAKAGE_COLS)

print(f"Train shape: {train_raw.shape}   Test shape: {test_raw.shape}")
print(train_raw.head())


# =============================================================================
# Step 2: Target distribution
# =============================================================================
# int_rate is moderately right-skewed (skew ~0.85). Tree models like XGBoost
# don't require a target transform - they split on quantiles natively - so we
# model int_rate on the raw percentage scale and report RMSE in percentage
# points.

TARGET = "int_rate"

print()
print("=" * 70)
print("Step 2: Target distribution")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].hist(train_raw[TARGET], bins=40, color="steelblue", edgecolor="white")
axes[0].set_xlabel("int_rate (%)")
axes[0].set_ylabel("Loans")
axes[0].set_title("Raw scale")

axes[1].hist(np.log1p(train_raw[TARGET]), bins=40, color="steelblue", edgecolor="white")
axes[1].set_xlabel("log1p(int_rate)")
axes[1].set_title("log1p (diagnostic only)")
plt.tight_layout()

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUTS_DIR / "w2_target_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

print(
    f"min={train_raw[TARGET].min():.2f}  max={train_raw[TARGET].max():.2f}  "
    f"mean={train_raw[TARGET].mean():.2f}  median={train_raw[TARGET].median():.2f}  "
    f"std={train_raw[TARGET].std():.2f}  skew={train_raw[TARGET].skew():.3f}"
)
print(f"Saved figure -> {OUTPUTS_DIR / 'w2_target_distribution.png'}")


# =============================================================================
# Step 3: Feature engineering and train/validation split
# =============================================================================
# Per the EDA report (sec 13): drop free-text and duplicate columns, parse
# `term` to integer months, map `emp_length` ordinally, collapse the two FICO
# columns to their midpoint, and add binary flags for the semantically-missing
# `mths_since_*` fields. There is no `issue_d` in this slice, so validation is
# a random 80/20 split.

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}

MTHS_SINCE_COLS = [
    "mths_since_last_record", "mths_since_recent_inq",
    "mths_since_rcnt_il", "mths_since_recent_bc",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering. Anything that requires fitting on training
    data (imputation, encoding) goes inside the sklearn Pipeline so each CV
    fold sees only its own training portion."""
    out = df.copy()

    # FICO midpoint (the two columns differ by 4 or 5 in every row)
    out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2

    # term -> integer months (" 36 months" / " 60 months")
    out["term_months"] = (
        out["term"].astype("string").str.extract(r"(\d+)", expand=False).astype("Int64")
    )

    # emp_length ordinal encoding (NA preserved as NaN for the imputer)
    out["emp_length_num"] = out["emp_length"].map(EMP_LENGTH_MAP)

    # "Has event" flags for informatively-missing mths_since_* columns
    for c in MTHS_SINCE_COLS:
        out[f"has_{c}"] = out[c].notna().astype(int)

    # loan-to-income (underwriting signal recommended by EDA)
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"].replace(0, np.nan)

    # Drop original columns now superseded or unusable
    drop_cols = [
        "fico_range_low", "fico_range_high",   # collapsed into fico
        "term",                                  # parsed into term_months
        "emp_length",                            # encoded as emp_length_num
        "emp_title",                             # 35k unique strings, weak signal
        "title",                                 # one-to-one duplicate of purpose
        "zip_code",                              # masked, low signal vs addr_state
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    return out


print()
print("=" * 70)
print("Step 3: Feature engineering and train/validation split")
print("=" * 70)

train_fe = engineer(train_raw)
test_fe  = engineer(test_raw)

# Keep the test ID column aside for the final submission CSV
test_ids = test_fe["ID"].copy() if "ID" in test_fe.columns else None
if "ID" in test_fe.columns:
    test_fe = test_fe.drop(columns=["ID"])

X = train_fe.drop(columns=[TARGET])
y = train_fe[TARGET]

# 80/20 random split (no temporal column available - see EDA sec 1).
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

print(f"X_train: {X_train.shape}   X_val: {X_val.shape}   test_fe: {test_fe.shape}")


# =============================================================================
# Step 4: Preprocessing pipeline
# =============================================================================
# Numeric branch: median imputation (XGBoost handles NaN natively, but median
# imputation keeps the pipeline general and stable across CV folds).
# Categorical branch: most-frequent imputation, then one-hot encoding with
# handle_unknown="ignore". Everything is wrapped in Pipeline + ColumnTransformer
# so the imputer/encoder are fit only on each CV fold's training portion.

categorical_features = [
    "addr_state",
    "application_type",
    "home_ownership",
    "purpose",
    "verification_status",
]

# Everything not categorical and not the target is numeric (engineered + raw)
numeric_features = [c for c in X_train.columns if c not in categorical_features]

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ]
)

print()
print("=" * 70)
print("Step 4: Preprocessing pipeline")
print("=" * 70)
print(f"{len(numeric_features)} numeric, {len(categorical_features)} categorical")


# =============================================================================
# Step 5: Baseline model
# =============================================================================
# XGBoost with default-ish hyperparameters. Report RMSE, MAE, and R^2 on both
# the training and validation portions. The train vs validation gap on each
# metric is the overfit signal that tuning needs to improve on.

def report(name, model, X_tr, y_tr, X_va, y_va):
    """All metrics on the raw int_rate scale (percentage points).
    Train + validation reported so the gap reads as an overfit signal."""
    pred_tr = model.predict(X_tr)
    pred_va = model.predict(X_va)
    return {
        "model":      name,
        "train_R2":   r2_score(y_tr, pred_tr),
        "val_R2":     r2_score(y_va, pred_va),
        "train_RMSE": np.sqrt(mean_squared_error(y_tr, pred_tr)),
        "val_RMSE":   np.sqrt(mean_squared_error(y_va, pred_va)),
        "train_MAE":  mean_absolute_error(y_tr, pred_tr),
        "val_MAE":    mean_absolute_error(y_va, pred_va),
    }


print()
print("=" * 70)
print("Step 5: Baseline model")
print("=" * 70)

xgb_baseline = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

xgb_baseline.fit(X_train, y_train)

defaults = xgb_baseline.named_steps["model"].get_params()
print("XGBoost baseline params:")
for k in ["n_estimators", "learning_rate", "max_depth", "subsample",
         "colsample_bytree", "reg_lambda", "reg_alpha", "min_child_weight"]:
    print(f"  {k}: {defaults.get(k)}")

baseline_row = report("Baseline XGB", xgb_baseline, X_train, y_train, X_val, y_val)
print()
print(pd.DataFrame([baseline_row]).to_string(index=False))


# =============================================================================
# IMPORTANT
# =============================================================================
# The W1 notebook used GridSearchCV and RandomizedSearchCV with a `scoring`
# argument that picked the metric to optimize. Here we use Optuna with the
# TPE sampler, which builds a probabilistic model of the loss surface and
# chooses each next configuration where the expected improvement is highest.
#
# The objective function below returns the mean RMSE across cv folds
# (lower is better). Optuna's direction="minimize" then drives the search
# toward smaller validation RMSE.


# =============================================================================
# Step 6: Hyperparameter tuning with Optuna (RMSE)
# =============================================================================
# The XGBoost hyperparameters we tune:
#   n_estimators      - number of boosting trees
#   learning_rate     - contribution of each tree (lower = more trees needed)
#   max_depth         - tree depth (deeper = more interactions, more overfit)
#   min_child_weight  - minimum sum of instance weight per leaf (higher = reg)
#   subsample         - fraction of rows used per tree (lower = more variance)
#   colsample_bytree  - fraction of features used per tree (lower = decorrelates)
#   reg_lambda        - L2 regularization on leaf weights
#   reg_alpha         - L1 regularization (sparsifies leaf weights)
#   gamma             - minimum loss reduction required to split

print()
print("=" * 70)
print(f"Step 6: Optuna tuning ({N_TRIALS} trials, minimize 5-fold CV RMSE)")
print("=" * 70)

# Same shuffled K-fold across the baseline gap analysis and the Optuna trials.
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def objective(trial: optuna.Trial) -> float:
    """Mean 5-fold CV RMSE for one XGBoost configuration. Optuna minimizes this."""
    params = {
        "model__n_estimators":     trial.suggest_int("n_estimators", 200, 1200),
        "model__learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "model__max_depth":        trial.suggest_int("max_depth", 3, 10),
        "model__min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "model__subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "model__reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "model__reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "model__gamma":            trial.suggest_float("gamma", 1e-3, 5.0, log=True),
    }

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    pipe.set_params(**params)

    # cross_val_score with scoring="neg_root_mean_squared_error" returns
    # negative RMSE per fold; flip the sign so we minimize positive RMSE.
    neg_rmse = cross_val_score(
        pipe, X_train, y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return -neg_rmse.mean()


# TPE = Tree-structured Parzen Estimator. The sampler models p(params | low loss)
# vs p(params | high loss) and suggests configurations where the ratio is largest.
sampler = TPESampler(seed=RANDOM_STATE)
study = optuna.create_study(direction="minimize", sampler=sampler, study_name="xgb_int_rate_rmse")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest CV RMSE: {study.best_value:.4f}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")


# =============================================================================
# Step 6b: Optuna diagnostics
# =============================================================================
# Optimization history shows how CV RMSE fell across trials; parameter
# importances rank which hyperparameters Optuna found most consequential for
# the loss.

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

values = [t.value for t in study.trials if t.value is not None]
best_so_far = np.minimum.accumulate(values)
axes[0].plot(values, "o", color="steelblue", alpha=0.5, label="trial")
axes[0].plot(best_so_far, "-", color="crimson", label="best so far")
axes[0].set_xlabel("trial")
axes[0].set_ylabel("CV RMSE")
axes[0].set_title("Optimization history")
axes[0].legend()

try:
    importances = optuna.importance.get_param_importances(study)
    names = list(importances.keys())[::-1]
    vals  = [importances[n] for n in names]
    axes[1].barh(names, vals, color="steelblue")
    axes[1].set_xlabel("importance")
    axes[1].set_title("Hyperparameter importance")
except Exception as exc:  # importance needs >=2 completed trials with variance
    axes[1].text(0.5, 0.5, f"importance unavailable\n({exc})", ha="center", va="center")
    axes[1].set_axis_off()

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "w2_optuna_diagnostics.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved figure -> {OUTPUTS_DIR / 'w2_optuna_diagnostics.png'}")


# =============================================================================
# Step 7: Refit best configuration on the full training fold
# =============================================================================
print()
print("=" * 70)
print("Step 7: Refit best configuration")
print("=" * 70)

best_params = {f"model__{k}": v for k, v in study.best_params.items()}

xgb_tuned = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])
xgb_tuned.set_params(**best_params)
xgb_tuned.fit(X_train, y_train)
print("Tuned model fit on X_train.")


# =============================================================================
# Step 8: Final validation-set comparison
# =============================================================================
# Baseline vs Optuna-tuned XGBoost on the held-out 20% validation slice. The
# train vs val gap on each metric reads the overfit signal: large gap = high
# variance, both equally poor = high bias, train and val close = healthier
# generalization.

print()
print("=" * 70)
print("Step 8: Final validation-set comparison")
print("=" * 70)

results = pd.DataFrame([
    report("Baseline XGB",     xgb_baseline, X_train, y_train, X_val, y_val),
    report("Optuna-tuned XGB", xgb_tuned,    X_train, y_train, X_val, y_val),
])

# Pretty-print with explicit formatting (parallel to the .style.format in W1)
display_df = results.copy()
for col in ["train_R2", "val_R2"]:
    display_df[col] = display_df[col].map(lambda v: f"{v:.4f}")
for col in ["train_RMSE", "val_RMSE", "train_MAE", "val_MAE"]:
    display_df[col] = display_df[col].map(lambda v: f"{v:.4f} pp")
print(display_df.to_string(index=False))

# Persist raw numeric results too
results.to_csv(OUTPUTS_DIR / "w2_results.csv", index=False)
print(f"Saved results -> {OUTPUTS_DIR / 'w2_results.csv'}")


# =============================================================================
# Step 9: Predict on the held-out test set
# =============================================================================
# true data/LC_test.csv has no int_rate; we generate predictions, attach them
# to the ID column, and save to disk for submission. Sanity check: distribution
# of predicted rates should look broadly similar to the training distribution
# (with the caveat from the EDA that the test slice is slightly higher-FICO /
# lower-loan-amount, which should push mean predicted rate down a touch).

print()
print("=" * 70)
print("Step 9: Predict on the held-out test set")
print("=" * 70)

test_pred = xgb_tuned.predict(test_fe)

if test_ids is not None:
    submission = pd.DataFrame({"ID": test_ids.values, "int_rate": test_pred})
else:
    submission = pd.DataFrame({"int_rate": test_pred})

out_path = OUTPUTS_DIR / "test_predictions_optuna_xgb.csv"
submission.to_csv(out_path, index=False)

print(f"Saved {len(submission):,} predictions to {out_path}")
print(
    f"Predicted int_rate -- mean={test_pred.mean():.3f}  median={np.median(test_pred):.3f}  "
    f"min={test_pred.min():.3f}  max={test_pred.max():.3f}"
)
print(submission.head().to_string(index=False))


# =============================================================================
# Step 10: Predicting a single loan
# =============================================================================
# Same workflow as the W1 demo: pick a single row and produce a point
# prediction. Useful for sanity-checking individual cases and for demonstrating
# the model on unseen application data with the same column structure.

print()
print("=" * 70)
print("Step 10: Predicting a single loan")
print("=" * 70)

sample_loan = X_val.iloc[[0]]
pred_rate   = xgb_tuned.predict(sample_loan)[0]
true_rate   = y_val.iloc[0]

print(f"Predicted int_rate: {pred_rate:.2f}%")
print(f"Actual int_rate:    {true_rate:.2f}%")
print(f"Residual:           {pred_rate - true_rate:+.2f} pp")
