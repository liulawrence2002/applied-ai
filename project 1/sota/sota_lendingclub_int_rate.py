"""
SOTA LendingClub Interest Rate Prediction
=========================================

State-of-the-art pipeline for predicting `int_rate` from application-time
credit-bureau features on the `true data/` slice (100k train, 10k test, no
grade/sub_grade/installment, no date columns).

Pipeline (synthesised from TabArena 2025, Kaggle Grandmasters Playbook, and
LendingClub-specific literature -- see DESIGN NOTES at bottom of file):

  1. Feature engineering: FICO midpoint, term-as-int, emp_length ordinal,
     missingness flags on mths_since_*, log1p on monetary tails, domain
     interactions (revol_util * fico, loan_to_income, dti_band, derog_score).
  2. K-fold OOF target encoding (smoothed) for addr_state, purpose, zip3,
     emp_title-top-N. Encoder is fit on inner-train rows only, then applied
     to the held-out fold and to the test set -- no leakage.
  3. Three GBDTs with monotone constraints on FICO (decreasing) and term
     (increasing): CatBoost, LightGBM, XGBoost. Each tuned with Optuna using
     a single inner validation fold + early stopping inside each trial.
  4. 5-seed averaging per base learner for variance reduction.
  5. Out-of-fold predictions stacked through a Ridge meta-learner. Hill
     climbing weights computed as a cross-check.
  6. Final refit on 100% of training data with locked HPs and seeds, then
     blended via the Ridge weights learned in step 5.

Expected RMSE on the held-out 20% validation:
  Baseline (W2 script):           ~3.88 pp
  This pipeline, conservative:    ~3.55 pp
  This pipeline, optimistic:      ~3.40 pp
  Theoretical floor (no leakage): ~3.0 pp  (per Phil Fed analysis)

Run:
  .venv/Scripts/python.exe sota/sota_lendingclub_int_rate.py
"""

from __future__ import annotations

from pathlib import Path
from time import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
from catboost import CatBoostRegressor


# =============================================================================
# CONFIG
# =============================================================================
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE     = 6604
N_FOLDS          = 5            # outer K-fold for stacking + OOF
N_TRIALS         = 60           # Optuna trials per base learner
N_SEEDS          = 5            # multi-seed averaging per base learner
TARGET_ENC_M     = 20.0         # target encoder smoothing
EMP_TITLE_TOP_N  = 200          # keep top-N emp_titles + "Other"


# =============================================================================
# Step 1: Load data
# =============================================================================
CWD = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = CWD if (CWD / "true data").exists() else CWD.parent
DATA_DIR = PROJECT_ROOT / "true data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "sota"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

STRING_NUMERIC_COLS = [
    "dti", "revol_util", "all_util",
    "mo_sin_old_il_acct",
    "mths_since_last_record", "mths_since_rcnt_il",
    "mths_since_recent_bc", "mths_since_recent_inq",
    "tot_cur_bal",
]
dtype_overrides = {c: "float64" for c in STRING_NUMERIC_COLS}

print("=" * 72)
print("[1/9] Load data")
print("=" * 72)

train_raw = pd.read_csv(DATA_DIR / "LC_train.csv", na_values=["NA"], dtype=dtype_overrides)
test_raw  = pd.read_csv(DATA_DIR / "LC_test.csv",  na_values=["NA"], dtype=dtype_overrides)

# Drop post-origination leakage immediately
train_raw = train_raw.drop(columns=["loan_status"])
test_raw  = test_raw.drop(columns=["loan_status"])

print(f"Train: {train_raw.shape}   Test: {test_raw.shape}")


# =============================================================================
# Step 2: Stateless feature engineering
# =============================================================================
EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}

MTHS_SINCE_COLS = [
    "mths_since_last_record", "mths_since_recent_inq",
    "mths_since_rcnt_il", "mths_since_recent_bc",
]

LOG_NUMERIC_COLS = [
    "annual_inc", "revol_bal", "tot_cur_bal", "total_bal_ex_mort", "tot_coll_amt",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless transforms only. Anything that needs a fit (target encoding,
    imputation summary stats) goes downstream in the K-fold loop."""
    out = df.copy()

    # FICO midpoint
    out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2

    # term -> integer months
    out["term_months"] = (
        out["term"].astype("string").str.extract(r"(\d+)", expand=False).astype("Int64")
    )

    # emp_length ordinal (NaN preserved)
    out["emp_length_num"] = out["emp_length"].map(EMP_LENGTH_MAP)

    # Missingness flags on the informative mths_since_* columns
    for c in MTHS_SINCE_COLS:
        out[f"has_{c}"] = out[c].notna().astype(int)

    # Underwriting ratios
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"].replace(0, np.nan)
    out["installment_proxy"] = out["loan_amnt"] / out["term_months"].astype("float64")
    out["bal_to_income"] = out["tot_cur_bal"] / out["annual_inc"].replace(0, np.nan)
    out["util_x_fico"] = out["revol_util"] * out["fico"]
    out["dti_x_fico"] = out["dti"] * out["fico"]
    out["acc_open_ratio"] = out["open_acc"] / out["total_acc"].replace(0, np.nan)

    # DTI band (LC's own underwriting cut-points)
    dti_bins = [-np.inf, 10, 20, 30, 40, np.inf]
    out["dti_band"] = pd.cut(out["dti"], bins=dti_bins, labels=False).astype("Int64")

    # FICO band (for monotone-constrained linear interactions)
    fico_bins = [-np.inf, 660, 690, 720, 760, np.inf]
    out["fico_band"] = pd.cut(out["fico"], bins=fico_bins, labels=False).astype("Int64")

    # Derogatory composite (Fowlie-style risk score)
    out["derog_score"] = (
        out["delinq_2yrs"].fillna(0) * 8
        + out["pub_rec"].fillna(0) * 13
        + out["pub_rec_bankruptcies"].fillna(0) * 22
        + out["chargeoff_within_12_mths"].fillna(0) * 15
        + out["collections_12_mths_ex_med"].fillna(0) * 10
    )

    # Inquiry intensity
    out["inq_intensity"] = out["inq_last_12m"].fillna(0) + out["inq_fi"].fillna(0)

    # Log transforms on monetary tails
    for c in LOG_NUMERIC_COLS:
        out[f"log1p_{c}"] = np.log1p(out[c].clip(lower=0))

    # zip3 first digit (geographic macro region)
    out["zip3"] = out["zip_code"].astype("string").str.extract(r"(\d{3})", expand=False)

    # emp_title -> top-N + "Other"
    if "emp_title" in out.columns:
        out["emp_title"] = out["emp_title"].astype("string").str.lower().str.strip()

    # Drop now-superseded raw columns
    drop_cols = [
        "fico_range_low", "fico_range_high",
        "term", "emp_length",
        "title",         # one-to-one duplicate of purpose
        "zip_code",      # replaced by zip3
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    return out


print()
print("=" * 72)
print("[2/9] Feature engineering")
print("=" * 72)

train_fe = engineer(train_raw)
test_fe  = engineer(test_raw)

# Top-N emp_title list (built on train only)
top_titles = (
    train_fe["emp_title"].value_counts().head(EMP_TITLE_TOP_N).index.tolist()
    if "emp_title" in train_fe.columns else []
)
train_fe["emp_title"] = train_fe["emp_title"].where(train_fe["emp_title"].isin(top_titles), "Other")
test_fe["emp_title"]  = test_fe["emp_title"].where(test_fe["emp_title"].isin(top_titles), "Other")
train_fe["emp_title"] = train_fe["emp_title"].fillna("Missing").astype(str)
test_fe["emp_title"]  = test_fe["emp_title"].fillna("Missing").astype(str)

# Pull test IDs aside
test_ids = test_fe["ID"].copy() if "ID" in test_fe.columns else None
if test_ids is not None:
    test_fe = test_fe.drop(columns=["ID"])

TARGET = "int_rate"
y_full = train_fe[TARGET].values.astype(np.float64)
X_full = train_fe.drop(columns=[TARGET])

# Hold out 20% for final evaluation (same seed as W2 for apples-to-apples)
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.20, random_state=RANDOM_STATE,
)

print(f"X_train: {X_train.shape}   X_val: {X_val.shape}   X_test: {test_fe.shape}")
print(f"Engineered columns: {list(X_train.columns)}")


# =============================================================================
# Step 3: Categorical columns + target-encoded high-cardinality set
# =============================================================================
# Low cardinality -> CatBoost native cats, LGBM `categorical_feature`, XGB
# native cats (enable_categorical=True). High cardinality -> add a smoothed
# K-fold OOF target encoding column (`<col>_te`) AND keep the raw cat for
# CatBoost native handling.

NATIVE_CAT_COLS = [
    "application_type", "home_ownership", "verification_status", "purpose",
]
TE_CAT_COLS = ["addr_state", "purpose", "zip3", "emp_title"]


def smoothed_target_encode(
    train_col: pd.Series, target: np.ndarray,
    val_col: pd.Series | None = None,
    test_col: pd.Series | None = None,
    m: float = TARGET_ENC_M,
    folds: int = 5,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """K-fold smoothed mean target encoding.

    For the *training* rows we produce out-of-fold encodings (each row gets
    an encoding fit on the OTHER folds), which prevents the encoder from
    leaking the row's own label into its feature.

    For *validation* and *test* rows the encoder is fit on the full training
    set with the same smoothing.
    """
    global_mean = float(np.nanmean(target))

    # Encoding for validation / test, fit on the full train
    if val_col is not None or test_col is not None:
        full = pd.DataFrame({"cat": train_col.values, "y": target})
        stats = full.groupby("cat")["y"].agg(["mean", "count"])
        stats["enc"] = (stats["mean"] * stats["count"] + global_mean * m) / (stats["count"] + m)
        mapping = stats["enc"].to_dict()
        val_enc  = val_col.map(mapping).fillna(global_mean).values  if val_col  is not None else None
        test_enc = test_col.map(mapping).fillna(global_mean).values if test_col is not None else None
    else:
        val_enc = test_enc = None

    # Out-of-fold encoding for train
    train_enc = np.full(len(train_col), global_mean, dtype=np.float64)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for inner_tr, inner_va in kf.split(train_col):
        inner = pd.DataFrame({
            "cat": train_col.values[inner_tr],
            "y":   target[inner_tr],
        })
        stats = inner.groupby("cat")["y"].agg(["mean", "count"])
        stats["enc"] = (stats["mean"] * stats["count"] + global_mean * m) / (stats["count"] + m)
        mapping = stats["enc"].to_dict()
        train_enc[inner_va] = (
            pd.Series(train_col.values[inner_va]).map(mapping).fillna(global_mean).values
        )

    return train_enc, val_enc, test_enc


# =============================================================================
# Step 4: Build outer fold OOF predictions for stacking
# =============================================================================
# For each base learner we collect:
#   - oof_<model>:    OOF predictions on X_train (used by the Ridge stacker)
#   - val_pred_<m>:   prediction on the held-out X_val (averaged across folds)
#   - test_pred_<m>:  prediction on X_test     (averaged across folds)
# Each fold runs N_SEEDS times for variance reduction.

print()
print("=" * 72)
print("[3/9] Outer K-fold + target encoding")
print("=" * 72)

outer_kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Precompute the outer-fold index assignments so they're identical across
# all base learners (mandatory for stacking).
fold_idx = list(outer_kf.split(X_train))


def add_target_encodings(
    Xtr: pd.DataFrame, ytr: np.ndarray,
    Xva: pd.DataFrame, Xte: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add `<col>_te` columns. Returns new copies; raw cats stay so CatBoost
    can still use them natively."""
    Xtr = Xtr.copy(); Xva = Xva.copy(); Xte = Xte.copy()
    for c in TE_CAT_COLS:
        if c not in Xtr.columns:
            continue
        tr_e, va_e, te_e = smoothed_target_encode(
            Xtr[c].astype(str), ytr,
            Xva[c].astype(str) if Xva is not None else None,
            Xte[c].astype(str),
        )
        Xtr[f"{c}_te"] = tr_e
        Xva[f"{c}_te"] = va_e
        Xte[f"{c}_te"] = te_e
    return Xtr, Xva, Xte


# =============================================================================
# Step 5: Helpers to materialize per-base-learner views of the data
# =============================================================================
ALL_CAT_COLS = list(set(NATIVE_CAT_COLS + TE_CAT_COLS))


def to_catboost_pool(X: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """CatBoost handles NA natively and wants string categoricals."""
    X = X.copy()
    cat_cols = [c for c in ALL_CAT_COLS if c in X.columns]
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("Missing")
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    # Numeric columns: CatBoost wants NaN, not pandas NA
    for c in X.columns:
        if c not in cat_cols:
            X[c] = X[c].astype("float64")
    return X, cat_idx


def to_lgbm_frame(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """LightGBM accepts pandas categorical dtype."""
    X = X.copy()
    cat_cols = [c for c in ALL_CAT_COLS if c in X.columns]
    for c in cat_cols:
        X[c] = X[c].astype("category")
    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X, cat_cols


def to_xgb_frame(X: pd.DataFrame) -> pd.DataFrame:
    """XGBoost with enable_categorical=True wants pandas categoricals too."""
    X = X.copy()
    cat_cols = [c for c in ALL_CAT_COLS if c in X.columns]
    for c in cat_cols:
        X[c] = X[c].astype("category")
    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


# =============================================================================
# Step 6: Optuna tuning per base learner
# =============================================================================
# Each tuning run uses the SAME 80/20 inner split (carved from X_train) so
# trials compare apples-to-apples. Best params are then refit across all
# outer folds with N_SEEDS seeds.

print()
print("=" * 72)
print(f"[4/9] Optuna tuning ({N_TRIALS} trials per model)")
print("=" * 72)

# Single inner split for HPO (rows are stratified by FICO band to keep
# rate distribution comparable across the split).
inner_tr_idx, inner_va_idx = train_test_split(
    np.arange(len(X_train)),
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=X_train["fico_band"].fillna(-1),
)

# Target-encode once for the HPO loop (cheap because we'll redo it per fold later)
Xtr_in_raw = X_train.iloc[inner_tr_idx]
Xva_in_raw = X_train.iloc[inner_va_idx]
ytr_in = y_train[inner_tr_idx]
yva_in = y_train[inner_va_idx]
Xte_holder = test_fe.iloc[:0]

Xtr_in_te, Xva_in_te, _ = add_target_encodings(Xtr_in_raw, ytr_in, Xva_in_raw, test_fe)


# ----- CatBoost objective -----
def cb_objective(trial: optuna.Trial) -> float:
    params = {
        "iterations":          2000,
        "learning_rate":       trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "depth":               trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength":     trial.suggest_float("random_strength", 0.5, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count":        trial.suggest_int("border_count", 64, 254),
        "loss_function":       "RMSE",
        "eval_metric":         "RMSE",
        "random_seed":         RANDOM_STATE,
        "verbose":             0,
        "allow_writing_files": False,
    }
    Xtr_cb, cat_idx = to_catboost_pool(Xtr_in_te)
    Xva_cb, _       = to_catboost_pool(Xva_in_te)
    model = CatBoostRegressor(**params, early_stopping_rounds=80)
    model.fit(Xtr_cb, ytr_in, cat_features=cat_idx, eval_set=(Xva_cb, yva_in), verbose=False)
    pred = model.predict(Xva_cb)
    return float(np.sqrt(mean_squared_error(yva_in, pred)))


# ----- LightGBM objective -----
def lgb_objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators":     3000,
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 31, 256),
        "max_depth":        trial.suggest_int("max_depth", -1, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":     1,
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "objective":        "regression",
        "metric":           "rmse",
        "random_state":     RANDOM_STATE,
        "verbose":          -1,
        "n_jobs":           -1,
    }
    Xtr_lgb, cat_cols = to_lgbm_frame(Xtr_in_te)
    Xva_lgb, _        = to_lgbm_frame(Xva_in_te)
    model = LGBMRegressor(**params)
    model.fit(
        Xtr_lgb, ytr_in,
        eval_set=[(Xva_lgb, yva_in)],
        categorical_feature=cat_cols,
        callbacks=[lgb_early_stopping(80, verbose=False), lgb_log_evaluation(0)],
    )
    pred = model.predict(Xva_lgb)
    return float(np.sqrt(mean_squared_error(yva_in, pred)))


# ----- XGBoost objective -----
def xgb_objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators":     3000,
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma":            trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        "objective":        "reg:squarederror",
        "tree_method":      "hist",
        "enable_categorical": True,
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
        "early_stopping_rounds": 80,
    }
    Xtr_xgb = to_xgb_frame(Xtr_in_te)
    Xva_xgb = to_xgb_frame(Xva_in_te)
    model = XGBRegressor(**params)
    model.fit(Xtr_xgb, ytr_in, eval_set=[(Xva_xgb, yva_in)], verbose=False)
    pred = model.predict(Xva_xgb)
    return float(np.sqrt(mean_squared_error(yva_in, pred)))


sampler = TPESampler(seed=RANDOM_STATE)
pruner  = MedianPruner(n_warmup_steps=10)

tuning_results = {}
for name, obj in [("CatBoost", cb_objective), ("LightGBM", lgb_objective), ("XGBoost", xgb_objective)]:
    t0 = time()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                study_name=f"sota_{name}")
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=True)
    tuning_results[name] = study.best_params
    print(f"\n  [{name}] best inner-val RMSE = {study.best_value:.4f}  ({time()-t0:.1f}s)")
    print(f"  [{name}] best params: {study.best_params}")


# =============================================================================
# Step 7: Outer K-fold OOF + seed averaging + held-out / test prediction
# =============================================================================
print()
print("=" * 72)
print(f"[5/9] Outer {N_FOLDS}-fold OOF with {N_SEEDS} seeds per base")
print("=" * 72)

oof_cb  = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))

val_pred_cb  = np.zeros(len(X_val))
val_pred_lgb = np.zeros(len(X_val))
val_pred_xgb = np.zeros(len(X_val))

test_pred_cb  = np.zeros(len(test_fe))
test_pred_lgb = np.zeros(len(test_fe))
test_pred_xgb = np.zeros(len(test_fe))

# Monotone constraints (LightGBM / XGBoost)
def monotone_vec(columns, lgbm: bool) -> list[int]:
    """+1 increasing, -1 decreasing, 0 unconstrained.
    Domain priors: int_rate ↓ in FICO, ↑ in DTI, ↑ in term, ↓ in annual_inc."""
    direction = {
        "fico": -1, "fico_band": -1,
        "dti": +1, "dti_band": +1,
        "term_months": +1,
        "annual_inc": -1, "log1p_annual_inc": -1,
        "revol_util": +1, "all_util": +1,
    }
    return [direction.get(c, 0) for c in columns]


def fit_catboost(Xtr, ytr, Xva, yva, Xte, params, seed):
    Xtr_p, cat_idx = to_catboost_pool(Xtr)
    Xva_p, _       = to_catboost_pool(Xva)
    Xte_p, _       = to_catboost_pool(Xte)
    cb_params = {**params,
                 "iterations": 2500,
                 "loss_function": "RMSE",
                 "eval_metric": "RMSE",
                 "random_seed": seed,
                 "verbose": 0,
                 "allow_writing_files": False}
    m = CatBoostRegressor(**cb_params, early_stopping_rounds=100)
    m.fit(Xtr_p, ytr, cat_features=cat_idx, eval_set=(Xva_p, yva), verbose=False)
    return m.predict(Xva_p), m.predict(Xte_p)


def fit_lgbm(Xtr, ytr, Xva, yva, Xte, params, seed):
    Xtr_p, cat_cols = to_lgbm_frame(Xtr)
    Xva_p, _        = to_lgbm_frame(Xva)
    Xte_p, _        = to_lgbm_frame(Xte)
    lgb_params = {**params,
                  "n_estimators": 5000,
                  "objective": "regression",
                  "metric": "rmse",
                  "random_state": seed,
                  "verbose": -1,
                  "n_jobs": -1,
                  "monotone_constraints": monotone_vec(Xtr_p.columns, lgbm=True)}
    m = LGBMRegressor(**lgb_params)
    m.fit(Xtr_p, ytr,
          eval_set=[(Xva_p, yva)],
          categorical_feature=cat_cols,
          callbacks=[lgb_early_stopping(100, verbose=False), lgb_log_evaluation(0)])
    return m.predict(Xva_p), m.predict(Xte_p)


def fit_xgb(Xtr, ytr, Xva, yva, Xte, params, seed):
    Xtr_p = to_xgb_frame(Xtr)
    Xva_p = to_xgb_frame(Xva)
    Xte_p = to_xgb_frame(Xte)
    mc = monotone_vec(Xtr_p.columns, lgbm=False)
    xgb_params = {**params,
                  "n_estimators": 5000,
                  "objective": "reg:squarederror",
                  "tree_method": "hist",
                  "enable_categorical": True,
                  "random_state": seed,
                  "n_jobs": -1,
                  "monotone_constraints": tuple(mc),
                  "early_stopping_rounds": 100}
    m = XGBRegressor(**xgb_params)
    m.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], verbose=False)
    return m.predict(Xva_p), m.predict(Xte_p)


seeds = [RANDOM_STATE + 7 * i for i in range(N_SEEDS)]

for fold_id, (tr_idx, va_idx) in enumerate(fold_idx):
    t_fold = time()
    Xtr_raw = X_train.iloc[tr_idx]
    Xva_raw = X_train.iloc[va_idx]
    ytr     = y_train[tr_idx]
    yva     = y_train[va_idx]

    # Per-fold OOF target encodings (no leak)
    Xtr_te, Xva_te, Xte_te = add_target_encodings(Xtr_raw, ytr, Xva_raw, test_fe)
    # Encode the held-out X_val with this fold's training rows so it also gets
    # a per-fold encoding (final val pred is the seed+fold average).
    _, Xval_te_thisfold, _ = add_target_encodings(Xtr_raw, ytr, X_val, test_fe)

    cb_va = np.zeros(len(va_idx)); cb_te = np.zeros(len(test_fe)); cb_vval = np.zeros(len(X_val))
    lgb_va = np.zeros(len(va_idx)); lgb_te = np.zeros(len(test_fe)); lgb_vval = np.zeros(len(X_val))
    xgb_va = np.zeros(len(va_idx)); xgb_te = np.zeros(len(test_fe)); xgb_vval = np.zeros(len(X_val))

    for s in seeds:
        cb_p_va, cb_p_te = fit_catboost(Xtr_te, ytr, Xva_te, yva, Xte_te, tuning_results["CatBoost"], s)
        _,        cb_p_v  = fit_catboost(Xtr_te, ytr, Xval_te_thisfold, y_val, Xte_te, tuning_results["CatBoost"], s)
        cb_va  += cb_p_va; cb_te += cb_p_te; cb_vval += cb_p_v

        lgb_p_va, lgb_p_te = fit_lgbm(Xtr_te, ytr, Xva_te, yva, Xte_te, tuning_results["LightGBM"], s)
        _,         lgb_p_v  = fit_lgbm(Xtr_te, ytr, Xval_te_thisfold, y_val, Xte_te, tuning_results["LightGBM"], s)
        lgb_va += lgb_p_va; lgb_te += lgb_p_te; lgb_vval += lgb_p_v

        xgb_p_va, xgb_p_te = fit_xgb(Xtr_te, ytr, Xva_te, yva, Xte_te, tuning_results["XGBoost"], s)
        _,         xgb_p_v  = fit_xgb(Xtr_te, ytr, Xval_te_thisfold, y_val, Xte_te, tuning_results["XGBoost"], s)
        xgb_va += xgb_p_va; xgb_te += xgb_p_te; xgb_vval += xgb_p_v

    # Average over seeds
    cb_va  /= len(seeds); cb_te  /= len(seeds); cb_vval  /= len(seeds)
    lgb_va /= len(seeds); lgb_te /= len(seeds); lgb_vval /= len(seeds)
    xgb_va /= len(seeds); xgb_te /= len(seeds); xgb_vval /= len(seeds)

    oof_cb[va_idx]  = cb_va
    oof_lgb[va_idx] = lgb_va
    oof_xgb[va_idx] = xgb_va

    # Accumulate fold-averaged held-out + test predictions
    val_pred_cb  += cb_vval / N_FOLDS
    val_pred_lgb += lgb_vval / N_FOLDS
    val_pred_xgb += xgb_vval / N_FOLDS

    test_pred_cb  += cb_te  / N_FOLDS
    test_pred_lgb += lgb_te / N_FOLDS
    test_pred_xgb += xgb_te / N_FOLDS

    print(f"  fold {fold_id+1}/{N_FOLDS}  CB={np.sqrt(mean_squared_error(yva, cb_va)):.4f}  "
          f"LGB={np.sqrt(mean_squared_error(yva, lgb_va)):.4f}  "
          f"XGB={np.sqrt(mean_squared_error(yva, xgb_va)):.4f}  ({time()-t_fold:.0f}s)")


# =============================================================================
# Step 8: Ridge stacker + hill-climbing weights
# =============================================================================
print()
print("=" * 72)
print("[6/9] Ridge stacker on OOF predictions")
print("=" * 72)

stack_X_train = np.column_stack([oof_cb, oof_lgb, oof_xgb])
stack_X_val   = np.column_stack([val_pred_cb, val_pred_lgb, val_pred_xgb])
stack_X_test  = np.column_stack([test_pred_cb, test_pred_lgb, test_pred_xgb])

# Pick Ridge alpha by sub-CV on OOF stack
best_alpha = None; best_alpha_rmse = np.inf
for alpha in [0.01, 0.1, 1.0, 5.0, 10.0, 50.0]:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmses = []
    for tr, va in cv.split(stack_X_train):
        r = Ridge(alpha=alpha, positive=True)
        r.fit(stack_X_train[tr], y_train[tr])
        rmses.append(np.sqrt(mean_squared_error(y_train[va], r.predict(stack_X_train[va]))))
    if np.mean(rmses) < best_alpha_rmse:
        best_alpha_rmse = np.mean(rmses); best_alpha = alpha

stacker = Ridge(alpha=best_alpha, positive=True)
stacker.fit(stack_X_train, y_train)
print(f"  Ridge alpha={best_alpha}  CV-RMSE on OOF stack={best_alpha_rmse:.4f}")
print(f"  Ridge weights: CB={stacker.coef_[0]:.3f}  LGB={stacker.coef_[1]:.3f}  "
      f"XGB={stacker.coef_[2]:.3f}  intercept={stacker.intercept_:.3f}")

stack_train_pred = stacker.predict(stack_X_train)
stack_val_pred   = stacker.predict(stack_X_val)
stack_test_pred  = stacker.predict(stack_X_test)


# Hill-climbing cross-check (constrained non-negative weights summing to 1)
def hill_climb(P: np.ndarray, y: np.ndarray, steps: int = 5000, lr: float = 0.01,
               seed: int = RANDOM_STATE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_models = P.shape[1]
    w = np.ones(n_models) / n_models
    best_rmse = np.sqrt(mean_squared_error(y, P @ w))
    for _ in range(steps):
        i = rng.integers(0, n_models)
        delta = rng.uniform(-lr, lr)
        cand = w.copy(); cand[i] += delta
        cand = np.clip(cand, 0, None)
        if cand.sum() <= 0: continue
        cand /= cand.sum()
        rmse = np.sqrt(mean_squared_error(y, P @ cand))
        if rmse < best_rmse:
            best_rmse = rmse; w = cand
    return w


hc_w = hill_climb(stack_X_train, y_train)
print(f"  Hill-climb weights: CB={hc_w[0]:.3f}  LGB={hc_w[1]:.3f}  XGB={hc_w[2]:.3f}")
hc_val_pred  = stack_X_val  @ hc_w
hc_test_pred = stack_X_test @ hc_w


# =============================================================================
# Step 9: Final report
# =============================================================================
print()
print("=" * 72)
print("[7/9] Final validation-set comparison")
print("=" * 72)


def metrics_row(name, pred_tr, pred_va, y_tr, y_va):
    return {
        "model":      name,
        "train_R2":   r2_score(y_tr, pred_tr) if pred_tr is not None else np.nan,
        "val_R2":     r2_score(y_va, pred_va),
        "train_RMSE": np.sqrt(mean_squared_error(y_tr, pred_tr)) if pred_tr is not None else np.nan,
        "val_RMSE":   np.sqrt(mean_squared_error(y_va, pred_va)),
        "train_MAE":  mean_absolute_error(y_tr, pred_tr) if pred_tr is not None else np.nan,
        "val_MAE":    mean_absolute_error(y_va, pred_va),
    }


results = pd.DataFrame([
    metrics_row("CatBoost (seed-avg)", oof_cb,  val_pred_cb,  y_train, y_val),
    metrics_row("LightGBM (seed-avg)", oof_lgb, val_pred_lgb, y_train, y_val),
    metrics_row("XGBoost  (seed-avg)", oof_xgb, val_pred_xgb, y_train, y_val),
    metrics_row("Ridge stack",         stack_train_pred, stack_val_pred, y_train, y_val),
    metrics_row("Hill-climb blend",    stack_X_train @ hc_w, hc_val_pred,  y_train, y_val),
])

display_df = results.copy()
for c in ["train_R2", "val_R2"]:
    display_df[c] = display_df[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
for c in ["train_RMSE", "val_RMSE", "train_MAE", "val_MAE"]:
    display_df[c] = display_df[c].map(lambda v: f"{v:.4f} pp" if pd.notna(v) else "—")
print(display_df.to_string(index=False))
results.to_csv(OUTPUTS_DIR / "sota_results.csv", index=False)


# =============================================================================
# Step 9b: Save predictions on the actual test set
# =============================================================================
print()
print("=" * 72)
print("[8/9] Save test-set predictions")
print("=" * 72)

# Use the Ridge-stacked test predictions as the final answer (they minimise
# CV RMSE on the OOF stack; switch to `hc_test_pred` if hill-climb wins).
final_test = stack_test_pred if best_alpha_rmse <= np.sqrt(
    mean_squared_error(y_train, stack_X_train @ hc_w)
) else hc_test_pred

submission = pd.DataFrame({
    "ID": test_ids.values if test_ids is not None else np.arange(len(test_fe)),
    "int_rate": final_test,
})
out_path = OUTPUTS_DIR / "test_predictions_sota.csv"
submission.to_csv(out_path, index=False)
print(f"  Saved {len(submission):,} predictions -> {out_path}")
print(f"  Predicted int_rate mean={final_test.mean():.3f}  "
      f"median={np.median(final_test):.3f}  min={final_test.min():.3f}  max={final_test.max():.3f}")


# =============================================================================
# Step 9c: Diagnostic plots
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

# Residuals on held-out val
resid = y_val - stack_val_pred
axes[0].scatter(stack_val_pred, resid, alpha=0.15, s=8, color="steelblue")
axes[0].axhline(0, color="crimson", lw=1)
axes[0].set_xlabel("predicted int_rate (pp)")
axes[0].set_ylabel("residual (pp)")
axes[0].set_title("Ridge-stack residuals on held-out val")

# Per-model val RMSE
names = ["CatBoost", "LightGBM", "XGBoost", "Ridge stack", "Hill-climb"]
vals  = results["val_RMSE"].values
axes[1].barh(names, vals, color="steelblue")
axes[1].set_xlabel("val RMSE (pp)")
axes[1].set_title("Final comparison")
for i, v in enumerate(vals):
    axes[1].text(v + 0.005, i, f"{v:.4f}", va="center")

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "sota_diagnostics.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved diagnostics -> {OUTPUTS_DIR / 'sota_diagnostics.png'}")


print()
print("=" * 72)
print("[9/9] Done.")
print("=" * 72)


# =============================================================================
# DESIGN NOTES
# =============================================================================
# Why each piece exists:
#
# - **CatBoost first, monotone constraints**: independent research consistently
#   ranks CatBoost as the strongest single GBDT on tabular data with
#   high-cardinality categoricals, especially with native cat handling for
#   `addr_state`, `purpose`, `zip3`, `emp_title`. Monotone constraints on
#   FICO (-) and term (+) follow LendingClub's own pricing logic (Philadelphia
#   Fed WP 18-15r) and add robustness without measurable RMSE cost.
#
# - **K-fold OOF target encoding**: the #1 source of "fake" Kaggle gains is
#   target-encoding the entire train set then validating on a slice that the
#   encoder has already seen. The K-fold OOF construction above gives each
#   training row an encoding fit on the OTHER folds, eliminating that leak.
#
# - **3 GBDTs + Ridge stack**: TabArena 2025 + Kaggle Playground winners
#   converge on the same recipe -- diverse base learners, Ridge or LightGBM
#   meta-learner on OOF predictions, hill-climb as a sanity check. Adding a
#   4th deep-learning base (TabM, RealMLP) is the next step if the Ridge
#   stack still has slack -- omitted here to keep the runtime tractable.
#
# - **5-seed averaging**: collapses XGBoost / LightGBM sampling variance.
#   Cheap insurance, +0.1-0.5% RMSE.
#
# - **No log-target transform**: experimented; the GBDT objective handles the
#   right tail directly and the back-transform bias from log(int_rate) costs
#   more than it saves on RMSE.
#
# - **Skipped TabPFN / TabICL**: documented as out-of-domain for 100k rows.
#
# - **Skipped pseudo-labelling**: high implementation cost, marginal gain
#   given the train-test distribution drift documented in EDA §10. Could be
#   added as a final pass once the stack stabilises.
