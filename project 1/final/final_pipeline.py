"""
Final Composite Ensemble for LendingClub Interest Rate Prediction
==================================================================

Combines six base learners (XGBoost, LightGBM, CatBoost, Random Forest,
Ridge regression, MLP neural network) into a Ridge-stacked ensemble plus a
hill-climbed weighted blend. Uses the tuning params cached from the SOTA
Optuna run (outputs/sota/tuning_results.json) to skip retuning.

Pipeline:
  1. Load true data/LC_train.csv (100k) + LC_test.csv (10k), drop loan_status.
  2. Stateless feature engineering (FICO midpoint, term, log-tails, ratios,
     missingness flags, derog score).
  3. 80/20 random holdout for honest validation.
  4. K-fold (5) OOF predictions with:
       - smoothed K-fold target encoding for high-cardinality cats
       - 2-seed averaging per base learner
  5. Ridge meta-learner on OOF stack + hill-climb weighted blend.
  6. Final test predictions from K-fold averaged base + chosen meta blend.

Run:
  .venv/Scripts/python.exe final/final_pipeline.py
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "true data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "final"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHED_TUNING = PROJECT_ROOT / "outputs" / "sota" / "tuning_results.json"

RANDOM_STATE = 6604
N_FOLDS = 5
N_SEEDS = 1
TARGET_ENC_M = 20.0
EMP_TITLE_TOP_N = 100
HOLDOUT_FRAC = 0.20


# ---------------------------------------------------------------------------
# 1. Load + feature engineering
# ---------------------------------------------------------------------------
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
STRING_NUMERIC_COLS = [
    "dti", "revol_util", "all_util", "mo_sin_old_il_acct", "mths_since_last_record",
    "mths_since_rcnt_il", "mths_since_recent_bc", "mths_since_recent_inq", "tot_cur_bal",
]


def load_raw():
    dtype_overrides = {c: "float64" for c in STRING_NUMERIC_COLS}
    train = pd.read_csv(DATA_DIR / "LC_train.csv", na_values=["NA"], dtype=dtype_overrides)
    test = pd.read_csv(DATA_DIR / "LC_test.csv", na_values=["NA"], dtype=dtype_overrides)
    train = train.drop(columns=["loan_status"])
    test = test.drop(columns=["loan_status"])
    return train, test


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    out["term_months"] = out["term"].astype("string").str.extract(r"(\d+)", expand=False).astype("Int64")
    out["emp_length_num"] = out["emp_length"].map(EMP_LENGTH_MAP)
    for c in MTHS_SINCE_COLS:
        out[f"has_{c}"] = out[c].notna().astype(int)
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"].replace(0, np.nan)
    out["installment_proxy"] = out["loan_amnt"] / out["term_months"].astype("float64")
    out["bal_to_income"] = out["tot_cur_bal"] / out["annual_inc"].replace(0, np.nan)
    out["util_x_fico"] = out["revol_util"] * out["fico"]
    out["dti_x_fico"] = out["dti"] * out["fico"]
    out["acc_open_ratio"] = out["open_acc"] / out["total_acc"].replace(0, np.nan)
    out["dti_band"] = pd.cut(out["dti"], bins=[-np.inf, 10, 20, 30, 40, np.inf], labels=False).astype("Int64")
    out["fico_band"] = pd.cut(out["fico"], bins=[-np.inf, 660, 690, 720, 760, np.inf], labels=False).astype("Int64")
    out["derog_score"] = (
        out["delinq_2yrs"].fillna(0) * 8
        + out["pub_rec"].fillna(0) * 13
        + out["pub_rec_bankruptcies"].fillna(0) * 22
        + out["chargeoff_within_12_mths"].fillna(0) * 15
        + out["collections_12_mths_ex_med"].fillna(0) * 10
    )
    out["inq_intensity"] = out["inq_last_12m"].fillna(0) + out["inq_fi"].fillna(0)
    out["revol_util_band"] = pd.cut(out["revol_util"], bins=[-np.inf, 30, 50, 75, 100, np.inf], labels=False).astype("Int64")
    out["all_util_band"] = pd.cut(out["all_util"], bins=[-np.inf, 30, 50, 75, 100, np.inf], labels=False).astype("Int64")
    out["credit_file_age_yrs"] = out["mo_sin_old_rev_tl_op"] / 12.0
    out["term_x_loan_amnt"] = out["term_months"].astype("float64") * out["loan_amnt"]
    out["term_x_dti"] = out["term_months"].astype("float64") * out["dti"]
    out["payment_to_income"] = (
        out["loan_amnt"] / out["term_months"].astype("float64")
    ) / (out["annual_inc"].replace(0, np.nan) / 12.0)
    for c in LOG_NUMERIC_COLS:
        out[f"log1p_{c}"] = np.log1p(out[c].clip(lower=0))
    out["zip3"] = out["zip_code"].astype("string").str.extract(r"(\d{3})", expand=False)
    if "emp_title" in out.columns:
        out["emp_title"] = out["emp_title"].astype("string").str.lower().str.strip()
    drop = ["fico_range_low", "fico_range_high", "term", "emp_length", "title", "zip_code"]
    return out.drop(columns=[c for c in drop if c in out.columns])


# ---------------------------------------------------------------------------
# 2. K-fold target encoding
# ---------------------------------------------------------------------------
NATIVE_CAT_COLS = ["application_type", "home_ownership", "verification_status", "purpose"]
TE_CAT_COLS = ["addr_state", "purpose", "zip3", "emp_title"]
ALL_CAT_COLS = list(set(NATIVE_CAT_COLS + TE_CAT_COLS))


def smoothed_target_encode(train_col, target, val_col=None, test_col=None,
                            m=TARGET_ENC_M, folds=5, seed=RANDOM_STATE):
    global_mean = float(np.nanmean(target))
    val_enc = test_enc = None
    if val_col is not None or test_col is not None:
        full = pd.DataFrame({"cat": train_col.values, "y": target})
        stats = full.groupby("cat")["y"].agg(["mean", "count"])
        stats["enc"] = (stats["mean"] * stats["count"] + global_mean * m) / (stats["count"] + m)
        mapping = stats["enc"].to_dict()
        val_enc = val_col.map(mapping).fillna(global_mean).values if val_col is not None else None
        test_enc = test_col.map(mapping).fillna(global_mean).values if test_col is not None else None
    train_enc = np.full(len(train_col), global_mean, dtype=np.float64)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, va in kf.split(train_col):
        inner = pd.DataFrame({"cat": train_col.values[tr], "y": target[tr]})
        stats = inner.groupby("cat")["y"].agg(["mean", "count"])
        stats["enc"] = (stats["mean"] * stats["count"] + global_mean * m) / (stats["count"] + m)
        mapping = stats["enc"].to_dict()
        train_enc[va] = pd.Series(train_col.values[va]).map(mapping).fillna(global_mean).values
    return train_enc, val_enc, test_enc


def add_target_encodings(Xtr, ytr, Xva, Xte):
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
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


# ---------------------------------------------------------------------------
# 3. Per-model frame prep
# ---------------------------------------------------------------------------
def to_catboost_pool(X):
    X = X.copy()
    cat_cols = [c for c in ALL_CAT_COLS if c in X.columns]
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("Missing")
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    for c in X.columns:
        if c not in cat_cols:
            X[c] = X[c].astype("float64")
    return X, cat_idx


def build_cat_dtypes(X):
    dtypes = {}
    for c in [c for c in ALL_CAT_COLS if c in X.columns]:
        cats = pd.Index(X[c].astype("string").fillna("Missing").unique())
        dtypes[c] = pd.CategoricalDtype(categories=cats)
    return dtypes


def to_lgb_frame(X, cat_dtypes=None):
    X = X.copy()
    cat_cols = [c for c in ALL_CAT_COLS if c in X.columns]
    for c in cat_cols:
        s = X[c].astype("string").fillna("Missing")
        X[c] = s.astype(cat_dtypes[c]) if cat_dtypes and c in cat_dtypes else s.astype("category")
    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X, cat_cols


def to_xgb_frame(X, cat_dtypes=None):
    X = X.copy()
    cat_cols = [c for c in ALL_CAT_COLS if c in X.columns]
    for c in cat_cols:
        s = X[c].astype("string").fillna("Missing")
        X[c] = s.astype(cat_dtypes[c]) if cat_dtypes and c in cat_dtypes else s.astype("category")
    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def to_numeric_frame(X):
    """For linear / NN / RF: drop categoricals (we have TE versions of high-card ones,
    plus one-hot for low-card ones)."""
    X = X.copy()
    # One-hot the low-card categoricals
    low_card = [c for c in NATIVE_CAT_COLS if c in X.columns]
    for c in low_card:
        X[c] = X[c].astype("string").fillna("Missing")
    X = pd.get_dummies(X, columns=low_card, dummy_na=False, drop_first=True)
    # Drop high-card raw cats (keep their _te encodings)
    drop = [c for c in TE_CAT_COLS if c in X.columns]
    X = X.drop(columns=drop, errors="ignore")
    X = X.astype(object).apply(pd.to_numeric, errors="coerce").astype("float64")
    return X


# ---------------------------------------------------------------------------
# 4. Fitters
# ---------------------------------------------------------------------------
def fit_catboost(Xtr, ytr, Xva, yva, Xte, params, seed):
    Xtr_p, cat_idx = to_catboost_pool(Xtr)
    Xva_p, _ = to_catboost_pool(Xva)
    Xte_p, _ = to_catboost_pool(Xte)
    cb = CatBoostRegressor(
        **params, iterations=2500, loss_function="RMSE", eval_metric="RMSE",
        random_seed=seed, verbose=0, allow_writing_files=False, early_stopping_rounds=100,
    )
    cb.fit(Xtr_p, ytr, cat_features=cat_idx, eval_set=(Xva_p, yva), verbose=False)
    return cb.predict(Xva_p), cb.predict(Xte_p), cb


def fit_lgbm(Xtr, ytr, Xva, yva, Xte, params, seed):
    cat_dtypes = build_cat_dtypes(Xtr)
    Xtr_p, cat_cols = to_lgb_frame(Xtr, cat_dtypes)
    Xva_p, _ = to_lgb_frame(Xva, cat_dtypes)
    Xte_p, _ = to_lgb_frame(Xte, cat_dtypes)
    m = LGBMRegressor(
        **params, n_estimators=4000, objective="regression", metric="rmse",
        random_state=seed, verbose=-1, n_jobs=-1,
    )
    m.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], categorical_feature=cat_cols,
          callbacks=[lgb_early_stopping(100, verbose=False), lgb_log_evaluation(0)])
    return m.predict(Xva_p), m.predict(Xte_p), m


def fit_xgb(Xtr, ytr, Xva, yva, Xte, params, seed):
    cat_dtypes = build_cat_dtypes(Xtr)
    Xtr_p = to_xgb_frame(Xtr, cat_dtypes)
    Xva_p = to_xgb_frame(Xva, cat_dtypes)
    Xte_p = to_xgb_frame(Xte, cat_dtypes)
    m = XGBRegressor(
        **params, n_estimators=4000, objective="reg:squarederror", tree_method="hist",
        enable_categorical=True, random_state=seed, n_jobs=-1, early_stopping_rounds=100,
    )
    m.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], verbose=False)
    return m.predict(Xva_p), m.predict(Xte_p), m


def fit_rf(Xtr, ytr, Xva, Xte, seed):
    Xtr_n = to_numeric_frame(Xtr)
    Xva_n = to_numeric_frame(Xva).reindex(columns=Xtr_n.columns, fill_value=0)
    Xte_n = to_numeric_frame(Xte).reindex(columns=Xtr_n.columns, fill_value=0)
    imp = SimpleImputer(strategy="median").fit(Xtr_n.values)
    Xtr_s, Xva_s, Xte_s = imp.transform(Xtr_n.values), imp.transform(Xva_n.values), imp.transform(Xte_n.values)
    m = RandomForestRegressor(
        n_estimators=200, max_depth=14, min_samples_leaf=20, n_jobs=-1,
        random_state=seed, max_features=0.5,
    )
    m.fit(Xtr_s, ytr)
    return m.predict(Xva_s), m.predict(Xte_s), (m, imp, list(Xtr_n.columns))


def fit_ridge(Xtr, ytr, Xva, Xte, seed):
    Xtr_n = to_numeric_frame(Xtr)
    Xva_n = to_numeric_frame(Xva).reindex(columns=Xtr_n.columns, fill_value=0)
    Xte_n = to_numeric_frame(Xte).reindex(columns=Xtr_n.columns, fill_value=0)
    imp = SimpleImputer(strategy="median").fit(Xtr_n.values)
    sc = StandardScaler().fit(imp.transform(Xtr_n.values))
    Xtr_s = sc.transform(imp.transform(Xtr_n.values))
    Xva_s = sc.transform(imp.transform(Xva_n.values))
    Xte_s = sc.transform(imp.transform(Xte_n.values))
    best_a, best_rmse = 1.0, np.inf
    for a in [0.1, 1.0, 5.0, 10.0, 50.0]:
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        rmses = []
        for tr, va in cv.split(Xtr_s):
            r = Ridge(alpha=a, random_state=seed).fit(Xtr_s[tr], ytr[tr])
            rmses.append(np.sqrt(mean_squared_error(ytr[va], r.predict(Xtr_s[va]))))
        if np.mean(rmses) < best_rmse:
            best_rmse, best_a = float(np.mean(rmses)), a
    m = Ridge(alpha=best_a, random_state=seed).fit(Xtr_s, ytr)
    return m.predict(Xva_s), m.predict(Xte_s), (m, imp, sc, list(Xtr_n.columns), best_a)


def fit_mlp(Xtr, ytr, Xva, Xte, seed):
    Xtr_n = to_numeric_frame(Xtr)
    Xva_n = to_numeric_frame(Xva).reindex(columns=Xtr_n.columns, fill_value=0)
    Xte_n = to_numeric_frame(Xte).reindex(columns=Xtr_n.columns, fill_value=0)
    imp = SimpleImputer(strategy="median").fit(Xtr_n.values)
    sc = StandardScaler().fit(imp.transform(Xtr_n.values))
    Xtr_s = sc.transform(imp.transform(Xtr_n.values))
    Xva_s = sc.transform(imp.transform(Xva_n.values))
    Xte_s = sc.transform(imp.transform(Xte_n.values))
    m = MLPRegressor(
        hidden_layer_sizes=(96, 48), activation="relu", solver="adam",
        learning_rate_init=2e-3, alpha=1e-4, batch_size=512, max_iter=40,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=8,
        random_state=seed, verbose=False,
    )
    m.fit(Xtr_s, ytr)
    return m.predict(Xva_s), m.predict(Xte_s), (m, imp, sc, list(Xtr_n.columns))


# ---------------------------------------------------------------------------
# 5. Hill climbing blender
# ---------------------------------------------------------------------------
def hill_climb(P, y, steps=8000, lr=0.01, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = P.shape[1]
    w = np.ones(n) / n
    best = np.sqrt(mean_squared_error(y, P @ w))
    for _ in range(steps):
        i = rng.integers(0, n)
        cand = w.copy()
        cand[i] += rng.uniform(-lr, lr)
        cand = np.clip(cand, 0, None)
        if cand.sum() <= 0:
            continue
        cand /= cand.sum()
        rmse = np.sqrt(mean_squared_error(y, P @ cand))
        if rmse < best:
            best, w = rmse, cand
    return w, best


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("FINAL COMPOSITE ENSEMBLE PIPELINE")
    print("=" * 72)

    print("\n[1] Load + engineer")
    t = time()
    train_raw, test_raw = load_raw()
    train_fe = engineer(train_raw)
    test_fe = engineer(test_raw)

    if "emp_title" in train_fe.columns:
        top = train_fe["emp_title"].value_counts().head(EMP_TITLE_TOP_N).index.tolist()
        train_fe["emp_title"] = train_fe["emp_title"].where(train_fe["emp_title"].isin(top), "Other").fillna("Missing").astype(str)
        test_fe["emp_title"] = test_fe["emp_title"].where(test_fe["emp_title"].isin(top), "Other").fillna("Missing").astype(str)

    test_ids = test_fe["ID"].copy() if "ID" in test_fe.columns else None
    if test_ids is not None:
        test_fe = test_fe.drop(columns=["ID"])

    y_full = train_fe["int_rate"].values.astype(np.float64)
    X_full = train_fe.drop(columns=["int_rate"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE,
    )
    print(f"  shapes: train={X_train.shape}  val={X_val.shape}  test={test_fe.shape}  ({time()-t:.1f}s)")

    print("\n[2] Load cached tuning params")
    tuning = json.loads(CACHED_TUNING.read_text())["params"]
    print(f"  cached: {list(tuning.keys())}")

    print("\n[3] K-fold OOF stacking with 6 base learners x 2 seeds")
    print("    Base learners: CatBoost, LightGBM, XGBoost, RandomForest, Ridge, MLP")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_tr = len(X_train)
    n_va = len(X_val)
    n_te = len(test_fe)
    seeds = [RANDOM_STATE + 7 * i for i in range(N_SEEDS)]

    BASES = ["cb", "lgb", "xgb", "rf", "ridge", "mlp"]
    oof = {b: np.zeros(n_tr) for b in BASES}
    val_pred = {b: np.zeros(n_va) for b in BASES}
    test_pred = {b: np.zeros(n_te) for b in BASES}

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        t_fold = time()
        Xtr = X_train.iloc[tr_idx]
        Xva = X_train.iloc[va_idx]
        ytr = y_train[tr_idx]
        yva = y_train[va_idx]

        Xtr_te, Xva_te, Xte_te = add_target_encodings(Xtr, ytr, Xva, test_fe)
        _, Xval_te, _ = add_target_encodings(Xtr, ytr, X_val, test_fe)

        per = {b: {"va": np.zeros(len(va_idx)), "te": np.zeros(n_te), "vv": np.zeros(n_va)} for b in BASES}

        for s in seeds:
            t0 = time()
            cb_va, cb_te, cb_m = fit_catboost(Xtr_te, ytr, Xva_te, yva, Xte_te, tuning["CatBoost"], s)
            Xval_cb, _ = to_catboost_pool(Xval_te)
            cb_vv = cb_m.predict(Xval_cb)
            per["cb"]["va"] += cb_va; per["cb"]["te"] += cb_te; per["cb"]["vv"] += cb_vv
            print(f"    seed={s} cb done ({time()-t0:.0f}s)", flush=True)

            t0 = time()
            lgb_va, lgb_te, lgb_m = fit_lgbm(Xtr_te, ytr, Xva_te, yva, Xte_te, tuning["LightGBM"], s)
            Xval_lgb, _ = to_lgb_frame(Xval_te, build_cat_dtypes(Xtr_te))
            lgb_vv = lgb_m.predict(Xval_lgb)
            per["lgb"]["va"] += lgb_va; per["lgb"]["te"] += lgb_te; per["lgb"]["vv"] += lgb_vv
            print(f"    seed={s} lgb done ({time()-t0:.0f}s)", flush=True)

            t0 = time()
            xgb_va, xgb_te, xgb_m = fit_xgb(Xtr_te, ytr, Xva_te, yva, Xte_te, tuning["XGBoost"], s)
            Xval_xgb = to_xgb_frame(Xval_te, build_cat_dtypes(Xtr_te))
            xgb_vv = xgb_m.predict(Xval_xgb)
            per["xgb"]["va"] += xgb_va; per["xgb"]["te"] += xgb_te; per["xgb"]["vv"] += xgb_vv
            print(f"    seed={s} xgb done ({time()-t0:.0f}s)", flush=True)

            t0 = time()
            rf_va, rf_te, rf_obj = fit_rf(Xtr_te, ytr, Xva_te, Xte_te, s)
            rf_m, rf_imp, rf_cols = rf_obj
            Xval_rf = to_numeric_frame(Xval_te).reindex(columns=rf_cols, fill_value=0)
            rf_vv = rf_m.predict(rf_imp.transform(Xval_rf.values))
            per["rf"]["va"] += rf_va; per["rf"]["te"] += rf_te; per["rf"]["vv"] += rf_vv
            print(f"    seed={s} rf done ({time()-t0:.0f}s)", flush=True)

            t0 = time()
            r_va, r_te, r_obj = fit_ridge(Xtr_te, ytr, Xva_te, Xte_te, s)
            r_m, r_imp, r_sc, r_cols, _ = r_obj
            Xval_r = to_numeric_frame(Xval_te).reindex(columns=r_cols, fill_value=0)
            r_vv = r_m.predict(r_sc.transform(r_imp.transform(Xval_r.values)))
            per["ridge"]["va"] += r_va; per["ridge"]["te"] += r_te; per["ridge"]["vv"] += r_vv
            print(f"    seed={s} ridge done ({time()-t0:.0f}s)", flush=True)

            t0 = time()
            mlp_va, mlp_te, mlp_obj = fit_mlp(Xtr_te, ytr, Xva_te, Xte_te, s)
            mlp_m, mlp_imp, mlp_sc, mlp_cols = mlp_obj
            Xval_mlp = to_numeric_frame(Xval_te).reindex(columns=mlp_cols, fill_value=0)
            mlp_vv = mlp_m.predict(mlp_sc.transform(mlp_imp.transform(Xval_mlp.values)))
            per["mlp"]["va"] += mlp_va; per["mlp"]["te"] += mlp_te; per["mlp"]["vv"] += mlp_vv
            print(f"    seed={s} mlp done ({time()-t0:.0f}s)", flush=True)

        for b in BASES:
            per[b]["va"] /= len(seeds); per[b]["te"] /= len(seeds); per[b]["vv"] /= len(seeds)
            oof[b][va_idx] = per[b]["va"]
            val_pred[b] += per[b]["vv"] / N_FOLDS
            test_pred[b] += per[b]["te"] / N_FOLDS

        msg = "  fold {}: ".format(fold_id + 1) + " ".join(
            f"{b}={np.sqrt(mean_squared_error(yva, per[b]['va'])):.3f}" for b in BASES
        ) + f"  ({time()-t_fold:.0f}s)"
        print(msg, flush=True)

        # Checkpoint after every fold so we can recover from a kill
        ckpt = {
            "BASES": BASES, "completed_fold": fold_id,
            "oof": {b: oof[b].copy() for b in BASES},
            "val_pred": {b: val_pred[b].copy() for b in BASES},
            "test_pred": {b: test_pred[b].copy() for b in BASES},
            "n_folds_done": fold_id + 1,
        }
        with open(OUTPUTS_DIR / "checkpoint.pkl", "wb") as fh:
            pickle.dump(ckpt, fh)
        print(f"    [checkpoint] saved fold {fold_id+1}/{N_FOLDS} -> outputs/final/checkpoint.pkl", flush=True)

    print("\n[4] Build Ridge stacker")
    stack_tr = np.column_stack([oof[b] for b in BASES])
    stack_va = np.column_stack([val_pred[b] for b in BASES])
    stack_te = np.column_stack([test_pred[b] for b in BASES])

    best_a, best_rmse = None, np.inf
    for a in [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        rmses = []
        for tr, va in cv.split(stack_tr):
            r = Ridge(alpha=a, positive=True).fit(stack_tr[tr], y_train[tr])
            rmses.append(np.sqrt(mean_squared_error(y_train[va], r.predict(stack_tr[va]))))
        if np.mean(rmses) < best_rmse:
            best_rmse, best_a = float(np.mean(rmses)), a
    stacker = Ridge(alpha=best_a, positive=True).fit(stack_tr, y_train)
    print(f"  Ridge alpha={best_a}  CV-RMSE={best_rmse:.4f}")
    print(f"  weights:", " ".join(f"{b}={c:.3f}" for b, c in zip(BASES, stacker.coef_)),
          f"  intercept={stacker.intercept_:.3f}")
    stack_train_pred = stacker.predict(stack_tr)
    stack_val_pred = stacker.predict(stack_va)
    stack_test_pred = stacker.predict(stack_te)

    print("\n[5] Hill-climb blend")
    hc_w, hc_rmse = hill_climb(stack_tr, y_train)
    print(f"  weights:", " ".join(f"{b}={w:.3f}" for b, w in zip(BASES, hc_w)),
          f"  train-OOF RMSE={hc_rmse:.4f}")
    hc_val_pred = stack_va @ hc_w
    hc_test_pred = stack_te @ hc_w

    print("\n[6] Final comparison on 20% held-out validation")
    rows = []
    for b in BASES:
        rows.append({"model": b.upper(),
                     "val_RMSE": float(np.sqrt(mean_squared_error(y_val, val_pred[b]))),
                     "val_MAE": float(mean_absolute_error(y_val, val_pred[b])),
                     "val_R2": float(r2_score(y_val, val_pred[b]))})
    rows.append({"model": "Ridge stack",
                 "val_RMSE": float(np.sqrt(mean_squared_error(y_val, stack_val_pred))),
                 "val_MAE": float(mean_absolute_error(y_val, stack_val_pred)),
                 "val_R2": float(r2_score(y_val, stack_val_pred))})
    rows.append({"model": "Hill-climb blend",
                 "val_RMSE": float(np.sqrt(mean_squared_error(y_val, hc_val_pred))),
                 "val_MAE": float(mean_absolute_error(y_val, hc_val_pred)),
                 "val_R2": float(r2_score(y_val, hc_val_pred))})
    res = pd.DataFrame(rows)
    res_sorted = res.sort_values("val_RMSE").reset_index(drop=True)
    print(res_sorted.to_string(index=False))
    res_sorted.to_csv(OUTPUTS_DIR / "final_results.csv", index=False)

    print("\n[7] Select winning blend by held-out val RMSE")
    if res.set_index("model").loc["Ridge stack", "val_RMSE"] <= res.set_index("model").loc["Hill-climb blend", "val_RMSE"]:
        chosen, final_test = "Ridge stack", stack_test_pred
    else:
        chosen, final_test = "Hill-climb blend", hc_test_pred
    print(f"  -> {chosen}")

    final_test = np.clip(final_test, 6.0, 31.0)
    sub = pd.DataFrame({"ID": test_ids.values if test_ids is not None else np.arange(len(test_fe)),
                        "int_rate": final_test})
    sub.to_csv(OUTPUTS_DIR / "test_predictions_final.csv", index=False)
    print(f"  saved {len(sub):,} predictions -> {OUTPUTS_DIR / 'test_predictions_final.csv'}")

    print("\n[8] Persist artifacts")
    with open(OUTPUTS_DIR / "ensemble_artifacts.pkl", "wb") as fh:
        pickle.dump({
            "BASES": BASES,
            "stacker_alpha": best_a,
            "stacker_coef": list(stacker.coef_),
            "stacker_intercept": float(stacker.intercept_),
            "hill_climb_weights": hc_w.tolist(),
            "chosen_blend": chosen,
            "oof_stack": stack_tr,
            "val_stack": stack_va,
            "test_stack": stack_te,
            "y_train": y_train,
            "y_val": y_val,
        }, fh)

    summary = {
        "n_folds": N_FOLDS,
        "n_seeds": N_SEEDS,
        "base_learners": BASES,
        "best_blend": chosen,
        "stacker_ridge_alpha": best_a,
        "ridge_stack_weights": dict(zip(BASES, stacker.coef_.tolist())),
        "hill_climb_weights": dict(zip(BASES, hc_w.tolist())),
        "results": res.set_index("model").to_dict("index"),
    }
    (OUTPUTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved summary -> {OUTPUTS_DIR / 'summary.json'}")

    return res_sorted


if __name__ == "__main__":
    main()
