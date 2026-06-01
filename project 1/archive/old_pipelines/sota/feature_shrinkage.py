"""
Feature importance + shrinkage follow-on for `sota_lendingclub_int_rate.py`.

Pipeline (corresponds to Section D of the review plan):
  1. Load the final-refit CatBoost / LightGBM / XGBoost models and the
     target-encoded train / val / test frames persisted by the main script.
  2. Compute native GAIN importance from each GBDT, normalise per model,
     ensemble-average across the three. Save.
  3. Compute PERMUTATION importance against each GBDT on X_val (the
     held-out 20%) with n_repeats=5 using neg-RMSE as the score. Ensemble
     average. Save.
  4. Combined score = mean of the two min-max-normalised vectors. Save the
     merged per-feature importance file.
  5. Shrinkage pass: drop features in the bottom 25% by combined score,
     *excluding* the target-encoded `_te` columns (which act as wide signal
     carriers regardless of permutation rank). Refit single-seed CB/LGB/XGB
     on the pruned feature set, blend through the stacker coefficients
     saved by the main script, and compare X_val RMSE before vs after.

Outputs (under outputs/sota/):
  feature_importance_gain.csv
  feature_importance_permutation.csv
  feature_importance.csv
  shrinkage_results.csv

Run:
  .venv/Scripts/python.exe sota/feature_shrinkage.py
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs" / "sota"
ARTIFACTS    = OUTPUTS_DIR / "artifacts"

RANDOM_STATE = 6604

print("=" * 72)
print("[1/5] Load artifacts persisted by sota_lendingclub_int_rate.py")
print("=" * 72)

with open(ARTIFACTS / "final_models.pkl", "rb") as fh:
    art = pickle.load(fh)

cb_final  = art["catboost"]
lgb_final = art["lightgbm"]
xgb_final = art["xgboost"]
lgb_cat_dtypes = art["lgb_cat_dtypes"]
xgb_cat_dtypes = art["xgb_cat_dtypes"]
feature_cols   = art["feature_cols"]
cat_cols       = art["cat_cols"]
tuning_results = art["tuning_results"]
stacker_coef   = np.asarray(art["stacker_coef"], dtype=np.float64)
stacker_intercept = art["stacker_intercept"]
stacker_ridge_alpha = art["stacker_ridge_alpha"]

Xtr = pd.read_parquet(ARTIFACTS / "X_train_te.parquet")
Xva = pd.read_parquet(ARTIFACTS / "X_val_te.parquet")
Xte = pd.read_parquet(ARTIFACTS / "X_test_te.parquet")
ytr = pd.read_csv(ARTIFACTS / "y_train.csv")["int_rate"].values.astype(np.float64)
yva = pd.read_csv(ARTIFACTS / "y_val.csv")["int_rate"].values.astype(np.float64)

print(f"  X_train: {Xtr.shape}   X_val: {Xva.shape}   X_test: {Xte.shape}")
print(f"  Engineered columns: {len(feature_cols)}   categoricals: {len(cat_cols)}")


# ---------------------------------------------------------------------------
# Helpers to materialize the per-model views (mirror the main script)
# ---------------------------------------------------------------------------
ALL_CAT_COLS = cat_cols


def to_catboost_pool(X):
    X = X.copy()
    cc = [c for c in ALL_CAT_COLS if c in X.columns]
    for c in cc:
        X[c] = X[c].astype("string").fillna("Missing")
    idx = [X.columns.get_loc(c) for c in cc]
    for c in X.columns:
        if c not in cc:
            X[c] = X[c].astype("float64")
    return X, idx


def to_lgbm_frame(X, cat_dtypes):
    X = X.copy()
    cc = [c for c in ALL_CAT_COLS if c in X.columns]
    for c in cc:
        s = X[c].astype("string").fillna("Missing")
        X[c] = s.astype(cat_dtypes[c]) if c in cat_dtypes else s.astype("category")
    for c in X.columns:
        if c not in cc:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X, cc


def to_xgb_frame(X, cat_dtypes):
    X = X.copy()
    cc = [c for c in ALL_CAT_COLS if c in X.columns]
    for c in cc:
        s = X[c].astype("string").fillna("Missing")
        X[c] = s.astype(cat_dtypes[c]) if c in cat_dtypes else s.astype("category")
    for c in X.columns:
        if c not in cc:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


Xtr_cb_p, _ = to_catboost_pool(Xtr)
Xva_cb_p, _ = to_catboost_pool(Xva)
Xtr_lgb_p, _ = to_lgbm_frame(Xtr, lgb_cat_dtypes)
Xva_lgb_p, _ = to_lgbm_frame(Xva, lgb_cat_dtypes)
Xtr_xgb_p    = to_xgb_frame(Xtr, xgb_cat_dtypes)
Xva_xgb_p    = to_xgb_frame(Xva, xgb_cat_dtypes)


# ---------------------------------------------------------------------------
# 2. Native gain importance
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("[2/5] Native gain importance (ensemble-averaged)")
print("=" * 72)


def _normalize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.sum() <= 0:
        return arr
    return arr / arr.sum()


def _aligned_importance(model_cols, model_imp, all_cols):
    s = pd.Series(model_imp, index=model_cols)
    return s.reindex(all_cols).fillna(0.0).values


cb_imp_raw  = cb_final.feature_importances_
lgb_imp_raw = lgb_final.feature_importances_
xgb_imp_raw = xgb_final.feature_importances_

# CatBoost / LightGBM / XGBoost all see the same column ordering because
# the main script built them from the same Xtr_full_te frame. Sanity-check.
assert len(cb_imp_raw)  == len(feature_cols), "CB importance length mismatch"
assert len(lgb_imp_raw) == len(feature_cols), "LGB importance length mismatch"
assert len(xgb_imp_raw) == len(feature_cols), "XGB importance length mismatch"

gain_df = pd.DataFrame({
    "feature": feature_cols,
    "cb_gain":  _normalize(cb_imp_raw),
    "lgb_gain": _normalize(lgb_imp_raw),
    "xgb_gain": _normalize(xgb_imp_raw),
})
gain_df["gain_ensemble"] = gain_df[["cb_gain", "lgb_gain", "xgb_gain"]].mean(axis=1)
gain_df = gain_df.sort_values("gain_ensemble", ascending=False).reset_index(drop=True)
gain_df.to_csv(OUTPUTS_DIR / "feature_importance_gain.csv", index=False)
print(gain_df.head(10).to_string(index=False))
print(f"  Saved -> {OUTPUTS_DIR / 'feature_importance_gain.csv'}")


# ---------------------------------------------------------------------------
# 3. Permutation importance on X_val
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("[3/5] Permutation importance on X_val  (n_repeats=5, scoring=neg_RMSE)")
print("=" * 72)


def _perm_for_model(model, X_p, y, label):
    t = time()
    pi = permutation_importance(
        model, X_p, y,
        n_repeats=5, random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,  # tree models already use threads internally
    )
    # neg_RMSE: positive importance == feature ablation INCREASED RMSE
    # (i.e. the feature was useful). We invert so higher = more important.
    imp = -pi.importances_mean
    imp = np.clip(imp, 0, None)
    print(f"  {label}: {time()-t:.0f}s")
    return imp


perm_cb  = _perm_for_model(cb_final,  Xva_cb_p,  yva, "CatBoost")
perm_lgb = _perm_for_model(lgb_final, Xva_lgb_p, yva, "LightGBM")
perm_xgb = _perm_for_model(xgb_final, Xva_xgb_p, yva, "XGBoost ")

perm_df = pd.DataFrame({
    "feature": feature_cols,
    "cb_perm":  _normalize(perm_cb),
    "lgb_perm": _normalize(perm_lgb),
    "xgb_perm": _normalize(perm_xgb),
})
perm_df["perm_ensemble"] = perm_df[["cb_perm", "lgb_perm", "xgb_perm"]].mean(axis=1)
perm_df = perm_df.sort_values("perm_ensemble", ascending=False).reset_index(drop=True)
perm_df.to_csv(OUTPUTS_DIR / "feature_importance_permutation.csv", index=False)
print()
print(perm_df.head(10).to_string(index=False))
print(f"  Saved -> {OUTPUTS_DIR / 'feature_importance_permutation.csv'}")


# ---------------------------------------------------------------------------
# 4. Combined importance + shrinkage candidate selection
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("[4/5] Combined importance + shrinkage candidate selection")
print("=" * 72)


def _minmax(arr):
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


combined = pd.merge(
    gain_df[["feature", "gain_ensemble"]],
    perm_df[["feature", "perm_ensemble"]],
    on="feature", how="outer",
)
combined["gain_norm"] = _minmax(combined["gain_ensemble"].values)
combined["perm_norm"] = _minmax(combined["perm_ensemble"].values)
combined["combined"]  = (combined["gain_norm"] + combined["perm_norm"]) / 2
combined = combined.sort_values("combined", ascending=False).reset_index(drop=True)
combined.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

# Shrinkage rule: drop bottom-quartile by combined score, BUT keep all
# target-encoded `_te` columns and all categorical columns regardless --
# their importance is structurally diluted by being one-of-many encodings.
keep_always = set(cat_cols) | {c for c in feature_cols if c.endswith("_te")}

# threshold = 25th percentile of the *eligible* features only
eligible = combined[~combined["feature"].isin(keep_always)]
threshold = float(eligible["combined"].quantile(0.25))

drop_cols = sorted(
    eligible.loc[eligible["combined"] < threshold, "feature"].tolist()
)
keep_cols = [c for c in feature_cols if c not in drop_cols]

print(f"  Combined-score 25th percentile threshold: {threshold:.5f}")
print(f"  Dropping {len(drop_cols)} features (of {len(feature_cols)}):")
for c in drop_cols:
    score = float(combined.loc[combined["feature"] == c, "combined"].iloc[0])
    print(f"    - {c:<35s} combined={score:.5f}")


# ---------------------------------------------------------------------------
# 5. Shrinkage pass: refit GBDTs on the pruned feature set, blend, compare
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("[5/5] Refit on pruned features and compare val RMSE")
print("=" * 72)

# Helpers reused, but on pruned frames
Xtr_pruned = Xtr[keep_cols]
Xva_pruned = Xva[keep_cols]

# Pruned categorical sets (drop_cols may have removed some)
pruned_cat = [c for c in cat_cols if c in keep_cols]
ALL_CAT_COLS = pruned_cat  # used by the local to_*_frame helpers below

# Rebuild cat dtypes against the pruned training fold so categories match
lgb_cat_dtypes_p = {
    c: pd.CategoricalDtype(categories=pd.Index(
        Xtr_pruned[c].astype("string").fillna("Missing").unique()
    ))
    for c in pruned_cat
}
xgb_cat_dtypes_p = dict(lgb_cat_dtypes_p)


def _to_cb(X):
    X = X.copy()
    for c in pruned_cat:
        X[c] = X[c].astype("string").fillna("Missing")
    idx = [X.columns.get_loc(c) for c in pruned_cat]
    for c in X.columns:
        if c not in pruned_cat:
            X[c] = X[c].astype("float64")
    return X, idx


def _to_lgb(X):
    X = X.copy()
    for c in pruned_cat:
        X[c] = X[c].astype("string").fillna("Missing").astype(lgb_cat_dtypes_p[c])
    for c in X.columns:
        if c not in pruned_cat:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def _to_xgb(X):
    X = X.copy()
    for c in pruned_cat:
        X[c] = X[c].astype("string").fillna("Missing").astype(xgb_cat_dtypes_p[c])
    for c in X.columns:
        if c not in pruned_cat:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


# Refit single-seed on Xtr_pruned (no hold-out -- the comparison set IS X_val)
t = time()
Xtr_cb2, idx2 = _to_cb(Xtr_pruned)
Xva_cb2, _    = _to_cb(Xva_pruned)
cb_p = CatBoostRegressor(
    **tuning_results["CatBoost"],
    iterations=1500, loss_function="RMSE", eval_metric="RMSE",
    random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False,
)
cb_p.fit(Xtr_cb2, ytr, cat_features=idx2)
pred_cb_pruned = cb_p.predict(Xva_cb2)
print(f"  CatBoost (pruned) fit: {time()-t:.0f}s")

t = time()
Xtr_lgb2 = _to_lgb(Xtr_pruned)
Xva_lgb2 = _to_lgb(Xva_pruned)
lgb_p = LGBMRegressor(
    **tuning_results["LightGBM"],
    n_estimators=1500, objective="regression", metric="rmse",
    random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
)
lgb_p.fit(Xtr_lgb2, ytr, categorical_feature=pruned_cat)
pred_lgb_pruned = lgb_p.predict(Xva_lgb2)
print(f"  LightGBM (pruned) fit: {time()-t:.0f}s")

t = time()
Xtr_xgb2 = _to_xgb(Xtr_pruned)
Xva_xgb2 = _to_xgb(Xva_pruned)
xgb_p = XGBRegressor(
    **tuning_results["XGBoost"],
    n_estimators=1500, objective="reg:squarederror", tree_method="hist",
    enable_categorical=True, random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_p.fit(Xtr_xgb2, ytr, verbose=False)
pred_xgb_pruned = xgb_p.predict(Xva_xgb2)
print(f"  XGBoost  (pruned) fit: {time()-t:.0f}s")

# Baseline (full-feature) predictions for each GBDT on X_val
pred_cb_full  = cb_final.predict(Xva_cb_p)
pred_lgb_full = lgb_final.predict(Xva_lgb_p)
pred_xgb_full = xgb_final.predict(Xva_xgb_p)

# The saved stacker was fit on a 4-wide stack (CB, LGB, XGB, LR). For this
# shrinkage probe we only refit the three GBDTs, so blend with renormalised
# stacker weights over the GBDT columns only.
gbdt_weights = stacker_coef[:3]
gbdt_norm    = gbdt_weights / gbdt_weights.sum() if gbdt_weights.sum() > 0 else np.full(3, 1/3)

full_blend   = (
    pred_cb_full * gbdt_norm[0]
    + pred_lgb_full * gbdt_norm[1]
    + pred_xgb_full * gbdt_norm[2]
)
pruned_blend = (
    pred_cb_pruned * gbdt_norm[0]
    + pred_lgb_pruned * gbdt_norm[1]
    + pred_xgb_pruned * gbdt_norm[2]
)

shrink_rows = []
for label, pred in [
    ("CatBoost_full",  pred_cb_full),
    ("CatBoost_pruned",pred_cb_pruned),
    ("LightGBM_full",  pred_lgb_full),
    ("LightGBM_pruned",pred_lgb_pruned),
    ("XGBoost_full",   pred_xgb_full),
    ("XGBoost_pruned", pred_xgb_pruned),
    ("GBDT_blend_full",  full_blend),
    ("GBDT_blend_pruned",pruned_blend),
]:
    shrink_rows.append({
        "model":    label,
        "val_RMSE": float(np.sqrt(mean_squared_error(yva, pred))),
        "n_features_used": int(len(keep_cols) if "pruned" in label else len(feature_cols)),
    })

shrink_df = pd.DataFrame(shrink_rows)
shrink_df["val_RMSE"] = shrink_df["val_RMSE"].round(4)
shrink_df.to_csv(OUTPUTS_DIR / "shrinkage_results.csv", index=False)

print()
print(shrink_df.to_string(index=False))
print()
delta_blend = (
    shrink_df.query("model == 'GBDT_blend_pruned'")["val_RMSE"].iloc[0]
    - shrink_df.query("model == 'GBDT_blend_full'")["val_RMSE"].iloc[0]
)
verdict = (
    "PRUNING IMPROVED RMSE" if delta_blend < -0.005
    else "no meaningful change" if abs(delta_blend) <= 0.005
    else "PRUNING HURT RMSE"
)
print(f"  GBDT-blend delta (pruned - full) = {delta_blend:+.4f} pp  -> {verdict}")
print(f"  Saved -> {OUTPUTS_DIR / 'shrinkage_results.csv'}")
