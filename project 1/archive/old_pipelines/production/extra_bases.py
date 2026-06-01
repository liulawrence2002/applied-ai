"""Two more orthogonal base learners: HistGradientBoosting + ExtraTrees.

LightAutoML 0.4.2 won't install on Python 3.14 (needs tabicl/tabm and pins
xgboost<3). These two sklearn-native models give similar ensemble diversity
without the install friction:

  - HistGradientBoostingRegressor: sklearn's binned GBDT. Uses a different
    histogram binning strategy than LightGBM (more conservative on rare
    categories), so its errors are partially decorrelated from lgb/cb/xgb.

  - ExtraTreesRegressor: extremely randomised trees (random split values, not
    optimal). Behaves like a high-variance, low-bias RF — captures wiggles
    that the smooth GBDT loss surface misses.

Both produce OOF / val / test predictions and are appended to aggregate.pkl's
BASES list so the stacker picks them up automatically.
"""
from __future__ import annotations

import pickle
from time import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from production.config import (
    CHECKPOINTS_DIR, EMP_TITLE_TOP_N, HOLDOUT_FRAC, N_FOLDS, N_SEEDS_GBDT,
    RANDOM_STATE,
)
from production.encoders import add_te_columns, to_numeric_frame
from production.features import compact_emp_title, engineer, load_raw


def _prepare_data():
    train_raw, test_raw = load_raw()
    train_fe = engineer(train_raw)
    test_fe = engineer(test_raw)
    train_fe, test_fe = compact_emp_title(train_fe, test_fe, EMP_TITLE_TOP_N)
    if "ID" in test_fe.columns:
        test_fe = test_fe.drop(columns=["ID"])
    y_full = train_fe["int_rate"].values.astype(np.float64)
    X_full = train_fe.drop(columns=["int_rate"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE,
    )
    return X_train, X_val, y_train, y_val, test_fe


def _fit_hgb(Xtr, ytr, Xva, Xval, Xte, seed):
    Xtr_n = to_numeric_frame(Xtr)
    Xva_n = to_numeric_frame(Xva).reindex(columns=Xtr_n.columns, fill_value=0)
    Xval_n = to_numeric_frame(Xval).reindex(columns=Xtr_n.columns, fill_value=0)
    Xte_n = to_numeric_frame(Xte).reindex(columns=Xtr_n.columns, fill_value=0)
    imp = SimpleImputer(strategy="median").fit(Xtr_n.values)
    Xtr_s = imp.transform(Xtr_n.values)
    Xva_s = imp.transform(Xva_n.values)
    Xval_s = imp.transform(Xval_n.values)
    Xte_s = imp.transform(Xte_n.values)
    m = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=2000,
        learning_rate=0.03,
        max_leaf_nodes=63,
        max_depth=10,
        min_samples_leaf=40,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=seed,
        verbose=0,
    )
    m.fit(Xtr_s, ytr)
    return m.predict(Xva_s), m.predict(Xval_s), m.predict(Xte_s)


def _fit_et(Xtr, ytr, Xva, Xval, Xte, seed):
    Xtr_n = to_numeric_frame(Xtr)
    Xva_n = to_numeric_frame(Xva).reindex(columns=Xtr_n.columns, fill_value=0)
    Xval_n = to_numeric_frame(Xval).reindex(columns=Xtr_n.columns, fill_value=0)
    Xte_n = to_numeric_frame(Xte).reindex(columns=Xtr_n.columns, fill_value=0)
    imp = SimpleImputer(strategy="median").fit(Xtr_n.values)
    Xtr_s = imp.transform(Xtr_n.values)
    Xva_s = imp.transform(Xva_n.values)
    Xval_s = imp.transform(Xval_n.values)
    Xte_s = imp.transform(Xte_n.values)
    m = ExtraTreesRegressor(
        n_estimators=400, max_depth=18, min_samples_leaf=15,
        max_features=0.5, n_jobs=-1, random_state=seed,
    )
    m.fit(Xtr_s, ytr)
    return m.predict(Xva_s), m.predict(Xval_s), m.predict(Xte_s)


def _run_one(name, fit_fn, n_seeds):
    agg_path = CHECKPOINTS_DIR / "aggregate.pkl"
    with open(agg_path, "rb") as f:
        agg = pickle.load(f)

    if name in agg.get("BASES", []):
        print(f"[{name}] already in aggregate, skipping", flush=True)
        return agg

    X_train, X_val, y_train, y_val, X_test = _prepare_data()
    n_tr, n_va, n_te = len(X_train), len(X_val), len(X_test)
    oof = np.zeros(n_tr); val_p = np.zeros(n_va); test_p = np.zeros(n_te)
    seeds = [RANDOM_STATE + 31 * i for i in range(n_seeds)]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        t = time()
        Xtr = X_train.iloc[tr_idx]; ytr = y_train[tr_idx]
        Xva = X_train.iloc[va_idx]; yva = y_train[va_idx]
        Xtr_te, Xva_te, Xte_te = add_te_columns(Xtr, ytr, Xva, X_test)
        _, Xval_te, _ = add_te_columns(Xtr, ytr, X_val, X_test)
        va_acc = np.zeros(len(va_idx)); vv_acc = np.zeros(n_va); te_acc = np.zeros(n_te)
        for s in seeds:
            va, vv, te = fit_fn(Xtr_te, ytr, Xva_te, Xval_te, Xte_te, s)
            va_acc += va; vv_acc += vv; te_acc += te
        va_acc /= len(seeds); vv_acc /= len(seeds); te_acc /= len(seeds)
        oof[va_idx] = va_acc
        val_p += vv_acc / N_FOLDS
        test_p += te_acc / N_FOLDS
        rmse = float(np.sqrt(mean_squared_error(yva, va_acc)))
        print(f"  fold {fold_id+1}/{N_FOLDS}: {name}={rmse:.3f}  ({time()-t:.0f}s)", flush=True)

    if name not in agg["BASES"]:
        agg["BASES"] = list(agg["BASES"]) + [name]
    agg["oof"][name] = oof
    agg["val_pred"][name] = val_p
    agg["test_pred"][name] = test_p
    with open(agg_path, "wb") as f:
        pickle.dump(agg, f)
    val_rmse = float(np.sqrt(mean_squared_error(agg["y_val"], val_p)))
    print(f"[{name}] held-out val RMSE = {val_rmse:.4f}", flush=True)
    return agg


def run():
    print("=" * 72)
    print("EXTRA BASES PHASE -> hgb + extra_trees")
    print("=" * 72, flush=True)
    _run_one("hgb", _fit_hgb, n_seeds=2)
    agg = _run_one("extra_trees", _fit_et, n_seeds=1)
    return agg


if __name__ == "__main__":
    run()
