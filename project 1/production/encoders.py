"""K-fold OOF target encoding + per-model frame preparation.

The target encoders are leak-safe by construction:
  - Training rows get OOF encodings (each row uses other folds' stats).
  - Val/test rows use the full-train encoder (single pass).

`prepare_for_*` helpers materialise per-model views of the data with the
correct categorical dtypes locked, so train / val / test all see the same
category indices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from production.config import TARGET_ENC_M, RANDOM_STATE

NATIVE_CAT_COLS = ["application_type", "home_ownership", "verification_status", "purpose"]
TE_CAT_COLS = ["addr_state", "purpose", "zip3", "emp_title"]
ALL_CAT_COLS = list(set(NATIVE_CAT_COLS + TE_CAT_COLS))


def smoothed_te(train_col, target, val_col=None, test_col=None,
                m=TARGET_ENC_M, folds=5, seed=RANDOM_STATE):
    """K-fold smoothed mean target encoding."""
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


def add_te_columns(Xtr, ytr, Xva, Xte):
    """Add `<col>_te` columns for high-cardinality cats. Raw cats stay so
    CatBoost can still use them natively."""
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in TE_CAT_COLS:
        if c not in Xtr.columns:
            continue
        tr_e, va_e, te_e = smoothed_te(
            Xtr[c].astype(str), ytr,
            Xva[c].astype(str) if Xva is not None else None,
            Xte[c].astype(str),
        )
        Xtr[f"{c}_te"] = tr_e
        Xva[f"{c}_te"] = va_e
        Xte[f"{c}_te"] = te_e
    return Xtr, Xva, Xte


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
    out = {}
    for c in [c for c in ALL_CAT_COLS if c in X.columns]:
        cats = pd.Index(X[c].astype("string").fillna("Missing").unique())
        out[c] = pd.CategoricalDtype(categories=cats)
    return out


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
    """Linear/NN frame: one-hot low-card cats, drop high-card raw cats (kept via TE)."""
    X = X.copy()
    low_card = [c for c in NATIVE_CAT_COLS if c in X.columns]
    for c in low_card:
        X[c] = X[c].astype("string").fillna("Missing")
    X = pd.get_dummies(X, columns=low_card, dummy_na=False, drop_first=True)
    drop = [c for c in TE_CAT_COLS if c in X.columns]
    X = X.drop(columns=drop, errors="ignore")
    X = X.astype(object).apply(pd.to_numeric, errors="coerce").astype("float64")
    return X
