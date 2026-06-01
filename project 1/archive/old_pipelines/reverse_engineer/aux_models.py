"""Auxiliary CatBoost models trained on archive data.

Each call to `train_all_aux(source_df)` returns 3 fitted models — one per
target (grade, sub_grade_ord, int_rate). Frames are aligned to
COMMON_FEATURES so the models can later be applied to `true data/` rows
that lack grade/sub_grade.
"""
from __future__ import annotations

from time import time
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split

from reverse_engineer.archive_loader import COMMON_FEATURES, add_subgrade_ordinal, split_X_y
from reverse_engineer.config import AUX_CB_PARAMS, AUX_VAL_FRAC, RANDOM_STATE


def _prepare_x(X: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """CatBoost wants string categoricals (filled NA), float numerics (NaN OK)."""
    X = X.copy()
    cat_cols = [c for c in ["addr_state", "application_type", "emp_length",
                            "emp_title", "home_ownership", "purpose",
                            "term", "title", "verification_status",
                            "zip_code"] if c in X.columns]
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("Missing")
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X, cat_idx


def _attach_extra_features(X: pd.DataFrame, extra: Optional[pd.DataFrame]) -> pd.DataFrame:
    """For iter >= 2, the prior-iteration's predictions on archive rows are
    attached as numeric features (named `prev_*`). They are numeric so
    `_prepare_x` treats them correctly."""
    if extra is None or extra.empty:
        return X
    extra = extra.reset_index(drop=True)
    X = X.reset_index(drop=True)
    return pd.concat([X, extra.add_prefix("prev_")], axis=1)


def train_aux_grade(X, y, params=AUX_CB_PARAMS, seed=RANDOM_STATE,
                    extra: Optional[pd.DataFrame] = None):
    X = _attach_extra_features(X, extra)
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=AUX_VAL_FRAC, random_state=seed, stratify=y,
    )
    Xtr_p, cat_idx = _prepare_x(Xtr)
    Xva_p, _ = _prepare_x(Xva)
    m = CatBoostClassifier(
        **{k: v for k, v in params.items() if k != "early_stopping_rounds"},
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        early_stopping_rounds=params.get("early_stopping_rounds", 100),
    )
    m.fit(Xtr_p, ytr, cat_features=cat_idx, eval_set=(Xva_p, yva), verbose=False)
    return m


def train_aux_subgrade_ord(X, y_ord, params=AUX_CB_PARAMS, seed=RANDOM_STATE,
                            extra: Optional[pd.DataFrame] = None):
    X = _attach_extra_features(X, extra)
    Xtr, Xva, ytr, yva = train_test_split(
        X, y_ord, test_size=AUX_VAL_FRAC, random_state=seed,
    )
    Xtr_p, cat_idx = _prepare_x(Xtr)
    Xva_p, _ = _prepare_x(Xva)
    m = CatBoostRegressor(
        **{k: v for k, v in params.items() if k != "early_stopping_rounds"},
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        early_stopping_rounds=params.get("early_stopping_rounds", 100),
    )
    m.fit(Xtr_p, ytr, cat_features=cat_idx, eval_set=(Xva_p, yva), verbose=False)
    return m


def train_aux_int_rate(X, y_rate, params=AUX_CB_PARAMS, seed=RANDOM_STATE,
                        extra: Optional[pd.DataFrame] = None):
    X = _attach_extra_features(X, extra)
    Xtr, Xva, ytr, yva = train_test_split(
        X, y_rate, test_size=AUX_VAL_FRAC, random_state=seed,
    )
    Xtr_p, cat_idx = _prepare_x(Xtr)
    Xva_p, _ = _prepare_x(Xva)
    m = CatBoostRegressor(
        **{k: v for k, v in params.items() if k != "early_stopping_rounds"},
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        early_stopping_rounds=params.get("early_stopping_rounds", 100),
    )
    m.fit(Xtr_p, ytr, cat_features=cat_idx, eval_set=(Xva_p, yva), verbose=False)
    return m


def train_all_aux(source_name: str, source_df: pd.DataFrame,
                  extra_features: Optional[pd.DataFrame] = None) -> dict:
    """Train the 3 aux targets on one archive source. Returns {target: model}.

    `extra_features` (optional): a same-length DataFrame of prior-iter
    predictions, attached to the source_df as `prev_*` columns. The
    classifier/regressors then learn the inverse mapping
    "given this predicted rate, what grade does it imply?"
    """
    source_df = add_subgrade_ordinal(source_df)
    X, _ = split_X_y(source_df, target="int_rate")  # gets COMMON_FEATURES only
    y_grade = source_df["grade"]
    y_subord = source_df["sub_grade_ord"]
    y_rate = source_df["int_rate"].astype("float64")

    print(f"  [{source_name}] training aux models on {len(X):,} rows, "
          f"{X.shape[1]} cols (+extra={extra_features is not None})", flush=True)

    t = time()
    m_grade = train_aux_grade(X, y_grade, extra=extra_features)
    print(f"    grade clf done ({time()-t:.0f}s)", flush=True)

    t = time()
    m_subord = train_aux_subgrade_ord(X, y_subord, extra=extra_features)
    print(f"    sub_grade_ord reg done ({time()-t:.0f}s)", flush=True)

    t = time()
    m_rate = train_aux_int_rate(X, y_rate, extra=extra_features)
    print(f"    int_rate reg done ({time()-t:.0f}s)", flush=True)

    return {"grade": m_grade, "sub_grade_ord": m_subord, "int_rate": m_rate}
