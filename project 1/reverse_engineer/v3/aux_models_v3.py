"""V3 aux models: 35-class sub_grade, issue_year regressor, within-grade
residual regressor. Trained on archive sources. Each fitter returns a fitted
CatBoost model whose feature_names_ list locks the column order used at
predict time.
"""
from __future__ import annotations

from time import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split

from reverse_engineer.archive_loader import (
    ARCHIVE_LC_TRAIN_CSV, ARCHIVE_LOAN_CSV, COMMON_FEATURES, NA_VALUES,
    _trim_revol_util, _coerce_numeric_strings, split_X_y, add_subgrade_ordinal,
)
from reverse_engineer.aux_models import _prepare_x
from reverse_engineer.v3.config_v3 import (
    AUX_V3_CB_PARAMS, AUX_V3_SG35_PARAMS, RANDOM_STATE, SKIP_SG35,
)


def load_loan_with_dates(max_rows: int | None = None) -> pd.DataFrame:
    """Load loan.csv with issue_d included alongside the COMMON_FEATURES + targets."""
    use_cols = list(set(COMMON_FEATURES + ["grade", "sub_grade", "int_rate", "issue_d"]))
    df = pd.read_csv(
        ARCHIVE_LOAN_CSV,
        usecols=lambda c: c in use_cols,
        nrows=max_rows,
        na_values=NA_VALUES,
        low_memory=False,
    )
    if "revol_util" in df.columns:
        df["revol_util"] = _trim_revol_util(df["revol_util"])
    df = _coerce_numeric_strings(df)
    df = df.dropna(subset=["grade", "sub_grade", "int_rate"]).reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def load_lct_with_dates() -> pd.DataFrame:
    df = pd.read_csv(ARCHIVE_LC_TRAIN_CSV, na_values=NA_VALUES, low_memory=False)
    if "revol_util" in df.columns:
        df["revol_util"] = _trim_revol_util(df["revol_util"])
    df = _coerce_numeric_strings(df)
    keep = list(set(COMMON_FEATURES + ["grade", "sub_grade", "int_rate", "issue_d"]) & set(df.columns))
    keep_targets = [t for t in ["grade", "sub_grade", "int_rate"] if t in df.columns]
    df = df[keep].dropna(subset=keep_targets).reset_index(drop=True)
    return df


def _split(X, y, seed=RANDOM_STATE, stratify=None):
    return train_test_split(X, y, test_size=0.10, random_state=seed, stratify=stratify)


def train_aux_sg35(X, y_sg, seed=RANDOM_STATE):
    """35-class CatBoost classifier on sub_grade."""
    Xtr, Xva, ytr, yva = _split(X, y_sg, seed=seed, stratify=y_sg)
    Xtr_p, cat_idx = _prepare_x(Xtr)
    Xva_p, _ = _prepare_x(Xva)
    m = CatBoostClassifier(
        **{k: v for k, v in AUX_V3_SG35_PARAMS.items() if k != "early_stopping_rounds"},
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        early_stopping_rounds=AUX_V3_SG35_PARAMS["early_stopping_rounds"],
    )
    m.fit(Xtr_p, ytr, cat_features=cat_idx, eval_set=(Xva_p, yva), verbose=False)
    return m


def train_aux_issue_year(X, y_issue_year, seed=RANDOM_STATE):
    Xtr, Xva, ytr, yva = _split(X, y_issue_year, seed=seed)
    Xtr_p, cat_idx = _prepare_x(Xtr)
    Xva_p, _ = _prepare_x(Xva)
    m = CatBoostRegressor(
        **{k: v for k, v in AUX_V3_CB_PARAMS.items() if k != "early_stopping_rounds"},
        loss_function="RMSE", eval_metric="RMSE",
        random_seed=seed, verbose=0, allow_writing_files=False,
        early_stopping_rounds=AUX_V3_CB_PARAMS["early_stopping_rounds"],
    )
    m.fit(Xtr_p, ytr, cat_features=cat_idx, eval_set=(Xva_p, yva), verbose=False)
    return m


def train_aux_within_grade_residual(X, y_residual, seed=RANDOM_STATE):
    Xtr, Xva, ytr, yva = _split(X, y_residual, seed=seed)
    Xtr_p, cat_idx = _prepare_x(Xtr)
    Xva_p, _ = _prepare_x(Xva)
    m = CatBoostRegressor(
        **{k: v for k, v in AUX_V3_CB_PARAMS.items() if k != "early_stopping_rounds"},
        loss_function="RMSE", eval_metric="RMSE",
        random_seed=seed, verbose=0, allow_writing_files=False,
        early_stopping_rounds=AUX_V3_CB_PARAMS["early_stopping_rounds"],
    )
    m.fit(Xtr_p, ytr, cat_features=cat_idx, eval_set=(Xva_p, yva), verbose=False)
    return m


def train_v3_aux_for_source(source_name: str, source_df: pd.DataFrame,
                             grade_mean_rate_lookup: dict) -> dict:
    """Train 3 v3 aux models on one archive source. Returns dict with keys
    {sg35, issue_year, within_grade_residual}."""
    source_df = add_subgrade_ordinal(source_df)
    X, _ = split_X_y(source_df, target="int_rate")    # COMMON_FEATURES only

    print(f"  [{source_name}] training v3 aux models on {len(X):,} rows", flush=True)

    # --- 35-class subgrade (skippable when SKIP_SG35) ---
    if SKIP_SG35:
        print(f"    sg35 SKIPPED (config_v3.SKIP_SG35=True)", flush=True)
        m_sg35 = None
    else:
        t = time()
        y_sg = source_df["sub_grade"].astype(str)
        m_sg35 = train_aux_sg35(X, y_sg)
        print(f"    sg35 done ({time()-t:.0f}s)", flush=True)

    # --- Issue-year ---
    if "issue_d" in source_df.columns:
        # loan.csv uses "%b-%Y", LC_train.csv uses "%Y-%m-%d" - try both
        years = pd.to_datetime(source_df["issue_d"], format="%b-%Y", errors="coerce").dt.year
        fallback = pd.to_datetime(source_df["issue_d"], errors="coerce").dt.year
        years = years.fillna(fallback).astype("float64")
        valid = years.notna()
        if valid.sum() < 1000:
            print(f"    issue_year: not enough valid dates ({valid.sum()}), skipping", flush=True)
            m_issue = None
        elif years[valid].nunique() < 2:
            print(f"    issue_year: constant target ({years[valid].iloc[0]}), skipping", flush=True)
            m_issue = None
        else:
            t = time()
            m_issue = train_aux_issue_year(X[valid], years[valid].values)
            print(f"    issue_year done ({time()-t:.0f}s)", flush=True)
    else:
        print(f"    issue_year: no issue_d column, skipping", flush=True)
        m_issue = None

    # --- Within-grade residual = int_rate - grade_mean_rate[grade] ---
    t = time()
    grade = source_df["grade"].astype(str)
    grade_mean = grade.map(grade_mean_rate_lookup).fillna(np.nanmean(list(grade_mean_rate_lookup.values())))
    residual = source_df["int_rate"].astype("float64") - grade_mean
    m_residual = train_aux_within_grade_residual(X, residual.values)
    print(f"    within_grade_residual done ({time()-t:.0f}s)", flush=True)

    return {"sg35": m_sg35, "issue_year": m_issue, "within_grade_residual": m_residual}
