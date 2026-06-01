"""Load archive sources + align their columns to what `true data/` exposes.

Archive `loan.csv` is 1.2 GB / 145 cols; we only read the subset of columns
that overlap with true data + the 3 targets. That keeps memory ~600 MB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from reverse_engineer.config import (
    ARCHIVE_LC_TRAIN_CSV, ARCHIVE_LOAN_CSV, LOAN_CSV_MAX_ROWS,
    RANDOM_STATE, SUBGRADE_TO_ORD, TRUE_DATA_DIR,
)

# Cols that exist in BOTH true data/LC_train.csv and the archive sources.
# Computed once by intersecting column lists; hardcoded here to avoid IO at
# every call.
COMMON_FEATURES = [
    "addr_state", "annual_inc", "application_type",
    "chargeoff_within_12_mths", "collections_12_mths_ex_med",
    "delinq_2yrs", "dti", "emp_length", "emp_title", "home_ownership",
    "loan_amnt", "mo_sin_old_rev_tl_op", "mort_acc",
    "open_acc", "pub_rec", "pub_rec_bankruptcies", "purpose",
    "revol_bal", "revol_util", "term", "title", "total_acc",
    "verification_status", "zip_code",
]

# Targets we care about (only int_rate is in true data — the others are what
# we'll learn to predict).
AUX_TARGETS = ["grade", "sub_grade", "int_rate"]

NA_VALUES = ["NA", "n/a", "N/A", "null", ""]


def _coerce_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["dti", "revol_util", "annual_inc", "loan_amnt",
              "delinq_2yrs", "open_acc", "pub_rec", "total_acc",
              "mort_acc", "pub_rec_bankruptcies",
              "chargeoff_within_12_mths", "collections_12_mths_ex_med",
              "mo_sin_old_rev_tl_op"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _trim_revol_util(s: pd.Series) -> pd.Series:
    """`revol_util` in loan.csv has '%' suffix in some rows."""
    if s.dtype == object:
        return pd.to_numeric(s.astype(str).str.rstrip("%"), errors="coerce")
    return s


def load_archive_loan(max_rows: int | None = LOAN_CSV_MAX_ROWS) -> pd.DataFrame:
    """Load loan.csv with only the columns we actually need + targets."""
    use_cols = list(set(COMMON_FEATURES + AUX_TARGETS))
    print(f"  [loan.csv] reading up to {max_rows or 'all'} rows, "
          f"{len(use_cols)} cols", flush=True)
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
    # Drop rows missing the targets
    df = df.dropna(subset=AUX_TARGETS).reset_index(drop=True)
    # Sample for diversity rather than head-only
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def load_archive_lc_train() -> pd.DataFrame:
    df = pd.read_csv(ARCHIVE_LC_TRAIN_CSV, na_values=NA_VALUES, low_memory=False)
    if "revol_util" in df.columns:
        df["revol_util"] = _trim_revol_util(df["revol_util"])
    df = _coerce_numeric_strings(df)
    keep = list(set(COMMON_FEATURES + AUX_TARGETS) & set(df.columns))
    df = df[keep].dropna(subset=[t for t in AUX_TARGETS if t in df.columns]).reset_index(drop=True)
    return df


def add_subgrade_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    if "sub_grade" in df.columns:
        df = df.copy()
        df["sub_grade_ord"] = df["sub_grade"].map(SUBGRADE_TO_ORD).astype("float64")
    return df


def split_X_y(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where X has ONLY the COMMON_FEATURES (intersected with df)."""
    X_cols = [c for c in COMMON_FEATURES if c in df.columns]
    return df[X_cols].copy(), df[target].copy()


def load_true_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return (X_train, X_test, y_train, test_ids) where the X frames are the
    raw `true data/` features (still containing FICO etc. — the final-model
    side will use those; aux models only get COMMON_FEATURES subset)."""
    train = pd.read_csv(TRUE_DATA_DIR / "LC_train.csv", na_values=NA_VALUES, low_memory=False)
    test = pd.read_csv(TRUE_DATA_DIR / "LC_test.csv", na_values=NA_VALUES, low_memory=False)
    if "loan_status" in train.columns:
        train = train.drop(columns=["loan_status"])
    if "loan_status" in test.columns:
        test = test.drop(columns=["loan_status"])
    y = train["int_rate"].astype("float64").values
    ids = test["ID"].copy() if "ID" in test.columns else pd.Series(np.arange(len(test)))
    X_train = train.drop(columns=["int_rate"])
    X_test = test.drop(columns=["ID"]) if "ID" in test.columns else test
    return X_train, X_test, pd.Series(y, name="int_rate"), ids
