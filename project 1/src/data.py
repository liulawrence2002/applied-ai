"""Data loading and leakage constants for the LendingClub int_rate pipeline.

Single source of truth for paths, leakage rules, and basic IO. Everything
downstream imports from here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "true data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRAIN_PATH = DATA_DIR / "LC_train.csv"
TEST_PATH = DATA_DIR / "LC_test.csv"

TARGET_COL = "int_rate"
ID_COL = "ID"
RANDOM_STATE = 42
N_FOLDS = 5

# Columns that must NEVER reach the model. The first is outcome leakage in this
# slice; the rest are well-known post-origination columns from the broader
# LendingClub schema, listed defensively so a future schema change can't sneak
# them in. The hardship_ prefix is checked separately in validate_no_leakage.
LEAKAGE_COLS = (
    "loan_status",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "debt_settlement_flag",
    "hardship_flag",
    "next_pymnt_d",
    "grade",
    "sub_grade",
    "installment",
)
HARDSHIP_PREFIX = "hardship_"

# Raw columns that aren't leakage but are dropped after being transformed into
# engineered equivalents. Keeping the raw and engineered side-by-side would
# double-feed the model the same signal.
RAW_REPLACED_COLS = (
    "fico_range_low",
    "fico_range_high",
    "emp_length",
    "term",
)

# Free-text / one-to-one duplicates of other features — drop unconditionally.
NOISE_TEXT_COLS = (
    "emp_title",  # high-cardinality free text, no signal worth the variance
    "title",      # one-to-one duplicate of purpose in this slice
)

# Belt-and-suspenders: assert these never appear in the final feature matrix.
FORBIDDEN_GUARD = frozenset({TARGET_COL, ID_COL, *LEAKAGE_COLS, *RAW_REPLACED_COLS, *NOISE_TEXT_COLS})


def read_train() -> pd.DataFrame:
    """Read the canonical train CSV, drop outcome leakage immediately."""
    df = pd.read_csv(TRAIN_PATH, na_values=["NA"])
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TRAIN_PATH} must contain {TARGET_COL!r}.")
    return _drop_leakage(df)


def read_test() -> pd.DataFrame:
    """Read the canonical test CSV, drop outcome leakage immediately."""
    df = pd.read_csv(TEST_PATH, na_values=["NA"])
    if TARGET_COL in df.columns:
        raise ValueError(f"{TEST_PATH} must not contain {TARGET_COL!r}.")
    if ID_COL not in df.columns:
        raise ValueError(f"{TEST_PATH} must contain {ID_COL!r}.")
    if len(df) != 10_000:
        raise ValueError(f"Expected 10,000 test rows, found {len(df):,}.")
    if not df[ID_COL].is_unique:
        raise ValueError("Test ID column must be unique.")
    return _drop_leakage(df)


def _drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in df.columns if c in LEAKAGE_COLS or c.startswith(HARDSHIP_PREFIX)]
    return df.drop(columns=to_drop) if to_drop else df


def validate_no_leakage(feature_columns: list[str] | set[str]) -> None:
    """Raise if any forbidden or hardship_* column slipped into the feature matrix."""
    cols = set(feature_columns)
    forbidden = cols & FORBIDDEN_GUARD
    hardship = {c for c in cols if c.startswith(HARDSHIP_PREFIX)}
    bad = forbidden | hardship
    if bad:
        raise ValueError(f"Forbidden feature(s) present in matrix: {sorted(bad)}")
