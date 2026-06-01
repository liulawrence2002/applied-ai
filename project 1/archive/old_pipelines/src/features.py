"""V1 feature engineering for the LendingClub int_rate pipeline.

This is the proven winner from prior ablation work. Every block here is either
EDA-validated or a domain-motivated interaction that survived the v2 ablation.
Anything that lost in ablation (FICO nonlinear terms, categorical crosses,
economics proxies, target encoding) is intentionally absent — do not add them
back without fresh evidence.

The Preprocessor is fit fold-locally: medians and categorical level sets are
learned on the training fold only and applied to validation + test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data import (
    NOISE_TEXT_COLS,
    RAW_REPLACED_COLS,
    TARGET_COL,
    validate_no_leakage,
)

EMP_LENGTH_MAP = {
    "< 1 year": 0.0,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}

# Native categorical columns that XGBoost / CatBoost handle directly.
CATEGORICAL_COLS = [
    "addr_state",
    "application_type",
    "home_ownership",
    "purpose",
    "verification_status",
    "zip_code",
    "term_cat",
]

# mths_since_* columns where NaN means "no such event ever" — that null is
# semantically meaningful, so encode it as a flag before any imputation.
MISSING_FLAG_COLS = {
    "mths_since_last_record": "has_derog_record",
    "mths_since_recent_inq": "has_recent_inq",
    "mths_since_rcnt_il": "has_recent_il",
    "mths_since_recent_bc": "has_recent_bc",
}

# Heavily zero-inflated counts: a binary "ever > 0" flag captures most of the
# signal cleanly while the raw count stays available for magnitude.
ZERO_FLAG_COLS = [
    "pub_rec",
    "delinq_2yrs",
    "chargeoff_within_12_mths",
    "collections_12_mths_ex_med",
    "tot_coll_amt",
]

# Right-skewed monetary columns — log1p compresses the tail for tree splits.
LOG1P_COLS = [
    "annual_inc",
    "revol_bal",
    "tot_cur_bal",
    "total_bal_ex_mort",
    "tot_coll_amt",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def engineer_v1(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Pure function: raw columns -> v1 engineered feature matrix.

    Drops the target if present, raw forms that have engineered replacements,
    and free-text/duplicate columns. Does NOT impute or encode categoricals —
    those are stateful and live in the Preprocessor below.
    """
    df = raw_df.copy()

    # FICO midpoint — the strongest legitimate signal in this slice.
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0

    # Parse term ("36 months" -> 36) and keep both numeric and categorical form.
    df["term_months"] = (
        df["term"].astype("string").str.extract(r"(\d+)", expand=False).astype(float)
    )
    df["term_cat"] = df["term"].astype("string").fillna("missing")

    # Ordinal employment length.
    df["emp_length_num"] = df["emp_length"].map(EMP_LENGTH_MAP).astype(float)

    # Missingness flags: NaN means "no such event" for these credit-history fields.
    for source, flag in MISSING_FLAG_COLS.items():
        df[flag] = df[source].notna().astype(np.int8)

    # Zero-inflation flags for delinquency / public-record / collections counts.
    for source in ZERO_FLAG_COLS:
        df[f"{source}_flag"] = (df[source].fillna(0) > 0).astype(np.int8)

    # Log transforms for right-skewed monetary columns.
    for source in LOG1P_COLS:
        df[f"{source}_log"] = np.log1p(df[source].clip(lower=0))

    # Domain interactions — every one of these survived v2 ablation.
    df["loan_to_income"] = _safe_divide(df["loan_amnt"], df["annual_inc"] + 1.0)
    df["dti_x_log_income"] = df["dti"] * df["annual_inc_log"]
    df["fico_x_term"] = df["fico"] * df["term_months"]
    df["fico_x_revol_util"] = df["fico"] * df["revol_util"]
    df["fico_x_all_util"] = df["fico"] * df["all_util"]
    df["inq_per_open_acc"] = _safe_divide(df["inq_last_12m"], df["open_acc"] + 1.0)
    df["delinq_per_credit_age"] = _safe_divide(
        df["delinq_2yrs"], (df["mo_sin_old_rev_tl_op"] / 12.0) + 1.0
    )
    df["pub_rec_bankruptcy_combo"] = (
        df["pub_rec"].fillna(0) + df["pub_rec_bankruptcies"].fillna(0)
    )
    df["revol_util_x_open_acc"] = df["revol_util"] * df["open_acc"]
    df["loan_amnt_x_term"] = df["loan_amnt"] * df["term_months"]
    df["loan_amnt_x_revol_util"] = df["loan_amnt"] * (df["revol_util"] / 100.0)

    drop_cols = [c for c in (*RAW_REPLACED_COLS, *NOISE_TEXT_COLS, TARGET_COL) if c in df.columns]
    return df.drop(columns=drop_cols)


@dataclass
class Preprocessor:
    """Fold-local preprocessor: median imputation + frozen categorical levels.

    fit() on the training fold, then transform() on validation and test. This
    ensures medians and category sets reflect only data the model has seen.
    """

    numeric_medians: dict[str, float] = field(default_factory=dict)
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)

    def fit(self, raw_df: pd.DataFrame) -> "Preprocessor":
        features = engineer_v1(raw_df)
        for col in CATEGORICAL_COLS:
            observed = features[col].astype("string").fillna("missing")
            levels = sorted(v for v in observed.unique().tolist() if v != "missing")
            self.category_levels[col] = ["missing", *levels]
        for col in features.select_dtypes(include=[np.number]).columns:
            median = features[col].median()
            self.numeric_medians[col] = 0.0 if pd.isna(median) else float(median)
        self.feature_columns = features.columns.tolist()
        return self

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns:
            raise RuntimeError("Preprocessor must be fit before transform.")
        features = engineer_v1(raw_df)
        # Ensure column alignment.
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = np.nan
        features = features[self.feature_columns]
        # Median impute numerics.
        for col, median in self.numeric_medians.items():
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors="coerce").fillna(median)
        # Apply frozen categorical levels (unseen -> "missing").
        for col, levels in self.category_levels.items():
            as_string = features[col].astype("string").fillna("missing")
            as_string = as_string.where(as_string.isin(levels), "missing")
            features[col] = pd.Categorical(as_string, categories=levels)
        validate_no_leakage(features.columns.tolist())
        return features

    def fit_transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(raw_df).transform(raw_df)
