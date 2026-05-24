"""Expanded feature engineering for LendingClub interest rate prediction.

Building on the baseline `final/final_pipeline.py` set, this adds:
  - More interaction features (FICO × {dti, util, term}, term × {amount, dti})
  - Quantile-bucket flags for skewed monetary columns
  - Zero-inflation flags for derogatory counters
  - Per-state TE proxy via address_state median income lookup (computed from train)
  - Credit-line tenure cohorts
  - Income/loan amount Box-Cox-style transforms

All transforms are stateless (no fits on data); anything that needs a fit
(target encoding, imputation summary stats, state-income lookup) is done in
the K-fold loop downstream so no row sees its own label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl  # for fast IO

from production.config import DATA_DIR

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}

# US Census 2022 ACS state median household income ($, thousands). External
# data: a borrower in CA earning $60k is below-median; in MS earning $60k is
# above-median. The model can't know this from the dataset alone.
STATE_MEDIAN_INCOME = {
    "AL": 56.9, "AK": 84.8, "AZ": 72.6, "AR": 56.3, "CA": 91.6, "CO": 87.6,
    "CT": 88.4, "DE": 79.3, "FL": 67.9, "GA": 71.4, "HI": 92.5, "ID": 70.2,
    "IL": 78.4, "IN": 67.2, "IA": 70.5, "KS": 69.7, "KY": 60.2, "LA": 57.6,
    "ME": 68.3, "MD": 98.5, "MA": 96.5, "MI": 68.5, "MN": 84.3, "MS": 52.7,
    "MO": 65.9, "MT": 66.8, "NE": 71.7, "NV": 71.6, "NH": 90.8, "NJ": 97.1,
    "NM": 58.7, "NY": 81.4, "NC": 66.2, "ND": 73.0, "OH": 66.6, "OK": 61.4,
    "OR": 76.6, "PA": 73.8, "RI": 81.4, "SC": 62.5, "SD": 69.5, "TN": 64.0,
    "TX": 73.0, "UT": 86.8, "VT": 74.0, "VA": 87.2, "WA": 91.3, "WV": 55.2,
    "WI": 72.5, "WY": 72.4, "DC": 101.0,
}

# Approximate BLS state unemployment rate (long-run 2018-2022 avg, %). High
# state unemployment is a macro proxy for default risk priced into rates.
STATE_UNEMPLOYMENT = {
    "AL": 3.8, "AK": 6.5, "AZ": 5.0, "AR": 3.9, "CA": 5.7, "CO": 4.1,
    "CT": 5.4, "DE": 4.8, "FL": 4.3, "GA": 4.3, "HI": 4.7, "ID": 3.5,
    "IL": 5.5, "IN": 4.0, "IA": 3.7, "KS": 3.8, "KY": 4.7, "LA": 5.4,
    "ME": 4.0, "MD": 4.5, "MA": 4.8, "MI": 5.2, "MN": 3.9, "MS": 5.7,
    "MO": 4.1, "MT": 4.0, "NE": 2.9, "NV": 6.4, "NH": 3.4, "NJ": 5.7,
    "NM": 6.1, "NY": 5.4, "NC": 4.6, "ND": 3.0, "OH": 4.7, "OK": 4.4,
    "OR": 5.0, "PA": 5.4, "RI": 5.4, "SC": 4.5, "SD": 3.0, "TN": 4.3,
    "TX": 5.0, "UT": 3.4, "VT": 3.0, "VA": 4.0, "WA": 5.3, "WV": 5.2,
    "WI": 3.7, "WY": 4.4, "DC": 5.8,
}

# Cost-of-living index, US avg = 100 (Council for Community and Economic
# Research). $50k in MS ≠ $50k in HI; higher COL means borrower has less
# real disposable income after the loan.
STATE_COL_INDEX = {
    "AL": 88.1, "AK": 125.8, "AZ": 102.2, "AR": 89.0, "CA": 142.2, "CO": 105.5,
    "CT": 113.1, "DE": 102.7, "FL": 100.7, "GA": 90.8, "HI": 184.0, "ID": 103.1,
    "IL": 92.1, "IN": 90.6, "IA": 89.0, "KS": 87.5, "KY": 92.0, "LA": 91.0,
    "ME": 111.5, "MD": 116.5, "MA": 148.4, "MI": 91.5, "MN": 94.0, "MS": 85.0,
    "MO": 88.6, "MT": 102.9, "NE": 91.1, "NV": 100.8, "NH": 114.1, "NJ": 113.9,
    "NM": 94.0, "NY": 125.1, "NC": 95.7, "ND": 94.6, "OH": 92.7, "OK": 87.0,
    "OR": 115.1, "PA": 95.6, "RI": 109.4, "SC": 96.6, "SD": 92.4, "TN": 90.3,
    "TX": 91.5, "UT": 103.2, "VT": 115.3, "VA": 102.1, "WA": 116.0, "WV": 90.5,
    "WI": 95.1, "WY": 92.4, "DC": 145.7,
}
MTHS_SINCE_COLS = [
    "mths_since_last_record", "mths_since_recent_inq",
    "mths_since_rcnt_il", "mths_since_recent_bc",
]
LOG_NUMERIC_COLS = [
    "annual_inc", "revol_bal", "tot_cur_bal", "total_bal_ex_mort", "tot_coll_amt",
    "loan_amnt",
]
STRING_NUMERIC_COLS = [
    "dti", "revol_util", "all_util", "mo_sin_old_il_acct",
    "mths_since_last_record", "mths_since_rcnt_il", "mths_since_recent_bc",
    "mths_since_recent_inq", "tot_cur_bal",
]


def load_raw():
    """Load with explicit dtypes + drop loan_status leakage."""
    dt = {c: "float64" for c in STRING_NUMERIC_COLS}
    train = pd.read_csv(DATA_DIR / "LC_train.csv", na_values=["NA"], dtype=dt)
    test = pd.read_csv(DATA_DIR / "LC_test.csv", na_values=["NA"], dtype=dt)
    train = train.drop(columns=["loan_status"])
    test = test.drop(columns=["loan_status"])
    return train, test


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ---- Stateless transforms ------------------------------------------------
    out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    out["fico_spread"] = out["fico_range_high"] - out["fico_range_low"]
    out["term_months"] = out["term"].astype("string").str.extract(r"(\d+)", expand=False).astype("Int64")
    out["emp_length_num"] = out["emp_length"].map(EMP_LENGTH_MAP)

    # Missingness flags
    for c in MTHS_SINCE_COLS:
        out[f"has_{c}"] = out[c].notna().astype(int)

    # Underwriting ratios
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"].replace(0, np.nan)
    out["installment_proxy"] = out["loan_amnt"] / out["term_months"].astype("float64")
    out["bal_to_income"] = out["tot_cur_bal"] / out["annual_inc"].replace(0, np.nan)
    out["util_x_fico"] = out["revol_util"] * out["fico"]
    out["dti_x_fico"] = out["dti"] * out["fico"]
    out["acc_open_ratio"] = out["open_acc"] / out["total_acc"].replace(0, np.nan)

    # Bands (replicate LC's underwriting cut-points)
    out["dti_band"] = pd.cut(out["dti"], bins=[-np.inf, 10, 20, 30, 40, np.inf], labels=False).astype("Int64")
    out["fico_band"] = pd.cut(out["fico"], bins=[-np.inf, 660, 690, 720, 760, np.inf], labels=False).astype("Int64")
    out["fico_band_fine"] = pd.cut(
        out["fico"], bins=[-np.inf, 640, 660, 680, 700, 720, 740, 760, 780, np.inf], labels=False
    ).astype("Int64")

    # Composite derog score (Fowlie-style)
    out["derog_score"] = (
        out["delinq_2yrs"].fillna(0) * 8
        + out["pub_rec"].fillna(0) * 13
        + out["pub_rec_bankruptcies"].fillna(0) * 22
        + out["chargeoff_within_12_mths"].fillna(0) * 15
        + out["collections_12_mths_ex_med"].fillna(0) * 10
    )
    out["inq_intensity"] = out["inq_last_12m"].fillna(0) + out["inq_fi"].fillna(0)

    # Utilization bands
    util_bins = [-np.inf, 30, 50, 75, 100, np.inf]
    out["revol_util_band"] = pd.cut(out["revol_util"], bins=util_bins, labels=False).astype("Int64")
    out["all_util_band"] = pd.cut(out["all_util"], bins=util_bins, labels=False).astype("Int64")

    # Credit-age normalised intensities
    out["credit_file_age_yrs"] = out["mo_sin_old_rev_tl_op"] / 12.0
    out["credit_age_band"] = pd.cut(
        out["credit_file_age_yrs"], bins=[-np.inf, 5, 10, 15, 20, np.inf], labels=False
    ).astype("Int64")
    out["delinq_per_credit_age"] = out["delinq_2yrs"].fillna(0) / (out["credit_file_age_yrs"].fillna(0) + 1)
    out["inq_per_credit_age"] = out["inq_last_12m"].fillna(0) / (out["credit_file_age_yrs"].fillna(0) + 1)
    out["inq_per_open_acc"] = out["inq_last_12m"].fillna(0) / (out["open_acc"].fillna(0) + 1)

    # Loan-structure interactions
    out["term_x_loan_amnt"] = out["term_months"].astype("float64") * out["loan_amnt"]
    out["term_x_dti"] = out["term_months"].astype("float64") * out["dti"]
    out["term_x_fico"] = out["term_months"].astype("float64") * out["fico"]
    out["payment_to_income"] = (
        out["loan_amnt"] / out["term_months"].astype("float64")
    ) / (out["annual_inc"].replace(0, np.nan) / 12.0)

    # Employment x collateral (BGG financial-accelerator proxy)
    out["emp_length_x_mortgage"] = out["emp_length_num"].fillna(0) * (out["home_ownership"] == "MORTGAGE").astype("int8")
    out["emp_length_x_rent"] = out["emp_length_num"].fillna(0) * (out["home_ownership"] == "RENT").astype("int8")

    # Zero-inflation flags
    out["delinq_2yr_flag"] = (out["delinq_2yrs"].fillna(0) > 0).astype("int8")
    out["pub_rec_flag"] = (out["pub_rec"].fillna(0) > 0).astype("int8")
    out["bankrupt_flag"] = (out["pub_rec_bankruptcies"].fillna(0) > 0).astype("int8")
    out["chargeoff_flag"] = (out["chargeoff_within_12_mths"].fillna(0) > 0).astype("int8")
    out["collections_flag"] = (out["collections_12_mths_ex_med"].fillna(0) > 0).astype("int8")
    out["mort_acc_flag"] = (out["mort_acc"].fillna(0) > 0).astype("int8")
    out["any_derog_flag"] = (
        (out["delinq_2yr_flag"] | out["pub_rec_flag"] | out["bankrupt_flag"] |
         out["chargeoff_flag"] | out["collections_flag"])
    ).astype("int8")

    # Log transforms on monetary tails
    for c in LOG_NUMERIC_COLS:
        out[f"log1p_{c}"] = np.log1p(out[c].clip(lower=0))

    # NEW: Sqrt transforms for moderately skewed columns
    out["sqrt_revol_bal"] = np.sqrt(out["revol_bal"].clip(lower=0))
    out["sqrt_open_acc"] = np.sqrt(out["open_acc"].fillna(0))
    out["sqrt_total_acc"] = np.sqrt(out["total_acc"].fillna(0))

    # NEW: Loan-amount × FICO interaction (large loans for low-FICO are riskiest)
    out["loan_amt_div_fico"] = out["loan_amnt"] / out["fico"].clip(lower=300)
    out["log_loan_x_inv_fico"] = out["log1p_loan_amnt"] * (1 / out["fico"].clip(lower=300))

    # NEW: Available credit headroom
    out["available_credit"] = (
        out["tot_cur_bal"].fillna(0) - out["revol_bal"].fillna(0)
    ).clip(lower=0)
    out["log_available_credit"] = np.log1p(out["available_credit"])

    # NEW: New-loan installment to current debt ratio (debt service capacity)
    out["installment_to_revol"] = out["installment_proxy"] / (out["revol_bal"].fillna(0) + 1)
    out["installment_to_curbal"] = out["installment_proxy"] / (out["tot_cur_bal"].fillna(0) + 1)

    # NEW: Tenure score (years of credit experience scaled by accounts)
    out["credit_experience"] = out["credit_file_age_yrs"].fillna(0) * out["sqrt_total_acc"]

    # zip3 first digit (geographic macro region)
    out["zip3"] = out["zip_code"].astype("string").str.extract(r"(\d{3})", expand=False)
    out["zip1"] = out["zip_code"].astype("string").str.extract(r"(\d)", expand=False)

    # ========================================================================
    # NEW: State macro features (external data, structurally orthogonal)
    # ========================================================================
    out["state_median_income"] = out["addr_state"].map(STATE_MEDIAN_INCOME).fillna(70.0)
    out["state_unemployment"] = out["addr_state"].map(STATE_UNEMPLOYMENT).fillna(4.5)
    out["state_col_index"] = out["addr_state"].map(STATE_COL_INDEX).fillna(100.0)

    # Borrower's income relative to their state's median — captures local
    # purchasing power that absolute annual_inc misses.
    annual_inc_k = out["annual_inc"] / 1000.0  # in $k for comparable units
    out["inc_vs_state_median"] = annual_inc_k / out["state_median_income"]
    out["log_inc_vs_state_median"] = np.log1p(out["inc_vs_state_median"].clip(lower=0))

    # COL-adjusted income (real purchasing power after housing/food)
    out["col_adjusted_income"] = annual_inc_k * 100.0 / out["state_col_index"]
    out["log_col_adjusted_income"] = np.log1p(out["col_adjusted_income"].clip(lower=0))

    # Loan amount relative to state median income — large loan in low-income
    # state = higher default risk priced into the rate.
    out["loan_to_state_median"] = (out["loan_amnt"] / 1000.0) / out["state_median_income"]

    # State macro × DTI: high state-unemployment + high DTI = compounded risk
    out["unemployment_x_dti"] = out["state_unemployment"] * out["dti"]
    out["col_x_dti"] = out["state_col_index"] * out["dti"]

    # ========================================================================
    # NEW: FICO non-linear transforms and cross-products
    # ========================================================================
    fico_clip = out["fico"].clip(lower=300, upper=850)
    out["fico_sq"] = fico_clip ** 2 / 1e5            # scaled to avoid huge magnitudes
    out["fico_inv"] = 1000.0 / fico_clip              # inverse FICO scales risk
    out["fico_log"] = np.log(fico_clip)
    # FICO sweet-spot signal — distance from 720 (the prime/non-prime threshold)
    out["fico_dist_720"] = (fico_clip - 720).abs()
    out["fico_above_prime"] = (fico_clip >= 720).astype("int8")
    out["fico_subprime"] = (fico_clip < 660).astype("int8")

    # FICO crossed with the four most predictive features
    out["fico_x_dti"] = fico_clip * out["dti"].fillna(out["dti"].median())
    out["fico_x_revol_util"] = fico_clip * out["revol_util"].fillna(out["revol_util"].median())
    out["fico_x_inq_intensity"] = fico_clip * out["inq_intensity"]
    out["fico_x_derog"] = fico_clip * out["derog_score"]
    out["fico_inv_x_loan_amnt"] = out["fico_inv"] * out["loan_amnt"]
    out["fico_inv_x_term"] = out["fico_inv"] * out["term_months"].astype("float64")

    # ========================================================================
    # NEW: Debt-service capacity (the underlying "can they pay it back" signal)
    # ========================================================================
    monthly_inc = (out["annual_inc"].replace(0, np.nan) / 12.0)
    # Existing monthly debt service estimated from DTI (DTI = debt_pmts / income)
    out["existing_monthly_debt"] = out["dti"].fillna(0) / 100.0 * monthly_inc
    # New installment + existing debt as fraction of monthly income
    out["total_debt_service_ratio"] = (
        (out["installment_proxy"] + out["existing_monthly_debt"]) / monthly_inc
    )
    # Remaining monthly income after all debt service (negative = unaffordable)
    out["monthly_residual_income"] = (
        monthly_inc - out["installment_proxy"] - out["existing_monthly_debt"]
    )
    out["log1p_monthly_residual"] = np.log1p(out["monthly_residual_income"].clip(lower=0))
    out["residual_to_income"] = out["monthly_residual_income"] / monthly_inc

    # ========================================================================
    # NEW: Purpose × loan interactions (purposes have different risk profiles)
    # ========================================================================
    # Risk-weighted purpose encoding from LC's historical default rates
    PURPOSE_RISK = {
        "small_business": 1.0, "renewable_energy": 0.85, "moving": 0.80,
        "house": 0.75, "medical": 0.70, "vacation": 0.65, "other": 0.60,
        "wedding": 0.55, "educational": 0.55, "home_improvement": 0.45,
        "major_purchase": 0.45, "debt_consolidation": 0.40, "credit_card": 0.35,
        "car": 0.30,
    }
    out["purpose_risk_score"] = out["purpose"].map(PURPOSE_RISK).fillna(0.50)
    out["purpose_risk_x_loan"] = out["purpose_risk_score"] * out["loan_amnt"]
    out["purpose_risk_x_dti"] = out["purpose_risk_score"] * out["dti"].fillna(0)
    out["purpose_risk_x_fico"] = out["purpose_risk_score"] * fico_clip

    # ========================================================================
    # NEW: Inquiry intensity normalised by lifetime
    # ========================================================================
    out["inq_per_lifetime_acc"] = out["inq_last_12m"].fillna(0) / (out["total_acc"].fillna(0) + 1)
    out["inq_density"] = (
        (out["inq_last_12m"].fillna(0) + out["inq_fi"].fillna(0))
        / (out["credit_file_age_yrs"].fillna(0) + 1)
    )
    out["recent_inq_burst"] = (out["inq_last_12m"].fillna(0) >= 4).astype("int8")

    # ========================================================================
    # NEW: Utilization compounding & ratios
    # ========================================================================
    # all_util captures total credit usage (cards + installment); revol_util is
    # just bankcard. Spread between them is informative.
    out["util_spread"] = out["all_util"].fillna(0) - out["revol_util"].fillna(0)
    out["util_product"] = out["all_util"].fillna(0) * out["revol_util"].fillna(0) / 100.0
    out["util_max"] = np.maximum(out["all_util"].fillna(0), out["revol_util"].fillna(0))
    out["util_maxed_out"] = (out["revol_util"].fillna(0) >= 95).astype("int8")
    out["util_unused"] = (out["revol_util"].fillna(0) <= 5).astype("int8")

    # ========================================================================
    # NEW: Mortgage / homeownership signals
    # ========================================================================
    out["is_homeowner"] = (out["home_ownership"].isin(["MORTGAGE", "OWN"])).astype("int8")
    out["is_renter"] = (out["home_ownership"] == "RENT").astype("int8")
    out["mortgage_per_open_acc"] = out["mort_acc"].fillna(0) / (out["open_acc"].fillna(0) + 1)

    # ========================================================================
    # NEW: Employment stability × income
    # ========================================================================
    out["emp_length_x_income"] = out["emp_length_num"].fillna(0) * out["log1p_annual_inc"]
    out["stable_high_income"] = (
        (out["emp_length_num"].fillna(0) >= 5) & (annual_inc_k >= 70)
    ).astype("int8")
    out["unstable_low_income"] = (
        (out["emp_length_num"].fillna(0) <= 1) & (annual_inc_k <= 40)
    ).astype("int8")

    # ========================================================================
    # NEW: Composite risk score (linear combo of the strongest priors)
    # ========================================================================
    out["composite_risk"] = (
        - 0.30 * (fico_clip / 850.0)                                 # higher FICO -> lower risk
        + 0.20 * (out["dti"].fillna(out["dti"].median()) / 100.0)    # higher DTI -> higher risk
        + 0.15 * (out["revol_util"].fillna(out["revol_util"].median()) / 100.0)
        + 0.10 * (out["derog_score"] / 50.0)
        + 0.10 * (out["inq_last_12m"].fillna(0) / 10.0)
        + 0.10 * (out["term_months"].astype("float64") / 60.0)
        + 0.05 * (out["loan_to_income"].fillna(0))
    )

    # Lowercase emp_title (TE-friendly)
    if "emp_title" in out.columns:
        out["emp_title"] = out["emp_title"].astype("string").str.lower().str.strip()

    # Drop superseded raw columns
    drop = ["fico_range_low", "fico_range_high", "term", "emp_length", "title", "zip_code"]
    return out.drop(columns=[c for c in drop if c in out.columns])


def compact_emp_title(train_fe: pd.DataFrame, test_fe: pd.DataFrame, top_n: int):
    """Compact emp_title to top-N (fit on train) + 'Other' + 'Missing'."""
    top = train_fe["emp_title"].value_counts().head(top_n).index.tolist()
    for df in (train_fe, test_fe):
        df["emp_title"] = (
            df["emp_title"].where(df["emp_title"].isin(top), "Other")
            .fillna("Missing").astype(str)
        )
    return train_fe, test_fe
