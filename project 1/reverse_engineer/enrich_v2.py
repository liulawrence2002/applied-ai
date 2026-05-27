"""V2 enrichment: cross the aux predictions with FICO + other true-data signals.

Run AFTER `production.features.engineer()` has added the 135 base engineered
features and AFTER the aux pipeline has added the 14 aux columns. This adds
~30 more interaction columns that capture the joint information of
(predicted grade/sub_grade/int_rate) × (FICO / dti / loan amount).

Why this is worth doing: tree models already learn interactions implicitly,
but cross-features sharpen the gradient signal and help the linear/NN
base learners exploit the aux columns directly. Each aux × FICO interaction
costs ~50 KB of memory and 0 training time (computed once).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRADE_LETTERS = ["A", "B", "C", "D", "E", "F", "G"]


def add_fico_aux_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Append FICO × aux interaction columns. Expects the enriched DataFrame
    that already has both `fico` (from production.features.engineer) and the
    aux_* columns (from reverse_engineer.enrich.enrich_with_aux)."""
    out = df.copy()

    # Sanity: skip silently if required columns missing
    if "fico" not in out.columns or "aux_grade_argmax" not in out.columns:
        return out

    fico = out["fico"].clip(lower=300, upper=850)
    fico_band = out["fico_band"] if "fico_band" in out.columns else None
    fico_band_fine = out["fico_band_fine"] if "fico_band_fine" in out.columns else None

    # ----- Predicted-grade × FICO crosses -----
    out["fico_x_aux_grade_argmax"] = fico * out["aux_grade_argmax"]
    out["fico_x_aux_subgrade_ord_avg"] = fico * out["aux_subgrade_ord_avg"]
    out["fico_inv_x_aux_subgrade_ord_avg"] = (1000.0 / fico) * out["aux_subgrade_ord_avg"]
    out["fico_x_aux_int_rate_avg"] = fico * out["aux_int_rate_avg"]
    out["fico_inv_x_aux_int_rate_avg"] = (1000.0 / fico) * out["aux_int_rate_avg"]
    if fico_band is not None:
        out["fico_band_x_aux_grade_argmax"] = fico_band.astype("float64") * out["aux_grade_argmax"]
    if fico_band_fine is not None:
        out["fico_band_fine_x_aux_subgrade_ord_avg"] = (
            fico_band_fine.astype("float64") * out["aux_subgrade_ord_avg"]
        )

    # ----- Grade-probability × FICO (per-grade) -----
    # If grade prob A is high but FICO is low → mispricing signal
    for g in GRADE_LETTERS:
        col = f"aux_grade_prob_{g}"
        if col in out.columns:
            out[f"fico_x_{col}"] = fico * out[col]

    # Expected FICO under predicted grade: weighted FICO by grade probabilities
    # Treat grade letters as ordinal 1..7; this approximates "what FICO band
    # does the model's grade distribution imply"
    grade_ords = np.arange(1, 8, dtype=np.float64)  # 1..7
    grade_prob_matrix = np.column_stack([
        out[f"aux_grade_prob_{g}"].values for g in GRADE_LETTERS
        if f"aux_grade_prob_{g}" in out.columns
    ])
    if grade_prob_matrix.shape[1] == 7:
        out["aux_grade_entropy"] = -np.sum(
            grade_prob_matrix * np.log(np.clip(grade_prob_matrix, 1e-9, 1.0)),
            axis=1,
        )
        out["aux_grade_max_prob"] = grade_prob_matrix.max(axis=1)
        out["aux_grade_top2_gap"] = (
            np.sort(grade_prob_matrix, axis=1)[:, -1]
            - np.sort(grade_prob_matrix, axis=1)[:, -2]
        )
        out["aux_grade_expected_ord"] = grade_prob_matrix @ grade_ords
        out["fico_x_aux_grade_expected_ord"] = fico * out["aux_grade_expected_ord"]

    # ----- Discrepancy features -----
    if fico_band is not None:
        out["aux_argmax_minus_fico_band"] = (
            out["aux_grade_argmax"] - fico_band.astype("float64")
        )
        # Absolute mispricing signal
        out["abs_aux_argmax_minus_fico_band"] = np.abs(out["aux_argmax_minus_fico_band"])

    # ----- Aux × DTI / loan_amnt -----
    if "dti" in out.columns:
        out["aux_subgrade_ord_x_dti"] = out["aux_subgrade_ord_avg"] * out["dti"].fillna(out["dti"].median())
        out["aux_int_rate_x_dti"] = out["aux_int_rate_avg"] * out["dti"].fillna(out["dti"].median())
    if "loan_amnt" in out.columns:
        out["aux_subgrade_ord_x_loan_amnt"] = out["aux_subgrade_ord_avg"] * out["loan_amnt"]
        out["aux_int_rate_x_loan_amnt"] = out["aux_int_rate_avg"] * out["loan_amnt"]

    # ----- Cross-source agreement / disagreement -----
    if "aux_subgrade_ord_loan" in out.columns and "aux_subgrade_ord_lct" in out.columns:
        out["aux_subgrade_source_disagreement"] = np.abs(
            out["aux_subgrade_ord_loan"] - out["aux_subgrade_ord_lct"]
        )
    if "aux_int_rate_loan" in out.columns and "aux_int_rate_lct" in out.columns:
        out["aux_int_rate_source_disagreement"] = np.abs(
            out["aux_int_rate_loan"] - out["aux_int_rate_lct"]
        )

    # ----- "Mispricing" residual: aux says X, the rate-from-fico would say Y -----
    # Crude estimate of FICO-implied rate: a linear inverse FICO mapping
    # (calibrated roughly from typical LendingClub rate ranges).
    # 850 FICO → ~6%, 600 FICO → ~20%; linear in (850-fico)
    fico_implied_rate = 6.0 + 0.056 * (850 - fico)
    out["aux_minus_fico_implied_rate"] = out["aux_int_rate_avg"] - fico_implied_rate
    out["abs_aux_minus_fico_implied_rate"] = np.abs(out["aux_minus_fico_implied_rate"])

    # ----- Composite risk score reusing aux + true data signals -----
    out["composite_risk_v2"] = (
        - 0.20 * (fico / 850.0)
        + 0.25 * (out["aux_subgrade_ord_avg"] / 35.0)
        + 0.15 * (out["aux_int_rate_avg"] / 30.0)
        + 0.15 * (out["aux_grade_argmax"] / 7.0)
        + 0.10 * (out["dti"].fillna(out["dti"].median()) / 100.0 if "dti" in out.columns else 0)
        + 0.10 * (out["revol_util"].fillna(out["revol_util"].median()) / 100.0 if "revol_util" in out.columns else 0)
        + 0.05 * (out["term_months"].astype("float64") / 60.0 if "term_months" in out.columns else 0)
    )

    return out
