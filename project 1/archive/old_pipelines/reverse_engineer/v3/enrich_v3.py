"""V3 enrichment: apply 35-class subgrade clf, issue_year aux, within-grade
residual aux to true data, plus add rate-lookup-derived features.

Output cols (~47):
  - aux_sg35_A1 ... aux_sg35_G5             (35 probability cols)
  - aux_sg35_argmax_ord, _entropy, _max_prob, _top2_gap   (4)
  - aux_lookup_expected_rate                            (1, prob-weighted sum from rate-card)
  - aux_lookup_rate_uncertainty                         (1, prob-weighted stddev)
  - aux_lookup_expected_rate_x_term                     (1, expected_rate × term_months)
  - aux_issue_year_avg / _loan / _lct                   (3)
  - aux_within_grade_residual                           (1, averaged across sources)
  - aux_grade_mean_rate_lookup                          (1, from predicted hard grade)
  - aux_reconstructed_rate                              (1, grade_mean + residual)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from reverse_engineer.aux_models import _prepare_x
from reverse_engineer.config import SUBGRADES

GRADE_LETTERS = ["A", "B", "C", "D", "E", "F", "G"]


def _apply_sg35(true_df: pd.DataFrame, sg35_model) -> np.ndarray:
    """Return (n, 35) array of probabilities in canonical SUBGRADES order."""
    train_feature_names = list(sg35_model.feature_names_)
    X = true_df.copy()
    for c in train_feature_names:
        if c not in X.columns:
            X[c] = np.nan
    X = X[train_feature_names].copy()
    X_p, _ = _prepare_x(X)
    proba = sg35_model.predict_proba(X_p)
    classes = list(sg35_model.classes_)
    # Re-order to canonical A1..G5 if needed
    if classes != SUBGRADES:
        idx = [classes.index(sg) if sg in classes else -1 for sg in SUBGRADES]
        ordered = np.zeros((proba.shape[0], len(SUBGRADES)), dtype=np.float64)
        for j, src in enumerate(idx):
            if src >= 0:
                ordered[:, j] = proba[:, src]
        proba = ordered
    return proba


def _apply_simple(true_df: pd.DataFrame, model) -> np.ndarray | None:
    if model is None:
        return None
    train_feature_names = list(model.feature_names_)
    X = true_df.copy()
    for c in train_feature_names:
        if c not in X.columns:
            X[c] = np.nan
    X = X[train_feature_names].copy()
    X_p, _ = _prepare_x(X)
    return model.predict(X_p)


def enrich_v3(true_df: pd.DataFrame, aux_v3_per_source: dict,
              rate_lookup: dict) -> pd.DataFrame:
    """
    true_df: a true-data frame (already enriched with v2 columns is fine — we
        just append v3 columns).
    aux_v3_per_source: {"loan": {sg35, issue_year, within_grade_residual},
                       "lct":  {sg35, issue_year, within_grade_residual}}
        sg35 may be None when SKIP_SG35=True; we then fall back to the
        existing v2 7-class grade probabilities broadcast over 5 subgrades
        each (uniform within each grade).
    rate_lookup: dict from rate_lookup.build_lookup()
    """
    out = true_df.copy()
    n = len(true_df)

    # --- 35-class sg probabilities, averaged across sources ---
    has_sg35 = any(models.get("sg35") is not None for models in aux_v3_per_source.values())
    if has_sg35:
        sg35_per_source = {tag: _apply_sg35(true_df, models["sg35"])
                           for tag, models in aux_v3_per_source.items()
                           if models.get("sg35") is not None}
        sg35_avg = np.mean(list(sg35_per_source.values()), axis=0)  # (n, 35)
    else:
        # Fallback: broadcast v2's 7-class grade probabilities to 35 columns
        # (uniform within each grade — each sub_grade in grade X gets prob/5)
        sg35_avg = np.zeros((n, 35), dtype=np.float64)
        for gi, g in enumerate(GRADE_LETTERS):
            col = f"aux_grade_prob_{g}"
            if col in true_df.columns:
                # 5 subgrades per grade
                sg35_avg[:, gi * 5:(gi + 1) * 5] = (
                    true_df[col].values.reshape(-1, 1) / 5.0
                )
        # If even the v2 grade probs are missing, fall back to a uniform prior
        row_sums = sg35_avg.sum(axis=1, keepdims=True)
        zero_rows = (row_sums.ravel() == 0)
        if zero_rows.any():
            sg35_avg[zero_rows, :] = 1.0 / 35
    for j, sg in enumerate(SUBGRADES):
        out[f"aux_sg35_{sg}"] = sg35_avg[:, j]

    # Summary stats over the 35-prob distribution
    out["aux_sg35_argmax_ord"] = (sg35_avg.argmax(axis=1) + 1).astype("float64")
    out["aux_sg35_max_prob"] = sg35_avg.max(axis=1)
    sorted_probs = np.sort(sg35_avg, axis=1)
    out["aux_sg35_top2_gap"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    out["aux_sg35_entropy"] = -np.sum(
        sg35_avg * np.log(np.clip(sg35_avg, 1e-9, 1.0)), axis=1,
    )

    # --- Rate-card lookup features ---
    # Build per-column vectors of mean and std rate aligned to SUBGRADES order
    mean_rate_vec = np.array([
        rate_lookup["subgrade_mean_rate"].get(sg, rate_lookup["global_mean_rate"])
        for sg in SUBGRADES
    ])
    std_rate_vec = np.array([
        rate_lookup["subgrade_std_rate"].get(sg, 0.0) for sg in SUBGRADES
    ])
    # Probability-weighted expected rate
    out["aux_lookup_expected_rate"] = sg35_avg @ mean_rate_vec
    # Probability-weighted std (uncertainty proxy)
    # Var = E[X^2] - (E[X])^2
    e_x2 = sg35_avg @ (mean_rate_vec ** 2 + std_rate_vec ** 2)
    e_x = out["aux_lookup_expected_rate"].values
    out["aux_lookup_rate_uncertainty"] = np.sqrt(np.maximum(e_x2 - e_x ** 2, 0.0))

    if "term_months" in out.columns:
        out["aux_lookup_expected_rate_x_term"] = (
            out["aux_lookup_expected_rate"] * out["term_months"].astype("float64")
        )

    # --- Issue-year aux ---
    iy_per = {}
    for tag, models in aux_v3_per_source.items():
        y = _apply_simple(true_df, models.get("issue_year"))
        if y is not None:
            iy_per[tag] = y
    if iy_per:
        iy_avg = np.mean(list(iy_per.values()), axis=0)
        out["aux_issue_year_avg"] = iy_avg
        for tag, y in iy_per.items():
            out[f"aux_issue_year_{tag}"] = y

    # --- Within-grade residual ---
    res_per = {tag: _apply_simple(true_df, models["within_grade_residual"])
               for tag, models in aux_v3_per_source.items()}
    res_avg = np.mean([r for r in res_per.values() if r is not None], axis=0)
    out["aux_within_grade_residual"] = res_avg

    # Grade-mean rate looked up from predicted hard grade (from existing aux_grade_argmax
    # if present in true_df, else from sg35_argmax-derived grade)
    if "aux_grade_argmax" in true_df.columns:
        grade_ord = true_df["aux_grade_argmax"].astype("Int64").astype(int)
    else:
        grade_ord = ((out["aux_sg35_argmax_ord"].values - 1) // 5).astype(int) + 1
        grade_ord = pd.Series(grade_ord, index=out.index)
    # Map 1..7 -> A..G then look up mean rate
    grade_letter_lookup = {i + 1: g for i, g in enumerate(GRADE_LETTERS)}
    grade_str = grade_ord.map(grade_letter_lookup)
    out["aux_grade_mean_rate_lookup"] = grade_str.map(
        rate_lookup["grade_mean_rate"]
    ).astype("float64")
    out["aux_reconstructed_rate"] = out["aux_grade_mean_rate_lookup"] + out["aux_within_grade_residual"]

    return out
