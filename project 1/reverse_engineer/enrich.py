"""Apply the trained aux models to true data → 14 new feature columns.

Output columns per call:
  - aux_grade_prob_A ... aux_grade_prob_G       (7, averaged across sources)
  - aux_subgrade_ord_avg / _loan / _lct         (3)
  - aux_int_rate_avg / _loan / _lct             (3)
  - aux_grade_argmax                            (1, hard predicted grade ordinal)
                                                Total = 14
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from reverse_engineer.archive_loader import COMMON_FEATURES
from reverse_engineer.aux_models import _prepare_x

GRADE_CLASSES = ["A", "B", "C", "D", "E", "F", "G"]
GRADE_TO_ORD = {g: i + 1 for i, g in enumerate(GRADE_CLASSES)}


def _apply_models_one_source(true_df: pd.DataFrame, source_models: dict,
                              source_tag: str) -> dict:
    """Predict on true_df with the 3 models from one source. Returns a dict of
    arrays keyed by ('grade_prob' -> (n,7), 'sub_grade_ord' -> (n,), 'int_rate' -> (n,)).

    Critical: the CatBoost model stores its feature names from training. We
    reindex `true_df` to that same column order before prediction so
    cat-vs-numeric position indices line up.
    """
    train_feature_names = list(source_models["grade"].feature_names_)
    X = true_df.copy()
    # Add any missing columns as NaN (will be imputed by _prepare_x)
    for c in train_feature_names:
        if c not in X.columns:
            X[c] = np.nan
    X = X[train_feature_names].copy()
    X_p, _ = _prepare_x(X)

    grade_proba = source_models["grade"].predict_proba(X_p)  # (n, 7)
    sub_ord = source_models["sub_grade_ord"].predict(X_p)    # (n,)
    int_rate = source_models["int_rate"].predict(X_p)         # (n,)

    # Make sure grade_proba columns are in canonical A..G order. CatBoost
    # orders classes by sorted label string, so for grade {A,B,C,D,E,F,G}
    # alphabetical == canonical. Defensively reindex.
    classes = list(source_models["grade"].classes_)
    if classes != GRADE_CLASSES:
        idx = [classes.index(g) for g in GRADE_CLASSES]
        grade_proba = grade_proba[:, idx]

    return {
        "grade_proba": grade_proba,
        "sub_grade_ord": sub_ord,
        "int_rate": int_rate,
        "_source_tag": source_tag,
    }


def enrich_with_aux(true_df: pd.DataFrame,
                    aux_per_source: dict) -> pd.DataFrame:
    """
    true_df: a true-data X frame (NO target, NO ID).
    aux_per_source: dict like {"loan": {grade, sub_grade_ord, int_rate}, "lct": {...}}.
    """
    preds = {tag: _apply_models_one_source(true_df, models, tag)
             for tag, models in aux_per_source.items()}

    # Average across sources for the per-class grade probs and the single-output ones
    grade_proba_avg = np.mean([p["grade_proba"] for p in preds.values()], axis=0)
    sub_ord_per = {p["_source_tag"]: p["sub_grade_ord"] for p in preds.values()}
    rate_per = {p["_source_tag"]: p["int_rate"] for p in preds.values()}

    sub_ord_avg = np.mean(list(sub_ord_per.values()), axis=0)
    rate_avg = np.mean(list(rate_per.values()), axis=0)
    grade_argmax_ord = grade_proba_avg.argmax(axis=1) + 1  # 1..7

    out = true_df.copy()
    for i, g in enumerate(GRADE_CLASSES):
        out[f"aux_grade_prob_{g}"] = grade_proba_avg[:, i]
    out["aux_subgrade_ord_avg"] = sub_ord_avg
    out["aux_int_rate_avg"] = rate_avg
    for tag in preds.keys():
        out[f"aux_subgrade_ord_{tag}"] = sub_ord_per[tag]
        out[f"aux_int_rate_{tag}"] = rate_per[tag]
    out["aux_grade_argmax"] = grade_argmax_ord

    return out


def predict_on_archive(archive_df: pd.DataFrame, source_models: dict) -> pd.DataFrame:
    """Used in iter >= 2 to feed the FINAL model's predictions back into aux training.
    Returns a DataFrame with `pred_int_rate` and `pred_grade_argmax_ord` columns,
    aligned to `archive_df` row order."""
    train_feature_names = list(source_models["grade"].feature_names_)
    X = archive_df.copy()
    for c in train_feature_names:
        if c not in X.columns:
            X[c] = np.nan
    X = X[train_feature_names].copy()
    X_p, _ = _prepare_x(X)
    rate = source_models["int_rate"].predict(X_p)
    proba = source_models["grade"].predict_proba(X_p)
    classes = list(source_models["grade"].classes_)
    if classes != GRADE_CLASSES:
        idx = [classes.index(g) for g in GRADE_CLASSES]
        proba = proba[:, idx]
    return pd.DataFrame({
        "pred_int_rate": rate,
        "pred_grade_argmax_ord": proba.argmax(axis=1) + 1,
        "pred_subgrade_ord": source_models["sub_grade_ord"].predict(X_p),
    })
