"""End-to-end training pipeline.

Pipeline:
  1. Load train + test (drop outcome leakage).
  2. Optuna-tune CatBoost (80/20 split, 100 trials by default).
  3. 5-fold CV with tuned CatBoost — save OOF + averaged test predictions.
  4. 5-fold CV with frozen XGBoost v1 params — save OOF + averaged test predictions.
  5. NNLS-blend the two OOF series; drop any model with weight < 0.05.
  6. Rate-snap experiment: apply only if it improves OOF RMSE by > 0.005.
  7. Write outputs/final_test_predictions.csv and outputs/final_report.json.

Run: ``python -m src.train``
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from scipy.optimize import nnls
from sklearn.model_selection import KFold, train_test_split

from src.data import (
    ID_COL,
    N_FOLDS,
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET_COL,
    read_test,
    read_train,
)
from src.features import Preprocessor
from src.models import (
    HISTORICAL_V1_XGB_PARAMS,
    catboost_search_space,
    fit_catboost,
    fit_xgb,
    predict_catboost,
    predict_xgb,
)

OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "100"))
STUDY_PATH = OUTPUT_DIR / "optuna_catboost.db"
STUDY_NAME = "lc_catboost_v1_features"

CAT_OOF_PATH = OUTPUT_DIR / "oof_catboost.csv"
XGB_OOF_PATH = OUTPUT_DIR / "oof_xgb.csv"
CAT_TEST_PATH = OUTPUT_DIR / "test_preds_catboost.csv"
XGB_TEST_PATH = OUTPUT_DIR / "test_preds_xgb.csv"
BLEND_META_PATH = OUTPUT_DIR / "blend_meta.json"
FINAL_REPORT_PATH = OUTPUT_DIR / "final_report.json"
FINAL_SUBMISSION_PATH = OUTPUT_DIR / "final_test_predictions.csv"
MODELS_DIR = OUTPUT_DIR / "models"

# Predictions clipped just outside the observed train range (6.46–30.99) to
# avoid clipping legitimate tail predictions while capping pathological ones.
PRED_CLIP = (6.0, 31.0)
# Rate-snap is kept ONLY if it improves OOF RMSE by more than this margin.
SNAP_THRESHOLD = 0.005
# Models with NNLS weight below this threshold are dropped from the blend.
MIN_BLEND_WEIGHT = 0.05


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "R2": r2(y_true, y_pred)}


def get_folds(n_rows: int, seed: int = RANDOM_STATE, n_splits: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Single source of truth for fold indices. Every model uses the same splits."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_rows)))


def _tune_catboost(train: pd.DataFrame) -> tuple[dict[str, Any], float]:
    """Optuna tuning on a single 80/20 split (same strategy as the v1 XGB tune)."""
    print(f"\n=== CatBoost Optuna tuning ({OPTUNA_TRIALS} trials) ===", flush=True)
    train_idx, val_idx = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    raw_fit = train.iloc[train_idx].reset_index(drop=True)
    raw_val = train.iloc[val_idx].reset_index(drop=True)
    y_fit = raw_fit[TARGET_COL].to_numpy()
    y_val = raw_val[TARGET_COL].to_numpy()

    pre = Preprocessor()
    X_fit = pre.fit_transform(raw_fit)
    X_val = pre.transform(raw_val)

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        params = catboost_search_space(trial)
        model = fit_catboost(X_fit, y_fit, X_val, y_val, params)
        preds = predict_catboost(model, X_val)
        trial_rmse = rmse(y_val, preds)
        print(
            f"  Trial {trial.number + 1}/{OPTUNA_TRIALS}: RMSE={trial_rmse:.5f} "
            f"depth={params['depth']} iters={params['iterations']} "
            f"lr={params['learning_rate']:.4f} elapsed={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return trial_rmse

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_STATE),
        storage=f"sqlite:///{STUDY_PATH.as_posix()}",
        study_name=STUDY_NAME,
        load_if_exists=True,
    )
    before = len(study.trials)
    print(f"Study: {STUDY_PATH}; existing trials: {before}", flush=True)
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    print(f"\nBest tuning RMSE: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")
    return dict(study.best_params), float(study.best_value)


def _run_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_kind: str,
    cat_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """5-fold CV: fit fold-local preprocessor + model, collect OOF + test averages."""
    y = train[TARGET_COL].to_numpy()
    oof = np.full(len(train), np.nan, dtype=float)
    test_mat = np.zeros((len(test), len(folds)), dtype=float)
    fold_rmse: list[float] = []

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for fold_no, (fit_idx, val_idx) in enumerate(folds, start=1):
        t0 = time.perf_counter()
        raw_fit = train.iloc[fit_idx].reset_index(drop=True)
        raw_val = train.iloc[val_idx].reset_index(drop=True)
        y_fit = raw_fit[TARGET_COL].to_numpy()
        y_val = raw_val[TARGET_COL].to_numpy()

        pre = Preprocessor()
        X_fit = pre.fit_transform(raw_fit)
        X_val = pre.transform(raw_val)
        X_test = pre.transform(test)

        if model_kind == "xgb":
            booster = fit_xgb(X_fit, y_fit, X_val, y_val)
            val_pred = predict_xgb(booster, X_val)
            test_pred = predict_xgb(booster, X_test)
            booster.save_model(str(MODELS_DIR / f"xgb_fold_{fold_no}.json"))
        elif model_kind == "catboost":
            assert cat_params is not None
            model = fit_catboost(X_fit, y_fit, X_val, y_val, cat_params)
            val_pred = predict_catboost(model, X_val)
            test_pred = predict_catboost(model, X_test)
            model.save_model(str(MODELS_DIR / f"catboost_fold_{fold_no}.cbm"))
        else:
            raise ValueError(f"Unknown model_kind: {model_kind!r}")

        oof[val_idx] = val_pred
        test_mat[:, fold_no - 1] = test_pred
        fold_rmse.append(rmse(y_val, val_pred))
        print(
            f"  [{model_kind}] Fold {fold_no}/{len(folds)}: "
            f"RMSE={fold_rmse[-1]:.5f} elapsed={time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    if np.isnan(oof).any() or not np.isfinite(oof).all():
        raise RuntimeError(f"{model_kind} OOF predictions are incomplete or non-finite.")

    test_pred = test_mat.mean(axis=1)
    return {
        "oof": oof,
        "test_pred": test_pred,
        "fold_rmse": fold_rmse,
        "metrics": metrics(y, oof),
        "fold_std": float(np.std(fold_rmse)),
    }


def _nnls_blend(
    y: np.ndarray, oof_xgb: np.ndarray, oof_cat: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    """Non-negative least squares blend; drop weights < MIN_BLEND_WEIGHT."""
    pred_matrix = np.column_stack([oof_xgb, oof_cat])
    weights, _ = nnls(pred_matrix, y)
    raw = {"xgb": float(weights[0]), "catboost": float(weights[1])}
    print(f"  Raw NNLS weights: {raw}")

    if raw["xgb"] < MIN_BLEND_WEIGHT and raw["catboost"] >= MIN_BLEND_WEIGHT:
        final = {"xgb": 0.0, "catboost": 1.0}
        blended = oof_cat.copy()
        print(f"  XGB weight below {MIN_BLEND_WEIGHT}; collapsing blend to CatBoost only.")
    elif raw["catboost"] < MIN_BLEND_WEIGHT and raw["xgb"] >= MIN_BLEND_WEIGHT:
        final = {"xgb": 1.0, "catboost": 0.0}
        blended = oof_xgb.copy()
        print(f"  CatBoost weight below {MIN_BLEND_WEIGHT}; collapsing blend to XGB only.")
    elif raw["xgb"] < MIN_BLEND_WEIGHT and raw["catboost"] < MIN_BLEND_WEIGHT:
        # Both tiny — keep raw rather than divide by zero.
        final = raw
        blended = pred_matrix @ weights
    else:
        final = raw
        blended = pred_matrix @ weights

    return final, blended


def _rate_snap_decision(
    train: pd.DataFrame, oof_blended: np.ndarray, test_blended: np.ndarray
) -> tuple[bool, np.ndarray, np.ndarray, float, float]:
    """Snap to nearest legal int_rate value; keep only if OOF RMSE improves > SNAP_THRESHOLD."""
    legal = np.sort(train[TARGET_COL].unique())
    y = train[TARGET_COL].to_numpy()

    def snap(values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(legal, values)
        idx = np.clip(idx, 1, len(legal) - 1)
        left = legal[idx - 1]
        right = legal[idx]
        return np.where(np.abs(values - left) <= np.abs(values - right), left, right)

    snapped_oof = snap(oof_blended)
    snapped_test = snap(test_blended)
    rmse_pre = rmse(y, oof_blended)
    rmse_post = rmse(y, snapped_oof)
    improvement = rmse_pre - rmse_post
    apply = improvement > SNAP_THRESHOLD
    print(
        f"  Rate-snap: OOF RMSE {rmse_pre:.5f} -> {rmse_post:.5f} "
        f"(delta={improvement:+.5f}; threshold={SNAP_THRESHOLD}); "
        f"decision={'APPLY' if apply else 'SKIP'}"
    )
    return apply, snapped_oof, snapped_test, rmse_pre, rmse_post


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    print("=== Loading data ===")
    train = read_train()
    test = read_test()
    print(f"Train: {train.shape}; Test: {test.shape}")

    folds = get_folds(len(train), seed=RANDOM_STATE, n_splits=N_FOLDS)
    print(f"Fold indices built: {N_FOLDS}-fold KFold, seed={RANDOM_STATE}")

    best_cat_params, tuning_best = _tune_catboost(train)

    print("\n=== CatBoost 5-fold CV (best params) ===")
    cat_result = _run_cv(train, test, folds, model_kind="catboost", cat_params=best_cat_params)
    print(f"CatBoost OOF: {cat_result['metrics']}; fold std={cat_result['fold_std']:.5f}")

    print("\n=== XGBoost v1 5-fold CV (frozen params) ===")
    xgb_result = _run_cv(train, test, folds, model_kind="xgb")
    print(f"XGB OOF: {xgb_result['metrics']}; fold std={xgb_result['fold_std']:.5f}")

    y = train[TARGET_COL].to_numpy()

    print("\n=== NNLS blend ===")
    weights, oof_blend = _nnls_blend(y, xgb_result["oof"], cat_result["oof"])
    test_blend = weights["xgb"] * xgb_result["test_pred"] + weights["catboost"] * cat_result["test_pred"]
    print(f"  Blend OOF: {metrics(y, oof_blend)}")

    print("\n=== Rate-snap test ===")
    apply_snap, snap_oof, snap_test, rmse_pre, rmse_post = _rate_snap_decision(
        train, oof_blend, test_blend
    )

    final_oof = snap_oof if apply_snap else oof_blend
    final_test = snap_test if apply_snap else test_blend
    final_test = np.clip(final_test, *PRED_CLIP)

    # Persist per-model OOF + test prediction CSVs (predict.py reads test files).
    pd.DataFrame({"row_id": np.arange(len(train)), TARGET_COL: y, "pred": cat_result["oof"]}).to_csv(
        CAT_OOF_PATH, index=False
    )
    pd.DataFrame({"row_id": np.arange(len(train)), TARGET_COL: y, "pred": xgb_result["oof"]}).to_csv(
        XGB_OOF_PATH, index=False
    )
    pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), "pred": cat_result["test_pred"]}).to_csv(
        CAT_TEST_PATH, index=False
    )
    pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), "pred": xgb_result["test_pred"]}).to_csv(
        XGB_TEST_PATH, index=False
    )

    BLEND_META_PATH.write_text(
        json.dumps(
            {
                "weights": weights,
                "apply_snap": apply_snap,
                "pred_clip": list(PRED_CLIP),
                "snap_threshold": SNAP_THRESHOLD,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    submission = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET_COL: final_test})
    assert len(submission) == 10_000, f"Expected 10,000 rows, got {len(submission)}"
    assert list(submission.columns) == [ID_COL, TARGET_COL]
    submission.to_csv(FINAL_SUBMISSION_PATH, index=False)

    report = {
        "tuning": {"best_rmse_holdout": tuning_best, "trials": OPTUNA_TRIALS},
        "catboost_params": best_cat_params,
        "catboost": {
            "oof": cat_result["metrics"],
            "fold_rmse": cat_result["fold_rmse"],
            "fold_std": cat_result["fold_std"],
        },
        "xgb": {
            "oof": xgb_result["metrics"],
            "fold_rmse": xgb_result["fold_rmse"],
            "fold_std": xgb_result["fold_std"],
            "params": HISTORICAL_V1_XGB_PARAMS,
        },
        "blend": {
            "weights": weights,
            "oof": metrics(y, oof_blend),
        },
        "snap": {
            "applied": apply_snap,
            "rmse_pre": rmse_pre,
            "rmse_post": rmse_post,
            "delta": rmse_pre - rmse_post,
            "threshold": SNAP_THRESHOLD,
        },
        "final_oof_rmse": rmse(y, final_oof),
        "rows": {"train": len(train), "test": len(test)},
        "pred_clip": list(PRED_CLIP),
        "total_runtime_sec": time.perf_counter() - t_total,
    }
    FINAL_REPORT_PATH.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    print("\n=== Done ===")
    print(f"Submission:    {FINAL_SUBMISSION_PATH}")
    print(f"Report:        {FINAL_REPORT_PATH}")
    print(f"Final OOF RMSE: {report['final_oof_rmse']:.5f}")
    print(f"Total runtime: {report['total_runtime_sec']:.1f}s")


if __name__ == "__main__":
    main()
