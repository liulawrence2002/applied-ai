"""Iteration loop: aux training → enrich → final model → convergence check.

To keep wall-clock reasonable across up to 5 iterations, the per-iter final
model is a multi-seed CatBoost on enriched true data. Once iteration
converges, `run_all.py` runs the FULL `production/` stack on the best
iteration's enriched data as a post-processing step (one-shot cost).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from reverse_engineer import enrich
from reverse_engineer.archive_loader import (
    add_subgrade_ordinal, load_archive_lc_train, load_archive_loan, load_true_data,
)
from reverse_engineer.aux_models import _prepare_x, train_all_aux
from reverse_engineer.config import (
    AUX_CB_PARAMS, HOLDOUT_FRAC, OUTPUTS_DIR, RANDOM_STATE,
)

ITER_FINAL_N_SEEDS = 3
ITER_FINAL_N_FOLDS = 5


def _ensure_iter_dir(iter_idx: int) -> Path:
    p = OUTPUTS_DIR / f"iter_{iter_idx}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _engineer_for_final(X_enriched: pd.DataFrame) -> pd.DataFrame:
    """Light feature engineering on enriched data: keep raw + aux cols, do the
    minimum prep so CatBoost can ingest them."""
    df = X_enriched.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string").fillna("Missing")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def _cat_idx(df: pd.DataFrame) -> list[int]:
    return [df.columns.get_loc(c) for c in df.columns if df[c].dtype.name == "string"]


def _fit_iter_final(X_train_enriched, y_train, X_val_enriched, y_val,
                     X_test_enriched, seed) -> tuple[np.ndarray, np.ndarray]:
    Xtr = _engineer_for_final(X_train_enriched)
    Xva = _engineer_for_final(X_val_enriched)
    Xte = _engineer_for_final(X_test_enriched)
    cat_idx = _cat_idx(Xtr)
    m = CatBoostRegressor(
        iterations=3000,
        depth=8,
        learning_rate=0.04,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        bagging_temperature=0.5,
        border_count=254,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        early_stopping_rounds=120,
    )
    m.fit(Xtr, y_train, cat_features=cat_idx, eval_set=(Xva, y_val), verbose=False)
    return m.predict(Xva), m.predict(Xte)


def _run_final_kfold(X_enriched_train, y_train, X_val_enriched, y_val,
                      X_test_enriched, seeds=None) -> dict:
    """K-fold OOF on enriched train, predict on val + test."""
    seeds = seeds or [RANDOM_STATE + 7 * i for i in range(ITER_FINAL_N_SEEDS)]
    n_tr = len(X_enriched_train); n_va = len(X_val_enriched); n_te = len(X_test_enriched)
    oof = np.zeros(n_tr); val_p = np.zeros(n_va); test_p = np.zeros(n_te)
    kf = KFold(n_splits=ITER_FINAL_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold_id, (tr, va) in enumerate(kf.split(X_enriched_train)):
        t = time()
        Xtr_f = X_enriched_train.iloc[tr]
        Xva_f = X_enriched_train.iloc[va]
        ytr_f = y_train[tr]; yva_f = y_train[va]
        va_seed = np.zeros(len(va)); vv_seed = np.zeros(n_va); te_seed = np.zeros(n_te)
        for s in seeds:
            va_pred, te_pred = _fit_iter_final(Xtr_f, ytr_f, Xva_f, yva_f,
                                                pd.concat([X_val_enriched, X_test_enriched]), s)
            # Split combined val+test
            n_val_block = len(X_val_enriched)
            vv_seed += te_pred[:n_val_block]
            te_seed += te_pred[n_val_block:]
            va_seed += va_pred
        va_seed /= len(seeds); vv_seed /= len(seeds); te_seed /= len(seeds)
        oof[va] = va_seed
        val_p += vv_seed / ITER_FINAL_N_FOLDS
        test_p += te_seed / ITER_FINAL_N_FOLDS
        fold_rmse = float(np.sqrt(mean_squared_error(yva_f, va_seed)))
        print(f"      fold {fold_id+1}/{ITER_FINAL_N_FOLDS}: {fold_rmse:.4f}  ({time()-t:.0f}s)", flush=True)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_p)))
    return {"oof": oof, "val_pred": val_p, "test_pred": test_p,
            "val_rmse": val_rmse,
            "val_mae": float(mean_absolute_error(y_val, val_p)),
            "val_r2": float(r2_score(y_val, val_p))}


def _train_aux_ensemble(prev_predictions_on_archive: dict | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Train aux models on BOTH archive sources. Returns:
        (aux_per_source, archive_loan_df, archive_lct_df)
    so the caller can re-apply the trained models back to archive rows for
    the next iteration's extra-feature input."""
    print("  Loading archive sources...", flush=True)
    t = time()
    archive_loan = load_archive_loan()
    print(f"    loan.csv: {archive_loan.shape}  ({time()-t:.0f}s)", flush=True)

    t = time()
    archive_lct = load_archive_lc_train()
    print(f"    LC_train.csv: {archive_lct.shape}  ({time()-t:.0f}s)", flush=True)

    archive_loan = add_subgrade_ordinal(archive_loan)
    archive_lct = add_subgrade_ordinal(archive_lct)

    extra_loan = prev_predictions_on_archive.get("loan") if prev_predictions_on_archive else None
    extra_lct = prev_predictions_on_archive.get("lct") if prev_predictions_on_archive else None

    aux_loan = train_all_aux("loan", archive_loan, extra_features=extra_loan)
    aux_lct = train_all_aux("lct", archive_lct, extra_features=extra_lct)
    return {"loan": aux_loan, "lct": aux_lct}, archive_loan, archive_lct


def run_one_iteration(iter_idx: int, prev_archive_predictions: dict | None = None,
                       prev_val_rmse: float | None = None) -> dict:
    """Returns dict with keys: val_rmse, val_pred, test_pred, archive_predictions, iter_dir."""
    iter_dir = _ensure_iter_dir(iter_idx)
    t_iter = time()
    print(f"\n=== ITERATION {iter_idx} ===", flush=True)

    # 1. Train aux ensemble
    aux_per_source, archive_loan, archive_lct = _train_aux_ensemble(prev_archive_predictions)

    # 2. Load true data + apply 80/20 holdout
    X_train, X_test, y, test_ids = load_true_data()
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y.values, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE,
    )
    X_train_split = X_train_split.reset_index(drop=True)
    X_val_split = X_val_split.reset_index(drop=True)

    # 3. Enrich
    print("  Enriching true data with aux predictions...", flush=True)
    t = time()
    X_train_enriched = enrich.enrich_with_aux(X_train_split, aux_per_source)
    X_val_enriched = enrich.enrich_with_aux(X_val_split, aux_per_source)
    X_test_enriched = enrich.enrich_with_aux(X_test, aux_per_source)
    print(f"    train enriched cols: {X_train_enriched.shape[1]} (+ {X_train_enriched.shape[1]-X_train.shape[1]} aux)  ({time()-t:.0f}s)", flush=True)

    # 4. Per-iter final model (multi-seed CatBoost K-fold)
    print(f"  Per-iter final model (CatBoost {ITER_FINAL_N_FOLDS}-fold x {ITER_FINAL_N_SEEDS}-seed)...", flush=True)
    result = _run_final_kfold(
        X_train_enriched, y_train_split, X_val_enriched, y_val_split,
        X_test_enriched,
    )
    print(f"  iter {iter_idx} val RMSE: {result['val_rmse']:.4f}", flush=True)

    # 5. Predict back to archive (only if more iterations may run)
    print("  Computing archive predictions for next iter...", flush=True)
    archive_predictions = {
        "loan": enrich.predict_on_archive(archive_loan, aux_per_source["loan"]),
        "lct": enrich.predict_on_archive(archive_lct, aux_per_source["lct"]),
    }

    # 6. Persist iter artifacts
    pickle.dump({
        "iter": iter_idx,
        "val_rmse": result["val_rmse"],
        "val_mae": result["val_mae"],
        "val_r2": result["val_r2"],
        "test_pred": result["test_pred"],
        "val_pred": result["val_pred"],
        "y_val": y_val_split,
        "y_train": y_train_split,
        "oof": result["oof"],
        "aux_per_source": aux_per_source,
        "test_ids": test_ids.values,
        "n_enriched_cols": X_train_enriched.shape[1],
    }, open(iter_dir / "iter_artifacts.pkl", "wb"))
    # Save enriched test for the post-iter production stack run
    X_test_enriched.to_parquet(iter_dir / "X_test_enriched.parquet")
    X_train_enriched.assign(int_rate=y_train_split).to_parquet(iter_dir / "X_train_enriched.parquet")
    X_val_enriched.assign(int_rate=y_val_split).to_parquet(iter_dir / "X_val_enriched.parquet")

    delta = (prev_val_rmse - result["val_rmse"]) if prev_val_rmse is not None else None
    print(f"  iter {iter_idx} runtime: {time()-t_iter:.0f}s, delta vs prev: {delta}", flush=True)
    return {
        "iter": iter_idx,
        "val_rmse": result["val_rmse"],
        "val_mae": result["val_mae"],
        "val_r2": result["val_r2"],
        "delta": delta,
        "test_pred": result["test_pred"],
        "val_pred": result["val_pred"],
        "test_ids": test_ids.values,
        "archive_predictions": archive_predictions,
        "runtime_sec": time() - t_iter,
        "n_enriched_cols": X_train_enriched.shape[1],
    }
