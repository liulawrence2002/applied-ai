"""
CatBoost-only pipeline for LendingClub Interest Rate Prediction
===============================================================

A trimmed copy of final_pipeline.py that trains ONLY CatBoost — no LightGBM,
XGBoost, RandomForest, Ridge, or MLP, and no stacking / hill-climb blend. Much
faster to run when you only need the CatBoost submission + honest validation.

Shared logic (load_raw, engineer, target encoding, CatBoost frame prep, constants)
is imported from final_pipeline so feature-engineering changes stay in one place.

Pipeline:
  1. Load true data/ (train 100k + test 10k), drop loan_status.
  2. Stateless feature engineering (fp.engineer).
  3. 80/20 random holdout for honest validation.
  4. K-fold (5) CatBoost:
       - smoothed K-fold target encoding refit inside each fold (no leakage)
       - OOF predictions on the train fold (honest CV estimate)
       - per-fold predictions on the 20% holdout and the test set, averaged
  5. Report OOF + held-out RMSE / MAE / R2 and write clipped test predictions.

Run:
  .venv/Scripts/python.exe final/final_pipeline_catboost.py
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from catboost import CatBoostRegressor

# Share one source of truth with the main pipeline module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_pipeline as fp

warnings.filterwarnings("ignore")

OUTPUTS_DIR = fp.OUTPUTS_DIR
CACHED_TUNING = fp.CACHED_TUNING
RANDOM_STATE = fp.RANDOM_STATE
N_FOLDS = fp.N_FOLDS
N_SEEDS = fp.N_SEEDS
EMP_TITLE_TOP_N = fp.EMP_TITLE_TOP_N
HOLDOUT_FRAC = fp.HOLDOUT_FRAC


def fit_catboost(Xtr, ytr, Xva, yva, Xte, params, seed):
    Xtr_p, cat_idx = fp.to_catboost_pool(Xtr)
    Xva_p, _ = fp.to_catboost_pool(Xva)
    Xte_p, _ = fp.to_catboost_pool(Xte)
    cb = CatBoostRegressor(
        **params, iterations=2500, loss_function="RMSE", eval_metric="RMSE",
        random_seed=seed, verbose=0, allow_writing_files=False, early_stopping_rounds=100,
    )
    cb.fit(Xtr_p, ytr, cat_features=cat_idx, eval_set=(Xva_p, yva), verbose=False)
    return cb.predict(Xva_p), cb.predict(Xte_p), cb


def main():
    print("=" * 72)
    print("CATBOOST-ONLY PIPELINE")
    print("=" * 72)

    print("\n[1] Load + engineer")
    t = time()
    train_raw, test_raw = fp.load_raw()
    train_fe = fp.engineer(train_raw)
    test_fe = fp.engineer(test_raw)

    if "emp_title" in train_fe.columns:
        top = train_fe["emp_title"].value_counts().head(EMP_TITLE_TOP_N).index.tolist()
        train_fe["emp_title"] = train_fe["emp_title"].where(train_fe["emp_title"].isin(top), "Other").fillna("Missing").astype(str)
        test_fe["emp_title"] = test_fe["emp_title"].where(test_fe["emp_title"].isin(top), "Other").fillna("Missing").astype(str)

    test_ids = test_fe["ID"].copy() if "ID" in test_fe.columns else None
    if test_ids is not None:
        test_fe = test_fe.drop(columns=["ID"])

    y_full = train_fe["int_rate"].values.astype(np.float64)
    X_full = train_fe.drop(columns=["int_rate"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE,
    )
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    print(f"  shapes: train={X_train.shape}  val={X_val.shape}  test={test_fe.shape}  ({time()-t:.1f}s)")

    print("\n[2] Load cached tuning params")
    tuning = json.loads(CACHED_TUNING.read_text())["params"]
    cb_params = tuning["CatBoost"]
    print(f"  CatBoost: {cb_params}")

    print("\n[3] K-fold OOF CatBoost")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_tr = len(X_train)
    n_va = len(X_val)
    n_te = len(test_fe)
    seeds = [RANDOM_STATE + 7 * i for i in range(N_SEEDS)]

    oof = np.zeros(n_tr)
    val_pred = np.zeros(n_va)
    test_pred = np.zeros(n_te)

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        t_fold = time()
        Xtr = X_train.iloc[tr_idx]
        Xva = X_train.iloc[va_idx]
        ytr = y_train[tr_idx]
        yva = y_train[va_idx]

        Xtr_te, Xva_te, Xte_te = fp.add_target_encodings(Xtr, ytr, Xva, test_fe)
        _, Xval_te, _ = fp.add_target_encodings(Xtr, ytr, X_val, test_fe)

        va_acc = np.zeros(len(va_idx))
        te_acc = np.zeros(n_te)
        vv_acc = np.zeros(n_va)
        for s in seeds:
            cb_va, cb_te, cb_m = fit_catboost(Xtr_te, ytr, Xva_te, yva, Xte_te, cb_params, s)
            Xval_cb, _ = fp.to_catboost_pool(Xval_te)
            va_acc += cb_va
            te_acc += cb_te
            vv_acc += cb_m.predict(Xval_cb)
        va_acc /= len(seeds); te_acc /= len(seeds); vv_acc /= len(seeds)

        oof[va_idx] = va_acc
        val_pred += vv_acc / N_FOLDS
        test_pred += te_acc / N_FOLDS

        print(f"  fold {fold_id+1}/{N_FOLDS}: OOF RMSE={np.sqrt(mean_squared_error(yva, va_acc)):.4f}  ({time()-t_fold:.0f}s)", flush=True)

    print("\n[4] Held-out validation metrics")
    oof_rmse = float(np.sqrt(mean_squared_error(y_train, oof)))
    rows = [{
        "model": "CatBoost (fold-averaged)",
        "OOF_RMSE": oof_rmse,
        "val_RMSE": float(np.sqrt(mean_squared_error(y_val, val_pred))),
        "val_MAE": float(mean_absolute_error(y_val, val_pred)),
        "val_R2": float(r2_score(y_val, val_pred)),
    }]
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(OUTPUTS_DIR / "catboost_results.csv", index=False)

    print("\n[5] Test predictions")
    final_test = np.clip(test_pred, 6.0, 31.0)
    sub = pd.DataFrame({"ID": test_ids.values if test_ids is not None else np.arange(n_te),
                        "int_rate": final_test})
    sub_path = OUTPUTS_DIR / "catboost_test_predictions_final.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  saved {len(sub):,} predictions -> {sub_path}")

    print("\n[6] Persist artifacts")
    with open(OUTPUTS_DIR / "catboost_artifacts.pkl", "wb") as fh:
        pickle.dump({
            "oof": oof, "val_pred": val_pred, "test_pred": test_pred,
            "y_train": y_train, "y_val": y_val,
        }, fh)
    summary = {
        "model": "CatBoost-only",
        "n_folds": N_FOLDS,
        "n_seeds": N_SEEDS,
        "cb_params": cb_params,
        "results": res.set_index("model").to_dict("index"),
    }
    (OUTPUTS_DIR / "catboost_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved summary -> {OUTPUTS_DIR / 'catboost_summary.json'}")

    return res


if __name__ == "__main__":
    main()
