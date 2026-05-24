"""Pseudo-labeling for an extra CatBoost base learner.

The idea:
  1. The existing 9-base ensemble produces a stable test-set prediction.
  2. Test rows where ALL base learners agree (low cross-model std) are the
     "easy" rows — the ensemble is confident in its prediction.
  3. We treat those high-confidence test predictions as pseudo-labels and
     append them to the training set with a reduced sample weight.
  4. Refit CatBoost on (train + pseudo-labeled test) per fold to produce a
     new `cb_pseudo` set of OOF / val / test predictions.
  5. The stacker picks `cb_pseudo` up as just another base learner.

Why this can help: the pseudo-labels expose CatBoost to the test-set feature
distribution during training, narrowing any train-vs-test drift. Reduced
sample weight (0.5) keeps the real labels dominant.
"""
from __future__ import annotations

import json
import pickle
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from production.config import (
    CHECKPOINTS_DIR, EMP_TITLE_TOP_N, HOLDOUT_FRAC, N_FOLDS, N_SEEDS_GBDT,
    OUTPUTS_DIR, RANDOM_STATE,
)
from production.encoders import add_te_columns, to_catboost_pool
from production.features import compact_emp_title, engineer, load_raw
from production.fitters import fit_catboost

CONFIDENCE_QUANTILE = 0.50    # use bottom-half cross-model std (most confident)
PSEUDO_WEIGHT = 0.5            # weight of pseudo-labels relative to real labels


def _select_pseudo_labels(agg):
    """Pick the top-confidence test rows + their mean ensemble prediction."""
    BASES = agg["BASES"]
    test_stack = np.column_stack([agg["test_pred"][b] for b in BASES])
    # Confidence = inverse of cross-model std (low std = all models agree)
    cross_std = test_stack.std(axis=1)
    mean_pred = test_stack.mean(axis=1)

    threshold = np.quantile(cross_std, CONFIDENCE_QUANTILE)
    mask = cross_std <= threshold
    n_pseudo = int(mask.sum())
    print(f"  [pseudo] using {n_pseudo:,}/{len(mask):,} test rows "
          f"(cross-model std <= {threshold:.3f})", flush=True)
    return mask, mean_pred


def _prepare_data():
    train_raw, test_raw = load_raw()
    train_fe = engineer(train_raw)
    test_fe = engineer(test_raw)
    train_fe, test_fe = compact_emp_title(train_fe, test_fe, EMP_TITLE_TOP_N)

    if "ID" in test_fe.columns:
        test_fe = test_fe.drop(columns=["ID"])

    y_full = train_fe["int_rate"].values.astype(np.float64)
    X_full = train_fe.drop(columns=["int_rate"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE,
    )
    return X_train, X_val, y_train, y_val, test_fe


def run():
    """Generate `cb_pseudo` OOF/val/test predictions using pseudo-labels."""
    print("=" * 72)
    print("PSEUDO-LABELING PHASE -> cb_pseudo base learner")
    print("=" * 72, flush=True)

    # 1. Load main aggregate
    agg_path = CHECKPOINTS_DIR / "aggregate.pkl"
    if not agg_path.exists():
        raise RuntimeError("Run oof.py first; aggregate.pkl required for pseudo-labels")
    with open(agg_path, "rb") as f:
        agg = pickle.load(f)

    # 2. Load tuning params
    tune_path = OUTPUTS_DIR / "tuning_v2.json"
    cb_params = json.loads(tune_path.read_text())["params"]["CatBoost"]

    # 3. Select confident test rows
    mask, pseudo_y_test = _select_pseudo_labels(agg)

    # 4. Prepare data
    X_train, X_val, y_train, y_val, X_test = _prepare_data()
    print(f"  base: train={X_train.shape}  val={X_val.shape}  test={X_test.shape}", flush=True)

    # 5. Build augmented train set per fold (need fresh TE per fold)
    n_tr = len(X_train)
    n_va = len(X_val)
    n_te = len(X_test)

    cb_pseudo_oof = np.zeros(n_tr)
    cb_pseudo_val = np.zeros(n_va)
    cb_pseudo_test = np.zeros(n_te)

    seeds = [RANDOM_STATE + 7 * i for i in range(N_SEEDS_GBDT)]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        t = time()
        Xtr = X_train.iloc[tr_idx]; ytr = y_train[tr_idx]
        Xva = X_train.iloc[va_idx]; yva = y_train[va_idx]

        # Augment train with pseudo-labeled test rows
        X_pseudo = X_test.iloc[mask].reset_index(drop=True)
        y_pseudo = pseudo_y_test[mask]
        Xtr_aug = pd.concat([Xtr, X_pseudo], axis=0).reset_index(drop=True)
        ytr_aug = np.concatenate([ytr, y_pseudo])

        # Build sample weights: 1.0 for real, PSEUDO_WEIGHT for pseudo
        weights = np.ones(len(Xtr_aug))
        weights[len(Xtr):] = PSEUDO_WEIGHT
        # CatBoost's fit_catboost helper doesn't pass weights — patch in here
        from catboost import CatBoostRegressor

        # Fresh TE built on the augmented train fold (no leak: pseudo-labels
        # are model outputs, not the true int_rate they don't know)
        Xtr_te, Xva_te, Xte_te = add_te_columns(Xtr_aug, ytr_aug, Xva, X_test)
        _, Xval_te, _ = add_te_columns(Xtr_aug, ytr_aug, X_val, X_test)

        va_pred = np.zeros(len(va_idx))
        te_pred = np.zeros(n_te)
        vv_pred = np.zeros(n_va)
        for s in seeds:
            Xtr_p, cat_idx = to_catboost_pool(Xtr_te)
            Xva_p, _ = to_catboost_pool(Xva_te)
            Xte_p, _ = to_catboost_pool(Xte_te)
            Xval_p, _ = to_catboost_pool(Xval_te)

            m = CatBoostRegressor(
                **cb_params, iterations=3000, loss_function="RMSE",
                eval_metric="RMSE", random_seed=s, verbose=0,
                allow_writing_files=False, early_stopping_rounds=120,
            )
            m.fit(Xtr_p, ytr_aug, sample_weight=weights, cat_features=cat_idx,
                  eval_set=(Xva_p, yva), verbose=False)
            va_pred += m.predict(Xva_p)
            te_pred += m.predict(Xte_p)
            vv_pred += m.predict(Xval_p)

        va_pred /= len(seeds); te_pred /= len(seeds); vv_pred /= len(seeds)
        cb_pseudo_oof[va_idx] = va_pred
        cb_pseudo_val += vv_pred / N_FOLDS
        cb_pseudo_test += te_pred / N_FOLDS

        fold_rmse = float(np.sqrt(mean_squared_error(yva, va_pred)))
        print(f"  fold {fold_id+1}/{N_FOLDS}: cb_pseudo={fold_rmse:.3f}  "
              f"({time()-t:.0f}s)", flush=True)

    # 6. Persist into the aggregate
    agg["BASES"] = list(agg["BASES"])
    if "cb_pseudo" not in agg["BASES"]:
        agg["BASES"].append("cb_pseudo")
    agg["oof"]["cb_pseudo"] = cb_pseudo_oof
    agg["val_pred"]["cb_pseudo"] = cb_pseudo_val
    agg["test_pred"]["cb_pseudo"] = cb_pseudo_test
    with open(agg_path, "wb") as f:
        pickle.dump(agg, f)

    val_rmse = float(np.sqrt(mean_squared_error(agg["y_val"], cb_pseudo_val)))
    print(f"\n[pseudo] cb_pseudo held-out val RMSE = {val_rmse:.4f}", flush=True)
    print(f"[pseudo] aggregate.pkl now has {len(agg['BASES'])} base learners", flush=True)
    return agg


if __name__ == "__main__":
    run()
