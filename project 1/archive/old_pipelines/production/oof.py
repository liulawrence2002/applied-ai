"""K-fold OOF prediction generation with multi-seed averaging + checkpointing.

Each fold:
  1. Build per-fold OOF target encodings on the fold's training rows.
  2. Fit each base learner with N_SEEDS seeds; average predictions across seeds.
  3. Store OOF, val (held-out 20%), and test predictions per learner.
  4. Checkpoint after every fold so a kill loses at most one fold.

Re-runs resume from the latest aggregate checkpoint. Delete
`outputs/production/checkpoints/aggregate.pkl` (and the fold_NN.pkl files) to
start over.
"""
from __future__ import annotations

import pickle
from time import time

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from production.config import (
    CHECKPOINTS_DIR, EMP_TITLE_TOP_N, HOLDOUT_FRAC, N_FOLDS, N_SEEDS_GBDT,
    N_SEEDS_NN, RANDOM_STATE,
)
from production.encoders import (
    add_te_columns, build_cat_dtypes,
    to_catboost_pool, to_lgb_frame, to_numeric_frame, to_xgb_frame,
)
from production.features import compact_emp_title, engineer, load_raw
from production.fitters import (
    fit_catboost, fit_lgbm, fit_lgbm_huber, fit_mlp, fit_rf, fit_ridge, fit_xgb,
)
from production.nn_models import fit_ft_transformer, fit_tab_mlp_deep

BASES = ["cb", "lgb", "xgb", "lgb_huber", "rf", "ridge", "mlp", "ft", "mlp_deep"]


def _prepare_data():
    train_raw, test_raw = load_raw()
    train_fe = engineer(train_raw)
    test_fe = engineer(test_raw)
    train_fe, test_fe = compact_emp_title(train_fe, test_fe, EMP_TITLE_TOP_N)

    test_ids = test_fe["ID"].copy() if "ID" in test_fe.columns else None
    if test_ids is not None:
        test_fe = test_fe.drop(columns=["ID"])

    y_full = train_fe["int_rate"].values.astype(np.float64)
    X_full = train_fe.drop(columns=["int_rate"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE,
    )
    return X_train, X_val, y_train, y_val, test_fe, test_ids


def _aggregator_path():
    return CHECKPOINTS_DIR / "aggregate.pkl"


def _load_aggregator():
    p = _aggregator_path()
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _save_aggregator(agg):
    with open(_aggregator_path(), "wb") as f:
        pickle.dump(agg, f)


def _seed_gbdt_block(per, b, va, te, vv, n_seeds_gbdt):
    """Accumulate one seed's GBDT predictions."""
    per[b]["va"] += va
    per[b]["te"] += te
    per[b]["vv"] += vv


def _seed_nonseeded_block(per, b, va, te, vv, n_seeds_gbdt):
    """For models that we run only on seed 0, scale by N_SEEDS_GBDT so the
    /= divisor in the average step cancels out."""
    per[b]["va"] += va * n_seeds_gbdt
    per[b]["te"] += te * n_seeds_gbdt
    per[b]["vv"] += vv * n_seeds_gbdt


def run(tuning_params):
    """Main loop. `tuning_params` is the dict from production.tune.run()."""
    X_train, X_val, y_train, y_val, test_fe, test_ids = _prepare_data()
    print(f"[oof] X_train={X_train.shape}  X_val={X_val.shape}  X_test={test_fe.shape}", flush=True)

    n_tr = len(X_train)
    n_va = len(X_val)
    n_te = len(test_fe)

    agg = _load_aggregator()
    if agg is None:
        agg = {
            "BASES": BASES,
            "oof": {b: np.zeros(n_tr) for b in BASES},
            "val_pred": {b: np.zeros(n_va) for b in BASES},
            "test_pred": {b: np.zeros(n_te) for b in BASES},
            "completed_folds": [],
            "y_train": y_train, "y_val": y_val,
            "test_ids": test_ids.values if test_ids is not None else np.arange(n_te),
        }
        print("[oof] starting fresh", flush=True)
    else:
        print(f"[oof] resuming  done folds: {agg['completed_folds']}", flush=True)

    seeds_gbdt = [RANDOM_STATE + 7 * i for i in range(N_SEEDS_GBDT)]
    seeds_nn = [RANDOM_STATE + 13 * i for i in range(N_SEEDS_NN)]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(kf.split(X_train))

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        if fold_id in agg["completed_folds"]:
            print(f"[oof] fold {fold_id+1}/{N_FOLDS} already done, skipping", flush=True)
            continue
        t_fold = time()
        Xtr = X_train.iloc[tr_idx]; Xva = X_train.iloc[va_idx]
        ytr = y_train[tr_idx];      yva = y_train[va_idx]

        # Target-encode (per-fold OOF for train rows; full-fit for val/test/Xval)
        Xtr_te, Xva_te, Xte_te = add_te_columns(Xtr, ytr, Xva, test_fe)
        _, Xval_te, _ = add_te_columns(Xtr, ytr, X_val, test_fe)
        cat_dt = build_cat_dtypes(Xtr_te)

        per = {b: {"va": np.zeros(len(va_idx)), "te": np.zeros(n_te), "vv": np.zeros(n_va)}
               for b in BASES}

        # --- GBDT block (each seed) ---
        for s in seeds_gbdt:
            t = time()
            va, te, m = fit_catboost(Xtr_te, ytr, Xva_te, yva, Xte_te, tuning_params["CatBoost"], s)
            Xval_p, _ = to_catboost_pool(Xval_te)
            _seed_gbdt_block(per, "cb", va, te, m.predict(Xval_p), N_SEEDS_GBDT)
            print(f"    [seed={s}] cb done  ({time()-t:.0f}s)", flush=True)

            t = time()
            va, te, m = fit_lgbm(Xtr_te, ytr, Xva_te, yva, Xte_te, tuning_params["LightGBM"], s)
            Xval_p, _ = to_lgb_frame(Xval_te, cat_dt)
            _seed_gbdt_block(per, "lgb", va, te, m.predict(Xval_p), N_SEEDS_GBDT)
            print(f"    [seed={s}] lgb done ({time()-t:.0f}s)", flush=True)

            t = time()
            va, te, m = fit_xgb(Xtr_te, ytr, Xva_te, yva, Xte_te, tuning_params["XGBoost"], s)
            Xval_p = to_xgb_frame(Xval_te, cat_dt)
            _seed_gbdt_block(per, "xgb", va, te, m.predict(Xval_p), N_SEEDS_GBDT)
            print(f"    [seed={s}] xgb done ({time()-t:.0f}s)", flush=True)

            t = time()
            va, te, m = fit_lgbm_huber(Xtr_te, ytr, Xva_te, yva, Xte_te, tuning_params["LightGBM"], s)
            Xval_p, _ = to_lgb_frame(Xval_te, cat_dt)
            _seed_gbdt_block(per, "lgb_huber", va, te, m.predict(Xval_p), N_SEEDS_GBDT)
            print(f"    [seed={s}] lgb_huber done ({time()-t:.0f}s)", flush=True)

            # RF, ridge, mlp use only seed 0 to save time
            if s == seeds_gbdt[0]:
                t = time()
                va, te, (m, imp, cols) = fit_rf(Xtr_te, ytr, Xva_te, Xte_te, s)
                Xval_n = to_numeric_frame(Xval_te).reindex(columns=cols, fill_value=0)
                _seed_nonseeded_block(per, "rf", va, te, m.predict(imp.transform(Xval_n.values)), N_SEEDS_GBDT)
                print(f"    [seed={s}] rf done ({time()-t:.0f}s)", flush=True)

                t = time()
                va, te, (m, imp, sc, cols, _) = fit_ridge(Xtr_te, ytr, Xva_te, Xte_te, s)
                Xval_n = to_numeric_frame(Xval_te).reindex(columns=cols, fill_value=0)
                vv = m.predict(sc.transform(imp.transform(Xval_n.values)))
                _seed_nonseeded_block(per, "ridge", va, te, vv, N_SEEDS_GBDT)
                print(f"    [seed={s}] ridge done ({time()-t:.0f}s)", flush=True)

                t = time()
                va, te, (m, imp, sc, cols) = fit_mlp(Xtr_te, ytr, Xva_te, Xte_te, s)
                Xval_n = to_numeric_frame(Xval_te).reindex(columns=cols, fill_value=0)
                vv = m.predict(sc.transform(imp.transform(Xval_n.values)))
                _seed_nonseeded_block(per, "mlp", va, te, vv, N_SEEDS_GBDT)
                print(f"    [seed={s}] mlp done ({time()-t:.0f}s)", flush=True)

        # --- NN block (each NN seed) ---
        for s in seeds_nn:
            t = time()
            va, te, vv, _ = fit_ft_transformer(Xtr_te, ytr, Xva_te, yva, Xte_te, s,
                                                Xval_extra=Xval_te)
            per["ft"]["va"] += va; per["ft"]["te"] += te; per["ft"]["vv"] += vv
            print(f"    [seed={s}] ft done ({time()-t:.0f}s)", flush=True)

            t = time()
            va, te, vv, _ = fit_tab_mlp_deep(Xtr_te, ytr, Xva_te, yva, Xte_te, s,
                                              Xval_extra=Xval_te)
            per["mlp_deep"]["va"] += va; per["mlp_deep"]["te"] += te; per["mlp_deep"]["vv"] += vv
            print(f"    [seed={s}] mlp_deep done ({time()-t:.0f}s)", flush=True)

        # Average over seeds
        for b in BASES:
            divisor = (
                N_SEEDS_NN if b in ("ft", "mlp_deep")
                else N_SEEDS_GBDT  # ("rf","ridge","mlp" pre-scaled; tree models naturally summed)
            )
            per[b]["va"] /= divisor; per[b]["te"] /= divisor; per[b]["vv"] /= divisor

            agg["oof"][b][va_idx] = per[b]["va"]
            agg["val_pred"][b] += per[b]["vv"] / N_FOLDS
            agg["test_pred"][b] += per[b]["te"] / N_FOLDS

        msg = "  fold {}: ".format(fold_id + 1) + " ".join(
            f"{b}={np.sqrt(mean_squared_error(yva, per[b]['va'])):.3f}" for b in BASES
        ) + f"  ({time()-t_fold:.0f}s)"
        print(msg, flush=True)

        agg["completed_folds"].append(fold_id)
        _save_aggregator(agg)
        print(f"    [ckpt] fold {fold_id+1}/{N_FOLDS} saved", flush=True)

    return agg
