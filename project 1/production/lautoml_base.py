"""LightAutoML wrapper as an additional base learner.

LightAutoML's TabularAutoML internally runs its own multi-level stacking
(LGBM + linear + denoising-AE + blending), so adding it as a single base
learner in our outer stack adds genuinely orthogonal signal compared to our
hand-rolled GBDTs.

Generates OOF / val / test predictions and appends `lautoml` to the
aggregate.pkl base list.
"""
from __future__ import annotations

import pickle
import warnings
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from production.config import (
    CHECKPOINTS_DIR, EMP_TITLE_TOP_N, HOLDOUT_FRAC, N_FOLDS, RANDOM_STATE,
)
from production.encoders import add_te_columns
from production.features import compact_emp_title, engineer, load_raw

warnings.filterwarnings("ignore")

# Time budget per LightAutoML fold (seconds). LightAutoML uses this as a hard
# cap on its internal stacking. 1200s per fold * 5 folds = 1.7 hrs.
LAUTOML_TIMEOUT_PER_FOLD = 1200


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


def _fit_lautoml_fold(Xtr, ytr, Xva, Xval, Xte, fold_id):
    """Fit one LightAutoML TabularAutoML on (Xtr, ytr); predict on Xva, Xval, Xte."""
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task

    df_tr = Xtr.copy()
    df_tr["int_rate"] = ytr

    task = Task("reg", loss="mse", metric="mse")
    automl = TabularAutoML(
        task=task,
        timeout=LAUTOML_TIMEOUT_PER_FOLD,
        cpu_limit=-1,
        reader_params={"n_jobs": 4, "cv": 5, "random_state": RANDOM_STATE + fold_id},
        general_params={
            "use_algos": [["lgb", "lgb_tuned", "cb", "linear_l2"]],
            "nested_cv": False,
        },
    )

    _ = automl.fit_predict(df_tr, roles={"target": "int_rate"}, verbose=0)

    va_pred = automl.predict(Xva).data.ravel()
    val_pred = automl.predict(Xval).data.ravel()
    test_pred = automl.predict(Xte).data.ravel()
    return va_pred, val_pred, test_pred


def run():
    print("=" * 72)
    print("LIGHTAUTOML PHASE -> lautoml base learner")
    print("=" * 72, flush=True)

    try:
        import lightautoml  # noqa: F401
    except ImportError:
        print("[lautoml] LightAutoML not installed. Install with:")
        print("    .venv/Scripts/pip install lightautoml")
        raise

    agg_path = CHECKPOINTS_DIR / "aggregate.pkl"
    if not agg_path.exists():
        raise RuntimeError("Run oof.py first; aggregate.pkl required")
    with open(agg_path, "rb") as f:
        agg = pickle.load(f)

    X_train, X_val, y_train, y_val, X_test = _prepare_data()
    print(f"  base: train={X_train.shape}  val={X_val.shape}  test={X_test.shape}", flush=True)

    n_tr = len(X_train); n_va = len(X_val); n_te = len(X_test)
    lautoml_oof = np.zeros(n_tr)
    lautoml_val = np.zeros(n_va)
    lautoml_test = np.zeros(n_te)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        t = time()
        Xtr = X_train.iloc[tr_idx].reset_index(drop=True)
        Xva = X_train.iloc[va_idx].reset_index(drop=True)
        ytr = y_train[tr_idx]
        yva = y_train[va_idx]
        Xtr_te, Xva_te, Xte_te = add_te_columns(Xtr, ytr, Xva, X_test)
        _, Xval_te, _ = add_te_columns(Xtr, ytr, X_val, X_test)
        va_pred, val_pred, test_pred = _fit_lautoml_fold(
            Xtr_te, ytr, Xva_te, Xval_te, Xte_te, fold_id,
        )
        lautoml_oof[va_idx] = va_pred
        lautoml_val += val_pred / N_FOLDS
        lautoml_test += test_pred / N_FOLDS
        rmse = float(np.sqrt(mean_squared_error(yva, va_pred)))
        print(f"  fold {fold_id+1}/{N_FOLDS}: lautoml={rmse:.3f}  ({time()-t:.0f}s)", flush=True)

    if "lautoml" not in agg["BASES"]:
        agg["BASES"] = list(agg["BASES"]) + ["lautoml"]
    agg["oof"]["lautoml"] = lautoml_oof
    agg["val_pred"]["lautoml"] = lautoml_val
    agg["test_pred"]["lautoml"] = lautoml_test
    with open(agg_path, "wb") as f:
        pickle.dump(agg, f)

    val_rmse = float(np.sqrt(mean_squared_error(agg["y_val"], lautoml_val)))
    print(f"\n[lautoml] held-out val RMSE = {val_rmse:.4f}", flush=True)
    print(f"[lautoml] aggregate.pkl now has {len(agg['BASES'])} base learners", flush=True)
    return agg


if __name__ == "__main__":
    run()
