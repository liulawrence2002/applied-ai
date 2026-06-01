"""Fresh Optuna tuning for CatBoost, LightGBM, XGBoost.

Replaces the cached 30-trial study from `outputs/sota/tuning_results.json`
with longer 60-100 trial runs against an 80/20 inner split of the active
80% train slice. Each trial uses early stopping inside the model so the
true cost is sub-linear in n_estimators.

Tuning results are persisted to `outputs/production/tuning_v2.json`.
Re-running this script reuses the cache; delete the JSON to retune.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from time import time

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation as lgb_log
from xgboost import XGBRegressor

from production.config import (
    HOLDOUT_FRAC, OUTPUTS_DIR, RANDOM_STATE,
    N_TRIALS_CB, N_TRIALS_LGB, N_TRIALS_XGB,
)
from production.encoders import (
    add_te_columns, build_cat_dtypes,
    to_catboost_pool, to_lgb_frame, to_xgb_frame,
)
from production.features import compact_emp_title, engineer, load_raw
from production.config import EMP_TITLE_TOP_N

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

CACHE = OUTPUTS_DIR / "tuning_v2.json"


def _prepare_inner_split():
    """Load + engineer + apply outer 80/20 holdout, then carve an inner 85/15
    HPO split from the 80% training slice."""
    train_raw, _ = load_raw()
    train_fe = engineer(train_raw)
    train_fe, _ = compact_emp_title(train_fe, train_fe.copy(), EMP_TITLE_TOP_N)

    y_full = train_fe["int_rate"].values.astype(np.float64)
    X_full = train_fe.drop(columns=["int_rate"])
    X_train, _, y_train, _ = train_test_split(
        X_full, y_full, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE,
    )

    inner_tr, inner_va = train_test_split(
        np.arange(len(X_train)), test_size=0.15,
        random_state=RANDOM_STATE, stratify=X_train["fico_band"].fillna(-1),
    )
    Xtr = X_train.iloc[inner_tr].reset_index(drop=True)
    Xva = X_train.iloc[inner_va].reset_index(drop=True)
    ytr = y_train[inner_tr]
    yva = y_train[inner_va]
    # Target-encode against the inner training fold only
    Xtr_te, Xva_te, _ = add_te_columns(Xtr, ytr, Xva, Xtr.iloc[:0])
    return Xtr_te, ytr, Xva_te, yva


def cb_objective(Xtr, ytr, Xva, yva):
    def _obj(trial):
        params = {
            "iterations": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.10, log=True),
            "depth": trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 12.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 64, 254),
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": RANDOM_STATE,
            "verbose": 0,
            "allow_writing_files": False,
        }
        Xtr_p, cat_idx = to_catboost_pool(Xtr)
        Xva_p, _ = to_catboost_pool(Xva)
        m = CatBoostRegressor(**params, early_stopping_rounds=100)
        m.fit(Xtr_p, ytr, cat_features=cat_idx, eval_set=(Xva_p, yva), verbose=False)
        return float(np.sqrt(mean_squared_error(yva, m.predict(Xva_p))))
    return _obj


def lgb_objective(Xtr, ytr, Xva, yva):
    def _obj(trial):
        params = {
            "n_estimators": 4000,
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 300),
            "max_depth": trial.suggest_int("max_depth", -1, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 250),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 15.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 15.0, log=True),
            "objective": "regression",
            "metric": "rmse",
            "random_state": RANDOM_STATE,
            "verbose": -1,
            "n_jobs": -1,
        }
        cat_dt = build_cat_dtypes(Xtr)
        Xtr_p, cat_cols = to_lgb_frame(Xtr, cat_dt)
        Xva_p, _ = to_lgb_frame(Xva, cat_dt)
        m = LGBMRegressor(**params)
        m.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)],
              categorical_feature=cat_cols,
              callbacks=[lgb_es(100, verbose=False), lgb_log(0)])
        return float(np.sqrt(mean_squared_error(yva, m.predict(Xva_p))))
    return _obj


def xgb_objective(Xtr, ytr, Xva, yva):
    def _obj(trial):
        params = {
            "n_estimators": 4000,
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.10, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 15.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 15.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "early_stopping_rounds": 100,
        }
        cat_dt = build_cat_dtypes(Xtr)
        Xtr_p = to_xgb_frame(Xtr, cat_dt)
        Xva_p = to_xgb_frame(Xva, cat_dt)
        m = XGBRegressor(**params)
        m.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], verbose=False)
        return float(np.sqrt(mean_squared_error(yva, m.predict(Xva_p))))
    return _obj


def run():
    # Load any existing cached params (partial cache OK)
    if CACHE.exists():
        cached = json.loads(CACHE.read_text()).get("params", {})
        if all(k in cached for k in ("CatBoost", "LightGBM", "XGBoost")):
            print(f"[tune] full cache hit -> {CACHE}", flush=True)
            return cached
        else:
            print(f"[tune] partial cache: {list(cached.keys())} done; resuming", flush=True)
    else:
        cached = {}

    print("[tune] preparing inner split", flush=True)
    Xtr, ytr, Xva, yva = _prepare_inner_split()
    print(f"  Xtr={Xtr.shape}  Xva={Xva.shape}", flush=True)

    # warn_independent_sampling=False silences LightGBM/XGB warnings about
    # max_depth/num_leaves conditional dependencies. Sampling falls back to
    # RandomSampler for those parameters — quality impact is marginal.
    sampler = TPESampler(seed=RANDOM_STATE, multivariate=True, warn_independent_sampling=False)
    pruner = MedianPruner(n_warmup_steps=10)
    results = dict(cached)  # preserve already-tuned models

    for name, obj_factory, n_trials in [
        ("CatBoost", cb_objective, N_TRIALS_CB),
        ("LightGBM", lgb_objective, N_TRIALS_LGB),
        ("XGBoost", xgb_objective, N_TRIALS_XGB),
    ]:
        if name in cached:
            print(f"  [{name}] skipping (cached)", flush=True)
            continue
        t0 = time()
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                    study_name=f"prod_{name}")
        study.optimize(obj_factory(Xtr, ytr, Xva, yva), n_trials=n_trials, show_progress_bar=False)
        results[name] = study.best_params
        print(f"  [{name}] best val RMSE = {study.best_value:.4f}  ({time()-t0:.0f}s, {n_trials} trials)", flush=True)
        print(f"  [{name}] params: {study.best_params}", flush=True)
        # Persist incrementally so we don't lose partial progress
        CACHE.write_text(json.dumps({"params": results, "ntrials": {name: n_trials}}, indent=2))

    print(f"[tune] saved -> {CACHE}")
    return results


if __name__ == "__main__":
    run()
