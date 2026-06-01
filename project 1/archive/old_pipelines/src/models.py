"""Model configurations and fit/predict helpers.

XGBoost params are FROZEN at the v1-tuned values — XGB serves as the diversity
contributor in the blend, not the primary, and prior ablation showed retuning
it on v2 features made things worse. CatBoost is the primary and gets Optuna
tuning (search space defined here, optimization driven from train.py).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data import RANDOM_STATE
from src.features import CATEGORICAL_COLS

NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 75

# Frozen v1 XGBoost params (sourced from the prior Optuna study that produced
# OOF RMSE ≈ 3.903). Do NOT retune.
HISTORICAL_V1_XGB_PARAMS: dict[str, Any] = {
    "learning_rate": 0.010188389076377652,
    "max_depth": 8,
    "min_child_weight": 26.66728772081506,
    "subsample": 0.935054932261084,
    "colsample_bytree": 0.6964907784550468,
    "reg_alpha": 8.928086127845225e-07,
    "reg_lambda": 23.816212859455263,
    "gamma": 0.21854027389704697,
    "max_cat_to_onehot": 5,
}


def _xgb_base_params(seed: int = RANDOM_STATE) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": seed,
        "nthread": max(os.cpu_count() or 1, 1),
    }


def _make_dmatrix(X: pd.DataFrame, y: np.ndarray | None = None) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, enable_categorical=True)


def _predict_best_iter(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    best = getattr(booster, "best_iteration", None)
    dmat = _make_dmatrix(X)
    if best is None:
        return booster.predict(dmat)
    return booster.predict(dmat, iteration_range=(0, int(best) + 1))


def fit_xgb(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int = RANDOM_STATE,
) -> xgb.Booster:
    """Fit a single XGBoost fold with the frozen v1 params."""
    return xgb.train(
        params={**_xgb_base_params(seed), **HISTORICAL_V1_XGB_PARAMS},
        dtrain=_make_dmatrix(X_fit, y_fit),
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(_make_dmatrix(X_val, y_val), "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )


def predict_xgb(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return _predict_best_iter(booster, X)


def _to_catboost_frame(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost wants categorical columns as string with no nulls."""
    out = X.copy()
    for col in CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("missing")
    return out


def catboost_search_space(trial: "Any") -> dict[str, Any]:
    """Optuna search space for CatBoost regressor."""
    return {
        "iterations": trial.suggest_int("iterations", 1500, 4000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 254),
    }


def fit_catboost(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
    seed: int = RANDOM_STATE,
) -> "Any":
    """Fit a single CatBoost fold with the provided params."""
    from catboost import CatBoostRegressor

    X_fit_cb = _to_catboost_frame(X_fit)
    X_val_cb = _to_catboost_frame(X_val)
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_fit_cb.columns]

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        od_type="Iter",
        od_wait=EARLY_STOPPING_ROUNDS,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
        **params,
    )
    model.fit(
        X_fit_cb,
        y_fit,
        cat_features=cat_cols,
        eval_set=(X_val_cb, y_val),
        use_best_model=True,
    )
    return model


def predict_catboost(model: "Any", X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(_to_catboost_frame(X)))
