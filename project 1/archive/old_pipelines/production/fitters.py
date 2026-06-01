"""Per-model fit-predict helpers used by oof.py.

Each fitter trains a single seed on (Xtr, ytr), early-stops on (Xva, yva)
when applicable, and returns:
  (val_pred, test_pred, fitted_model)
The caller is responsible for averaging across seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation as lgb_log
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from production.config import RANDOM_STATE
from production.encoders import (
    build_cat_dtypes, to_catboost_pool, to_lgb_frame, to_xgb_frame, to_numeric_frame,
)


def _monotone_vec(cols, kind="lgbm"):
    """+1 increasing in target, -1 decreasing, 0 unconstrained.
    Domain priors: int_rate ↓ in FICO, ↑ in DTI, ↑ in term, ↑ in util."""
    d = {
        "fico": -1, "fico_band": -1, "fico_band_fine": -1,
        "fico_log": -1, "fico_inv": +1, "fico_sq": -1,
        "fico_above_prime": -1, "fico_subprime": +1,
        "dti": +1, "dti_band": +1,
        "term_months": +1,
        "annual_inc": -1, "log1p_annual_inc": -1,
        "revol_util": +1, "all_util": +1, "util_max": +1, "util_maxed_out": +1,
        "revol_util_band": +1, "all_util_band": +1,
        "term_x_loan_amnt": +1, "term_x_dti": +1, "term_x_fico": -1,
        "payment_to_income": +1,
        "credit_file_age_yrs": -1, "credit_age_band": -1,
        # New monotone priors
        "composite_risk": +1,
        "total_debt_service_ratio": +1,
        "monthly_residual_income": -1,
        "residual_to_income": -1,
        "state_unemployment": +1,
        "inc_vs_state_median": -1,
        "log_inc_vs_state_median": -1,
        "col_adjusted_income": -1,
        "log_col_adjusted_income": -1,
        "loan_to_state_median": +1,
        "purpose_risk_score": +1,
        "recent_inq_burst": +1,
        "inq_density": +1,
        "inq_per_lifetime_acc": +1,
        "fico_inv_x_loan_amnt": +1,
        "fico_inv_x_term": +1,
    }
    return [d.get(c, 0) for c in cols]


# ----- CatBoost -----
def fit_catboost(Xtr, ytr, Xva, yva, Xte, params, seed):
    Xtr_p, cat_idx = to_catboost_pool(Xtr)
    Xva_p, _ = to_catboost_pool(Xva)
    Xte_p, _ = to_catboost_pool(Xte)
    m = CatBoostRegressor(
        **params, iterations=3000, loss_function="RMSE", eval_metric="RMSE",
        random_seed=seed, verbose=0, allow_writing_files=False, early_stopping_rounds=120,
    )
    m.fit(Xtr_p, ytr, cat_features=cat_idx, eval_set=(Xva_p, yva), verbose=False)
    return m.predict(Xva_p), m.predict(Xte_p), m


# ----- LightGBM -----
def fit_lgbm(Xtr, ytr, Xva, yva, Xte, params, seed):
    cat_dt = build_cat_dtypes(Xtr)
    Xtr_p, cat_cols = to_lgb_frame(Xtr, cat_dt)
    Xva_p, _ = to_lgb_frame(Xva, cat_dt)
    Xte_p, _ = to_lgb_frame(Xte, cat_dt)
    m = LGBMRegressor(
        **params, n_estimators=5000, objective="regression", metric="rmse",
        random_state=seed, verbose=-1, n_jobs=-1,
        monotone_constraints=_monotone_vec(Xtr_p.columns, "lgbm"),
    )
    m.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], categorical_feature=cat_cols,
          callbacks=[lgb_es(120, verbose=False), lgb_log(0)])
    return m.predict(Xva_p), m.predict(Xte_p), m


# ----- XGBoost -----
def fit_xgb(Xtr, ytr, Xva, yva, Xte, params, seed):
    cat_dt = build_cat_dtypes(Xtr)
    Xtr_p = to_xgb_frame(Xtr, cat_dt)
    Xva_p = to_xgb_frame(Xva, cat_dt)
    Xte_p = to_xgb_frame(Xte, cat_dt)
    mc = _monotone_vec(Xtr_p.columns, "xgb")
    m = XGBRegressor(
        **params, n_estimators=5000, objective="reg:squarederror", tree_method="hist",
        enable_categorical=True, random_state=seed, n_jobs=-1,
        monotone_constraints=tuple(mc), early_stopping_rounds=120,
    )
    m.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], verbose=False)
    return m.predict(Xva_p), m.predict(Xte_p), m


# ----- LightGBM Huber (robust to outliers) -----
def fit_lgbm_huber(Xtr, ytr, Xva, yva, Xte, params, seed, alpha=0.9):
    cat_dt = build_cat_dtypes(Xtr)
    Xtr_p, cat_cols = to_lgb_frame(Xtr, cat_dt)
    Xva_p, _ = to_lgb_frame(Xva, cat_dt)
    Xte_p, _ = to_lgb_frame(Xte, cat_dt)
    params_h = {k: v for k, v in params.items() if k not in ("objective", "metric")}
    m = LGBMRegressor(
        **params_h, n_estimators=5000,
        objective="huber", alpha=alpha, metric="rmse",
        random_state=seed, verbose=-1, n_jobs=-1,
    )
    m.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], categorical_feature=cat_cols,
          callbacks=[lgb_es(120, verbose=False), lgb_log(0)])
    return m.predict(Xva_p), m.predict(Xte_p), m


# ----- Random Forest -----
def fit_rf(Xtr, ytr, Xva, Xte, seed):
    Xtr_n = to_numeric_frame(Xtr)
    Xva_n = to_numeric_frame(Xva).reindex(columns=Xtr_n.columns, fill_value=0)
    Xte_n = to_numeric_frame(Xte).reindex(columns=Xtr_n.columns, fill_value=0)
    imp = SimpleImputer(strategy="median").fit(Xtr_n.values)
    Xtr_s = imp.transform(Xtr_n.values)
    Xva_s = imp.transform(Xva_n.values)
    Xte_s = imp.transform(Xte_n.values)
    m = RandomForestRegressor(
        n_estimators=300, max_depth=16, min_samples_leaf=15, n_jobs=-1,
        random_state=seed, max_features=0.5,
    )
    m.fit(Xtr_s, ytr)
    return m.predict(Xva_s), m.predict(Xte_s), (m, imp, list(Xtr_n.columns))


# ----- Ridge linear -----
def fit_ridge(Xtr, ytr, Xva, Xte, seed):
    Xtr_n = to_numeric_frame(Xtr)
    Xva_n = to_numeric_frame(Xva).reindex(columns=Xtr_n.columns, fill_value=0)
    Xte_n = to_numeric_frame(Xte).reindex(columns=Xtr_n.columns, fill_value=0)
    imp = SimpleImputer(strategy="median").fit(Xtr_n.values)
    sc = StandardScaler().fit(imp.transform(Xtr_n.values))
    Xtr_s = sc.transform(imp.transform(Xtr_n.values))
    Xva_s = sc.transform(imp.transform(Xva_n.values))
    Xte_s = sc.transform(imp.transform(Xte_n.values))
    best_a, best_rmse = 1.0, np.inf
    for a in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        rmses = []
        for tr, va in cv.split(Xtr_s):
            r = Ridge(alpha=a, random_state=seed).fit(Xtr_s[tr], ytr[tr])
            rmses.append(np.sqrt(mean_squared_error(ytr[va], r.predict(Xtr_s[va]))))
        if np.mean(rmses) < best_rmse:
            best_rmse, best_a = float(np.mean(rmses)), a
    m = Ridge(alpha=best_a, random_state=seed).fit(Xtr_s, ytr)
    return m.predict(Xva_s), m.predict(Xte_s), (m, imp, sc, list(Xtr_n.columns), best_a)


# ----- sklearn MLP -----
def fit_mlp(Xtr, ytr, Xva, Xte, seed):
    Xtr_n = to_numeric_frame(Xtr)
    Xva_n = to_numeric_frame(Xva).reindex(columns=Xtr_n.columns, fill_value=0)
    Xte_n = to_numeric_frame(Xte).reindex(columns=Xtr_n.columns, fill_value=0)
    imp = SimpleImputer(strategy="median").fit(Xtr_n.values)
    sc = StandardScaler().fit(imp.transform(Xtr_n.values))
    Xtr_s = sc.transform(imp.transform(Xtr_n.values))
    Xva_s = sc.transform(imp.transform(Xva_n.values))
    Xte_s = sc.transform(imp.transform(Xte_n.values))
    m = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
        learning_rate_init=2e-3, alpha=1e-4, batch_size=512, max_iter=60,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
        random_state=seed, verbose=False,
    )
    m.fit(Xtr_s, ytr)
    return m.predict(Xva_s), m.predict(Xte_s), (m, imp, sc, list(Xtr_n.columns))
