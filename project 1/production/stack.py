"""Meta-learning layer: Ridge stacker, ElasticNet stacker, LightGBM stacker,
and a hill-climb weighted blend. The final submission uses whichever meta
strategy minimises held-out X_val RMSE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from production.config import RANDOM_STATE


def _hill_climb(P, y, steps=10000, lr=0.01, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = P.shape[1]
    w = np.ones(n) / n
    best = float(np.sqrt(mean_squared_error(y, P @ w)))
    for _ in range(steps):
        i = rng.integers(0, n)
        cand = w.copy()
        cand[i] += rng.uniform(-lr, lr)
        cand = np.clip(cand, 0, None)
        if cand.sum() <= 0:
            continue
        cand /= cand.sum()
        rmse = float(np.sqrt(mean_squared_error(y, P @ cand)))
        if rmse < best:
            best, w = rmse, cand
    return w, best


def _fit_ridge_stack(stack_tr, y_tr, alphas=(0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0)):
    best_a, best_rmse = None, np.inf
    for a in alphas:
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        rmses = []
        for tr, va in kf.split(stack_tr):
            r = Ridge(alpha=a, positive=True).fit(stack_tr[tr], y_tr[tr])
            rmses.append(np.sqrt(mean_squared_error(y_tr[va], r.predict(stack_tr[va]))))
        if np.mean(rmses) < best_rmse:
            best_rmse, best_a = float(np.mean(rmses)), a
    return Ridge(alpha=best_a, positive=True).fit(stack_tr, y_tr), best_a, best_rmse


def _fit_en_stack(stack_tr, y_tr,
                  alphas=(0.001, 0.01, 0.1, 1.0),
                  l1_ratios=(0.1, 0.3, 0.5, 0.7, 0.9)):
    best_params, best_rmse, best_model = None, np.inf, None
    for a in alphas:
        for l1 in l1_ratios:
            kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            rmses = []
            for tr, va in kf.split(stack_tr):
                m = ElasticNet(alpha=a, l1_ratio=l1, positive=True,
                               max_iter=5000, random_state=RANDOM_STATE)
                m.fit(stack_tr[tr], y_tr[tr])
                rmses.append(np.sqrt(mean_squared_error(y_tr[va], m.predict(stack_tr[va]))))
            mean = float(np.mean(rmses))
            if mean < best_rmse:
                best_rmse, best_params = mean, (a, l1)
    a, l1 = best_params
    m = ElasticNet(alpha=a, l1_ratio=l1, positive=True,
                   max_iter=5000, random_state=RANDOM_STATE).fit(stack_tr, y_tr)
    return m, best_params, best_rmse


def _fit_lgb_stack(stack_tr, y_tr):
    """Non-linear meta-learner. Lightly tuned LightGBM on the OOF predictions."""
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmses = []
    for tr, va in kf.split(stack_tr):
        m = LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=15, max_depth=4,
            min_child_samples=30, feature_fraction=0.9, bagging_fraction=0.9,
            bagging_freq=1, reg_alpha=0.1, reg_lambda=0.1,
            objective="regression", metric="rmse",
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        )
        m.fit(stack_tr[tr], y_tr[tr])
        rmses.append(np.sqrt(mean_squared_error(y_tr[va], m.predict(stack_tr[va]))))
    m = LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=15, max_depth=4,
        min_child_samples=30, feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=1, reg_alpha=0.1, reg_lambda=0.1,
        objective="regression", metric="rmse",
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    ).fit(stack_tr, y_tr)
    return m, float(np.mean(rmses))


def build_and_select(agg):
    """Train all meta-learners and pick the one minimising val RMSE.

    Also reports per-base RMSE so the caller can decide if a single base
    learner actually beats the blend (the lendingclub_project's lesson)."""
    BASES = agg["BASES"]
    y_tr = agg["y_train"]
    y_va = agg["y_val"]

    stack_tr = np.column_stack([agg["oof"][b] for b in BASES])
    stack_va = np.column_stack([agg["val_pred"][b] for b in BASES])
    stack_te = np.column_stack([agg["test_pred"][b] for b in BASES])

    contenders = []

    # Each base learner is a contender too
    for i, b in enumerate(BASES):
        contenders.append({
            "name": f"base_{b}",
            "val_pred": stack_va[:, i],
            "test_pred": stack_te[:, i],
        })

    # Ridge stacker
    ridge_m, ridge_a, ridge_cv = _fit_ridge_stack(stack_tr, y_tr)
    contenders.append({
        "name": f"ridge_stack(alpha={ridge_a})",
        "val_pred": ridge_m.predict(stack_va),
        "test_pred": ridge_m.predict(stack_te),
        "weights": dict(zip(BASES, ridge_m.coef_.tolist())),
    })

    # ElasticNet stacker
    en_m, en_params, en_cv = _fit_en_stack(stack_tr, y_tr)
    contenders.append({
        "name": f"en_stack(alpha={en_params[0]}, l1={en_params[1]})",
        "val_pred": en_m.predict(stack_va),
        "test_pred": en_m.predict(stack_te),
        "weights": dict(zip(BASES, en_m.coef_.tolist())),
    })

    # LightGBM stacker
    lgb_m, lgb_cv = _fit_lgb_stack(stack_tr, y_tr)
    contenders.append({
        "name": "lgb_stack",
        "val_pred": lgb_m.predict(stack_va),
        "test_pred": lgb_m.predict(stack_te),
    })

    # Hill-climb blend
    hc_w, hc_rmse = _hill_climb(stack_tr, y_tr)
    contenders.append({
        "name": "hill_climb_blend",
        "val_pred": stack_va @ hc_w,
        "test_pred": stack_te @ hc_w,
        "weights": dict(zip(BASES, hc_w.tolist())),
    })

    # Mean of top-k stackers (rank-averaged ensemble of ensembles)
    rs_val, en_val, lgb_val, hc_val = (
        ridge_m.predict(stack_va), en_m.predict(stack_va),
        lgb_m.predict(stack_va), stack_va @ hc_w,
    )
    rs_te, en_te, lgb_te, hc_te = (
        ridge_m.predict(stack_te), en_m.predict(stack_te),
        lgb_m.predict(stack_te), stack_te @ hc_w,
    )
    contenders.append({
        "name": "mean_of_meta_learners",
        "val_pred": (rs_val + en_val + lgb_val + hc_val) / 4,
        "test_pred": (rs_te + en_te + lgb_te + hc_te) / 4,
    })

    # Score all contenders on held-out val
    rows = []
    for c in contenders:
        rmse = float(np.sqrt(mean_squared_error(y_va, c["val_pred"])))
        mae = float(mean_absolute_error(y_va, c["val_pred"]))
        r2 = float(r2_score(y_va, c["val_pred"]))
        rows.append({"model": c["name"], "val_RMSE": rmse, "val_MAE": mae, "val_R2": r2})
    rank = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)

    # Pick winner by val RMSE
    best_name = rank.iloc[0]["model"]
    best_contender = next(c for c in contenders if c["name"] == best_name)

    return {
        "ranking": rank,
        "winner_name": best_name,
        "winner_val_pred": best_contender["val_pred"],
        "winner_test_pred": best_contender["test_pred"],
        "winner_weights": best_contender.get("weights"),
        "ridge_stack": {"alpha": ridge_a, "cv_rmse": ridge_cv,
                        "weights": dict(zip(BASES, ridge_m.coef_.tolist()))},
        "en_stack": {"params": en_params, "cv_rmse": en_cv,
                     "weights": dict(zip(BASES, en_m.coef_.tolist()))},
        "lgb_stack": {"cv_rmse": lgb_cv},
        "hill_climb": {"weights": dict(zip(BASES, hc_w.tolist())), "train_rmse": hc_rmse},
    }
