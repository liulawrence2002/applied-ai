"""V3 orchestrator: rate-lookup → aux training → enrich → retune → 5-fold stack.

Each phase is checkpointed; re-running resumes from the latest cache.
"""
from __future__ import annotations

import io
import json
import pickle
import sys
import warnings
from pathlib import Path
from time import time

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from reverse_engineer.archive_loader import add_subgrade_ordinal, load_true_data
from reverse_engineer.v3 import aux_models_v3, enrich_v3, rate_lookup
from reverse_engineer.v3.aux_models_v3 import load_loan_with_dates, load_lct_with_dates
from reverse_engineer.v3.config_v3 import (
    AUX_MODELS_PKL, ENRICHED_V3_TEST, ENRICHED_V3_TRAIN, ENRICHED_V3_VAL,
    HOLDOUT_FRAC, N_TRIALS_CB_V3, N_TRIALS_LGB_V3, N_TRIALS_XGB_V3,
    RATE_LOOKUP_JSON, TUNING_V3_JSON, V2_TEST_PARQUET, V2_TRAIN_PARQUET,
    V2_VAL_PARQUET, V3_CKPT, V3_DIR,
)
from production import features as pfeat, stack
from production.config import N_FOLDS, N_SEEDS_GBDT, RANDOM_STATE
from production.encoders import (
    add_te_columns, build_cat_dtypes, to_catboost_pool, to_lgb_frame,
    to_numeric_frame, to_xgb_frame,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# PHASE 1: Build rate lookup from loan.csv
# ============================================================================
def phase1_lookup():
    print("\n[phase 1] Building rate lookup", flush=True)
    return rate_lookup.build_lookup(force=False)


# ============================================================================
# PHASE 2: Train v3 aux models (sg35, issue_year, within_grade_residual)
# ============================================================================
def phase2_aux(rl: dict):
    if AUX_MODELS_PKL.exists():
        print("\n[phase 2] aux_models_v3.pkl exists, loading", flush=True)
        with open(AUX_MODELS_PKL, "rb") as f:
            return pickle.load(f)

    print("\n[phase 2] Training v3 aux models", flush=True)
    t0 = time()

    print("  Loading archive sources (with issue_d)...", flush=True)
    from reverse_engineer.v3.config_v3 import LOAN_CSV_MAX_ROWS
    t = time()
    loan_df = load_loan_with_dates(LOAN_CSV_MAX_ROWS)
    print(f"    loan.csv: {loan_df.shape}  (has issue_d: {'issue_d' in loan_df.columns})  ({time()-t:.0f}s)", flush=True)
    t = time()
    lct_df = load_lct_with_dates()
    print(f"    LC_train.csv: {lct_df.shape}  (has issue_d: {'issue_d' in lct_df.columns})  ({time()-t:.0f}s)", flush=True)

    grade_mean = rl["grade_mean_rate"]

    aux_loan = aux_models_v3.train_v3_aux_for_source("loan", loan_df, grade_mean)
    aux_lct = aux_models_v3.train_v3_aux_for_source("lct", lct_df, grade_mean)
    aux = {"loan": aux_loan, "lct": aux_lct}

    with open(AUX_MODELS_PKL, "wb") as f:
        pickle.dump(aux, f)
    print(f"  saved -> {AUX_MODELS_PKL}  ({time()-t0:.0f}s total)", flush=True)
    return aux


# ============================================================================
# PHASE 3: Enrich v2 parquets with v3 features
# ============================================================================
def phase3_enrich(aux: dict, rl: dict):
    if all(p.exists() for p in (ENRICHED_V3_TRAIN, ENRICHED_V3_VAL, ENRICHED_V3_TEST)):
        print("\n[phase 3] enriched_v3 parquets exist, skipping", flush=True)
        return

    print("\n[phase 3] Building enriched_v3 frames", flush=True)
    t = time()

    xtr = pd.read_parquet(V2_TRAIN_PARQUET)
    xva = pd.read_parquet(V2_VAL_PARQUET)
    xte = pd.read_parquet(V2_TEST_PARQUET)
    y_tr = xtr["int_rate"].values.astype(np.float64)
    y_va = xva["int_rate"].values.astype(np.float64)
    xtr = xtr.drop(columns=["int_rate"])
    xva = xva.drop(columns=["int_rate"])
    print(f"  loaded v2: train={xtr.shape}  val={xva.shape}  test={xte.shape}", flush=True)

    xtr_v3 = enrich_v3.enrich_v3(xtr, aux, rl)
    xva_v3 = enrich_v3.enrich_v3(xva, aux, rl)
    xte_v3 = enrich_v3.enrich_v3(xte, aux, rl)
    print(f"  v3 cols: {xtr_v3.shape[1]} (+{xtr_v3.shape[1] - xtr.shape[1]} new)", flush=True)

    xtr_v3.assign(int_rate=y_tr).to_parquet(ENRICHED_V3_TRAIN)
    xva_v3.assign(int_rate=y_va).to_parquet(ENRICHED_V3_VAL)
    xte_v3.to_parquet(ENRICHED_V3_TEST)
    print(f"  saved v3 parquets ({time()-t:.0f}s)", flush=True)


# ============================================================================
# Helpers shared by phases 4 & 5
# ============================================================================
def _load_v3():
    xtr = pd.read_parquet(ENRICHED_V3_TRAIN)
    xva = pd.read_parquet(ENRICHED_V3_VAL)
    xte = pd.read_parquet(ENRICHED_V3_TEST)
    y_tr = xtr["int_rate"].values.astype(np.float64)
    y_va = xva["int_rate"].values.astype(np.float64)
    xtr = xtr.drop(columns=["int_rate"])
    xva = xva.drop(columns=["int_rate"])
    return xtr, xva, xte, y_tr, y_va


def _build_inner_split(xtr_v3, y_tr):
    inner_tr, inner_va = train_test_split(
        np.arange(len(xtr_v3)), test_size=0.15, random_state=RANDOM_STATE,
        stratify=xtr_v3["fico_band"].fillna(-1) if "fico_band" in xtr_v3.columns else None,
    )
    Xtr_i = xtr_v3.iloc[inner_tr].reset_index(drop=True)
    Xva_i = xtr_v3.iloc[inner_va].reset_index(drop=True)
    ytr_i = y_tr[inner_tr]
    yva_i = y_tr[inner_va]
    Xtr_te, Xva_te, _ = add_te_columns(Xtr_i, ytr_i, Xva_i, xtr_v3.iloc[:0])
    return Xtr_te, Xva_te, ytr_i, yva_i


# ============================================================================
# PHASE 4: Re-tune Optuna on enriched_v3
# ============================================================================
def phase4_tune():
    if TUNING_V3_JSON.exists():
        cached = json.loads(TUNING_V3_JSON.read_text()).get("params", {})
        if all(k in cached for k in ("CatBoost", "LightGBM", "XGBoost")):
            print(f"\n[phase 4] full tuning cache hit -> {TUNING_V3_JSON}", flush=True)
            return cached
    else:
        cached = {}

    print("\n[phase 4] Tuning Optuna on enriched_v3", flush=True)
    xtr_v3, _, _, y_tr, _ = _load_v3()
    Xtr, Xva, ytr, yva = _build_inner_split(xtr_v3, y_tr)
    print(f"  inner split: train={Xtr.shape}  val={Xva.shape}", flush=True)

    sampler = TPESampler(seed=RANDOM_STATE, multivariate=True, warn_independent_sampling=False)
    pruner = MedianPruner(n_warmup_steps=10)
    results = dict(cached)

    # ----- CatBoost -----
    if "CatBoost" not in results:
        from catboost import CatBoostRegressor
        def cb_obj(trial):
            params = {
                "iterations": 3000,
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.10, log=True),
                "depth": trial.suggest_int("depth", 5, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 12.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "border_count": trial.suggest_int("border_count", 64, 254),
                "loss_function": "RMSE", "eval_metric": "RMSE",
                "random_seed": RANDOM_STATE, "verbose": 0, "allow_writing_files": False,
            }
            Xtr_p, cat_idx = to_catboost_pool(Xtr)
            Xva_p, _ = to_catboost_pool(Xva)
            m = CatBoostRegressor(**params, early_stopping_rounds=100)
            m.fit(Xtr_p, ytr, cat_features=cat_idx, eval_set=(Xva_p, yva), verbose=False)
            return float(np.sqrt(mean_squared_error(yva, m.predict(Xva_p))))
        t = time()
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                    study_name="v3_CatBoost")
        study.optimize(cb_obj, n_trials=N_TRIALS_CB_V3, show_progress_bar=False)
        results["CatBoost"] = study.best_params
        print(f"  [CatBoost] best val RMSE = {study.best_value:.4f}  ({time()-t:.0f}s)", flush=True)
        TUNING_V3_JSON.write_text(json.dumps({"params": results}, indent=2))

    # ----- LightGBM -----
    if "LightGBM" not in results:
        from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation as lgb_log
        def lgb_obj(trial):
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
                "objective": "regression", "metric": "rmse",
                "random_state": RANDOM_STATE, "verbose": -1, "n_jobs": -1,
            }
            cat_dt = build_cat_dtypes(Xtr)
            Xtr_p, cat_cols = to_lgb_frame(Xtr, cat_dt)
            Xva_p, _ = to_lgb_frame(Xva, cat_dt)
            m = LGBMRegressor(**params)
            m.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], categorical_feature=cat_cols,
                  callbacks=[lgb_es(100, verbose=False), lgb_log(0)])
            return float(np.sqrt(mean_squared_error(yva, m.predict(Xva_p))))
        t = time()
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                    study_name="v3_LightGBM")
        study.optimize(lgb_obj, n_trials=N_TRIALS_LGB_V3, show_progress_bar=False)
        results["LightGBM"] = study.best_params
        print(f"  [LightGBM] best val RMSE = {study.best_value:.4f}  ({time()-t:.0f}s)", flush=True)
        TUNING_V3_JSON.write_text(json.dumps({"params": results}, indent=2))

    # ----- XGBoost -----
    if "XGBoost" not in results:
        from xgboost import XGBRegressor
        def xgb_obj(trial):
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
                "objective": "reg:squarederror", "tree_method": "hist",
                "enable_categorical": True, "random_state": RANDOM_STATE,
                "n_jobs": -1, "early_stopping_rounds": 100,
            }
            cat_dt = build_cat_dtypes(Xtr)
            Xtr_p = to_xgb_frame(Xtr, cat_dt)
            Xva_p = to_xgb_frame(Xva, cat_dt)
            m = XGBRegressor(**params)
            m.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], verbose=False)
            return float(np.sqrt(mean_squared_error(yva, m.predict(Xva_p))))
        t = time()
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                    study_name="v3_XGBoost")
        study.optimize(xgb_obj, n_trials=N_TRIALS_XGB_V3, show_progress_bar=False)
        results["XGBoost"] = study.best_params
        print(f"  [XGBoost] best val RMSE = {study.best_value:.4f}  ({time()-t:.0f}s)", flush=True)
        TUNING_V3_JSON.write_text(json.dumps({"params": results}, indent=2))

    return results


# ============================================================================
# PHASE 5: 5-fold OOF stack
# ============================================================================
def phase5_stack(tuning: dict):
    from production.fitters import (
        fit_catboost, fit_lgbm, fit_lgbm_huber, fit_mlp, fit_rf, fit_ridge, fit_xgb,
    )
    from production.nn_models import fit_ft_transformer, fit_tab_mlp_deep

    xtr_v3, xva_v3, xte_v3, y_tr, y_va = _load_v3()
    print(f"\n[phase 5] 5-fold OOF stack on enriched_v3: train={xtr_v3.shape}", flush=True)

    BASES = ["cb", "lgb", "xgb", "lgb_huber", "rf", "ridge", "mlp", "ft", "mlp_deep"]
    n_tr, n_va, n_te = len(xtr_v3), len(xva_v3), len(xte_v3)
    oof = {b: np.zeros(n_tr) for b in BASES}
    val_pred = {b: np.zeros(n_va) for b in BASES}
    test_pred = {b: np.zeros(n_te) for b in BASES}

    ckpt_path = V3_CKPT / "aggregate.pkl"
    completed_folds = []
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        completed_folds = ckpt.get("completed_folds", [])
        oof = {**oof, **ckpt.get("oof", {})}
        val_pred = {**val_pred, **ckpt.get("val_pred", {})}
        test_pred = {**test_pred, **ckpt.get("test_pred", {})}
        print(f"  resumed: completed folds = {completed_folds}", flush=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    seeds_gbdt = [RANDOM_STATE + 7 * i for i in range(N_SEEDS_GBDT)]
    seeds_nn = [RANDOM_STATE + 13 * i for i in range(2)]

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(xtr_v3)):
        if fold_id in completed_folds:
            print(f"  fold {fold_id+1}/{N_FOLDS} cached, skipping", flush=True)
            continue
        t_fold = time()
        Xtr_f = xtr_v3.iloc[tr_idx]; ytr_f = y_tr[tr_idx]
        Xva_f = xtr_v3.iloc[va_idx]; yva_f = y_tr[va_idx]
        Xtr_te, Xva_te, Xte_te = add_te_columns(Xtr_f, ytr_f, Xva_f, xte_v3)
        _, Xval_te, _ = add_te_columns(Xtr_f, ytr_f, xva_v3, xte_v3)
        cat_dt = build_cat_dtypes(Xtr_te)
        per = {b: {"va": np.zeros(len(va_idx)), "te": np.zeros(n_te), "vv": np.zeros(n_va)} for b in BASES}

        for s in seeds_gbdt:
            t = time()
            va, te, m = fit_catboost(Xtr_te, ytr_f, Xva_te, yva_f, Xte_te, tuning["CatBoost"], s)
            Xval_p, _ = to_catboost_pool(Xval_te)
            per["cb"]["va"] += va; per["cb"]["te"] += te; per["cb"]["vv"] += m.predict(Xval_p)
            print(f"    [s={s}] cb done ({time()-t:.0f}s)", flush=True)
            t = time()
            va, te, m = fit_lgbm(Xtr_te, ytr_f, Xva_te, yva_f, Xte_te, tuning["LightGBM"], s)
            Xval_p, _ = to_lgb_frame(Xval_te, cat_dt)
            per["lgb"]["va"] += va; per["lgb"]["te"] += te; per["lgb"]["vv"] += m.predict(Xval_p)
            print(f"    [s={s}] lgb done ({time()-t:.0f}s)", flush=True)
            t = time()
            va, te, m = fit_xgb(Xtr_te, ytr_f, Xva_te, yva_f, Xte_te, tuning["XGBoost"], s)
            Xval_p = to_xgb_frame(Xval_te, cat_dt)
            per["xgb"]["va"] += va; per["xgb"]["te"] += te; per["xgb"]["vv"] += m.predict(Xval_p)
            print(f"    [s={s}] xgb done ({time()-t:.0f}s)", flush=True)
            t = time()
            va, te, m = fit_lgbm_huber(Xtr_te, ytr_f, Xva_te, yva_f, Xte_te, tuning["LightGBM"], s)
            Xval_p, _ = to_lgb_frame(Xval_te, cat_dt)
            per["lgb_huber"]["va"] += va; per["lgb_huber"]["te"] += te; per["lgb_huber"]["vv"] += m.predict(Xval_p)
            print(f"    [s={s}] lgb_huber done ({time()-t:.0f}s)", flush=True)
            if s == seeds_gbdt[0]:
                t = time()
                va, te, (m, imp, cols) = fit_rf(Xtr_te, ytr_f, Xva_te, Xte_te, s)
                Xval_n = to_numeric_frame(Xval_te).reindex(columns=cols, fill_value=0)
                vv = m.predict(imp.transform(Xval_n.values))
                per["rf"]["va"] += va * N_SEEDS_GBDT; per["rf"]["te"] += te * N_SEEDS_GBDT
                per["rf"]["vv"] += vv * N_SEEDS_GBDT
                print(f"    [s={s}] rf done ({time()-t:.0f}s)", flush=True)
                t = time()
                va, te, (m, imp, sc, cols, _) = fit_ridge(Xtr_te, ytr_f, Xva_te, Xte_te, s)
                Xval_n = to_numeric_frame(Xval_te).reindex(columns=cols, fill_value=0)
                vv = m.predict(sc.transform(imp.transform(Xval_n.values)))
                per["ridge"]["va"] += va * N_SEEDS_GBDT; per["ridge"]["te"] += te * N_SEEDS_GBDT
                per["ridge"]["vv"] += vv * N_SEEDS_GBDT
                print(f"    [s={s}] ridge done ({time()-t:.0f}s)", flush=True)
                t = time()
                va, te, (m, imp, sc, cols) = fit_mlp(Xtr_te, ytr_f, Xva_te, Xte_te, s)
                Xval_n = to_numeric_frame(Xval_te).reindex(columns=cols, fill_value=0)
                vv = m.predict(sc.transform(imp.transform(Xval_n.values)))
                per["mlp"]["va"] += va * N_SEEDS_GBDT; per["mlp"]["te"] += te * N_SEEDS_GBDT
                per["mlp"]["vv"] += vv * N_SEEDS_GBDT
                print(f"    [s={s}] mlp done ({time()-t:.0f}s)", flush=True)

        for s in seeds_nn:
            t = time()
            va, te, vv, _ = fit_ft_transformer(Xtr_te, ytr_f, Xva_te, yva_f, Xte_te, s, Xval_extra=Xval_te)
            per["ft"]["va"] += va; per["ft"]["te"] += te; per["ft"]["vv"] += vv
            print(f"    [s={s}] ft done ({time()-t:.0f}s)", flush=True)
            t = time()
            va, te, vv, _ = fit_tab_mlp_deep(Xtr_te, ytr_f, Xva_te, yva_f, Xte_te, s, Xval_extra=Xval_te)
            per["mlp_deep"]["va"] += va; per["mlp_deep"]["te"] += te; per["mlp_deep"]["vv"] += vv
            print(f"    [s={s}] mlp_deep done ({time()-t:.0f}s)", flush=True)

        for b in BASES:
            div = (2 if b in ("ft", "mlp_deep") else N_SEEDS_GBDT)
            per[b]["va"] /= div; per[b]["te"] /= div; per[b]["vv"] /= div
            oof[b][va_idx] = per[b]["va"]
            val_pred[b] += per[b]["vv"] / N_FOLDS
            test_pred[b] += per[b]["te"] / N_FOLDS

        msg = "  fold {}: ".format(fold_id + 1) + " ".join(
            f"{b}={np.sqrt(mean_squared_error(yva_f, per[b]['va'])):.3f}" for b in BASES
        ) + f"  ({time()-t_fold:.0f}s)"
        print(msg, flush=True)
        completed_folds.append(fold_id)
        with open(ckpt_path, "wb") as f:
            pickle.dump({
                "completed_folds": completed_folds,
                "oof": oof, "val_pred": val_pred, "test_pred": test_pred,
                "y_train": y_tr, "y_val": y_va,
            }, f)
        print(f"  [ckpt] fold {fold_id+1}/{N_FOLDS} saved", flush=True)

    agg = {
        "BASES": BASES, "oof": oof, "val_pred": val_pred, "test_pred": test_pred,
        "y_train": y_tr, "y_val": y_va, "test_ids": np.arange(n_te),
    }
    print("\n  Building meta-learner stack...", flush=True)
    result = stack.build_and_select(agg)
    print("\n=== V3 RANKING ===")
    print(result["ranking"].to_string(index=False))
    print(f"\nWinner: {result['winner_name']}")
    best_rmse = float(result["ranking"].iloc[0]["val_RMSE"])
    print(f"Held-out val RMSE: {best_rmse:.4f}")
    print("\nComparison:")
    print(f"  production/ (no aux):                    3.8365")
    print(f"  reverse_engineer/post_production:        3.8331")
    print(f"  reverse_engineer/v2 (FICO×aux + retune): 3.8297")
    print(f"  reverse_engineer/v3 (this run):          {best_rmse:.4f}")

    final_test = np.clip(result["winner_test_pred"], 6.0, 31.0)
    _, _, _, test_ids = load_true_data()
    sub = pd.DataFrame({"ID": test_ids.values, "int_rate": final_test})
    sub.to_csv(V3_DIR / "test_predictions_v3.csv", index=False)
    print(f"\nSubmission: {len(sub):,} rows -> {V3_DIR / 'test_predictions_v3.csv'}")

    pickle.dump({
        "ranking": result["ranking"].to_dict("records"),
        "winner": result["winner_name"],
        "winner_val_rmse": best_rmse,
        "agg": agg,
    }, open(V3_DIR / "post_artifacts_v3.pkl", "wb"))
    result["ranking"].to_csv(V3_DIR / "ranking_v3.csv", index=False)


def main():
    t0 = time()
    print("=" * 72)
    print("REVERSE-ENGINEER V3 PIPELINE (35-class sg + rate lookup + issue_year + residual)")
    print("=" * 72, flush=True)
    rl = phase1_lookup()
    aux = phase2_aux(rl)
    phase3_enrich(aux, rl)
    tuning = phase4_tune()
    phase5_stack(tuning)
    print(f"\nTotal runtime: {time()-t0:.0f}s ({(time()-t0)/3600:.2f} hrs)")


if __name__ == "__main__":
    main()
