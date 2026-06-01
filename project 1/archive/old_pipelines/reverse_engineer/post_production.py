"""Post-iteration: run production's full 12-base stack on the winning iter's
enriched data. The per-iter CB-only baseline showed 3.92 but production's
stack hits 3.84 on raw data — applying it on top of the 14 aux features
SHOULD give us the actual lift the reverse-engineer approach can deliver.

Reuses production/oof.run + stack.build_and_select but with the enriched
dataframes as input. We bypass production's data-loading code by monkey-
patching the loader.
"""
from __future__ import annotations

import io
import json
import pickle
import sys
from pathlib import Path
from time import time

# Force UTF-8 stdout for Windows
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from reverse_engineer.config import OUTPUTS_DIR
from reverse_engineer.archive_loader import load_true_data

ITER_DIR = OUTPUTS_DIR / "iter_1"   # winning iteration from convergence
POST_OUT_DIR = OUTPUTS_DIR / "post_production"
POST_OUT_DIR.mkdir(parents=True, exist_ok=True)
POST_CKPT_DIR = POST_OUT_DIR / "checkpoints"
POST_CKPT_DIR.mkdir(parents=True, exist_ok=True)


def _load_enriched():
    xtr = pd.read_parquet(ITER_DIR / "X_train_enriched.parquet")
    xva = pd.read_parquet(ITER_DIR / "X_val_enriched.parquet")
    xte = pd.read_parquet(ITER_DIR / "X_test_enriched.parquet")
    y_tr = xtr["int_rate"].values.astype(np.float64)
    y_va = xva["int_rate"].values.astype(np.float64)
    xtr = xtr.drop(columns=["int_rate"])
    xva = xva.drop(columns=["int_rate"])
    return xtr, xva, xte, y_tr, y_va


def _run_engineered_stack():
    """Apply production.features.engineer() on the enriched data, then run
    the full OOF + stack pipeline manually."""
    from production import features as pfeat, encoders, fitters, stack
    from production.config import RANDOM_STATE, N_FOLDS, N_SEEDS_GBDT
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error

    print("=" * 72)
    print("POST-PRODUCTION: full 12-base stack on enriched data (iter_1)")
    print("=" * 72, flush=True)

    xtr_raw, xva_raw, xte_raw, y_tr, y_va = _load_enriched()
    print(f"  loaded enriched: train={xtr_raw.shape}  val={xva_raw.shape}  test={xte_raw.shape}", flush=True)

    # Apply production feature engineering on top of the enriched data
    print("  Running production.features.engineer() to add 135 engineered cols...", flush=True)
    t = time()
    # Re-attach int_rate temporarily because engineer() doesn't touch it (it's just kept untouched)
    # Then drop after. Engineer expects the raw schema, so the aux_* columns pass through unchanged.
    xtr_eng = pfeat.engineer(xtr_raw)
    xva_eng = pfeat.engineer(xva_raw)
    xte_eng = pfeat.engineer(xte_raw)
    # Compact emp_title
    xtr_eng, xte_eng = pfeat.compact_emp_title(xtr_eng, xte_eng, top_n=200)
    xva_eng, _ = pfeat.compact_emp_title(xva_eng, xva_eng.copy(), top_n=200)
    print(f"  enriched + engineered cols: train={xtr_eng.shape[1]}  ({time()-t:.0f}s)", flush=True)

    aux_cols = [c for c in xtr_eng.columns if c.startswith("aux_")]
    print(f"  aux cols preserved: {len(aux_cols)}", flush=True)

    # Build OOF aggregator manually (we can't reuse production.oof.run because
    # it has its own load_true_data hardcoded).
    from production.fitters import (
        fit_catboost, fit_lgbm, fit_lgbm_huber, fit_mlp, fit_rf, fit_ridge, fit_xgb,
    )
    from production.nn_models import fit_ft_transformer, fit_tab_mlp_deep
    from production.encoders import (
        add_te_columns, build_cat_dtypes, to_catboost_pool, to_lgb_frame,
        to_numeric_frame, to_xgb_frame,
    )

    # Load tuning params from production
    tuning_path = PROJECT_ROOT / "outputs" / "production" / "tuning_v2.json"
    tuning = json.loads(tuning_path.read_text())["params"]

    BASES = ["cb", "lgb", "xgb", "lgb_huber", "rf", "ridge", "mlp", "ft", "mlp_deep"]
    n_tr = len(xtr_eng); n_va = len(xva_eng); n_te = len(xte_eng)
    oof = {b: np.zeros(n_tr) for b in BASES}
    val_pred = {b: np.zeros(n_va) for b in BASES}
    test_pred = {b: np.zeros(n_te) for b in BASES}

    # Use full y_full = concat(y_tr, y_va) because production used train_test_split
    # internally. Here we already have the split — use xtr_eng/y_tr for OOF.
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    seeds_gbdt = [RANDOM_STATE + 7 * i for i in range(N_SEEDS_GBDT)]
    seeds_nn = [RANDOM_STATE + 13 * i for i in range(2)]

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(xtr_eng)):
        t_fold = time()
        Xtr_f = xtr_eng.iloc[tr_idx]; ytr_f = y_tr[tr_idx]
        Xva_f = xtr_eng.iloc[va_idx]; yva_f = y_tr[va_idx]

        Xtr_te, Xva_te, Xte_te = add_te_columns(Xtr_f, ytr_f, Xva_f, xte_eng)
        _, Xval_te, _ = add_te_columns(Xtr_f, ytr_f, xva_eng, xte_eng)
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
                per["rf"]["va"] += va * N_SEEDS_GBDT
                per["rf"]["te"] += te * N_SEEDS_GBDT
                per["rf"]["vv"] += vv * N_SEEDS_GBDT
                print(f"    [s={s}] rf done ({time()-t:.0f}s)", flush=True)

                t = time()
                va, te, (m, imp, sc, cols, _) = fit_ridge(Xtr_te, ytr_f, Xva_te, Xte_te, s)
                Xval_n = to_numeric_frame(Xval_te).reindex(columns=cols, fill_value=0)
                vv = m.predict(sc.transform(imp.transform(Xval_n.values)))
                per["ridge"]["va"] += va * N_SEEDS_GBDT
                per["ridge"]["te"] += te * N_SEEDS_GBDT
                per["ridge"]["vv"] += vv * N_SEEDS_GBDT
                print(f"    [s={s}] ridge done ({time()-t:.0f}s)", flush=True)

                t = time()
                va, te, (m, imp, sc, cols) = fit_mlp(Xtr_te, ytr_f, Xva_te, Xte_te, s)
                Xval_n = to_numeric_frame(Xval_te).reindex(columns=cols, fill_value=0)
                vv = m.predict(sc.transform(imp.transform(Xval_n.values)))
                per["mlp"]["va"] += va * N_SEEDS_GBDT
                per["mlp"]["te"] += te * N_SEEDS_GBDT
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

        from sklearn.metrics import mean_squared_error
        msg = "  fold {}: ".format(fold_id + 1) + " ".join(
            f"{b}={np.sqrt(mean_squared_error(yva_f, per[b]['va'])):.3f}" for b in BASES
        ) + f"  ({time()-t_fold:.0f}s)"
        print(msg, flush=True)

        # Checkpoint per fold
        ckpt = {
            "BASES": BASES, "completed_fold": fold_id,
            "oof": {b: oof[b].copy() for b in BASES},
            "val_pred": {b: val_pred[b].copy() for b in BASES},
            "test_pred": {b: test_pred[b].copy() for b in BASES},
            "y_train": y_tr, "y_val": y_va,
            "test_ids": np.arange(n_te),
        }
        pickle.dump(ckpt, open(POST_CKPT_DIR / "aggregate.pkl", "wb"))

    # Build the aggregator dict that stack.build_and_select expects
    agg = {
        "BASES": BASES,
        "oof": oof,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "y_train": y_tr,
        "y_val": y_va,
        "test_ids": np.arange(n_te),
    }

    print("\n  Building meta-learner stack...", flush=True)
    result = stack.build_and_select(agg)
    print("\n=== POST-PRODUCTION RANKING ===")
    print(result["ranking"].to_string(index=False))
    print(f"\nWinner: {result['winner_name']}")
    best_rmse = float(result["ranking"].iloc[0]["val_RMSE"])
    print(f"Held-out val RMSE: {best_rmse:.4f}")

    # Compare to baselines
    print("\nComparison:")
    print(f"  production/ (no aux): 3.8365")
    print(f"  reverse_engineer iter 1 (CB-only): 3.9234")
    print(f"  post_production (12-base on enriched): {best_rmse:.4f}")

    # Save submission + artifacts
    final_test = np.clip(result["winner_test_pred"], 6.0, 31.0)
    sub = pd.DataFrame({"ID": np.arange(1, len(final_test) + 1), "int_rate": final_test})
    # Use real test IDs
    _, _, _, test_ids = load_true_data()
    sub["ID"] = test_ids.values
    sub_path = POST_OUT_DIR / "test_predictions_post_production.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission: {len(sub):,} rows -> {sub_path}")

    pickle.dump({
        "ranking": result["ranking"].to_dict("records"),
        "winner": result["winner_name"],
        "winner_val_rmse": best_rmse,
        "agg": agg,
        "stacker_weights": result["ridge_stack"]["weights"],
    }, open(POST_OUT_DIR / "post_artifacts.pkl", "wb"))
    result["ranking"].to_csv(POST_OUT_DIR / "ranking.csv", index=False)


if __name__ == "__main__":
    _run_engineered_stack()
