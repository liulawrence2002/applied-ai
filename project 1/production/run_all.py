"""Orchestrator: tune → OOF → stack → predict.

Phases are checkpointed independently:
  1. `outputs/production/tuning_v2.json`           (from tune.run())
  2. `outputs/production/checkpoints/aggregate.pkl` (from oof.run())
  3. `outputs/production/final_artifacts.pkl`      (from stack.build_and_select())

Re-running picks up where it left off. Safe to kill at any point.
"""
from __future__ import annotations

import io
import json
import pickle
import sys
from pathlib import Path
from time import time

# Force UTF-8 stdout/stderr on Windows so unicode chars in any nested print
# (e.g. progress bars) don't crash the pipeline mid-run.
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
except Exception:
    pass

# Make `production` importable when run as a script from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from production import oof, stack, tune
from production.config import (
    CHECKPOINTS_DIR, INT_RATE_MAX, INT_RATE_MIN, OUTPUTS_DIR,
)

# Optional extra base learners.
ENABLE_PSEUDO = True       # adds cb_pseudo (CatBoost trained on confident test pseudo-labels)
ENABLE_EXTRA = True        # adds hgb (HistGradientBoosting) + extra_trees
ENABLE_LAUTOML = False     # LightAutoML 0.4.2 doesn't install on Python 3.14 (needs xgboost<3)


def main():
    t0_all = time()

    print("=" * 72)
    print("PRODUCTION PIPELINE -- target val RMSE <= 3.55")
    print("=" * 72, flush=True)

    # --- Phase 1: Optuna tuning ---
    print("\n[phase 1/3] Optuna tuning", flush=True)
    t = time()
    tuning_params = tune.run()
    print(f"  done in {time()-t:.0f}s", flush=True)

    # --- Phase 2: OOF stacking ---
    print("\n[phase 2] K-fold OOF + multi-seed averaging", flush=True)
    t = time()
    agg = oof.run(tuning_params)
    print(f"  done in {time()-t:.0f}s", flush=True)

    # --- Phase 2b: Extra base learners (hgb + extra_trees) ---
    if ENABLE_EXTRA:
        print("\n[phase 2b] Extra bases -> hgb + extra_trees", flush=True)
        from production import extra_bases
        t = time()
        agg = extra_bases.run()
        print(f"  done in {time()-t:.0f}s", flush=True)

    # --- Phase 2c: Pseudo-labeling (CatBoost trained on confident test rows) ---
    if ENABLE_PSEUDO and "cb_pseudo" not in agg.get("BASES", []):
        print("\n[phase 2c] Pseudo-labeling -> cb_pseudo", flush=True)
        from production import pseudo_labels
        t = time()
        agg = pseudo_labels.run()
        print(f"  done in {time()-t:.0f}s", flush=True)
    elif ENABLE_PSEUDO:
        print("\n[phase 2c] cb_pseudo already in aggregate, skipping", flush=True)

    # --- Phase 2d: LightAutoML (disabled by default — see comment above) ---
    if ENABLE_LAUTOML and "lautoml" not in agg.get("BASES", []):
        print("\n[phase 2d] LightAutoML -> lautoml base learner", flush=True)
        try:
            from production import lautoml_base
            t = time()
            agg = lautoml_base.run()
            print(f"  done in {time()-t:.0f}s", flush=True)
        except Exception as e:
            print(f"  [skip] LightAutoML failed: {e}", flush=True)

    # --- Phase 3: Meta-learner selection ---
    print("\n[phase 3] Meta-learner stack + winner selection", flush=True)
    t = time()
    result = stack.build_and_select(agg)
    print(f"  done in {time()-t:.0f}s", flush=True)

    print("\n=== RANKING ===")
    print(result["ranking"].to_string(index=False))
    print(f"\nWinner: {result['winner_name']}")
    print(f"Held-out val RMSE: {result['ranking'].iloc[0]['val_RMSE']:.4f}")

    # --- Write submission ---
    final_test = np.clip(result["winner_test_pred"], INT_RATE_MIN, INT_RATE_MAX)
    sub = pd.DataFrame({"ID": agg["test_ids"], "int_rate": final_test})
    sub_path = OUTPUTS_DIR / "test_predictions_production.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission: {len(sub):,} rows -> {sub_path}")
    print(f"pred stats: mean={final_test.mean():.3f}  median={np.median(final_test):.3f}  "
          f"min={final_test.min():.3f}  max={final_test.max():.3f}")

    # --- Persist artifacts ---
    art = {
        "BASES": agg["BASES"],
        "oof_stack": np.column_stack([agg["oof"][b] for b in agg["BASES"]]),
        "val_stack": np.column_stack([agg["val_pred"][b] for b in agg["BASES"]]),
        "test_stack": np.column_stack([agg["test_pred"][b] for b in agg["BASES"]]),
        "y_train": agg["y_train"], "y_val": agg["y_val"],
        "test_ids": agg["test_ids"],
        "tuning_params": tuning_params,
        "ranking": result["ranking"].to_dict("records"),
        "winner_name": result["winner_name"],
        "winner_val_pred": result["winner_val_pred"],
        "winner_test_pred": result["winner_test_pred"],
        "winner_weights": result.get("winner_weights"),
        "ridge_stack": result["ridge_stack"],
        "en_stack": result["en_stack"],
        "lgb_stack": result["lgb_stack"],
        "hill_climb": result["hill_climb"],
    }
    art_path = OUTPUTS_DIR / "final_artifacts.pkl"
    with open(art_path, "wb") as f:
        pickle.dump(art, f)
    print(f"Artifacts: {art_path}")

    result["ranking"].to_csv(OUTPUTS_DIR / "ranking.csv", index=False)
    summary = {
        "winner": result["winner_name"],
        "winner_val_RMSE": float(result["ranking"].iloc[0]["val_RMSE"]),
        "ranking": result["ranking"].to_dict("records"),
        "ridge_stack_weights": result["ridge_stack"]["weights"],
        "en_stack_weights": result["en_stack"]["weights"],
        "hill_climb_weights": result["hill_climb"]["weights"],
        "total_runtime_sec": time() - t0_all,
    }
    (OUTPUTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary: {OUTPUTS_DIR / 'summary.json'}")
    print(f"\nTotal runtime: {time()-t0_all:.0f}s ({(time()-t0_all)/3600:.1f} hrs)")


if __name__ == "__main__":
    main()
