"""Orchestrator: run iterations 1..MAX_ITERATIONS with convergence check.

Per iteration:
  - aux ensemble training (CatBoost on archive_loan + archive_lct)
  - enrich true data with 14 aux columns
  - per-iter final model (multi-seed CatBoost K-fold on enriched true data)
  - persist iter artifacts under outputs/reverse_engineer/iter_k/

After iterations finish (convergence or max), pick the winning iter and
write the final submission CSV.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from time import time

# Force utf-8 stdout for Windows
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

from reverse_engineer import iterate
from reverse_engineer.config import (
    CONVERGENCE_THRESHOLD, MAX_ITERATIONS, OUTPUTS_DIR,
)

# Test-set clipping (matches production)
INT_RATE_MIN = 6.0
INT_RATE_MAX = 31.0


def main():
    t0_all = time()
    print("=" * 72)
    print("REVERSE-ENGINEER PIPELINE")
    print("=" * 72, flush=True)

    history = []
    prev_val_rmse = None
    prev_archive_predictions = None
    best = None
    stopped_reason = "max_iterations"

    for k in range(1, MAX_ITERATIONS + 1):
        result = iterate.run_one_iteration(
            iter_idx=k,
            prev_archive_predictions=prev_archive_predictions,
            prev_val_rmse=prev_val_rmse,
        )

        history.append({
            "iter": k,
            "val_RMSE": result["val_rmse"],
            "val_MAE": result["val_mae"],
            "val_R2": result["val_r2"],
            "delta_vs_prev": result["delta"],
            "n_enriched_cols": result["n_enriched_cols"],
            "runtime_sec": result["runtime_sec"],
        })

        # Save best
        if best is None or result["val_rmse"] < best["val_rmse"]:
            best = {
                "iter": k,
                "val_rmse": result["val_rmse"],
                "val_pred": result["val_pred"],
                "test_pred": result["test_pred"],
                "test_ids": result["test_ids"],
            }

        # Persist running summary
        summary = {
            "iter_history": history,
            "current_best_iter": best["iter"],
            "current_best_val_RMSE": float(best["val_rmse"]),
            "total_runtime_sec": time() - t0_all,
        }
        (OUTPUTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

        # Convergence check
        if prev_val_rmse is not None and result["delta"] is not None:
            if result["delta"] < CONVERGENCE_THRESHOLD:
                stopped_reason = "convergence"
                print(f"\n[converged] delta {result['delta']:.4f} < {CONVERGENCE_THRESHOLD}, stopping after iter {k}", flush=True)
                break

        prev_val_rmse = result["val_rmse"]
        prev_archive_predictions = result["archive_predictions"]

    # ----- Final submission from best iter -----
    final_test = np.clip(best["test_pred"], INT_RATE_MIN, INT_RATE_MAX)
    sub = pd.DataFrame({"ID": best["test_ids"], "int_rate": final_test})
    sub_path = OUTPUTS_DIR / "test_predictions_reverse_engineer.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission: {len(sub):,} rows -> {sub_path}")
    print(f"pred stats: mean={final_test.mean():.3f}  median={np.median(final_test):.3f}  "
          f"min={final_test.min():.3f}  max={final_test.max():.3f}")

    summary = {
        "iter_history": history,
        "winning_iter": best["iter"],
        "winning_val_RMSE": float(best["val_rmse"]),
        "total_runtime_sec": time() - t0_all,
        "stopped_reason": stopped_reason,
    }
    (OUTPUTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary: {OUTPUTS_DIR / 'summary.json'}")
    print(f"Total runtime: {time()-t0_all:.0f}s ({(time()-t0_all)/3600:.2f} hrs)")


if __name__ == "__main__":
    main()
