"""Regenerate the submission CSV from saved per-model test predictions.

This is a thin transformer: it reads the per-model test prediction files and
blend metadata that train.py writes, then applies the same blend weights,
optional rate-snap, and clipping to produce ``outputs/final_test_predictions.csv``.
It does NOT re-run any model — use ``python -m src.train`` for that.

Run: ``python -m src.predict``
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data import ID_COL, TARGET_COL, read_test, read_train
from src.train import (
    BLEND_META_PATH,
    CAT_TEST_PATH,
    FINAL_SUBMISSION_PATH,
    XGB_TEST_PATH,
)


def _snap_to_legal(values: np.ndarray, legal: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(legal, values)
    idx = np.clip(idx, 1, len(legal) - 1)
    left = legal[idx - 1]
    right = legal[idx]
    return np.where(np.abs(values - left) <= np.abs(values - right), left, right)


def main() -> None:
    for path in (CAT_TEST_PATH, XGB_TEST_PATH, BLEND_META_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {path}. Run `python -m src.train` first."
            )

    meta = json.loads(BLEND_META_PATH.read_text(encoding="utf-8"))
    weights = meta["weights"]
    apply_snap = bool(meta["apply_snap"])
    pred_clip = tuple(meta["pred_clip"])

    cat = pd.read_csv(CAT_TEST_PATH)
    xgb = pd.read_csv(XGB_TEST_PATH)
    test = read_test()

    if not cat[ID_COL].equals(test[ID_COL]):
        raise ValueError(f"{CAT_TEST_PATH} ID column does not match LC_test.csv order.")
    if not xgb[ID_COL].equals(test[ID_COL]):
        raise ValueError(f"{XGB_TEST_PATH} ID column does not match LC_test.csv order.")

    blended = weights["xgb"] * xgb["pred"].to_numpy() + weights["catboost"] * cat["pred"].to_numpy()

    if apply_snap:
        legal = np.sort(read_train()[TARGET_COL].unique())
        blended = _snap_to_legal(blended, legal)

    blended = np.clip(blended, *pred_clip)

    submission = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET_COL: blended})
    assert len(submission) == 10_000, f"Expected 10,000 rows, got {len(submission)}"
    assert list(submission.columns) == [ID_COL, TARGET_COL]
    submission.to_csv(FINAL_SUBMISSION_PATH, index=False)
    print(
        f"Wrote {FINAL_SUBMISSION_PATH} "
        f"(rows={len(submission)}, weights={weights}, snap={apply_snap})"
    )


if __name__ == "__main__":
    main()
