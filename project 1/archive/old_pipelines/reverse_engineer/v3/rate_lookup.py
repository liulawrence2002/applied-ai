"""Build static rate lookup tables from loan.csv.

Three lookups:
  - subgrade -> {mean_rate, std_rate, n}
  - grade -> {mean_rate, std_rate, n}
  - (subgrade, term_months) -> mean_rate

Persisted as JSON. Loaded by enrich_v3 to build the
"expected_rate from probability-weighted lookup" feature.
"""
from __future__ import annotations

import json
from time import time

import numpy as np
import pandas as pd

from reverse_engineer.archive_loader import load_archive_loan
from reverse_engineer.v3.config_v3 import RATE_LOOKUP_JSON


def build_lookup(force: bool = False) -> dict:
    if RATE_LOOKUP_JSON.exists() and not force:
        print(f"[rate_lookup] cached -> {RATE_LOOKUP_JSON}", flush=True)
        return json.loads(RATE_LOOKUP_JSON.read_text())

    print("[rate_lookup] loading loan.csv...", flush=True)
    t = time()
    df = load_archive_loan()
    print(f"  loaded {len(df):,} rows ({time()-t:.0f}s)", flush=True)

    # Coerce term to int months
    df["term_months"] = (
        df["term"].astype("string").str.extract(r"(\d+)", expand=False).astype("Int64")
    )

    sg = df.groupby("sub_grade")["int_rate"].agg(["mean", "std", "count"])
    g = df.groupby("grade")["int_rate"].agg(["mean", "std", "count"])
    sg_term = df.groupby(["sub_grade", "term_months"])["int_rate"].mean()

    lookup = {
        "subgrade_mean_rate": sg["mean"].to_dict(),
        "subgrade_std_rate": sg["std"].fillna(0.0).to_dict(),
        "subgrade_n": sg["count"].astype(int).to_dict(),
        "grade_mean_rate": g["mean"].to_dict(),
        "grade_std_rate": g["std"].fillna(0.0).to_dict(),
        "grade_n": g["count"].astype(int).to_dict(),
        # Tuple keys aren't JSON-serializable — flatten as "{sg}_{term}"
        "subgrade_term_mean_rate": {
            f"{sg_val}_{int(t_val)}": float(rate)
            for (sg_val, t_val), rate in sg_term.dropna().items()
            if pd.notna(t_val)
        },
        "global_mean_rate": float(df["int_rate"].mean()),
    }
    RATE_LOOKUP_JSON.write_text(json.dumps(lookup, indent=2))
    print(f"  saved -> {RATE_LOOKUP_JSON}", flush=True)
    print(f"  35 subgrades, {len(lookup['subgrade_term_mean_rate'])} subgrade×term combos", flush=True)
    return lookup


if __name__ == "__main__":
    build_lookup(force=True)
