"""Data-driven pairwise interaction discovery for the LendingClub true-data slice.

Goal: rank ~10 high-value pairwise interaction features that reduce held-out
RMSE when added to a tuned LightGBM regressor on the reduced feature set.

Recipe:
  1. Load true data/LC_train.csv, drop loan_status leakage.
  2. Apply the reduced FE from build_simplified_notebook.py.
  3. 80/20 holdout, random_state=6604; smoothed K-fold OOF target encoding (m=20)
     for addr_state, purpose, zip3.
  4. Cached LightGBM params (from build_final_notebook.py); cap each fit at 3000
     iters with early stopping. Train baseline -> RMSE_base.
  5. Generate candidate pairwise interactions from the top-15 features by gain.
  6. For each candidate, add column, refit LightGBM, record val_RMSE.
     delta = RMSE_base - RMSE_with_interaction (positive == helpful).
  7. Write outputs/simplification/interaction_candidates_ranked.csv.

Wall-clock budget: ~15 minutes.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation as lgb_log

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "true data"
OUT_DIR = PROJECT_ROOT / "outputs" / "simplification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 6604
HOLDOUT_FRAC = 0.20
N_FOLDS_TE = 5
TARGET_ENC_M = 20.0
MAX_ITER = 3000
EARLY_STOP = 60

STRING_NUMERIC_COLS = [
    "dti", "revol_util", "all_util", "mo_sin_old_il_acct",
    "mths_since_last_record", "mths_since_rcnt_il", "mths_since_recent_bc",
    "mths_since_recent_inq", "tot_cur_bal",
]

LGB_PARAMS = {
    "learning_rate": 0.015518526303667327,
    "num_leaves": 182,
    "max_depth": 14,
    "min_child_samples": 47,
    "feature_fraction": 0.4003725559330725,
    "bagging_fraction": 0.6305049835350598,
    "bagging_freq": 1,
    "reg_alpha": 1.416786428180489,
    "reg_lambda": 0.001022839594333141,
    "n_estimators": MAX_ITER,
    "objective": "regression",
    "metric": "rmse",
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "n_jobs": -1,
}

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


# -------------------------------------------------------------
# Feature engineering (from build_simplified_notebook.py)
# -------------------------------------------------------------
def engineer_reduced(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    out["fico_band"] = pd.cut(
        out["fico"], bins=[-np.inf, 660, 690, 720, 760, np.inf], labels=False
    ).astype("Int64")
    out["fico_above_prime"] = (out["fico"] >= 720).astype("int8")
    out["term_months"] = (
        out["term"].astype("string").str.extract(r"(\d+)", expand=False).astype("Int64")
    )
    out["installment_proxy"] = out["loan_amnt"] / out["term_months"].astype("float64")
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"].replace(0, np.nan)
    out["payment_to_income"] = out["installment_proxy"] / (
        out["annual_inc"].replace(0, np.nan) / 12
    )
    out["util_max"] = np.maximum(
        out["all_util"].fillna(0), out["revol_util"].fillna(0)
    )
    out["term_x_dti"] = out["term_months"].astype("float64") * out["dti"]
    out["fico_x_dti"] = out["fico"] * out["dti"]
    out["fico_revol_interaction"] = out["fico"] * out["revol_util"]
    out["credit_file_age_yrs"] = out["mo_sin_old_rev_tl_op"] / 12.0
    out["inq_per_credit_age"] = out["inq_last_12m"].fillna(0) / (
        out["credit_file_age_yrs"].fillna(0) + 1
    )
    out["log1p_annual_inc"] = np.log1p(out["annual_inc"].clip(lower=0))
    out["log1p_loan_amnt"] = np.log1p(out["loan_amnt"].clip(lower=0))
    out["log1p_revol_bal"] = np.log1p(out["revol_bal"].clip(lower=0))
    out["zip3"] = out["zip_code"].astype("string").str.extract(r"(\d{3})", expand=False)
    out["emp_length_num"] = out["emp_length"].map(EMP_LENGTH_MAP)

    drop = ["fico_range_low", "fico_range_high", "term", "emp_length",
            "title", "zip_code", "emp_title"]
    return out.drop(columns=[c for c in drop if c in out.columns])


FINAL_FEATURES = [
    "fico", "fico_band", "fico_above_prime",
    "dti", "revol_util", "all_util", "util_max",
    "term_months", "loan_amnt", "installment_proxy",
    "loan_to_income", "payment_to_income", "log1p_loan_amnt",
    "annual_inc", "revol_bal", "tot_cur_bal", "total_bal_ex_mort",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
    "mths_since_rcnt_il", "mths_since_recent_bc", "mths_since_recent_inq",
    "credit_file_age_yrs",
    "term_x_dti", "fico_x_dti", "fico_revol_interaction",
    "inq_fi", "inq_per_credit_age",
    "addr_state", "purpose", "application_type", "zip3",
    "log1p_annual_inc", "log1p_revol_bal",
    "emp_length_num",
]

NATIVE_CAT = ["application_type", "purpose"]
TE_CAT = ["addr_state", "purpose", "zip3"]
ALL_CAT = list(set(NATIVE_CAT + TE_CAT))


# -------------------------------------------------------------
# Target encoding
# -------------------------------------------------------------
def smoothed_te(train_col, target, val_col=None, folds=N_FOLDS_TE, seed=RANDOM_STATE):
    global_mean = float(np.nanmean(target))
    val_enc = None
    if val_col is not None:
        stats = pd.DataFrame({"cat": train_col.values, "y": target}).groupby("cat")["y"].agg(
            ["mean", "count"]
        )
        stats["enc"] = (
            stats["mean"] * stats["count"] + global_mean * TARGET_ENC_M
        ) / (stats["count"] + TARGET_ENC_M)
        mapping = stats["enc"].to_dict()
        val_enc = val_col.map(mapping).fillna(global_mean).values
    train_enc = np.full(len(train_col), global_mean, dtype=np.float64)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, va in kf.split(train_col):
        inner = pd.DataFrame({"cat": train_col.values[tr], "y": target[tr]}).groupby("cat")["y"].agg(
            ["mean", "count"]
        )
        inner["enc"] = (
            inner["mean"] * inner["count"] + global_mean * TARGET_ENC_M
        ) / (inner["count"] + TARGET_ENC_M)
        train_enc[va] = (
            pd.Series(train_col.values[va]).map(inner["enc"].to_dict()).fillna(global_mean).values
        )
    return train_enc, val_enc


def add_te(Xtr, ytr, Xva):
    Xtr, Xva = Xtr.copy(), Xva.copy()
    for c in TE_CAT:
        if c not in Xtr.columns:
            continue
        tr_e, va_e = smoothed_te(Xtr[c].astype(str), ytr, Xva[c].astype(str))
        Xtr[f"{c}_te"] = tr_e
        Xva[f"{c}_te"] = va_e
    return Xtr, Xva


def prep_lgb(df, cat_dtypes=None):
    df = df.copy()
    cat_cols = [c for c in ALL_CAT if c in df.columns]
    for c in cat_cols:
        s = df[c].astype("string").fillna("Missing")
        df[c] = s.astype(cat_dtypes[c]) if cat_dtypes and c in cat_dtypes else s.astype("category")
    for c in df.columns:
        if c not in cat_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df, cat_cols


def build_cat_dtypes(df):
    return {
        c: pd.CategoricalDtype(
            categories=pd.Index(df[c].astype("string").fillna("Missing").unique())
        )
        for c in ALL_CAT
        if c in df.columns
    }


# -------------------------------------------------------------
# Training helper
# -------------------------------------------------------------
def fit_eval(Xtr_te, Xva_te, ytr, yva, extra_cols=None):
    """Fit LightGBM on (optionally augmented) features. Return val_RMSE and best_iter."""
    if extra_cols is not None:
        Xtr_te = Xtr_te.assign(**extra_cols["train"])
        Xva_te = Xva_te.assign(**extra_cols["val"])

    cat_dt = build_cat_dtypes(Xtr_te)
    Xtr_lgb, cat_cols = prep_lgb(Xtr_te, cat_dt)
    Xva_lgb, _ = prep_lgb(Xva_te, cat_dt)

    m = LGBMRegressor(**LGB_PARAMS)
    m.fit(
        Xtr_lgb, ytr,
        eval_set=[(Xva_lgb, yva)],
        categorical_feature=cat_cols,
        callbacks=[lgb_es(EARLY_STOP, verbose=False), lgb_log(0)],
    )
    pred = m.predict(Xva_lgb)
    rmse = float(np.sqrt(mean_squared_error(yva, pred)))
    return rmse, int(m.best_iteration_ or m.n_estimators)


# -------------------------------------------------------------
# Candidate generation
# -------------------------------------------------------------
def _safe_div(a, b):
    a = a.astype("float64")
    b = b.astype("float64").replace(0, np.nan)
    return a / b


def _add(cands, name, tr_series, va_series, formula, group):
    cands[name] = {
        "train": tr_series,
        "val": va_series,
        "formula": formula,
        "group": group,
    }


def build_candidates(Xtr_te, Xva_te):
    """Return an *ordered* dict: name -> {'train', 'val', 'formula', 'group'}.

    Ordering matters because the script has a 13-min wall-clock guard. The most
    theoretically promising candidates (aux-grade analogues, FICO band products,
    high-gain numeric pairs) go FIRST so they survive even if the budget cuts in.
    """
    cands: dict = {}

    # ---- (A) HIGHEST PRIORITY: TE x FICO (aux-grade analogue) ----
    for te_col in ["zip3_te", "purpose_te", "addr_state_te"]:
        if te_col not in Xtr_te.columns:
            continue
        _add(
            cands, f"{te_col}_x_fico",
            Xtr_te[te_col].astype("float64") * Xtr_te["fico"].astype("float64"),
            Xva_te[te_col].astype("float64") * Xva_te["fico"].astype("float64"),
            f"{te_col} * fico", "te_x_fico",
        )

    # ---- (B) FICO band x utilisation / dti ----
    fb_tr = Xtr_te["fico_band"].astype("float64")
    fb_va = Xva_te["fico_band"].astype("float64")
    for partner in ["util_max", "revol_util", "dti", "all_util"]:
        if partner not in Xtr_te.columns:
            continue
        _add(
            cands, f"fico_band_x_{partner}",
            fb_tr * Xtr_te[partner].astype("float64"),
            fb_va * Xva_te[partner].astype("float64"),
            f"fico_band * {partner}", "fico_band_x_numeric",
        )

    # ---- (C) Highest-gain numeric products involving FICO ----
    fico_partners_top = [
        "util_max", "all_util", "term_months", "installment_proxy",
        "loan_amnt", "mths_since_rcnt_il", "mths_since_recent_bc",
        "mths_since_recent_inq", "annual_inc", "mo_sin_old_il_acct",
    ]
    for partner in fico_partners_top:
        if partner not in Xtr_te.columns:
            continue
        _add(
            cands, f"fico_x_{partner}",
            Xtr_te["fico"].astype("float64") * Xtr_te[partner].astype("float64"),
            Xva_te["fico"].astype("float64") * Xva_te[partner].astype("float64"),
            f"fico * {partner}", "product_numeric",
        )

    # ---- (D) Key ratios ----
    ratio_pairs = [
        ("fico", "dti"),
        ("fico", "util_max"),
        ("fico", "revol_util"),
        ("installment_proxy", "dti"),
        ("util_max", "term_months"),
        ("dti", "fico"),
        ("loan_amnt", "fico"),
        ("util_max", "fico"),
        ("all_util", "term_months"),
        ("loan_amnt", "term_months"),
    ]
    for a, b in ratio_pairs:
        if a not in Xtr_te.columns or b not in Xtr_te.columns:
            continue
        _add(
            cands, f"{a}_over_{b}",
            _safe_div(Xtr_te[a], Xtr_te[b]),
            _safe_div(Xva_te[a], Xva_te[b]),
            f"{a} / {b}", "ratio",
        )

    # ---- (E) TE x other numerics ----
    for te_col in ["zip3_te", "purpose_te", "addr_state_te"]:
        if te_col not in Xtr_te.columns:
            continue
        for partner in ["dti", "util_max"]:
            _add(
                cands, f"{te_col}_x_{partner}",
                Xtr_te[te_col].astype("float64") * Xtr_te[partner].astype("float64"),
                Xva_te[te_col].astype("float64") * Xva_te[partner].astype("float64"),
                f"{te_col} * {partner}", "te_x_numeric",
            )

    # ---- (F) Remaining numeric x numeric (lower priority) ----
    other_pairs = [
        ("dti", "revol_util"),
        ("dti", "all_util"),
        ("dti", "util_max"),
        ("dti", "loan_amnt"),
        ("dti", "annual_inc"),
        ("dti", "installment_proxy"),
        ("revol_util", "all_util"),
        ("revol_util", "term_months"),
        ("revol_util", "loan_amnt"),
        ("util_max", "term_months"),
        ("util_max", "loan_amnt"),
        ("util_max", "installment_proxy"),
        ("term_months", "loan_amnt"),
        ("term_months", "installment_proxy"),
        ("loan_amnt", "annual_inc"),
        ("installment_proxy", "annual_inc"),
        ("mths_since_rcnt_il", "mths_since_recent_bc"),
        ("mths_since_rcnt_il", "mths_since_recent_inq"),
        ("mths_since_recent_bc", "mths_since_recent_inq"),
        ("mo_sin_old_il_acct", "mths_since_rcnt_il"),
    ]
    for a, b in other_pairs:
        if a not in Xtr_te.columns or b not in Xtr_te.columns:
            continue
        name = f"{a}_x_{b}"
        if name in cands:
            continue
        _add(
            cands, name,
            Xtr_te[a].astype("float64") * Xtr_te[b].astype("float64"),
            Xva_te[a].astype("float64") * Xva_te[b].astype("float64"),
            f"{a} * {b}", "product_numeric",
        )

    return cands


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    t_global = time.time()

    print(f"[{time.strftime('%H:%M:%S')}] Loading data ...")
    dt_overrides = {c: "float64" for c in STRING_NUMERIC_COLS}
    train_raw = pd.read_csv(
        DATA_DIR / "LC_train.csv", na_values=["NA"], dtype=dt_overrides, low_memory=False
    )
    train_raw = train_raw.drop(columns=["loan_status"])
    print(f"  train_raw: {train_raw.shape}")

    print(f"[{time.strftime('%H:%M:%S')}] Engineering reduced features ...")
    train_fe = engineer_reduced(train_raw)
    y = train_fe["int_rate"].values.astype(np.float64)
    X = train_fe.drop(columns=["int_rate"])
    present = [c for c in FINAL_FEATURES if c in X.columns]
    missing = [c for c in FINAL_FEATURES if c not in X.columns]
    if missing:
        print(f"  WARNING: missing features: {missing}")
    X = X[present].copy()
    print(f"  X shape: {X.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE
    )
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

    print(f"[{time.strftime('%H:%M:%S')}] Applying target encoding (single-pass, 5-fold OOF) ...")
    Xtr_te, Xva_te = add_te(X_train, y_train, X_val)
    print(f"  Xtr_te columns: {Xtr_te.shape[1]}")

    # ---- Baseline ----
    print(f"[{time.strftime('%H:%M:%S')}] Training baseline LightGBM ...")
    t0 = time.time()
    base_rmse, base_iter = fit_eval(Xtr_te, Xva_te, y_train, y_val)
    print(f"  baseline RMSE = {base_rmse:.5f}  (best_iter={base_iter}, {time.time()-t0:.1f}s)")

    # ---- Candidate generation ----
    print(f"[{time.strftime('%H:%M:%S')}] Building candidate interactions ...")
    cands = build_candidates(Xtr_te, Xva_te)
    print(f"  generated {len(cands)} candidates across groups: "
          f"{sorted(set(v['group'] for v in cands.values()))}")

    # ---- Individual ablation ----
    print(f"[{time.strftime('%H:%M:%S')}] Running individual ablation (one candidate at a time) ...")
    records = []
    for i, (name, spec) in enumerate(cands.items(), 1):
        t0 = time.time()
        extra = {
            "train": {name: spec["train"].astype("float64").values},
            "val": {name: spec["val"].astype("float64").values},
        }
        try:
            rmse, n_iter = fit_eval(Xtr_te, Xva_te, y_train, y_val, extra_cols=extra)
            delta = base_rmse - rmse
            elapsed = time.time() - t0
            status = "OK"
        except Exception as e:
            rmse = float("nan")
            delta = float("nan")
            n_iter = -1
            elapsed = time.time() - t0
            status = f"ERR:{type(e).__name__}"
        records.append({
            "interaction": name,
            "formula": spec["formula"],
            "group": spec["group"],
            "rmse_with": rmse,
            "individual_delta_rmse": delta,
            "best_iter": n_iter,
            "elapsed_s": elapsed,
            "status": status,
        })
        sign = "+" if delta > 0 else " "
        print(f"  [{i:2d}/{len(cands)}] {name:40s} rmse={rmse:.5f} delta={sign}{delta:+.5f}  ({elapsed:.1f}s)", flush=True)
        # Persist running results every 5 candidates so a budget cut still produces output
        if i % 5 == 0:
            pd.DataFrame(records).sort_values(
                "individual_delta_rmse", ascending=False
            ).to_csv(OUT_DIR / "interaction_candidates_ranked.csv", index=False)
        # Budget guard
        if time.time() - t_global > 14 * 60:
            print(f"  WARNING: budget limit reached at candidate {i}/{len(cands)}; stopping early.", flush=True)
            break

    df = pd.DataFrame(records).sort_values("individual_delta_rmse", ascending=False).reset_index(drop=True)
    df["notes"] = df["status"].where(df["status"] != "OK", "")

    out_path = OUT_DIR / "interaction_candidates_ranked.csv"
    df_out = df[[
        "interaction", "formula", "group",
        "individual_delta_rmse", "rmse_with", "best_iter", "elapsed_s", "notes",
    ]]
    df_out.to_csv(out_path, index=False)
    print(f"\n[{time.strftime('%H:%M:%S')}] Wrote {out_path}")

    # Persist baseline & summary
    summary = {
        "random_state": RANDOM_STATE,
        "baseline_val_rmse": base_rmse,
        "baseline_best_iter": base_iter,
        "n_candidates_tested": int((df["status"] == "OK").sum()),
        "n_candidates_total": len(cands),
        "top10": df.head(10)[["interaction", "formula", "group", "individual_delta_rmse"]].to_dict("records"),
        "worst5": df.tail(5)[["interaction", "formula", "group", "individual_delta_rmse"]].to_dict("records"),
        "wall_clock_s": time.time() - t_global,
    }
    with open(OUT_DIR / "interaction_discovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n=== TOP 10 (positive lift) ===")
    print(df.head(10).to_string(index=False))
    print("\n=== BOTTOM 5 (most harmful) ===")
    print(df.tail(5).to_string(index=False))
    print(f"\nBaseline RMSE: {base_rmse:.5f}")
    print(f"Total wall-clock: {time.time() - t_global:.0f}s")


if __name__ == "__main__":
    main()
