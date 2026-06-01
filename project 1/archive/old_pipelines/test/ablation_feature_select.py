"""Feature ablation + importance experiment for the simplification review.

Loads the true-data slice, applies the same feature engineering as
build_final_notebook.py (without the aux-grade enrichment so this stays
fast and reproducible), fits a single tuned LightGBM with 5-fold OOF,
and dumps:

    outputs/simplification/lgb_full_fe_oof_rmse.txt
    outputs/simplification/lgb_full_fe_importance_gain.csv
    outputs/simplification/lgb_full_fe_importance_perm.csv
    outputs/simplification/reduced_feature_list.json

Runtime ~3-5 min on CPU. Used only to inform the
MODEL_SIMPLIFICATION_REVIEW.md and the reduced feature set; the final
simplified notebook re-tunes Optuna for its own RMSE numbers.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation as lgb_log
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "true data"
OUT_DIR = PROJECT_ROOT / "outputs" / "simplification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 6604
N_FOLDS = 5
HOLDOUT_FRAC = 0.20
TARGET_ENC_M = 20.0

STRING_NUMERIC_COLS = [
    "dti", "revol_util", "all_util", "mo_sin_old_il_acct",
    "mths_since_last_record", "mths_since_rcnt_il", "mths_since_recent_bc",
    "mths_since_recent_inq", "tot_cur_bal",
]
DT_OVERRIDES = {c: "float64" for c in STRING_NUMERIC_COLS}

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the distilled FE in build_final_notebook.py, minus aux-grade."""
    out = df.copy()
    out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    out["fico_band"] = pd.cut(out["fico"], bins=[-np.inf, 660, 690, 720, 760, np.inf], labels=False).astype("Int64")
    out["fico_above_prime"] = (out["fico"] >= 720).astype("int8")
    out["fico_subprime"] = (out["fico"] < 660).astype("int8")

    out["term_months"] = out["term"].astype("string").str.extract(r"(\d+)", expand=False).astype("Int64")
    out["emp_length_num"] = out["emp_length"].map(EMP_LENGTH_MAP)

    for c in ["mths_since_last_record", "mths_since_recent_inq",
              "mths_since_rcnt_il", "mths_since_recent_bc"]:
        out[f"has_{c}"] = out[c].notna().astype(int)

    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"].replace(0, np.nan)
    out["installment_proxy"] = out["loan_amnt"] / out["term_months"].astype("float64")
    out["payment_to_income"] = out["installment_proxy"] / (out["annual_inc"].replace(0, np.nan) / 12)
    out["acc_open_ratio"] = out["open_acc"] / out["total_acc"].replace(0, np.nan)

    # Hannah's interactions
    out["dti_income_interaction"] = out["dti"] * out["annual_inc"]
    out["fico_revol_interaction"] = out["fico"] * out["revol_util"]

    out["fico_x_dti"] = out["fico"] * out["dti"]
    out["term_x_dti"] = out["term_months"].astype("float64") * out["dti"]

    out["dti_band"] = pd.cut(out["dti"], bins=[-np.inf, 10, 20, 30, 40, np.inf], labels=False).astype("Int64")

    out["derog_score"] = (
        out["delinq_2yrs"].fillna(0) * 8 +
        out["pub_rec"].fillna(0) * 13 +
        out["pub_rec_bankruptcies"].fillna(0) * 22 +
        out["chargeoff_within_12_mths"].fillna(0) * 15 +
        out["collections_12_mths_ex_med"].fillna(0) * 10
    )
    out["inq_intensity"] = out["inq_last_12m"].fillna(0) + out["inq_fi"].fillna(0)

    out["delinq_flag"] = (out["delinq_2yrs"].fillna(0) > 0).astype("int8")
    out["pub_rec_flag"] = (out["pub_rec"].fillna(0) > 0).astype("int8")
    out["bankrupt_flag"] = (out["pub_rec_bankruptcies"].fillna(0) > 0).astype("int8")
    out["any_derog"] = (out["derog_score"] > 0).astype("int8")

    out["revol_util_band"] = pd.cut(out["revol_util"], bins=[-np.inf, 30, 50, 75, 100, np.inf], labels=False).astype("Int64")
    out["util_max"] = np.maximum(out["all_util"].fillna(0), out["revol_util"].fillna(0))
    out["util_maxed_out"] = (out["revol_util"].fillna(0) >= 95).astype("int8")

    out["credit_file_age_yrs"] = out["mo_sin_old_rev_tl_op"] / 12.0
    out["inq_per_credit_age"] = out["inq_last_12m"].fillna(0) / (out["credit_file_age_yrs"].fillna(0) + 1)

    for c in ["annual_inc", "revol_bal", "tot_cur_bal", "loan_amnt"]:
        out[f"log1p_{c}"] = np.log1p(out[c].clip(lower=0))

    out["zip3"] = out["zip_code"].astype("string").str.extract(r"(\d{3})", expand=False)
    out["emp_title"] = out["emp_title"].astype("string").str.lower().str.strip()

    drop = ["fico_range_low", "fico_range_high", "term", "emp_length",
            "title", "zip_code"]
    return out.drop(columns=[c for c in drop if c in out.columns])


NATIVE_CAT = ["application_type", "home_ownership", "verification_status", "purpose"]
TE_CAT = ["addr_state", "purpose", "zip3", "emp_title"]
ALL_CAT = list(set(NATIVE_CAT + TE_CAT))


def smoothed_te(train_col, target, val_col=None, test_col=None, folds=5,
                seed=RANDOM_STATE):
    global_mean = float(np.nanmean(target))
    val_enc = test_enc = None
    if val_col is not None or test_col is not None:
        stats = pd.DataFrame({"cat": train_col.values, "y": target}).groupby("cat")["y"].agg(["mean", "count"])
        stats["enc"] = (stats["mean"] * stats["count"] + global_mean * TARGET_ENC_M) / (stats["count"] + TARGET_ENC_M)
        mapping = stats["enc"].to_dict()
        val_enc = val_col.map(mapping).fillna(global_mean).values if val_col is not None else None
        test_enc = test_col.map(mapping).fillna(global_mean).values if test_col is not None else None
    train_enc = np.full(len(train_col), global_mean, dtype=np.float64)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, va in kf.split(train_col):
        inner = pd.DataFrame({"cat": train_col.values[tr], "y": target[tr]}).groupby("cat")["y"].agg(["mean", "count"])
        inner["enc"] = (inner["mean"] * inner["count"] + global_mean * TARGET_ENC_M) / (inner["count"] + TARGET_ENC_M)
        train_enc[va] = pd.Series(train_col.values[va]).map(inner["enc"].to_dict()).fillna(global_mean).values
    return train_enc, val_enc, test_enc


def add_te(Xtr, ytr, Xva, Xte):
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in TE_CAT:
        if c not in Xtr.columns:
            continue
        tr_e, va_e, te_e = smoothed_te(
            Xtr[c].astype(str), ytr,
            Xva[c].astype(str) if Xva is not None else None,
            Xte[c].astype(str) if Xte is not None else None,
        )
        Xtr[f"{c}_te"] = tr_e
        if va_e is not None:
            Xva[f"{c}_te"] = va_e
        if te_e is not None:
            Xte[f"{c}_te"] = te_e
    return Xtr, Xva, Xte


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
    return {c: pd.CategoricalDtype(categories=pd.Index(df[c].astype("string").fillna("Missing").unique()))
            for c in ALL_CAT if c in df.columns}


def main():
    print("=== loading true data ===")
    train_raw = pd.read_csv(DATA_DIR / "LC_train.csv", na_values=["NA"], dtype=DT_OVERRIDES, low_memory=False)
    test_raw = pd.read_csv(DATA_DIR / "LC_test.csv", na_values=["NA"], dtype=DT_OVERRIDES, low_memory=False)
    train_raw = train_raw.drop(columns=["loan_status"])
    test_raw = test_raw.drop(columns=["loan_status"])

    train_fe = engineer(train_raw)
    test_fe = engineer(test_raw)

    top_titles = train_fe["emp_title"].value_counts().head(100).index.tolist()
    train_fe["emp_title"] = train_fe["emp_title"].where(train_fe["emp_title"].isin(top_titles), "Other").fillna("Missing").astype(str)
    test_fe["emp_title"] = test_fe["emp_title"].where(test_fe["emp_title"].isin(top_titles), "Other").fillna("Missing").astype(str)

    test_ids = test_fe["ID"].copy()
    test_fe = test_fe.drop(columns=["ID"])
    y = train_fe["int_rate"].values.astype(np.float64)
    X = train_fe.drop(columns=["int_rate"])
    print(f"engineered: X={X.shape}  test={test_fe.shape}")

    # 80/20 holdout
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE)
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)

    # Single-pass TE using train -> val/test (faster, OK for importance ranking)
    Xtr_te, Xval_te, Xte_te = add_te(X_train, y_train, X_val, test_fe)
    cat_dt = build_cat_dtypes(Xtr_te)
    Xtr_lgb, cat_cols = prep_lgb(Xtr_te, cat_dt)
    Xval_lgb, _ = prep_lgb(Xval_te, cat_dt)

    LGB_PARAMS = {
        "learning_rate": 0.015518526303667327,
        "num_leaves": 182,
        "max_depth": 14,
        "min_child_samples": 47,
        "feature_fraction": 0.4003725559330725,
        "bagging_fraction": 0.6305049835350598,
        "reg_alpha": 1.416786428180489,
        "reg_lambda": 0.001022839594333141,
    }

    print("=== fitting LightGBM (single) ===")
    t = time()
    lgb = LGBMRegressor(**LGB_PARAMS, n_estimators=4000, objective="regression", metric="rmse",
                        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1, bagging_freq=1)
    lgb.fit(Xtr_lgb, y_train,
            eval_set=[(Xval_lgb, y_val)],
            categorical_feature=cat_cols,
            callbacks=[lgb_es(120, verbose=False), lgb_log(0)])
    val_pred = lgb.predict(Xval_lgb)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    print(f"single-LGB val_RMSE (full FE, no aux): {val_rmse:.4f}  ({time()-t:.0f}s)")

    # Gain importance
    gain = pd.DataFrame({
        "feature": Xtr_lgb.columns,
        "gain": lgb.booster_.feature_importance(importance_type="gain"),
        "split": lgb.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    gain["gain_share"] = gain["gain"] / gain["gain"].sum()
    gain.to_csv(OUT_DIR / "lgb_full_fe_importance_gain.csv", index=False)
    print(f"\nTop 25 gain features:")
    print(gain.head(25).to_string(index=False))

    # Permutation importance (on val, n=3 for speed)
    print("\n=== permutation importance (n_repeats=3) ===")
    t = time()
    perm = permutation_importance(lgb, Xval_lgb, y_val, n_repeats=3,
                                   random_state=RANDOM_STATE, n_jobs=1,
                                   scoring="neg_root_mean_squared_error")
    perm_df = pd.DataFrame({
        "feature": Xval_lgb.columns,
        "perm_mean": perm.importances_mean,
        "perm_std": perm.importances_std,
    }).sort_values("perm_mean", ascending=False).reset_index(drop=True)
    perm_df.to_csv(OUT_DIR / "lgb_full_fe_importance_perm.csv", index=False)
    print(f"perm done in {time()-t:.0f}s")
    print(f"\nTop 25 perm features:")
    print(perm_df.head(25).to_string(index=False))

    # Reduced set: keep top 25 by gain (cumulative gain share usually >= 95%)
    top_gain = gain.head(30)["feature"].tolist()
    top_perm = perm_df.head(20)["feature"].tolist()
    reduced = sorted(set(top_gain) | set(top_perm))

    # Always keep these even if missing: structural drivers
    keep = ["fico", "term_months", "dti", "revol_util", "annual_inc",
            "loan_amnt", "purpose", "addr_state_te", "purpose_te",
            "zip3_te", "emp_title_te", "fico_band", "loan_to_income",
            "installment_proxy", "payment_to_income", "all_util",
            "log1p_annual_inc", "log1p_loan_amnt", "credit_file_age_yrs"]
    for k in keep:
        if k in Xtr_lgb.columns and k not in reduced:
            reduced.append(k)

    print(f"\nfull FE column count: {Xtr_lgb.shape[1]}")
    print(f"reduced feature count: {len(reduced)}")
    cum_share = gain[gain["feature"].isin(reduced)]["gain_share"].sum()
    print(f"reduced set captures {cum_share*100:.2f}% of total gain")

    with open(OUT_DIR / "reduced_feature_list.json", "w") as f:
        json.dump({
            "val_rmse_full_fe_single_lgb": val_rmse,
            "n_features_full": int(Xtr_lgb.shape[1]),
            "n_features_reduced": len(reduced),
            "reduced_features": reduced,
            "cum_gain_share_in_reduced": float(cum_share),
        }, f, indent=2)
    with open(OUT_DIR / "lgb_full_fe_oof_rmse.txt", "w") as f:
        f.write(f"val_RMSE (single LightGBM, full FE, no aux): {val_rmse:.6f}\n")
        f.write(f"n_features: {Xtr_lgb.shape[1]}\n")

    print(f"\nWrote reduced_feature_list.json with {len(reduced)} features.")
    print(f"Captures {cum_share*100:.2f}% of total gain in single LGB.")


if __name__ == "__main__":
    main()
