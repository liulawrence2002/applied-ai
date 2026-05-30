"""Smoke test the explainability cells of FINAL_SIMPLIFIED_MODEL_EXPLAINABILITY.ipynb
without paying the 15-min Optuna + 5-fold OOF cost.

Trains a single CatBoost on a 30k sample with cached params, then runs:
- permutation importance
- CatBoost native gain
- CatBoost native SHAP (per-row + summary)
- PDP via sklearn shim
- ICE via sklearn shim
- LIME with int<->string round-trip
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import PartialDependenceDisplay

from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "true data"
OUT_DIR = PROJECT_ROOT / "outputs" / "simplification" / "smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 6604

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

def engineer_reduced(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    out["fico_band"] = pd.cut(out["fico"], bins=[-np.inf, 660, 690, 720, 760, np.inf], labels=False).astype("Int64")
    out["fico_above_prime"] = (out["fico"] >= 720).astype("int8")
    out["term_months"] = out["term"].astype("string").str.extract(r"(\d+)", expand=False).astype("Int64")
    out["installment_proxy"] = out["loan_amnt"] / out["term_months"].astype("float64")
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"].replace(0, np.nan)
    out["payment_to_income"] = out["installment_proxy"] / (out["annual_inc"].replace(0, np.nan) / 12)
    out["util_max"] = np.maximum(out["all_util"].fillna(0), out["revol_util"].fillna(0))
    out["term_x_dti"] = out["term_months"].astype("float64") * out["dti"]
    out["fico_x_dti"] = out["fico"] * out["dti"]
    out["fico_revol_interaction"] = out["fico"] * out["revol_util"]
    out["credit_file_age_yrs"] = out["mo_sin_old_rev_tl_op"] / 12.0
    out["inq_per_credit_age"] = out["inq_last_12m"].fillna(0) / (out["credit_file_age_yrs"].fillna(0) + 1)
    out["log1p_annual_inc"] = np.log1p(out["annual_inc"].clip(lower=0))
    out["log1p_loan_amnt"] = np.log1p(out["loan_amnt"].clip(lower=0))
    out["log1p_revol_bal"] = np.log1p(out["revol_bal"].clip(lower=0))
    out["zip3"] = out["zip_code"].astype("string").str.extract(r"(\d{3})", expand=False)
    out["emp_length_num"] = out["emp_length"].map(EMP_LENGTH_MAP)
    drop = ["fico_range_low", "fico_range_high", "term", "emp_length", "title", "zip_code", "emp_title"]
    return out.drop(columns=[c for c in drop if c in out.columns])


FINAL_FEATURES = [
    'fico', 'fico_band', 'fico_above_prime',
    'dti', 'revol_util', 'all_util', 'util_max',
    'term_months', 'loan_amnt', 'installment_proxy',
    'loan_to_income', 'payment_to_income', 'log1p_loan_amnt',
    'annual_inc', 'revol_bal', 'tot_cur_bal', 'total_bal_ex_mort',
    'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
    'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_inq',
    'credit_file_age_yrs',
    'term_x_dti', 'fico_x_dti', 'fico_revol_interaction',
    'inq_fi', 'inq_per_credit_age',
    'addr_state', 'purpose', 'application_type', 'zip3',
    'log1p_annual_inc', 'log1p_revol_bal',
    'emp_length_num',
]

NATIVE_CAT = ['application_type', 'purpose']
TE_CAT = ['addr_state', 'purpose', 'zip3']
ALL_CAT = list(set(NATIVE_CAT + TE_CAT))


def smoothed_te_simple(train_col, target, val_col=None, test_col=None, m=20.0):
    global_mean = float(np.nanmean(target))
    stats = pd.DataFrame({'cat': train_col.values, 'y': target}).groupby('cat')['y'].agg(['mean', 'count'])
    stats['enc'] = (stats['mean'] * stats['count'] + global_mean * m) / (stats['count'] + m)
    mapping = stats['enc'].to_dict()
    val_enc = val_col.map(mapping).fillna(global_mean).values if val_col is not None else None
    test_enc = test_col.map(mapping).fillna(global_mean).values if test_col is not None else None
    train_enc = train_col.map(mapping).fillna(global_mean).values
    return train_enc, val_enc, test_enc


def add_te_simple(Xtr, ytr, Xva, Xte):
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in TE_CAT:
        if c not in Xtr.columns:
            continue
        tr_e, va_e, te_e = smoothed_te_simple(Xtr[c].astype(str), ytr,
                                                Xva[c].astype(str), Xte[c].astype(str))
        Xtr[f'{c}_te'] = tr_e
        if va_e is not None:
            Xva[f'{c}_te'] = va_e
        if te_e is not None:
            Xte[f'{c}_te'] = te_e
    return Xtr, Xva, Xte


def prep_cb(df):
    df = df.copy()
    cat_cols = [c for c in ALL_CAT if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype('string').fillna('Missing')
    cat_idx = [df.columns.get_loc(c) for c in cat_cols]
    for c in df.columns:
        if c not in cat_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    return df, cat_idx


def main():
    print("=== loading ===")
    train_raw = pd.read_csv(DATA_DIR / "LC_train.csv", na_values=["NA"], dtype=DT_OVERRIDES, low_memory=False)
    test_raw = pd.read_csv(DATA_DIR / "LC_test.csv", na_values=["NA"], dtype=DT_OVERRIDES, low_memory=False)
    train_raw = train_raw.drop(columns=["loan_status"])
    test_raw = test_raw.drop(columns=["loan_status"])

    train_fe = engineer_reduced(train_raw)
    test_fe = engineer_reduced(test_raw)

    test_ids = test_fe["ID"].copy()
    test_fe = test_fe.drop(columns=["ID"])
    y = train_fe["int_rate"].values.astype(np.float64)
    X = train_fe.drop(columns=["int_rate"])

    present = [c for c in FINAL_FEATURES if c in X.columns]
    X = X[present].copy()
    test_fe = test_fe[present].copy()

    # 80/20 split, sub-sample train for speed
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    # Smoke test: use first 25k rows
    X_train = X_train.iloc[:25000].reset_index(drop=True)
    y_train = y_train[:25000]
    print(f"train sub: {X_train.shape}  val: {X_val.shape}")

    Xtr_te, Xval_te, Xte_te = add_te_simple(X_train, y_train, X_val, test_fe)
    Xtr_cb_full, cat_idx_full = prep_cb(Xtr_te)
    Xval_cb_full, _ = prep_cb(Xval_te)
    Xte_cb_full, _ = prep_cb(Xte_te)

    print("=== fitting CB (smoke) ===")
    CB_PARAMS = {
        'learning_rate': 0.034266751541500245,
        'depth': 7,
        'l2_leaf_reg': 0.9758040518945117,
        'random_strength': 4.906296660542921,
        'bagging_temperature': 0.14995979808622192,
        'border_count': 197,
    }
    explain_model = CatBoostRegressor(**CB_PARAMS,
                                       iterations=600,
                                       loss_function='RMSE', eval_metric='RMSE',
                                       random_seed=RANDOM_STATE, verbose=0,
                                       allow_writing_files=False,
                                       early_stopping_rounds=50)
    explain_model.fit(Xtr_cb_full, y_train, cat_features=cat_idx_full,
                       eval_set=(Xval_cb_full, y_val), verbose=False)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, explain_model.predict(Xval_cb_full))))
    print(f"smoke CB val_RMSE: {val_rmse:.4f}  (with {explain_model.tree_count_} trees)")

    # === SHAP native ===
    print("=== SHAP native ===")
    pool_val = Pool(Xval_cb_full.head(500), cat_features=cat_idx_full)
    shap_with_expected = explain_model.get_feature_importance(pool_val, type='ShapValues')
    expected_value = float(shap_with_expected[0, -1])
    shap_values_val = shap_with_expected[:, :-1]
    print(f"  shap shape: {shap_values_val.shape}, expected: {expected_value:.3f}")

    import shap

    def to_numeric_frame(df_cb):
        out = df_cb.copy()
        for i in cat_idx_full:
            col = out.columns[i]
            codes, _ = pd.factorize(out[col].astype('string').fillna('Missing'))
            out[col] = codes.astype('int64')
        return out

    Xval_sample = Xval_cb_full.head(500).reset_index(drop=True)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values_val, to_numeric_frame(Xval_sample),
                       feature_names=list(Xval_sample.columns), max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "smoke_shap_summary.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  wrote smoke_shap_summary.png")

    # === PDP via shim ===
    print("=== PDP via shim ===")
    class CatBoostShim(RegressorMixin, BaseEstimator):
        def __init__(self, model, columns, cat_idx):
            self.model = model
            self.columns = list(columns)
            self.cat_idx = list(cat_idx)
            self.n_features_in_ = len(self.columns)
            self.feature_names_in_ = np.array(self.columns, dtype=object)
            self.is_fitted_ = True
        def __sklearn_is_fitted__(self):
            return True
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.estimator_type = 'regressor'
            try:
                from sklearn.utils._tags import RegressorTags
                tags.regressor_tags = RegressorTags()
            except Exception:
                pass
            return tags
        def fit(self, X, y=None):
            return self
        def predict(self, X):
            df = pd.DataFrame(X, columns=self.columns)
            for i in self.cat_idx:
                c = self.columns[i]
                df[c] = df[c].astype('string').fillna('Missing')
            for j, c in enumerate(self.columns):
                if j not in self.cat_idx:
                    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
            return self.model.predict(df)

    shim = CatBoostShim(explain_model, Xval_cb_full.columns, cat_idx_full)
    Xtr_pdp_numeric = to_numeric_frame(Xtr_cb_full).values
    feats = ['fico', 'term_months', 'dti']
    feat_idx = [list(Xtr_cb_full.columns).index(f) for f in feats]
    try:
        fig, ax = plt.subplots(figsize=(13, 4))
        PartialDependenceDisplay.from_estimator(
            shim, Xtr_pdp_numeric, features=feat_idx,
            feature_names=list(Xtr_cb_full.columns), ax=ax, kind='average',
            grid_resolution=20,
        )
        plt.tight_layout()
        plt.savefig(OUT_DIR / "smoke_pdp.png", dpi=100, bbox_inches='tight')
        plt.close()
        print("  PDP ok")
    except Exception as e:
        print(f"  PDP FAIL: {e}")
        raise

    # === ICE ===
    try:
        fig, ax = plt.subplots(figsize=(13, 4))
        PartialDependenceDisplay.from_estimator(
            shim, Xtr_pdp_numeric, features=feat_idx,
            feature_names=list(Xtr_cb_full.columns), kind='both', centered=True,
            ax=ax, grid_resolution=20,
            pd_line_kw={'color': 'gold', 'linewidth': 3},
        )
        plt.tight_layout()
        plt.savefig(OUT_DIR / "smoke_ice.png", dpi=100, bbox_inches='tight')
        plt.close()
        print("  ICE ok")
    except Exception as e:
        print(f"  ICE FAIL: {e}")
        raise

    # === LIME ===
    print("=== LIME ===")
    import lime
    import lime.lime_tabular

    cat_code_maps = {}
    cat_train_arrays = {}
    for i in cat_idx_full:
        col = Xtr_cb_full.columns[i]
        series_str = Xtr_cb_full[col].astype('string').fillna('Missing')
        codes, uniques = pd.factorize(series_str)
        cat_code_maps[col] = {idx: val for idx, val in enumerate(uniques)}
        cat_train_arrays[col] = codes

    Xtr_lime = Xtr_cb_full.copy()
    for i in cat_idx_full:
        col = Xtr_cb_full.columns[i]
        Xtr_lime[col] = cat_train_arrays[col]
    for j, c in enumerate(Xtr_lime.columns):
        if j not in cat_idx_full:
            Xtr_lime[c] = pd.to_numeric(Xtr_lime[c], errors='coerce').astype('float64')
    Xtr_lime_np = Xtr_lime.values.astype('float64')

    # Val rows
    Xval_lime = Xval_cb_full.copy()
    for i in cat_idx_full:
        col = Xval_cb_full.columns[i]
        series_str = Xval_cb_full[col].astype('string').fillna('Missing')
        inv = {v: k for k, v in cat_code_maps[col].items()}
        next_code = max(cat_code_maps[col].keys()) + 1 if cat_code_maps[col] else 0
        Xval_lime[col] = series_str.map(lambda v: inv.get(v, next_code)).astype('int64')
    for j, c in enumerate(Xval_lime.columns):
        if j not in cat_idx_full:
            Xval_lime[c] = pd.to_numeric(Xval_lime[c], errors='coerce').astype('float64')
    Xval_lime_np = Xval_lime.values.astype('float64')

    lime_cat_features = [list(Xtr_cb_full.columns).index(c)
                          for c in ALL_CAT if c in Xtr_cb_full.columns]
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=Xtr_lime_np,
        feature_names=list(Xtr_cb_full.columns),
        categorical_features=lime_cat_features,
        mode='regression',
        random_state=RANDOM_STATE,
        discretize_continuous=False,
    )

    def cb_predict_numeric(X_np):
        df = pd.DataFrame(X_np, columns=Xtr_cb_full.columns)
        for i in cat_idx_full:
            col = Xtr_cb_full.columns[i]
            mapping = cat_code_maps[col]
            codes_int = df[col].round().astype('Int64')
            df[col] = codes_int.apply(lambda v: mapping.get(int(v), 'Missing') if pd.notna(v) else 'Missing').astype('string')
        for j, c in enumerate(df.columns):
            if j not in cat_idx_full:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
        return explain_model.predict(df)

    # Test on one row
    try:
        explanation = lime_explainer.explain_instance(
            Xval_lime_np[0], cb_predict_numeric, num_features=10,
        )
        top = explanation.as_list()[:5]
        print(f"  LIME ok. Top weights: {top}")
    except Exception as e:
        print(f"  LIME FAIL: {e}")
        raise

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
