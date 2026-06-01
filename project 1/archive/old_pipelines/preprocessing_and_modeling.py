"""
True-data LendingClub interest-rate pipeline.

This script trains an XGBoost regressor on true data/LC_train.csv and writes
5-fold ensemble predictions for true data/LC_test.csv.

Outputs:
  - outputs/test_predictions.csv
  - outputs/modeling_report.md
  - outputs/oof_predictions.csv
  - outputs/feature_importance.csv
  - outputs/xgb_fold_1.json ... outputs/xgb_fold_5.json
  - outputs/optuna_study_true_data_v2.db

Runtime knobs:
  - OPTUNA_TRIALS: defaults to 100
  - N_FOLDS: defaults to 5
  - XGB_NUM_BOOST_ROUND: defaults to 3000
  - XGB_EARLY_STOPPING_ROUNDS: defaults to 75
  - TARGET_ENCODING_SMOOTHING: defaults to 50
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


RANDOM_STATE = 42
TARGET_COL = "int_rate"
ID_COL = "ID"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "true data"
TRAIN_PATH = DATA_DIR / "LC_train.csv"
TEST_PATH = DATA_DIR / "LC_test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FEATURE_SET_VERSION = "v2_fico_econ_target_encoding"
STUDY_PATH = OUTPUT_DIR / "optuna_study_true_data_v2.db"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
REPORT_PATH = OUTPUT_DIR / "modeling_report.md"
OOF_PATH = OUTPUT_DIR / "oof_predictions.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"

N_FOLDS = int(os.environ.get("N_FOLDS", "5"))
OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "100"))
NUM_BOOST_ROUND = int(os.environ.get("XGB_NUM_BOOST_ROUND", "3000"))
EARLY_STOPPING_ROUNDS = int(os.environ.get("XGB_EARLY_STOPPING_ROUNDS", "75"))
TARGET_ENCODING_SMOOTHING = float(os.environ.get("TARGET_ENCODING_SMOOTHING", "50"))
XGB_VERBOSE_EVAL = int(os.environ.get("XGB_VERBOSE_EVAL", "0"))

LEAKAGE_OR_UNUSED_COLS = [
    "loan_status",
    "emp_title",
    "title",
    "fico_range_low",
    "fico_range_high",
    "emp_length",
    "term",
]
STRICTLY_FORBIDDEN_FEATURES = {
    TARGET_COL,
    ID_COL,
    "loan_status",
    "grade",
    "sub_grade",
    "installment",
}

CATEGORICAL_COLS = [
    "addr_state",
    "application_type",
    "home_ownership",
    "purpose",
    "verification_status",
    "zip_code",
    "term_cat",
    "fico_band",
    "dti_band",
    "util_band",
    "all_util_band",
    "loan_amount_band",
    "income_band",
    "emp_tenure_band",
    "purpose_term",
    "purpose_fico_band",
    "state_term",
    "zip_term",
    "home_term",
    "verification_term",
    "application_term",
    "home_emp_band",
    "fico_util_band",
]

TARGET_ENCODING_COLS = [
    "addr_state",
    "application_type",
    "home_ownership",
    "purpose",
    "verification_status",
    "zip_code",
    "term_cat",
    "fico_band",
    "dti_band",
    "util_band",
    "all_util_band",
    "loan_amount_band",
    "income_band",
    "purpose_term",
    "purpose_fico_band",
    "state_term",
    "zip_term",
    "home_term",
    "verification_term",
    "application_term",
    "home_emp_band",
    "fico_util_band",
]

MISSING_FLAG_COLS = {
    "mths_since_last_record": "has_derog_record",
    "mths_since_recent_inq": "has_recent_inq",
    "mths_since_rcnt_il": "has_recent_il",
    "mths_since_recent_bc": "has_recent_bc",
}

ZERO_FLAG_COLS = [
    "pub_rec",
    "delinq_2yrs",
    "chargeoff_within_12_mths",
    "collections_12_mths_ex_med",
    "tot_coll_amt",
]

LOG1P_COLS = [
    "annual_inc",
    "revol_bal",
    "tot_cur_bal",
    "total_bal_ex_mort",
    "tot_coll_amt",
]

EMP_LENGTH_MAP = {
    "< 1 year": 0.0,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".5f") -> str:
    """Small dependency-free Markdown table renderer."""
    if df.empty:
        return "_No rows._"

    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def safe_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def read_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH, na_values=["NA"])
    test = pd.read_csv(TEST_PATH, na_values=["NA"])
    validate_raw_contract(train, test)
    return train, test


def validate_raw_contract(train: pd.DataFrame, test: pd.DataFrame) -> None:
    missing_paths = [str(p) for p in [TRAIN_PATH, TEST_PATH] if not p.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing input file(s): {missing_paths}")

    if TARGET_COL not in train.columns:
        raise ValueError(f"{TRAIN_PATH} must contain target column {TARGET_COL!r}.")
    if TARGET_COL in test.columns:
        raise ValueError(f"{TEST_PATH} must not contain target column {TARGET_COL!r}.")
    if ID_COL not in test.columns:
        raise ValueError(f"{TEST_PATH} must contain ID column {ID_COL!r}.")
    if not test[ID_COL].is_unique:
        raise ValueError("Test ID column must be unique.")
    if len(test) != 10_000:
        raise ValueError(f"Expected 10,000 test rows, found {len(test):,}.")


@dataclass
class LendingClubPreprocessor:
    """Fold-local preprocessing: feature engineering, imputation, categories."""

    numeric_medians: dict[str, float] = field(default_factory=dict)
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    target_mean: float = 0.0
    target_encoding_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    categorical_columns_: list[str] = field(default_factory=list)

    def fit(self, raw_df: pd.DataFrame) -> "LendingClubPreprocessor":
        features = self._engineer(raw_df)
        self._fit_target_encodings(features, raw_df[TARGET_COL])
        features = self._apply_target_encodings(features)
        self.categorical_columns_ = [c for c in CATEGORICAL_COLS if c in features.columns]

        for col in self.categorical_columns_:
            observed = features[col].astype("string").fillna("missing")
            levels = sorted(v for v in observed.unique().tolist() if v != "missing")
            self.category_levels[col] = ["missing", *levels]

        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median = features[col].median()
            self.numeric_medians[col] = 0.0 if pd.isna(median) else float(median)

        self.feature_columns = features.columns.tolist()
        return self

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns:
            raise RuntimeError("Preprocessor must be fit before transform.")

        features = self._engineer(raw_df)
        features = self._apply_target_encodings(features)

        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = np.nan
        features = features[self.feature_columns]

        for col, median in self.numeric_medians.items():
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors="coerce").fillna(median)

        for col in self.categorical_columns_:
            levels = self.category_levels[col]
            as_string = features[col].astype("string").fillna("missing")
            as_string = as_string.where(as_string.isin(levels), "missing")
            features[col] = pd.Categorical(as_string, categories=levels)

        self._validate_feature_matrix(features)
        return features

    def fit_transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(raw_df).transform(raw_df)

    def _fit_target_encodings(self, features: pd.DataFrame, y: pd.Series) -> None:
        """Fit smoothed category-to-target maps using only the current fold's fit rows."""
        self.target_mean = float(y.mean())
        self.target_encoding_maps = {}
        for col in TARGET_ENCODING_COLS:
            if col not in features.columns:
                continue
            tmp = pd.DataFrame(
                {
                    "key": features[col].astype("string").fillna("missing"),
                    "target": y.to_numpy(),
                }
            )
            grouped = tmp.groupby("key", dropna=False)["target"].agg(["count", "mean"])
            smoothed = (
                grouped["count"] * grouped["mean"]
                + TARGET_ENCODING_SMOOTHING * self.target_mean
            ) / (grouped["count"] + TARGET_ENCODING_SMOOTHING)
            self.target_encoding_maps[col] = smoothed.to_dict()

    def _apply_target_encodings(self, features: pd.DataFrame) -> pd.DataFrame:
        for col, mapping in self.target_encoding_maps.items():
            if col not in features.columns:
                continue
            key = features[col].astype("string").fillna("missing")
            features[f"{col}_target_mean"] = key.map(mapping).astype(float).fillna(self.target_mean)
        return features

    def _engineer(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
        df["term_months"] = (
            df["term"].astype("string").str.extract(r"(\d+)", expand=False).astype(float)
        )
        df["emp_length_num"] = df["emp_length"].map(EMP_LENGTH_MAP).astype(float)
        df["term_cat"] = df["term"].astype("string").fillna("missing")

        df["fico_centered"] = df["fico"] - 700.0
        df["fico_centered_sq"] = df["fico_centered"] ** 2
        df["fico_inv"] = 1000.0 / (df["fico"] + 1.0)
        df["fico_shortfall_680"] = (680.0 - df["fico"]).clip(lower=0)
        df["fico_shortfall_700"] = (700.0 - df["fico"]).clip(lower=0)
        df["fico_shortfall_720"] = (720.0 - df["fico"]).clip(lower=0)
        df["fico_shortfall_740"] = (740.0 - df["fico"]).clip(lower=0)
        df["fico_surplus_740"] = (df["fico"] - 740.0).clip(lower=0)

        df["fico_band"] = pd.cut(
            df["fico"],
            bins=[0, 680, 700, 720, 740, 760, 800, np.inf],
            labels=["lt680", "680_699", "700_719", "720_739", "740_759", "760_799", "800_plus"],
            include_lowest=True,
        ).astype("string")
        df["dti_band"] = pd.cut(
            df["dti"],
            bins=[-np.inf, 10, 20, 30, 40, np.inf],
            labels=["dti_lt10", "dti_10_20", "dti_20_30", "dti_30_40", "dti_40_plus"],
        ).astype("string")
        df["util_band"] = pd.cut(
            df["revol_util"],
            bins=[-np.inf, 20, 40, 60, 80, 100, np.inf],
            labels=["util_lt20", "util_20_40", "util_40_60", "util_60_80", "util_80_100", "util_100_plus"],
        ).astype("string")
        df["all_util_band"] = pd.cut(
            df["all_util"],
            bins=[-np.inf, 20, 40, 60, 80, 100, np.inf],
            labels=[
                "all_util_lt20",
                "all_util_20_40",
                "all_util_40_60",
                "all_util_60_80",
                "all_util_80_100",
                "all_util_100_plus",
            ],
        ).astype("string")
        df["loan_amount_band"] = pd.cut(
            df["loan_amnt"],
            bins=[-np.inf, 5_000, 10_000, 20_000, 30_000, np.inf],
            labels=["loan_lt5k", "loan_5_10k", "loan_10_20k", "loan_20_30k", "loan_30k_plus"],
        ).astype("string")
        df["income_band"] = pd.cut(
            df["annual_inc"],
            bins=[-np.inf, 40_000, 70_000, 100_000, 150_000, np.inf],
            labels=["inc_lt40k", "inc_40_70k", "inc_70_100k", "inc_100_150k", "inc_150k_plus"],
        ).astype("string")
        df["emp_tenure_band"] = pd.cut(
            df["emp_length_num"],
            bins=[-np.inf, 0, 2, 5, 9, np.inf],
            labels=["emp_unknown_or_lt1", "emp_1_2", "emp_3_5", "emp_6_9", "emp_10_plus"],
        ).astype("string")

        df["purpose_term"] = df["purpose"].astype("string").fillna("missing") + "__" + df["term_cat"]
        df["purpose_fico_band"] = (
            df["purpose"].astype("string").fillna("missing")
            + "__"
            + df["fico_band"].fillna("missing")
        )
        df["state_term"] = df["addr_state"].astype("string").fillna("missing") + "__" + df["term_cat"]
        df["zip_term"] = df["zip_code"].astype("string").fillna("missing") + "__" + df["term_cat"]
        df["home_term"] = df["home_ownership"].astype("string").fillna("missing") + "__" + df["term_cat"]
        df["verification_term"] = (
            df["verification_status"].astype("string").fillna("missing") + "__" + df["term_cat"]
        )
        df["application_term"] = (
            df["application_type"].astype("string").fillna("missing") + "__" + df["term_cat"]
        )
        df["home_emp_band"] = (
            df["home_ownership"].astype("string").fillna("missing")
            + "__"
            + df["emp_tenure_band"].fillna("missing")
        )
        df["fico_util_band"] = df["fico_band"].fillna("missing") + "__" + df["util_band"].fillna("missing")

        for source, flag in MISSING_FLAG_COLS.items():
            df[flag] = df[source].notna().astype(np.int8)

        for source in ZERO_FLAG_COLS:
            df[f"{source}_flag"] = (df[source].fillna(0) > 0).astype(np.int8)

        for source in LOG1P_COLS:
            df[f"{source}_log"] = np.log1p(df[source].clip(lower=0))

        df["loan_to_income"] = safe_divide(df["loan_amnt"], df["annual_inc"] + 1.0)
        df["dti_x_log_income"] = df["dti"] * df["annual_inc_log"]
        df["fico_x_term"] = df["fico"] * df["term_months"]
        df["fico_x_revol_util"] = df["fico"] * df["revol_util"]
        df["fico_x_all_util"] = df["fico"] * df["all_util"]
        df["fico_x_dti"] = df["fico"] * df["dti"]
        df["fico_x_loan_to_income"] = df["fico"] * df["loan_to_income"]
        df["fico_x_inq_last_12m"] = df["fico"] * df["inq_last_12m"]
        df["fico_x_inq_fi"] = df["fico"] * df["inq_fi"]
        df["inq_per_open_acc"] = safe_divide(df["inq_last_12m"], df["open_acc"] + 1.0)
        df["total_inquiries"] = df["inq_last_12m"].fillna(0) + df["inq_fi"].fillna(0)
        df["delinq_per_credit_age"] = safe_divide(
            df["delinq_2yrs"], (df["mo_sin_old_rev_tl_op"] / 12.0) + 1.0
        )
        df["pub_rec_bankruptcy_combo"] = (
            df["pub_rec"].fillna(0) + df["pub_rec_bankruptcies"].fillna(0)
        )
        df["revol_util_x_open_acc"] = df["revol_util"] * df["open_acc"]
        df["loan_amnt_x_term"] = df["loan_amnt"] * df["term_months"]
        df["loan_amnt_x_revol_util"] = df["loan_amnt"] * (df["revol_util"] / 100.0)

        df["credit_age_years"] = df["mo_sin_old_rev_tl_op"] / 12.0
        df["credit_age_log"] = np.log1p(df["credit_age_years"].clip(lower=0))
        df["thin_file_flag"] = (df["credit_age_years"] < 3).astype(np.int8)
        df["fico_x_credit_age_log"] = df["fico"] * df["credit_age_log"]
        df["fico_shortfall_720_x_revol_util"] = df["fico_shortfall_720"] * df["revol_util"]
        df["fico_shortfall_720_x_all_util"] = df["fico_shortfall_720"] * df["all_util"]
        df["fico_shortfall_720_x_dti"] = df["fico_shortfall_720"] * df["dti"]
        df["fico_shortfall_720_x_inquiries"] = df["fico_shortfall_720"] * df["total_inquiries"]
        df["fico_shortfall_720_x_pub_rec"] = df["fico_shortfall_720"] * df["pub_rec"].fillna(0)
        df["term_x_fico_shortfall_720"] = df["term_months"] * df["fico_shortfall_720"]
        df["term_x_util_pressure"] = df["term_months"] * ((df["revol_util"].fillna(0) + df["all_util"].fillna(0)) / 200.0)

        # Economics-inspired proxies using only origination-time borrower data.
        df["capacity_stress"] = df["loan_to_income"] * (df["dti"] / 100.0)
        df["liquidity_stress"] = (df["revol_util"].fillna(0) / 100.0) * (df["all_util"].fillna(0) / 100.0)
        df["leverage_stress"] = (df["all_util"].fillna(0) / 100.0) * (df["dti"].fillna(0) / 100.0)
        df["term_premium_proxy"] = df["term_months"] * (df["loan_to_income"].fillna(0) + df["dti"].fillna(0) / 100.0)
        df["adverse_selection_proxy"] = (
            df["fico_shortfall_720"].fillna(0) / 100.0
            + df["total_inquiries"].fillna(0) / 10.0
            + df["delinq_2yrs"].fillna(0)
            + df["pub_rec_bankruptcy_combo"].fillna(0)
        )
        df["credit_rationing_score"] = (
            df["fico_shortfall_720"].fillna(0) / 100.0
            + df["dti"].fillna(0) / 50.0
            + df["revol_util"].fillna(0) / 100.0
            + df["inq_per_open_acc"].fillna(0)
            + df["pub_rec_flag"].fillna(0)
            + df["delinq_2yrs_flag"].fillna(0)
        )
        df["renter_short_emp_flag"] = (
            (df["home_ownership"].astype("string") == "RENT") & (df["emp_length_num"].fillna(0) <= 2)
        ).astype(np.int8)
        df["mortgage_long_emp_flag"] = (
            (df["home_ownership"].astype("string") == "MORTGAGE") & (df["emp_length_num"].fillna(0) >= 10)
        ).astype(np.int8)
        df["home_capacity_stress"] = df["capacity_stress"] * (
            df["home_ownership"].astype("string").isin(["RENT", "OWN"]).astype(float)
        )

        drop_cols = [c for c in LEAKAGE_OR_UNUSED_COLS + [TARGET_COL, ID_COL] if c in df.columns]
        df = df.drop(columns=drop_cols)

        return df

    @staticmethod
    def _validate_feature_matrix(features: pd.DataFrame) -> None:
        forbidden = STRICTLY_FORBIDDEN_FEATURES.intersection(features.columns)
        if forbidden:
            raise ValueError(f"Forbidden feature(s) present: {sorted(forbidden)}")

        non_numeric_non_cat = [
            c
            for c in features.columns
            if not pd.api.types.is_numeric_dtype(features[c])
            and not isinstance(features[c].dtype, pd.CategoricalDtype)
        ]
        if non_numeric_non_cat:
            raise TypeError(f"Unexpected non-numeric/non-categorical columns: {non_numeric_non_cat}")


def make_dmatrix(X: pd.DataFrame, y: np.ndarray | None = None) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, enable_categorical=True)


def predict_with_best_iteration(booster: xgb.Booster, dmatrix: xgb.DMatrix) -> np.ndarray:
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is None:
        return booster.predict(dmatrix)
    return booster.predict(dmatrix, iteration_range=(0, int(best_iteration) + 1))


def base_xgb_params() -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": RANDOM_STATE,
        "nthread": max(os.cpu_count() or 1, 1),
    }


def train_booster(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
    verbose_eval: bool | int = False,
) -> xgb.Booster:
    dtrain = make_dmatrix(X_train, y_train)
    dval = make_dmatrix(X_val, y_val)
    return xgb.train(
        params={**base_xgb_params(), **params},
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dval, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=verbose_eval,
    )


def tune_hyperparameters(train: pd.DataFrame) -> tuple[dict[str, Any], dict[str, float]]:
    print("\n=== Optuna tuning ===")
    train_idx, val_idx = train_test_split(
        np.arange(len(train)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    raw_train = train.iloc[train_idx].reset_index(drop=True)
    raw_val = train.iloc[val_idx].reset_index(drop=True)
    y_train = raw_train[TARGET_COL].to_numpy()
    y_val = raw_val[TARGET_COL].to_numpy()

    preprocessor = LendingClubPreprocessor()
    X_train = preprocessor.fit_transform(raw_train)
    X_val = preprocessor.transform(raw_val)

    def objective(trial: optuna.Trial) -> float:
        trial_t0 = time.perf_counter()
        trial_run_no = trial.number - before_trials + 1
        print(
            f"  Trial {trial_run_no}/{OPTUNA_TRIALS} "
            f"(study trial #{trial.number}) starting...",
            flush=True,
        )
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 1, 8),
        }
        booster = train_booster(
            X_train,
            y_train,
            X_val,
            y_val,
            params,
            verbose_eval=XGB_VERBOSE_EVAL if XGB_VERBOSE_EVAL > 0 else False,
        )
        preds = predict_with_best_iteration(booster, make_dmatrix(X_val))
        trial_rmse = rmse(y_val, preds)
        print(
            f"  Trial {trial_run_no}/{OPTUNA_TRIALS} complete: "
            f"RMSE={trial_rmse:.5f}, "
            f"best_iter={getattr(booster, 'best_iteration', 'n/a')}, "
            f"elapsed={time.perf_counter() - trial_t0:.1f}s",
            flush=True,
        )
        return trial_rmse

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_STATE),
        storage=f"sqlite:///{STUDY_PATH.as_posix()}",
        study_name=f"lc_true_data_xgb_{FEATURE_SET_VERSION}",
        load_if_exists=True,
    )
    before_trials = len(study.trials)
    print(f"Study: {STUDY_PATH}")
    print(f"Running {OPTUNA_TRIALS} trial(s); existing trials: {before_trials}", flush=True)
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best_params = dict(study.best_params)
    validation_metrics = {
        "tuning_best_rmse": float(study.best_value),
        "tuning_trials_total": float(len(study.trials)),
    }
    print(f"Best tuning RMSE: {study.best_value:.5f}")
    print(f"Best params: {json.dumps(best_params, sort_keys=True)}")
    return best_params, validation_metrics


def train_kfold_ensemble(
    train: pd.DataFrame,
    test: pd.DataFrame,
    best_params: dict[str, Any],
) -> dict[str, Any]:
    print("\n=== 5-fold ensemble ===")
    ids = test[ID_COL].to_numpy()
    y = train[TARGET_COL].to_numpy()
    oof_pred = np.full(len(train), np.nan, dtype=float)
    test_pred_matrix = np.zeros((len(test), N_FOLDS), dtype=float)

    fold_rows: list[dict[str, Any]] = []
    feature_importance_frames: list[pd.DataFrame] = []
    feature_columns: list[str] | None = None

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold_no, (fit_idx, val_idx) in enumerate(kfold.split(train), start=1):
        print(f"Starting fold {fold_no}/{N_FOLDS}...", flush=True)
        fold_t0 = time.perf_counter()
        raw_fit = train.iloc[fit_idx].reset_index(drop=True)
        raw_val = train.iloc[val_idx].reset_index(drop=True)

        y_fit = raw_fit[TARGET_COL].to_numpy()
        y_val = raw_val[TARGET_COL].to_numpy()

        preprocessor = LendingClubPreprocessor()
        X_fit = preprocessor.fit_transform(raw_fit)
        X_val = preprocessor.transform(raw_val)
        X_test = preprocessor.transform(test)

        if feature_columns is None:
            feature_columns = preprocessor.feature_columns
        elif feature_columns != preprocessor.feature_columns:
            raise ValueError("Feature columns changed across folds.")

        booster = train_booster(X_fit, y_fit, X_val, y_val, best_params)

        val_pred = predict_with_best_iteration(booster, make_dmatrix(X_val))
        test_pred = predict_with_best_iteration(booster, make_dmatrix(X_test))

        oof_pred[val_idx] = val_pred
        test_pred_matrix[:, fold_no - 1] = test_pred

        model_path = OUTPUT_DIR / f"xgb_fold_{fold_no}.json"
        booster.save_model(model_path)

        fold_metrics = {
            "fold": fold_no,
            "rmse": rmse(y_val, val_pred),
            "mae": mae(y_val, val_pred),
            "r2": r2_score_np(y_val, val_pred),
            "best_iteration": int(getattr(booster, "best_iteration", NUM_BOOST_ROUND - 1)),
            "seconds": time.perf_counter() - fold_t0,
            "model_path": model_path.name,
        }
        fold_rows.append(fold_metrics)
        print(
            f"Fold {fold_no}: RMSE={fold_metrics['rmse']:.5f} "
            f"MAE={fold_metrics['mae']:.5f} R2={fold_metrics['r2']:.5f} "
            f"best_iter={fold_metrics['best_iteration']}",
            flush=True,
        )

        importance = booster.get_score(importance_type="gain")
        fold_importance = pd.DataFrame(
            [{"feature": k, "gain": v, "fold": fold_no} for k, v in importance.items()]
        )
        feature_importance_frames.append(fold_importance)

    if np.isnan(oof_pred).any():
        raise RuntimeError("OOF predictions contain missing values.")

    ensemble_test_pred = test_pred_matrix.mean(axis=1)
    oof_metrics = {
        "oof_rmse": rmse(y, oof_pred),
        "oof_mae": mae(y, oof_pred),
        "oof_r2": r2_score_np(y, oof_pred),
    }

    predictions = pd.DataFrame({ID_COL: ids, "int_rate_pred": ensemble_test_pred})
    validate_outputs(test, predictions, oof_pred)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    oof = pd.DataFrame(
        {
            "row_id": np.arange(len(train)),
            "int_rate": y,
            "oof_pred": oof_pred,
            "residual": y - oof_pred,
        }
    )
    oof.to_csv(OOF_PATH, index=False)

    feature_importance = aggregate_feature_importance(feature_importance_frames, feature_columns or [])
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    return {
        "folds": fold_rows,
        "oof_metrics": oof_metrics,
        "feature_columns": feature_columns or [],
        "feature_importance": feature_importance,
        "prediction_summary": {
            "min": float(ensemble_test_pred.min()),
            "max": float(ensemble_test_pred.max()),
            "mean": float(ensemble_test_pred.mean()),
            "std": float(ensemble_test_pred.std()),
        },
    }


def aggregate_feature_importance(
    fold_importance_frames: list[pd.DataFrame],
    feature_columns: list[str],
) -> pd.DataFrame:
    if not fold_importance_frames:
        return pd.DataFrame(columns=["feature", "mean_gain", "folds_present"])

    all_importance = pd.concat(fold_importance_frames, ignore_index=True)
    if all_importance.empty:
        return pd.DataFrame({"feature": feature_columns, "mean_gain": 0.0, "folds_present": 0})

    summary = (
        all_importance.groupby("feature", as_index=False)
        .agg(mean_gain=("gain", "mean"), folds_present=("fold", "nunique"))
        .sort_values("mean_gain", ascending=False)
    )
    missing = sorted(set(feature_columns) - set(summary["feature"]))
    if missing:
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    {"feature": missing, "mean_gain": 0.0, "folds_present": 0}
                ),
            ],
            ignore_index=True,
        )
    return summary


def validate_outputs(test: pd.DataFrame, predictions: pd.DataFrame, oof_pred: np.ndarray) -> None:
    if len(predictions) != 10_000:
        raise ValueError(f"Expected 10,000 predictions, found {len(predictions):,}.")
    if not predictions[ID_COL].equals(test[ID_COL].reset_index(drop=True)):
        raise ValueError("Prediction IDs do not match test IDs in order.")
    if not np.isfinite(predictions["int_rate_pred"].to_numpy()).all():
        raise ValueError("Test predictions contain non-finite values.")
    if not np.isfinite(oof_pred).all():
        raise ValueError("OOF predictions contain non-finite values.")


def render_report(
    best_params: dict[str, Any],
    tuning_metrics: dict[str, float],
    ensemble_results: dict[str, Any],
    elapsed_seconds: float,
) -> str:
    fold_df = pd.DataFrame(ensemble_results["folds"])
    feature_importance = ensemble_results["feature_importance"]
    top_features = feature_importance.head(25)
    pred_summary = ensemble_results["prediction_summary"]
    oof = ensemble_results["oof_metrics"]
    feature_columns = ensemble_results["feature_columns"]

    prediction_warning = ""
    if pred_summary["min"] < 0 or pred_summary["max"] > 40:
        prediction_warning = (
            "\n\n**Warning:** prediction range is outside the expected 0-40 percentage "
            "point band. Values were not clipped."
        )

    fold_table = dataframe_to_markdown(
        fold_df[["fold", "rmse", "mae", "r2", "best_iteration", "seconds", "model_path"]]
    )
    top_feature_table = dataframe_to_markdown(top_features)
    params_json = json.dumps(best_params, indent=2, sort_keys=True)

    return f"""# True-Data XGBoost Modeling Report

Generated by `src/preprocessing_and_modeling.py`.

## Summary

- Input train: `true data/LC_train.csv`
- Input test: `true data/LC_test.csv`
- Model: XGBoost regression with native categorical support
- Feature set: `{FEATURE_SET_VERSION}`
- Tuning: Optuna TPE, `{int(tuning_metrics["tuning_trials_total"])}` total trial(s) in study
- Final prediction: `{N_FOLDS}`-fold ensemble averaged on test
- Runtime: `{elapsed_seconds:.1f}` seconds

## Validation Metrics

- Optuna holdout best RMSE: `{tuning_metrics["tuning_best_rmse"]:.5f}`
- OOF RMSE: `{oof["oof_rmse"]:.5f}`
- OOF MAE: `{oof["oof_mae"]:.5f}`
- OOF R2: `{oof["oof_r2"]:.5f}`

The provided test set has no `int_rate` labels, so test RMSE/R2 cannot be
computed locally. The OOF metrics above are the leakage-safe estimate of
generalization performance on labeled data.

## Fold Metrics

{fold_table}

## Best Parameters

```json
{params_json}
```

## Test Prediction Summary

- Rows: `10000`
- Min: `{pred_summary["min"]:.5f}`
- Max: `{pred_summary["max"]:.5f}`
- Mean: `{pred_summary["mean"]:.5f}`
- Std: `{pred_summary["std"]:.5f}`{prediction_warning}

## Top Feature Importance

Mean gain across fold models.

{top_feature_table}

## Feature Contract

- Feature count: `{len(feature_columns)}`
- Excluded leakage/unavailable fields: `loan_status`, `grade`, `sub_grade`, `installment`
- Dropped free-text/duplicates: `emp_title`, `title`, raw `fico_range_low`, raw `fico_range_high`
- Invalid doc interactions intentionally excluded: `payment_to_income`, macro interactions, grade/sub-grade crosses, `earliest_cr_line_age`
- Added FICO-heavy terms: score bands, score shortfalls, nonlinear score transforms, and FICO crosses with term, DTI, utilization, inquiries, and credit age
- Added economics-inspired proxies: capacity stress, liquidity stress, leverage stress, term premium proxy, adverse-selection proxy, credit-rationing score, and collateral/home-stability flags
- Added fold-local smoothed target encodings for categorical pricing patterns and categorical crosses

## Outputs

- `outputs/test_predictions.csv`
- `outputs/oof_predictions.csv`
- `outputs/feature_importance.csv`
- `outputs/xgb_fold_1.json` through `outputs/xgb_fold_{N_FOLDS}.json`
"""


def main() -> None:
    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Load data ===")
    train, test = read_raw_data()
    print(f"Train: {train.shape}, Test: {test.shape}")

    best_params, tuning_metrics = tune_hyperparameters(train)
    ensemble_results = train_kfold_ensemble(train, test, best_params)

    elapsed = time.perf_counter() - t0
    report = render_report(best_params, tuning_metrics, ensemble_results, elapsed)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\n=== Complete ===")
    print(f"OOF RMSE: {ensemble_results['oof_metrics']['oof_rmse']:.5f}")
    print(f"OOF MAE:  {ensemble_results['oof_metrics']['oof_mae']:.5f}")
    print(f"OOF R2:   {ensemble_results['oof_metrics']['oof_r2']:.5f}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Report:      {REPORT_PATH}")
    print(f"Elapsed:     {elapsed:.1f}s")


if __name__ == "__main__":
    main()
