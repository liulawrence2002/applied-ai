"""
Leakage-safe v2 ablation and ensemble runner for LendingClub int_rate.

This script is for controlled experiments, not the final production pipeline.
It keeps validation fixed at 5-fold random CV with seed=42 unless a smoke-test
sample is explicitly requested.

Primary modes:
  - contract: validate feature engineering/leakage contract quickly
  - ablate: run v1 baseline plus one feature block at a time with XGBoost
  - ensemble: train XGBoost, LightGBM, CatBoost, simple average, and NNLS blend
  - all: run ablation, then ensemble using the blocks selected by ablation

Outputs:
  - outputs/experiments_log.md
  - outputs/v2_ablation_results.csv
  - outputs/v2_selected_blocks.txt
  - outputs/v2_model_oof_predictions.csv
  - outputs/v2_ensemble_test_predictions.csv
  - outputs/v2_ensemble_results.csv
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import nnls
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=FutureWarning)


RANDOM_STATE = 42
TARGET_COL = "int_rate"
ID_COL = "ID"
N_FOLDS = 5
TARGET_ENCODING_SMOOTHING = 50.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "true data"
TRAIN_PATH = DATA_DIR / "LC_train.csv"
TEST_PATH = DATA_DIR / "LC_test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

EXPERIMENT_LOG_PATH = OUTPUT_DIR / "experiments_log.md"
ABLATION_RESULTS_PATH = OUTPUT_DIR / "v2_ablation_results.csv"
SELECTED_BLOCKS_PATH = OUTPUT_DIR / "v2_selected_blocks.txt"
OOF_PREDICTIONS_PATH = OUTPUT_DIR / "v2_model_oof_predictions.csv"
ENSEMBLE_TEST_PREDICTIONS_PATH = OUTPUT_DIR / "v2_ensemble_test_predictions.csv"
ENSEMBLE_RESULTS_PATH = OUTPUT_DIR / "v2_ensemble_results.csv"

NUM_BOOST_ROUND = int(os.environ.get("V2_NUM_BOOST_ROUND", "3000"))
EARLY_STOPPING_ROUNDS = int(os.environ.get("V2_EARLY_STOPPING_ROUNDS", "75"))
CATBOOST_ITERATIONS = int(os.environ.get("V2_CATBOOST_ITERATIONS", str(NUM_BOOST_ROUND)))
SAMPLE_ROWS = int(os.environ.get("V2_SAMPLE_ROWS", "0"))

HISTORICAL_V1_RMSE = 3.90322

FORBIDDEN_FEATURES = {
    TARGET_COL,
    ID_COL,
    "loan_status",
    "grade",
    "sub_grade",
    "installment",
    "title",
    "emp_title",
    "fico_range_low",
    "fico_range_high",
    "term",
    "emp_length",
}

DROP_COLS = sorted(FORBIDDEN_FEATURES)

BASE_CATEGORICAL_COLS = [
    "addr_state",
    "application_type",
    "home_ownership",
    "purpose",
    "verification_status",
    "zip_code",
    "term_cat",
]

LOW_CARD_TE_COLS = ("purpose", "term_cat", "application_type", "home_ownership")
HIGH_CARD_TE_COLS = ("zip_code", "addr_state", "emp_tenure_band")
CROSSED_TE_COLS = ("purpose_term", "purpose_fico_band", "zip_term", "state_term")

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

HISTORICAL_V1_XGB_PARAMS = {
    "learning_rate": 0.010188389076377652,
    "max_depth": 8,
    "min_child_weight": 26.66728772081506,
    "subsample": 0.935054932261084,
    "colsample_bytree": 0.6964907784550468,
    "reg_alpha": 8.928086127845225e-07,
    "reg_lambda": 23.816212859455263,
    "gamma": 0.21854027389704697,
    "max_cat_to_onehot": 5,
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "OOF_RMSE": rmse(y_true, y_pred),
        "OOF_MAE": mae(y_true, y_pred),
        "OOF_R2": r2_score_np(y_true, y_pred),
    }


def safe_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    add_fico_nonlinear: bool = False
    add_fico_crosses: bool = False
    add_categorical_crosses: bool = False
    add_economics_proxies: bool = False
    add_low_card_te: bool = False
    add_high_card_te: bool = False
    add_crossed_te: bool = False

    @property
    def target_encoding_cols(self) -> tuple[str, ...]:
        cols: list[str] = []
        if self.add_low_card_te:
            cols.extend(LOW_CARD_TE_COLS)
        if self.add_high_card_te:
            cols.extend(HIGH_CARD_TE_COLS)
        if self.add_crossed_te:
            cols.extend(CROSSED_TE_COLS)
        return tuple(dict.fromkeys(cols))

    @property
    def categorical_columns(self) -> list[str]:
        cols = list(BASE_CATEGORICAL_COLS)
        if self.needs_fico_band:
            cols.append("fico_band")
        if self.needs_emp_tenure_band:
            cols.append("emp_tenure_band")
        if self.add_categorical_crosses or self.add_crossed_te:
            cols.extend(["purpose_term", "purpose_fico_band", "zip_term", "state_term"])
        return list(dict.fromkeys(cols))

    @property
    def needs_fico_band(self) -> bool:
        return self.add_fico_nonlinear or self.add_categorical_crosses or self.add_crossed_te

    @property
    def needs_emp_tenure_band(self) -> bool:
        return self.add_high_card_te


V1_FEATURES = FeatureConfig(name="v1_base")

BLOCK_CONFIGS = {
    "fico_nonlinear": FeatureConfig(name="v1_plus_fico_nonlinear", add_fico_nonlinear=True),
    "fico_crosses": FeatureConfig(name="v1_plus_fico_crosses", add_fico_crosses=True),
    "categorical_crosses": FeatureConfig(
        name="v1_plus_categorical_crosses",
        add_categorical_crosses=True,
    ),
    "economics_proxies": FeatureConfig(
        name="v1_plus_economics_proxies",
        add_economics_proxies=True,
    ),
    "low_card_te": FeatureConfig(name="v1_plus_low_card_te", add_low_card_te=True),
    "high_card_te": FeatureConfig(name="v1_plus_high_card_te", add_high_card_te=True),
    "crossed_te": FeatureConfig(
        name="v1_plus_crossed_te",
        add_categorical_crosses=True,
        add_crossed_te=True,
    ),
}


def combine_feature_blocks(blocks: list[str]) -> FeatureConfig:
    valid = set(BLOCK_CONFIGS)
    unknown = sorted(set(blocks) - valid)
    if unknown:
        raise ValueError(f"Unknown feature block(s): {unknown}. Valid blocks: {sorted(valid)}")

    return FeatureConfig(
        name="v1_5_" + ("_".join(blocks) if blocks else "base"),
        add_fico_nonlinear=any(BLOCK_CONFIGS[b].add_fico_nonlinear for b in blocks),
        add_fico_crosses=any(BLOCK_CONFIGS[b].add_fico_crosses for b in blocks),
        add_categorical_crosses=any(BLOCK_CONFIGS[b].add_categorical_crosses for b in blocks),
        add_economics_proxies=any(BLOCK_CONFIGS[b].add_economics_proxies for b in blocks),
        add_low_card_te=any(BLOCK_CONFIGS[b].add_low_card_te for b in blocks),
        add_high_card_te=any(BLOCK_CONFIGS[b].add_high_card_te for b in blocks),
        add_crossed_te=any(BLOCK_CONFIGS[b].add_crossed_te for b in blocks),
    )


@dataclass
class LendingClubPreprocessor:
    config: FeatureConfig
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
        self.categorical_columns_ = [
            c for c in self.config.categorical_columns if c in features.columns
        ]

        for col in self.categorical_columns_:
            observed = features[col].astype("string").fillna("missing")
            levels = sorted(v for v in observed.unique().tolist() if v != "missing")
            self.category_levels[col] = ["missing", *levels]

        for col in features.select_dtypes(include=[np.number]).columns:
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
        self.target_mean = float(y.mean())
        self.target_encoding_maps = {}
        for col in self.config.target_encoding_cols:
            if col not in features.columns:
                raise ValueError(f"Target encoding column is unavailable: {col}")
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
        df["inq_per_open_acc"] = safe_divide(df["inq_last_12m"], df["open_acc"] + 1.0)
        df["delinq_per_credit_age"] = safe_divide(
            df["delinq_2yrs"], (df["mo_sin_old_rev_tl_op"] / 12.0) + 1.0
        )
        df["pub_rec_bankruptcy_combo"] = (
            df["pub_rec"].fillna(0) + df["pub_rec_bankruptcies"].fillna(0)
        )
        df["revol_util_x_open_acc"] = df["revol_util"] * df["open_acc"]
        df["loan_amnt_x_term"] = df["loan_amnt"] * df["term_months"]
        df["loan_amnt_x_revol_util"] = df["loan_amnt"] * (df["revol_util"] / 100.0)

        if self.config.needs_fico_band:
            df["fico_band"] = pd.cut(
                df["fico"],
                bins=[0, 680, 700, 720, 740, 760, 800, np.inf],
                labels=[
                    "lt680",
                    "680_699",
                    "700_719",
                    "720_739",
                    "740_759",
                    "760_799",
                    "800_plus",
                ],
                include_lowest=True,
            ).astype("string")

        if self.config.needs_emp_tenure_band:
            df["emp_tenure_band"] = pd.cut(
                df["emp_length_num"],
                bins=[-np.inf, 0, 2, 5, 9, np.inf],
                labels=["emp_unknown_or_lt1", "emp_1_2", "emp_3_5", "emp_6_9", "emp_10_plus"],
            ).astype("string")

        if self.config.add_fico_nonlinear:
            df["fico_centered"] = df["fico"] - 700.0
            df["fico_centered_sq"] = df["fico_centered"] ** 2
            df["fico_inv"] = 1000.0 / (df["fico"] + 1.0)
            df["fico_shortfall_680"] = (680.0 - df["fico"]).clip(lower=0)
            df["fico_shortfall_700"] = (700.0 - df["fico"]).clip(lower=0)
            df["fico_shortfall_720"] = (720.0 - df["fico"]).clip(lower=0)
            df["fico_shortfall_740"] = (740.0 - df["fico"]).clip(lower=0)
            df["fico_surplus_740"] = (df["fico"] - 740.0).clip(lower=0)

        if self.config.add_fico_crosses:
            fico_shortfall_720 = (720.0 - df["fico"]).clip(lower=0)
            total_inquiries = df["inq_last_12m"].fillna(0) + df["inq_fi"].fillna(0)
            credit_age_years = df["mo_sin_old_rev_tl_op"] / 12.0
            credit_age_log = np.log1p(credit_age_years.clip(lower=0))
            df["fico_x_dti"] = df["fico"] * df["dti"]
            df["fico_x_loan_to_income"] = df["fico"] * df["loan_to_income"]
            df["fico_x_inq_last_12m"] = df["fico"] * df["inq_last_12m"]
            df["fico_x_inq_fi"] = df["fico"] * df["inq_fi"]
            df["fico_x_credit_age_log"] = df["fico"] * credit_age_log
            df["fico_shortfall_720_x_revol_util"] = fico_shortfall_720 * df["revol_util"]
            df["fico_shortfall_720_x_all_util"] = fico_shortfall_720 * df["all_util"]
            df["fico_shortfall_720_x_dti"] = fico_shortfall_720 * df["dti"]
            df["fico_shortfall_720_x_inquiries"] = fico_shortfall_720 * total_inquiries
            df["fico_shortfall_720_x_pub_rec"] = fico_shortfall_720 * df["pub_rec"].fillna(0)
            df["term_x_fico_shortfall_720"] = df["term_months"] * fico_shortfall_720

        if self.config.add_categorical_crosses or self.config.add_crossed_te:
            df["purpose_term"] = df["purpose"].astype("string").fillna("missing") + "__" + df["term_cat"]
            df["purpose_fico_band"] = (
                df["purpose"].astype("string").fillna("missing")
                + "__"
                + df["fico_band"].fillna("missing")
            )
            df["zip_term"] = df["zip_code"].astype("string").fillna("missing") + "__" + df["term_cat"]
            df["state_term"] = df["addr_state"].astype("string").fillna("missing") + "__" + df["term_cat"]

        if self.config.add_economics_proxies:
            total_inquiries = df["inq_last_12m"].fillna(0) + df["inq_fi"].fillna(0)
            fico_shortfall_720 = (720.0 - df["fico"]).clip(lower=0)
            df["capacity_stress"] = df["loan_to_income"] * (df["dti"] / 100.0)
            df["liquidity_stress"] = (
                df["revol_util"].fillna(0) / 100.0
            ) * (df["all_util"].fillna(0) / 100.0)
            df["leverage_stress"] = (
                df["all_util"].fillna(0) / 100.0
            ) * (df["dti"].fillna(0) / 100.0)
            df["term_premium_proxy"] = df["term_months"] * (
                df["loan_to_income"].fillna(0) + df["dti"].fillna(0) / 100.0
            )
            df["adverse_selection_proxy"] = (
                fico_shortfall_720.fillna(0) / 100.0
                + total_inquiries / 10.0
                + df["delinq_2yrs"].fillna(0)
                + df["pub_rec_bankruptcy_combo"].fillna(0)
            )
            df["credit_rationing_score"] = (
                fico_shortfall_720.fillna(0) / 100.0
                + df["dti"].fillna(0) / 50.0
                + df["revol_util"].fillna(0) / 100.0
                + df["inq_per_open_acc"].fillna(0)
                + df["pub_rec_flag"].fillna(0)
                + df["delinq_2yrs_flag"].fillna(0)
            )
            df["renter_short_emp_flag"] = (
                (df["home_ownership"].astype("string") == "RENT")
                & (df["emp_length_num"].fillna(0) <= 2)
            ).astype(np.int8)
            df["mortgage_long_emp_flag"] = (
                (df["home_ownership"].astype("string") == "MORTGAGE")
                & (df["emp_length_num"].fillna(0) >= 10)
            ).astype(np.int8)
            df["home_capacity_stress"] = df["capacity_stress"] * (
                df["home_ownership"].astype("string").isin(["RENT", "OWN"]).astype(float)
            )

        return df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    @staticmethod
    def _validate_feature_matrix(features: pd.DataFrame) -> None:
        forbidden = FORBIDDEN_FEATURES.intersection(features.columns)
        if forbidden:
            raise ValueError(f"Forbidden feature(s) present: {sorted(forbidden)}")

        bad_columns = [
            c
            for c in features.columns
            if not pd.api.types.is_numeric_dtype(features[c])
            and not isinstance(features[c].dtype, pd.CategoricalDtype)
        ]
        if bad_columns:
            raise TypeError(f"Unexpected non-numeric/non-categorical columns: {bad_columns}")


def read_data(sample_rows: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing train file: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_PATH}")

    train = pd.read_csv(TRAIN_PATH, na_values=["NA"])
    test = pd.read_csv(TEST_PATH, na_values=["NA"])
    if sample_rows > 0:
        train = train.head(sample_rows).copy()
        test = test.head(min(sample_rows, len(test))).copy()

    if TARGET_COL not in train.columns:
        raise ValueError(f"{TRAIN_PATH} must contain {TARGET_COL!r}.")
    if TARGET_COL in test.columns:
        raise ValueError(f"{TEST_PATH} must not contain {TARGET_COL!r}.")
    if ID_COL not in test.columns:
        raise ValueError(f"{TEST_PATH} must contain {ID_COL!r}.")
    if sample_rows == 0 and len(test) != 10_000:
        raise ValueError(f"Expected 10,000 test rows, found {len(test):,}.")
    if not test[ID_COL].is_unique:
        raise ValueError("Test ID column must be unique.")
    return train, test


def make_xgb_dmatrix(X: pd.DataFrame, y: np.ndarray | None = None) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, enable_categorical=True)


def predict_xgb(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    best_iteration = getattr(booster, "best_iteration", None)
    dmatrix = make_xgb_dmatrix(X)
    if best_iteration is None:
        return booster.predict(dmatrix)
    return booster.predict(dmatrix, iteration_range=(0, int(best_iteration) + 1))


def xgb_base_params(model_seed: int = RANDOM_STATE) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": model_seed,
        "nthread": max(os.cpu_count() or 1, 1),
    }


def fit_predict_xgb(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame | None,
    model_seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray | None]:
    booster = xgb.train(
        params={**xgb_base_params(model_seed), **HISTORICAL_V1_XGB_PARAMS},
        dtrain=make_xgb_dmatrix(X_fit, y_fit),
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(make_xgb_dmatrix(X_val, y_val), "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    val_pred = predict_xgb(booster, X_val)
    test_pred = predict_xgb(booster, X_test) if X_test is not None else None
    return val_pred, test_pred


def fit_predict_lgbm(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame | None,
    categorical_cols: list[str],
    model_seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray | None]:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        n_estimators=NUM_BOOST_ROUND,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=10.0,
        reg_alpha=0.0,
        random_state=model_seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        categorical_feature=categorical_cols,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
    test_pred = (
        model.predict(X_test, num_iteration=model.best_iteration_) if X_test is not None else None
    )
    return np.asarray(val_pred), None if test_pred is None else np.asarray(test_pred)


def catboost_frame(X: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    converted = X.copy()
    for col in categorical_cols:
        converted[col] = converted[col].astype("string").fillna("missing")
    return converted


def fit_predict_catboost(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame | None,
    categorical_cols: list[str],
    model_seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray | None]:
    from catboost import CatBoostRegressor

    X_fit_cb = catboost_frame(X_fit, categorical_cols)
    X_val_cb = catboost_frame(X_val, categorical_cols)
    X_test_cb = catboost_frame(X_test, categorical_cols) if X_test is not None else None

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=CATBOOST_ITERATIONS,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=10.0,
        random_seed=model_seed,
        od_type="Iter",
        od_wait=EARLY_STOPPING_ROUNDS,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
    model.fit(
        X_fit_cb,
        y_fit,
        cat_features=categorical_cols,
        eval_set=(X_val_cb, y_val),
        use_best_model=True,
    )
    val_pred = model.predict(X_val_cb)
    test_pred = model.predict(X_test_cb) if X_test_cb is not None else None
    return np.asarray(val_pred), None if test_pred is None else np.asarray(test_pred)


def run_cv(
    train: pd.DataFrame,
    feature_config: FeatureConfig,
    model_names: list[str],
    test: pd.DataFrame | None = None,
    fold_seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    y = train[TARGET_COL].to_numpy()
    ids = test[ID_COL].to_numpy() if test is not None else None
    oof = {model_name: np.full(len(train), np.nan, dtype=float) for model_name in model_names}
    test_pred_mats = {
        model_name: np.zeros((len(test), N_FOLDS), dtype=float)
        for model_name in model_names
        if test is not None
    }
    fold_rows: list[dict[str, Any]] = []
    feature_columns: list[str] | None = None
    categorical_columns: list[str] = []

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=fold_seed)
    for fold_no, (fit_idx, val_idx) in enumerate(kfold.split(train), start=1):
        print(f"  Fold {fold_no}/{N_FOLDS}")
        fold_t0 = time.perf_counter()
        raw_fit = train.iloc[fit_idx].reset_index(drop=True)
        raw_val = train.iloc[val_idx].reset_index(drop=True)
        y_fit = raw_fit[TARGET_COL].to_numpy()
        y_val = raw_val[TARGET_COL].to_numpy()

        preprocessor = LendingClubPreprocessor(feature_config)
        X_fit = preprocessor.fit_transform(raw_fit)
        X_val = preprocessor.transform(raw_val)
        X_test = preprocessor.transform(test) if test is not None else None

        if feature_columns is None:
            feature_columns = preprocessor.feature_columns
            categorical_columns = preprocessor.categorical_columns_
        elif feature_columns != preprocessor.feature_columns:
            raise ValueError("Feature columns changed across folds.")

        for model_name in model_names:
            model_t0 = time.perf_counter()
            if model_name == "xgb":
                val_pred, test_pred = fit_predict_xgb(X_fit, y_fit, X_val, y_val, X_test)
            elif model_name == "lgbm":
                val_pred, test_pred = fit_predict_lgbm(
                    X_fit,
                    y_fit,
                    X_val,
                    y_val,
                    X_test,
                    categorical_columns,
                )
            elif model_name == "catboost":
                val_pred, test_pred = fit_predict_catboost(
                    X_fit,
                    y_fit,
                    X_val,
                    y_val,
                    X_test,
                    categorical_columns,
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            oof[model_name][val_idx] = val_pred
            if test is not None and test_pred is not None:
                test_pred_mats[model_name][:, fold_no - 1] = test_pred
            fold_metric = metric_dict(y_val, val_pred)
            fold_rows.append(
                {
                    "fold": fold_no,
                    "model": model_name,
                    **fold_metric,
                    "seconds": time.perf_counter() - model_t0,
                }
            )
            print(
                f"    {model_name}: RMSE={fold_metric['OOF_RMSE']:.5f} "
                f"MAE={fold_metric['OOF_MAE']:.5f} R2={fold_metric['OOF_R2']:.5f}"
            )

        print(f"  Fold {fold_no} elapsed: {time.perf_counter() - fold_t0:.1f}s")

    for model_name, pred in oof.items():
        if np.isnan(pred).any() or not np.isfinite(pred).all():
            raise RuntimeError(f"{model_name} OOF predictions are incomplete/non-finite.")

    averaged_test_preds = {
        model_name: matrix.mean(axis=1) for model_name, matrix in test_pred_mats.items()
    }
    return {
        "y": y,
        "ids": ids,
        "oof": oof,
        "test_preds": averaged_test_preds,
        "folds": fold_rows,
        "feature_columns": feature_columns or [],
        "categorical_columns": categorical_columns,
    }


def compute_blends(y: np.ndarray, oof: dict[str, np.ndarray]) -> dict[str, Any]:
    model_names = list(oof)
    pred_matrix = np.column_stack([oof[name] for name in model_names])
    simple_avg = pred_matrix.mean(axis=1)
    weights, _ = nnls(pred_matrix, y)
    if float(weights.sum()) <= 0:
        weights = np.ones(len(model_names), dtype=float) / len(model_names)
    nnls_blend = pred_matrix @ weights
    return {
        "simple_avg": simple_avg,
        "nnls_blend": nnls_blend,
        "nnls_weights": dict(zip(model_names, [float(w) for w in weights])),
    }


def apply_test_blends(
    test_preds: dict[str, np.ndarray],
    nnls_weights: dict[str, float],
) -> dict[str, np.ndarray]:
    model_names = list(test_preds)
    pred_matrix = np.column_stack([test_preds[name] for name in model_names])
    simple_avg = pred_matrix.mean(axis=1)
    weights = np.array([nnls_weights[name] for name in model_names], dtype=float)
    nnls_blend = pred_matrix @ weights
    return {"simple_avg": simple_avg, "nnls_blend": nnls_blend}


def markdown_row(values: list[Any]) -> str:
    rendered = []
    for value in values:
        if isinstance(value, float):
            rendered.append(f"{value:.5f}")
        else:
            rendered.append(str(value).replace("|", "\\|"))
    return "| " + " | ".join(rendered) + " |"


def append_experiment_rows(rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not EXPERIMENT_LOG_PATH.exists():
        EXPERIMENT_LOG_PATH.write_text(
            "# Leakage-Safe RMSE Experiments\n\n"
            "| step | config | OOF_RMSE | OOF_MAE | OOF_R2 | delta_vs_v1 | notes |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | --- |\n",
            encoding="utf-8",
        )

    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(
                markdown_row(
                    [
                        row["step"],
                        row["config"],
                        row["OOF_RMSE"],
                        row["OOF_MAE"],
                        row["OOF_R2"],
                        row["delta_vs_v1"],
                        row["notes"],
                    ]
                )
                + "\n"
            )


def run_contract(sample_rows: int) -> None:
    train, test = read_data(sample_rows=sample_rows or 1000)
    configs = [V1_FEATURES, *BLOCK_CONFIGS.values(), combine_feature_blocks(list(BLOCK_CONFIGS))]
    for config in configs:
        raw_fit = train.iloc[: max(100, int(len(train) * 0.8))].reset_index(drop=True)
        raw_val = train.iloc[max(100, int(len(train) * 0.8)) :].reset_index(drop=True)
        preprocessor = LendingClubPreprocessor(config)
        X_fit = preprocessor.fit_transform(raw_fit)
        X_val = preprocessor.transform(raw_val)
        X_test = preprocessor.transform(test)
        if list(X_fit.columns) != list(X_val.columns) or list(X_fit.columns) != list(X_test.columns):
            raise RuntimeError(f"Feature columns do not align for {config.name}.")
        print(
            f"{config.name}: features={X_fit.shape[1]} "
            f"categoricals={len(preprocessor.categorical_columns_)} "
            f"target_encodings={len(config.target_encoding_cols)}"
        )
    print("Contract check passed: feature matrices align and forbidden columns are absent.")


def run_ablation(min_improvement: float, sample_rows: int) -> list[str]:
    train, _ = read_data(sample_rows=sample_rows)
    smoke = sample_rows > 0
    ablation_results_path = (
        OUTPUT_DIR / "v2_ablation_results_smoke.csv" if smoke else ABLATION_RESULTS_PATH
    )
    selected_blocks_path = (
        OUTPUT_DIR / "v2_selected_blocks_smoke.txt" if smoke else SELECTED_BLOCKS_PATH
    )
    print(f"=== Ablation ===\nTrain rows: {len(train):,}")
    if smoke:
        print("Smoke mode: writing _smoke artifacts and not appending experiments_log.md")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    print("\n=== Baseline: v1_base ===")
    baseline_cv = run_cv(train, V1_FEATURES, ["xgb"], test=None)
    y = baseline_cv["y"]
    baseline_metrics = metric_dict(y, baseline_cv["oof"]["xgb"])
    baseline_rmse = baseline_metrics["OOF_RMSE"]
    rows.append(
        {
            "step": "ablate",
            "config": "v1_base_xgb",
            **baseline_metrics,
            "delta_vs_v1": 0.0,
            "notes": (
                f"feature_count={len(baseline_cv['feature_columns'])}; "
                f"elapsed_sec={time.perf_counter() - started:.1f}; baseline"
            ),
        }
    )

    for block_name, config in BLOCK_CONFIGS.items():
        print(f"\n=== Block: {block_name} ===")
        block_t0 = time.perf_counter()
        cv_result = run_cv(train, config, ["xgb"], test=None)
        metrics = metric_dict(cv_result["y"], cv_result["oof"]["xgb"])
        delta = metrics["OOF_RMSE"] - baseline_rmse
        keep = delta <= -min_improvement
        rows.append(
            {
                "step": "ablate",
                "config": f"{config.name}_xgb",
                **metrics,
                "delta_vs_v1": delta,
                "notes": (
                    f"block={block_name}; feature_count={len(cv_result['feature_columns'])}; "
                    f"elapsed_sec={time.perf_counter() - block_t0:.1f}; "
                    f"{'KEEP' if keep else 'REMOVE'} threshold={min_improvement:.5f}"
                ),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(ablation_results_path, index=False)
    if not smoke:
        append_experiment_rows(rows)

    selected_blocks = []
    for _, row in results.iloc[1:].iterrows():
        block = str(row["notes"]).split("block=", 1)[1].split(";", 1)[0]
        if float(row["delta_vs_v1"]) <= -min_improvement:
            selected_blocks.append(block)
    selected_blocks_path.write_text(",".join(selected_blocks), encoding="utf-8")

    print("\nAblation results:")
    print(results[["config", "OOF_RMSE", "OOF_MAE", "OOF_R2", "delta_vs_v1", "notes"]])
    print(f"\nSelected blocks: {selected_blocks if selected_blocks else '[none]'}")
    print(f"Wrote {ablation_results_path}")
    print(f"Wrote {selected_blocks_path}")
    return selected_blocks


def load_selected_blocks(blocks_arg: str) -> list[str]:
    if blocks_arg and blocks_arg != "auto":
        return [b.strip() for b in blocks_arg.split(",") if b.strip()]
    if SELECTED_BLOCKS_PATH.exists():
        text = SELECTED_BLOCKS_PATH.read_text(encoding="utf-8").strip()
        return [b.strip() for b in text.split(",") if b.strip()]
    return []


def run_ensemble(blocks_arg: str, sample_rows: int) -> None:
    selected_blocks = load_selected_blocks(blocks_arg)
    feature_config = combine_feature_blocks(selected_blocks)
    train, test = read_data(sample_rows=sample_rows)
    smoke = sample_rows > 0
    results_path = OUTPUT_DIR / "v2_ensemble_results_smoke.csv" if smoke else ENSEMBLE_RESULTS_PATH
    oof_path = OUTPUT_DIR / "v2_model_oof_predictions_smoke.csv" if smoke else OOF_PREDICTIONS_PATH
    test_predictions_path = (
        OUTPUT_DIR / "v2_ensemble_test_predictions_smoke.csv"
        if smoke
        else ENSEMBLE_TEST_PREDICTIONS_PATH
    )
    print(f"=== Ensemble ===")
    print(f"Train rows: {len(train):,}; Test rows: {len(test):,}")
    print(f"Feature blocks: {selected_blocks if selected_blocks else '[v1 base only]'}")
    if smoke:
        print("Smoke mode: writing _smoke artifacts and not appending experiments_log.md")

    cv_result = run_cv(train, feature_config, ["xgb", "lgbm", "catboost"], test=test)
    y = cv_result["y"]
    model_oof = cv_result["oof"]
    blends = compute_blends(y, model_oof)

    all_oof = {
        **model_oof,
        "simple_avg": blends["simple_avg"],
        "nnls_blend": blends["nnls_blend"],
    }

    result_rows = []
    log_rows = []
    for name, pred in all_oof.items():
        metrics = metric_dict(y, pred)
        result_rows.append(
            {
                "model": name,
                **metrics,
                "delta_vs_v1": metrics["OOF_RMSE"] - HISTORICAL_V1_RMSE,
                "feature_config": feature_config.name,
                "blocks": ",".join(selected_blocks),
            }
        )
        log_rows.append(
            {
                "step": "ensemble",
                "config": f"{feature_config.name}_{name}",
                **metrics,
                "delta_vs_v1": metrics["OOF_RMSE"] - HISTORICAL_V1_RMSE,
                "notes": (
                    f"blocks={selected_blocks if selected_blocks else '[v1 base only]'}; "
                    f"feature_count={len(cv_result['feature_columns'])}; "
                    f"cat_cols={len(cv_result['categorical_columns'])}"
                ),
            }
        )

    result_df = pd.DataFrame(result_rows).sort_values("OOF_RMSE")
    result_df.to_csv(results_path, index=False)
    if not smoke:
        append_experiment_rows(log_rows)

    oof_df = pd.DataFrame(
        {
            "row_id": np.arange(len(train)),
            TARGET_COL: y,
            **{f"{name}_oof": pred for name, pred in all_oof.items()},
        }
    )
    oof_df.to_csv(oof_path, index=False)

    test_blends = apply_test_blends(cv_result["test_preds"], blends["nnls_weights"])
    test_df = pd.DataFrame(
        {
            ID_COL: cv_result["ids"],
            **{f"{name}_pred": pred for name, pred in cv_result["test_preds"].items()},
            "simple_avg_pred": test_blends["simple_avg"],
            "nnls_blend_pred": test_blends["nnls_blend"],
        }
    )
    test_df.to_csv(test_predictions_path, index=False)

    print("\nEnsemble results:")
    print(result_df)
    print(f"\nNNLS weights: {blends['nnls_weights']}")
    print(f"Wrote {results_path}")
    print(f"Wrote {oof_path}")
    print(f"Wrote {test_predictions_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["contract", "ablate", "ensemble", "all"],
        default="contract",
        help="What to run.",
    )
    parser.add_argument(
        "--blocks",
        default="auto",
        help=(
            "Comma-separated feature blocks for ensemble, or 'auto' to read "
            "outputs/v2_selected_blocks.txt. Valid blocks: "
            + ",".join(BLOCK_CONFIGS)
        ),
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.01,
        help="Minimum RMSE improvement required to keep an ablation block.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=SAMPLE_ROWS,
        help="Use the first N train rows for smoke testing only. Full experiments should leave this at 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "contract":
        run_contract(args.sample_rows)
    elif args.mode == "ablate":
        run_ablation(args.min_improvement, args.sample_rows)
    elif args.mode == "ensemble":
        run_ensemble(args.blocks, args.sample_rows)
    elif args.mode == "all":
        selected_blocks = run_ablation(args.min_improvement, args.sample_rows)
        run_ensemble(",".join(selected_blocks), args.sample_rows)


if __name__ == "__main__":
    main()
