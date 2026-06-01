"""
Leakage-safe LendingClub RMSE experiment runner.

This file is intentionally separate from preprocessing_and_modeling.py while
the feature/model search is active. It appends live experiment results to
outputs/experiments_log.md and leaves the final production pipeline untouched
until the winner is known.

Current implemented step:
  - Step 0: reproduce v1 with historical tuned params and a fixed default-ish
    XGBoost baseline.
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
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=FutureWarning)


RANDOM_STATE = 42
TARGET_COL = "int_rate"
ID_COL = "ID"
N_FOLDS = 5
NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 75
TARGET_ENCODING_SMOOTHING = 50.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "true data"
TRAIN_PATH = DATA_DIR / "LC_train.csv"
TEST_PATH = DATA_DIR / "LC_test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXPERIMENT_LOG_PATH = OUTPUT_DIR / "experiments_log.md"

HISTORICAL_V1_RMSE = 3.90621
REPRO_TOLERANCE = 0.01

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

DEFAULTISH_XGB_PARAMS = {
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 10,
    "reg_alpha": 0,
    "gamma": 0,
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
    target_encoding_cols: tuple[str, ...] = ()

    @property
    def categorical_columns(self) -> list[str]:
        cols = list(BASE_CATEGORICAL_COLS)
        if self.add_fico_nonlinear:
            cols.append("fico_band")
        if self.add_categorical_crosses:
            cols.extend(["purpose_term", "purpose_fico_band", "zip_term", "state_term"])
        return cols


V1_FEATURES = FeatureConfig(name="v1_base")


@dataclass
class LendingClubExperimentPreprocessor:
    config: FeatureConfig
    numeric_medians: dict[str, float] = field(default_factory=dict)
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    target_mean: float = 0.0
    target_encoding_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    categorical_columns_: list[str] = field(default_factory=list)

    def fit(self, raw_df: pd.DataFrame) -> "LendingClubExperimentPreprocessor":
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

        if self.config.add_fico_nonlinear or self.config.add_fico_crosses or self.config.add_categorical_crosses:
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

        if self.config.add_categorical_crosses:
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

        drop_cols = [c for c in DROP_COLS if c in df.columns]
        return df.drop(columns=drop_cols)

    @staticmethod
    def _validate_feature_matrix(features: pd.DataFrame) -> None:
        forbidden = FORBIDDEN_FEATURES.intersection(features.columns)
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


def read_train() -> pd.DataFrame:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing train file: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_PATH}")

    train = pd.read_csv(TRAIN_PATH, na_values=["NA"])
    test_header = pd.read_csv(TEST_PATH, na_values=["NA"], nrows=0)
    if TARGET_COL not in train.columns:
        raise ValueError(f"{TRAIN_PATH} must contain {TARGET_COL!r}.")
    if TARGET_COL in test_header.columns:
        raise ValueError(f"{TEST_PATH} must not contain {TARGET_COL!r}.")
    if ID_COL not in test_header.columns:
        raise ValueError(f"{TEST_PATH} must contain {ID_COL!r}.")
    return train


def base_xgb_params(model_seed: int = RANDOM_STATE) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": model_seed,
        "nthread": max(os.cpu_count() or 1, 1),
    }


def make_dmatrix(X: pd.DataFrame, y: np.ndarray | None = None) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, enable_categorical=True)


def predict_with_best_iteration(booster: xgb.Booster, dmatrix: xgb.DMatrix) -> np.ndarray:
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is None:
        return booster.predict(dmatrix)
    return booster.predict(dmatrix, iteration_range=(0, int(best_iteration) + 1))


def train_xgb_fold(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
    model_seed: int = RANDOM_STATE,
) -> xgb.Booster:
    return xgb.train(
        params={**base_xgb_params(model_seed), **params},
        dtrain=make_dmatrix(X_fit, y_fit),
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(make_dmatrix(X_val, y_val), "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )


def run_xgb_cv(
    train: pd.DataFrame,
    feature_config: FeatureConfig,
    xgb_params: dict[str, Any],
    fold_seed: int = RANDOM_STATE,
    model_seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    y = train[TARGET_COL].to_numpy()
    oof_pred = np.full(len(train), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    feature_columns: list[str] | None = None

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=fold_seed)
    for fold_no, (fit_idx, val_idx) in enumerate(kfold.split(train), start=1):
        fold_t0 = time.perf_counter()
        raw_fit = train.iloc[fit_idx].reset_index(drop=True)
        raw_val = train.iloc[val_idx].reset_index(drop=True)
        y_fit = raw_fit[TARGET_COL].to_numpy()
        y_val = raw_val[TARGET_COL].to_numpy()

        preprocessor = LendingClubExperimentPreprocessor(feature_config)
        X_fit = preprocessor.fit_transform(raw_fit)
        X_val = preprocessor.transform(raw_val)

        if feature_columns is None:
            feature_columns = preprocessor.feature_columns
        elif feature_columns != preprocessor.feature_columns:
            raise ValueError("Feature columns changed across folds.")

        booster = train_xgb_fold(X_fit, y_fit, X_val, y_val, xgb_params, model_seed)
        val_pred = predict_with_best_iteration(booster, make_dmatrix(X_val))
        oof_pred[val_idx] = val_pred

        fold_metrics = {
            "fold": fold_no,
            "rmse": rmse(y_val, val_pred),
            "mae": mae(y_val, val_pred),
            "r2": r2_score_np(y_val, val_pred),
            "best_iteration": int(getattr(booster, "best_iteration", NUM_BOOST_ROUND - 1)),
            "seconds": time.perf_counter() - fold_t0,
        }
        fold_rows.append(fold_metrics)
        print(
            f"  Fold {fold_no}: RMSE={fold_metrics['rmse']:.5f} "
            f"MAE={fold_metrics['mae']:.5f} R2={fold_metrics['r2']:.5f} "
            f"best_iter={fold_metrics['best_iteration']}"
        )

    if np.isnan(oof_pred).any():
        raise RuntimeError("OOF predictions contain missing values.")
    if not np.isfinite(oof_pred).all():
        raise RuntimeError("OOF predictions contain non-finite values.")

    return {
        "oof_pred": oof_pred,
        "folds": fold_rows,
        "feature_count": len(feature_columns or []),
        "metrics": {
            "OOF_RMSE": rmse(y, oof_pred),
            "OOF_MAE": mae(y, oof_pred),
            "OOF_R2": r2_score_np(y, oof_pred),
        },
    }


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


def run_step0() -> None:
    print("=== Step 0: Reproduce v1 cleanly ===")
    train = read_train()
    print(f"Train: {train.shape}")

    runs = [
        ("0A", "v1_xgb_historical_tuned_params", HISTORICAL_V1_XGB_PARAMS),
        ("0B", "v1_xgb_defaultish_params", DEFAULTISH_XGB_PARAMS),
    ]
    results: list[dict[str, Any]] = []

    for step, config_name, params in runs:
        print(f"\n=== {step}: {config_name} ===")
        started = time.perf_counter()
        result = run_xgb_cv(train, V1_FEATURES, params, fold_seed=RANDOM_STATE)
        elapsed = time.perf_counter() - started
        metrics = result["metrics"]
        delta_vs_reference = metrics["OOF_RMSE"] - HISTORICAL_V1_RMSE
        pass_gate = abs(delta_vs_reference) <= REPRO_TOLERANCE
        notes = (
            f"feature_count={result['feature_count']}; "
            f"elapsed_sec={elapsed:.1f}; "
            f"repro_gate={'PASS' if pass_gate else 'FAIL'} vs {HISTORICAL_V1_RMSE:.5f}"
        )
        results.append(
            {
                "step": step,
                "config": config_name,
                "OOF_RMSE": metrics["OOF_RMSE"],
                "OOF_MAE": metrics["OOF_MAE"],
                "OOF_R2": metrics["OOF_R2"],
                "delta_vs_v1": delta_vs_reference,
                "notes": notes,
            }
        )
        print(
            f"{step} OOF: RMSE={metrics['OOF_RMSE']:.5f} "
            f"MAE={metrics['OOF_MAE']:.5f} R2={metrics['OOF_R2']:.5f} "
            f"delta_vs_v1={delta_vs_reference:+.5f} "
            f"gate={'PASS' if pass_gate else 'FAIL'}"
        )

    append_experiment_rows(results)
    any_passed = any(abs(row["delta_vs_v1"]) <= REPRO_TOLERANCE for row in results)
    if not any_passed:
        raise RuntimeError(
            "Step 0 failed: neither v1 run reproduced the historical RMSE within "
            f"{REPRO_TOLERANCE:.2f}. Stop before continuing."
        )
    print(f"\nStep 0 complete. Appended results to {EXPERIMENT_LOG_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        default="0",
        choices=["0"],
        help="Experiment step to run. Only Step 0 is implemented so far.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step == "0":
        run_step0()


if __name__ == "__main__":
    main()
