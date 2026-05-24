"""Reverse-engineer pipeline config.

The goal: recover the `grade` / `sub_grade` / `int_rate` signal that the
LendingClub pricing model uses internally, by training auxiliary models on
the historical full-disclosure data in `achive_data/`, then enriching the
`true data/` rows with predicted versions of those columns.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# True-data (target dataset — what we actually have to predict)
TRUE_DATA_DIR = PROJECT_ROOT / "true data"

# Archive source A: historical 100k slice that DOES have grade/sub_grade
ARCHIVE_LC_TRAIN_CSV = PROJECT_ROOT / "achive_data" / "archive" / "LC_train.csv"

# Archive source B: the full 2.26M-row Kaggle dump
ARCHIVE_LOAN_CSV = PROJECT_ROOT / "achive_data" / "archive" / "loan.csv"

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "reverse_engineer"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 6604
HOLDOUT_FRAC = 0.20

# Iteration parameters
MAX_ITERATIONS = 5
CONVERGENCE_THRESHOLD = 0.005     # stop when val_RMSE delta < this

# Aux model parameters (default — CatBoost on common-features only)
AUX_CB_PARAMS = {
    "iterations": 2000,
    "depth": 8,
    "learning_rate": 0.04,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 0.5,
    "border_count": 254,
    "early_stopping_rounds": 100,
}
AUX_VAL_FRAC = 0.10            # inner val split for aux model early stopping

# Sub-grade ordinal encoding (35 levels)
SUBGRADES = [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)]
SUBGRADE_TO_ORD = {sg: i + 1 for i, sg in enumerate(SUBGRADES)}
ORD_TO_SUBGRADE = {v: k for k, v in SUBGRADE_TO_ORD.items()}

# Loan.csv is huge — cap rows to keep memory reasonable. Set to None for full.
LOAN_CSV_MAX_ROWS = 800_000     # ~600MB pandas frame; still 8x more than LC_train
