"""v3 reverse-engineer config.

Builds on v2's enriched parquets. Adds:
  - 35-class sub_grade aux model
  - Rate-card lookup table from loan.csv
  - Issue-year aux model
  - Within-grade residual aux model
"""
from pathlib import Path

from reverse_engineer.config import (
    ARCHIVE_LC_TRAIN_CSV, ARCHIVE_LOAN_CSV, OUTPUTS_DIR as REVERSE_OUT,
    RANDOM_STATE, SUBGRADES, SUBGRADE_TO_ORD,
)

V3_DIR = REVERSE_OUT / "v3"
V3_DIR.mkdir(parents=True, exist_ok=True)
V3_CKPT = V3_DIR / "checkpoints"
V3_CKPT.mkdir(parents=True, exist_ok=True)

# Source: v2's enriched parquets (already production-engineered + FICO×aux'd)
V2_ITER_DIR = REVERSE_OUT / "iter_1"
V2_DIR = REVERSE_OUT / "v2"
V2_TRAIN_PARQUET = V2_DIR / "X_train_enriched_v2.parquet"
V2_VAL_PARQUET = V2_DIR / "X_val_enriched_v2.parquet"
V2_TEST_PARQUET = V2_DIR / "X_test_enriched_v2.parquet"

# v3 output artifacts
RATE_LOOKUP_JSON = V3_DIR / "rate_lookup.json"
AUX_MODELS_PKL = V3_DIR / "aux_models_v3.pkl"
ENRICHED_V3_TRAIN = V3_DIR / "X_train_enriched_v3.parquet"
ENRICHED_V3_VAL = V3_DIR / "X_val_enriched_v3.parquet"
ENRICHED_V3_TEST = V3_DIR / "X_test_enriched_v3.parquet"
TUNING_V3_JSON = V3_DIR / "tuning_v3.json"

# Use the same 800k-row cap as v2 for loan.csv
LOAN_CSV_MAX_ROWS = 800_000

# CatBoost params for the new aux models
AUX_V3_CB_PARAMS = {
    "iterations": 2000,
    "depth": 8,
    "learning_rate": 0.04,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 0.5,
    "border_count": 254,
    "early_stopping_rounds": 100,
}
# The 35-class classifier is heavier — keep it slightly trimmer
AUX_V3_SG35_PARAMS = {
    **AUX_V3_CB_PARAMS,
    "iterations": 1500,
    "depth": 7,    # 7 levels keeps wall-clock under 4 hr on 800k rows
}

# Optuna trial counts for v3 retune
N_TRIALS_CB_V3 = 30      # CB is slow; we already have v2 params as warm start
N_TRIALS_LGB_V3 = 80
N_TRIALS_XGB_V3 = 80
HOLDOUT_FRAC = 0.20

# Skip the 35-class sub_grade CB classifier. After 20+ hours of training it
# never completed on 800k rows. The other 3 v3 levers (rate_lookup,
# issue_year, within_grade_residual) capture most of the same signal at a
# fraction of the compute cost: the existing v2 7-class grade probabilities
# combined with within_grade_residual already gives us "predicted rate from
# grade lookup" via aux_reconstructed_rate.
SKIP_SG35 = True
