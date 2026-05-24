"""Production pipeline config + shared utilities."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "true data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "production"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
RANDOM_STATE = 6604
HOLDOUT_FRAC = 0.20

# K-fold OOF
N_FOLDS = 5
N_SEEDS_GBDT = 3       # 3-seed averaging for tree models (CB/LGB/XGB)
N_SEEDS_NN = 2         # 2-seed averaging for FT-Transformer / MLP

# Optuna tuning
N_TRIALS_CB = 60       # CatBoost is slow → fewer trials
N_TRIALS_LGB = 100
N_TRIALS_XGB = 100

# Target encoding smoothing
TARGET_ENC_M = 20.0
EMP_TITLE_TOP_N = 200

# Pseudo-labeling
PSEUDO_LABEL_QUANTILE = 0.70   # take top 70% confidence test rows as pseudo-labels

# Target clipping bounds (observed range in train)
INT_RATE_MIN = 6.0
INT_RATE_MAX = 31.0
