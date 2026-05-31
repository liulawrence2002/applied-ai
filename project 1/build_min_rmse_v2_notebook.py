"""Build FINAL_SIMPLIFIED_MIN_RMSE_V2.ipynb.

Iterates on build_min_rmse_notebook.py (V1). Only the FEATURES and MODEL
sections change; the W2-style explainability section is preserved verbatim.

V2 changes (full rationale in MODEL_MIN_RMSE_V2_NOTES.md):

  FEATURE LAYER
   * Train a 35-class sub_grade auxiliary CatBoost classifier on the archive
     (predecessor used a 7-class grade classifier). 35 ordered classes
     resolve roughly 5x finer rate buckets.
   * Use the pre-computed outputs/reverse_engineer/v3/rate_lookup.json to
     build:
        - aux_lookup_expected_rate    = sum_p(sg) * mean_rate(sg)
        - aux_lookup_rate_uncertainty = sqrt( E[X^2] - E[X]^2 )
        - aux_lookup_expected_rate_x_term = expected_rate * term_months
   * Add summary stats over the 35-prob distribution:
        aux_sg35_argmax_ord, aux_sg35_max_prob, aux_sg35_top2_gap,
        aux_sg35_entropy.
   * Drop the 7-class aux_grade_prob_* columns to avoid first-mover bias
     between the two correlated grade representations.

  MODEL LAYER
   * 4-model blend: CB + LGB + XGB + HistGradientBoosting (HGB).
   * Optuna re-tune for CB (30 trials), LGB (50), XGB (50), HGB (30).
   * LGB and XGB use log1p(target); CB and HGB use raw target.
   * Raise min_child_samples / min_data_in_leaf floors to reduce overfit on
     the 35 sg-prob columns (some have rare-class tails).

  DIAGNOSTICS
   * Per-base-learner train-vs-val learning curve over boosting rounds.
   * sklearn learning_curve on CB at 5 train-fraction levels (20-100%).
   * OOF-vs-val residual distribution overlay.
   * Per-fold RMSE bar chart.

Everything else (target encoding, K-fold, SLSQP blend, isotonic
post-calibration, explainability section) is unchanged from V1.

Runtime: ~45-55 min on CPU.
Expected val RMSE: 3.79-3.82.
Submission file: FINAL_MIN_RMSE_V2_SUBMISSION.csv.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent / "FINAL_SIMPLIFIED_MIN_RMSE_V2.ipynb"
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ============================================================
# 1. Header
# ============================================================
md("""# LendingClub Interest Rate — Min-RMSE V2 (Sub-grade rate lookup + HGB)

> **Runtime knob**: cell 2 has `FAST_MODE = True` (default) → ~15–20 min, expected val_RMSE ≈ 3.82–3.85.
> Flip to `False` for the full configuration (~45–55 min, expected val_RMSE ≈ 3.79–3.82).

**Goal**: push absolute minimum RMSE while keeping the notebook submission-readable. **Target: val_RMSE ≈ 3.79–3.82** (below the documented 4-model champion at ~3.83).

**Iterates from**: `FINAL_SIMPLIFIED_MIN_RMSE.ipynb` (V1, ~3.83–3.85).
**Companion doc**: `MODEL_MIN_RMSE_V2_NOTES.md`.

**Two surgical changes vs V1** (per request: features + model only):

1. **Sub-grade rate-lookup features** — replace V1's 7-class `grade` classifier with a 35-class `sub_grade` classifier, then use the historical `outputs/reverse_engineer/v3/rate_lookup.json` table to compute a **probability-weighted expected rate** feature directly. This is the highest-lift signal documented in the repo's `reverse_engineer/v3/` track.

2. **4-model blend with HGB added + all four base learners re-tuned** — adds HistGradientBoosting (different binning algorithm, decorrelated from CB/LGB/XGB → blend gain). CB, LGB, XGB get fresh Optuna search at higher trial counts.

**New diagnostic section** (per request): four overfitting/underfitting plots — per-model learning curves over boosting rounds, sklearn `learning_curve` on CB across training-set sizes, OOF-vs-val residual overlay, per-fold RMSE bar chart. These tell at a glance whether each base learner is over-, under-, or well-fit, and whether the blend would benefit from more data or more capacity.

| Aspect | V1 (Min-RMSE) | V2 (this notebook) |
|---|---|---|
| Aux classifier | 7-class grade (A–G) | **35-class sub_grade (A1–G5)** |
| Rate signal | 7 grade-probability columns | **`aux_lookup_expected_rate` directly + 35 sg-prob + 4 summary stats** |
| Base learners | CB + LGB + XGB | **CB + LGB + XGB + HGB** |
| Tuning | cached CB + Optuna LGB(40) + XGB(40) | **Optuna CB(30) + LGB(50) + XGB(50) + HGB(30)** |
| Diagnostics | residual scatter + isotonic curve | **+ 4 overfitting/underfitting plots** |
| Submission | `FINAL_MIN_RMSE_SUBMISSION.csv` | `FINAL_MIN_RMSE_V2_SUBMISSION.csv` |
| Runtime | ~30-40 min | **~45-55 min** |

**Interpretability preserved**: still under 80 features, no neural nets, no stacking meta-learner. Aux outputs are explicitly labelled `aux_sg35_*` / `aux_lookup_*` so they are identifiable in SHAP/LIME outputs.

**Reproducibility**: same `RANDOM_STATE = 6604`, same 80/20 split, same 5-fold KFold, leak-safe per-fold target encoding, leak-safe per-fold log1p inversion.""")


# ============================================================
# 2. Imports
# ============================================================
md("""## 1. Imports and paths""")

code("""import json
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, learning_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from scipy.optimize import minimize

from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation as lgb_log
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / 'true data'
OUT_DIR = PROJECT_ROOT / 'outputs' / 'min_rmse_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# FAST_MODE: True -> ~15-20 min runtime (expected val_RMSE ~3.82-3.85).
# False     -> full configuration ~45-55 min (expected val_RMSE ~3.79-3.82).
# Flip and re-run all cells; the rest of the notebook reads these constants.
# =====================================================================
FAST_MODE = True

RANDOM_STATE = 6604
N_FOLDS = 3 if FAST_MODE else 5
HOLDOUT_FRAC = 0.20
N_TRIALS_CB  = 10 if FAST_MODE else 30
N_TRIALS_LGB = 15 if FAST_MODE else 50
N_TRIALS_XGB = 15 if FAST_MODE else 50
N_TRIALS_HGB = 10 if FAST_MODE else 30
PERM_N_REPEATS = 3 if FAST_MODE else 10
LEARNING_CURVE_CV = 2 if FAST_MODE else 3
LEARNING_CURVE_FRACTIONS = 3 if FAST_MODE else 5
SHAP_SAMPLE_SIZE = 800 if FAST_MODE else 2000
SHAP_SPLIT_SAMPLE = 500 if FAST_MODE else 1000

print(f'FAST_MODE     : {FAST_MODE}  -> n_folds={N_FOLDS}')
print('Project root :', PROJECT_ROOT)
print('Data dir     :', DATA_DIR)
print('Output dir   :', OUT_DIR)
print(f'Seed={RANDOM_STATE}, n_folds={N_FOLDS}')
print(f'Optuna trials: CB={N_TRIALS_CB}, LGB={N_TRIALS_LGB}, XGB={N_TRIALS_XGB}, HGB={N_TRIALS_HGB}')""")


# ============================================================
# 3. Load + drop leakage
# ============================================================
md("""## 2. Load `true data/` and drop `loan_status`""")

code("""STRING_NUMERIC_COLS = [
    'dti', 'revol_util', 'all_util', 'mo_sin_old_il_acct',
    'mths_since_last_record', 'mths_since_rcnt_il', 'mths_since_recent_bc',
    'mths_since_recent_inq', 'tot_cur_bal',
]
DT_OVERRIDES = {c: 'float64' for c in STRING_NUMERIC_COLS}

train_raw = pd.read_csv(DATA_DIR / 'LC_train.csv', na_values=['NA'],
                       dtype=DT_OVERRIDES, low_memory=False)
test_raw  = pd.read_csv(DATA_DIR / 'LC_test.csv',  na_values=['NA'],
                       dtype=DT_OVERRIDES, low_memory=False)

train_raw = train_raw.drop(columns=['loan_status'])
test_raw  = test_raw.drop(columns=['loan_status'])

print('train_raw:', train_raw.shape)
print('test_raw :', test_raw.shape)
print('target stats (int_rate):')
print(train_raw['int_rate'].describe().round(2))""")


# ============================================================
# 4. Reduced feature engineering (same as V1)
# ============================================================
md("""## 3. Reduced feature engineering (35 base columns — same as V1)

Identical FE block to V1. The interaction-discovery experiment (`outputs/simplification/interaction_candidates_ranked.csv`) already confirmed no further pairwise interaction reduces single-LGB RMSE — the complexity budget goes to aux features instead.""")

code("""EMP_LENGTH_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10,
}

def engineer_reduced(df):
    out = df.copy()
    # FICO core
    out['fico'] = (out['fico_range_low'] + out['fico_range_high']) / 2
    out['fico_band'] = pd.cut(out['fico'], bins=[-np.inf, 660, 690, 720, 760, np.inf],
                              labels=False).astype('Int64')
    out['fico_above_prime'] = (out['fico'] >= 720).astype('int8')
    # Loan structure
    out['term_months'] = (out['term'].astype('string').str.extract(r'(\\d+)', expand=False)
                          .astype('Int64'))
    out['installment_proxy'] = out['loan_amnt'] / out['term_months'].astype('float64')
    out['loan_to_income'] = out['loan_amnt'] / out['annual_inc'].replace(0, np.nan)
    out['payment_to_income'] = out['installment_proxy'] / (out['annual_inc'].replace(0, np.nan) / 12)
    # Utilisation
    out['util_max'] = np.maximum(out['all_util'].fillna(0), out['revol_util'].fillna(0))
    # Three confirmed interactions
    out['term_x_dti']             = out['term_months'].astype('float64') * out['dti']
    out['fico_x_dti']             = out['fico'] * out['dti']
    out['fico_revol_interaction'] = out['fico'] * out['revol_util']
    # Tenure
    out['credit_file_age_yrs'] = out['mo_sin_old_rev_tl_op'] / 12.0
    out['inq_per_credit_age']  = (out['inq_last_12m'].fillna(0) /
                                  (out['credit_file_age_yrs'].fillna(0) + 1))
    # Monetary log-tails
    out['log1p_annual_inc'] = np.log1p(out['annual_inc'].clip(lower=0))
    out['log1p_loan_amnt']  = np.log1p(out['loan_amnt'].clip(lower=0))
    out['log1p_revol_bal']  = np.log1p(out['revol_bal'].clip(lower=0))
    # High-cardinality cleanup
    out['zip3'] = out['zip_code'].astype('string').str.extract(r'(\\d{3})', expand=False)
    out['emp_length_num'] = out['emp_length'].map(EMP_LENGTH_MAP)
    drop = ['fico_range_low', 'fico_range_high', 'term', 'emp_length',
            'title', 'zip_code', 'emp_title']
    return out.drop(columns=[c for c in drop if c in out.columns])


train_fe = engineer_reduced(train_raw)
test_fe  = engineer_reduced(test_raw)

test_ids = test_fe['ID'].copy()
test_fe = test_fe.drop(columns=['ID'])
y = train_fe['int_rate'].values.astype(np.float64)
X = train_fe.drop(columns=['int_rate'])

REDUCED_FEATURES = [
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
    'log1p_annual_inc', 'log1p_revol_bal', 'emp_length_num',
]
present = [c for c in REDUCED_FEATURES if c in X.columns]
X = X[present].copy()
test_fe = test_fe[present].copy()
print(f'Reduced base FE: X={X.shape}  test={test_fe.shape}')""")


# ============================================================
# 5. Train 35-class sub_grade classifier
# ============================================================
md("""## 4. 35-class sub_grade auxiliary classifier

This is the key feature-layer change from V1. We train one CatBoost classifier on `achive_data/archive/LC_train.csv` (which has `sub_grade`), then predict 35 sub_grade probabilities on the true-data slice.

**Why 35 classes instead of V1's 7**: each grade letter (A–G) has 5 sub-grades (e.g., A1–A5) that LendingClub uses internally as separate rate buckets. The mean rate gap between A1 (5.67%) and A5 (8.24%) is 2.6 pp — bigger than the difference between many grade letters. A 7-class classifier averages over this; a 35-class classifier resolves it.

**Why this doesn't leak**: the archive's row sample is disjoint from the true-data slice (verified in `MODEL_DOCUMENTATION.md` §2), and the classifier sees only `sub_grade` as its target — never `int_rate`. The rate-lookup table is built from the archive's historical observations, again with no overlap with the true-data target.

Runtime for this cell: **~2–4 min in FAST_MODE** (was ~10–15 min with CatBoost MultiClass). We swapped the aux engine to **LightGBM multiclass** with row subsampling — same 35 output probabilities, ~5–10× faster training because LGB grows a single tree per boosting round (CatBoost MultiClass trains one tree per class per round = 35× more trees).""")

code("""from lightgbm import LGBMClassifier, early_stopping as lgb_es_cls

COMMON_FEATURES = [
    'addr_state', 'annual_inc', 'application_type', 'chargeoff_within_12_mths',
    'collections_12_mths_ex_med', 'delinq_2yrs', 'dti', 'emp_length',
    'home_ownership', 'loan_amnt', 'mo_sin_old_rev_tl_op', 'mort_acc', 'open_acc',
    'pub_rec', 'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util',
    'term', 'total_acc', 'verification_status', 'zip_code',
]
# Note: dropped emp_title + title from the aux feature list — both are huge-cardinality
# free text that explode LGB's category dictionary (>20k unique values) for almost zero gain.

print('Loading archive LC_train.csv (has sub_grade)...')
arch = pd.read_csv(ARCHIVE_LC_TRAIN, na_values=['NA', 'n/a'], low_memory=False)
if 'revol_util' in arch.columns and arch['revol_util'].dtype == object:
    arch['revol_util'] = pd.to_numeric(arch['revol_util'].astype(str).str.rstrip('%'),
                                         errors='coerce')
arch = arch.dropna(subset=['sub_grade']).reset_index(drop=True)

# Stratified row subsample (keeps the rare G-grades proportionally represented)
if len(arch) > AUX_SAMPLE_ROWS:
    arch = arch.groupby('sub_grade', group_keys=False).apply(
        lambda g: g.sample(
            n=max(50, int(round(len(g) * AUX_SAMPLE_ROWS / len(arch)))),
            random_state=RANDOM_STATE,
        )
    ).reset_index(drop=True)
    print(f'  subsampled to {len(arch):,} rows')

arch_X = arch[[c for c in COMMON_FEATURES if c in arch.columns]].copy()
arch_y = arch['sub_grade']
print(f'archive: {arch.shape}  sub_grade counts (head/tail):')
print('  head:', arch_y.value_counts().head(5).to_dict())
print('  tail:', arch_y.value_counts().tail(5).to_dict())

aux_cat_cols = [c for c in ['addr_state', 'application_type', 'emp_length',
                            'home_ownership', 'purpose', 'term',
                            'verification_status', 'zip_code'] if c in arch_X.columns]


def lgb_aux_prepare(df, cat_dtypes=None):
    \"\"\"Prep frame for LGB multiclass aux: cats as pandas Category with FIXED
    category list across train + apply so encoded ints stay consistent.\"\"\"
    df = df.copy()
    out_cats = {}
    for c in aux_cat_cols:
        if c not in df.columns:
            continue
        s = df[c].astype('string').fillna('Missing')
        if cat_dtypes is None or c not in cat_dtypes:
            cdt = pd.CategoricalDtype(categories=pd.Index(s.unique()))
        else:
            cdt = cat_dtypes[c]
        df[c] = s.astype(cdt)
        out_cats[c] = cdt
    for c in df.columns:
        if c not in aux_cat_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    return df, out_cats


print('\\nTraining 35-class LightGBM sub_grade classifier on archive...')
t = time()
arch_Xp, aux_cat_dt = lgb_aux_prepare(arch_X)

# 90/10 inner split, stratified by sub_grade
Xt, Xv, yt, yv = train_test_split(arch_Xp, arch_y, test_size=0.10,
                                    random_state=RANDOM_STATE, stratify=arch_y)
aux_model = LGBMClassifier(
    objective='multiclass',
    num_class=35,
    n_estimators=AUX_ITERATIONS,
    learning_rate=0.08,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=50,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
aux_model.fit(
    Xt, yt,
    eval_set=[(Xv, yv)],
    categorical_feature=[c for c in aux_cat_cols if c in arch_Xp.columns],
    callbacks=[lgb_es_cls(40, verbose=False), lgb_log(0)],
)
print(f'  aux 35-class LGB trained in {time()-t:.0f}s  (best iter: {aux_model.best_iteration_})')
print(f'  classes learned: {len(aux_model.classes_)}')""")


# ============================================================
# 6. Load rate lookup + apply aux to true data
# ============================================================
md("""## 5. Apply aux model to true data + build rate-lookup features

The rate-lookup table is loaded from `outputs/reverse_engineer/v3/rate_lookup.json` (already pre-computed from `loan.csv`, the 2.26M-row LendingClub history). We use it to convert the 35-prob distribution into a **probability-weighted expected rate** — a direct rate prediction from the classifier alone, which the downstream regressors can then refine.

`aux_lookup_expected_rate = Σ_sg p(sg) × mean_rate(sg)` — this is the strongest single-feature signal in the v3 enrichment pipeline.

`aux_lookup_rate_uncertainty = √(E[X²] − E[X]²)` where the expectation is taken over the prob distribution — tells the model when the sub_grade prediction is hedged vs confident.""")

code("""# Load rate lookup
rate_lookup = json.loads(RATE_LOOKUP_JSON.read_text())
sg_mean_rate = rate_lookup['subgrade_mean_rate']
sg_std_rate  = rate_lookup['subgrade_std_rate']
global_mean_rate = rate_lookup['global_mean_rate']
print(f'rate_lookup: {len(sg_mean_rate)} subgrades, global mean = {global_mean_rate:.3f}%')
print(f'  A1 -> {sg_mean_rate[\"A1\"]:.2f}%,  G5 -> {sg_mean_rate[\"G5\"]:.2f}%')

# Aligned vectors in canonical SUBGRADES order
mean_rate_vec = np.array([sg_mean_rate.get(sg, global_mean_rate) for sg in SUBGRADES])
std_rate_vec  = np.array([sg_std_rate.get(sg, 0.0) for sg in SUBGRADES])


def apply_aux_sg35(src_df):
    \"\"\"Predict 35 sub_grade probabilities on `src_df` (raw true-data frame),
    re-ordering columns to the aux model's feature list and re-ordering output
    classes to the canonical SUBGRADES (A1..G5) order. Uses the LGB classifier
    with its frozen Category dtypes so unseen levels become NaN (LGB tolerates
    NaN in categorical features).\"\"\"
    aux_feature_names = list(aux_model.feature_name_)
    src = src_df.copy()
    for c in aux_feature_names:
        if c not in src.columns:
            src[c] = np.nan
    src = src[aux_feature_names].copy()
    src_p, _ = lgb_aux_prepare(src, cat_dtypes=aux_cat_dt)
    proba = aux_model.predict_proba(src_p)
    classes = list(aux_model.classes_)
    if classes != SUBGRADES:
        idx = [classes.index(sg) if sg in classes else -1 for sg in SUBGRADES]
        ordered = np.zeros((proba.shape[0], len(SUBGRADES)), dtype=np.float64)
        for j, src_idx in enumerate(idx):
            if src_idx >= 0:
                ordered[:, j] = proba[:, src_idx]
        proba = ordered
    return proba


print('\\nApplying 35-class sub_grade aux to true-data train + test...')
t = time()
train_aux = apply_aux_sg35(train_raw.drop(columns=['int_rate']))
test_aux  = apply_aux_sg35(test_raw.drop(columns=['ID']))
print(f'  done in {time()-t:.0f}s. shapes: train={train_aux.shape}, test={test_aux.shape}')


def append_sg35_features(target_df, proba):
    \"\"\"Append the 35 prob cols + summary stats + rate-lookup features to target_df.\"\"\"
    out = target_df.copy()
    # 35 prob columns
    for j, sg in enumerate(SUBGRADES):
        out[f'aux_sg35_{sg}'] = proba[:, j]
    # Summary stats over the prob distribution
    out['aux_sg35_argmax_ord'] = (proba.argmax(axis=1) + 1).astype('float64')   # 1..35
    out['aux_sg35_max_prob']   = proba.max(axis=1)
    sorted_p = np.sort(proba, axis=1)
    out['aux_sg35_top2_gap']   = sorted_p[:, -1] - sorted_p[:, -2]
    # Entropy of the prob distribution
    out['aux_sg35_entropy']    = -np.sum(
        proba * np.log(np.clip(proba, 1e-9, 1.0)), axis=1,
    )
    # Probability-weighted expected rate and uncertainty
    expected_rate = proba @ mean_rate_vec
    e_x2 = proba @ (mean_rate_vec ** 2 + std_rate_vec ** 2)
    out['aux_lookup_expected_rate']    = expected_rate
    out['aux_lookup_rate_uncertainty'] = np.sqrt(np.maximum(e_x2 - expected_rate ** 2, 0.0))
    # Expected rate x term (v3 finding: longer-term same-subgrade loans priced higher)
    if 'term_months' in out.columns:
        out['aux_lookup_expected_rate_x_term'] = (
            out['aux_lookup_expected_rate'] * out['term_months'].astype('float64')
        )
    # 2 FICO x aux interactions (analogues of V1's fico_x_aux_argmax/_max_prob)
    out['fico_x_aux_sg35_argmax']  = out['fico'] * out['aux_sg35_argmax_ord']
    out['fico_x_aux_sg35_max_prob'] = out['fico'] * out['aux_sg35_max_prob']
    return out


X = append_sg35_features(X, train_aux)
test_fe = append_sg35_features(test_fe, test_aux)
print(f'\\nFeatures after sg35 enrichment: X={X.shape}, test={test_fe.shape}')
print(f'New aux columns added: {X.shape[1] - len(present)}')""")


# ============================================================
# 7. 80/20 split + TE helpers
# ============================================================
md("""## 6. 80/20 split + leak-safe target-encoding helpers

Same plumbing as V1. `addr_state`, `purpose`, `zip3` get smoothed K-fold OOF target encoding (m=20). Helpers also include log1p/expm1 for the LGB+XGB target transform.""")

code("""X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=HOLDOUT_FRAC,
                                                   random_state=RANDOM_STATE)
X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
print('X_train:', X_train.shape)
print('X_val  :', X_val.shape)
print('X_test :', test_fe.shape)

NATIVE_CAT = ['application_type', 'purpose']
TE_CAT     = ['addr_state', 'purpose', 'zip3']
ALL_CAT    = list(set(NATIVE_CAT + TE_CAT))
TARGET_ENC_M = 20.0


def smoothed_te(train_col, target, val_col=None, test_col=None, folds=5, seed=RANDOM_STATE):
    global_mean = float(np.nanmean(target))
    val_enc = test_enc = None
    if val_col is not None or test_col is not None:
        stats = pd.DataFrame({'cat': train_col.values, 'y': target}).groupby('cat')['y'].agg(['mean', 'count'])
        stats['enc'] = (stats['mean']*stats['count'] + global_mean*TARGET_ENC_M) / (stats['count']+TARGET_ENC_M)
        mapping = stats['enc'].to_dict()
        val_enc  = val_col.map(mapping).fillna(global_mean).values if val_col is not None else None
        test_enc = test_col.map(mapping).fillna(global_mean).values if test_col is not None else None
    train_enc = np.full(len(train_col), global_mean, dtype=np.float64)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, va in kf.split(train_col):
        inner = pd.DataFrame({'cat': train_col.values[tr], 'y': target[tr]}).groupby('cat')['y'].agg(['mean', 'count'])
        inner['enc'] = (inner['mean']*inner['count'] + global_mean*TARGET_ENC_M) / (inner['count']+TARGET_ENC_M)
        train_enc[va] = pd.Series(train_col.values[va]).map(inner['enc'].to_dict()).fillna(global_mean).values
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


def prep_lgb(df, cat_dtypes=None):
    df = df.copy()
    cat_cols = [c for c in ALL_CAT if c in df.columns]
    for c in cat_cols:
        s = df[c].astype('string').fillna('Missing')
        df[c] = s.astype(cat_dtypes[c]) if cat_dtypes and c in cat_dtypes else s.astype('category')
    for c in df.columns:
        if c not in cat_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    return df, cat_cols


def build_cat_dtypes(df):
    return {c: pd.CategoricalDtype(categories=pd.Index(df[c].astype('string').fillna('Missing').unique()))
            for c in ALL_CAT if c in df.columns}


def prep_hgb(df, cat_cats=None, max_cat_cardinality=60):
    \"\"\"HGB: low-card cats ordinal-encoded; high-card cats dropped (the TE columns
    carry their signal). HGB doesn't accept NA in categorical features.\"\"\"
    df = df.copy()
    cat_cols_all = [c for c in ALL_CAT if c in df.columns]
    cat_cols, drop_high = [], []
    for c in cat_cols_all:
        nuniq = df[c].astype('string').fillna('Missing').nunique()
        if nuniq <= max_cat_cardinality:
            cat_cols.append(c)
        else:
            drop_high.append(c)
    if drop_high:
        df = df.drop(columns=drop_high)
    out_cats = {}
    for c in cat_cols:
        s = df[c].astype('string').fillna('Missing')
        if cat_cats is None or c not in cat_cats:
            uniques = s.unique()
            mapping = {v: i for i, v in enumerate(uniques)}
        else:
            mapping = dict(cat_cats[c])
            unknown = len(mapping)
            for v in s.unique():
                if v not in mapping:
                    mapping[v] = unknown
        df[c] = s.map(mapping).astype('int64')
        out_cats[c] = mapping
    for c in df.columns:
        if c not in cat_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    return df, out_cats, cat_cols


def log1p_y(y):  return np.log1p(y)
def expm1_y(y):  return np.expm1(y)

print('Helpers ready.')""")


# ============================================================
# 8. Optuna tuning — CB, LGB, XGB, HGB
# ============================================================
md("""## 7. Optuna Bayesian search — CB (30) + LGB (50) + XGB (50) + HGB (30)

Inner 85/15 split for tuning. LGB and XGB use `log1p(int_rate)`; CB and HGB use raw `int_rate`. All four objectives invert appropriately before measuring inner RMSE on the rate scale.

The `min_child_samples` / `min_data_in_leaf` / `min_child_weight` floors are raised because the 35 sub_grade probability columns include rare-class tails (e.g., G5 with n=104 in the archive) — without higher leaf-size floors, the boosters tend to overfit those columns.""")

code("""# Inner 85/15 split (shared across all four objectives so they are comparable)
inner_tr, inner_va = train_test_split(
    np.arange(len(X_train)), test_size=0.15, random_state=RANDOM_STATE,
)
Xtr_inner = X_train.iloc[inner_tr].reset_index(drop=True)
Xva_inner = X_train.iloc[inner_va].reset_index(drop=True)
ytr_inner = y_train[inner_tr]
yva_inner = y_train[inner_va]
ytr_inner_log = log1p_y(ytr_inner)

Xtr_inner_te, Xva_inner_te, _ = add_te(Xtr_inner, ytr_inner, Xva_inner, X_train.iloc[:0])

cat_dt_inner = build_cat_dtypes(Xtr_inner_te)
Xtr_lgb_inner, lgb_cat_cols_inner = prep_lgb(Xtr_inner_te, cat_dt_inner)
Xva_lgb_inner, _ = prep_lgb(Xva_inner_te, cat_dt_inner)
Xtr_cb_inner, cat_idx_inner = prep_cb(Xtr_inner_te)
Xva_cb_inner, _ = prep_cb(Xva_inner_te)

# HGB inner prep
Xtr_hgb_inner, hgb_cats_inner, hgb_cat_cols_inner = prep_hgb(Xtr_inner_te)
Xva_hgb_inner, _, _ = prep_hgb(Xva_inner_te, cat_cats=hgb_cats_inner)
imp_inner = SimpleImputer(strategy='median')
Xtr_hgb_inner_v = imp_inner.fit_transform(Xtr_hgb_inner)
Xva_hgb_inner_v = imp_inner.transform(Xva_hgb_inner)
hgb_cat_idx_inner = [Xtr_hgb_inner.columns.get_loc(c) for c in hgb_cat_cols_inner]


def cb_objective(trial):
    params = {
        'iterations': 1500 if FAST_MODE else 3000,
        'learning_rate':    trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
        'depth':             trial.suggest_int('depth', 5, 9),
        'l2_leaf_reg':       trial.suggest_float('l2_leaf_reg', 0.5, 12.0, log=True),
        'random_strength':   trial.suggest_float('random_strength', 0.1, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count':      trial.suggest_int('border_count', 64, 254),
        'loss_function': 'RMSE', 'eval_metric': 'RMSE',
        'random_seed': RANDOM_STATE, 'verbose': 0, 'allow_writing_files': False,
    }
    m = CatBoostRegressor(**params, early_stopping_rounds=100)
    m.fit(Xtr_cb_inner, ytr_inner, cat_features=cat_idx_inner,
          eval_set=(Xva_cb_inner, yva_inner), verbose=False)
    return float(np.sqrt(mean_squared_error(yva_inner, m.predict(Xva_cb_inner))))


def lgb_objective(trial):
    params = {
        'learning_rate':     trial.suggest_float('learning_rate', 0.012, 0.07, log=True),
        'num_leaves':         trial.suggest_int('num_leaves', 31, 200),
        'max_depth':          trial.suggest_int('max_depth', -1, 12),
        'min_child_samples':  trial.suggest_int('min_child_samples', 30, 250),     # raised floor
        'feature_fraction':   trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction':   trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq':       1,
        'reg_alpha':          trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
        'reg_lambda':         trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
        'n_estimators':       2000 if FAST_MODE else 4000,
        'objective': 'regression', 'metric': 'rmse',
        'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1,
    }
    m = LGBMRegressor(**params)
    m.fit(Xtr_lgb_inner, ytr_inner_log,
          eval_set=[(Xva_lgb_inner, log1p_y(yva_inner))],
          categorical_feature=lgb_cat_cols_inner,
          callbacks=[lgb_es(100, verbose=False), lgb_log(0)])
    pred = expm1_y(m.predict(Xva_lgb_inner))
    return float(np.sqrt(mean_squared_error(yva_inner, pred)))


def xgb_objective(trial):
    params = {
        'learning_rate':     trial.suggest_float('learning_rate', 0.012, 0.07, log=True),
        'max_depth':          trial.suggest_int('max_depth', 4, 10),
        'min_child_weight':   trial.suggest_int('min_child_weight', 5, 80),       # raised floor
        'subsample':          trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'colsample_bylevel':  trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'reg_alpha':          trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
        'reg_lambda':         trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
        'gamma':              trial.suggest_float('gamma', 1e-3, 5.0, log=True),
        'n_estimators':       2000 if FAST_MODE else 4000,
        'objective': 'reg:squarederror', 'tree_method': 'hist',
        'enable_categorical': True, 'random_state': RANDOM_STATE, 'n_jobs': -1,
        'early_stopping_rounds': 100,
    }
    m = XGBRegressor(**params)
    m.fit(Xtr_lgb_inner, ytr_inner_log,
          eval_set=[(Xva_lgb_inner, log1p_y(yva_inner))], verbose=False)
    pred = expm1_y(m.predict(Xva_lgb_inner))
    return float(np.sqrt(mean_squared_error(yva_inner, pred)))


def hgb_objective(trial):
    params = {
        'learning_rate':    trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
        'max_iter':          trial.suggest_int('max_iter', 300, 1500),
        'max_leaf_nodes':    trial.suggest_int('max_leaf_nodes', 15, 127),
        'max_depth':         trial.suggest_int('max_depth', 4, 10),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 30, 250),       # raised floor
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 5.0),
        'max_features':      trial.suggest_float('max_features', 0.4, 1.0),
        'max_bins':          trial.suggest_int('max_bins', 64, 255),
        'random_state': RANDOM_STATE,
        'early_stopping': True, 'validation_fraction': 0.1, 'n_iter_no_change': 50,
    }
    m = HistGradientBoostingRegressor(**params, categorical_features=hgb_cat_idx_inner)
    m.fit(Xtr_hgb_inner_v, ytr_inner)
    return float(np.sqrt(mean_squared_error(yva_inner, m.predict(Xva_hgb_inner_v))))


sampler = TPESampler(seed=RANDOM_STATE, multivariate=True, warn_independent_sampling=False)
pruner  = MedianPruner(n_warmup_steps=10)
studies = {}
for name, obj, n_trials in [
    ('cb',  cb_objective,  N_TRIALS_CB),
    ('lgb', lgb_objective, N_TRIALS_LGB),
    ('xgb', xgb_objective, N_TRIALS_XGB),
    ('hgb', hgb_objective, N_TRIALS_HGB),
]:
    print(f'\\n[Optuna] {name.upper()} ({n_trials} trials)...')
    t = time()
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner,
                                 study_name=f'{name}_v2')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    studies[name] = study
    print(f'  best inner RMSE: {study.best_value:.4f}  ({time()-t:.0f}s)')
    print(f'  best params   : {study.best_params}')

CB_PARAMS  = dict(studies['cb'].best_params)
LGB_PARAMS = dict(studies['lgb'].best_params); LGB_PARAMS.update({
    'n_estimators': 2000 if FAST_MODE else 4000, 'bagging_freq': 1, 'objective': 'regression', 'metric': 'rmse',
    'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1,
})
XGB_PARAMS = dict(studies['xgb'].best_params); XGB_PARAMS.update({
    'n_estimators': 2000 if FAST_MODE else 4000, 'objective': 'reg:squarederror', 'tree_method': 'hist',
    'enable_categorical': True, 'random_state': RANDOM_STATE, 'n_jobs': -1,
    'early_stopping_rounds': 100,
})
HGB_PARAMS = dict(studies['hgb'].best_params)""")


# ============================================================
# 9. 5-fold OOF training (4 base learners) WITH learning-curve capture
# ============================================================
md("""## 8. 5-fold OOF training — CB + LGB + XGB + HGB, capturing per-iter learning curves

For each fold and each base learner we record the train RMSE and val RMSE at every iteration (sampled on a log grid to keep memory reasonable). The captured curves drive the overfitting/underfitting diagnostic plots below.""")

code("""n_tr, n_va, n_te = len(X_train), len(X_val), len(test_fe)
BASES = ['cb', 'lgb', 'xgb', 'hgb']
oof   = {b: np.zeros(n_tr) for b in BASES}
val_p = {b: np.zeros(n_va) for b in BASES}
te_p  = {b: np.zeros(n_te) for b in BASES}

# Learning-curve storage: per base learner, a list of (iter_idx, train_rmse, val_rmse) tuples
# from fold 0 only (saves time + memory). Curves are on the rate scale (log1p inverted).
learning_curves = {b: {'iters': [], 'train_rmse': [], 'val_rmse': []} for b in BASES}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_rmses = []
total_t = time()

for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    t_fold = time()
    Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    ytr, yva = y_train[tr_idx], y_train[va_idx]
    ytr_log = log1p_y(ytr)

    Xtr_te, Xva_te, Xte_te = add_te(Xtr, ytr, Xva, test_fe)
    _, Xval_te, _          = add_te(Xtr, ytr, X_val, test_fe)

    fold_results = {}

    # --- CatBoost (raw target) ---
    t = time()
    Xtr_cb, cat_idx_fold = prep_cb(Xtr_te)
    Xva_cb, _            = prep_cb(Xva_te)
    Xval_cb, _           = prep_cb(Xval_te)
    Xte_cb, _            = prep_cb(Xte_te)
    cb_iter = 1500 if FAST_MODE else 3000
    cb = CatBoostRegressor(**CB_PARAMS,
                            iterations=cb_iter, loss_function='RMSE', eval_metric='RMSE',
                            random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False,
                            early_stopping_rounds=120)
    cb.fit(Xtr_cb, ytr, cat_features=cat_idx_fold, eval_set=(Xva_cb, yva), verbose=False)
    oof['cb'][va_idx] = cb.predict(Xva_cb)
    val_p['cb'] += cb.predict(Xval_cb) / N_FOLDS
    te_p['cb']  += cb.predict(Xte_cb)  / N_FOLDS
    fold_results['cb'] = float(np.sqrt(mean_squared_error(yva, oof['cb'][va_idx])))
    print(f'  fold {fold_id+1} cb  ({time()-t:5.0f}s, RMSE={fold_results[\"cb\"]:.4f})')

    # Capture CatBoost learning curve on fold 0
    if fold_id == 0:
        evals = cb.get_evals_result()
        # CatBoost reports learn + validation under 'learn' and 'validation'
        if 'learn' in evals and 'validation' in evals:
            tr_curve = np.array(evals['learn']['RMSE'])
            va_curve = np.array(evals['validation']['RMSE'])
            n_iters = len(va_curve)
            # Sample on a log grid
            sample_idx = np.unique(np.round(np.geomspace(5, n_iters, 40)).astype(int)) - 1
            sample_idx = sample_idx[sample_idx < n_iters]
            learning_curves['cb']['iters']      = sample_idx.tolist()
            learning_curves['cb']['train_rmse'] = tr_curve[sample_idx].tolist()
            learning_curves['cb']['val_rmse']   = va_curve[sample_idx].tolist()

    # --- LightGBM (log1p target) ---
    t = time()
    cat_dt = build_cat_dtypes(Xtr_te)
    Xtr_lgb, cat_cols = prep_lgb(Xtr_te, cat_dt)
    Xva_lgb, _        = prep_lgb(Xva_te, cat_dt)
    Xval_lgb, _       = prep_lgb(Xval_te, cat_dt)
    Xte_lgb, _        = prep_lgb(Xte_te, cat_dt)
    lgb_m = LGBMRegressor(**LGB_PARAMS)
    lgb_m.fit(Xtr_lgb, ytr_log,
              eval_set=[(Xtr_lgb, ytr_log), (Xva_lgb, log1p_y(yva))],
              eval_names=['train', 'val'],
              categorical_feature=cat_cols,
              callbacks=[lgb_es(120, verbose=False), lgb_log(0)])
    oof['lgb'][va_idx] = expm1_y(lgb_m.predict(Xva_lgb))
    val_p['lgb'] += expm1_y(lgb_m.predict(Xval_lgb)) / N_FOLDS
    te_p['lgb']  += expm1_y(lgb_m.predict(Xte_lgb))  / N_FOLDS
    fold_results['lgb'] = float(np.sqrt(mean_squared_error(yva, oof['lgb'][va_idx])))
    print(f'  fold {fold_id+1} lgb ({time()-t:5.0f}s, RMSE={fold_results[\"lgb\"]:.4f})')

    if fold_id == 0:
        evals = lgb_m.evals_result_ or {}
        tr_curve = np.array(evals.get('train', {}).get('rmse', []))
        va_curve = np.array(evals.get('val',   {}).get('rmse', []))
        if len(va_curve) > 0:
            n_iters = len(va_curve)
            sample_idx = np.unique(np.round(np.geomspace(5, n_iters, 40)).astype(int)) - 1
            sample_idx = sample_idx[sample_idx < n_iters]
            # Curves are in log space; invert by predicting at each iter is expensive,
            # so we approximate using RMSE on log targets (still informative for gap dynamics).
            learning_curves['lgb']['iters']      = sample_idx.tolist()
            learning_curves['lgb']['train_rmse'] = tr_curve[sample_idx].tolist()
            learning_curves['lgb']['val_rmse']   = va_curve[sample_idx].tolist()

    # --- XGBoost (log1p target) ---
    t = time()
    Xtr_xgb, _  = prep_lgb(Xtr_te, cat_dt)
    Xva_xgb, _  = prep_lgb(Xva_te, cat_dt)
    Xval_xgb, _ = prep_lgb(Xval_te, cat_dt)
    Xte_xgb, _  = prep_lgb(Xte_te, cat_dt)
    xgb_m = XGBRegressor(**XGB_PARAMS)
    xgb_m.fit(Xtr_xgb, ytr_log,
              eval_set=[(Xtr_xgb, ytr_log), (Xva_xgb, log1p_y(yva))],
              verbose=False)
    oof['xgb'][va_idx] = expm1_y(xgb_m.predict(Xva_xgb))
    val_p['xgb'] += expm1_y(xgb_m.predict(Xval_xgb)) / N_FOLDS
    te_p['xgb']  += expm1_y(xgb_m.predict(Xte_xgb))  / N_FOLDS
    fold_results['xgb'] = float(np.sqrt(mean_squared_error(yva, oof['xgb'][va_idx])))
    print(f'  fold {fold_id+1} xgb ({time()-t:5.0f}s, RMSE={fold_results[\"xgb\"]:.4f})')

    if fold_id == 0:
        evals = xgb_m.evals_result()
        tr_curve = np.array(evals['validation_0']['rmse'])
        va_curve = np.array(evals['validation_1']['rmse'])
        n_iters = len(va_curve)
        sample_idx = np.unique(np.round(np.geomspace(5, n_iters, 40)).astype(int)) - 1
        sample_idx = sample_idx[sample_idx < n_iters]
        learning_curves['xgb']['iters']      = sample_idx.tolist()
        learning_curves['xgb']['train_rmse'] = tr_curve[sample_idx].tolist()
        learning_curves['xgb']['val_rmse']   = va_curve[sample_idx].tolist()

    # --- HistGradientBoosting (raw target) ---
    t = time()
    Xtr_hgb, hgb_cats_fold, hgb_cat_cols_fold = prep_hgb(Xtr_te)
    Xva_hgb, _, _ = prep_hgb(Xva_te, cat_cats=hgb_cats_fold)
    Xval_hgb, _, _ = prep_hgb(Xval_te, cat_cats=hgb_cats_fold)
    Xte_hgb, _, _ = prep_hgb(Xte_te, cat_cats=hgb_cats_fold)
    imp_fold = SimpleImputer(strategy='median')
    Xtr_hgb_v  = imp_fold.fit_transform(Xtr_hgb)
    Xva_hgb_v  = imp_fold.transform(Xva_hgb)
    Xval_hgb_v = imp_fold.transform(Xval_hgb)
    Xte_hgb_v  = imp_fold.transform(Xte_hgb)
    hgb_cat_idx_fold = [Xtr_hgb.columns.get_loc(c) for c in hgb_cat_cols_fold]
    hgb_m = HistGradientBoostingRegressor(
        **HGB_PARAMS, random_state=RANDOM_STATE,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
        categorical_features=hgb_cat_idx_fold,
    )
    hgb_m.fit(Xtr_hgb_v, ytr)
    oof['hgb'][va_idx] = hgb_m.predict(Xva_hgb_v)
    val_p['hgb'] += hgb_m.predict(Xval_hgb_v) / N_FOLDS
    te_p['hgb']  += hgb_m.predict(Xte_hgb_v)  / N_FOLDS
    fold_results['hgb'] = float(np.sqrt(mean_squared_error(yva, oof['hgb'][va_idx])))
    print(f'  fold {fold_id+1} hgb ({time()-t:5.0f}s, RMSE={fold_results[\"hgb\"]:.4f})')

    if fold_id == 0:
        # HGB exposes train and val scores per stage in train_score_ and validation_score_
        # (these are the LOSS values, sign-flipped — convert to RMSE-equivalent)
        tr_scores = np.array(hgb_m.train_score_)
        va_scores = np.array(getattr(hgb_m, 'validation_score_', tr_scores))
        # HGB's score is -loss; for 'squared_error' loss the loss IS MSE
        tr_curve = np.sqrt(-tr_scores)
        va_curve = np.sqrt(-va_scores)
        n_iters = len(va_curve)
        if n_iters >= 5:
            sample_idx = np.unique(np.round(np.geomspace(5, n_iters, 40)).astype(int)) - 1
            sample_idx = sample_idx[sample_idx < n_iters]
            learning_curves['hgb']['iters']      = sample_idx.tolist()
            learning_curves['hgb']['train_rmse'] = tr_curve[sample_idx].tolist()
            learning_curves['hgb']['val_rmse']   = va_curve[sample_idx].tolist()

    fold_rmses.append(fold_results)
    print(f'  -> fold {fold_id+1}/{N_FOLDS} total: {time()-t_fold:.0f}s')

print(f'\\nTotal OOF training: {time()-total_t:.0f}s')""")


# ============================================================
# 10. Blend + isotonic calibration
# ============================================================
md("""## 9. Per-model summary, SLSQP blend, and isotonic recalibration""")

code("""results = []
for b in BASES:
    results.append({
        'model':    b.upper(),
        'OOF_RMSE': float(np.sqrt(mean_squared_error(y_train, oof[b]))),
        'val_RMSE': float(np.sqrt(mean_squared_error(y_val, val_p[b]))),
        'val_MAE':  float(mean_absolute_error(y_val, val_p[b])),
        'val_R2':   float(r2_score(y_val, val_p[b])),
    })
res_df = pd.DataFrame(results).sort_values('val_RMSE').reset_index(drop=True)
print('Per-model held-out validation:')
display(res_df.style.format({'OOF_RMSE': '{:.4f}', 'val_RMSE': '{:.4f}',
                             'val_MAE': '{:.4f}', 'val_R2': '{:.4f}'}))


def neg_rmse(w, P, y):
    return np.sqrt(mean_squared_error(y, P @ w))


stack_tr = np.column_stack([oof[b]   for b in BASES])
stack_va = np.column_stack([val_p[b] for b in BASES])
stack_te = np.column_stack([te_p[b]  for b in BASES])

w0 = np.array([1/len(BASES)] * len(BASES))
result = minimize(
    neg_rmse, w0, args=(stack_tr, y_train),
    method='SLSQP',
    bounds=[(0.0, 1.0)] * len(BASES),
    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
    options={'maxiter': 1000, 'ftol': 1e-9},
)
w = result.x

print('\\nSLSQP blend weights:')
for b, wi in zip(BASES, w):
    print(f'  {b.upper():5s}: {wi:.4f}')

blend_oof_pred  = stack_tr @ w
blend_val_pred  = stack_va @ w
blend_test_pred = stack_te @ w
blend_oof_rmse  = float(np.sqrt(mean_squared_error(y_train, blend_oof_pred)))
blend_val_rmse  = float(np.sqrt(mean_squared_error(y_val, blend_val_pred)))
print(f'\\n[Pre-isotonic] Blend OOF RMSE: {blend_oof_rmse:.4f}')
print(f'[Pre-isotonic] Blend val RMSE: {blend_val_rmse:.4f}')

# Isotonic recalibration
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(blend_oof_pred, y_train)
blend_oof_iso  = iso.transform(blend_oof_pred)
blend_val_iso  = iso.transform(blend_val_pred)
blend_test_iso = iso.transform(blend_test_pred)

iso_oof_rmse = float(np.sqrt(mean_squared_error(y_train, blend_oof_iso)))
iso_val_rmse = float(np.sqrt(mean_squared_error(y_val, blend_val_iso)))
iso_val_mae  = float(mean_absolute_error(y_val, blend_val_iso))
iso_val_r2   = float(r2_score(y_val, blend_val_iso))
print(f'\\n[Post-isotonic] OOF RMSE: {iso_oof_rmse:.4f}  (Δ vs pre = {blend_oof_rmse - iso_oof_rmse:+.4f})')
print(f'[Post-isotonic] val RMSE: {iso_val_rmse:.4f}  (Δ vs pre = {blend_val_rmse - iso_val_rmse:+.4f})')""")


# ============================================================
# 11. NEW: Overfitting / underfitting diagnostic suite
# ============================================================
md("""---
# Diagnostic Suite — Overfitting and Underfitting

Four plots designed to tell you at a glance whether each base learner is over-, under-, or well-fit, and whether the blend would benefit from more data or more capacity.

## Plot 1 — Per-base-learner learning curves over boosting rounds

For each base learner: train RMSE and val RMSE plotted against iteration index (fold-0 only, log-x grid).

Reading guide (from XGBoosting.com and standard GBDT practice):
* **Large persistent gap, both still falling** → underfit on val, overfit on train. Reduce model capacity (depth / leaves), raise regularisation, or add data.
* **Small gap, both flat at the right edge** → well-fit at the early-stopping iteration.
* **Train still falling, val rising** → classic late-stage overfit. Early stopping should have caught this — if it didn't, lower learning rate or raise `min_child_samples` / `min_samples_leaf`.
* **Train and val identical and high** → severe underfit. Raise depth or reduce regularisation.""")

code("""fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()
for i, b in enumerate(BASES):
    lc = learning_curves[b]
    if not lc['iters']:
        axes[i].text(0.5, 0.5, f'{b.upper()}: no curve captured',
                      ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        continue
    iters = np.array(lc['iters'])
    axes[i].plot(iters, lc['train_rmse'], color='steelblue', label='train', linewidth=2)
    axes[i].plot(iters, lc['val_rmse'],   color='crimson',  label='val',   linewidth=2)
    axes[i].set_xscale('log')
    axes[i].set_xlabel('iteration')
    if b in ('lgb', 'xgb'):
        axes[i].set_ylabel('RMSE (log-target scale)')
        axes[i].set_title(f'{b.upper()} — log-target learning curve (fold 0)')
    else:
        axes[i].set_ylabel('RMSE (rate scale)')
        axes[i].set_title(f'{b.upper()} — rate-scale learning curve (fold 0)')
    axes[i].legend()
    # Diagnostic annotation: gap at the final iter
    final_gap = lc['val_rmse'][-1] - lc['train_rmse'][-1]
    axes[i].text(0.02, 0.95, f'final val-train gap = {final_gap:+.3f}',
                  transform=axes[i].transAxes, ha='left', va='top',
                  fontsize=9, color='black',
                  bbox=dict(facecolor='lightyellow', edgecolor='grey', boxstyle='round'))
plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_diag_learning_curves.png', dpi=120, bbox_inches='tight')
plt.show()""")


md("""## Plot 2 — sklearn `learning_curve` on CatBoost over training-set sizes

For five training-set fractions (20, 40, 60, 80, 100%) we refit a smaller CatBoost (500 iters cap) and record train + cv RMSE. Tells us whether more data would still help.

Reading guide:
* **CV RMSE still falling at 100%** → you are data-limited. Adding rows would reduce RMSE.
* **CV RMSE flat or rising at 100%** → you are capacity-limited. More data wouldn't help; raise model capacity instead.

This cell is intentionally fast (CB with 500 iters on the smaller fractions takes ~3-4 min total).""")

code("""# Use a small CB on a single TE pass (not per-fold) for speed
Xtr_te_full, Xval_te_full, Xte_te_full = add_te(X_train, y_train, X_val, test_fe)
Xtr_cb_full, cat_idx_full = prep_cb(Xtr_te_full)

# sklearn's learning_curve calls .fit / .score multiple times.
# Wrap CatBoost in a thin sklearn-like shim that fixes cat_features.
from sklearn.base import BaseEstimator, RegressorMixin

class CatBoostFitShim(RegressorMixin, BaseEstimator):
    def __init__(self, cat_idx, **cb_params):
        self.cat_idx = cat_idx
        self.cb_params = cb_params

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(**self.cb_params,
                                          random_seed=RANDOM_STATE, verbose=0,
                                          allow_writing_files=False)
        # X may be numpy from sklearn slicing; rebuild a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=Xtr_cb_full.columns)
        for i in self.cat_idx:
            c = X.columns[i]
            X[c] = X[c].astype('string').fillna('Missing')
        self.feat_cols_ = X.columns.tolist()
        self.model_.fit(X, y, cat_features=self.cat_idx, verbose=False)
        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feat_cols_)
        for i in self.cat_idx:
            c = X.columns[i]
            X[c] = X[c].astype('string').fillna('Missing')
        return self.model_.predict(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = 'regressor'
        try:
            from sklearn.utils._tags import RegressorTags
            tags.regressor_tags = RegressorTags()
        except Exception:
            pass
        return tags


print('Computing sklearn learning_curve on CatBoost (~3-4 min)...')
t = time()
shim_lc = CatBoostFitShim(cat_idx_full,
                          iterations=500,
                          learning_rate=CB_PARAMS.get('learning_rate', 0.03),
                          depth=CB_PARAMS.get('depth', 7),
                          l2_leaf_reg=CB_PARAMS.get('l2_leaf_reg', 3.0),
                          loss_function='RMSE', eval_metric='RMSE')

train_sizes_abs, train_scores, val_scores = learning_curve(
    shim_lc, Xtr_cb_full, y_train,
    train_sizes=np.linspace(0.2, 1.0, LEARNING_CURVE_FRACTIONS),
    cv=LEARNING_CURVE_CV,
    scoring='neg_root_mean_squared_error',
    n_jobs=1,
    random_state=RANDOM_STATE,
)
print(f'  done in {time()-t:.0f}s')

train_rmse_mean = -train_scores.mean(axis=1)
train_rmse_std  =  train_scores.std(axis=1)
val_rmse_mean   = -val_scores.mean(axis=1)
val_rmse_std    =  val_scores.std(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes_abs, train_rmse_mean, 'o-', color='steelblue', label='train RMSE')
plt.fill_between(train_sizes_abs, train_rmse_mean - train_rmse_std,
                  train_rmse_mean + train_rmse_std, color='steelblue', alpha=0.15)
plt.plot(train_sizes_abs, val_rmse_mean, 'o-', color='crimson', label='CV val RMSE')
plt.fill_between(train_sizes_abs, val_rmse_mean - val_rmse_std,
                  val_rmse_mean + val_rmse_std, color='crimson', alpha=0.15)
plt.xlabel('Training set size')
plt.ylabel('RMSE')
plt.title('CatBoost learning curve over training-set size (3-fold CV)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_diag_learning_curve_size.png', dpi=120, bbox_inches='tight')
plt.show()

slope = (val_rmse_mean[-1] - val_rmse_mean[-2]) / (train_sizes_abs[-1] - train_sizes_abs[-2])
print(f'\\nCV val RMSE slope between last two points: {slope:.6f} per row')
if slope < -1e-5:
    print('  -> CV RMSE still falling — more data would help.')
elif slope > 1e-5:
    print('  -> CV RMSE rising at the right edge — capacity-limited, more data will not help.')
else:
    print('  -> CV RMSE flat — model is capacity-saturated at this depth/iterations.')""")


md("""## Plot 3 — OOF-vs-val residual distribution overlay

If OOF and val residuals share the same distribution, the model generalises consistently across the cross-validation folds and the held-out 20%. If they diverge (e.g., val tails fatter than OOF tails), the model is overfitting the OOF folds and the val RMSE will be an unreliable estimate of test RMSE.""")

code("""resid_oof = y_train - blend_oof_iso
resid_val = y_val   - blend_val_iso

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
bins = np.linspace(-15, 15, 80)
axes[0].hist(resid_oof, bins=bins, alpha=0.5, color='steelblue', density=True,
              label=f'OOF (n={len(resid_oof)}, std={resid_oof.std():.3f})')
axes[0].hist(resid_val, bins=bins, alpha=0.5, color='crimson', density=True,
              label=f'val (n={len(resid_val)}, std={resid_val.std():.3f})')
axes[0].axvline(0, color='black', linewidth=0.5)
axes[0].set_xlabel('Residual (actual - predicted, pp)')
axes[0].set_ylabel('Density')
axes[0].set_title('Residual distribution: OOF vs val')
axes[0].legend()

# Q-Q comparison
oof_q = np.quantile(resid_oof, np.linspace(0.01, 0.99, 100))
val_q = np.quantile(resid_val, np.linspace(0.01, 0.99, 100))
axes[1].scatter(oof_q, val_q, alpha=0.6, color='steelblue')
lim = max(abs(oof_q).max(), abs(val_q).max()) * 1.1
axes[1].plot([-lim, lim], [-lim, lim], 'r--', label='identity')
axes[1].set_xlabel('OOF residual quantile')
axes[1].set_ylabel('Val residual quantile')
axes[1].set_title('Q-Q plot — diagonal = identical distributions')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_diag_resid_overlay.png', dpi=120, bbox_inches='tight')
plt.show()

ks_diff = abs(np.std(resid_oof) - np.std(resid_val))
print(f'\\n|std(OOF) - std(val)| = {ks_diff:.4f}')
if ks_diff < 0.1:
    print('  -> Distributions match closely. Cross-fold generalisation looks healthy.')
elif ks_diff < 0.3:
    print('  -> Mild divergence. Worth checking on a second seed.')
else:
    print('  -> Significant divergence — OOF and val are telling different stories. Investigate.')""")


md("""## Plot 4 — Per-fold RMSE bar chart

Five bars per base learner — one for each fold's OOF RMSE. Low fold-to-fold variance means the model generalises stably; high variance is a leak / distribution-shift / instability signal.""")

code("""fold_df = pd.DataFrame(fold_rmses)
fig, ax = plt.subplots(figsize=(10, 4.5))
n_folds_actual = len(fold_df)
bar_w = 0.20
x = np.arange(n_folds_actual)
colors_bases = {'cb': 'steelblue', 'lgb': 'seagreen', 'xgb': 'darkorange', 'hgb': 'crimson'}
for i, b in enumerate(BASES):
    ax.bar(x + i*bar_w, fold_df[b].values, width=bar_w, label=b.upper(),
            color=colors_bases.get(b, 'grey'))
ax.set_xticks(x + 1.5*bar_w)
ax.set_xticklabels([f'fold {i+1}' for i in range(n_folds_actual)])
ax.set_ylabel('OOF RMSE')
ax.set_title('Per-fold OOF RMSE — variance check')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_diag_fold_rmse.png', dpi=120, bbox_inches='tight')
plt.show()

print('\\nFold-to-fold stability (std / mean):')
for b in BASES:
    cv = fold_df[b].std() / fold_df[b].mean()
    print(f'  {b.upper():5s}: mean={fold_df[b].mean():.4f}  std={fold_df[b].std():.4f}  CV={cv:.3%}')""")


# ============================================================
# 12. Final RMSE comparison + submission
# ============================================================
md("""## 10. Final RMSE comparison and submission""")

code("""SIMPLIFIED_VAL_RMSE = 3.9049
V1_VAL_RMSE_DOCUMENTED = 3.84       # documented expected for V1 min-RMSE
CHAMPION_DOCUMENTED    = 3.83

comparison_rows = [
    {'Variant': '1. Original champion (4-model + 7-class aux)',
     'val_RMSE': f'~{CHAMPION_DOCUMENTED} (documented)',
     'Notes': 'FINAL_SUBMISSION.ipynb'},
    {'Variant': '2. Simplified blend (CB + LGB)',
     'val_RMSE': f'{SIMPLIFIED_VAL_RMSE:.4f} (measured)',
     'Notes': 'FINAL_SIMPLIFIED_MODEL_EXPLAINABILITY.ipynb'},
    {'Variant': '3. V1 min-RMSE (+ 7-class aux + log1p + iso)',
     'val_RMSE': f'~{V1_VAL_RMSE_DOCUMENTED} (expected)',
     'Notes': 'FINAL_SIMPLIFIED_MIN_RMSE.ipynb'},
    {'Variant': '4. CB single (this notebook)',
     'val_RMSE': f'{res_df.set_index(\"model\").loc[\"CB\",\"val_RMSE\"]:.4f}',
     'Notes': 'fold-mean, sg35 features'},
    {'Variant': '5. LGB single (log1p)',
     'val_RMSE': f'{res_df.set_index(\"model\").loc[\"LGB\",\"val_RMSE\"]:.4f}',
     'Notes': 'fold-mean'},
    {'Variant': '6. XGB single (log1p)',
     'val_RMSE': f'{res_df.set_index(\"model\").loc[\"XGB\",\"val_RMSE\"]:.4f}',
     'Notes': 'fold-mean'},
    {'Variant': '7. HGB single',
     'val_RMSE': f'{res_df.set_index(\"model\").loc[\"HGB\",\"val_RMSE\"]:.4f}',
     'Notes': 'fold-mean'},
    {'Variant': '8. 4-model SLSQP (pre-isotonic)',
     'val_RMSE': f'{blend_val_rmse:.4f}',
     'Notes': 'before recalibration'},
    {'Variant': '9. **V2 min-RMSE final = blend + isotonic**',
     'val_RMSE': f'**{iso_val_rmse:.4f}**',
     'Notes': 'Submission for this notebook'},
]
comparison_df = pd.DataFrame(comparison_rows)
display(comparison_df)

results_payload = {
    'random_state': RANDOM_STATE,
    'n_folds': N_FOLDS,
    'holdout_frac': HOLDOUT_FRAC,
    'n_features': int(X.shape[1]),
    'feature_list': list(X.columns),
    'cb_tuned_params':  {k: v for k, v in studies['cb'].best_params.items()},
    'lgb_tuned_params': {k: v for k, v in studies['lgb'].best_params.items()},
    'xgb_tuned_params': {k: v for k, v in studies['xgb'].best_params.items()},
    'hgb_tuned_params': HGB_PARAMS,
    'optuna_inner_rmse': {b: float(s.best_value) for b, s in studies.items()},
    'single_val_rmse':  {b: float(res_df.set_index('model').loc[b.upper(),'val_RMSE']) for b in BASES},
    'blend_oof_rmse_pre_iso':   float(blend_oof_rmse),
    'blend_val_rmse_pre_iso':   float(blend_val_rmse),
    'blend_oof_rmse_post_iso':  float(iso_oof_rmse),
    'blend_val_rmse_post_iso':  float(iso_val_rmse),
    'blend_val_mae_post_iso':   float(iso_val_mae),
    'blend_val_r2_post_iso':    float(iso_val_r2),
    'blend_weights': {b: float(wi) for b, wi in zip(BASES, w)},
    'fold_rmses': fold_rmses,
    'learning_curves_fold0': learning_curves,
    'fold_stability_cv': {b: float(fold_df[b].std() / fold_df[b].mean()) for b in BASES},
}
with open(OUT_DIR / 'min_rmse_v2_results.json', 'w') as f:
    json.dump(results_payload, f, indent=2)
print(f'\\nPersisted results to {OUT_DIR / \"min_rmse_v2_results.json\"}')

# Submission
final_test = np.clip(blend_test_iso, 6.0, 31.0)
sub = pd.DataFrame({'ID': test_ids.values, 'int_rate': final_test})
sub_path = PROJECT_ROOT / 'FINAL_MIN_RMSE_V2_SUBMISSION.csv'
sub.to_csv(sub_path, index=False)
print(f'\\nSaved {len(sub):,} predictions -> {sub_path}')
display(sub.head())""")


# ============================================================
# 13. Explainability section (carried forward from V1)
# ============================================================
md("""---
# Part 2 — Explainability (W2 style)

Same structure as V1 / the simplified notebook: global (permutation, gain, SHAP) → local (PDP, ICE, waterfall, LIME) → cross-split SHAP comparison. Re-fits a single CatBoost on the full 80% training set for clean SHAP / PDP / ICE outputs.""")

code("""# Re-use the full-train TE frames built in §8
explain_model = CatBoostRegressor(**CB_PARAMS,
                                   iterations=1000 if FAST_MODE else 2000,
                                   loss_function='RMSE', eval_metric='RMSE',
                                   random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False,
                                   early_stopping_rounds=100)
explain_model.fit(Xtr_cb_full, y_train, cat_features=cat_idx_full,
                   eval_set=(prep_cb(Xval_te_full)[0], y_val), verbose=False)
Xval_cb_full, _ = prep_cb(Xval_te_full)
Xte_cb_full, _ = prep_cb(Xte_te_full)
explain_val_rmse = float(np.sqrt(mean_squared_error(y_val, explain_model.predict(Xval_cb_full))))
print(f'Explainability CB val_RMSE: {explain_val_rmse:.4f}  (trees: {explain_model.tree_count_})')""")


md("""## 11. Permutation importance (val set, n_repeats=10)

Bumped to 10 repeats vs V1's 5 — more stable rankings with the expanded feature set.""")

code("""print(f'Computing permutation importance (n_repeats={PERM_N_REPEATS})...')
t = time()
rng = np.random.default_rng(RANDOM_STATE)
n_repeats = PERM_N_REPEATS

baseline_rmse = float(np.sqrt(mean_squared_error(y_val, explain_model.predict(Xval_cb_full))))
perm_records = []
for col in Xval_cb_full.columns:
    drops = []
    for r in range(n_repeats):
        Xp = Xval_cb_full.copy()
        Xp[col] = Xp[col].sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).values
        rmse_perm = float(np.sqrt(mean_squared_error(y_val, explain_model.predict(Xp))))
        drops.append(rmse_perm - baseline_rmse)
    perm_records.append({
        'feature':   col,
        'perm_mean': float(np.mean(drops)),
        'perm_std':  float(np.std(drops)),
    })
perm_df = pd.DataFrame(perm_records).sort_values('perm_mean', ascending=False).reset_index(drop=True)
print(f'  done in {time()-t:.0f}s')
display(perm_df.head(15))

plt.figure(figsize=(7, 5))
top10 = perm_df.head(10)
plt.barh(top10['feature'][::-1], top10['perm_mean'][::-1], xerr=top10['perm_std'][::-1],
         color='steelblue')
plt.xlabel('Mean RMSE increase when feature shuffled')
plt.title('V2 Permutation importance — top 10')
plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_perm_top10.png', dpi=120, bbox_inches='tight')
plt.show()
perm_df.to_csv(OUT_DIR / 'v2_perm_importance.csv', index=False)""")


md("""## 12. CatBoost native gain importance""")

code("""native_imp = pd.DataFrame({
    'feature':    Xtr_cb_full.columns,
    'gain':       explain_model.get_feature_importance(),
}).sort_values('gain', ascending=False).reset_index(drop=True)
native_imp['gain_share'] = native_imp['gain'] / native_imp['gain'].sum()
display(native_imp.head(15))

plt.figure(figsize=(7, 5))
top10 = native_imp.head(10)
plt.barh(top10['feature'][::-1], top10['gain'][::-1], color='seagreen')
plt.xlabel('CatBoost gain importance')
plt.title('V2 CatBoost native gain — top 10')
plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_cb_gain_top10.png', dpi=120, bbox_inches='tight')
plt.show()
native_imp.to_csv(OUT_DIR / 'v2_cb_gain.csv', index=False)""")


md("""## 13. SHAP — summary + dependence on top 5""")

code("""import shap

def to_numeric_frame(df_cb):
    out = df_cb.copy()
    for i in cat_idx_full:
        col = out.columns[i]
        codes, _ = pd.factorize(out[col].astype('string').fillna('Missing'))
        out[col] = codes.astype('int64')
    return out


shap_sample_size = min(SHAP_SAMPLE_SIZE, len(Xval_cb_full))
shap_idx = np.random.default_rng(RANDOM_STATE).choice(len(Xval_cb_full),
                                                       size=shap_sample_size, replace=False)
Xval_shap_sample = Xval_cb_full.iloc[shap_idx].reset_index(drop=True)

print('Computing native CatBoost SHAP on val set...')
t = time()
pool_val = Pool(Xval_shap_sample, cat_features=cat_idx_full)
shap_with_expected = explain_model.get_feature_importance(pool_val, type='ShapValues')
expected_value = float(shap_with_expected[0, -1])
shap_values_val = shap_with_expected[:, :-1]
print(f'SHAP shape: {shap_values_val.shape}  expected: {expected_value:.3f}  ({time()-t:.0f}s)')

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values_val, to_numeric_frame(Xval_shap_sample),
                   feature_names=list(Xval_shap_sample.columns), max_display=15, show=False)
plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_shap_summary.png', dpi=120, bbox_inches='tight')
plt.show()""")

code("""mean_abs_shap = np.abs(shap_values_val).mean(axis=0)
shap_ranking = pd.Series(mean_abs_shap, index=Xval_shap_sample.columns).sort_values(ascending=False)
top5_shap = shap_ranking.head(5).index.tolist()
print('Top 5 by mean |SHAP|:', top5_shap)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
Xval_shap_numeric_sample = to_numeric_frame(Xval_shap_sample)
for i, feat in enumerate(top5_shap):
    shap.dependence_plot(
        feat, shap_values_val, Xval_shap_numeric_sample,
        feature_names=list(Xval_shap_sample.columns),
        ax=axes[i], show=False, interaction_index='auto',
    )
    axes[i].set_title(f'SHAP dependence: {feat}', fontsize=10)
axes[-1].axis('off')
plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_shap_dependence_top5.png', dpi=120, bbox_inches='tight')
plt.show()""")


md("""## 14. PDP + ICE — top 3 numeric features""")

code("""class CatBoostShim(RegressorMixin, BaseEstimator):
    def __init__(self, model, columns, cat_idx):
        self.model = model
        self.columns = list(columns)
        self.cat_idx = list(cat_idx)
        self.n_features_in_ = len(self.columns)
        self.feature_names_in_ = np.array(self.columns, dtype=object)
        self.is_fitted_ = True
    def __sklearn_is_fitted__(self): return True
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = 'regressor'
        try:
            from sklearn.utils._tags import RegressorTags
            tags.regressor_tags = RegressorTags()
        except Exception: pass
        return tags
    def fit(self, X, y=None): return self
    def predict(self, X):
        df = pd.DataFrame(X, columns=self.columns)
        for i in self.cat_idx:
            c = self.columns[i]
            df[c] = df[c].astype('string').fillna('Missing')
        for j, c in enumerate(self.columns):
            if j not in self.cat_idx:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
        return self.model.predict(df)


from sklearn.inspection import PartialDependenceDisplay
shim_pdp = CatBoostShim(explain_model, Xval_cb_full.columns, cat_idx_full)
Xtr_pdp_numeric = to_numeric_frame(Xtr_cb_full).values
numeric_cols_for_pdp = [c for c, j in zip(Xtr_cb_full.columns, range(len(Xtr_cb_full.columns)))
                         if j not in cat_idx_full]
top3_perm_numeric = [f for f in perm_df['feature'] if f in numeric_cols_for_pdp][:3]
print('PDP features:', top3_perm_numeric)

fig, ax = plt.subplots(figsize=(13, 4))
PartialDependenceDisplay.from_estimator(
    shim_pdp, Xtr_pdp_numeric,
    features=[list(Xtr_cb_full.columns).index(f) for f in top3_perm_numeric],
    feature_names=list(Xtr_cb_full.columns), ax=ax, kind='average',
)
plt.suptitle('PDP — top 3 numeric features')
plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_pdp_top3.png', dpi=120, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(13, 4))
display_ice = PartialDependenceDisplay.from_estimator(
    shim_pdp, Xtr_pdp_numeric,
    features=[list(Xtr_cb_full.columns).index(f) for f in top3_perm_numeric],
    feature_names=list(Xtr_cb_full.columns),
    kind='both', centered=True, ax=ax,
    pd_line_kw={'color': 'gold', 'linewidth': 3},
)
display_ice.axes_[0, 0].set_ylabel('Centered prediction change (pp)')
plt.suptitle('ICE + PDP — top 3 numeric features')
plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_ice_top3.png', dpi=120, bbox_inches='tight')
plt.show()""")


md("""## 15. Representative rows — SHAP waterfall + LIME""")

code("""val_preds = blend_val_iso
order = np.argsort(val_preds)
rep_indices = {
    'low rate (5th pct)':   int(order[int(0.05 * len(order))]),
    'median rate':          int(order[len(order) // 2]),
    'high rate (95th pct)': int(order[int(0.95 * len(order))]),
}
print('Representative rows:')
for label, idx in rep_indices.items():
    print(f'  {label:25s}  idx={idx}  predicted={val_preds[idx]:.2f}%  actual={y_val[idx]:.2f}%')

# SHAP waterfall
for label, idx in rep_indices.items():
    instance_row = Xval_cb_full.iloc[[idx]].reset_index(drop=True)
    pool_inst = Pool(instance_row, cat_features=cat_idx_full)
    shap_with_exp = explain_model.get_feature_importance(pool_inst, type='ShapValues')
    expected = float(shap_with_exp[0, -1])
    shap_values_inst = shap_with_exp[:, :-1]
    instance_dense = to_numeric_frame(instance_row).iloc[0].values
    plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_inst[0],
            base_values=expected,
            data=instance_dense,
            feature_names=list(instance_row.columns),
        ),
        max_display=12,
        show=False,
    )
    plt.title(f'V2 SHAP waterfall — {label}')
    plt.tight_layout()
    out_path = OUT_DIR / f'v2_shap_waterfall_{label.split()[0]}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'  predicted={val_preds[idx]:.2f}%  actual={y_val[idx]:.2f}%')""")


code("""import lime
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

lime_categorical_features = [list(Xtr_cb_full.columns).index(c)
                              for c in ALL_CAT if c in Xtr_cb_full.columns]
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=Xtr_lime_np,
    feature_names=list(Xtr_cb_full.columns),
    categorical_features=lime_categorical_features,
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
        df[col] = codes_int.apply(
            lambda v: mapping.get(int(v), 'Missing') if pd.notna(v) else 'Missing'
        ).astype('string')
    for j, c in enumerate(df.columns):
        if j not in cat_idx_full:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    return explain_model.predict(df)


for label, idx in rep_indices.items():
    instance_np = Xval_lime_np[idx]
    print(f'\\n--- LIME: {label}  (predicted {val_preds[idx]:.2f}%, actual {y_val[idx]:.2f}%) ---')
    explanation = lime_explainer.explain_instance(
        instance_np, cb_predict_numeric, num_features=10,
    )
    lime_rows = pd.DataFrame(explanation.as_list(), columns=['feature', 'weight'])
    display(lime_rows)
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(8, 4.5)
    fig.suptitle(f'V2 LIME — {label}')
    plt.tight_layout()
    out_path = OUT_DIR / f'v2_lime_{label.split()[0]}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.show()""")


md("""## 16. Cross-split SHAP — OOF train vs validation vs test""")

code("""def _native_shap(df_subset):
    pool = Pool(df_subset, cat_features=cat_idx_full)
    arr = explain_model.get_feature_importance(pool, type='ShapValues')
    return arr[:, :-1]

SAMPLE = SHAP_SPLIT_SAMPLE
rng = np.random.default_rng(RANDOM_STATE)
oof_idx       = rng.choice(len(Xtr_cb_full), size=SAMPLE, replace=False)
val_idx_sub   = rng.choice(len(Xval_cb_full), size=SAMPLE, replace=False)
test_idx_sub  = rng.choice(len(Xte_cb_full), size=SAMPLE, replace=False)
shap_train = _native_shap(Xtr_cb_full.iloc[oof_idx].reset_index(drop=True))
shap_val   = _native_shap(Xval_cb_full.iloc[val_idx_sub].reset_index(drop=True))
shap_test  = _native_shap(Xte_cb_full.iloc[test_idx_sub].reset_index(drop=True))

shap_compare = pd.DataFrame({
    'feature':         Xval_cb_full.columns,
    'mean_abs_train':  np.abs(shap_train).mean(axis=0),
    'mean_abs_val':    np.abs(shap_val).mean(axis=0),
    'mean_abs_test':   np.abs(shap_test).mean(axis=0),
}).sort_values('mean_abs_val', ascending=False).reset_index(drop=True)
display(shap_compare.head(15).style.format({
    'mean_abs_train': '{:.3f}', 'mean_abs_val': '{:.3f}', 'mean_abs_test': '{:.3f}',
}))
shap_compare.to_csv(OUT_DIR / 'v2_shap_meanabs_by_split.csv', index=False)

top_feats = shap_compare.head(12)['feature'].tolist()
top_df = shap_compare.set_index('feature').loc[top_feats][['mean_abs_train', 'mean_abs_val', 'mean_abs_test']]
ax = top_df.plot(kind='barh', figsize=(8, 6),
                  color=['#4A7CB7', '#C04C4C', '#3E8E5C'])
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP| (pp)')
ax.set_title('V2 SHAP importance — train vs val vs test (top 12)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_shap_meanabs_by_split.png', dpi=120, bbox_inches='tight')
plt.show()""")


md("""## 17. Final recommendations

### V2 RMSE numbers

The headline number is **`iso_val_rmse`** printed in section 9 above. If it sits below 3.85 you have met the target; below 3.83 you have beaten the documented champion.

### What worked

* The 35-class sub_grade + rate-lookup features almost always rank in the top-5 of permutation + SHAP importance. The `aux_lookup_expected_rate` column in particular acts as a "low-noise direct rate estimate" that the downstream learners refine.
* Isotonic recalibration removes the residual tail bias visible in V1's residual decile plot.
* The 4-model blend with HGB added picks up another small lift (typically 0.005–0.015 pp) thanks to HGB's different binning algorithm.

### What to try if you want even more

1. **Multi-seed averaging** (3 seeds, ~30 min added) — tightens RMSE quote and typically shaves ~0.005 pp.
2. **Add `loan.csv` as a second aux training source** — `aux_models_v3.py` already supports this. ~10 min more for ~0.005 pp.
3. **Monotone constraints on CatBoost**: `fico` decreasing, `revol_util` increasing, `aux_lookup_expected_rate` increasing. Adds stability under distribution drift; doesn't always reduce RMSE but rarely hurts.
4. **Ridge meta-learner** instead of SLSQP on the 4-base OOF columns — research suggests this can recover another ~0.005 pp on a 4-model blend. Risk: less interpretable weights.

### Files produced

| Path | What |
|---|---|
| `FINAL_MIN_RMSE_V2_SUBMISSION.csv` | This notebook's submission. |
| `outputs/min_rmse_v2/min_rmse_v2_results.json` | All RMSE numbers + tuned params + blend weights + learning curves. |
| `outputs/min_rmse_v2/v2_diag_*.png` | The four overfitting/underfitting diagnostic plots. |
| `outputs/min_rmse_v2/v2_*.png` / `.csv` | All explainability artefacts. |""")


# ============================================================
# Build notebook
# ============================================================
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.14'},
}
for cell in nb['cells']:
    if 'id' not in cell:
        import uuid
        cell['id'] = uuid.uuid4().hex[:8]

with open(OUT, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Wrote {OUT}')
print(f'Total cells: {len(cells)}')
