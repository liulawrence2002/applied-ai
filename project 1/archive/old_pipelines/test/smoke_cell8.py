"""Smoke test for cell 8 (Optuna HGB tuning) in FINAL_SUBMISSION.ipynb.

Runs the prerequisite setup (load -> FE -> split -> helpers) but SKIPS the
slow aux-grade classifier (cell 5), since that's not needed for cell 8 to run
and would add ~5 min to the test. Then runs cell 8 with N_TRIALS_HGB=2 to
verify it executes end-to-end with no errors.

This is a smoke test, not a correctness test — we only check that:
  1. Cell 8 runs without exception
  2. study.best_params is non-empty and the right keys
  3. HGB_PARAMS gets populated for downstream cells
"""
import sys
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'true data'
RANDOM_STATE = 6604
HOLDOUT_FRAC = 0.20
N_TRIALS_HGB = 2  # smoke-test value

# ----- cell 2: load -----
STRING_NUMERIC_COLS = [
    'dti', 'revol_util', 'all_util', 'mo_sin_old_il_acct',
    'mths_since_last_record', 'mths_since_rcnt_il', 'mths_since_recent_bc',
    'mths_since_recent_inq', 'tot_cur_bal',
]
DT_OVERRIDES = {c: 'float64' for c in STRING_NUMERIC_COLS}
train_raw = pd.read_csv(DATA_DIR / 'LC_train.csv', na_values=['NA'], dtype=DT_OVERRIDES, low_memory=False)
test_raw = pd.read_csv(DATA_DIR / 'LC_test.csv', na_values=['NA'], dtype=DT_OVERRIDES, low_memory=False)
train_raw = train_raw.drop(columns=['loan_status'])
test_raw = test_raw.drop(columns=['loan_status'])
print(f'[load] train={train_raw.shape}  test={test_raw.shape}')

# ----- cell 4: FE (paste verbatim from build_final_notebook.py) -----
EMP_LENGTH_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10,
}
STATE_MEDIAN_INCOME = {'AL':56.9,'AK':84.8,'AZ':72.6,'AR':56.3,'CA':91.6,'CO':87.6,'CT':88.4,'DE':79.3,'FL':67.9,'GA':71.4,'HI':92.5,'ID':70.2,'IL':78.4,'IN':67.2,'IA':70.5,'KS':69.7,'KY':60.2,'LA':57.6,'ME':68.3,'MD':98.5,'MA':96.5,'MI':68.5,'MN':84.3,'MS':52.7,'MO':65.9,'MT':66.8,'NE':71.7,'NV':71.6,'NH':90.8,'NJ':97.1,'NM':58.7,'NY':81.4,'NC':66.2,'ND':73.0,'OH':66.6,'OK':61.4,'OR':76.6,'PA':73.8,'RI':81.4,'SC':62.5,'SD':69.5,'TN':64.0,'TX':73.0,'UT':86.8,'VT':74.0,'VA':87.2,'WA':91.3,'WV':55.2,'WI':72.5,'WY':72.4,'DC':101.0}
STATE_UNEMPLOYMENT = {'AL':3.8,'AK':6.5,'AZ':5.0,'AR':3.9,'CA':5.7,'CO':4.1,'CT':5.4,'DE':4.8,'FL':4.3,'GA':4.3,'HI':4.7,'ID':3.5,'IL':5.5,'IN':4.0,'IA':3.7,'KS':3.8,'KY':4.7,'LA':5.4,'ME':4.0,'MD':4.5,'MA':4.8,'MI':5.2,'MN':3.9,'MS':5.7,'MO':4.1,'MT':4.0,'NE':2.9,'NV':6.4,'NH':3.4,'NJ':5.7,'NM':6.1,'NY':5.4,'NC':4.6,'ND':3.0,'OH':4.7,'OK':4.4,'OR':5.0,'PA':5.4,'RI':5.4,'SC':4.5,'SD':3.0,'TN':4.3,'TX':5.0,'UT':3.4,'VT':3.0,'VA':4.0,'WA':5.3,'WV':5.2,'WI':3.7,'WY':4.4,'DC':5.8}


def engineer(df):
    out = df.copy()
    out['fico'] = (out['fico_range_low'] + out['fico_range_high']) / 2
    out['fico_band'] = pd.cut(out['fico'], bins=[-np.inf, 660, 690, 720, 760, np.inf], labels=False).astype('Int64')
    out['fico_above_prime'] = (out['fico'] >= 720).astype('int8')
    out['fico_subprime'] = (out['fico'] < 660).astype('int8')
    out['term_months'] = out['term'].astype('string').str.extract(r'(\d+)', expand=False).astype('Int64')
    out['emp_length_num'] = out['emp_length'].map(EMP_LENGTH_MAP)
    for c in ['mths_since_last_record', 'mths_since_recent_inq', 'mths_since_rcnt_il', 'mths_since_recent_bc']:
        out[f'has_{c}'] = out[c].notna().astype(int)
    out['loan_to_income'] = out['loan_amnt'] / out['annual_inc'].replace(0, np.nan)
    out['installment_proxy'] = out['loan_amnt'] / out['term_months'].astype('float64')
    out['payment_to_income'] = (out['installment_proxy']) / (out['annual_inc'].replace(0, np.nan) / 12)
    out['acc_open_ratio'] = out['open_acc'] / out['total_acc'].replace(0, np.nan)
    out['dti_income_interaction'] = out['dti'] * out['annual_inc']
    out['fico_revol_interaction'] = out['fico'] * out['revol_util']
    out['fico_x_dti'] = out['fico'] * out['dti']
    out['term_x_dti'] = out['term_months'].astype('float64') * out['dti']
    out['dti_band'] = pd.cut(out['dti'], bins=[-np.inf, 10, 20, 30, 40, np.inf], labels=False).astype('Int64')
    out['derog_score'] = (out['delinq_2yrs'].fillna(0)*8 + out['pub_rec'].fillna(0)*13 + out['pub_rec_bankruptcies'].fillna(0)*22 + out['chargeoff_within_12_mths'].fillna(0)*15 + out['collections_12_mths_ex_med'].fillna(0)*10)
    out['inq_intensity'] = out['inq_last_12m'].fillna(0) + out['inq_fi'].fillna(0)
    out['delinq_flag'] = (out['delinq_2yrs'].fillna(0) > 0).astype('int8')
    out['pub_rec_flag'] = (out['pub_rec'].fillna(0) > 0).astype('int8')
    out['bankrupt_flag'] = (out['pub_rec_bankruptcies'].fillna(0) > 0).astype('int8')
    out['any_derog'] = (out['derog_score'] > 0).astype('int8')
    out['revol_util_band'] = pd.cut(out['revol_util'], bins=[-np.inf, 30, 50, 75, 100, np.inf], labels=False).astype('Int64')
    out['util_max'] = np.maximum(out['all_util'].fillna(0), out['revol_util'].fillna(0))
    out['util_maxed_out'] = (out['revol_util'].fillna(0) >= 95).astype('int8')
    out['credit_file_age_yrs'] = out['mo_sin_old_rev_tl_op'] / 12.0
    out['inq_per_credit_age'] = out['inq_last_12m'].fillna(0) / (out['credit_file_age_yrs'].fillna(0) + 1)
    for c in ['annual_inc', 'revol_bal', 'tot_cur_bal', 'loan_amnt']:
        out[f'log1p_{c}'] = np.log1p(out[c].clip(lower=0))
    out['state_median_income'] = out['addr_state'].map(STATE_MEDIAN_INCOME).fillna(70.0)
    out['state_unemployment'] = out['addr_state'].map(STATE_UNEMPLOYMENT).fillna(4.5)
    out['inc_vs_state_median'] = (out['annual_inc'] / 1000.0) / out['state_median_income']
    out['zip3'] = out['zip_code'].astype('string').str.extract(r'(\d{3})', expand=False)
    if 'emp_title' in out.columns:
        out['emp_title'] = out['emp_title'].astype('string').str.lower().str.strip()
    drop = ['fico_range_low', 'fico_range_high', 'term', 'emp_length', 'title', 'zip_code']
    return out.drop(columns=[c for c in drop if c in out.columns])


train_fe = engineer(train_raw)
test_fe = engineer(test_raw)
top_titles = train_fe['emp_title'].value_counts().head(100).index.tolist()
train_fe['emp_title'] = train_fe['emp_title'].where(train_fe['emp_title'].isin(top_titles), 'Other').fillna('Missing').astype(str)
test_fe['emp_title'] = test_fe['emp_title'].where(test_fe['emp_title'].isin(top_titles), 'Other').fillna('Missing').astype(str)
test_fe = test_fe.drop(columns=['ID'])
y = train_fe['int_rate'].values.astype(np.float64)
X = train_fe.drop(columns=['int_rate'])
print(f'[FE] X={X.shape}  X_test={test_fe.shape}')

# ----- SKIP cell 5 (aux grade classifier) -----
# Adds 11 columns we won't have, but cell 8 doesn't depend on them.

# ----- cell 6: split + helpers -----
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE)
X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
print(f'[split] X_train={X_train.shape}  X_val={X_val.shape}')

NATIVE_CAT = ['application_type', 'home_ownership', 'verification_status', 'purpose']
TE_CAT = ['addr_state', 'purpose', 'zip3', 'emp_title']
ALL_CAT = list(set(NATIVE_CAT + TE_CAT))
TARGET_ENC_M = 20.0


def smoothed_te(train_col, target, val_col=None, test_col=None, folds=5, seed=RANDOM_STATE):
    global_mean = float(np.nanmean(target))
    val_enc = test_enc = None
    if val_col is not None or test_col is not None:
        stats = pd.DataFrame({'cat': train_col.values, 'y': target}).groupby('cat')['y'].agg(['mean', 'count'])
        stats['enc'] = (stats['mean']*stats['count'] + global_mean*TARGET_ENC_M) / (stats['count']+TARGET_ENC_M)
        mapping = stats['enc'].to_dict()
        val_enc = val_col.map(mapping).fillna(global_mean).values if val_col is not None else None
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
            Xte[c].astype(str),
        )
        Xtr[f'{c}_te'] = tr_e
        Xva[f'{c}_te'] = va_e
        Xte[f'{c}_te'] = te_e
    return Xtr, Xva, Xte


def prep_hgb(df, cat_cats=None, max_cat_cardinality=60):
    df = df.copy()
    cat_cols_all = [c for c in ALL_CAT if c in df.columns]
    cat_cols = []
    drop_high_card = []
    for c in cat_cols_all:
        nuniq = df[c].astype('string').fillna('Missing').nunique()
        if nuniq <= max_cat_cardinality:
            cat_cols.append(c)
        else:
            drop_high_card.append(c)
    if drop_high_card:
        df = df.drop(columns=drop_high_card)
    out_cats = {}
    for c in cat_cols:
        s = df[c].astype('string').fillna('Missing')
        if cat_cats is None or c not in cat_cats:
            uniques = s.unique()
            mapping = {v: i for i, v in enumerate(uniques)}
        else:
            mapping = cat_cats[c]
            unknown = len(mapping)
            mapping = dict(mapping)
            for v in s.unique():
                if v not in mapping:
                    mapping[v] = unknown
        df[c] = s.map(mapping).astype('int64')
        out_cats[c] = mapping
    for c in df.columns:
        if c not in cat_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    return df, out_cats, cat_cols


# ----- cell 8 verbatim (with N_TRIALS_HGB=2) -----
print('\n[Optuna] Building inner 85/15 split for HGB tuning...')

inner_tr, inner_va = train_test_split(
    np.arange(len(X_train)), test_size=0.15, random_state=RANDOM_STATE,
)
Xtr_inner = X_train.iloc[inner_tr].reset_index(drop=True)
Xva_inner = X_train.iloc[inner_va].reset_index(drop=True)
ytr_inner = y_train[inner_tr]
yva_inner = y_train[inner_va]

Xtr_inner_te, Xva_inner_te, _ = add_te(Xtr_inner, ytr_inner, Xva_inner, X_train.iloc[:0])

Xtr_hgb_inner, cat_cats_inner, hgb_cat_cols = prep_hgb(Xtr_inner_te)
Xva_hgb_inner, _, _ = prep_hgb(Xva_inner_te, cat_cats=cat_cats_inner)
imp = SimpleImputer(strategy='median')
Xtr_hgb_inner_imp = imp.fit_transform(Xtr_hgb_inner)
Xva_hgb_inner_imp = imp.transform(Xva_hgb_inner)
hgb_cat_idx_inner = [Xtr_hgb_inner.columns.get_loc(c) for c in hgb_cat_cols]

print(f'  Xtr_hgb_inner_imp shape  : {Xtr_hgb_inner_imp.shape}')
print(f'  Xva_hgb_inner_imp shape  : {Xva_hgb_inner_imp.shape}')
print(f'  hgb_cat_cols             : {hgb_cat_cols}')
print(f'  hgb_cat_idx_inner        : {hgb_cat_idx_inner}')
for c in hgb_cat_cols:
    nuniq = Xtr_hgb_inner[c].nunique()
    cmax = Xtr_hgb_inner[c].max()
    print(f'    {c:25s}: n_unique={nuniq:4d}  max_value={cmax}')


def hgb_objective(trial):
    params = {
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.10, log=True),
        'max_iter':         trial.suggest_int('max_iter', 300, 1500),
        'max_leaf_nodes':   trial.suggest_int('max_leaf_nodes', 15, 127),
        'max_depth':        trial.suggest_int('max_depth', 4, 12),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 200),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 5.0),
        'max_features':     trial.suggest_float('max_features', 0.4, 1.0),
        'max_bins':         trial.suggest_int('max_bins', 64, 255),
        'random_state': RANDOM_STATE,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 50,
    }
    m = HistGradientBoostingRegressor(**params, categorical_features=hgb_cat_idx_inner)
    m.fit(Xtr_hgb_inner_imp, ytr_inner)
    pred = m.predict(Xva_hgb_inner_imp)
    return float(np.sqrt(mean_squared_error(yva_inner, pred)))


t = time()
print(f'\n[Optuna] Running TPE search for HGB ({N_TRIALS_HGB} trials)...')
study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=RANDOM_STATE, multivariate=True, warn_independent_sampling=False),
    pruner=MedianPruner(n_warmup_steps=10),
    study_name='hgb_bayesian_search',
)
study.optimize(hgb_objective, n_trials=N_TRIALS_HGB, show_progress_bar=False)

HGB_PARAMS = study.best_params
print(f'\n[HGB] Optuna done in {time()-t:.0f}s')
print(f'[HGB] Best inner-val RMSE: {study.best_value:.4f}')
print(f'[HGB] Best params:')
for k, v in HGB_PARAMS.items():
    print(f'    {k}: {v}')

expected = {'learning_rate', 'max_iter', 'max_leaf_nodes', 'max_depth',
            'min_samples_leaf', 'l2_regularization', 'max_features', 'max_bins'}
assert set(HGB_PARAMS) == expected, f'HGB_PARAMS keys differ: {set(HGB_PARAMS) ^ expected}'
print('\n[OK] Cell 8 smoke test PASSED')
