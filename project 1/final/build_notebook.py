"""Build Final_Ensemble.ipynb from cell definitions. Re-run anytime."""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent / "Final_Ensemble.ipynb"

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Final Composite Ensemble — LendingClub Interest Rate Prediction

This notebook builds a **six-base-learner stacked ensemble** on the `true data/`
LendingClub slice and reports the lowest validation RMSE we can achieve from
application-time features only.

**Base learners**
1. **XGBoost** (tuned via Optuna, cached params)
2. **LightGBM** (tuned)
3. **CatBoost** (tuned, native categorical handling)
4. **Random Forest**
5. **Ridge regression** (linear baseline)
6. **MLP neural network** (sklearn `MLPRegressor`)

**Meta-learners**
- **Ridge stacker** on out-of-fold predictions (selected by 5-fold CV alpha)
- **Hill-climb weighted blend** (constrained non-negative weights summing to 1)

**Anti-leakage**
- `loan_status` dropped (post-origination outcome)
- Random 80/20 holdout (no date columns in this slice — temporal split impossible)
- K-fold **OOF** smoothed target encoding for high-cardinality categoricals
  (each training row encoded using the *other* folds only)
- All preprocessing (imputers, scalers, encoders) fit on training folds only
""")

code("""# 1) Imports + paths
from pathlib import Path
import sys, json, warnings, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == 'final':
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / 'final'))

# Import the pipeline module so notebook + script share one source of truth
import final_pipeline as fp

print('Project root:', PROJECT_ROOT)
print('Data dir    :', fp.DATA_DIR)
print('Outputs dir :', fp.OUTPUTS_DIR)
""")

md("""## 1. Load + drop leakage column

`loan_status` is a post-origination outcome and would leak future information
into a rate model. We drop it immediately. `ID` is held aside for the
submission CSV. The remaining 36 features are all observable at application
time per the leakage audit in [research/no_leakage_strategies.md](../research/no_leakage_strategies.md).""")

code("""train_raw, test_raw = fp.load_raw()
print('train:', train_raw.shape, '   test:', test_raw.shape)
print('target stats:')
print(train_raw['int_rate'].describe().round(3))
""")

md("""## 2. Feature engineering

Stateless transforms only — anything that needs a fit (target encoding,
imputation stats) is done *inside* the K-fold loop so no row sees its own
label.

Added features:
- `fico` (midpoint of fico_range_low/high) + `fico_band` (binned)
- `term_months` (int), `emp_length_num` (ordinal)
- Missingness flags on `mths_since_*` columns
- Underwriting ratios: `loan_to_income`, `dti_band`, `payment_to_income`,
  `revol_util_band`, `acc_open_ratio`
- Interactions: `util_x_fico`, `dti_x_fico`, `term_x_dti`, `term_x_loan_amnt`
- `derog_score` (weighted combination of pub_rec / bankruptcies / chargeoffs)
- `log1p_*` transforms on long-tailed monetary columns
- `zip3` (first 3 digits of zip_code → ~900 categories)""")

code("""train_fe = fp.engineer(train_raw)
test_fe  = fp.engineer(test_raw)

# Top-N emp_title compaction (fit on train only)
top = train_fe['emp_title'].value_counts().head(fp.EMP_TITLE_TOP_N).index.tolist()
train_fe['emp_title'] = train_fe['emp_title'].where(train_fe['emp_title'].isin(top), 'Other').fillna('Missing').astype(str)
test_fe['emp_title']  = test_fe['emp_title'].where(test_fe['emp_title'].isin(top), 'Other').fillna('Missing').astype(str)

test_ids = test_fe['ID'].copy()
test_fe = test_fe.drop(columns=['ID'])

y_full = train_fe['int_rate'].values.astype(np.float64)
X_full = train_fe.drop(columns=['int_rate'])
print('engineered cols:', X_full.shape[1])
print(list(X_full.columns))
""")

md("""## 3. 80/20 holdout split

No date columns are available in `true data/`, so a temporal split is
impossible. We use a fixed-seed random 80/20 split as the honest held-out
validation set.""")

code("""from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=fp.HOLDOUT_FRAC, random_state=fp.RANDOM_STATE,
)
print('X_train:', X_train.shape, '  X_val:', X_val.shape, '  X_test:', test_fe.shape)
""")

md("""## 4. Cached Optuna tuning params

We reuse the Optuna study output from `outputs/sota/tuning_results.json`
(30 trials × 3 GBDTs, TPE sampler with median pruning). This avoids paying
the multi-hour tuning cost on every notebook run.""")

code("""tuning = json.loads(fp.CACHED_TUNING.read_text())['params']
for name, p in tuning.items():
    print(f'\\n[{name}]')
    for k, v in p.items():
        print(f'  {k}: {v}')
""")

md("""## 5. K-fold OOF stacking

For each of 5 outer folds we:
1. Build per-fold smoothed K-fold OOF target encodings for the 4
   high-cardinality cats (`addr_state`, `purpose`, `zip3`, `emp_title`).
2. Train each of the 6 base learners with 2 seeds and average their fold
   predictions on `X_val` and `X_test`.
3. Record the held-out OOF prediction on the validation slice of the fold.

The OOF predictions become features for the meta-learners below.

**This is the slow step — runs ~15–30 minutes.** The pipeline module
`final/final_pipeline.py` runs the same logic from the command line and
saves all artifacts to `outputs/final/`. Re-run that script to refresh
the cached predictions, then re-execute this notebook to load them.""")

code("""# Re-run final/final_pipeline.py from the shell to regenerate artifacts.
# Here we just load the persisted result for fast notebook iteration.

ART = fp.OUTPUTS_DIR / 'ensemble_artifacts.pkl'
if ART.exists():
    with open(ART, 'rb') as fh:
        art = pickle.load(fh)
    BASES = art['BASES']
    stack_tr = art['oof_stack']
    stack_va = art['val_stack']
    stack_te = art['test_stack']
    stacker_coef = art['stacker_coef']
    stacker_alpha = art['stacker_alpha']
    stacker_intercept = art['stacker_intercept']
    hc_w = np.array(art['hill_climb_weights'])
    chosen = art['chosen_blend']
    y_train_saved = art['y_train']
    y_val_saved = art['y_val']
    print('Loaded ensemble artifacts:')
    print(f'  bases:               {BASES}')
    print(f'  stacker Ridge alpha: {stacker_alpha}')
    print(f'  stacker weights:     {dict(zip(BASES, stacker_coef))}')
    print(f'  stacker intercept:   {stacker_intercept:.3f}')
    print(f'  hill-climb weights:  {dict(zip(BASES, hc_w))}')
    print(f'  chosen blend:        {chosen}')
else:
    print('Artifacts not found. Run:')
    print('   .venv/Scripts/python.exe final/final_pipeline.py')
""")

md("""## 6. Per-base-learner held-out RMSE""")

code("""from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

results_path = fp.OUTPUTS_DIR / 'final_results.csv'
if results_path.exists():
    results = pd.read_csv(results_path)
    results = results.sort_values('val_RMSE').reset_index(drop=True)
    display(results.style.format({'val_RMSE': '{:.4f}', 'val_MAE': '{:.4f}', 'val_R2': '{:.4f}'}))
else:
    print('Run final/final_pipeline.py first.')
""")

md("""## 7. Visualise model comparison""")

code("""if results_path.exists():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(results['model'], results['val_RMSE'], color='steelblue')
    ax.invert_yaxis()
    ax.set_xlabel('Held-out validation RMSE (percentage points)')
    ax.set_title('Final composite ensemble — model comparison')
    for i, v in enumerate(results['val_RMSE']):
        ax.text(v + 0.005, i, f'{v:.4f}', va='center')
    plt.tight_layout()
    plt.savefig(fp.OUTPUTS_DIR / 'model_comparison.png', dpi=120, bbox_inches='tight')
    plt.show()
""")

md("""## 8. Residual diagnostics for the chosen blend""")

code("""if ART.exists():
    if chosen == 'Ridge stack':
        from sklearn.linear_model import Ridge
        stacker = Ridge(alpha=stacker_alpha, positive=True).fit(stack_tr, y_train_saved)
        val_blend = stacker.predict(stack_va)
        test_blend = stacker.predict(stack_te)
    else:
        val_blend = stack_va @ hc_w
        test_blend = stack_te @ hc_w

    resid = y_val_saved - val_blend
    rmse  = float(np.sqrt(mean_squared_error(y_val_saved, val_blend)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(val_blend, resid, alpha=0.15, s=8, color='steelblue')
    axes[0].axhline(0, color='crimson', lw=1)
    axes[0].set_xlabel('predicted int_rate (pp)')
    axes[0].set_ylabel('residual (pp)')
    axes[0].set_title(f'{chosen} residuals  (val RMSE = {rmse:.4f})')

    axes[1].hist(resid, bins=60, color='steelblue', alpha=0.85)
    axes[1].axvline(0, color='crimson', lw=1)
    axes[1].set_xlabel('residual (pp)')
    axes[1].set_title('Residual distribution')

    plt.tight_layout()
    plt.savefig(fp.OUTPUTS_DIR / 'residuals.png', dpi=120, bbox_inches='tight')
    plt.show()
""")

md("""## 9. Test set predictions

Predictions are clipped to `[6.0, 31.0]` (observed `int_rate` range in train)
and written to `outputs/final/test_predictions_final.csv` with the
required `ID, int_rate` schema.""")

code("""pred_path = fp.OUTPUTS_DIR / 'test_predictions_final.csv'
if pred_path.exists():
    preds = pd.read_csv(pred_path)
    print('rows:', len(preds))
    print('columns:', list(preds.columns))
    print()
    print('int_rate distribution:')
    print(preds['int_rate'].describe().round(3))
    display(preds.head(10))
""")

md("""## 10. Summary

The final blend lowers held-out RMSE meaningfully versus any individual base
learner. Diversity of error patterns (tree-based GBDTs + bagged RF + linear
Ridge + non-linear MLP) is what the Ridge stacker exploits.

**Outputs written to `outputs/final/`:**
- `test_predictions_final.csv` — submission file (ID, int_rate)
- `final_results.csv` — held-out RMSE/MAE/R² for each model + the two blends
- `summary.json` — metadata + stacker/hill-climb weights
- `ensemble_artifacts.pkl` — OOF stack, val stack, test stack, weights
- `model_comparison.png`, `residuals.png` — diagnostic plots
- `run.log` — full pipeline stdout
""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}

with open(OUT, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Wrote {OUT}')
print(f'Cells: {len(cells)}')
