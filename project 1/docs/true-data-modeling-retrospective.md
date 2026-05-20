# True-Data Modeling Retrospective

This note summarizes the modeling work performed after switching the project to the assignment-provided files in `true data/`. It is the current source of truth for what was tried, what worked, what did not, and what could realistically improve the model.

## Current Dataset Context

- Train file: `true data/LC_train.csv`
  - 100,000 rows, 39 columns
  - Target: `int_rate`
- Test file: `true data/LC_test.csv`
  - 10,000 rows, 39 columns
  - Includes `ID`
  - Does not include `int_rate`, so local test RMSE/R2 cannot be computed
- Data dictionary: `true data/LCDataDictionary.xlsx`
- Validation policy: random folds, because the true-data slice has no date fields such as `issue_d` or `earliest_cr_line`.

Important leakage rules:

- Drop `loan_status`: it is post-origination loan outcome/status.
- Do not use `grade`, `sub_grade`, or `installment`; they are absent here and should not be reintroduced.
- Drop `title`: in this slice it is a one-to-one human-readable duplicate of `purpose`.
- Drop `emp_title`: high-cardinality free text.

## Results Summary

| Attempt | Feature set | Tuning | Final model | OOF RMSE | OOF MAE | OOF R2 | Verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| v1 | EDA-approved transformations + valid domain interactions | 100 Optuna trials | 5-fold XGBoost ensemble | **3.9062** | **2.9239** | **0.4476** | Best current honest run |
| v2 | v1 + FICO-heavy nonlinear terms + economics proxies + fold-local target encodings | 100 Optuna trials | 5-fold XGBoost ensemble | 3.9442 | 2.9518 | 0.4368 | More complex, but worse |

The current generated `outputs/modeling_report.md` reflects the latest v2 run. The best observed modeling result so far is the v1 run, not v2.

## What Was Tried

### 1. True-data EDA

The notebook `notebooks/01_eda.ipynb` was rebuilt around `true data/`.

Key EDA findings used for modeling:

- FICO columns exist and are the strongest non-leakage signal.
- `loan_status` exists but is leakage and must be dropped.
- No temporal columns exist, so temporal validation is impossible.
- `loan_amnt` and `mths_since_recent_bc` have the largest train/test distribution drift.
- `term`, `purpose`, `application_type`, `addr_state`, and `zip_code` have useful pricing signal.
- Several numeric columns arrive as string-like values with `"NA"` and need explicit null handling.

### 2. v1 XGBoost Pipeline

Implemented an active modeling script at `src/preprocessing_and_modeling.py`.

The v1 feature system included:

- `fico = (fico_range_low + fico_range_high) / 2`
- `term_months`
- `emp_length_num`
- `log1p` monetary transforms
- missingness flags for selected `mths_since_*` columns
- zero flags for delinquency/public-record/count fields
- valid interaction terms from the research docs:
  - `loan_to_income`
  - `dti_x_log_income`
  - `fico_x_term`
  - `fico_x_revol_util`
  - `fico_x_all_util`
  - `inq_per_open_acc`
  - `delinq_per_credit_age`
  - `pub_rec_bankruptcy_combo`
  - `revol_util_x_open_acc`
  - `loan_amnt_x_term`
  - `loan_amnt_x_revol_util`

Training design:

- 100-trial Optuna TPE search.
- 5-fold XGBoost ensemble.
- Fold-local median imputation and categorical handling.
- Native XGBoost categorical support.

Result:

- OOF RMSE: `3.9062`
- OOF MAE: `2.9239`
- OOF R2: `0.4476`

### 3. v2 FICO/Economic/Target-Encoding Expansion

The user asked to use FICO more aggressively and add economic-theory-inspired interactions.

Added:

- FICO bands, centered FICO, squared FICO, inverse FICO.
- FICO shortfall thresholds at 680, 700, 720, and 740.
- FICO crosses with term, DTI, utilization, inquiries, public records, and credit age.
- Categorical crosses such as purpose x term, purpose x FICO band, ZIP x term, state x term.
- Economics-inspired proxy variables:
  - `capacity_stress`
  - `liquidity_stress`
  - `leverage_stress`
  - `term_premium_proxy`
  - `adverse_selection_proxy`
  - `credit_rationing_score`
  - home/collateral stability flags
- Fold-local smoothed target encodings for categorical columns and categorical crosses.

Result:

- OOF RMSE: `3.9442`
- OOF MAE: `2.9518`
- OOF R2: `0.4368`

Interpretation:

- v2 made the model worse by about `0.038` RMSE.
- XGBoost was already learning many nonlinear relationships from the v1 features.
- The extra target encodings/crosses likely added variance/noise.
- More complex features did not translate into better generalization.

## What The Model Is Learning

For v2, the top gain features were dominated by:

- `term_cat`
- `term_months`
- `purpose_fico_band_target_mean`
- `term_x_fico_shortfall_720`
- `purpose_fico_band`
- `fico_shortfall_720_x_dti`
- `purpose_term`
- `zip_code`
- `application_type`
- `loan_amnt`

This confirms that the richer feature set was being used, but use does not equal improvement. The OOF metrics worsened, so these features should be treated as candidates for ablation rather than automatic keepers.

## Why RMSE 1.0 / R2 > 0.7 Is Unlikely

The true-data target standard deviation is about `5.26`. An RMSE of `1.0` would imply:

```text
R2 ~= 1 - (1.0^2 / 5.26^2) ~= 0.96
```

That level is not realistic for an honest model on this 39-column schema unless one of the following is true:

- leakage is introduced,
- hidden test labels are available,
- LendingClub's internal grade/sub-grade/pricing score is included,
- the target is nearly deterministic from fields not currently available.

Even R2 `0.70` would require RMSE around `2.88`, which is far below the best honest observed result so far (`3.906` OOF RMSE).

## What Can Improve The Model

### Highest Priority: Ablation Study

Run controlled comparisons and keep only feature blocks that improve OOF RMSE.

Recommended blocks to test:

1. v1 base features only.
2. v1 + FICO nonlinear terms.
3. v1 + economics proxies.
4. v1 + categorical crosses, no target encoding.
5. v1 + low-cardinality target encodings only.
6. v1 + ZIP/state target encodings.
7. v1 + all v2 features.

Expected outcome:

- v1 likely remains strongest or near strongest.
- Some FICO nonlinear terms may help slightly.
- High-cardinality target encodings and crossed categoricals are likely the first things to remove.

### Stronger But Still Honest Modeling Options

1. **CatBoost comparison**
   - CatBoost often handles categorical borrower data better than XGBoost.
   - This would require adding `catboost` to `requirements.txt`.
   - Compare OOF RMSE/R2 against v1 XGBoost.

2. **LightGBM comparison**
   - Fast, strong tabular baseline.
   - Useful as a sanity check against XGBoost.
   - Requires adding `lightgbm`.

3. **Repeated K-fold or seed ensemble**
   - Train v1 across multiple random seeds and average predictions.
   - May reduce prediction variance but probably will not move RMSE dramatically.

4. **Stratified validation diagnostics**
   - Report RMSE by term, FICO band, purpose, and loan amount band.
   - Helps identify where the model fails and where targeted features may help.

5. **Residual model**
   - Train v1 XGBoost.
   - Train a second model on residuals using only features that explain systematic misses.
   - Keep only if OOF residual improvement is real.

### Data Improvements That Would Help Most

The current schema is the biggest bottleneck. Better features would likely help more than more tuning.

Potentially valuable data additions:

- more credit-history recency fields,
- more utilization/account composition fields,
- actual origination date,
- macro variables at origination date,
- state-level unemployment or home-price indices,
- LendingClub internal risk grade only if the assignment permits it, though it is likely target leakage for rate prediction.

### Things That Are Unlikely To Help

- More Optuna trials alone.
- More tree depth alone.
- More hand-built interactions without ablation.
- Neural networks/LSTMs on this static tabular schema.
- Training metrics as a success metric; they can be made high by overfitting and do not measure generalization.

## Recommended Next Action

Create an ablation mode in `src/preprocessing_and_modeling.py` that can run feature blocks independently and write a table like:

| Feature block | OOF RMSE | OOF MAE | OOF R2 | Delta vs v1 |
| --- | ---: | ---: | ---: | ---: |
| v1_base | TBD | TBD | TBD | 0.000 |
| v1_plus_fico_nonlinear | TBD | TBD | TBD | TBD |
| v1_plus_econ_proxies | TBD | TBD | TBD | TBD |
| v1_plus_target_encoding | TBD | TBD | TBD | TBD |
| v2_all | 3.9442 | 2.9518 | 0.4368 | +0.0380 RMSE |

Use fewer Optuna trials for ablation, such as `OPTUNA_TRIALS=15`, then rerun the best candidate with 100 trials.

## Files Of Record

- `notebooks/01_eda.ipynb`: true-data EDA.
- `EDA_REPORT.md`: EDA report.
- `src/preprocessing_and_modeling.py`: active modeling pipeline.
- `outputs/modeling_report.md`: latest generated modeling report.
- `outputs/test_predictions.csv`: latest test predictions.
- `outputs/oof_predictions.csv`: latest out-of-fold predictions.
- `outputs/feature_importance.csv`: latest feature importance.
- `docs/interaction-terms-and-economic-theory.md`: source for economic interaction ideas.
- `docs/loan-interest-rate-prediction-research.md`: supplementary ML research notes.

