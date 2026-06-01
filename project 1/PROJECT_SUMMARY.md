# LendingClub Interest Rate Prediction — Project Summary

Predict a loan's `int_rate` from **application-time features only**, using the
assignment-provided dataset in `true data/`. This document describes the data,
the exploratory analysis, and the performance of the final CatBoost model.

---

## 1. Deliverables in this repo

After cleanup, the repo contains only the final model, the EDA, and the data:

```text
project 1/
├── final/
│   ├── final_catboost.ipynb              # FINAL MODEL — end-to-end CatBoost notebook
│   ├── final_pipeline_catboost.py        # CatBoost-only training pipeline (script form)
│   ├── final_pipeline.py                 # shared load + feature-engineering logic
│   └── build_final_catboost_notebook.py  # regenerates final_catboost.ipynb
├── notebooks/
│   ├── 01_eda.ipynb                      # full exploratory data analysis
│   └── _build_eda_notebook.py            # regenerates 01_eda.ipynb
├── true data/                            # LC_train.csv (100k), LC_test.csv (10k)
├── outputs/                              # generated figures + model artifacts
├── EDA_REPORT.md                         # detailed written EDA report
├── PROJECT_SUMMARY.md                    # this file
├── CLAUDE.md                             # data-loading + modeling conventions
└── archive/                             # all superseded work (prior models, decks, teammate folders)
```

Everything that was not part of the final CatBoost pipeline or the EDA was moved
to `archive/` (prior ensemble/min-RMSE notebooks, presentation decks, teammate
folders, old pipelines, research docs). Nothing was deleted.

---

## 2. The data

| | Train | Test |
| --- | --- | --- |
| Rows × cols | 100,000 × 39 | 10,000 × 39 |
| `int_rate` (target) | present | absent (to predict) |
| `ID` | absent | present (1–10,000) |
| `loan_status` | present → **dropped (leakage)** | present → dropped |
| Date columns | **none** | none |

Target `int_rate` ranges **6.46 – 30.99 %** (mean 12.95, median 11.71, std 5.26,
right-skewed, skew 0.85).

Key structural facts that shaped modeling:

- **`loan_status` is post-origination leakage** (loan lifecycle outcome) and is
  dropped immediately at load.
- The classic leakage trio (`grade`, `sub_grade`, `installment`) is **absent**
  from this slice.
- **FICO is present** (`fico_range_low`, `fico_range_high`) — the dominant signal.
- **No date columns** (`issue_d`, `earliest_cr_line` absent) → temporal
  validation is impossible, so a **random 80/20 split** is used.
- Eight numeric columns arrive as strings with literal `"NA"`; cast to float at load.

---

## 3. EDA — what the notebooks found

Full analysis: `notebooks/01_eda.ipynb`; written report: `EDA_REPORT.md`;
figures: `outputs/eda/`.

**Target.** Right-skewed (skew 0.85); no transform needed for tree models.

**Missingness.** Only 10 columns have nulls. `mths_since_last_record` is 90.9%
missing and **informatively missing** ("no derogatory record ever") — handled
with a binary `has_*` flag rather than median imputation. Same logic for the
other `mths_since_*` columns.

**FICO (the headline signal).** `fico_range_high − low` is always 4–5, so the two
columns are one signal — collapsed to the midpoint `fico = (low + high) / 2`.
Pearson **−0.43**, Spearman **−0.46** with `int_rate` (higher FICO → lower rate).
Population is pre-screened (no sub-660 applicants).

**Other predictors (after FICO).** Strongest remaining signals are utilization
(`all_util` +0.33, `revol_util` +0.30) and `dti` (+0.18); then credit-shopping
intensity (`inq_fi`, `inq_last_12m`). `term` is the cleanest categorical
separator (60-month loans price higher); `purpose` separates well
(small-business / educational price above debt-consolidation / credit-card).

**Multicollinearity.** Only real issue is the FICO pair (handled via midpoint);
`revol_util ↔ all_util` correlate ~0.7–0.8 but trees handle the redundancy.

**Leakage audit.** `loan_status` shows a 3.42 pp mean-rate spread
(ANOVA F=88.7, eta²=0.0044 → 0.44% of variance). Small in magnitude but
principled to exclude; dropped unconditionally before any modeling.

**Cardinality / drops.** `emp_title` (35k unique free text) and `title`
(one-to-one duplicate of `purpose`) are dropped. `addr_state`, `purpose`,
`home_ownership`, `verification_status`, `term`, `application_type` kept as
categoricals.

**Train vs test drift.** Test set is a more conservative subset: `loan_amnt`
shifts **down** (~$16.8k → $13.8k, PSI 0.17) and FICO shifts **up** ~5 points.
Worth monitoring — a naïve model may over-estimate rates on the test set.

---

## 4. Feature engineering

Built statelessly (no leakage) in `final_pipeline.engineer`, applied identically
to train and test:

- `fico` midpoint (drop raw pair); `fico_band`, `dti_band`, utilization bands.
- `term_months` parsed from `term`; ordinal `emp_length_num`.
- `log1p` of monetary tails (`annual_inc`, `revol_bal`, `tot_cur_bal`,
  `total_bal_ex_mort`, `tot_coll_amt`).
- Ratios: `loan_to_income`, `bal_to_income`, `acc_open_ratio`, `inq_intensity`.
- Interactions: `util_x_fico`, `dti_x_fico`, `fico_x_term`, `fico_x_dti_band`,
  `term_x_dti`.
- `derog_score` (weighted public-record / delinquency composite).
- `has_*` missingness flags for the `mths_since_*` columns.

High-cardinality categoricals use **smoothed K-fold target encoding** refit
**inside each fold** so no target information leaks across folds.

---

## 5. Final model — CatBoost

Implemented in `final/final_catboost.ipynb` (script: `final_pipeline_catboost.py`).

**Training scheme**
- 80/20 random holdout for honest validation.
- 5-fold `KFold` (shuffle, seed 6604); per-fold target encoding refit; OOF
  predictions for an honest CV estimate; fold-averaged predictions on the
  holdout and test set.
- `CatBoostRegressor`, RMSE loss, 2,500 iterations, early stopping 100.

**Hyperparameters** (cached from an Optuna search):

| param | value |
| --- | --- |
| learning_rate | 0.03 |
| depth | 7 |
| l2_leaf_reg | 5.0 |
| random_strength | 1.0 |
| bagging_temperature | 0.5 |
| border_count | 200 |

Test predictions clipped to the observed target range `[6.0, 31.0]`.

### Results (20% held-out validation)

| Metric | Value |
| --- | --- |
| **OOF RMSE** | **3.863** |
| **Validation RMSE** | **3.872** |
| Validation MAE | 2.902 |
| Validation R² | 0.465 |

OOF and held-out RMSE agree closely (3.863 vs 3.872), indicating the CV estimate
is honest and the model is not over-fit. The R² of 0.465 reflects the dataset's
real ceiling: with the pre-origination grade/sub-grade columns absent, FICO plus
utilization and term carry most of the recoverable signal.

### Most important features

By both CatBoost gain and permutation importance, the model leans on the same
drivers the EDA flagged:

1. `fico` / `fico_band` (credit quality — the dominant signal)
2. `term_x_dti`, `loan_amnt`, `term_months` (loan structure)
3. `purpose`, `application_type` (loan-type categoricals)
4. `mths_since_rcnt_il`, `all_util`, `revol_bal` (utilization / credit history)
5. `fico_x_term` interaction

Explainability artifacts (SHAP summary/waterfalls, permutation importance, PDP/ICE,
LIME, residual and learning-curve diagnostics) are in `outputs/final/catboost_*`.

---

## 6. Reproducing

```bash
# EDA
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute \
  notebooks/01_eda.ipynb --output 01_eda.ipynb

# Final CatBoost model (script)
.venv/Scripts/python.exe final/final_pipeline_catboost.py

# Rebuild the final notebook
.venv/Scripts/python.exe final/build_final_catboost_notebook.py
```

Data-loading and modeling conventions (string-numeric casts, leakage drop, etc.)
are documented in `CLAUDE.md`.
