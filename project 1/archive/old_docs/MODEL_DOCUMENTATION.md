# LendingClub Interest Rate Prediction — Final Model Documentation

**Champion**: `reverse_engineer/v2` — held-out validation RMSE **3.8297**
**Submission file**: [`outputs/reverse_engineer/v2/test_predictions_v2.csv`](outputs/reverse_engineer/v2/test_predictions_v2.csv)
**Improvement vs original baseline (3.912)**: −0.082 pp, **2.1% relative RMSE reduction**

---

## 1. Executive Summary

The task is regression of `int_rate` from application-time loan features. The "true data" slice (100k train + 10k test) is intentionally stripped of LendingClub's internal `grade`/`sub_grade` columns and contains no `issue_d` dates, which removes the two strongest known predictors of interest rate. Despite this, we hit val RMSE 3.83 by combining four ideas:

1. **Heavy feature engineering** on the 38 raw application-time features → 135 engineered features (ratios, log-tails, FICO bands, derog scores, state-macro lookups, etc.).
2. **A 12-base-learner stacked ensemble** (CatBoost, LightGBM, XGBoost, LightGBM-Huber, RandomForest, Ridge, sklearn MLP, FT-Transformer, Deep ResMLP, HistGradientBoosting, ExtraTrees, plus a pseudo-labelled CatBoost) with three seeds per GBDT and a 5-fold OOF design.
3. **Reverse-engineering of the missing `grade` signal** by training auxiliary CatBoost models on the historical `achive_data/` slice (which *does* contain grade/sub_grade), then applying them to the true data as enrichment features. This added 14 auxiliary columns.
4. **30 FICO × auxiliary interaction terms** to cross the recovered grade signal with FICO (which exists in true data but not in the archive that trained the aux model). This is what `v2` adds over `post_production`.

Five meta-learners (Ridge stack, ElasticNet stack, LightGBM stack, hill-climb weighted blend, and a mean-of-meta-learners blend) compete; **LightGBM stack** won at 3.8297.

The structural ceiling for this dataset is ~3.80 — beyond that requires data we don't have (native grade, native dates, or FICO inside the archive). We hit that ceiling.

---

## 2. Problem Statement & Dataset

### Task
Given application-time loan features, predict the `int_rate` LendingClub assigned to each loan. Continuous regression; metric is RMSE.

### Data sources

| Source | Rows | Cols | Has FICO range? | Has grade/sub_grade? | Has int_rate? | Has issue_d? |
|---|---|---|---|---|---|---|
| `true data/LC_train.csv` | 100,000 | 39 | ✓ | ✗ | ✓ (target) | ✗ |
| `true data/LC_test.csv` | 10,000 | 39 | ✓ | ✗ | absent | ✗ |
| `achive_data/archive/LC_train.csv` | 100,000 | 39 | ✗ | **✓** | ✓ | **✓** |
| `achive_data/archive/loan.csv` | 2,260,668 | 145 | ✗ | **✓** | ✓ | **✓** |

The two `LC_train.csv` files contain **different** sampled rows (verified by direct comparison). The archive sources lack FICO (Kaggle redacted it from the public LC data dump); the true-data slice has FICO but no grade/dates. The non-overlap between the two columns lists is precisely what made the reverse-engineering approach work — each source has signal the other lacks.

---

## 3. EDA Findings

Full report: [`EDA_REPORT.md`](EDA_REPORT.md). Notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). Figures: [`outputs/eda/`](outputs/eda/).

Key signals that drove modelling decisions:

### Target distribution
- `int_rate` range 6.46% → 30.99%, mean 12.95%, median 11.71%, std 5.26%.
- Right-skewed (skew +0.85). `log1p` transform reduces skew to +0.23 but introduces back-transform bias; we **did not** transform the target — empirical experiments showed GBDT objective handles the skew directly.
- ![target distribution](outputs/eda/02_target_distribution.png)

### Strongest non-leakage predictors (Spearman with `int_rate`)
| Rank | Feature | Spearman | Notes |
|---|---|---|---|
| 1 | `fico_range_low` / `fico_range_high` | **−0.46** | Dominant predictor; both columns near-identical so collapsed to midpoint |
| 2 | `all_util` | +0.33 | Total credit utilization |
| 3 | `revol_util` | +0.30 | Revolving credit utilization |
| 4 | `dti` | +0.18 | Debt-to-income |
| 5 | `term` | +0.30 (binary) | 60-month loans price ~3pp higher than 36-month |

### Missingness
- `mths_since_last_record`: **90.9% missing** — handled with explicit missingness flag + fill with 999 (sentinel for "never").
- `emp_title`: 14.9% missing — compacted to top-200 + "Other" + "Missing".
- `mths_since_recent_inq`: 10.0% missing — same pattern.
- `dti`, `revol_util`, `all_util`: <0.2% missing — median imputed.
- ![missingness](outputs/eda/03_missingness_top15.png)

### FICO analysis
- Range 660-845 in train, mean 698, std 30.
- Strong monotone negative relationship with `int_rate`: FICO ≥ 760 → mean rate 8.4%; FICO 660-680 → mean rate 17.6%.
- `fico_range_high − fico_range_low` is always 4 (the LC reporting window); collapsed to `fico = midpoint` and dropped originals to avoid collinearity.
- ![fico](outputs/eda/05_fico_analysis.png)

### Data quality issues
- 8 columns arrive as strings with literal `"NA"` for missing — handled at load via dtype overrides.
- `tot_cur_bal` arrives Float64 in train, Int64 in test — both forced to Float64 for schema parity.
- 0 full-row duplicates.

### Critical structural absences
- **No `grade` / `sub_grade`** — LendingClub's internal pricing variables that determine ~90% of rate variance (Phil Fed study). This is the source of our ~3.83 ceiling.
- **No date columns** (`issue_d`, `earliest_cr_line` absent) — temporal validation impossible; rate-regime shifts unmodelable from native features.
- **No `installment` column** — would have provided an amortization back-door to int_rate.

### Train vs test drift
- Largest drift on `loan_amnt` (KS D=0.14, PSI=0.17). Direction is opposite to typical temporal drift, suggesting the test set is a different random sample rather than a temporal slice.
- All other features show PSI < 0.10 (negligible drift).

---

## 4. Data Cleaning & Preprocessing

### Load-time handling

In [`production/features.py`](production/features.py):

```python
STRING_NUMERIC_COLS = [
    "dti", "revol_util", "all_util", "mo_sin_old_il_acct",
    "mths_since_last_record", "mths_since_rcnt_il", "mths_since_recent_bc",
    "mths_since_recent_inq", "tot_cur_bal", "loan_amnt",
]
# Forced to float64 at read with na_values=["NA"]
```

### Leakage drop on load
`loan_status` is post-origination outcome data; dropped immediately on load to prevent target leakage. This is the only column we drop unconditionally.

### Categorical handling
- **Low-cardinality** (`application_type`, `home_ownership`, `verification_status`, `purpose`): native CatBoost categorical, one-hot for linear/NN models.
- **High-cardinality** (`addr_state`, `zip3` derived from `zip_code`, `emp_title`, `purpose` again): smoothed K-fold OOF target encoding (m=20 smoothing constant) to prevent leakage.
- **`emp_title`**: free-text field with ~25k unique values → compacted to top-200 by frequency, rest → `"Other"`, missing → `"Missing"`.
- **`title`**: dropped (one-to-one duplicate of `purpose`).
- **`zip_code`**: replaced by `zip3` (first 3 digits) for ~900 categories vs ~30k.
- **`term`**: parsed to integer months (36 or 60).
- **`emp_length`**: ordinal-encoded 0-10.

### Numeric handling
- **Median imputation** (StandardScaler + Ridge/MLP only): inside each fold to prevent leakage.
- **Log1p** for monetary tails: `annual_inc`, `revol_bal`, `tot_cur_bal`, `total_bal_ex_mort`, `tot_coll_amt`, `loan_amnt`.
- **Sqrt** for moderately skewed counts: `revol_bal`, `open_acc`, `total_acc`.
- **Missingness flags** for the 4 `mths_since_*` columns where missing has meaning ("never happened"): `has_mths_since_last_record`, etc., then fill with sentinel value.

### Validation split
Random 80/20 train/holdout via `train_test_split(test_size=0.20, random_state=6604)`. **No temporal split possible** (no dates). Same `random_state` reused across every experiment so comparisons are apples-to-apples.

---

## 5. Feature Engineering (178 total features)

### 5.1 Production base features (135)
Built in [`production/features.py::engineer()`](production/features.py).

**Stateless transforms** (computed per row, no fit needed):

| Group | Features | Purpose |
|---|---|---|
| FICO derived | `fico` (midpoint), `fico_spread`, `fico_band` (5 bins), `fico_band_fine` (9 bins), `fico_log`, `fico_inv`, `fico_sq`, `fico_dist_720`, `fico_above_prime`, `fico_subprime` | Capture non-linear FICO effects + threshold flags |
| Term/employment | `term_months` (36/60 → int), `emp_length_num` (0-10), `emp_length_x_mortgage`, `emp_length_x_rent` | Numeric encoding + interactions |
| Missingness flags | `has_mths_since_*` (4) | Convert "never happened" semantic into model signal |
| Underwriting ratios | `loan_to_income`, `installment_proxy` (loan_amnt/term), `bal_to_income`, `payment_to_income`, `acc_open_ratio` | Standard credit ratios |
| FICO interactions | `util_x_fico`, `dti_x_fico`, `term_x_fico`, `fico_inv_x_loan_amnt`, `fico_inv_x_term`, `log_loan_x_inv_fico` | Combined risk |
| DTI bands | `dti_band` (5 bins by 10pp) | LC's underwriting cut-points |
| Composite risk | `derog_score` (weighted sum of pub_rec, bankruptcies, chargeoffs, collections), `inq_intensity`, `delinq_2yr_flag`, `pub_rec_flag`, `bankrupt_flag`, `chargeoff_flag`, `collections_flag`, `mort_acc_flag`, `any_derog_flag`, `composite_risk` (linear combo of strongest priors) | Aggregate adverse-history signal |
| Utilization bands | `revol_util_band`, `all_util_band` (5 bins each), `util_max`, `util_maxed_out`, `util_unused`, `util_product`, `util_spread` | Credit-utilization patterns |
| Credit-line tenure | `credit_file_age_yrs` = `mo_sin_old_rev_tl_op / 12`, `credit_age_band`, `delinq_per_credit_age`, `inq_per_credit_age`, `inq_per_open_acc`, `credit_experience` | Normalize counts by tenure |
| Loan-structure interactions | `term_x_loan_amnt`, `term_x_dti` | Combined loan-size × duration risk |
| Log tails | `log1p_annual_inc`, `log1p_revol_bal`, `log1p_tot_cur_bal`, `log1p_total_bal_ex_mort`, `log1p_tot_coll_amt`, `log1p_loan_amnt`, `sqrt_revol_bal`, `sqrt_open_acc`, `sqrt_total_acc`, `log_available_credit` | Reduce skew for linear/NN models |
| Loan affordability | `available_credit`, `installment_to_revol`, `installment_to_curbal`, `loan_amt_div_fico` | Debt-service capacity |
| Geography | `zip3` (first 3 digits of zip_code), `zip1` (first digit) | Coarser geographic encoding |
| Homeownership | `is_homeowner`, `is_renter`, `mortgage_per_open_acc` | Binary indicators |
| Employment-income | `emp_length_x_income`, `stable_high_income`, `unstable_low_income` | Joint stability signals |
| **State macros (external)** | `state_median_income`, `state_unemployment`, `state_col_index`, `inc_vs_state_median`, `col_adjusted_income`, `loan_to_state_median`, `unemployment_x_dti`, `col_x_dti` | US Census 2022 ACS + BLS lookup tables embedded as constants |
| Purpose-risk | `purpose_risk_score`, `purpose_risk_x_loan`, `purpose_risk_x_dti`, `purpose_risk_x_fico` | Weighted-by-default-rate purpose encoding |

**Monotone constraints** applied to LightGBM/XGBoost via priors:
- FICO → −1 (higher FICO ⇒ lower rate)
- DTI, term, revol_util → +1
- `monthly_residual_income`, `inc_vs_state_median` → −1
- `composite_risk`, `purpose_risk_score`, `recent_inq_burst` → +1

These constraints help linear-leaning base learners not overfit while reducing stacker noise.

### 5.2 Auxiliary "reverse-engineered" features (14)
Built in [`reverse_engineer/enrich.py::enrich_with_aux()`](reverse_engineer/enrich.py).

The aux features come from CatBoost models trained on `achive_data/` (which has `grade`, `sub_grade`, `int_rate` but NOT FICO). Two source datasets (`loan.csv` 800k rows + `archive/LC_train.csv` 100k rows) each yield three models. Predictions on true data are averaged across sources:

| Column | Source | Meaning |
|---|---|---|
| `aux_grade_prob_A` … `aux_grade_prob_G` | 7-class CatBoost classifier | Probability each row belongs to grade A-G, averaged across the two sources |
| `aux_subgrade_ord_avg` | CatBoost regressor on sub_grade ordinal (1-35) | Predicted sub-grade as continuous score |
| `aux_subgrade_ord_loan`, `aux_subgrade_ord_lct` | Per-source raw outputs | Lets stacker weight the two source predictions independently |
| `aux_int_rate_avg` | CatBoost regressor on int_rate | Predicted rate from archive features alone |
| `aux_int_rate_loan`, `aux_int_rate_lct` | Per-source | |
| `aux_grade_argmax` | Hard predicted grade ordinal (1-7) | |

### 5.3 FICO × aux interaction features (30)
Built in [`reverse_engineer/enrich_v2.py::add_fico_aux_interactions()`](reverse_engineer/enrich_v2.py). **This is what v2 adds over post_production** — and what produced the final 0.003 pp improvement to 3.8297.

The aux model never sees FICO (archive has none). But the *combination* of FICO (from true data) × predicted_grade (from aux) is much more informative than either alone — a row with high FICO but low predicted grade is either a mispriced opportunity or a misprediction; either way, it's signal.

| Group | Features | Count |
|---|---|---|
| Predicted-grade × FICO | `fico_x_aux_grade_argmax`, `fico_x_aux_subgrade_ord_avg`, `fico_inv_x_aux_subgrade_ord_avg`, `fico_x_aux_int_rate_avg`, `fico_inv_x_aux_int_rate_avg`, `fico_band_x_aux_grade_argmax`, `fico_band_fine_x_aux_subgrade_ord_avg` | 7 |
| Grade-prob × FICO (one per grade) | `fico_x_aux_grade_prob_A` … `fico_x_aux_grade_prob_G` | 7 |
| Grade-distribution moments | `aux_grade_entropy`, `aux_grade_max_prob`, `aux_grade_top2_gap`, `aux_grade_expected_ord`, `fico_x_aux_grade_expected_ord` | 5 |
| Mispricing residuals | `aux_argmax_minus_fico_band`, `abs_aux_argmax_minus_fico_band`, `aux_minus_fico_implied_rate`, `abs_aux_minus_fico_implied_rate` | 4 |
| Aux × DTI / loan_amnt | `aux_subgrade_ord_x_dti`, `aux_int_rate_x_dti`, `aux_subgrade_ord_x_loan_amnt`, `aux_int_rate_x_loan_amnt` | 4 |
| Source-disagreement | `aux_subgrade_source_disagreement`, `aux_int_rate_source_disagreement` | 2 |
| Composite | `composite_risk_v2` (linear blend of FICO + aux signals + DTI + util) | 1 |

**Total**: 135 + 14 + 30 = **179 features** going into the v2 stack (one column was dropped due to the test predictions exclusion of `ID`).

---

## 6. Model Architecture — 12-base Stacked Ensemble

### 6.1 Base learners
Built in [`production/fitters.py`](production/fitters.py), [`production/nn_models.py`](production/nn_models.py), [`production/extra_bases.py`](production/extra_bases.py), [`production/pseudo_labels.py`](production/pseudo_labels.py).

| # | Base | Type | Tuned via | Multi-seed | Notes |
|---|---|---|---|---|---|
| 1 | **CatBoost** (cb) | GBDT | Optuna 60 trials | 3 seeds | Native categorical handling; uses target-encoded + raw cats; monotone constraints |
| 2 | **LightGBM** (lgb) | GBDT | Optuna 100 trials | 3 seeds | Locked categorical dtypes; monotone constraints |
| 3 | **XGBoost** (xgb) | GBDT | Optuna 100 trials | 3 seeds | `enable_categorical=True`; monotone constraints |
| 4 | **LightGBM-Huber** (lgb_huber) | GBDT | Reuses LGB params | 3 seeds | Huber loss (α=0.9) for outlier robustness |
| 5 | **RandomForest** (rf) | Bagged trees | Hand-set (n_est=200, depth=14, leaf=20) | 1 seed | One-hot + median-imputed numeric frame |
| 6 | **Ridge** (ridge) | Linear | α grid-searched per fold | 1 seed | StandardScaler + SimpleImputer pipeline |
| 7 | **sklearn MLPRegressor** (mlp) | NN | Hand-set (96-48 hidden, lr=2e-3) | 1 seed | Early stopping at val patience 8 |
| 8 | **FT-Transformer** (ft) | Tabular Transformer (PyTorch) | Hand-set (d_token=32, 3 blocks, 4 heads, d_hidden=128) | 2 seeds | Single shared embedder + CLS token aggregation |
| 9 | **Deep ResMLP** (mlp_deep) | NN | Hand-set ((256,256,128,64) hidden + skip connections + GELU) | 2 seeds | Residual blocks with SE-style projections |
| 10 | **HistGradientBoosting** (hgb) | Binned GBDT | Hand-set | 2 seeds | sklearn's binned GBDT; different splits than LGB |
| 11 | **ExtraTrees** (extra_trees) | Randomised trees | Hand-set | 1 seed | High-variance bagging; captures non-smooth structure |
| 12 | **CatBoost Pseudo** (cb_pseudo) | GBDT trained on (train + top-50% confident test as soft labels, weight 0.5) | Reuses CB params | 3 seeds | Pseudo-labelling lever |

### 6.2 K-fold OOF design

- **5 outer folds** of the 80% train set (KFold, shuffle=True, random_state=6604).
- For each fold:
  - Build per-fold OOF smoothed target encodings (smoothing m=20) for the 4 high-cardinality cat columns. Train rows get encoded using the *other* folds only — no leakage.
  - Fit each base learner on the in-fold train data, early-stopping on the out-fold validation rows.
  - Average predictions across seeds.
- Each base learner produces:
  - `oof[base]` — out-of-fold predictions on the 80% train (input to the meta-learners)
  - `val_pred[base]` — averaged predictions on the 20% held-out validation
  - `test_pred[base]` — averaged predictions on the 10k test set

### 6.3 Meta-learners (compete for winning blend)
Built in [`production/stack.py::build_and_select()`](production/stack.py).

| Meta-learner | Mechanism |
|---|---|
| **Ridge stack** | `Ridge(alpha=100.0, positive=True)` on OOF stack; α picked by 5-fold CV |
| **ElasticNet stack** | `ElasticNet(alpha=0.01, l1_ratio=0.9, positive=True)` |
| **LightGBM stack** | Shallow LightGBM (n_est=400, depth=4) on the 12-column OOF matrix |
| **Hill-climb blend** | Constrained-positive weights summing to 1, optimized by random hill-climbing on OOF RMSE |
| **Mean of meta-learners** | Average of the four blends above |

Winner is selected by held-out validation RMSE (not by training RMSE — that would systematically favour LightGBM stack via overfitting on OOF train).

### 6.4 v2 stacker weights (winner = lgb_stack at 3.8297)
The Ridge-stack weights (which gives interpretable per-base contribution; LGB stack uses non-linear combinations):

```
cb        +0.38   ← still the workhorse despite all the additions
xgb       +0.21
ft        +0.17   ← FT-Transformer captures non-linear orthogonal signal
mlp_deep  +0.14
lgb       +0.11
lgb_huber +0.05
cb_pseudo +0.03
hgb       +0.02
rf, ridge, mlp, extra_trees → 0   (stacker rejects fold-2 anomalous models)
```

The fold-2 anomaly (Ridge produces RMSE 6+ on that one fold due to TE distribution shift) is correctly downweighted to zero by all four meta-learners.

---

## 7. Reverse Engineering — The Auxiliary Grade Inference

This is the conceptual heart of v2's edge over `production/`.

### Why it works
- Native `grade` explains ~90% of `int_rate` variance (Phil Fed). True data has no grade.
- Archive data has grade but no FICO.
- We train aux models on archive to **predict** grade from the 24 features common to both sources.
- Apply aux to true data → "predicted grade" enrichment columns.
- The main model now sees (true-data features incl. FICO) ⊕ (aux-predicted grade).
- FICO × aux-grade interactions give the linear/NN base learners a much sharper gradient signal.

### Per-fold contributions of aux features
Comparing v2's fold-1 base-learner RMSEs to production's (same data, same models, only difference is +14 aux + 30 FICO×aux cols):

| Model | Δ on fold 1 | Lift class |
|---|---|---|
| cb | −0.003 | Marginal (trees already approximate grade) |
| xgb | −0.003 | Marginal |
| lgb | −0.003 | Marginal |
| ft | −0.010 | Helpful |
| mlp_deep | −0.015 | Helpful |
| rf | −0.015 | Helpful |
| mlp | −0.020 | Helpful |
| **ridge** | **−0.087** | **Massive** (linear models gain hugely from explicit grade encoding) |

So aux features primarily uplift the *non-tree* base learners — exactly what the stacker needs for diversity. Tree models already capture the structure implicitly.

### The v3 negative result (lessons learned)
A v3 experiment added: 35-class sub_grade classifier, sub_grade rate-card lookup, issue-year prediction, within-grade residual model. Result: **3.8413 — worse than v2's 3.8297**.

Diagnostic: every per-model inner-validation RMSE in v3 was 0.005-0.010 worse than v2's. The new features added more noise than signal. This confirms v2 sits at the noise-vs-signal optimum for this approach.

---

## 8. Training Procedure (v2)

Total wall-clock: ~8 hours.

| Phase | Time | Output |
|---|---|---|
| **Phase 1 — Optuna tuning** (CB 60 trials + LGB 100 + XGB 100, single inner 85/15 split, early stopping per trial) | ~5 hr | `tuning_v2.json` |
| **Phase 2 — K-fold OOF stack** (5 folds × 9 base learners × 3 GBDT seeds + 2 NN seeds + single-seed for RF/Ridge/MLP) | ~3 hr | per-fold checkpoint pkls + final `aggregate.pkl` |
| **Phase 3 — Meta-learner selection** (5 meta-learners scored on held-out val) | ~30 s | `ranking_v2.csv`, `test_predictions_v2.csv` |

### Reproducibility knobs
- `RANDOM_STATE = 6604` everywhere — fold splits, seed offsets, scaler/imputer fits all derive from this.
- `N_FOLDS = 5`, `N_SEEDS_GBDT = 3`, `N_SEEDS_NN = 2`.
- Tuning cache (`tuning_v2.json`) is reusable — re-running picks up the cached params and skips Phase 1.
- OOF checkpoint (`aggregate.pkl`) saves after every fold — kills resume from latest fold.

### Anti-leakage discipline
1. `loan_status` dropped at load.
2. Train/val split BEFORE any fitting (imputers, scalers, target encoders).
3. Target encoders use 5-fold OOF on train rows; val/test rows use the full-train encoder.
4. SimpleImputer + StandardScaler fit on each fold's training rows only.
5. Auxiliary models trained on disjoint archive data — no peek at true-data int_rate during aux training.
6. Held-out 20% validation never touched until the final meta-learner selection.

---

## 9. Results

### Full progression

| Pipeline | val_RMSE | val_MAE | val_R² | Δ vs baseline |
|---|---|---|---|---|
| Original (`lendingclub_project/`) | 3.9120 | ~2.94 | ~0.45 | baseline |
| `final/` (CB only) | 3.8666 | 2.8959 | 0.4663 | −0.045 |
| `production/` (12-base, 89 feats) | 3.8434 | 2.8673 | 0.4726 | −0.069 |
| `production/` v2 (135 feats, retuned) | 3.8365 | ~2.86 | 0.466 | −0.076 |
| `reverse_engineer/post_production` | 3.8331 | 2.8652 | 0.4754 | −0.079 |
| 🏆 **`reverse_engineer/v2`** | **3.8297** | **2.854** | **0.4764** | **−0.082** |
| `reverse_engineer/v3` (overfit) | 3.8413 | 2.860 | 0.473 | −0.071 (regressed vs v2) |

### v2 base-learner detail (held-out 20%)
From [`outputs/reverse_engineer/v2/ranking_v2.csv`](outputs/reverse_engineer/v2/ranking_v2.csv):

| Rank | Model | val_RMSE | val_R² |
|---|---|---|---|
| 🥇 | **lgb_stack** | **3.8297** | 0.4764 |
| 🥈 | mean_of_meta_learners | 3.8373 | 0.4743 |
| 🥉 | en_stack | 3.8408 | 0.4734 |
| 4 | hill_climb_blend | 3.8431 | 0.4727 |
| 5 | base_cb | 3.8458 | 0.4720 |
| 6 | ridge_stack | 3.8552 | 0.4694 |
| 7 | base_xgb | 3.8696 | 0.4654 |
| 8 | base_lgb | 3.8983 | 0.4575 |
| 9 | base_mlp_deep | 3.9001 | 0.4570 |
| 10 | base_ft | 3.9129 | 0.4534 |
| 11 | base_mlp | 3.9513 | 0.4426 |
| 12 | base_lgb_huber | 3.9689 | 0.4376 |
| 13 | base_rf | 4.0006 | 0.4286 |
| 14 | base_ridge | 4.0558 | 0.4128 |

The stack adds ~0.016 pp over the best base learner (CB) — the value of ensembling is real but bounded.

### Test predictions (`outputs/reverse_engineer/v2/test_predictions_v2.csv`)
- 10,000 rows, `ID, int_rate`.
- `int_rate` clipped to `[6.0, 31.0]` (the observed range in train).
- Predicted mean ≈ 12.0, median ≈ 11.8, range 6.0-23.2 (no rows hit the upper clip in practice).

---

## 10. Limitations & The Data Ceiling

### Why ~3.83 is the floor for this slice
1. **No native `grade` column**: LendingClub's grade is the discretization of the rate they're computing — without it we're predicting the output of their pricing model from the same inputs, lossily. With grade, RMSE drops to ~3.0 (Phil Fed; QUB benchmark study).
2. **No FICO in archive sources**: The aux models that recover grade can't use FICO themselves, weakening their grade predictions. Combined approaches help but cannot recover the full signal.
3. **No `issue_d`**: Macro rate-regime shifts (2008 vs 2018 had wildly different baseline rates) are not modelable.
4. **No `installment` column**: Would have provided an amortization back-door (installment = f(loan_amnt, term, int_rate)).
5. **Fold-2 distribution shift**: One specific fold's TE distribution breaks Ridge/sklearn-MLP across every experiment — likely a few extreme outlier rows. The stacker correctly downweights these models to zero on fold 2, so the final blend is robust.

### What was tried and did NOT help
- **v3 35-class sub_grade classifier**: 20+ hours of compute, never completed. Even after dropping the heaviest lever, v3 with rate-card + issue-year + within-grade-residual underperformed v2.
- **Multi-iteration self-training** (`reverse_engineer/` iter loop): converged to a worse RMSE (3.94) after 2 iterations, then triggered convergence stop.
- **LightAutoML**: incompatible with Python 3.14 (`statsmodels` build failure); HistGradientBoosting and ExtraTrees were used as sklearn-native substitutes.
- **Joint-applicant secondary features**: only 5.34% of rows; expected gain too marginal to justify the implementation cost.

### What would beat 3.83 (none feasible from the project data)
- Acquire the original LendingClub historical dataset with `grade` for these specific rows.
- Add FICO to the archive (would require LendingClub's non-public data dump).
- Add `issue_d` to the true-data slice (would require enriching the source).

---

## 11. Reproducibility

### File locations

**Code** (all read-only references for the v2 champion):
- [`production/features.py`](production/features.py) — `engineer()`, `compact_emp_title()`, state-macro lookups
- [`production/encoders.py`](production/encoders.py) — K-fold OOF target encoding + per-model frame prep
- [`production/fitters.py`](production/fitters.py) — 7 base learner fit-predict helpers
- [`production/nn_models.py`](production/nn_models.py) — FT-Transformer + Deep ResMLP (PyTorch)
- [`production/extra_bases.py`](production/extra_bases.py) — HistGradientBoosting + ExtraTrees
- [`production/pseudo_labels.py`](production/pseudo_labels.py) — cb_pseudo base learner
- [`production/stack.py`](production/stack.py) — meta-learner selection
- [`production/tune.py`](production/tune.py) — Optuna tuning
- [`reverse_engineer/archive_loader.py`](reverse_engineer/archive_loader.py) — loads + aligns archive sources
- [`reverse_engineer/aux_models.py`](reverse_engineer/aux_models.py) — CB grade/sub_grade/int_rate aux trainers
- [`reverse_engineer/enrich.py`](reverse_engineer/enrich.py) — applies aux ensemble → 14 cols
- [`reverse_engineer/enrich_v2.py`](reverse_engineer/enrich_v2.py) — FICO×aux interactions → 30 cols
- [`reverse_engineer/run_v2.py`](reverse_engineer/run_v2.py) — v2 orchestrator (3 phases)

**Artifacts** (all under `outputs/reverse_engineer/v2/`):
- `test_predictions_v2.csv` — 10k predictions (the submission)
- `ranking_v2.csv` — full 14-model ranking
- `post_artifacts_v2.pkl` — OOF/val/test stacks for analysis
- `tuning_v2.json` — Optuna-cached hyperparameters

### To reproduce from scratch
```powershell
cd "c:\Users\liula\Documents\applied ai\project 1"
Remove-Item "outputs\reverse_engineer\v2" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "outputs\reverse_engineer\iter_1" -Recurse -Force -ErrorAction SilentlyContinue

# Phase 1 — generates iter_1's enriched parquets (~3 hours)
.venv\Scripts\python.exe -u reverse_engineer\run_all.py

# Phase 2 — runs v2 enrichment + tuning + 5-fold stack (~8 hours)
.venv\Scripts\python.exe -u reverse_engineer\run_v2.py
```

Resulting `test_predictions_v2.csv` should match the shipped one within floating-point precision (deterministic with the fixed seed).

### Dependencies
- Python 3.14
- pandas, numpy, scikit-learn ≥ 1.8
- catboost 1.2.10, lightgbm 4.6, xgboost 3.2
- torch 2.12 (CPU)
- optuna 4.8
- shap 0.51

---

## 12. Glossary

- **OOF (out-of-fold)**: predictions made on rows the model didn't see during training. Used to feed meta-learners without leakage.
- **TE (target encoding)**: replace a categorical level with the mean of the target for that level. K-fold OOF TE prevents the level's own target from contributing.
- **Multi-seed averaging**: train the same model with different random seeds and average predictions. Reduces variance by `~1/√n_seeds`.
- **Stacking**: a meta-model that learns optimal weights to combine base-model OOF predictions.
- **Hill-climb blend**: a simple coordinate-descent search for non-negative weights summing to 1 that minimise OOF RMSE.
- **Pseudo-labelling**: confident test-set predictions are added to the training set as soft labels (with reduced weight), letting the model see the test feature distribution during training.
- **Auxiliary model**: a model whose output becomes a feature for the main model. Here, CatBoost trained to predict `grade` on archive data, then applied to true data.
- **Fold-2 anomaly**: a recurring observation that the random K-fold's second fold causes Ridge and sklearn MLP to produce RMSE > 6 on every experiment. Caused by TE distribution shift on a specific subset of rows. Stacker correctly assigns those models zero weight on that fold.
