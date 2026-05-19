# LendingClub Interest Rate Prediction — EDA Report

**Notebook:** [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) · **Figures:** [`outputs/eda/`](outputs/eda/) · **Predecessor pipeline (archived for reference):** [`src/archive/preprocessing_and_modeling.py`](src/archive/preprocessing_and_modeling.py)

---

## Executive summary

100,000 application-time loan records (Jun 2007 → Jan 2013) with 39 columns; 10,000 chronologically-later test records (Jan-Feb 2013). Target `int_rate` is mildly right-skewed (skew 0.288, mean 13.04%, IQR 10.00–15.80). Three columns leak the target and must be dropped: `grade` (Pearson r = 0.922), `sub_grade` (0.939), and `installment` — the last is *algebraically* the monthly PMT of (`loan_amnt`, `term`, `int_rate`) and recovers the observed installment to within $3 mean absolute error. ~70% of credit-bureau enrichment columns (`mo_sin_*`, `num_*`, `pct_tl_nvr_dlq`, `percent_bc_gt_75`) are MNAR-structural — LC introduced them mid-2012. Test drift is meaningful but tolerable: `loan_amnt` mean moves +26% (train→test), `installment` PSI 0.28. Strongest non-leakage Spearman correlations are `revol_util` (0.47) and `percent_bc_gt_75` (0.40); the rest of the feature set provides modest, additive signal. Honest temporal R² ceiling is ≈ 0.55–0.60 (FINDINGS retrospective: 0.516).

---

## 1. Data overview

| | Train | Test |
| --- | --- | --- |
| Rows × cols | 100,000 × 39 | 10,000 × 39 |
| `int_rate` | present (target) | absent |
| `ID` | absent | present (1–10,000, unique) |
| `issue_d` range | 2007-06-01 → 2013-01-01 | 2013-01-01 → 2013-02-01 |
| `issue_d` monotonic | ✓ | ✓ |
| Full-row duplicates | 0 | — |
| Dtype mismatches (shared cols) | 0 | |

The split is purely temporal — test sits immediately after train, with overlap only at the boundary day 2013-01-01. No post-origination columns (`loan_status`, `total_pymnt`, `recoveries`, `last_pymnt_*`, etc.) are present, so the slice is clean for application-time modeling. The 4 known pre-origination leakage candidates (`grade`, `sub_grade`, `installment`, and indirectly `issue_d` itself) are addressed below.

---

## 2. Target distribution

`int_rate` ranges 5.42 → 24.89 %, mean 13.04, median 13.11, std 4.16. The distribution is bimodal-ish (clusters around 11% and 16%) with a mild right tail.

![target distribution](outputs/eda/02_target_distribution.png)

Transform diagnostics (skew | kurtosis):

| Transform | skew | kurtosis |
| --- | --- | --- |
| identity | +0.288 | −0.447 |
| log1p | −0.338 | −0.564 |
| sqrt | −0.054 | −0.619 |
| Box-Cox (λ=0.52) | −0.039 | −0.617 |

A transform isn't needed for tree models. The XGBoost baseline in the archived pipeline operates on raw `int_rate`; that's the right call.

---

## 3. Missingness — MCAR / MAR / MNAR verdict

23 columns have any missingness. Two tiers dominate:

![top-15 missingness](outputs/eda/03_missingness_top15.png)

| Tier | Pct missing | Columns | Verdict |
| --- | --- | --- | --- |
| Heavy | ~70% | `mo_sin_old_rev_tl_op`, `mo_sin_rcnt_tl`, `num_accts_ever_120_pd`, `num_tl_90g_dpd_24m`, `pct_tl_nvr_dlq` | **MNAR-structural** — LC introduced these columns in mid-2012; they are ~100% null pre-2012 and ~0% null after |
| Moderate | ~50% | `percent_bc_gt_75`, `mort_acc`, `total_bc_limit` | MNAR-structural, slightly earlier rollout |
| Light | 1–6% | `emp_title`, `emp_length`, `pub_rec_bankruptcies` | MAR/MCAR — sporadic rather than systematic |

The missingness matrix (rows sorted chronologically) shows a clean horizontal split — the columns flip from "all null" to "almost all populated" at a single boundary, confirming the structural-rollout story.

![missingness × year](outputs/eda/03_missingness_by_year.png)

Co-occurrence is near-perfect within each tier:

![missingness co-occurrence](outputs/eda/03_missingness_cooccurrence.png)

**Implication:** the ~70% block isn't lost data — it's data that doesn't exist for older loans. Median imputation is reasonable, but adding a missingness flag (`is_missing_<col>`) lets the model learn the "pre-2012 vs post-2012" macro regime indirectly. Point-biserial correlation of `is_null` indicators with `int_rate` is meaningful for several columns and worth using as a feature.

---

## 4. Univariate distributions

![numeric histograms](outputs/eda/04_numeric_histograms.png)

Highly-skewed monetary columns (`annual_inc` skew 9.4, `revol_bal` 6.6, `total_bc_limit` 4.2, `loan_amnt` 0.5) benefit from log transforms in linear-model contexts; for XGBoost the log mostly improves SHAP interpretability rather than accuracy. Zero-inflated columns include `pub_rec` (78% zero), `delinq_2yrs` (84%), `tax_liens` (99%), `acc_now_delinq` (99%), `chargeoff_within_12_mths` (99%) — these are good candidates for binary "any" flags alongside the raw counts.

![categorical counts](outputs/eda/04_categorical_counts.png)

`application_type` is constant (`Individual`) — drop. `initial_list_status` has only 2 levels but the train/test split is highly skewed (chi² 7,688) — important to keep. `emp_length` is a textual ordinal that needs explicit mapping (`< 1 year` → 0, …, `10+ years` → 10) per the archived `EMP_LENGTH_MAP`.

---

## 5. Temporal analysis — the macro story

Loan volume grew ~200× over the train period, on a roughly exponential trajectory:

![volume drift](outputs/eda/05_volume_drift.png)

Mean rate drifted from 9.81% in Q3 2007 to 14.36% in Q1 2013 — a 4.5 pp move:

![target drift](outputs/eda/05_target_drift.png)

Several application features trend alongside this — average `loan_amnt`, `dti`, `revol_util`, and `mort_acc` all rise with issue date:

![feature drift](outputs/eda/05_feature_drift.png)

**Implication:** features that trend with time are *partially confounded* with the macro drift. Under random validation, they look much more powerful than they are; under honest temporal validation (last 20% of train by `issue_d`), the model can't extrapolate beyond the training time window and these features are correctly discounted. The retrospective in `FINDINGS.md` shows random splits hide ~1.5 pp of RMSE error from this effect — temporal validation is non-negotiable.

---

## 6. Leakage audit

**`grade` and `sub_grade`** are LC's internal risk grade letters (A–G) and sub-grades (A1–G5). Each is derived from the same underwriting model that sets the rate, so they correlate ~0.92 / ~0.94 with the target — catastrophic separation:

![grade separation](outputs/eda/08_grade_separation.png)

**`installment`** looks innocent — marginal correlation with `int_rate` is just 0.34. But `installment` is the **monthly PMT** computed from `loan_amnt`, `term`, and `int_rate`:

$$\text{installment} = \frac{r \cdot P}{1 - (1+r)^{-n}}, \quad r = \frac{\text{int\_rate}}{100 \cdot 12}, \quad n = \text{term months}$$

Computing the implied PMT from the data and comparing to the observed `installment`:

| | Value |
| --- | --- |
| corr(implied, observed) | 0.994 |
| mean abs. error | $3.20 |
| max abs. error | $745.72 |

![installment PMT scatter](outputs/eda/08_installment_pmt.png)

A linear partial correlation `corr(installment, int_rate \| loan_amnt, term)` reports 0.49 — substantial on its own. The relationship is non-linear (PMT depends on $(1+r)^{-n}$), so linear partial correlation under-states the deterministic dependency that the scatter plot makes obvious: given the other two inputs, knowing `installment` exactly recovers `int_rate`.

**Drop list:** `LEAKAGE_COLS = ["grade", "sub_grade", "installment"]`. The archived pipeline already does this — keep it that way.

---

## 7. Correlations & multicollinearity

![correlation heatmap](outputs/eda/07_correlation_heatmap.png)

Non-leakage Spearman ranking against `int_rate`:

| Rank | Feature | Spearman | Pearson | n |
| --- | --- | --- | --- | --- |
| 1 | `revol_util` | +0.468 | +0.466 | 99,862 |
| 2 | `percent_bc_gt_75` | +0.402 | +0.410 | 49,416 |
| 3 | `loan_amnt` | +0.287 | +0.340 | 100,000 |
| 4 | `inq_last_6mths` | +0.210 | +0.157 | 99,971 |
| 5 | `pct_tl_nvr_dlq` | −0.209 | −0.183 | 29,724 |
| 6 | `total_bc_limit` | −0.172 | −0.183 | 49,970 |
| 7 | `dti` | +0.170 | +0.175 | 100,000 |
| 8 | `revol_bal` | +0.169 | +0.090 | 100,000 |
| 9 | `delinq_2yrs` | +0.159 | +0.144 | 99,971 |
| 10 | `mo_sin_rcnt_tl` | −0.155 | −0.096 | 29,724 |

Multicollinearity is mild — only 2 numeric pairs exceed |r| = 0.7:

| Feature A | Feature B | r |
| --- | --- | --- |
| `pub_rec` | `pub_rec_bankruptcies` | 0.858 |
| `revol_util` | `percent_bc_gt_75` | 0.726 |

XGBoost is robust to these; SHAP attributions may split across the pair in ways that distort feature importance but not prediction accuracy.

---

## 8. Bivariate analysis vs target

![bivariate hexbin grid](outputs/eda/06_bivariate_hexbin.png)

The binned-mean overlays show that `revol_util` and `percent_bc_gt_75` have clean monotonic relationships with `int_rate`; `loan_amnt` is also monotonic. `pct_tl_nvr_dlq` is inversely monotonic (higher percentage of never-delinquent accounts → lower rate). These are the columns that survive into the SHAP top-10 of the archived pipeline.

Categorical separations are real but smaller:

![categorical boxplots](outputs/eda/06_categorical_boxplots.png)

`term` ("60 months" carries higher rate than "36 months") and `purpose` (small_business and renewable_energy carry the highest rates) are the cleanest separators among categoricals.

---

## 9. Geographic analysis

The `addr_state` mean-rate spread is **3.9 pp** between the lowest (DC) and highest (NV) — larger than the 0.4 pp note in FINDINGS suggested, though the *median* spread is much tighter (most states cluster within 1 pp of each other). State is worth keeping as a native XGBoost categorical for interaction potential, not as a strong main effect.

![state bar](outputs/eda/06_state_bar.png)

Zip-code first-digit signal is genuinely flat (0.26 pp spread):

![zip first digit](outputs/eda/11_zip_first_digit.png)

Of 850 unique `zip_code` 3-digit prefixes, 127 map to more than one state — this is an artifact of LC's masking (the trailing two digits are replaced with `xx`, conflating distinct full ZIPs). Combined with the flat signal, `zip_code` should be dropped.

---

## 10. Train vs test drift

KS test and PSI on each numeric feature; chi-square on each categorical. The split is temporal, so drift is expected — the question is how much.

Top 10 by KS D-statistic:

| Feature | KS D | PSI | train mean | test mean | Δ mean |
| --- | --- | --- | --- | --- | --- |
| `installment` | 0.221 | 0.278 | 381.84 | 491.48 | +109.6 |
| `loan_amnt` | 0.185 | 0.212 | 12,524 | 15,800 | +3,276 |
| `revol_bal` | 0.156 | 0.206 | 14,867 | 19,092 | +4,225 |
| `revol_util` | 0.139 | 0.134 | 54.36 | 62.25 | +7.89 |
| `open_acc` | 0.129 | 0.105 | 10.11 | 11.40 | +1.29 |
| `dti` | 0.111 | 0.133 | 15.29 | 17.79 | +2.50 |
| `annual_inc` | 0.089 | 0.062 | 69,624 | 76,828 | +7,204 |
| `total_acc` | 0.088 | 0.059 | 23.04 | 25.26 | +2.22 |
| `pct_tl_nvr_dlq` | 0.079 | 0.026 | 95.07 | 96.11 | +1.04 |
| `mort_acc` | 0.077 | 0.019 | 1.63 | 1.89 | +0.26 |

PSI > 0.2 (significant shift) for `installment`, `loan_amnt`, `revol_bal`. PSI > 0.1 (monitor) for `revol_util`, `open_acc`, `dti`. This drift is consistent with the macro story in §5: loans got bigger and borrowers more leveraged as LC scaled.

![drift KDE](outputs/eda/10_drift_kde.png)

Categorical chi² results are dominated by `initial_list_status` (D-stat 7,688 on 2 levels — distribution shifted dramatically as LC's whole-vs-fractional loan policy changed mid-period), `purpose` (1,480 on 14 levels), and `verification_status` (1,145). `application_type` is constant — drop.

**Implication:** the validation set should be the last 20% of train (chronologically), not the test set. The test set is a *more drifted* slice than even the val fold — this is why the FINDINGS retrospective notes that real-world deployment performance may be ~0.1 worse RMSE than the val-set estimate.

---

## 11. Data quality issues

| Issue | Count | Recommendation |
| --- | --- | --- |
| `annual_inc` > $1M (self-reported, no LC verification pre-2014) | 22 | XGBoost handles via splits; consider winsorize for linear models |
| `annual_inc` == 0 | 0 | n/a |
| `dti` == 0 | 244 | Legitimate (zero non-mortgage debt); keep |
| `revol_util` == 0 | 1,349 | Legitimate; keep |
| `loan_amnt` > 5 × `annual_inc` | 21 | Edge case — LC's actual underwriting cap is ~10× |
| `earliest_cr_line` parse failures | 29 / 100,000 | Negligible; impute |
| `term` raw values | "36 months" / "60 months" | Extract integer months |
| `revol_util` parsed as float | ✓ | No cleaning needed |
| `application_type` unique levels | 1 (`Individual`) | Drop column |

`earliest_cr_line` parses cleanly as `%b-%Y`, range Jan 1946 → Dec 2009. Derive `credit_history_length = issue_d - earliest_cr_line` — this appears in SHAP top features in the archived run.

---

## 12. Feature engineering recommendations

Each hypothesis is cross-referenced to its existing implementation in `engineer_features()` at [`src/archive/preprocessing_and_modeling.py`](src/archive/preprocessing_and_modeling.py).

| Hypothesis | Motivation | Prod implementation |
| --- | --- | --- |
| `log1p(annual_inc)`, `log1p(revol_bal)`, `log1p(total_bc_limit)` | Heavy right tail (§4) | `annual_inc_log`, `revol_bal_log`, `total_bc_limit_log` |
| `loan_to_income = loan_amnt / annual_inc` | Underwriting signal (§11) | `loan_to_income` |
| `inq_per_open_acc` | Credit-shopping density (§7) | `inq_per_open_acc` |
| `credit_history_length` (days) | Strong SHAP feature in retrospective (§11) | `credit_history_length` |
| `loan_amnt × revol_util`, `dti × total_bc_limit`, etc. | Non-linear bivariate patterns (§7, §8) | 13 interaction terms |
| Zero-inflated flags (`pub_rec_flag`, `delinq_flag`, …) | Long zero spikes (§4) | `pub_rec_flag`, `delinq_flag`, `tax_liens_flag`, `collections_flag` |
| **`is_missing_<col>` for the ~70% block** | Point-biserial r with target is meaningful (§3); structural rollout (§3) | **Not in prod — recommended addition** |
| `dti_bucket`, `loan_amnt_bucket` | Discrete tree splits help non-monotone effects | `dti_bucket`, `loan_amnt_bucket` |
| `addr_state` as native categorical | 3.9 pp main effect + interaction potential (§9) | `pl.Categorical` |
| Drop `emp_title`, `zip_code` | High cardinality, weak signal (§4, §9) | In `DROP_COLS` |
| Drop `grade`, `sub_grade`, `installment` | Leakage (§6) | In `LEAKAGE_COLS` |

The only new recommendation vs. the archived pipeline is **missingness indicators for the ~70% MNAR block**. The pipeline currently median-imputes silently; surfacing the "this loan is from the pre-2012 regime" signal as an explicit binary should help.

---

## 13. Modeling implications & next steps

### Pre-modeling checklist (refines the archived pipeline)

1. Drop `["grade", "sub_grade", "installment"]` unconditionally.
2. Drop `["emp_title", "zip_code", "application_type"]` — no signal.
3. Parse `term` to integer months, `emp_length` via `EMP_LENGTH_MAP`, `earliest_cr_line` via `pd.to_datetime(..., format="%b-%Y")`.
4. Apply `log1p` to `annual_inc`, `revol_bal`, `total_bc_limit`.
5. Compute `credit_history_length = issue_d - earliest_cr_line` (days).
6. **Add `is_missing_<col>` flags** for the 8 columns with > 30% missingness — new vs current pipeline.
7. Fit median imputation **on train only**; apply to val and test.
8. Native XGBoost categoricals for `addr_state`, `home_ownership`, `verification_status`, `purpose`, `initial_list_status` — no one-hot.
9. Validation = last 20% of train by `issue_d`. Do not random-split.

### Honest expectations

- **R² ceiling ≈ 0.55–0.60** under honest temporal validation (FINDINGS final: R² 0.516, RMSE 2.97).
- **Hyperparameter tuning** moves RMSE by ~0.02 across a 100-trial TPE study; the feature set is the binding constraint, not the optimizer.
- The richest unrecoverable signal is **FICO** (absent from public LC dumps under TransUnion licensing) and **`issue_year`** (out-of-distribution in val; trees can't extrapolate).
- Of what's left, `revol_util`, `percent_bc_gt_75`, `credit_history_length`, and `loan_amnt` carry the bulk of the SHAP attribution.

### Risk register

Features most exposed to train→test drift (PSI ≥ 0.1): `installment` (already dropped), `loan_amnt`, `revol_bal`, `revol_util`, `open_acc`, `dti`. If the model is deployed on a moving window, **recompute imputation medians per retrain** rather than freezing them.

---

*Generated from [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). To regenerate, run `.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb`.*
