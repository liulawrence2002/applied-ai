# LendingClub Interest Rate Prediction — EDA Report

**Dataset:** [`true data/LC_train.csv`](true%20data/LC_train.csv) (100k rows × 39 cols) · [`true data/LC_test.csv`](true%20data/LC_test.csv) (10k rows × 39 cols, no target, `ID` column instead)

**Notebook:** [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) · **Figures:** [`outputs/eda/`](outputs/eda/) · **Predecessor pipeline (archived for reference):** [`src/archive/preprocessing_and_modeling.py`](src/archive/preprocessing_and_modeling.py)

---

## Executive summary

100,000 application-time loan records with 39 columns; 10,000 held-out test records with the same 38 features plus an `ID` column (and no `int_rate`). Target `int_rate` is moderately right-skewed (skew 0.851, mean 12.95%, range 6.46–30.99). **FICO scores are present** (`fico_range_low`, `fico_range_high`) — the dominant non-leakage predictor (Spearman −0.46 with the target). **`loan_status` is post-origination leakage and is dropped immediately on load**; the side audit shows it explains 0.44% of `int_rate` variance via ANOVA (modest in magnitude but principled to exclude). The classic pre-origination leakage trio (`grade`, `sub_grade`, `installment`) is absent from this slice. **There are no date columns** — neither `issue_d` nor `earliest_cr_line` — so temporal validation is impossible; modeling must use a random split. Eight numeric columns arrived as strings with literal `"NA"` for nulls and are cast at load. After FICO, the strongest signals are `all_util` (Spearman +0.33), `revol_util` (+0.30), and `dti` (+0.18). Train and test differ most on `loan_amnt` (KS D=0.14, PSI=0.17, opposite direction to typical temporal drift). With FICO present, validation R² should be materially higher than the FICO-stripped `data/` slice (R² ~0.52 per `FINDINGS.md`), but the ceiling should be quoted only after a random-holdout modeling run.

---

## 1. Data overview

| | Train | Test |
| --- | --- | --- |
| Rows × cols | 100,000 × 39 | 10,000 × 39 |
| `int_rate` | present (target) | absent |
| `ID` | absent | present (1–10,000, unique) |
| `loan_status` | present → **dropped at load (leakage)** | present → dropped at load |
| Date columns | **none** (no `issue_d`, no `earliest_cr_line`) | none |
| Full-row duplicates | 0 | — |
| Dtype mismatches (shared cols) | 0 after Float64 override on `tot_cur_bal` | |

The dataset has **no temporal information** — neither `issue_d` nor `earliest_cr_line` exists. This is the single most important structural difference from the prior `data/` slice. There is no temporal split to honor, no macro drift to discount, and no in-time vs out-of-time generalization gap to measure. Validation must use a random hold-out.

Eight columns arrive as strings with literal `"NA"` for missingness — handled at load time via `pl.read_csv(null_values=["NA"], schema_overrides={...: pl.Float64, ...})`:

`dti`, `revol_util`, `all_util`, `mo_sin_old_il_acct`, `mths_since_last_record`, `mths_since_rcnt_il`, `mths_since_recent_bc`, `mths_since_recent_inq`.

`tot_cur_bal` arrives as Float64 in train but Int64 in test — both forced to Float64 for parity.

---

## 2. Target distribution

`int_rate` ranges 6.46 → 30.99 %, mean 12.95, median 11.71, std 5.26, skew 0.851. The distribution is right-skewed with a long upper tail.

![target distribution](outputs/eda/02_target_distribution.png)

Transform diagnostics (skew | kurtosis):

| Transform | skew | kurtosis |
| --- | --- | --- |
| identity | +0.851 | +0.100 |
| log1p | +0.232 | −0.903 |
| sqrt | +0.505 | −0.570 |

A transform isn't required for tree models. The skew here is larger than the prior `data/` slice (0.288), but XGBoost handles asymmetric targets natively via quantile splits.

---

## 3. Missingness

Only 10 columns have any missingness — far fewer than the prior `data/` slice (23 columns). The largest by far is `mths_since_last_record` at 90.9%:

| Column | n_missing | % |
| --- | --- | --- |
| `mths_since_last_record` | 90,875 | 90.88 |
| `emp_title` | 14,920 | 14.92 |
| `mths_since_recent_inq` | 9,975 | 9.98 |
| `emp_length` | 8,669 | 8.67 |
| `mo_sin_old_il_acct` | 2,204 | 2.20 |
| `mths_since_rcnt_il` | 2,204 | 2.20 |
| `mths_since_recent_bc` | 920 | 0.92 |
| `dti` | 186 | 0.19 |
| `revol_util` | 123 | 0.12 |
| `all_util` | 16 | 0.02 |

![top-15 missingness](outputs/eda/03_missingness_top15.png)

`mths_since_last_record` is **informatively missing**: most borrowers have no derogatory public record, so "months since last record" is undefined for them. Treat as MNAR (semantic null = "no event") and add a `has_record` binary flag rather than imputing a median that would falsely imply a recent event. The same logic applies to the other `mths_since_*` columns.

![missingness co-occurrence](outputs/eda/03_missingness_cooccurrence.png)

Without `issue_d` we cannot test whether missingness has a temporal mechanism (as it did in the prior `data/` slice where the ~70% block was rolled out in 2012). The co-occurrence heatmap shows `mo_sin_old_il_acct` and `mths_since_rcnt_il` are perfectly co-missing (both 2,204 rows) — same underlying "no installment account ever opened" semantics.

---

## 4. Univariate distributions

![numeric histograms](outputs/eda/04_numeric_histograms.png)

The most-skewed numerics (skew > 5) are the monetary tails: `tot_coll_amt`, `annual_inc`, `revol_bal`, `tot_cur_bal`, `total_bal_ex_mort` — log-transform candidates. Zero-inflated counters include `tot_coll_amt` (88% zero), `pub_rec`, `delinq_2yrs`, `chargeoff_within_12_mths`, `collections_12_mths_ex_med` — good candidates for binary "any" flags alongside the raw counts.

![categorical counts](outputs/eda/04_categorical_counts.png)

`application_type` has a meaningful split (`Individual` 87.3%, `Joint App` 12.7%), so do not drop it as constant without validation. `term` has only "36 months" / "60 months" (extract integer months). `purpose` has 11 levels with `debt_consolidation` and `credit_card` dominant — the typical LC distribution.

`emp_length` has 11 ordinal levels plus ~8.7% NA; map via `EMP_LENGTH_MAP = {"< 1 year": 0, …, "10+ years": 10}`.

---

## 5. FICO score analysis

The headline structural difference from the prior `data/` slice: **FICO is present here**.

- `fico_range_high − fico_range_low` is either 4 or 5 in every row, so the two columns are effectively the same signal — use the midpoint `fico = (low + high) / 2` and drop the originals.
- FICO range: **662 – 848** (mean 708.6, median 702, std 35.6). The lower bound shows LC pre-screens applicants — sub-660 FICO never makes it into the population.
- **Pearson r with `int_rate` = −0.430, Spearman = −0.462** — higher FICO → lower rate, the canonical credit-quality signal.

![FICO vs int_rate](outputs/eda/05_fico_analysis.png)

The bivariate plot shows the canonical monotonic inverse relationship: rates drop from ~16% at FICO 670 to ~9% at FICO 800+. The relationship is roughly piecewise-linear with a steeper slope below 720 — exactly what a tree model will discover.

**No temporal section.** There is no `issue_d` in this dataset. Validation must be a random split, not a temporal split, and out-of-time generalization is unmeasured.

---

## 6. Bivariate analysis vs target

Spearman ranking against `int_rate` (top 15):

| Rank | Feature | Spearman | Pearson | n |
| --- | --- | --- | --- | --- |
| 1 | `fico_range_high` | −0.462 | −0.430 | 100,000 |
| 1 | `fico_range_low` | −0.462 | −0.430 | 100,000 |
| 1 | `fico` (midpoint) | −0.462 | −0.430 | 100,000 |
| 4 | `all_util` | +0.334 | +0.316 | 99,984 |
| 5 | `revol_util` | +0.300 | +0.290 | 99,877 |
| 6 | `dti` | +0.179 | +0.098 | 99,814 |
| 7 | `mo_sin_old_rev_tl_op` | −0.140 | −0.120 | 100,000 |
| 8 | `inq_fi` | +0.125 | +0.124 | 100,000 |
| 9 | `mths_since_rcnt_il` | −0.124 | −0.074 | 97,796 |
| 10 | `inq_last_12m` | +0.116 | +0.106 | 100,000 |
| 11 | `mort_acc` | −0.109 | −0.102 | 100,000 |
| 12 | `annual_inc` | −0.106 | −0.058 | 100,000 |
| 13 | `mths_since_last_record` | −0.106 | −0.090 | 9,125 |
| 14 | `delinq_2yrs` | +0.104 | +0.093 | 100,000 |
| 15 | `tot_coll_amt` | +0.089 | +0.027 | 100,000 |

![bivariate hexbin grid](outputs/eda/06_bivariate_hexbin.png)

After FICO, the strongest signals are utilization-based (`all_util`, `revol_util`) and shopping-intensity (`inq_fi`, `inq_last_12m`). Credit-history-length (`mo_sin_old_rev_tl_op`) carries modest negative signal.

![categorical boxplots](outputs/eda/06_categorical_boxplots.png)

`term` is the cleanest categorical separator: 60-month loans carry visibly higher rates than 36-month. `purpose` separates well — small-business and educational purposes price higher than debt consolidation and credit card.

![state bar](outputs/eda/06_state_bar.png)

State spread is 3.24 pp — meaningful enough to keep as a native XGBoost categorical.

---

## 7. Correlation & multicollinearity

![correlation heatmap](outputs/eda/07_correlation_heatmap.png)

The only true multicollinearity issue is `fico_range_low ↔ fico_range_high ↔ fico` — a near-duplicate by construction (`high - low` is 4 for 99,986 rows and 5 for 14 rows). Use only the midpoint. Beyond that, `revol_util ↔ all_util` correlate around 0.7-0.8 (both measure utilization at slightly different scopes) and would benefit from keeping only one in a linear model — but XGBoost handles the redundancy fine.

---

## 8. Leakage audit

This dataset's leakage story differs sharply from the prior `data/` slice:

| Column | Prior `data/` slice | `true data/` slice |
| --- | --- | --- |
| `grade` | present, r=0.92 leakage | **absent** |
| `sub_grade` | present, r=0.94 leakage | **absent** |
| `installment` | present, PMT-derived leakage | **absent** |
| `loan_status` | absent | **present → DROPPED at load** |

`loan_status` is the loan's lifecycle outcome (Current, Fully Paid, Charged Off, Default, In Grace Period, Late N days). At origination, every loan's status is "Current" — any predictive power that `loan_status` shows in EDA comes from the loan's *later* performance correlating with the rate that was set originally. Using it as a feature would inflate accuracy on this static dataset in a way that fails in production.

![loan_status leakage](outputs/eda/08_loan_status_leakage.png)

Audit numbers (computed on a side copy before the column is dropped, then the side copy discarded):

- **Mean-rate spread across `loan_status` levels: 3.42 pp** (highest mean: defaulted; lowest: fully paid)
- **One-way ANOVA F(5, 99994) = 88.69, p < 1e-90**
- **Eta-squared = 0.0044** → `loan_status` explains 0.44% of `int_rate` variance

The magnitude is small compared to the prior dataset's grade-trio leakage (90%+ correlation), but the principle is the same: a column whose value at scoring time is not what's in the training data is leakage. Dropped unconditionally.

The notebook drops `loan_status` at the top of section 1, before any of the bivariate, correlation, drift, or cardinality analysis touches it. The audit in §8 uses a side copy that is then deleted.

---

## 9. Cardinality

| Column | n_unique | dtype | Recommendation |
| --- | --- | --- | --- |
| `tot_cur_bal`, `total_bal_ex_mort`, `revol_bal` | 40k–84k | numeric | KEEP — continuous |
| `emp_title` | 35,337 | str | **DROP** — free text, weak signal |
| `title` | 11 labels | str | **DROP** — one-to-one duplicate of `purpose` |
| `zip_code` | 873 | masked 3-digit | Optional — see §11 |
| `addr_state` | 50 | str | KEEP — native XGBoost categorical |
| `purpose` | 11 | str | KEEP — native categorical |
| `emp_length` | 11 | ordinal-text | ENCODE via `EMP_LENGTH_MAP` |
| `home_ownership` | 3 | str | KEEP — native categorical |
| `verification_status` | 2 | str | KEEP |
| `term` | 2 | str | EXTRACT integer months |
| `application_type` | 2 | str | KEEP or validate — not constant (87/13 split) |

---

## 10. Train vs test drift

Without `issue_d`, drift cannot be attributed to time. Strong drift on a non-temporal split usually indicates the test set was sampled with different weights or rejection-inference treatment. Top 10 by KS D:

| Feature | KS D | PSI | train mean | test mean | Δ mean |
| --- | --- | --- | --- | --- | --- |
| `loan_amnt` | 0.142 | 0.169 | 16,833 | 13,794 | **−3,039** |
| `mths_since_recent_bc` | 0.140 | 0.116 | 24.5 | 28.4 | +3.97 |
| `mths_since_rcnt_il` | 0.078 | 0.072 | 17.3 | 16.1 | −1.28 |
| `fico_range_high` | 0.069 | 0.026 | 710.6 | 716.1 | +5.52 |
| `fico_range_low` | 0.069 | 0.026 | 706.6 | 712.1 | +5.52 |
| `total_acc` | 0.054 | 0.020 | 23.9 | 25.3 | +1.43 |
| `revol_bal` | 0.049 | 0.011 | 19,093 | 17,257 | −1,836 |
| `mths_since_recent_inq` | 0.048 | 0.020 | 7.0 | 7.5 | +0.45 |
| `all_util` | 0.048 | 0.020 | 55.1 | 52.7 | −2.37 |
| `revol_util` | 0.047 | 0.017 | 44.9 | 41.9 | −3.00 |

![drift KDE](outputs/eda/10_drift_kde.png)

**Notably**: `loan_amnt` shifts **down** in test (train mean $16,833 → test mean $13,794, PSI 0.17) and **FICO shifts up** by ~5 points. The test set looks like a more conservative, smaller-loan, higher-credit-quality subset than train — consistent with a stratified sample weighted toward lower-risk applications. This is the opposite direction of the prior dataset's drift (where larger loans appeared in test). It suggests the model will *over*-estimate rates on the test set if it learns the train distribution naïvely.

Categorical chi² results (most-drifted first):

| Feature | chi² | dof | n_levels |
| --- | --- | --- | --- |
| `term` | 587.1 | 1 | 2 |
| `emp_length` | 395.3 | 11 | 12 |
| `verification_status` | 332.6 | 1 | 2 |
| `purpose` | 276.1 | 10 | 11 |
| `application_type` | 149.0 | 1 | 2 |
| `addr_state` | 137.2 | 49 | 50 |
| `home_ownership` | 16.7 | 2 | 3 |

The `verification_status` and `term` shifts are large enough to matter for SHAP attributions in test.

---

## 11. Geographic analysis

State spread = 3.24 pp — non-trivial main effect. Keep `addr_state` as a native XGBoost categorical.

Zip-first-digit spread = 1.04 pp (vs 0.26 pp in the prior dataset's `zip_code` field) — more signal here, but still much smaller than the state-level effect. Of 873 unique `zip3` prefixes, 115 map to >1 state — a masking artifact (the trailing two digits are replaced with `xx`, conflating distinct full ZIPs).

![zip first digit](outputs/eda/11_zip_first_digit.png)

---

## 12. Data quality issues

| Check | n | Recommendation |
| --- | --- | --- |
| `annual_inc` > $1M | small count | Long self-reported tail; XGBoost robust, winsorize for linear |
| `annual_inc` == 0 | (see notebook) | Imputation needed if any |
| `dti` == 0 | small count | Legitimate (zero non-mortgage debt) |
| `dti` > 100 | (see notebook) | Possible encoding artifact; investigate |
| `revol_util` > 150 | small count | Overlimit accounts; legitimate but rare |
| `loan_amnt` > 5× `annual_inc` | small count | Edge of underwriting cap |
| `fico_range_low` outside 600-850 | 0 | All in-range as expected |
| 8 string-encoded numerics | all parsed cleanly with `null_values=["NA"]` | Successful at load |

`emp_title` has 35k unique values — recommend drop (high cardinality, weak signal, no clean grouping without an external NER or industry-mapping step).

---

## 13. Feature engineering recommendations

| Hypothesis | Motivation | Implementation note |
| --- | --- | --- |
| **Use `fico = (low + high)/2`; drop the raw pair** | §5 — perfect linear duplicate | new column, drop originals |
| `log1p` on `annual_inc`, `revol_bal`, `tot_cur_bal`, `total_bal_ex_mort`, `tot_coll_amt` | §4 heavy skew | standard log transform |
| `loan_to_income = loan_amnt / annual_inc` | §12 underwriting signal | mirrors archived `loan_to_income` |
| `inq_per_open_acc` | §6 shopping density | mirrors archived feature |
| Interaction `fico × revol_util` | Credit quality × leverage (§5, §6) | new |
| Zero-inflated flags for `pub_rec`, `delinq_2yrs`, `tot_coll_amt` | §4 zero spikes | binary `*_flag` |
| **`has_derog_record` flag from `mths_since_last_record`** | §3 — 91% NA means "no event ever" | `mths_since_last_record.notna().astype(int)` |
| Similar `has_*` flags for `mths_since_recent_inq`, `mths_since_rcnt_il`, `mths_since_recent_bc` | §3 — same semantics | binary flags |
| `addr_state` native XGBoost categorical | §6, §11 — 3.24 pp spread | `pl.Categorical` |
| Drop `emp_title`, `title` | §9: `emp_title` high cardinality; `title` duplicates `purpose` | in `DROP_COLS` |
| Drop `loan_status` | §8 leakage | already done at load |
| **NOT POSSIBLE: temporal features** | §1 no `issue_d`, no `earliest_cr_line` | — |

---

## 14. Modeling implications & next steps

### Pre-modeling checklist

1. **Drop `loan_status`** (done at load — see §8).
2. Drop `emp_title` (high-cardinality free text) and `title` (one-to-one duplicate of `purpose`).
3. Keep or validate `application_type`; it is not constant in this slice.
4. Parse `term` to integer months; map `emp_length` via `EMP_LENGTH_MAP`.
5. Use `fico = (fico_range_low + fico_range_high) / 2`; drop the two raw columns to avoid perfect collinearity.
6. Apply `log1p` to monetary tails: `annual_inc`, `revol_bal`, `tot_cur_bal`, `total_bal_ex_mort`, `tot_coll_amt`.
7. Add `has_derog_record`, `has_recent_inq`, `has_recent_il`, `has_recent_bc` from the 4 `mths_since_*` columns (treat the NA as "never happened" rather than imputing a number that implies a recent event).
8. Add zero-inflated flags for `pub_rec`, `delinq_2yrs`, `chargeoff_within_12_mths`, `collections_12_mths_ex_med`, `tot_coll_amt`.
9. Fit median imputation on **train only**; apply to val and test.
10. Native XGBoost categoricals for `addr_state`, `home_ownership`, `verification_status`, `purpose`, `term`.
11. **Validation: random 80/20 split** (no `issue_d` available for temporal validation).

### Honest expectations

- With FICO present, validation R² should be materially higher than the FICO-stripped prior slice's 0.52 ceiling, but do not quote a new ceiling until a random-holdout model is run.
- The largest residual signal lives in `all_util`, `revol_util`, and `dti` after FICO is conditioned out.
- **Out-of-time generalization is unmeasured** on this slice. Monitor live performance against expected RMSE — drift can only be detected post-hoc, since the test set already shows substantial *non-temporal* drift on `loan_amnt` (PSI 0.17).

### Risk register

Features with PSI ≥ 0.1 on the train-vs-test split (deserve generalization monitoring): `loan_amnt`, `mths_since_recent_bc`. The `loan_amnt` shift is large (−$3,039 mean) and in the opposite direction from the prior dataset's drift — a model trained on this `true data/` train slice will see different loan-size distributions in deployment than it saw during fit.

---

*Generated from [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). To regenerate, run `.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb`.*
