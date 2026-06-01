# CatBoost Model — Feature Engineering Reference

Documents **every engineered feature** produced for the final CatBoost model
(`final/final_catboost.ipynb`). All features are built in
`final_pipeline.engineer()` (stateless, applied identically to train and test)
plus `final_pipeline.add_target_encodings()` (refit inside each CV fold).

- **Raw inputs:** 39-column `true data/` schema, minus `loan_status` (leakage, dropped at load).
- **New features created:** **36** (32 stateless + 4 leakage-safe target encodings).
- **Raw columns dropped after engineering:** `fico_range_low`, `fico_range_high`,
  `term`, `emp_length`, `title`, `zip_code` (replaced by engineered equivalents).

Each new feature below lists its **formula**, **why** it was created, and the
**EDA finding** it responds to (see `EDA_REPORT.md`).

---

## 1. Credit-quality / FICO features (3)

FICO is the single strongest non-leakage signal (Spearman −0.46). The raw pair is
collapsed and re-expressed several ways.

| Feature | Formula | Rationale |
| --- | --- | --- |
| `fico` | `(fico_range_low + fico_range_high) / 2` | The two raw columns differ by only 4–5 points (perfect collinearity); the midpoint is the clean single signal. Replaces both raw columns. |
| `fico_band` | `cut(fico, [-∞,660,680,700,720,740,760,780,∞])` | Discretizes FICO into 8 risk tiers; lets the model capture the steeper rate slope below 720 the EDA flagged as piecewise-linear. |
| `fico_range_width` | `fico_range_high − fico_range_low` | The 4-vs-5 width is a minor scoring-model artifact; kept as a low-cost flag in case it encodes which FICO model version scored the applicant. |

`fico` and `fico_band` are consistently the #1 and #2 features by both CatBoost
gain and permutation importance.

---

## 2. Parsed / re-encoded raw columns (3)

Raw text fields converted to model-usable form.

| Feature | Formula | Rationale |
| --- | --- | --- |
| `term_months` | integer extracted from `term` (`"36 months"` → 36) | `term` is a clean 2-level categorical separator (60-month loans price higher); numeric form enables interactions. Replaces `term`. |
| `emp_length_num` | ordinal map `"< 1 year"→0 … "10+ years"→10` | Preserves the ordering employment length implies. NA left as null (informative). Replaces `emp_length`. |
| `zip3` | first 3 digits of `zip_code` | Coarse geographic signal (zip-first-digit spread ~1 pp). Feeds target encoding; replaces masked `zip_code`. |

---

## 3. Missingness flags (4)

The `mths_since_*` columns are **informatively missing** ("event never happened",
not "value unknown"). EDA §3 showed `mths_since_last_record` is 90.9% null. Each
gets a binary "did this event ever occur" flag instead of imputing a number that
would falsely imply a recent event.

| Feature | Formula |
| --- | --- |
| `has_mths_since_last_record` | `mths_since_last_record.notna()` |
| `has_mths_since_recent_inq` | `mths_since_recent_inq.notna()` |
| `has_mths_since_rcnt_il` | `mths_since_rcnt_il.notna()` |
| `has_mths_since_recent_bc` | `mths_since_recent_bc.notna()` |

The raw `mths_since_*` values are also kept (CatBoost handles NaN natively).

---

## 4. Affordability / leverage ratios (5)

Underwriting-style ratios that normalize loan size against income and capacity —
the EDA's underwriting-signal recommendations (§12–13).

| Feature | Formula | Rationale |
| --- | --- | --- |
| `loan_to_income` | `loan_amnt / annual_inc` | Core affordability ratio; how big the loan is relative to income. |
| `installment_proxy` | `loan_amnt / term_months` | Stand-in for the absent `installment` column — approximate monthly principal. |
| `payment_to_income` | `(loan_amnt / term_months) / (annual_inc / 12)` | Monthly-payment-to-monthly-income (DTI-of-this-loan); the affordability ratio lenders price on. |
| `bal_to_income` | `tot_cur_bal / annual_inc` | Total existing debt burden relative to income. |
| `acc_open_ratio` | `open_acc / total_acc` | Share of accounts currently open — credit-activity proxy. |

(`annual_inc == 0` mapped to NaN before division to avoid infinities.)

---

## 5. Utilization & DTI banding (3)

Discretize the strongest post-FICO signals (utilization +0.30/+0.33, dti +0.18)
into risk bands so the model can split on regime boundaries cleanly.

| Feature | Formula |
| --- | --- |
| `dti_band` | `cut(dti, [-∞,10,20,30,40,∞])` |
| `revol_util_band` | `cut(revol_util, [-∞,30,50,75,100,∞])` |
| `all_util_band` | `cut(all_util, [-∞,30,50,75,100,∞])` |

---

## 6. Interaction features (6)

Cross-terms encoding the economic interactions the EDA hypothesized — credit
quality × leverage, and credit quality × loan structure.

| Feature | Formula | Rationale |
| --- | --- | --- |
| `util_x_fico` | `revol_util × fico` | Credit quality × revolving leverage — a high-FICO borrower running hot on cards is priced differently. |
| `dti_x_fico` | `dti × fico` | Credit quality × debt burden interaction. |
| `fico_x_term` | `fico × term_months` | Credit quality × loan duration; risk of a long 60-month term depends on borrower quality. Top-10 feature by importance. |
| `fico_x_dti_band` | `fico × dti_band` | Credit quality × discretized debt burden. |
| `term_x_loan_amnt` | `term_months × loan_amnt` | Total exposure (size × duration). |
| `term_x_dti` | `term_months × dti` | Duration × debt burden; a strong feature (top 3 by importance). |

---

## 7. Risk composites (2)

Hand-built aggregates of sparse, zero-inflated derogatory/inquiry counters.

| Feature | Formula | Rationale |
| --- | --- | --- |
| `derog_score` | `8·delinq_2yrs + 13·pub_rec + 22·pub_rec_bankruptcies + 15·chargeoff_within_12_mths + 10·collections_12_mths_ex_med` | Weighted severity composite of derogatory events; concentrates several sparse zero-inflated counters into one continuous risk score (weights approximate event severity). |
| `inq_intensity` | `inq_last_12m + inq_fi` | Combined credit-shopping intensity; both raw inquiry counts carried positive signal in EDA §6. |

---

## 8. Credit-history length (1)

| Feature | Formula | Rationale |
| --- | --- | --- |
| `credit_file_age_yrs` | `mo_sin_old_rev_tl_op / 12` | Oldest revolving account age in years — credit-history length (modest negative signal, Spearman −0.14). |

---

## 9. Log-transformed monetary tails (5)

EDA §4 flagged five heavily right-skewed monetary columns (skew > 5).
`log1p` compresses the tail; raw columns are also retained.

| Feature | Formula |
| --- | --- |
| `log1p_annual_inc` | `log1p(clip(annual_inc, ≥0))` |
| `log1p_revol_bal` | `log1p(clip(revol_bal, ≥0))` |
| `log1p_tot_cur_bal` | `log1p(clip(tot_cur_bal, ≥0))` |
| `log1p_total_bal_ex_mort` | `log1p(clip(total_bal_ex_mort, ≥0))` |
| `log1p_tot_coll_amt` | `log1p(clip(tot_coll_amt, ≥0))` |

---

## 10. Target encodings (4) — leakage-safe

High-cardinality categoricals are encoded with **smoothed, out-of-fold K-fold
target encoding** (`add_target_encodings`), refit **inside each CV fold** so no
target information leaks across folds.

| Feature | Source column | Cardinality |
| --- | --- | --- |
| `addr_state_te` | `addr_state` | 50 |
| `purpose_te` | `purpose` | 11 |
| `zip3_te` | `zip3` | ~870 |
| `emp_title_te` | `emp_title` | 35k (capped to top-100 + "Other"/"Missing" before encoding) |

**Encoding formula** (smoothing toward the global mean, `m = 20`):

```
enc(category) = (mean_y(category) · count + global_mean · m) / (count + m)
```

- Out-of-fold for the training rows: a 5-fold inner split assigns each row an
  encoding computed only from the *other* folds.
- Validation/test rows use a full-train mapping; unseen categories fall back to
  the global mean.

This gives high-cardinality fields (state, zip, job title) usable numeric signal
without the optimistic bias of naïve in-fold target encoding.

---

## 11. How CatBoost consumes the features

`to_catboost_pool()` separates columns into two groups:

- **Native categoricals** (passed to CatBoost as `cat_features`, NaN → `"Missing"`):
  `application_type`, `home_ownership`, `verification_status`, `purpose`,
  `addr_state`, `zip3`, `emp_title`.
- **Everything else** cast to `float64` (CatBoost handles NaN natively, so raw
  `mths_since_*` and ratio NaNs are left in place alongside their `has_*` flags).

Note CatBoost uses the high-cardinality raw categoricals **and** their `_te`
encodings together — its ordered-boosting target statistics complement the
explicit smoothed encodings.

---

## 12. Feature summary

| Group | Count | Examples |
| --- | --- | --- |
| FICO / credit-quality | 3 | `fico`, `fico_band`, `fico_range_width` |
| Parsed / re-encoded | 3 | `term_months`, `emp_length_num`, `zip3` |
| Missingness flags | 4 | `has_mths_since_last_record`, … |
| Affordability ratios | 5 | `loan_to_income`, `payment_to_income`, … |
| Utilization / DTI bands | 3 | `dti_band`, `revol_util_band`, `all_util_band` |
| Interactions | 6 | `fico_x_term`, `term_x_dti`, `util_x_fico`, … |
| Risk composites | 2 | `derog_score`, `inq_intensity` |
| Credit-history length | 1 | `credit_file_age_yrs` |
| Log monetary tails | 5 | `log1p_annual_inc`, … |
| Target encodings | 4 | `addr_state_te`, `purpose_te`, `zip3_te`, `emp_title_te` |
| **Total new features** | **36** | |

The engineered features the model leans on most (by CatBoost gain & permutation
importance) are `fico`, `fico_band`, `term_x_dti`, `fico_x_term`, and the
target-encoded categoricals — confirming that the FICO re-expressions and the
credit-quality × loan-structure interactions carry the engineered signal.
