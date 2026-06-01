# LendingClub Interest-Rate Prediction — Findings

End-of-project notes summarizing what the data supports, what it doesn't, and where the model ended up. Read this before sinking more time into the pipeline.

## Final numbers (honest temporal validation)

| Metric | Baseline (untuned XGB) | Tuned (Optuna 100 trials) |
|---|---|---|
| **RMSE** | 2.99 | **2.97** |
| **MAE** | 2.27 | 2.28 |
| **R²** | 0.510 | 0.516 |

Validation set: the chronologically-last 20% of `LC_train.csv` (mid-2012 → 2013), held out from training. Test predictions on `LC_test.csv` (10k loans, 2013) are in `outputs/test_predictions.csv` — `int_rate_pred` summary `min=5.3, max=24.6, mean=14.8, std=3.5`, which matches the empirical range of `int_rate` in LendingClub data.

## What we proved during the project

1. **Three columns are textbook leakage** and were dropped:
   - `grade` (corr with `int_rate` ≈ 0.92)
   - `sub_grade` (corr ≈ 0.94)
   - `installment` — algebraic function of the target: `installment = PMT(loan_amnt, term, int_rate)`. The marginal correlation looks innocent (~0.34) because installment is jointly determined by `loan_amnt`, `term`, and the target. Earlier pipeline runs that kept it produced misleadingly low RMSE (~1.4) and SHAP confirmed `installment` was the #1 feature — a tell.

2. **Random validation splits on this data are dishonest.** The source CSV is sorted by `issue_d`. A `train_test_split` with `random_state=42` mixes 2007-2013 loans into both folds, leaking macro information. Switching to a temporal tail split raised honest RMSE from ~1.43 (random) → ~2.99 (temporal) — the 1.43 was a lie.

3. **Hyperparameter tuning is not the bottleneck.** A 100-trial Optuna TPE study with MedianPruner and `XGBoostPruningCallback` over an 8-dimensional search space moved RMSE from 2.989 → 2.971 — a 0.018 improvement (0.6%). The TPE search is doing real work, but the surface is flat — every reasonable hyperparameter setting produces nearly the same result. The signal ceiling is determined by the features, not the model.

4. **The data dictionary describes columns that aren't in our data.** `LCDataDictionary.xlsx` documents ~150 columns including FICO (`fico_range_low`, `fico_range_high`), `bc_util`, `avg_cur_bal`, `mths_since_last_delinq`, `inq_last_12m`, etc. Of these, only ~30 of the documented columns actually appear in `LC_train.csv`; the rest were stripped by whoever produced this CSV.

5. **FICO is gone from the upstream Kaggle dump too.** We checked the raw 145-column `data/archive/loan.csv` — there are zero columns matching `fico`, `score`, or `credit_score`. LendingClub's public data releases strip FICO because TransUnion (the supplier) doesn't allow free redistribution of FICO-band data. The data dictionary lists FICO columns as a courtesy; they aren't recoverable from this source.

6. **`issue_d` is the largest empirical signal we can't use.** Mean `int_rate` rises from 11.83% (2007) → 13.64% (2012) → 14.36% (2013) — a 2.5 pp macro drift. Adding `issue_year` as a feature looks attractive on paper, but under temporal validation the year is effectively a constant in val (year=2012/2013) that the model only saw values ≤2012 of in train. Trees can't extrapolate; adding the feature didn't move validation RMSE. **The 2.5 pp macro drift exists in the data but is unrecoverable under honest temporal validation.**

7. **`addr_state` and `zip_code` are nearly flat marginally.** State means span 0.40 percentage points; zip-first-digit spans 0.27 pp. We added `addr_state` as a native XGBoost categorical (in case sub-population interactions exist); zip stays dropped. Marginal lift was zero.

8. **Target skewness is mild** (skew = 0.288 on the full 100k). Box-Cox with λ ≈ 0.52 or sqrt would bring skew to ~0, but the expected RMSE gain is <0.05. Not worth the bookkeeping of inverse-transforming predictions.

9. **The new engineered features the dictionary suggested mostly didn't help.** We added `acc_age_ratio`, `delinq_severity`, `inq_per_open_acc`, and 6 zero-inflation flags. Two of the new interactions made the SHAP top 10 (`inq_per_open_acc`, `delinq_severity`), proving they're being used — but RMSE moved by less than 0.01. The information was already learnable from existing columns at sufficient tree depth.

## What the model actually learns from

SHAP top 10 (final tuned model, 100 trials):

```
term_numeric                 1.13   # 60-month loans cost meaningfully more than 36-month
loan_amnt_x_revol_util       0.75   # loan size × revolving utilization
percent_bc_gt_75_x_revol_util 0.66  # concentration × utilization
revol_util                   0.64   # raw revolving utilization
inq_per_open_acc             0.47   # credit-seeking intensity (NEW)
revol_util_x_open_acc        0.36   # utilization × number of open accounts
credit_history_length        0.35   # months from earliest credit line to issuance
loan_amnt                    0.33   # raw loan size
pct_tl_nvr_dlq               0.33   # percent of trades never delinquent
delinq_severity              0.33   # weighted delinquency aggregate (NEW)
```

Six of the top 10 involve revolving credit (utilization, open accounts, percent of bankcards >75% utilized). The model is essentially a "revolving credit health" scorer plus loan-term length, with credit-history-length as a tie-breaker. This matches what the 5 C's framework would predict: in the absence of FICO, the next-best risk signals are utilization and account-history breadth.

## Honest assessment

RMSE 2.97 on a 5-25% target = predictions typically off by ~3 percentage points, sometimes 5+. R² 0.52 means the model explains slightly more than half of the variance. This is mid-tier for an academic exercise and **at the ceiling for what this column set can deliver** without FICO.

For comparison, peer-reviewed LendingClub interest-rate models that include FICO routinely report R² of 0.85-0.95 because grade/sub_grade and FICO together encode 90%+ of LC's own pricing formula. Strip those out (legitimately, since they're either the target itself in disguise or supplied by the lender) and you land in the 0.45-0.60 range. We're at 0.516 — right where the literature predicts a leakage-free, FICO-free model should sit.

## What could improve this further (and what realistically won't)

### Would likely help (if you want to extend the project)

1. **Regenerate `LC_train.csv` / `LC_test.csv` from `data/archive/loan.csv` with a wider column slice.** The raw 145-column file has signals the current 39-column slice doesn't:
   - `mths_since_last_delinq`, `mths_since_last_major_derog`, `mths_since_recent_inq`, `mths_since_recent_bc_dlq`, `mths_since_recent_revol_delinq` — recency of bad events is consistently stronger than counts.
   - `bc_util`, `avg_cur_bal`, `tot_cur_bal`, `tot_hi_cred_lim`, `total_bal_ex_mort`, `bc_open_to_buy`, `max_bal_bc`, `all_util`, `il_util` — additional utilization/balance detail.
   - `num_actv_rev_tl`, `num_actv_bc_tl`, `num_bc_sats`, `num_op_rev_tl`, `num_rev_accts`, `num_sats`, `num_rev_tl_bal_gt_0` — account composition.
   - `inq_last_12m`, `inq_fi`, `open_acc_6m`, `acc_open_past_24mths`, `num_tl_op_past_12m`, `open_rv_12m/24m`, `open_il_12m/24m` — credit-seeking over multiple windows.

   Realistic ceiling with these added: RMSE 2.5-2.7 / R² 0.55-0.65. Solid but not transformational.

2. **External macro joins.** Map `issue_d` → 10Y Treasury yield or Fed Funds rate at issuance. A *continuous* macro variable generalizes to out-of-distribution years in a way that an integer year does not. Map `addr_state` → state unemployment rate, HPI growth at origination. Would require pulling FRED data and a state-time merge.

3. **Quantile regression.** `reg:quantileerror` with multiple quantile heads (e.g., 0.1 / 0.5 / 0.9) gives you confidence bands instead of a point estimate. Doesn't improve RMSE but improves *decision utility*.

### Won't move the needle

- More Optuna trials. The tuning surface is flat; we've seen baseline (default) within 0.02 RMSE of best-tuned.
- More tree depth / wider models. Already capped at max_depth=10; tuning consistently picks the cap, suggesting we'd just memorize training.
- LSTM / neural nets. The data is tabular and at the small scale (80k rows train); XGBoost is at or near optimal for this regime.
- More interaction features. We tested 10+ economically-motivated interactions; the marginal lift per added feature is ≤0.005 RMSE.

### Don't do

- **Don't reintroduce `installment`, `grade`, or `sub_grade`.** RMSE will look better but the model is solving a degenerate problem.
- **Don't switch back to a random validation split.** The honest temporal split is what makes the test-set predictions meaningful.

## Files of record

- `src/preprocessing_and_modeling.py` — final pipeline (Polars + native XGBoost + Optuna TPE).
- `src/create_train_test_split.py` — one-time script that produced `LC_train.csv` / `LC_test.csv` from `data/archive/loan.csv`. Don't re-run unless you're regenerating the data.
- `outputs/xgb_final.json` — saved XGBoost booster (load with `xgb.Booster().load_model(...)`).
- `outputs/test_predictions.csv` — final predictions on the 10k-row held-out test set.
- `outputs/shap_summary.png` — SHAP summary plot for the final model.
- `docs/interaction-terms-and-economic-theory.md` — the economic theory we drew from.
- `docs/loan-interest-rate-prediction-research.md` — SOTA ML survey for this problem.
- `docs/pipeline-attempt-summary.md` — log of an earlier abandoned attempt (Kaggle-sourced data, kept for reference).
- `data/archive/LCDataDictionary.xlsx` — official LendingClub data dictionary (describes ~150 columns; only ~30 actually present in `LC_train.csv`).
- `data/archive/loan.csv` — 1.2 GB raw Kaggle dump (145 columns; source for the column slice).
