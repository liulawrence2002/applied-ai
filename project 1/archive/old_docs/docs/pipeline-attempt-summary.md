# Pipeline Attempt Summary

> Log of the automated pipeline execution attempt before data source swap.

## What Was Attempted

An end-to-end LendingClub interest rate prediction pipeline was implemented and partially executed. The user opted to source raw LendingClub data from Kaggle because the custom assignment files (`LC_train.csv`, `LC_test.csv`, `LCDataDictionary.xlsx`) were not present in the workspace.

## Data Acquisition

- **Source**: KaggleHub dataset `wordsforthewise/lending-club`
- **Raw file**: `accepted_2007_to_2018Q4.csv.gz` (~1.26 GB)
- **Sampling**: 110,000 rows sampled from 2.26M accepted loans
  - Train: 100,000 loans (2007–2017)
  - Test: 10,000 loans (2018)
- **Schema**: 39 variables selected to approximate the assignment description

## Pipeline Architecture Implemented

| Phase | Status | Notes |
|-------|--------|-------|
| **Phase 0: Data Audit & Leakage Surgery** | ✅ Completed | Dropped `installment` (PMT correlation = 0.9998), `grade` (corr = 0.9531), `sub_grade` (corr = 0.9771). Temporal sanity check passed. |
| **Phase 1: Feature Engineering** | ✅ Completed | 15 engineered features including `loan_to_income`, `fico_x_term`, `revol_util_x_fico`, `inq_intensity`, `home_own_x_emp_length`, plus categorical encodings. |
| **Phase 2: Baseline Models** | ✅ Completed | Naive Mean RMSE: 5.09; OLS Core RMSE: 4.09; XGBoost Default RMSE: 3.77 |
| **Phase 3: Optuna Optimization** | ✅ Completed | 15 trials; best RMSE: **3.7507** (marginal 0.5% improvement over default). Best params: max_depth=4, lr=0.013, n_estimators=2211. |
| **Phase 4: Diagnostics** | ✅ Completed | Residual plots and XGBoost gain importance saved. |
| **Phase 5: SHAP** | ✅ Completed | SHAP summary and bar plots saved. Top features: `term_numeric`, `fico_range_low`, `purpose_enc`, `loan_to_income`. No leakage detected in top 3. |
| **Phase 6: Submission** | ❌ Not reached | Pipeline aborted before test predictions due to user request to revert and swap data source. |

## Key Results

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Naive Mean | 5.0868 | 4.1250 | -0.0170 |
| OLS (Core 4) | 4.0862 | 3.1354 | 0.3438 |
| XGBoost Default | 3.7682 | 2.8600 | 0.4420 |
| XGBoost + Optuna | **3.7507** | **2.8446** | **0.4471** |

## Why the User Is Swapping Data

The Kaggle-sourced data covers **2007–2018Q4**, while the assignment specifies **2007–2020**. Additionally, the custom assignment files likely have a curated 39-variable schema and a specific train/test chronological split that differs from our programmatic sampling. To match the assignment exactly, the user will provide the original `LC_train.csv` and `LC_test.csv` files.

## Files Generated (Now Deleted)

- `lendingclub_pipeline.py` — full Python pipeline script
- `LC_train.csv` / `LC_test.csv` — sampled Kaggle data
- `submission.csv` — never generated
- `xgb_model.json` — never generated
- `leakage_audit_log.txt` — never generated
- `phase0_int_rate_over_time.png`
- `phase4_diagnostics.png`
- `phase4_xgb_importance.png`
- `phase5_shap_summary.png`
- `phase5_shap_bar.png`
- `archive/loan.csv` — intermediate raw data
- `archive/LCDataDictionary.xlsx` — placeholder data dictionary

## Files Retained

- `loan-interest-rate-prediction-research.md`
- `lstm-xgboost-deep-dive.md`
- `interaction-terms-and-economic-theory.md`
- `image.png` — assignment instructions screenshot

## Next Steps (for user-provided data)

1. Place custom `LC_train.csv`, `LC_test.csv`, and `LCDataDictionary.xlsx` in the project root.
2. Re-run a cleaned version of the pipeline script (without Kaggle download/sampling logic).
3. The pipeline script is structurally sound; only the data loader needs to change.
4. Consider increasing Optuna trials to 30–50 if runtime allows, or use the best default+features model which already performs well.
