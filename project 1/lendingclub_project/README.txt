# LendingClub Interest Rate Prediction - Project Deliverables

## Files Included

### 1. Predictions CSV (REQUIRED)
- **predictions.csv** - Test predictions with columns: ID, int_rate
  - Rename to YOUR_SECTION+GROUP.csv (e.g., HOYA1.csv)
  - int_rate values are numeric percentages between 0 and 100

### 2. Slide Deck PDF (REQUIRED)
- **slides.pdf** - 8 slides + 2 appendix slides
  - Slide 1: Title slide (update team info)
  - Slide 2: Problem Statement & Approach
  - Slide 3: Data Processing & Anti-Leakage Measures
  - Slide 4: Modeling Approach
  - Slide 5: Model Comparison Results
  - Slide 6: SHAP Explainability
  - Slide 7: Business Interpretation
  - Slide 8: Conclusion & Lessons Learned
  - Appendix A: Partial Dependence Plots
  - Appendix B: Model Diagnostics (Residuals)

### 3. Python Notebook (REQUIRED)
- **LC_InterestRate_Prediction.ipynb** - Fully annotated, reproducible code
  - Loads LC_train.csv from working directory
  - Runs complete pipeline from preprocessing to predictions
  - All steps clearly documented with markdown cells

## Model Performance

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|-----|
| XGBoost | 3.940 | 2.961 | 23.76% | 0.448 |
| LightGBM | 3.932 | 2.959 | 23.73% | 0.450 |
| CatBoost | 3.930 | 2.962 | 23.81% | 0.451 |
| **Simple Average (BEST)** | **3.912** | **2.942** | **23.61%** | **0.456** |

## Anti-Leakage Measures
- loan_status REMOVED (post-origination outcome)
- title REMOVED (redundant with purpose)
- Temporal split (70/15/15) - no random shuffle
- All preprocessing fit on training data only
- No target encoding used

## Explainability Techniques Used
1. **SHAP (TreeSHAP)**: Global beeswarm plot + local waterfall plot
2. **Permutation Importance**: Model-agnostic feature validation
3. **Partial Dependence Plots**: Marginal effects for top 4 features

## Instructions for Renaming
Update the following in slides.pdf and predictions.csv:
- [Your Team Number] -> Your actual team number
- [Member Names] -> Your team member names
- [Your Section] -> Your section (e.g., HOYA)
- [Your Group Number] -> Your group number
- Rename predictions.csv to SECTIONGROUP.csv format (e.g., HOYA1.csv)
