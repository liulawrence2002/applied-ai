# LendingClub Interest Rate Prediction - Project Plan

## Goal
Build a state-of-the-art ML pipeline to predict loan interest rates on the LendingClub dataset, with zero data leakage, no temporal bias, and minimized RMSE/MAE/MAPE. Deliver: predictions CSV, slide deck (PDF), and clean Python code.

## Stage 1: Data Exploration & Research (Parallel)
- **1A**: Load data, profile distributions, identify target leakage risks, understand temporal structure
- **1B**: Research SOTA approaches for LendingClub interest rate prediction via deep-research-swarm
- **1C**: Research best practices for no-leakage, no-temporal-bias loan pricing models
- Output: Data profile report, research findings, feature availability audit

## Stage 2: Data Preprocessing & Feature Engineering
- Filter features to those available at loan origination (no leakage)
- Handle missing values (imputation strategy per feature type)
- Encode categoricals (target encoding, ordinal, one-hot as appropriate)
- Feature engineering (ratios, bins, interactions, domain-knowledge features)
- Temporal train/validation/test split strategy
- Output: Clean train/val/test sets, preprocessing pipeline

## Stage 3: Model Training & Hyperparameter Tuning
- Model 1: Gradient Boosting (XGBoost/LightGBM/CatBoost) with Optuna tuning
- Model 2: Neural Network / TabNet or Random Forest with tuning
- Model 3: Ensemble / Stacking if beneficial
- Time-aware cross-validation
- Output: Trained models, performance comparison table, best model selected

## Stage 4: Explainability Analysis
- SHAP values for global and local feature importance
- Permutation importance for validation
- Partial dependence plots for key features
- Output: Explainability visualizations, business interpretation

## Stage 5: Test Prediction Generation
- Apply best model to LC_test.csv
- Format output as required (ID, int_rate)
- Validate output format
- Output: SECTIONNAMEGROUPNUMBER.csv

## Stage 6: Slide Deck Creation (pptx-swarm skill)
- 8 slides max + 2 appendix slides
- Title slide, Problem Statement, Data Processing, Modeling Approach, 
  Model Comparison, Explainability, Conclusion, Lessons Learned
- Output: PDF slides

## Stage 7: Code Packaging & Final Deliverables
- Clean, annotated Jupyter notebook
- Ensure reproducibility (single entry point, reads LC_train.csv)
- Package all deliverables

## Skills Used
- deep-research-swarm: Route B - Focused research on SOTA approaches
- vibecoding-general-swarm: ML pipeline coding
- pptx-swarm: Slide deck creation

## Anti-Leakage Checklist
- [ ] Remove post-origination variables (loan_status, collections after origination, etc.)
- [ ] Remove variables derived from target or future information
- [ ] Verify all features are known at application time
- [ ] Use time-based splits (not random)
- [ ] No target encoding leakage (fit on train only)

## Anti-Temporal-Bias Checklist
- [ ] Sort by loan issue date before splitting
- [ ] Train on older loans, validate on newer loans
- [ ] Check for temporal distribution shifts
- [ ] Evaluate model stability across time periods
