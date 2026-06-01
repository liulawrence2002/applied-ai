# LendingClub Interest Rate Prediction - ML Pipeline Spec

## Project Structure
```
/mnt/agents/output/lendingclub_project/
├── LC_InterestRate_Prediction.ipynb    # Main notebook (deliverable)
├── predictions.csv                      # Test predictions (deliverable)
├── slides.pdf                           # Slide deck (deliverable)
├── models/                              # Saved model artifacts
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── oof_predictions.csv
├── figures/                             # Generated visualizations
│   ├── feature_importance_shap.png
│   ├── permutation_importance.png
│   ├── pdp_plots.png
│   ├── model_comparison.png
│   ├── residuals.png
│   └── correlation_heatmap.png
└── data/                                # (data read from /mnt/agents/upload/)
```

## Pipeline Architecture

### Stage 1: Data Loading & Leakage Audit
- Load LC_train.csv, LC_test.csv
- Remove leakage features: loan_status, title (redundant with purpose)
- Verify all remaining features are available at origination
- Document leakage audit decisions

### Stage 2: Temporal Train/Validation/Test Split
- Use index as temporal proxy (data ordered by time)
- Split: 70% train / 15% validation / 15% test (temporal order)
- All preprocessing fit on train only, transform on val/test

### Stage 3: Preprocessing Pipeline
- Missing value imputation (train statistics only)
- Categorical encoding (ordinal for ordered, one-hot for nominal)
- Feature engineering (FICO midpoint, log transforms, ratios)
- No target encoding (leakage risk)

### Stage 4: Model Training & Tuning
- Model A: XGBoost with Optuna (100 trials)
- Model B: LightGBM with Optuna (100 trials)
- Model C: CatBoost with Optuna (100 trials)
- Model D: Stacked Ensemble (if improvement)
- Evaluation: RMSE, MAE, MAPE on temporal test set

### Stage 5: Explainability
- SHAP global feature importance (beeswarm plot)
- SHAP local explanations (waterfall plots)
- Permutation importance on test set
- PDP with ICE curves for top 4 features

### Stage 6: Test Predictions
- Apply best model to LC_test.csv
- Format: ID, int_rate (numeric, 0-100)

### Stage 7: Slides
- 8 slides max + 2 appendix
- PDF format

## Anti-Leakage Checklist
- [x] loan_status removed (post-origination outcome)
- [x] title removed (redundant with purpose)
- [x] No grade/sub_grade (directly determine rate)
- [x] Temporal split (not random)
- [x] All preprocessing fit on train only
- [x] No target encoding

## Models & Hyperparameters

### XGBoost
```python
params = {
    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
    'max_depth': trial.suggest_int('max_depth', 3, 10),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1
}
```

### LightGBM
```python
params = {
    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
    'max_depth': trial.suggest_int('max_depth', 3, 12),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    'objective': 'regression',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
```

### CatBoost
```python
params = {
    'iterations': trial.suggest_int('iterations', 100, 2000),
    'depth': trial.suggest_int('depth', 4, 10),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
    'border_count': trial.suggest_int('border_count', 32, 255),
    'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': 0,
    'thread_count': -1
}
```

## Evaluation Metrics
- RMSE (primary - competition metric)
- MAE (tiebreaker)
- MAPE (secondary tiebreaker)
- R² (goodness of fit)
