# Data Leakage Prevention and Temporal Bias Elimination for Loan Interest Rate Prediction

## Comprehensive Research Findings

---

## 1. Feature Availability at Loan Origination

### Critical Principle
> "A major goal is to keep the prediction problem realistic: keep only features available at origination; avoid post-origination variables (payments, recoveries, etc.)." - LendingClub Default Risk Screening Research

### 1.1 loan_status - KNOWN AT ORIGINATION? **NO - POST-ORIGINATION**

**VERDICT: `loan_status` is a POST-ORIGINATION variable and introduces SEVERE data leakage if used as a feature.**

| Aspect | Details |
|--------|---------|
| **Definition** | "Current status of the loan" - LendingClub Data Dictionary |
| **Values** | Current, Fully Paid, Charged Off, Late (16-30 days), Late (31-120 days), Default, In Grace Period |
| **Why it's post-origination** | These statuses describe the loan's **performance after funding**. "Current" means payments are up to date (post-origination tracking). "Fully Paid" means all payments completed (post-origination). "Charged Off" means the loan was written off as a loss (post-origination). |
| **Impact if used** | Using `loan_status` as a feature for interest rate prediction is **circular**: `loan_status` is a *consequence* of the interest rate, not a predictor of it. Higher interest rates are assigned to riskier borrowers, who then have higher rates of "Charged Off" or "Late" status. |
| **Risk** | **CRITICAL LEAKAGE** - This would create perfect-looking correlations that are entirely artifacts of the temporal sequence. |

**When can loan_status be used?** ONLY as a target variable for default prediction models (trained on historical loans to predict outcomes of future loans). NEVER as a feature for interest rate prediction or origination-time decisions.

### 1.2 title vs. purpose - BOTH AVAILABLE AT APPLICATION

| Feature | Availability | Description |
|---------|-------------|-------------|
| `purpose` | **Origination** | "A category provided by the borrower for the loan request" - LendingClub Data Dictionary. Standardized categories like 'debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase'. |
| `title` | **Origination** | "The loan title provided by the borrower" - LendingClub Data Dictionary. Free-text description of the loan purpose. |

**Key Differences:**
- `purpose` is a **standardized categorical variable** with a fixed set of values
- `title` is a **free-text field** that borrowers fill in (e.g., "Debt consolidation", "Credit Card Consolidation", "Medical expenses")
- `title` is essentially a user-provided description of `purpose`
- Both are collected **at application time** and are safe to use as features

**Best Practice:** `purpose` and `title` are highly correlated. Using both may be redundant. `purpose` is preferred because it's standardized and has fewer unique categories. `title` can be used for NLP feature extraction if the text adds information beyond `purpose`.

### 1.3 mths_since_* Features - AVAILABLE AT ORIGINATION (Credit Report)

All `mths_since_*` features in LendingClub data come from the **credit report pulled at application time** and are available at origination:

| Feature | Definition | Available at Origination? |
|---------|-----------|---------------------------|
| `mths_since_last_record` | "The number of months since the last public record" | **YES** - from credit report at application |
| `mths_since_rcnt_il` | "Months since most recent installment accounts opened" | **YES** - from credit history at application |
| `mths_since_recent_bc` | "Months since most recent bankcard account opened" | **YES** - from credit report |
| `mths_since_recent_inq` | "Months since most recent inquiry" | **YES** - from credit report at application |
| `mths_since_recent_bc_dlq` | "Months since most recent bankcard delinquency" | **YES** - from credit report |
| `mths_since_recent_revol_delinq` | "Months since most recent revolving delinquency" | **YES** - from credit report |

**Important Note:** These features come from the **credit report pulled at the time of application**. LendingClub pulls the borrower's credit report during underwriting, and these values represent the state of the credit file at that moment. They are **not** updated after origination.

**Handling Missing Values:** Missing values in `mths_since_*` features typically mean "never happened" (e.g., no delinquencies = no months since last delinquency). These should be imputed with a large number or handled with an indicator variable for "no event."

### 1.4 chargeoff_within_12_mths - AVAILABLE AT ORIGINATION

| Aspect | Details |
|--------|---------|
| **Definition** | "Number of charge-offs within 12 months" - LendingClub Data Dictionary |
| **Source** | Credit report at time of application |
| **Time Reference** | "within 12 months" refers to **the 12 months prior to the credit pull date** (application time) |
| **Available at Origination?** | **YES** |

**Critical Distinction:** This feature captures the borrower's **pre-application charge-off history** - how many of their accounts were charged off in the 12 months before applying. It does NOT refer to charge-offs of the current loan being applied for. This is legitimate historical data available at origination time.

**Risk Note:** If the dataset contains `chargeoff_within_12_mths` values that update after origination, those would be post-origination and must be excluded. Ensure you're using the origination-snapshot values only.

### 1.5 Complete Feature Classification for Your Dataset

| Feature | Category | Available at Origination? | Risk Level |
|---------|----------|---------------------------|------------|
| `addr_state` | Application info | **YES** | Low |
| `all_util` | Credit report | **YES** | Low |
| `annual_inc` | Application info | **YES** | Low |
| `application_type` | Application info | **YES** | Low |
| `chargeoff_within_12_mths` | Credit report (pre-app history) | **YES** | Low |
| `collections_12_mths_ex_med` | Credit report | **YES** | Low |
| `delinq_2yrs` | Credit report | **YES** | Low |
| `dti` | Application info | **YES** | Low |
| `emp_length` | Application info | **YES** | Low |
| `emp_title` | Application info | **YES** | Low |
| `fico_range_high` | Credit report (at origination) | **YES** | Low |
| `fico_range_low` | Credit report (at origination) | **YES** | Low |
| `home_ownership` | Application info | **YES** | Low |
| `inq_fi` | Credit report | **YES** | Low |
| `inq_last_12m` | Credit report | **YES** | Low |
| `loan_amnt` | Application info | **YES** | Low |
| **loan_status** | **Post-origination performance** | **NO** | **CRITICAL** |
| `mo_sin_old_il_acct` | Credit report | **YES** | Low |
| `mo_sin_old_rev_tl_op` | Credit report | **YES** | Low |
| `mort_acc` | Credit report | **YES** | Low |
| `mths_since_last_record` | Credit report | **YES** | Low |
| `mths_since_rcnt_il` | Credit report | **YES** | Low |
| `mths_since_recent_bc` | Credit report | **YES** | Low |
| `mths_since_recent_inq` | Credit report | **YES** | Low |
| `open_acc` | Credit report | **YES** | Low |
| `pub_rec` | Credit report | **YES** | Low |
| `pub_rec_bankruptcies` | Credit report | **YES** | Low |
| `purpose` | Application info | **YES** | Low |
| `revol_bal` | Credit report | **YES** | Low |
| `revol_util` | Credit report | **YES** | Low |
| `term` | Application info | **YES** | Low |
| `title` | Application info | **YES** | Low |
| `tot_coll_amt` | Credit report | **YES** | Low |
| `tot_cur_bal` | Credit report | **YES** | Low |
| `total_acc` | Credit report | **YES** | Low |
| `total_bal_ex_mort` | Credit report | **YES** | Low |
| `verification_status` | Application info | **YES** | Low |
| `zip_code` | Application info | **YES** | Low |
| `int_rate` | **TARGET VARIABLE** | Determined at origination | N/A |

### 1.6 Post-Origination Variables to ALWAYS Exclude

The following LendingClub variables (not in your current dataset but commonly found in full datasets) are **post-origination and must be excluded**:

| Variable | Why it's post-origination |
|----------|---------------------------|
| `out_prncp` / `out_prncp_inv` | Outstanding principal - changes as borrower pays |
| `total_pymnt` / `total_pymnt_inv` | Total payments received - post-origination |
| `total_rec_prncp` | Principal received to date - post-origination |
| `total_rec_int` | Interest received to date - post-origination |
| `total_rec_late_fee` | Late fees received - post-origination |
| `recoveries` | Recovery amount after charge-off - post-origination |
| `collection_recovery_fee` | Collection recovery fee - post-origination |
| `last_pymnt_d` / `last_pymnt_amnt` | Last payment information - post-origination |
| `next_pymnt_d` | Next scheduled payment - post-origination |
| `last_credit_pull_d` | Date of most recent credit pull - updated post-origination |
| `last_fico_range_high` / `last_fico_range_low` | Updated FICO score - post-origination |

---

## 2. Common Data Leakage Pitfalls in Loan Pricing Models

### 2.1 Using Future Information

**The Golden Rule:** "For time series or any temporal data, always train on the past and validate/test on the future to simulate the actual deployment." - GTR Academy

| Pitfall | Description | Example in LendingClub |
|---------|-------------|----------------------|
| **Post-origination performance as features** | Using loan repayment behavior to predict the interest rate that was already set | Using `loan_status` (Current, Fully Paid, Charged Off) as a feature for `int_rate` prediction |
| **Updated credit scores** | Using credit information pulled after origination | Using `last_fico_range_high` instead of `fico_range_high` (at origination) |
| **Payment history features** | Using cumulative payment amounts | `total_pymnt`, `total_rec_int`, `last_pymnt_amnt` |
| **Future default knowledge** | Using outcome-derived features | `recoveries`, `collection_recovery_fee`, `debt_settlement_flag` |

**Impact:** Models using future information will show impossibly good performance in validation but completely fail in production when future data is not available.

### 2.2 Target Leakage Through Engineered Features

| Leakage Type | Description | LendingClub Example |
|-------------|-------------|-------------------|
| **Direct target leakage** | Feature is derived from or equal to the target | Using `grade`/`sub_grade` to predict `int_rate` when grade is assigned using interest rate |
| **Derived leakage** | Feature contains target information indirectly | Using `installment` (which is calculated FROM `int_rate` and `loan_amnt`) to predict `int_rate` |
| **Temporal leakage** | Feature from future used for past prediction | Using credit report updated after loan funding |

**Critical LendingClub-Specific Leakage:**

```
grade/sub_grade -> int_rate LEAKAGE:
- LendingClub assigns grade and sub_grade based on borrower risk profile
- interest_rate is SET based on grade/sub_grade
- grade/sub_grade are essentially discretized versions of int_rate
- Using them to predict int_rate is circular/self-fulfilling
- If grade/sub_grade MUST be used, acknowledge they encode
  LendingClub's internal risk pricing model
```

### 2.3 Leakage in Cross-Validation

| Pitfall | Why it's wrong | Correct Approach |
|---------|---------------|------------------|
| **Random k-fold CV** | Shuffles temporal order; validation folds may contain loans from 2015 while training folds contain loans from 2020 | **Time-based splitting**: Train on earlier periods, validate on later periods |
| **Stratified CV on time series** | Preserves class distribution but breaks temporal ordering | **Walk-forward validation** or **expanding window CV** |
| **Purging insufficient** | Even with time-based splits, adjacent time periods may leak information (e.g., economic conditions) | Add **embargo periods** between train and validation |

**Key Insight from Research:** "In credit risk, borrower behavior changes with interest rates, employment trends, and inflation. A model trained during stable economic conditions may perform poorly during a downturn." - Medium/Concept Drift

### 2.4 Feature Engineering Leakage

| Operation | Leakage Risk | Safe Alternative |
|-----------|-------------|-----------------|
| Computing aggregate statistics across full dataset | Global mean/std includes test set info | Compute statistics on training set only, apply to test |
| Target encoding with full data | Uses target values from test set | Cross-validated target encoding (compute within folds) |
| Imputation before split | Imputed values influenced by test set | Split first, impute using training statistics only |
| One-hot encoding before split | Learns all categories from full data | Generally safe, but rare categories may leak frequency info |
| Feature selection before split | Selects features based on full data correlations | Select features using only training data |

---

## 3. Temporal Validation Best Practices

### 3.1 Why Temporal Validation is Critical for Lending

> "We construct models using past data to predict future applicants' data." - LendingClub Credit Scoring Project

> "Out-of-time train-test split is considered the best approach for PD, EAD, and LGD Modeling." - Credit Scoring Best Practices

Financial data is inherently temporal:
- Economic conditions change over time (recessions, interest rate cycles)
- Lending standards evolve (LendingClub tightened standards post-2015)
- Borrower demographics shift
- Regulatory environment changes

**A model trained on 2015 loans and tested on 2020 loans faces a fundamentally different distribution than random splitting would suggest.**

### 3.2 Walk-Forward Validation for Loans

**Standard Walk-Forward (Expanding Window):**

```
Period 1: Train [Jan-Mar 2015] -> Validate [Apr-Jun 2015]
Period 2: Train [Jan-Jun 2015] -> Validate [Jul-Sep 2015]
Period 3: Train [Jan-Sep 2015] -> Validate [Oct-Dec 2015]
Period 4: Train [Jan-Dec 2015] -> Validate [Jan-Mar 2016]
...and so on
```

**Advantages:**
- Simulates real deployment: model trained on all past data predicts next period
- Training set grows over time (more data for recent periods)
- Tests model stability across different economic environments
- Detects concept drift naturally (performance drops = drift)

**For Loan Data Without Explicit Date:**

If `issue_d` (loan issue date) is missing but you have temporal ordering:
1. Sort loans by implicit order (row index may represent chronological order)
2. Use the row index as a proxy for time
3. Apply expanding or sliding window validation

### 3.3 Sliding Window (Rolling Window) Validation

```
Period 1: Train [Jan-Dec 2015] -> Validate [Jan-Mar 2016]
Period 2: Train [Apr 2015-Mar 2016] -> Validate [Apr-Jun 2016]
Period 3: Train [Jul 2015-Jun 2016] -> Validate [Jul-Sep 2016]
...and so on
```

**When to use sliding vs. expanding window:**
- **Expanding window**: When you expect older data to still be relevant (stable relationships)
- **Sliding window**: When you expect older data to become irrelevant (rapidly changing conditions)
- For LendingClub, **expanding window** is generally preferred as credit relationships are relatively stable

### 3.4 Time-Based Split Implementation (No Explicit Date)

If your dataset lacks an explicit date column:

1. **Check if row order is meaningful** - LendingClub datasets are often ordered by `issue_d`
2. **Create a synthetic time index** - Use row position as proxy for time
3. **Use sklearn's TimeSeriesSplit** - Even with synthetic ordering

```python
from sklearn.model_selection import TimeSeriesSplit

# If data is ordered chronologically (even without explicit date)
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    # Train and evaluate
```

### 3.5 Handling Concept Drift in Lending

**Types of drift in loan pricing:**

| Drift Type | Cause | Detection | Mitigation |
|------------|-------|-----------|------------|
| **Sudden drift** | Regulatory changes, economic shocks (COVID-19, 2008 crisis) | Sharp performance drops in walk-forward validation | Trigger-based retraining |
| **Gradual drift** | Slowly changing economic conditions, evolving borrower base | Slowly declining performance in monitoring | Periodic retraining (quarterly/annually) |
| **Seasonal drift** | Holiday spending, tax season borrowing patterns | Cyclical performance variations | Include seasonal features, use full-year training windows |

**Best Practice for Loan Models:**

> "Periodic retraining is the simplest approach. Models are retrained at fixed intervals using the latest data. This works reasonably well for gradual drift but may be insufficient for sudden changes." - Concept Drift in Finance Research

Recommended monitoring:
- Track prediction error by month/quarter
- Monitor feature distributions for shifts
- Use population stability index (PSI) to detect distribution changes
- Champion-challenger framework: maintain a challenger model trained on recent data

### 3.6 Purged Cross-Validation (Advanced)

From Marcos Lopez de Prado's "Advances in Financial Machine Learning":

**Combinatorial Purged Cross-Validation (CPCV):**
- Splits data into K folds
- Creates multiple paths to assess model performance
- "Purges" observations between train and test that could leak information
- Critical when adjacent observations are not independent (e.g., overlapping loan performance periods)

**When to use:** For high-frequency or when loan performance periods overlap significantly.

---

## 4. Preprocessing Without Data Leakage

### 4.1 Imputation Strategies That Don't Leak

**The Golden Rule:** "Always split your data into training and test sets before any imputation. Use only training data statistics for imputation." - SAS Communities

**Correct Pipeline:**
```
1. Split data (temporally) into train/validation/test
2. Compute imputation statistics ONLY from training data
3. Apply those statistics to ALL sets (train, val, test)
```

| Imputation Method | How to Do It Safely | LendingClub Application |
|-------------------|---------------------|------------------------|
| **Mean/Median imputation** | Compute mean on training set only; apply to all sets | `annual_inc`, `dti`, `revol_util` |
| **Mode (most frequent) imputation** | Compute mode on training set only | `emp_title`, `verification_status` |
| **Constant imputation** | Use domain-knowledge constant (e.g., 0) | `chargeoff_within_12_mths` (0 = no charge-offs) |
| **Indicator variable + imputation** | Create "is_missing" flag, then impute | `mths_since_*` features (missing has meaning) |
| **Model-based imputation** | Train imputation model on training set only | Complex missing patterns |

**LendingClub-Specific Imputation Guidance:**

Based on the NYC Data Science analysis:
- `annual_inc`: Impute with mean (self-reported income, some missing)
- `bc_util`, `delinq_2yrs`: Impute with mean (credit report features)
- `mths_since_*` features: **Impute with a large number OR 0** depending on interpretation:
  - Missing = "never happened" -> impute with large number (e.g., 999 for "never delinquent")
  - Missing = "no event in recent history" -> impute with 0 or use indicator
- Consider missing values as a **separate category** when they carry information (e.g., "no delinquency ever" vs "delinquency 12 months ago")

**CRITICAL:** Never impute using the full dataset. Never use the validation or test set mean/median for imputation.

### 4.2 Encoding Categorical Variables (Target Encoding Risks)

**Target Encoding Leakage Risk:**

```python
# WRONG - Leaks test set information
means = df.groupby('category')['target'].mean()  # Uses ALL data including test
df['category_encoded'] = df['category'].map(means)
```

**Safe Target Encoding Methods:**

**Method 1: Cross-Validated Target Encoding**
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
encoded = pd.Series(index=df.index, dtype=float)

for train_idx, val_idx in kf.split(X_train):  # ONLY on training data
    train_fold = X_train.iloc[train_idx]
    val_fold = X_train.iloc[val_idx]
    
    means = train_fold.groupby('category')['target'].mean()
    val_encoded = val_fold['category'].map(means)
    val_encoded.fillna(train_fold['target'].mean(), inplace=True)
    encoded.iloc[val_idx] = val_encoded
```

**Method 2: Expanding Window Target Encoding (for time series)**
```python
# For each row, use only PAST data to compute category mean
encoded = []
for i in range(len(df)):
    past_data = df.iloc[:i]  # Only past observations
    category_mean = past_data.groupby('category')['target'].mean()
    current_value = df.iloc[i]['category']
    encoded_value = category_mean.get(current_value, global_mean)
    encoded.append(encoded_value)
```

**Method 3: Regularization (Smoothing)**
```python
# Blend category mean with global mean based on category size
def smoothed_target_encode(category, target, alpha=10):
    global_mean = target.mean()
    category_stats = df.groupby(category)[target].agg(['mean', 'count'])
    smoothed = (category_stats['count'] * category_stats['mean'] + alpha * global_mean) / \
               (category_stats['count'] + alpha)
    return smoothed
```

**For LendingClub Features:**

| Feature | Encoding Strategy | Risk Level |
|---------|-------------------|------------|
| `purpose` | One-hot encoding or target encoding with CV | Low |
| `title` | Text vectorization (TF-IDF) or drop (redundant with purpose) | Low |
| `addr_state` | Target encoding with CV (50+ categories) or one-hot | Medium |
| `zip_code` | Target encoding or embeddings (900+ categories) | Medium |
| `emp_title` | Target encoding or text features (high cardinality) | Medium |
| `home_ownership` | One-hot encoding (4 categories) | Low |
| `verification_status` | One-hot or ordinal encoding | Low |
| `application_type` | One-hot encoding (2 categories) | Low |

### 4.3 Scaling and Normalization Across Time

**The Leakage Risk:** "When we normalize the input variables, this requires that we first calculate the minimum and maximum values for each variable before using these values to scale the variables. The dataset is then split into train and test datasets, but the examples in the training dataset know something about the data in the test dataset." - Machine Learning Mastery

**Correct Approach:**
```python
from sklearn.preprocessing import StandardScaler

# 1. Split first (temporally)
X_train, X_test, y_train, y_test = temporal_split(X, y)

# 2. Fit scaler ONLY on training data
scaler = StandardScaler()
scaler.fit(X_train)  # Only training statistics!

# 3. Transform both sets with training statistics
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # NOT fit_transform!
```

**Key Rules:**
1. `fit()` only on training data
2. `transform()` on all data (train, val, test)
3. NEVER `fit_transform()` on the full dataset before splitting
4. Save the scaler with the model for production use

**For Cross-Validation with Pipelines:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Safe: scaler is fit within each CV fold on training data only
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=tscv)
```

### 4.4 Feature Engineering Without Leakage

| Operation | Safe Practice | Leaky Practice |
|-----------|--------------|----------------|
| **Aggregation features** | Compute aggregates only from past data | Compute from full dataset including future |
| **Ratio features** | Compute numerator and denominator from same time period | Mix different time periods |
| **Binning/Discretization** | Learn bin edges from training data only | Learn from full dataset |
| **Interaction features** | Create interactions after split | Pre-compute interactions before split |
| **Dimensionality reduction** | Fit PCA on training, apply to all | Fit PCA on full dataset |

---

## 5. Proper Train/Validation/Test Split Structure

### 5.1 Recommended Split for Loan Pricing Models

```
Timeline: |------- Training (70%) --------|-- Validation (15%) --|-- Test (15%) -->
           [Oldest loans]                   [Recent loans]         [Newest loans]
           Jan 2012 - Jun 2015             Jul 2015 - Dec 2015   Jan 2016 - Jun 2016
```

**Rationale:**
- Train on the oldest data (build model on historical patterns)
- Validate on recent-but-not-newest data (tune hyperparameters, select features)
- Test on the newest data (final unbiased performance estimate)

### 5.2 Walk-Forward Cross-Validation for Hyperparameter Tuning

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# For hyperparameter tuning on training set
tscv = TimeSeriesSplit(n_splits=5)

best_score = -np.inf
best_params = None

for params in parameter_grid:
    fold_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = Model(**params)
        model.fit(X_tr, y_tr)
        score = model.score(X_val, y_val)
        fold_scores.append(score)
    
    avg_score = np.mean(fold_scores)
    if avg_score > best_score:
        best_score = avg_score
        best_params = params
```

### 5.3 Final Model Training and Evaluation

```python
# 1. Train final model on ALL training data with best parameters
final_model = Model(**best_params)
final_model.fit(X_train, y_train)

# 2. Evaluate on validation set (used during development, not for final report)
val_score = final_model.score(X_val, y_val)

# 3. Evaluate on test set (ONE TIME ONLY, for final performance estimate)
test_score = final_model.score(X_test, y_test)

# 4. For production, retrain on all available data (train + val + test)
production_model = Model(**best_params)
production_model.fit(pd.concat([X_train, X_val, X_test]),
                    pd.concat([y_train, y_val, y_test]))
```

### 5.4 Monitoring and Retraining Strategy

| Trigger | Action | Frequency |
|---------|--------|-----------|
| **Scheduled** | Retrain on new data | Quarterly or annually |
| **Performance degradation** | Retrain when test error exceeds threshold | Continuous monitoring |
| **Feature drift** | Retrain when PSI > 0.25 | Monthly check |
| **Economic events** | Immediate retrain after major events | Event-driven |

---

## 6. Summary Checklist for Zero-Leakage Loan Pricing Model

### Data Preparation
- [ ] Explicitly identify and exclude all post-origination variables
- [ ] Remove `loan_status` from features (it's an outcome, not a predictor)
- [ ] Confirm all `mths_since_*` features are from origination-time credit report
- [ ] Verify `chargeoff_within_12_mths` refers to pre-application history
- [ ] Remove or carefully handle `grade`/`sub_grade` if predicting `int_rate` (circular)

### Data Splitting
- [ ] Sort data chronologically by loan issue date
- [ ] Use time-based split (train on past, test on future)
- [ ] Never use random splitting for loan data
- [ ] Consider walk-forward validation for hyperparameter tuning
- [ ] Add embargo periods if adjacent observations may leak information

### Preprocessing
- [ ] Split data BEFORE any preprocessing
- [ ] Compute imputation statistics on training set only
- [ ] Use `fit()` on training, `transform()` on all sets
- [ ] Use pipelines for cross-validation
- [ ] Apply target encoding with cross-validation or expanding windows only

### Feature Engineering
- [ ] Only use features known at application/ origination time
- [ ] Document availability timestamp for each feature
- [ ] Avoid derived features that encode target information
- [ ] Create missing indicators when missingness carries information

### Validation
- [ ] Use expanding or sliding window cross-validation
- [ ] Monitor for concept drift across validation folds
- [ ] Report performance by time period
- [ ] Test on truly held-out recent data (never seen during any tuning)

---

## Sources

1. SadraDaneshvar/lendingclub-default-risk-screening - "Default-Risk Prediction & Screening at Loan Origination in P2P Consumer Lending" (GitHub)
2. allmeidaapedro/Lending-Club-Credit-Scoring - "PD, EAD, LGD Modeling with Lending Club Data" (GitHub)
3. PDx Adaptive Credit Risk Forecasting Model - "Columns prone to data leakage like pymnt_plan, funded_amnt, payment_differment, and int_rate were removed" (arXiv)
4. NYC Data Science - "LendingClub Loan Default Predictions with Machine Learning"
5. LendingClub Data Dictionary - Kaggle/SEC filings
6. Machine Learning Mastery - "How To Backtest Machine Learning Models for Time Series Forecasting"
7. Machine Learning Mastery - "How to Avoid Data Leakage When Performing Data Preparation"
8. H2O.ai - "What is Target Leakage and how can you Stop it?"
9. GTR Academy - "Train/Validation/Test Splits and Data Leakage in Practice"
10. Medium - "When the world changes: How Machine Learning Handles Concept Drift" (Banking focus)
11. FinTech Weekly - "How to Manage AI Model Drift in FinTech Applications"
12. Stats StackExchange - "Time series split (expanding window) vs k-fold"
13. SAS Communities - "How to Avoid Data Leakage When Imputing for Predictive Modeling"
14. Medium/Prathik - "How to Do Target Encoding Without Data Leakage"
15. Stats StackExchange - "Target encoding in test data and target leakage"
16. scikit-learn documentation - "Common pitfalls and recommended practices"
17. PMC/DataSAIL - "Data splitting to avoid information leakage"
18. Redalyc - "Loan Default Prediction: A Complete Revision"
19. Michael Toth - "Analyzing Historical Default Rates of Lending Club Notes"
20. RahulDhanasiri - "Loan default prediction and credit risk analysis" (GitHub)

---

*Document compiled for production-grade machine learning model development with strict zero-leakage requirements.*
