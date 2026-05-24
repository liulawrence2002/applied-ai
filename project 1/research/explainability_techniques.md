# ML Explainability Techniques for Loan Interest Rate Prediction Models

## Comprehensive Research Findings

**Date:** 2025-01-21
**Context:** Class project building a model to predict loan interest rates. Required to use at least 2 explainability techniques and interpret results in business terms.

---

## Table of Contents

1. [SHAP (SHapley Additive exPlanations)](#1-shap)
2. [Permutation Importance](#2-permutation-importance)
3. [Partial Dependence Plots (PDP) and ICE Plots](#3-partial-dependence-plots-and-ice-plots)
4. [Additional Explainability Techniques](#4-additional-explainability-techniques)
5. [Business Interpretation Guidelines](#5-business-interpretation-guidelines)
6. [Recommended Approach for Class Project](#6-recommended-approach-for-class-project)
7. [References](#7-references)

---

## 1. SHAP (SHapley Additive exPlanations)

### 1.1 Overview and Theory

SHAP (SHapley Additive exPlanations) is a method based on cooperative game theory that explains how individual predictions are made by a machine learning model. SHAP deconstructs a prediction into a sum of contributions from each of the model's input variables, rooted in Shapley values from coalition game theory [Lundberg & Lee, 2017].

For every prediction, SHAP distributes the model's output among features according to their marginal contributions across all possible feature coalitions. This provides:
- **Local interpretability**: Why was this specific loan assigned this interest rate?
- **Global interpretability**: Which features matter most across all loans?

The key mathematical property:
```
f(x) = base_value + sum(SHAP values for all features)
```

Where the `base_value` is the average prediction across the training dataset, and each SHAP value represents a feature's contribution to moving the prediction away from this baseline [Molnar, 2021].

### 1.2 TreeSHAP for Tree-Based Models

**TreeSHAP** (introduced by Lundberg, Erion, and Lee, 2018) is the specialized algorithm for tree-based models (Random Forests, XGBoost, LightGBM, CatBoost) [MetricGate, 2026].

**Key advantages of TreeSHAP:**

| Property | TreeSHAP | KernelSHAP (generic) |
|----------|----------|---------------------|
| Exactness | Exact Shapley values | Sampling approximation |
| Speed | Polynomial O(T*L*D^2) | Exponential in features |
| Model support | Trees only | Any model |
| Deterministic | Yes | No (has variance) |
| Consistency guarantees | Yes | Approximate |

Where T = number of trees, L = number of leaves, D = maximum depth [Yang, 2021].

**Axioms satisfied by TreeSHAP:**
1. **Local accuracy** -- the sum of SHAP values plus baseline equals the model prediction
2. **Missingness** -- features absent from a tree path receive zero attribution
3. **Consistency** -- if a model changes so a feature contributes more, its SHAP value never decreases
4. **Symmetry** -- features that contribute equally receive equal attribution [Lundberg et al., 2018]

### 1.3 Global Feature Importance Interpretation

**Mean absolute SHAP value** is the standard global importance metric:

```
Global Importance_j = (1/n) * sum(|phi_j^(i)|) for i = 1 to n
```

**Visualization options for global interpretation:**

1. **SHAP Bar Plot**: Ranks features by mean absolute SHAP value. Simple, clear, shows which features drive the model overall.

2. **SHAP Beeswarm/Summary Plot**: The most informative global visualization. Each dot represents one SHAP value for one instance. The color encodes the feature value (red = high, blue = low), and the x-position shows the SHAP value. This reveals:
   - Feature importance ranking (top to bottom)
   - Direction of effect (positive SHAP = increases interest rate)
   - Non-linear relationships (color gradients)
   - Feature value distributions [Molnar, 2021; Pastor et al., 2024]

**Interpreting the beeswarm plot for loan pricing:**
- If FICO score dots are mostly on the left (negative SHAP) and blue (low FICO), this means low FICO scores increase predicted interest rates
- If debt-to-income dots are on the right (positive SHAP) and red (high DTI), high DTI increases interest rates
- Wide spread = strong effect; narrow cluster around zero = weak effect

### 1.4 Local Explanations for Individual Predictions

Local explanations answer: *"Why was this specific applicant offered a 12.5% interest rate?"*

**Available visualizations:**

1. **Waterfall Plot**: The most complete single-prediction display. Starts at the base value (average prediction), then adds each feature's SHAP value (positive = red, negative = blue) step-by-step until reaching the final prediction. Shows exactly how each feature pushed the rate up or down from average [SHAP Documentation, 2025].

2. **Force Plot**: A compact version of the waterfall plot. Shows features as arrows pushing the prediction away from the base value. Useful for comparing multiple instances side-by-side [Cooper, 2021].

3. **Decision Plot**: Shows how predictions are built step-by-step for one or more instances. Lines trace how each feature adds to the cumulative prediction. Useful for comparing multiple loan applicants.

**Example interpretation for a loan applicant:**
```
Base interest rate (average): 10.2%
+ FICO 680 (below average): +1.5%
+ DTI 42% (high): +0.8%
+ Loan amount $35K (moderate): +0.2%
+ Term 60 months: +0.3%
- Income $85K (above average): -0.5%
= Final rate: 12.5%
```

### 1.5 Computational Considerations for 100K+ Row Datasets

TreeSHAP computational performance for large datasets [Yang, 2021; Mitchell, 2022]:

| Scenario | Dataset Size | Trees | Max Depth | CPU Time | Optimization |
|----------|-------------|-------|-----------|----------|-------------|
| Small model | 100K rows | 100 | 6 | ~2 minutes | Standard TreeSHAP |
| Medium model | 100K rows | 400 | 12 | ~30 minutes | Standard TreeSHAP |
| Large model | 20M rows | 400 | 12 | ~14.7 hours (100 cores) | Fast TreeSHAP v2 (~4.9 hours) |

**Best practices for large datasets:**

1. **Use TreeSHAP, NOT KernelSHAP** for tree models -- polynomial vs. exponential complexity
2. **Sample a background dataset** -- use 100-1000 representative rows as background instead of the full dataset for calculating the base value
3. **Use Fast TreeSHAP v1/v2** -- achieves 1.5-3x speedup over standard TreeSHAP [Yang, 2021]
4. **GPU acceleration (GPUTreeSHAP)** -- achieves 13-19x speedup vs. 40 CPU cores on large models [Mitchell, 2022]
5. **Explain a stratified sample** -- for global interpretation, compute SHAP values on a representative subset (e.g., 10K rows stratified by interest rate bins) rather than all 100K rows
6. **Use `shap.sample()`** -- the SHAP library provides methods for efficient sampling
7. **Parallel processing** -- TreeSHAP is embarrassingly parallel across dataset rows [Yang, 2021]

**Practical recommendation for 100K row dataset:**
- For global interpretation: Sample 5,000-10,000 rows stratified across the interest rate distribution. Compute time: 30 seconds - 2 minutes
- For local explanations: Compute SHAP values on-demand for individual predictions (near-instantaneous)
- For model development: Use a 5,000-row validation set for iterative SHAP analysis

### 1.6 SHAP Interaction Values

TreeSHAP can compute interaction values that split the effect into main effects and pairwise interactions [Lundberg et al., 2018]:

```
phi_{i,j} = interaction effect between feature i and j
```

**Key interactions for loan pricing models:**
- FICO score x Loan term: Does the effect of FICO differ for 36 vs. 60-month loans?
- DTI x Income: Is high DTI more concerning for lower-income borrowers?
- Loan amount x Term: Does the effect of loan size depend on repayment period?

**Visualizing interactions:**
- SHAP dependence plots with color encoding a second feature
- `shap.dependence_plot("feature_A", shap_values, X, interaction_index="feature_B")`
- Clear color separation indicates meaningful interaction

> **Important finding from research:** SHAP and ALE plots can disagree on interaction direction when features are correlated. In a Lending Club dataset analysis, ALE showed that longer terms + high interest rates increased default risk, while SHAP showed the opposite. This occurred because SHAP accounts for feature correlations differently. Both answers are "correct for what they measure" -- agreement across methods strengthens confidence [Klaas, 2023].

---

## 2. Permutation Importance

### 2.1 How to Properly Compute Without Leakage

Permutation importance is a **model-agnostic**, **global** method that measures how much a model's error increases when a feature's values are randomly shuffled [Breiman, 2001; Altmann et al., 2010].

**Formula:**
```
PI_j = (1/K) * sum[L(y, f(X^(pi_k^(j)))) - L(y, f(X))] for k = 1 to K
```

**Step-by-step computation:**

1. **Split first** -- Train on training data, compute importance on a **held-out validation/test set** (never training data alone)
2. **Compute baseline** -- Record model performance on the untouched validation set
3. **Permute** -- For feature j, randomly shuffle its values across all instances, breaking the feature-target relationship
4. **Re-evaluate** -- Compute performance on the permuted dataset
5. **Repeat** -- Run K >= 10 repetitions and report mean AND standard deviation
6. **Rank** -- Features with the largest performance drop are most important

**Critical anti-leakage rules:**
- **Never compute on training data** -- scores will be inflated due to overfitting
- **Permute AFTER train/test split** -- shuffling before splitting leaks information
- **Use consistent random seeds** for reproducibility
- **Report variance** across multiple permutation runs -- high variance means your dataset may be too small [Molnar, 2021; MetricGate, 2026]

```python
from sklearn.inspection import permutation_importance

# Proper implementation
result = permutation_importance(
    model, X_test, y_test,  # <-- TEST SET, not training
    n_repeats=10,           # <-- Multiple repetitions
    random_state=42,
    scoring='neg_mean_squared_error'
)

# Report with confidence
for i in result.importances_mean.argsort()[::-1]:
    print(f"{feature_names[i]:>20}: "
          f"{result.importances_mean[i]:.3f} "
          f"+/- {result.importances_std[i]:.3f}")
```

### 2.2 Comparison with SHAP for Feature Ranking

Permutation importance and SHAP answer **subtly different questions** [MetricGate, 2026; Molnar, 2021]:

| Dimension | Permutation Importance | SHAP |
|-----------|----------------------|------|
| **Question answered** | "If I destroy this feature's info, how much worse does the model perform?" | "How much does this feature move a prediction away from average?" |
| **Scope** | Global only | Local + Global |
| **Units** | Change in loss (e.g., MSE increase) | Model output units (e.g., percentage points of interest rate) |
| **Model-agnostic** | Yes | TreeSHAP is tree-specific; KernelSHAP is agnostic |
| **Direction** | Always non-negative | Signed (positive/negative) |
| **Local explanations** | No | Yes |
| **Computational cost** | O(K * p * n) | TreeSHAP: O(T*L*D^2) |
| **Overfitting detection** | Compare train vs. test importance | Not natively supported |
| **Interactions** | Collapses into single score | Explicit interaction values |

**Key insight:** Permutation importance is about **model performance**; SHAP is about **prediction explanation**. Conflating them is the #1 source of incorrect feature importance stories [MetricGate, 2026].

**Validation workflow:**
1. Compute both permutation importance and mean absolute SHAP values
2. Calculate Spearman correlation between the two rankings
3. Correlation >= 0.8: Green light -- both methods agree
4. Correlation < 0.5: Investigate -- usually indicates correlated features or sampling issues [MetricGate, 2026]

### 2.3 Handling Correlated Features

The **correlated feature trap** is the most significant challenge for both methods [MetricGate, 2026; Strobl et al., 2008]:

**Scenario:** Features x1 and x2 have correlation r=0.9, but only x1 is causally linked to interest rate.

| Method | Behavior | Result |
|--------|----------|--------|
| Permutation Importance | Permuting x1 alone barely hurts because x2 is a backup | **Both features show LOW importance** -- signal is hidden |
| SHAP (standard) | Splits credit between both features via coalition averaging | **Both show moderate importance** -- attribution is diluted |

**Neither answer is wrong** -- both are correct for what they measure. The problem is interpretation.

**Best practices for correlated features:**

1. **Check correlations first** -- compute feature correlation matrix. Flag pairs with |r| > 0.7.
2. **Use grouped permutation** -- permute correlated features *together*. If joint permutation shows high importance but individual show low, that's "shadow feature" behavior.
3. **Apply domain knowledge** -- if FICO and "credit tier" are highly correlated, consider dropping the derived feature.
4. **Use regularization** -- tree-based models with `colsample_bytree < 1.0` reduce correlation effects.
5. **Compare rankings** -- disagreement between permutation importance and SHAP rankings is a diagnostic signal for multicollinearity [MetricGate, 2026].

**Example grouped permutation:**
```python
# Permute correlated features together
from sklearn.inspection import permutation_importance

# Group FICO and credit_tier together
groups = [['fico', 'credit_tier'], 'dti', 'income', 'loan_amount', 'term']

result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42
)
```

---

## 3. Partial Dependence Plots (PDP) and ICE Plots

### 3.1 Theory and Computation

**Partial Dependence Plots (PDP)** show the marginal effect of one or two features on the predicted outcome, averaging over all other features [Friedman, 2001; scikit-learn, 2025].

**Mathematical definition:**
```
PDP_s(x_s) = E_{x_c}[f(x_s, x_c)] = (1/n) * sum_{i=1}^{n} f(x_s, x_c^(i))
```

Where x_s are the features of interest and x_c are the complement features (held constant at their observed values).

**Individual Conditional Expectation (ICE) plots** show the same relationship but for individual instances rather than the average [Goldstein et al., 2015]. A PDP is simply the average of all ICE curves.

### 3.2 Best Practices for Continuous Features

**For loan pricing continuous features (FICO, DTI, loan amount):**

1. **Always plot ICE + PDP together** -- PDPs can hide heterogeneous effects. If ICE curves show diverging patterns, the feature has interactions with others [MathWorks, 2025; Goldstein et al., 2015].

2. **Choose appropriate grid points**:
   - Use the observed feature quantiles rather than uniform grid
   - For FICO (300-850), use 20-50 points spanning the range
   - For DTI (0-50%), focus on the 10-45% range where most data exists

3. **Center ICE plots (c-ICE)** -- subtract individual predictions from each ICE curve to make heterogeneity easier to see [Goldstein et al., 2015].

4. **Use percentiles for x-axis** -- when feature distributions are skewed, plotting against percentiles rather than raw values makes patterns clearer.

```python
from sklearn.inspection import PartialDependenceDisplay

# Recommended approach for loan pricing
features_to_plot = ['fico', 'dti', 'loan_amount', 'term_months']

fig, ax = plt.subplots(figsize=(16, 10))
PartialDependenceDisplay.from_estimator(
    model, X_train, features_to_plot,
    kind='both',           # Shows both PDP (line) and ICE (translucent lines)
    subsample=500,         # Limit ICE curves to avoid overcrowding
    random_state=42,
    grid_resolution=50,    # Number of grid points
    ax=ax
)
```

### 3.3 Interpreting Non-Linear Relationships

**Common PDP patterns and their business meaning** [Sun, 2025]:

| Pattern | Shape | Business Interpretation |
|---------|-------|------------------------|
| Monotonic | Steady increase/decrease | Strong, consistent effect (e.g., FICO vs. rate) |
| Saturating | Steep then flat | Diminishing returns (e.g., FICO above 750 has minimal additional benefit) |
| Non-monotonic | U-shaped or inverted U | Complex relationship requiring investigation |
| Flat | Horizontal line | Feature has little marginal effect |
| Step function | Sudden jumps | Tree model split points (e.g., FICO threshold at 620) |

**Key features to analyze for loan pricing:**

1. **FICO Score** -- Expected monotonic decreasing: higher FICO = lower rate. Look for:
   - Threshold effects (e.g., big drop at 620, 680, 740)
   - Saturation above 760 (diminishing risk reduction)
   - Non-linear regions where the model applies different risk premiums

2. **Debt-to-Income (DTI) Ratio** -- Expected increasing: higher DTI = higher rate. Check:
   - Regulatory thresholds (36%, 43% for QM loans)
   - Whether the model correctly captures DTI risk
   - Interaction with income level

3. **Loan Amount** -- Relationship depends on product. Analyze:
   - Jumbo loan thresholds ($766,550 in 2024)
   - Whether larger loans get better rates (economies of scale) or worse (concentration risk)

4. **Loan Term** -- Typically 36 vs. 60 months. Examine:
   - Term premium for longer loans
   - Interaction with borrower quality

### 3.4 Two-Way PDP for Interactions

Two-way PDPs show how two features jointly affect predictions. Critical interactions for loan pricing:

- **FICO x Term**: Is the FICO discount larger for 60-month loans?
- **DTI x Income**: Does high DTI matter more for low-income borrowers?
- **Loan Amount x Term**: Do large, long-term loans get special pricing?

```python
# Two-way interaction PDP
PartialDependenceDisplay.from_estimator(
    model, X_train,
    features=[('fico', 'dti')],  # Two-way interaction
    grid_resolution=20,
    contourf_kw={"cmap": "RdBu_r"}
)
```

### 3.5 ALE Plots: Alternative to PDP for Correlated Features

**Accumulated Local Effects (ALE)** plots address PDP's main limitation: bias from correlated features [Apley & Zhu, 2016].

**When to use ALE vs. PDP:**

| Scenario | Recommendation |
|----------|---------------|
| Features uncorrelated | Use PDP (simpler interpretation) |
| Features correlated (r > 0.5) | Use ALE |
| PDP and ALE look similar | Use PDP |
| PDP and ALE differ significantly | Use ALE, investigate correlation |

**ALE advantages:**
- Uses only instances with similar feature values (conditional distribution)
- Unbiased when features are correlated
- Faster to compute: O(n) vs. O(n * grid_points) for PDP
- Centered at zero: ALE value = difference from mean prediction [Molnar, 2021]

**ALE interpretation:**
- ALE curve at x = 0 means the feature value has average effect
- Positive ALE = feature value increases prediction above average
- Slope of ALE curve is analogous to a regression coefficient
- 2D ALE shows only the *interaction effect*, not total effect [Molnar, 2021]

```python
# Using PyALE for ALE plots
from PyALE import ale

# 1D ALE plot
ale_eff = ale(
    X=X_test, model=model, feature=["fico"],
    feature_type="continuous", grid_size=50
)

# 2D ALE for interactions
ale_eff = ale(
    X=X_test, model=model, feature=["fico", "dti"],
    feature_type=["continuous", "continuous"], grid_size=20
)
```

---

## 4. Additional Explainability Techniques

### 4.1 LIME for Local Explanations

**Local Interpretable Model-agnostic Explanations (LIME)** approximates any black-box model with a local, interpretable surrogate (typically linear regression) around a specific prediction [Ribeiro et al., 2016].

**How LIME works:**
1. Select an instance to explain
2. Generate perturbed samples around that instance
3. Weight samples by proximity to the original instance
4. Fit an interpretable model (e.g., linear regression) on weighted samples
5. Use the interpretable model's coefficients as local feature importances

**LIME for loan pricing example** [Bhatnagar, 2024]:
```python
import lime
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    mode='regression'
)

# Explain a specific loan
exp = explainer.explain_instance(
    X_test[0], 
    model.predict,
    num_features=5
)

# Output example:
# "DTI = 42%: +0.8% to interest rate"
# "FICO = 680: +1.2% to interest rate"
# "Income = $85K: -0.4% from interest rate"
```

**SHAP vs. LIME comparison:**

| Property | SHAP | LIME |
|----------|------|------|
| Theoretical foundation | Cooperative game theory (exact) | Local linear approximation |
| Consistency | Guaranteed | Not guaranteed |
| Stability | Stable across runs | Sensitive to perturbations |
| Computational cost | TreeSHAP: exact, fast; KernelSHAP: slower | Faster for single predictions |
| Global explanation | Yes (aggregate SHAP values) | No (local only) |
| Additive property | Exact sum equals prediction | Approximate |
| Output | Feature contributions in model units | Linear coefficients |

**Recommendation:** For loan pricing models using tree-based algorithms, **prefer TreeSHAP over LIME** due to its stronger theoretical guarantees, exactness, and consistency. SHAP has demonstrated "significant value across different financial use cases" with "more consistent and robust" explanations compared to LIME's "unstable nature" [Tang et al., 2024; Tyagi, 2022].

### 4.2 Feature Interaction Analysis

Three complementary methods for detecting feature interactions:

**1. Friedman's H-Statistic**
Measures what fraction of the model's prediction variance depends on interactions between features [Friedman & Popescu, 2008]:
- H = 0: No interaction (purely additive model)
- H = 1: All variance comes from interactions
- Typically, H > 0.1 indicates meaningful interaction

**2. SHAP Interaction Values**
TreeSHAP computes pairwise interaction effects explicitly:
```python
explainer = shap.TreeExplainer(model)
shap_interaction = explainer.shap_interaction_values(X_sample)

# Summarize interactions
interaction_matrix = np.abs(shap_interaction).mean(axis=0)
```

**3. Two-way ALE Plots**
Show only the pure interaction effect (after removing main effects), making interactions easy to identify visually [Molnar, 2021; Halac & Chang, 2023].

**SHAP Dependence Plots for interactions:**
```python
# Shows how FICO effect varies by loan term
shap.dependence_plot("fico", shap_values, X, interaction_index="term")
```
- If colors (representing the interaction feature) separate into distinct bands = strong interaction
- If colors are randomly mixed = no interaction

### 4.3 Model-Agnostic vs. Model-Specific Approaches

| Approach | Techniques | When to Use |
|----------|-----------|-------------|
| **Model-specific** | TreeSHAP, Gain importance, Built-in feature importance | When using tree-based models; faster, more accurate |
| **Model-agnostic** | Permutation importance, LIME, PDP, ALE, SHAP (KernelSHAP) | When explaining any model type; for model comparison; for regulatory documentation |

**Hybrid approach (recommended):**
1. Use TreeSHAP for primary interpretation (exact, efficient)
2. Use permutation importance as a validation check
3. Use PDP/ALE for understanding feature shapes
4. Use LIME only if stakeholder needs require a different explanation format [BIS, 2024]

**Global surrogate models** offer another approach: train an interpretable model (decision tree, linear regression) to approximate the black-box model's predictions [Molnar, 2021].
- Pros: Simple, easy to communicate, can provide rule-based explanations
- Cons: Approximation quality must be validated (check R^2 of surrogate vs. original)

### 4.4 Feature Importance Validation Techniques

A defensible feature importance workflow [MetricGate, 2026; Halac & Chang, 2023]:

**Step 1: Compute multiple importance metrics**
- Built-in model importance (gain)
- Permutation importance (test set)
- Mean absolute SHAP values

**Step 2: Compare rankings**
- Spearman correlation between rankings should be >= 0.5
- Large discrepancies indicate issues (correlated features, overfitting, data leakage)

**Step 3: Examine top features with PDP/ALE**
- Confirm relationships make business sense
- Check for non-linear patterns and thresholds

**Step 4: Check for stability**
- Re-run on bootstrap samples
- Important features should appear consistently
- Report confidence intervals for importance scores

**Step 5: Domain expert review**
- Validate that top features align with lending expertise
- Flag any counter-intuitive patterns for investigation

---

## 5. Business Interpretation Guidelines

### 5.1 Translating SHAP Values into Business Insights

**For loan interest rate prediction, SHAP values translate directly to basis points:**

| SHAP Value | Business Translation |
|-----------|---------------------|
| +0.015 (in log space) | Feature increases interest rate by ~1.5 percentage points |
| -0.008 | Feature decreases interest rate by ~0.8 percentage points |
| Near zero | Feature has minimal effect on this loan's rate |

**Example business interpretation:**

*Waterfall plot for Applicant #12345:*
```
Base rate (portfolio average): 9.50%
FICO 710 (good, not great): +0.80% 
DTI 38% (concerning): +0.95%
Income $72K (moderate): +0.10%
Loan amount $28K (small, low risk): -0.20%
Term 36 months (shorter = less risk): -0.25%
Previous defaults 0 (clean history): -0.30%
Employment 5 years (stable): -0.10%
------------------------------------
Final predicted rate: 10.50%
```

**Business insight:** "This applicant's rate is 1.0% above average primarily due to two factors: a FICO score of 710 (adding 80 bps) and a debt-to-income ratio of 38% (adding 95 bps). The combination of moderate credit quality with elevated leverage drives the risk premium. If the applicant reduced DTI to 30%, we estimate the rate would decrease by approximately 45 bps."

### 5.2 Communicating Model Behavior to Non-Technical Stakeholders

**Key principles** [Various sources, 2024-2025]:

1. **Lead with business value**, not methodology
   - Good: "FICO score is our strongest pricing factor, driving up to 3 percentage points of rate variation"
   - Bad: "SHAP values showed the highest mean absolute attribution for fico_score"

2. **Use analogies**
   - "SHAP values are like a restaurant bill split among friends -- each person pays their fair share based on what they ordered"
   - "The waterfall plot shows how each factor 'votes' for a higher or lower rate"

3. **Focus on actionable insights**
   - "Loans with DTI above 43% receive an average 85 bps rate premium"
   - "FICO scores below 620 drive the majority of high-rate predictions"

4. **Use appropriate visualizations**
   - **Executives**: SHAP bar plot (top 10 features), simple PDPs
   - **Underwriting managers**: Beeswarm plots, dependence plots
   - **Compliance officers**: Full waterfall plots, feature importance comparison
   - **Loan officers**: Simplified force plots for individual explanations

5. **Create layered communication**
   - Layer 1 (30 seconds): Top 3 factors and their direction
   - Layer 2 (5 minutes): Feature importance ranking with business interpretation
   - Layer 3 (30 minutes): Full technical documentation with plots

6. **Use scenario-based explanations**
   - "A borrower with FICO 750 and DTI 25% would receive ~X% rate"
   - "The same borrower with FICO 650 would pay ~Y% more"

### 5.3 Regulatory Considerations (Fair Lending, ECOA, Adverse Action)

**Equal Credit Opportunity Act (ECOA) and Regulation B** require creditors to provide written notification with "specific reasons" when taking adverse action against a consumer [CFPB, 2022; Skadden, 2024].

**Key regulatory requirements:**

1. **Adverse Action Notices (ECOA/Regulation B)**
   - Must provide "statement of specific reasons for the action taken"
   - Must disclose "principal reasons" for denial or adverse terms
   - While no specific number is mandated, "disclosure of more than four reasons is not likely to be helpful" [Reg B Commentary]
   - Reasons must be specific: "debt-to-income ratio too high" not just "did not meet criteria"

2. **CFPB Circular 2022-03** -- Critical guidance for AI models:
   > "ECOA and Regulation B do not permit creditors to use complex algorithms when doing so means they cannot provide the specific and accurate reasons for adverse actions." [CFPB, 2022]

   **Key implications:**
   - Post-hoc explanations (SHAP, LIME) are approximations
   - Creditors must be able to **validate the accuracy** of these approximations
   - For black-box models, "true" exact explanations are unknowable
   - This creates compliance risk for highly complex models [Pace Analytics, 2022]

3. **Fair Lending (ECOA's anti-discrimination provisions)**
   - Prohibits discrimination on basis of race, color, religion, national origin, sex, marital status, age
   - Model explanations must not reveal proxy variables for protected characteristics
   - SHAP analysis should include fairness auditing: check if protected class proxies (zip code, income source) have disproportionate impact
   - Document that explainability techniques were used to test for disparate impact

4. **Practical compliance recommendations** [Debevoise, 2023; CFPB, 2020; Carrington Labs, 2025]:

   - **Use inherently interpretable models where possible** (e.g., logistic regression, scorecards) for final decision
   - **If using complex models**: Document that explanations are accurate approximations
   - **Provide multiple explanation methods** -- BIS recommends "a suite of methods to address differing needs of stakeholders" [BIS, 2024]
   - **Validate surrogate model accuracy** -- if using LIME/SHAP for adverse action reasons, document fidelity to the original model
   - **Maintain explanation audit trail** -- save SHAP values for every adverse action decision
   - **Train staff** -- ensure loan officers can interpret and communicate AI-driven explanations

5. **Adverse Action Best Practices:**
   - Use SHAP to identify top 4 factors pushing the rate up (or causing denial)
   - Map SHAP features to Regulation B-compliant reason codes
   - Example mapping: "fico_score" -> "Credit history" or "Credit score"; "dti_ratio" -> "Excessive obligations relative to income"

**Important note:** For a class project, regulatory compliance is illustrative. In production lending, these requirements are legally binding and subject to CFPB enforcement.

---

## 6. Recommended Approach for Class Project

### 6.1 Minimum Viable Explainability Pipeline

**For your loan interest rate prediction class project, use at least these 2 techniques:**

**Technique 1: SHAP (TreeSHAP)**
```python
import shap

# After training your XGBoost/LightGBM model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global: Summary/beeswarm plot
shap.summary_plot(shap_values, X_test)

# Global: Bar plot (simpler)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Local: Waterfall for individual predictions
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns
))
```

**Technique 2: Partial Dependence Plots**
```python
from sklearn.inspection import PartialDependenceDisplay

# PDP for top 4 features
features_to_plot = ['fico', 'dti', 'loan_amount', 'term_months']
PartialDependenceDisplay.from_estimator(
    model, X_train, features_to_plot,
    kind='both',  # Shows PDP + ICE
    subsample=500,
    grid_resolution=50
)
```

**Optional: Permutation Importance (as validation)**
```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10)
# Compare with SHAP rankings
```

### 6.2 Business Interpretation Template

For each explainability technique, provide:

**SHAP Global Analysis:**
- "The top 5 features driving interest rate predictions are: [list]"
- "FICO score has the largest impact, with a range of X basis points between the lowest and highest scores"
- "Higher DTI consistently pushes rates higher, with the strongest effect above 40%"

**SHAP Local Analysis:**
- "Applicant #X was offered Y% because: [list top 3-4 factors with magnitudes]"
- "If their FICO were 50 points higher, the model predicts a rate Z basis points lower"

**PDP Analysis:**
- "The relationship between FICO and rate is monotonically decreasing with diminishing returns above 750"
- "DTI shows a threshold effect at 43%, with sharply higher rates above this level"
- "[Feature X] shows [pattern], which aligns/doesn't align with business expectations because..."

### 6.3 Grading Rubric Alignment

| Criterion | How to Address |
|-----------|---------------|
| Use >= 2 explainability techniques | SHAP + PDP/ICE (add permutation importance for bonus) |
| Interpret results in business terms | Translate SHAP values to bps, use business language |
| Demonstrate understanding | Compare methods, discuss strengths/limitations |
| Actionable insights | Provide specific recommendations based on findings |

---

## 7. References

### Primary Sources

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

2. Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 2522-5839.

3. Lundberg, S. M., Erion, G., & Lee, S. I. (2018). Consistent individualized feature attribution for tree ensembles. *arXiv preprint arXiv:1802.03888*.

4. Molnar, C. (2021). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. 2nd Edition. https://christophm.github.io/interpretable-ml-book/

5. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 1189-1232.

6. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

7. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD*, 1135-1144.

8. Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015). Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation. *Journal of Computational and Graphical Statistics*, 24(1), 44-65.

9. Apley, D. W., & Zhu, J. (2016). Visualizing the effects of predictor variables in black box supervised learning models. *arXiv preprint arXiv:1612.08468*.

10. Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles. *Journal of the American Statistical Association*, 516-584.

### Regulatory and Industry Sources

11. Consumer Financial Protection Bureau. (2022). CFPB Circular 2022-03: Adverse action notification requirements in connection with credit decisions based on complex algorithms.

12. Consumer Financial Protection Bureau. (2020). Innovation Spotlight: Providing adverse action notices when using AI/ML models. https://www.consumerfinance.gov/about-us/blog/innovation-spotlight-providing-adverse-action-notices-when-using-ai-ml-models/

13. Skadden, Arps, Slate, Meagher & Flom LLP. (2024). CFPB Applies Adverse Action Notification Requirement to Artificial Intelligence Models.

14. Debevoise & Plimpton. (2023). Adverse Action Notice Compliance Considerations for Creditors That Use AI.

15. Pace Analytics LLC. (2022). Using Explainable AI to Produce ECOA Adverse Action Reasons.

16. Bank for International Settlements (BIS). (2024). *How regulators can address AI explainability*. FSI Papers No. 24.

17. Carrington Labs. (2025). A practical guide to explainability in lending.

18. SBS Software. (2026). Explainability in financial AI: Why trust is the new currency.

19. Congressional Research Service. (2024). Artificial Intelligence and Machine Learning in Financial Services. R47997.

### Technical Implementation Sources

20. Yang, J. (2021). Fast TreeSHAP: Accelerating SHAP value computation for trees. *arXiv preprint arXiv:2109.09847*.

21. Mitchell, R., et al. (2022). GPUTreeShap: massively parallel exact calculation of SHAP scores for tree ensembles. *Bioinformatics*, 38(Supplement_1), i127-i134.

22. scikit-learn. (2025). Partial Dependence and Individual Conditional Expectation plots. https://scikit-learn.org/stable/modules/partial_dependence.html

23. SHAP Documentation. (2025). https://shap.readthedocs.io/

24. MathWorks. (2025). Interpretability and Explainability for Credit Scoring. https://www.mathworks.com/help/risk/interpretability-and-explainability-for-credit-scoring.html

### Feature Importance and Comparison Sources

25. MetricGate. (2026). Permutation Importance vs SHAP. https://metricgate.com/blogs/permutation-importance-vs-shap/

26. MetricGate. (2026). SHAP Tree Explainer (TreeSHAP) Calculator. https://metricgate.com/docs/shap-tree-explainer/

27. BugFree.AI. (2026). Feature Importance Metrics: Gain, Permutation, and SHAP Values.

28. Altmann, A., Tolosi, L., Sander, O., & Lengauer, T. (2010). Permutation importance: A corrected feature importance measure. *Bioinformatics*, 26(10), 1340-1347.

29. Halac, H. & Chang, R. (2023). A Model Explainability Toolbox: Tips and Techniques to Interpret Black Box Models. FI Consulting.

### SHAP Interpretation and Communication Sources

30. Cooper, A. (2021). A Non-Technical Guide to Interpreting SHAP Analyses. https://www.aidancooper.co.uk/

31. Pastor, E. (2024). Post-modeling Explainability. Politecnico di Torino.

32. Sun, P. (2025). Partial Dependence Plots Will Expose Your Model. Medium.

33. OneModel. (2024). What is the SHAP beeswarm chart? https://help.onemodel.co/

### SHAP Interaction Sources

34. Klaas, M. (2023). SHAP vs. ALE for Feature Interactions: Understanding Conflicting Results. Towards Data Science.

35. Foster, L. (2024). Feature Interaction Detection with SHAP. Medium.

### LIME and Surrogate Model Sources

36. Bhatnagar, N. (2024). Decoding closed box Models with LIME. Medium.

37. C3 AI. (2026). LIME: Local Interpretable Model-agnostic Explanations. https://c3.ai/glossary/

38. OECD.AI. (2023). Local Interpretable Model-agnostic Explanation (LIME). https://oecd.ai/en/catalogue/metrics/

39. Medium. (2022). Explainable AI (XAI) Methods Part 5 -- Global Surrogate Models.

40. Milvus. (2026). What is the role of surrogate models in Explainable AI?

### Communication Sources

41. Medium. (2024). How to Effectively Communicate Complex ML Model Results to Non-Technical Stakeholders.

42. IgniteSAP. (2024). Communicating with Non-Technical Stakeholders.

43. Statsig. (2024). How to communicate findings to non-technical teams.

44. arXiv. (2026). Trade-offs in Financial AI: Explainability in a Trilemma with Accuracy and Ease of Understanding.

### Lending-Specific SHAP Sources

45. Tang, F. K., et al. (2024). A SHAP-Based Comparative Analysis of Machine Learning. RITHA EU.

46. Misheva, B. H., et al. (2021). Explainable AI and SHAP for credit risk. *Finance Research Letters*.

47. Tyagi, A. (2022). SHAP for credit scoring and investment decisions.

---

## Appendix A: Quick Reference Card

### SHAP Interpretation Cheat Sheet

| Plot Type | Use Case | Audience |
|-----------|----------|----------|
| Bar plot (mean abs SHAP) | Feature ranking | Everyone |
| Beeswarm/Summary | Feature ranking + direction + non-linearity | Technical |
| Waterfall | Single prediction breakdown | Compliance, loan officers |
| Force plot | Multiple prediction comparison | Technical |
| Decision plot | Multiple prediction comparison | Technical |
| Dependence plot | Feature shape + interactions | Technical |

### PDP/ICE Interpretation Cheat Sheet

| Pattern | Meaning | Action |
|---------|---------|--------|
| Monotonic | Consistent effect | Validate direction matches business logic |
| Saturating | Diminishing returns | Note threshold where effect plateaus |
| Non-monotonic | Complex relationship | Investigate -- may indicate overfitting or real interaction |
| Flat | Weak marginal effect | Consider removing or transforming feature |
| Step function | Tree split point | Note threshold value; may be business-relevant |

### Regulatory Mapping Cheat Sheet

| Model Feature | ECOA/Reg B Reason Code Category |
|--------------|--------------------------------|
| fico_score | Credit history / Credit score |
| dti_ratio | Excessive obligations relative to income |
| income | Insufficient income for amount of credit requested |
| loan_amount | Unacceptable loan amount or terms |
| employment_length | Length of employment too short |
| inquiries_last_6mo | Number of recent credit inquiries |

---

*End of Research Document*
