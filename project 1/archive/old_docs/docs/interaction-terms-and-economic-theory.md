# Interaction Terms & Economic Theory for Loan Interest Rate Prediction

> A synthesis of domain-driven feature engineering interaction terms and macroeconomic theories that underpin loan pricing. Drawn from credit-risk literature, the "5 C's" framework, Keynesian/Neo-Keynesian monetary transmission, and empirical ML studies on LendingClub data.

---

## Part I: Domain-Driven Interaction Terms

### 1. The "5 C's of Credit" Framework

Credit analysts evaluate borrowers on five dimensions. Each dimension suggests natural interaction terms because risk is rarely additive—it is **multiplicative**.

| Dimension | Features | Interaction Rationale |
|-----------|----------|----------------------|
| **Character** | FICO, delinquencies, inquiries, public records, credit history length | A low-FICO borrower with *no* delinquencies is different from a low-FICO borrower with *many* delinquencies |
| **Capacity** | DTI, income, employment length, loan amount, installment | High DTI is dangerous only if income is also low; a $2M-income borrower with 40% DTI is not the same as a $40k borrower with 40% DTI |
| **Capital** | Revolving utilization, revolving balance, total leverage | High utilization + low FICO = liquidity crisis; high utilization + high FICO = temporary cash-flow timing |
| **Conditions** | Loan term, purpose, grade, macro environment | 60-month debt-consolidation loans carry different risk than 36-month home-improvement loans |
| **Collateral** | Home ownership, mortgage status | Renters with short employment are riskier than renters with 10+ years at the same job |

### 2. Recommended Interaction Terms

Based on peer-reviewed LendingClub studies (Wide & Deep Learning for P2P Lending, 2024; MDPI 2025; GitHub credit-risk models), the following interactions consistently improve predictive power:

#### 2.1 Borrower-Level Interactions

| Interaction Term | Formula | Economic Rationale |
|-----------------|---------|-------------------|
| **FICO × Term** | `fico_range_low * term_numeric` | Longer terms amplify credit risk for low-FICO borrowers (more time for default) |
| **FICO × Purpose** | Embedding or one-hot cross | Debt consolidation at low FICO is riskier than medical loans at the same FICO |
| **DTI × Income** | `dti * annual_inc` (or ratio) | Absolute debt burden matters, not just the ratio; DTI of 40% on $200k is very different from 40% on $30k |
| **Loan-to-Income** | `loan_amnt / annual_inc` | Affordability ratio; captures whether the loan is disproportionate to earnings |
| **Payment-to-Income** | `installment / (annual_inc / 12)` | Monthly debt service burden; central to underwriting capacity analysis |
| **Revolving Util × FICO** | `revol_util * fico_range_low` | High credit-card usage is benign for high-FICO borrowers but catastrophic for subprime |
| **Employment × Home Ownership** | `emp_length_int * home_ownership_enc` | Stable employment mitigates renter risk; mortgage + short job = recent mover risk |
| **Inquiries × Open Accounts** | `inq_last_6mths / (open_acc + 1)` | Credit-seeking intensity relative to existing credit lines |
| **Delinquency × Credit Age** | `delinq_2yrs / (earliest_cr_line_age + 1)` | Recent delinquencies on a thin file are worse than on a 20-year file |
| **Public Records × Bankruptcy** | `pub_rec + pub_rec_bankruptcies` (or interaction) | Multiple public records compound legal/financial distress |

#### 2.2 Loan-Structure Interactions

| Interaction Term | Formula | Economic Rationale |
|-----------------|---------|-------------------|
| **Grade × Term** | Cross-product or embedding | LendingClub already encodes this, but re-deriving it from fundamentals removes circularity |
| **Interest Rate × Term** | `int_rate * term_numeric` | Total interest cost; 15% over 60 months is much more expensive than 15% over 36 months |
| **Loan Amount × Term** | `loan_amnt * term_numeric` | Exposure duration; large long-term loans are riskier than small short-term loans |
| **Sub-grade × Purpose** | Cross-product | Within the same sub-grade, small-business loans default more than credit-card refinance |

#### 2.3 Macro-Borrower Interactions

| Interaction Term | Formula | Economic Rationale |
|-----------------|---------|-------------------|
| **FICO × Fed Funds Rate** | `fico_range_low * fed_funds_rate` | Tight monetary policy hits low-FICO borrowers harder (credit rationing channel) |
| **DTI × Unemployment Rate** | `dti * unemployment_rate` | High DTI during recessions is far riskier than high DTI during expansions |
| **Loan Amount × GDP Growth** | `loan_amnt * gdp_growth` | Large loans originated just before recessions have higher default risk |
| **Income × Inflation** | `annual_inc * cpi_inflation` | Real purchasing power erosion affects lower-income borrowers more |

### 3. Implementation Notes

- **For tree models (XGBoost, LightGBM, CatBoost)**: Many interactions are discovered automatically via tree splits, but explicit interaction terms still help when the signal is weak or the tree depth is constrained.
- **For linear models**: Interaction terms are **essential**; without them, linear models miss threshold effects entirely.
- **For neural networks / LSTM**: Interaction terms can be fed as auxiliary static features alongside sequential data.

**Pro tip from the literature**: The Wide & Deep Learning for P2P Lending paper (2018, updated 2024) explicitly defines `AND(term, FICO)` and `AND(housing, emp_length)` as wide-component cross-features, which improved default prediction AUC.

---

## Part II: Economic Theories & Loan Pricing

### 1. Keynesian Theory & the IS-LM Framework

#### 1.1 Liquidity Preference Theory of Interest

> *"The rate of interest is a monetary concept... determined by the supply of money and liquidity preference."* — Keynes, *The General Theory* (1936)

**Core idea**: Interest rates are not just determined by the "real" productivity of capital (classical view), but by the supply of money and the public's desire to hold liquid cash.

**Implication for loan pricing**:
- When central banks expand money supply (quantitative easing), liquidity preference falls and loan rates drop.
- When uncertainty rises (e.g., 2008 crisis, COVID-19), liquidity preference spikes—even if central banks cut rates, banks may not pass cuts through to borrowers.

**Feature engineering insight**: Include a **liquidity stress proxy** (e.g., VIX, TED spread, interbank rate volatility) as a macro feature. During high-liquidity-preference periods, the spread between policy rates and loan rates widens.

#### 1.2 The IS-LM Model & Credit Transmission

```
M ↑ ⇒ r ↓ ⇒ I ↑ ⇒ Y ↑
```

Where:
- `M` = money supply
- `r` = real interest rate
- `I` = investment (including loans)
- `Y` = output / aggregate demand

**The Bernanke-Blinder CC-LM Extension** (1988):

Bernanke and Blinder added a **credit channel** to Keynesian IS-LM. They showed that monetary policy affects the economy not just through the interest rate, but through the **availability of bank credit**:

```
M ↑ ⇒ bank reserves ↑ ⇒ loanable funds ↑ ⇒ L ↑ ⇒ Y ↑
```

**Implication for loan pricing**:
- Loan rates depend on both the **policy rate** (cost of funds) and the **supply of credit**.
- During credit crunches (2008, 2020), loan rates rise even if policy rates fall because banks ration credit.

**Feature engineering insight**: Add a **credit availability index** or bank lending standards (from Fed Senior Loan Officer Survey) as a macro predictor.

### 2. The Phillips Curve

#### 2.1 Original & Expectations-Augmented Phillips Curve

| Version | Equation | Meaning |
|---------|----------|---------|
| Original (Phillips, 1958) | `π = f(u)` | Inflation (`π`) is a negative function of unemployment (`u`) |
| Expectations-Augmented (Friedman, 1968) | `π = πᵉ + f(u)` | Inflation depends on **expected inflation** (`πᵉ`) plus the unemployment gap |
| New Keynesian (Taylor, 1980; Calvo, 1983) | `πₜ = βEₜπₜ₊₁ + γyₜ` | Inflation is forward-looking, driven by expected future inflation and the output gap (`y`) |

**Why it matters for loan rates**:
- The Phillips curve is the **core input to central bank reaction functions** (Taylor rules).
- When unemployment is low and inflation is high, central banks raise policy rates → loan rates rise.
- When unemployment is high and inflation is low, central banks cut rates → loan rates fall.

**Feature engineering insight**:
- Include **inflation expectations** (breakeven rates, survey data) and **output gap** (real GDP vs. potential GDP) as macro features.
- Interaction: `unemployment_rate × cpi_inflation` captures the Phillips curve tradeoff directly.

#### 2.2 Neural Phillips Curves (Modern ML Twist)

Recent research (arXiv 2024) uses **Hemisphere Neural Networks (HNN)** to learn a nonlinear Phillips curve:

```
πₜ = F(output_gap_hemisphere, price_hemisphere)
```

Where `F(·)` is a deep neural network restricted to sum components from two "hemispheres" of predictors. This architecture forces the model to learn economically interpretable latent states.

**Implication**: You can use ML to discover **nonlinear Phillips curve dynamics** and feed the predicted inflation path into your loan rate model.

### 3. The Taylor Rule

#### 3.1 The Original Taylor Rule (1993)

```
i* = r* + π + 0.5(π - π*) + 0.5(y - y*)
```

Where:
- `i*` = target nominal policy rate
- `r*` = equilibrium real rate (~2%)
- `π` = current inflation
- `π*` = inflation target (~2%)
- `y - y*` = output gap

**What it means**: Central banks set rates based on **two gaps**:
1. Inflation gap (`π - π*`)
2. Output gap (`y - y*`)

#### 3.2 Taylor Rule for Loan Pricing

LendingClub operates in a market-rate environment, not a policy-rate environment, but the Taylor rule still matters because:

- **When the Taylor rule signals tightening**: Loan rates rise, especially for riskier borrowers.
- **When the Taylor rule signals easing**: Loan rates fall, and credit availability expands.

**Feature engineering insight**:
- Compute a **Taylor rule residual**: `actual_fed_rate - taylor_rule_predicted_rate`
- A large positive residual means the Fed is unusually tight → expect higher loan rates.
- A large negative residual means the Fed is unusually loose → expect lower loan rates.

### 4. The Fisher Effect

#### 4.1 Fisher Equation

```
i = r + πᵉ
```

Where:
- `i` = nominal interest rate
- `r` = real interest rate
- `πᵉ` = expected inflation

**Key insight**: Nominal loan rates must compensate lenders for **expected inflation** plus a **real return** plus a **risk premium**.

#### 4.2 Fisher Effect + Credit Risk Premium

The full loan pricing equation becomes:

```
loan_rate = r* + πᵉ + credit_risk_premium + liquidity_premium + term_premium
```

Where:
- `credit_risk_premium` = f(FICO, DTI, utilization, delinquencies)
- `liquidity_premium` = f(market volatility, bank funding costs)
- `term_premium` = f(loan term, yield curve slope)

**Feature engineering insight**:
- Include **expected inflation** (TIPS breakeven, CPI forecasts) as a macro feature.
- Include **real rate proxies** (TIPS yield, Treasury yield minus inflation).
- The difference between loan rates and Treasury rates (the **credit spread**) is the empirical credit risk premium.

### 5. Credit Rationing & the Stiglitz-Weiss Model (1981)

#### 5.1 Information Asymmetry & Adverse Selection

> *"Interest rates may not always function as an efficient allocator of credits... banks will rather ration credit to borrowers instead of pricing them strictly against the signals of monetary policy."* — Stiglitz & Weiss (1981)

**Core idea**: Because of **information asymmetry**, raising interest rates can actually *increase* default risk (adverse selection). Banks therefore **ration credit** rather than price all risk into the rate.

**Implication for loan pricing**:
- Loan rates are **not a continuous function of risk**; there are discrete cutoffs.
- Some borrowers are simply denied credit rather than charged a higher rate.
- This creates a **selection bias** in observed loan rates: we only see rates for *approved* borrowers.

**Feature engineering insight**:
- Model **approval probability** separately from **rate prediction** (Heckman-style selection model).
- Include a **credit rationing proxy** (e.g., Fed lending standards survey) as a macro feature.
- During tight credit periods, approved borrowers are "safer" than during loose periods, even at the same FICO.

### 6. The Financial Accelerator (Bernanke, Gertler, Gilchrist — BGG)

#### 6.1 Net Worth → Borrowing Costs → Investment

```
Economic shock ↓
    ⇒ Borrower net worth ↓
    ⇒ Collateral value ↓
    ⇒ External finance premium ↑
    ⇒ Investment ↓
    ⇒ Output ↓ (amplifies the shock)
```

**Core idea**: Borrowing costs depend on **borrower net worth** (collateral). When net worth falls during recessions, the external finance premium rises, amplifying downturns.

**Implication for loan pricing**:
- Loan rates are **procyclical** through the collateral channel: in booms, collateral values rise and rates fall; in busts, collateral values fall and rates rise.
- Home ownership and home equity are not just static features—they interact with the business cycle.

**Feature engineering insight**:
- Include **home price index** (Case-Shiller, FHFA) as a macro feature.
- Interaction: `home_ownership * hpi_growth` — homeowners benefit from rising home prices; renters do not.

---

## Part III: Putting It All Together — A Theory-Informed Feature Set

### For Your LendingClub Assignment

| Feature Group | Specific Features / Interactions | Economic Theory |
|--------------|----------------------------------|-----------------|
| **Character** | FICO, delinquencies, inquiries, pub_rec | Stiglitz-Weiss: info asymmetry drives adverse selection |
| **Capacity** | DTI, income, `loan_amnt / income`, `installment / income` | Keynesian IS-LM: investment depends on cost of capital |
| **Capital** | Revolving utilization, `revol_util × FICO` | BGG Financial Accelerator: collateral / net worth matters |
| **Conditions** | Term, purpose, `FICO × purpose`, `grade × term` | Credit channel: loan structure affects risk |
| **Collateral** | Home ownership, `home_own × emp_length` | BGG: collateral values amplify cycles |
| **Macro / Policy** | Fed funds rate, unemployment, CPI, GDP growth | Taylor Rule, Phillips Curve, Fisher Effect |
| **Macro-Borrower** | `FICO × fed_funds`, `DTI × unemployment` | Credit rationing: monetary tightening hits subprime harder |
| **Credit Spread** | Loan rate − Treasury yield (if available) | Fisher Effect + credit risk premium decomposition |

---

## Part IV: Quick-Start Code Template

```python
import pandas as pd
import numpy as np

# ============================================================
# 1. RAW FEATURES (from LC_train.csv)
# ============================================================
# fico_range_low, annual_inc, dti, loan_amnt, installment,
# term_numeric, revol_util, emp_length_int, home_ownership,
# purpose, delinq_2yrs, inq_last_6mths, open_acc, pub_rec,
# earliest_cr_line_age

# ============================================================
# 2. INTERACTION TERMS (domain-driven)
# ============================================================
def engineer_interactions(df):
    df = df.copy()
    
    # Capacity interactions
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['payment_to_income'] = df['installment'] / (df['annual_inc'] / 12 + 1)
    df['dti_x_income'] = df['dti'] * np.log1p(df['annual_inc'])
    
    # Character × Capital interactions
    df['fico_x_term'] = df['fico_range_low'] * df['term_numeric']
    df['revol_util_x_fico'] = df['revol_util'] * df['fico_range_low']
    df['delinq_x_credit_age'] = df['delinq_2yrs'] / (df['earliest_cr_line_age'] + 1)
    
    # Collateral × Capacity
    df['home_own_x_emp_length'] = df['home_ownership_enc'] * df['emp_length_int']
    
    # Credit-seeking intensity
    df['inq_intensity'] = df['inq_last_6mths'] / (df['open_acc'] + 1)
    
    # Macro-borrower interactions (if macro data merged)
    # df['fico_x_fed_funds'] = df['fico_range_low'] * df['fed_funds_rate']
    # df['dti_x_unemployment'] = df['dti'] * df['unemployment_rate']
    
    return df

# ============================================================
# 3. THEORY-INFORMED TRANSFORMATIONS
# ============================================================
def theory_transforms(df):
    df = df.copy()
    
    # Stiglitz-Weiss: credit rationing proxy
    # (tight credit periods = higher selection bias)
    # df['credit_rationing_proxy'] = df['fed_lending_standards_index']
    
    # Fisher Effect: real rate proxy
    # df['real_rate_proxy'] = df['treasury_10y'] - df['expected_inflation']
    
    # Taylor Rule residual (how much is Fed deviating from rule?)
    # taylor_target = 2.0 + df['cpi_inflation'] + 0.5*(df['cpi_inflation'] - 2.0) + 0.5*df['output_gap']
    # df['taylor_residual'] = df['fed_funds_rate'] - taylor_target
    
    # BGG Financial Accelerator: collateral amplification
    # df['collateral_cycle'] = df['home_ownership_enc'] * df['hpi_growth']
    
    return df
```

---

## References

### Interaction Terms & Feature Engineering

1. **Wide & Deep Learning for Peer-to-Peer Lending** (2018/2024). ResearchGate. Defines `AND(term, FICO)` and `AND(housing, emp_length)` as explicit wide-component interactions.
2. **MDPI** (2025). *Optimizing Credit Risk Prediction for Peer-to-Peer Lending*. Introduces `DTI × annual_income` interaction term.
3. **johndmcmillin / credit-risk-model** (GitHub, 2026). "5 C's" framework: Capacity, Character, Capital, Conditions, Collateral.
4. **bertiedickinson / LendingClub** (GitHub). Feature engineering: `loan_to_income_ratio`, `interest_to_income_ratio`, `fico_risk_group`, `high_dti_risk`.

### Economic Theory

5. **Keynes, J. M.** (1936). *The General Theory of Employment, Interest and Money*. Liquidity preference theory of interest.
6. **Bernanke, B. S., & Blinder, A. S.** (1988). Credit, Money, and Aggregate Demand. *American Economic Review*.
7. **Taylor, J. B.** (1993). Discretion versus Policy Rules in Practice. *Carnegie-Rochester Conference Series on Public Policy*.
8. **Fisher, I.** (1930). *The Theory of Interest*. Nominal rate = real rate + expected inflation.
9. **Stiglitz, J. E., & Weiss, A.** (1981). Credit Rationing in Markets with Imperfect Information. *American Economic Review*, 71(3), 393–410.
10. **Bernanke, B. S., Gertler, M., & Gilchrist, S.** (1999). The Financial Accelerator in a Quantitative Business Cycle Framework. *Handbook of Macroeconomics*.
11. **Phillips, A. W.** (1958). The Relation between Unemployment and the Rate of Change of Money Wage Rates in the United Kingdom. *Economica*.
12. **Mishkin, F. S.** (1996). *The Economics of Money, Banking, and Financial Markets*. Monetary policy transmission mechanisms.

---

*Last updated: 2026-05-18*
