# SOTA Machine Learning for Loan Interest Rate Prediction

> **Research synthesis** covering state-of-the-art ML models, open-source implementations, and practical recommendations for predicting interest rates on loans. Compiled from 2024–2025 academic literature, pre-prints, and active GitHub repositories.

---

## 1. Executive Summary

Predicting loan interest rates sits at the intersection of **credit risk assessment** and **dynamic pricing**. While traditional lenders rely on scorecards and macroeconomic benchmarks, modern ML approaches treat interest rate prediction as either:

- A **regression problem** (predict the exact rate given borrower and market features), or
- A **two-stage optimization** (predict default probability → set risk-adjusted rate).

The current SOTA landscape is dominated by **gradient-boosted decision trees** (XGBoost, LightGBM, CatBoost) for tabular borrower data, **LSTM/GRU networks** for sequential behavioral data, and **hybrid architectures** (e.g., LSTM + XGBoost stacking) that capture both temporal dependencies and complex feature interactions. Emerging transformer-based models (TabTransformer, FT-Transformer) are challenging tree ensembles on high-cardinality categorical financial data.

**Bottom line:** Start with CatBoost or XGBoost for baseline tabular models; add LSTM/GRU or hybrid stacks if you have time-series behavioral data; consider TabTransformer only when categorical feature interactions are critical and compute budget allows.

---

## 2. Problem Framing

### 2.1 Regression vs. Classification

| Framing | Target | Use Case |
|---------|--------|----------|
| **Regression** | `interest_rate` (continuous) | Directly predict the rate a lender will offer given borrower profile |
| **Classification** | `isDefault` (binary) | Predict default risk, then map to a rate via risk-based pricing |
| **Joint / Multi-task** | Both | Simultaneously predict default probability and optimal rate (e.g., Utility-Engine, Interest-Rate-Optimization-System) |

In practice, most commercial systems use a **two-stage approach**: a default probability model (PD) feeds into a pricing engine that adds spreads for operating costs, profit margin, and macroeconomic conditions.

### 2.2 Static vs. Dynamic Pricing

- **Static models** predict rates based solely on borrower characteristics at application time.
- **Dynamic / regime-aware models** incorporate macroeconomic cycles (GDP growth, unemployment, Fed funds rate) to adjust pricing. Recent work (Villalobos 2025) shows **516% profit improvement** over static baselines by optimizing rates per economic regime.

---

## 3. Feature Engineering Best Practices

### 3.1 Core Feature Categories

Based on Lending Club analyses, Kaggle credit-risk datasets, and 2025 literature, the most predictive features are:

| Category | Key Features |
|----------|-------------|
| **Credit History** | FICO score (range low/high), delinquencies (30+ days), public records, bankruptcies, earliest credit line age |
| **Utilization & Debt** | Revolving balance, revolving utilization (`revol_util`), debt-to-income ratio (`dti`), total open accounts |
| **Income & Employment** | Annual income, employment length, employment title, home ownership status |
| **Loan Characteristics** | Loan amount, term (36 vs. 60 months), purpose, grade/sub-grade, application type (individual vs. joint) |
| **Behavioral / Temporal** | Monthly repayment patterns, cash flow trajectories, credit utilization trends over time |
| **Macroeconomic** | Fed funds rate, unemployment, GDP growth, inflation, 10-year Treasury yield |

### 3.2 Preprocessing Pipeline

1. **Missing value imputation**
   - Numerical: median or KNN imputation
   - Categorical: mode imputation or "Unknown" category
2. **Outlier handling**: Cap extreme values (e.g., DTI > 100%) or use robust scalers
3. **Encoding**
   - **CatBoost**: Native categorical handling (no encoding needed)
   - **XGBoost / LightGBM**: Label encoding or target encoding for high-cardinality features
   - **Deep learning**: Embedding layers for categorical variables
4. **Class imbalance** (if default prediction is part of the pipeline)
   - **SMOTE**, **TomekLinks**, or **SMOTE–TomekLinks** hybrid
   - Note: Over-sampling can inflate precision but hurt recall on imbalanced credit data
5. **Time-series aware splitting**
   - Use **rolling-window CV** or **temporal train-test splits** to prevent look-ahead bias
   - Never shuffle time-ordered financial data

---

## 4. SOTA Model Comparison

### 4.1 Gradient-Boosted Trees (Tabular Baselines)

These remain the **practical SOTA** for structured borrower data due to speed, interpretability, and robustness.

| Model | Strengths | Weaknesses | Best For |
|-------|-----------|------------|----------|
| **XGBoost** | Excellent regularization, widely supported, strong feature importance | Slower training than LightGBM; requires more tuning | General-purpose tabular regression/classification |
| **LightGBM** | Fastest training (leaf-wise growth), GOSS + EFB for large data | Prone to overfitting on small data | Large-scale datasets (>100k rows) |
| **CatBoost** | Native categorical handling, ordered boosting reduces overfitting, often best accuracy | Slightly slower prediction throughput | Datasets with many categorical features (loan purpose, state, home ownership) |

**2025 Comparative Results** (Nuraliyudin & Utomo, 2026 — default prediction on 255k Indonesian loans):

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|-------|----------|--------|----------|---------|
| CatBoost (TomekLinks) | **88.68%** | 5.80% | 10.63% | 75.66% |
| LightGBM (TomekLinks) | 88.65% | 5.80% | 10.61% | **75.69%** |
| XGBoost (TomekLinks) | 88.63% | 5.51% | 10.12% | 75.67% |

*Note:* On imbalanced data, accuracy is misleading. CatBoost and LightGBM are nearly tied; CatBoost slightly edges out on precision, while LightGBM has marginally better AUC.

### 4.2 Deep Learning for Sequential Data

When borrower **behavioral time series** are available (e.g., monthly cash flows, credit utilization trends), RNNs and Transformers capture temporal dependencies that tree models miss.

| Model | Strengths | Weaknesses | Best For |
|-------|-----------|------------|----------|
| **LSTM** | Captures long-term dependencies; robust to vanishing gradients | Slow to train; needs large sequential datasets | Monthly repayment patterns, macroeconomic time series |
| **GRU** | Faster than LSTM with comparable performance | Slightly less expressive on very long sequences | Medium-length behavioral sequences |
| **BiLSTM + Attention** | Bidirectional context + attention weights | Higher compute cost | Complex temporal credit scoring (Yang et al. 2025: AUC = 0.982) |

**2022 P2P Lending Results** (PMC study — average borrowing interest rate prediction):

| Model | RMSE | MAPE | SMAPE |
|-------|------|------|-------|
| **LSTM** | **0.372** | **15.45%** | **14.11%** |
| AttLSTM | 0.392 | 16.67% | 14.91% |
| SVR | 1.578 | 40.27% | 28.03% |
| Random Forest | 1.934 | 66.48% | 41.23% |

### 4.3 Transformer-Based Models for Tabular Finance

Originally designed for NLP, Transformers are increasingly adapted for financial tabular data.

| Model | Key Innovation | Performance |
|-------|---------------|-------------|
| **TabTransformer** (Huang et al. 2020) | Contextual embeddings for categorical features | Improves over MLP and tree baselines on high-cardinality data |
| **FT-Transformer** (Gorishniy et al. 2021) | Attention blocks applied to tabular features | Matches or surpasses CatBoost/XGBoost on nonlinear tasks |
| **Gated Transformer** (PeerJ 2025) | Gating mechanism + self-attention for tabular credit data | AUROC 0.852, beating FT-Transformer (0.829) and SAINT (0.817) |

**When to use:** Transformers shine when feature **interactions** are complex and high-cardinality (e.g., 50+ categorical variables). For typical lending datasets with 10–30 features, the overhead may not justify the gain over CatBoost.

### 4.4 Hybrid & Stacked Architectures

Recent 2025–2026 research consistently shows that **hybrid models** outperform individual learners.

| Architecture | Components | Use Case |
|--------------|------------|----------|
| **LSTM → XGBoost** | LSTM extracts temporal features → XGBoost regressor predicts rate | Macro + borrower time series |
| **LSTM + XGBoost (stacking)** | Base learners: LSTM & ANN → Meta-model: XGBoost | Central bank policy rate forecasting (R² = 0.936) |
| **Transformer + Tree Ensemble** | Self-attention for feature interaction → GBDT for final prediction | High-cardinality tabular + behavioral fusion |
| **Regime-Aware Optimization** | XGBoost (default PD) + macro regime classifier + scalar optimizer | Dynamic loan pricing (516% profit improvement) |

**2025 Mongolia Policy Rate Study** (preprint):

| Model | R² | RMSE | Rank |
|-------|-----|------|------|
| **Hybrid LSTM–XGBoost** | **0.9355** | **Lowest** | 1st |
| XGBoost | ~0.89 | Medium | 2nd |
| LSTM | ~0.87 | Medium | 3rd |
| Random Forest | ~0.85 | Higher | 4th |
| Linear Regression | ~0.72 | Highest | Baseline |

---

## 5. Notable GitHub Repositories

| Repository | Author | Highlights | Link |
|------------|--------|------------|------|
| **Interest-Rate-Optimization-System** | SebastianVillalobosAlva | XGBoost default prediction + macroeconomic regime optimization (FRED data); 516% profit improvement over static pricing | [GitHub](https://github.com/SebastianVillalobosAlva/Interest-Rate-Optimization-System) |
| **Machine-Learning-Loan-Lending-Club** | ragraw26 | Classic Lending Club pipeline: classification (approve/decline) + regression (interest rate) with clustering (manual, k-means, no-cluster) | [GitHub](https://github.com/ragraw26/Machine-Learning-Loan-Lending-Club) |
| **Utility-Engine** | VrajPatel0220 | End-to-end XGBoost credit risk system with Streamlit UI, SHAP explainability, real-time rates, and LangChain policy compliance | [GitHub](https://github.com/VrajPatel0220/Utility-Engine) |
| **Loan-interest-rate-prediction** | Asadtafheem05 | Streamlit app predicting loan interest rate from borrower details (income, credit score, employment length) | [GitHub](https://github.com/Asadtafheem05/Loan-interest-rate-prediction) |
| **Dynamic-Interest-Rate-Prediction-for-Loans-Using-PySpark-Machine-Learning** | ekhosravie | PySpark Random Forest regression for large-scale loan pricing | [GitHub](https://github.com/ekhosravie/Dynamic-Interest-Rate-Prediction-for-Loans-Using-PySpark-Machine-Learning) |
| **Loan-Insight-AI-Predictor** | YaswanthaGoteka | Dual-model: XGBoost regressor (EMI, interest rate, term) + Random Forest classifier (approval) | [GitHub](https://github.com/YaswanthaGoteka/Loan-Insight-AI-Predictor) |
| **loan-interest-rate-prediction-ml** | LovelyPS | Linear Regression baseline with R² and MAPE metrics; simple reference implementation | [GitHub](https://github.com/LovelyPS/loan-interest-rate-prediction-ml) |
| **Credit-Risk-PD-Model-Loan-Default-Prediction** | hemanthreddyaeddulla | PD model on 32k loans; SMOTE balancing; benchmarks LR (78%), RF (94%), XGBoost (95%) | [GitHub](https://github.com/hemanthreddyaeddulla/Credit-Risk-PD-Model-Loan-Default-Prediction) |

---

## 6. Evaluation Metrics

| Metric | Formula / Meaning | When to Use |
|--------|-------------------|-------------|
| **RMSE** | Sqrt(mean squared error) | Penalizes large errors; good for regression |
| **MAE** | Mean absolute error | Robust to outliers; interpretable in rate points |
| **MAPE** | Mean absolute percentage error | Relative error; useful across different rate scales |
| **R²** | Coefficient of determination | Explained variance; compare across models |
| **AUC-ROC** | Area under ROC curve | Classification (default risk) discrimination |
| **Brier Score** | Mean squared probability error | Calibration of predicted default probabilities |
| **SHAP Values** | Shapley additive explanations | Feature importance and regulatory explainability |

**Critical:** For time-dependent data, always use **rolling-window cross-validation** or **walk-forward validation**. Random k-fold CV on financial data leaks future information into training.

---

## 7. Practical Recommendations

### 7.1 If You Are Starting Today

1. **Baseline:** Train **CatBoost** or **XGBoost** on your tabular borrower data.
   - CatBoost wins if you have many categorical features and want minimal preprocessing.
   - XGBoost wins if you need maximum community support and deployment tooling.
2. **Feature engineering:** Focus on FICO, DTI, revolving utilization, loan term, and income. These consistently dominate SHAP importance rankings.
3. **Validation:** Use temporal splits. Report RMSE, MAE, and R² for rate prediction; AUC-ROC and Brier Score for default probability.
4. **Explainability:** Integrate **SHAP** early. Regulators and business stakeholders require transparency in lending decisions.

### 7.2 If You Have Behavioral / Time-Series Data

1. Add **LSTM** or **GRU** layers to model sequential features (e.g., 12–24 months of credit utilization, payment history).
2. Use a **hybrid architecture**: LSTM extracts temporal embeddings → XGBoost/CatBoost predicts the rate.
3. Consider **attention mechanisms** (BiLSTM + Multi-Head Attention) if sequence length > 50 and interactions are non-local.

### 7.3 If You Need Dynamic / Market-Responsive Pricing

1. Build a **two-stage system**:
   - Stage 1: XGBoost predicts Probability of Default (PD).
   - Stage 2: Macroeconomic regime classifier (Crisis / Recovery / Expansion) + bounded optimizer sets the rate.
2. Integrate **FRED macro indicators** (Fed funds rate, unemployment, GDP) as real-time features.
3. Re-train or fine-tune quarterly to adapt to regime shifts.

### 7.4 When to Consider Transformers

- You have **>30 categorical features** with high cardinality (e.g., merchant categories, geographic regions, loan purpose sub-types).
- You have already maximized tree-based performance and need marginal gains.
- Compute budget allows longer training times (TabTransformer/FT-Transformer are slower than LightGBM).

---

## 8. Key References

### Papers & Preprints

1. Nuraliyudin, S. S., & Utomo, W. H. (2026). *A Comparative Analysis of Boosting and Transformers Models For Loan Default Risk Prediction*. Jurnal Perspektif, 10(1), 40–54.
2. Preprint (2026). *Forecasting Central Bank Policy Rates Using Machine Learning in Mongolia*. Preprints.org. Hybrid LSTM–XGBoost achieves R² = 0.936.
3. AMRO Working Paper (2025). *Forecasting Federal Fund Rates with AI: LSTM, GRU, and LLMs*.
4. MDPI (2025). *Data-Driven Loan Default Prediction: A Machine Learning Approach*. Systems, 13(7), 581.
5. PMC (2022). *Investor sentiment-aware prediction model for P2P lending indicators based on LSTM*. AttLSTM vs. LSTM evaluation.
6. MDPI (2026). *Deep Learning for Credit Risk Prediction: A Survey of Methods, Applications, and Challenges*. Information, 17(4), 395.
7. PeerJ (2025). *A deep learning model for predicting loan repayment capacity*. Gated Transformer vs. FT-Transformer, TabNet, SAINT.
8. ResearchSquare (2026). *Transformer architectures for sequential credit risk*. LSTM/GRU/Transformer review.

### GitHub Repositories

- [Interest-Rate-Optimization-System](https://github.com/SebastianVillalobosAlva/Interest-Rate-Optimization-System)
- [Machine-Learning-Loan-Lending-Club](https://github.com/ragraw26/Machine-Learning-Loan-Lending-Club)
- [Utility-Engine](https://github.com/VrajPatel0220/Utility-Engine)
- [Loan-interest-rate-prediction](https://github.com/Asadtafheem05/Loan-interest-rate-prediction)
- [Dynamic-Interest-Rate-Prediction-for-Loans-Using-PySpark-Machine-Learning](https://github.com/ekhosravie/Dynamic-Interest-Rate-Prediction-for-Loans-Using-PySpark-Machine-Learning)
- [Loan-Insight-AI-Predictor](https://github.com/YaswanthaGoteka/Loan-Insight-AI-Predictor)
- [loan-interest-rate-prediction-ml](https://github.com/LovelyPS/loan-interest-rate-prediction-ml)
- [Credit-Risk-PD-Model-Loan-Default-Prediction](https://github.com/hemanthreddyaeddulla/Credit-Risk-PD-Model-Loan-Default-Prediction)

---

*Last updated: 2026-05-18*
