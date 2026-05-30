# LendingClub Modeling Deck — Change Summary

New file: `LendingClub_Modeling_Deck.pptx` (17 slides). Built to match the
original `LendingClub_Slide_Template.pptx` — same Cambria/Calibri type, navy
(`1E2761`) / ice-blue (`CADCFC`) / amber accent palette, section tag chips,
title divider rule, stat boxes, and card layout.

## 1. What changed from the original deck

The original template was a single-track XGBoost story with `[X.XX]`
placeholders throughout. The new deck is reframed as a **team bake-off across
five model families**, with every metric filled in from the actual notebooks.

- Kept and adapted: Title, Overview/Problem statement, Preprocessing, and a
  Conclusion slide — placeholders replaced with real figures (target range
  6.46–30.99%, mean 12.95%, etc.).
- Replaced the single-model comparison table with a **side-by-side comparison of
  all teammates' models** (slide 11).
- Added a **model-selection narrative**: ANN limitations → the two best trees →
  why we blend them → final blended metric.
- Expanded EDA from one slide into a **dedicated EDA section** (slides 3–8) using
  the existing figures in `outputs/eda/`.
- Added an explainability slide and an appendix of per-model tuning detail.
- All numbers are sourced from the notebooks/STATUS files — none were invented.
  Where a metric was missing it is flagged on the slide (see §4).

## 2. Which two tree-based models were blended

**CatBoost + XGBoost** (Andrew's track).

- CatBoost — 5-fold OOF RMSE **3.845** (best single model)
- XGBoost — 5-fold OOF RMSE **3.867** (second best)
- Final blend — honest held-out RMSE **3.841**, weights CatBoost ≈ 0.72,
  XGBoost ≈ 0.27 (together ~99%).

LightGBM (3.945) was evaluated in the same blend but received a near-zero weight
(~0.013), so the shipped model is effectively a two-tree blend. Blending lowers
variance (in-sample 3.8413 vs. honest held-out 3.8414 — an optimism gap of just
0.0001) and edges past the best single model, while CatBoost↔XGBoost error
correlation (~0.99) caps the size of the gain.

## 3. Why the ANN was rejected

Will's neural network (sklearn `MLPRegressor`) was **unstable across folds**. In
5-fold cross-validation, fold 3 diverged — RMSE **7.96** and R² **−1.2** while
the other four folds sat near 4.0 — dragging the out-of-fold RMSE to **5.10**
(R² 0.06). This exploding-gradient-style instability made the aggregate metrics
unreliable and irreproducible. Even the best *stable* network (~4.0 RMSE) trailed
every boosted tree. Neural networks need heavy scaling/tuning to be stable on
this mostly-tabular, mixed-type data with strong monotonic signals (FICO↓,
utilization↑), so the team moved to tree-based models. This is the transition
into slide 13 (the two best models are both trees).

## 4. EDA slides — added vs. still needing support

**Added (all backed by existing plots in `outputs/eda/`):**

- Target variable distribution — `02_target_distribution.png`
- Missingness patterns — `03_missingness_top15.png`
- Feature distributions & outliers — `04_numeric_histograms.png`
- Correlation / multicollinearity & top drivers — `07_correlation_heatmap.png`
- Leakage checks & train/test consistency — `08_loan_status_leakage.png`
- Feature-selection rationale (why the final features make sense) — text cards

**Explainability** — `outputs/simplification/explain_cb_gain_top10.png`.

**Covered in text but could use a dedicated plot if you want more visual depth:**

- *Most predictive features* — currently shown as a Spearman table on the
  correlation slide and the gain chart on the explainability slide. A standalone
  SHAP beeswarm (`explain_shap_summary_val.png` exists) could be promoted to its
  own slide.
- *Train/test (fold) consistency* — described from the drift table; the KDE plot
  `10_drift_kde.png` is available if you'd prefer to show it.

**Data gaps flagged on the slides (not invented):**

- **Jaci — Decision Tree:** the notebook ran GridSearchCV but **no RMSE/R² was
  saved**; the comparison table and appendix say "not saved / not recorded."
- **Hannah — Hist-Gradient-Boosting (3.834) and her blend (3.832):** these ran
  on a **~19.5k-row subset**, not the full 100k, so they are **not directly
  comparable** to the 5-fold OOF numbers. This is marked with the ✱ footnote on
  the comparison slide. Re-running her Hist-GB on the full 100k with the shared
  folds would make it directly comparable.
- Andrew's tree R² values weren't recorded in STATUS, so the table reports RMSE
  (the competition metric) rather than guessing R².
