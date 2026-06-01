# Min-RMSE V2 — Design Notes

**Notebook**: `FINAL_SIMPLIFIED_MIN_RMSE_V2.ipynb`
**Build script**: `build_min_rmse_v2_notebook.py`
**Submission file**: `FINAL_MIN_RMSE_V2_SUBMISSION.csv`
**Predecessor**: `FINAL_SIMPLIFIED_MIN_RMSE.ipynb` (the V1 min-RMSE notebook, expected val_RMSE ~3.83–3.85).

## What this iteration changes

V2 keeps V1's plumbing — same 80/20 split, same 5-fold OOF, same isotonic post-calibration, same W2-style explainability — and changes only **the features and the model layer** (per request). Every other section is left intact.

### Feature changes

| Change | Why |
|---|---|
| **Add 35-class sub_grade auxiliary classifier** | The v3 reverse-engineer experiment (`reverse_engineer/v3/aux_models_v3.py`) showed sub_grade carries strictly more information than the 7-class grade — 35 ordered classes resolve roughly 5× finer rate buckets. The classifier is trained on `achive_data/archive/LC_train.csv` (which has `sub_grade`) and applied to the true-data slice. |
| **Add rate-lookup expected-rate feature** | `outputs/reverse_engineer/v3/rate_lookup.json` is a static historical mapping `subgrade → mean_rate` (e.g., A1 → 5.67%, G5 → 29.69%). We compute `aux_lookup_expected_rate = Σ_sg p(sg) × mean_rate(sg)` — a probability-weighted direct rate prediction. The v3 codebase calls this the "highest individual feature contribution" in its analysis. |
| **Add rate-lookup uncertainty feature** | `aux_lookup_rate_uncertainty = √(Σ p(sg) × (mean_rate(sg)² + std_rate(sg)²) − expected_rate²)`. Tells the model when the sub_grade prediction is confident vs hedged. |
| **Add `aux_sg35_argmax_ord` + `_max_prob` + `_top2_gap` + `_entropy`** | Summary stats over the 35-class probability distribution. The entropy and top-2 gap are model-confidence proxies — particularly useful in residual buckets. |
| **Drop the old 7-class `aux_grade_prob_*` columns from V1** | The 35-class sub_grade prediction subsumes them. Keeping both creates first-mover bias (Mar 2026 arXiv 2603.22346) where SHAP arbitrarily picks one of the two correlated representations. |
| **Keep `fico_x_aux_argmax` + `fico_x_aux_max_prob`** | Reformulated to use the 35-class argmax-ordinal and max-prob (range 1–35, finer granularity). |
| **Add `aux_lookup_expected_rate × term_months`** | The v3 enrich module shows this interaction is particularly useful: longer-term loans at the same sub_grade are systematically priced higher. |

Total feature count grows from V1's ~48 to V2's **~75** (35 sg-prob cols + 4 summary stats + 2 lookup cols + 1 lookup×term interaction + 2 fico×aux interactions + the 35 reduced-FE base columns), still well under 100 and still no neural-net opaqueness.

### Model layer changes

| Change | Why |
|---|---|
| **Add a 4th base learner: HistGradientBoosting (HGB)** | V1 used CB + LGB + XGB. HGB has a structurally different binning algorithm and tends to be ~0.005–0.015 pp decorrelated from the other three in blends (per Hannah's notebook). At V2's larger feature set the blend gain is worth the +5 min wall-clock. |
| **CatBoost gets `bootstrap_type=Bernoulli` + `subsample` re-tune** | V1 used the cached production-v2 CB params (`bagging_temperature` style). For the larger V2 feature set we Optuna-retune CB at 30 trials to recover its optimum on the enriched space. |
| **All three boosters: `min_child_samples` / `min_data_in_leaf` / `min_child_weight` lower bound raised** | Wider feature set → some new features have noisy tail buckets (e.g., sg35 prob columns for rare classes like G5 with n=104). Higher min-samples-in-leaf cuts the bias toward over-fitting those tails. |
| **Optuna search expanded to 50 trials each for LGB / XGB** (vs 40 in V1) | Same logic — wider search space deserves more trials. |
| **`log1p(int_rate)` target retained for LGB + XGB only** | CB stays on raw target (decorrelation). The log transform's biggest win is in the right tail (high-rate subprime loans), and adding the rate-lookup expected-rate gives the model an even more direct handle on that tail — keeping log1p doubles down on it. |

### Diagnostic additions

| Plot | What it tells you |
|---|---|
| **Learning curve vs boosting rounds** (per base learner) | The train RMSE vs val RMSE curves across iterations. Train < val by a large persistent gap → overfitting. Train ≈ val and both still falling at the stopping point → underfitting / too few iterations. Train ≈ val and both flat → well-fit. Plotted for CB, LGB, XGB, HGB separately. Source: XGBoosting.com learning-curve guide. |
| **Train-set-size learning curve** | sklearn's `learning_curve` run on CB at 5 training fractions (20/40/60/80/100%). Shows whether more data would still reduce val_RMSE (curve still descending = underfit; flat = capacity-bound). |
| **OOF-vs-val residual distribution overlay** | Confirms the OOF and held-out residuals share the same shape — if they diverge, the model is overfitting the OOF folds. |
| **Per-fold RMSE bar chart** | Five bars per model, each fold's RMSE. High variance across folds → unstable model / leak risk. Low variance → solid generalisation. |
| **SHAP value distribution: train vs val vs test** | Already in V1; carried forward unchanged. Confirms drivers are stable across splits. |

## Expected impact

Per the v3 documentation (`outputs/reverse_engineer/v3/ranking_v3.csv`), the 35-class sub_grade + rate_lookup additions contributed roughly:

* **Single LGB**: 3.901 → 3.873 (~−0.03 pp) when sg35 prob cols are added.
* **Single CB**: 3.852 → 3.846 in the v3 stack (marginal).
* **Best stack (lgb_stack meta-learner)**: 3.841 in v3 vs 3.830 in v2 — i.e., v3's full stack actually under-performed v2 because the production stack used a Ridge meta-learner that got confused by the redundant features.

The V2 design extracts the v3 *features* without the v3 *meta-learner* problem: we use a simple 4-model SLSQP blend + isotonic, not a Ridge stack. The expected combined lift is:

| V1 baseline | V2 with rate-lookup features + HGB | Lift |
|---|---|---|
| ~3.83–3.85 (expected) | **~3.79–3.82 (expected)** | ~0.03–0.05 pp |

This pushes the notebook a bit below the documented champion (3.83), making it among the strongest interpretable single-notebook submissions in the repo.

## What we deliberately do NOT add

* **Issue-year aux models** from v3 — true data has no `issue_d`, so the issue-year prediction would be 100% extrapolation and the model treats it as noise.
* **Within-grade residual aux model** from v3 — same data dependency, plus the production analysis showed it adds <0.005 pp.
* **`loan.csv` (2.26M row) as a second aux training source** — the v3 ensemble uses both `LC_train.csv` and `loan.csv`. Doubling training adds ~10 min compute for ~0.005 pp marginal lift. Skip in the interest of clarity.
* **Multi-seed averaging** — would add ~30 min for ~0.005 pp; outside the budget.

## Reproducibility

Same `RANDOM_STATE = 6604`, same 80/20 split, same 5-fold KFold, leak-safe per-fold target encoding, leak-safe per-fold log1p inversion.

## Sources

| Source | What we use |
|---|---|
| `reverse_engineer/v3/rate_lookup.py` + `outputs/reverse_engineer/v3/rate_lookup.json` | Pre-computed `subgrade_mean_rate` and `subgrade_std_rate` tables. |
| `reverse_engineer/v3/aux_models_v3.py` | Reference implementation of the 35-class CatBoost classifier. We re-train it inline rather than depending on the pickled `aux_models_v3.pkl` (which the repo does not ship). |
| `reverse_engineer/v3/enrich_v3.py` | The probability-weighted lookup expression for `aux_lookup_expected_rate` and `aux_lookup_rate_uncertainty`. |
| Pargent et al. 2022 (Comput. Stat.) | TargetEncoder recommendations — already applied in V1. |
| APAR (NeurIPS 2024, arXiv 2412.10941) | `log1p` target for GBDT on skewed regression — already in V1. |
| First-mover bias (arXiv 2603.22346, Apr 2026) | Why we drop the 7-class grade probs when the 35-class set is present (avoids redundant correlated representations). |
| XGBoosting.com learning-curve guide | Diagnostic plot design for the new overfitting/underfitting section. |

## How to run

```powershell
# Regenerate the notebook from the build script (idempotent)
python build_min_rmse_v2_notebook.py
# Then open the notebook and run-all, or:
jupyter nbconvert --to notebook --execute FINAL_SIMPLIFIED_MIN_RMSE_V2.ipynb --inplace --ExecutePreprocessor.timeout=4200
```

Wall-clock estimate: **45–55 min** on CPU (the 35-class sub_grade classifier alone takes ~10–15 min; the rest matches V1 timings).
