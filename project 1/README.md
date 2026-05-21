# LendingClub `int_rate` Prediction

Predicts the loan interest rate (`int_rate`) for LendingClub applications using
application-time features only. The model is a 5-fold NNLS blend of a tuned
CatBoost regressor (Optuna, 100 trials) and a fixed XGBoost regressor (v1-tuned
params), trained on `true data/LC_train.csv` and scored on `true data/LC_test.csv`.
Submission is `outputs/final_test_predictions.csv` (10,000 rows, columns
`ID, int_rate`).

## Reproduce

```bash
pip install -r requirements.txt
python -m src.train     # trains both models, blends, snaps, writes final CSV (~15 hours)
python -m src.predict   # regenerates outputs/final_test_predictions.csv from saved artifacts
```

`src/train.py` writes per-model OOF predictions, fold models in `outputs/models/`,
blend metadata in `outputs/blend_meta.json`, and a full run report in
`outputs/final_report.json`. `src/predict.py` is a thin transformer that re-applies
the saved blend weights and clip range — use it to regenerate the submission CSV
without re-training.

## Results

5-fold KFold, seed=42. All metrics are out-of-fold (no leakage).

| Model                    | OOF RMSE | OOF MAE | OOF R² | Fold std |
| ------------------------ | -------: | ------: | -----: | -------: |
| CatBoost (Optuna-tuned)  |   3.8484 |  2.8809 | 0.4638 |    0.016 |
| XGBoost v1 (frozen)      |   3.9034 |  2.9215 | 0.4484 |    0.019 |
| **NNLS blend**           | **3.8469** | **2.8795** | **0.4642** | — |

NNLS blend weights: XGBoost 0.141, CatBoost 0.859.

Rate-snap (snap predictions to the nearest legal `int_rate` value seen in train)
was tested and **skipped** — it worsened OOF RMSE by 0.004, below the 0.005
keep-only threshold.

CatBoost params from Optuna (100 trials, single 80/20 split objective):
`iterations=3924, learning_rate=0.0307, depth=8, l2_leaf_reg=2.44,
random_strength=3.10, bagging_temperature=0.39, border_count=180`.

## Leakage exclusions

Columns dropped unconditionally before any feature engineering:

| Column                                                                 | Reason                                                              |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `loan_status`                                                          | Post-origination outcome                                            |
| `grade`, `sub_grade`                                                   | LendingClub's own pricing grade — direct proxy of `int_rate`        |
| `installment`                                                          | Deterministic function of (loan_amnt, term, int_rate)               |
| `out_prncp*`, `total_pymnt*`, `total_rec_*`, `recoveries*`             | Post-origination payment / recovery history                         |
| `last_pymnt_*`, `last_credit_pull_d`, `next_pymnt_d`                   | Post-origination dates / amounts                                    |
| `last_fico_range_low/high`                                             | FICO measured **after** origination (pre-origination FICO is kept)  |
| `debt_settlement_flag`, `hardship_flag`, `hardship_*`                  | Post-origination distress / settlement flags                        |
| `title`                                                                | One-to-one duplicate of `purpose` in this slice                     |
| `emp_title`                                                            | High-cardinality free text, no signal worth the variance            |
| `fico_range_low`, `fico_range_high`, `emp_length`, `term` (raw forms) | Replaced by engineered versions (`fico` midpoint, `emp_length_num`, `term_months`/`term_cat`) |

A `FORBIDDEN_GUARD` set in `src/data.py` is asserted against the final feature
matrix on every fold transform — the pipeline hard-fails if any of these slip
through.

## Validation: random 5-fold, not temporal

The dataset has no date columns (`issue_d` and `earliest_cr_line` are both
absent), so temporal cross-validation is impossible. Random KFold with
`n_splits=5, seed=42` is used for every model, and the same fold indices are
shared across CatBoost and XGBoost so OOF blending is valid.

## Limitations and floor

- **Fold-to-fold std ≈ 0.016-0.019 RMSE.** Any "improvement" smaller than ~0.02
  RMSE is noise. The 100-trial Optuna run on CatBoost moved OOF RMSE by 0.001
  vs the prior hand-set defaults — basically nothing — which is consistent with
  the search space already being near its ceiling.
- **Achievable floor on this slice ≈ 3.75-3.9 RMSE.** Without `grade`/`sub_grade`
  (LendingClub's own price decision; leakage) and without any temporal/macro
  features, ~46% R² is roughly the ceiling for this feature set.
- **Train/test drift exists.** The test slice has lower mean `loan_amnt` and
  slightly higher mean FICO than train (KS D≈0.14 on `loan_amnt`). Predictions
  are calibrated for train and may be slightly conservative on test.
- **Prior v2 feature ablation was conclusive** — FICO nonlinear terms,
  categorical crosses, economics proxies, and target encodings all lost in
  ablation. They are intentionally absent from `src/features.py`.

## Repository layout

```text
src/
  data.py        # load + leakage constants + guard
  features.py    # v1 feature engineering, fold-local preprocessor
  models.py      # frozen XGB v1 params, CatBoost search space + fit helpers
  train.py       # Optuna tune CatBoost, 5-fold CV both models, NNLS blend, snap, submission
  predict.py     # regenerate submission CSV from saved artifacts
true data/       # canonical train + test CSVs + data dictionary
outputs/
  final_test_predictions.csv    # SUBMISSION
  final_report.json             # metrics, params, weights, snap decision
  blend_meta.json               # weights + snap flag for predict.py
  oof_*.csv, test_preds_*.csv   # per-model OOF and averaged-fold test predictions
  models/                       # saved per-fold model files (xgb_fold_*.json, catboost_fold_*.cbm)
  optuna_catboost.db            # Optuna study (resumable)
  archive/                      # superseded artifacts from earlier iterations
archive/                        # earlier pipeline scripts kept for reference
notebooks/01_eda.ipynb          # current EDA on true data/
docs/                           # assignment instructions, retrospective, research notes
```
