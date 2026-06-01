"""
Builds notebooks/01_eda.ipynb programmatically using nbformat.
Run once to (re)generate the notebook. The notebook itself is the deliverable.

Cells are appended to CELLS in section order. Each section has 1 markdown
header cell followed by 1+ code cells. Executing this script writes a fresh
.ipynb; subsequent `jupyter nbconvert --execute` runs it and captures outputs.

Dataset: true data/LC_train.csv and true data/LC_test.csv
  - 100k train, 10k test
  - 39 columns each (train has int_rate target; test has ID, no target)
  - data dictionary covers every train column exactly
  - FICO scores PRESENT (fico_range_low, fico_range_high)
  - loan_status PRESENT in both — current status, unavailable at application time; dropped immediately after load
  - No date columns (no issue_d, no earliest_cr_line) — no temporal validation possible
  - Several numerics stored as strings with "NA" nulls — cleaned at load time
"""
import os
from pathlib import Path
import nbformat as nbf

NB = nbf.v4.new_notebook()
CELLS = []


def md(text: str):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(src: str):
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n")))


# ============================================================================
# SECTION 0 — Setup
# ============================================================================
md("""
# LendingClub Interest Rate EDA — `true data/`

**Goal:** Understand the structure, quality, and signal in `true data/LC_train.csv` (100k loans, 39 cols) before modeling `int_rate`. This EDA is application-time honest: it uses the data dictionary to verify what each field means, keeps raw inputs separate from safe analysis frames, and excludes fields unavailable at origination.

This dataset differs materially from `data/LC_*.csv`:

- **FICO scores are present** (`fico_range_low`, `fico_range_high`) — they were stripped from the prior slice
- **`loan_status` is present** — a post-origination status field; excluded because it is unavailable at application time
- **No date columns** (no `issue_d`, no `earliest_cr_line`) — temporal validation is not possible on this slice
- **No `grade`, `sub_grade`, or `installment`** — the obvious pre-origination leakage trio is gone
- Several numerics arrive as strings (`dti`, `revol_util`, `all_util`, `mths_since_*`, `mo_sin_old_il_acct`) with literal `"NA"` for nulls — cast at load time

This notebook does NOT fit models. It uses only diagnostic derived fields needed to audit relationships and recommend modeling choices.
""")

md("## 0. Setup")
code("""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

np.random.seed(42)
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# Path resolution works whether run from notebooks/ or project root.
PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "true data"
DICTIONARY_PATH = DATA_DIR / "LCDataDictionary.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(name: str, fig=None):
    fig = fig or plt.gcf()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}.png")
    return OUTPUT_DIR / f"{name}.png"

print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DATA_DIR     = {DATA_DIR}")
print(f"DICTIONARY   = {DICTIONARY_PATH}")
print(f"OUTPUT_DIR   = {OUTPUT_DIR}")
""")

code("""
# Numeric columns the CSV stores as strings (e.g. "26.33", "NA"). Polars
# reads them as String unless overridden. The "NA" literal is handled by
# passing null_values=["NA"] to pl.read_csv.
STRING_NUMERIC_COLS = [
    "dti", "revol_util", "all_util",
    "mo_sin_old_il_acct",
    "mths_since_last_record", "mths_since_rcnt_il",
    "mths_since_recent_bc", "mths_since_recent_inq",
]
# tot_cur_bal arrives as Float64 in train but Int64 in test — force Float64
# in both for parity.
NUMERIC_OVERRIDES = {c: pl.Float64 for c in STRING_NUMERIC_COLS + ["tot_cur_bal"]}

# loan_status is the current lifecycle status of the loan. The data dictionary
# confirms it is not an application input, so it is unavailable at origination.
# It must be excluded from application-time EDA and modeling even if its
# marginal relationship with int_rate is modest.
LEAKAGE_COLS_POST_ORIGINATION = ["loan_status"]

# Categoricals to treat as native XGBoost categoricals (no one-hot).
CATEGORICAL_COLS = [
    "home_ownership", "verification_status", "purpose",
    "application_type", "addr_state", "term",
]

# emp_length ordinal map — same as the archived pipeline.
EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10, "n/a": None,
}
""")

# ============================================================================
# SECTION 1 — Schema audit
# ============================================================================
md("""
## 1. Schema & data quality audit

Load raw train/test with `null_values=["NA"]` and Float64 overrides for the string-encoded numeric columns. Bridge to pandas. Verify shapes, dictionary coverage, dtype parity, ID uniqueness, target absence in test, and observed encodings. (No `issue_d` exists, so no chronological ordering check.)
""")
code("""
train_raw = pl.read_csv(
    DATA_DIR / "LC_train.csv",
    schema_overrides=NUMERIC_OVERRIDES,
    null_values=["NA"],
    infer_schema_length=20000,
).to_pandas()

test_raw = pl.read_csv(
    DATA_DIR / "LC_test.csv",
    schema_overrides=NUMERIC_OVERRIDES,
    null_values=["NA"],
    infer_schema_length=20000,
).to_pandas()

data_dict = pd.read_excel(DICTIONARY_PATH, sheet_name="LoanStats")
data_dict["Variable Name"] = data_dict["Variable Name"].astype(str)
dict_cols = set(data_dict["Variable Name"])

print(f"train_raw shape: {train_raw.shape}")
print(f"test_raw  shape: {test_raw.shape}")
print(f"dictionary rows: {len(data_dict)}")

assert train_raw.shape == (100_000, 39), f"unexpected train shape: {train_raw.shape}"
assert test_raw.shape == (10_000, 39),   f"unexpected test shape: {test_raw.shape}"
assert "int_rate" in train_raw.columns and "int_rate" not in test_raw.columns
assert "ID" in test_raw.columns and "ID" not in train_raw.columns
assert test_raw["ID"].is_unique and test_raw["ID"].nunique() == len(test_raw)
assert len(data_dict) == 39
assert set(train_raw.columns) == dict_cols
assert set(test_raw.columns) - dict_cols == {"ID"}
assert dict_cols - set(test_raw.columns) == {"int_rate"}
print("Raw schema and dictionary assertions passed.")

schema_audit = pd.DataFrame({
    "check": [
        "train columns covered by dictionary",
        "dictionary columns missing from train",
        "test-only columns outside dictionary",
        "dictionary columns absent from test",
        "test ID unique",
    ],
    "result": [
        len(set(train_raw.columns) & dict_cols),
        sorted(dict_cols - set(train_raw.columns)),
        sorted(set(test_raw.columns) - dict_cols),
        sorted(dict_cols - set(test_raw.columns)),
        bool(test_raw["ID"].is_unique),
    ],
})
display(schema_audit)
""")

code("""
# Dtype parity check between train and test (excluding the asymmetric int_rate / ID columns)
dtype_train = train_raw.drop(columns=["int_rate"]).dtypes
dtype_test  = test_raw.drop(columns=["ID"]).dtypes
parity = pd.DataFrame({"train": dtype_train, "test": dtype_test})
parity["match"] = parity["train"].astype(str) == parity["test"].astype(str)
mismatches = parity[~parity["match"]]
print(f"Dtype mismatches: {len(mismatches)}")
if len(mismatches):
    display(mismatches)
else:
    print("All shared columns have matching dtypes.")
""")

code("""
def observed_values(series, max_items=12):
    vals = series.dropna().astype(str).value_counts().head(max_items)
    return ", ".join([f"{repr(k)} ({v:,})" for k, v in vals.items()])

dict_lookup = data_dict.set_index("Variable Name")["Description"].to_dict()
encoding_audit = pd.DataFrame([
    {
        "column": "term",
        "dictionary_expectation": dict_lookup["term"],
        "observed_train_encoding": observed_values(train_raw["term"]),
        "implication": "Strip leading spaces and parse 36/60 into term_months before modeling.",
    },
    {
        "column": "emp_length",
        "dictionary_expectation": dict_lookup["emp_length"],
        "observed_train_encoding": observed_values(train_raw["emp_length"]),
        "implication": "Observed as LendingClub text labels; map with EMP_LENGTH_MAP and treat missing explicitly.",
    },
    {
        "column": "home_ownership",
        "dictionary_expectation": dict_lookup["home_ownership"],
        "observed_train_encoding": observed_values(train_raw["home_ownership"]),
        "implication": "Dictionary allows OTHER, but train/test observe only MORTGAGE/RENT/OWN; encoder should still tolerate OTHER.",
    },
    {
        "column": "loan_status",
        "dictionary_expectation": dict_lookup["loan_status"],
        "observed_train_encoding": observed_values(train_raw["loan_status"]),
        "implication": "Current loan status is not an application-time input; audit only, then drop.",
    },
])
display(encoding_audit)
""")

md("""
**Schema takeaway:** The raw train file exactly matches the 39-column data dictionary; test differs only by replacing `int_rate` with `ID`. From here on, all signal analysis uses `train_safe` / `test_safe`, where post-origination `loan_status` has been removed.
""")

code("""
# *** LEAKAGE GUARD ***
# loan_status is the loan's current lifecycle status, not a borrower/application
# input available when the interest rate is set. We exclude it from all safe
# bivariate/correlation/drift/feature recommendation analysis.
#
# We KEEP a side copy (loan_status_audit) for use only in §8 (leakage audit),
# where we quantify the leakage explicitly. That copy is discarded after §8.
loan_status_audit = train_raw[["loan_status", "int_rate"]].copy()
train_safe = train_raw.drop(columns=LEAKAGE_COLS_POST_ORIGINATION).copy()
test_safe  = test_raw.drop(columns=LEAKAGE_COLS_POST_ORIGINATION).copy()
print(f"Dropped: {LEAKAGE_COLS_POST_ORIGINATION}")
print(f"train_safe shape after availability drop: {train_safe.shape}")
print(f"test_safe shape after availability drop:  {test_safe.shape}")
assert "loan_status" not in train_safe.columns
assert "loan_status" not in test_safe.columns

# Downstream cells use short aliases for the safe application-time frames.
# The raw frames remain available as train_raw/test_raw for schema audits only.
train = train_safe
test = test_safe
""")

code("""
# Full-row duplicate check
n_dupes = train_raw.duplicated().sum()
print(f"Full-row duplicates in train: {n_dupes}")

# Confirm no date columns exist (different from the prior data/ slice)
date_like = [c for c in train_raw.columns if any(s in c.lower() for s in ("date", "_d", "issue", "earliest"))]
print(f"Date-like columns: {date_like or 'none — no temporal structure available'}")
""")

# ============================================================================
# SECTION 2 — Target distribution
# ============================================================================
md("""
## 2. Target distribution (`int_rate`)

`int_rate` is the APR percentage on each loan. We check shape, central tendency, tail behaviour, and whether a transform (log / sqrt / Box-Cox) would meaningfully reduce skew before modeling.
""")
code("""
y = train["int_rate"]
desc = y.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(4)
desc["skew"] = y.skew()
desc["kurtosis"] = y.kurtosis()
print(desc.to_string())
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram + KDE
sns.histplot(y, bins=50, kde=True, ax=axes[0], color="steelblue")
for q in [0.05, 0.95]:
    axes[0].axvline(y.quantile(q), color="orange", linestyle="--", linewidth=1)
axes[0].set_title(f"Histogram + KDE (mean={y.mean():.2f}, median={y.median():.2f})")
axes[0].set_xlabel("int_rate (%)")

# Boxplot
sns.boxplot(x=y, ax=axes[1], color="steelblue")
axes[1].set_title("Boxplot")
axes[1].set_xlabel("int_rate (%)")

# Q-Q vs normal
stats.probplot(y, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q vs Normal")
axes[2].get_lines()[0].set_markersize(2)

save_fig("02_target_distribution")
plt.show()
""")

code("""
def report_skew(name, vals):
    print(f"  {name:<15s} skew={stats.skew(vals):+.4f}  kurt={stats.kurtosis(vals):+.4f}")

print("Skew before / after candidate transforms:")
report_skew("identity",  y.values)
report_skew("log1p",     np.log1p(y.values))
report_skew("sqrt",      np.sqrt(y.values))
bc, lam = stats.boxcox(y.values + 1e-6)
report_skew(f"box-cox(λ={lam:.2f})", bc)
print()
print("Recommendation: keep target-transform decisions empirical. Box-Cox/log reduce univariate skew,")
print("but tree ensembles should compare raw vs transformed targets by cross-validation RMSE.")
""")

md("""
**Target takeaway:** `int_rate` is moderately right-skewed but bounded and discrete. A transform can make residuals nicer for some learners, but it is not an EDA mandate; accept it only if validation RMSE improves.
""")

# ============================================================================
# SECTION 3 — Missingness
# ============================================================================
md("""
## 3. Missingness analysis

Per-column null counts after the `"NA"` literals have been promoted to true nulls at load time. Without `issue_d` we can't probe the temporal mechanism of missingness — we test the remaining mechanisms: co-occurrence patterns and target correlation of missingness indicators.
""")
code("""
null_counts = train.isna().sum().sort_values(ascending=False)
null_pct = (null_counts / len(train) * 100).round(2)
null_table = pd.DataFrame({"n_missing": null_counts, "pct": null_pct})
null_table = null_table[null_table["n_missing"] > 0]
print(f"Columns with any missingness: {len(null_table)}")
display(null_table.head(20))
""")

code("""
fig, ax = plt.subplots(figsize=(10, 6))
top = null_table.sort_values("pct").tail(15)
ax.barh(top.index, top["pct"], color="steelblue")
ax.set_xlabel("% missing")
ax.set_title("Top-15 columns by missingness (train)")
for i, (col, pct) in enumerate(zip(top.index, top["pct"])):
    ax.text(pct + 0.5, i, f"{pct:.1f}%", va="center", fontsize=9)
save_fig("03_missingness_top15")
plt.show()
""")

code("""
fig, ax = plt.subplots(figsize=(14, 6))
msno.matrix(train, sparkline=False, ax=ax, fontsize=9)
ax.set_title("Missingness matrix (row order = file order)")
save_fig("03_missingness_matrix")
plt.show()
""")

code("""
fig, ax = plt.subplots(figsize=(12, 9))
msno.heatmap(train, ax=ax, fontsize=9)
ax.set_title("Null co-occurrence heatmap (Spearman of is-null indicators)")
save_fig("03_missingness_cooccurrence")
plt.show()
""")

code("""
# Point-biserial: does missingness in column X correlate with int_rate?
pb_rows = []
for col in null_table.index:
    is_null = train[col].isna().astype(int)
    if is_null.sum() < 50 or is_null.sum() > len(train) - 50:
        continue
    r, p = stats.pointbiserialr(is_null, train["int_rate"])
    pb_rows.append({"column": col, "pct_missing": null_table.loc[col, "pct"],
                    "pointbiserial_r": r, "p_value": p})
pb_df = pd.DataFrame(pb_rows)
pb_df["abs_r"] = pb_df["pointbiserial_r"].abs()
pb_df["flag_action"] = np.select(
    [
        (pb_df["pct_missing"] >= 5) & (pb_df["abs_r"] >= 0.02),
        (pb_df["pct_missing"] >= 1) & (pb_df["abs_r"] >= 0.02),
        (pb_df["pct_missing"] >= 5),
    ],
    [
        "add missingness flag",
        "consider flag; validate in CV",
        "usually impute only; weak target association",
    ],
    default="impute only; tiny missingness/effect",
)
pb_df = pb_df.sort_values(["abs_r", "pct_missing"], ascending=False)
print("Missingness indicator relationship with int_rate (effect-size first; p-values are secondary at n=100k):")
display(pb_df.head(15).round(4))
""")

md("""
**Missingness takeaway:** Use missingness flags only where missingness is both material and measurably associated with `int_rate`. In this slice, the strongest candidates are `mths_since_recent_inq` and `mths_since_last_record`; tiny-missingness fields like `dti` and `revol_util` should normally be imputed without extra flags.
""")

# ============================================================================
# SECTION 4 — Univariate
# ============================================================================
md("""
## 4. Univariate distributions

Distribution shape for every numeric column, value counts for every categorical, plus a focused look at the `emp_length` and `term` text encodings.
""")
code("""
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "int_rate"]
print(f"{len(numeric_cols)} numeric columns to plot")

skew_table = pd.DataFrame({
    "skew":      [train[c].skew() for c in numeric_cols],
    "n_unique":  [train[c].nunique() for c in numeric_cols],
    "n_zero":    [(train[c] == 0).sum() for c in numeric_cols],
    "pct_zero":  [(train[c] == 0).mean() * 100 for c in numeric_cols],
}, index=numeric_cols).round(3)
skew_table = skew_table.sort_values("skew", key=abs, ascending=False)
print("Skewness ranking (by |skew|):")
display(skew_table.head(20))
""")

code("""
# Grid of numeric histograms — log scale where skew > 3 and positive
ncols = 5
nrows = int(np.ceil(len(numeric_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.4))
for i, col in enumerate(numeric_cols):
    ax = axes.flat[i]
    series = train[col].dropna()
    if len(series) == 0:
        ax.axis("off"); continue
    log_scale = abs(series.skew()) > 3 and (series > 0).all()
    if log_scale:
        ax.hist(np.log1p(series), bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(f"{col}\\n(log1p, skew={series.skew():.1f})", fontsize=8)
    else:
        ax.hist(series, bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(f"{col}\\n(skew={series.skew():.2f})", fontsize=8)
    ax.tick_params(labelsize=7)
for j in range(len(numeric_cols), nrows * ncols):
    axes.flat[j].axis("off")
save_fig("04_numeric_histograms")
plt.show()
""")

code("""
# Categorical value-count bars
cat_cols = [c for c in ["home_ownership", "verification_status", "purpose",
                        "application_type", "term"] if c in train.columns]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, cat_cols):
    vc = train[col].value_counts(dropna=False).head(15)
    ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color="steelblue")
    ax.set_title(f"{col} ({train[col].nunique(dropna=False)} unique)")
    ax.tick_params(labelsize=8)
for j in range(len(cat_cols), 6):
    axes.flat[j].axis("off")
plt.suptitle("Categorical value counts (low-cardinality fields)")
save_fig("04_categorical_counts")
plt.show()
""")

code("""
# emp_length raw values + ordinal mapping proposal
emp_vc = train["emp_length"].value_counts(dropna=False)
print("emp_length raw value counts:")
print(emp_vc.to_string())
print("\\nProposed ordinal mapping (mirrors src/archive/preprocessing_and_modeling.py):")
for k, v in EMP_LENGTH_MAP.items():
    print(f"  {k:<12s} -> {v}")
""")

# ============================================================================
# SECTION 5 — FICO score analysis  (replaces temporal analysis from prior EDA)
# ============================================================================
md("""
## 5. FICO score analysis

The prior `data/` slice had no FICO columns — they were stripped under TransUnion licensing. This `true data/` slice has both `fico_range_low` and `fico_range_high`. FICO is the canonical credit-quality signal in consumer lending; expect it to be the strongest non-leakage predictor.

(This section replaces the temporal analysis from the prior notebook — there is no `issue_d` here, so temporal validation is impossible. **The validation strategy must be a random split** on this dataset, with the caveat that out-of-time generalization cannot be measured.)
""")
code("""
# fico_range_high - fico_range_low is almost always 4 in this slice
# (14 rows are 5), so the two columns carry the same information.
fico_diff = (train["fico_range_high"] - train["fico_range_low"]).unique()
print(f"Unique values of (fico_range_high - fico_range_low): {sorted(fico_diff.tolist())}")
print()
# Add diagnostic derived fields to the safe application-time frames only.
train["fico"] = (train["fico_range_low"] + train["fico_range_high"]) / 2.0
test["fico"]  = (test["fico_range_low"]  + test["fico_range_high"])  / 2.0
train["term_months"] = train["term"].astype(str).str.extract(r"(36|60)", expand=False).astype("int16")
test["term_months"]  = test["term"].astype(str).str.extract(r"(36|60)", expand=False).astype("int16")
train["zip3"] = train["zip_code"].astype(str).str[:3]
test["zip3"]  = test["zip_code"].astype(str).str[:3]
train["zip_first_digit"] = train["zip_code"].astype(str).str[0]
test["zip_first_digit"]  = test["zip_code"].astype(str).str[0]
print(f"FICO range: {train['fico'].min():.0f} - {train['fico'].max():.0f}")
print(f"FICO mean:  {train['fico'].mean():.1f}, median: {train['fico'].median():.0f}, std: {train['fico'].std():.1f}")
print("Derived fields added: fico, term_months, zip3, zip_first_digit")
""")

code("""
# FICO vs int_rate — the headline relationship
sub = train[["fico", "int_rate"]].dropna()
pr, _ = stats.pearsonr(sub["fico"], sub["int_rate"])
sr, _ = stats.spearmanr(sub["fico"], sub["int_rate"])

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Hexbin scatter
axes[0].hexbin(sub["fico"], sub["int_rate"], gridsize=40, cmap="Blues", mincnt=1)
# Binned mean overlay
bins = pd.qcut(sub["fico"], q=20, duplicates="drop")
binned = sub.groupby(bins, observed=True).agg({"fico": "mean", "int_rate": "mean"})
axes[0].plot(binned["fico"], binned["int_rate"], color="orange", linewidth=2.5, marker="o", markersize=4)
axes[0].set_xlabel("FICO score")
axes[0].set_ylabel("int_rate (%)")
axes[0].set_title(f"int_rate vs FICO (pearson={pr:+.3f}, spearman={sr:+.3f})")

# FICO distribution
axes[1].hist(train["fico"].dropna(), bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
axes[1].set_xlabel("FICO score")
axes[1].set_ylabel("count")
axes[1].set_title(f"FICO score distribution (n={train['fico'].notna().sum():,})")

save_fig("05_fico_analysis")
plt.show()
print(f"\\nFICO is the dominant non-leakage signal: pearson r = {pr:.4f}")
""")

md("""
**FICO takeaway:** `fico_range_low` and `fico_range_high` are effectively duplicate boundaries, so the midpoint `fico` is the clean modeling representation. Keep the midpoint and drop the raw pair before modeling to avoid duplicated attribution.
""")

# ============================================================================
# SECTION 6 — Bivariate vs target
# ============================================================================
md("""
## 6. Bivariate analysis vs target

Spearman correlation handles the skew in monetary columns better than Pearson. The categorical boxplots and the `addr_state` bar chart probe what each non-numeric column buys you.
""")
code("""
# Pearson + Spearman ranking. Use the FICO midpoint to avoid triple-counting
# fico_range_low, fico_range_high, and fico as three copies of the same signal.
CORR_EXCLUDE = {"int_rate", "ID", "fico_range_low", "fico_range_high"}
features_for_corr = [c for c in numeric_cols if c not in CORR_EXCLUDE]
# Add the derived FICO midpoint we computed in §5.
features_for_corr = list(dict.fromkeys(features_for_corr + ["fico"]))
assert "loan_status" not in features_for_corr

rows = []
for c in features_for_corr:
    valid = train[[c, "int_rate"]].dropna()
    if len(valid) < 100:
        continue
    pr, _ = stats.pearsonr(valid[c], valid["int_rate"])
    sr, _ = stats.spearmanr(valid[c], valid["int_rate"])
    rows.append({"feature": c, "pearson": pr, "spearman": sr, "n": len(valid)})
corr_df = pd.DataFrame(rows).set_index("feature")
corr_df["abs_spearman"] = corr_df["spearman"].abs()
corr_df = corr_df.sort_values("abs_spearman", ascending=False)
print("Top 15 features by |Spearman| with int_rate:")
display(corr_df.head(15).round(4))
""")

code("""
# Hexbin grid for top-15 most-correlated numerics
top15_feats = corr_df.head(15).index.tolist()
fig, axes = plt.subplots(3, 5, figsize=(18, 10))
for ax, col in zip(axes.flat, top15_feats):
    sub = train[[col, "int_rate"]].dropna()
    if len(sub) < 100:
        ax.axis("off"); continue
    lo, hi = sub[col].quantile([0.01, 0.99])
    sub = sub[(sub[col] >= lo) & (sub[col] <= hi)]
    ax.hexbin(sub[col], sub["int_rate"], gridsize=30, cmap="Blues", mincnt=1)
    try:
        bins = pd.qcut(sub[col], q=20, duplicates="drop")
        binned = sub.groupby(bins, observed=True).agg({col: "mean", "int_rate": "mean"})
        ax.plot(binned[col], binned["int_rate"], color="orange", linewidth=2, marker="o", markersize=3)
    except ValueError:
        pass
    sp = corr_df.loc[col, "spearman"]
    ax.set_title(f"{col}\\n(spearman={sp:+.3f})", fontsize=9)
    ax.tick_params(labelsize=7)
save_fig("06_bivariate_hexbin")
plt.show()
""")

code("""
# Categorical boxplots vs target
cat_for_box = [c for c in ["home_ownership", "verification_status", "purpose",
                           "term", "application_type"] if c in train.columns]
n = len(cat_for_box)
rows = int(np.ceil(n / 3))
fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 4.5))
for ax, col in zip(axes.flat, cat_for_box):
    order = train.groupby(col)["int_rate"].median().sort_values().index
    sns.boxplot(data=train, x=col, y="int_rate", order=order, ax=ax,
                color="steelblue", fliersize=1)
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.set_xlabel("")
for j in range(n, rows * 3):
    axes.flat[j].axis("off")
plt.suptitle("int_rate by categorical (sorted by median)")
save_fig("06_categorical_boxplots")
plt.show()
""")

code("""
# addr_state sorted bar with sample-size annotations
state_stats = train.groupby("addr_state")["int_rate"].agg(["mean", "count"]).sort_values("mean")
spread = state_stats['mean'].max() - state_stats['mean'].min()
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(state_stats.index, state_stats["mean"], color="steelblue")
ax.set_title(f"Mean int_rate by addr_state (spread = {spread:.2f} pp)")
ax.set_ylabel("mean int_rate (%)")
ax.tick_params(axis="x", labelsize=7)
ax.set_ylim(state_stats["mean"].min() - 0.5, state_stats["mean"].max() + 0.5)
save_fig("06_state_bar")
plt.show()
""")

md("""
**Bivariate takeaway:** Once FICO is represented once, the next strongest raw numeric signals are utilization and debt-burden fields. Treat state and term as useful categorical structure, but validate their incremental value in the modeling pipeline.
""")

# ============================================================================
# SECTION 7 — Correlation & multicollinearity
# ============================================================================
md("""
## 7. Correlation & multicollinearity

Pairwise feature correlations expose redundancy clusters. Pairs with |r|>0.85 are candidates for dropping the less interpretable member; XGBoost is robust to multicollinearity for predictive accuracy, but SHAP attributions split across correlated features.
""")
code("""
corr_cols = [c for c in numeric_cols if c not in {"ID"}] + ["fico"]
corr_cols = list(dict.fromkeys(corr_cols))
corr_mat = train[corr_cols + ["int_rate"]].corr()
order = ["int_rate"] + [c for c in corr_mat.columns if c != "int_rate"]
corr_mat = corr_mat.loc[order, order]

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_mat, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=False, cbar_kws={"label": "Pearson r"}, ax=ax)
ax.set_title("Pearson correlation (int_rate pulled to top)")
save_fig("07_correlation_heatmap")
plt.show()
""")

code("""
# High-correlation pairs (|r|>0.7) excluding self-pairs and int_rate
pair_corr = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool)).stack()
pair_corr = pair_corr[pair_corr.abs() > 0.7]
pair_corr = pair_corr.sort_values(key=abs, ascending=False)
pair_df = pair_corr.reset_index()
pair_df.columns = ["feature_a", "feature_b", "pearson_r"]
pair_df = pair_df[~pair_df.apply(lambda r: r["feature_a"] == "int_rate" or r["feature_b"] == "int_rate", axis=1)]
print(f"Numeric pairs with |Pearson r| > 0.7: {len(pair_df)}")
display(pair_df.round(4))
""")

# ============================================================================
# SECTION 8 — Leakage audit (focus on loan_status; not grade/installment which are absent)
# ============================================================================
md("""
## 8. Leakage audit

This dataset's leakage story is different from the prior `data/` slice:

- **`grade`, `sub_grade`, `installment`** — **ABSENT**. The classic pre-origination leakage trio doesn't exist here.
- **`loan_status`** — **PRESENT** and **already dropped** at the top of the notebook. It is the current status of the loan (Current, Fully Paid, Charged Off, In Grace Period, Late, Issued), which is not available as an application-time borrower feature. We quantify its association here using the side copy `loan_status_audit` saved before the drop, then discard the copy.

The exclusion is an availability-timing decision, not a claim that `loan_status` is the strongest predictor. Its observed association with `int_rate` is small but invalid for application-time modeling.
""")
code("""
# Compute the leakage using the side copy preserved before the drop
ls_audit = loan_status_audit.copy()
status_means = ls_audit.groupby("loan_status")["int_rate"].agg(["mean", "median", "std", "count"]).sort_values("mean")
print("Mean int_rate by loan_status:")
display(status_means.round(3))
print()
print(f"loan_status mean-rate spread (max − min): "
      f"{status_means['mean'].max() - status_means['mean'].min():.2f} pp")
""")

code("""
# Effect size: one-way ANOVA F-statistic
groups = [g["int_rate"].dropna().values for _, g in ls_audit.groupby("loan_status")]
groups = [g for g in groups if len(g) > 1]
F, p = stats.f_oneway(*groups)

# Eta squared: proportion of int_rate variance explained by loan_status
grand_mean = ls_audit["int_rate"].mean()
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
ss_total = ((ls_audit["int_rate"] - grand_mean) ** 2).sum()
eta_sq = ss_between / ss_total

print(f"One-way ANOVA F({len(groups)-1}, {sum(len(g)-1 for g in groups)}) = {F:.2f}")
print(f"p-value: {p:.3e}")
print(f"Eta-squared (variance explained by loan_status): {eta_sq:.4f}")
print()
print(f"→ loan_status explains {eta_sq*100:.2f}% of int_rate variance.")
print("  The effect is modest; the exclusion is because status is unavailable at application time.")
""")

code("""
# Visualize separation
fig, ax = plt.subplots(figsize=(12, 5))
order = ls_audit.groupby("loan_status")["int_rate"].median().sort_values().index
sns.boxplot(data=ls_audit, x="loan_status", y="int_rate", order=order,
            ax=ax, color="steelblue", fliersize=1)
ax.set_title("int_rate by loan_status (availability audit — excluded from safe analysis)")
ax.tick_params(axis="x", rotation=30, labelsize=9)
ax.set_xlabel("")
save_fig("08_loan_status_leakage")
plt.show()
""")

code("""
# Discard the audit copy — leakage column never re-enters working dataframes
del loan_status_audit, ls_audit
print("loan_status audit copy discarded.")
print()
print("LEAKAGE_COLS_POST_ORIGINATION =", LEAKAGE_COLS_POST_ORIGINATION)
print("(grade, sub_grade, installment from the prior dataset are NOT present here.)")
""")

# ============================================================================
# SECTION 9 — Cardinality
# ============================================================================
md("""
## 9. Cardinality analysis

High-cardinality string columns need an explicit encoding strategy. `emp_title` is free text; `title` is a one-to-one human-readable duplicate of `purpose` in this slice; `zip_code` is a masked 3-digit prefix; `addr_state` is ~50 levels.
""")
code("""
card_rows = []
exclude = {"int_rate"}
for c in train.columns:
    if c in exclude:
        continue
    nunique = train[c].nunique(dropna=True)
    n_missing = train[c].isna().sum()
    top = train[c].value_counts(dropna=True).head(3)
    top_str = ", ".join(f"{repr(v)[:20]}:{n}" for v, n in top.items())
    card_rows.append({
        "column": c,
        "dtype": str(train[c].dtype),
        "n_unique": nunique,
        "pct_missing": round(n_missing / len(train) * 100, 2),
        "top_3": top_str[:80],
    })
card_df = pd.DataFrame(card_rows).sort_values("n_unique", ascending=False)
print("Cardinality table (sorted by n_unique):")
display(card_df.head(20))
""")

code("""
title_purpose_pairs = (
    train.groupby(["purpose", "title"], dropna=False)
    .size()
    .reset_index(name="n")
    .sort_values(["purpose", "title"])
)
n_titles_per_purpose = title_purpose_pairs.groupby("purpose")["title"].nunique(dropna=False)
n_purposes_per_title = title_purpose_pairs.groupby("title")["purpose"].nunique(dropna=False)

print(f"Observed purpose/title pairs: {len(title_purpose_pairs)}")
print(f"purpose unique values: {train['purpose'].nunique(dropna=False)}")
print(f"title unique values:   {train['title'].nunique(dropna=False)}")
print(f"max titles per purpose: {n_titles_per_purpose.max()}")
print(f"max purposes per title: {n_purposes_per_title.max()}")
assert len(title_purpose_pairs) == train["purpose"].nunique(dropna=False) == train["title"].nunique(dropna=False)
assert n_titles_per_purpose.max() == 1 and n_purposes_per_title.max() == 1
display(title_purpose_pairs)
""")

code("""
recommendations = pd.DataFrame([
    ["emp_title",   "35k unique free text",    "DROP or hash — no clean grouping"],
    ["title",       "11 labels",               "DROP — one-to-one duplicate of purpose"],
    ["zip_code",    "masked 3-digit prefix",   "Optional — usually flat signal"],
    ["addr_state",  "~50 US states",           "KEEP — native XGBoost categorical"],
    ["purpose",     "11 levels",               "KEEP — native categorical"],
    ["emp_length",  "11 ordinal-like levels",  "ENCODE via EMP_LENGTH_MAP"],
    ["home_ownership",      "3 levels",        "KEEP — native categorical"],
    ["verification_status", "2 levels",        "KEEP — native categorical"],
    ["term",                "2 levels",        "EXTRACT integer months"],
    ["application_type",    "2 levels (87/13 split)", "KEEP or validate; not constant"],
], columns=["column", "cardinality", "recommendation"])
display(recommendations)
""")

md("""
**Cardinality takeaway:** `title` is not merely semantically similar to `purpose`; it is an exact 11-pair one-to-one recoding in this train slice. Drop `title`, keep `purpose`, and avoid spending model capacity on `emp_title` unless a later text-specific experiment proves value.
""")

# ============================================================================
# SECTION 10 — Train vs test parity (no temporal interpretation)
# ============================================================================
md("""
## 10. Train vs test distribution parity (drift check)

Without `issue_d` we cannot say whether train→test drift is temporal. We still quantify it: Kolmogorov–Smirnov per numeric, Population Stability Index (PSI, 10-bin), chi-square per categorical. Strong drift on a non-temporal split usually means the test split was sampled differently (different sampling weights, different rejection-inference treatment, etc.).
""")
code("""
def psi(expected, actual, bins=10):
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    if len(expected) < 50 or len(actual) < 50:
        return np.nan
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = pd.cut(expected, edges).value_counts(normalize=True).sort_index()
    a_pct = pd.cut(actual, edges).value_counts(normalize=True).sort_index()
    eps = 1e-6
    return float(((a_pct - e_pct) * np.log((a_pct + eps) / (e_pct + eps))).sum())

drift_rows = []
shared_numeric = [c for c in numeric_cols if c in test.columns and c != "ID"]
assert "loan_status" not in shared_numeric
for c in shared_numeric:
    a, b = train[c].dropna(), test[c].dropna()
    if len(a) < 50 or len(b) < 50:
        continue
    ks_stat, ks_p = stats.ks_2samp(a, b)
    drift_rows.append({
        "feature": c,
        "ks_D": ks_stat,
        "ks_p": ks_p,
        "psi": psi(a, b),
        "train_mean": a.mean(),
        "test_mean":  b.mean(),
        "delta_mean": b.mean() - a.mean(),
    })
drift_df = pd.DataFrame(drift_rows).set_index("feature").sort_values("ks_D", ascending=False)
print("Numeric drift between train and test (sorted by KS D-statistic):")
display(drift_df.round(4).head(20))
print()
print("PSI interpretation: <0.1 = stable, 0.1-0.2 = monitor, >0.2 = significant shift")
""")

code("""
top5_drift = drift_df.head(5).index.tolist()
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, col in zip(axes, top5_drift):
    a, b = train[col].dropna(), test[col].dropna()
    lo, hi = np.quantile(np.concatenate([a, b]), [0.01, 0.99])
    a_c, b_c = a[(a >= lo) & (a <= hi)], b[(b >= lo) & (b <= hi)]
    sns.kdeplot(a_c, ax=ax, label="train", color="steelblue", linewidth=2)
    sns.kdeplot(b_c, ax=ax, label="test",  color="orange",    linewidth=2)
    ax.set_title(f"{col}\\n(KS D={drift_df.loc[col, 'ks_D']:.3f}, PSI={drift_df.loc[col, 'psi']:.3f})",
                 fontsize=9)
    ax.legend(fontsize=8)
save_fig("10_drift_kde")
plt.show()
""")

code("""
chi_rows = []
shared_cats = [c for c in CATEGORICAL_COLS + ["emp_length", "purpose"]
               if c in train.columns and c in test.columns]
shared_cats = list(dict.fromkeys(shared_cats))
assert "loan_status" not in shared_cats

def cramers_v_bias_corrected(contingency):
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.sum()
    r, k = contingency.shape
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((k_corr - 1), (r_corr - 1))
    v = np.sqrt(phi2_corr / denom) if denom > 0 else np.nan
    return chi2, p, dof, v

for c in shared_cats:
    a_vc = train[c].value_counts(dropna=False)
    b_vc = test[c].value_counts(dropna=False)
    common = sorted(set(a_vc.index) | set(b_vc.index), key=lambda x: str(x))
    a_arr = np.array([a_vc.get(k, 0) for k in common])
    b_arr = np.array([b_vc.get(k, 0) for k in common])
    contingency = np.array([a_arr, b_arr])
    if contingency.sum() == 0 or (contingency.sum(axis=0) == 0).any():
        continue
    chi2, p, dof, cramers_v = cramers_v_bias_corrected(contingency)
    chi_rows.append({"feature": c, "cramers_v": cramers_v, "chi2": chi2, "p": p, "dof": dof, "n_levels": len(common)})
chi_df = pd.DataFrame(chi_rows).set_index("feature").sort_values("cramers_v", ascending=False)
print("Categorical drift between train and test (sorted by bias-corrected Cramer's V effect size):")
display(chi_df.round(4))
""")

md("""
**Drift takeaway:** The largest train/test shift is practical, not just statistical: test loans are smaller and have a different term mix. Monitor `loan_amnt`, `term`, and recent-credit-history fields most closely when interpreting test predictions.
""")

# ============================================================================
# SECTION 11 — Geographic analysis
# ============================================================================
md("""
## 11. Geographic analysis

State and ZIP-code level signal in `int_rate`.
""")
code("""
state_stats = train.groupby("addr_state")["int_rate"].agg(["mean", "median", "std", "count"]).sort_values("mean")
print(f"State spread: {state_stats['mean'].max() - state_stats['mean'].min():.3f} pp")
print(f"Top 5 highest mean rate:")
display(state_stats.tail(5).round(3))
print(f"Bottom 5 lowest mean rate:")
display(state_stats.head(5).round(3))
""")

code("""
# Zip first-digit signal
zip_first = train.groupby("zip_first_digit")["int_rate"].agg(["mean", "count"]).sort_index()
print(f"zip first-digit spread: {zip_first['mean'].max() - zip_first['mean'].min():.3f} pp")
display(zip_first.round(3))

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(zip_first.index, zip_first["mean"], color="steelblue")
ax.set_title(
    f"Mean int_rate by zip-first-digit "
    f"(spread {zip_first['mean'].max() - zip_first['mean'].min():.2f} pp)"
)
ax.set_ylabel("mean int_rate (%)")
ax.set_xlabel("first digit of zip_code")
save_fig("11_zip_first_digit")
plt.show()
""")

code("""
# Zip nests within state? Check whether each zip3 maps to a single state.
nest = train.groupby("zip3")["addr_state"].nunique()
print(f"zip3 unique values: {len(nest)}")
print(f"zip3 mapping to >1 state: {(nest > 1).sum()}")
print(f"→ {'Perfectly nested' if (nest > 1).sum() == 0 else 'NOT perfectly nested'}")
""")

# ============================================================================
# SECTION 12 — Data quality
# ============================================================================
md("""
## 12. Data quality issues

The string-encoded columns have already been cast to numeric at load time (with `"NA"` → null). This section surveys the remaining quality concerns: extreme values, suspicious zeros, encoding edge cases.
""")
code("""
ai = train["annual_inc"]
print(f"annual_inc summary:")
print(f"  median = ${ai.median():,.0f}")
print(f"  99th pctile = ${ai.quantile(0.99):,.0f}")
print(f"  max = ${ai.max():,.0f}")
print(f"  count > $1M = {(ai > 1_000_000).sum()}  ({(ai > 1_000_000).mean()*100:.2f}%)")
print()
print("Top 10 annual_inc values:")
display(train.nlargest(10, "annual_inc")[["annual_inc", "loan_amnt", "verification_status", "purpose", "int_rate"]])
""")

code("""
checks = pd.DataFrame([
    ["annual_inc == 0",   (train["annual_inc"] == 0).sum()],
    ["annual_inc isna",   train["annual_inc"].isna().sum()],
    ["dti == 0",          (train["dti"] == 0).sum()],
    ["dti < 0",           (train["dti"] < 0).sum()],
    ["dti > 100",         (train["dti"] > 100).sum()],
    ["revol_util == 0",   (train["revol_util"] == 0).sum()],
    ["revol_util > 150",  (train["revol_util"] > 150).sum()],
    ["loan_amnt == 0",    (train["loan_amnt"] == 0).sum()],
    ["loan > 5×annual_inc",
        ((train["loan_amnt"] > 5 * train["annual_inc"]) & (train["annual_inc"] > 0)).sum()],
    ["int_rate < 5",      (train["int_rate"] < 5).sum()],
    ["int_rate > 30",     (train["int_rate"] > 30).sum()],
    ["fico_range_low < 600",  (train["fico_range_low"] < 600).sum()],
    ["fico_range_low > 850",  (train["fico_range_low"] > 850).sum()],
], columns=["check", "n"])
display(checks)
""")

code("""
# Confirm string-encoded numerics were parsed correctly
print("Dtype of formerly-string-encoded numeric columns:")
print(train[STRING_NUMERIC_COLS].dtypes.to_string())
print()
print("Their null counts (driven by 'NA' literals at load time):")
print(train[STRING_NUMERIC_COLS].isna().sum().to_string())
""")

# ============================================================================
# SECTION 13 — Feature engineering hypotheses
# ============================================================================
md("""
## 13. Feature engineering hypotheses (catalog, no implementation)

What the EDA suggests building. Differences from the prior catalog: FICO becomes the primary main effect, `loan_status` does NOT appear (unavailable at application time), date-derived features are impossible (no `issue_d` / `earliest_cr_line`).
""")
code("""
fe_hypotheses = pd.DataFrame([
    ["use fico (midpoint of low/high)",   "Largest non-leakage signal (§5)",            "fico = (low + high) / 2"],
    ["log transform: annual_inc",          "Skew >> 3 (§4)",                              "annual_inc_log"],
    ["log transform: revol_bal, tot_cur_bal, tot_coll_amt",
                                           "Heavy right tails (§4)",                      "*_log"],
    ["ratio: loan_to_income",              "Underwriting signal (§12)",                   "loan_to_income"],
    ["ratio: revol_util × loan_amnt",      "Bivariate non-linear (§6, §7)",               "interaction term"],
    ["interaction: fico × revol_util",     "Credit quality × utilization (§5, §6)",       "interaction term"],
    ["flag: selected is_missing_<col>",
                                           "Only when missingness rate and effect size justify it (§3)", "validate binary flags in CV"],
    ["flag: zero-inflated counters",       "pub_rec, delinq_2yrs, chargeoff/collection counters, tot_coll_amt (§4)",
                                                                                          "*_flag binaries"],
    ["bin: dti_bucket, loan_amnt_bucket",  "Discrete tree splits (§6)",                   "categorical bins"],
    ["addr_state native categorical",      "0.4-3pp main effect + interactions (§6, §11)", "pl.Categorical"],
    ["DROP: emp_title, title",             "emp_title high-cardinality; title duplicates purpose (§9)", "in DROP_COLS"],
    ["DROP: loan_status",                  "Unavailable at application time (§8)",         "already removed"],
    ["NOT POSSIBLE: temporal features",    "No issue_d in this dataset (§1)",             "—"],
], columns=["hypothesis", "motivation (EDA section)", "implementation"])
display(fe_hypotheses)
""")

# ============================================================================
# SECTION 14 — Summary & modeling implications
# ============================================================================
md("""
## 14. Summary & modeling implications

### Top findings

1. **The data dictionary fully reconciles to train** — all 39 train columns are documented; test differs only by replacing `int_rate` with `ID` (§1).
2. **FICO is present and dominates non-leakage signal** — the prior `data/` slice was missing this column and the modeling ceiling was R² ≈ 0.52 as a result. FICO Spearman with `int_rate` is strongly negative; this dataset should support a substantially higher ceiling (§5).
3. **`loan_status` is unavailable at application time** — its marginal ANOVA effect is modest (eta-squared ≈ 0.44%), but it is current loan status rather than a borrower input. Dropped immediately on load (§8).
4. **`grade`, `sub_grade`, `installment` are absent** — the obvious pre-origination leakage trio is not in this slice. No need to drop them; the audit code from the prior notebook does not apply.
5. **No date columns** — neither `issue_d` nor `earliest_cr_line` exists. Temporal validation is impossible on this slice; use a random hold-out and accept that out-of-time generalization cannot be measured (§1, §5).
6. **Several numerics were string-encoded with literal `"NA"`** — `dti`, `revol_util`, `all_util`, `mo_sin_old_il_acct`, and four `mths_since_*` columns. Cast at load with `null_values=["NA"]` and Float64 overrides (§1, §12).
7. **`tot_cur_bal` dtype divergence between train and test** — Float64 vs Int64. Forced Float64 in both via `NUMERIC_OVERRIDES` (§1).
8. **Missingness flags should be selective** — `mths_since_recent_inq` and `mths_since_last_record` are the clearest candidates; low-missingness fields generally need imputation only (§3).
9. **Multicollinearity is mild** — only a few numeric pairs above |r|>0.7; `fico_range_low ↔ fico_range_high` is a near-duplicate by construction (diff is 4 for 99,986 rows and 5 for 14 rows); use the midpoint instead (§5, §7).
10. **State mean-rate spread is non-trivial** — but train/test drift is stronger for `loan_amnt`, `term`, and recent-credit-history fields (§10, §11).

### Pre-modeling checklist

- Drop `loan_status` unconditionally (already done at load).
- Drop `emp_title` (high-cardinality free text) and `title` (one-to-one duplicate of `purpose`).
- Parse `term` to integer months; map `emp_length` via `EMP_LENGTH_MAP`.
- Use `fico = (fico_range_low + fico_range_high) / 2` and drop the two raw columns to avoid the near-duplicate pair.
- Apply `log1p` to `annual_inc`, `revol_bal`, `tot_cur_bal`, `tot_coll_amt`.
- Add zero-inflated flags for `pub_rec`, `delinq_2yrs`, `chargeoff_within_12_mths`, `collections_12_mths_ex_med`, and `tot_coll_amt`.
- Add `is_missing_<col>` flags only for material missingness with non-trivial effect size, then keep only if CV improves (§3).
- Compute median imputation on train only; apply to val and test.
- Native XGBoost categoricals: `addr_state`, `home_ownership`, `verification_status`, `purpose`, `term`.
- **Validation: random split** (e.g. 80/20). No temporal column exists.

### Honest expectations

- With FICO present, validation R² should be materially higher than the prior FICO-stripped slice, but quote the ceiling only after running the random holdout model.
- The largest residual signal lives in `revol_util` and `dti` after FICO is conditioned out.
- Out-of-time generalization is **unmeasured** on this slice. If the model is deployed, monitor live performance against expected RMSE — drift can only be detected post-hoc.

### Risk register

Features with PSI ≥ 0.1 (train vs test, §10) are most exposed to whatever sampling mechanism produced the test split. If the model is retrained, recompute imputation medians.
""")

code("""
print("EDA complete. Figures written to outputs/eda/:")
for p in sorted(OUTPUT_DIR.glob('*.png')):
    print(f"  {p.name}")
print(f"\\nTotal: {len(list(OUTPUT_DIR.glob('*.png')))} figures")
""")
# ----------------------------------------------------------------------------

NB["cells"] = CELLS
NB["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out_path = Path(__file__).parent / "01_eda.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(NB, f)
print(f"Wrote {out_path} with {len(CELLS)} cells")
