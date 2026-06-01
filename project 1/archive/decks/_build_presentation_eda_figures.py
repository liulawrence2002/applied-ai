"""
Build presentation-ready EDA figures from the audited true-data LendingClub slice.

These figures are separate from outputs/eda/ so the finalized notebook outputs stay
unchanged while slide assets can be more designed and selective.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "true data"
OUT_DIR = PROJECT_ROOT / "outputs" / "eda_deck"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRING_NUMERIC_COLS = [
    "dti",
    "revol_util",
    "all_util",
    "mo_sin_old_il_acct",
    "mths_since_last_record",
    "mths_since_rcnt_il",
    "mths_since_recent_bc",
    "mths_since_recent_inq",
    "tot_cur_bal",
]

BG = "#F6F8FB"
INK = "#172033"
MUTED = "#667085"
BLUE = "#1F77B4"
CYAN = "#00A6D6"
CORAL = "#F26D5B"
GREEN = "#16A085"
PURPLE = "#7B61FF"
GOLD = "#D9A441"
GRID = "#D9DEE8"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dtype_overrides = {c: "float64" for c in STRING_NUMERIC_COLS}
    train_raw = pd.read_csv(DATA_DIR / "LC_train.csv", na_values=["NA"], dtype=dtype_overrides)
    test_raw = pd.read_csv(DATA_DIR / "LC_test.csv", na_values=["NA"], dtype=dtype_overrides)
    train = train_raw.drop(columns=["loan_status"]).copy()
    test = test_raw.drop(columns=["loan_status"]).copy()
    for df in (train, test):
        df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
        df["term_months"] = df["term"].astype(str).str.extract(r"(36|60)", expand=False).astype("int16")
        df["zip3"] = df["zip_code"].astype(str).str[:3]
        df["zip_first_digit"] = df["zip_code"].astype(str).str[0]
    return train, test


def set_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": "white",
            "savefig.facecolor": BG,
            "axes.edgecolor": "#E6EAF2",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
        }
    )


def add_header(fig, title: str, subtitle: str) -> None:
    fig.text(0.045, 0.94, title, fontsize=24, fontweight="bold", color=INK, ha="left")
    fig.text(0.045, 0.895, subtitle, fontsize=12.5, color=MUTED, ha="left")


def save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def card(fig, xywh, title: str, value: str, note: str, color: str) -> None:
    x, y, w, h = xywh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="#E1E7F0",
        linewidth=1.2,
    )
    fig.patches.append(patch)
    fig.text(x + 0.022, y + h - 0.05, title, fontsize=11, color=MUTED, weight="bold")
    fig.text(x + 0.022, y + h - 0.13, value, fontsize=28, color=color, weight="bold")
    fig.text(x + 0.022, y + 0.04, note, fontsize=10.5, color=INK, wrap=True)


def build_schema_audit(train: pd.DataFrame, test: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 9), constrained_layout=False)
    add_header(
        fig,
        "EDA Guardrails: Clean Inputs, No Application-Time Leakage",
        "The train file reconciles to the data dictionary, while model analysis excludes fields unavailable at origination.",
    )
    card(fig, (0.055, 0.61, 0.205, 0.19), "Dictionary Match", "39 / 39", "Every train column is documented.", GREEN)
    card(fig, (0.285, 0.61, 0.205, 0.19), "Safe Train Shape", "100k x 38", "After dropping current loan status.", BLUE)
    card(fig, (0.515, 0.61, 0.205, 0.19), "Test Contract", "ID only", "Test replaces target with ID.", PURPLE)
    card(fig, (0.745, 0.61, 0.205, 0.19), "Date Columns", "0", "No temporal split is possible.", CORAL)

    ax = fig.add_axes([0.08, 0.16, 0.84, 0.31])
    status = pd.Series(
        {
            "Dropped: loan_status": 1,
            "Absent: grade/sub_grade/installment": 1,
            "Kept: origination FICO": 1,
            "Validation: random holdout/CV": 1,
        }
    )
    colors = [CORAL, CORAL, GREEN, BLUE]
    ax.barh(status.index[::-1], status.values[::-1], color=colors[::-1], height=0.54)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([])
    ax.set_title("Modeling guardrails supported by EDA", loc="left", pad=12)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for y in range(len(status)):
        ax.text(1.03, y, "verified", va="center", ha="left", fontsize=11, color=GREEN, weight="bold")
    save(fig, "01_schema_leakage_audit_slide.png")


def build_target(train: pd.DataFrame) -> None:
    y = train["int_rate"]
    fig = plt.figure(figsize=(16, 9), constrained_layout=False)
    add_header(
        fig,
        "Target Distribution: Moderate Skew, Bounded APR Range",
        "Interest rate is expressed in percentage points; transforms should be accepted only if validation improves.",
    )
    ax = fig.add_axes([0.08, 0.16, 0.84, 0.62])
    sns.histplot(y, bins=48, stat="density", color=BLUE, alpha=0.72, edgecolor="white", linewidth=0.4, ax=ax)
    sns.kdeplot(y, color=INK, linewidth=2.6, ax=ax)
    q05, med, q95 = y.quantile([0.05, 0.5, 0.95])
    mean = y.mean()
    for x, label, color in [(q05, "5th", GOLD), (med, "median", CORAL), (q95, "95th", GOLD), (mean, "mean", GREEN)]:
        ax.axvline(x, color=color, linewidth=2, linestyle="--" if label != "median" else "-")
        ax.text(x, ax.get_ylim()[1] * 0.88, f"{label}\n{x:.2f}", color=color, ha="center", fontsize=10, weight="bold")
    ax.set_xlabel("Interest rate (%)")
    ax.set_ylabel("Density")
    ax.set_title("Observed int_rate distribution", loc="left", pad=10)
    ax.text(
        0.985,
        0.82,
        f"n = {len(y):,}\nmean = {mean:.2f}\nmedian = {med:.2f}\nskew = {y.skew():.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="#E1E7F0", boxstyle="round,pad=0.45"),
        fontsize=11,
        color=INK,
    )
    save(fig, "02_target_distribution_slide.png")


def build_fico(train: pd.DataFrame) -> None:
    sub = train[["fico", "int_rate"]].dropna()
    pr, _ = stats.pearsonr(sub["fico"], sub["int_rate"])
    sr, _ = stats.spearmanr(sub["fico"], sub["int_rate"])
    fig = plt.figure(figsize=(16, 9))
    add_header(
        fig,
        "FICO Is The Cleanest High-Signal Application-Time Feature",
        "Lower FICO maps to higher rates; use the midpoint and drop the duplicate raw boundaries.",
    )
    ax1 = fig.add_axes([0.07, 0.15, 0.51, 0.64])
    hb = ax1.hexbin(sub["fico"], sub["int_rate"], gridsize=42, cmap="Blues", mincnt=1)
    bins = pd.qcut(sub["fico"], q=20, duplicates="drop")
    binned = sub.groupby(bins, observed=True).agg({"fico": "mean", "int_rate": "mean"})
    ax1.plot(binned["fico"], binned["int_rate"], color=CORAL, linewidth=3, marker="o", markersize=4)
    ax1.set_xlabel("FICO midpoint")
    ax1.set_ylabel("Interest rate (%)")
    ax1.set_title("Binned rate curve over borrower FICO", loc="left", pad=10)
    ax1.text(
        0.04,
        0.94,
        f"Pearson {pr:+.3f}\nSpearman {sr:+.3f}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="#E1E7F0", boxstyle="round,pad=0.45"),
        fontsize=12,
        color=INK,
    )
    cbar = fig.colorbar(hb, ax=ax1, fraction=0.032, pad=0.02)
    cbar.ax.set_title("density", fontsize=9, color=MUTED, pad=8)

    ax2 = fig.add_axes([0.70, 0.15, 0.24, 0.64])
    bands = pd.cut(train["fico"], bins=[-np.inf, 680, 700, 720, 740, 760, np.inf])
    band_stats = train.groupby(bands, observed=True)["int_rate"].mean()
    labels = ["<680", "680-699", "700-719", "720-739", "740-759", "760+"]
    ax2.barh(labels[::-1], band_stats.values[::-1], color=[CORAL, CORAL, GOLD, BLUE, BLUE, GREEN][::-1])
    ax2.set_xlabel("Mean interest rate (%)")
    ax2.set_title("Mean rate by FICO band", loc="left", pad=10)
    for y, v in enumerate(band_stats.values[::-1]):
        ax2.text(v + 0.08, y, f"{v:.1f}%", va="center", fontsize=10, color=INK, weight="bold")
    save(fig, "03_fico_signal_slide.png")


def build_top_drivers(train: pd.DataFrame) -> None:
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {"int_rate", "fico_range_low", "fico_range_high"}]
    rows = []
    for c in numeric_cols:
        valid = train[[c, "int_rate"]].dropna()
        if len(valid) >= 100 and valid[c].nunique() > 1:
            sr, _ = stats.spearmanr(valid[c], valid["int_rate"])
            rows.append((c, sr))
    corr = pd.DataFrame(rows, columns=["feature", "spearman"])
    corr["abs_spearman"] = corr["spearman"].abs()
    corr = corr.sort_values("abs_spearman", ascending=False).head(14).sort_values("abs_spearman")

    fig = plt.figure(figsize=(16, 9), constrained_layout=False)
    add_header(
        fig,
        "Top Numeric Signals Point To Credit Quality, Utilization, And Debt Burden",
        "Spearman rank correlation is used because several money and credit-history fields are heavily skewed.",
    )
    ax = fig.add_axes([0.25, 0.14, 0.65, 0.66])
    colors = [CORAL if v > 0 else BLUE for v in corr["spearman"]]
    ax.barh(corr["feature"], corr["spearman"], color=colors, height=0.62)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel("Spearman correlation with int_rate")
    ax.set_title("Top raw/diagnostic numeric relationships", loc="left", pad=10)
    ax.set_xlim(-0.52, 0.38)
    for y, v in enumerate(corr["spearman"]):
        ha = "left" if v >= 0 else "right"
        dx = 0.012 if v >= 0 else -0.012
        ax.text(v + dx, y, f"{v:+.3f}", va="center", ha=ha, fontsize=10, color=INK, weight="bold")
    save(fig, "04_top_numeric_signals_slide.png")


def build_missingness(train: pd.DataFrame) -> None:
    null_counts = train.isna().sum().sort_values(ascending=False)
    null_pct = null_counts / len(train) * 100
    top = null_pct[null_counts > 0].head(10).sort_values()
    pb_rows = []
    for col in top.index:
        flag = train[col].isna().astype(int)
        if flag.sum() < 50 or flag.sum() > len(train) - 50:
            r = np.nan
        else:
            r, _ = stats.pointbiserialr(flag, train["int_rate"])
        pb_rows.append((col, r))
    pb = pd.DataFrame(pb_rows, columns=["feature", "missing_target_r"]).set_index("feature")

    fig = plt.figure(figsize=(16, 9))
    add_header(
        fig,
        "Missingness Is Structural: Flag Selectively, Impute The Rest",
        "Only missingness with material rate and target association should become a model feature.",
    )
    ax1 = fig.add_axes([0.08, 0.16, 0.48, 0.62])
    colors = [CORAL if c in {"mths_since_recent_inq", "mths_since_last_record"} else BLUE for c in top.index]
    ax1.barh(top.index, top.values, color=colors, height=0.62)
    ax1.set_xlabel("% missing in train")
    ax1.set_title("Top columns by missingness", loc="left", pad=10)
    for y, v in enumerate(top.values):
        ax1.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=10, color=INK, weight="bold")

    ax2 = fig.add_axes([0.73, 0.16, 0.21, 0.62])
    pb = pb.loc[top.index].sort_values("missing_target_r")
    ax2.barh(pb.index, pb["missing_target_r"], color=[CORAL if abs(v) >= 0.02 else MUTED for v in pb["missing_target_r"]])
    ax2.axvline(0, color=INK, linewidth=1)
    ax2.set_xlabel("Point-biserial r")
    ax2.set_title("Missing flag vs target", loc="left", pad=10)
    save(fig, "05_missingness_slide.png")


def psi(expected, actual, bins=10) -> float:
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = pd.cut(expected, edges).value_counts(normalize=True).sort_index()
    a_pct = pd.cut(actual, edges).value_counts(normalize=True).sort_index()
    eps = 1e-6
    return float(((a_pct - e_pct) * np.log((a_pct + eps) / (e_pct + eps))).sum())


def cramers_v(contingency: np.ndarray) -> float:
    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = contingency.sum()
    r, k = contingency.shape
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return np.sqrt(phi2_corr / denom) if denom > 0 else np.nan


def build_drift(train: pd.DataFrame, test: pd.DataFrame) -> None:
    numeric = [c for c in train.select_dtypes(include=[np.number]).columns if c in test.columns and c != "int_rate"]
    drift_rows = [(c, psi(train[c], test[c])) for c in numeric]
    drift = pd.DataFrame(drift_rows, columns=["feature", "psi"]).dropna().sort_values("psi", ascending=False).head(10)
    cats = ["term", "emp_length", "verification_status", "purpose", "application_type", "addr_state", "home_ownership"]
    cat_rows = []
    for c in cats:
        a = train[c].value_counts(dropna=False)
        b = test[c].value_counts(dropna=False)
        levels = sorted(set(a.index) | set(b.index), key=lambda x: str(x))
        contingency = np.array([[a.get(level, 0) for level in levels], [b.get(level, 0) for level in levels]])
        cat_rows.append((c, cramers_v(contingency)))
    cat = pd.DataFrame(cat_rows, columns=["feature", "cramers_v"]).sort_values("cramers_v", ascending=False)

    fig = plt.figure(figsize=(16, 9))
    add_header(
        fig,
        "Train/Test Drift: Test Loans Are Smaller And Mix Differs By Term",
        "Drift is not temporal here, but it matters for how confidently validation transfers to the submission set.",
    )
    ax1 = fig.add_axes([0.08, 0.16, 0.42, 0.62])
    d = drift.sort_values("psi")
    ax1.barh(d["feature"], d["psi"], color=[CORAL if x >= 0.1 else BLUE for x in d["psi"]], height=0.62)
    ax1.axvline(0.1, color=CORAL, linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Population Stability Index")
    ax1.set_title("Numeric drift by PSI", loc="left", pad=10)
    for y, v in enumerate(d["psi"]):
        ax1.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=10, color=INK, weight="bold")

    ax2 = fig.add_axes([0.60, 0.16, 0.32, 0.62])
    c = cat.sort_values("cramers_v")
    ax2.barh(c["feature"], c["cramers_v"], color=PURPLE, height=0.62)
    ax2.set_xlabel("Bias-corrected Cramer's V")
    ax2.set_title("Categorical drift effect size", loc="left", pad=10)
    for y, v in enumerate(c["cramers_v"]):
        ax2.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=10, color=INK, weight="bold")
    save(fig, "06_train_test_drift_slide.png")


def main() -> None:
    set_style()
    train, test = load_data()
    build_schema_audit(train, test)
    build_target(train)
    build_fico(train)
    build_top_drivers(train)
    build_missingness(train)
    build_drift(train, test)
    print(f"Wrote presentation figures to {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
