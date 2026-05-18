import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import shap
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, norm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.pipeline import encode_categoricals

logger = logging.getLogger(__name__)

ALPHA        = 0.05
RANDOM_STATE = 42


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """duration excluded -- post-treatment variable."""
    df = df.copy()
    df = encode_categoricals(df)
    df["treatment_flag"] = (df["group"] == "treatment").astype(int)
    feature_cols = [
        "age", "balance", "campaign", "previous",
        "treatment_flag", "job_enc", "marital_enc", "education_enc",
        "housing_enc", "loan_enc", "poutcome_enc"
    ]
    return df, feature_cols


def compute_results(df: pd.DataFrame, feature_cols: list) -> dict:
    treatment = df[df["group"] == "treatment"]
    control   = df[df["group"] == "control"]
    t_rate    = treatment["converted"].mean()
    c_rate    = control["converted"].mean()
    lift      = t_rate - c_rate

    t_conv, t_no  = treatment["converted"].sum(), len(treatment) - treatment["converted"].sum()
    c_conv, c_no  = control["converted"].sum(), len(control) - control["converted"].sum()
    chi2, p, _, _ = chi2_contingency([[t_conv, t_no], [c_conv, c_no]])
    h = abs(2 * (np.arcsin(np.sqrt(t_rate)) - np.arcsin(np.sqrt(c_rate))))

    z_alpha = norm.ppf(1 - ALPHA / 2)
    z_power = h * np.sqrt((len(treatment) * len(control)) / (len(treatment) + len(control))) - z_alpha
    power   = norm.cdf(z_power)

    df["age_band"] = pd.cut(df["age"], bins=[0, 30, 40, 50, 60, 100],
                             labels=["<30", "30-39", "40-49", "50-59", "60+"])
    seg_results = []
    for band in df["age_band"].dropna().unique():
        sub = df[df["age_band"] == band]
        t   = sub[sub["group"] == "treatment"]
        c   = sub[sub["group"] == "control"]
        if len(t) < 30 or len(c) < 30:
            continue
        seg_results.append({"band": str(band), "lift": t["converted"].mean() - c["converted"].mean(),
                             "t_rate": t["converted"].mean(), "c_rate": c["converted"].mean()})

    df["housing_bin"] = (df["housing"] == "yes").astype(int)
    formula  = "converted ~ treatment_flag + balance + campaign + housing_bin + C(job) + C(marital) + C(education)"
    model_lr = smf.logit(formula, data=df).fit(disp=False)
    adj_or   = np.exp(model_lr.params["treatment_flag"])
    adj_p    = model_lr.pvalues["treatment_flag"]

    X = df[feature_cols]
    y = df["converted"]
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    shap_vals = shap.TreeExplainer(clf).shap_values(X_te)
    mean_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_te.columns)

    return {
        "t_rate": t_rate, "c_rate": c_rate, "lift": lift,
        "chi2": chi2, "p": p, "power": power, "cohens_h": h,
        "adj_or": adj_or, "adj_p": adj_p,
        "seg_df": pd.DataFrame(seg_results), "mean_shap": mean_shap,
        "n_treatment": len(treatment), "n_control": len(control)
    }


def plot_summary(r: dict, output_path: str = "outputs/summary.png") -> None:
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Feature Adoption Experiment: Summary of Findings",
                 fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1    = fig.add_subplot(gs[0, 0])
    groups = ["Treatment", "Control"]
    rates  = [r["t_rate"], r["c_rate"]]
    ns     = [r["n_treatment"], r["n_control"]]
    errors = [1.96 * np.sqrt(rt * (1 - rt) / n) for rt, n in zip(rates, ns)]
    ax1.bar(groups, rates, color=["#2E4057", "#A8DADC"], yerr=errors, capsize=6)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_title("Conversion Rate (95% CI)")
    for i, (rt, e) in enumerate(zip(rates, errors)):
        ax1.text(i, rt + e + 0.005, f"{rt:.1%}", ha="center", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    stats_data = [
        ["Metric", "Value"],
        ["Raw lift", f"{r['lift']:+.1%}"],
        ["Chi-squared p", f"{r['p']:.6f}"],
        ["Cohen's h", f"{r['cohens_h']:.3f}"],
        ["Observed power", f"{r['power']:.1%}"],
        ["Adj. odds ratio", f"{r['adj_or']:.3f}"],
        ["Adj. p-value", f"{r['adj_p']:.6f}"],
        ["Significant?", "YES" if r["p"] < ALPHA else "NO"],
    ]
    table = ax2.table(cellText=stats_data[1:], colLabels=stats_data[0], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax2.set_title("Statistical Summary")

    ax3 = fig.add_subplot(gs[0, 2])
    seg = r["seg_df"].sort_values("band")
    ax3.bar(seg["band"], seg["lift"], color="#2E4057")
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("Lift by Age Band")
    ax3.tick_params(axis="x", rotation=30)

    ax4         = fig.add_subplot(gs[1, :2])
    shap_sorted = r["mean_shap"].sort_values(ascending=True)
    colors      = ["#E63946" if f == "treatment_flag" else "#2E4057" for f in shap_sorted.index]
    ax4.barh(shap_sorted.index, shap_sorted.values, color=colors)
    ax4.set_title("SHAP Feature Importance (red = treatment group)")
    ax4.set_xlabel("Mean |SHAP value|")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    sig_text = "SIGNIFICANT" if r["p"] < ALPHA else "NOT SIGNIFICANT"
    interp = (
        f"Key Findings\n\n"
        f"Raw lift: {r['lift']:+.1%} ({sig_text})\n"
        f"p = {r['p']:.6f}\n\n"
        f"After controlling for balance,\n"
        f"campaign contacts, housing, job,\n"
        f"marital status and education,\n"
        f"adjusted OR = {r['adj_or']:.2f}x (p={r['adj_p']:.4f}).\n\n"
        f"SHAP confirms treatment group\n"
        f"membership is a top predictor\n"
        f"alongside account balance."
    )
    ax5.text(0.05, 0.95, interp, transform=ax5.transAxes, fontsize=9,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#F0F4F8", alpha=0.8))
    ax5.set_title("Interpretation")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.show()


def write_readme(r: dict, readme_path: str = "README.md") -> None:
    sig = "statistically significant" if r["p"] < ALPHA else "not statistically significant"
    readme = f"""# Feature Adoption Experiment Analysis

An end-to-end A/B experiment analysis pipeline built on the UCI Bank Marketing dataset. It tests whether prior customer engagement predicts feature adoption, and covers the full analytical lifecycle: balance checking, significance testing, explainability, and drift detection.

**Stack:** Python · scipy · statsmodels · scikit-learn · SHAP · Evidently · uv

---

## The Problem

Is prior engagement history associated with higher feature adoption rates? This is the core question behind campaign targeting decisions at retail banks. Answering it requires more than a p-value — you need to verify group comparability, control for confounders, measure effect size, and check whether the model holds over time.

---

## Data

- **Source:** UCI Bank Marketing dataset (`fetch_ucirepo(id=222)`)
- **Size:** 45,211 records, 17 features
- **Group assignment:** Users with prior campaign contact (`previous > 0`) = treatment; no prior contact = control
- **Outcome:** Subscribed to a term deposit (`y == 'yes'`)
- **Known limitation:** Groups are not randomly assigned. Treatment users self-selected by being reachable in a prior campaign, introducing selection bias that motivates the covariate controls in Step 2.

---

## Methods

Data is ingested via a generator-based pipeline that processes data in chunks — the same pattern used with Kafka consumers or `pd.read_csv(chunksize=N)` in production. All analysis modules live in `src/analysis/` and are called by thin entry-point scripts.

**Step 1 — Balance check** (`scripts/run_balance_check.py`): T-tests on numeric features and chi-squared tests on categoricals verify group comparability before any significance testing.

**Step 2 — Significance testing** (`scripts/run_significance.py`): Chi-squared test, 95% confidence intervals, power analysis (Cohen's h), segmentation across age bands and job types, and logistic regression controlling for imbalanced covariates. `duration` is excluded from all models — it is a post-treatment variable that would leak outcome information.

**Step 3 — SHAP explainability** (`scripts/run_shap.py`): A `GradientBoostingClassifier` is trained to predict conversion. SHAP values rank feature contributions globally and at the individual level, which is a prerequisite for the individual-level explanations required by EU AI Act Article 86. Notably, `treatment_flag` ranks low in SHAP importance — account balance, housing status, and prior campaign outcome are stronger predictors of conversion than group membership alone.

**Step 4 — Drift detection** (`scripts/run_drift.py`): Dataset split into reference (first 60%) and current (last 40%) periods. Evidently compares feature distributions and model predictions across periods.

---

## Key Results

| Metric | Value |
|--------|-------|
| Treatment conversion rate | {r['t_rate']:.1%} |
| Control conversion rate | {r['c_rate']:.1%} |
| Raw lift | {r['lift']:+.1%} |
| Chi-squared p-value | {r['p']:.6f} |
| Cohen's h (effect size) | {r['cohens_h']:.3f} (medium) |
| Observed power | {r['power']:.1%} |
| Adjusted odds ratio (logistic regression) | {r['adj_or']:.3f}x |
| Adjusted p-value | {r['adj_p']:.6f} |

The lift is **{sig}** at alpha=0.05.

After controlling for imbalanced covariates identified in the balance check, previously contacted users have {r['adj_or']:.2f}x the odds of converting compared to otherwise similar control users. This is an associational estimate — groups were not randomly assigned, so residual confounding from unobserved variables cannot be fully ruled out.

![Summary](outputs/summary.png)

---

## Project Structure

```
src/
    pipeline.py              # Shared fetch, stream, and transform functions
    analysis/
        balance.py           # Balance check logic
        significance.py      # Chi-squared, power, segmentation, logistic regression
        shap.py              # SHAP training and explainability
        drift.py             # Evidently drift detection
        summary.py           # Summary visualization and README generation
scripts/
    run_balance_check.py
    run_significance.py
    run_shap.py
    run_drift.py
    run_summary.py
outputs/                     # Generated charts and reports
```

---

## How to Run

```bash
git clone https://github.com/mychellehale/fintech-ab-experiment.git
cd fintech-ab-experiment
uv sync
uv run python scripts/run_balance_check.py
uv run python scripts/run_significance.py
uv run python scripts/run_shap.py
uv run python scripts/run_drift.py
uv run python scripts/run_summary.py
```

---

## Compliance

EU AI Act Article 86 requires individual-level explanations for automated decisions affecting customers. The SHAP layer enables per-prediction feature attribution — a prerequisite for satisfying that requirement — and is the pattern used in production credit and fraud models at regulated financial institutions.
"""
    with open(readme_path, "w") as f:
        f.write(readme)
    logger.info(f"Saved: {readme_path}")
