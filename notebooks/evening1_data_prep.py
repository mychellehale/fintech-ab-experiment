"""
Feature Adoption Experiment Analysis
Evening 1: Data Loading, Group Definition, and Balance Checking

Dataset: UCI Bank Marketing
Download from: https://archive.ics.uci.edu/dataset/222/bank+marketing
Use the file: bank-full.csv (semicolon-delimited)

What we're simulating:
    A fintech product team ran an outreach campaign (treatment) to encourage
    users to adopt a savings feature. We want to know if it worked, and for whom.

    treatment group  = contacted via campaign (contact != 'unknown')
    control group    = not contacted (contact == 'unknown')
    primary metric   = feature adoption (y == 'yes')
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────

def load_data(filepath="bank-full.csv"):
    df = pd.read_csv(filepath, sep=";")
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(df.head(3))
    return df


# ─── 2. DEFINE TREATMENT AND CONTROL ──────────────────────────────────────────

def assign_groups(df):
    """
    Treatment: users who were contacted (campaign outreach)
    Control:   users who were not contacted
    Outcome:   subscribed to term deposit (proxy for feature adoption)
    """
    df = df.copy()
    df["group"] = np.where(df["contact"] != "unknown", "treatment", "control")
    df["converted"] = np.where(df["y"] == "yes", 1, 0)

    treatment = df[df["group"] == "treatment"]
    control   = df[df["group"] == "control"]

    print(f"\nGroup sizes:")
    print(f"  Treatment : {len(treatment):,}")
    print(f"  Control   : {len(control):,}")
    print(f"\nConversion rates:")
    print(f"  Treatment : {treatment['converted'].mean():.1%}")
    print(f"  Control   : {control['converted'].mean():.1%}")

    return df, treatment, control


# ─── 3. BALANCE CHECK ─────────────────────────────────────────────────────────
# This is the step most tutorials skip. A real A/B test requires the groups
# to be comparable on key covariates before we can attribute any difference
# in outcome to the treatment itself.

def check_balance(treatment, control):
    print("\n" + "="*55)
    print("BALANCE CHECK")
    print("="*55)
    print(f"{'Variable':<20} {'Treatment':>12} {'Control':>12} {'p-value':>10}")
    print("-"*55)

    # Numeric variables: compare means via t-test
    numeric_vars = ["age", "balance", "duration", "campaign", "previous"]
    for var in numeric_vars:
        t_mean = treatment[var].mean()
        c_mean = control[var].mean()
        _, p = stats.ttest_ind(treatment[var], control[var])
        flag = " *" if p < 0.05 else ""
        print(f"{var:<20} {t_mean:>12.2f} {c_mean:>12.2f} {p:>10.4f}{flag}")

    print()

    # Categorical variables: compare distributions via chi-squared
    cat_vars = ["job", "marital", "education", "housing"]
    for var in cat_vars:
        t_dist = treatment[var].value_counts(normalize=True)
        c_dist = control[var].value_counts(normalize=True)
        all_cats = set(t_dist.index) | set(c_dist.index)
        t_counts = [treatment[var].value_counts().get(c, 0) for c in all_cats]
        c_counts = [control[var].value_counts().get(c, 0) for c in all_cats]
        _, p = stats.chi2_contingency([t_counts, c_counts])[:2]
        flag = " *" if p < 0.05 else ""
        print(f"{var:<20} {'(categorical)':>12} {'':>12} {p:>10.4f}{flag}")

    print("\n* = significant imbalance (p < 0.05). Note these for Evening 2.")


# ─── 4. VISUALISE GROUPS ──────────────────────────────────────────────────────

def plot_group_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Experiment Group Overview", fontsize=13, fontweight="bold", y=1.01)

    # Group sizes
    counts = df["group"].value_counts()
    axes[0].bar(counts.index, counts.values, color=["#2E4057", "#A8DADC"])
    axes[0].set_title("Group Size")
    axes[0].set_ylabel("Users")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, f"{v:,}", ha="center", fontsize=9)

    # Conversion rate by group
    conv = df.groupby("group")["converted"].mean()
    bars = axes[1].bar(conv.index, conv.values, color=["#2E4057", "#A8DADC"])
    axes[1].set_title("Conversion Rate by Group")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for bar, val in zip(bars, conv.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.002,
                     f"{val:.1%}", ha="center", fontsize=9)

    # Age distribution
    for grp, color in zip(["treatment", "control"], ["#2E4057", "#A8DADC"]):
        subset = df[df["group"] == grp]["age"]
        axes[2].hist(subset, bins=20, alpha=0.6, label=grp, color=color)
    axes[2].set_title("Age Distribution by Group")
    axes[2].set_xlabel("Age")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("evening1_overview.png", dpi=150, bbox_inches="tight")
    print("\nSaved: evening1_overview.png")
    plt.show()


# ─── 5. SUMMARY FOR EVENING 2 ─────────────────────────────────────────────────

def evening1_summary(df, treatment, control):
    print("\n" + "="*55)
    print("READY FOR EVENING 2")
    print("="*55)
    print(f"Treatment conversion : {treatment['converted'].mean():.1%}")
    print(f"Control conversion   : {control['converted'].mean():.1%}")
    lift = treatment['converted'].mean() - control['converted'].mean()
    print(f"Raw lift             : {lift:+.1%}")
    print("""
Evening 2 will:
  - Run chi-squared significance test on this lift
  - Calculate confidence intervals
  - Run a power analysis (was our sample big enough?)
  - Check if the lift holds across user segments (age, job, housing)
""")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data("bank-full.csv")
    df, treatment, control = assign_groups(df)
    check_balance(treatment, control)
    plot_group_overview(df)
    evening1_summary(df, treatment, control)
