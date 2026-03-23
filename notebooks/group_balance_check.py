"""
Feature Adoption Experiment Analysis
Step 1: Group Assignment and Balance Check

Loads the dataset, assigns treatment/control groups, then checks whether
the groups are comparable on key covariates. Imbalanced variables are flagged
for covariate control in the significance analysis step.
"""

import sys
import logging
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from src.pipeline import fetch_raw, process_stream

# --- LOGGING ------------------------------------------------------------------
# basicConfig sets the format for all loggers in this process.
# In production you would replace StreamHandler with a structured handler
# (e.g. JSON to stdout for ingestion by Datadog or CloudWatch).

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- 1. SUMMARISE GROUPS ------------------------------------------------------

def summarise_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    treatment = df[df["group"] == "treatment"]
    control   = df[df["group"] == "control"]
    lift      = treatment["converted"].mean() - control["converted"].mean()

    logger.info(f"Treatment | n={len(treatment):,} | conversion={treatment['converted'].mean():.1%}")
    logger.info(f"Control   | n={len(control):,} | conversion={control['converted'].mean():.1%}")
    logger.info(f"Raw lift  | {lift:+.1%}")

    return treatment, control


# --- 2. BALANCE CHECK ---------------------------------------------------------

def balance_check(df: pd.DataFrame) -> None:
    """
    Check whether treatment and control groups are comparable on key covariates.
    Logs a WARNING for any variable with significant imbalance (p < 0.05) --
    these need to be controlled for in the regression models in significance_analysis.py.
    """
    logger.info("Running balance check")

    numeric_vars     = ["age", "balance", "duration", "campaign", "previous"]
    categorical_vars = ["job", "marital", "education", "housing"]

    treatment = df[df["group"] == "treatment"]
    control   = df[df["group"] == "control"]

    for var in numeric_vars:
        t_mean = treatment[var].mean()
        c_mean = control[var].mean()
        _, p   = stats.ttest_ind(treatment[var], control[var])
        msg    = f"Balance | {var:<20} | t_mean={t_mean:.2f} c_mean={c_mean:.2f} p={p:.4f}"
        logger.warning(msg + " | IMBALANCED") if p < 0.05 else logger.info(msg)

    for var in categorical_vars:
        all_cats = sorted(set(df[var].dropna().unique()))
        t_counts = [treatment[var].value_counts().get(c, 0) for c in all_cats]
        c_counts = [control[var].value_counts().get(c, 0) for c in all_cats]
        _, p, _, _ = stats.chi2_contingency([t_counts, c_counts])
        msg = f"Balance | {var:<20} | (categorical) p={p:.4f}"
        logger.warning(msg + " | IMBALANCED") if p < 0.05 else logger.info(msg)

    logger.info("Balance check complete -- IMBALANCED variables must be controlled for in significance_analysis.py")


# --- 3. VISUALIZE -------------------------------------------------------------

def plot_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Experiment Group Overview", fontsize=13, fontweight="bold", y=1.01)

    # Group sizes
    counts = df["group"].value_counts()
    axes[0].bar(counts.index, counts.values, color=["#2E4057", "#A8DADC"])
    axes[0].set_title("Group Size")
    axes[0].set_ylabel("Users")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 80, f"{v:,}", ha="center", fontsize=9)

    # Conversion rate
    conv = df.groupby("group")["converted"].mean()
    bars = axes[1].bar(conv.index, conv.values, color=["#2E4057", "#A8DADC"])
    axes[1].set_title("Conversion Rate by Group")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for bar, val in zip(bars, conv.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                     f"{val:.1%}", ha="center", fontsize=9)

    # Age distribution
    for grp, color in zip(["treatment", "control"], ["#2E4057", "#A8DADC"]):
        axes[2].hist(df[df["group"] == grp]["age"], bins=20,
                     alpha=0.6, label=grp, color=color)
    axes[2].set_title("Age Distribution by Group")
    axes[2].set_xlabel("Age")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("group_balance_overview.png", dpi=150, bbox_inches="tight")
    logger.info("Saved: group_balance_overview.png")
    plt.show()


# --- 4. HANDOFF ---------------------------------------------------------------

def log_handoff(treatment: pd.DataFrame, control: pd.DataFrame) -> None:
    lift = treatment["converted"].mean() - control["converted"].mean()
    logger.info("Group balance check complete")
    logger.info(f"Treatment conversion : {treatment['converted'].mean():.1%}")
    logger.info(f"Control conversion   : {control['converted'].mean():.1%}")
    logger.info(f"Raw lift             : {lift:+.1%}")
    logger.info("Next: significance_analysis.py -- significance testing, power analysis, segmentation")


# --- MAIN ---------------------------------------------------------------------

if __name__ == "__main__":
    raw                = fetch_raw()
    df                 = process_stream(raw)
    treatment, control = summarise_groups(df)
    balance_check(df)
    plot_overview(df)
    log_handoff(treatment, control)
