import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats

logger = logging.getLogger(__name__)


def summarise_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    treatment = df[df["group"] == "treatment"]
    control   = df[df["group"] == "control"]
    lift      = treatment["converted"].mean() - control["converted"].mean()

    logger.info(f"Treatment | n={len(treatment):,} | conversion={treatment['converted'].mean():.1%}")
    logger.info(f"Control   | n={len(control):,} | conversion={control['converted'].mean():.1%}")
    logger.info(f"Raw lift  | {lift:+.1%}")

    return treatment, control


def balance_check(df: pd.DataFrame) -> None:
    """
    T-tests on numeric features and chi-squared on categoricals.
    Logs WARNING for any variable with p < 0.05 -- these must be controlled
    for in significance.py.
    """
    logger.info("Running balance check")

    # duration is included here to characterise the groups, not as a model feature.
    # It is excluded from all models because it is a post-treatment variable.
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

    logger.info("Balance check complete -- IMBALANCED variables must be controlled for in significance.py")


def plot_overview(df: pd.DataFrame, output_path: str = "outputs/group_balance_overview.png") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Experiment Group Overview", fontsize=13, fontweight="bold", y=1.01)

    counts = df["group"].value_counts()
    axes[0].bar(counts.index, counts.values, color=["#2E4057", "#A8DADC"])
    axes[0].set_title("Group Size")
    axes[0].set_ylabel("Users")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 80, f"{v:,}", ha="center", fontsize=9)

    conv = df.groupby("group")["converted"].mean()
    bars = axes[1].bar(conv.index, conv.values, color=["#2E4057", "#A8DADC"])
    axes[1].set_title("Conversion Rate by Group")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for bar, val in zip(bars, conv.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                     f"{val:.1%}", ha="center", fontsize=9)

    for grp, color in zip(["treatment", "control"], ["#2E4057", "#A8DADC"]):
        axes[2].hist(df[df["group"] == grp]["age"], bins=20,
                     alpha=0.6, label=grp, color=color)
    axes[2].set_title("Age Distribution by Group")
    axes[2].set_xlabel("Age")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.show()
