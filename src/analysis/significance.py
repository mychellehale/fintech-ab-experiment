import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

ALPHA = 0.05


def chi_squared_test(df: pd.DataFrame) -> dict:
    treatment  = df[df["group"] == "treatment"]
    control    = df[df["group"] == "control"]

    t_conv     = treatment["converted"].sum()
    t_no_conv  = len(treatment) - t_conv
    c_conv     = control["converted"].sum()
    c_no_conv  = len(control) - c_conv

    chi2, p, dof, _ = chi2_contingency([[t_conv, t_no_conv], [c_conv, c_no_conv]])

    t_rate    = treatment["converted"].mean()
    c_rate    = control["converted"].mean()
    lift      = t_rate - c_rate
    cohens_h  = 2 * (np.arcsin(np.sqrt(t_rate)) - np.arcsin(np.sqrt(c_rate)))

    results = {
        "t_rate": t_rate, "c_rate": c_rate, "lift": lift,
        "chi2": chi2, "p": p, "dof": dof,
        "cohens_h": cohens_h, "significant": p < ALPHA
    }

    logger.info(f"Chi-squared | chi2={chi2:.2f} p={p:.6f} dof={dof}")
    logger.info(f"Treatment={t_rate:.1%} | Control={c_rate:.1%} | Lift={lift:+.1%}")
    logger.info(f"Cohen's h={cohens_h:.3f}")
    if results["significant"]:
        logger.info(f"Result: SIGNIFICANT at alpha={ALPHA}")
    else:
        logger.warning(f"Result: NOT significant at alpha={ALPHA}")

    return results


def confidence_intervals(df: pd.DataFrame) -> None:
    for group in ["treatment", "control"]:
        subset = df[df["group"] == group]
        n      = len(subset)
        p_hat  = subset["converted"].mean()
        se     = np.sqrt(p_hat * (1 - p_hat) / n)
        logger.info(f"95% CI | {group:<12} rate={p_hat:.1%} [{p_hat - 1.96*se:.1%}, {p_hat + 1.96*se:.1%}] n={n:,}")


def power_analysis(df: pd.DataFrame) -> None:
    treatment = df[df["group"] == "treatment"]
    control   = df[df["group"] == "control"]

    t_rate = treatment["converted"].mean()
    c_rate = control["converted"].mean()
    n_t    = len(treatment)
    n_c    = len(control)

    h         = abs(2 * (np.arcsin(np.sqrt(t_rate)) - np.arcsin(np.sqrt(c_rate))))
    z_alpha   = norm.ppf(1 - ALPHA / 2)
    z_power   = h * np.sqrt((n_t * n_c) / (n_t + n_c)) - z_alpha
    observed_power = norm.cdf(z_power)

    z_beta     = norm.ppf(0.80)
    n_required = int(np.ceil(((z_alpha + z_beta) / h) ** 2 * 2))

    logger.info(f"Power | effect_size(h)={h:.3f}")
    logger.info(f"Power | observed_power={observed_power:.1%}")
    logger.info(f"Power | n_required_for_80pct={n_required:,} per group")
    if observed_power >= 0.80:
        logger.info("Power: ADEQUATE")
    else:
        logger.warning("Power: UNDERPOWERED")


def segmented_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_band"] = pd.cut(
        df["age"], bins=[0, 30, 40, 50, 60, 100],
        labels=["Under 30", "30-39", "40-49", "50-59", "60+"]
    )

    segment_results = []
    for segment_col in ["age_band", "job", "housing"]:
        for segment_val in df[segment_col].dropna().unique():
            subset    = df[df[segment_col] == segment_val]
            treatment = subset[subset["group"] == "treatment"]
            control   = subset[subset["group"] == "control"]
            if len(treatment) < 30 or len(control) < 30:
                continue

            t_rate = treatment["converted"].mean()
            c_rate = control["converted"].mean()
            t_conv, t_no = treatment["converted"].sum(), len(treatment) - treatment["converted"].sum()
            c_conv, c_no = control["converted"].sum(), len(control) - control["converted"].sum()

            try:
                _, p, _, _ = chi2_contingency([[t_conv, t_no], [c_conv, c_no]])
            except ValueError:
                p = 1.0

            segment_results.append({
                "segment": segment_col, "value": segment_val,
                "n_treatment": len(treatment), "n_control": len(control),
                "t_rate": t_rate, "c_rate": c_rate, "lift": t_rate - c_rate,
                "p_value": p
            })

    results_df = pd.DataFrame(segment_results)

    # Benjamini-Hochberg correction across all segment tests
    _, corrected, _, _ = multipletests(results_df["p_value"], alpha=ALPHA, method="fdr_bh")
    results_df["p_corrected"] = corrected
    results_df["significant"]  = results_df["p_corrected"] < ALPHA

    n_raw       = (results_df["p_value"] < ALPHA).sum()
    n_corrected = results_df["significant"].sum()
    logger.info(f"Segmentation: {n_raw} significant before BH correction, {n_corrected} after")

    for _, row in results_df.iterrows():
        sig = "SIGNIFICANT" if row["significant"] else "not significant"
        logger.info(
            f"{row['segment']:<12} {str(row['value']):<20} "
            f"lift={row['lift']:+.1%} p={row['p_value']:.4f} p_bh={row['p_corrected']:.4f} {sig}"
        )
    return results_df


def logistic_regression(df: pd.DataFrame) -> None:
    """
    Controls for imbalanced covariates identified in the balance check.
    duration excluded -- post-treatment variable.
    """
    df = df.copy()
    df["treatment_flag"] = (df["group"] == "treatment").astype(int)
    df["housing_bin"]    = (df["housing"] == "yes").astype(int)

    formula = "converted ~ treatment_flag + balance + campaign + housing_bin + C(job) + C(marital) + C(education)"  # noqa: E501
    try:
        model  = smf.logit(formula, data=df).fit(disp=False)
        coef   = model.params["treatment_flag"]
        pval   = model.pvalues["treatment_flag"]
        or_val = np.exp(coef)

        logger.info(f"Logistic regression | odds_ratio={or_val:.3f} p={pval:.6f}")
        if pval < ALPHA:
            logger.info(f"Treatment effect REMAINS significant after covariate control. OR={or_val:.2f}x")
        else:
            logger.warning("Treatment effect NOT significant after covariate control.")
    except Exception as e:
        logger.error(f"Logistic regression failed: {e}")


def plot_results(df: pd.DataFrame, segment_results: pd.DataFrame,
                 output_path: str = "outputs/significance_results.png") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Significance Testing and Segmentation", fontsize=13, fontweight="bold")

    groups = ["treatment", "control"]
    rates  = [df[df["group"] == g]["converted"].mean() for g in groups]
    ns     = [len(df[df["group"] == g]) for g in groups]
    errors = [1.96 * np.sqrt(r * (1 - r) / n) for r, n in zip(rates, ns)]

    axes[0].bar(groups, rates, color=["#2E4057", "#A8DADC"], yerr=errors, capsize=6)
    axes[0].set_title("Conversion Rate with 95% CI")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for i, (r, e) in enumerate(zip(rates, errors)):
        axes[0].text(i, r + e + 0.005, f"{r:.1%}", ha="center", fontsize=9)

    age_segs = segment_results[segment_results["segment"] == "age_band"].copy()
    age_segs = age_segs.sort_values("value")
    colors   = ["#2E4057" if s else "#A8DADC" for s in age_segs["significant"]]
    axes[1].bar(age_segs["value"].astype(str), age_segs["lift"], color=colors)
    axes[1].set_title("Lift by Age Band\n(dark = significant)")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=30)

    house_segs = segment_results[segment_results["segment"] == "housing"].copy()
    colors     = ["#2E4057" if s else "#A8DADC" for s in house_segs["significant"]]
    axes[2].bar(house_segs["value"].astype(str), house_segs["lift"], color=colors)
    axes[2].set_title("Lift by Housing Status\n(dark = significant)")
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[2].axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.show()
