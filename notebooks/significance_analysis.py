"""
Feature Adoption Experiment Analysis
Step 2: Significance Testing, Power Analysis, and Segmentation

Building on the balance check, we now answer three questions:
    1. Is the observed lift statistically significant?
    2. Was our sample large enough to detect this effect? (power analysis)
    3. Does the lift hold across user segments, or is it driven by one group?

The imbalanced covariates from group_balance_check.py (balance, campaign, job,
marital, education, housing) are controlled for in a logistic regression model
so we can estimate the "true" lift after accounting for group differences.
"""

import sys
import logging
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, norm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from src.pipeline import fetch_raw, process_stream

# --- LOGGING ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALPHA = 0.05  # significance threshold


# --- 1. CHI-SQUARED SIGNIFICANCE TEST ----------------------------------------

def chi_squared_test(df: pd.DataFrame) -> dict:
    """
    Test whether the conversion rate difference between treatment and control
    is statistically significant using a chi-squared test of independence.

    Returns a dict of results for use in downstream reporting.
    """
    treatment  = df[df["group"] == "treatment"]
    control    = df[df["group"] == "control"]

    t_conv     = treatment["converted"].sum()
    t_no_conv  = len(treatment) - t_conv
    c_conv     = control["converted"].sum()
    c_no_conv  = len(control) - c_conv

    contingency_table = [[t_conv, t_no_conv], [c_conv, c_no_conv]]
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    t_rate = treatment["converted"].mean()
    c_rate = control["converted"].mean()
    lift   = t_rate - c_rate

    # Cohen's h -- effect size for proportions
    cohens_h = 2 * (np.arcsin(np.sqrt(t_rate)) - np.arcsin(np.sqrt(c_rate)))

    results = {
        "t_rate": t_rate,
        "c_rate": c_rate,
        "lift":   lift,
        "chi2":   chi2,
        "p":      p,
        "dof":    dof,
        "cohens_h": cohens_h,
        "significant": p < ALPHA
    }

    logger.info(f"Chi-squared test | chi2={chi2:.2f} p={p:.6f} dof={dof}")
    logger.info(f"Treatment rate={t_rate:.1%} | Control rate={c_rate:.1%} | Lift={lift:+.1%}")
    logger.info(f"Cohen's h (effect size)={cohens_h:.3f}")

    if results["significant"]:
        logger.info(f"Result: SIGNIFICANT at alpha={ALPHA}. Lift is real.")
    else:
        logger.warning(f"Result: NOT significant at alpha={ALPHA}. Lift may be noise.")

    return results


# --- 2. CONFIDENCE INTERVALS -------------------------------------------------

def confidence_intervals(df: pd.DataFrame) -> None:
    """
    Calculate 95% confidence intervals for conversion rates in each group.
    Uses the normal approximation (Wilson would be more robust at extreme
    proportions, but is fine here given our sample sizes).
    """
    for group in ["treatment", "control"]:
        subset = df[df["group"] == group]
        n      = len(subset)
        p_hat  = subset["converted"].mean()
        se     = np.sqrt(p_hat * (1 - p_hat) / n)
        ci_low = p_hat - 1.96 * se
        ci_hi  = p_hat + 1.96 * se
        logger.info(f"95% CI | {group:<12} rate={p_hat:.1%} [{ci_low:.1%}, {ci_hi:.1%}] n={n:,}")


# --- 3. POWER ANALYSIS -------------------------------------------------------

def power_analysis(df: pd.DataFrame) -> None:
    """
    Retrospective power analysis: given our observed effect size and sample
    size, what was the probability of detecting this effect if it is real?

    Also answers: how large a sample would we have needed for 80% power?
    """
    treatment = df[df["group"] == "treatment"]
    control   = df[df["group"] == "control"]

    t_rate  = treatment["converted"].mean()
    c_rate  = control["converted"].mean()
    n_t     = len(treatment)
    n_c     = len(control)

    # Cohen's h
    h = abs(2 * (np.arcsin(np.sqrt(t_rate)) - np.arcsin(np.sqrt(c_rate))))

    # Observed power
    z_alpha = norm.ppf(1 - ALPHA / 2)
    z_power = h * np.sqrt((n_t * n_c) / (n_t + n_c)) - z_alpha
    observed_power = norm.cdf(z_power)

    # Required sample size for 80% power
    z_beta       = norm.ppf(0.80)
    n_required   = ((z_alpha + z_beta) / h) ** 2 * 2
    n_required   = int(np.ceil(n_required))

    logger.info(f"Power analysis | effect_size(h)={h:.3f}")
    logger.info(f"Power analysis | observed_power={observed_power:.1%} (want >80%)")
    logger.info(f"Power analysis | sample_size_for_80pct_power={n_required:,} per group")
    logger.info(f"Power analysis | our_sample treatment={n_t:,} control={n_c:,}")

    if observed_power >= 0.80:
        logger.info("Power: ADEQUATE. Sample was large enough to detect this effect.")
    else:
        logger.warning("Power: UNDERPOWERED. Results should be interpreted cautiously.")


# --- 4. SEGMENTATION ---------------------------------------------------------

def segmented_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether the lift holds across key user segments.
    Segments: age band, job type, housing status.

    This is where real insights live -- a global lift can mask
    a strong effect in one segment and none in others.
    """
    logger.info("Running segmented analysis")

    df = df.copy()
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
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
            lift   = t_rate - c_rate

            t_conv    = treatment["converted"].sum()
            t_no_conv = len(treatment) - t_conv
            c_conv    = control["converted"].sum()
            c_no_conv = len(control) - c_conv

            try:
                _, p, _, _ = chi2_contingency([[t_conv, t_no_conv], [c_conv, c_no_conv]])
            except ValueError:
                p = 1.0

            segment_results.append({
                "segment":    segment_col,
                "value":      segment_val,
                "n_treatment": len(treatment),
                "n_control":  len(control),
                "t_rate":     t_rate,
                "c_rate":     c_rate,
                "lift":       lift,
                "p_value":    p,
                "significant": p < ALPHA
            })

    results_df = pd.DataFrame(segment_results)

    print("\nSegmented Analysis Results:")
    print("=" * 80)
    for _, row in results_df.iterrows():
        sig = "SIGNIFICANT" if row["significant"] else "not significant"
        logger.info(
            f"{row['segment']:<12} {str(row['value']):<20} "
            f"lift={row['lift']:+.1%} p={row['p_value']:.4f} {sig}"
        )

    return results_df


# --- 5. LOGISTIC REGRESSION WITH CONTROLS ------------------------------------

def logistic_regression(df: pd.DataFrame) -> None:
    """
    Logistic regression controlling for the imbalanced covariates identified
    in group_balance_check.py (balance, campaign, job, marital, education, housing).

    This gives us the "true" treatment effect after controlling for the fact
    that treatment users were systematically different from control users.
    The coefficient on 'treatment' is what we care about.
    """
    logger.info("Running logistic regression with covariate controls")

    df = df.copy()
    df["treatment_flag"] = (df["group"] == "treatment").astype(int)

    # Encode categoricals
    df["housing_bin"] = (df["housing"] == "yes").astype(int)

    formula = "converted ~ treatment_flag + balance + campaign + housing_bin + C(job) + C(marital) + C(education)"

    try:
        model  = smf.logit(formula, data=df).fit(disp=False)
        coef   = model.params["treatment_flag"]
        pval   = model.pvalues["treatment_flag"]
        or_val = np.exp(coef)

        logger.info(f"Logistic regression | treatment coefficient={coef:.4f}")
        logger.info(f"Logistic regression | odds_ratio={or_val:.3f}")
        logger.info(f"Logistic regression | p_value={pval:.6f}")

        if pval < ALPHA:
            logger.info(f"After controlling for covariates, treatment effect REMAINS significant.")
            logger.info(f"Odds ratio {or_val:.2f}x means treated users are {or_val:.2f}x more likely to convert.")
        else:
            logger.warning("After controlling for covariates, treatment effect is NOT significant.")
            logger.warning("The raw lift may be explained by group composition differences.")

    except Exception as e:
        logger.error(f"Logistic regression failed: {e}")


# --- 6. VISUALISE ------------------------------------------------------------

def plot_results(df: pd.DataFrame, segment_results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Significance Testing and Segmentation", fontsize=13, fontweight="bold")

    # Conversion rates with CI
    groups    = ["treatment", "control"]
    rates     = [df[df["group"] == g]["converted"].mean() for g in groups]
    ns        = [len(df[df["group"] == g]) for g in groups]
    errors    = [1.96 * np.sqrt(r * (1 - r) / n) for r, n in zip(rates, ns)]

    axes[0].bar(groups, rates, color=["#2E4057", "#A8DADC"], yerr=errors, capsize=6)
    axes[0].set_title("Conversion Rate with 95% CI")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for i, (r, e) in enumerate(zip(rates, errors)):
        axes[0].text(i, r + e + 0.005, f"{r:.1%}", ha="center", fontsize=9)

    # Lift by age band
    age_segs = segment_results[segment_results["segment"] == "age_band"].copy()
    age_segs = age_segs.sort_values("value")
    colors   = ["#2E4057" if s else "#A8DADC" for s in age_segs["significant"]]
    axes[1].bar(age_segs["value"].astype(str), age_segs["lift"], color=colors)
    axes[1].set_title("Lift by Age Band\n(dark = significant)")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=30)

    # Lift by housing status
    house_segs = segment_results[segment_results["segment"] == "housing"].copy()
    colors     = ["#2E4057" if s else "#A8DADC" for s in house_segs["significant"]]
    axes[2].bar(house_segs["value"].astype(str), house_segs["lift"], color=colors)
    axes[2].set_title("Lift by Housing Status\n(dark = significant)")
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[2].axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig("significance_results.png", dpi=150, bbox_inches="tight")
    logger.info("Saved: significance_results.png")
    plt.show()


# --- 7. HANDOFF --------------------------------------------------------------

def log_handoff(df: pd.DataFrame) -> None:
    logger.info("Significance analysis complete")
    logger.info("Next: SHAP-based segmentation using a trained classifier")
    logger.info("This will show WHICH FEATURES drive conversion probability at the individual level")


# --- MAIN --------------------------------------------------------------------

if __name__ == "__main__":
    raw             = fetch_raw()
    df              = process_stream(raw)
    results         = chi_squared_test(df)
    confidence_intervals(df)
    power_analysis(df)
    segment_results = segmented_analysis(df)
    logistic_regression(df)
    plot_results(df, segment_results)
    log_handoff(df)
