"""
Feature Adoption Experiment Analysis
Evening 1: Data Loading, Group Definition, and Balance Checking

Uses a generator pipeline to stream and process data in chunks rather than
loading everything into memory at once -- a pattern common in production
fintech data systems where datasets are large and memory is constrained.

Dataset: UCI Bank Marketing (fetched via ucimlrepo)
Experiment framing:
    treatment = users contacted via campaign (contact 'previous' > 0)
    control   = users not contacted (contact == 'unknown')
    metric    = feature adoption / subscription (y == 'yes')
"""

import logging
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm
from typing import Generator

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

CHUNK_SIZE = 1000


# --- 1. FETCH AND STREAM ------------------------------------------------------

def fetch_raw() -> pd.DataFrame:
    """Fetch the dataset from UCI and return as a single DataFrame."""
    logger.info("Fetching Bank Marketing dataset from UCI...")
    repo = fetch_ucirepo(id=222)
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    logger.info(f"Fetched {len(df):,} rows, {df.shape[1]} columns")
    return df


def stream_chunks(df: pd.DataFrame, chunk_size: int = CHUNK_SIZE) -> Generator[pd.DataFrame, None, None]:
    """
    Yield DataFrame chunks of chunk_size rows.
    Simulates streaming from a database or Kafka topic -- in production you
    would replace this with a real source (e.g. pd.read_csv with chunksize,
    or a Kafka consumer).
    """
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start: start + chunk_size].copy()


# --- 2. TRANSFORM -------------------------------------------------------------

def transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering and group assignment to a single chunk.
    Keeping transforms here makes them easy to unit test and reuse in Evening 3.
    """
    chunk = chunk.copy()
    chunk["group"]     = np.where(chunk["previous"] > 0, "treatment", "control")
    chunk["converted"] = np.where(chunk["y"] == "yes", 1, 0)
    return chunk


def process_stream(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drive the stream: pull chunks, transform each one, concatenate.
    tqdm provides a human-readable progress bar in the terminal.
    Logging captures structured milestones for monitoring systems.
    In production the final concat would be replaced with writes to a sink.
    """
    logger.info(f"Starting stream processing | chunk_size={CHUNK_SIZE}")
    processed_chunks = []
    total = 0
    n_chunks = (len(df) // CHUNK_SIZE) + 1

    for chunk in tqdm(stream_chunks(df), total=n_chunks, desc="Processing chunks", unit="chunk"):
        transformed = transform_chunk(chunk)
        processed_chunks.append(transformed)
        total += len(chunk)

    logger.info(f"Stream complete | total_rows={total:,}")
    return pd.concat(processed_chunks, ignore_index=True)


# --- 3. SUMMARISE GROUPS ------------------------------------------------------

def summarise_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    treatment = df[df["group"] == "treatment"]
    control   = df[df["group"] == "control"]
    lift      = treatment["converted"].mean() - control["converted"].mean()

    logger.info(f"Treatment | n={len(treatment):,} | conversion={treatment['converted'].mean():.1%}")
    logger.info(f"Control   | n={len(control):,} | conversion={control['converted'].mean():.1%}")
    logger.info(f"Raw lift  | {lift:+.1%}")

    return treatment, control


# --- 4. BALANCE CHECK ---------------------------------------------------------

def balance_check_stream(df: pd.DataFrame) -> None:
    """
    Check whether treatment and control groups are comparable on key covariates.
    Logs a WARNING for any variable with significant imbalance (p < 0.05) --
    these need to be controlled for in the regression models in Evening 2.
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

    logger.info("Balance check complete -- IMBALANCED variables must be controlled for in Evening 2")


# --- 5. VISUALIZE -------------------------------------------------------------

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
    plt.savefig("evening1_overview.png", dpi=150, bbox_inches="tight")
    logger.info("Saved: evening1_overview.png")
    plt.show()


# --- 6. EVENING 2 HANDOFF -----------------------------------------------------

def log_handoff(treatment: pd.DataFrame, control: pd.DataFrame) -> None:
    lift = treatment["converted"].mean() - control["converted"].mean()
    logger.info("Evening 1 complete")
    logger.info(f"Treatment conversion : {treatment['converted'].mean():.1%}")
    logger.info(f"Control conversion   : {control['converted'].mean():.1%}")
    logger.info(f"Raw lift             : {lift:+.1%}")
    logger.info("Next: Evening 2 -- significance testing, power analysis, segmentation")


# --- MAIN ---------------------------------------------------------------------

if __name__ == "__main__":
    raw                = fetch_raw()
    df                 = process_stream(raw)
    treatment, control = summarise_groups(df)
    balance_check_stream(df)
    plot_overview(df)
    log_handoff(treatment, control)
