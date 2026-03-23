"""
Feature Adoption Experiment Analysis
Step 4: Model and Data Drift Detection with Evidently

Production ML models degrade over time as the real world changes.
Drift detection is how you catch this before it causes silent failures.

We simulate temporal drift by:
    - Splitting our dataset into an "early" reference period and a
      "recent" current period
    - Training a model on the reference period
    - Checking whether feature distributions and model predictions
      have shifted in the current period

This maps directly to what Monzo's ML platform team does to monitor
production models across fraud, credit, and personalisation systems.
"""

import sys
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesSummaryMetric
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from src.pipeline import fetch_raw, process_stream

# --- LOGGING ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
REFERENCE_SPLIT = 0.6  # first 60% = reference, last 40% = current


# --- 1. PREPARE FEATURES -----------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categoricals and add treatment flag.
    duration is excluded -- it is a post-treatment variable and would leak
    outcome information into the model.
    """
    df = df.copy()
    df["treatment_flag"] = (df["group"] == "treatment").astype(int)

    cat_cols = ["job", "marital", "education", "contact", "housing", "loan", "poutcome"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    return df


# --- 2. SIMULATE TEMPORAL SPLIT ----------------------------------------------

def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a reference period (early data) and current period (recent data).
    In production this would be an actual time-based split on event timestamps.
    Here we use row order as a proxy for time.
    """
    split_idx = int(len(df) * REFERENCE_SPLIT)
    reference = df.iloc[:split_idx].copy()
    current   = df.iloc[split_idx:].copy()

    logger.info(f"Reference period: {len(reference):,} rows ({REFERENCE_SPLIT:.0%} of data)")
    logger.info(f"Current period:   {len(current):,} rows ({1 - REFERENCE_SPLIT:.0%} of data)")
    return reference, current


# --- 3. TRAIN ON REFERENCE ---------------------------------------------------

def train_on_reference(reference: pd.DataFrame) -> tuple:
    feature_cols = [
        "age", "balance", "campaign", "previous",
        "treatment_flag", "job_enc", "marital_enc", "education_enc",
        "housing_enc", "loan_enc", "poutcome_enc"
    ]

    X_ref = reference[feature_cols]
    y_ref = reference["converted"]

    logger.info("Training model on reference period...")
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4,
        learning_rate=0.1, random_state=RANDOM_STATE
    )
    model.fit(X_ref, y_ref)
    logger.info("Model trained on reference period")
    return model, feature_cols


# --- 4. GENERATE PREDICTIONS -------------------------------------------------

def add_predictions(df: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    df["prediction"]       = model.predict(df[feature_cols])
    df["prediction_proba"] = model.predict_proba(df[feature_cols])[:, 1]
    return df


# --- 5. EVIDENTLY DRIFT REPORT -----------------------------------------------

def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, feature_cols: list) -> None:
    """
    Generate an Evidently drift report comparing feature distributions
    between reference and current periods.

    In production this report would run automatically on a schedule and
    alert the DS team when drift exceeds a threshold.
    """
    logger.info("Generating Evidently drift report...")

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
        DatasetMissingValuesSummaryMetric(),
    ])

    report.run(
        reference_data=reference[feature_cols + ["converted"]],
        current_data=current[feature_cols + ["converted"]]
    )

    report.save_html("drift_report.html")
    logger.info("Saved: drift_report.html -- open in browser to explore")

    drift_result = report.as_dict()
    n_drifted = drift_result["metrics"][1]["result"]["number_of_drifted_columns"]
    n_total   = drift_result["metrics"][1]["result"]["number_of_columns"]
    logger.info(f"Drift summary: {n_drifted}/{n_total} features show significant drift")

    if n_drifted > 0:
        logger.warning(f"{n_drifted} features drifted. Model retraining may be needed.")
    else:
        logger.info("No significant drift detected. Model remains valid.")


# --- 6. PREDICTION DRIFT PLOT ------------------------------------------------

def plot_prediction_drift(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Prediction Drift Detection", fontsize=13, fontweight="bold")

    # Predicted probability distributions
    axes[0].hist(reference["prediction_proba"], bins=30, alpha=0.6,
                 label="Reference", color="#2E4057")
    axes[0].hist(current["prediction_proba"], bins=30, alpha=0.6,
                 label="Current", color="#A8DADC")
    axes[0].set_title("Predicted Probability Distribution")
    axes[0].set_xlabel("Predicted conversion probability")
    axes[0].legend()

    # Actual vs predicted conversion rates across periods
    df_all       = pd.concat([reference.assign(period="Reference"),
                               current.assign(period="Current")])
    period_rates = df_all.groupby("period")["converted"].mean()
    period_pred  = df_all.groupby("period")["prediction_proba"].mean()

    x      = [0, 1]
    labels = ["Reference", "Current"]
    axes[1].plot(x, [period_rates["Reference"], period_rates["Current"]],
                 "o-", color="#2E4057", label="Actual conversion rate")
    axes[1].plot(x, [period_pred["Reference"], period_pred["Current"]],
                 "s--", color="#E63946", label="Predicted conversion rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Actual vs Predicted Rate: Reference vs Current")
    axes[1].set_ylabel("Conversion rate")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("drift_results.png", dpi=150, bbox_inches="tight")
    logger.info("Saved: drift_results.png")
    plt.show()


# --- 7. HANDOFF --------------------------------------------------------------

def log_handoff() -> None:
    logger.info("Drift detection complete")
    logger.info("Next: experiment_summary.py -- README and clean summary visualization")
    logger.info("That is the artifact a hiring manager will actually see on GitHub")


# --- MAIN --------------------------------------------------------------------

if __name__ == "__main__":
    raw                 = fetch_raw()
    df                  = process_stream(raw)
    df                  = prepare_features(df)
    reference, current  = temporal_split(df)
    model, feature_cols = train_on_reference(reference)
    reference           = add_predictions(reference, model, feature_cols)
    current             = add_predictions(current, model, feature_cols)
    run_drift_report(reference, current, feature_cols)
    plot_prediction_drift(reference, current)
    log_handoff()
