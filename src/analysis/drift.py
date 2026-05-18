import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset

from src.pipeline import encode_categoricals

logger = logging.getLogger(__name__)

RANDOM_STATE    = 42
REFERENCE_SPLIT = 0.6


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categoricals and add treatment flag.
    duration excluded -- post-treatment variable.
    """
    df = encode_categoricals(df.copy())
    df["treatment_flag"] = (df["group"] == "treatment").astype(int)
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate reference (early) and current (recent) periods using row order
    as a proxy for time. In production this would be a real timestamp split.
    """
    split_idx = int(len(df) * REFERENCE_SPLIT)
    reference = df.iloc[:split_idx].copy()
    current   = df.iloc[split_idx:].copy()
    logger.info(f"Reference: {len(reference):,} rows | Current: {len(current):,} rows")
    return reference, current


def train_on_reference(reference: pd.DataFrame) -> tuple:
    feature_cols = [
        "age", "balance", "campaign", "previous",
        "treatment_flag", "job_enc", "marital_enc", "education_enc",
        "housing_enc", "loan_enc", "poutcome_enc"
    ]
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE
    )
    model.fit(reference[feature_cols], reference["converted"])
    logger.info("Model trained on reference period")
    return model, feature_cols


def add_predictions(df: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    df["prediction"]       = model.predict(df[feature_cols])
    df["prediction_proba"] = model.predict_proba(df[feature_cols])[:, 1]
    return df


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                     feature_cols: list,
                     output_path: str = "outputs/drift_report.html") -> None:
    logger.info("Generating Evidently drift report...")
    report   = Report([DataDriftPreset()])
    cols     = feature_cols + ["converted"]
    snapshot = report.run(
        reference_data=Dataset.from_pandas(reference[cols]),
        current_data=Dataset.from_pandas(current[cols])
    )
    snapshot.save_html(output_path)
    logger.info(f"Saved: {output_path}")

    # Extract drifted column count from metric results
    n_drifted = 0
    n_total   = len(cols)
    for result in snapshot.metric_results.values():
        if hasattr(result, "count") and hasattr(result.count, "value"):
            n_drifted = int(result.count.value)
            break

    logger.info(f"Drift summary: {n_drifted}/{n_total} features show significant drift")
    if n_drifted > 0:
        logger.warning(f"{n_drifted} features drifted. Model retraining may be needed.")
    else:
        logger.info("No significant drift detected.")


def plot_prediction_drift(reference: pd.DataFrame, current: pd.DataFrame,
                          output_path: str = "outputs/drift_results.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Prediction Drift Detection", fontsize=13, fontweight="bold")

    axes[0].hist(reference["prediction_proba"], bins=30, alpha=0.6, label="Reference", color="#2E4057")
    axes[0].hist(current["prediction_proba"], bins=30, alpha=0.6, label="Current", color="#A8DADC")
    axes[0].set_title("Predicted Probability Distribution")
    axes[0].set_xlabel("Predicted conversion probability")
    axes[0].legend()

    df_all       = pd.concat([reference.assign(period="Reference"), current.assign(period="Current")])
    period_rates = df_all.groupby("period")["converted"].mean()
    period_pred  = df_all.groupby("period")["prediction_proba"].mean()
    x            = [0, 1]
    labels       = ["Reference", "Current"]
    axes[1].plot(x, [period_rates["Reference"], period_rates["Current"]], "o-", color="#2E4057", label="Actual")
    axes[1].plot(x, [period_pred["Reference"], period_pred["Current"]], "s--", color="#E63946", label="Predicted")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Actual vs Predicted: Reference vs Current")
    axes[1].set_ylabel("Conversion rate")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.show()
