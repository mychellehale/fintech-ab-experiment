"""
Feature Adoption Experiment Analysis
Step 3: SHAP-Based Feature Importance and Individual-Level Explainability

We train a lightweight classifier to predict conversion, then use SHAP to:
    1. Identify which features drive conversion probability globally
    2. Show how individual users' features push their prediction up or down
    3. Identify whether treatment/control group membership itself is a driver

This connects directly to production ML practice at fintech companies like
Monzo, where explainability is required for credit and fraud models under
EU AI Act Articles 6, 10, 14 and 86.
"""

import sys
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import shap
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


# --- 1. FEATURE ENGINEERING --------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Encode categorical variables and prepare feature matrix for the classifier.
    We include treatment group as a feature so SHAP can show its relative
    importance compared to demographic and behavioral features.

    Note: duration is intentionally excluded -- it is a post-treatment variable
    (you only know call length after the call happened) and would leak information
    about the outcome into the feature matrix.
    """
    df = df.copy()
    df["treatment_flag"] = (df["group"] == "treatment").astype(int)
    df["age_band"]       = pd.cut(df["age"], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4]).astype(int)

    cat_cols = ["job", "marital", "education", "contact", "housing", "loan", "poutcome"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    feature_cols = [
        "age", "age_band", "balance", "campaign", "previous",
        "treatment_flag", "job_enc", "marital_enc", "education_enc",
        "housing_enc", "loan_enc", "poutcome_enc"
    ]

    X = df[feature_cols]
    y = df["converted"]

    logger.info(f"Feature matrix shape: {X.shape}")
    return X, y


# --- 2. TRAIN CLASSIFIER ------------------------------------------------------

def train_classifier(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train a gradient boosting classifier. We use a lightweight model here
    since the goal is explainability, not maximum predictive performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    logger.info("Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"Model AUC-ROC: {auc:.4f}")

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(f"Precision (converted): {report['1']['precision']:.3f}")
    logger.info(f"Recall (converted): {report['1']['recall']:.3f}")

    return model, X_train, X_test, y_train, y_test


# --- 3. SHAP ANALYSIS --------------------------------------------------------

def run_shap(model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Compute SHAP values for the test set.
    SHAP (SHapley Additive exPlanations) assigns each feature a contribution
    value for each prediction, grounded in cooperative game theory.

    At Monzo this matters because EU AI Act Article 86 requires meaningful
    explanations for automated decisions affecting customers.
    """
    logger.info("Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    logger.info("SHAP values computed")
    return explainer, shap_values


# --- 4. VISUALISE ------------------------------------------------------------

def plot_shap(shap_values: np.ndarray, X_test: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("SHAP Feature Importance", fontsize=13, fontweight="bold")

    # Global feature importance -- mean absolute SHAP value per feature
    mean_shap   = np.abs(shap_values).mean(axis=0)
    feature_imp = pd.Series(mean_shap, index=X_test.columns).sort_values(ascending=True)

    axes[0].barh(feature_imp.index, feature_imp.values, color="#2E4057")
    axes[0].set_title("Global Feature Importance\n(mean |SHAP value|)")
    axes[0].set_xlabel("Mean |SHAP value|")

    # Treatment flag importance highlighted
    if "treatment_flag" in feature_imp.index:
        t_rank = list(feature_imp.index).index("treatment_flag")
        axes[0].get_children()[t_rank].set_color("#E63946")

    # SHAP summary: distribution of SHAP values per feature
    plt.sca(axes[1])
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
    axes[1].set_title("SHAP Value Distribution\n(red=high feature value, blue=low)")

    plt.tight_layout()
    plt.savefig("shap_results.png", dpi=150, bbox_inches="tight")
    logger.info("Saved: shap_results.png")
    plt.show()


def log_treatment_importance(shap_values: np.ndarray, X_test: pd.DataFrame) -> None:
    """
    Report specifically on how much treatment group membership contributes
    to conversion probability, relative to other features.
    """
    feature_names = list(X_test.columns)
    mean_shap     = np.abs(shap_values).mean(axis=0)
    ranked        = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)

    logger.info("Feature importance ranking (by mean |SHAP|):")
    for rank, (feat, val) in enumerate(ranked, 1):
        marker = " <-- treatment group" if feat == "treatment_flag" else ""
        logger.info(f"  {rank:>2}. {feat:<25} {val:.4f}{marker}")


# --- 5. HANDOFF --------------------------------------------------------------

def log_handoff() -> None:
    logger.info("SHAP analysis complete")
    logger.info("Next: drift_detection.py -- Evidently drift detection")
    logger.info("We will simulate temporal drift by splitting data into early/late periods")
    logger.info("and check whether feature distributions and model predictions shift over time")


# --- MAIN --------------------------------------------------------------------

if __name__ == "__main__":
    raw                          = fetch_raw()
    df                           = process_stream(raw)
    X, y                         = prepare_features(df)
    model, X_train, X_test, _, _ = train_classifier(X, y)
    explainer, shap_values       = run_shap(model, X_train, X_test)
    plot_shap(shap_values, X_test)
    log_treatment_importance(shap_values, X_test)
    log_handoff()
