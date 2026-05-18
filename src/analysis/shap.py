import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Encode categoricals and build feature matrix.
    duration excluded -- post-treatment variable.
    treatment_flag included so SHAP can rank its importance relative to
    demographic and behavioral features.
    """
    df = df.copy()
    df["treatment_flag"] = (df["group"] == "treatment").astype(int)
    df["age_band"]       = pd.cut(df["age"], bins=[0, 30, 40, 50, 60, 100],
                                   labels=[0, 1, 2, 3, 4]).astype(int)

    cat_cols = ["job", "marital", "education", "contact", "housing", "loan", "poutcome"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    feature_cols = [
        "age", "age_band", "balance", "campaign", "previous",
        "treatment_flag", "job_enc", "marital_enc", "education_enc",
        "housing_enc", "loan_enc", "poutcome_enc"
    ]

    logger.info(f"Feature matrix shape: {df[feature_cols].shape}")
    return df[feature_cols], df["converted"]


def train_classifier(X: pd.DataFrame, y: pd.Series) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    logger.info("Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    logger.info(f"AUC-ROC={auc:.4f} | Precision={report['1']['precision']:.3f} | Recall={report['1']['recall']:.3f}")

    return model, X_train, X_test


def run_shap(model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    logger.info("Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    logger.info("SHAP values computed")
    return explainer, shap_values


def log_treatment_importance(shap_values: np.ndarray, X_test: pd.DataFrame) -> None:
    mean_shap = np.abs(shap_values).mean(axis=0)
    ranked    = sorted(zip(X_test.columns, mean_shap), key=lambda x: x[1], reverse=True)
    logger.info("Feature importance ranking (by mean |SHAP|):")
    for rank, (feat, val) in enumerate(ranked, 1):
        marker = " <-- treatment group" if feat == "treatment_flag" else ""
        logger.info(f"  {rank:>2}. {feat:<25} {val:.4f}{marker}")


def plot_shap(shap_values: np.ndarray, X_test: pd.DataFrame,
              output_path: str = "outputs/shap_results.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("SHAP Feature Importance", fontsize=13, fontweight="bold")

    mean_shap   = np.abs(shap_values).mean(axis=0)
    feature_imp = pd.Series(mean_shap, index=X_test.columns).sort_values(ascending=True)

    axes[0].barh(feature_imp.index, feature_imp.values, color="#2E4057")
    axes[0].set_title("Global Feature Importance\n(mean |SHAP value|)")
    axes[0].set_xlabel("Mean |SHAP value|")
    if "treatment_flag" in feature_imp.index:
        t_rank = list(feature_imp.index).index("treatment_flag")
        axes[0].get_children()[t_rank].set_color("#E63946")

    plt.sca(axes[1])
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
    axes[1].set_title("SHAP Value Distribution\n(red=high feature value, blue=low)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.show()
