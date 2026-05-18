import logging

from src.pipeline import fetch_or_load, process_stream
from src.analysis.shap import prepare_features, train_classifier, run_shap, plot_shap, log_treatment_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    raw                    = fetch_or_load()
    df                     = process_stream(raw)
    X, y                   = prepare_features(df)
    model, X_train, X_test = train_classifier(X, y)
    explainer, shap_values = run_shap(model, X_train, X_test)
    plot_shap(shap_values, X_test)
    log_treatment_importance(shap_values, X_test)
