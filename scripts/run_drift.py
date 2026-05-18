import logging

from src.pipeline import fetch_or_load, process_stream
from src.analysis.drift import (
    prepare_features, temporal_split, train_on_reference,
    add_predictions, run_drift_report, plot_prediction_drift
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    raw                 = fetch_or_load()
    df                  = process_stream(raw)
    df                  = prepare_features(df)
    reference, current  = temporal_split(df)
    model, feature_cols = train_on_reference(reference)
    reference           = add_predictions(reference, model, feature_cols)
    current             = add_predictions(current, model, feature_cols)
    run_drift_report(reference, current, feature_cols)
    plot_prediction_drift(reference, current)
