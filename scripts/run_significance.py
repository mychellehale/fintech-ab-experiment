import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import fetch_raw, process_stream
from src.analysis.significance import (
    chi_squared_test, confidence_intervals, power_analysis,
    segmented_analysis, logistic_regression, plot_results
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    raw             = fetch_raw()
    df              = process_stream(raw)
    chi_squared_test(df)
    confidence_intervals(df)
    power_analysis(df)
    segment_results = segmented_analysis(df)
    logistic_regression(df)
    plot_results(df, segment_results)
