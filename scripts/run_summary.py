import logging

from src.pipeline import fetch_or_load, process_stream
from src.analysis.summary import encode_features, compute_results, plot_summary, write_readme

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    raw           = fetch_or_load()
    df            = process_stream(raw)
    df, feat_cols = encode_features(df)
    results       = compute_results(df, feat_cols)
    plot_summary(results)
    write_readme(results)
