import logging

from src.pipeline import fetch_or_load, process_stream
from src.analysis.balance import summarise_groups, balance_check, plot_overview

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    raw                = fetch_or_load()
    df                 = process_stream(raw)
    treatment, control = summarise_groups(df)
    balance_check(df)
    plot_overview(df)
