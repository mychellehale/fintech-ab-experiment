"""
Shared fixtures. Uses a small synthetic dataset that mirrors the
UCI Bank Marketing schema so tests run offline and fast.
"""

import pandas as pd
import numpy as np
import pytest

CATS = {
    "job":       ["management", "technician", "blue-collar", "services", "retired"],
    "marital":   ["married", "single", "divorced"],
    "education": ["secondary", "tertiary", "primary"],
    "contact":   ["cellular", "telephone", "unknown"],
    "housing":   ["yes", "no"],
    "loan":      ["yes", "no"],
    "poutcome":  ["success", "failure", "unknown"],
}


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """
    Synthetic raw dataset before group assignment — mirrors UCI schema.
    previous is biased toward 0 (~40% control) so segments have enough rows
    in both groups to pass the n>=30 filter in segmented_analysis.
    """
    rng = np.random.default_rng(42)
    n   = 1000

    df = pd.DataFrame({
        "age":      rng.integers(18, 80, n),
        "balance":  rng.integers(-500, 5000, n),
        "duration": rng.integers(0, 600, n),
        "campaign": rng.integers(1, 10, n),
        "previous": rng.choice([0, 1, 2, 3], n, p=[0.4, 0.3, 0.2, 0.1]),
        "y":        rng.choice(["yes", "no"], n, p=[0.12, 0.88]),
    })
    for col, cats in CATS.items():
        df[col] = rng.choice(cats, n)

    return df


@pytest.fixture
def processed_df(raw_df) -> pd.DataFrame:
    """Synthetic dataset after group assignment and conversion labelling."""
    from src.pipeline import process_stream
    return process_stream(raw_df)


@pytest.fixture
def encoded_df(processed_df) -> pd.DataFrame:
    """Processed dataset with categorical encodings applied."""
    from src.pipeline import encode_categoricals
    return encode_categoricals(processed_df)
