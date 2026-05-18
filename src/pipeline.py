"""
Shared pipeline for the Feature Adoption Experiment.

Handles data fetching (with local cache), chunk-based processing,
group/conversion assignment, and categorical encoding.
Imported by all analysis modules to avoid duplication.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import Generator
from ucimlrepo import fetch_ucirepo

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CACHE_PATH = Path(".cache/data.parquet")

CAT_COLS = ["job", "marital", "education", "contact", "housing", "loan", "poutcome"]


def fetch_raw() -> pd.DataFrame:
    logger.info("Fetching Bank Marketing dataset from UCI...")
    repo = fetch_ucirepo(id=222)
    df   = pd.concat([repo.data.features, repo.data.targets], axis=1)
    logger.info(f"Fetched {len(df):,} rows, {df.shape[1]} columns")
    return df


def fetch_or_load() -> pd.DataFrame:
    """Return cached parquet if available, otherwise fetch from UCI and cache it."""
    if CACHE_PATH.exists():
        logger.info(f"Loading cached data from {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)
    df = fetch_raw()
    CACHE_PATH.parent.mkdir(exist_ok=True)
    df.to_parquet(CACHE_PATH)
    logger.info(f"Cached raw data to {CACHE_PATH}")
    return df


def stream_chunks(df: pd.DataFrame, chunk_size: int = CHUNK_SIZE) -> Generator[pd.DataFrame, None, None]:
    """
    Yield DataFrame chunks of chunk_size rows.
    In production this generator would wrap a Kafka consumer or
    pd.read_csv(chunksize=N) rather than slicing an in-memory DataFrame.
    """
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start: start + chunk_size].copy()


def transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Assign treatment/control groups and conversion label.
        treatment = users previously contacted (previous > 0)
        control   = users not previously contacted
        converted = subscribed to term deposit (y == 'yes')
    """
    chunk = chunk.copy()
    chunk["group"]     = np.where(chunk["previous"] > 0, "treatment", "control")
    chunk["converted"] = np.where(chunk["y"] == "yes", 1, 0)
    return chunk


def process_stream(df: pd.DataFrame) -> pd.DataFrame:
    """Drive the stream: pull chunks, transform, concatenate."""
    logger.info(f"Starting stream processing | chunk_size={CHUNK_SIZE}")
    processed_chunks = []
    total    = 0
    n_chunks = (len(df) // CHUNK_SIZE) + 1

    for chunk in tqdm(stream_chunks(df), total=n_chunks, desc="Processing chunks", unit="chunk"):
        processed_chunks.append(transform_chunk(chunk))
        total += len(chunk)

    logger.info(f"Stream complete | total_rows={total:,}")
    return pd.concat(processed_chunks, ignore_index=True)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode all categorical columns in CAT_COLS and return the updated DataFrame.
    Centralised here so encoding is consistent across all analysis modules.
    """
    df = df.copy()
    le = LabelEncoder()
    for col in CAT_COLS:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    return df
