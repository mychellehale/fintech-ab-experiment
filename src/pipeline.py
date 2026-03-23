"""
Shared streaming pipeline for the Feature Adoption Experiment.

Handles data fetching, chunk-based stream processing, and group/conversion
assignment. Imported by all analysis scripts to avoid duplication.
"""

import logging
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm
from typing import Generator

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000


def fetch_raw() -> pd.DataFrame:
    """Fetch the Bank Marketing dataset from UCI and return as a single DataFrame."""
    logger.info("Fetching Bank Marketing dataset from UCI...")
    repo = fetch_ucirepo(id=222)
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    logger.info(f"Fetched {len(df):,} rows, {df.shape[1]} columns")
    return df


def stream_chunks(df: pd.DataFrame, chunk_size: int = CHUNK_SIZE) -> Generator[pd.DataFrame, None, None]:
    """
    Yield DataFrame chunks of chunk_size rows.
    Simulates streaming from a database or Kafka topic -- in production you
    would replace this with a real source (e.g. pd.read_csv with chunksize,
    or a Kafka consumer).
    """
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start: start + chunk_size].copy()


def transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Apply group assignment and conversion label to a single chunk.
        treatment = users previously contacted (previous > 0)
        control   = users not previously contacted
        converted = subscribed to term deposit (y == 'yes')
    """
    chunk = chunk.copy()
    chunk["group"]     = np.where(chunk["previous"] > 0, "treatment", "control")
    chunk["converted"] = np.where(chunk["y"] == "yes", 1, 0)
    return chunk


def process_stream(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drive the stream: pull chunks, transform each one, concatenate.
    tqdm provides a progress bar; logging captures structured milestones.
    In production the final concat would be replaced with writes to a sink.
    """
    logger.info(f"Starting stream processing | chunk_size={CHUNK_SIZE}")
    processed_chunks = []
    total = 0
    n_chunks = (len(df) // CHUNK_SIZE) + 1

    for chunk in tqdm(stream_chunks(df), total=n_chunks, desc="Processing chunks", unit="chunk"):
        transformed = transform_chunk(chunk)
        processed_chunks.append(transformed)
        total += len(chunk)

    logger.info(f"Stream complete | total_rows={total:,}")
    return pd.concat(processed_chunks, ignore_index=True)
