import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from src.pipeline import transform_chunk, stream_chunks, process_stream, encode_categoricals, CAT_COLS


def test_transform_chunk_group_assignment(raw_df):
    result = transform_chunk(raw_df.copy())
    assert set(result["group"].unique()).issubset({"treatment", "control"})
    assert (result.loc[result["previous"] > 0, "group"] == "treatment").all()
    assert (result.loc[result["previous"] == 0, "group"] == "control").all()


def test_transform_chunk_conversion_label(raw_df):
    result = transform_chunk(raw_df.copy())
    assert set(result["converted"].unique()).issubset({0, 1})
    assert (result.loc[result["y"] == "yes", "converted"] == 1).all()
    assert (result.loc[result["y"] == "no", "converted"] == 0).all()


def test_transform_chunk_does_not_mutate_input(raw_df):
    original_cols = set(raw_df.columns)
    transform_chunk(raw_df)
    assert set(raw_df.columns) == original_cols


def test_stream_chunks_covers_all_rows(raw_df):
    chunks = list(stream_chunks(raw_df, chunk_size=100))
    assert sum(len(c) for c in chunks) == len(raw_df)


def test_stream_chunks_respects_chunk_size(raw_df):
    chunks = list(stream_chunks(raw_df, chunk_size=50))
    for chunk in chunks[:-1]:
        assert len(chunk) == 50


def test_process_stream_returns_all_rows(raw_df):
    assert len(process_stream(raw_df)) == len(raw_df)


def test_process_stream_adds_required_columns(raw_df):
    result = process_stream(raw_df)
    assert "group" in result.columns
    assert "converted" in result.columns


def test_encode_categoricals_adds_encoded_columns(processed_df):
    result = encode_categoricals(processed_df)
    for col in CAT_COLS:
        assert f"{col}_enc" in result.columns


def test_encode_categoricals_no_nulls(processed_df):
    result = encode_categoricals(processed_df)
    for col in CAT_COLS:
        assert result[f"{col}_enc"].isna().sum() == 0


def test_encode_categoricals_does_not_mutate_input(processed_df):
    original_cols = set(processed_df.columns)
    encode_categoricals(processed_df)
    assert set(processed_df.columns) == original_cols


def test_fetch_or_load_uses_cache_when_present(raw_df, tmp_path):
    """Should load from parquet and never call fetch_raw when cache exists."""
    from src.pipeline import fetch_or_load, CACHE_PATH
    import src.pipeline as pipeline_module

    cache = tmp_path / "data.parquet"
    raw_df.to_parquet(cache)

    with patch.object(pipeline_module, "CACHE_PATH", cache), \
         patch.object(pipeline_module, "fetch_raw") as mock_fetch:
        result = fetch_or_load()
        mock_fetch.assert_not_called()
        assert len(result) == len(raw_df)


def test_fetch_or_load_writes_cache_when_missing(raw_df, tmp_path):
    """Should call fetch_raw and write a parquet when no cache exists."""
    from src.pipeline import fetch_or_load
    import src.pipeline as pipeline_module

    cache = tmp_path / "data.parquet"

    with patch.object(pipeline_module, "CACHE_PATH", cache), \
         patch.object(pipeline_module, "fetch_raw", return_value=raw_df):
        fetch_or_load()
        assert cache.exists()
