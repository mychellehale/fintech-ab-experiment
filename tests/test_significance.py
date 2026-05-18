import pandas as pd
import numpy as np
import pytest

from src.analysis.significance import (
    chi_squared_test, confidence_intervals, power_analysis,
    segmented_analysis, logistic_regression
)


def test_chi_squared_returns_required_keys(processed_df):
    result = chi_squared_test(processed_df)
    for key in ["t_rate", "c_rate", "lift", "chi2", "p", "cohens_h", "significant"]:
        assert key in result


def test_chi_squared_rates_are_proportions(processed_df):
    result = chi_squared_test(processed_df)
    assert 0 <= result["t_rate"] <= 1
    assert 0 <= result["c_rate"] <= 1


def test_chi_squared_detects_obvious_effect():
    """A dataset with a very large conversion gap should always be significant."""
    df = pd.DataFrame({
        "group":     ["treatment"] * 200 + ["control"] * 200,
        "converted": [1] * 180 + [0] * 20 + [0] * 180 + [1] * 20,
    })
    result = chi_squared_test(df)
    assert result["significant"]
    assert result["lift"] > 0
    assert result["t_rate"] > result["c_rate"]


def test_chi_squared_no_effect_not_significant():
    """Identical conversion rates should not be significant."""
    df = pd.DataFrame({
        "group":     ["treatment"] * 200 + ["control"] * 200,
        "converted": [1] * 50 + [0] * 150 + [1] * 50 + [0] * 150,
    })
    result = chi_squared_test(df)
    assert not result["significant"]
    assert result["lift"] == 0.0


def test_confidence_intervals_runs_without_error(processed_df):
    confidence_intervals(processed_df)


def test_power_analysis_runs_without_error(processed_df):
    power_analysis(processed_df)


def test_segmented_analysis_returns_dataframe(processed_df):
    result = segmented_analysis(processed_df)
    assert isinstance(result, pd.DataFrame)


def test_segmented_analysis_has_bh_correction(processed_df):
    result = segmented_analysis(processed_df)
    assert "p_corrected" in result.columns
    assert "significant" in result.columns


def test_segmented_analysis_bh_is_conservative(processed_df):
    """BH-corrected p-values should be >= raw p-values."""
    result = segmented_analysis(processed_df)
    assert (result["p_corrected"] >= result["p_value"]).all()


def test_segmented_analysis_bh_produces_segments(processed_df):
    """With 1000 rows and ~40% control the n>=30 filter should pass for several segments."""
    result = segmented_analysis(processed_df)
    assert len(result) > 0


def test_logistic_regression_runs_without_error(processed_df):
    logistic_regression(processed_df)


def test_logistic_regression_requires_correct_columns(processed_df):
    """Should raise if required columns are missing."""
    bad_df = processed_df.drop(columns=["balance"])
    with pytest.raises(Exception):
        logistic_regression(bad_df)
