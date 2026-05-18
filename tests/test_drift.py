import pytest

from src.analysis.drift import prepare_features, temporal_split, train_on_reference, add_predictions
from src.pipeline import CAT_COLS


def test_prepare_features_adds_treatment_flag(processed_df):
    result = prepare_features(processed_df)
    assert "treatment_flag" in result.columns
    assert set(result["treatment_flag"].unique()).issubset({0, 1})


def test_prepare_features_adds_encoded_columns(processed_df):
    result = prepare_features(processed_df)
    for col in CAT_COLS:
        assert f"{col}_enc" in result.columns


def test_prepare_features_does_not_mutate_input(processed_df):
    original_cols = set(processed_df.columns)
    prepare_features(processed_df)
    assert set(processed_df.columns) == original_cols


def test_temporal_split_covers_all_rows(processed_df):
    df        = prepare_features(processed_df)
    ref, cur  = temporal_split(df)
    assert len(ref) + len(cur) == len(df)


def test_temporal_split_ratio(processed_df):
    df       = prepare_features(processed_df)
    ref, cur = temporal_split(df)
    assert len(ref) == int(len(df) * 0.6)


def test_temporal_split_no_overlap(processed_df):
    df       = prepare_features(processed_df)
    ref, cur = temporal_split(df)
    assert set(ref.index).isdisjoint(set(cur.index))


def test_train_on_reference_returns_model_and_features(processed_df):
    df         = prepare_features(processed_df)
    ref, _     = temporal_split(df)
    model, feature_cols = train_on_reference(ref)
    assert hasattr(model, "predict")
    assert isinstance(feature_cols, list)
    assert len(feature_cols) > 0


def test_add_predictions_adds_columns(processed_df):
    df                  = prepare_features(processed_df)
    ref, cur            = temporal_split(df)
    model, feature_cols = train_on_reference(ref)
    result              = add_predictions(ref, model, feature_cols)
    assert "prediction" in result.columns
    assert "prediction_proba" in result.columns


def test_add_predictions_proba_in_range(processed_df):
    df                  = prepare_features(processed_df)
    ref, cur            = temporal_split(df)
    model, feature_cols = train_on_reference(ref)
    result              = add_predictions(ref, model, feature_cols)
    assert result["prediction_proba"].between(0, 1).all()
