import pytest

from src.analysis.shap import prepare_features, train_classifier


def test_prepare_features_excludes_duration(processed_df):
    X, y = prepare_features(processed_df)
    assert "duration" not in X.columns


def test_prepare_features_includes_treatment_flag(processed_df):
    X, y = prepare_features(processed_df)
    assert "treatment_flag" in X.columns


def test_prepare_features_no_nulls(processed_df):
    X, y = prepare_features(processed_df)
    assert X.isna().sum().sum() == 0


def test_prepare_features_target_is_binary(processed_df):
    X, y = prepare_features(processed_df)
    assert set(y.unique()).issubset({0, 1})


def test_prepare_features_row_count_preserved(processed_df):
    X, y = prepare_features(processed_df)
    assert len(X) == len(processed_df)
    assert len(y) == len(processed_df)


def test_train_classifier_returns_correct_shapes(processed_df):
    X, y               = prepare_features(processed_df)
    model, X_train, X_test = train_classifier(X, y)
    assert len(X_train) + len(X_test) == len(X)


def test_train_classifier_model_can_predict(processed_df):
    X, y               = prepare_features(processed_df)
    model, X_train, X_test = train_classifier(X, y)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})
