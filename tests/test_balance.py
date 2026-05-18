import pytest

from src.analysis.balance import summarise_groups, balance_check


def test_summarise_groups_returns_correct_split(processed_df):
    treatment, control = summarise_groups(processed_df)
    assert (treatment["group"] == "treatment").all()
    assert (control["group"] == "control").all()
    assert len(treatment) + len(control) == len(processed_df)


def test_summarise_groups_no_empty_groups(processed_df):
    treatment, control = summarise_groups(processed_df)
    assert len(treatment) > 0
    assert len(control) > 0


def test_balance_check_runs_without_error(processed_df):
    balance_check(processed_df)


def test_balance_check_does_not_mutate_input(processed_df):
    original_shape = processed_df.shape
    balance_check(processed_df)
    assert processed_df.shape == original_shape
