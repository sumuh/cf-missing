import pandas as pd

from src.data_utils import get_cols_with_missing_values


def test_when_no_missing_values_then_no_columns_with_missing_values_found():
    sample = pd.DataFrame({"a": [0.5], "b": [0], "c": ["xyz"]})
    missing_col_names = get_cols_with_missing_values(sample)
    assert missing_col_names == []


def test_when_one_missing_value_then_one_column_with_missing_value_found():
    sample = pd.DataFrame({"a": [pd.NA], "b": [0], "c": ["xyz"]})
    missing_col_names = get_cols_with_missing_values(sample)
    assert missing_col_names == ["a"]


def test_when_two_missing_values_then_two_columns_with_missing_values_found():
    sample = pd.DataFrame({"a": [pd.NA], "b": [pd.NA], "c": ["xyz"]})
    missing_col_names = get_cols_with_missing_values(sample)
    assert missing_col_names == ["a", "b"]


def test_when_three_missing_values_then_three_columns_with_missing_values_found():
    sample = pd.DataFrame({"a": [pd.NA], "b": [pd.NA], "c": [pd.NA]})
    missing_col_names = get_cols_with_missing_values(sample)
    assert missing_col_names == ["a", "b", "c"]
