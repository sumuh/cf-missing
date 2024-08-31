import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

from src.utils.data_utils import get_indices_with_missing_values, _calculate_mad


def test_when_no_missing_values_then_no_indices_with_missing_values_found():
    sample = pd.DataFrame({"a": [0.5], "b": [0], "c": [5]}).to_numpy().ravel()
    missing_value_indices = get_indices_with_missing_values(sample)
    assert len(missing_value_indices) == 0


def test_when_one_missing_value_then_one_index_with_missing_value_found():
    sample = pd.DataFrame({"a": [np.nan], "b": [0], "c": [5]}).to_numpy().ravel()
    missing_value_indices = get_indices_with_missing_values(sample)
    assert_array_equal(missing_value_indices, np.array([0]))


def test_when_two_missing_values_then_two_columns_with_missing_values_found():
    sample = pd.DataFrame({"a": [np.nan], "b": [np.nan], "c": [5]}).to_numpy().ravel()
    missing_value_indices = get_indices_with_missing_values(sample)
    assert_array_equal(missing_value_indices, np.array([0, 1]))


def test_when_three_missing_values_then_three_columns_with_missing_values_found():
    sample = (
        pd.DataFrame({"a": [np.nan], "b": [np.nan], "c": [np.nan]}).to_numpy().ravel()
    )
    missing_value_indices = get_indices_with_missing_values(sample)
    assert_array_equal(missing_value_indices, np.array([0, 1, 2]))


def test_when_mad_is_integer_then_correct_value_calculated():
    arr = np.array([1, 1, 3, 3])
    mad = _calculate_mad(arr)
    assert mad == 1


def test_when_mad_is_fraction_then_correct_value_calculated():
    arr = np.array([1, 1, 2, 2])
    mad = _calculate_mad(arr)
    assert mad == 0.5


def test_when_mad_is_zero_then_correct_value_calculated():
    arr = np.array([1, 1, 1, 1])
    mad = _calculate_mad(arr)
    assert mad == 0


def test_when_input_to_mad_is_empty_then_None_returned():
    arr = np.array([])
    mad = _calculate_mad(arr)
    assert mad == None
