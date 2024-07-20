import pandas as pd

from src.imputer import Imputer


def test_when_mean_imputation_then_correct_value_is_imputed():
    imputer = Imputer()
    data = pd.DataFrame({"a": [1, 1, 3, 3], "b": [5, 6, 7, 8]})
    sample = pd.DataFrame([{"a": [pd.NA], "b": [6]}])
    imputed_input = imputer.mean_imputation(data, sample, ["a"])
    assert imputed_input.loc[0, "a"] == 2


def test_when_mean_imputation_then_other_value_is_not_changed():
    imputer = Imputer()
    data = pd.DataFrame({"a": [1, 1, 3, 3], "b": [5, 6, 7, 8]})
    sample = pd.DataFrame({"a": [pd.NA], "b": [6]})
    imputed_input = imputer.mean_imputation(data, sample, ["a"])
    assert imputed_input.loc[0, "b"] == 6
