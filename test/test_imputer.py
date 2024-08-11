import pandas as pd
import numpy as np

from src.imputer import Imputer


def test_when_mean_imputation_then_correct_value_is_imputed():
    data = pd.DataFrame({"a": [1, 1, 3, 3], "b": [5, 6, 7, 8]}).to_numpy()
    imputer = Imputer(data)
    sample = pd.DataFrame([{"a": [np.nan], "b": [6]}]).to_numpy().ravel()
    imputed_input = imputer.mean_imputation(sample, [0])
    assert imputed_input[0] == 2


def test_when_mean_imputation_then_other_value_is_not_changed():
    data = pd.DataFrame({"a": [1, 1, 3, 3], "b": [5, 6, 7, 8]}).to_numpy()
    imputer = Imputer(data)
    sample = pd.DataFrame({"a": [np.nan], "b": [6]}).to_numpy().ravel()
    imputed_input = imputer.mean_imputation(sample, [0])
    assert imputed_input[1] == 6
