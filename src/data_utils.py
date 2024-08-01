import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from .constants import *


def get_wine_dataset_config() -> dict:
    """Return details of wine dataset.

    :return dict: config dictionary
    """
    return {
        config_dataset_name: "WineQuality red",
        config_file_path: "winequality/winequality-red.csv",
        config_target_name: "quality",
        config_multiclass_target: True,
        config_missing_values: False,
        config_multiclass_threshold: 5,
        config_separator: ";",
    }


def get_diabetes_dataset_config() -> dict:
    """Return details of diabetes dataset.

    :return dict: config dictionary
    """
    return {
        config_dataset_name: "Pima Indians Diabetes",
        config_file_path: "diabetes.csv",
        config_target_name: "Outcome",
        config_multiclass_target: False,
        config_missing_values: True,
        config_separator: ",",
    }


def load_data(relative_path: str, separator: str) -> pd.DataFrame:
    """Loads the given dataset.

    :param str path: path to dataset .csv; relative to data dir
    :param str separator: csv separator
    :return pd.DataFrame: raw dataset
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = f"{dir_path}/../data/{relative_path}"
    data = pd.read_csv(data_path, sep=separator)
    return data


def transform_target_to_binary_class(
    data: pd.DataFrame, target_name: str, threshold: int
) -> pd.DataFrame:
    """Transforms multiclass target column to binary class.

    :param pd.DataFrame data: data
    :param str target_name: target column name
    :param int threshold: cutoff point
    :return pd.DataFrame: data with target column replaced
    """
    data[target_name] = np.where(data[target_name] <= threshold, 0, 1)
    return data


def explore_data(data: pd.DataFrame):
    """Print auto-generated summaries and plots for dataframe.

    :param pd.DataFrame data: dataset to summarize
    """
    symbol = "="
    print(f"{symbol*12} Data exploration {symbol*12}")
    print("\nColumn types:\n")
    print(data.info())
    print("\nStatistics:\n")
    print(data.describe())
    print("\nCorrelations:")
    print(data.corr())
    data.hist()
    plt.show()
    print(f"{symbol*42}")


def get_target_index(data: pd.DataFrame, target_name: str) -> int:
    """Return index of target variable based on name.

    :param pd.DataFrame data: dataset
    :param str target_name: target feature name
    :return int: target feature index
    """
    return data.columns.get_loc(target_name)


def get_cat_indices(data: pd.DataFrame, target_index: int) -> np.array:
    """Returns indices of categorical predictors.

    :param pd.DataFrame data: dataset
    :param int: target feature index
    :return np.array: indices of categorical predictors
    """
    predictors = data.drop(data.columns[target_index], axis=1)
    cat_predictor_names = predictors.select_dtypes(include=["object"]).columns
    return np.array(
        [data.columns.get_loc(predictor_name) for predictor_name in cat_predictor_names]
    )


def get_num_indices(data: pd.DataFrame, target_index: int) -> np.array:
    """Returns indices of numeric predictors.

    :param pd.DataFrame data: dataset
    :param int: target feature index
    :return np.array: indices of numeric predictors
    """
    predictors = data.drop(data.columns[target_index], axis=1)
    num_predictor_names = predictors.select_dtypes(
        include=["int", "float"]
    ).columns.to_list()
    return np.array(
        [data.columns.get_loc(predictor_name) for predictor_name in num_predictor_names]
    )


def get_indices_with_missing_values(sample: np.array) -> np.array:
    """Returns indices where the value is missing.

    :param np.array sample: 1D input array
    :return np.array: indices of features with missing values
    """
    return np.where(np.isnan(sample))[0]


def get_test_input_1() -> dict:
    """Test input for which we expect classification outcome=0.

    :return dict: test input dict
    """
    return {
        "Pregnancies": 2,
        "Glucose": 50,
        "BloodPressure": 50,
        "SkinThickness": 20,
        "Insulin": 0,
        "BMI": 23,
        "DiabetesPedigreeFunction": 0.356,
        "Age": 34,
    }


def get_test_input_2() -> dict:
    """Test input for which we expect classification outcome=1.

    :return dict: test input dict
    """
    return {
        "Pregnancies": 6,
        "Glucose": 150,
        "BloodPressure": 70,
        "SkinThickness": 30,
        "Insulin": 140,
        "BMI": 29,
        "DiabetesPedigreeFunction": 0.90,
        "Age": 50,
    }
