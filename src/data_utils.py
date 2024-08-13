import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from .constants import *


def get_wine_dataset_config() -> dict:
    """Return details of wine dataset.

    :return dict: config dictionary
    """
    return {
        config_dataset_name: "WineQuality red",
        config_file_path: "winequality/winequality-red.csv",
        config_target_name: "quality",
        config_target_class: 1,
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
        config_target_class: 0,
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


def show_correlation_matrix(data: pd.DataFrame):
    """Plots correlation matrix.

    :param pd.DataFrame data: data
    """
    f = plt.figure(figsize=(12, 10))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(
        range(data.select_dtypes(["number"]).shape[1]),
        data.select_dtypes(["number"]).columns,
        fontsize=14,
        rotation=45,
    )
    plt.yticks(
        range(data.select_dtypes(["number"]).shape[1]),
        data.select_dtypes(["number"]).columns,
        fontsize=14,
    )
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()


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
    show_correlation_matrix(data)
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


def get_feature_min_values(data: np.array) -> float:
    return np.min(data, axis=0)


def get_feature_max_values(data: np.array) -> float:
    return np.max(data, axis=0)


def _calculate_mad(values: np.array) -> float:
    """Calculates the mean absolute deviation (MAD) of values.

    :param np.array values: vector of values
    :return float: MAD of values
    """
    if len(values) == 0:
        return None
    return np.sum(abs(values - np.mean(values))) / len(values)


def get_feature_mads(data: np.array) -> np.array:
    """Get MAD (mean absolute deviation) for each column (feature) in data.

    :param np.array data: data to calculate MADS for
    :return np.array: MAD for each feature
    """
    return np.apply_along_axis(_calculate_mad, 0, data)


def get_indices_with_missing_values(sample: np.array) -> np.array:
    """Returns indices where the value is missing.

    :param np.array sample: 1D input array
    :return np.array: indices of features with missing values
    """
    return np.where(np.isnan(sample))[0]


def get_averages_from_dict_of_arrays(arr_dict: dict[str, np.array]) -> dict[str, float]:
    """Given a dict of numpy arrays, returns a new dict with averages
    for each array.

    :param dict[str, np.array] arr_dict: dict with numpy arrays
    :return dict[str, float]: dict with array averages
    """
    avg_metric_names = [
        "avg_" + metric_name if not metric_name.startswith("avg_") else metric_name
        for metric_name in arr_dict.keys()
    ]
    averages = [np.mean(np.array(metric_arr)) for metric_arr in arr_dict.values()]
    return {
        metric_name: avg_metric
        for metric_name, avg_metric in zip(avg_metric_names, averages)
    }


def get_counts_of_values_in_arrays(list_of_arrs: list[np.array]) -> dict[str, int]:
    """Given a list of numpy arrays, returns a dict where keys are unique values
    and values are occurrence counts of those values across all arrays.

    :param list[np.array] list_of_arrs: list of numpy arrays
    :return dict[str, int]: dict mapping values to occurrence counts
    """
    count_dict = defaultdict(int)
    for arr in list_of_arrs:
        arr = arr.flatten()
        for value in np.unique(arr):
            count_dict[str(value)] += int(np.sum(arr == value))

    return dict(count_dict)


def plot_metric_histograms(
    metrics: dict[str, np.array], save_to_file: bool, file_path: str = None
):
    """Plots histograms of each metric and saves image to given path if save_to_file is True.

    :param dict[str, np.array] metrics: dict of metrics
    :param bool save_to_file: whether to save image
    :param str file_path: file path for saving image, defaults to None
    """
    fig, axes = plt.subplots(2, 4, figsize=(19, 15))
    axes = axes.flatten()
    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        axes[i].hist(metric_values)
        axes[i].set_title(metric_name)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_to_file:
        plt.savefig(file_path)
    plt.show()


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
