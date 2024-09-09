import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict


class Config:
    """Config object for accessing config with dot notation."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def get_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.get_dict()
            else:
                result[key] = value
        return result


def load_data(relative_path: str, separator: str) -> pd.DataFrame:
    """Loads the given dataset.

    :param str path: path to dataset .csv; relative to data dir
    :param str separator: csv separator
    :return pd.DataFrame: raw dataset
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = f"{dir_path}/../../data/{relative_path}"
    data = pd.read_csv(data_path, sep=separator)
    return data


def transform_data(data: pd.DataFrame, data_config: Config) -> pd.DataFrame:
    # For now multiclass classification is not supported so convert to binary class if needed
    if data_config.multiclass_target:
        data = transform_target_to_binary_class(
            data,
            data_config.target_name,
            data_config.multiclass_threshold,
        )

    print(f"Rows before drop: {len(data)}")

    if data_config.dataset_name == "Pima Indians Diabetes":
        data = drop_rows_with_missing_values(data)

    print(f"Rows after drop: {len(data)}")

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
    data[target_name] = np.where(data[target_name] < threshold, 0, 1)
    return data


def drop_rows_with_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Drops rows where any of the specified features
    has value zero (assuming Diabetes dataset).

    :param pd.DataFrame data: diabetes dataset
    :return pd.DataFrame: dataset with no missing values
    """
    data = data[
        (data.Glucose != 0)
        & (data.BloodPressure != 0)
        & (data.SkinThickness != 0)
        & (data.Insulin != 0)
        & (data.BMI != 0)
    ]
    return data


def get_X_y(
    data: np.array, predictor_indices: list[int], target_index: int
) -> tuple[np.array, np.array]:
    """Get X train and y train.

    :param np.array data: data
    :param list[int]: indices of predictor variables
    :param int: index of target variable
    :return tuple[np.array, np.array]: X_train, y_train
    """
    X_train = data[:, predictor_indices]
    y_train = data[:, target_index].ravel()
    return X_train, y_train


def get_data_metrics(data: pd.DataFrame, target_name: str) -> dict[str, int]:
    """Obtain metrics from dataset.

    :param pd.DataFrame data: dataset
    :param str target_name: name of target variable
    :return dict[str, int]: metrics
    """
    total_rows = len(data)
    label_0 = len(data[data[target_name] == 0])
    label_1 = len(data[data[target_name] == 1])
    return {
        "total_rows": total_rows,
        "label_0": label_0,
        "label_1": label_1,
    }


def get_feature_min_values(data: np.array) -> np.array:
    """For each column in data returns mininum value.

    :param np.array data: data
    :return np.array: array of min values
    """
    return np.min(data, axis=0)


def get_feature_max_values(data: np.array) -> np.array:
    """For each column in data returns maximum value.

    :param np.array data: data
    :return np.array: array of max values
    """
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


def get_str_from_dict(dict_to_save: dict, dict_name: str) -> str:
    """Utility method for generating string based on dictionary.

    :param dict dict_to_save: dictionary to stringify
    :param str dict_name: name of dictionary
    :return str: string representation of dict name and contents
    """
    title = f"{dict_name}\n"
    content = "\n".join([f"{item[0]}\t{item[1]}" for item in dict_to_save.items()])
    return title + content
