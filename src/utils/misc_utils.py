import numpy as np
import pandas as pd
import yaml
import json

from ..evaluation.evaluation_results_container import (
    EvaluationResultsContainer,
    SingleEvaluationResultsContainer,
)
from .data_utils import Config, get_str_from_dict


def write_run_configuration_to_file(
    data: pd.DataFrame, params: Config, results_file_path: str
):
    params_dict = {"all_params": params.get_dict()}
    data_stats_dict = {
        "data_stats": {
            "column_names": data.columns.to_list(),
            "rows": len(data),
        }
    }
    with open(results_file_path, "a") as file:
        yaml.dump(params_dict, file, default_flow_style=False, sort_keys=False)
        yaml.dump(data_stats_dict, file, default_flow_style=False, sort_keys=False)


def get_example_df_for_input_with_missing_values(
    test_instance_complete: np.array,
    test_instance_with_missing_values: np.array,
    counterfactuals: np.array,
) -> pd.DataFrame:
    """Creates a neat dataframe that shows the original input,
    input with missing values, (mean) imputed input, and counterfactuals.
    For qualitative evaluation of counterfactuals.

    :param np.array test_instance_complete: complete test instance
    :param np.array test_instance_with_missing_values: test instance with missing value(s)
    :param np.array test_instance_imputed: mean imputed test instance (input to classifier)
    :param np.array counterfactuals: counterfactuals
    :return pd.DataFrame: example df
    """
    if len(counterfactuals) > 0:
        example_df = np.vstack(
            (
                test_instance_complete,
                test_instance_with_missing_values,
                counterfactuals,
            )
        )
    else:
        example_df = np.vstack(
            (
                test_instance_complete,
                test_instance_with_missing_values,
            )
        )
    index = ["complete input", "input with missing"]

    for _ in range(len(counterfactuals)):
        index.append("counterfactual")
    return pd.DataFrame(example_df, index=index)


def get_example_df_for_complete_input(
    test_instance_complete: np.array,
    counterfactuals: np.array,
) -> pd.DataFrame:
    """Creates a neat dataframe that shows the original input
    and counterfactuals.
    For qualitative evaluation of counterfactuals.

    :param np.array test_instance_complete: complete test instance
    :param np.array counterfactuals: counterfactuals
    :return pd.DataFrame: example df
    """
    if len(counterfactuals) > 0:
        example_df = np.vstack(
            (
                test_instance_complete,
                counterfactuals,
            )
        )
    else:
        example_df = test_instance_complete
    index = ["complete input"]

    for _ in range(len(counterfactuals)):
        index.append("counterfactual")
    return pd.DataFrame(example_df, index=index)


def get_missing_indices_for_multiple_missing_values(
    num_missing_values: int,
) -> list[int]:
    """Given number of missing values, returns indices up to that value.

    :param int num_missing_values: number of missing values
    :return list[int]: indices up to num_missing_values
    """
    return [i for i in range(num_missing_values)]


def parse_results_file(
    file_path: str,
) -> tuple[dict, EvaluationResultsContainer]:
    """Parses given results.txt file and returns data info + evaluation results object.
    Purpose is to be able to make visualizations on previous runs.

    :param str file_path: path to results.txt
    :return EvaluationResultsContainer: results container with file contents parsed
    """
    with open(file_path, "r") as file:
        yaml_content = yaml.safe_load(file)

    data_stats = yaml_content.get("data_stats", {})
    all_params = yaml_content.get("all_params", {})

    evaluation_results_container = EvaluationResultsContainer(Config(all_params))

    for run_data in yaml_content.get("runs", []):
        run = run_data.get("run", {})
        run_params = run.get("run_params", {})
        run_metrics = run.get("run_metrics", {})
        single_results_container = SingleEvaluationResultsContainer(Config(run_params))
        single_results_container.set_counterfactual_metrics(run_metrics)
        evaluation_results_container.add_evaluation(single_results_container)
    return data_stats, evaluation_results_container
