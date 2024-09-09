import numpy as np
import pandas as pd

from ..evaluation.evaluation_metrics import (
    get_distance,
    get_sparsity,
)
from ..evaluation.evaluation_results_container import EvaluationResultsContainer, SingleEvaluationResultsContainer
from .data_utils import Config

def get_example_df_for_input_with_missing_values(
    test_instance_complete: np.array,
    test_instance_with_missing_values: np.array,
    test_instance_imputed: np.array,
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
    example_df = np.vstack(
        (
            test_instance_complete,
            test_instance_with_missing_values,
            test_instance_imputed,
            counterfactuals,
        )
    )
    index = ["complete input", "input with missing", "imputed input"]

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
    example_df = np.vstack(
        (
            test_instance_complete,
            counterfactuals,
        )
    )
    index = ["complete input"]

    for _ in range(len(counterfactuals)):
        index.append("counterfactual")
    return pd.DataFrame(example_df, index=index)


def print_counterfactual_generation_debug_info(
    input_for_explanations, explanations, final_explanations, mads
):
    print("input_for_explanations:")
    print(input_for_explanations)

    print(f"All k*n explanations:")
    df = pd.DataFrame(explanations)
    df.loc[-1] = pd.Series(input_for_explanations)
    df.index = df.index + 1
    df.sort_index(inplace=True)
    df["sparsity"] = df.apply(
        lambda row: get_sparsity(row[:-1].to_numpy(), input_for_explanations), axis=1
    )
    df["distance"] = df[1:].apply(
        lambda row: get_distance(row[:-2].to_numpy(), input_for_explanations, mads), axis=1
    )
    print(df)

    print(f"Final k selections:")
    df = pd.DataFrame(final_explanations)
    df.loc[-1] = pd.Series(input_for_explanations)
    df.index = df.index + 1
    df.sort_index(inplace=True)
    df["sparsity"] = df.apply(lambda row: get_sparsity(row.to_numpy(), input_for_explanations), axis=1)
    df["distance"] = df.apply(
        lambda row: get_distance(row[:-1].to_numpy(), input_for_explanations, mads), axis=1
    )
    print(df)


def get_missing_indices_for_multiple_missing_values(
    num_missing_values: int,
) -> list[int]:
    """Given number of missing values, returns indices up to that value.

    :param int num_missing_values: number of missing values
    :return list[int]: indices up to num_missing_values
    """
    return [i for i in range(num_missing_values)]

def parse_partial_evaluation_results_object_from_file(
        file_path: str
) -> EvaluationResultsContainer:
    """Returns evaluation results container object based on results saved to file.
    Purpose is to be able to make visualizations on previous runs.

    :param str file_path: path to results.txt or cf_rolling_results.txt
    :return EvaluationResultsContainer: results container with file contents parsed
    """
    params_list = []
    results_list = []
    current_section = None
    current_dict = {}
    unique_params = {}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            
            if line == "PARAMS":
                if current_dict:
                    if current_section == "RESULTS":
                        results_list.append(current_dict)
                    elif current_section == "PARAMS":
                        params_list.append(current_dict)
                current_section = "PARAMS"
                current_dict = {}

            elif line == "RESULTS":
                if current_dict:
                    params_list.append(current_dict)
                current_section = "RESULTS"
                current_dict = {}

            elif line and current_section:
                key, value = line.split("\t")
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                current_dict[key] = value

                if current_section == "PARAMS":
                    if key not in unique_params:
                        unique_params[key] = []
                    if value not in unique_params[key]:
                        unique_params[key].append(value)

        if current_section == "RESULTS":
            results_list.append(current_dict)
        elif current_section == "PARAMS":
            params_list.append(current_dict)

    container_all = EvaluationResultsContainer(Config(unique_params))
    for i in range(len(params_list)):
        container = SingleEvaluationResultsContainer(Config(params_list[i]))
        container.set_counterfactual_metrics(results_list[i])
        container_all.add_evaluation(container)

    return container_all