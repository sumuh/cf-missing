import numpy as np
import pandas as pd

from ..evaluation.evaluation_metrics import (
    get_distance,
    get_sparsity,
)

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
