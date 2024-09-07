import numpy as np
import pandas as pd

def get_example_df(
                   indices_with_missing_values: np.array,
                   test_instance_complete: np.array,
                   test_instance_with_missing_values: np.array,
                   test_instance_imputed: np.array,
                   counterfactuals: np.array,
                   ) -> pd.DataFrame:
        """Creates a neat dataframe that shows the original input,
        input with missing values, (mean) imputed input, and counterfactuals. 
        For qualitative evaluation of counterfactuals.

        :param np.array indices_with_missing_values: indices that missing values were introduced to
        :param np.array test_instance_complete: complete test instance
        :param np.array test_instance_with_missing_values: test instance with missing value(s)
        :param np.array test_instance_imputed: mean imputed test instance (input to classifier)
        :param np.array counterfactuals: counterfactuals
        :return pd.DataFrame: example df
        """
        if len(indices_with_missing_values) > 0:
            example_df = np.vstack(
                (
                    test_instance_complete,
                    test_instance_with_missing_values,
                    test_instance_imputed,
                    counterfactuals,
                )
            )
            index = ["complete input", "input with missing", "imputed input"]
        else:
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

def get_missing_indices_for_multiple_missing_values(num_missing_values: int) -> list[int]:
     return [i for i in range(num_missing_values - 1)]