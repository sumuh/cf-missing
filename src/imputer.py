import pandas as pd
import numpy as np

from .sampler import Sampler


class Imputer:

    def __init__(self):
        self.sampler = Sampler()

    def check_number_of_missing_values(self, indices_with_missing_values: np.array):
        """Ensure number of missing values is within allowed limits.

        :param np.array indices_with_missing_values: indices of input where value is missing
        :raises RuntimeError: if no missing values were provided
        :raises NotImplementedError: for now, if more than 1 missing values provided
        """
        if len(indices_with_missing_values) == 0:
            raise RuntimeError("Sampler invoked but no missing values identified!")
        elif len(indices_with_missing_values) != 1:
            raise NotImplementedError(
                f"Sampling implemented for precisely one missing value, were: {indices_with_missing_values}"
            )

    def mean_imputation(
        self,
        train_data: np.array,
        input: np.array,
        indices_with_missing_values: np.array,
    ) -> np.array:
        """Returns input with missing value imputed with dataset mean.

        :param np.array train_data: dataset
        :param np.array input: 1D input array
        :param np.array indices_with_missing_values: indices of input where value is missing
        :return np.array: input imputed with mean
        """
        self.check_number_of_missing_values(indices_with_missing_values)
        index_of_missing_feature = indices_with_missing_values[0]
        feature_mean = np.mean(train_data[:, index_of_missing_feature])
        input[index_of_missing_feature] = feature_mean
        return input

    def subgroup_mean_imputation(
        self,
        train_data: np.array,
        input: np.array,
        indices_with_missing_values: np.array,
    ) -> np.array:
        raise NotImplementedError

    def regression_imputation(
        self,
        train_data: np.array,
        input: np.array,
        indices_with_missing_values: np.array,
        target_feature: str,
    ) -> np.array:
        raise NotImplementedError

    def multiple_imputation(
        self,
        train_data: np.array,
        input: np.array,
        indices_with_missing_values: np.array,
        target_feature_index: int,
        n: int,
    ) -> np.array:
        """For given input, generate n imputed versions.

        :param np.array train_data: dataset
        :param np.array input: 1D input array
        :param np.array indices_with_missing_values: indices of input where value is missing
        :param int target_feature_index: index of original target feature to predict
        :param int n: number of imputed inputs to create
        :return np.array: array with n rows, each an imputed version of input
        """
        self.check_number_of_missing_values(indices_with_missing_values)
        missing_feature = indices_with_missing_values[0]
        sampled_values = [
            self.sampler.sample_regression_with_noise(
                train_data,
                input,
                indices_with_missing_values,
                target_feature_index,
            )
            for _ in range(n)
        ]
        imputed_inputs = np.array(
            np.repeat(input.values, n, axis=0), columns=input.columns
        )
        imputed_inputs[missing_feature] = sampled_values
        return imputed_inputs
