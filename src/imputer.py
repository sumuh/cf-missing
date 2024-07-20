import pandas as pd
import numpy as np

from .sampler import Sampler


class Imputer:

    def __init__(self):
        self.sampler = Sampler()

    def check_number_of_missing_values(self, cols_with_missing_values: list[str]):
        """Ensure number of missing values is within allowed limits.

        :param list[str] cols_with_missing_values: columns that contain missing value in input
        :raises RuntimeError: if no missing values were provided
        :raises NotImplementedError: for now, if more than 1 missing values provided
        """
        if len(cols_with_missing_values) == 0:
            raise RuntimeError("Sampler invoked but no missing values identified!")
        elif len(cols_with_missing_values) != 1:
            raise NotImplementedError(
                f"Sampling implemented for precisely one missing value, were: {cols_with_missing_values}"
            )

    def mean_imputation(
        self,
        train_data: pd.DataFrame,
        input: pd.DataFrame,
        cols_with_missing_values: list[str],
    ) -> pd.DataFrame:
        """Returns input with missing value imputed with dataset mean.

        :param pd.DataFrame train_data: dataset
        :param pd.DataFrame input: input dataframe with one row
        :param list[str] cols_with_missing_values: columns that contain missing value in input
        :return pd.DataFrame: input imputed with mean
        """
        self.check_number_of_missing_values(cols_with_missing_values)
        missing_feature = cols_with_missing_values[0]
        feature_mean = train_data[missing_feature].mean()
        input.at[0, missing_feature] = feature_mean
        return input

    def subgroup_mean_imputation(
        self,
        train_data: pd.DataFrame,
        input: pd.DataFrame,
        cols_with_missing_values: list[str],
        target_feature: str,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def regression_imputation(
        self,
        train_data: pd.DataFrame,
        input: pd.DataFrame,
        cols_with_missing_values: list[str],
        target_feature: str,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def multiple_imputation(
        self,
        train_data: pd.DataFrame,
        input: pd.DataFrame,
        cols_with_missing_values: list[str],
        target_feature: str,
        n: int,
    ) -> pd.DataFrame:
        """For given input, generate n imputed versions.

        :param pd.DataFrame train_data: dataset
        :param pd.DataFrame input: input dataframe with one row
        :param list[str] cols_with_missing_values: columns that contain missing value in input
        :param str target_feature: original target feature to predict
        :param int n: number of imputed inputs to create
        :return pd.DataFrame: dataframe with n rows, each an imputed version of input
        """
        self.check_number_of_missing_values(cols_with_missing_values)
        missing_feature = cols_with_missing_values[0]
        sampled_values = [
            self.sampler.sample_regression_with_noise(
                self.train_data,
                input,
                cols_with_missing_values,
                target_feature,
            )
            for _ in range(n)
        ]
        imputed_inputs = pd.DataFrame(
            np.repeat(input.values, n, axis=0), columns=input.columns
        )
        imputed_inputs[missing_feature] = sampled_values
        return imputed_inputs
