import pandas as pd
import numpy as np
import random

from classifier import Classifier
from sampler import Sampler
from data_utils import get_col_names_with_missing_values

class CounterfactualGenerator:

    def __init__(self, classifier: Classifier, train_data: pd.DataFrame, target_feature: str):
        self.classifier = classifier
        self.train_data = train_data
        self.sampler = Sampler()
        self.target_feature = target_feature

    def get_imputed_inputs(self, input: pd.DataFrame, n: int, 
                           col_names_with_missing_values: list[str]) -> pd.DataFrame:
        """For given input, generate n imputed versions.

        :param pd.DataFrame input: input dataframe with one row
        :param int n: number of imputed inputs to create
        :param list[str] col_names_with_missing_values: columns that contain missing value in input
        :return pd.DataFrame: dataframe with n rows, each an imputed version of input
        """
        missing_feature = col_names_with_missing_values[0]
        sampled_values = [self.sampler.sample_regression_with_noise(self.train_data, input, col_names_with_missing_values, self.target_feature) for _ in range(n)]
        imputed_inputs = pd.DataFrame(np.repeat(input.values, n, axis=0), columns=input.columns)
        imputed_inputs[missing_feature] = sampled_values
        return imputed_inputs

    def generate_explanations(self, input: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate explanation for input vector.

        :param pd.DataFrame input: input dataframe with one row
        :param int n: number of explanations to generate
        :return pd.DataFrame: dataframe with n rows
        """
        col_names_with_missing_values = get_col_names_with_missing_values(input)
        if len(col_names_with_missing_values) > 0:
            inputs_to_explain = self.get_imputed_inputs(input, n, col_names_with_missing_values)
        else:
            inputs_to_explain = pd.DataFrame(np.repeat(input.values, n, axis=0), columns=input.columns)
        print("inputs to explain:")
        print(inputs_to_explain)
        # Todo: actually implement CF generation
        explanations = inputs_to_explain
        return explanations