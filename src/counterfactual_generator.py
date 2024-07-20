import pandas as pd
import numpy as np

from .classifier import Classifier
from .sampler import Sampler


class CounterfactualGenerator:

    def __init__(
        self, classifier: Classifier, train_data: pd.DataFrame, target_feature: str
    ):
        self.classifier = classifier
        self.train_data = train_data
        self.target_feature = target_feature

    def search_counterfactuals(self, input: pd.DataFrame):
        pass

    def generate_explanations(
        self, input: pd.DataFrame, cols_with_missing_values: list[str], n: int
    ) -> pd.DataFrame:
        """Generate explanation for input vector(s).

        :param pd.DataFrame input: input dataframe with one or multiple rows
        :param list[str] cols_with_missing_values: features with missing values
        :param int n: number of explanations to generate
        :return pd.DataFrame: dataframe with n rows
        """
        if len(input) == 1:
            # Generate n explanations based on single input vector
            explanations = self.search_counterfactuals(input)
        else:
            # Generate n explanations based on multiple input vecotrs (from multiple imputation)
            explanations = self.search_counterfactuals(input)
        return explanations
