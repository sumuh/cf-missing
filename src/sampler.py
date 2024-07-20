import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random


class Sampler:

    def __init__(self):
        pass

    def add_noise(
        self, original_value: float, feature_name: str, train_data: pd.DataFrame
    ) -> float:
        """Adds noise to the original value by sampling from normal distribution
        with mean zero and standard deviation as estimated from training dataset for that feature.

        :param float original_value: value to add noise to
        :param str feature_name: feature name of original_value
        :param pd.DataFrame train_data: original training dataset
        :return float: original value with noise
        """
        mu = 0
        sigma = train_data[feature_name].std()
        noise = np.random.normal(mu, sigma)
        is_positive_noise = random.randint(0, 1)
        if is_positive_noise == 0:
            return original_value - noise
        else:
            return original_value + noise

    def sample_regression_with_noise(
        self,
        train_data: pd.DataFrame,
        input: pd.DataFrame,
        cols_with_missing_values: list[str],
        target_feature: str,
    ) -> float:
        """Samples imputation candidates via regression and adding noise.
        This corresponds to the predict + noise method in Van Buuren (2018) ch. 3.1.2

        :param pd.DataFrame train_data: training data
        :param pd.DataFrame input: input dataframe with one row with the missing value(s)
        :param list[str] cols_with_missing_values: columns with missing values (for now assumes exactly one)
        :param str target_feature: original target feature to predict
        :return float: sampled value
        """
        missing_feature = cols_with_missing_values[0]
        X_train = train_data.loc[
            :,
            (train_data.columns != missing_feature)
            & (train_data.columns != target_feature),
        ]
        y_train = train_data.loc[:, missing_feature]
        regression_model = LinearRegression().fit(X_train, y_train)
        sampled = regression_model.predict(
            input.loc[:, (input.columns != missing_feature)]
        )[0]
        return self.add_noise(sampled, missing_feature, train_data)
