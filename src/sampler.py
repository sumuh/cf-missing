import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random


class Sampler:

    def __init__(self):
        pass

    def add_noise(
        self, original_value: float, feature_name: str, train_data: np.array
    ) -> float:
        """Adds noise to the original value by sampling from normal distribution
        with mean zero and standard deviation as estimated from training dataset for that feature.

        :param float original_value: value to add noise to
        :param int target_feature_index: index of original target feature to predict
        :param np.array train_data: original training dataset
        :return float: original value with noise
        """
        mu = 0
        sigma = np.std(train_data[:, feature_name])
        noise = np.random.normal(mu, sigma)
        is_positive_noise = random.randint(0, 1)
        if is_positive_noise == 0:
            return original_value - noise
        else:
            return original_value + noise

    def sample_regression_with_noise(
        self,
        train_data: np.array,
        input: np.array,
        indices_with_missing_values: np.array,
        target_feature_index: int,
    ) -> float:
        """Samples imputation candidates via regression and adding noise.
        This corresponds to the predict + noise method in Van Buuren (2018) ch. 3.1.2

        :param np.array train_data: dataset
        :param np.array input: 1D input array
        :param np.array indices_with_missing_values: indices of input where value is missing
        :param int target_feature_index: index of original target feature to predict
        :return float: sampled value
        """
        missing_feature_index = indices_with_missing_values[0]
        mask = np.ones(train_data.shape[1], dtype=bool)
        mask[missing_feature_index] = False
        mask[target_feature_index] = False
        X_train = train_data[:, mask]
        y_train = train_data[:, missing_feature_index]
        regression_model = LinearRegression().fit(X_train, y_train)

        input_mask = np.ones(len(input), dtype=bool)
        input_mask[missing_feature_index] = False
        sampled = regression_model.predict([input[input_mask]])[0]
        return self.add_noise(sampled, missing_feature_index, train_data)
