import pandas as pd
import numpy as np
import time
import json
from typing import Callable

from ..classifier import Classifier
from ..constants import *
from ..counterfactual_generator import CounterfactualGenerator
from ..imputer import Imputer
from ..data_utils import (
    get_cat_indices,
    get_num_indices,
    get_indices_with_missing_values,
    get_target_index,
    get_feature_mads,
)
from .evaluation_metrics import (
    get_average_sparsity,
    get_diversity,
    get_average_distance_from_original,
    get_valid_ratio,
)


class Evaluator:

    def __init__(self, data_pd: pd.DataFrame, config: dict):
        self.config = config
        self.debug = config[config_debug]
        self.target_index = get_target_index(data_pd, config.get(config_target_name))
        self.cat_indices = get_cat_indices(data_pd, self.target_index)
        self.num_indices = get_num_indices(data_pd, self.target_index)
        self.predictor_indices = np.sort(
            np.concatenate([self.cat_indices, self.num_indices])
        ).astype(int)
        self.target_class = self.config[config_target_class]

        if self.config[config_classifier] == config_logistic_regression:
            self.classifier = Classifier(self.num_indices)

        if self.debug:
            print(data_pd.head())

        self.data_pd = data_pd
        self.data = data_pd.copy().to_numpy()

        self.X_train_current = None
        self.indices_with_missing_values_current = None
        self.test_instance_complete_current = None
        self.test_instance_with_missing_values_current = None

        self.cf_generator = CounterfactualGenerator(
            self.classifier, self.target_class, self.debug
        )

        self.metric_names = [
            "n_vectors",
            "valid_ratio",
            "avg_dist_from_original",
            "diversity",
            "avg_sparsity",
            "runtime_seconds",
            "num_missing_values",
        ]

        if self.debug:
            print(f"\ncat_indices: {type(self.cat_indices)} {self.cat_indices}")
            print(f"num_indices: {type(self.num_indices)} {self.num_indices}")
            print(
                f"predictor_indices: {type(self.predictor_indices)} {self.predictor_indices}"
            )
            print(f"target_index: {type(self.target_index)} {self.target_index}\n")

    def _evaluate_counterfactuals(
        self,
        counterfactuals: np.array,
        prediction_func: Callable,
    ) -> dict[str, np.array]:
        """Return evaluation metrics for a set of counterfactuals.

        :param np.array counterfactuals: set of counterfactuals
        :param Callable prediction_func: classifier prediction function
        :return dict[str, np.array]: dict of metrics
        """
        n_cfs = len(counterfactuals)
        valid_ratio = get_valid_ratio(
            counterfactuals, prediction_func, self.target_class
        )
        mads = get_feature_mads(self.X_train_current)
        avg_dist_from_original = get_average_distance_from_original(
            self.test_instance_complete_current, counterfactuals, mads
        )
        diversity = get_diversity(counterfactuals, mads)
        # TODO: calculate sparsity from complete or imputed?
        avg_sparsity = get_average_sparsity(
            self.test_instance_complete_current, counterfactuals
        )
        return {
            "n_vectors": n_cfs,
            "valid_ratio": valid_ratio,
            "avg_dist_from_original": avg_dist_from_original,
            "diversity": diversity,
            "avg_sparsity": avg_sparsity,
        }

    def _introduce_missing_values_to_test_instance(self):
        """Change values in test instance to nan according to specified
        missing data mechanism.

        :return np.array: test instance with missing value(s)
        """
        test_instance_with_missing_values = self.test_instance_complete_current.copy()
        mechanism = self.config[config_missing_data_mechanism]
        if mechanism == config_MCAR:
            test_instance_with_missing_values[3] = np.nan
        elif mechanism == config_MAR:
            # if sulphates is small, volatile acidity is missing
            if self.test_instance_complete_current[9] < 0.9:
                test_instance_with_missing_values[1] = np.nan
        elif mechanism == config_MNAR:
            # if pH is large, pH is missing
            if self.test_instance_complete_current[8] > 3.4:
                test_instance_with_missing_values[8] = np.nan
        else:
            raise RuntimeError(
                f"Invalid missing data mechanism '{mechanism}'; excpected one of MCAR, MAR, MNAR"
            )
        return test_instance_with_missing_values

    def _get_test_instance(self, row_ind: int) -> np.array:
        """Create test instance from data.

        :param int row_ind: row index in data to test
        :return np.array: test instance with target feature removed
        """
        test_instance = self.data[row_ind, :].ravel()
        test_instance = np.delete(test_instance, self.target_index)
        return test_instance

    def _get_X_train_y_train(
        self, data_without_test_instance: np.array
    ) -> tuple[np.array, np.array]:
        """Get X train and y train.

        :param np.array data_without_test_instance: data with test instance row removed
        :return tuple[np.array, np.array]: X_train, y_train
        """
        X_train = data_without_test_instance[:, self.predictor_indices]
        y_train = data_without_test_instance[:, self.target_index].ravel()
        return X_train, y_train

    def _impute_test_instance(self) -> np.array:
        """Impute test instance with missing values using chosen imputation technique.

        :return np.array: test instance with missing values imputed
        """
        imputer = Imputer(self.X_train_current)
        return imputer.mean_imputation(
            self.test_instance_with_missing_values_current,
            self.indices_with_missing_values_current,
        )

    def _get_counterfactuals(self) -> tuple[np.array, float]:
        """Generate counterfactuals.

        :return tuple[np.array, float]: set of alternative vectors and generation wall clock time
        """
        time_start = time.time()
        counterfactuals = self.cf_generator.generate_explanations(
            self.test_instance_with_missing_values_current,
            self.X_train_current,
            self.indices_with_missing_values_current,
            3,
            self.data_pd,
            "DICE",
        )
        time_end = time.time()
        wall_time = time_end - time_start
        return counterfactuals, wall_time

    def _evaluate_single_instance(self, row_ind: int) -> dict[str, float]:
        """Return metrics for counterfactuals generated for single test instance.
        Introduces potential missing values to test instance, imputes them,
        generates counterfactual and evaluates it.

        :param int row_ind: row index of test instance in test data
        :return dict[str, float]: metrics for evaluated instance
        """
        self.test_instance_complete_current = self._get_test_instance(row_ind)
        self.test_instance_with_missing_values_current = (
            self._introduce_missing_values_to_test_instance()
        )
        if self.debug:
            print(
                f"test_instance_with_missing_values: {self.test_instance_with_missing_values_current}"
            )
        self.indices_with_missing_values_current = get_indices_with_missing_values(
            self.test_instance_with_missing_values_current
        )

        test_instance_metrics = {}
        if len(self.indices_with_missing_values_current) == 0:
            test_instance_metrics = {"num_missing_values": 0}
            # Skip rest as no missing values were generated (happens in MAR or MNAR case)
            # Todo: should MCAR also contain vecs with no missing values?
        else:
            data_without_test_instance = np.delete(self.data, row_ind, 0)
            X_train, y_train = self._get_X_train_y_train(data_without_test_instance)
            self.X_train_current = X_train
            self.classifier.train(X_train, y_train)
            test_instance_imputed = self._impute_test_instance()
            # Make prediction
            prediction = self.classifier.predict(test_instance_imputed)
            if prediction != self.target_class:
                # Generate counterfactuals
                counterfactuals, wall_time = self._get_counterfactuals()
                debug_arr = np.vstack(
                    (
                        self.test_instance_complete_current,
                        test_instance_imputed,
                        counterfactuals,
                    )
                )
                debug_df = pd.DataFrame(debug_arr)
                # print("debug_df")
                # print(debug_df)
                # Evaluate generated counterfactuals vs. original vector before introducing missing values
                test_instance_metrics = self._evaluate_counterfactuals(
                    counterfactuals,
                    self.classifier.predict,
                )
                test_instance_metrics["runtime_seconds"] = wall_time
                test_instance_metrics["num_missing_values"] = len(
                    self.indices_with_missing_values_current
                )
                if self.debug:
                    print(
                        f"test_instance_metrics: {json.dumps(test_instance_metrics, indent=2)}"
                    )

        return test_instance_metrics

    def perform_evaluation(self):
        """Evaluate counterfactual generation process for configurations given on init.

        :return dict metrics: dict containing metric arrays for all test instances
        """
        num_rows = len(self.data)
        metrics = {metric: [] for metric in self.metric_names}
        for row_ind in range(num_rows):
            single_instance_metrics = self._evaluate_single_instance(row_ind)
            for metric_name, metric_value in single_instance_metrics.items():
                metrics[metric_name].append(metric_value)
            if row_ind % 100 == 0:
                print(f"Evaluated {row_ind}/{num_rows} rows")
        print(f"Evaluated {num_rows}/{num_rows} rows")
        return metrics
