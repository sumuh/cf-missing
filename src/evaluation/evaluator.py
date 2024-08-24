import pandas as pd
import numpy as np
import time
import json
import sys
import random
from typing import Callable

from ..classifiers.classifier_interface import Classifier
from ..constants import *
from ..counterfactual_generator import CounterfactualGenerator
from ..imputer import Imputer
from ..data_utils import (
    get_cat_indices,
    get_num_indices,
    get_indices_with_missing_values,
    get_target_index,
    get_feature_mads,
    get_averages_from_dict_of_arrays,
)
from .evaluation_metrics import (
    get_average_sparsity,
    get_diversity,
    get_average_distance_from_original,
    get_valid_ratio,
    get_count_diversity,
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
        self.classifier = Classifier(self.config[config_classifier], self.num_indices)

        if self.debug:
            print(data_pd.head())

        self.data_pd = data_pd
        self.data = data_pd.copy().to_numpy()

        self.X_train_current = None
        self.indices_with_missing_values_current = None
        self.test_instance_complete_current = None
        self.test_instance_with_missing_values_current = None

        self.cf_generator = CounterfactualGenerator(
            self.classifier,
            self.target_class,
            config.get(config_target_name),
            self.debug,
        )

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
        n_unique_cfs = np.unique(counterfactuals, axis=0).shape[0]
        valid_ratio = get_valid_ratio(
            counterfactuals, prediction_func, self.target_class
        )
        mads = get_feature_mads(self.X_train_current)
        avg_dist_from_original = get_average_distance_from_original(
            self.test_instance_complete_current, counterfactuals, mads
        )
        diversity = get_diversity(counterfactuals, mads)
        count_diversity = get_count_diversity(counterfactuals)
        diversity_missing_values = get_diversity(
            counterfactuals[:, self.indices_with_missing_values_current],
            mads[self.indices_with_missing_values_current],
        )
        count_diversity_missing_values = get_count_diversity(
            counterfactuals[:, self.indices_with_missing_values_current]
        )
        # Calculate sparsity from imputed test instance
        avg_sparsity = get_average_sparsity(
            self.test_instance_complete_current, counterfactuals
        )
        return {
            "n_vectors": n_cfs,
            "n_unique_vectors": n_unique_cfs,
            "valid_ratio": valid_ratio,
            "avg_dist_from_original": avg_dist_from_original,
            "diversity": diversity,
            "count_diversity": count_diversity,
            "diversity_missing_values": diversity_missing_values,
            "count_diversity_missing_values": count_diversity_missing_values,
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
            ind_missing = random.sample(
                range(0, len(test_instance_with_missing_values)),
                self.config[config_number_of_missing_values],
            )
        elif mechanism == config_MAR:
            # if pedigree fun is below 0.5, insuling is missing
            if self.test_instance_complete_current[6] < 0.5:
                test_instance_with_missing_values[4] = np.nan
        elif mechanism == config_MNAR:
            # if BMI is over 30, BMI is missing
            if self.test_instance_complete_current[5] > 30:
                test_instance_with_missing_values[5] = np.nan
        else:
            raise RuntimeError(
                f"Invalid missing data mechanism '{mechanism}'; excpected one of MCAR, MAR, MNAR"
            )
        if ind_missing is not None:
            test_instance_with_missing_values[ind_missing] = np.nan
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
            self.test_instance_imputed_current,
            self.X_train_current,
            self.indices_with_missing_values_current,
            3,
            self.data_pd,
            "DiCE",
        )
        time_end = time.time()
        wall_time = time_end - time_start
        return counterfactuals, wall_time

    def _get_example_df(self, counterfactuals: np.array) -> pd.DataFrame:
        """Creates dataframe from one test instance and the counterfactuals
         generated for it. For qualitative evaluation of counterfactuals.

        :param np.array counterfactuals: counterfactuals
        :return pd.DataFrame: example df
        """
        example_df = np.vstack(
            (
                self.test_instance_complete_current,
                self.test_instance_with_missing_values_current,
                self.test_instance_imputed_current,
                counterfactuals,
            )
        )
        index = ["complete input", "input with missing", "imputed input"]
        for _ in range(len(counterfactuals)):
            index.append("counterfactual")
        return pd.DataFrame(example_df, index=index)

    def _evaluate_single_instance(
        self, row_ind: int, return_example: bool
    ) -> tuple[dict[str, float], pd.DataFrame]:
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
        example_df = None
        if len(self.indices_with_missing_values_current) == 0:
            test_instance_metrics = {"num_missing_values": 0}
            # Skip rest as no missing values were generated (happens in MAR or MNAR case)
            # Todo: should MCAR also contain vecs with no missing values?
        else:
            data_without_test_instance = np.delete(self.data, row_ind, 0)
            X_train, y_train = self._get_X_train_y_train(data_without_test_instance)
            self.X_train_current = X_train
            self.classifier.train(X_train, y_train)
            self.test_instance_imputed_current = self._impute_test_instance()
            # Make prediction
            prediction = self.classifier.predict(self.test_instance_imputed_current)
            if prediction != self.target_class:
                # Generate counterfactuals
                counterfactuals, wall_time = self._get_counterfactuals()
                if return_example:
                    example_df = self._get_example_df(counterfactuals)
                if len(counterfactuals) > 0:
                    # Evaluate generated counterfactuals vs. original vector before introducing missing values
                    test_instance_metrics = self._evaluate_counterfactuals(
                        counterfactuals,
                        self.classifier.predict,
                    )
                else:
                    test_instance_metrics["no_cf_found"] = 1
                test_instance_metrics["runtime_seconds"] = wall_time
                test_instance_metrics["total_undesired_class"] = 1
                test_instance_metrics["num_missing_values"] = len(
                    self.indices_with_missing_values_current
                )
                if self.debug:
                    print(
                        f"test_instance_metrics: {json.dumps(test_instance_metrics, indent=2)}"
                    )

        return (test_instance_metrics, example_df)

    def aggregate_results(self, metrics_to_average, sum_metrics):
        final_dict = get_averages_from_dict_of_arrays(metrics_to_average)
        # round
        for metric_name, metric in final_dict.items():
            final_dict[metric_name] = round(metric, 3)
        for metric_name, metric in sum_metrics.items():
            final_dict[metric_name] = metric
        return final_dict

    def perform_evaluation(self):
        """Evaluate counterfactual generation process for configurations given on init.

        :return dict metrics: dict containing metric arrays for all test instances
        """
        metrics_to_average_names = [
            "n_vectors",
            "n_unique_vectors",
            "valid_ratio",
            "avg_dist_from_original",
            "diversity",
            "count_diversity",
            "diversity_missing_values",
            "count_diversity_missing_values",
            "avg_sparsity",
            "runtime_seconds",
            "num_missing_values",
        ]
        metrics_to_sum_names = ["total_undesired_class", "no_cf_found"]
        metrics_for_hist_names = [
            "avg_dist_from_original",
            "diversity",
            "count_diversity",
            "diversity_missing_values",
            "count_diversity_missing_values",
            "avg_sparsity",
            "runtime_seconds",
        ]

        num_rows = len(self.data)
        metrics_to_average = {metric: [] for metric in metrics_to_average_names}
        sum_metrics = {metric: 0 for metric in metrics_to_sum_names}

        if self.debug:
            show_example = True
        else:
            show_example = False
        for row_ind in range(num_rows):
            show_example = row_ind % 100 == 0 or show_example
            result = self._evaluate_single_instance(row_ind, show_example)
            single_instance_metrics = result[0]
            for metric_name, metric_value in single_instance_metrics.items():
                if metric_name in metrics_to_average_names:
                    metrics_to_average[metric_name].append(metric_value)
                else:
                    sum_metrics[metric_name] += metric_value

            if row_ind % 100 == 0:
                print(f"Evaluated {row_ind}/{num_rows} rows")
            # Every 100 rows find example to show
            if show_example:
                if result[1] is not None:
                    print(result[1])
                    if not self.debug:
                        show_example = False
                    print(json.dumps(single_instance_metrics, indent=2))
            if self.debug:
                s = "~"
                for _ in range(4):
                    print(f"{(s*20)}")

        print(f"Evaluated {num_rows}/{num_rows} rows")
        histogram_dict = {
            metric_name: metric_value
            for metric_name, metric_value in metrics_to_average.items()
            if metric_name in metrics_for_hist_names
        }
        aggregated_results = self.aggregate_results(metrics_to_average, sum_metrics)
        return histogram_dict, aggregated_results
