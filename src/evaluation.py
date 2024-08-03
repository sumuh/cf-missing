import pandas as pd
import numpy as np
import time
from typing import Callable

from .classifier import Classifier
from .constants import *
from .counterfactual_generator import CounterfactualGenerator
from .imputer import Imputer
from .data_utils import (
    get_cat_indices,
    get_num_indices,
    get_indices_with_missing_values,
    get_target_index,
)


class CounterfactualEvaluator:

    def __init__(self, X_train: np.array):
        # Calculate mean absolute deviation (MAD) for each predictor in train data
        self.mads = np.apply_along_axis(self._calculate_mad, 0, X_train)

    def _calculate_l0_norm(self, vector_1: np.array, vector_2: np.array) -> int:
        """Calculates l0 norm (sparsity). Tells
        how many features have different values between vectors.
        Counterfactuals should have low sparsity.

        :param np.array vector_1: vector 1
        :param np.array vector_2: vector 2
        :return int: l0 norm
        """
        diff = abs(vector_1 - vector_2)
        return np.linalg.norm(diff, ord=0)

    def _calculate_l1_norm(self, vector_1: np.array, vector_2: np.array) -> float:
        """Calculates l1 norm (Manhattan distance).

        :param np.array vector_1: vector 1
        :param np.array vector_2: vector 2
        :return float: l1 norm
        """
        return np.sum(abs(vector_1 - vector_2))

    def _calculate_l2_norm(self, vector_1: np.array, vector_2: np.array) -> float:
        """Calculates l2 norm (Euclidean distance).

        :param np.array vector_1: vector 1
        :param np.array vector_2: vector 2
        :return float: l2 norm
        """
        return np.sqrt(np.sum((vector_1 - vector_2) ** 2))

    def _calculate_mad(self, values: np.array) -> float:
        """Calculates the mean absolute deviation (MAD) of values.

        :param np.array values: vector of values
        :return float: mean absolute deviation
        """
        if len(values) == 0:
            return None
        return np.sum(abs(values - np.mean(values))) / len(values)

    def _get_mad_weighted_distance(
        self, vector_1: np.array, vector_2: np.array
    ) -> float:
        """Calculates distance between two vectors weighted by each feature's
        mean absolute deviation. (Weighted l1 norm)

        :param np.array vector_1: vector 1
        :param np.array vector_2: vector 2
        :return float: MAD weighted distance
        """
        return np.sum(abs(vector_1 - vector_2) / self.mads)

    def get_average_distance_from_original(
        self, original_vector: np.array, cf_vectors: np.array
    ) -> float:
        """Calculate average distance of counterfactual vectors in array
        from original vector.

        :param np.array original_vector: original input vector
        :param np.array cf_vectors: counterfactual vectors
        :return float: average distance of vectors from original
        """
        return np.mean(
            np.apply_along_axis(
                self._get_mad_weighted_distance, 1, cf_vectors, vector_2=original_vector
            )
        )

    def get_diversity(self, cf_vectors: np.array) -> float:
        """Diversity metric from Mothilal et al. (2020). Quantifies diversity
        within set of vectors.

        :param np.array cf_vectors: counterfactual vectors
        :return float: diversity measure
        """
        if len(cf_vectors) == 0:
            return 0
        sum_distances = 0
        for i in range(len(cf_vectors)):
            for j in range(i + 1, len(cf_vectors)):
                sum_distances += self._get_mad_weighted_distance(
                    cf_vectors[i, :], cf_vectors[j, :]
                )
        return sum_distances / (len(cf_vectors) ** 2)

    def get_count_diversity(self, cf_vectors: np.array) -> float:
        """Count diversity metric from Mothilal et al. (2020).

        :param np.array cf_vectors: counterfactual vectors
        :return float: count diversity
        """
        pass

    def get_valid_ratio(
        self, cf_vectors: np.array, prediction_func: Callable, target_class: int
    ) -> float:
        """Return ratio of valid alternative vectors.
        0 = none are valid, 1 = all are valid.

        :param np.array cf_vectors: counterfactual vectors
        :param Callable prediction_func: prediction function that returns a class
        :param int target_class: target class
        :return float: valid ratio
        """
        classes = np.apply_along_axis(prediction_func, 1, cf_vectors)
        n_valid = len(np.where(classes == target_class)[0])
        return n_valid / len(cf_vectors)

    def get_average_sparsity(
        self, original_vector: np.array, cf_vectors: np.array
    ) -> float:
        """Calculate average sparsity (how many features have different
        values between vectors). Counterfactuals should have low sparsity,
        but always more than zero.

        :param np.array original_vector: original input vector
        :param np.array cf_vectors: counterfactual vectors
        :return float: average sparsity
        """
        return np.mean(
            np.apply_along_axis(
                self._calculate_l0_norm, 1, cf_vectors, vector_2=original_vector
            )
        )

    def evaluate_explanation(
        self,
        original_vector: np.array,
        explanation: np.array,
        prediction_func: Callable,
        target_class: int,
    ) -> dict[str, np.array]:
        """Return evaluation metrics for a set of alternative vectors.

        :param np.array original_vector: original input vector with complete values
        :param np.array explanation: set of alternative vectors
        :param Callable prediction_func: classifier prediction function
        :param int target_class: class that counterfactual should predict
        :return dict[str, np.array]: dict of metrics
        """
        n_vectors = len(explanation)
        valid_ratio = self.get_valid_ratio(explanation, prediction_func, target_class)

        avg_dist_from_original = self.get_average_distance_from_original(
            original_vector, explanation
        )
        diversity = self.get_diversity(explanation)
        avg_sparsity = self.get_average_sparsity(original_vector, explanation)
        return {
            "n_vectors": n_vectors,
            "valid_ratio": valid_ratio,
            "avg_dist_from_original": avg_dist_from_original,
            "diversity": diversity,
            "avg_sparsity": avg_sparsity,
        }


class LoocvEvaluator:

    def __init__(self, data: pd.DataFrame, config: dict):
        self.config = config
        self.debug = config[config_debug]
        self.target_index = get_target_index(data, config.get(config_target_name))
        self.cat_indices = get_cat_indices(data, self.target_index)
        self.num_indices = get_num_indices(data, self.target_index)
        self.predictor_indices = np.sort(
            np.concatenate([self.cat_indices, self.num_indices])
        ).astype(int)

        if self.config[config_classifier] == config_logistic_regression:
            self.classifier = Classifier(self.num_indices)

        if self.debug:
            print(data.head())

        self.data = data.to_numpy()

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

    def _introduce_missing_values(self, test_instance: np.array) -> np.array:
        """Change values in test instance to nan according to specified
        missing data mechanism.

        :param np.array test_instance: test instance
        :return np.array: test instance with missing value(s)
        """
        test_instance_with_missing_values = test_instance.copy()
        mechanism = self.config[config_missing_data_mechanism]
        if mechanism == config_MCAR:
            test_instance_with_missing_values[3] = np.nan
        elif mechanism == config_MAR:
            # if sulphates is small, volatile acidity is missing
            if test_instance[9] < 0.9:
                test_instance_with_missing_values[1] = np.nan
        elif mechanism == config_MNAR:
            # if pH is large, pH is missing
            if test_instance[8] > 3.4:
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

    def _impute_test_instance(
        self,
        data_without_test_instance: np.array,
        test_instance: np.array,
        indices_with_missing_values: np.array,
    ) -> np.array:
        """Impute test instance with missing values using chosen imputation technique.

        :param np.array data_without_test_instance: dataset with test instance removed
        :param np.array test_instance: test instance
        :param np.array indices_with_missing_values: indices with missing values in test instance
        :return np.array: test instance with missing values imputed
        """
        imputer = Imputer()
        return imputer.mean_imputation(
            data_without_test_instance,
            test_instance,
            indices_with_missing_values,
        )

    def _get_counterfactual(self, test_instance: np.array) -> tuple[np.array, float]:
        """Generate counterfactual with alternative vectors.

        :param np.array test_instance: original vector
        :return tuple[np.array, float]: set of alternative vectors and generation wall clock time
        """
        cf_generator = CounterfactualGenerator(self.classifier, None, None)
        time_start = time.time()
        counterfactual = cf_generator.generate_explanations(
            test_instance, None, 3, "GS", self.debug
        )
        time_end = time.time()
        wall_time = time_end - time_start
        return counterfactual, wall_time

    def _evaluate_single_instance(self, row_ind: int) -> dict[str, float]:
        """Return metrics for counterfactual generated for single test instance.
        Introduces potential missing values to test instance, imputes them,
        generates counterfactual and evaluates it.

        :param int row_ind: row index of test instance in test data
        :return dict[str, float]: metrics for evaluated instance
        """
        test_instance_complete = self._get_test_instance(row_ind)
        test_instance_with_missing_values = self._introduce_missing_values(
            test_instance_complete
        )
        indices_with_missing_values = get_indices_with_missing_values(
            test_instance_with_missing_values
        )

        test_instance_metrics = {}
        if len(indices_with_missing_values) == 0:
            test_instance_metrics = {"num_missing_values": 0}
            # Skip rest as no missing values were generated (happens in MAR or MNAR case)
            # Todo: should MCAR also contain vecs with no missing values?
        else:
            data_without_test_instance = np.delete(self.data, row_ind, 0)
            X_train, y_train = self._get_X_train_y_train(data_without_test_instance)
            self.classifier.train(X_train, y_train)
            test_instance_imputed = self._impute_test_instance(
                data_without_test_instance,
                test_instance_with_missing_values,
                indices_with_missing_values,
            )
            # Make prediction
            prediction = self.classifier.predict(test_instance_imputed)
            if prediction == 1:
                # Generate counterfactual
                counterfactual, wall_time = self._get_counterfactual(
                    test_instance_imputed
                )
                # Evaluate generated counterfactual vs. original vector before introducing missing values
                evaluator = CounterfactualEvaluator(X_train)
                test_instance_metrics = evaluator.evaluate_explanation(
                    test_instance_complete, counterfactual, self.classifier.predict, 0
                )
                test_instance_metrics["runtime_seconds"] = wall_time
                test_instance_metrics["num_missing_values"] = len(
                    indices_with_missing_values
                )

        return test_instance_metrics

    def perform_loocv_evaluation(self):
        """Evaluate counterfactual generation process for configurations given on init.

        :return dict avg_metrics: average metrics
        """
        metrics = {metric: [] for metric in self.metric_names}
        for row_ind in range(len(self.data)):
            single_instance_metrics = self._evaluate_single_instance(row_ind)
            for metric_name, metric_value in single_instance_metrics.items():
                metrics[metric_name].append(metric_value)
        return metrics
