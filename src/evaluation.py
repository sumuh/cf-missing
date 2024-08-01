import pandas as pd
import numpy as np
import time
import random
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
        imputed_vector: np.array,
        explanation: np.array,
        prediction_func: Callable,
        target_class: int,
        wall_time: float,
    ) -> tuple:
        """Return evaluation metrics for a set of alternative vectors.

        :param np.array original_vector: original input vector with complete values
        :param np.array imputed_vector: input vector with imputed missing value(s)
        :param np.array explanation: set of alternative vectors
        :param Callable prediction_func: classifier prediction function
        :param int target_class: class that counterfactual should predict
        :param float wall_time: wall clock runtime for generating cf
        :return tuple: set of metrics
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
            "runtime_seconds": wall_time,
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
        ]
        self.metrics = {metric: [] for metric in self.metric_names}

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
        mechanism = self.config[config_missing_data_mechanism]
        if mechanism == config_MCAR:
            index = random.randint(0, len(test_instance) - 1)
            test_instance[index] = np.nan
        elif mechanism == config_MAR:
            pass
        elif mechanism == config_MNAR:
            pass
        else:
            raise RuntimeError(
                f"Invalid missing data mechanism '{mechanism}'; excpected one of MCAR, MAR, MNAR"
            )
        return test_instance

    def _get_and_impute_test_instance(
        self, row_ind: int
    ) -> tuple[np.array, np.array, np.array]:
        """Create test instance from data. Introduce missing values according
        to mechanism defined in config.

        :param int row_ind: row index in data to test
        :return tuple[np.array, np.array, np.array]: (original complete test instance,
        test instance with missing values, indices of missing values)
        """
        test_instance = self.data[row_ind, :].ravel()
        test_instance = np.delete(test_instance, self.target_index)

        # Save test instance without missing values for evaluation
        test_instance_complete = test_instance.copy()

        # Artificially create missing value(s) in test instance
        test_instance_with_missing_values = self._introduce_missing_values(
            test_instance
        )

        if self.debug:
            print(f"test_instance after introducing missing values {test_instance}")

        indices_with_missing_values = get_indices_with_missing_values(test_instance)

        return (
            test_instance_complete,
            test_instance_with_missing_values,
            indices_with_missing_values,
        )

    def _train_and_generate(self, row_ind: int, classifier: Classifier):
        """Train model without selected test row and create counterfactuals
        for test row. Store metrics in dict.

        :param int row_ind: row index in data to test
        :param Classifier classifier: classifier
        """

        # Get test instance and copy with missing values introduced
        (
            test_instance_complete,
            test_instance_with_missing_values,
            indices_with_missing_values,
        ) = self._get_and_impute_test_instance(row_ind)

        # Create train and test datasets
        data_without_test_instance = np.delete(self.data, row_ind, 0)
        X_train = data_without_test_instance[:, self.predictor_indices]
        y_train = data_without_test_instance[:, self.target_index].ravel()
        classifier.train(X_train, y_train)
        cf_generator = CounterfactualGenerator(classifier, None, None)

        # Impute missing value(s) in test instance
        imputer = Imputer()
        test_instance_imputed = imputer.mean_imputation(
            data_without_test_instance,
            test_instance_with_missing_values,
            indices_with_missing_values,
        )

        prediction = classifier.predict(test_instance_imputed)
        if prediction == 1:
            time_1 = time.time()
            counterfactuals = cf_generator.generate_explanations(
                test_instance_imputed, None, 3, "GS", self.debug
            )
            time_2 = time.time()
            wall_time = time_2 - time_1

            evaluator = CounterfactualEvaluator(X_train)
            test_instance_metrics = evaluator.evaluate_explanation(
                test_instance_complete,
                test_instance_imputed,
                counterfactuals,
                classifier.predict,
                0,
                wall_time,
            )
            for metric_name in self.metric_names:
                self.metrics.get(metric_name).append(
                    test_instance_metrics.get(metric_name)
                )

    def perform_loocv_evaluation(self):
        """Evaluate counterfactual generation process for configurations given on init.

        :return dict avg_metrics: average metrics
        """

        if self.config[config_classifier] == config_logistic_regression:
            classifier = Classifier(self.num_indices)

        for row_ind in range(len(self.data)):
            self._train_and_generate(row_ind, classifier)

        averages = [
            np.mean(np.array(metric_arr)) for metric_arr in self.metrics.values()
        ]
        final_metric_names = [
            "avg_" + metric_name if not metric_name.startswith("avg_") else metric_name
            for metric_name in self.metric_names
        ]
        avg_metrics = {
            metric_name: avg_metric
            for metric_name, avg_metric in zip(final_metric_names, averages)
        }
        return avg_metrics
