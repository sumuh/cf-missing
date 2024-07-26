import pandas as pd
import numpy as np
from typing import Callable

from .classifier import Classifier
from .counterfactual_generator import CounterfactualGenerator
from .imputer import Imputer
from .data_utils import (
    get_cat_indices,
    get_num_indices,
    get_indices_with_missing_values,
    get_target_index,
)


class Evaluator:

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
    ) -> tuple:
        """Return evaluation metrics for a set of alternative vectors.

        :param np.array original_vector: original input vector
        :param np.array explanation: set of alternative vectors
        :param Callable prediction_func: classifier prediction function
        :param int target_class: class that counterfactual should predict
        :return tuple: set of metrics
        """
        n_vectors = len(explanation)
        valid_ratio = self.get_valid_ratio(explanation, prediction_func, target_class)
        avg_dist_from_original = self.get_average_distance_from_original(
            original_vector, explanation
        )
        diversity = self.get_diversity(explanation)
        avg_sparsity = self.get_average_sparsity(original_vector, explanation)
        return (n_vectors, valid_ratio, avg_dist_from_original, diversity, avg_sparsity)


def perform_loocv_evaluation(data: pd.DataFrame, config: dict):
    """Evaluate counterfactual generation process for given configurations.

    :param pd.DataFrame data: dataset to use
    :param dict config: config parameters
    """
    # TODO: handle config params and setup accordingly

    print(data.head())

    target_index = get_target_index(data, "Outcome")
    cat_indices = get_cat_indices(data, target_index)
    num_indices = get_num_indices(data, target_index)
    predictor_indices = np.sort(np.concatenate([cat_indices, num_indices])).astype(int)

    print(f"cat_indices: {type(cat_indices)} {cat_indices}")
    print(f"num_indices: {type(num_indices)} {num_indices}")
    print(f"predictor_indices: {type(predictor_indices)} {predictor_indices}")
    print(f"target_index: {type(target_index)} {target_index}")

    classifier = Classifier(num_indices)
    data = data.to_numpy()

    for row_ind in range(len(data)):
        test_instance = data[row_ind, :].ravel()
        test_instance = np.array([np.delete(test_instance, target_index)])

        # TODO: introduce missing values to test_instance
        # with missing data mechanism from config

        # indices_with_missing_values = get_indices_with_missing_values(test_instance)

        # Create train and test datasets
        data_without_test_instance = np.delete(data, row_ind, 0)
        X_train = data_without_test_instance[:, predictor_indices]
        y_train = data_without_test_instance[:, target_index].ravel()
        classifier.train(X_train, y_train)
        cf_generator = CounterfactualGenerator(classifier, None, None)

        # TODO: impute test_instance with missing data
        # imputer = Imputer()
        # test_instance = imputer.mean_imputation(data_without_test_instance, test_instance, indices_with_missing_values, 1)

        prediction = classifier.predict(test_instance)
        if prediction == 1:
            counterfactuals = cf_generator.generate_explanations(
                test_instance, None, None
            )

            if counterfactuals.ndim == 1:
                counterfactuals = np.array([counterfactuals])

            evaluator = Evaluator(X_train)
            metrics = evaluator.evaluate_explanation(
                test_instance[0], counterfactuals, classifier.predict, 0
            )
            print(metrics)
