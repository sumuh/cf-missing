import pandas as pd
import numpy as np


class Evaluator:

    def __init__(self, train_data: np.array):
        # Calculate mean absolute deviation (MAD) for each feature in train data
        self.mads = np.apply_along_axis(self._calculate_mad, 1, train_data)
        print(self.mads)

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
        :return float: _description_
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

    def get_diversity(self, vectors: np.array) -> float:
        """Diversity metric from Mothilal et al. (2020). Quantifies diversity
        within set of vectors.

        :param np.array vectors: counterfactual vectors
        :return float: diversity measure
        """
        sum_distances = 0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sum_distances += self._get_mad_weighted_distance(
                    vectors[i, :], vectors[j, :]
                )
        return sum_distances / (len(vectors) ** 2)

    def get_count_diversity(self, vectors: np.array) -> float:
        """Count diversity metric from Mothilal et al. (2020).

        :param np.array vectors: counterfactual vectors
        :return float: count diversity
        """
        pass
