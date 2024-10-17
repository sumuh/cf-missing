import numpy as np
from typing import Callable


def _get_l0_norm(vector_1: np.array, vector_2: np.array) -> int:
    """Calculates l0 norm of difference of two vectors.

    :param np.array vector_1: vector 1
    :param np.array vector_2: vector 2
    :return int: l0 norm
    """
    diff = abs(vector_1 - vector_2)
    return np.linalg.norm(diff, ord=0)


def _get_l0_norm_normalized_by_number_of_features(
    vector_1: np.array, vector_2: np.array
) -> float:
    """Calculates l0 norm of difference of two vectors.

    :param np.array vector_1: vector 1
    :param np.array vector_2: vector 2
    :return float: l0 norm
    """
    return _get_l0_norm(vector_1, vector_2) / vector_1.size


def _get_distance_normalized_by_mads(
    vector_1: np.array, vector_2: np.array, mads: np.array
) -> float:
    """Calculates distance between two vectors weighted by each feature's
    mean absolute deviation.

    :param np.array vector_1: vector 1
    :param np.array vector_2: vector 2
    :param np.array mads: mean absolute deviations for each feature
    :return float: distance weighted by MADs
    """
    distances = 0
    num_features = vector_1.size
    for i in range(num_features):
        distances += abs(vector_1[i] - vector_2[i]) / mads[i]
    return distances


def get_distance_normalized_by_number_of_features(
    vector_1: np.array, vector_2: np.array, mads: np.array
) -> float:
    """Calculates distance between two vectors weighted by each feature's
    mean absolute deviation and normalized by number of features.

    :param np.array vector_1: vector 1
    :param np.array vector_2: vector 2
    :param np.array mads: mean absolute deviations for each feature
    :return float: distance weighted by number of features (and MADs)
    """
    num_features = vector_1.size
    return _get_distance_normalized_by_mads(vector_1, vector_2, mads) / num_features


def get_average_distance_from_original(
    original_vector: np.array, cf_vectors: np.array, mads: np.array
) -> float:
    """Calculate average distance of counterfactual vectors in array
    from original vector.

    :param np.array original_vector: original input vector
    :param np.array cf_vectors: counterfactual vectors
    :param np.array mads: mean absolute deviations for each feature
    :return float: average distance of vectors from original
    """
    distances = np.apply_along_axis(
        get_distance_normalized_by_number_of_features,
        1,
        cf_vectors,
        vector_2=original_vector,
        mads=mads,
    )
    return np.sum(distances) / len(cf_vectors)


def get_average_sparsity(original_vector: np.array, cf_vectors: np.array) -> float:
    """Calculate average sparsity. Counterfactuals should have high sparsity.

    :param np.array original_vector: original input vector
    :param np.array cf_vectors: counterfactual vectors
    :return float: average sparsity
    """
    sparsities = np.apply_along_axis(
        _get_l0_norm_normalized_by_number_of_features,
        1,
        cf_vectors,
        vector_2=original_vector,
    )
    return 1 - (np.sum(sparsities) / len(cf_vectors))


def get_diversity(cf_vectors: np.array, mads: np.array) -> float:
    """Diversity metric from Mothilal et al. (2020). Quantifies diversity
    within set of vectors.

    :param np.array cf_vectors: counterfactual vectors
    :param np.array mads: mean absolute deviations for each feature
    :return float: diversity normalized by number of vectors
    """
    if len(cf_vectors) < 2:
        return 0
    sum_distances = 0
    count = 0
    for i in range(len(cf_vectors)):
        for j in range(i + 1, len(cf_vectors)):
            count += 1
            sum_distances += get_distance_normalized_by_number_of_features(
                cf_vectors[i, :], cf_vectors[j, :], mads
            )
    return sum_distances / count


def get_count_diversity(cf_vectors: np.array) -> float:
    """Count diversity metric from Mothilal et al. (2020):
    'the fraction of features that are different
    between any two pair of counterfactual examples'

    :param np.array cf_vectors: counterfactual vectors
    :return float: count diversity normalized by number of vectors
    """
    if len(cf_vectors) < 2:
        return 0
    sum_norms = 0
    count = 0
    for i in range(len(cf_vectors)):
        for j in range(i + 1, len(cf_vectors)):
            count += 1
            sum_norms += _get_l0_norm_normalized_by_number_of_features(
                cf_vectors[i, :], cf_vectors[j, :]
            )
    return sum_norms / count
