import pandas as pd
import numpy as np
from typing import Callable


def _calculate_l0_norm(vector_1: np.array, vector_2: np.array) -> int:
    """Calculates l0 norm (sparsity). Tells
    how many features have different values between vectors.
    Counterfactuals should have low sparsity.

    :param np.array vector_1: vector 1
    :param np.array vector_2: vector 2
    :return int: l0 norm
    """
    diff = abs(vector_1 - vector_2)
    return np.linalg.norm(diff, ord=0)


def get_distance(vector_1: np.array, vector_2: np.array, mads: np.array) -> float:
    """Calculates distance between two vectors weighted by each feature's
    mean absolute deviation and normalized by number of features. (Weighted l1 norm)

    :param np.array vector_1: vector 1
    :param np.array vector_2: vector 2
    :param np.array mads: mean absolute deviations for each feature
    :return float: MAD weighted distance
    """
    distances = 0
    num_features = vector_1.size
    for i in range(num_features):
        distances += abs(vector_1[i] - vector_2[i]) / mads[i]
    return distances


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
    num_features = cf_vectors.shape[1]
    distances = np.apply_along_axis(
        get_distance,
        1,
        cf_vectors,
        vector_2=original_vector,
        mads=mads,
    )
    return (1 / (len(cf_vectors) * num_features)) * np.sum(distances)


def get_average_sparsity(original_vector: np.array, cf_vectors: np.array) -> float:
    """Calculate average sparsity (how many features have different
    values between vectors). Counterfactuals should have low sparsity,
    but always more than zero.

    :param np.array original_vector: original input vector
    :param np.array cf_vectors: counterfactual vectors
    :return float: average sparsity
    """
    num_features = cf_vectors.shape[1]
    sparsities = np.apply_along_axis(
        _calculate_l0_norm, 1, cf_vectors, vector_2=original_vector
    )
    return (1 / (len(cf_vectors) * num_features)) * np.sum(sparsities)


def get_diversity(cf_vectors: np.array, mads: np.array) -> float:
    """Diversity metric from Mothilal et al. (2020). Quantifies diversity
    within set of vectors.

    :param np.array cf_vectors: counterfactual vectors
    :param np.array mads: mean absolute deviations for each feature
    :return float: diversity measure
    """
    if len(cf_vectors) == 0:
        return 0
    num_features = cf_vectors.shape[1]
    sum_distances = 0
    for i in range(len(cf_vectors)):
        for j in range(i + 1, len(cf_vectors)):
            sum_distances += get_distance(cf_vectors[i, :], cf_vectors[j, :], mads)
    return 1 / ((len(cf_vectors) ** 2) * num_features) * sum_distances


def get_count_diversity(cf_vectors: np.array) -> float:
    """Count diversity metric from Mothilal et al. (2020):
    'the fraction of features that are different
    between any two pair of counterfactual examples'

    :param np.array cf_vectors: counterfactual vectors
    :return float: count diversity
    """
    if len(cf_vectors) == 0:
        return 0
    sum_norms = 0
    num_features = cf_vectors.shape[1]
    for i in range(len(cf_vectors)):
        for j in range(i + 1, len(cf_vectors)):
            sum_norms += _calculate_l0_norm(cf_vectors[i, :], cf_vectors[j, :])
    return 1 / (len(cf_vectors) ** 2 * num_features) * sum_norms


def get_valid_ratio(
    cf_vectors: np.array, prediction_func: Callable, target_class: int
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
