import pandas as pd
import numpy as np

from src.evaluation.evaluation_metrics import (
    get_l0_norm,
    get_average_sparsity,
    get_distance,
    get_diversity,
    get_average_distance_from_original,
    get_average_sparsity,
    get_count_diversity,
)


def test_when_vectors_are_equal_then_l0_norm_is_zero():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    l0_norm = get_l0_norm(vector_1, vector_2)
    assert l0_norm == 0


def test_when_vectors_have_one_different_feature_then_l0_norm_is_one():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([2, 1, 1, 1])
    l0_norm = get_l0_norm(vector_1, vector_2)
    assert l0_norm == 1


def test_when_vectors_have_two_different_features_then_l0_norm_is_two():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([2, 2, 1, 1])
    l0_norm = get_l0_norm(vector_1, vector_2)
    assert l0_norm == 2


def test_when_vectors_have_one_different_feature_and_difference_is_more_than_one_then_l0_norm_is_one():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([3, 1, 1, 1])
    l0_norm = get_l0_norm(vector_1, vector_2)
    assert l0_norm == 1


def test_when_vectors_are_equal_then_l0_norm_is_zero():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    l0_norm = get_l0_norm(vector_1, vector_2)
    assert l0_norm == 0


def test_when_vectors_have_distance_zero_then_mad_weighted_distance_returns_zero():
    mads = np.array([1, 1, 1, 1])
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    dist = get_distance(vector_1, vector_2, mads)
    assert dist == 0


def test_when_vectors_have_integer_distance_then_mad_weighted_distance_returns_correct_number():
    mads = np.array([1, 1, 1, 1])
    vector_1 = np.array([2, 2, 2, 2])
    vector_2 = np.array([1, 1, 1, 1])
    dist = get_distance(vector_1, vector_2, mads)
    assert dist == 4


def test_when_vectors_have_fraction_distance_then_get_distance_returns_correct_number():
    mads = np.array([1, 1, 1])
    vector_1 = np.array([0.5, 0.5, 0.5])
    vector_2 = np.array([1, 1, 1])
    dist = get_distance(vector_1, vector_2, mads)
    assert dist == 1.5


def test_when_vectors_are_equal_diversity_is_zero():
    mads = np.array([1, 1, 1])
    vector_1 = np.array([[1, 1, 1]])
    vector_2 = np.array([[1, 1, 1]])
    vectors = np.concatenate((vector_1, vector_2), axis=0)
    diversity = get_diversity(vectors, mads)
    assert diversity == 0


def test_when_vector_diversity_is_integer_then_correct_diversity_is_returned():
    mads = np.array([1, 1])
    vector_1 = np.array([[1, 1]])
    vector_2 = np.array([[5, 5]])
    vectors = np.concatenate((vector_1, vector_2), axis=0)
    diversity = get_diversity(vectors, mads)
    assert diversity == 1


def test_when_vector_diversity_is_fraction_then_correct_diversity_is_returned():
    mads = np.array([1, 1])
    vector_1 = np.array([[1, 1]])
    vector_2 = np.array([[2, 2]])
    vectors = np.concatenate((vector_1, vector_2), axis=0)
    diversity = get_diversity(vectors, mads)
    assert diversity == 0.25


def test_average_distance_from_original():
    mads = np.array([1, 1])
    cf_1 = np.array([[1, 1]])
    cf_2 = np.array([[2, 3]])
    vectors = np.concatenate((cf_1, cf_2), axis=0)
    original = np.array([3, 3])
    avg_distance = get_average_distance_from_original(original, vectors, mads)
    assert avg_distance == 1.25


def test_get_average_sparsity():
    cf_1 = np.array([[1, 2]])
    cf_2 = np.array([[2, 3]])
    vectors = np.concatenate((cf_1, cf_2), axis=0)
    original = np.array([1, 1])
    avg_sparsity = get_average_sparsity(original, vectors)
    assert avg_sparsity == 0.25


def test_get_count_diversity():
    cf_1 = np.array([[1, 1]])
    cf_2 = np.array([[1, 2]])
    vectors = np.concatenate((cf_1, cf_2), axis=0)
    count_diversity = get_count_diversity(vectors)
    assert count_diversity == 0.125