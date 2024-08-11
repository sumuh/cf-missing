import pandas as pd
import numpy as np

from src.evaluation.evaluation_metrics import _calculate_l0_norm, _calculate_l1_norm, _calculate_l2_norm, get_mad_weighted_distance, get_diversity

def test_when_vectors_are_equal_then_l0_norm_is_zero():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    l0_norm = _calculate_l0_norm(vector_1, vector_2)
    assert l0_norm == 0

def test_when_vectors_have_one_different_feature_then_l0_norm_is_one():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([2, 1, 1, 1])
    l0_norm = _calculate_l0_norm(vector_1, vector_2)
    assert l0_norm == 1

def test_when_vectors_have_two_different_features_then_l0_norm_is_two():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([2, 2, 1, 1])
    l0_norm = _calculate_l0_norm(vector_1, vector_2)
    assert l0_norm == 2

def test_when_vectors_have_one_different_feature_and_difference_is_more_than_one_then_l0_norm_is_one():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([3, 1, 1, 1])
    l0_norm = _calculate_l0_norm(vector_1, vector_2)
    assert l0_norm == 1

def test_when_vectors_are_equal_then_l0_norm_is_zero():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    l0_norm = _calculate_l0_norm(vector_1, vector_2)
    assert l0_norm == 0

def test_when_vectors_are_equal_then_l1_norm_is_zero():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    l1_norm = _calculate_l1_norm(vector_1, vector_2)
    assert l1_norm == 0

def test_when_vectors_l1_norm_is_integer_then_correct_norm_is_returned():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 2])
    l1_norm = _calculate_l1_norm(vector_1, vector_2)
    assert l1_norm == 1

def test_when_vectors_l1_norm_is_fraction_then_correct_norm_is_returned():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1.5])
    l1_norm = _calculate_l1_norm(vector_1, vector_2)
    assert l1_norm == 0.5

def test_when_vectors_are_equal_then_l2_norm_is_zero():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    l2_norm = _calculate_l2_norm(vector_1, vector_2)
    assert l2_norm == 0

def test_when_vectors_l2_norm_is_integer_then_correct_norm_is_returned():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1.5, 1.5, 1.5, 1.5])
    l2_norm = _calculate_l2_norm(vector_1, vector_2)
    assert l2_norm == 1

def test_when_vectors_l2_norm_is_fraction_then_correct_norm_is_returned():
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1.25, 1.25, 1.25, 1.25])
    l2_norm = _calculate_l2_norm(vector_1, vector_2)
    assert l2_norm == 0.5

def test_when_vectors_have_distance_zero_then_mad_weighted_distance_returns_zero():
    mads = np.array([1, 1, 1, 1])
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    dist = get_mad_weighted_distance(vector_1, vector_2, mads)
    assert dist == 0

def test_when_vectors_have_integer_distance_then_mad_weighted_distance_returns_correct_number():
    mads = np.array([1, 1, 1, 1])
    vector_1 = np.array([2, 2, 2, 2])
    vector_2 = np.array([1, 1, 1, 1])
    dist = get_mad_weighted_distance(vector_1, vector_2, mads)
    assert dist == 4

def test_when_vectors_have_fraction_distance_then_get_distance_returns_correct_number():
    mads = np.array([1, 1, 1])
    vector_1 = np.array([0.5, 0.5, 0.5])
    vector_2 = np.array([1, 1, 1])
    dist = get_mad_weighted_distance(vector_1, vector_2, mads)
    assert dist == 1.5

def test_when_vectors_are_equal_diversity_is_zero():
    mads = np.array([1, 1, 1])
    vector_1 = np.array([[1, 1, 1]])
    vector_2 = np.array([[1, 1, 1]])
    vectors = np.concatenate((vector_1, vector_2), axis=0)
    diversity = get_diversity(vectors, mads)
    assert diversity == 0

def test_when_vector_diversity_is_integer_then_correct_diversity_is_returned():
    mads = np.array([1, 1, 1, 1])
    vector_1 = np.array([[1, 1, 1, 1]])
    vector_2 = np.array([[2, 2, 2, 2]])
    vectors = np.concatenate((vector_1, vector_2), axis=0)
    diversity = get_diversity(vectors, mads)
    assert diversity == 1

def test_when_vector_diversity_is_fraction_then_correct_diversity_is_returned():
    mads = np.array([1, 1, 1])
    vector_1 = np.array([[1, 2, 3]])
    vector_2 = np.array([[2, 3, 4]])
    vectors = np.concatenate((vector_1, vector_2), axis=0)
    diversity = get_diversity(vectors, mads)
    assert diversity == 0.75
