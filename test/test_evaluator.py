import pandas as pd
import numpy as np

from src.evaluation import Evaluator

def test_when_mad_is_integer_then_correct_value_calculated():
    evaluator = Evaluator(np.array([]))
    arr = np.array([1, 1, 3, 3])
    mad = evaluator._calculate_mad(arr)
    assert mad == 1

def test_when_mad_is_fraction_then_correct_value_calculated():
    evaluator = Evaluator(np.array([]))
    arr = np.array([1, 1, 2, 2])
    mad = evaluator._calculate_mad(arr)
    assert mad == 0.5

def test_when_mad_is_zero_then_correct_value_calculated():
    evaluator = Evaluator(np.array([]))
    arr = np.array([1, 1, 1, 1])
    mad = evaluator._calculate_mad(arr)
    assert mad == 0

def test_when_input_to_mad_is_empty_then_None_returned():
    evaluator = Evaluator(np.array([]))
    arr = np.array([])
    mad = evaluator._calculate_mad(arr)
    assert mad == None

def test_when_vectors_are_equal_then_l0_norm_is_zero():
    evaluator = Evaluator(np.array([]))
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    l0_norm = evaluator._calculate_l0_norm(vector_1, vector_2)
    assert l0_norm == 0

def test_when_vectors_have_one_different_feature_then_l0_norm_is_one():
    evaluator = Evaluator(np.array([]))
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([2, 1, 1, 1])
    l0_norm = evaluator._calculate_l0_norm(vector_1, vector_2)
    assert l0_norm == 1

def test_when_vectors_have_two_different_features_then_l0_norm_is_two():
    evaluator = Evaluator(np.array([]))
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([2, 2, 1, 1])
    l0_norm = evaluator._calculate_l0_norm(vector_1, vector_2)
    assert l0_norm == 2

def test_when_vectors_have_one_different_feature_and_difference_is_more_than_one_then_l0_norm_is_one():
    evaluator = Evaluator(np.array([]))
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([3, 1, 1, 1])
    l0_norm = evaluator._calculate_l0_norm(vector_1, vector_2)
    assert l0_norm == 1

def test_when_vectors_are_equal_then_l0_norm_is_zero():
    evaluator = Evaluator(np.array([]))
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    l0_norm = evaluator._calculate_l0_norm(vector_1, vector_2)
    assert l0_norm == 0

def test_when_vectors_are_equal_then_l1_norm_is_zero():
    evaluator = Evaluator(np.array([]))
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    l1_norm = evaluator._calculate_l1_norm(vector_1, vector_2)
    assert l1_norm == 0

def test_when_vectors_l1_norm_is_integer_then_correct_norm_is_returned():
    evaluator = Evaluator(np.array([]))
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 2])
    l1_norm = evaluator._calculate_l1_norm(vector_1, vector_2)
    assert l1_norm == 1

def test_when_vectors_l1_norm_is_fraction_then_correct_norm_is_returned():
    evaluator = Evaluator(np.array([]))
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1.5])
    l1_norm = evaluator._calculate_l1_norm(vector_1, vector_2)
    assert l1_norm == 0.5

def test_when_vectors_are_equal_then_l2_norm_is_zero():
    evaluator = Evaluator(np.array([]))
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    l2_norm = evaluator._calculate_l2_norm(vector_1, vector_2)
    assert l2_norm == 0

def test_when_vectors_l2_norm_is_integer_then_correct_norm_is_returned():
    evaluator = Evaluator(np.array([]))
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1.5, 1.5, 1.5, 1.5])
    l2_norm = evaluator._calculate_l2_norm(vector_1, vector_2)
    assert l2_norm == 1

def test_when_vectors_l2_norm_is_fraction_then_correct_norm_is_returned():
    evaluator = Evaluator(np.array([]))
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1.25, 1.25, 1.25, 1.25])
    l2_norm = evaluator._calculate_l2_norm(vector_1, vector_2)
    assert l2_norm == 0.5

def test_when_vectors_have_distance_zero_then_mad_weighted_distance_returns_zero():
    evaluator = Evaluator(np.array([]))
    evaluator.mads = np.array([1, 1, 1, 1])
    vector_1 = np.array([1, 1, 1, 1])
    vector_2 = np.array([1, 1, 1, 1])
    dist = evaluator._get_mad_weighted_distance(vector_1, vector_2)
    assert dist == 0

def test_when_vectors_have_integer_distance_then_mad_weighted_distance_returns_correct_number():
    evaluator = Evaluator(np.array([]))
    evaluator.mads = np.array([1, 1, 1, 1])
    vector_1 = np.array([2, 2, 2, 2])
    vector_2 = np.array([1, 1, 1, 1])
    dist = evaluator._get_mad_weighted_distance(vector_1, vector_2)
    assert dist == 4

def test_when_vectors_have_fraction_distance_then_get_distance_returns_correct_number():
    evaluator = Evaluator(np.array([]))
    evaluator.mads = np.array([1, 1, 1])
    vector_1 = np.array([0.5, 0.5, 0.5])
    vector_2 = np.array([1, 1, 1])
    dist = evaluator._get_mad_weighted_distance(vector_1, vector_2)
    assert dist == 1.5

def test_when_vectors_are_equal_diversity_is_zero():
    evaluator = Evaluator(np.array([]))
    evaluator.mads = np.array([1, 1, 1])
    vector_1 = np.array([[1, 1, 1]])
    vector_2 = np.array([[1, 1, 1]])
    vectors = np.concatenate((vector_1, vector_2), axis=0)
    diversity = evaluator.get_diversity(vectors)
    assert diversity == 0

def test_when_vector_diversity_is_integer_then_correct_diversity_is_returned():
    evaluator = Evaluator(np.array([]))
    evaluator.mads = np.array([1, 1, 1, 1])
    vector_1 = np.array([[1, 1, 1, 1]])
    vector_2 = np.array([[2, 2, 2, 2]])
    vectors = np.concatenate((vector_1, vector_2), axis=0)
    diversity = evaluator.get_diversity(vectors)
    assert diversity == 1

def test_when_vector_diversity_is_fraction_then_correct_diversity_is_returned():
    evaluator = Evaluator(np.array([]))
    evaluator.mads = np.array([1, 1, 1])
    vector_1 = np.array([[1, 2, 3]])
    vector_2 = np.array([[2, 3, 4]])
    vectors = np.concatenate((vector_1, vector_2), axis=0)
    diversity = evaluator.get_diversity(vectors)
    assert diversity == 0.75
