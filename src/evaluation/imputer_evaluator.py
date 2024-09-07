import pandas as pd
import numpy as np
import time
import json
import sys
import random
from typing import Callable
from sklearn.model_selection import train_test_split
from ..classifiers.classifier_interface import Classifier
from ..counterfactual_generator import CounterfactualGenerator
from ..imputer import Imputer
from ..utils.data_utils import (
    get_feature_mads,
)
from ..hyperparams.hyperparam_optimization import HyperparamOptimizer
from .evaluation_metrics import (
    get_average_sparsity,
    get_diversity,
    get_average_distance_from_original,
    get_count_diversity,
    get_distance,
)


class ImputerEvaluator:
    """Class for evaluating imputation methods."""

    def __init__(self, data: pd.DataFrame, hyperparam_opt: HyperparamOptimizer, debug: bool):
        train_data, test_data = train_test_split(data, test_size=0.3)
        self.imputer = Imputer(
            train_data.to_numpy()[:, :-1], hyperparam_opt, True, debug
        )
        self.test_data = test_data.to_numpy()[:, :-1]
        self.mads = get_feature_mads(train_data.to_numpy()[:, :-1])
        self.debug = debug

    def run_imputer_evaluation(self) -> dict:
        """Run evaluation for all imputation methods.
        For multiple imputation, separate evaluation is ran for
        different numbers of imputations created (n).

        :return dict: dictionary with results
        """
        results_dict = {}
        single_missing_value_results = (
            self.evaluate_mean_imputation_for_single_missing_value()
        )
        results_dict["mean"] = single_missing_value_results
        for n in [1, 10, 50]:
            results_dict.update(
                {
                    f"multiple_{n}": self.evaluate_multiple_imputation_for_single_missing_value(
                        n
                    )
                }
            )
        return results_dict

    def evaluate_mean_imputation_for_single_missing_value(self) -> dict[int, float]:
        """Calculates the mean error (difference between actual and imputed value)
        for each feature when using mean imputation.

        :return dict[int, float]: results dict where key is feature index and value is mean error
        """
        mean_total_errors_per_feature = [0 for x in range(self.test_data.shape[1])]
        for row_ind in range(self.test_data.shape[0]):
            test_input = self.test_data[row_ind, :]
            for i in range(test_input.size):
                test_input_mis = test_input.copy()
                test_input_mis[i] = np.nan
                mean_imputed = self.imputer.mean_imputation(test_input_mis, [i])
                error_mean = abs(mean_imputed[i] - test_input[i])
                mean_total_errors_per_feature[i] += np.mean(error_mean)
        mean_avg_error_per_feature = (
            np.array(mean_total_errors_per_feature) / self.test_data.shape[0]
        )
        return mean_avg_error_per_feature

    def evaluate_multiple_imputation_for_single_missing_value(self, n: int):
        """Calculates the mean error (difference between actual and imputed value)
        for each feature when using multiple imputation.

        :param int n: number of multiply imputed vectors to create
        :return dict[int, float]: results dict where key is feature index and value is mean error
        """
        multiple_total_errors_per_feature = [0 for x in range(self.test_data.shape[1])]
        for row_ind in range(self.test_data.shape[0]):
            test_input = self.test_data[row_ind, :]
            for i in range(test_input.size):
                test_input_mis = test_input.copy()
                test_input_mis[i] = np.nan
                multiple_imputed_arr = self.imputer.multiple_imputation(
                    test_input_mis, [i], n
                )
                errors_multiple = abs(multiple_imputed_arr[:, i] - test_input[i])
                error_multiple_avg = np.mean(errors_multiple)
                multiple_total_errors_per_feature[i] += np.mean(error_multiple_avg)
        multiple_avg_error_per_feature = (
            np.array(multiple_total_errors_per_feature) / self.test_data.shape[0]
        )
        return multiple_avg_error_per_feature

    # def evaluate_mean_imputation_multiple_missing_values(self):
    #    total_errors_per_feature = [0 for 0 in range(self.test_data.shape[1])]
    #    for row_ind in range(self.test_data.shape[0]):
    #        test_input = self.test_data[row_ind, :]
    #        for i in range(test_input.size):
    #            test_input_mis = test_input.copy()
    #            test_input_mis[i] = np.nan
    #            imputed = self.imputer.mean_imputation(test_input_mis, [i])
    #            error = abs(imputed[i], test_input[i])
    #            total_errors_per_feature[i] += error
    #    avg_errors_per_feature = np.array(total_errors_per_feature) / self.test_data.shape[0]
    #    return avg_errors_per_feature


#
# def evaluate_multiple_imputation_multiple_missing_values(self):
#    total_errors_per_feature = [0 for 0 in range(self.test_data.shape[1])]
#    for row_ind in range(self.test_data.shape[0]):
#        test_input = self.test_data[row_ind, :]
#        for i in range(test_input.size):
#            test_input_mis = test_input.copy()
#            test_input_mis[i] = np.nan
#            imputed_arr = self.imputer.multiple_imputation(test_input_mis, [i], 10)
#            errors = imputed_arr[:, i] - test_input[i]
#            total_errors_per_feature[i] += np.mean(errors)
#    avg_errors_per_feature = np.array(total_errors_per_feature) / self.test_data.shape[0]
#    return avg_errors_per_feature
