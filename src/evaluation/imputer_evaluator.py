import pandas as pd
import numpy as np
from typing import Union
from sklearn.model_selection import train_test_split
from ..imputer import Imputer
from ..utils.data_utils import (
    get_feature_mads,
)
from ..hyperparams.hyperparam_optimization import HyperparamOptimizer


class ImputerEvaluator:
    """Class for evaluating imputation methods."""

    def __init__(
        self, data: pd.DataFrame, hyperparam_opt: HyperparamOptimizer, debug: bool
    ):
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=12)
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
        print("Running imputer evaluation.")
        results_dict = {}
        single_missing_value_results = (
            self.evaluate_mean_imputation_for_single_missing_value()
        )
        results_dict["avg_errors"] = {}
        results_dict["avg_errors"]["mean"] = single_missing_value_results
        for n in [1, 5, 50]:
            print(f"n: {n}")
            results_dict["avg_errors"].update(
                {
                    f"multiple_{n}": self.evaluate_multiple_imputation_for_single_missing_value(
                        n
                    )
                }
            )
        for n in [100]:
            samples_distribution = (
                self.evaluate_multiple_imputation_samples_distribution(n)
            )
            results_dict.update({f"samples_distribution_{n}": samples_distribution})
        # print(json.dumps(results_dict, indent=2))
        return results_dict

    def evaluate_mean_imputation_for_single_missing_value(self) -> list[float]:
        """Calculates the mean error (difference between actual and imputed value)
        for each feature when using mean imputation.

        :return list[float]: average error for each feature
        """
        mean_total_errors_per_feature = [0 for x in range(self.test_data.shape[1])]
        for row_ind in range(self.test_data.shape[0]):
            test_input = self.test_data[row_ind, :]
            for i in range(test_input.size):
                test_input_mis = test_input.copy()
                test_input_mis[i] = np.nan
                mean_imputed = self.imputer.mean_imputation(test_input_mis, [i])
                error_mean = abs(mean_imputed[i] - test_input[i]) / self.mads[i]
                mean_total_errors_per_feature[i] += np.mean(error_mean)
        mean_avg_error_per_feature = (
            np.array(mean_total_errors_per_feature) / self.test_data.shape[0]
        )
        return mean_avg_error_per_feature

    def evaluate_multiple_imputation_for_single_missing_value(
        self, n: int
    ) -> list[float]:
        """Calculates the mean error (difference between actual and imputed value)
        for each feature when using multiple imputation.

        :param int n: number of multiply imputed vectors to create
        :return list[float]: average error for each feature
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
                errors_multiple = (
                    abs(multiple_imputed_arr[:, i] - test_input[i]) / self.mads[i]
                )
                multiple_total_errors_per_feature[i] += np.mean(errors_multiple)
        multiple_avg_error_per_feature = (
            np.array(multiple_total_errors_per_feature) / self.test_data.shape[0]
        )
        return multiple_avg_error_per_feature

    def evaluate_multiple_imputation_samples_distribution(
        self, n: int
    ) -> dict[int, dict[str, Union[float, np.array]]]:
        """Evaluates how sampled values are distributed around mean and std of distribution.

        :param int n: number of samples
        :return dict[int, dict[str, Union[float, np.array]]]: results, key is feature index and dict has values for mu, sigma and samples
        """
        results = {}
        test_input = self.test_data[0, :]
        for i in range(test_input.size):
            test_input_mis = test_input.copy()
            test_input_mis[i] = np.nan
            mu, sigma = self.imputer._get_fcs_model_predicted_mu_and_sigma_for_feat(
                i, test_input_mis
            )
            samples = self.imputer.multiple_imputation(test_input_mis, [i], n)[:, i]
            results[i] = {}
            results[i]["mu"] = mu
            results[i]["sigma"] = sigma
            results[i]["samples"] = samples
            a, b = self.imputer._get_feat_min_max(i)
            results[i]["a"] = a
            results[i]["b"] = b
        return results

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
