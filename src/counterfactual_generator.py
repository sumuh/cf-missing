import pandas as pd
import numpy as np
import dice_ml
import time
import sys, os
from itertools import combinations

from .logging.cf_logger import CfLogger
from .hyperparams.hyperparam_optimization import HyperparamOptimizer
from .classifiers.classifier_interface import Classifier
from .classifiers.classifier_sklearn import ClassifierSklearn
from .classifiers.classifier_tensorflow import ClassifierTensorFlow
from .imputer import Imputer
from .evaluation.evaluation_metrics import (
    get_average_distance_from_original,
    get_diversity,
    get_average_sparsity,
)
from .utils.data_utils import get_feature_mads


class CounterfactualGenerator:

    def __init__(
        self,
        classifier: Classifier,
        target_class: int,
        target_variable_name: str,
        hyperparam_opt: HyperparamOptimizer,
        distance_lambda_generation: float,
        diversity_lambda_generation: float,
        sparsity_lambda_generation: float,
        distance_lambda_selection: float,
        diversity_lambda_selection: float,
        sparsity_lambda_selection: float,
        selection_alg: str,
        logger: CfLogger,
    ):
        self.classifier = classifier
        self.target_class = target_class
        self.target_variable_name = target_variable_name
        self.hyperparam_opt = hyperparam_opt
        self.distance_lambda_generation = distance_lambda_generation
        self.diversity_lambda_generation = diversity_lambda_generation
        self.sparsity_lambda_generation = sparsity_lambda_generation
        self.distance_lambda_selection = distance_lambda_selection
        self.diversity_lambda_selection = diversity_lambda_selection
        self.sparsity_lambda_selection = sparsity_lambda_selection
        self.selection_alg = selection_alg
        self.logger = logger

    def block_stderr(self):
        sys.stderr = open(os.devnull, "w")

    def enable_stderr(self):
        sys.stderr = sys.__stderr__

    def _get_dice_counterfactuals(
        self, input: np.array, data: pd.DataFrame, k: int
    ) -> np.array:
        """Returns counterfactuals generated with the DiCE library.

        :param np.array input: input array
        :param pd.DataFrame data: dataset
        :param int k: number of counterfactuals to generate
        :return np.array: generated counterfactuals
        """
        self.logger.log_debug("Getting DiCE counterfactuals")
        self.logger.log_debug(f"input: {input}")
        predictor_col_names = [
            name for name in data.columns.to_list() if name != self.target_variable_name
        ]
        dice_data = dice_ml.Data(
            dataframe=data,
            outcome_name=self.target_variable_name,
            continuous_features=predictor_col_names,
        )
        # Suppress some sklearn(?) progress bar that is being output to stderr for some reason
        self.block_stderr()
        if isinstance(self.classifier.get_classifier(), ClassifierSklearn):
            model = dice_ml.Model(model=self.classifier.get_model(), backend="sklearn")
            exp = dice_ml.Dice(dice_data, model, method="genetic")
            e1 = exp.generate_counterfactuals(
                pd.DataFrame(
                    input.reshape(-1, len(input)), columns=predictor_col_names
                ),
                total_CFs=k,
                desired_class=self.target_class,
                proximity_weight=self.distance_lambda_generation,
                diversity_weight=self.diversity_lambda_generation,
                sparsity_weight=self.sparsity_lambda_generation,
                verbose=False,
            )
        elif isinstance(self.classifier.get_classifier(), ClassifierTensorFlow):
            model = dice_ml.Model(
                model=self.classifier.get_model(), backend="TF2", func="ohe-min-max"
            )
            exp = dice_ml.Dice(dice_data, model, method="gradient")
            e1 = exp.generate_counterfactuals(
                pd.DataFrame(
                    input.reshape(-1, len(input)), columns=predictor_col_names
                ),
                total_CFs=k,
                proximity_weight=self.distance_lambda_generation,
                diversity_weight=self.diversity_lambda_generation,
                desired_class=self.target_class,
                verbose=False,
            )
        else:
            raise RuntimeError(
                f"Expected one of [ClassifierSklearn, ClassifierTensorFlow], got {self.classifier}"
            )
        self.enable_stderr()
        self.logger.log_debug("e1.cf_examples_list[0]")
        self.logger.log_debug(e1.cf_examples_list[0].final_cfs_df_sparse)
        return e1.cf_examples_list[0].final_cfs_df_sparse.to_numpy()

    def _selection_loss_function(
        self,
        candidate_counterfactuals: np.array,
        comparison_vector: np.array,
        mads: np.array,
    ) -> float:
        """Loss function modified from Mothilal et al. (2020)
        for selecting best set of counterfactuals.

        :param np.array candidate_counterfactuals: set of counterfactuals to score
        :param np.array comparison_vector: vector to compare to (complete vector in case of no missing values)
        :param np.array mads: mean absolute deviationd for each predictor
        :return float: value of loss function
        """
        dist = get_average_distance_from_original(
            comparison_vector, candidate_counterfactuals, mads
        )
        div = get_diversity(candidate_counterfactuals, mads)
        sparsity = get_average_sparsity(comparison_vector, candidate_counterfactuals)
        return (
            (self.distance_lambda_selection * dist)
            - (self.diversity_lambda_selection * div)
            - (self.sparsity_lambda_selection * sparsity)
        )

    def _perform_final_selection_naive(
        self,
        counterfactuals: np.array,
        comparison_vector: np.array,
        final_set_size: int,
        mads: np.array,
    ) -> np.array:
        """Select a limited number of counterfactuals based on those
        that minimize the loss function: naive approach that iterates over all possible combinations.

        :param np.array counterfactuals: counterfactuals
        :param np.array comparison_vector: vector to compare to (complete vector in case of no missing values)
        :param int final_set_size: number of counterfactuals to return
        :param np.array mads: mean absolute deviations for each predictor
        :return np.array: set of counterfactuals of size final_set_size
        """
        best_loss = float("inf")
        best_set = None
        if len(counterfactuals) <= final_set_size:
            return counterfactuals
        for comb in combinations(counterfactuals, final_set_size):
            current_set = np.array(comb)
            loss = self._selection_loss_function(current_set, comparison_vector, mads)
            if loss < best_loss:
                best_loss = loss
                best_set = current_set

        return best_set

    def _perform_final_selection_greedy(
        self,
        counterfactuals: np.array,
        comparison_vector: np.array,
        final_set_size: int,
        mads: np.array,
    ) -> np.array:
        """Select a limited number of counterfactuals based on those
        that minimize the loss function: greedy approach.

        :param np.array counterfactuals: counterfactuals
        :param np.array comparison_vector: vector to compare to (complete vector in case of no missing values)
        :param int final_set_size: number of counterfactuals to return
        :param np.array mads: mean absolute deviations for each predictor
        :return np.array: set of counterfactuals of size final_set_size
        """
        if len(counterfactuals) <= final_set_size:
            return counterfactuals
        best_set = np.array([])
        added_cf_indices = set()
        while len(best_set) < final_set_size:
            best_loss = float("inf")
            # Find cf that minimises loss when added to set
            for row_ind in range(len(counterfactuals)):
                if row_ind not in added_cf_indices:
                    candidate = counterfactuals[row_ind, :]
                    if len(best_set) == 0:
                        set_with_candidate = np.array([candidate])
                    else:
                        set_with_candidate = np.vstack((best_set, candidate))

                    loss_with_candidate = self._selection_loss_function(
                        set_with_candidate, comparison_vector, mads
                    )
                    if loss_with_candidate < best_loss:
                        best_loss = loss_with_candidate
                        best_set_with_candidate = set_with_candidate
                        best_cf_ind = row_ind

            best_set = best_set_with_candidate
            added_cf_indices.add(best_cf_ind)
        return best_set

    def _perform_final_selection_greedy_unlimited(
        self,
        counterfactuals: np.array,
        input: np.array,
        mads: np.array,
    ) -> np.array:
        """Select a limited number of counterfactuals based on those
        that minimize the loss function: greedy approach.

        :param np.array counterfactuals: counterfactuals
        :param np.array input: original imputed input
        :param np.array mads: mean absolute deviations for each predictor
        :return np.array: set of counterfactuals of size final_set_size
        """
        best_loss = float("inf")
        best_set = counterfactuals[0, :]
        for row_ind in range(1, len(counterfactuals)):
            candidate = counterfactuals[row_ind, :]
            set_with_candidate = np.vstack((best_set, candidate))
            new_loss = self._selection_loss_function(set_with_candidate, input, mads)
            if new_loss < best_loss:
                best_loss = new_loss
                best_set = set_with_candidate
        return best_set

    def _filter_out_non_valid(self, counterfactuals: np.array) -> np.array:
        """Filter out counterfactuals that are not valid (they don't belong to the target class).

        :param np.array counterfactuals: counterfactuals
        :return np.array: valid counterfactuals
        """
        if self.target_class == 0:
            return counterfactuals[
                counterfactuals[:, -1] < self.classifier.get_threshold()
            ]
        elif self.target_class == 1:
            return counterfactuals[
                counterfactuals[:, -1] >= self.classifier.get_threshold()
            ]

    def _filter_out_duplicates(self, counterfactuals: np.array) -> np.array:
        """Returns unique counterfactuals.

        :param np.array counterfactuals: counterfactuals
        :return np.array: unique counterfactuals
        """
        return np.unique(counterfactuals, axis=0)

    def _perform_imputation(
        self,
        imputer: Imputer,
        imputation_type: str,
        input: np.array,
        indices_with_missing_values: np.array,
        n: int,
    ) -> np.array:
        """Imputes missing values with specified imputation method.

        :param Imputer imputer: Imputer instance
        :param str imputation_type: imputation type e.g. mean, multiple
        :param np.array input: input to impute
        :param np.array indices_with_missing_values: indices that have missing values in input
        :param int n: how many imputations to create if imputation_type is 'multiple'
        :raises RuntimeError: raised if unknown imputation_type
        :return np.array: 1D or 2D array of imputed versions of input
        """
        if imputation_type == "multiple":
            imputed = imputer.multiple_imputation(input, indices_with_missing_values, n)
        elif imputation_type == "mean":
            imputed = np.array(
                [imputer.mean_imputation(input, indices_with_missing_values)]
            )
            imputed = np.repeat(imputed, n, axis=0)
        elif imputation_type == "regression":
            imputed = np.array(
                [imputer.regression_imputation(input, indices_with_missing_values)]
            )
        else:
            raise RuntimeError(
                f"Unexpected imputation type '{imputation_type}', expected one of: ['multiple', 'mean']"
            )
        return imputed

    def _get_explanations(
        self, imputed_inputs: np.array, data_pd: pd.DataFrame, k: int
    ) -> np.array:
        """Returns counterfactual explanations for each input.

        :param np.array imputed_inputs: inputs
        :param pd.DataFrame data_pd: data
        :param int k: how many counterfactuals to generate per input
        :return np.array: counterfactual array of size k * len(imputed_inputs)
        """
        explanations = None
        for imputed_input in imputed_inputs:
            if explanations is None:
                explanations = self._get_dice_counterfactuals(imputed_input, data_pd, k)
            else:
                explanations = np.vstack(
                    (
                        explanations,
                        self._get_dice_counterfactuals(imputed_input, data_pd, k),
                    )
                )
        return explanations

    def generate_explanations(
        self,
        input: np.array,
        X_train: np.array,
        indices_with_missing_values: np.array,
        k: int,
        n: int,
        data_pd: pd.DataFrame,
        imputation_type: str,
    ) -> tuple[np.array, dict[str, float]]:
        """Generate explanation for input vector(s).

        :param np.array input: input array
        :param np.array X_train: train data
        :param np.array indices_with_missing_values: indices of columns missing values
        :param int k: number of explanations to generate
        :param int n: number of imputed vectors to create in multiple imputation
        :param pd.DataFrame data_pd: data
        :param str imputation_type: imputation method name
        :return tuple[np.array, dict[str, float]]: counterfactuals and runtimes of different parts
        """
        imputer = Imputer(X_train, self.hyperparam_opt, True)
        mads = get_feature_mads(X_train)
        if len(indices_with_missing_values) > 0:
            imputation_start = time.time()
            # Evaluate input with missing values
            input_for_explanations = self._perform_imputation(
                imputer, imputation_type, input, indices_with_missing_values, n
            )
            imputation_end = time.time()
            imputation_runtime = imputation_end - imputation_start
            selection_algo_comparison_point = imputer.mean_imputation(
                input, indices_with_missing_values
            )
        else:
            # Evaluate complete input
            input_for_explanations = input.copy()
            imputation_runtime = 0
            selection_algo_comparison_point = input_for_explanations

        counterfactual_generation_start = time.time()
        explanations = self._get_explanations(input_for_explanations, data_pd, k)
        counterfactual_generation_end = time.time()
        counterfactual_generation_runtime = (
            counterfactual_generation_end - counterfactual_generation_start
        )

        filtering_start = time.time()
        valid_explanations = self._filter_out_non_valid(explanations)
        valid_unique_explanations = self._filter_out_duplicates(valid_explanations)
        filtering_end = time.time()
        filtering_runtime = filtering_end - filtering_start

        if len(valid_unique_explanations) == 0:
            return np.array([]), {
                "imputation": imputation_runtime,
                "counterfactual_generation": counterfactual_generation_runtime,
                "filtering": filtering_runtime,
                "selection": 0,
            }

        selection_start = time.time()
        if self.selection_alg == "naive":
            final_explanations = self._perform_final_selection_naive(
                valid_unique_explanations[:, :-1],
                selection_algo_comparison_point,
                k,
                mads,
            )
        elif self.selection_alg == "greedy":
            final_explanations = self._perform_final_selection_greedy(
                valid_unique_explanations[:, :-1],
                selection_algo_comparison_point,
                k,
                mads,
            )
        else:
            raise RuntimeError(f"Invalid selection algorithm {self.selection_alg}")
        selection_end = time.time()
        selection_runtime = selection_end - selection_start

        return final_explanations, {
            "imputation": imputation_runtime,
            "counterfactual_generation": counterfactual_generation_runtime,
            "filtering": filtering_runtime,
            "selection": selection_runtime,
        }
