import pandas as pd
import numpy as np
import dice_ml
import sys, os
from itertools import combinations

from .classifiers.classifier_interface import Classifier
from .classifiers.classifier_sklearn import ClassifierSklearn
from .classifiers.classifier_tensorflow import ClassifierTensorFlow
from .imputer import Imputer
from .evaluation.evaluation_metrics import get_distance, get_diversity
from .data_utils import get_feature_mads


class CounterfactualGenerator:

    def __init__(
        self,
        classifier: Classifier,
        target_class: int,
        target_variable_name: str,
        debug: bool,
    ):
        self.classifier = classifier
        self.target_class = target_class
        self.target_variable_name = target_variable_name
        self.debug = debug

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
        if self.debug:
            print("Getting DiCE counterfactuals")
            print(f"input: {input}")
        predictor_col_names = [
            name for name in data.columns.to_list() if name != self.target_variable_name
        ]
        dice_data = dice_ml.Data(
            dataframe=data,
            outcome_name=self.target_variable_name,
            continuous_features=predictor_col_names,
        )
        if isinstance(self.classifier.get_classifier(), ClassifierSklearn):
            model = dice_ml.Model(model=self.classifier.get_model(), backend="sklearn")
            exp = dice_ml.Dice(dice_data, model, method="genetic")
        elif isinstance(self.classifier.get_classifier(), ClassifierTensorFlow):
            model = dice_ml.Model(
                model=self.classifier.get_model(), backend="TF2", func="ohe-min-max"
            )
            exp = dice_ml.Dice(dice_data, model, method="gradient")
        else:
            raise RuntimeError(
                f"Expected one of [ClassifierSklearn, ClassifierTensorFlow], got {self.classifier}"
            )
        # Suppress some sklearn(?) progress bar that is being output to stderr for some reason
        self.block_stderr()
        e1 = exp.generate_counterfactuals(
            pd.DataFrame(input.reshape(-1, len(input)), columns=predictor_col_names),
            total_CFs=k,
            desired_class=self.target_class,
            verbose=False,
        )
        self.enable_stderr()
        return e1.cf_examples_list[0].final_cfs_df.to_numpy()

    def _selection_loss_function(
        self,
        candidate_counterfactuals: np.array,
        input: np.array,
        mads: np.array,
        lambda_1: float = 0.5,
        lambda_2: float = 1,
    ) -> float:
        """Loss function modified from Mothilal et al. (2020)
        for selecting best set of counterfactuals.

        :param np.array candidate_counterfactuals: set of counterfactuals to score
        :param np.array input: original imputed input
        :param np.array mads: mean absolute deviationd for each predictor
        :param float lambda_1: hyperparam, defaults to 0.5
        :param float lambda_2: hyperparam, defaults to 1
        :return float: value of loss function
        """
        dist_sum = 0
        for cf in candidate_counterfactuals:
            dist_sum += get_distance(cf, input, mads)
        div = get_diversity(candidate_counterfactuals, mads)
        return lambda_1 / len(candidate_counterfactuals) * dist_sum - lambda_2 * div

    def _perform_final_selection(
        self,
        counterfactuals: np.array,
        input: np.array,
        final_set_size: int,
        mads: np.array,
    ) -> np.array:
        """Select a limited number of counterfactuals based on those
        that minimize the loss function.

        :param np.array counterfactuals: counterfactuals
        :param np.array input: original imputed input
        :param int final_set_size: number of counterfactuals to return
        :param np.array mads: mean absolute deviations for each predictor
        :return np.array: set of counterfactuals of size final_set_size
        """
        best_loss = float("inf")
        best_set = None
        for comb in combinations(counterfactuals, final_set_size):
            current_set = np.array(comb)
            loss = self._selection_loss_function(current_set, input, mads)
            if loss < best_loss:
                best_loss = loss
                best_set = current_set

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

    def generate_explanations(
        self,
        input: np.array,
        X_train: np.array,
        indices_with_missing_values: np.array,
        k: int,
        n: int,
        data_pd: pd.DataFrame,
    ) -> np.array:
        """Generate explanation for input vector(s).

        :param np.array input: input array
        :param np.array X_train: train data
        :param np.array indices_with_missing_values: indices of columns missing values
        :param int k: number of explanations to generate
        :param int n: number of imputed vectors to create in multiple imputation
        :param str method: counterfactual generation method
        :return np.array: array with n rows
        """
        imputer = Imputer(X_train, True, self.debug)
        mads = get_feature_mads(X_train)
        imputed_inputs = imputer.multiple_imputation(
            input, indices_with_missing_values, n
        )
        if self.debug:
            print("Multiply imputed inputs:")
            print(imputed_inputs)
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

        if self.debug:
            print(f"All k*n explanations:")
            print(pd.DataFrame(explanations))
        valid_explanations = self._filter_out_non_valid(explanations)
        valid_unique_explanations = self._filter_out_duplicates(valid_explanations)
        if len(valid_unique_explanations) == 0:
            return np.array([])
        final_explanations = self._perform_final_selection(
            valid_unique_explanations[:, :-1], input, k, mads
        )
        if self.debug:
            print(f"Final k selections:")
            print(pd.DataFrame(final_explanations))
        return final_explanations
