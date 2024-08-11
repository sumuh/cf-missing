import pandas as pd
import numpy as np
from itertools import combinations

from .classifier import Classifier
from .libraries.growingspheres import growingspheres as gs
from .imputer import Imputer
from .evaluation.evaluation_metrics import get_mad_weighted_distance, get_diversity
from .data_utils import get_feature_mads


class CounterfactualGenerator:

    def __init__(self, classifier: Classifier, target_class: int, debug: bool):
        self.classifier = classifier
        self.target_class = target_class
        self.debug = debug

    def _get_growing_spheres_counterfactual(self, input: np.array) -> np.array:
        """Returns a counterfactual generated with the GrowingSpheres algorithm.

        :param np.array input: input array
        :return np.array: generated counterfactual
        """
        gs_model = gs.GrowingSpheres(
            obs_to_interprete=np.array([input]),
            prediction_fn=self.classifier.predict,
            target_class=self.target_class,
            caps=None,
            n_in_layer=1000,
            first_radius=0.1,
            dicrease_radius=3.0,  # TODO: hyperparam estimation
            layer_shape="sphere",
            verbose=False,
        )
        return gs_model.find_counterfactual()

    def _selection_loss_function(
        self,
        candidate_counterfactuals: np.array,
        input: np.array,
        mads: np.array,
        lambda_1: float = 0.5,
        lambda_2: float = 1,
    ) -> float:
        dist_sum = 0
        for cf in candidate_counterfactuals:
            dist_sum += get_mad_weighted_distance(cf, input, mads)
        div = get_diversity(candidate_counterfactuals, mads)
        return lambda_1 / len(candidate_counterfactuals) * dist_sum - lambda_2 * div

    def _perform_final_selection(
        self,
        counterfactuals: np.array,
        input: np.array,
        final_set_size: int,
        mads: np.array,
    ):
        best_loss = float("inf")
        best_set = None
        for comb in combinations(counterfactuals, final_set_size):
            current_set = np.array(comb)
            loss = self._selection_loss_function(current_set, input, mads)
            if loss < best_loss:
                best_loss = loss
                best_set = current_set

        return best_set

    def generate_explanations(
        self,
        input: np.array,
        X_train: np.array,
        indices_with_missing_values: np.array,
        n: int,
        method: str = "GS",
    ) -> np.array:
        """Generate explanation for input vector(s).

        :param np.array input: input array
        :param np.array X_train: train data
        :param np.array indices_with_missing_values: indices of columns missing values
        :param int n: number of explanations to generate
        :param str method: counterfactual generation method
        :return np.array: array with n rows
        """
        n = 50
        imputer = Imputer(X_train, True, self.debug)
        mads = get_feature_mads(X_train)
        if self.debug:
            print(f"Input: {input}")
            print(f"Indices with missing values: {indices_with_missing_values}")
        imputed_inputs = imputer.multiple_imputation(
            input, indices_with_missing_values, n
        )
        if self.debug:
            print("Multiply imputed inputs:")
            print(imputed_inputs)
        if method == "GS":
            explanations = np.array(
                [
                    self._get_growing_spheres_counterfactual(imputed_input)
                    for imputed_input in imputed_inputs
                ]
            )
            if self.debug:
                print(f"Explanations: {explanations}")
        else:
            raise NotImplementedError("Only method GS (GrowingSpheres) is implemented!")
        final_set_size = 3
        final_explanations = self._perform_final_selection(
            explanations, input, final_set_size, mads
        )
        if self.debug:
            print(f"Final selections: {final_explanations}")
        return final_explanations
