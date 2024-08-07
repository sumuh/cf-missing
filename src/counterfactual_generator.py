import pandas as pd
import numpy as np

from .classifier import Classifier
from .libraries.growingspheres import growingspheres as gs
from .imputer import Imputer


class CounterfactualGenerator:

    def __init__(self, classifier: Classifier, target_class: int, debug: bool):
        self.classifier = classifier
        self.target_class = target_class
        self.debug = debug

    def get_growing_spheres_counterfactuals(self, input: np.array) -> np.array:
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
            verbose=self.debug,
        )
        return gs_model.find_counterfactual()

    def generate_explanations(
        self,
        input: np.array,
        train_data_without_target: np.array,
        indices_with_missing_values: np.array,
        n: int,
        method: str = "GS",
    ) -> np.array:
        """Generate explanation for input vector(s).

        :param np.array input: input array
        :param np.array train_data_without_target: train data without target column
        :param np.array indices_with_missing_values: indices of columns missing values
        :param int n: number of explanations to generate
        :param str method: counterfactual generation method
        :return np.array: array with n rows
        """
        imputer = Imputer(train_data_without_target, True, self.debug)
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
                    self.get_growing_spheres_counterfactuals(imputed_input)
                    for imputed_input in imputed_inputs
                ]
            )
            if self.debug:
                print(f"Explanations: {explanations}")
        else:
            raise NotImplementedError("Only method GS (GrowingSpheres) is implemented!")
        return explanations
