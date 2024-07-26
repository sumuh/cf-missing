import pandas as pd
import numpy as np

from .classifier import Classifier
from .sampler import Sampler
from .libraries.growingspheres import growingspheres as gs


class CounterfactualGenerator:

    def __init__(
        self, classifier: Classifier, train_data: np.array, target_feature_index: int
    ):
        self.classifier = classifier
        self.train_data = train_data
        self.target_feature_index = target_feature_index

    def get_growing_spheres_counterfactual(self, input: np.array) -> np.array:
        gs_model = gs.GrowingSpheres(
            obs_to_interprete=np.array([input]),
            prediction_fn=self.classifier.predict,
            target_class=0,
            caps=[0.0, 200.0],
            n_in_layer=1000,
            first_radius=5.0,
            dicrease_radius=5.0,
            layer_shape="sphere",
            # verbose=True,
        )
        return gs_model.find_counterfactual()

    def get_counterfactuals(self, input: np.array, method: str) -> np.array:
        if method == "GS":
            explanation = self.get_growing_spheres_counterfactual(input)
        else:
            raise NotImplementedError("Only method GS (GrowingSpheres) is implemented!")
        return explanation

    def generate_explanations(
        self,
        input: np.array,
        indices_with_missing_values: np.array,
        n: int,
        method: str = "GS",
    ) -> np.array:
        """Generate explanation for input vector(s).

        :param np.array input: input array with one or multiple rows
        :param np.array indices_with_missing_values: indices of columns missing values
        :param int n: number of explanations to generate
        :return np.array: array with n rows
        """
        counterfactuals = self.get_counterfactuals(input[0], method)
        return counterfactuals
