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

    def get_growing_spheres_counterfactuals(
        self, input: np.array, n: int, debug: bool
    ) -> np.array:
        """Returns n counterfactuals generated with the DiverseGrowingSpheres algorithm.

        :param np.array input: input array
        :param int n: number of explanations to generate
        :param bool debug: enable debug/verbose mode
        :return np.array: array with n rows
        """
        gs_model = gs.DiverseGrowingSpheres(
            obs_to_interprete=np.array([input]),
            prediction_fn=self.classifier.predict,
            target_class=0,
            caps=[0.0, 200.0],
            n_in_layer=1000,
            first_radius=5.0,
            dicrease_radius=5.0,
            layer_shape="sphere",
            n_results=n,
            verbose=debug,
            debug=debug,
        )
        return gs_model.find_counterfactual()

    def generate_explanations(
        self,
        input: np.array,
        indices_with_missing_values: np.array,
        n: int,
        method: str = "GS",
        debug: bool = False,
    ) -> np.array:
        """Generate explanation for input vector(s).

        :param np.array input: input array
        :param np.array indices_with_missing_values: indices of columns missing values
        :param int n: number of explanations to generate
        :param str method: counterfactual generation method
        :param bool debug: enable debug/verbose mode
        :return np.array: array with n rows
        """
        if method == "GS":
            explanation = self.get_growing_spheres_counterfactuals(input, n, debug)
        else:
            raise NotImplementedError("Only method GS (GrowingSpheres) is implemented!")
        return explanation
