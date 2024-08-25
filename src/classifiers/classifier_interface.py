import numpy as np
import pandas as pd

from .classifier_sklearn import ClassifierSklearn
from .classifier_tensorflow import ClassifierTensorFlow


class Classifier:
    """Abstracts details of different classifier implementations."""

    def __init__(
        self,
        implementation: str,
        predictor_indices: list[int],
        threshold: float = 0.5,
    ):
        if implementation == "sklearn":
            self.classifier = ClassifierSklearn(predictor_indices, threshold)
        elif implementation == "tensorflow":
            self.classifier = ClassifierTensorFlow(predictor_indices)
        else:
            raise RuntimeError(
                f"Classifier implementation must be one of [sklearn, tensorflow]; was {implementation}"
            )
        self.threshold = threshold

    def get_threshold(self) -> float:
        """Get threshold of classification.

        :return float: threshold
        """
        return self.threshold

    def train(self, X_train: np.array, y_train: np.array):
        """Train model on given data.

        :param np.array X_train: predictor training data for model
        :param np.array y_train: target training data for model
        """
        self.classifier.train(X_train, y_train)

    def predict(self, input: np.array) -> int:
        """Get classification (0/1) for new instance from trained model.

        :param np.array input: 1D input array to predict
        :return int: predicted class
        """
        return self.classifier.predict(input)

    def predict_with_proba(self, input: np.array) -> tuple[int, float]:
        """Get classification (0/1) and probability of 1 for a new instance from trained model.

        :param np.array input: 1D input array to predict
        :return tuple[int, float]: tuple with predicted class and probability that predicted class was 1
        """
        return self.classifier.predict_with_proba(input)

    def get_classifier(self):
        return self.classifier

    def get_model(self):
        return self.classifier.get_model()
