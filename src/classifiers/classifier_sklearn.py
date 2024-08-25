import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError


class ClassifierSklearn:
    """Abstracts details of using the sklearn LogisticRegression classifier."""

    def __init__(
        self,
        num_predictor_indices: list[int],
        threshold: float = 0.5,
    ):
        self.pipeline = self.init_pipeline(num_predictor_indices)
        self.threshold = threshold

    def init_pipeline(self, predictor_indices: np.array) -> Pipeline:
        """Build pipeline for logistic regression.

        :param np.array predictor_indices: indices of predictors in training data
        :return Pipeline: Pipeline with scaler and logistic regression model
        """
        num_pipeline = Pipeline(steps=[("scale", MinMaxScaler())])
        col_trans = ColumnTransformer(
            transformers=[("num_pipeline", num_pipeline, predictor_indices)],
            remainder="drop",
        )
        return Pipeline(
            steps=[
                ("col_trans", col_trans),
                ("model", LogisticRegression(class_weight="balanced")),
            ]
        )

    def train(self, X_train: np.array, y_train: np.array):
        """Train model on given data.

        :param np.array X_train: predictor training data for model
        :param np.array y_train: target training data for model
        """
        self.pipeline.fit(X_train, y_train)

    def predict(self, input: np.array) -> int:
        """Get classification (0/1) for new instance from trained model.

        :param np.array input: 1D input array to predict
        :return int: predicted class
        """
        prediction = self.predict_with_proba(input)
        return prediction[0]

    def predict_with_proba(self, input: np.array) -> tuple[int, float]:
        """Get classification (0/1) and probability of 1 for a new instance from trained model.

        :param np.array input: 1D input array to predict
        :return tuple[int, float]: tuple with predicted class and probability that predicted class was 1
        """
        try:
            if len(input.shape) == 1:
                input = [input]
            res = self.pipeline.predict_proba(input)[0]
        except NotFittedError:
            raise NotFittedError("Classifier.train() must be called before predict()!")

        prob_positive = res[1]
        if prob_positive >= self.threshold:
            predicted_class = 1
        else:
            predicted_class = 0
        return (predicted_class, prob_positive)

    def get_model(self):
        return self.pipeline
