import numpy as np
import tensorflow as tf
import os
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

from ..imputer import Imputer
from ..utils.data_utils import get_indices_with_missing_values

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ClassifierTensorFlow:
    """Abstracts details of using a TensorFlow binary classifier."""

    def __init__(
        self,
        predictor_indices: list[int],
        threshold: float = 0.5,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        self.predictor_indices = predictor_indices
        self.scaler = MinMaxScaler()
        self.threshold = threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def build_model(self, input_dim: int) -> tf.keras.Model:
        """Builds a simple logistic regression model using TensorFlow.

        :param int input_dim: dimension of the input features
        :return tf.keras.Model: compiled model
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_dim,)),
                tf.keras.layers.Dense(10, activation="relu"),
                # tf.keras.layers.Dense(5, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def train(self, X_train: np.array, y_train: np.array):
        """Train the TensorFlow model on given data.

        :param np.array X_train: predictor training data for model
        :param np.array y_train: target training data for model
        """
        X_train_scaled = self.scaler.fit_transform(X_train[:, self.predictor_indices])

        input_dim = X_train_scaled.shape[1]
        self.model = self.build_model(input_dim)

        self.model.fit(
            X_train_scaled,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

    def _handle_missing_values(self, input: np.array, X_train: np.array) -> np.array:
        """Imputes missing values with mean imputation.

        :param np.array X_train: train dataset for calculating means
        :param np.array input: input with potentially missing values
        :return np.array: imputed input
        """
        indices_with_missing_values = get_indices_with_missing_values(input)
        if len(indices_with_missing_values) > 0:
            imputer = Imputer(X_train)
            return imputer.mean_imputation(input, indices_with_missing_values)
        else:
            return input

    def predict(self, input: np.array, X_train: np.array) -> int:
        """Get classification (0/1) for a new instance from trained model.

        :param np.array input: 1D input array to predict
        :param np.array X_train: training data
        :return int: predicted class
        """
        input_handled = self._handle_missing_values(input, X_train)
        prediction = self.predict_with_proba(input_handled, X_train)
        return prediction[0]

    def predict_with_proba(
        self, input: np.array, X_train: np.array
    ) -> tuple[int, float]:
        """Get classification (0/1) and probability of 1 for a new instance from trained model.

        :param np.array input: 1D input array to predict
        :param np.array X_train: training data
        :return tuple[int, float]: tuple with predicted class and probability that predicted class was 1
        """
        input_handled = self._handle_missing_values(input, X_train)
        input_handled = np.array(input_handled).reshape(1, -1)
        input_scaled = self.scaler.transform(input_handled[:, self.predictor_indices])

        try:
            prob_positive = self.model.predict(input_scaled, verbose=0)[0][0]
        except NotFittedError:
            raise NotFittedError("Classifier.train() must be called before predict()!")

        if prob_positive >= self.threshold:
            predicted_class = 1
        else:
            predicted_class = 0
        return (predicted_class, prob_positive)

    def get_model(self):
        return self.model
