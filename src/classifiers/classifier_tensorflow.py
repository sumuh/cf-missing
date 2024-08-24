import numpy as np
import tensorflow as tf
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError


class ClassifierTensorFlow:
    """Abstracts details of using a TensorFlow binary classifier."""

    def __init__(
        self,
        num_predictor_indices: list[int],
        threshold: float = 0.5,
        epochs: int = 10,
        batch_size: int = 20,
    ):
        self.num_predictor_indices = num_predictor_indices
        self.scaler = MinMaxScaler()
        self.threshold = threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    def build_model(self, input_dim: int) -> tf.keras.Model:
        """Builds a simple logistic regression model using TensorFlow.

        :param int input_dim: dimension of the input features
        :return tf.keras.Model: compiled model
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_dim,)),
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
        X_train_scaled = self.scaler.fit_transform(
            X_train[:, self.num_predictor_indices]
        )

        input_dim = X_train_scaled.shape[1]
        self.model = self.build_model(input_dim)

        self.model.fit(
            X_train_scaled,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

    def predict(self, input: np.array) -> int:
        """Get classification (0/1) for a new instance from trained model.

        :param np.array input: 1D input array to predict
        :return int: predicted class
        """
        prediction = self.predict_with_proba(input)
        return prediction[0]

    def predict_with_proba(self, input: np.array) -> tuple[int, float]:
        """Get classification (0/1) for a new instance from trained model.

        :param np.array input: 1D input array to predict
        :return tuple[int, float]: tuple with predicted class and probability that predicted class was 1
        """
        input = np.array(input).reshape(1, -1)
        input_scaled = self.scaler.transform(input[:, self.num_predictor_indices])

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
