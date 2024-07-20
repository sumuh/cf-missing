import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class Classifier:
    """Abstracts details of using the sklearn LogisticRegression classifier.
    """
    def __init__(self, train_data: pd.DataFrame, predictor_names: list[str], target_name: str, threshold: float = 0.5):
        self.pipeline = self.init_pipeline(predictor_names)
        self.train_data = train_data
        self.predictor_names = predictor_names
        self.target_name = target_name
        self.threshold = threshold

    def init_pipeline(self, predictor_names: list[str]) -> Pipeline:
        """Build pipeline for logistic regression.

        :param list[str] predictor_names: predictor names
        :return Pipeline: Pipeline with scaler and logistic regression model
        """
        num_pipeline = Pipeline(steps=[
            ("scale", MinMaxScaler())
        ])
        col_trans = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, predictor_names)
            ],
            remainder="drop"
        )
        return Pipeline(steps=[
            ("col_trans", col_trans),
            ("model", LogisticRegression())
        ])

    def train(self):
        """Train model on data given to class on init.
        """
        X = self.train_data[self.predictor_names]
        y = np.array(self.train_data[self.target_name]).ravel()
        self.pipeline.fit(X, y)

    def predict(self, input: pd.DataFrame) -> tuple[int, float]:
        """Get classification for new instance from trained model.
        Class 1 means person is predicted to have diabetes,
        class 0 means person is predicted to not have diabetes. 

        :param pd.DataFrame input: input df with one row
        :return tuple[int, float]: tuple with predicted class and probability that predicted class was 1
        """
        res = self.pipeline.predict_proba(input)[0]
        prob_positive = res[1]
        if prob_positive >= self.threshold:
            predicted_class = 1
        else:
            predicted_class = 0
        return (predicted_class, prob_positive)
