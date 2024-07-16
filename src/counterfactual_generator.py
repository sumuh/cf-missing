import pandas as pd
import random

from classifier import Classifier

class CounterfactualGenerator:

    def __init__(self, classifier: Classifier, train_data: pd.DataFrame):
        self.classifier = classifier
        self.train_data = train_data

    def generate_explanation(self, input: dict, n_vectors: int) -> pd.DataFrame:
        """Dummy method"""
        result = []
        for i in range(n_vectors):
            result.append({"age": random.randint(20, 40), "HDL": random.randint(5, 20), "LDL": random.randint(5, 20)})
        return pd.DataFrame(result)