from sklearn.model_selection import train_test_split
from sklearn import metrics as metrics

from ..utils.data_utils import (
    Config,
    get_indices_with_missing_values,
    get_feature_mads,
    get_averages_from_dict_of_arrays,
    get_X_y,
)


class ClassifierEvaluator:
    """Class for evaluating classifiers."""

    def __init__(self, data_config, train_data, test_data, classifier):
        self.data_config = data_config
        self.train_data = train_data.to_numpy()
        self.test_data = test_data.to_numpy()
        self.classifier = classifier

    def get_y_pred_y_true(self) -> tuple[list, list]:
        """Get predicted labels and true labels for test data.

        :return tuple[list, list]: predicted labels and true labels
        """
        X_train, y_train = get_X_y(
            self.train_data,
            self.data_config.predictor_indices,
            self.data_config.target_index,
        )
        X_test, y_test = get_X_y(
            self.test_data,
            self.data_config.predictor_indices,
            self.data_config.target_index,
        )
        self.classifier.train(X_train, y_train)
        y_pred = [self.classifier.predict(X_test_row) for X_test_row in X_test]
        return (y_pred, y_test)

    def evaluate_classifier(self) -> dict[str, float]:
        """Calculate various performance metrics for classifier.

        :return dict[str, float]: dict where key is metric name and value is metric value
        """
        y_pred, y_true = self.get_y_pred_y_true()
        accuracy = metrics.accuracy_score(y_true, y_pred)
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        return {
            "accuracy": round(accuracy, 3),
            "roc_auc": round(roc_auc, 3),
            "f1": round(f1, 3),
        }
