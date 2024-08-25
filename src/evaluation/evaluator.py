import pandas as pd
import numpy as np
import time
import json
import sys
import random
from typing import Callable
from sklearn.model_selection import train_test_split
from sklearn import metrics as metrics
from tensorflow.keras import metrics as tf_metrics

from ..classifiers.classifier_interface import Classifier
from ..counterfactual_generator import CounterfactualGenerator
from ..imputer import Imputer
from ..data_utils import (
    Config,
    get_indices_with_missing_values,
    get_feature_mads,
    get_averages_from_dict_of_arrays,
    get_X_y,
)
from .evaluation_metrics import (
    get_average_sparsity,
    get_diversity,
    get_average_distance_from_original,
    get_valid_ratio,
    get_count_diversity,
)


class Evaluator:
    """Class used to evaluate classifier and counterfactual generation process."""

    def __init__(
        self, data_pd: pd.DataFrame, data_config: Config, evaluation_config: Config
    ):
        self.data_config = data_config
        self.evaluation_config = evaluation_config
        self.classifier = Classifier(
            evaluation_config.classifier, data_config.predictor_indices
        )
        self.train_data, self.test_data = train_test_split(data_pd, test_size=0.2)

    def evaluate_classifier(self) -> dict[str, float]:
        """Evaluates classifier on test data.

        :return dict[str, float]: dict where key is metric name and value is metric value
        """
        classifier_evaluator = ClassifierEvaluator(
            self.data_config, self.train_data, self.test_data, self.classifier
        )
        return classifier_evaluator.evaluate_classifier()

    def evaluate_counterfactual_generation(self) -> tuple[dict, dict]:
        """Evaluates counterfactual generation.

        :return tuple[dict, dict]: dict where key is metric name and value is metric value
        """
        counterfactual_evaluator = CounterfactualEvaluator(
            self.data_config, self.evaluation_config, self.train_data, self.classifier
        )
        histogram_dict, aggregated_results = (
            counterfactual_evaluator.perform_evaluation()
        )
        return histogram_dict, aggregated_results


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


class CounterfactualEvaluator:
    """Class for evaluating counterfactual generation."""

    def __init__(
        self,
        data_config: Config,
        evaluation_config: Config,
        data_pd: pd.DataFrame,
        classifier: Classifier,
    ):
        self.data_config = data_config
        self.evaluation_config = evaluation_config
        self.debug = evaluation_config.debug
        self.classifier = classifier

        if self.debug:
            print(data_pd.head())

        self.data_pd = data_pd
        self.data_np = data_pd.copy().to_numpy()

        self.X_train_current = None
        self.indices_with_missing_values_current = None
        self.test_instance_complete_current = None
        self.test_instance_with_missing_values_current = None

        self.cf_generator = CounterfactualGenerator(
            self.classifier,
            data_config.target_class,
            data_config.target_name,
            self.debug,
        )

    def _evaluate_counterfactuals(
        self, counterfactuals: np.array, prediction_func: Callable
    ) -> dict[str, np.array]:
        """Return evaluation metrics for a set of counterfactuals.

        :param np.array counterfactuals: set of counterfactuals
        :param Callable prediction_func: classifier prediction function
        :return dict[str, np.array]: dict where key is metric name and value is metric value
        """
        n_cfs = len(counterfactuals)
        if n_cfs > 0:
            valid_ratio = get_valid_ratio(
                counterfactuals, prediction_func, self.data_config.target_class
            )
            mads = get_feature_mads(self.X_train_current)
            avg_dist_from_original = get_average_distance_from_original(
                self.test_instance_complete_current, counterfactuals, mads
            )
            diversity = get_diversity(counterfactuals, mads)
            count_diversity = get_count_diversity(counterfactuals)
            diversity_missing_values = get_diversity(
                counterfactuals[:, self.indices_with_missing_values_current],
                mads[self.indices_with_missing_values_current],
            )
            count_diversity_missing_values = get_count_diversity(
                counterfactuals[:, self.indices_with_missing_values_current]
            )
            # Calculate sparsity from imputed test instance
            avg_sparsity = get_average_sparsity(
                self.test_instance_complete_current, counterfactuals
            )

        return {
            "n_vectors": n_cfs,
            "no_cf_found": n_cfs == 0,
            "valid_ratio": valid_ratio,
            "avg_dist_from_original": avg_dist_from_original,
            "diversity": diversity,
            "count_diversity": count_diversity,
            "diversity_missing_values": diversity_missing_values,
            "count_diversity_missing_values": count_diversity_missing_values,
            "avg_sparsity": avg_sparsity,
        }

    def _introduce_missing_values_to_test_instance(self):
        """Change values in test instance to nan according to specified
        missing data mechanism.

        :return np.array: test instance with missing value(s)
        """
        test_instance_with_missing_values = self.test_instance_complete_current.copy()
        mechanism = self.evaluation_config.missing_data.mechanism
        if mechanism == "MCAR":
            ind_missing = random.sample(
                range(0, len(test_instance_with_missing_values)),
                self.evaluation_config.missing_data.number_of_missing,
            )
        elif mechanism == "MAR":
            # if pedigree fun is below 0.5, insuling is missing
            if self.test_instance_complete_current[6] < 0.5:
                test_instance_with_missing_values[4] = np.nan
        elif mechanism == "MNAR":
            # if BMI is over 30, BMI is missing
            if self.test_instance_complete_current[5] > 30:
                test_instance_with_missing_values[5] = np.nan
        else:
            raise RuntimeError(
                f"Invalid missing data mechanism '{mechanism}'; excpected one of MCAR, MAR, MNAR"
            )
        if ind_missing is not None:
            test_instance_with_missing_values[ind_missing] = np.nan
        return test_instance_with_missing_values

    def _get_test_instance(self, row_ind: int) -> np.array:
        """Create test instance from data.

        :param int row_ind: row index in data to test
        :return np.array: test instance with target feature removed
        """
        test_instance = self.data_np[row_ind, :].ravel()
        test_instance = np.delete(test_instance, self.data_config.target_index)
        return test_instance

    def _impute_test_instance(self) -> np.array:
        """Impute test instance with missing values using chosen imputation technique.

        :return np.array: test instance with missing values imputed
        """
        imputer = Imputer(self.X_train_current)
        return imputer.mean_imputation(
            self.test_instance_with_missing_values_current,
            self.indices_with_missing_values_current,
        )

    def _get_counterfactuals(self) -> tuple[np.array, float]:
        """Generate counterfactuals.

        :return tuple[np.array, float]: set of alternative vectors and generation wall clock time
        """
        time_start = time.time()
        counterfactuals = self.cf_generator.generate_explanations(
            self.test_instance_imputed_current,
            self.X_train_current,
            self.indices_with_missing_values_current,
            self.evaluation_config.counterfactuals.counterfactuals_to_return,
            self.evaluation_config.multiple_imputation.imputations_to_create,
            self.data_pd,
        )
        time_end = time.time()
        wall_time = time_end - time_start
        return counterfactuals, wall_time

    def _get_example_df(self, counterfactuals: np.array) -> pd.DataFrame:
        """Creates dataframe from one test instance and the counterfactuals
         generated for it. For qualitative evaluation of counterfactuals.

        :param np.array counterfactuals: counterfactuals
        :return pd.DataFrame: example df
        """
        example_df = np.vstack(
            (
                self.test_instance_complete_current,
                self.test_instance_with_missing_values_current,
                self.test_instance_imputed_current,
                counterfactuals,
            )
        )
        index = ["complete input", "input with missing", "imputed input"]
        for _ in range(len(counterfactuals)):
            index.append("counterfactual")
        return pd.DataFrame(example_df, index=index)

    def _evaluate_single_instance(
        self, row_ind: int
    ) -> tuple[dict[str, float], pd.DataFrame]:
        """Return metrics for counterfactuals generated for single test instance.
        Introduces potential missing values to test instance, imputes them,
        generates counterfactual and evaluates it.

        :param int row_ind: row index of test instance in test data
        :return tuple[dict[str, float], pd.DataFrame]: metrics for evaluated instance 
        and dataframe comparing inputs with counterfactuals
        """
        self.test_instance_complete_current = self._get_test_instance(row_ind)
        self.test_instance_with_missing_values_current = (
            self._introduce_missing_values_to_test_instance()
        )
        self.indices_with_missing_values_current = get_indices_with_missing_values(
            self.test_instance_with_missing_values_current
        )

        example_df = None
        test_instance_metrics = {
            "num_missing_values": len(self.indices_with_missing_values_current)
        }

        if len(self.indices_with_missing_values_current) > 0:
            data_without_test_instance = np.delete(self.data_np, row_ind, 0)
            X_train, y_train = get_X_y(
                data_without_test_instance,
                self.data_config.predictor_indices,
                self.data_config.target_index,
            )
            self.X_train_current = X_train
            self.classifier.train(X_train, y_train)
            self.test_instance_imputed_current = self._impute_test_instance()
            # Make prediction
            prediction = self.classifier.predict(self.test_instance_imputed_current)
            test_instance_metrics.update(
                {"undesired_class": prediction != self.data_config.target_class}
            )
            if prediction != self.data_config.target_class:
                # Generate counterfactuals
                counterfactuals, wall_time = self._get_counterfactuals()
                test_instance_metrics.update({"runtime_seconds": wall_time})

                example_df = self._get_example_df(counterfactuals)

                # Evaluate generated counterfactuals vs. original vector
                test_instance_metrics.update(
                    self._evaluate_counterfactuals(
                        counterfactuals, self.classifier.predict
                    )
                )

        return (test_instance_metrics, example_df)

    def _aggregate_results(self, metrics: dict):
        """Aggregate results for each test instance to obtain averages.

        :param dict metrics: all metrics
        :return dict: aggregated metrics; averages for numeric fields and total of true values for boolean fields
        """
        numeric_metrics = {
            k: v for k, v in metrics.items() if k in self.evaluation_config.numeric_metrics
        }
        boolean_metrics = {
            k: v for k, v in metrics.items() if k in self.evaluation_config.boolean_metrics
        }
        numeric_averages = {
            k: round(v, 3) for k, v in get_averages_from_dict_of_arrays(numeric_metrics).items()
        }
        boolean_total_trues = {k: sum(v) for k, v in boolean_metrics.items()}
        final_dict = numeric_averages
        final_dict.update(boolean_total_trues)
        return final_dict

    def perform_evaluation(self):
        """Evaluate counterfactual generation process for configurations given on init.

        :return dict metrics: dict containing metric arrays for all test instances
        """

        def debug_info(info_frequency, row_ind, show_example, metrics):
            if row_ind % info_frequency == 0:
                print(f"Evaluated {row_ind}/{num_rows} rows")
            # Every info_frequency rows find example to show
            if show_example:
                if result[1] is not None:
                    print(result[1])
                    if not self.debug:
                        show_example = False
                    print(json.dumps(metrics, indent=2))
            if self.debug:
                for _ in range(4):
                    print(f"{('~'*20)}")
            return show_example

        info_frequency = 20

        num_rows = len(self.data_np)
        all_metric_names = (
            self.evaluation_config.numeric_metrics
            + self.evaluation_config.boolean_metrics
        )
        metrics = {metric: [] for metric in all_metric_names}

        if self.debug:
            show_example = True
        else:
            show_example = False

        for row_ind in range(num_rows):
            show_example = row_ind % info_frequency == 0 or show_example
            result = self._evaluate_single_instance(row_ind)
            single_instance_metrics = result[0]
            if self.debug:
                print(
                    f"test_instance_metrics: {json.dumps(single_instance_metrics, indent=2)}"
                )

            for metric_name, metric_value in single_instance_metrics.items():
                metrics[metric_name].append(metric_value)

            show_example = debug_info(
                info_frequency, row_ind, show_example, single_instance_metrics
            )

        print(f"Evaluated {num_rows}/{num_rows} rows")
        histogram_dict = {
            metric_name: metric_value
            for metric_name, metric_value in metrics.items()
            if metric_name in self.evaluation_config.metrics_for_histograms
        }
        aggregated_results = self._aggregate_results(metrics)
        return histogram_dict, aggregated_results
