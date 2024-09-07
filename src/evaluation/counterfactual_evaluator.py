import pandas as pd
import numpy as np
import time
import json
import sys
import random

from ..hyperparams.hyperparam_optimization import HyperparamOptimizer
from ..classifiers.classifier_interface import Classifier
from ..counterfactual_generator import CounterfactualGenerator
from ..imputer import Imputer
from ..utils.misc_utils import get_example_df, get_missing_indices_for_multiple_missing_values
from ..utils.data_utils import (
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
    get_count_diversity,
    get_distance,
)


class CounterfactualEvaluator:
    """Class for evaluating counterfactual generation."""

    def __init__(
        self,
        data_config: Config,
        evaluation_config: Config,
        data_pd: pd.DataFrame,
        classifier: Classifier,
        hyperparam_opt: HyperparamOptimizer,
    ):
        self.data_config = data_config
        self.evaluation_config = evaluation_config
        self.debug = evaluation_config.debug
        self.hyperparam_opt = hyperparam_opt
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
            self.hyperparam_opt,
            self.debug,
        )

    def _evaluate_counterfactuals(
        self, counterfactuals: np.array
    ) -> dict[str, np.array]:
        """Return evaluation metrics for a set of counterfactuals.

        :param np.array counterfactuals: set of counterfactuals
        :return dict[str, np.array]: dict where key is metric name and value is metric value
        """
        n_cfs = len(counterfactuals)
        if n_cfs > 0:
            mads = get_feature_mads(self.X_train_current)
            avg_dist_from_original = get_average_distance_from_original(
                self.test_instance_complete_current, counterfactuals, mads
            )
            diversity = get_diversity(counterfactuals, mads)
            count_diversity = get_count_diversity(counterfactuals)
            if len(self.indices_with_missing_values_current) > 0:
                diversity_missing_values = get_diversity(
                    counterfactuals[:, self.indices_with_missing_values_current],
                    mads[self.indices_with_missing_values_current],
                )
                count_diversity_missing_values = get_count_diversity(
                    counterfactuals[:, self.indices_with_missing_values_current]
                )
                # Calculate sparsity from imputed test instance
                avg_sparsity = get_average_sparsity(
                    self.test_instance_imputed_current, counterfactuals
                )
            else:
                diversity_missing_values = 0
                count_diversity_missing_values = 0
                avg_sparsity = get_average_sparsity(
                    self.test_instance_complete_current, counterfactuals
                )

        return {
            "n_vectors": n_cfs,
            "avg_dist_from_original": avg_dist_from_original,
            "diversity": diversity,
            "count_diversity": count_diversity,
            "diversity_missing_values": diversity_missing_values,
            "count_diversity_missing_values": count_diversity_missing_values,
            "avg_sparsity": avg_sparsity,
        }

    def _introduce_missing_values_to_test_instance(self):
        """Change specified value(s) in test instance to nan according to evaluation config params.

        :return np.array: test instance with missing value(s)
        """
        test_instance_with_missing_values = self.test_instance_complete_current.copy()
        if self.evaluation_config.current_params.ind_missing != "None":
            # Single missing value according to specified index
            ind_missing = self.evaluation_config.current_params.ind_missing
        elif self.evaluation_config.current_params.num_missing != "None":
            # Multiple missing values, indexes up to num_missing
            ind_missing = get_missing_indices_for_multiple_missing_values(self.evaluation_config.current_params.num_missing)
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
        if len(self.indices_with_missing_values_current) > 0:
            # Evaluate input with missing values
            input = self.test_instance_imputed_current
        else:
            # Evaluate complete input
            input = self.test_instance_complete_current

        counterfactuals = self.cf_generator.generate_explanations(
            input,
            self.X_train_current,
            self.indices_with_missing_values_current,
            self.evaluation_config.current_params.k,
            self.evaluation_config.current_params.n,
            self.data_pd,
            self.evaluation_config.current_params.imputation_type,
        )
        time_end = time.time()
        wall_time = time_end - time_start
        return counterfactuals, wall_time


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

        data_without_test_instance = np.delete(self.data_np, row_ind, 0)
        X_train, y_train = get_X_y(
            data_without_test_instance,
            self.data_config.predictor_indices,
            self.data_config.target_index,
        )
        self.X_train_current = X_train
        self.classifier.train(X_train, y_train)

        example_df = None

        test_instance_metrics = {}

        if (
            self.evaluation_config.current_params.num_missing != "None" and 
            self.evaluation_config.current_params.num_missing > 0
        ):
            # Evaluate input with missing values
            self.test_instance_with_missing_values_current = (
                self._introduce_missing_values_to_test_instance()
            )
            self.indices_with_missing_values_current = get_indices_with_missing_values(
                self.test_instance_with_missing_values_current
            )
            self.test_instance_imputed_current = self._impute_test_instance()
            prediction = self.classifier.predict(self.test_instance_imputed_current)
        else:
            # Evaluate complete input
            self.indices_with_missing_values_current = np.array([])
            prediction = self.classifier.predict(self.test_instance_complete_current)

        if prediction != self.data_config.target_class:
            # Generate counterfactuals
            counterfactuals, wall_time = self._get_counterfactuals()
            test_instance_metrics.update({"runtime_seconds": wall_time})

            example_df = get_example_df(
                self.indices_with_missing_values_current,
                self.test_instance_complete_current,
                self.test_instance_with_missing_values_current,
                self.test_instance_imputed_current,
                counterfactuals
            )

            # Evaluate generated counterfactuals vs. original vector
            test_instance_metrics.update(
                self._evaluate_counterfactuals(counterfactuals)
            )

        test_instance_metrics.update(
            {"num_missing": len(self.indices_with_missing_values_current)}
        )

        return (test_instance_metrics, example_df)

    def _aggregate_results(self, metrics: dict):
        """Aggregate results for each test instance to obtain averages.

        :param dict metrics: all metrics
        :return dict: aggregated metrics; averages for numeric fields and total of true values for boolean fields
        """
        numeric_metrics = {
            k: v
            for k, v in metrics.items()
            if k in self.evaluation_config.numeric_metrics
        }
        boolean_metrics = {
            k: v
            for k, v in metrics.items()
            if k in self.evaluation_config.boolean_metrics
        }
        numeric_averages = {
            k: round(v, 3)
            for k, v in get_averages_from_dict_of_arrays(numeric_metrics).items()
        }
        boolean_total_trues = {k: sum(v) for k, v in boolean_metrics.items()}
        final_dict = numeric_averages
        final_dict.update(boolean_total_trues)
        # Add coverage: ratio of test inputs for which at least one cf was found
        num_inputs_cf_found = filter(lambda x: x > 0, metrics["n_vectors"])
        total_inputs_cf_required = metrics["n_vectors"]
        final_dict.update(
            {"coverage": len(list(num_inputs_cf_found)) / len(total_inputs_cf_required)}
        )
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

        # Print status info & example df every info_frequency rows
        info_frequency = 100

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
            show_example = (
                row_ind % info_frequency == 0 or show_example
            ) and self.debug
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