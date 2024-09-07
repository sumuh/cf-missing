import json
import yaml
import sys
import itertools
import time
import pandas as pd
from ..evaluation.counterfactual_evaluator import CounterfactualEvaluator
from ..evaluation.imputer_evaluator import ImputerEvaluator
from ..evaluation.evaluation_results_container import (
    EvaluationResultsContainer,
    SingleEvaluationResultsContainer,
)
from ..classifiers.classifier_interface import Classifier
from ..evaluation.results_visualizer import ResultsVisualizer
from ..utils.data_utils import (
    Config,
    load_data,
    transform_data,
)
from ..hyperparams.hyperparam_optimization import HyperparamOptimizer

class EvaluationRunner():

    def __init__(self, config_file_path: str, results_dir: str):
        self.results_dir = results_dir
        config_all = Config(self.load_config(config_file_path))
        self.data_config = config_all.data.diabetes
        self.evaluation_config = config_all.evaluation
        self.hyperparam_opt = HyperparamOptimizer()
        self.debug = config_all.evaluation.debug
        self.rolling_results_file = f"{results_dir}/cf_rolling_results.txt"
        self.params_to_vary_dict = self.evaluation_config.params.get_dict()
        self.params_to_vary_names = self.evaluation_config.params.get_dict().keys()
        self.time_start = time.time()

    def load_config(self, config_file_path: str) -> dict:
        """Load configuration.

        :param str config_file_path: path to config.yaml
        :return dict: config as dictionary
        """
        with open(config_file_path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)


    def run_counterfactual_evaluation_with_config(
        self, current_evaluation_config: Config,
    ) -> SingleEvaluationResultsContainer:
        """Runs evaluation for config settings and writes results to file if so specified.

        :param Config current_evaluation_config: evaluation configuration with current_params set to those to evaluate
        :return SingleEvaluationResultsContainer: object containing evaluation results for this run
        """

        results_container = SingleEvaluationResultsContainer(current_evaluation_config, self.data)
        classifier = Classifier(
            current_evaluation_config.current_params.classifier, self.data_config.predictor_indices
        )
        # Evaluate counterfactual generation for each row in dataset and return average metrics
        evaluator = CounterfactualEvaluator(
            self.data_config, self.evaluation_config, self.data, classifier, self.hyperparam_opt
        )
        histogram_dict, aggregated_results = evaluator.perform_evaluation()
        if self.debug:
            print(json.dumps(aggregated_results, indent=2))
        results_container.set_counterfactual_metrics(aggregated_results)
        results_container.set_counterfactual_histogram_dict(histogram_dict)

        return results_container

    def run_single_evaluation(
        self, param_combination: tuple, type: str,
    ):
        """Runs evaluation for specified parameters.

        :param tuple param_combination: combination of params to evaluate with
        :return SingleEvaluationResultsContainer: object containing evaluation results for this run
        """
        current_params_dict = dict(zip(self.params_to_vary_names, param_combination))
        extra_param = param_combination[len(param_combination) - 1]
        print("extra_param")
        print(extra_param)
        # Set current_params as the current combinations
        current_evaluation_config_dict = self.evaluation_config.get_dict().copy()
        for k, v in current_params_dict.items():
            current_evaluation_config_dict["current_params"][k] = v
            if type == "one_by_one":
                current_evaluation_config_dict["current_params"]["ind_missing"] = extra_param
            elif type == "increasing_index":
                current_evaluation_config_dict["current_params"]["num_missing"] = extra_param

        print("current_evaluation_config_dict")
        print(json.dumps(current_evaluation_config_dict, indent=2))

        evaluation_results_obj = self.run_counterfactual_evaluation_with_config(
            Config(current_evaluation_config_dict),
        )
        evaluation_results_obj.set_evaluation_params_dict(current_params_dict)
        # Save results in file after each single evaluation in case of run failure
        evaluation_results_obj.append_results_to_file(self.rolling_results_file)
        time_so_far_min = round((time.time() - self.time_start) / 60, 1)
        time_so_far_h = round(time_so_far_min / 60, 1)
        print(
            f"Completed evaluation run. Time since start: {time_so_far_min} minutes ({time_so_far_h} hours)"
        )
        return evaluation_results_obj


    def run_counterfactual_evaluation_with_different_configs(self) -> EvaluationResultsContainer:
        """Runs counterfactual evaluation with different parameters.

        :param Config evaluation_config: evaluation configuration
        :param Config data_config: data configuration
        :param pd.DataFrame data: dataset
        :param str results_dir: directory to write results to
        :param HyperparamOptimizer hyperparam_opt: object containing optimized hyperparams
        :return tuple[EvaluationResultsContainer, float]: object containing results for run + runtime
        """
        print("Running evaluation...")
        time_start = time.time()

        # Object to store all evaluation results in
        all_results_container = EvaluationResultsContainer(self.params_to_vary_dict)
        lists = self.params_to_vary_dict.values()
        param_combinations_one_by_one_run = list(itertools.product(*lists))
        param_combinations_increasing_index_run = list(itertools.product(*lists))
        total_combinations = len(list(itertools.product(*lists))) * 2

        all_results_container = EvaluationResultsContainer(self.evaluation_config)

        iteration = 1
        ind_missing = list(itertools.product([i for i in range(self.data.shape[1] - 1)]))
        ind_missing_all_product = itertools.product(param_combinations_one_by_one_run, ind_missing)
        ind_missing_all_combs = [tuple_1 + tuple_2 for tuple_1, tuple_2 in ind_missing_all_product]
        for param_combination in ind_missing_all_combs:
            print("Running evaluation for missing values so that each feature is missing at a time")
            print(f"Running evaluation {iteration}/{total_combinations}")
            print(f"param combination: {param_combination}")
            iteration += 1
            all_results_container.add_evaluation(self.run_single_evaluation(param_combination, "one_by_one"))

        num_missing = list(itertools.product([i for i in range(self.data.shape[1] - 2)]))
        num_missing_all_product = itertools.product(param_combinations_increasing_index_run, num_missing)
        num_missing_all_combs = [tuple_1 + tuple_2 for tuple_1, tuple_2 in num_missing_all_product]
        for param_combination in num_missing_all_combs:
            print("Running evaluation for increasing number of missing values")
            print(f"Running evaluation {iteration}/{total_combinations}")
            print(f"param combination: {param_combination}")
            iteration += 1
            all_results_container.add_evaluation(self.run_single_evaluation(param_combination, "increasing_index"))

        time_end = time.time()
        total_time_min = round((time_end - time_start) / 60, 1)
        total_time_h = round(total_time_min / 60, 1)
        print(f"Evaluation done. Took {total_time_min} minutes ({total_time_h} hours)")
        all_results_container.set_runtime(time_end - time_start)
        return all_results_container


    def run_evaluations(self):
        # Load and transform data
        data_raw = load_data(self.data_config.file_path, self.data_config.separator)
        self.data = transform_data(data_raw, self.data_config)
        # For testing
        self.data = self.data[:75]
        # explore_data(data)

        # Find best hyperparameters for models
        # self.hyperparam_opt.run_hyperparam_optimization_for_imputation_models(data.to_numpy()[:, :-1])

        cf_results = self.run_counterfactual_evaluation_with_different_configs()
        cf_results.save_stats_to_file(f"{self.results_dir}/config.txt")
        cf_results.save_all_results_to_file(f"{self.results_dir}/results.txt")

        # Evaluate imputation methods
        imputater_evaluator = ImputerEvaluator(self.data, self.hyperparam_opt, self.evaluation_config.debug)
        imputater_results = imputater_evaluator.run_imputer_evaluation()

        # Save result visualizations
        results_visualizer = ResultsVisualizer(self.data, self.results_dir, self.evaluation_config)
        results_visualizer.save_counterfactual_results_visualizations(cf_results)
        results_visualizer.save_imputer_evaluation_results_visualizations(imputater_results)
        results_visualizer.save_data_visualizations()