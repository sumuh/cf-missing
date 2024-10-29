import json
import yaml
import sys
import itertools
import time
from ..evaluation.counterfactual_evaluator import CounterfactualEvaluator
from ..evaluation.imputer_evaluator import ImputerEvaluator
from ..evaluation.evaluation_results_container import (
    EvaluationResultsContainer,
    SingleEvaluationResultsContainer,
)
from ..classifiers.classifier_interface import Classifier
from ..utils.data_utils import (
    Config,
    load_data,
    transform_data,
)
from ..utils.misc_utils import write_run_configuration_to_file
from ..utils.visualization_utils import explore_data
from ..hyperparams.hyperparam_optimization import HyperparamOptimizer
from ..logging.cf_logger import CfLogger


class EvaluationRunner:

    def __init__(self, config_file_path: str, results_dir: str, logger: CfLogger):
        self.results_dir = results_dir
        config_all = Config(self.load_config(config_file_path))
        self.data_config = config_all.data.diabetes
        self.evaluation_config = config_all.evaluation
        self.hyperparam_opt = HyperparamOptimizer()
        self.time_start = time.time()
        self.logger = logger

    def load_config(self, config_file_path: str) -> dict:
        """Load configuration.

        :param str config_file_path: path to config.yaml
        :return dict: config as dictionary
        """
        with open(config_file_path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def run_counterfactual_evaluation_with_config(
        self,
        current_evaluation_config: Config,
    ) -> SingleEvaluationResultsContainer:
        """Runs evaluation for config settings and writes results to file if so specified.

        :param Config current_evaluation_config: evaluation configuration with current_params set to those to evaluate
        :return SingleEvaluationResultsContainer: object containing evaluation results for this run
        """

        results_container = SingleEvaluationResultsContainer(
            current_evaluation_config.current_params
        )
        classifier = Classifier(
            current_evaluation_config.current_params.classifier,
            self.data_config.predictor_indices,
        )
        # Evaluate counterfactual generation for each row in dataset and return average metrics
        evaluator = CounterfactualEvaluator(
            self.data_config,
            current_evaluation_config,
            self.data,
            classifier,
            self.hyperparam_opt,
            self.logger,
        )
        aggregated_results = evaluator.perform_evaluation()
        self.logger.log_debug(json.dumps(aggregated_results, indent=2))
        results_container.set_counterfactual_metrics(aggregated_results)

        return results_container

    def run_single_evaluation(self, params: dict[str, any], results_file_path: str):
        """Runs evaluation for specified parameters.

        :param dict[str, any] params: dictionary of params to evaluate with
        :return SingleEvaluationResultsContainer: object containing evaluation results for this run
        """
        print("current_params_dict")
        print(json.dumps(params, indent=2))
        # Set current_params as the current combinations
        current_evaluation_config_dict = self.evaluation_config.get_dict().copy()
        for k, v in params.items():
            current_evaluation_config_dict["current_params"][k] = v

        self.logger.log_debug("current_evaluation_config_dict")
        self.logger.log_debug(json.dumps(current_evaluation_config_dict, indent=2))

        evaluation_results_obj = self.run_counterfactual_evaluation_with_config(
            Config(current_evaluation_config_dict),
        )
        # Save results in file after each single evaluation in case of run failure
        evaluation_results_obj.append_results_to_file(results_file_path)
        time_so_far_min = round((time.time() - self.time_start) / 60, 1)
        time_so_far_h = round(time_so_far_min / 60, 1)
        print(
            f"Completed evaluation run. Time since start: {time_so_far_min} minutes ({time_so_far_h} hours)"
        )
        return evaluation_results_obj

    def _get_param_combinations_to_run(self) -> tuple[list, list]:
        """Create two separate combinations of params:
        one for runs where single index is set to missing at a time,
        and one for runs where multiple indices are set to missing at a time.

        :return tuple[list, list]: param combinations for single missing value runs, param combinations for multiple missing values run
        """
        param_lists = [
            v
            for k, v in self.evaluation_config.params.get_dict().items()
            if k not in ["ind_missing", "num_missing"]
        ]
        param_combinations_one_by_one_run = list(itertools.product(*param_lists))
        param_combinations_increasing_index_run = list(itertools.product(*param_lists))

        if self.evaluation_config.params.ind_missing is not None:
            ind_missing = list(
                itertools.product(self.evaluation_config.params.ind_missing)
            )
        else:
            ind_missing = []

        ind_missing_all_product = itertools.product(
            param_combinations_one_by_one_run, ind_missing
        )
        ind_missing_all_combs = [
            tuple_1 + tuple_2 for tuple_1, tuple_2 in ind_missing_all_product
        ]

        if self.evaluation_config.params.num_missing is not None:
            num_missing = list(
                itertools.product(self.evaluation_config.params.num_missing)
            )
        else:
            num_missing = []

        num_missing_all_product = itertools.product(
            param_combinations_increasing_index_run, num_missing
        )
        num_missing_all_combs = [
            tuple_1 + tuple_2 for tuple_1, tuple_2 in num_missing_all_product
        ]
        return ind_missing_all_combs, num_missing_all_combs

    def run_counterfactual_evaluation_with_different_configs(
        self, results_file_path: str
    ) -> EvaluationResultsContainer:
        """Runs counterfactual evaluation with different parameters.

        :return EvaluationResultsContainer: object containing results for all runs
        """
        print("Running countercactual evaluation.")
        time_start = time.time()

        # Object to store all evaluation results in
        all_results_container = EvaluationResultsContainer(
            self.evaluation_config.params
        )

        write_run_configuration_to_file(
            self.data, self.evaluation_config.params, results_file_path
        )

        ind_missing_all_combs, num_missing_all_combs = (
            self._get_param_combinations_to_run()
        )

        if len(ind_missing_all_combs) > 0 and len(num_missing_all_combs) > 0:
            total_combinations = len(ind_missing_all_combs) + len(num_missing_all_combs)
        elif len(ind_missing_all_combs) > 0:
            total_combinations = len(ind_missing_all_combs)
        elif len(num_missing_all_combs) > 0:
            total_combinations = len(num_missing_all_combs)

        iteration = 1
        if len(ind_missing_all_combs) > 0:
            print(
                "Running evaluation for missing values so that each feature is missing at a time"
            )
            for param_combination in ind_missing_all_combs:
                print(f"Running evaluation {iteration}/{total_combinations}")
                self.logger.log_debug(f"param combination: {param_combination}")
                iteration += 1
                param_dict = {
                    "classifier": param_combination[0],
                    "imputation_type": param_combination[1],
                    "n": param_combination[2],
                    "k": param_combination[3],
                    "distance_lambda_generation": param_combination[4],
                    "diversity_lambda_generation": param_combination[5],
                    "sparsity_lambda_generation": param_combination[6],
                    "distance_lambda_selection": param_combination[7],
                    "diversity_lambda_selection": param_combination[8],
                    "sparsity_lambda_selection": param_combination[9],
                    "selection_alg": param_combination[10],
                    "ind_missing": param_combination[11],
                    "num_missing": None,
                }
                all_results_container.add_evaluation(
                    self.run_single_evaluation(param_dict, results_file_path)
                )

        if len(num_missing_all_combs) > 0:
            print("Running evaluation for increasing number of missing values")
            for param_combination in num_missing_all_combs:
                print(f"Running evaluation {iteration}/{total_combinations}")
                self.logger.log_debug(f"param combination: {param_combination}")
                iteration += 1
                param_dict = {
                    "classifier": param_combination[0],
                    "imputation_type": param_combination[1],
                    "n": param_combination[2],
                    "k": param_combination[3],
                    "distance_lambda_generation": param_combination[4],
                    "diversity_lambda_generation": param_combination[5],
                    "sparsity_lambda_generation": param_combination[6],
                    "distance_lambda_selection": param_combination[7],
                    "diversity_lambda_selection": param_combination[8],
                    "sparsity_lambda_selection": param_combination[9],
                    "selection_alg": param_combination[10],
                    "num_missing": param_combination[11],
                    "ind_missing": None,
                }
                all_results_container.add_evaluation(
                    self.run_single_evaluation(param_dict, results_file_path)
                )

        time_end = time.time()
        total_time_min = round((time_end - time_start) / 60, 1)
        total_time_h = round(total_time_min / 60, 1)
        print(f"Evaluation done. Took {total_time_min} minutes ({total_time_h} hours)")
        return all_results_container

    def run_evaluations(self):
        """Loads config and data, and initiates evaluation runs. Saves results."""
        # Load and transform data
        data_raw = load_data(self.data_config.file_path, self.data_config.separator)
        self.data = transform_data(data_raw, self.data_config)
        # For testing
        # self.data = self.data[:75]
        # explore_data(self.data)

        # Find best hyperparameters for models
        # self.hyperparam_opt.run_hyperparam_optimization_for_imputation_models(data.to_numpy()[:, :-1])

        self.run_counterfactual_evaluation_with_different_configs(
            f"{self.results_dir}/results.yaml"
        )
