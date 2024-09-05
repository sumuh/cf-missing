import os
import json
import yaml
import sys
import itertools
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from .evaluation.counterfactual_evaluator import CounterfactualEvaluator
from .evaluation.imputer_evaluator import ImputerEvaluator
from .evaluation.evaluation_results_container import (
    EvaluationResultsContainer,
    SingleEvaluationResultsContainer,
)
from .classifiers.classifier_interface import Classifier
from .evaluation.results_visualizer import ResultsVisualizer
from .utils.data_utils import (
    Config,
    load_data,
    transform_data,
)
from .hyperparams.hyperparam_optimization import HyperparamOptimizer


# Todo: remove after fixing sklearn UserWarning: X has feature names, but MinMaxScaler was fitted without feature names
# (maybe fit with pd df)
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def load_config(config_file_path: str) -> dict:
    """Load configuration.

    :param str config_file_path: path to config.yaml
    :return dict: config as dictionary
    """
    with open(config_file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def run_counterfactual_evaluation_with_config(
    data: pd.DataFrame,
    data_config: Config,
    evaluation_config: Config,
    hyperparam_opt: HyperparamOptimizer,
    debug: bool,
) -> SingleEvaluationResultsContainer:
    """Runs evaluation for config settings and writes results to file if so specified.

    :param pd.DataFrame data: dataset
    :param Config data_config: data configuration
    :param Config evaluation_config: evaluation configuration with current_params set to those to evaluate
    :param HyperparamOptimizer hyperparam_opt: object containing optimized hyperparams
    :param bool debug: debug/verbose mode
    :return SingleEvaluationResultsContainer: results container object
    """

    results_container = SingleEvaluationResultsContainer(evaluation_config, data)
    classifier = Classifier(
        evaluation_config.current_params.classifier, data_config.predictor_indices
    )
    # Evaluate counterfactual generation for each row in dataset and return average metrics
    evaluator = CounterfactualEvaluator(
        data_config, evaluation_config, data, classifier, hyperparam_opt
    )
    histogram_dict, aggregated_results = evaluator.perform_evaluation()
    if debug:
        print(json.dumps(aggregated_results, indent=2))
    results_container.set_counterfactual_metrics(aggregated_results)
    results_container.set_counterfactual_histogram_dict(histogram_dict)

    return results_container


def run_counterfactual_evaluation_with_different_configs(
    evaluation_config: Config,
    data_config: Config,
    data: pd.DataFrame,
    results_dir: str,
    hyperparam_opt: HyperparamOptimizer,
) -> EvaluationResultsContainer:
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
    debug = (evaluation_config.debug,)
    params_to_vary_dict = evaluation_config.params.get_dict()
    params_to_vary_names = params_to_vary_dict.keys()

    # Object to store all evaluation results in
    all_results_container = EvaluationResultsContainer(params_to_vary_dict)
    lists = params_to_vary_dict.values()
    param_combinations = itertools.product(*lists)
    total_combinations = len(list(itertools.product(*lists)))

    # For testing
    # data = data[:75]

    # For running many evaluation with different configs,
    # save each one's results to file as they are obtained
    rolling_results_file = f"{results_dir}/cf_rolling_results.txt"

    iteration = 1
    # Loop combinations of params specified in config.yaml
    for param_combination in param_combinations:
        print(f"Running evaluation {iteration}/{total_combinations}")
        iteration += 1
        current_params_dict = dict(zip(params_to_vary_names, param_combination))
        # Skip unnecessary combinations
        if (
            current_params_dict["num_missing"] > 1
            and current_params_dict["ind_missing"] != "random"
        ):
            continue
        # Set current_params as the current combinations
        current_evaluation_config_dict = evaluation_config.get_dict().copy()
        for k, v in current_params_dict.items():
            current_evaluation_config_dict["current_params"][k] = v
        if debug:
            print(
                f"Running evaluation for params: {json.dumps(current_params_dict, indent=2)}"
            )
        evaluation_obj = run_counterfactual_evaluation_with_config(
            data,
            data_config,
            Config(current_evaluation_config_dict),
            hyperparam_opt,
            debug,
        )
        evaluation_obj.set_evaluation_params_dict(current_params_dict)
        all_results_container.add_evaluation(evaluation_obj)
        evaluation_obj.append_results_to_file(rolling_results_file)
        time_so_far_min = round((time.time() - time_start) / 60, 1)
        time_so_far_h = round(time_so_far_min / 60, 1)
        print(
            f"Completed evaluation run. Time since start: {time_so_far_min} minutes ({time_so_far_h} hours)"
        )

    time_end = time.time()
    total_time_min = round((time_end - time_start) / 60, 1)
    total_time_h = round(total_time_min / 60, 1)
    print(f"Evaluation done. Took {total_time_min} minutes ({total_time_h} hours)")
    all_results_container.set_runtime(time_end - time_start)
    return all_results_container


def main():
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    config_all = Config(load_config(f"{current_file_path}/../config/config.yaml"))
    data_config = config_all.data.diabetes
    evaluation_config = config_all.evaluation

    # Load and transform data
    data = load_data(data_config.file_path, data_config.separator)
    data = transform_data(data, data_config)
    # explore_data(data)

    # Find best hyperparameters for models
    hyperparam_opt = HyperparamOptimizer()
    # hyperparam_opt.run_hyperparam_optimization_for_imputation_models(data.to_numpy()[:, :-1])

    current_time = datetime.now()
    formatted_time_day = current_time.strftime("%d-%m-%Y")
    formatted_time_sec = current_time.strftime("%d-%m-%Y-%H-%M-%S")
    results_dir = f"{current_file_path}/../evaluation_results/{formatted_time_day}/run_{formatted_time_sec}"
    Path(results_dir).mkdir(parents=True)

    cf_results = (
        run_counterfactual_evaluation_with_different_configs(
            evaluation_config, data_config, data, results_dir, hyperparam_opt
        )
    )

    cf_results.save_stats_to_file(f"{results_dir}/config.txt")
    cf_results.save_all_results_to_file(f"{results_dir}/results.txt")

    # Evaluate imputation methods
    imputater_evaluator = ImputerEvaluator(data, hyperparam_opt)
    imputater_results = imputater_evaluator.run_imputer_evaluation()
    
    # Save result visualizations
    results_visualizer = ResultsVisualizer(data, results_dir, evaluation_config)
    results_visualizer.save_counterfactual_results_visualizations(cf_results)
    results_visualizer.save_imputer_evaluation_results_visualizations(imputater_results)
    results_visualizer.save_data_visualizations()


if __name__ == "__main__":
    main()
