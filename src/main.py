import os
import json
import yaml
import sys
import itertools
from datetime import datetime
from pathlib import Path
from .evaluation.evaluator import Evaluator
from .evaluation.evaluation_results_container import (
    EvaluationResultsContainer,
    SingleEvaluationResultsContainer,
)
from .utils.data_utils import (
    Config,
    load_data,
    transform_target_to_binary_class,
    drop_rows_with_missing_values,
    get_data_metrics,
)
from .utils.visualization_utils import (
    explore_data,
    plot_imputation_type_results_per_missing_value_count,
)


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


def run_evaluation_with_config(
    data_config: Config,
    evaluation_config: Config,
) -> dict:
    """Runs evaluation for config settings and writes results to file if so specified.

    :param str results_main_dir: directory for results
    :param str file_prefix: prefix for results file that contains info of important run parameters
    :param Config data_config: data configuration
    :param Config evaluation_config: evaluation configuration
    """

    # Choose and load dataset to use
    data = load_data(data_config.file_path, data_config.separator)

    # For now multiclass classification is not supported so convert to binary class if needed
    if data_config.multiclass_target:
        data = transform_target_to_binary_class(
            data,
            data_config.target_name,
            data_config.multiclass_threshold,
        )

    if data_config.dataset_name == "Pima Indians Diabetes":
        data = drop_rows_with_missing_values(data)

    # data = data[:50]

    results_container = SingleEvaluationResultsContainer(
        data_config, evaluation_config, data
    )

    if evaluation_config.show_plots:
        explore_data(data)

    data_metrics = get_data_metrics(data, data_config.target_name)
    results_container.set_data_metrics(data_metrics)

    # Evaluate counterfactual generation for each row in dataset and return average metrics
    evaluator = Evaluator(data, data_config, evaluation_config)
    classifier_metrics = evaluator.evaluate_classifier()
    results_container.set_classifier_metrics(classifier_metrics)
    histogram_dict, aggregated_results = evaluator.evaluate_counterfactual_generation()
    results_container.set_counterfactual_metrics(aggregated_results)
    results_container.set_counterfactual_histogram_dict(histogram_dict)

    return results_container


def main():
    print("Running evaluation...")
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    config_all = Config(load_config(f"{current_file_path}/../config/config.yaml"))
    data_config = Config(config_all.data.diabetes.get_dict())
    evaluation_config_dict = config_all.evaluation.get_dict()

    current_time = datetime.now()
    formatted_time_day = current_time.strftime("%d-%m-%Y")
    formatted_time_sec = current_time.strftime("%d-%m-%Y-%H-%M-%S")
    results_dir = f"{current_file_path}/../evaluation_results/{formatted_time_day}/run_{formatted_time_sec}"

    params_to_vary_dict = evaluation_config_dict["params"]
    params_to_vary_names = params_to_vary_dict.keys()

    # Object to store all evaluation results in
    all_results_container = EvaluationResultsContainer(params_to_vary_dict)
    lists = params_to_vary_dict.values()
    param_combinations = itertools.product(*lists)

    # Loop combinations of params specified in config.yaml
    for param_combination in param_combinations:
        current_params_dict = dict(zip(params_to_vary_names, param_combination))
        # Set current_params as the current combinations
        for k, v in current_params_dict.items():
            evaluation_config_dict["current_params"][k] = v
        print(
            f"Running evaluation for params: {json.dumps(current_params_dict, indent=2)}"
        )
        evaluation_results = run_evaluation_with_config(
            data_config, Config(evaluation_config_dict)
        )
        evaluation_results.pretty_print_all()
        evaluation_results.set_evaluation_params_dict(current_params_dict)
        all_results_container.add_evaluation(evaluation_results)

    Path(results_dir).mkdir(parents=True)
    plot_imputation_type_results_per_missing_value_count(
        all_results_container, results_dir
    )


if __name__ == "__main__":
    main()
