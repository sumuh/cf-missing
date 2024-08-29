import os
import json
import yaml
import sys
from datetime import datetime
from pathlib import Path
from .evaluation.evaluator import Evaluator
from .data_utils import (
    Config,
    load_data,
    explore_data,
    plot_metric_histograms,
    transform_target_to_binary_class,
    drop_rows_with_missing_values,
    get_data_metrics,
    save_data_histogram,
    get_str_from_dict_for_saving,
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
    results_main_dir: str,
    file_prefix: str,
    data_config: Config,
    evaluation_config: Config,
):
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

    if evaluation_config.show_plots:
        explore_data(data)

    data_metrics = get_data_metrics(data, data_config.target_name)
    print(data_metrics)

    # Evaluate counterfactual generation for each row in dataset and return average metrics
    evaluator = Evaluator(data, data_config, evaluation_config)
    classifier_metrics = evaluator.evaluate_classifier()
    print(classifier_metrics)
    histogram_dict, aggregated_results = evaluator.evaluate_counterfactual_generation()

    print(f"config\n{json.dumps(evaluation_config.get_dict(), indent=2)}")
    print(f"data metrics\n{json.dumps(data_metrics, indent=2)}")
    print(f"classifier metrics\n{json.dumps(classifier_metrics, indent=2)}")
    print(f"counterfactual metrics\n{json.dumps(aggregated_results, indent=2)}")

    if evaluation_config.save_results:
        # Create new dir for each test
        results_dir = f"{results_main_dir}/{file_prefix}"
        Path(results_dir).mkdir(parents=True)
        results_filename = f"{results_dir}/{file_prefix}.txt"

    if evaluation_config.save_results:
        metrics_histogram_file = f"{results_dir}/{file_prefix}-results-hist.png"
        data_histogram_file = f"{results_dir}/{file_prefix}-data-hist.png"
        plot_metric_histograms(
            histogram_dict,
            save_to_file=True,
            show=False,
            file_path=metrics_histogram_file,
        )
        save_data_histogram(data, data_histogram_file)
    elif evaluation_config.show_plots:
        plot_metric_histograms(histogram_dict, save_to_file=False, show=True)

    if evaluation_config.save_results:
        # Pretty print config and results to file
        with open(results_filename, "w") as results_file:
            data_config_str = get_str_from_dict_for_saving(
                data_config.get_dict(), "data config"
            )
            evaluation_config_str = get_str_from_dict_for_saving(
                evaluation_config.get_dict(), "evaluation config"
            )
            data_metrics_str = get_str_from_dict_for_saving(
                data_metrics, "data metrics"
            )
            classifier_metrics_str = get_str_from_dict_for_saving(
                classifier_metrics, "classifier metrics"
            )
            counterfactual_metrics_str = get_str_from_dict_for_saving(
                aggregated_results, "counterfactual metrics"
            )
            all_to_write = [
                data_config_str,
                evaluation_config_str,
                data_metrics_str,
                classifier_metrics_str,
                counterfactual_metrics_str,
            ]
            results_file.write("\n\n".join(all_to_write))


def main():
    print("Running evaluation...")
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    config_all = Config(load_config(f"{current_file_path}/../config/config.yaml"))
    data_config = Config(config_all.data.diabetes.get_dict())
    evaluation_config = Config(config_all.evaluation.get_dict())

    RUN_EVALUATION_LOOP = True

    current_time = datetime.now()
    formatted_time_day = current_time.strftime("%d-%m-%Y")
    formatted_time_sec = current_time.strftime("%d-%m-%Y-%H-%M-%S")
    results_dir = f"{current_file_path}/../evaluation_results/{formatted_time_day}/run_{formatted_time_sec}"

    if RUN_EVALUATION_LOOP:
        # Override config file options with looping different values
        num_missing_values_to_test = [2, 3]
        imputation_types_to_test = ["multiple", "mean"]
        for n in num_missing_values_to_test:
            evaluation_config.missing_data.number_of_missing = n
            for t in imputation_types_to_test:
                evaluation_config.imputation.type = t
                file_prefix = f"{n}_missing_{t}"
                print(
                    f"Running evaluation for {n} missing values, imputation type: {t}"
                )
                run_evaluation_with_config(
                    results_dir, file_prefix, data_config, evaluation_config
                )

    else:
        # Use params from config file
        file_prefix = f"test_run"
        run_evaluation_with_config(
            results_dir, file_prefix, data_config, evaluation_config
        )


if __name__ == "__main__":
    main()
