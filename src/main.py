import os
import json
import sys
from datetime import datetime
from pathlib import Path
from .evaluation.evaluator import Evaluator
from .data_utils import (
    load_data,
    explore_data,
    get_diabetes_dataset_config,
    plot_metric_histograms,
    transform_target_to_binary_class,
)
from .constants import *


# Todo: remove after fixing sklearn UserWarning: X has feature names, but MinMaxScaler was fitted without feature names
# (maybe fit with pd df)
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def main():
    # Enable saving evaluation results to file, if False will only print to console
    SAVE_RESULTS = False
    # Enable debug/verbose mode
    DEBUG = False
    # Show plots while running
    SHOW_PLOTS = False
    print("Running evaluation...")

    if SAVE_RESULTS:
        current_time = datetime.now()
        formatted_time_day = current_time.strftime("%d-%m-%Y")
        formatted_time_sec = current_time.strftime("%d-%m-%Y-%H-%M-%S")
        results_dir = f"{os.path.dirname(os.path.realpath(__file__))}/../evaluation_results/{formatted_time_day}/{formatted_time_sec}"
        # Create new dir for each test
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        results_filename = f"{results_dir}/results-{formatted_time_sec}.txt"

    # Choose and load dataset to use
    data_config = get_diabetes_dataset_config()
    data = load_data(data_config[config_file_path], data_config[config_separator])

    print(f"rows originally: {len(data)}")
    # Drop rows with missing values
    data = data[
        (data.Glucose != 0)
        & (data.BloodPressure != 0)
        & (data.SkinThickness != 0)
        & (data.Insulin != 0)
        & (data.BMI != 0)
    ]
    print(f"rows after drop: {len(data)}")

    if SHOW_PLOTS:
        explore_data(data)

    # For now multiclass classification is not supported so convert to binary class if needed
    if data_config[config_multiclass_target]:
        data = transform_target_to_binary_class(
            data,
            data_config[config_target_name],
            data_config[config_multiclass_threshold],
        )

    evaluation_config = {
        config_classifier: config_logistic_regression,
        config_missing_data_mechanism: config_MCAR,
        config_debug: DEBUG,
        config_number_of_missing_values: 3,  # only used if MCAR
    }
    evaluation_config.update(data_config)

    # Evaluate counterfactual generation for each row in dataset and return average metrics
    evaluator = Evaluator(data, evaluation_config)
    histogram_dict, aggregated_results = evaluator.perform_evaluation()

    print(f"config\n{json.dumps(evaluation_config, indent=2)}")
    print(f"results\n{json.dumps(aggregated_results, indent=2)}")

    if SAVE_RESULTS:
        histogram_file = f"{results_dir}/results-hist-{formatted_time_sec}.png"
        plot_metric_histograms(histogram_dict, True, histogram_file)
    elif SHOW_PLOTS:
        plot_metric_histograms(histogram_dict, False)

    if SAVE_RESULTS:
        # Pretty print config and results to file
        with open(results_filename, "w") as results_file:
            config_str = "config\n" + "\n".join(
                [f"{item[0]}\t{item[1]}" for item in evaluation_config.items()]
            )
            results_str = "results\n" + "\n".join(
                [f"{item[0]}\t{item[1]}" for item in aggregated_results.items()]
            )
            results_file.write(f"{config_str}\n\n{results_str}")


if __name__ == "__main__":
    main()
