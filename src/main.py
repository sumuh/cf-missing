import os
import sys
from datetime import datetime
from pathlib import Path
from .evaluation.evaluation_runner import EvaluationRunner
from .visualization.results_visualizer import ResultsVisualizer
import warnings
import random
from .logging.cf_logger import CfLogger


# Todo: remove after fixing sklearn UserWarning: X has feature names, but MinMaxScaler was fitted without feature names
# (maybe fit with pd df)
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def run_evaluations(results_dir: str, config_file_path: str, logger: CfLogger) -> str:
    evaluation_runner = EvaluationRunner(config_file_path, results_dir, logger)
    evaluation_runner.run_evaluations()


def make_visualizations(results_file_path: str, visualizations_dir_path: str):
    results_visualizer = ResultsVisualizer(results_file_path, visualizations_dir_path)
    # results_visualizer.gradient_vs_genetic()
    # results_visualizer.multiple_imputation_effect_of_n()
    # results_visualizer.multiple_imputation_effect_of_selection_algo()
    results_visualizer.runtime_distributions_per_selection_alg_and_n()
    # results_visualizer.mean_vs_multiple_imputation_single_missing_value()
    # results_visualizer.mean_vs_multiple_imputation_multiple_missing_values()
    # results_visualizer.mean_and_multiple_imputation_lambda_grid_search()


def main():
    random.seed(12)
    current_time = datetime.now()
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = f"{current_file_path}/../config/config.yaml"

    # Make results directory
    formatted_time_day = current_time.strftime("%d-%m-%Y")
    formatted_time_sec = current_time.strftime("%d-%m-%Y-%H-%M-%S")
    results_dir = f"{current_file_path}/../evaluation_results/{formatted_time_day}/run_{formatted_time_sec}"
    Path(results_dir).mkdir(parents=True)

    # Init logger
    logger = CfLogger(True, results_dir)

    # Run evaluations
    run_evaluations(results_dir, config_file_path, logger)
    sys.exit()

    # Make visualizations
    results_dir = (
        f"{current_file_path}/../evaluation_results/09-10-2024/run_09-10-2024-18-24-13"
    )
    make_visualizations(f"{results_dir}/results.yaml", results_dir)


if __name__ == "__main__":
    main()
