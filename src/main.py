import os
import sys
from datetime import datetime
from pathlib import Path
from .evaluation.evaluation_runner import EvaluationRunner
from .visualization.results_visualizer import ResultsVisualizer
import warnings


# Todo: remove after fixing sklearn UserWarning: X has feature names, but MinMaxScaler was fitted without feature names
# (maybe fit with pd df)
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def run_evaluations(current_file_path: str) -> str:
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = f"{current_file_path}/../config/config.yaml"
    current_time = datetime.now()
    formatted_time_day = current_time.strftime("%d-%m-%Y")
    formatted_time_sec = current_time.strftime("%d-%m-%Y-%H-%M-%S")
    results_dir = f"{current_file_path}/../evaluation_results/{formatted_time_day}/run_{formatted_time_sec}"
    Path(results_dir).mkdir(parents=True)

    evaluation_runner = EvaluationRunner(config_file_path, results_dir)
    evaluation_runner.run_evaluations()
    return results_dir


def make_visualizations(
    current_file_path: str, results_file_path: str, visualizations_dir_path: str
):
    results_visualizer = ResultsVisualizer(results_file_path, visualizations_dir_path)
    results_visualizer.save_runtime_distribution_plot()


def main():
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    results_dir = run_evaluations(current_file_path)
    # results_dir = f"{current_file_path}/../evaluation_results/14-09-2024/run_14-09-2024-20-44-45"
    # print(results_dir)
    # make_visualizations(current_file_path, f"{results_dir}/results.yaml", results_dir)


if __name__ == "__main__":
    main()
