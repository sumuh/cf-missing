import os
import sys
from datetime import datetime
from pathlib import Path
from .evaluation.evaluation_runner import EvaluationRunner
import warnings


# Todo: remove after fixing sklearn UserWarning: X has feature names, but MinMaxScaler was fitted without feature names
# (maybe fit with pd df)
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def main():
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = f"{current_file_path}/../config/config.yaml"
    current_time = datetime.now()
    formatted_time_day = current_time.strftime("%d-%m-%Y")
    formatted_time_sec = current_time.strftime("%d-%m-%Y-%H-%M-%S")
    results_dir = f"{current_file_path}/../evaluation_results/{formatted_time_day}/run_{formatted_time_sec}"
    Path(results_dir).mkdir(parents=True)

    evaluation_runner = EvaluationRunner(config_file_path, results_dir)
    evaluation_runner.run_evaluations()
    # evaluation_runner.save_visualizations_from_results_file(f"{current_file_path}/../evaluation_results/08-09-2024/run_08-09-2024-23-32-13/results.txt")


if __name__ == "__main__":
    main()
