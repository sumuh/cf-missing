import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ..utils.data_utils import Config, get_str_from_dict


class SingleEvaluationResultsContainer:
    """Stores all relevant configurations and results
    for an evaluation run.
    Has utils for saving and displaying results.
    """

    def __init__(self, evaluation_config, data_pd):
        self.evaluation_config = evaluation_config
        self.data_pd = data_pd

    def set_evaluation_params_dict(self, params_dict: dict):
        self.evaluation_params_dict = params_dict

    def get_evaluation_params_dict(self):
        return self.evaluation_params_dict

    def get_data_config(self):
        return self.data_config

    def get_evaluation_config(self):
        return self.evaluation_config

    def set_classifier_metrics(self, classifier_metrics: dict):
        self.classifier_metrics = classifier_metrics

    def get_classifier_metrics(self):
        return self.classifier_metrics

    def set_counterfactual_metrics(self, counterfactual_metrics: dict):
        self.counterfactual_metrics = counterfactual_metrics

    def get_counterfactual_metrics(self):
        return self.counterfactual_metrics

    def get_counterfactual_plot_metrics(self):
        plot_metrics = [
            "avg_n_vectors",
            "avg_dist_from_original",
            "avg_diversity",
            "avg_count_diversity",
            "avg_diversity_missing_values",
            "avg_count_diversity_missing_values",
            "avg_sparsity",
            "avg_runtime_seconds",
        ]
        return {
            k: v for k, v in self.counterfactual_metrics.items() if k in plot_metrics
        }

    def set_counterfactual_histogram_dict(self, histogram_dict: dict):
        self.counterfactual_histogram_dict = histogram_dict

    def get_counterfactual_histogram_dict(self):
        return self.counterfactual_histogram_dict


class EvaluationResultsContainer:

    def __init__(self, all_evaluation_params_dict: dict[str, list]):
        self.evaluations = []
        self.all_evaluation_params_dict = all_evaluation_params_dict

    def add_evaluation(self, evaluation: SingleEvaluationResultsContainer):
        self.evaluations.append(evaluation)

    def get_evaluations(self):
        return self.evaluations

    def get_evaluation_for_params(self, param_dict):
        matches = False
        for evaluation in self.evaluations:
            for k, v in param_dict.items():
                if evaluation.get_evaluation_params_dict()[k] == v:
                    matches = True
                else:
                    matches = False
                    break
            if matches:
                return evaluation
        raise RuntimeError(f"Did not find evaluation for params {param_dict}")

    def get_all_evaluation_params_dict(self):
        return self.all_evaluation_params_dict

    def set_data_metrics(self, data_metrics: dict):
        self.data_metrics = data_metrics

    def get_data_metrics(self) -> dict:
        return self.data_metrics

    def save_stats_to_file(self, file_path: str, runtime: float):
        with open(file_path, "w") as file:
            config_dict = self.all_evaluation_params_dict
            config_dict.update({"total_runtime": f"{runtime / 60} minutes"})
            str_to_write = get_str_from_dict(config_dict, "evaluation parameters")
            file.write(str_to_write)