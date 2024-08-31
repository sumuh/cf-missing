import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ..utils.data_utils import Config


class SingleEvaluationResultsContainer:
    """Stores all relevant configurations and results
    for an evaluation run.
    Has utils for saving and displaying results.
    """

    def __init__(self, data_config, evaluation_config, data_pd):
        self.data_config = data_config
        self.evaluation_config = evaluation_config
        self.data_pd = data_pd

    def _save_metric_histograms(self, file_path: str):
        """Plots histograms of each counterfactual metric and saves image to given path.

        :param str file_path: file path for saving image
        """
        fig, axes = plt.subplots(2, 4, figsize=(19, 15))
        axes = axes.flatten()
        for i, (metric_name, metric_values) in enumerate(
            self.counterfactual_metrics.items()
        ):
            axes[i].hist(metric_values)
            axes[i].set_title(metric_name)

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(file_path)

    def _save_data_histograms(data: pd.DataFrame, file_path: str = None):
        """Plots histograms of each feature in data and saves image to given path.

        :param pd.DataFrame data: dataframe
        :param str file_path: file path for saving image, defaults to None
        """
        data.hist(grid=False, figsize=(19, 15))
        plt.savefig(file_path)

    def _get_str_from_dict_for_saving(self, dict_to_save: dict, dict_name: str) -> str:
        """Utility method for generating string based on dictionary.

        :param dict dict_to_save: dictionary to stringify
        :param str dict_name: name of dictionary
        :return str: string representation of dict name and contents
        """
        s = "-"
        title = f"{s*12} {dict_name} {s*12}\n"
        content = "\n".join([f"{item[0]}\t{item[1]}" for item in dict_to_save.items()])
        return title + content

    def pretty_print_all(self):
        print(f"config\n{json.dumps(self.evaluation_config.get_dict(), indent=2)}")
        print(f"data metrics\n{json.dumps(self.data_metrics, indent=2)}")
        print(f"classifier metrics\n{json.dumps(self.classifier_metrics, indent=2)}")
        print(
            f"counterfactual metrics\n{json.dumps(self.counterfactual_metrics, indent=2)}"
        )

    def save_all_to_dir(self, dir_name: str):
        metrics_histogram_file = f"{dir_name}/results-hist.png"
        data_histogram_file = f"{dir_name}/data-hist.png"
        self._save_metric_histograms(
            self.counterfactual_histogram_dict,
            metrics_histogram_file,
        )
        self._save_data_histograms(self.data_pd, data_histogram_file)
        Path(dir_name).mkdir(parents=True)
        text_results_filename = f"{dir_name}/results.txt"
        with open(text_results_filename, "w") as results_file:
            data_config_str = self._get_str_from_dict_for_saving(
                self.data_config.get_dict(), "data config"
            )
            evaluation_config_str = self._get_str_from_dict_for_saving(
                self.evaluation_config.get_dict(), "evaluation config"
            )
            data_metrics_str = self._get_str_from_dict_for_saving(
                self.data_metrics, "data metrics"
            )
            classifier_metrics_str = self._get_str_from_dict_for_saving(
                self.classifier_metrics, "classifier metrics"
            )
            counterfactual_metrics_str = self._get_str_from_dict_for_saving(
                self.counterfactual_metrics, "counterfactual metrics"
            )
            all_to_write = [
                data_config_str,
                evaluation_config_str,
                data_metrics_str,
                classifier_metrics_str,
                counterfactual_metrics_str,
            ]
            results_file.write("\n\n".join(all_to_write))

    def set_evaluation_params_dict(self, params_dict: dict):
        self.evaluation_params_dict = params_dict

    def get_evaluation_params_dict(self):
        return self.evaluation_params_dict

    def set_data_metrics(self, data_metrics: dict):
        self.data_metrics = data_metrics

    def get_data_metrics(self):
        return self.data_metrics

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
