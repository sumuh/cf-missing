import json
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

    def set_counterfactual_histogram_dict(self, histogram_dict: dict):
        self.counterfactual_histogram_dict = histogram_dict

    def get_counterfactual_histogram_dict(self):
        return self.counterfactual_histogram_dict

    def append_results_to_file(self, file_path: str):
        params_str = get_str_from_dict(
            self.evaluation_config.get_dict()["current_params"], "evaluation parameters"
        )
        results_str = get_str_from_dict(self.counterfactual_metrics, "results")
        with open(file_path, "a") as file:
            file.write(params_str + "\n" + results_str + "\n\n")


class EvaluationResultsContainer:

    def __init__(self, all_evaluation_params: Config):
        self.evaluations = []
        self.all_evaluation_params = all_evaluation_params

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

    def get_all_evaluation_params(self) -> Config:
        return self.all_evaluation_params

    def get_all_evaluation_params_dict(self) -> dict:
        return self.all_evaluation_params.get_dict()

    def set_data_metrics(self, data_metrics: dict):
        self.data_metrics = data_metrics

    def get_data_metrics(self) -> dict:
        return self.data_metrics

    def set_runtime(self, runtime: float):
        self.runtime = runtime

    def get_runtime(self) -> float:
        return self.runtime

    def save_stats_to_file(self, file_path: str):
        with open(file_path, "w") as file:
            config_dict = self.get_all_evaluation_params_dict()
            config_dict.update({"total_runtime": f"{self.get_runtime() / 60} minutes"})
            str_to_write = get_str_from_dict(config_dict, "evaluation parameters")
            file.write(str_to_write)

    def save_all_results_to_file(self, file_path: str):
        for evaluation in self.evaluations:
            evaluation.append_results_to_file(file_path)
