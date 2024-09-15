import json
import yaml
from ..utils.data_utils import Config


class SingleEvaluationResultsContainer:
    """Stores all relevant configurations and results
    for an evaluation run.
    Has utils for saving and displaying results.
    """

    def __init__(self, params: Config):
        self.params = params

    def get_params(self):
        return self.params

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
        with open(file_path, "r") as results_file:
            current_content = yaml.safe_load(results_file)
            if "runs" not in current_content:
                current_content["runs"] = []

        formatted_metrics = self.counterfactual_metrics.copy()
        for k, v in formatted_metrics.items():
            if k == "runtimes":
                for r_k, r_v in v.items():
                    formatted_metrics["runtimes"][r_k] = float(r_v)
            else:
                formatted_metrics[k] = float(v)

        run = {
            "run": {
                "run_params": self.params.get_dict(),
                "run_metrics": formatted_metrics,
            }
        }
        current_content["runs"].append(run)
        with open(file_path, "w") as file:
            yaml.dump(current_content, file, default_flow_style=False, sort_keys=False)


class EvaluationResultsContainer:

    def __init__(self, all_evaluation_params: Config):
        self.evaluations = []
        self.all_evaluation_params = all_evaluation_params

    def add_evaluation(self, evaluation: SingleEvaluationResultsContainer):
        self.evaluations.append(evaluation)

    def get_evaluations(self):
        return self.evaluations

    def get_evaluation_for_params(self, param_dict) -> SingleEvaluationResultsContainer:
        matches = False
        for evaluation in self.evaluations:
            for k, v in param_dict.items():
                if evaluation.get_params().get_dict()[k] == v:
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
