import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
from typing import Union
from ..evaluation.evaluation_results_container import (
    EvaluationResultsContainer,
)
from ..utils.visualization_utils import (
    get_sns_palette,
    save_data_boxplots,
    save_data_histograms,
)
from ..utils.data_utils import Config


class ResultsVisualizer:

    def __init__(
        self,
        data: pd.DataFrame,
        results_dir: str,
        evaluation_config: Config,
    ):
        self.data = data
        self.results_dir = results_dir
        self.evaluation_config = evaluation_config
        self.predictor_names = self.data.columns.to_list()[:-1]

    def save_counterfactual_results_visualizations(
        self, all_results_container: EvaluationResultsContainer
    ):
        """Saves visualizations for counterfactual evaluations.

        :param EvaluationResultsContainer all_results_container: obj with all evaluation results
        """
        self._save_imputation_type_results_per_missing_value_count_plot(
            all_results_container,
            f"{self.results_dir}/imputation_type_results_per_missing_value_count.png",
        )
        self._save_imputation_type_results_per_feature_with_missing_value(
            all_results_container,
            f"{self.results_dir}/imputation_type_results_per_feature_with_missing_value.png",
        )

    def save_data_visualizations(self):
        """Saves visualizations related to data used."""
        save_data_histograms(self.data, f"{self.results_dir}/data_hists.png")
        save_data_boxplots(self.data, f"{self.results_dir}/data_boxplots.png")

    def save_imputer_evaluation_results_visualizations(
        self, imputation_results: dict[str, list], result_file_path: str
    ):
        """Saves plot comparing results of different imputation methods.

        :param dict[str, list] imputation_results: key: imputation method name, value: list of average error for each feature
        :param str result_file_path: file to save plot to
        """

        data = []
        for feat_ind in range(len(self.data.columns) - 1):
            feat_error_mean = imputation_results["mean"][feat_ind]
            multiple_results = filter(
                lambda key: key.split("_")[0] == "multiple", imputation_results.keys()
            )
            ns = [name.split("_")[1] for name in list(multiple_results)]
            for n in ns:
                n = int(n)
                data.append(
                    {
                        "type": "mean",
                        "n": n,
                        "error": feat_error_mean,
                        "feat_name": self.predictor_names[feat_ind],
                    }
                )
                feat_error_multiple = imputation_results[f"multiple_{n}"][feat_ind]
                data.append(
                    {
                        "type": "multiple",
                        "n": n,
                        "error": feat_error_multiple,
                        "feat_name": self.predictor_names[feat_ind],
                    }
                )

        df = pd.DataFrame(data)

        g = sns.catplot(
            data=df,
            x="feat_name",
            y="error",
            hue="type",
            col="n",
            kind="bar",
            palette=get_sns_palette(),
            sharex=False,
            sharey=False,
        )

        for ax in g.axes.flat:
            ax.set_title(ax.get_title().split(" = ")[-1])
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", fontsize=10
            )
            ax.set_xlabel("Predictor with missing value")

        g.figure.tight_layout()

        g.figure.savefig(
            result_file_path,
            bbox_inches="tight",
            dpi=300,
        )

    def save_imputation_samples_distribution_plot(
        self,
        results: dict[int, dict[str, Union[float, np.array]]],
        result_file_path: str,
    ):
        """Saves plot for the result of imputating a single value for one test row.
        Used for debugging, not for actual evaluation.

        :param dict[int, dict[str, Union[float, np.array]]] results: dict of results
        :param str result_file_path: file to save plot to
        """
        data = []
        for i in range(0, len(self.predictor_names)):
            results_for_feat = results[i]
            samples = results_for_feat["samples"]
            for sample in samples:
                data.append(
                    {
                        "sample_value": sample,
                        "feature": self.predictor_names[i],
                    }
                )

        df = pd.DataFrame(data)

        g = sns.catplot(
            data=df,
            x="sample_value",
            col="feature",
            palette=get_sns_palette(),
            ci=None,
            height=4,
            aspect=1.5,
            col_wrap=2,
            sharey=False,
            sharex=False,
        )

        g.figure.subplots_adjust(top=0.85, hspace=0.6)

        for i, ax in enumerate(g.axes.flat):
            results_for_feat = results[i]
            mu_value = results_for_feat["mu"]
            sigma_value = results_for_feat["sigma"]
            a = results_for_feat["a"]
            b = results_for_feat["b"]

            feature_samples = df[df["feature"] == self.predictor_names[i]][
                "sample_value"
            ]
            avg_sample_value = feature_samples.mean()

            ax.axvline(
                x=mu_value, color="red", linestyle="--", label=f"mu = {mu_value}"
            )
            ax.axvline(x=a, color="black", linestyle="--", label=f"a = {a}")
            ax.axvline(x=b, color="black", linestyle="--", label=f"b = {b}")

            ax.axvline(
                x=avg_sample_value,
                color="blue",
                linestyle="--",
                label=f"avg = {avg_sample_value:.2f}",
            )

            x_min, x_max = ax.get_xlim()
            x_values = np.linspace(x_min, x_max, 100)

            y_values = norm.pdf(x_values, loc=mu_value, scale=sigma_value)

            y_max = ax.get_ylim()[1]
            y_values_scaled = y_values * (y_max / max(y_values))

            ax.plot(x_values, y_values_scaled, color="green", label="Normal Dist.")

            ax.yaxis.set_tick_params(labelsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)

            ax.legend(loc="upper right", fontsize=10)

        for ax in g.axes.flat:
            ax.yaxis.set_tick_params(labelsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)

        g.figure.savefig(
            result_file_path,
            bbox_inches="tight",
            dpi=300,
        )

    def _get_pretty_title(self, metric_name: str) -> str:
        """Maps metric name to string representation used in plot title.

        :param str metric_name: metric name used as dict key
        :return str: pretty name
        """
        title_dict = {
            "avg_n_vectors": "Average # counterfactuals found",
            "avg_dist_from_original": "Average distance from original",
            "avg_diversity": "Average diversity",
            "avg_count_diversity": "Average count diversity",
            "avg_diversity_missing_values": "Average diversity (missing values)",
            "avg_count_diversity_missing_values": "Average count diversity (missing values)",
            "avg_sparsity": "Average sparsity",
            "avg_runtime_seconds": "Average runtime (s)",
            "coverage": "Coverage",
        }
        return title_dict[metric_name]

    def _save_imputation_type_results_per_missing_value_count_plot(
        self, all_results: EvaluationResultsContainer, file_path: str
    ):
        """Plots metrics of each counterfactual metric per number of missing values.

        :param EvaluationResultsContainer all_results: all results dictionary
        :param str file_path: path to save plot to
        """

        plot_metrics = [
            "avg_n_vectors",
            "avg_dist_from_original",
            "avg_diversity",
            "avg_count_diversity",
            "avg_diversity_missing_values",
            "avg_count_diversity_missing_values",
            "avg_sparsity",
            "avg_runtime_seconds",
            "coverage",
        ]

        all_evaluation_params_dict = all_results.get_all_evaluation_params_dict()

        data = []
        for m in range(1, len(self.predictor_names)):
            for imputation_type in all_evaluation_params_dict["imputation_type"]:
                evaluation_obj = all_results.get_evaluation_for_params(
                    {"imputation_type": imputation_type, "num_missing": m}
                )
                results_dict = {
                    k: v
                    for k, v in evaluation_obj.get_counterfactual_metrics().items()
                    if k in plot_metrics
                }
                for k, v in results_dict.items():
                    data.append(
                        {"m": m, "type": imputation_type, "metric": k, "value": v}
                    )

        df = pd.DataFrame(data)

        g = sns.catplot(
            data=df,
            x="m",
            y="value",
            hue="type",
            col="metric",
            kind="bar",
            palette=get_sns_palette(),
            ci=None,
            height=4,
            aspect=1.5,
            col_wrap=2,
            sharey=False,
            sharex=False,
        )

        g.figure.subplots_adjust(top=0.85, hspace=0.6)

        for ax in g.axes.flat:
            metric_name = ax.get_title().split(" = ")[-1]
            ax.set_title(self._get_pretty_title(metric_name), fontsize=14)
            ax.set_ylabel("")
            ax.set_xlabel("# missing values")

        for ax in g.axes.flat:
            ax.yaxis.set_tick_params(labelsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)

        g.figure.savefig(
            file_path,
            bbox_inches="tight",
            dpi=300,
        )

    def _save_imputation_type_results_per_feature_with_missing_value(
        self, all_results: EvaluationResultsContainer, file_path: str
    ):
        """Plots metrics of each counterfactual metric per feature with missing value.

        :param EvaluationResultsContainer all_results: all results dictionary
        :param str file_path: path to save plot to
        """

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

        all_evaluation_params_dict = all_results.get_all_evaluation_params_dict()

        data = []

        feat_indices = [i for i in range(len(self.predictor_names))]

        for f_ind in feat_indices:
            f_name = self.predictor_names[int(f_ind)]

            for imputation_type in all_evaluation_params_dict["imputation_type"]:
                evaluation_obj = all_results.get_evaluation_for_params(
                    {"imputation_type": imputation_type, "ind_missing": f_ind}
                )
                results_dict = {
                    k: v
                    for k, v in evaluation_obj.get_counterfactual_metrics().items()
                    if k in plot_metrics
                }
                for k, v in results_dict.items():
                    data.append(
                        {
                            "f_ind": f_ind,
                            "type": imputation_type,
                            "metric": k,
                            "value": v,
                            "f_name": f_name,
                        }
                    )

        df = pd.DataFrame(data)

        g = sns.catplot(
            data=df,
            x="f_name",
            y="value",
            hue="type",
            col="metric",
            kind="bar",
            palette=get_sns_palette(),
            ci=None,
            height=4,
            aspect=1.5,
            col_wrap=2,
            sharey=False,
            sharex=False,
        )

        g.figure.subplots_adjust(top=0.85, hspace=0.6)

        for ax in g.axes.flat:
            metric_name = ax.get_title().split(" = ")[-1]
            ax.set_title(self._get_pretty_title(metric_name), fontsize=14)
            ax.set_ylabel("")
            ax.set_xlabel("Predictor with missing value")
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", fontsize=10
            )

            if not ax.has_data():
                ax.remove()

        for ax in g.axes.flat:
            ax.yaxis.set_tick_params(labelsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)

        g.figure.tight_layout()

        g.figure.savefig(
            file_path,
            bbox_inches="tight",
            dpi=300,
        )
