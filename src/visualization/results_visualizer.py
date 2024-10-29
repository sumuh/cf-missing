import pandas as pd
import numpy as np
import seaborn as sns
import json
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Union
from ..evaluation.evaluation_results_container import (
    EvaluationResultsContainer,
)
from ..utils.visualization_utils import (
    get_sns_palette,
    get_sns_palette_alt,
    save_data_boxplots,
    save_data_histograms,
    get_pretty_title,
    get_plot_metrics_names,
    get_average_metric_values,
    get_gradient_genetic_colors,
    get_selection_algo_colors,
    get_imputation_type_colors,
    get_runtime_distribution_colors,
    get_custom_palette_colorbrewer_vibrant,
)
from ..utils.data_utils import Config
from ..utils.misc_utils import parse_results_file


class ResultsVisualizer:

    def __init__(self, results_file_path: str, visualizations_dir_path: str):
        data_stats, results_container = parse_results_file(results_file_path)
        self.results_container = results_container
        self.predictor_names = data_stats["column_names"][:-1]
        self.visualizations_dir_path = visualizations_dir_path

    def save_data_visualizations(self, data: pd.DataFrame):
        """Saves visualizations related to data used."""
        save_data_histograms(data, f"{self.results_dir}/data_hists.png")
        save_data_boxplots(data, f"{self.results_dir}/data_boxplots.png")

    def gradient_vs_genetic(self):
        """Saves plot comparing performance of gradient and genetic algorithms."""
        all_evaluation_params_dict = (
            self.results_container.get_all_evaluation_params_dict()
        )
        data = []

        for n in all_evaluation_params_dict["n"]:
            for classifier in all_evaluation_params_dict["classifier"]:
                results_for_all_missing_inds = []
                for ind_missing in all_evaluation_params_dict["ind_missing"]:
                    evaluation_obj = self.results_container.get_evaluation_for_params(
                        {
                            "classifier": classifier,
                            "imputation_type": "multiple",
                            "ind_missing": ind_missing,
                            "num_missing": None,
                            "n": n,
                            "k": 3,
                            "selection_alg": "naive",
                        }
                    )
                    results_dict = {
                        k: v
                        for k, v in evaluation_obj.get_counterfactual_metrics().items()
                        if k in get_plot_metrics_names()
                    }
                    results_dict[
                        "avg_runtime_seconds"
                    ] = evaluation_obj.get_counterfactual_metrics()["runtimes"][
                        "avg_total"
                    ]
                    results_for_all_missing_inds.append(results_dict)

                average_results = get_average_metric_values(
                    results_for_all_missing_inds
                )
                for k, v in average_results.items():
                    data.append(
                        {
                            "n": n,
                            "metric": k,
                            "value": v,
                            "classifier": evaluation_obj.get_params().classifier,
                        }
                    )

        df = pd.DataFrame(data)
        sns.set_theme()

        g = sns.FacetGrid(
            data=df,
            col="metric",
            col_wrap=3,
            sharex=False,
            sharey=False,
            height=4,
            legend_out=True,
            col_order=get_plot_metrics_names(),
        )

        g.map_dataframe(
            sns.lineplot,
            x="n",
            y="value",
            hue="classifier",
            style="classifier",
            palette=get_gradient_genetic_colors(),
            markers=["s", "o"],
            dashes=False,
            legend="brief",
        )

        g.figure.subplots_adjust(top=0.85, hspace=0.6)

        # Tune axes

        g.set(xticks=[1, 5, 10])
        # g.set(xlim=(1, 11))

        for ax in g.axes.flat:
            metric_name = ax.get_title().split(" = ")[-1]

            if metric_name == "avg_n_vectors":
                ax.set_ylim(ymax=3.1)

            if metric_name in [
                "coverage",
                "avg_count_diversity",
                "avg_count_diversity_missing_values",
                "avg_sparsity",
            ]:
                ax.set_ylim(ymax=1.1)

            if metric_name == "avg_runtime_seconds":
                ax.set_title("Average runtime (log seconds)", fontsize=14)
                ax.set_yscale("log")
            else:
                ax.set_title(get_pretty_title(metric_name), fontsize=14)

            ax.set_xlabel("n")
            ax.yaxis.set_tick_params(labelsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            ax.set_ylim(bottom=0)

        # Make legend

        legend_name_map = {"sklearn": "Genetic", "tensorflow": "Gradient"}
        legend = plt.legend()
        for text in legend.get_texts():
            classifier_name = text.get_text()
            text.set_text(legend_name_map[classifier_name])

        # Save figure

        g.figure.savefig(
            f"{self.visualizations_dir_path}/gradient_vs_genetic.png",
            bbox_inches="tight",
            dpi=300,
        )

        with open(f"{self.visualizations_dir_path}/gradient_vs_genetic.csv", "w") as f:
            f.write(df.to_csv())

    def multiple_imputation_effect_of_n(
        self,
    ):
        """Saves plot visualizing effect of different n values on multiple imputation."""
        all_evaluation_params_dict = (
            self.results_container.get_all_evaluation_params_dict()
        )
        data = []

        for n in all_evaluation_params_dict["n"]:
            results_for_all_missing_inds = []
            for ind_missing in all_evaluation_params_dict["ind_missing"]:
                evaluation_obj = self.results_container.get_evaluation_for_params(
                    {
                        "classifier": "sklearn",
                        "imputation_type": "multiple",
                        "ind_missing": ind_missing,
                        "num_missing": None,
                        "n": n,
                        "k": 3,
                        "selection_alg": "naive",
                    }
                )
                results_dict = {
                    k: v
                    for k, v in evaluation_obj.get_counterfactual_metrics().items()
                    if k in get_plot_metrics_names()
                }
                results_dict["avg_runtime_seconds"] = (
                    evaluation_obj.get_counterfactual_metrics()["runtimes"]["avg_total"]
                )
                results_for_all_missing_inds.append(results_dict)

            average_results = get_average_metric_values(results_for_all_missing_inds)
            for k, v in average_results.items():
                data.append(
                    {
                        "n": n,
                        "metric": k,
                        "value": v,
                    }
                )

        df = pd.DataFrame(data)
        sns.set_theme()
        g = sns.FacetGrid(
            data=df,
            col="metric",
            col_wrap=3,
            palette=get_sns_palette(),
            sharex=False,
            sharey=False,
            height=4,
            col_order=get_plot_metrics_names(),
        )

        g.map_dataframe(
            sns.lineplot,
            x="n",
            y="value",
            palette=get_sns_palette(),
            marker="s",
            dashes=False,
        )
        x_ticks = [1] + [*range(10, 101, 10)]
        g.set(xticks=x_ticks)

        g.set(xlim=(1, 101))

        g.figure.subplots_adjust(top=0.85, hspace=0.6)

        for ax in g.axes.flat:
            metric_name = ax.get_title().split(" = ")[-1]
            ax.set_title(get_pretty_title(metric_name), fontsize=12)
            ax.set_ylabel("")
            ax.set_xlabel("n")

            if not ax.has_data():
                ax.remove()

        for ax in g.axes.flat:
            ax.yaxis.set_tick_params(labelsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            ax.set_ylim(ymin=0)

        g.figure.savefig(
            f"{self.visualizations_dir_path}/multiple_imputation_effect_of_n.png",
            bbox_inches="tight",
            dpi=300,
        )

        with open(
            f"{self.visualizations_dir_path}/multiple_imputation_effect_of_n.csv", "w"
        ) as f:
            f.write(df.to_csv())

    def multiple_imputation_effect_of_selection_algo(self):
        """Saves plot visualizing effect of selection algorithm on multiple imputation,
        on different values of n.
        """
        all_evaluation_params_dict = (
            self.results_container.get_all_evaluation_params_dict()
        )
        data = []

        for n in all_evaluation_params_dict["n"]:
            for selection_alg in all_evaluation_params_dict["selection_alg"]:
                results_for_all_missing_inds = []
                for ind_missing in all_evaluation_params_dict["ind_missing"]:
                    evaluation_obj = self.results_container.get_evaluation_for_params(
                        {
                            "classifier": "sklearn",
                            "imputation_type": "multiple",
                            "ind_missing": ind_missing,
                            "num_missing": None,
                            "n": n,
                            "k": 3,
                            "selection_alg": selection_alg,
                        }
                    )
                    results_dict = {
                        k: v
                        for k, v in evaluation_obj.get_counterfactual_metrics().items()
                        if k in get_plot_metrics_names()
                    }
                    results_dict[
                        "avg_runtime_seconds"
                    ] = evaluation_obj.get_counterfactual_metrics()["runtimes"][
                        "avg_total"
                    ]
                    results_for_all_missing_inds.append(results_dict)

                # Obtain average for one missing feature
                average_results = get_average_metric_values(
                    results_for_all_missing_inds
                )
                # if n == 5 and selection_alg == "greedy":
                #    print(json.dumps(average_results, indent=2))
                for k, v in average_results.items():
                    data.append(
                        {
                            "metric": k,
                            "n": n,
                            "value": v,
                            "selection_alg": selection_alg,
                        }
                    )

        df = pd.DataFrame(data)
        sns.set_theme()
        g = sns.FacetGrid(
            data=df,
            col="metric",
            col_wrap=3,
            sharex=False,
            sharey=False,
            height=4,
            legend_out=True,
            col_order=get_plot_metrics_names(),
        )

        g.map_dataframe(
            sns.lineplot,
            x="n",
            y="value",
            hue="selection_alg",
            style="selection_alg",
            palette=get_selection_algo_colors(),
            markers=["s", "o"],
            dashes=False,
            legend="brief",
        )

        g.figure.subplots_adjust(top=0.85, hspace=0.6)

        # Tune axes

        x_ticks = [1] + [*range(10, 101, 10)]
        g.set(xticks=x_ticks)
        g.set(xlim=(1, 101))

        for ax in g.axes.flat:
            metric_name = ax.get_title().split(" = ")[-1]

            if metric_name == "avg_n_vectors":
                ax.set_ylim(ymax=3.1)

            if metric_name in [
                "coverage",
                "avg_count_diversity",
                "avg_count_diversity_missing_values",
                "avg_sparsity",
            ]:
                ax.set_ylim(ymax=1.1)

            ax.set_title(get_pretty_title(metric_name), fontsize=12)
            ax.set_xlabel("n")

            ax.yaxis.set_tick_params(labelsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            ax.set_ylim(ymin=0)

            if not ax.has_data():
                ax.remove()

        # Make legend

        plt.legend()

        # Save figure

        g.figure.savefig(
            f"{self.visualizations_dir_path}/multiple_imputation_effect_of_selection_algo.png",
            bbox_inches="tight",
            dpi=300,
        )

        with open(
            f"{self.visualizations_dir_path}/multiple_imputation_effect_of_selection_algo.csv",
            "w",
        ) as f:
            f.write(df.to_csv())

    def mean_vs_multiple_imputation_multiple_missing_values(self):
        """Saves plot comparing mean and multiple imputation for given n,
        for different number of missing values.
        """

        plot_metrics = get_plot_metrics_names()

        all_evaluation_params_dict = (
            self.results_container.get_all_evaluation_params_dict()
        )

        data = []
        for m in all_evaluation_params_dict["num_missing"]:
            for imputation_type in all_evaluation_params_dict["imputation_type"]:
                evaluation_obj = self.results_container.get_evaluation_for_params(
                    {
                        "classifier": "sklearn",
                        "imputation_type": imputation_type,
                        "ind_missing": None,
                        "num_missing": m,
                        "n": 20,
                        "k": 3,
                        "selection_alg": "greedy",
                    }
                )
                results_dict = {
                    k: v
                    for k, v in evaluation_obj.get_counterfactual_metrics().items()
                    if k in plot_metrics
                }
                results_dict["avg_runtime_seconds"] = (
                    evaluation_obj.get_counterfactual_metrics()["runtimes"]["avg_total"]
                )
                for k, v in results_dict.items():
                    data.append(
                        {
                            "m": m,
                            "imputation_type": imputation_type,
                            "metric": k,
                            "value": v,
                        }
                    )

        df = pd.DataFrame(data)

        sns.set_theme()
        g = sns.FacetGrid(
            data=df,
            col="metric",
            col_wrap=3,
            sharex=False,
            sharey=False,
            height=4,
            legend_out=True,
            col_order=get_plot_metrics_names(),
        )

        g.map_dataframe(
            sns.lineplot,
            x="m",
            y="value",
            hue="imputation_type",
            style="imputation_type",
            palette=get_imputation_type_colors(),
            markers=["s", "o"],
            dashes=False,
            legend="brief",
        )

        g.figure.subplots_adjust(top=0.85, hspace=0.6)

        # legend = g._legend
        # legend.set_title("Imputation type")

        # Tune axes

        x_ticks = [*range(1, 8)]
        g.set(xticks=x_ticks)

        g.set(xlim=(0.9, 7.1))

        for ax in g.axes.flat:
            metric_name = ax.get_title().split(" = ")[-1]

            ax.set_ylim(ymin=0.0)

            if metric_name == "avg_n_vectors":
                ax.set_ylim(ymax=3.1)

            if metric_name in [
                "coverage",
                "avg_count_diversity",
                "avg_count_diversity_missing_values",
                "avg_sparsity",
            ]:
                ax.set_ylim(ymax=1.1)

            ax.set_title(get_pretty_title(metric_name), fontsize=14)
            ax.set_xlabel("# missing values")
            ax.yaxis.set_tick_params(labelsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)

        # sns.move_legend(g, "upper center")

        # Make legend

        plt.legend()

        # Save figure

        g.figure.savefig(
            f"{self.visualizations_dir_path}/mean_vs_multiple_imputation_multiple_missing_values.png",
            bbox_inches="tight",
            dpi=300,
        )

        with open(
            f"{self.visualizations_dir_path}/mean_vs_multiple_imputation_multiple_missing_values.csv",
            "w",
        ) as f:
            f.write(df.to_csv())

    def mean_vs_multiple_imputation_single_missing_value(self):
        """Saves plot comparing mean and multiple imputation for given n,
        for different missing feature.
        """
        all_evaluation_params_dict = (
            self.results_container.get_all_evaluation_params_dict()
        )
        data = []

        for f_ind in all_evaluation_params_dict["ind_missing"]:
            f_name = self.predictor_names[int(f_ind)]
            for imputation_type in all_evaluation_params_dict["imputation_type"]:
                evaluation_obj = self.results_container.get_evaluation_for_params(
                    {
                        "classifier": "sklearn",
                        "imputation_type": imputation_type,
                        "ind_missing": f_ind,
                        "num_missing": None,
                        "n": 20,  # TODO: change to appropriate
                        "k": 3,
                        "selection_alg": "greedy",
                    }
                )
                results_dict = {
                    k: v
                    for k, v in evaluation_obj.get_counterfactual_metrics().items()
                    if k in get_plot_metrics_names()
                }
                results_dict["avg_runtime_seconds"] = (
                    evaluation_obj.get_counterfactual_metrics()["runtimes"]["avg_total"]
                )
                for k, v in results_dict.items():
                    data.append(
                        {
                            "f_ind": f_ind,
                            "imputation_type": imputation_type,
                            "metric": k,
                            "value": v,
                            "f_name": f_name,
                        }
                    )

        df = pd.DataFrame(data)
        sns.set_theme()
        g = sns.catplot(
            data=df,
            x="f_name",
            y="value",
            hue="imputation_type",
            col="metric",
            kind="bar",
            palette=get_imputation_type_colors(),
            ci=None,
            height=4,
            aspect=1.5,
            col_wrap=2,
            sharey=False,
            sharex=False,
            col_order=get_plot_metrics_names(),
        )

        g.figure.subplots_adjust(top=0.85, hspace=0.6)
        # legend = g._legend
        # legend.set_title("Imputation type")
        # sns.move_legend(g, "lower right")

        for ax in g.axes.flat:
            metric_name = ax.get_title().split(" = ")[-1]

            if metric_name == "avg_n_vectors":
                ax.set_ylim(ymax=3.1)

            if metric_name in [
                "coverage",
                "avg_count_diversity",
                "avg_count_diversity_missing_values",
                "avg_sparsity",
            ]:
                ax.set_ylim(ymax=1.1)

            ax.set_title(get_pretty_title(metric_name), fontsize=14)
            ax.yaxis.set_tick_params(labelsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            ax.set_xlabel("")
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", fontsize=10
            )

            if not ax.has_data():
                ax.remove()

        # Make legend

        plt.legend()

        # Save figure

        g.figure.tight_layout()

        g.figure.savefig(
            f"{self.visualizations_dir_path}/mean_vs_multiple_imputation_single_missing_value.png",
            bbox_inches="tight",
            dpi=300,
        )

        with open(
            f"{self.visualizations_dir_path}/mean_vs_multiple_imputation_single_missing_value.csv",
            "w",
        ) as f:
            f.write(df.to_csv())

    def mean_and_multiple_imputation_lambda_grid_search(self):
        """Saves plot comparing mean and multiple imputation for given n,
        for different number of missing values.
        """

        plot_metrics = get_plot_metrics_names()

        all_evaluation_params_dict = (
            self.results_container.get_all_evaluation_params_dict()
        )

        data = []
        for n in all_evaluation_params_dict["n"]:
            for imputation_type in all_evaluation_params_dict["imputation_type"]:
                for lambda_dist in all_evaluation_params_dict[
                    "distance_lambda_selection"
                ]:
                    for lambda_div in all_evaluation_params_dict[
                        "diversity_lambda_selection"
                    ]:
                        for lambda_spar in all_evaluation_params_dict[
                            "sparsity_lambda_selection"
                        ]:
                            results_for_all_missing_inds = []
                            for ind_missing in all_evaluation_params_dict[
                                "ind_missing"
                            ]:
                                evaluation_obj = (
                                    self.results_container.get_evaluation_for_params(
                                        {
                                            "classifier": "sklearn",
                                            "imputation_type": imputation_type,
                                            "ind_missing": ind_missing,
                                            "num_missing": None,
                                            "n": n,
                                            "k": 3,
                                            "selection_alg": "greedy",
                                            "distance_lambda_selection": lambda_dist,
                                            "diversity_lambda_selection": lambda_div,
                                            "sparsity_lambda_selection": lambda_spar,
                                        }
                                    )
                                )
                                results_dict = {
                                    k: v
                                    for k, v in evaluation_obj.get_counterfactual_metrics().items()
                                    if k in plot_metrics
                                }
                                results_dict[
                                    "avg_runtime_seconds"
                                ] = evaluation_obj.get_counterfactual_metrics()[
                                    "runtimes"
                                ][
                                    "avg_total"
                                ]
                                results_for_all_missing_inds.append(results_dict)

                            average_results = get_average_metric_values(
                                results_for_all_missing_inds
                            )
                            for k, v in average_results.items():
                                data.append(
                                    {
                                        "n": n,
                                        "imputation_type": imputation_type,
                                        "metric": k,
                                        "value": v,
                                        "Lambdas": f"dist={lambda_dist},div={lambda_div},spar={lambda_spar}",
                                    }
                                )

        df = pd.DataFrame(data)

        sns.set_theme()
        g = sns.FacetGrid(
            data=df,
            col="metric",
            col_wrap=3,
            sharex=False,
            sharey=False,
            height=4,
            legend_out=True,
            col_order=get_plot_metrics_names(),
        )

        g.map_dataframe(
            sns.lineplot,
            x="n",
            y="value",
            hue="Lambdas",
            style="imputation_type",
            palette=get_custom_palette_colorbrewer_vibrant(),
            # markers=["s", "o"],
            legend="brief",
        )

        # g.figure.subplots_adjust(top=0.85, hspace=0.6)
        g.set(xticks=[1, 5, 10, 15, 20])

        # Tune axes

        for ax in g.axes.flat:
            metric_name = ax.get_title().split(" = ")[-1]

            if metric_name in [
                "coverage",
                "avg_n_vectors",
                "avg_runtime_seconds",
            ]:
                ax.set_ylim(ymin=0.0)

            if metric_name == "avg_n_vectors":
                ax.set_ylim(ymax=3.1)

            if metric_name == "coverage":
                ax.set_ylim(ymax=1.1)

            ax.set_title(get_pretty_title(metric_name), fontsize=14)
            ax.set_xlabel("n")
            ax.yaxis.set_tick_params(labelsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)

        # Make legend

        plt.legend()

        # Save figure

        g.figure.savefig(
            f"{self.visualizations_dir_path}/lambda_grid_search.png",
            bbox_inches="tight",
            dpi=300,
        )

        with open(
            f"{self.visualizations_dir_path}/lambda_grid_search.csv",
            "w",
        ) as f:
            f.write(df.to_csv())

    def runtime_distributions_per_selection_alg_and_n(self):
        all_evaluation_params_dict = (
            self.results_container.get_all_evaluation_params_dict()
        )

        data_list = []
        ns = all_evaluation_params_dict["n"]
        selection_algs = all_evaluation_params_dict["selection_alg"]

        for alg in selection_algs:
            for n in ns:
                results_for_all_missing_inds = []
                for ind_missing in all_evaluation_params_dict["ind_missing"]:
                    evaluation_obj = self.results_container.get_evaluation_for_params(
                        {
                            "classifier": "sklearn",
                            "imputation_type": "multiple",
                            "ind_missing": ind_missing,
                            "num_missing": None,
                            "n": n,
                            "k": 3,
                            "selection_alg": alg,
                        }
                    )
                    runtime_results = {
                        k: v
                        for k, v in evaluation_obj.get_counterfactual_metrics()[
                            "runtimes"
                        ].items()
                        if not k == "avg_total"
                    }
                    results_for_all_missing_inds.append(runtime_results)

                average_results = get_average_metric_values(
                    results_for_all_missing_inds
                )
                data_item = average_results
                data_item["n"] = n
                data_item["selection_alg"] = alg
                data_list.append(data_item)

        names = {
            "avg_imputation": "Imputation",
            "avg_counterfactual_generation": "Counterfactual generation",
            "avg_filtering": "Filtering",
            "avg_selection": "Selection",
        }

        naive_data = [d for d in data_list if d["selection_alg"] == "naive"]
        greedy_data = [d for d in data_list if d["selection_alg"] == "greedy"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        palette = get_runtime_distribution_colors()
        sns.set_theme()

        def plot_data(ax, data, title):
            bar_positions = np.arange(len(ns))
            bar_width = 0.5

            for i, data_item in enumerate(data):
                bottom = 0

                for j, (key, value) in enumerate(data_item.items()):
                    if key == "n" or key == "selection_alg":
                        continue

                    ax.bar(
                        bar_positions[i],
                        value,
                        bottom=bottom,
                        color=palette[j],
                        width=bar_width,
                        label=names[key] if i == 0 else "",
                    )
                    bottom += value

            ax.set_xlabel("n")
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(ns)
            ax.set_title(title)

        plot_data(ax1, naive_data, "Runtime distribution (naive)")

        plot_data(ax2, greedy_data, "Runtime distribution (greedy)")

        ax1.set_ylabel("Runtime (s)")

        handles, labels = ax1.get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            title="",  # bbox_to_anchor=(1.05, 0.5),
        )

        # Adjust layout to leave space for the legend
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Make space on the right for the legend

        plt.savefig(
            f"{self.visualizations_dir_path}/runtime_distribution_naive_vs_greedy.png",
            bbox_inches="tight",
            dpi=500,
        )

        with open(
            f"{self.visualizations_dir_path}/runtime_distribution_naive_vs_greedy.csv",
            "w",
        ) as f:
            f.write(pd.DataFrame(data_list).to_csv())

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
