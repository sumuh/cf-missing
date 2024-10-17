import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from ..evaluation.evaluation_results_container import (
    EvaluationResultsContainer,
    SingleEvaluationResultsContainer,
)


def get_gradient_genetic_colors() -> list[tuple]:
    return [
        get_custom_palette_colorbrewer_vibrant()[0],
        get_custom_palette_colorbrewer_vibrant()[1],
    ]


def get_selection_algo_colors() -> list[tuple]:
    return [
        get_custom_palette_colorbrewer_vibrant()[2],
        get_custom_palette_colorbrewer_vibrant()[3],
    ]


def get_imputation_type_colors() -> list[tuple]:
    return [
        # get_custom_palette_colorbrewer_vibrant()[4],
        get_custom_palette_colorbrewer_vibrant_mega()[4],
        get_custom_palette_colorbrewer_vibrant_mega()[5],
    ]


def get_runtime_distribution_colors() -> list[tuple]:
    return [
        get_custom_palette_colorbrewer_pastel()[1],
        get_custom_palette_colorbrewer_pastel()[0],
        get_custom_palette_colorbrewer_pastel()[3],
        get_custom_palette_colorbrewer_pastel()[2],
    ]


def get_custom_palette_colorbrewer_vibrant() -> list[tuple]:
    return [
        (252 / 255, 141 / 255, 98 / 255),
        (102 / 255, 194 / 255, 165 / 255),
        (141 / 255, 160 / 255, 203 / 255),
        (231 / 255, 138 / 255, 195 / 255),
        (166 / 255, 216 / 255, 84 / 255),
        (255 / 255, 217 / 255, 47 / 255),
        (229 / 255, 196 / 255, 148 / 255),
        (179 / 255, 179 / 255, 179 / 255),
    ]


def get_custom_palette_colorbrewer_vibrant_mega() -> list[tuple]:
    return [
        (141 / 255, 211 / 255, 199 / 255),
        (255 / 255, 255 / 255, 179 / 255),
        (190 / 255, 186 / 255, 218 / 255),
        (251 / 255, 128 / 255, 114 / 255),
        (128 / 255, 177 / 255, 211 / 255),
        (253 / 255, 180 / 255, 98 / 255),
        (179 / 255, 222 / 255, 105 / 255),
        (252 / 255, 205 / 255, 229 / 255),
        (217 / 255, 217 / 255, 217 / 255),
        (188 / 255, 128 / 255, 189 / 255),
        (204 / 255, 235 / 255, 197 / 255),
        (255 / 255, 237 / 255, 111 / 255),
    ]


def get_custom_palette_colorbrewer_pastel() -> list[tuple]:
    return [
        (179 / 255, 226 / 255, 205 / 255),
        (253 / 255, 205 / 255, 172 / 255),
        (203 / 255, 213 / 255, 232 / 255),
        (244 / 255, 202 / 255, 228 / 255),
        (230 / 255, 245 / 255, 201 / 255),
        (255 / 255, 242 / 255, 174 / 255),
        (241 / 255, 226 / 255, 204 / 255),
        (204 / 255, 204 / 255, 204 / 255),
    ]


def get_sns_palette() -> str:
    return "muted"


def get_sns_palette_alt() -> str:
    palette_alt = sns.color_palette(get_sns_palette())
    first = palette_alt[0]
    second = palette_alt[1]
    third = palette_alt[2]
    fourth = palette_alt[3]
    palette_alt[0] = second
    palette_alt[1] = first
    palette_alt[2] = fourth
    palette_alt[3] = third
    return palette_alt


def get_pretty_title(metric_name: str) -> str:
    """Maps metric name to string representation used in plot title.
    :param str metric_name: metric name used as dict key
    :return str: pretty name
    """
    title_dict = {
        "avg_n_vectors": "Size",
        "avg_dist_from_original": "Distance from original",
        "avg_diversity": "Diversity",
        "avg_count_diversity": "Count diversity",
        "avg_diversity_missing_values": "Diversity within missing feature(s)",
        "avg_count_diversity_missing_values": "Count diversity within missing feature(s)",
        "avg_sparsity": "Sparsity",
        "avg_runtime_seconds": "Runtime (seconds)",
        "coverage": "Coverage",
    }
    return title_dict[metric_name]


def get_plot_metrics_names():
    return [
        "avg_dist_from_original",
        "avg_diversity",
        "avg_count_diversity",
        "avg_diversity_missing_values",
        "avg_count_diversity_missing_values",
        "avg_sparsity",
        "avg_runtime_seconds",
        "avg_n_vectors",
        "coverage",
    ]


def show_correlation_matrix(data: pd.DataFrame):
    """Plots correlation matrix.

    :param pd.DataFrame data: data
    """
    f = plt.figure(figsize=(12, 10))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(
        range(data.select_dtypes(["number"]).shape[1]),
        data.select_dtypes(["number"]).columns,
        fontsize=14,
        rotation=45,
    )
    plt.yticks(
        range(data.select_dtypes(["number"]).shape[1]),
        data.select_dtypes(["number"]).columns,
        fontsize=14,
    )
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()


def explore_data(data: pd.DataFrame):
    """Print auto-generated summaries and plots for dataframe.

    :param pd.DataFrame data: dataset to summarize
    """
    symbol = "="
    print(f"{symbol*12} Data exploration {symbol*12}")
    print("\nColumn types:\n")
    print(data.info())
    print("\nStatistics:\n")
    print(data.describe())
    print("\nCorrelations:")
    print(data.corr())
    data.hist()
    plt.show()
    show_correlation_matrix(data)
    print(f"{symbol*42}")
    len_negative = len(data[data["Outcome"] == 0])
    len_positive = len(data[data["Outcome"] == 1])
    print(f"Outcome 0: {len_negative}")
    print(f"Outcome 1: {len_positive}")


def save_data_histograms(data: pd.DataFrame, file_path: str):
    """Plots histograms of each feature in data and saves image to given path.

    :param pd.DataFrame data: dataframe
    :param str file_path: file path for saving image
    """
    sns.set_palette(get_sns_palette())
    data_long = data.melt(var_name="Feature", value_name="Value")
    print(data_long)

    # fig, axes = plt.subplots(2, 2)
    # sns.histplot(data=data, x="Pregnancies")
    g = sns.FacetGrid(
        data_long,
        col="Feature",
        col_wrap=2,
        height=3.5,
        aspect=1.5,
        sharex=False,
        sharey=False,
    )
    g.map(sns.histplot, "Value")
    g.set_titles("{col_name}")

    for ax in g.axes.flat:
        feature = ax.get_title()

        if feature == "Pregnancies":
            ax.set_xticks(np.arange(0, 18, 1))
            ax.set_xlim(-0.5, 18)

        if feature == "Outcome":
            ax.set_xticks([0, 1])
            ax.set_xlim(-0.5, 1.5)

        # Ensure y-axis only shows integer labels
        ax.yaxis.get_major_locator().set_params(integer=True)
    # plt.show()
    plt.savefig(file_path)


def save_data_boxplots(data: pd.DataFrame, file_path: str):
    """Plots boxplots of each feature in data and saves image to given path.

    :param pd.DataFrame data: dataframe
    :param str file_path: file path for saving image
    """
    sns.set_palette(get_sns_palette())

    data_predictors_only = data.iloc[:, :-1]
    data_long = data_predictors_only.melt(var_name="Feature", value_name="Value")

    g = sns.FacetGrid(
        data_long,
        col="Feature",
        col_wrap=2,
        height=3.5,
        aspect=1.5,
        sharex=False,
        sharey=False,
    )
    g.map(sns.boxplot, "Value", orient="v")
    g.set_titles("{col_name}")

    plt.savefig(file_path)


def get_average_metric_values(
    metrics_dict_list: list[dict[str, float]]
) -> dict[str, float]:
    """Returns average values from list of metric dictionaries.

    :param list[dict[str, float]] metrics_dict_list: dictionaries of metrics
    :return dict[str, float]: averages of given list
    """
    sum_dict = defaultdict(float)
    for metrics_dict in metrics_dict_list:
        for k, v in metrics_dict.items():
            sum_dict[k] += v

    return {k: sum_dict[k] / len(metrics_dict_list) for k in sum_dict.keys()}
