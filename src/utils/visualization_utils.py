import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from ..evaluation.evaluation_results_container import (
    EvaluationResultsContainer,
    SingleEvaluationResultsContainer,
)


def get_gradient_genetic_colors() -> list[tuple]:
    return [
        (216 / 255, 172 / 255, 174 / 255),
        (141 / 255, 211 / 255, 199 / 255),
    ]


def get_naive_greedy_colors() -> list[tuple]:
    return [(190, 186, 218), (255, 237, 111)]


def get_mean_multiple_colors() -> list[tuple]:
    return [
        # (253, 180, 98),
        (204, 235, 197),
        (128, 177, 211),
    ]


def get_runtime_distribution_colors() -> list[tuple]:
    return [
        (166 / 255, 206 / 255, 227 / 255),
        (31 / 255, 120 / 255, 180 / 255),
        (178 / 255, 223 / 255, 138 / 255),
        (51 / 255, 160 / 255, 44 / 255),
    ]


def get_custom_palette_colorbrewer_vibrant() -> list[tuple]:
    return [
        (102 / 255, 194 / 255, 165 / 255),
        (252 / 255, 141 / 255, 98 / 255),
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
        "avg_n_vectors": "Average size",
        "avg_dist_from_original": "Average distance from original",
        "avg_diversity": "Average diversity",
        "avg_count_diversity": "Average count diversity",
        "avg_diversity_missing_values": "Average diversity (missing values)",
        "avg_count_diversity_missing_values": "Average count diversity (missing values)",
        "avg_sparsity": "Average sparsity",
        "avg_runtime_seconds": "Average runtime (seconds)",
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


def save_data_histograms(data: pd.DataFrame, file_path: str):
    """Plots histograms of each feature in data and saves image to given path.

    :param pd.DataFrame data: dataframe
    :param str file_path: file path for saving image
    """
    sns.set_palette(get_sns_palette())
    data_long = data.melt(var_name="Feature", value_name="Value")
    g = sns.FacetGrid(
        data_long,
        col="Feature",
        col_wrap=2,
        height=3.5,
        aspect=1.5,
        sharex=False,
        sharey=False,
    )
    g.map(sns.histplot, "Value", bins=30, kde=False)
    g.set_titles("{col_name}")

    for ax in g.axes.flat:
        feature = ax.get_title()

        if feature == "Pregnancies":
            pass
            # ax.set_xticks(np.arange(0, 18, 2))  # Set x-ticks from 0 to 17 with a step of 1
            # Align bins with integer values, including 17
            # min_val = 0
            # max_val = 18
            # n_bins = 18
            # val_width = max_val - min_val
            # bin_width = 1
            # ax.set_xticks(np.arange(min_val-bin_width/2, max_val+bin_width/2, bin_width))
            # ax.set_xlim(-0.5, 17.5)

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
