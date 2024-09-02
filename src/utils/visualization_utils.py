import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ..evaluation.evaluation_results_container import (
    EvaluationResultsContainer,
)


def get_sns_palette() -> str:
    return "muted"


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


def save_metric_histograms(metrics_dict: dict, file_path: str):
    """Plots histograms of each counterfactual metric and saves image to given path.

    :param str file_path: file path for saving image
    """
    sns.set_palette(get_sns_palette())
    fig, axes = plt.subplots(2, 4, figsize=(19, 15))
    axes = axes.flatten()
    for i, (metric_name, metric_values) in enumerate(metrics_dict.items()):
        sns.histplot(metric_values, bins=30, ax=axes[i])
        axes[i].set_title(metric_name)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(file_path)


def save_imputation_type_results_per_missing_value_count_plot(
    all_results: EvaluationResultsContainer, file_path: str
):

    all_evaluation_params_dict = all_results.get_all_evaluation_params_dict()

    data = []
    for m in all_evaluation_params_dict["num_missing"]:
        for imputation_type in all_evaluation_params_dict["imputation_type"]:
            evaluation_obj = all_results.get_evaluation_for_params(
                {"imputation_type": imputation_type, "num_missing": m}
            )
            results_dict = evaluation_obj.get_counterfactual_plot_metrics()
            for k, v in results_dict.items():
                data.append({"m": m, "type": imputation_type, "metric": k, "value": v})

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

    # g.figure.suptitle("Experiment results", fontsize=16)
    g.figure.subplots_adjust(top=0.85, hspace=0.6)

    for ax in g.axes.flat:
        metric_name = ax.get_title().split(" = ")[-1]
        ax.set_title(f"{metric_name}", fontsize=14)
        ax.set_ylabel("Value")
        ax.set_xlabel("# missing values")

    for ax in g.axes.flat:
        ax.yaxis.set_tick_params(labelsize=10)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)

    g.figure.savefig(
        file_path,
        bbox_inches="tight",
        dpi=300,
    )
    # plt.show()


def save_imputation_type_results_per_feature_with_missing_value(
    all_results: EvaluationResultsContainer, predictor_names: list[str], file_path: str
):

    all_evaluation_params_dict = all_results.get_all_evaluation_params_dict()

    data = []

    feat_indices = all_evaluation_params_dict["ind_missing"]

    for f_ind in feat_indices:
        if f_ind == "random":
            f_name = "random"
        else:
            f_name = predictor_names[int(f_ind)]

        for imputation_type in all_evaluation_params_dict["imputation_type"]:
            evaluation_obj = all_results.get_evaluation_for_params(
                {
                    "imputation_type": imputation_type,
                    "ind_missing": f_ind,
                    "num_missing": 1,
                }
            )
            results_dict = evaluation_obj.get_counterfactual_plot_metrics()
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

    # g.figure.suptitle("Experiment results", fontsize=16)
    g.figure.subplots_adjust(top=0.85, hspace=0.6)

    for ax in g.axes.flat:
        metric_name = ax.get_title().split(" = ")[-1]
        ax.set_title(f"{metric_name}", fontsize=14)
        ax.set_ylabel("Value")
        ax.set_xlabel("Predictor with missing value")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

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
    # plt.show()
