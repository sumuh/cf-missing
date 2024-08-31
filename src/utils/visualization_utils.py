import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ..evaluation.evaluation_results_container import (
    EvaluationResultsContainer,
    SingleEvaluationResultsContainer,
)


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


def plot_imputation_type_results_per_missing_value_count(
    all_results: EvaluationResultsContainer, dir_name: str
):

    all_evaluation_params_dict = all_results.get_all_evaluation_params_dict()
    print("all_evaluation_params_dict")
    print(all_evaluation_params_dict)

    data = []
    for m in all_evaluation_params_dict["num_missing"]:
        for imputation_type in all_evaluation_params_dict["imputation_type"]:
            evaluation_obj = all_results.get_evaluation_for_params(
                {"imputation_type": imputation_type, "num_missing": m}
            )
            results_dict = evaluation_obj.get_counterfactual_metrics()
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
        palette="viridis",
        ci=None,
        height=4,
        aspect=1.5,
        col_wrap=4,
        sharey=False,
    )

    g.figure.suptitle("Experiment results", fontsize=16)
    g.figure.subplots_adjust(top=0.85)

    for ax in g.axes.flat:
        metric_name = ax.get_title().split(" = ")[-1]
        ax.set_title(f"{metric_name}", fontsize=14)
        ax.set_ylabel("Value")
        ax.set_xlabel("Experiment m")

    for ax in g.axes.flat:
        ax.yaxis.set_tick_params(labelsize=10)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)

    g.figure.savefig(
        f"{dir_name}/imputation_type_results_per_missing_value_count.png",
        bbox_inches="tight",
        dpi=300,
    )
    # plt.show()
