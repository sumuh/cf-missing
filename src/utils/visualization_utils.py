import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
