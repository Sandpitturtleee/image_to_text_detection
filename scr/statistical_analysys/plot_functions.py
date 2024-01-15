import pandas as pd
from matplotlib.pyplot import show
from pandas import cut


def create_bar_plots(percentages: list, percentages_sum: list):
    """
    Creates plots for percentages and percentages_sum

    Parameters:
    :param percentages: A list of percentages of areas detected/labeled for every img
    :type percentages: list
    :param percentages_sum: A list of percentages of summed areas detected/labeled for every file
    :type percentages_sum: list
    """
    create_bar_plot(
        data=percentages, bin_edges=area_bin_edges, bin_labels=area_bin_labels
    )
    create_bar_plot(
        data=percentages_sum, bin_edges=area_bin_edges, bin_labels=area_bin_labels
    )


def create_bar_plot(data: list, bin_edges: list, bin_labels: list):
    """
    Creates bar plot based on data, bin edges and bin labels

    Parameters:
    :param data: A list of percentages, data for creating plot
    :type data: list
    :param bin_edges: A list of bin_edges - ranges (from x to y) on which data is put in correct bins
    :type bin_edges: list
    :param bin_labels: A list of bin_labels than represents middle point in bins
    :type bin_labels: list
    """
    columns = f"{data=}".split("=")[0]
    df = pd.DataFrame(data, columns=[columns])
    df["bin"] = cut(
        df[columns],
        [
            *bin_edges,
            float("inf"),
        ],
        labels=bin_labels,
    )
    df["bin"].value_counts().sort_index().plot.bar()
    show()


area_bin_edges = [
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2.0,
]
area_bin_labels = [
    "0.05",
    "0.15",
    "0.25",
    "0.35",
    "0.45",
    "0.55",
    "0.65",
    "0.75",
    "0.85",
    "0.95",
    "1.05",
    "1.15",
    "1.25",
    "1.35",
    "1.45",
    "1.55",
    "1.65",
    "1.75",
    "1.85",
    "1.95",
    "2.0 +",
]
