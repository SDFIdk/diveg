import pathlib
from typing import (
    Union,
)

import matplotlib as mpl
from matplotlib import style

style.use("bmh")
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd


def plot_points_wih_grid(
    points: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    *,
    column: str,
    fname: Union[str, pathlib.Path],
) -> None:
    """
    Plot the cell grid with the data

    Notes:
        Takes about a minute on the Linux Server

    """
    fig, ax = plt.subplots(figsize=(20, 20))
    points.plot(ax=ax, markersize=0.1, column=column, legend=True)
    grid.plot(ax=ax, facecolor="none", edgecolor="#cccccc")
    fig.tight_layout()
    plt.savefig(f"{fname}.pdf")
    plt.savefig(f"{fname}.png")


def plot_ecdfs(points: gpd.GeoSeries, fname: Union[str, pathlib.Path]) -> None:
    """
    Product: ECDF for each cell to see how each cell's data compares, and if there are outliers.

    """
    fig, ax = plt.subplots(figsize=(20, 10))
    for (ix, row) in points.iteritems():
        sns.ecdfplot(row, ax=ax, color="#ff0000", alpha=0.1)
    fig.tight_layout()
    plt.savefig(f"{fname}.pdf")
