from functools import partial
from typing import (
    Union,
    Iterable,
)

import numpy as np
import geopandas as gpd
from shapely import geometry
from rasterio.transform import (
    Affine,
    from_bounds,
)
from rasterio.crs import CRS
from rasterio.features import rasterize
import pyproj


def get_grid_copy(
    grid: gpd.GeoDataFrame,
    dissolved: gpd.GeoDataFrame,
    source_column: str,
    stat_columns: Iterable,
) -> gpd.GeoDataFrame:
    """
    Creates a grid copy with the selected data.

    This step is needed, if the data are to be rasterised.

    Associate aggregated values of dissolved dataframe
    to the dataframe of the grid cells containing them.

    Column names for the values are prefixed with a '_' to
    avoid clash with existing properties of the dataframe.

    """
    grid_copy = grid.copy()
    for stat_column in stat_columns:
        grid_copy.loc[dissolved.index, f"_{stat_column}"] = dissolved[
            (source_column, stat_column)
        ].values
    return grid_copy


def get_transform(
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> Affine:
    """
    Use the from_bounds function in rasterio.transform module

    TODO: Better/non-trivial docs

    """
    return from_bounds(*bounds, *shape[::-1])



def _visualisation():
    if False:
        # ECDFs
        points = dissolved.dropna()[("VEL_V", "list")]
        points.name = "VEL_V_data"
        plot_ecdfs(
            points,
            f"VEL_V_binned_cell-ecdf_{info.cell_width:.0f}x{info.cell_height:.0f}",
        )
