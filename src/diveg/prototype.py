import pathlib
from functools import partial
from typing import (
    Union,
    Iterable,
)

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
import rasterio
from rasterio.transform import (
    Affine,
    from_bounds,
)
from rasterio.crs import CRS
from rasterio.features import rasterize
import pyproj

from diveg.data import (
    Grid2DInfo,
)


def load_insar(ifname: str, *, layer: str = "2D") -> gpd.GeoDataFrame:
    """
    Load the specified Geo Package (.gpkg) file's layer.

    """
    assert pathlib.Path(ifname).is_file()
    cache = pathlib.Path.home() / ".diveg/gdf.gz"
    cache.parent.mkdir(exist_ok=True)
    if cache.is_file():
        print("Load cache")
        return gpd.GeoDataFrame(pd.read_pickle(cache))
    gdf = gpd.read_file(ifname, layer=layer)
    gdf.to_pickle(cache)
    return gdf


def load_adm() -> gpd.GeoDataFrame:
    """
    Load commune borders to see the land around the cells

    """
    ifname = pathlib.Path.home() / "data" / "KOMMUNE.shp"
    assert pathlib.Path(ifname).is_file()
    return gpd.read_file(ifname).to_crs("epsg:4326")


def iqr(a: np.ndarray) -> float:
    """
    Calculate the range covered by the middle 50 % of the input values.

    Using statistical terms, calculate the difference between the third
    and the first quartiles (the inter-quartile range, IQR).

    Example
    -------

    >>> x = np.arange(6) * 5
    # array([ 0,  5, 10, 15, 20])
    >>> np.cumsum(x)
    # array([ 0,  5, 15, 30, 50], dtype=int32)
    >>> a, b = np.percentile(x, [25, 75])
    >>> (a, b)
    # (5.0, 15.0)
    >>> IQR = b - a
    >>> IQR
    # 10.0

    """
    a, b = np.percentile(np.asarray(a), [25, 75])
    return b - a


def build_grid(
    bounds: tuple,
    *,
    crs: pyproj.CRS = pyproj.CRS.from_epsg("25832"),
    N_points_x: int = 30,
    N_points_y: int = 30,
    point_distance: float = 80,  # [m]
) -> tuple[gpd.GeoDataFrame, Grid2DInfo]:
    """
    Args:
        bounds
            The boundary of the data that are to be gridded.
            Assumed to be a tuple containing (`minx`, `miny`, `maxx`, `maxy`)
            as returned by the property `gpd.GeoDataFrame.total_bounds` .
        crs
            The coordinate system that the geometry should use.
            The assumption is that the coordinated are already projected in this system.

    Assumptions:
        This is so far only used for grids of certain size covering a land the size of Denmark.
        What is reasonable in terms of resolution and number of points to each cell may need to be reconsidered for other use cases.

    """
    minx, miny, maxx, maxy = bounds

    # Cell resolution

    # The cells's side length in a given direction (x or y) is the point distance times the number of points
    cell_width = N_points_x * point_distance
    cell_height = N_points_y * point_distance

    # Subtract half the point_distance to have this distance as padding from the left and bottom-most poinst.
    grid_x = (
        np.arange(minx, maxx + cell_width, cell_width) - point_distance / 2
    )
    grid_y = (
        np.arange(miny, maxy + cell_height, cell_height) - point_distance / 2
    )

    # Create bounding-box coordinates for each cell in the grid
    # Make a cell (geometry) grid off these coordinates

    cells = [
        geometry.box(
            cell_minx,
            cell_miny,
            cell_minx + cell_width,
            cell_miny + cell_height,
        )
        for cell_minx in grid_x
        for cell_miny in grid_y
    ]

    # Build geometries into geopandas dataframe
    return (
        gpd.GeoDataFrame(cells, columns=["geometry"], crs=crs),
        Grid2DInfo(
            N_points_x,
            N_points_y,
            point_distance,
            # cell_width,
            # cell_height,
            grid_x,
            grid_y,
        ),
    )


def get_grid_copy(
    grid: gpd.GeoDataFrame,
    dissolved: gpd.GeoDataFrame,
    source_column: str,
    stat_columns: Iterable,
) -> gpd.GeoDataFrame:
    """
    Creates a grid copy with the selected aggregation products.

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


def get_shapes(gdf: gpd.GeoDataFrame, colname: str) -> list:
    """
    Extract and associate and format geometry and data in GeoDataFrame.

    """
    assert "geometry" in gdf.columns, f"Expected `geometry` in {gdf.columns=}"
    assert colname in gdf.columns, f"Expected {colname!r} in {gdf.columns=}."
    return [
        (shape, value) for (shape, value) in zip(gdf.geometry, gdf[colname])
    ]
