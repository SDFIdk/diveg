import pathlib
from typing import (
    Iterable,
    Tuple,
    Union,
)

import click
import pandas as pd
import geopandas as gpd

from diveg.statistics import iqr
from diveg.grid import (
    Grid,
    build_grid,
)

POINT_DISTANCE_OBSERVED = 0.000720

STAT_LABELS_DEFAULT = [
    "count",
    "mean",
    "std",
    "median",
    "min",
    "max",
    "iqr",
    # "data",
]
"Default command-line input with names of the statistical functions to use."

N_POINTS_DEFAULT = [6]
"Default command-line input with desired grid-cell size in number of points along the x and y direction."

COLUMNS_DEFAULT = [
    "VEL_V",
    "VEL_V_NOUPLIFT",
]
"Default command-line input with names of the input-data columns to use."

OPATH_DEFAULT = pathlib.Path("diveg_output")
"Default command-line input with directory for output files"

AGGFUNC_MAP = dict(
    count="count",
    mean="mean",
    std="std",
    median="median",
    min="min",
    max="max",
    iqr=("iqr", iqr),
    data=("data", list),
)
"A mapping between a string value to a format for `pandas`'s `.agg` method."


def build_aggfunc(
    *, columns: Iterable[str], stat_labels: Iterable[str]
) -> dict:
    """
    Constructs a dictionary with each data column as kay and the selected aggregates 

    """
    # TODO: Split this into two functions to separate problem-agnostic dict comprehension.
    aggregates = [
        aggregate
        for label in stat_labels
        if (aggregate := AGGFUNC_MAP.get(label)) is not None
    ]
    return {column: aggregates for column in columns}


def ofname_tif(
    column: Union[str, Tuple[str, str]], size_x: int, size_y: int
) -> str:
    """
    Return a filename for an output GeoTIFF file.

    """
    if isinstance(column, tuple):
        output_name = '_'.join(column)
    else:
        output_name = column
    return f"insar_grid_{size_x}x{size_y}_{output_name}.tif"


@click.command()
@click.argument("fname")
@click.option("-l", "--layer", default="2D", help="Layer name in InSAR file.")
@click.option(
    "-n",
    "--n-points-set",
    multiple=True,
    default=N_POINTS_DEFAULT,
    type=int,
    help=f"Number of points to include in each dimension. Example: `-n 2` provides a grid with cells covering 2x2 points. Default values `{N_POINTS_DEFAULT}`.",
)
@click.option(
    "-c",
    "--layer-columns",
    multiple=True,
    default=COLUMNS_DEFAULT,
    help=f"Columns to aggregate. Default values `{COLUMNS_DEFAULT}`.",
)
@click.option(
    "-s",
    "--stat-labels",
    multiple=True,
    default=STAT_LABELS_DEFAULT,
    help=f"Columns to aggregate. Default values `{STAT_LABELS_DEFAULT}`.",
)
@click.option(
    "-d",
    "--point-distance",
    default=POINT_DISTANCE_OBSERVED,
    help=f"Columns to aggregate. Default value `{POINT_DISTANCE_OBSERVED}`.",
)
@click.option(
    "-O",
    "--output-path",
    default=OPATH_DEFAULT,
    help=f"Path for output data. Default value `{OPATH_DEFAULT}`.",
)
# TODO: Add alternative cache path as option.
def main(
    fname: str,
    layer: str,
    n_points_set: Iterable[int],
    layer_columns: Iterable[str],
    stat_labels: Iterable[str],
    point_distance: float,
    output_path: Union[str, pathlib.Path],
) -> None:
    """
    Load InSAR data and aggregate them over grids
    of selected resolutions. Save the output as
    raster images.

    """
    # Remove duplicate input
    n_points_set = sorted(set(n_points_set))
    layer_columns = list(set(layer_columns))
    stat_labels = list(set(stat_labels))

    # Prepare output destination
    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True)
    click.secho(f"Output data are saved to `{output_path}` .", fg="yellow")

    # Load data
    click.secho(f"Load {layer=} from {fname=}.")
    assert pathlib.Path(fname).is_file()
    cache = pathlib.Path(f"__diveg__/{pathlib.Path(fname).stem}.gz")
    cache.parent.mkdir(exist_ok=True)
    if cache.is_file():
        click.secho("Load existing cache", fg="green")
        gdf = gpd.GeoDataFrame(pd.read_pickle(cache))
    else:
        gdf = gpd.read_file(fname, layer=layer)
        gdf.to_pickle(cache)

    # TODO: REMOVE THIS
    # gdf = gdf.to_crs(CRS_WORK_DEFAULT)
    # gdf.to_pickle(cache)
    # click.secho(f'{gdf.crs.name}', fg='red')
    # gdf = gdf.iloc[:100]

    # Verify that desired columns actually exist in the loaded data
    click.secho("Validate desired column names.")
    difference = set(layer_columns) - set(gdf.columns)
    verb = "are" if len(difference) > 1 else "is"
    assert not difference, f"Columns {difference!r} {verb} not available."

    # Select columns to aggregate
    aggfunc = build_aggfunc(columns=layer_columns, stat_labels=stat_labels)

    click.secho(f"Grid operations", fg="yellow")

    # Cache point boundary
    click.secho(f"Get total bounds of points for grid-creation.")
    bounds_points = gdf.total_bounds

    click.secho(f"Build low-resolution grid.")
    grid_lo = Grid(*build_grid(bounds_points, crs=gdf.crs, N_points_x=6, N_points_y=6, point_distance=point_distance))
    click.secho(f"Build high-resolution grid.")
    grid_hi = Grid(*build_grid(bounds_points, crs=gdf.crs, N_points_x=3, N_points_y=3, point_distance=point_distance))

    click.secho(f"Process low-resolution grid.")
    grid_lo.process_points(gdf, aggfunc)
    # grid_lo.save('test2.tif', 'VEL_V', 'mean')
    click.secho(f"Process high-resolution grid.")
    grid_hi.process_points(gdf, aggfunc)

    # percent_covered = .5
    filters = [
        lambda row: row[('VEL_V', 'iqr')] > 2,
        # lambda row: row[('VEL_V', 'count')] < max(2, grid_hi.info.nrows * grid_hi.info.ncols * percent_covered)
    ]
    # These are the columns we *want* to overwrite, when the criteria are met.
    stat_columns_wanted = ('mean', 'std', 'iqr')
    columns_imposable = grid_hi.get_columns_imposable(stat_columns_wanted)

    click.secho(f"Impose low-resolution stats onto to overwritable columns in high-resolution grid.")
    grid_hi.impose(grid_lo, filters=filters, columns=columns_imposable)

    # Save output
    click.secho(f"Save output", fg='yellow')
    size_x, size_y = grid_hi.info.n_points_x, grid_hi.info.n_points_y
    for column in grid_hi.columns_imposed:
        ofname = output_path / ofname_tif(column, size_x=size_x, size_y=size_y)
        click.secho(f'Writing output to file {ofname.name!r}', fg='green')
        grid_hi.save(ofname, column, crs=gdf.crs)

    # Save data from the high-resolution grid
    size_x, size_y = grid_lo.info.n_points_x, grid_lo.info.n_points_y
    for column in grid_lo.data_columns:
        ofname = output_path / ofname_tif(column, size_x=size_x, size_y=size_y)
        click.secho(f'Writing output to file {ofname.name!r}', fg='green')
        grid_lo.save(ofname, column, crs=gdf.crs)

    click.secho("[Done]", fg="white", bold=True)
