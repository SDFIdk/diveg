import pathlib
from functools import partial
from typing import (
    Iterable,
    Union,
)

import click
import geopandas as gpd

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from IPython import embed

from diveg.prototype import (
    load_insar,
    build_grid,
    get_grid_copy,
    iqr,
    get_shapes,
    get_transform,
)
from diveg.plots import (
    plot_ecdfs,
)


STAT_LABELS_DEFAULT = [
    "count",
    "mean",
    "std",
    "median",
    "min",
    "max",
    "iqr",
    "data",
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

CRS_OUTPUT = "epsg:4326"
"WSG84"
CRS_WORK = "epsg:25832"
"Universal Transverse Mercator (UTM) 32"

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
    aggregates = [
        aggregate
        for label in stat_labels
        if (aggregate := AGGFUNC_MAP.get(label)) is not None
    ]
    return {column: aggregates for column in columns}


def ofname_tif(
    layer_column: str, aggfunc: str, size_x: int, size_y: int
) -> str:
    """
    Return a filename for an output GeoTIFF file.

    """
    return f"insar_grid_{size_x}x{size_y}_{layer_column}_{aggfunc}.tif"


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
    "--columns",
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
    columns: Iterable[str],
    stat_labels: Iterable[str],
    output_path: Union[str, pathlib.Path],
) -> None:
    """
    Load InSAR data and aggregate them over grids of selected resolutions.
    Save the output as raster images.

    Todo:
        Make the boundary of the smaller-resolution grids
        encapsulate the cells of higher-resolution grids.

    """
    n_points_set = sorted(set(n_points_set))
    columns = list(set(columns))
    stat_labels = list(set(stat_labels))

    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True)
    click.secho(f"Output data are saved to `{output_path}` .", fg="yellow")

    click.secho(f"Load {layer=} from {fname=}.")
    gdf = load_insar(fname, layer=layer)

    click.secho("Validate desired column names.")
    difference = set(columns) - set(gdf.columns)
    verb = "are" if len(difference) > 1 else "is"
    assert not difference, f"Columns {difference!r} {verb} not available."

    click.secho(f"Convert coordinates from {gdf.crs.srs=} to {CRS_WORK}")
    gdf = gdf.to_crs(CRS_WORK)

    click.secho(f"Grid operations", fg="yellow")

    kwargs = dict(
        bounds=gdf.total_bounds,
        crs=gdf.crs,
    )
    cell_sizes = [f"({n}x{n})" for n in n_points_set]
    click.secho(
        f"Build grids with the following cell sizes / points = {cell_sizes!r} ."
    )
    grids = [
        build_grid(N_points_x=n_points, N_points_y=n_points, **kwargs)
        for n_points in n_points_set
    ]

    # Select columns to aggregate
    aggfunc = build_aggfunc(columns=columns, stat_labels=stat_labels)

    click.secho(f"Merge points with grids.")
    for (grid, info) in grids:
        click.secho(f"Cell size: {info.cell_width}x{info.cell_height} m^2 with <= {info.n_points_x}x{info.n_points_y} points .")
        click.secho(f"Bin points using the grid cells (spatial join).")
        merged = gpd.sjoin(gdf, grid, how="left", op="within")
 
        click.secho(f"Obtain summary statistics for each grid cell.")
        dissolved = merged.dissolve(by="index_right", aggfunc=aggfunc)

        click.secho(f"Prepare output", fg="yellow")

        click.secho(
            f"Convert grid coordinates from {grid.crs.srs=} to {CRS_OUTPUT}"
        )
        grid = grid.to_crs(CRS_OUTPUT)

        click.secho(
            f"Create separate grid copies with summary statistics for each column in the input layer."
        )
        for layer_column in columns:

            column_data = get_grid_copy(
                grid, dissolved, layer_column, stat_labels
            ).dropna()

            click.secho(f"Calculate affine transformation parameters.")
            transform = get_transform(column_data.total_bounds, info.shape)

            click.secho(f"Build rasteriser")
            rasterise = partial(
                rasterize,
                out_shape=info.shape,
                transform=transform,
                all_touched=True,
                # TODO: Find a nicer way to get the dtype (in case _mean is not a column).
                dtype=column_data._mean.dtype,
            )

            click.secho(f"Build raster writer")
            raster_writer = partial(
                rasterio.open,
                driver="GTiff",
                height=info.nrows,
                width=info.ncols,
                count=1,
                crs=grid.crs,
                transform=transform,
                # TODO: Find a nicer way to get the dtype (in case _mean is not a column).
                dtype=column_data._mean.dtype,
            )

            click.secho(f"Extract (shape, data) pairs.")
            shape_data_pairs = dict(
                mean=get_shapes(column_data, "_mean"),
                std=get_shapes(column_data, "_std"),
                iqr=get_shapes(column_data, "_iqr"),
                count=get_shapes(column_data, "_count"),
            )
            # output_stats = stat_labels.copy().pop('data')
            # shape_data_pairs = {
            #     stat_label: get_shapes(column_data, f"_{stat_label}")
            #     for  stat_label in output_stats
            # }

            click.secho(
                f"Rasterise stat values `{set(shape_data_pairs.keys())}`"
            )
            rasterised_data = {
                statistic: rasterise(shapes=shapes)
                for (statistic, shapes) in shape_data_pairs.items()
            }

            click.secho(f"Build raster images", fg="yellow")
            for (statistic, rasterised) in rasterised_data.items():
                ofname = output_path / ofname_tif(
                    layer_column,
                    statistic,
                    size_x=info.n_points_x,
                    size_y=info.n_points_y,
                )
                click.secho(f"Burn {ofname}")
                with raster_writer(ofname, "w+") as output:
                    output.write(rasterised, indexes=1)

    click.secho("[Done]", fg="white", bold=True)


def _visualisation():
    if False:
        # ECDFs
        points = dissolved.dropna()[("VEL_V", "list")]
        points.name = "VEL_V_data"
        plot_ecdfs(
            points,
            f"VEL_V_binned_cell-ecdf_{info.cell_width:.0f}x{info.cell_height:.0f}",
        )
