
import pathlib
from functools import partial
from typing import (
    Union,
)

import click
import geopandas as gpd

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds


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

AGGREGATES_DEFAULT = (
    'count',
    'mean',
    'std',
    'median',
    'min',
    'max',
    ('iqr', iqr),
    ('data', list),
)

COLUMNS_DEFAULT = (
    'VEL_V',
    'VEL_V_NOUPLIFT',
    'VEL_E',
)

OPATH_DEFAULT = pathlib.Path('diveg_output')

CRS_OUTPUT = 'epsg:4326'
CRS_WORK = 'epsg:25832'


def build_aggfunc(columns: tuple, *, aggregates: tuple = AGGREGATES_DEFAULT) -> dict:
    return {column: aggregates for column in columns}


@click.command()
@click.argument('fname')
@click.option('-l', '--layer', default='2D', help='Layer name in InSAR file.')
# @click.option('-c', '--columns', multiple=True, default=COLUMNS_DEFAULT, help=f'Columns to aggregate. Default values `{COLUMNS_DEFAULT}`.')
@click.option('-O', '--output-path', default=OPATH_DEFAULT, help=f'Path for output data. Default value `{OPATH_DEFAULT}`.')
def main(
    fname: str,
    layer: str,
    # columns: Iterable,
    output_path: Union[str, pathlib.Path]
) -> None:
    """

    Todo:
        Make a loop that creates grids with different resolutions
        and merges the points with each of these.

        Make the boundary of the smaller-resolution grids
        encapsulate the cells of higher-resolution grids.

        Allow for grid sizes to be set at the command line.

    """
    columns = COLUMNS_DEFAULT

    # N_points = (
    #     {'N_points_x': n, 'N_points_y': n}
    #     for n in (2, 4, 6, 8, )
    # )

    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True)
    click.secho(f'Output data are saved to `{output_path}` .', fg='yellow')

    click.secho(f'Load {layer=} from {fname=}.')
    gdf = load_insar(fname, layer=layer)

    click.secho('Validate desired column names.')
    difference = set(columns) - set(gdf.columns)
    assert not difference, f'Expected {difference=} to be an empty set.'

    click.secho(f'Convert coordinates from {gdf.crs.srs=} to {CRS_WORK}')
    gdf = gdf.to_crs(CRS_WORK)

    click.secho(f'Grid operations', fg='yellow')

    click.secho(f'Build grid')
    # TODO: Cache grids based on bounds
    grid, info = build_grid(gdf.total_bounds, crs=gdf.crs)
    click.secho(f'Grid info:\n{info}', fg='red')

    click.secho(f'Bin points using the grid cells (spatial join).')
    merged = gpd.sjoin(gdf, grid, how='left', op='within')

    # Select columns to aggregate
    aggfunc = build_aggfunc(columns)

    click.secho(f'Obtain summary statistics for each grid cell.')
    dissolved = merged.dissolve(by='index_right', aggfunc=aggfunc)

    click.secho(f'Prepare output', fg='yellow')

    click.secho(f'Convert grid coordinates from {grid.crs.srs=} to {CRS_OUTPUT}')
    grid = grid.to_crs(CRS_OUTPUT)

    click.secho(f'Create separate grid copies with summary statistics for each column in the input layer.')
    stat_columns = ('count', 'mean', 'std', 'median', 'min', 'max', 'iqr', 'data')

    # TODO: Make the loop over selected input columns
    # Maybe as a dictionary with each grid copy.
    vel_v = get_grid_copy(grid, dissolved, 'VEL_V', stat_columns).dropna()

    click.secho(f'Calculate necessary affine transformation.')
    transform = get_transform(vel_v.total_bounds, info.shape)

    click.secho(f'Extract shapes')
    shapes_mean = get_shapes(vel_v, '_mean')
    shapes_std = get_shapes(vel_v, '_std')

    click.secho(f'Build rasteriser')
    rasterise = partial(
        rasterize,
        out_shape=info.shape,
        transform=transform,
        all_touched=True,
        # TODO: parameterise dtype?
        dtype=vel_v._mean.dtype
    )

    click.secho(f'Rasterise means')
    raster_mean = rasterise(shapes=shapes_mean)
    click.secho(f'Rasterise stds')
    raster_std = rasterise(shapes=shapes_std)

    click.secho(f'Build raster images', fg='yellow')

    click.secho(f'Build raster writer')
    raster_writer = partial(
        rasterio.open,
        driver='GTiff',
        height=info.nrows,
        width=info.ncols,
        count=1,
        # TODO: variable dtypes and crs
        crs=grid.crs,
        transform=transform,
    )

    ofname = output_path / f'insar_grid_mean_{info.cell_width:.0f}x{info.cell_height:.0f}.tif'
    click.secho(f'Burn {ofname=}')
    with raster_writer(ofname, 'w+', dtype=vel_v._mean.dtype) as output:
        output.write(raster_mean, indexes=1)

    ofname = output_path / f'insar_grid_std_{info.cell_width:.0f}x{info.cell_height:.0f}.tif'
    click.secho(f'Burn {ofname=}')
    with raster_writer(ofname, 'w+', dtype=vel_v._std.dtype) as output:
        output.write(raster_std, indexes=1)

    click.secho('[Done]', fg='white', bold=True)
    

def _visualisation():
    if False:
        # ECDFs
        points = dissolved.dropna()[('VEL_V', 'list')]
        points.name = 'VEL_V_data'
        plot_ecdfs(points, f'VEL_V_binned_cell-ecdf_{info.cell_width:.0f}x{info.cell_height:.0f}')
