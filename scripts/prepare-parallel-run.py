import pickle
import pathlib

import pandas as pd
import geopandas as gpd

from diveg.grid import(
    Grid,
    build_grid,
)
from diveg.statistics import iqr

from cwd import cwd


def main():
    POINT_DISTANCE_OBSERVED = 0.000720
    # POINT_DISTANCE_OBSERVED = 0.000719856

    # Load from original file or (faster) load cache

    # ifname = '/home/e088195/git/diveg/data/insar.gpkg'
    ifname = '/home/e088195/data/insar/2021-08-05/insar.gpkg'
    assert pathlib.Path(ifname).is_file()

    # Load data
    cache = cwd / '/insar.pkl'
    if cache.is_file():
        print('Reading data from cache')
        points = gpd.GeoDataFrame(pd.read_pickle(cache))
    else:
        print('Reading data from geo package')
        points = gpd.read_file(ifname, layer='2D')
        points.to_pickle(cache)

    point_bounds = points.total_bounds

    # Select what columns in the input data to get what statistical 
    layer_columns = 'VEL_V', 'VEL_V_NOUPLIFT'
    aggregates = ('count', 'mean', 'std', 'median', 'min', 'max', ('iqr', iqr),)
    aggfunc = {column: aggregates for column in layer_columns}

    # Create a high- and low-resolution grid
    print('6x6: Build grid')
    grid_lo = Grid(*build_grid(point_bounds, crs=points.crs, N_points_x=6, N_points_y=6, point_distance=POINT_DISTANCE_OBSERVED))
    print('3x3: Build grid')
    grid_hi = Grid(*build_grid(point_bounds, crs=points.crs, N_points_x=3, N_points_y=3, point_distance=POINT_DISTANCE_OBSERVED))

    # Spatially, join the data values in points with the grid,
    # group by the grid-cell geometry and, finally,
    # aggregate the data values from the points in each cell.
    print('6x6: Process')
    grid_lo.process_points(points, aggfunc)
    print('3x3: Process')
    grid_hi.process_points(points, aggfunc)

    # Reduce amount of data to process
    columns_to_mark_empty_cells = [
        ('VEL_V', 'count'),  # This should be the same for VEL_V_NOUPLIFT
    ]
    print(f'6x6: Reduce by {1 - grid_lo.reduction(columns_to_mark_empty_cells): >5%}')
    grid_lo.reduce(columns_to_mark_empty_cells)
    print('3x3: Reduce by {1 - grid_hi.reduction(columns_to_mark_empty_cells): >5%}')
    grid_hi.reduce(columns_to_mark_empty_cells)

    # Store the grid that is to be imposed, so this can be re-used
    fname_grid_hi = cwd / 'grid_hi.pkl'
    with open(fname_grid_hi, 'wb') as fsock:
        pickle.dump(grid_hi, fsock)

    # Store the lo-res grid as well
    fname_grid_lo = cwd / 'grid_lo.pkl'
    with open(fname_grid_lo, 'wb') as fsock:
        pickle.dump(grid_lo, fsock)


if __name__ == '__main__':
    main()


