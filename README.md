# [Danish InSAR Velocity and Error Grid (DIVEG)](https://github.com/Kortforsyningen/diveg)

by [Joachim](joamo@sdfe.dk)

The goal of this package is to provide an easy-to-use command-line tool that takes a geopackage file with InSAR-velocity data as input and produces geotiff-raster files with points gridded and aggregated in the desired grid resolution.

## User installation

Requirements

*   Access to the data that the package is made for.
*   Access to [this repository](https://github.com/Kortforsyningen/diveg) on GitHub.
*   [`conda`](https://docs.conda.io/en/latest/miniconda.html) | [repo](https://repo.anaconda.com/miniconda/)
*   [`git`](https://git-scm.com/)

With the requirements, run the following commands in your environment:

```sh
cd your/git/repos
# As user
git clone https://github.com/Kortforsyningen/diveg.git
# As developer
git clone git@github.com:Kortforsyningen/diveg.git

# Install the environment
sh conda-setup.sh
```

## Usage

### CLI Example: Create a .tif file with the default settings using the command-line application `diveg`:

```sh
(diveg) $ ls
# (nothing to begin with)
(diveg) $ diveg your/data/insar.gpkg
# (...)
(diveg) $ ls
# diveg_output
(diveg) $ ls diveg_output
# insar_grid_mean_2400x2400.tif    insar_grid_std_2400x2400.tif
```

### API Example: using `diveg` as a module:

```python
import pathlib

import pandas as pd
import geopandas as gpd

from diveg.grid import(
    Grid,
    build_grid,
)
from diveg.statistics import iqr


POINT_DISTANCE_OBSERVED = 0.000720

ifname = 'path/to/insar.gpkg'
assert pathlib.Path(ifname).is_file()

# Load data
cache = pathlib.Path('__cache__/points.gz')
cache.parent.mkdir(exist_ok=True)
if cache.is_file():
    points = gpd.GeoDataFrame(pd.read_pickle(cache))
else:
    points = gpd.read_file(ifname, layer='2D')
    points.to_pickle(cache)

point_bounds = points.total_bounds

# Create a high- and low-resolution grid
grid_lo = Grid(*build_grid(point_bounds, crs=points.crs, N_points_x=6, N_points_y=6, point_distance=POINT_DISTANCE_OBSERVED))
grid_hi = Grid(*build_grid(point_bounds, crs=points.crs, N_points_x=3, N_points_y=3, point_distance=POINT_DISTANCE_OBSERVED))

# Select what columns in the input data to get what statistical 
layer_columns = 'VEL_V', 'VEL_V_NOUPLIFT'
aggregates = ('count', 'mean', 'std', 'median', 'min', 'max', ('iqr', iqr), ('data', list),)
aggfunc = {column: aggregates for column in layer_columns}

# Spatially, join the data values in points with the grid,
# group by the grid-cell geometry and, finally,
# aggregate the data values from the points in each cell.
grid_lo.process_points(points, aggfunc)
grid_hi.process_points(points, aggfunc)

# Specify criteria for overwriting the 
filters = [
    # From the experimental cummulative distribution function of the IQR values,
    # more than 90 % of the cells have an IQR lower than 2 [mm / yr].
    lambda row: row[('VEL_V', 'iqr')] > 2,
]

# Shortcut: specify only the labels of the aggregate methods specified above
stat_columns_wanted = ('mean', 'std', 'iqr')
columns_imposable = grid_hi.get_columns_imposable(stat_columns_wanted)

# Using the filter, and the lower-resolution grid,
# overwrite the cells in the higher-resolution grid.
grid_hi.impose(grid_lo, filters=filters, columns=columns_imposable)

# Save data from the high-resolution grid
size_x, size_y = grid_hi.info.n_points_x, grid_hi.info.n_points_y
for column in grid_hi.columns_imposed:
    output_name = column if isinstance(column, str) else '_'.join(column)
    ofname = f'insar_grid_{size_x}x{size_y}_{output_name}.tif'
    grid_hi.save(ofname, column)

# Save data from the high-resolution grid
size_x, size_y = grid_lo.info.n_points_x, grid_lo.info.n_points_y
for column in grid_lo.data_columns:
    output_name = column if isinstance(column, str) else '_'.join(column)
    ofname = f'insar_grid_{size_x}x{size_y}_{output_name}.tif'
    grid_lo.save(ofname, column)

```

