from typing import (
    Any,
    Iterable,
    Callable,
    Union,
    Optional,
    Sequence,
)

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
import rasterio
from rasterio.transform import (
    from_bounds,
)
from rasterio.features import rasterize
import pyproj
from IPython import embed

from diveg.data import (
    Grid2DInfo,
)


CRS_OUTPUT_DEFAULT: str = "epsg:4326"
"WSG84"
CRS_WORK_DEFAULT: str = "epsg:25832"
"Universal Transverse Mercator (UTM) 32"


def build_grid(
    bounds: tuple,
    *,
    crs: str = CRS_WORK_DEFAULT,
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
    grid_x = np.arange(minx, maxx + cell_width, cell_width) - point_distance / 2
    grid_y = np.arange(miny, maxy + cell_height, cell_height) - point_distance / 2

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
            grid_x,
            grid_y,
        ),
    )


class Grid:
    """
    Container for gridded data of a given resolution.

    Methods are able to impose values from a larger
    scale to a smaller, if certain criteria are met
    (measurements of bad statsistics).

    Methods can store information about which cells
    of the higher-resolution (smaller cell size) grid
    have had their values overridden.

    """

    def __init__(
        self, grid: gpd.GeoDataFrame, info: Grid2DInfo, output_crs: Union[str, pyproj.CRS] = CRS_OUTPUT_DEFAULT, work_crs: Union[str, pyproj.CRS] = CRS_WORK_DEFAULT):
        """
        Args:
            TODO: Rewrite docstring
            grid: GeoDataFrame, where geometry column
            contains the grid cell geometry, and the rest
            of the columns have desired statistical data.
            The data should be what needs to be imposed,
            and should thus be filterable by the method
            `impose_values()`.

        """
        self._grid = grid
        "The geometry and data of aggregated points."

        self.info = info
        "Information about the generated grid."

        self._output_crs = output_crs
        "Coordinates for generated output."

        self._work_crs = work_crs
        "Coordinates in which the work is performed."

        self._total_bounds: tuple[float] = None
        "Cached value of total bounds for the geometry."

        self._merged: gpd.GeoDataFrame = None
        "The resulting `GeoDataFrame` of the spatial join (binning) of input points with the grid."

        self._dissolved: gpd.GeoDataFrame = None
        "The resulting `GeoDataFrame` after aggregating merged points."

        self._imposed: gpd.GeoDataFrame = None
        "The resulting `GeoDataFrame` after imposing values from a lower-resolution grid on the original data in this instance."

        self._rasterised: dict[tuple[str, str, Optional[Union[str, pyproj.CRS]]], np.ndarray] = {}
        "Cache for rasterised data."

    @property
    def total_bounds(self) -> tuple[float]:
        """
        Return total bounds for the grid.

        """
        if self._total_bounds is None:
            self._total_bounds = self._grid.total_bounds
        return self._total_bounds

    def _merge(self, points: gpd.GeoDataFrame) -> "Grid":
        """
        Used input:
            left: Keep the rows of the left `GeoDataFrame` of the two that are given.
                  A separate column `index_right` is produced with the index of the
                  rows of the right `GeoDataFrame` that are included in the join.
            within: the points must be inside (not touching) the boundary of the geometry `_grid`.

        """
        self._merged = gpd.sjoin(points, self._grid, how="left", op="within")
        return self

    def _dissolve(self, aggfunc: dict) -> "Grid":
        """
        Used input:
            index_right: The index that was automatically created+named when
            spatially joining the input points with the instances's (grid) geometry.

        """
        self._dissolved = self._merged.dissolve(by="index_right", aggfunc=aggfunc)
        return self

    def _attach_to_grid_geometry(self) -> "Grid":
        """
        This step is needed, if the data are to be rasterised.

        Associate aggregated values of dissolved dataframe
        to the dataframe of the grid cells containing them.

        In other words, associate the aggregated data with the
        geometry of the grid cells (`Polygon` rectangles) of
        `self._grid` rather than the `MultiPoint`s of
        `self._dissolved`.

        The index values of `_dissolved` is a subset of the index
        of `_grid`, so the data have to be assigned using the
        index of `_dissolved` to keep the data aligned.

        """
        columns = self._dissolved.columns[1:]
        self._grid.loc[self._dissolved.index, columns] = self._dissolved[columns].values
        "Example source-column index: [('VEL_V', 'iqr'), ('VEL_V', 'mean'), ('VEL_V', 'std')]"
        return self

    def process_points(self, points: gpd.GeoDataFrame, aggfunc: dict) -> None:
        """
        Perform the process from spatially binning point-data values
        to the grid-cell geometry, calculating statistical results and,
        finally, attach these results to their respective grid cells (bins).

        In other words, proces input data by:

        *   spatially, joining point-data values with the grid geometry
        *   dissolving (pandas : groupby) the joined data using the desired aggregation methods
        *   adding the results of dissolving as newd columns to the grid

        """
        self._merge(points)
        self._dissolve(aggfunc)
        self._attach_to_grid_geometry()

    def select(self, column: tuple[str, str]) -> gpd.GeoDataFrame:
        """
        Return selected grid data as a GeoDataFrame with generic columns (`geometry', 'data').

        """
        return self._grid[['geometry', column]].rename(columns={column: 'data'})

    @property
    def _transform(self) -> rasterio.Affine:
        return from_bounds(*self.total_bounds, self.info.ncols, self.info.nrows)

    def _rasterise(self, column: Union[str, tuple[str, str]], crs: Union[str, pyproj.CRS] = None) -> np.ndarray:
        """
        Return matrix (numpy array) of rasterised data.

        This method always re-produces the output, since it is used by
        `Grid.rasterise(...)` which caches previously-created output.

        """
        grid_data = self.select(column).dropna()
        if crs is not None:
            grid_data = grid_data.to_crs(crs)
        return rasterize(
            shapes=get_shapes(grid_data, 'data'),
            out_shape=self.info.shape,
            transform=self._transform,
            all_touched=True,
            dtype=grid_data['data'].dtype,
        )

    def rasterise(self, column: Union[str, tuple[str, str]], crs: Union[str, pyproj.CRS] = None) -> np.ndarray:
        if isinstance(column, tuple):
            key = column + (crs,)
        else:
            crs_str = crs if isinstance(crs, (str)) else crs.srs if isinstance(crs, pyproj.CRS) else None
            key = f'{column}_{crs_str}'
        print(key)
        if key not in self._rasterised:
            self._rasterised[key] = self._rasterise(column, crs)
        return self._rasterised[key]

    def save(self, filename: str, column: Union[str, tuple[str, str]], crs: Union[str, pyproj.CRS] = None) -> None:
        """
        Save grid data given by `layer_column` and `stat_column` as `filename`.

        A default CRS is applied to the underlying data from which the selection draws,
        but alternative coordinates can be given.

        """
        rasterised = self.rasterise(column, crs)
        with rasterio.open(
            filename,
            "w+",
            driver="GTiff",
            height=self.info.nrows,
            width=self.info.ncols,
            count=1,
            crs=crs,
            transform=self._transform,
            dtype=rasterised.dtype,
        ) as output:
            output.write(rasterised, indexes=1)

    def get_columns_imposable(self, stats: Iterable[str]) -> list[tuple[str, str]]:
        """
        Returns all column names in the grid with the selected statistics.

        """
        return self._grid.columns.drop(get_columns_immutable(self._grid, stats))

    def impose(self, lo: "Grid", columns: Iterable[Any] = None, filters: Iterable[Callable[[gpd.GeoSeries], bool]] = None) -> None:
        """
        Overwrite (impose) specified values in the calling Grid instance's dataset with
        those of another Grid instance with lower resolution than the calling instance.

        Args:
            lo: Another Grid instance assumed to have the following properties:
                *   It has the same columns as the calling instance (otherwise this method is useless).
                *   It must also have a grid of lower resolution than the calling instance.
                *   It should be based on the same underlying point data.
                *   It should cover at least the same area as the calling instance.
            columns: Columns in the grid whose values should be overwritten by the lower-resolution grid (if criteria met)
            filters: A list of functions that can be used with (Geo)Pandas's (g)df.apply() method.

        TODO: Rewrite the rest of the docstring.

        Returns:
            Geodataframe with the grid data for each of the following quantities:

            *   The corrected data
                -   Example: VEL_V as input column. This has two output frames: VEL_V_mean, and VEL_V_std.
                    Each of these two separate geodataframes will have their cvalues corrected at the places and in the same way with their corresponding lower-resolution grid-cell valuess.
            *   Data (0: unchanged, 1: changed) about which cells were changed.
            *   

        Assumptions:

            We need good local statistics, i.e. data with well-defined location should not be changed, unless the foundation for these values are too bad to be useful.
            Thus, we need to defined `too bad to be useful`.

            We assume, for now, that the grid cells that have their values changed, can have the value be that of the containing lower-resolution cell.
            TODO: Consider using some average value of the lower-resolution cells's values (when possible), when the higher-resolution cell is close to the edge of the containing lower-resolution cell.

        Conclusions from initial data exploration of the InSAR dataset:

        *   At ceell-resolution 6x6, the distribution of interquartile ranges (IQRs) is closely centered around 1.2 and 95 % of the values below 2.
            Some go all the way up to 5, but setting the acceptable upper limit for these values to 2 seems reasonable.

        *   The number of cells with the possible number of points is vey small. Most cells cover (from visual inspection) less than 10 % of the capacity of the containing cell.
            With a cell capacity of 6x6 = 36 points, less than 10 % would give, say, 3 points in a cell, which are too few do get accurate statistics from.
            TODO: Where should we draw the line?
            TODO: How much difference can there be between the low- and high-resolution cell capacities before their sizes can not meaningfully produce a local value?

        Non-implementation-specific condiserations:

        *   Should data only be changed, given independently given criteria, such as IQR is larger than some value?
        *   Should data be changed, if the lower-resolution cell is just better than the higher-resolution cell's values?
            This would seem to always be the case, since quality of the statistical results increases with more data which
            is the case, when aggregating over larger-sized (ower-resolution) cells. The drawback of this is that is does
            not take into account our need for good local statistics.

        Aggregated data in the original dataset are used as input for the filtering,
        and the rest are considered data products that will be used in further analysis.

            Aggregated data used for filtering, refering to the high-resolution data:

            *   iqr: The interquartile range.
            *   count: The number of points in the cells.

            Aggregated data that will get copied and potentially overwritten with the lower-resolution grid data in this process:

            *   mean of the layer column, e.g. VEL_V
            *   std of the layer column, e.g. VEL_V (not VEL_V_STD which we do not use)

            Data that will be produced in order to illustrate the process:

            *   What rows were changed?
            *   (What columns changed?)
            *   (considering) What is the difference between the changed values and the new (imposed) values?

        Procedure:
            Select the (layer, stat) data that need to be manipulated.
    
                Example: Select layer_column `VEL_V` and its stat_columns,
                         excluding the information/filter columns `iqr` and `count`.)

            Make a copy of the selected data, with the suffix to each column

            Then, using the filter functions, determine the cells that need to have their data overwritten.

            Choices:
                Work with empty cells (np.nan)? These will be overwritten by values or NaN.
                Also, the fewer changes to manage, the simpler (better) the maintainance.

        """
        # Input validation or correction

        # Get columns that should have their values overwritten
        columns_imposable = self._grid.columns.drop('geometry') if columns is None else columns 

        # TODO: IMPLEMENTATION CONSIDERATION: IF NOT FILTER IS GIVEN; THIS CURRENTLY MEANS THAT NOTHNG IS OVERWRITTEN:
        # TODO: WHAT SHOULD BE THE BEHAVIOUR FOR THIS CASE? THAT ALL DATA SHOULD BE OVERWRITTEN. BUT THEN THERE IS NO
        # TODO: NEED TO IMPOSE UNLESS ONE WANTS A HIGH-RESOLUTION GRID WITH LOW-RESOLUTION RESULTS.

        # Avoid looping over `None`
        filters = filters or []

        ## Steps:

        # 0. Copy data (stat_columns) to new columns in _grid and suffix column names with, say, _changed.

        # Goal: Focus on the statistics (denoted by the column names), we are interested in.

        # Copy selected (imposable) columns to a new container, keeping the same columns names.
        imposed = self._grid[columns_imposable].copy()

        # 0.5 Create boolean array that filters out the rows, we do not need to change.

        # Now that the program knows which columns in the higher-resolution
        # grid that we want to overwrite with the lower-resolution grid data.

        # Look at what what rows of data in the filter columns and decide, which
        # cells whose stats should be overwritten with values that we assume are better.

        # Apply each filter to boolean array that starts with all (i.e. none selected) values false. 
        boolean_index_imposed_overwrite = np.zeros_like(self._grid.index.values).astype(bool)
        
        for filter_func in filters:
            boolean_index_imposed_overwrite |= self._grid.apply(filter_func, axis=1).values

        # Now we know what rows in the data that need to be overwritten.

        # Add a special column to imposed that contains, again, False as default.
        imposed['overwritten'] = 0
        # When the change has actually been made, each row that was changed will
        # have this corresponding changed-bit set True as well.

        # We need indices into lower-resolution grid to know what cells to copy from.

        # Loop over rows to overwrite and get the index of each cell in the lower-resolution grid that contains the points in the filtered data.

        # Search for the cell-index (replacement_index) of the lower-resolution grid whose corresponding
        # value should replace the 'bad' values in the high-resolution cell.
        # NOTE: self._dissolved containes the original geometry of the point locations as multipoints
        # since _dissolved only has the geometry of parts of the grid, we need to
        # * convert the boolean index of overwritable columns for the complete grid (as exemplified by _grid and imposed)
        #   to the indices of the rows that are to be overwritten.
        # * This index is the same for all, including _dissolved.
        # From this index, we can loop over just the cells in the that wee need to change.
        index_overwrite = self._grid.index[boolean_index_imposed_overwrite]
        for (index_points, points) in self._dissolved.loc[index_overwrite].iterrows():
            # Which cell in the lower-resolution grid are the points in the higher-resolution grid within?
            lo_indices = lo._grid.sindex.query(points.geometry, predicate="within")
            assert len(lo_indices) == 1
            # Assign the selected statistical values of the lower-resolution cell to the higher-resolution cell.
            imposed.loc[index_points, columns_imposable] = lo._grid.loc[lo_indices[0], columns_imposable].values
            # Mark row as overwritten
            imposed.loc[index_points, 'overwritten'] += 1

        # TODO: Try to do the above in a bulk operation instead of a Python for loop over each row.
        # Which cell in the lower-resolution grid covers the multipoint in the higher-resolution grid?
        # self_indices_covered = np.unique(self._dissolved.sindex.query_bulk(lo._grid.geometry, predicate='covers')[0])

        # Save the results in _grid?
        new_column_names = [
            (layer_column, f'{stat_column}_imposed')
            for (layer_column, stat_column)
            in imposed.columns.drop(['overwritten'])
        ] + ['overwritten']

        self._grid[new_column_names] = imposed[imposed.columns].values


def get_columns_immutable(gdf: gpd.GeoDataFrame, stat_columns_mutable: Iterable[str]) -> list[Union[str, tuple[str, str]]]:
    """
    Return list of column names in the given dataframe that may not be edited.

    Use case:
        For imposing values from one datafram to another, it is
        easier to specify what columns to change than the opposite.

        Generally, when what can be edited is known,
        the immutable columns can be obtained.

    Assumptions:
        `geometry` column is immutable
        Only 'geometry' column is a string, the rest are of the form
        
            ('name_of_layer_column_of_input_data', 'label_for_some_aggregated_value')

        For this reason, the geometry column is dropped from the
        column list and added afterwards in front of the resulting list.

    """
    columns_immutable = ['geometry'] + [
        (layer_column, stat_column)
        for (layer_column, stat_column)
        in gdf.columns.drop('geometry')
        # Here, we only check the value of the stat_column,
        # since these are the same for each layer_column
        if stat_column not in stat_columns_mutable
    ]
    return columns_immutable


# def get_grid_copy(
#     grid: gpd.GeoDataFrame,
#     dissolved: gpd.GeoDataFrame,
#     source_column: str,
#     stat_columns: Iterable,
# ) -> gpd.GeoDataFrame:
#     """
#     Creates a grid copy with the selected data.

#     This step is needed, if the data are to be rasterised.

#     Associate aggregated values of dissolved dataframe
#     to the dataframe of the grid cells containing them.

#     Column names for the values are prefixed with a '_' to
#     avoid clash with existing properties of the dataframe.

#     """
#     grid_copy = grid.copy()
#     for stat_column in stat_columns:
#         grid_copy.loc[dissolved.index, f"_{stat_column}"] = dissolved[
#             (source_column, stat_column)
#         ].values
#     return grid_copy


def get_shapes(gdf: gpd.GeoDataFrame, colname: str) -> list:
    """
    Extract and associate and format geometry and data in GeoDataFrame.

    """
    assert "geometry" in gdf.columns, f"Expected `geometry` in {gdf.columns=}"
    assert colname in gdf.columns, f"Expected {colname!r} in {gdf.columns=}."
    return [(shape, value) for (shape, value) in zip(gdf.geometry, gdf[colname])]
