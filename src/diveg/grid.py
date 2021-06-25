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
        bounds: The bounding-box coordinates of the data that are to be gridded.
            Assumed to be a tuple containing (`minx`, `miny`, `maxx`, `maxy`)
            as returned by the property `gpd.GeoDataFrame.total_bounds` .
        crs: The coordinate system that the geometry should use.
            This must match the CRS of `bounds`.
        N_points_x: Number of points to cover along the x axis.
        N_points_y: Number of points to cover along the y axis.
        point_distance: The distance between two adjacent points.
            For now, this must be worked out beforehand.

    Returns:
        A tuple with the following instances:

            *   A GeoDataFrame with the grid.
            *   A Grid2DInfo instance with information about the grid.

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

    _NODATA_VALUE: int = -9999

    def __init__(self, grid: gpd.GeoDataFrame, info: Grid2DInfo):
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

        self._total_bounds: tuple[float] = None
        "Cached value of total bounds for the geometry."

        self._merged: gpd.GeoDataFrame = None
        "The resulting `GeoDataFrame` of the spatial join (binning) of input points with the grid."

        self._dissolved: gpd.GeoDataFrame = None
        "The resulting `GeoDataFrame` after aggregating merged points."

        self._rasterised: dict[tuple[str, str, Optional[Union[str, pyproj.CRS]]], np.ndarray] = {}
        "Cache for rasterised data."

        self._columns_imposed: list[Union[str, tuple[str, str]]] = None
        "List of column names for data with imposed values."

    @property
    def total_bounds(self) -> tuple[float]:
        """
        Return total bounds for the grid.

        A cached value is returned after the first call.

        """
        if self._total_bounds is None:
            self._total_bounds = self._grid.total_bounds
        return self._total_bounds

    def _merge(self, points: gpd.GeoDataFrame) -> "Grid":
        """
        Perform a spatial join between the grid and the given
        points, merging the points with the grid geometry.

        Args:
            points: A GeoDataFrame with the points that should be merged with the grid.
        
        Returns:
            The `Grid` instance itself.

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
        Groups the data by the Grid instance's geometry
        and calculates the specified aggregate values.
        
        Args:
            aggfunc: Specified aggregation methods as a dict.

        Returns:
            The `Grid` instance itself.

        Used input:
            index_right: The index that was automatically created+named when
            spatially joining the input points with the instances's (grid) geometry.

        """
        self._dissolved = self._merged.dissolve(by="index_right", aggfunc=aggfunc)
        return self

    def _attach_to_grid_geometry(self) -> "Grid":
        """
        Associate aggregated values of dissolved dataframe
        to the dataframe of the grid cells containing them.

        In other words, associate the aggregated data with the
        geometry of the grid cells (`Polygon` rectangles) of
        `self._grid` rather than the `MultiPoint`s of
        `self._dissolved`.

        This step is needed, if the data are to be rasterised.

        The index values of `_dissolved` is a subset of the index
        of `_grid`, so the data have to be assigned using the
        index of `_dissolved` to keep the data aligned.

        Returns:
            The `Grid` instance itself.

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

        Args:
            points: A GeoDataFrame with the points that should be merged with the grid.
            aggfunc: Specified aggregation methods as a dict.
                Note: This more limited use compared to the underlying
                Pandas functionality is to avoid operations on irrelevant data.
        
        """
        self._merge(points)
        self._dissolve(aggfunc)
        self._attach_to_grid_geometry()

    def select(self, column: Union[str, tuple[str, str]]) -> gpd.GeoDataFrame:
        """
        Return selected grid data as a GeoDataFrame
        with generic columns (`geometry', 'data').

        Args:
            column: The data column to extract.

        """
        return self._grid[['geometry', column]].rename(columns={column: 'data'})

    @property
    def _transform(self) -> rasterio.Affine:
        """
        Returns:
            An instance of `rasterio`'s `Affine` object with
            transformation data for raster-image production.

        """
        return from_bounds(*self.total_bounds, self.info.ncols, self.info.nrows)

    def _rasterise(self, column: Union[str, tuple[str, str]], crs: Union[str, pyproj.CRS] = None) -> tuple[np.ndarray, rasterio.Affine]:
        """
        Return array of rasterised data with affine-transformation data.

        This method always re-produces the output, since it is used by
        `Grid.rasterise(...)` which caches previously-created output.

        Args:
            column: The data column to rasterise.
            crs: Coordinates in which to project the data.

        Returns:
            A tuple with the following instances:
                *   The array of rasterised data.
                *   An Affine instance with transformation data.

        """
        grid_data = self.select(column).fillna(self._NODATA_VALUE)
        if crs is not None:
            grid_data = grid_data.to_crs(crs)
            transform = from_bounds(*grid_data.total_bounds, self.info.ncols, self.info.nrows)
        else:
            transform = self._transform
        raster = rasterize(
            shapes=get_shapes(grid_data, 'data'),
            out_shape=self.info.shape,
            transform=transform,
            all_touched=True,
            dtype=grid_data['data'].dtype,
        )
        return raster, transform

    def rasterise(self, column: Union[str, tuple[str, str]], crs: Union[str, pyproj.CRS] = None) -> tuple[np.ndarray, rasterio.Affine]:
        """
        Return possibly cached array of rasterised data with affine-transformation data.

        Args:
            column: The data column to rasterise.
            crs: Coordinates in which to project the data.

        Returns:
            A tuple with the following instances:
                *   The array of rasterised data.
                *   An Affine instance with transformation data.

        """
        if isinstance(column, tuple):
            key = column + (crs,)
        else:
            crs_str = crs if isinstance(crs, (str)) else crs.srs if isinstance(crs, pyproj.CRS) else None
            key = f'{column}_{crs_str}'
        if key not in self._rasterised:
            self._rasterised[key] = self._rasterise(column, crs)
        return self._rasterised[key]

    def save(self, filename: str, column: Union[str, tuple[str, str]], crs: Union[str, pyproj.CRS] = None) -> None:
        """
        Save selected grid data as a GeoTIFF raster image.

        A default CRS is applied to the underlying data from which
        the selection draws, but alternative coordinates can be given.

        Args:
            filename: The complete output path of the filename to save the data.
            column: The column name of the grid data to save.
            crs: Alternative coordinates to project the data in, before saving the raster image.

        Side effects:
            Writes an image file to the selected destination path.

        """
        if crs is None:
            crs = self._grid.crs
        rasterised, transform = self.rasterise(column, crs)
        kwargs = dict(
            driver="GTiff",
            height=self.info.nrows,
            width=self.info.ncols,
            count=1,
            crs=crs,
            transform=transform,
            dtype=rasterised.dtype,
        )
        with rasterio.open(filename, "w+", **kwargs) as output:
            output.write(rasterised, indexes=1)

    def get_columns_imposable(self, stats: Iterable[str]) -> list[tuple[str, str]]:
        """
        Returns all column names in the grid with the selected statistics.

        Args:
            stats: A container of strings with names of the stat columns in the grid.

        Returns:
            A list of column indices pointing to the data in the grid that can have be overwritten.

        """
        return self._grid.columns.drop(get_columns_immutable(self._grid, stats))

    def impose(self, lo: "Grid", columns: Iterable[Any] = None, filters: Iterable[Callable[[gpd.GeoSeries], bool]] = None) -> None:
        """
        Overwrite (impose) values of specified columns in the calling Grid instance's
        dataset with those of another Grid instance with lower resolution than the
        calling instance, given the truth value returned by the specified filters.

        Args:
            lo: Another Grid instance assumed to have the following properties:
                *   It has the same columns as the calling instance (otherwise this method is useless).
                *   It must also have a grid of lower resolution than the calling instance.
                *   It should be based on the same underlying point data.
                *   It should cover at least the same area as the calling instance.
            columns: Columns in the grid whose values should be overwritten by the lower-resolution grid (if criteria met)
                If no column names are given, all the data columns in the grid will be imposable.
            filters: A list of functions that can be used with (Geo)Pandas's (g)df.apply() method.
                If no filter is given, no cells will be imposed.

                TODO: IMPLEMENTATION CONSIDERATION: IF NOT FILTER IS GIVEN; THIS CURRENTLY MEANS THAT NOTHNG IS OVERWRITTEN:
                TODO: WHAT SHOULD BE THE BEHAVIOUR FOR THIS CASE? THAT ALL DATA SHOULD BE OVERWRITTEN. BUT THEN THERE IS NO
                TODO: NEED TO IMPOSE UNLESS ONE WANTS A HIGH-RESOLUTION GRID WITH LOW-RESOLUTION RESULTS.

        Side effects:
            Stores a copy of specified data with filtered rows of data overwritten in the grid data frame.
                The imposed data will have '_imposed' as a suffix to the column names in the following way:
                
                        Before         ->            After
                    ('VEL_V', 'std')        ('VEL_V', 'std_imposed')

                Additionally, a new column `overwritten` is added, which contains the value 1,
                if the original data in a given row was overwritten, and zero otherwise.

        """
        # Input validation or correction

        # Get columns that should have their values overwritten
        columns_imposable = self._grid.columns.drop(['geometry', 'data']) if columns is None else columns 

        # Avoid looping over `None`
        filters = filters or []

        ## Steps:

        # Copy selected (imposable) columns to a new container, keeping the same columns names.
        imposed = self._grid[columns_imposable].copy()

        # Now the program knows which columns in the higher-resolution
        # grid that we want to overwrite with the lower-resolution grid data.
        # Look at what what rows of data in the filter columns and decide, which
        # cells whose stats should be overwritten with values that we assume are better.

        # Apply each filter to boolean array that starts with all (i.e. none selected) values false. 
        boolean_index_imposed_overwrite = np.zeros_like(self._grid.index.values).astype(bool)
        
        for filter_func in filters:
            boolean_index_imposed_overwrite |= self._grid.apply(filter_func, axis=1).values

        # Add a special column that signifies what rows in the copied data that were overwritten (imposed)
        imposed['overwritten'] = 0
        # A value of one is added to this column of a given row, when its values have been overwritten.
        # NOTE: This happens, everytime the values are overwritten. Thus, the overwritten value is the
        # number of times the data were overwritten. It may therefore serve as a quality check of the
        # resulting imposed values. If more than one of the low-resolution cells contain the points
        # that are in the higher-resolution cell, the values will be overwritten for each match.

        # Next, we need indices into lower-resolution grid to know what cells to copy from.
        # Loop over rows to overwrite and get the index of each cell in the lower-resolution
        # grid that contains the points in the filtered data.

        # Search for the row-index of the lower-resolution grid whose corresponding
        # value should replace the 'bad' values in the high-resolution cell.

        # NOTE: self._dissolved containes the original geometry of the point locations as multipoints
        # since _dissolved only has the geometry of parts of the grid, we need to
        # * convert the boolean index of overwritable columns for the complete grid (as exemplified
        #   by `_grid` and `imposed`) to the indices of the rows that are to be overwritten.
        # * This index is the same for all, including `_dissolved`.
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

        # Save the results in `_grid`
        new_column_names = [
            (layer_column, f'{stat_column}_imposed')
            for (layer_column, stat_column)
            in imposed.columns.drop(['overwritten'])
        ] + ['overwritten']

        self._grid[new_column_names] = imposed[imposed.columns].values
        self._columns_imposed = new_column_names

    @property
    def columns_imposed(self) -> list:
        return self._columns_imposed

    @property
    def data_columns(self) -> list:
        return self._grid.columns.drop(['geometry', 'data'])


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


def get_shapes(gdf: gpd.GeoDataFrame, colname: str) -> list:
    """
    Extract and associate and format geometry and data in GeoDataFrame.

    """
    assert "geometry" in gdf.columns, f"Expected `geometry` in {gdf.columns=}"
    assert colname in gdf.columns, f"Expected {colname!r} in {gdf.columns=}."
    return [(shape, value) for (shape, value) in zip(gdf.geometry, gdf[colname])]
