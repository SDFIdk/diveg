from typing import (
    Iterable,
    Callable,
    Union,
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
        self, grid: gpd.GeoDataFrame, info: Grid2DInfo, output_crs: pyproj.CRS = CRS_OUTPUT_DEFAULT, work_crs: pyproj.CRS = CRS_WORK_DEFAULT):
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

        self._total_bounds: tuple[float, float, float, float] = None
        "Cached value of total bounds for the geometry."

        self._merged: gpd.GeoDataFrame = None
        "The resulting `GeoDataFrame` of the spatial join (binning) of input points with the grid."

        self._dissolved: gpd.GeoDataFrame = None
        "The resulting `GeoDataFrame` after aggregating merged points."

        self._imposed: gpd.GeoDataFrame = None
        "The resulting `GeoDataFrame` after imposing values from a lower-resolution grid on the original data in this instance."

        self._rasterised: dict[tuple[str, str, Union[str, None]], np.ndarray] = {}
        "Cache for rasterised data."

    @property
    def total_bounds(self) -> tuple[float]:
        """
        Return total bounds for the grid.

        """
        if self._total_bounds is None:
            self._total_bounds = self._grid.total_bounds
        return self._total_bounds

    # @property
    # def output_grid(self) -> gpd.GeoDataFrame:
    #     """
    #     True if selected output coordinates.

    #     """
    #     if (
    #         self._output_grid is None
    #         or
    #         set(self._grid.columns) != set(self._output_grid.columns)
    #     ):
    #         self._output_grid = self._grid.to_crs(self._output_crs)
    #     return self._output_grid

    def _merge(self, points: gpd.GeoDataFrame) -> "Grid":
        """
        TODO: Explain left and within

        """
        # TODO (VERIFY): how="left" means that it is a left join, i.e. only the cells that cover at least one point are part of the result.
        self._merged = gpd.sjoin(points, self._grid, how="left", op="within")
        return self

    def _dissolve(self, aggfunc: dict) -> "Grid":
        """
        TODO: Explain index_right

        """
        self._dissolved = self._merged.dissolve(by="index_right", aggfunc=aggfunc)
        # self._dissolved.rename(columns={'geometry': ('geometry', '')}, inplace=True)
        # self._dissolved.columns = pd.MultiIndex.from_tuples(self._dissolved.columns)
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

        """
        # source_column_index_without_geometry = self._dissolved.columns[1:]
        # "Example source-column index: [('VEL_V', 'iqr'), ('VEL_V', 'mean'), ('VEL_V', 'std')]"

        # for source_column in source_column_index_without_geometry:
        #     # Note to self: when creating a new column with a tuple as column name, provide tuple in a list.
        #     self._grid.loc[self._dissolved.index, [source_column]] = self._dissolved[source_column].values

        # Alternative: could I just assign _grid.geometry to _dissolved.geometry and re-assign _grid with this?
        columns = self._dissolved.columns[1:]
        self._grid.loc[self._dissolved.index, columns] = self._dissolved[columns].values
        "Example source-column index: [('VEL_V', 'iqr'), ('VEL_V', 'mean'), ('VEL_V', 'std')]"

        # DEPRECATED: # Create a MultiIndex, so that the data 
        # DEPRECATED: self._grid.rename(columns={'geometry': ('geometry', '')}, inplace=True)
        # DEPRECATED: self._grid.columns = pd.MultiIndex.from_tuples(self._grid.columns)

        return self

    def process_points(self, points: gpd.GeoDataFrame, aggfunc: dict) -> None:
        """
        Perform the process from spatially binning point-data values
        to the grid-cell geometry, calculating statistical results and,
        finally, attach these results to their respective grid cells (bins).

        """
        self._merge(points)
        self._dissolve(aggfunc)
        self._attach_to_grid_geometry()

    def select(self, layer_column: str, stat_column: str) -> gpd.GeoDataFrame:
        """
        Return selected grid data as a GeoDataFrame with generic columns (`geometry', 'data').

        """
        # DEPRECATED: grid_data = self._grid[[('geometry', ''), (layer_column, stat_column)]]
        column = (layer_column, stat_column)
        return self._grid[['geometry', column]].rename(columns={column: 'data'})
        # DEPRECATED: The geometry needs to have its coordinates set or the CRS is assumed
        # DEPRECATED: to be 'naive' which cannot be transformed later with `.to_crs()`.
        # DEPRECATED: Note: This took some trial and error to figure out.
        # DEPRECATED: grid_data.geometry.set_crs(grid_data.crs, inplace=True)

    @property
    def _transform(self) -> rasterio.Affine:
        return from_bounds(*self.total_bounds, self.info.ncols, self.info.nrows)

    def _rasterise(self, layer_column: str, stat_column: str, crs: pyproj.CRS = None) -> np.ndarray:
        grid_data = self.select(layer_column, stat_column).dropna()
        if crs is not None:
            grid_data = grid_data.to_crs(crs)
        return rasterize(
            shapes=get_shapes(grid_data, 'data'),
            out_shape=self.info.shape,
            transform=self._transform,
            all_touched=True,
            dtype=grid_data['data'].dtype,
        )

    def rasterise(self, layer_column: str, stat_column: str, crs: pyproj.CRS = None) -> np.ndarray:
        key = (layer_column, stat_column, crs)
        if key not in self._rasterised:
            self._rasterised[key] = self._rasterise(layer_column, stat_column, crs)
        return self._rasterised[key]

    def save(self, filename: str, layer_column: str, stat_column: str, crs: pyproj.CRS = None) -> None:
        """
        Save grid data given by `layer_column` and `stat_column` as `filename`.

        A default CRS is applied to the underlying data from which the selection draws,
        but alternative coordinates can be given.

        """
        rasterised = self.rasterise(layer_column, stat_column, crs)
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

    def impose(self, lo: "Grid", filters: Iterable[Callable[[gpd.GeoSeries], bool]] = None) -> None:
        """
        Impose grid-cell values of given column to those of a
        grid with higher resolution, where the cells have worse
        data quality than the lower resolution grid.

        Returns:
            Geodataframe with the grid data for each of the following quantities:

            *   The corrected data
                -   Example: VEL_V as input column. This has two output frames: VEL_V_mean, and VEL_V_std.
                    Each of these two separate geodataframes will have their cvalues corrected at the places and in the same way with their corresponding lower-resolution grid-cell valuess.
            *   Data (0: unchanged, 1: changed) about which cells were changed.
            *   

        Assumptions and criteria:

        *   We need good local statistics, i.e. data with well-defined location should not be changed, unless the foundation for these values are too bad to be useful.
            Thus, we need to defined `too bad to be useful`.

        *   We assume, for now, that the grid cells that have their values changed, can have the value be that of the containing lower-resolution cell.
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
            is the case, when aggregating over larger-sized (ower-resolution) cells. The drawback of this is that is does not take into account our need for good local statistics.
        *   

        Aggregated data in the original dataset are used as input for the filtering, and the rest are considered data products that will be used in further analysis.

            Aggregated data used for filtering, refering to the high-resolution data:

            *   iqr: The interquartile range.
            *   count: The number of points in the cells.

            Aggregated data that will get copied and potentially overwritten with the lower-resolution grid data in this process:

            *   mean of the layer column, e.g. VEL_V
            *   std of the layer column, e.g. VEL_V (not VEL_V_STD which we do not use)

            Data that will be produced in order to illustrate the process:

            *    What columns changed?
            *   (considering) What is the difference between the changed values and the new (imposed) values?
            *   

        Select the (layer, stat) data that need to be manipulated. Example: Select layer_column `VEL_V` and its stat_columns, excluding the information/filter columns `iqr` and `count`.)

            Make a copy of the selected data, with the suffix to each column

            Then, using the filter functions, determine the cells that need to have their data overwritten.

        """

        # Work with empty cells (np.nan)? These will, presumably, be overwritten by values or NaN.
        # Also, the fewer changes to manage, the better (simpler) the maintainance.

        # Steps:
        # 0. Copy data (stat_columns) to new columns in _grid and suffix column names with, say, _changed.

        # Goal: Focus on the statistics (denoted by the column names), we are interested in.

        # Get columns that should have their values overwritten
        stat_columns_wanted = ('mean', 'std',)
        # Only 'geometry' column is a string, the rest are of the form
        # ('name_of_layer_column_of_input_data', 'label_for_some_aggregated_value')
        # For this reason, the geometry column (of the dataframe) is dropped from the looped-over list,
        # and it is added again (in front) to the resulting list created with the list comprehension.
        columns_immutable = ['geometry'] + [
            (layer_column, stat_column)
            for (layer_column, stat_column)
            in self._grid.columns.drop('geometry')
            # Here, we only check the value of the stat_column,
            # since these are the same for each layer_column
            if stat_column not in stat_columns_wanted
        ]
        # Now, we can remove thos columns from the list of columns
        # that we *want* to overwrite, when the criteria are met.
        columns_imposable = self._grid.columns.drop(columns_immutable)
        
        # Copy selected (imposable) columns to a new container, keeping the same columns names.
        imposed = self._grid[columns_imposable].copy()
        # Add the new data columns
        # Or we could do a comparison indices = (_imposed == _grid)

        """
        There are columns to copy from _grid, which are all imposable plus geometry.
        Imposable columns should be a list of the columns, excluding geometry, that can be changed.
        
        """

        # TODO: DO NOT ADD STAT_COLUMNS TO THE IMPOSED DATASET, SINCE THESE ARE NOT CHANED AND NEED NOT BE COMPIED OVER AS THEY ARE INCLUDED IN _grid.

        # 0.5 Create boolean array that filters out the rows, we do not need to change.

        # Now that the program knows which columns in the higher-resolution grid
        # that we want to overwrite (what we decided to call imposed) with the
        # lower-resolution grid data, we can look at what data values (i.e. rows)
        # in the columns that we use to decide, which cells whose stats should be
        # overwritten with values that we assume are better.

        # START BY HARDCODING THIS IN, BEFORE ADDING IT AS A METHOD PARAMETER
        filters_overwrite_when_true = [
            lambda row: row[('VEL_V', 'iqr')] > 3,
        ]

        # Apply each filter to boolean array that starts with all (i.e. none selected) values false. 
        boolean_index_imposed_overwrite = np.zeros_like(self._grid.index.values).astype(bool)
        for filter_func in filters_overwrite_when_true:
            boolean_index_imposed_overwrite |= self._grid.apply(filter_func, axis=1).values

        # Now we know what rows in the data that need to be overwritten.

        # Add a special column to imposed that contains, again, False as default.
        imposed['overwritten'] = False
        # When the change has actually been made, each row that was changed will
        # have this corresponding changed-bit set True as well.

        # We also need indices into lower-resolution grid to know what cells to copy from.

        # Loop over rows to overwrite and get the index of each cell in the lower-resolution grid that contains the points in the filtered data.

        embed(header=f'DEBUG: impose')
        # Search for the cell-index (replacement_index) of the lower-resolution grid whose corresponding
        # value should replace the 'bad' values in the high-resolution cell.
        # NOTE: self._dissolved containes the original geometry of the point locations as multipoints
        for (index_points, points) in self._dissolved.iloc[boolean_index_imposed_overwrite].iterrows():
            # Which cell in the lower-resolution grid are the points in the higher-resolution grid within?
            lo_indices = lo.sindex.query(points.geometry, predicate="within")
            assert len(lo_indices) == 1
            # Assign the selected statistical values of the lower-resolution cell to the higher-resolution cell.
            imposed.loc[index_points, columns_imposable] = lo.loc[lo_indices[0], columns_imposable].values
            imposed.iloc[index_points].overwritten = True

        # TODO: Try to do the above in a bulk operation instead of a Python for loop over each row.
        # Which cell in the lower-resolution grid covers the multipoint in the higher-resolution grid?
        # self_indices_covered = np.unique(self._dissolved.sindex.query_bulk(lo._grid.geometry, predicate='covers')[0])

        # Store the results in the instance.
        # self._imposed = imposed

        # Save the results in _grid?
        new_column_names = [
            (layer_column, f'{stat_column}_changed')
            for (layer_column, stat_column)
            in imposed.columns.drop(['geometry', 'overwritten'])
        ] + ['overwritten']

        self._grid[new_column_names] = imposed[imposed.columns.drop('geometry')].values

        embed(header=f'DEBUG: impose')



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


def get_shapes(gdf: gpd.GeoDataFrame, colname: str) -> list:
    """
    Extract and associate and format geometry and data in GeoDataFrame.

    """
    assert "geometry" in gdf.columns, f"Expected `geometry` in {gdf.columns=}"
    assert colname in gdf.columns, f"Expected {colname!r} in {gdf.columns=}."
    return [(shape, value) for (shape, value) in zip(gdf.geometry, gdf[colname])]
