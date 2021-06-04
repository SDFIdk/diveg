
from dataclasses import dataclass

import numpy as np


@dataclass
class Grid2DInfo:
    """
    An object with grid metadata.

    Attributes need to be set when instantiating.

    Properties are derived.

    Attributes:
        cell_width: The width of the grid cell in meters.
        cell_height: The height of the grid cell in meters.
        grid_x: The coordinate range of xmin values for the cells.
        grid_y: The coordinate range of ymin values for the cells.

    Properties:
        shape: The number of rows and columns in the grid in that order.
        nrows: The number of rows (number of cells) along the height of the grid (y direction).
        ncols: The number of columns (number of cells) along the width of the grid (x direction).

    """
    cell_width: float
    cell_height: float
    grid_x: np.ndarray
    grid_y: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (
            self.grid_y.size,
            self.grid_x.size,
        )

    @property
    def nrows(self) -> int:
        return self.grid_y.size

    @property
    def ncols(self) -> int:
        return self.grid_x.size

