
import geopandas as gpd
from shapely.geometry import (
    MultiPoint,
    Point,
)


from diveg.grid import (
        get_columns_immutable,
        # get_grid_copy,
        get_shapes,
)


def test_get_columns_immutable():
    # Arrange
    gdf = gpd.GeoDataFrame(
        data={
            'geometry': [
                MultiPoint([
                    Point(0, 0),
                    Point(1, 1),
                    Point(2, 2),
                    Point(3, 3),
                ]),
                MultiPoint([
                    Point(2, 2),
                    Point(4, 4),
                    Point(6, 6),
                    Point(8, 8),
                ]),
            ],
            ('LAYER_COLUMN_1', 'mean'): [
                0,
                1,
            ],
            ('LAYER_COLUMN_1', 'std'): [
                2,
                4,
            ],
            ('LAYER_COLUMN_1', 'iqr'): [
                10,
                20,
            ],
            ('LAYER_COLUMN_2', 'mean'): [
                0,
                1,
            ],
            ('LAYER_COLUMN_2', 'std'): [
                2,
                4,
            ],
            ('LAYER_COLUMN_2', 'iqr'): [
                30,
                60,
            ],
        }
    )

    # expected = [
    #     ('LAYER_COLUMN_1', 'mean'),
    #     ('LAYER_COLUMN_1', 'std'),
    #     ('LAYER_COLUMN_2', 'mean'),
    #     ('LAYER_COLUMN_2', 'std'),
    # ]
    expected = [
        'geometry',
        ('LAYER_COLUMN_1', 'iqr'),
        ('LAYER_COLUMN_2', 'iqr'),
    ]

    stat_columns_wanted = ('mean', 'std')

    # Act
    result = get_columns_immutable(gdf, stat_columns_wanted)
    result = list(result)

    # Assert
    assert result == expected, f'Expected `{result!r}` to be `{expected!r}`'


def test_get_shapes():
    raise NotImplementedError


def test_build_grid():
    raise NotImplementedError

