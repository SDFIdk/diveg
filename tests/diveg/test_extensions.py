
import pandas as pd
import geopandas as gpd


from diveg.extensions import (
    safe_drop,
)


def test_safe_drop():
    raw = dict(x=[1, 2, 3], y=[4, 5, 6])
    df = pd.DataFrame(data=raw)
    gdf = gpd.GeoDataFrame(df)

    expected = pd.Index(['y'])
    columns = ['x', 'z']
    for frame in [df, gdf]:
        result = safe_drop(frame, columns)
        assert result == expected, f'Expected {result!r} to be {expected!r}'

