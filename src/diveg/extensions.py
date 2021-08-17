
from typing import (
    Union,
    Iterable,
)

import pandas as pd
import geopandas as gpd


def safe_drop(df: Union[pd.DataFrame, gpd.GeoDataFrame], columns: Iterable[str]) -> pd.Index:
    existing = [column for column in columns if column in df.columns]
    return df.columns.drop(existing)

