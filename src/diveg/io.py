import pathlib

import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds


def load_insar(ifname: str, *, layer: str = "2D") -> gpd.GeoDataFrame:
    """
    Load the specified Geo Package (.gpkg) file's layer.

    """
    assert pathlib.Path(ifname).is_file()
    cache = pathlib.Path.home() / ".diveg/gdf.gz"
    cache.parent.mkdir(exist_ok=True)
    if cache.is_file():
        print("Load cache")
        return gpd.GeoDataFrame(pd.read_pickle(cache))
    gdf = gpd.read_file(ifname, layer=layer)
    gdf.to_pickle(cache)
    return gdf


def load_adm() -> gpd.GeoDataFrame:
    """
    Load commune borders to see the land around the cells

    """
    ifname = pathlib.Path.home() / "data" / "KOMMUNE.shp"
    assert pathlib.Path(ifname).is_file()
    return gpd.read_file(ifname).to_crs("epsg:4326")
