"""Build CBSA_20m.geojson from the Census Bureau cartographic boundary file.

Downloads the 1:20m CBSA shapefile (cb_2020_us_cbsa_20m), filters to the
CBSAs present in Modeling_Table.csv, and writes Methoden Data/Geo Data/CBSA_20m.geojson.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MODELING_TABLE = ROOT / "Methoden Data" / "Modeling_Table.csv"
OUT_GEOJSON = ROOT / "Methoden Data" / "Geo Data" / "CBSA_20m.geojson"
CENSUS_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_cbsa_20m.zip"


def main() -> None:
    cbsas = (
        pd.read_csv(MODELING_TABLE, usecols=["CBSA_CODE"])
        .drop_duplicates()["CBSA_CODE"]
        .astype(str)
        .str.zfill(5)
    )
    print(f"Modeling table has {cbsas.nunique()} unique CBSA codes.")

    print(f"Downloading {CENSUS_URL} ...")
    resp = requests.get(CENSUS_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        tmp = OUT_GEOJSON.parent / "_cb_2020_us_cbsa_20m"
        tmp.mkdir(parents=True, exist_ok=True)
        zf.extractall(tmp)
        shp = next(tmp.glob("*.shp"))
        gdf = gpd.read_file(shp).to_crs(epsg=4326)

    gdf = gdf.rename(columns={"CBSAFP": "CBSA_CODE", "NAME": "CBSA_TITLE"})
    gdf["CBSA_CODE"] = gdf["CBSA_CODE"].astype(str).str.zfill(5)
    gdf = gdf[gdf["CBSA_CODE"].isin(cbsas)][["CBSA_CODE", "CBSA_TITLE", "geometry"]]
    print(f"Filtered shapefile to {len(gdf)} CBSAs (target: {cbsas.nunique()}).")

    OUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    if OUT_GEOJSON.exists():
        OUT_GEOJSON.unlink()
    gdf.to_file(OUT_GEOJSON, driver="GeoJSON")
    print(f"Wrote {OUT_GEOJSON} ({OUT_GEOJSON.stat().st_size / 1024:.1f} KB).")

    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    main()
