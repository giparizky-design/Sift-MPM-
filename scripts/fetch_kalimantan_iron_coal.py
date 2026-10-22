"""
scripts/fetch_kalimantan_iron_coal.py
=======================================
Automated retrieval and spatial auditing of official Iron (Fe) and Coal
deposits and mines across Kalimantan (Indonesian Borneo).

Source:
Kementerian Energi dan Sumber Daya Mineral (ESDM) / Badan Geologi
ArcGIS REST Services:
1. Metallic Minerals Layer (Iron / Besi):
   https://geoportal.esdm.go.id/gis4/rest/services/BGD_TU/Potensi_Sumber_Daya_dan_Cadangan_Mineral_Logam/MapServer/0
2. Coal Potential & Deposits Layer (Batubara):
   https://geoportal.esdm.go.id/gis4/rest/services/BGD_TU/Potensi_Batubara/MapServer/0

Outputs:
- data/borneo/kalimantan_iron_deposits.gpkg & .csv
- data/borneo/kalimantan_coal_deposits.gpkg & .csv
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fetch_kalimantan_fe_coal")

KALIMANTAN_BBOX = "108.5,-4.5,119.5,4.5"
ESDM_METAL_URL = "https://geoportal.esdm.go.id/gis4/rest/services/BGD_TU/Potensi_Sumber_Daya_dan_Cadangan_Mineral_Logam/MapServer/0/query"
ESDM_COAL_URL = "https://geoportal.esdm.go.id/gis4/rest/services/BGD_TU/Potensi_Batubara/MapServer/0/query"


def fetch_kalimantan_iron(dest_dir: Path) -> gpd.GeoDataFrame:
    """Fetch official iron ore deposits and occurrences across Kalimantan."""
    logger.info("Fetching Kalimantan Iron (Fe) deposits from ESDM Badan Geologi...")
    params = {
        "where": "lbunsur = 'Fe' OR jnskom LIKE '%Besi%'",
        "geometry": KALIMANTAN_BBOX,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "resultRecordCount": "2000",
        "f": "json",
    }
    url = f"{ESDM_METAL_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (MPM Geospatial Scraper)"})

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    features = data.get("features", [])
    logger.info("Retrieved %d iron features from ESDM", len(features))

    records: list[dict[str, Any]] = []
    points: list[Point] = []
    for f in features:
        attrs = f["attributes"]
        geom = f.get("geometry", {})
        x, y = geom.get("x"), geom.get("y")
        if x is not None and y is not None:
            attrs["longitude"] = float(x)
            attrs["latitude"] = float(y)
            points.append(Point(float(x), float(y)))
            records.append(attrs)

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(df, geometry=points, crs="EPSG:4326")

    # Standardize column naming
    gdf = gdf.rename(
        columns={
            "namobj": "deposit_name",
            "lokasilgm": "location",
            "jnskom": "commodity_type",
            "lbunsur": "chemical_symbol",
            "kellgm": "metal_group",
            "statdiklgm": "exploration_status",
            "bjhhip": "ore_hypothetical_mt",
            "bjhtrka": "ore_inferred_mt",
            "bjhtjuk": "ore_indicated_mt",
            "bjhtkur": "ore_measured_mt",
        }
    )

    dest_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = dest_dir / "kalimantan_iron_deposits.gpkg"
    csv_path = dest_dir / "kalimantan_iron_deposits.csv"

    gdf.to_file(gpkg_path, driver="GPKG")
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d iron deposit records to %s and %s", len(gdf), gpkg_path, csv_path)
    return gdf


def fetch_kalimantan_coal(dest_dir: Path) -> gpd.GeoDataFrame:
    """Fetch official coal deposits and active mine sites across Kalimantan."""
    logger.info("Fetching Kalimantan Coal (Batubara) deposits from ESDM Badan Geologi...")
    params = {
        "where": "1=1",
        "geometry": KALIMANTAN_BBOX,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "resultRecordCount": "2000",
        "f": "json",
    }
    url = f"{ESDM_COAL_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (MPM Geospatial Scraper)"})

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    features = data.get("features", [])
    logger.info("Retrieved %d coal features from ESDM", len(features))

    records: list[dict[str, Any]] = []
    points: list[Point] = []
    for f in features:
        attrs = f["attributes"]
        geom = f.get("geometry", {})
        x, y = geom.get("x"), geom.get("y")
        if x is not None and y is not None:
            attrs["longitude"] = float(x)
            attrs["latitude"] = float(y)
            points.append(Point(float(x), float(y)))
            records.append(attrs)

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(df, geometry=points, crs="EPSG:4326")

    # Standardize column naming
    gdf = gdf.rename(
        columns={
            "namobj": "deposit_name",
            "klsbb": "coal_calorific_class",
            "statdikbb": "exploration_status",
            "hipbb": "resources_hypothetical_mt",
            "trekabb": "resources_inferred_mt",
            "tjukbb": "resources_indicated_mt",
            "tkurbb": "resources_measured_mt",
            "totsdbb": "total_resources_mt",
            "totcadbb": "total_reserves_mt",
        }
    )

    dest_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = dest_dir / "kalimantan_coal_deposits.gpkg"
    csv_path = dest_dir / "kalimantan_coal_deposits.csv"

    gdf.to_file(gpkg_path, driver="GPKG")
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d coal deposit records to %s and %s", len(gdf), gpkg_path, csv_path)
    return gdf


def main() -> None:
    dest_dir = Path("data/borneo")
    print("==================================================================")
    print(" FETCHING KALIMANTAN IRON & COAL SITES (ESDM BADAN GEOLOGI API)   ")
    print("==================================================================")

    # 1. Iron deposits
    gdf_fe = fetch_kalimantan_iron(dest_dir)
    print(f"\n[IRON SUMMARY] Total Sites: {len(gdf_fe)}")
    print(gdf_fe["commodity_type"].value_counts())
    print("\nExploration Status:")
    print(gdf_fe["exploration_status"].value_counts())

    # 2. Coal deposits
    gdf_coal = fetch_kalimantan_coal(dest_dir)
    print(f"\n[COAL SUMMARY] Total Sites: {len(gdf_coal)}")
    print(gdf_coal["coal_calorific_class"].value_counts())
    print(f"Total Resources: {gdf_coal['total_resources_mt'].sum():,.1f} Million Tonnes")
    print(f"Total Reserves:  {gdf_coal['total_reserves_mt'].sum():,.1f} Million Tonnes")


if __name__ == "__main__":
    main()
