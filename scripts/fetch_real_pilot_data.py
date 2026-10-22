"""
scripts/fetch_real_pilot_data.py
==================================
Automated retrieval of authentic open geospatial exploration data for the
Eastern Goldfields (Yilgarn Craton, WA) pilot area.

Sources:
1. GSWA MINEDEX: Official Western Australia mineral deposits and mines via
   DMIRS SLIP Public ArcGIS REST Service (Layer 0).
2. GSWA WAMEX: Official open-file mineral exploration drillhole collars via
   DMIRS SLIP Public ArcGIS REST Service (Layer 28).
3. GSWA Bedrock Geology: 1:500 000 State interpreted bedrock geology polygons via
   DMIRS SLIP Public ArcGIS REST Service (Layer 4).
4. GSWA Faults & Shear Zones: 1:500 000 State linear structures via
   DMIRS SLIP Public ArcGIS REST Service (Layer 1).
5. Copernicus GLO-30 DEM: 30 m resolution digital surface model streamed directly
   from the AWS Open Data public repository and reprojected to the 50 m MGA Zone 51 grid.

CRS: EPSG:7851 (GDA2020 / MGA Zone 51)
Grid AOI: [350000.0, 6590000.0, 360000.0, 6600000.0]
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon


def esri_to_shapely(geom: dict[str, Any] | None) -> Any:
    """Convert Esri JSON geometry (rings / paths) to a Shapely geometry."""
    if not geom:
        return None
    if "rings" in geom:
        rings = geom["rings"]
        polys = [Polygon(r) for r in rings if len(r) >= 3]
        if not polys:
            return None
        return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    elif "paths" in geom:
        paths = geom["paths"]
        lines = [LineString(p) for p in paths if len(p) >= 2]
        if not lines:
            return None
        return MultiLineString(lines) if len(lines) > 1 else lines[0]
    return None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fetch_real_data")

# AOI definition in EPSG:7851 (MGA Zone 51)
AOI_BBOX_EPSG7851 = (350000.0, 6590000.0, 360000.0, 6600000.0)
GRID_RES_M = 50.0

SLIP_BASE_MINING = "https://public-services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/Industry_and_Mining/MapServer"
SLIP_BASE_GEOLOGY = "https://public-services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/Geology_and_Soils_Map/MapServer"
COPERNICUS_DEM_S3 = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/Copernicus_DSM_COG_10_S31_00_E121_00_DEM/Copernicus_DSM_COG_10_S31_00_E121_00_DEM.tif"


def query_arcgis_rest(
    service_url: str,
    layer_id: int,
    bbox: tuple[float, float, float, float],
    in_sr: int = 7851,
    out_sr: int = 7851,
    where: str = "1=1",
    max_records: int = 2000,
) -> list[dict[str, Any]]:
    """Query an ArcGIS REST layer with pagination support."""
    endpoint = f"{service_url}/{layer_id}/query"
    geom_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    
    all_features: list[dict[str, Any]] = []
    offset = 0
    batch_size = 1000

    while offset < max_records:
        params = {
            "where": where,
            "geometry": geom_str,
            "geometryType": "esriGeometryEnvelope",
            "inSR": str(in_sr),
            "outSR": str(out_sr),
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "resultOffset": str(offset),
            "resultRecordCount": str(batch_size),
            "f": "json",
        }
        url = f"{endpoint}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (MPM Ingestion)"})
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.error("ArcGIS REST request failed: %s", exc)
            break

        features = data.get("features", [])
        if not features:
            break
        all_features.extend(features)
        offset += len(features)
        
        # Stop if server returned fewer than requested batch
        if len(features) < batch_size:
            break

    logger.info("Retrieved %d features from layer %d", len(all_features), layer_id)
    return all_features


def fetch_minedex_deposits(dest_path: Path) -> gpd.GeoDataFrame:
    """Fetch real MINEDEX gold deposits and mines in the AOI."""
    logger.info("Fetching real MINEDEX gold occurrences from DMIRS SLIP Layer 0...")
    features = query_arcgis_rest(
        service_url=SLIP_BASE_MINING,
        layer_id=0,
        bbox=AOI_BBOX_EPSG7851,
        where="target_com LIKE '%GOLD%'",
        max_records=5000,
    )
    
    records = []
    geoms = []
    for f in features:
        attrs = f["attributes"]
        geom = f.get("geometry")
        if geom and "x" in geom and "y" in geom:
            geoms.append(Point(geom["x"], geom["y"]))
            records.append(attrs)

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:7851")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dest_path, driver="GPKG")
    logger.info("Saved %d MINEDEX features to %s", len(gdf), dest_path)
    return gdf


def fetch_wamex_drillholes(dest_path: Path, max_holes: int = 5000) -> gpd.GeoDataFrame:
    """Fetch real open-file WAMEX drillholes in the AOI."""
    logger.info("Fetching real WAMEX drillholes from DMIRS SLIP Layer 28...")
    features = query_arcgis_rest(
        service_url=SLIP_BASE_MINING,
        layer_id=28,
        bbox=AOI_BBOX_EPSG7851,
        where="1=1",
        max_records=max_holes,
    )

    records = []
    geoms = []
    for f in features:
        attrs = f["attributes"]
        geom = f.get("geometry")
        if geom and "x" in geom and "y" in geom:
            geoms.append(Point(geom["x"], geom["y"]))
            records.append(attrs)

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:7851")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dest_path, driver="GPKG")
    logger.info("Saved %d WAMEX drillholes to %s", len(gdf), dest_path)
    return gdf


def fetch_bedrock_geology(dest_path: Path) -> gpd.GeoDataFrame:
    """Fetch real GSWA 1:500 000 interpreted bedrock geology polygons."""
    logger.info("Fetching GSWA 1:500k Bedrock Geology from DMIRS SLIP Layer 4...")
    features = query_arcgis_rest(
        service_url=SLIP_BASE_GEOLOGY,
        layer_id=4,
        bbox=AOI_BBOX_EPSG7851,
        where="1=1",
        max_records=1000,
    )

    records = []
    geoms = []
    for f in features:
        attrs = f["attributes"]
        geom = f.get("geometry")
        poly = esri_to_shapely(geom)
        if poly is not None and not poly.is_empty:
            geoms.append(poly)
            records.append(attrs)

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:7851")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dest_path, driver="GPKG")
    logger.info("Saved %d geology polygons to %s", len(gdf), dest_path)
    return gdf


def fetch_linear_structures(dest_path: Path) -> gpd.GeoDataFrame:
    """Fetch real GSWA linear fault/shear structures."""
    logger.info("Fetching GSWA Linear Structures from DMIRS SLIP Layer 1...")
    features = query_arcgis_rest(
        service_url=SLIP_BASE_GEOLOGY,
        layer_id=1,
        bbox=AOI_BBOX_EPSG7851,
        where="1=1",
        max_records=1000,
    )

    records = []
    geoms = []
    for f in features:
        attrs = f["attributes"]
        geom = f.get("geometry")
        line = esri_to_shapely(geom)
        if line is not None and not line.is_empty:
            geoms.append(line)
            records.append(attrs)

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:7851")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dest_path, driver="GPKG")
    logger.info("Saved %d fault line features to %s", len(gdf), dest_path)
    return gdf


def fetch_copernicus_dem(dest_path: Path) -> None:
    """Stream real Copernicus 30 m DEM and reproject to 50 m grid in EPSG:7851."""
    logger.info("Fetching and reprojecting real Copernicus 30 m DEM from AWS...")
    xmin, ymin, xmax, ymax = AOI_BBOX_EPSG7851
    width = int((xmax - xmin) / GRID_RES_M)
    height = int((ymax - ymin) / GRID_RES_M)
    dst_transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    dst_data = np.empty((height, width), dtype=np.float32)

    with rasterio.open(COPERNICUS_DEM_S3) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:7851",
            resampling=Resampling.bilinear,
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:7851",
        "transform": dst_transform,
    }
    with rasterio.open(dest_path, "w", **meta) as dst:
        dst.write(dst_data, 1)

    logger.info(
        "Saved reprojected DEM to %s (min=%.1f m, max=%.1f m, mean=%.1f m)",
        dest_path,
        float(dst_data.min()),
        float(dst_data.max()),
        float(dst_data.mean()),
    )


def main() -> None:
    dest_dir = Path("data/raw_real")
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print(" FETCHING REAL GEOSPATIAL EXPLORATION DATA (YILGARN CRATON, WA)   ")
    print("==================================================================")
    
    # 1. MINEDEX Deposits
    fetch_minedex_deposits(dest_dir / "minedex_deposits.gpkg")

    # 2. WAMEX Drillholes
    fetch_wamex_drillholes(dest_dir / "wamex_drillholes.gpkg", max_holes=5000)

    # 3. Bedrock Geology
    fetch_bedrock_geology(dest_dir / "geology.gpkg")

    # 4. Linear Structures (Faults/Shears)
    fetch_linear_structures(dest_dir / "faults.gpkg")

    # 5. Copernicus DEM
    fetch_copernicus_dem(dest_dir / "dem.tif")

    print("\n[SUCCESS] All real geospatial datasets successfully acquired and archived.")


if __name__ == "__main__":
    main()
