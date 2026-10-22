"""
scripts/fetch_borneo_data.py
==============================
Scrape and extract open mineral occurrences and geospatial exploration data
for the Borneo region (Kalimantan, Sarawak, Sabah).

Sources:
1. USGS MRDS (Mineral Resources Data System) via WFS 1.1.0 API.
   Extracts all recorded mineral deposits and occurrences across Borneo:
   - Latitude: -4.5 to 7.5
   - Longitude: 108.5 to 119.5
2. Audit of coordinates, identifying:
   - Genuine geolocated deposits (e.g. Kelian Au, Mamut Cu-Au, Bau Au-Sb, Busang Au)
   - Centroid artifacts (-2.0°, 114.0°) where historical records were assigned
     arbitrary island centroids by database compilers.
"""

from __future__ import annotations

import logging
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fetch_borneo")

BORNEO_BBOX_WGS84 = (-4.5, 108.5, 7.5, 119.5)  # minlat, minlon, maxlat, maxlon
USGS_WFS_URL = (
    "https://mrdata.usgs.gov/services/wfs/mrds?"
    "request=GetFeature&service=WFS&version=1.1.0&typeName=mrds&"
    "BBOX=-4.5,108.5,7.5,119.5,urn:ogc:def:crs:EPSG::4326&maxFeatures=500"
)


def fetch_borneo_mrds(dest_dir: Path) -> gpd.GeoDataFrame:
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Querying USGS MRDS WFS for Borneo mineral occurrences...")

    req = urllib.request.Request(USGS_WFS_URL, headers={"User-Agent": "Mozilla/5.0 (Borneo Prospectivity Scraper)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_str = resp.read().decode("utf-8", errors="ignore")

    root = ET.fromstring(xml_str)
    members = root.findall(".//{http://www.opengis.net/gml}featureMember")
    logger.info("Found %d features in WFS response", len(members))

    records = []
    points = []
    for m in members:
        f = list(m)[0]
        rec = {child.tag.split("}")[-1]: child.text for child in f}
        pos = f.find(".//{http://www.opengis.net/gml}pos")
        if pos is not None and pos.text:
            coords = [float(x) for x in pos.text.split()]
            lat, lon = coords[0], coords[1]
            rec["latitude"] = lat
            rec["longitude"] = lon
            points.append(Point(lon, lat))
            
            # Detect island-centroid artifact
            is_centroid = (abs(lat - (-1.9982)) < 0.05) and (abs(lon - 113.9995) < 0.05)
            rec["is_centroid_artifact"] = is_centroid
            records.append(rec)

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(df, geometry=points, crs="EPSG:4326")

    # Export to GeoPackage and CSV
    gpkg_path = dest_dir / "borneo_mrds_occurrences.gpkg"
    csv_path = dest_dir / "borneo_mrds_occurrences.csv"
    
    gdf.to_file(gpkg_path, driver="GPKG")
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d records to %s and %s", len(gdf), gpkg_path, csv_path)

    # Summary
    n_artifacts = df["is_centroid_artifact"].sum()
    n_genuine = len(df) - n_artifacts
    logger.info("Genuine geolocated deposits: %d | Centroid artifacts: %d", n_genuine, n_artifacts)
    return gdf


if __name__ == "__main__":
    out_dir = Path("data/borneo")
    fetch_borneo_mrds(out_dir)
