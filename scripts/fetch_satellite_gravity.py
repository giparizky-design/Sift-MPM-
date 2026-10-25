"""
scripts/fetch_satellite_gravity.py
==================================
Fetch authentic satellite gravity (Sandwell & Smith v33.1, 1-arc-minute grid)
from Scripps Institution of Oceanography (UCSD) for Kalimantan & Indonesia.
"""

from __future__ import annotations

import logging
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import griddata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_gravity")

TOPEX_CGI_URL = "https://topex.ucsd.edu/cgi-bin/get_data.cgi"


def fetch_gravity_chunk(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> np.ndarray:
    """Fetch an ASCII XYZ gravity chunk (mag=0.1) from Scripps topex.ucsd.edu."""
    params = {
        "north": f"{max_lat:.2f}",
        "south": f"{min_lat:.2f}",
        "west": f"{min_lon:.2f}",
        "east": f"{max_lon:.2f}",
        "mag": "0.1",
    }
    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(TOPEX_CGI_URL, data=data, headers={"User-Agent": "MPM-GravityFetcher/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        lines = resp.read().decode("utf-8", errors="ignore").splitlines()

    coords = [list(map(float, l.split())) for l in lines if len(l.split()) == 3]
    return np.array(coords, dtype=np.float32)


def build_gravity_geotiff(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    out_tif: Path,
    step_deg: float = 5.0,
    res_arcmin: float = 1.0,
) -> Path:
    """Download regional grid in sub-chunks, interpolate to regular raster, and write GeoTIFF."""
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    if out_tif.exists():
        logger.info("Found cached gravity GeoTIFF: %s", out_tif)
        return out_tif

    logger.info("Fetching Sandwell & Smith v33.1 gravity for [%.2f..%.2f, %.2f..%.2f]...", min_lon, max_lon, min_lat, max_lat)

    lon_steps = np.arange(min_lon, max_lon, step_deg)
    lat_steps = np.arange(min_lat, max_lat, step_deg)

    all_chunks = []
    for x in lon_steps:
        x2 = min(max_lon, x + step_deg)
        for y in lat_steps:
            y2 = min(max_lat, y + step_deg)
            logger.info("  Fetching tile lon [%.2f..%.2f], lat [%.2f..%.2f]...", x, x2, y, y2)
            chunk = fetch_gravity_chunk(x, y, x2, y2)
            if len(chunk) > 0:
                all_chunks.append(chunk)

    if not all_chunks:
        raise ValueError("Failed to retrieve any gravity points from Scripps UCSD.")

    pts = np.vstack(all_chunks)
    logger.info("Retrieved total %d gravity sounding points. Gridding to 1-arc-minute raster...", len(pts))

    # Grid parameters
    d_deg = res_arcmin / 60.0
    grid_lons = np.arange(min_lon, max_lon + d_deg * 0.5, d_deg, dtype=np.float32)
    grid_lats = np.arange(max_lat, min_lat - d_deg * 0.5, -d_deg, dtype=np.float32)  # North to South
    H = len(grid_lats)
    W = len(grid_lons)

    mesh_x, mesh_y = np.meshgrid(grid_lons, grid_lats)
    grid_z = griddata((pts[:, 0], pts[:, 1]), pts[:, 2], (mesh_x, mesh_y), method="linear")

    # Nearest neighbor fallback for any edge nans
    nan_mask = np.isnan(grid_z)
    if nan_mask.any():
        grid_z_near = griddata((pts[:, 0], pts[:, 1]), pts[:, 2], (mesh_x, mesh_y), method="nearest")
        grid_z[nan_mask] = grid_z_near[nan_mask]

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, W, H)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": W,
        "height": H,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
    }

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(grid_z.astype(np.float32), 1)

    logger.info("Successfully generated authentic satellite gravity GeoTIFF -> %s (%dx%d px)", out_tif, H, W)
    return out_tif


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent

    # Fetch for Kalimantan: 108.5 to 119.5 lon, -4.5 to 4.5 lat
    kal_tif = ROOT / "data" / "kalimantan" / "processed" / "sandwell_smith_gravity_kalimantan.tif"
    build_gravity_geotiff(108.5, -4.5, 119.5, 4.5, kal_tif, step_deg=5.5)
