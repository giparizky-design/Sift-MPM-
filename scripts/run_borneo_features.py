"""
scripts/run_borneo_features.py
================================
100% Clean, Leak-Free Feature Engineering Pipeline for the Borneo evaluation
region (Kelian Epithermal Au-Ag corridor, East Kalimantan).

STRICT NO-LEAKAGE GUARANTEE:
- Input Features (X): Derived 100% from the authentic 30 m Copernicus GLO-30
  Satellite Radar DSM (AWS Open Data) and structural/geomechanical physics.
  ZERO mine coordinates (no dist_kelian_center_m, no regional_fe_dist_m,
  no regional_coal_dist_m) are ever used in X.
- Target Labels (y): Ground-truth mine locations (USGS MRDS Kelian + official
  ESDM GeoRIMA deposits inside the 22 km x 22 km window) are used ONLY as
  positive labels (y = 1) for PU training and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter
from shapely.geometry import Point

from src.features.feature_stack import FeatureStack, flatten_to_dataframe
from src.features.structural import (
    aspect,
    dilation_tendency,
    euclidean_distance_to_features,
    plan_curvature,
    resolve_stress_on_vertical_fault,
    slip_tendency,
    slope,
    topographic_position_index,
)
from src.utils.io import save_raster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("borneo_features")

# Kelian AOI configuration in EPSG:32750
XMIN, YMIN, XMAX, YMAX = 316000.0, 9983000.0, 338000.0, 10005000.0
RES_M = 50.0
WIDTH = int((XMAX - XMIN) / RES_M)   # 440
HEIGHT = int((YMAX - YMIN) / RES_M)  # 440
TARGET_CRS = "EPSG:32750"

COPERNICUS_DEM_S01_URL = (
    "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
    "Copernicus_DSM_COG_10_S01_00_E115_00_DEM/Copernicus_DSM_COG_10_S01_00_DEM.tif"
)


def stream_borneo_dem(dest_path: Path) -> np.ndarray:
    """Load cached or stream 30 m Copernicus DEM for the Kelian AOI."""
    if dest_path.exists():
        with rasterio.open(dest_path) as src:
            dem = src.read(1).astype(np.float32)
        logger.info("Loaded authentic Copernicus GLO-30 DEM from %s", dest_path)
        return dem

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dst_transform = from_bounds(XMIN, YMIN, XMAX, YMAX, WIDTH, HEIGHT)
    dem = np.full((HEIGHT, WIDTH), -9999.0, dtype=np.float32)

    logger.info("Streaming Copernicus DEM (S01 E115) from AWS Open Data...")
    with rasterio.open(COPERNICUS_DEM_S01_URL) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dem,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
        )

    valid = dem > -500.0
    if not valid.all():
        dem[~valid] = np.nanmedian(dem[valid])

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": WIDTH,
        "height": HEIGHT,
        "count": 1,
        "crs": TARGET_CRS,
        "transform": dst_transform,
    }
    save_raster(dem, profile, str(dest_path))
    return dem


def compute_borneo_features() -> None:
    interim_dir = Path("data/borneo/interim")
    processed_dir = Path("data/borneo/processed")
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dst_transform = from_bounds(XMIN, YMIN, XMAX, YMAX, WIDTH, HEIGHT)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": WIDTH,
        "height": HEIGHT,
        "count": 1,
        "crs": TARGET_CRS,
        "transform": dst_transform,
    }

    # 1. Pure Satellite Topographic Derivatives from authentic Copernicus GLO-30 DSM
    dem_path = interim_dir / "dem_50m.tif"
    dem = stream_borneo_dem(dem_path)

    f_slope = slope(dem, resolution_m=RES_M).astype(np.float32)
    f_aspect = aspect(dem, resolution_m=RES_M).astype(np.float32)
    f_tpi_local = topographic_position_index(dem, radius_px=10).astype(np.float32)
    f_tpi_dome = topographic_position_index(dem, radius_px=35).astype(np.float32)
    f_curv = plan_curvature(dem, resolution_m=RES_M).astype(np.float32)

    # Terrain Ruggedness Index (local relief standard deviation over 350m window)
    mean_dem = uniform_filter(dem, size=7)
    mean_sq = uniform_filter(dem ** 2, size=7)
    f_tri = np.sqrt(np.maximum(mean_sq - mean_dem ** 2, 0.0)).astype(np.float32)

    # 2. Structural Lineaments & Geomechanical Stress Tensor from DEM Curvature/Gradient
    curv_smooth = gaussian_filter(np.abs(f_curv), sigma=1.2)
    lineament_thresh = np.percentile(curv_smooth, 88)
    lineament_mask = (curv_smooth >= lineament_thresh).astype(np.uint8)
    f_dist_lineaments = euclidean_distance_to_features(lineament_mask, resolution_m=RES_M).astype(np.float32)
    f_lineament_density = gaussian_filter(lineament_mask.astype(np.float32), sigma=8.0).astype(np.float32)

    # Resolve fault strike from topographic gradient orientation and compute Ts, Td under N35E Kutai compression
    gy, gx = np.gradient(gaussian_filter(dem, sigma=2.0), RES_M)
    strike_deg = (np.degrees(np.arctan2(gx, -gy)) % 180.0).astype(np.float32)
    prox_struct = np.exp(-f_dist_lineaments / 400.0).astype(np.float32)
    sigma_n, tau = resolve_stress_on_vertical_fault(
        strike_deg=strike_deg,
        sigma1=80.0,
        sigma3=25.0,
        sigma1_azimuth_deg=35.0,
    )
    f_slip_ts = (slip_tendency(sigma_n, tau) * (0.35 + 0.65 * prox_struct)).astype(np.float32)
    f_dilation_td = (dilation_tendency(sigma_n, 80.0, 25.0) * (0.35 + 0.65 * prox_struct)).astype(np.float32)

    # 3. Regolith / Alluvial Cover Thickness Proxy (purely from valley floor geometry & slope)
    norm_elev = (dem - dem.min()) / (np.ptp(dem) + 1e-6)
    norm_slope = np.clip(f_slope / 35.0, 0.0, 1.0)
    f_regolith_cover_m = np.clip((1.0 - norm_slope) * (1.0 - 0.5 * norm_elev) * 65.0, 2.0, 80.0).astype(np.float32)

    # 100% Leak-Free Feature Dictionary (ZERO mine coordinates or distances!)
    features_dict = {
        "dilation_tendency_Td": f_dilation_td,
        "slip_tendency_Ts": f_slip_ts,
        "dem_elevation": dem,
        "dem_slope": f_slope,
        "dem_aspect": f_aspect,
        "dem_tpi_local": f_tpi_local,
        "dem_tpi_dome": f_tpi_dome,
        "dem_plan_curvature": f_curv,
        "terrain_ruggedness_tri": f_tri,
        "dist_lineaments_m": f_dist_lineaments,
        "lineament_density": f_lineament_density,
        "regolith_cover_m": f_regolith_cover_m,
    }

    # Remove old leaky GeoTIFFs if present
    for old_leaky in ("dist_kelian_center_m.tif", "regional_fe_dist_m.tif", "regional_coal_dist_m.tif"):
        p_old = processed_dir / old_leaky
        if p_old.exists():
            p_old.unlink()

    feature_arrays = []
    feature_names = list(features_dict.keys())
    for name, arr in features_dict.items():
        save_raster(arr, profile, str(processed_dir / f"{name}.tif"))
        feature_arrays.append(arr)

    stacked_array = np.stack(feature_arrays, axis=0).astype(np.float32)
    quality_mask = np.isnan(stacked_array).any(axis=0)

    stack = FeatureStack(
        array=stacked_array,
        feature_names=feature_names,
        profile=profile,
        data_quality_mask=quality_mask,
    )
    joblib.dump(stack, processed_dir / "feature_stack.joblib")
    logger.info("Saved 100%% leak-free Borneo feature stack (%d channels) to %s", len(feature_names), processed_dir / "feature_stack.joblib")

    # 4. Ground-Truth Label Grid (y ONLY — never in X!)
    # Includes Kelian USGS MRDS + official ESDM GeoRIMA gold/mineral occurrences inside the AOI
    mine_pts_lonlat = [(115.45167, -0.02063)]  # Kelian Deposit
    georima_csv = Path("data/indonesia/georima_all_indonesia_nodes.csv")
    if georima_csv.exists():
        df_g = pd.read_csv(georima_csv)
        for _, r in df_g.iterrows():
            lon, lat = float(r["lon"]), float(r["lat"])
            if 115.30 <= lon <= 115.60 and -0.20 <= lat <= 0.10:
                mine_pts_lonlat.append((lon, lat))

    gdf_pts = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in mine_pts_lonlat],
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)

    yy, xx = np.indices((HEIGHT, WIDTH))
    x_coords = XMIN + (xx + 0.5) * RES_M
    y_coords = YMAX - (yy + 0.5) * RES_M

    label_array = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
    for pt in gdf_pts.geometry:
        if XMIN <= pt.x <= XMAX and YMIN <= pt.y <= YMAX:
            dist = np.hypot(x_coords - pt.x, y_coords - pt.y)
            label_array[dist <= 350.0] = 1

    np.save(processed_dir / "labels.npy", label_array)
    df = flatten_to_dataframe(stack, label_array=label_array)
    df.to_parquet(processed_dir / "dataset.parquet", index=False)

    p_count = int((label_array == 1).sum())
    u_count = int((label_array == -1).sum())
    logger.info("Borneo Clean Feature Engineering Complete: %d cells | P=%d, U=%d", df.shape[0], p_count, u_count)


if __name__ == "__main__":
    compute_borneo_features()
