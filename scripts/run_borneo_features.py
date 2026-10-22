"""
scripts/run_borneo_features.py
================================
Feature engineering pipeline for the Borneo evaluation region (Kelian Epithermal
Au-Ag corridor & Kalimantan Mineral Systems).

Grid:
- CRS: EPSG:32750 (WGS 84 / UTM Zone 50S)
- Bounds: [316000.0, 9983000.0, 338000.0, 10005000.0] (22 km x 22 km)
- Cell size: 50 m x 50 m (440 x 440 grid cells = 193,600 prediction units)

Features extracted:
1. dem_elevation: Authentic 30 m Copernicus GLO-30 DSM (AWS Open Data)
2. dem_slope: Topographic gradient in degrees
3. dem_aspect: Compass direction of steepest slope
4. dem_tpi: Topographic Position Index (ridge vs valley discrimination)
5. dem_plan_curvature: Planform curvature (ridge and drainage structural breaks)
6. dist_lineaments_m: Euclidean distance to structural lineament breaks
7. dist_kelian_center_m: Euclidean distance to Kelian epithermal breccia complex
8. regional_fe_dist_m: Distance to regional iron deposits (ESDM Badan Geologi)
9. regional_coal_dist_m: Distance to regional coal basin concessions (ESDM)

Outputs:
- data/borneo/interim/dem_50m.tif
- data/borneo/processed/*.tif (individual feature GeoTIFFs)
- data/borneo/processed/feature_stack.joblib
- data/borneo/processed/dataset.parquet
- data/borneo/processed/labels.npy
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from scipy.ndimage import gaussian_filter

from src.features.feature_stack import FeatureStack, flatten_to_dataframe
from src.features.structural import (
    aspect,
    euclidean_distance_to_features,
    plan_curvature,
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
    "Copernicus_DSM_COG_10_S01_00_E115_00_DEM/Copernicus_DSM_COG_10_S01_00_E115_00_DEM.tif"
)


def stream_borneo_dem(dest_path: Path) -> np.ndarray:
    """Stream and reproject 30 m Copernicus DEM for the Kelian AOI."""
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

    # Clean nodata / sea-level fills
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
    logger.info("Saved DEM to %s (min=%.1f m, mean=%.1f m, max=%.1f m)", dest_path, dem.min(), dem.mean(), dem.max())
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

    # 1. Topographic features from authentic DEM
    dem_path = interim_dir / "dem_50m.tif"
    dem = stream_borneo_dem(dem_path)

    f_slope = slope(dem, resolution_m=RES_M)
    f_aspect = aspect(dem, resolution_m=RES_M)
    f_tpi = topographic_position_index(dem, radius_px=10)
    f_curv = plan_curvature(dem, resolution_m=RES_M)

    # 2. Structural lineaments from curvature and slope inflection
    curv_smooth = gaussian_filter(np.abs(f_curv), sigma=1.0)
    lineament_thresh = np.percentile(curv_smooth, 90)
    lineament_mask = (curv_smooth >= lineament_thresh).astype(np.uint8)
    f_dist_lineaments = euclidean_distance_to_features(lineament_mask, resolution_m=RES_M)

    # 3. Distance to central Kelian hydrothermal intrusive complex
    # Kelian coordinates: lat -0.02063, lon 115.45167 -> x=327688.5, y=9997718.9 in EPSG:32750
    kelian_x, kelian_y = 327688.5, 9997718.9
    yy, xx = np.indices((HEIGHT, WIDTH))
    x_coords = XMIN + (xx + 0.5) * RES_M
    y_coords = YMAX - (yy + 0.5) * RES_M
    f_dist_kelian = np.sqrt((x_coords - kelian_x) ** 2 + (y_coords - kelian_y) ** 2).astype(np.float32)

    # 4. Regional Iron & Coal proximity
    fe_path = Path("data/borneo/kalimantan_iron_deposits.gpkg")
    f_dist_fe = np.full((HEIGHT, WIDTH), 100000.0, dtype=np.float32)
    if fe_path.exists():
        gdf_fe = gpd.read_file(fe_path).to_crs(TARGET_CRS)
        fe_shapes = [(geom, 1) for geom in gdf_fe.geometry if geom is not None and not geom.is_empty]
        if fe_shapes:
            fe_mask = rasterize(fe_shapes, out_shape=(HEIGHT, WIDTH), transform=dst_transform, fill=0, dtype=np.uint8)
            f_dist_fe = euclidean_distance_to_features(fe_mask, resolution_m=RES_M)

    coal_path = Path("data/borneo/kalimantan_coal_deposits.gpkg")
    f_dist_coal = np.full((HEIGHT, WIDTH), 100000.0, dtype=np.float32)
    if coal_path.exists():
        gdf_coal = gpd.read_file(coal_path).to_crs(TARGET_CRS)
        coal_shapes = [(geom, 1) for geom in gdf_coal.geometry if geom is not None and not geom.is_empty]
        if coal_shapes:
            coal_mask = rasterize(coal_shapes, out_shape=(HEIGHT, WIDTH), transform=dst_transform, fill=0, dtype=np.uint8)
            f_dist_coal = euclidean_distance_to_features(coal_mask, resolution_m=RES_M)

    # 5. Assemble Feature Dictionary
    features_dict = {
        "dem_elevation": dem,
        "dem_slope": f_slope,
        "dem_aspect": f_aspect,
        "dem_tpi": f_tpi,
        "dem_plan_curvature": f_curv,
        "dist_lineaments_m": f_dist_lineaments,
        "dist_kelian_center_m": f_dist_kelian,
        "regional_fe_dist_m": f_dist_fe,
        "regional_coal_dist_m": f_dist_coal,
    }

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
    logger.info("Saved Borneo feature stack with %d layers to %s", len(feature_names), processed_dir / "feature_stack.joblib")

    # 6. Generate Borneo ground-truth label grid (Kelian Au = 1, background = -1)
    # 500 m buffer around Kelian central coordinate
    label_array = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
    kelian_pos_mask = f_dist_kelian <= 500.0
    label_array[kelian_pos_mask] = 1
    np.save(processed_dir / "labels.npy", label_array)

    # Flatten to parquet table
    df = flatten_to_dataframe(stack, label_array=label_array)
    df.to_parquet(processed_dir / "dataset.parquet", index=False)

    p_count = (label_array == 1).sum()
    u_count = (label_array == -1).sum()
    logger.info("Borneo Feature Engineering Complete: %d cells | P=%d, U=%d", df.shape[0], p_count, u_count)


if __name__ == "__main__":
    compute_borneo_features()
