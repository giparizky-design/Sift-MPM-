"""scripts/build_kutai_basin_100m.py
==================================
Builds the 100m Ultra-HD Feature Cube and Multimodal Dataset for the entire Kutai Basin.
Spatial Extent: Longitude [115.0..118.0], Latitude [-2.0..1.4]
Spatial Resolution: 100m (0.0009 degrees, 3778 rows x 3333 cols = 12,592,074 cells)
Domain: Sedimentary Coal Basin Systems (SIFT K=3 Multi-Rank)
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as win_from_bounds

from src.features.feature_stack import FeatureStack
from src.models.pu_pipeline import verify_no_raw_pixel_leakage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_kutai_basin_100m")

ROOT_DIR = Path(__file__).resolve().parent.parent
KAL_PROC = ROOT_DIR / "data" / "kalimantan" / "processed"
OUT_PROC = ROOT_DIR / "data" / "kutai_basin_100m" / "processed"

MIN_LON = 115.0
MAX_LON = 118.0
MIN_LAT = -2.0
MAX_LAT = 1.4
RES_DEG = 0.0009  # ~100.18 meters at equator
W = int(round((MAX_LON - MIN_LON) / RES_DEG))  # 3333
H = int(round((MAX_LAT - MIN_LAT) / RES_DEG))  # 3778

CATEGORICAL_OR_INT_LAYERS = {
    "active_mine_mask",
    "formation_polygon_id",
    "in_coal_bearing_formation",
    "in_holocene_peat_or_karst_mask",
    "in_ophiolite_unit",
    "label",
    "label_coal",
    "label_coal_k0_high_cv",
    "label_coal_k1_med_cv",
    "label_coal_k2_low_cv",
    "label_k0_porphyry_cu",
    "label_k1_skarn_fe_cu",
    "label_k2_epithermal_au",
    "label_k3_laterite_fe_ni",
    "coal_parent_concession_id",
    "explored_footprint",
}


def build_kutai_basin_100m_cube() -> Path:
    start_t = time.time()
    OUT_PROC.mkdir(parents=True, exist_ok=True)
    dst_transform = from_bounds(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, W, H)

    out_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": W,
        "height": H,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": dst_transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "predictor": 2,
    }

    logger.info("================================================================================")
    logger.info("BUILDING KUTAI BASIN 100m FEATURE CUBE & DATASET")
    logger.info("Extent: Lon [%.2f..%.2f], Lat [%.2f..%.2f]", MIN_LON, MAX_LON, MIN_LAT, MAX_LAT)
    logger.info("Dimensions: %d rows x %d cols (%d total cells) @ ~100m resolution", H, W, H * W)
    logger.info("================================================================================")

    # 1. Resample all 75 processed channels from data/kalimantan/processed/
    tif_paths = sorted(KAL_PROC.glob("*.tif"))
    if not tif_paths:
        raise FileNotFoundError(f"No GeoTIFFs found in {KAL_PROC}")

    # Track 2D arrays in memory for dataset dataframe construction
    land_mask: np.ndarray | None = None
    raster_data: dict[str, np.ndarray] = {}

    for i, p in enumerate(tif_paths, 1):
        name = p.stem
        # Skip multi-band prospectivity exports from prior runs
        if name in ("prospectivity_k3_coal", "prospectivity_k4_metals", "fused_raw_score", "conformal_bounds_90ci"):
            continue

        is_cat = name in CATEGORICAL_OR_INT_LAYERS
        resample_method = Resampling.nearest if is_cat else Resampling.bilinear

        with rasterio.open(p) as src:
            win = win_from_bounds(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, src.transform)
            arr = src.read(1, window=win, out_shape=(H, W), resampling=resample_method)

        # Ensure consistent nodata
        arr_clean = np.where(np.isnan(arr) | (arr == -9999.0) | (arr == -9000.0), -9999.0, arr).astype(np.float32)

        out_tif = OUT_PROC / f"{name}.tif"
        p_curr = dict(out_profile)
        if is_cat:
            p_curr["dtype"] = "int32" if "id" in name else "int16"
            p_curr["nodata"] = -1
            p_curr["predictor"] = 1
            arr_to_write = np.where(arr_clean == -9999.0, -1, arr_clean).astype(p_curr["dtype"])
        else:
            arr_to_write = arr_clean

        with rasterio.open(out_tif, "w", **p_curr) as dst:
            dst.write(arr_to_write, 1)

        raster_data[name] = arr_clean
        if name == "dem_elevation":
            land_mask = (arr_clean >= 0.0) & (arr_clean != -9999.0)
            logger.info("[%d/%d] DEM Elevation processed: %d onshore land cells (%.1f%%)",
                        i, len(tif_paths), int(land_mask.sum()), float(land_mask.sum() / (H * W) * 100))

    if land_mask is None:
        raise RuntimeError("dem_elevation.tif was not processed; cannot establish land mask.")

    # 2. Extract dataframe of valid onshore cells
    rows_land, cols_land = np.where(land_mask)
    n_land = len(rows_land)
    logger.info("Extracting onshore tabular dataset for %d valid cells...", n_land)

    lons_1d = np.linspace(MIN_LON, MAX_LON, W, endpoint=False, dtype=np.float32) + (RES_DEG / 2.0)
    lats_1d = np.linspace(MAX_LAT, MIN_LAT, H, endpoint=False, dtype=np.float32) - (RES_DEG / 2.0)

    df_dict: dict[str, np.ndarray] = {
        "row": rows_land.astype(np.int32),
        "col": cols_land.astype(np.int32),
        "lon": lons_1d[cols_land],
        "lat": lats_1d[rows_land],
    }

    # Add all raster channels
    for name, arr in raster_data.items():
        vals = arr[rows_land, cols_land]
        if name in CATEGORICAL_OR_INT_LAYERS:
            vals_clean = np.where(vals == -9999.0, 0, vals)
            df_dict[name] = vals_clean.astype(np.int32)
        else:
            vals_clean = np.where(vals == -9999.0, np.nan, vals)
            df_dict[name] = vals_clean.astype(np.float32)

    # Clean composite coal label
    if "label_coal" not in df_dict:
        k0 = df_dict.get("label_coal_k0_high_cv", np.zeros(n_land, dtype=np.int32))
        k1 = df_dict.get("label_coal_k1_med_cv", np.zeros(n_land, dtype=np.int32))
        k2 = df_dict.get("label_coal_k2_low_cv", np.zeros(n_land, dtype=np.int32))
        df_dict["label_coal"] = ((k0 == 1) | (k1 == 1) | (k2 == 1)).astype(np.int32)

    df = pd.DataFrame(df_dict)
    del df_dict, raster_data
    gc.collect()

    # Verify no raw sensor leakage
    feature_cols = [
        c for c in df.columns
        if c not in (
            "row", "col", "lon", "lat", "label", "label_coal",
            "label_coal_k0_high_cv", "label_coal_k1_med_cv", "label_coal_k2_low_cv",
            "label_k0_porphyry_cu", "label_k1_skarn_fe_cu", "label_k2_epithermal_au", "label_k3_laterite_fe_ni",
            "coal_tonnage_weight", "coal_parent_concession_id", "active_mine_mask", "formation_polygon_id",
            "in_coal_bearing_formation", "in_holocene_peat_or_karst_mask", "in_ophiolite_unit", "explored_footprint",
            "dist_transport_km", "dist_major_rivers_km", "dist_haul_roads_km", "gfw_forest_disturbance",
            "forest_disturbance_rate", "w_canopy", "block_id", "alteration_benchmark",
            "geology_coal_contact_dist_km",
        )
        and not c.startswith("label_")
    ]
    verify_no_raw_pixel_leakage(feature_cols)

    # Cap distance features at local physical limit (15.0 km / 15000.0 m) to prevent GPS grid triangulation
    for c in feature_cols:
        if c.startswith("dist_") or c.endswith("_dist_km"):
            if c.endswith("_m"):
                df[c] = np.clip(df[c].to_numpy(dtype=np.float32), 0.0, 15000.0)
            else:
                df[c] = np.clip(df[c].to_numpy(dtype=np.float32), 0.0, 15.0)

    parquet_p = OUT_PROC / "dataset.parquet"
    logger.info("Saving Kutai Basin 100m dataset (%d rows, %d feature columns) -> %s", len(df), len(feature_cols), parquet_p)
    df.to_parquet(parquet_p, index=False)

    # 3. Create FeatureStack reference
    # Pick 24 essential continuous physical features to build FeatureStack
    core_features = [
        f for f in feature_cols
        if (OUT_PROC / f"{f}.tif").exists()
    ][:24]

    layers_list = []
    for f in core_features:
        with rasterio.open(OUT_PROC / f"{f}.tif") as src:
            layers_list.append(src.read(1))
    stack_arr = np.stack(layers_list, axis=0).astype(np.float32)

    f_stack = FeatureStack(
        array=stack_arr,
        feature_names=core_features,
        profile=out_profile,
    )
    joblib.dump(f_stack, OUT_PROC / "feature_stack.joblib")
    logger.info("Saved FeatureStack (%d channels) -> %s", len(core_features), OUT_PROC / "feature_stack.joblib")

    # Positive summary
    k0_pos = int((df["label_coal_k0_high_cv"] == 1).sum())
    k1_pos = int((df["label_coal_k1_med_cv"] == 1).sum())
    k2_pos = int((df["label_coal_k2_low_cv"] == 1).sum())
    coal_any_pos = int((df["label_coal"] == 1).sum())

    elapsed = time.time() - start_t
    logger.info("================================================================================")
    logger.info("KUTAI BASIN 100m CUBE BUILD COMPLETE in %.1fs", elapsed)
    logger.info("Total Grid Pixels: %d x %d = %d", H, W, H * W)
    logger.info("Onshore Land Pixels: %d (%.1f%%)", n_land, (n_land / (H * W)) * 100)
    logger.info("High-CV Coal Positives (Head 0): %d cells", k0_pos)
    logger.info("Med-CV Coal Positives (Head 1): %d cells", k1_pos)
    logger.info("Low-CV Coal Positives (Head 2): %d cells", k2_pos)
    logger.info("Composite Ground-Truth Coal Cells: %d cells", coal_any_pos)
    logger.info("Deliverables stored at: %s", OUT_PROC)
    logger.info("================================================================================")
    return OUT_PROC


if __name__ == "__main__":
    build_kutai_basin_100m_cube()
