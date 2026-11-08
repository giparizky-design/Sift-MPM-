"""Build 250m Resolution Dataset and Feature Stack for East Kalimantan (Kalimantan Timur).

Interpolates the 21 physical and tectonic channels from 500m to 250m ground resolution
(2400 x 2000 grid across 114.50°E–119.50°E, -2.80°S–3.20°N, EPSG:4326) and rasterizes
all K=4 deposit mechanics ground-truth labels and exploration footprints.
"""

import logging
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom
from pathlib import Path

from src.features.feature_stack import FeatureStack
from src.features.geophysics import load_and_compute_real_gravity_channels
from src.features.structural import plan_curvature
from src.labels.borneo_labels import build_k4_deposit_targets, rasterize_borneo_k4_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_east_kalimantan_250m")

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "data" / "east_kalimantan" / "processed"
DEST_DIR = ROOT_DIR / "data" / "east_kalimantan" / "processed"


def main() -> None:
    logger.info("Loading 250m/500m East Kalimantan feature stack...")
    fs_in = joblib.load(SRC_DIR / "feature_stack.joblib")
    feature_names = list(fs_in.feature_names)
    C, H_in, W_in = fs_in.array.shape
    logger.info("Found %d channels on %dx%d grid", C, H_in, W_in)

    H_250 = 2400
    W_250 = 2000
    new_transform = Affine(0.0025, 0.0, 114.50, 0.0, -0.0025, 3.20)
    profile_250 = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": W_250,
        "height": H_250,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": new_transform,
    }

    if (H_in, W_in) == (H_250, W_250):
        arr_250 = fs_in.array.astype(np.float32, copy=True)
    else:
        logger.info("Interpolating %d physical channels to 250m (%dx%d)...", C, H_250, W_250)
        arr_250 = np.zeros((C, H_250, W_250), dtype=np.float32)
        for ch in range(C):
            arr_250[ch] = zoom(fs_in.array[ch], (H_250 / H_in, W_250 / W_in), order=1)

    valid_mask = np.isfinite(arr_250).all(axis=0) & (arr_250[0] > -9000.0)
    valid_count = int(valid_mask.sum())
    logger.info("Valid land cells at 250m: %d (%.1f%% of grid)", valid_count, (valid_count / (H_250 * W_250)) * 100.0)

    # Recompute Evans (1980) plan curvature and real satellite Bouguer gravity / Parker-Oldenburg Moho
    idx_dem = feature_names.index("dem_elevation")
    dem_clean = np.where(valid_mask, arr_250[idx_dem], 0.0).astype(np.float32)

    logger.info("Computing Evans (1980) plan curvature and topographic lineament distance at 250m...")
    f_curv = plan_curvature(dem_clean, resolution_m=250.0).astype(np.float32)
    curv_smooth = gaussian_filter(np.abs(f_curv), sigma=1.2)
    lin_mask = curv_smooth >= np.percentile(curv_smooth[valid_mask], 86)
    f_dist_topo_lin_km = (distance_transform_edt(~lin_mask) * 0.25).astype(np.float32)

    logger.info("Ingesting real satellite gravity (Sandwell & Smith v33.1) and Parker-Oldenburg Moho inversion...")
    f_bouguer, f_crustal, f_moho_step = load_and_compute_real_gravity_channels(
        dem=dem_clean,
        dst_crs="EPSG:4326",
        dst_transform=new_transform,
        res_m=250.0,
    )

    channel_updates = {
        "dem_plan_curvature": f_curv,
        "dist_topo_lineaments_km": f_dist_topo_lin_km,
        "bouguer_gravity_mgal": f_bouguer,
        "crustal_thickness_km": f_crustal,
        "moho_gravity_step_mgal_km": f_moho_step,
    }
    for ch_name, ch_arr in channel_updates.items():
        if ch_name in feature_names:
            ch_idx = feature_names.index(ch_name)
            arr_ch = np.where(valid_mask, ch_arr, np.nan).astype(np.float32)
            arr_250[ch_idx] = arr_ch
            logger.info(
                " - Updated %s [min=%.4f, max=%.4f]",
                ch_name,
                float(np.nanmin(arr_ch)),
                float(np.nanmax(arr_ch)),
            )

    # 1. Save 250m Feature Stack
    new_stack_profile = dict(profile_250)
    new_stack_profile["count"] = C
    stack_250 = FeatureStack(
        array=arr_250,
        feature_names=feature_names,
        profile=new_stack_profile,
    )
    joblib.dump(stack_250, DEST_DIR / "feature_stack.joblib")
    logger.info("Saved 250m feature_stack.joblib (%dx%dx%d)", C, H_250, W_250)

    # 2. Rasterize K=4 Deposit Mechanics Targets at 250m
    logger.info("Rasterizing K=4 deposit targets onto 250m grid (buffer=1000m / 4px radius)...")
    deposit_targets = build_k4_deposit_targets()
    rasters_k4 = rasterize_borneo_k4_labels(deposit_targets, profile_250, buffer_m=1000.0)

    rows, cols = np.where(valid_mask)
    rows = rows.astype(np.int32)
    cols = cols.astype(np.int32)

    data_dict = {
        "row": rows,
        "col": cols,
    }
    for ch, name in enumerate(feature_names):
        data_dict[name] = arr_250[ch, rows, cols]

    k4_col_names = [
        "label_k0_porphyry_cu",
        "label_k1_skarn_fe_cu",
        "label_k2_epithermal_au",
        "label_k3_laterite_fe_ni",
    ]
    for c in k4_col_names:
        data_dict[c] = rasters_k4[c][rows, cols].astype(np.int32)
        pos = int((data_dict[c] == 1).sum())
        logger.info(" - %s positives: %d", c, pos)

    data_dict["explored_footprint"] = rasters_k4["explored_footprint"][rows, cols].astype(np.int32)
    data_dict["alteration_benchmark"] = rasters_k4["alteration_benchmark"][rows, cols].astype(np.int32)

    pos_any = (
        (data_dict["label_k0_porphyry_cu"] == 1) |
        (data_dict["label_k1_skarn_fe_cu"] == 1) |
        (data_dict["label_k2_epithermal_au"] == 1) |
        (data_dict["label_k3_laterite_fe_ni"] == 1)
    )
    data_dict["label"] = np.where(pos_any, 1, -1).astype(np.int32)
    logger.info("Total positive deposit cells: %d, Unlabeled cells: %d", int(pos_any.sum()), len(rows) - int(pos_any.sum()))

    df_250 = pd.DataFrame(data_dict)
    df_250.to_parquet(DEST_DIR / "dataset.parquet", index=False)
    logger.info("Saved 250m dataset.parquet with %d rows and %d columns", len(df_250), len(df_250.columns))

    # 3. Export all 21 channel GeoTIFFs at 250m
    logger.info("Exporting all 21 channel GeoTIFFs at 250m...")
    for ch, name in enumerate(feature_names):
        tif_p = DEST_DIR / f"{name}.tif"
        with rasterio.open(
            tif_p,
            "w",
            driver="GTiff",
            height=H_250,
            width=W_250,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=new_transform,
            nodata=-9999.0,
            compress="deflate",
        ) as dst:
            dst.write(arr_250[ch], 1)

    logger.info("Successfully generated 250m East Kalimantan dataset!")


if __name__ == "__main__":
    main()
