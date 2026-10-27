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
from scipy.ndimage import zoom
from pathlib import Path

from src.features.feature_stack import FeatureStack
from src.labels.borneo_labels import build_k4_deposit_targets, rasterize_borneo_k4_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_east_kalimantan_250m")

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "data" / "east_kalimantan" / "processed"
DEST_DIR = ROOT_DIR / "data" / "east_kalimantan" / "processed"


def main() -> None:
    logger.info("Loading 500m East Kalimantan feature stack...")
    fs_500 = joblib.load(SRC_DIR / "feature_stack.joblib")
    feature_names = fs_500.feature_names
    C, H_500, W_500 = fs_500.array.shape
    logger.info("Found %d channels on %dx%d 500m grid", C, H_500, W_500)

    H_250 = H_500 * 2  # 2400
    W_250 = W_500 * 2  # 2000
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

    logger.info("Interpolating 21 physical channels to 250m (%dx%d)...", H_250, W_250)
    arr_250 = np.zeros((C, H_250, W_250), dtype=np.float32)
    for ch in range(C):
        arr_250[ch] = zoom(fs_500.array[ch], 2.0, order=1)
        logger.info(" - Channel %2d/%2d: %s [min=%.2f, max=%.2f]", ch + 1, C, feature_names[ch], np.nanmin(arr_250[ch]), np.nanmax(arr_250[ch]))

    valid_mask = np.isfinite(arr_250).all(axis=0) & (arr_250[0] > -9000.0)
    valid_count = int(valid_mask.sum())
    logger.info("Valid land cells at 250m: %d (%.1f%% of grid)", valid_count, (valid_count / (H_250 * W_250)) * 100.0)

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
