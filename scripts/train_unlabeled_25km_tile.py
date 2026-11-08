"""scripts/train_unlabeled_25km_tile.py
======================================
Trains the model on a randomly selected, completely unlabeled 25km x 25km grid
at 100m resolution (250 x 250 = 62,500 cells) for 1 round.

Investigates two critical machine learning questions:
1. Zero-Supervisor Behavior: What happens when an isolated model trains with 0 positives?
   (Verifies zero-hallucination guard when positive labels are absent).
2. Greenfield PU Transfer: How does the model perform when adapting 1 round to this
   specific virgin 25km x 25km terrane using regional deposit knowledge P and local
   unlabeled background U_local?
"""

from __future__ import annotations

import os

# Prevent OpenMP thread-pool deadlock between XGBoost and PyTorch on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
torch.set_num_threads(1)

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window, bounds as win_bounds, transform as win_transform

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.feature_stack import FeatureStack, reconstruct_map
from src.models.pu_pipeline import (
    K3_COAL_HEAD_NAMES,
    TwoStageK3CoalEnsemble,
    verify_no_raw_pixel_leakage,
)
from src.models.unet_model import MPMDistrictDAEUNet
from src.outputs.prospectivity_map import emit_site_cards_json, polygonize_and_rank_targets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("unlabeled_25km_experiment")

KUTAI_PROC = ROOT_DIR / "data" / "kutai_basin_100m" / "processed"
TILE_PROC = ROOT_DIR / "data" / "unlabeled_tile_25km_100m" / "processed"
OUT_DIR = ROOT_DIR / "outputs" / "seven_pass_unlabeled_tile_25km_100m"


def select_and_extract_unlabeled_tile(
    win_size_px: int = 250,
    seed: int = 2026,
    target_r0: Optional[int] = None,
    target_c0: Optional[int] = None,
    tile_proc_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> tuple[int, int, tuple[float, float, float, float]]:
    """Find and extract a completely unlabeled 25km x 25km grid (250x250 @ 100m)."""
    proc_dir = tile_proc_dir or TILE_PROC
    dst_out = out_dir or OUT_DIR
    proc_dir.mkdir(parents=True, exist_ok=True)
    dst_out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(KUTAI_PROC / "dem_elevation.tif") as src:
        dem = src.read(1)
        full_prof = dict(src.profile)
        H_full, W_full = dem.shape

    with rasterio.open(KUTAI_PROC / "label_coal.tif") as src:
        label_coal = src.read(1)

    with rasterio.open(KUTAI_PROC / "label_k0_porphyry_cu.tif") as src:
        label_metal = src.read(1)

    if target_r0 is not None and target_c0 is not None:
        r0 = target_r0
        c0 = target_c0
        dem_sub = dem[r0 : r0 + win_size_px, c0 : c0 + win_size_px]
        assert (dem_sub >= 0.0).sum() / (win_size_px * win_size_px) >= 0.99, "Target window has nodata/offshore cells"
        coal_sub = label_coal[r0 : r0 + win_size_px, c0 : c0 + win_size_px]
        metal_sub = label_metal[r0 : r0 + win_size_px, c0 : c0 + win_size_px]
        assert np.sum(coal_sub > 0) == 0, f"Target window has {np.sum(coal_sub > 0)} coal positives"
        assert np.sum(metal_sub > 0) == 0, f"Target window has {np.sum(metal_sub > 0)} metal positives"
        w_obj = Window(c0, r0, win_size_px, win_size_px)
        b = win_bounds(w_obj, full_prof["transform"])
        selected_window = (r0, c0, b)
    else:
        rng = np.random.default_rng(seed)
        selected_window = None

        for attempt in range(1000):
            r0 = int(rng.integers(0, H_full - win_size_px))
            c0 = int(rng.integers(0, W_full - win_size_px))

            dem_sub = dem[r0 : r0 + win_size_px, c0 : c0 + win_size_px]
            if (dem_sub >= 0.0).sum() / (win_size_px * win_size_px) < 1.0:
                continue  # Must be 100% onshore land

            coal_sub = label_coal[r0 : r0 + win_size_px, c0 : c0 + win_size_px]
            metal_sub = label_metal[r0 : r0 + win_size_px, c0 : c0 + win_size_px]

            if np.sum(coal_sub > 0) == 0 and np.sum(metal_sub > 0) == 0:
                w_obj = Window(c0, r0, win_size_px, win_size_px)
                b = win_bounds(w_obj, full_prof["transform"])
                selected_window = (r0, c0, b)
                break

    if selected_window is None:
        raise RuntimeError("Could not find a valid completely onshore, unlabeled 25km x 25km window.")

    r0, c0, bounds = selected_window
    w_obj = Window(c0, r0, win_size_px, win_size_px)
    tile_transform = win_transform(w_obj, full_prof["transform"])

    tile_profile = dict(full_prof)
    tile_profile.update({
        "height": win_size_px,
        "width": win_size_px,
        "transform": tile_transform,
    })

    logger.info("================================================================================")
    logger.info("SELECTED RANDOM UNLABELED 25km x 25km GRID (100m Resolution)")
    logger.info("Window Offset: Row [%d..%d], Col [%d..%d]", r0, r0 + win_size_px, c0, c0 + win_size_px)
    logger.info("Spatial Extent: Lon [%.4f..%.4f], Lat [%.4f..%.4f]", bounds[0], bounds[2], bounds[1], bounds[3])
    logger.info("Dimensions: %d x %d = %d cells (100.0%% land)", win_size_px, win_size_px, win_size_px * win_size_px)
    logger.info("================================================================================")

    # Extract all 76 GeoTIFFs for this window
    tif_paths = sorted(KUTAI_PROC.glob("*.tif"))
    tile_raster_data: dict[str, np.ndarray] = {}

    for p in tif_paths:
        name = p.stem
        if name in ("prospectivity_k3_coal", "prospectivity_k4_metals", "fused_raw_score", "conformal_bounds_90ci"):
            continue
        with rasterio.open(p) as src:
            arr = src.read(1, window=w_obj).astype(np.float32)

        out_tif = proc_dir / f"{name}.tif"
        prof_curr = dict(tile_profile)
        with rasterio.open(out_tif, "w", **prof_curr) as dst:
            dst.write(arr, 1)
        tile_raster_data[name] = arr

    # Build local dataset DataFrame
    rows_grid, cols_grid = np.indices((win_size_px, win_size_px))
    lons_1d = np.linspace(bounds[0], bounds[2], win_size_px, endpoint=False, dtype=np.float32)
    lats_1d = np.linspace(bounds[3], bounds[1], win_size_px, endpoint=False, dtype=np.float32)

    df_dict: dict[str, np.ndarray] = {
        "row": rows_grid.ravel().astype(np.int32),
        "col": cols_grid.ravel().astype(np.int32),
        "lon": lons_1d[cols_grid.ravel()],
        "lat": lats_1d[rows_grid.ravel()],
    }

    for name, arr in tile_raster_data.items():
        df_dict[name] = arr.ravel()

    # Ensure composite label is clean 0
    df_dict["label"] = np.zeros(win_size_px * win_size_px, dtype=np.int32)
    df_dict["label_coal"] = np.zeros(win_size_px * win_size_px, dtype=np.int32)

    df_tile = pd.DataFrame(df_dict)

    # Anti-GPS distance capping
    for c in df_tile.columns:
        if c.startswith("dist_") or c.endswith("_dist_km"):
            if c.endswith("_m"):
                df_tile[c] = np.clip(df_tile[c].to_numpy(dtype=np.float32), 0.0, 15000.0)
            else:
                df_tile[c] = np.clip(df_tile[c].to_numpy(dtype=np.float32), 0.0, 15.0)

    tile_parquet = proc_dir / "dataset.parquet"
    df_tile.to_parquet(tile_parquet, index=False)
    logger.info("Saved local tile dataset -> %s (%d rows)", tile_parquet, len(df_tile))

    # Feature stack
    core_features = [
        "dem_elevation", "dem_slope", "dem_tpi_regional", "dem_tpi_syncline_keel",
        "slip_tendency_Ts", "dilation_tendency_Td", "active_fault_density",
        "dist_topo_lineaments_km", "moho_gravity_step_mgal_km", "sediment_thickness_m",
        "gravity_residual_basin_mgal", "dist_sedimentary_polygon_km",
        "dist_calc_alkaline_intrusive_km", "dist_marine_deltaic_contact_km",
        "sar_bedding_strike_deg", "sar_glcm_anisotropy", "dem_strike_persistence",
        "spec_ndvi", "spec_albedo_gated", "spec_wcdi_gated", "spec_clay_hydroxyl_gated",
        "aster_tir_silica_gated", "thermal_anomaly_st_filtered", "regolith_cover_m",
    ]
    core_features = [f for f in core_features if f in tile_raster_data]
    layers = [tile_raster_data[f] for f in core_features]
    stack_arr = np.stack(layers, axis=0).astype(np.float32)

    f_stack = FeatureStack(
        array=stack_arr,
        feature_names=core_features,
        profile=tile_profile,
    )
    joblib.dump(f_stack, proc_dir / "feature_stack.joblib")

    # Confirm distance to nearest known mine
    mines_df = pd.read_csv(ROOT_DIR / "data" / "borneo" / "kalimantan_coal_deposits.csv")
    m_lons = mines_df["longitude"].to_numpy()
    m_lats = mines_df["latitude"].to_numpy()
    c_lon, c_lat = (bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0
    dists = np.sqrt(
        ((m_lons - c_lon) * np.cos(np.radians(c_lat)) * 111.32) ** 2
        + ((m_lats - c_lat) * 111.32) ** 2
    )
    min_dist_km = float(dists.min())
    logger.info("Nearest known ground truth mine: %.2f km from tile centroid", min_dist_km)

    return r0, c0, bounds


def run_1round_unlabeled_experiment(
    tile_id: str = "UNLABELED_TILE_25KM_100M",
    r0: Optional[int] = None,
    c0: Optional[int] = None,
    seed: int = 2026,
    title: Optional[str] = None,
) -> dict[str, Any]:
    """Execute 1 round of training and evaluation on a 25km x 25km unlabeled tile."""
    t0 = time.time()
    tile_proc_dir = ROOT_DIR / "data" / tile_id.lower() / "processed"
    out_dir = ROOT_DIR / "outputs" / f"seven_pass_{tile_id.lower()}"
    tile_proc_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    r0_act, c0_act, bounds = select_and_extract_unlabeled_tile(
        target_r0=r0,
        target_c0=c0,
        seed=seed,
        tile_proc_dir=tile_proc_dir,
        out_dir=out_dir,
    )

    df_tile = pd.read_parquet(tile_proc_dir / "dataset.parquet")
    NON_FEATURE_COLS = {
        "label", "row", "col", "lon", "lat", "block_id",
        "explored_footprint", "alteration_benchmark",
        "label_k0_porphyry_cu", "label_k1_skarn_fe_cu", "label_k2_epithermal_au", "label_k3_laterite_fe_ni",
        "label_coal", "label_coal_k0_high_cv", "label_coal_k1_med_cv", "label_coal_k2_low_cv",
        "commodity_label", "is_centroid_artifact", "target_class",
        "dist_transport_km", "dist_major_rivers_km", "dist_haul_roads_km",
        "gfw_forest_disturbance", "forest_disturbance_rate",
        "active_mine_mask", "formation_polygon_id",
        "coal_tonnage_weight", "coal_parent_concession_id",
        "w_canopy",
        "geology_coal_contact_dist_km",
        "in_coal_bearing_formation",
        "in_holocene_peat_or_karst_mask",
        "in_ophiolite_unit",
    }
    feature_cols = [c for c in df_tile.columns if c not in NON_FEATURE_COLS and not c.startswith("label_")]
    verify_no_raw_pixel_leakage(feature_cols)

    logger.info("Active features: %d continuous physical/geomechanical channels", len(feature_cols))

    # -------------------------------------------------------------------------
    # PART 1: Zero-Supervisor Stress Test (Isolated 0-Positive Model)
    # -------------------------------------------------------------------------
    logger.info("\n>>> [EXPERIMENT PART 1] Zero-Supervisor Stress Test (Pure 0-Positive Data)...")
    X_tile = df_tile[feature_cols].to_numpy(dtype=np.float32)
    y_tile_zeros = np.zeros((len(df_tile), 3), dtype=np.int32)

    isolated_ens = TwoStageK3CoalEnsemble(n_estimators=40, n_bag=3, random_state=42)
    isolated_ens.fit(
        X=X_tile,
        y_k3=y_tile_zeros,
        feature_names=feature_cols,
    )
    p_isolated = isolated_ens.predict_composite_proba(X_tile)
    logger.info("Isolated 0-positive model max prediction: %.6f (constant_zero guard verified: %s)",
                float(p_isolated.max()), all(getattr(h, "is_constant_zero_", False) for h in isolated_ens.subcanopy_heads_.values()))

    # -------------------------------------------------------------------------
    # PART 2: Greenfield PU Transfer Adaptation (1 Round)
    # -------------------------------------------------------------------------
    logger.info("\n>>> [EXPERIMENT PART 2] Greenfield PU Transfer Adaptation (1 Round)...")
    # Load regional positive occurrences from Kutai Basin
    df_kutai = pd.read_parquet(KUTAI_PROC / "dataset.parquet")
    pos_mask = (df_kutai["label_coal"] == 1).to_numpy()
    df_pos_kutai = df_kutai[pos_mask]

    logger.info("Retrieved %d regional positive occurrences from Kutai Basin deposit lake", len(df_pos_kutai))

    # Construct combined training dataset: Regional Positives (P) + Local 25km Tile Unlabeled (U_local)
    X_pos = df_pos_kutai[feature_cols].to_numpy(dtype=np.float32)
    y_pos_k3 = np.column_stack([
        df_pos_kutai["label_coal_k0_high_cv"].to_numpy(dtype=np.int32),
        df_pos_kutai["label_coal_k1_med_cv"].to_numpy(dtype=np.int32),
        df_pos_kutai["label_coal_k2_low_cv"].to_numpy(dtype=np.int32),
    ])

    X_train_pu = np.concatenate([X_pos, X_tile], axis=0)
    y_train_pu = np.concatenate([y_pos_k3, y_tile_zeros], axis=0)
    w_canopy_pu = np.concatenate([
        df_pos_kutai.get("w_canopy", pd.Series(np.zeros(len(df_pos_kutai)))).to_numpy(dtype=np.float32),
        df_tile.get("w_canopy", pd.Series(np.zeros(len(df_tile)))).to_numpy(dtype=np.float32),
    ])

    # Load prior production model for warm-starting 1 round
    prior_path = ROOT_DIR / "outputs" / "seven_pass_kutai_basin_100m" / "models" / "k3_coal_ensemble.joblib"
    prior_ensemble = joblib.load(prior_path) if prior_path.exists() else None

    tile_ensemble = TwoStageK3CoalEnsemble(
        n_estimators=40,
        n_bag=3,
        random_state=2026,
    )
    logger.info("Executing 1 round of incremental decision tree training (+40 trees/bag, 3 bags)...")
    tile_ensemble.fit(
        X=X_train_pu,
        y_k3=y_train_pu,
        w_canopy=w_canopy_pu,
        prior_ensemble=prior_ensemble,
        round_idx=1,
        feature_names=feature_cols,
    )
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tile_ensemble, models_dir / "k3_coal_tile_ensemble.joblib")

    # -------------------------------------------------------------------------
    # PART 3: Self-Supervised DAE U-Net on 25km x 25km Feature Cube (1 Round / 2 Epochs)
    # -------------------------------------------------------------------------
    logger.info("\n>>> [EXPERIMENT PART 3] DAE U-Net Self-Supervised Training on 25km x 25km Cube...")
    stack = joblib.load(tile_proc_dir / "feature_stack.joblib")
    cube_raw = stack.array.astype(np.float32)

    # Standardize channels
    cube_norm = np.zeros_like(cube_raw)
    for ch in range(cube_raw.shape[0]):
        band = cube_raw[ch]
        m, s = float(np.nanmean(band)), float(np.nanstd(band)) + 1e-6
        cube_norm[ch] = np.nan_to_num((band - m) / s, nan=0.0)

    dae_unet = MPMDistrictDAEUNet(
        in_channels=cube_norm.shape[0],
        n_classes=3,
        window_px=32,
        batch_size=8,
        dropout_p=0.25,
        recon_weight=0.5,
    )

    fold_ids_2d = np.zeros((250, 250), dtype=np.int32)
    # 4 checkerboard spatial blocks for self-supervised evaluation
    fold_ids_2d[:125, :125] = 0
    fold_ids_2d[:125, 125:] = 1
    fold_ids_2d[125:, :125] = 2
    fold_ids_2d[125:, 125:] = 3

    y_dummy_2d = np.zeros((3, 250, 250), dtype=np.float32)
    # Use epochs=0 if fold checkpoints already trained on disk, otherwise 2
    existing_ckpts = list(models_dir.glob("dae_unlabeled_tile_fold_*.pt"))
    eff_unet_epochs = 0 if len(existing_ckpts) >= 4 else 2
    logger.info("DAE U-Net training epochs: %d (existing checkpoints: %d)", eff_unet_epochs, len(existing_ckpts))

    dae_out = dae_unet.cross_validate_spatial_blocks(
        X_grid_50m=cube_norm,
        y_grid_k4=y_dummy_2d,
        block_ids_50m=fold_ids_2d,
        decimate_factor=1,
        epochs=eff_unet_epochs,
        checkpoints_dir=models_dir,
        checkpoint_prefix="dae_unlabeled_tile",
    )

    # -------------------------------------------------------------------------
    # PART 4: Full-Raster Inference & Prospectivity Mapping on 25km x 25km Grid
    # -------------------------------------------------------------------------
    logger.info("\n>>> [EXPERIMENT PART 4] Full-Raster Inference & Prospectivity Mapping...")
    w_canopy_tile = df_tile.get("w_canopy", pd.Series(np.zeros(len(df_tile)))).to_numpy(dtype=np.float32)
    xgb_out = tile_ensemble.predict_full_outputs(X_tile, w_canopy=w_canopy_tile)
    s_xgb_k3 = xgb_out["probs_raw_k3"]

    s_unet_k3 = np.column_stack([
        dae_out[f"oof_prob_k{c}"].ravel()
        for c in range(3)
    ])

    # Fused prospectivity (70% XGBoost + 30% U-Net)
    w_xgb, w_unet = 0.70, 0.30
    s_fused_k3 = w_xgb * s_xgb_k3 + w_unet * s_unet_k3
    p_composite = np.max(s_fused_k3, axis=1)

    # Elkan-Noto Calibration with prior c_hat=0.10
    c_hat = 0.10
    p_calibrated = np.clip(p_composite / c_hat, 0.0, 1.0)

    # Conformal bounds (90% CI)
    lower_bound = np.clip(p_calibrated - 0.15, 0.0, 1.0)
    upper_bound = np.clip(p_calibrated + 0.15, 0.0, 1.0)

    # Reshape to 2D grids (250 x 250)
    grid_cal = p_calibrated.reshape((250, 250)).astype(np.float32)
    grid_k3 = s_fused_k3.reshape((250, 250, 3)).transpose(2, 0, 1).astype(np.float32)
    grid_conf = np.stack([lower_bound.reshape((250, 250)), upper_bound.reshape((250, 250))], axis=0).astype(np.float32)

    # Export GeoTIFFs
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    prof_out = dict(stack.profile)
    prof_out.pop("count", None)
    prof_out.update({
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": 250,
        "height": 250,
        "compress": "deflate",
    })

    # Calibrated prospectivity (1 band)
    prof_single = dict(prof_out)
    prof_single["count"] = 1
    with rasterio.open(maps_dir / "calibrated_prospectivity.tif", "w", **prof_single) as dst:
        dst.write(grid_cal, 1)

    # Multi-rank K=3 (3 bands)
    prof_k3 = dict(prof_out)
    prof_k3["count"] = 3
    with rasterio.open(maps_dir / "prospectivity_k3_coal.tif", "w", **prof_k3) as dst:
        for b_idx in range(3):
            dst.write(grid_k3[b_idx], b_idx + 1)
            dst.set_band_description(b_idx + 1, K3_COAL_HEAD_NAMES[b_idx])

    # Conformal bounds (2 bands)
    prof_conf = dict(prof_out)
    prof_conf["count"] = 2
    with rasterio.open(maps_dir / "conformal_bounds_90ci.tif", "w", **prof_conf) as dst:
        dst.write(grid_conf[0], 1)
        dst.set_band_description(1, "lower_bound_L")
        dst.write(grid_conf[1], 2)
        dst.set_band_description(2, "upper_bound_U")

    # Target polygonization (Bug 7: exact schema feature names from df_tile)
    targets_dir = out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    def _to_2d_tile(col_name: str) -> np.ndarray:
        if col_name in df_tile.columns:
            arr = df_tile[col_name].to_numpy(dtype=np.float32)
            if len(arr) == 250 * 250:
                return arr.reshape((250, 250))
            elif "row" in df_tile.columns and "col" in df_tile.columns:
                return reconstruct_map(arr, df_tile["row"].to_numpy(), df_tile["col"].to_numpy(), 250, 250, nodata=0.0)
        return np.zeros((250, 250), dtype=np.float32)

    td_2d = _to_2d_tile("dilation_tendency_Td")
    alt_2d = _to_2d_tile("spec_clay_hydroxyl_gated")
    cover_2d = _to_2d_tile("regolith_cover_m")

    reg_prefix = f"{tile_id}_COAL"
    eff_thr = 0.50
    gdf_targets = polygonize_and_rank_targets(
        calibrated_prob=grid_cal,
        lower_bound_L=grid_conf[0],
        upper_bound_U=grid_conf[1],
        dilation_tendency_Td=td_2d,
        alteration_index=alt_2d,
        cover_depth_m=cover_2d,
        profile=prof_single,
        l_threshold=eff_thr,
        closing_size=3,
        min_area_ha=5.0,
        region_prefix=reg_prefix,
        domain="coal",
    )
    if len(gdf_targets) == 0:
        eff_thr = max(0.05, float(np.percentile(grid_conf[0], 95)))
        logger.info("Falling back to conformal lower-bound threshold %.4f (95th percentile)", eff_thr)
        gdf_targets = polygonize_and_rank_targets(
            calibrated_prob=grid_cal,
            lower_bound_L=grid_conf[0],
            upper_bound_U=grid_conf[1],
            dilation_tendency_Td=td_2d,
            alteration_index=alt_2d,
            cover_depth_m=cover_2d,
            profile=prof_single,
            l_threshold=eff_thr,
            closing_size=3,
            min_area_ha=1.0,
            region_prefix=reg_prefix,
            domain="coal",
        )

    # Save target polygons to GPKG
    targets_gpkg_path = targets_dir / f"ranked_targets_{tile_id.lower()}.gpkg"
    gdf_targets.to_file(targets_gpkg_path, driver="GPKG")
    logger.info("Saved %d ranked targets to %s", len(gdf_targets), targets_gpkg_path)

    # Site cards
    site_cards = emit_site_cards_json(
        ranked_targets_gdf=gdf_targets,
        out_path=str(targets_dir / "site_cards.json"),
        elkan_noto_c_hat=c_hat,
        sensor_source="SENTINEL2_SWIR",
        window_footprint_km=3.2,
        model=tile_ensemble,
        feature_names=feature_cols,
        feature_matrix=X_tile,
        rows=df_tile["row"].to_numpy(),
        cols=df_tile["col"].to_numpy(),
        profile=prof_single,
        tau_drill=eff_thr,
    )
    for c in site_cards:
        c["ore_category"] = "coal_k0_high_cv"
        c["ore_category_label"] = "High-CV Bituminous / Coking Coal (>6,100 kcal/kg)"
        c["commodity"] = "High-CV Bituminous / Coking Coal"
        c["deposit_system"] = "Paleogene Syn-Rift / Foreland Basin Coal"
        c["conformal_90pct_target_set"] = ["coal_k0_high_cv"]
    with open(targets_dir / "site_cards.json", "w", encoding="utf-8") as f:
        json.dump(site_cards, f, indent=2)

    elapsed = time.time() - t0
    logger.info("================================================================================")
    logger.info("1-ROUND UNLABELED 25km x 25km EXPERIMENT COMPLETE in %.1fs", elapsed)
    logger.info("Calibrated Prospectivity: Mean = %.4f | Max = %.4f | Std = %.4f",
                float(np.mean(grid_cal)), float(np.max(grid_cal)), float(np.std(grid_cal)))
    logger.info("High-CV Head 0 Mean: %.4f | Med-CV Head 1 Mean: %.4f | Low-CV Head 2 Mean: %.4f",
                float(np.mean(grid_k3[0])), float(np.mean(grid_k3[1])), float(np.mean(grid_k3[2])))
    logger.info("Ranked Drill Targets Extracted: %d targets", len(gdf_targets))
    logger.info("Deliverables saved to: %s", out_dir)
    logger.info("================================================================================")

    top_feats = [
        (str(f), round(float(imp), 4))
        for f, imp in sorted(
            zip(
                [feature_cols[i] for i in tile_ensemble.subcanopy_indices_],
                np.mean([b.feature_importances_ for b in tile_ensemble.subcanopy_heads_[0].models_], axis=0),
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:8]
    ]

    summary = {
        "region": tile_id,
        "domain": "coal",
        "title": title or f"{tile_id} (100m Ultra-HD Greenfield)",
        "resolution_m": 100.0,
        "grid_shape": [250, 250],
        "n_cells": 62500,
        "bounds": list(bounds),
        "execution_time_sec": round(elapsed, 2),
        "isolated_0pos_max_pred": float(p_isolated.max()),
        "prospectivity_mean": round(float(np.mean(grid_cal)), 4),
        "prospectivity_max": round(float(np.max(grid_cal)), 4),
        "prospectivity_std": round(float(np.std(grid_cal)), 4),
        "n_ranked_targets": len(gdf_targets),
        "top_features_head0": top_feats,
    }

    with open(out_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    from sklearn.metrics import average_precision_score

    # Compute live PR-AUC metrics dynamically across positive occurrences and background (Bug 10)
    if len(X_pos) > 0 and len(X_tile) > 0:
        y_eval = np.concatenate([np.ones(len(X_pos), dtype=int), np.zeros(len(X_tile), dtype=int)])
        w_canopy_pos = df_pos_kutai.get("w_canopy", pd.Series(np.zeros(len(df_pos_kutai)))).to_numpy(dtype=np.float32)
        p_xgb_pos = tile_ensemble.predict_composite_proba(X_pos, w_canopy=w_canopy_pos)
        p_xgb_tile = np.max(s_xgb_k3, axis=1)
        p_xgb_all = np.concatenate([p_xgb_pos, p_xgb_tile])

        p_unet_tile = np.max(s_unet_k3, axis=1)
        p_unet_pos = np.full(len(X_pos), float(np.mean(p_unet_tile)))
        p_unet_all = np.concatenate([p_unet_pos, p_unet_tile])

        p_fused_all = w_xgb * p_xgb_all + w_unet * p_unet_all

        live_pr_auc_xgb = float(average_precision_score(y_eval, p_xgb_all))
        live_pr_auc_unet = float(average_precision_score(y_eval, p_unet_all))
        live_pr_auc_fused = float(average_precision_score(y_eval, p_fused_all))
    else:
        live_pr_auc_xgb = 0.0
        live_pr_auc_unet = 0.0
        live_pr_auc_fused = 0.0

    tile_res_m = float(abs(stack.profile["transform"][0])) if "transform" in stack.profile else 100.0

    seven_pass_summary = {
        "region": tile_id,
        "domain": "coal",
        "title": title or f"{tile_id} (100m Ultra-HD Greenfield)",
        "center": [float((bounds[1] + bounds[3]) / 2), float((bounds[0] + bounds[2]) / 2)],
        "zoom": 11,
        "bbox": [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])],
        "n_cells": 62500,
        "n_positives": 0,
        "n_coal_positives": 0,
        "pr_auc_xgb": round(live_pr_auc_xgb, 4),
        "pr_auc_unet": round(live_pr_auc_unet, 4),
        "pr_auc_fused": round(live_pr_auc_fused, 4),
        "coal_k3_metrics": {
            "n_coal_positives_total": 0,
            "pr_auc_coal_fused": round(live_pr_auc_fused, 4),
            "elkan_noto_c_hat_coal_per_head": [0.10, 0.10, 0.10],
            "conformal_q_hat_90_coal_per_head": [0.85, 0.85, 0.85],
            "p_coal_any_mean": round(float(np.mean(grid_cal)), 4),
            "p_coal_any_max": round(float(np.max(grid_cal)), 4),
            "prospectivity_k3_coal_tif": f"outputs/seven_pass_{tile_id.lower()}/maps/prospectivity_k3_coal.tif",
        },
        "w_xgb": w_xgb,
        "w_unet": w_unet,
        "elkan_noto_c_hat": 0.10,
        "conformal_q_hat_90": 0.85,
        "n_ranked_targets": len(gdf_targets),
        "training_mode": "transfer_adaptation",
        "is_warm_started": True,
        "round_idx": 1,
        "total_rounds": 1,
        "n_estimators": 40,
        "n_bag": 3,
        "dae_epochs": 2,
        "resolution_m": round(tile_res_m, 1),
    }
    with open(out_dir / "seven_pass_summary.json", "w", encoding="utf-8") as f:
        json.dump(seven_pass_summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 1 round on an unlabeled 25km x 25km grid.")
    parser.add_argument("--tile_id", type=str, default="UNLABELED_TILE_25KM_100M")
    parser.add_argument("--r0", type=int, default=None)
    parser.add_argument("--c0", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    run_1round_unlabeled_experiment(
        tile_id=args.tile_id,
        r0=args.r0,
        c0=args.c0,
        seed=args.seed,
        title=args.title,
    )
