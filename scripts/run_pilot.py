#!/usr/bin/env python3
"""End-to-end pipeline orchestration script for the Mineral Prospectivity Mapping system.

Usage:
    python scripts/run_pilot.py --config config/pilot.yaml --stage all --version-tag v1

Pipeline execution sequence:
    harmonize → features → labels → train → evaluate → outputs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio

from src.features.aster_indices import (
    aloh_content,
    carbonate_chlorite,
    ferric_oxide_content,
    silica_index,
)
from src.features.feature_stack import FeatureStack, flatten_to_dataframe, reconstruct_map
from src.features.geophysics import (
    analytic_signal,
    first_vertical_derivative,
    horizontal_gradient_magnitude,
    tilt_derivative,
)
from src.features.sentinel2_indices import clay_index, iron_oxide_s2, ndvi
from src.features.structural import (
    aspect,
    euclidean_distance_to_features,
    slope,
    topographic_position_index,
)
from src.harmonization.cover_mask import build_vegetation_mask
from src.harmonization.reproject import (
    build_pilot_grid,
    reproject_raster,
    reproject_vector_to_raster,
)
from src.labels.deposit_labels import (
    classify_drillhole_cells,
    load_minedex,
    load_wamex_drillholes,
    rasterise_labels,
)
from src.labels.spatial_splits import (
    assign_spatial_blocks,
    blind_holdout_masks,
    enforce_cluster_integrity,
    spatial_block_kfold,
)
from src.models.pu_pipeline import fit_pu_model_from_frame, score_metrics, spatial_cv_pu
from src.models.xgboost_model import MPMXGBoost
from src.outputs.prospectivity_map import (
    generate_target_polygons,
    save_prospectivity_geotiff,
    save_shap_geotiff,
    save_uncertainty_geotiff,
)
from src.utils.io import load_raster, save_raster

PIPELINE_STAGES: list[str] = [
    "harmonize",
    "features",
    "labels",
    "train",
    "evaluate",
    "outputs",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Harmonize
# ---------------------------------------------------------------------------
def stage_harmonize(config: dict) -> None:
    logger.info("Starting Stage 1: Harmonization...")
    grid_cfg = config["grid"]
    pilot_profile = build_pilot_grid(grid_cfg)

    raw_dir = Path("data/raw")
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Continuous raster layers to reproject to 50 m grid
    rasters_to_harmonize = {
        "dem": raw_dir / "dem" / "dem.tif",
        "mag_tmi": raw_dir / "magnetics" / "tmi.tif",
        "mag_rtp": raw_dir / "magnetics" / "rtp.tif",
        "rad_k": raw_dir / "radiometrics" / "potassium.tif",
        "rad_u": raw_dir / "radiometrics" / "uranium.tif",
        "rad_th": raw_dir / "radiometrics" / "thorium.tif",
        "grav_bouguer": raw_dir / "gravity" / "bouguer.tif",
        "s2_b4": raw_dir / "sentinel2" / "s2_b4.tif",
        "s2_b8": raw_dir / "sentinel2" / "s2_b8.tif",
        "s2_b11": raw_dir / "sentinel2" / "s2_b11.tif",
        "s2_b12": raw_dir / "sentinel2" / "s2_b12.tif",
        "ast_b1": raw_dir / "aster" / "aster_b1.tif",
        "ast_b2": raw_dir / "aster" / "aster_b2.tif",
        "ast_b5": raw_dir / "aster" / "aster_b5.tif",
        "ast_b6": raw_dir / "aster" / "aster_b6.tif",
        "ast_b7": raw_dir / "aster" / "aster_b7.tif",
        "ast_b8": raw_dir / "aster" / "aster_b8.tif",
        "ast_b9": raw_dir / "aster" / "aster_b9.tif",
        "ast_b11": raw_dir / "aster" / "aster_b11.tif",
        "ast_b13": raw_dir / "aster" / "aster_b13.tif",
    }

    for key, src_path in rasters_to_harmonize.items():
        if src_path.exists():
            dst_path = interim_dir / f"{key}_50m.tif"
            reproject_raster(str(src_path), str(dst_path), pilot_profile, resampling_method="bilinear")

    # 2. Vector rasterization (Geology prospective units)
    geology_path = raw_dir / "geology" / "geology.gpkg"
    if geology_path.exists():
        geo_gdf = gpd.read_file(geology_path)
        geo_arr = reproject_vector_to_raster(
            geo_gdf, attribute="is_prospective", target_profile=pilot_profile, dtype="float32"
        )
        save_raster(geo_arr, pilot_profile, str(interim_dir / "geology_prospective_50m.tif"))

    logger.info("Stage 1 (Harmonization) complete. All layers resampled to 50 m grid.")


def _read_2d(path: str | Path) -> np.ndarray:
    arr, _ = load_raster(str(path))
    return np.squeeze(arr)


# ---------------------------------------------------------------------------
# 2. Features
# ---------------------------------------------------------------------------
def stage_features(config: dict) -> None:
    logger.info("Starting Stage 2: Feature Engineering...")
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    pilot_profile = build_pilot_grid(config["grid"])
    dx = float(config["grid"].get("cell_size_m", 50.0))
    dy = dx

    # 1. Spectral alteration indices
    b4 = _read_2d(interim_dir / "s2_b4_50m.tif")
    b8 = _read_2d(interim_dir / "s2_b8_50m.tif")
    s2_ndvi = ndvi(b4, b8)
    veg_mask = build_vegetation_mask(s2_ndvi, threshold=0.20)

    b5 = _read_2d(interim_dir / "ast_b5_50m.tif")
    b6 = _read_2d(interim_dir / "ast_b6_50m.tif")
    b7 = _read_2d(interim_dir / "ast_b7_50m.tif")
    b8_ast = _read_2d(interim_dir / "ast_b8_50m.tif")
    b9 = _read_2d(interim_dir / "ast_b9_50m.tif")
    b1 = _read_2d(interim_dir / "ast_b1_50m.tif")
    b2 = _read_2d(interim_dir / "ast_b2_50m.tif")
    b11 = _read_2d(interim_dir / "ast_b11_50m.tif")
    b13 = _read_2d(interim_dir / "ast_b13_50m.tif")

    f_aloh = aloh_content(b5, b6, b7)
    f_carb_chlor = carbonate_chlorite(b6, b7, b8_ast, b9)
    f_fe_oxide = ferric_oxide_content(b1, b2)
    f_silica = silica_index(b11, b13)

    b11_s2 = _read_2d(interim_dir / "s2_b11_50m.tif")
    b12_s2 = _read_2d(interim_dir / "s2_b12_50m.tif")
    f_clay_s2 = clay_index(b11_s2, b12_s2)

    # Mask all optical/SWIR alteration indices over dense vegetation (XGBoost handles NaN natively)
    for opt_arr in (f_aloh, f_carb_chlor, f_fe_oxide, f_silica, f_clay_s2):
        opt_arr[veg_mask] = np.nan

    # 2. Geophysical derivatives
    rtp = _read_2d(interim_dir / "mag_rtp_50m.tif")
    f_mag_tdr = tilt_derivative(rtp, dx, dy)
    f_mag_as = analytic_signal(rtp, dx, dy)
    f_mag_1vd = first_vertical_derivative(rtp, dx, dy)

    grav = _read_2d(interim_dir / "grav_bouguer_50m.tif")
    f_grav_hgm = horizontal_gradient_magnitude(grav, dx, dy)

    rad_k = _read_2d(interim_dir / "rad_k_50m.tif")
    rad_u = _read_2d(interim_dir / "rad_u_50m.tif")
    rad_th = _read_2d(interim_dir / "rad_th_50m.tif")
    f_th_u = rad_th / np.maximum(rad_u, 1e-4)

    # 3. Structural & Topographic features
    real_dem_path = (
        Path("data/raw_real/dem/dem.tif")
        if Path("data/raw_real/dem/dem.tif").exists()
        else Path("data/raw_real/dem.tif")
    )
    if real_dem_path.exists():
        logger.info("Using authentic Copernicus DEM from %s", real_dem_path)
        dem = _read_2d(real_dem_path)
    else:
        dem = _read_2d(interim_dir / "dem_50m.tif")

    f_slope = slope(dem, dx)
    f_aspect = aspect(dem, dx)
    f_tpi = topographic_position_index(dem, radius_px=10)

    # Faults from real GSWA linear structures if available
    faults_path = (
        Path("data/raw_real/geology/faults.gpkg")
        if Path("data/raw_real/geology/faults.gpkg").exists()
        else Path("data/raw_real/faults.gpkg")
    )
    f_dist_faults = None
    if faults_path.exists():
        logger.info("Computing Euclidean distance to authentic GSWA faults (%s)...", faults_path)
        faults_gdf = gpd.read_file(faults_path)
        fault_shapes = [(geom, 1) for geom in faults_gdf.geometry if geom is not None and not geom.is_empty]
        if fault_shapes:
            fault_mask = rasterio.features.rasterize(
                fault_shapes,
                out_shape=(pilot_profile["height"], pilot_profile["width"]),
                transform=pilot_profile["transform"],
                fill=0,
                dtype=np.uint8,
            )
            f_dist_faults = euclidean_distance_to_features(fault_mask, resolution_m=dx)

    # Structural proxy: proximity to prospective geology contacts
    geo = _read_2d(interim_dir / "geology_prospective_50m.tif")
    geo_edges = (geo == 1).astype(np.uint8)
    f_dist_contact = euclidean_distance_to_features(geo_edges, resolution_m=dx)

    # Save feature rasters and assemble stack
    feature_dict = {
        "aster_aloh": f_aloh,
        "aster_carb_chlor": f_carb_chlor,
        "aster_fe_oxide": f_fe_oxide,
        "aster_silica": f_silica,
        "s2_clay": f_clay_s2,
        "mag_tdr": f_mag_tdr,
        "mag_as": f_mag_as,
        "mag_1vd": f_mag_1vd,
        "grav_hgm": f_grav_hgm,
        "rad_k": rad_k,
        "rad_th_u": f_th_u,
        "dem_slope": f_slope,
        "dem_aspect": f_aspect,
        "dem_tpi": f_tpi,
        "dist_contact_m": f_dist_contact,
    }
    if f_dist_faults is not None:
        feature_dict["dist_faults_m"] = f_dist_faults

    feature_arrays = []
    feature_names = list(feature_dict.keys())
    for fname, arr in feature_dict.items():
        save_raster(arr, pilot_profile, str(processed_dir / f"{fname}.tif"))
        feature_arrays.append(arr)

    stacked_array = np.stack(feature_arrays, axis=0).astype(np.float32)
    # Only mask pixels where non-optical (geophysical/structural) channels are NaN
    non_opt_indices = [
        i for i, n in enumerate(feature_names)
        if not n.startswith(("aster_", "s2_"))
    ]
    quality_mask = np.isnan(stacked_array[non_opt_indices]).any(axis=0) if non_opt_indices else np.isnan(stacked_array).all(axis=0)

    stack = FeatureStack(
        array=stacked_array,
        feature_names=feature_names,
        profile=pilot_profile,
        data_quality_mask=quality_mask,
    )
    joblib.dump(stack, processed_dir / "feature_stack.joblib")
    logger.info("Stage 2 (Features) complete. Assembled %d feature layers.", len(feature_names))


# ---------------------------------------------------------------------------
# 3. Labels
# ---------------------------------------------------------------------------
def stage_labels(config: dict) -> None:
    logger.info("Starting Stage 3: Label Preparation...")
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    pilot_profile = build_pilot_grid(config["grid"])

    # Load Positives (MINEDEX) - prioritize authentic real data if available
    real_minedex = (
        Path("data/raw_real/labels/minedex/deposits.gpkg")
        if Path("data/raw_real/labels/minedex/deposits.gpkg").exists()
        else Path("data/raw_real/minedex_deposits.gpkg")
    )
    if real_minedex.exists():
        logger.info("Loading authentic MINEDEX deposits from %s", real_minedex)
        positives_gdf = load_minedex(str(real_minedex), min_au_koz=100.0)
    else:
        minedex_path = str(raw_dir / "labels" / "minedex" / "deposits.gpkg")
        positives_gdf = load_minedex(minedex_path, min_au_koz=100.0)

    # Load Drillholes & Assays (WAMEX)
    real_collars = (
        Path("data/raw_real/labels/wamex/collars.gpkg")
        if Path("data/raw_real/labels/wamex/collars.gpkg").exists()
        else Path("data/raw_real/wamex_drillholes.gpkg")
    )
    collars_path = str(real_collars if real_collars.exists() else raw_dir / "labels" / "wamex" / "collars.csv")
    assays_path = raw_dir / "labels" / "wamex" / "assays.csv"
    drillholes_gdf = load_wamex_drillholes(collars_path)

    if assays_path.exists():
        try:
            rn_gdf = classify_drillhole_cells(
                drillholes=drillholes_gdf,
                assay_path=str(assays_path),
                min_depth_m=50.0,
                barren_threshold_gt=0.05,
            )
        except Exception as exc:
            logger.warning("classify_drillhole_cells failed (%s); falling back to deep collars.", exc)
            rn_gdf = gpd.GeoDataFrame()
    else:
        rn_gdf = gpd.GeoDataFrame()

    # If no holes matched the assay table, qualify deep exploration holes outside deposit buffers
    if len(rn_gdf) == 0 and "maxdepth" in drillholes_gdf.columns:
        logger.info("Qualifying deep exploration drillholes (maxdepth >= 50m) as RN candidates...")
        rn_gdf = drillholes_gdf[drillholes_gdf["maxdepth"] >= 50.0].copy()
        rn_gdf["classification"] = "RN"

    # Rasterise labels to pilot grid (P=1, RN=0, U=-1)
    label_array = rasterise_labels(
        positives=positives_gdf,
        reliable_negatives=rn_gdf,
        target_profile=pilot_profile,
        buffer_p_m=100.0,
        exclusion_buffer_m=250.0,
    )

    np.save(processed_dir / "labels.npy", label_array)

    # Also build flattened training table
    stack: FeatureStack = joblib.load(processed_dir / "feature_stack.joblib")
    df = flatten_to_dataframe(stack, label_array=label_array)
    df.to_parquet(processed_dir / "dataset.parquet", index=False)

    p_count = (label_array == 1).sum()
    rn_count = (label_array == 0).sum()
    u_count = (label_array == -1).sum()
    logger.info("Stage 3 (Labels) complete: P=%d, RN=%d, U=%d cells.", p_count, rn_count, u_count)


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
def _holdout_masks(df: pd.DataFrame, profile: dict, config: dict):
    """Return (holdout_mask, train_mask) for the configured blind district.

    If no holdout is configured, nothing is held out and a warning is emitted,
    because without one there is no unbiased estimate of transfer to new geology.
    """
    cfg = config.get("evaluation", {}).get("blind_holdout")
    if not cfg:
        logger.warning("No evaluation.blind_holdout configured: NO blind test set exists.")
        return np.zeros(len(df), dtype=bool), np.ones(len(df), dtype=bool)

    from shapely.geometry import box as shp_box

    xmin, ymin, xmax, ymax = cfg["bounds_proj"]
    poly = gpd.GeoDataFrame(
        {"name": [cfg.get("name", "holdout")]},
        geometry=[shp_box(xmin, ymin, xmax, ymax)],
        crs=profile["crs"],
    )
    return blind_holdout_masks(
        df["row"].to_numpy(), df["col"].to_numpy(), profile, poly,
        buffer_m=float(cfg.get("buffer_m", 0.0)),
    )


def stage_train(config: dict, version_tag: str) -> None:
    logger.info("Starting Stage 4: Model Training (version: %s)...", version_tag)
    processed_dir = Path("data/processed")
    models_dir = Path("outputs/models") / version_tag
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_dir / "dataset.parquet")
    feature_cols = [c for c in df.columns if c not in ("label", "row", "col")]
    stack: FeatureStack = joblib.load(processed_dir / "feature_stack.joblib")

    # The blind district and its buffer ring never reach training.
    _, train_mask = _holdout_masks(df, stack.profile, config)
    pool = df[train_mask].reset_index(drop=True)
    logger.info("Training pool: %d of %d cells (blind district + buffer excluded)", len(pool), len(df))

    model = fit_pu_model_from_frame(pool, feature_cols, config)
    model.save(str(models_dir / "xgboost_ensemble.joblib"))
    logger.info("Stage 4 (Train) complete. Model ensemble saved to %s", models_dir)


# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
def stage_evaluate(config: dict, version_tag: str) -> None:
    logger.info("Starting Stage 5: Leakage-Resistant Spatial Evaluation...")
    processed_dir = Path("data/processed")
    models_dir = Path("outputs/models") / version_tag
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_dir / "dataset.parquet")
    feature_cols = [c for c in df.columns if c not in ("label", "row", "col")]
    stack: FeatureStack = joblib.load(processed_dir / "feature_stack.joblib")
    H, W = stack.array.shape[1], stack.array.shape[2]
    cell_m = float(config["grid"].get("cell_size_m", 50.0))

    holdout_mask, train_mask = _holdout_masks(df, stack.profile, config)
    pool = df[train_mask].reset_index(drop=True)

    # Block size: configured km, capped so the AOI always holds >= ~4 blocks per side.
    eval_cfg = config.get("evaluation", {})
    block_px = int(round(float(eval_cfg.get("block_size_km", 50)) * 1000.0 / cell_m))
    block_px = max(10, min(block_px, min(H, W) // 4))
    logger.info("Spatial CV block size: %d px (%.1f km)", block_px, block_px * cell_m / 1000.0)

    block_ids = assign_spatial_blocks(
        pool["row"].to_numpy(), pool["col"].to_numpy(), H, W, block_size_px=block_px
    )
    positive_raster = np.zeros((H, W), dtype=np.uint8)
    pos = pool[pool["label"] == 1]
    positive_raster[pos["row"].to_numpy(), pos["col"].to_numpy()] = 1
    block_ids = enforce_cluster_integrity(
        block_ids, pool["row"].to_numpy(), pool["col"].to_numpy(), positive_raster
    )

    n_folds = min(int(eval_cfg.get("n_folds", 5)), len(np.unique(block_ids)))
    if n_folds < 2:
        raise RuntimeError("Spatial CV needs at least 2 spatial blocks; enlarge the AOI.")
    folds = spatial_block_kfold(block_ids, n_folds=n_folds, random_state=42)

    metrics_df = spatial_cv_pu(pool, feature_cols, folds, config)
    if len(metrics_df) < 2:
        raise RuntimeError(
            f"Only {len(metrics_df)} spatial-CV fold(s) had both verified positives and "
            "verified negatives in the held-out blocks. No metric is reported because it "
            "would not be meaningful; add labelled deposits/drillholes across more districts."
        )
    metrics_df.to_csv(reports_dir / f"spatial_cv_metrics_{version_tag}.csv", index=False)
    logger.info(
        "Spatial CV: %d valid folds | PR-AUC %.3f +/- %.3f | prevalence %.3f | recall@FPR10 %.3f",
        len(metrics_df), metrics_df["pr_auc"].mean(), metrics_df["pr_auc"].std(ddof=0),
        metrics_df["prevalence"].mean(), metrics_df["recall_at_fpr_0.1"].mean(),
    )

    # Blind district: scored once with the model trained without it.
    holdout = df[holdout_mask]
    holdout = holdout[holdout["label"].isin([0, 1])]
    report: dict = {"version": version_tag, "evaluated": False}
    y_blind = (holdout["label"] == 1).astype(int).to_numpy()
    if len(holdout) and 0 < y_blind.sum() < len(y_blind):
        model = MPMXGBoost(config)
        model.load(str(models_dir / "xgboost_ensemble.joblib"))
        report.update(
            evaluated=True,
            n_pos=int(y_blind.sum()),
            n_rn=int((1 - y_blind).sum()),
            **score_metrics(y_blind, model.predict_proba(holdout[feature_cols].to_numpy())),
        )
        logger.info("Blind holdout: PR-AUC %.3f (prevalence %.3f)", report["pr_auc"], report["prevalence"])
    else:
        logger.warning("Blind holdout has no verified P+RN mix; cannot be scored.")
    import json
    (reports_dir / f"blind_holdout_{version_tag}.json").write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# 6. Outputs
# ---------------------------------------------------------------------------
def stage_outputs(config: dict, version_tag: str) -> None:
    logger.info("Starting Stage 6: Generating GIS-Ready Outputs...")
    processed_dir = Path("data/processed")
    models_dir = Path("outputs/models") / version_tag
    maps_dir = Path("outputs/maps")
    targets_dir = Path("outputs/targets")
    maps_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    stack: FeatureStack = joblib.load(processed_dir / "feature_stack.joblib")
    df = pd.read_parquet(processed_dir / "dataset.parquet")
    feature_cols = [c for c in df.columns if c not in ("label", "row", "col")]

    model = MPMXGBoost(config)
    model.load(str(models_dir / "xgboost_ensemble.joblib"))

    X_all = df[feature_cols].values
    scores = model.predict_proba(X_all)
    variance = model.predict_uncertainty(X_all)
    shap_vals = model.compute_shap(X_all)

    H, W = stack.array.shape[1], stack.array.shape[2]
    row_idx = df["row"].values
    col_idx = df["col"].values
    profile = stack.profile

    # 1. Prospectivity GeoTIFF (0-1)
    prospectivity_tif = maps_dir / f"prospectivity_{version_tag}.tif"
    save_prospectivity_geotiff(scores, row_idx, col_idx, H, W, profile, str(prospectivity_tif))

    # 2. Epistemic Uncertainty GeoTIFF
    uncertainty_tif = maps_dir / f"uncertainty_{version_tag}.tif"
    save_uncertainty_geotiff(variance, row_idx, col_idx, H, W, profile, str(uncertainty_tif))

    # 3. Top-5 SHAP Attribution Channels GeoTIFF
    shap_tif = maps_dir / f"shap_top5_{version_tag}.tif"
    save_shap_geotiff(
        shap_values=shap_vals,
        row_idx=row_idx,
        col_idx=col_idx,
        H=H, W=W,
        profile=profile,
        feature_names=feature_cols,
        top_n=5,
        out_path=str(shap_tif),
    )

    # 4. Target Exploration Polygons GeoPackage
    prospectivity_2d = reconstruct_map(scores, row_idx, col_idx, H, W, nodata=np.nan)
    uncertainty_2d = reconstruct_map(variance, row_idx, col_idx, H, W, nodata=np.nan)

    targets_gdf = generate_target_polygons(
        prospectivity=prospectivity_2d,
        uncertainty=uncertainty_2d,
        profile=profile,
        min_score=0.70,
        max_uncertainty=0.20,
        min_area_ha=15.0,  # 15 ha minimum target area
    )

    targets_gpkg = targets_dir / f"targets_{version_tag}.gpkg"
    if len(targets_gdf) > 0:
        targets_gdf.to_file(targets_gpkg, driver="GPKG")
        logger.info("Saved %d drill targets to %s", len(targets_gdf), targets_gpkg)
    else:
        logger.warning("No targets met the strict prospectivity (>=0.70) and uncertainty (<=0.20) thresholds.")

    logger.info("Stage 6 (Outputs) complete. GIS deliverables ready in outputs/maps/ and outputs/targets/.")


# ---------------------------------------------------------------------------
# CLI & Dispatch
# ---------------------------------------------------------------------------
def _setup_logging(config: dict) -> None:
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = reports_dir / f"pipeline_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-35s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), mode="w", encoding="utf-8"),
        ],
    )
    logger.info("Logging initialised | log_file=%s", log_file)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MPM End-to-End Orchestrator")
    parser.add_argument("--config", default="config/pilot.yaml", help="Path to pilot.yaml")
    parser.add_argument("--stage", default="all", choices=["all"] + PIPELINE_STAGES)
    parser.add_argument("--version-tag", default="v1", help="Version tag, e.g. v1")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    _setup_logging(config)
    logger.info("=== MPM Pipeline: stage=%s | version=%s ===", args.stage, args.version_tag)

    stages_to_run = PIPELINE_STAGES[:] if args.stage == "all" else [args.stage]

    stage_dispatch = {
        "harmonize": lambda: stage_harmonize(config),
        "features":  lambda: stage_features(config),
        "labels":    lambda: stage_labels(config),
        "train":     lambda: stage_train(config, args.version_tag),
        "evaluate":  lambda: stage_evaluate(config, args.version_tag),
        "outputs":   lambda: stage_outputs(config, args.version_tag),
    }

    for stage_name in stages_to_run:
        logger.info("--- Stage: %s ---", stage_name.upper())
        try:
            stage_dispatch[stage_name]()
            logger.info("--- Stage %s: COMPLETED ---", stage_name.upper())
        except Exception:
            logger.error("--- Stage %s: FAILED ---\n%s", stage_name.upper(), traceback.format_exc())
            return 1

    logger.info("=== MPM Pipeline finished successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
