"""
scripts/run_seven_pass_training.py
====================================
End-to-end 7-Pass Mineral Prospectivity & PU-Learning training, calibration,
spatial conformal bounding, multi-criteria target ranking, and LLM Site Card
emission on both pilot regions:
  1. Yilgarn Craton (WA Goldfields, EPSG:32751)
  2. Borneo / East Kalimantan (Kelian Corridor + ESDM Iron/Coal, EPSG:32750)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from src.evaluation.metrics import (
    pr_auc,
    spatial_conformal_bounds,
    spatial_conformal_calibrate,
)
from src.features.coal_and_multimodal_features import compute_continuous_canopy_weight
from src.features.feature_stack import FeatureStack, reconstruct_map
from src.labels.borneo_labels import (
    K4_LABEL_COLS,
    build_k3_coal_rank_targets,
    build_k4_deposit_targets,
    load_kalimantan_coal_sources,
    rasterize_borneo_k4_labels,
)
from src.labels.pu_negatives import (
    OUTMOGCGate,
    compute_asymmetric_head_weights,
    compute_inverse_propensity_weights,
    compute_k4_exclusion_mask,
    fit_exploration_propensity,
    fit_pooled_exploration_propensity,
)
from src.labels.spatial_splits import assign_spatial_blocks, spatial_block_kfold
from src.schema.canonical import (
    NON_FEATURE_COLS as CANONICAL_NON_FEATURE_COLS,
    EXTENDED_NON_FEATURE_COLS,
    NON_FEATURE_COLS_SET,
    _FORBIDDEN_RAW_BAND_PATTERNS,
    FORBIDDEN_RAW_BAND_REGEXES,
    K4_HEAD_NAMES,
    K3_COAL_HEAD_NAMES,
    WARM_START_SCHEMA_POLICIES,
    resolve_warm_start_feature_schema,
    verify_no_raw_pixel_leakage,
)
from src.models.coal import compute_hierarchical_coal_probabilities
from src.models.pu_pipeline import (
    LABEL_P,
    LABEL_RN,
    LABEL_U,
    MacroBasinSequenceGate,
    TwoStageK3CoalEnsemble,
    TwoStageK4Ensemble,
    blend_predictions_by_pr_auc,
    calibrate_closed_boundary_elkan_noto,
    compute_multi_mineral_conformal_bounds,
    estimate_k4_discovery_priors,
    score_metrics,
)
from src.models.unet_model import MPMDistrictDAEUNet
from src.outputs.prospectivity_map import (
    emit_site_cards_json,
    polygonize_and_rank_targets,
    split_targets_by_known_ground,
    resolve_physical_layers,
    save_conformal_bounds_geotiff,
    save_prospectivity_geotiff,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seven_pass_train")


def _extract_windows_around_cells(
    cube: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    window_px: int = 16,
) -> np.ndarray:
    """Extract (N, C, window_px, window_px) spatial context patches centered at (rows, cols)."""
    C, H, W = cube.shape
    half = window_px // 2
    padded = np.pad(
        np.nan_to_num(cube, nan=0.0),
        ((0, 0), (half, half), (half, half)),
        mode="reflect",
    ).astype(np.float32)

    patches = np.empty((len(rows), C, window_px, window_px), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        patches[i] = padded[:, r : r + window_px, c : c + window_px]
    return patches


def _save_multiband_geotiff(
    bands_2d: list[np.ndarray],
    band_descriptions: list[str],
    profile: dict,
    out_path: Path,
) -> None:
    """Write a multi-band float32 GeoTIFF with band descriptions."""
    import rasterio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(profile)
    w = int(profile.get("width", bands_2d[0].shape[1]))
    h = int(profile.get("height", bands_2d[0].shape[0]))
    bs = min(256, w, h)
    bs = max(16, (bs // 16) * 16)
    prof.update(
        driver="GTiff",
        dtype="float32",
        count=len(bands_2d),
        nodata=np.nan,
        compress="lzw",
        tiled=True,
        blockxsize=bs,
        blockysize=bs,
    )
    with rasterio.open(out_path, "w", **prof) as dst:
        for b_idx, (arr, desc) in enumerate(zip(bands_2d, band_descriptions), start=1):
            dst.write(np.asarray(arr, dtype=np.float32), b_idx)
            dst.set_band_description(b_idx, desc)


def _write_singleband_cog(
    grid_2d: np.ndarray,
    profile: dict,
    out_path: Path,
    description: str = "MPM prospectivity map",
) -> None:
    """Write a single-band Cloud-Optimised float32 GeoTIFF with DEFLATE compression."""
    import rasterio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(profile)
    w = int(profile.get("width", grid_2d.shape[1]))
    h = int(profile.get("height", grid_2d.shape[0]))
    bs = min(256, w, h)
    bs = max(16, (bs // 16) * 16)
    prof.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
        tiled=True,
        blockxsize=bs,
        blockysize=bs,
    )
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(np.asarray(grid_2d, dtype=np.float32)[np.newaxis, :, :])
        dst.update_tags(description=description)


def _write_conformal_bounds_cog(
    lower_2d: np.ndarray,
    upper_2d: np.ndarray,
    profile: dict,
    out_path: Path,
) -> None:
    """Write a 2-band float32 GeoTIFF containing lower and upper conformal bounds."""
    import rasterio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(profile)
    w = int(profile.get("width", lower_2d.shape[1]))
    h = int(profile.get("height", lower_2d.shape[0]))
    bs = min(256, w, h)
    bs = max(16, (bs // 16) * 16)
    prof.update(
        driver="GTiff",
        dtype="float32",
        count=2,
        nodata=np.nan,
        compress="deflate",
        tiled=True,
        blockxsize=bs,
        blockysize=bs,
    )
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(np.asarray(lower_2d, dtype=np.float32), 1)
        dst.set_band_description(1, "lower_bound_L_90ci")
        dst.write(np.asarray(upper_2d, dtype=np.float32), 2)
        dst.set_band_description(2, "upper_bound_U_90ci")


def _apply_border_taper(arr: np.ndarray, margin: int = 4) -> np.ndarray:
    """Taper outer border margin to prevent DEM edge-filtering reflection artifacts."""
    if arr is None or arr.ndim != 2:
        return arr
    H, W = arr.shape
    if H <= 2 * margin or W <= 2 * margin:
        return arr
    out = arr.copy()
    for i in range(margin):
        weight = float((i + 1) / (margin + 1))
        out[i, :] = np.where(np.isnan(out[i, :]), np.nan, out[i, :] * weight)
        out[H - 1 - i, :] = np.where(np.isnan(out[H - 1 - i, :]), np.nan, out[H - 1 - i, :] * weight)
        out[:, i] = np.where(np.isnan(out[:, i]), np.nan, out[:, i] * weight)
        out[:, W - 1 - i] = np.where(np.isnan(out[:, W - 1 - i]), np.nan, out[:, W - 1 - i] * weight)
    return out


# v4.8 "find new ground" settings. Each one was accepted or rejected with the
# deposit-level harness (scripts/run_deposit_cv.py, 15 km buffer); see audit v4.8.
# The legacy (v4.7) behaviour of every switch stays available through the CLI.
# Smooth regional fields that act as location surrogates for the metals heads (audit v4.8:
# dropping them raised held-out any-metal AUC by ~0.05; for coal they help and are kept).
METALS_REGIONAL_PROXY_FEATURES: tuple[str, ...] = (
    "bouguer_gravity_mgal", "crustal_thickness_km", "moho_gravity_step_mgal_km",
    "dist_plate_boundary_km", "dist_volcanic_arc_km", "volcanic_arc_halo", "coastal_shelf_dist_km",
)
GENERALISATION_DEFAULTS: dict[str, Any] = {
    "weight_scheme": "hajek_neutral",     # legacy: "production"
    "gate_mode": "standardized",          # legacy: "production"; also "off"
    "head0_stacking": True,               # no measurable effect either way (kept)
    "min_child_weight_metals": 10.0,      # legacy: 0 = per-head defaults (1.0 stage-1, 3.0 porphyry)
    "coal_disk_mode": "exclude",          # legacy: "production_keep"
    "coal_buffer_m": 3500.0,              # legacy: 0.0
    "drop_features": (),                  # dropped for BOTH domains
    "drop_features_metals": METALS_REGIONAL_PROXY_FEATURES,  # legacy: ()
    "k4_exclusion_scale": 1.0,            # 0 / 0.5 / 2 made no difference (kept)
}


GENERALISATION_LEGACY_V47: dict[str, Any] = {
    "weight_scheme": "production", "gate_mode": "production", "head0_stacking": True,
    "min_child_weight_metals": 0.0, "coal_disk_mode": "production_keep", "coal_buffer_m": 0.0,
    "drop_features": (), "drop_features_metals": (), "k4_exclusion_scale": 1.0,
}
GREENFIELD_KM = 5.0  # targets farther than this from any known deposit are "greenfield"


def _prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Logit of a probability array; the rank softmax needs logits, not probabilities."""
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def execute_full_raster_chunked_inference(
    parquet_path: Path,
    feature_cols: list[str],
    H: int,
    W: int,
    profile: dict,
    # Metals models & calibration
    k4_ensemble: Optional[TwoStageK4Ensemble] = None,
    dae_out: Optional[dict[str, np.ndarray]] = None,
    c_hat_k4: Optional[list[float]] = None,
    q_hat_k4_list: Optional[list[float]] = None,
    w_xgb_k4: Optional[list[float]] = None,
    w_unet_k4: Optional[list[float]] = None,
    outmogc_gate_models: Optional[list[Any]] = None,
    # Coal models & calibration
    k3_coal_ensemble: Optional[TwoStageK3CoalEnsemble] = None,
    dae_coal_out: Optional[dict[str, np.ndarray]] = None,
    c_hat_k3: Optional[list[float]] = None,
    q_hat_k3_list: Optional[list[float]] = None,
    w_xgb_coal: Optional[list[float]] = None,
    w_unet_coal: Optional[list[float]] = None,
    # Propensity model
    prop_pipe: Optional[Any] = None,
    explicit_access_cols: Optional[list[str]] = None,
    access_col_indices: Optional[list[int]] = None,
    coarse_scale: int = 1,
    chunk_size: int = 250_000,
) -> dict[str, Any]:
    """Execute memory-bounded streaming chunked inference over 100% of onshore cells.

    Populates continuous 2D rasters covering every valid onshore pixel in dataset.parquet,
    guaranteeing 0% NaN holes on land while remaining strictly under 600 MB RAM footprint.
    """
    import pyarrow.parquet as pq

    out: dict[str, Any] = {}
    ex_2d = np.full((H, W), np.nan, dtype=np.float32)
    out["propensity_e_x_2d"] = ex_2d
    out["ex_2d"] = ex_2d

    cal_metals_2d = np.full((H, W), np.nan, dtype=np.float32)
    fused_metals_2d = np.full((H, W), np.nan, dtype=np.float32)
    lower_metals_2d = np.full((H, W), np.nan, dtype=np.float32)
    upper_metals_2d = np.full((H, W), np.nan, dtype=np.float32)
    err_metals_2d = np.full((H, W), np.nan, dtype=np.float32)
    cal_k4_bands_2d = [np.full((H, W), np.nan, dtype=np.float32) for _ in range(4)]
    out["cal_metals_composite_2d"] = cal_metals_2d
    out["fused_metals_composite_2d"] = fused_metals_2d
    out["lower_metals_composite_2d"] = lower_metals_2d
    out["upper_metals_composite_2d"] = upper_metals_2d
    out["dae_recon_err_metals_2d"] = err_metals_2d
    out["cal_k4_bands_2d"] = cal_k4_bands_2d

    cal_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
    fused_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
    lower_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
    upper_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
    err_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
    expected_cv_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
    epistemic_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
    cal_k3_bands_2d = [np.full((H, W), np.nan, dtype=np.float32) for _ in range(3)]
    out["cal_coal_composite_2d"] = cal_coal_2d
    out["fused_coal_composite_2d"] = fused_coal_2d
    out["lower_coal_composite_2d"] = lower_coal_2d
    out["upper_coal_composite_2d"] = upper_coal_2d
    out["dae_recon_err_coal_2d"] = err_coal_2d
    out["expected_cv_class_coal_2d"] = expected_cv_coal_2d
    out["epistemic_coal_composite_2d"] = epistemic_coal_2d
    out["cal_k3_bands_2d"] = cal_k3_bands_2d

    if k4_ensemble is not None and dae_out is not None:
        H_c_m, W_c_m = dae_out["dae_recon_err"].shape
        scale_r_m = max(1, int(round(H / H_c_m)))
        scale_c_m = max(1, int(round(W / W_c_m)))

    if k3_coal_ensemble is not None and dae_coal_out is not None:
        H_c_c, W_c_c = dae_coal_out["dae_recon_err"].shape
        scale_r_c = max(1, int(round(H / H_c_c)))
        scale_c_c = max(1, int(round(W / W_c_c)))

    pf = pq.ParquetFile(parquet_path)
    parquet_cols = set(pf.schema.names)
    read_cols = ["row", "col"] + [c for c in feature_cols if c in parquet_cols]
    if explicit_access_cols:
        for c in explicit_access_cols:
            if c in parquet_cols and c not in read_cols:
                read_cols.append(c)
    for c in ("w_canopy", "spec_ndvi", "dem_slope"):
        if c in parquet_cols and c not in read_cols:
            read_cols.append(c)

    n_total_rows = pf.metadata.num_rows
    logger.info(
        "Streaming chunked inference across %d onshore cells (%d cells/chunk)...",
        n_total_rows, chunk_size,
    )

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=chunk_size, columns=read_cols)):
        chunk_df = batch.to_pandas()
        r_ch = chunk_df["row"].to_numpy(dtype=np.int32)
        c_ch = chunk_df["col"].to_numpy(dtype=np.int32)
        # Cap distance features at local physical limit (15.0 km / 15000.0 m) matching training
        for c in feature_cols:
            if c in chunk_df.columns and (c.startswith("dist_") or c.endswith("_dist_km")):
                if c.endswith("_m"):
                    chunk_df[c] = np.clip(chunk_df[c].to_numpy(dtype=np.float32), 0.0, 15000.0)
                else:
                    chunk_df[c] = np.clip(chunk_df[c].to_numpy(dtype=np.float32), 0.0, 15.0)
        # v4.8: keep NaN, as the in-memory path does (XGBoost routes missing values natively,
        # the OUTMOGC gate and basin gate impute their own defaults). nan_to_num(0) used to
        # send missing cells down the "value = 0" branches.
        X_ch = chunk_df[feature_cols].to_numpy(dtype=np.float32)

        # Propensity scoring
        if prop_pipe is not None:
            if explicit_access_cols:
                X_acc_ch = np.nan_to_num(chunk_df[explicit_access_cols].to_numpy(dtype=np.float64), nan=0.0)
            elif access_col_indices:
                X_acc_ch = X_ch[:, access_col_indices].astype(np.float64)
            else:
                X_acc_ch = X_ch[:, :1].astype(np.float64)
            e_x_ch = np.clip(prop_pipe.predict_proba(X_acc_ch)[:, 1], 0.05, 1.0)
            ex_2d[r_ch, c_ch] = e_x_ch.astype(np.float32)
        else:
            ex_2d[r_ch, c_ch] = 0.20

        # Metals scoring
        # v4.8: metals/coal maps are scored even when the U-Net is disabled (its weight is 0 then);
        # previously the whole block was skipped and the maps stayed NaN.
        if k4_ensemble is not None:
            s_xgb_m = k4_ensemble.predict_proba(X_ch)
            std_xgb_m = k4_ensemble.predict_std(X_ch)

            if dae_out is not None:
                r_cm = np.clip(r_ch // scale_r_m, 0, H_c_m - 1)
                c_cm = np.clip(c_ch // scale_c_m, 0, W_c_m - 1)
                s_unet_m = np.column_stack([dae_out[f"oof_prob_k{k}"][r_cm, c_cm] for k in range(4)])
                err_m = dae_out["dae_recon_err"][r_cm, c_cm]
            else:
                s_unet_m = s_xgb_m.copy()
                err_m = np.full(len(X_ch), np.nan, dtype=np.float32)

            epistemic_m = np.sqrt(0.5 * (std_xgb_m ** 2) + 0.5 * ((s_xgb_m - s_unet_m) ** 2) + 1e-4)

            fused_m = np.zeros_like(s_xgb_m)
            for k in range(4):
                w_x = w_xgb_k4[k] if (w_xgb_k4 is not None and len(w_xgb_k4) > k) else 1.0
                w_u = w_unet_k4[k] if (w_unet_k4 is not None and len(w_unet_k4) > k) else 0.0
                if dae_out is None:
                    w_x, w_u = 1.0, 0.0
                fused_m[:, k] = w_x * s_xgb_m[:, k] + w_u * s_unet_m[:, k]

            cal_m = np.zeros_like(fused_m)
            for k in range(4):
                gamma_k = (
                    outmogc_gate_models[k].score(X_ch)
                    if (outmogc_gate_models is not None and len(outmogc_gate_models) > k and outmogc_gate_models[k] is not None)
                    else np.ones(len(X_ch), dtype=np.float64)
                )
                c_h = c_hat_k4[k] if (c_hat_k4 is not None and len(c_hat_k4) > k) else 0.5
                cal_m[:, k] = calibrate_closed_boundary_elkan_noto(
                    p_raw_k=fused_m[:, k],
                    c_hat_k=float(c_h),
                    gamma_closed_k=gamma_k,
                    min_c=0.10,
                )

            q_res_k4 = np.array([
                getattr(q, "q_res", float(q)) for q in q_hat_k4_list
            ], dtype=np.float64) if (q_hat_k4_list is not None and len(q_hat_k4_list) > 0) else np.full(4, 0.5)
            lower_m = np.clip(cal_m - q_res_k4 * epistemic_m, 0.0, 1.0)
            upper_m = np.clip(cal_m + 0.5 * np.max(epistemic_m, axis=1, keepdims=True), 0.0, 1.0)

            cal_metals_2d[r_ch, c_ch] = np.max(cal_m, axis=1)
            fused_metals_2d[r_ch, c_ch] = np.max(fused_m, axis=1)
            lower_metals_2d[r_ch, c_ch] = np.max(lower_m, axis=1)
            upper_metals_2d[r_ch, c_ch] = np.max(upper_m, axis=1)
            err_metals_2d[r_ch, c_ch] = err_m
            for k in range(4):
                cal_k4_bands_2d[k][r_ch, c_ch] = cal_m[:, k]

        # Coal scoring
        if k3_coal_ensemble is not None:
            if "w_canopy" in chunk_df.columns:
                w_c_ch = chunk_df["w_canopy"].to_numpy(dtype=np.float64)
            elif "spec_ndvi" in chunk_df.columns:
                w_c_ch = compute_continuous_canopy_weight(
                    np.nan_to_num(chunk_df["spec_ndvi"].to_numpy(dtype=np.float32), nan=0.5)
                ).astype(np.float64)
            else:
                w_c_ch = np.zeros(len(chunk_df), dtype=np.float64)

            coal_out_ch = k3_coal_ensemble.predict_full_outputs(X_ch, w_canopy=w_c_ch)
            s_xgb_c = coal_out_ch["probs_raw_k3"]  # Uncalibrated observation probabilities!
            std_xgb_c = coal_out_ch["std_k3"]

            if dae_coal_out is not None:
                r_cc = np.clip(r_ch // scale_r_c, 0, H_c_c - 1)
                c_cc = np.clip(c_ch // scale_c_c, 0, W_c_c - 1)
                s_unet_c = np.column_stack([dae_coal_out[f"oof_prob_k{c}"][r_cc, c_cc] for c in range(3)])
                err_c = dae_coal_out["dae_recon_err"][r_cc, c_cc]
            else:
                s_unet_c = s_xgb_c.copy()
                err_c = np.full(len(X_ch), np.nan, dtype=np.float32)

            epistemic_c = np.sqrt(0.5 * (std_xgb_c ** 2) + 0.5 * ((s_xgb_c - s_unet_c) ** 2) + 1e-4)

            fused_c = np.zeros_like(s_xgb_c)
            for c in range(3):
                w_x = w_xgb_coal[c] if (w_xgb_coal is not None and len(w_xgb_coal) > c) else 0.75
                w_u = w_unet_coal[c] if (w_unet_coal is not None and len(w_unet_coal) > c) else (
                    0.25 if dae_coal_out is not None else 0.0)
                if dae_coal_out is None:
                    w_x, w_u = 1.0, 0.0
                fused_c[:, c] = w_x * s_xgb_c[:, c] + w_u * s_unet_c[:, c]

            cal_c = np.zeros_like(fused_c)
            for c in range(3):
                c_h = c_hat_k3[c] if (c_hat_k3 is not None and len(c_hat_k3) > c) else 0.5
                cal_c[:, c] = calibrate_closed_boundary_elkan_noto(
                    p_raw_k=fused_c[:, c],
                    c_hat_k=float(c_h),
                    min_c=0.10,
                )

            q_res_k3 = np.array([
                getattr(q, "q_res", float(q)) for q in q_hat_k3_list
            ], dtype=np.float64) if (q_hat_k3_list is not None and len(q_hat_k3_list) > 0) else np.full(3, 0.5)
            lower_c = np.clip(cal_c - q_res_k3 * epistemic_c, 0.0, 1.0)
            upper_c = np.clip(cal_c + 0.5 * np.max(epistemic_c, axis=1, keepdims=True), 0.0, 1.0)
            if hasattr(k3_coal_ensemble, "basin_gate_") and k3_coal_ensemble.basin_gate_ is not None:
                g_basin_ch = k3_coal_ensemble.basin_gate_.compute_gate(X_ch, feature_names=feature_cols)
                p_cal_any_ch = 1.0 - np.prod(1.0 - np.clip(cal_c, 0.0, 1.0), axis=1)
                cal_c, p_coal_any_ch = compute_hierarchical_coal_probabilities(
                    p_cal_coal_any=p_cal_any_ch,
                    rank_logits=_prob_to_logit(fused_c),
                    basin_gate=g_basin_ch,
                    temperature=1.0,
                )
            else:
                p_coal_any_ch = 1.0 - np.prod(1.0 - np.clip(cal_c, 0.0, 1.0), axis=1)

            cal_coal_2d[r_ch, c_ch] = p_coal_any_ch
            fused_coal_2d[r_ch, c_ch] = np.max(fused_c, axis=1)
            lower_coal_2d[r_ch, c_ch] = np.max(lower_c, axis=1)
            upper_coal_2d[r_ch, c_ch] = np.max(upper_c, axis=1)
            err_coal_2d[r_ch, c_ch] = err_c
            expected_cv_coal_2d[r_ch, c_ch] = np.argmax(cal_c, axis=1).astype(np.float32)
            epistemic_coal_2d[r_ch, c_ch] = np.max(epistemic_c, axis=1)
            for c in range(3):
                cal_k3_bands_2d[c][r_ch, c_ch] = cal_c[:, c]

    logger.info("Chunked inference complete: 100% of onshore cells populated.")
    return out


def _emit_telemetry(
    phase: str,
    pass_idx: int,
    round_idx: Optional[int],
    total_rounds: Optional[int],
    progress_pct: float,
    metrics: Optional[dict] = None,
    message: Optional[str] = None,
) -> None:
    """Emit structured JSON telemetry line for real-time UI dashboard tracking."""
    r_curr = max(1, int(round_idx or 1))
    r_tot = max(1, int(total_rounds or 1))
    overall_pct = ((r_curr - 1) * 100.0 + float(np.clip(progress_pct, 0.0, 100.0))) / r_tot
    payload = {
        "phase": phase,
        "pass_idx": int(pass_idx),
        "round": r_curr,
        "total_rounds": r_tot,
        "progress_pct": round(float(np.clip(overall_pct, 0.0, 100.0)), 1),
        "round_progress_pct": round(float(np.clip(progress_pct, 0.0, 100.0)), 1),
        "metrics": metrics or {},
        "message": message or "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(f">>> [TELEMETRY] {json.dumps(payload)}", flush=True)


def run_region_seven_pass(
    region_name: str,
    data_dir: Path,
    out_dir: Path,
    block_size_m: float,
    l_threshold: float = 0.50,
    min_area_ha: float = 5.0,
    reset_from_zero: bool = False,
    train_10x: bool = False,
    incremental: bool = True,
    n_estimators: Optional[int] = None,
    n_bag: Optional[int] = None,
    dae_epochs: Optional[int] = None,
    round_idx: Optional[int] = None,
    total_rounds: Optional[int] = None,
    run_label: Optional[str] = None,
    domain: str = "both",
    base_model_dir: Optional[Union[Path, str]] = None,
    learning_rate: float = 0.05,
    max_depth: Optional[int] = None,
    disable_unet: bool = False,
    active_heads: Optional[list[str]] = None,
    scan_only: bool = False,
    on_schema_mismatch: str = "retrain",
    generalisation: Optional[dict] = None,
) -> dict:
    """Execute the complete Dual-Domain 7-Pass pipeline strictly specialized on Kalimantan."""
    gen = dict(GENERALISATION_DEFAULTS)
    gen.update({k: v for k, v in (generalisation or {}).items() if v is not None})
    unknown_gen = set(gen) - set(GENERALISATION_DEFAULTS)
    if unknown_gen:
        raise ValueError(f"unknown generalisation settings: {sorted(unknown_gen)}")
    logger.info("[%s] Generalisation settings: %s", region_name, gen)
    domain_clean = domain.lower().strip()
    if domain_clean not in ("metals", "coal", "both"):
        raise ValueError(f"domain must be 'metals', 'coal', or 'both', got '{domain}'")
    run_metals = domain_clean in ("metals", "both")
    run_coal = domain_clean in ("coal", "both")

    reg_clean = region_name.upper().strip()
    VALID_KALIMANTAN_REGIONS = {
        "KALIMANTAN", "BORNEO", "KALIMANTAN_50M", "KALIMANTAN_500M",
        "EAST_KALIMANTAN", "KALTIM", "KALIMANTAN_TIMUR",
        "HOTSPOT_SCHWANER", "HOTSPOT_KAYAN_NORTH", "HOTSPOT_KAHAYAN",
        "HOTSPOT_SCHWANER_50M", "HOTSPOT_KAYAN_50M",
        "KUTAI_BASIN", "KUTAI_BASIN_100M",
        "MAHAKAM_SUTURE", "SANGATTA_TREND", "KELIAN_CORRIDOR",
        "BERAU_FOLD_BELT", "ADANG_FAULT", "MALINAU_CORRIDOR",
        "CUSTOM_POLYGON",
    }
    if (
        reg_clean not in VALID_KALIMANTAN_REGIONS
        and not reg_clean.startswith("HOTSPOT_")
        and not reg_clean.startswith("KUTAI_")
        and not reg_clean.startswith("POLYGON_")
        and not reg_clean.startswith("SCAN_")
        and not reg_clean.startswith("CUSTOM_")
    ):
        raise ValueError(
            f"Strict Kalimantan-only policy enforced: Training outside Kalimantan is prohibited. "
            f"Requested region: '{region_name}'. Models are specialized exclusively for Kalimantan deposit mechanics (K=4)."
        )

    import shutil
    from src.models.training_run_manager import archive_existing_run, register_completed_run

    out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = out_dir / "maps"
    targets_dir = out_dir / "targets"
    models_dir = out_dir / "models"
    for d in (maps_dir, targets_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Copy polygonal boundary into out_dir if present in data_dir
    src_boundary = data_dir / "scanned_boundary.geojson"
    if not src_boundary.exists() and reg_clean in ("EAST_KALIMANTAN", "KALTIM", "KALIMANTAN_TIMUR"):
        ek_b = Path("data/east_kalimantan/east_kalimantan_boundary.geojson")
        if ek_b.exists():
            src_boundary = ek_b
    if src_boundary.exists():
        try:
            shutil.copy2(src_boundary, out_dir / "scanned_boundary.geojson")
        except Exception:
            pass

    ensemble_checkpoint = models_dir / "k4_ensemble.joblib"
    coal_ensemble_checkpoint = models_dir / "k3_coal_ensemble.joblib"
    prior_ensemble: Optional[TwoStageK4Ensemble] = None
    prior_coal_ensemble: Optional[TwoStageK3CoalEnsemble] = None

    # Resolve source checkpoint directory if --base-model-dir was provided
    src_models_dir = models_dir
    if base_model_dir is not None:
        cand_dir = Path(base_model_dir)
        if (cand_dir / "models").exists():
            src_models_dir = cand_dir / "models"
        elif cand_dir.exists():
            src_models_dir = cand_dir

    if (reset_from_zero or not incremental) and not scan_only and base_model_dir is None:
        logger.info("[%s] Resetting model weights and caches to learn from zero...", region_name)
        if round_idx is None or round_idx <= 1:
            archive_existing_run(out_dir, region_name)
        for d in (models_dir, maps_dir, targets_dir):
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()
    else:
        src_k4 = src_models_dir / "k4_ensemble.joblib"
        if not src_k4.exists() and ensemble_checkpoint.exists():
            src_k4 = ensemble_checkpoint
        src_k3 = src_models_dir / "k3_coal_ensemble.joblib"
        if not src_k3.exists() and coal_ensemble_checkpoint.exists():
            src_k3 = coal_ensemble_checkpoint

        if src_k4.exists() and run_metals:
            try:
                prior_ensemble = TwoStageK4Ensemble.load(src_k4)
                logger.info(
                    "[%s] Loaded prior K=4 ensemble from %s",
                    region_name, src_k4,
                )
            except Exception as e:
                logger.warning("[%s] Could not load prior K=4 ensemble from %s: %s", region_name, src_k4, e)
                prior_ensemble = None
        if src_k3.exists() and run_coal:
            try:
                prior_coal_ensemble = TwoStageK3CoalEnsemble.load(src_k3)
                logger.info(
                    "[%s] Loaded prior K=3 coal ensemble from %s",
                    region_name, src_k3,
                )
            except Exception as e:
                logger.warning("[%s] Could not load prior K=3 coal ensemble: %s", region_name, e)
                prior_coal_ensemble = None

        if not scan_only and (round_idx is None or round_idx <= 1):
            archive_existing_run(out_dir, region_name)

        # If warm-starting from an external base_model_dir, copy DAE U-Net weights into models_dir
        if src_models_dir.resolve() != models_dir.resolve() and src_models_dir.exists():
            for pt_file in src_models_dir.glob("*.pt"):
                try:
                    shutil.copy2(pt_file, models_dir / pt_file.name)
                except Exception:
                    pass
            if scan_only:
                for jb_file in src_models_dir.glob("*.joblib"):
                    try:
                        shutil.copy2(jb_file, models_dir / jb_file.name)
                    except Exception:
                        pass

    _emit_telemetry(
        phase="pass1",
        pass_idx=1,
        round_idx=round_idx,
        total_rounds=total_rounds,
        progress_pct=8.0,
        message=f"[{region_name}] Pass 1: Loading polygonal dataset & verifying zero raw pixel leakage",
    )

    df_full = pd.read_parquet(data_dir / "dataset.parquet")

    # Resolve profile and spatial dimensions without loading multi-GB feature_stack.joblib when GeoTIFFs exist
    import rasterio

    ref_tif = data_dir / "dem_elevation.tif"
    stack: Optional[FeatureStack] = None
    if ref_tif.exists() and len(df_full) > 150_000:
        with rasterio.open(ref_tif) as src:
            profile = dict(src.profile)
            H, W = int(src.height), int(src.width)
    else:
        stack = joblib.load(data_dir / "feature_stack.joblib")
        profile = dict(stack.profile)
        _, H, W = stack.array.shape

    # Subsample massive island-scale parquet (>150k rows) while keeping 100% of all metallic & coal positives
    is_subsampled = len(df_full) > 150_000
    if is_subsampled:
        pos_cols_check = [
            c for c in (
                "label",
                "label_k0_porphyry_cu", "label_k1_skarn_fe_cu", "label_k2_epithermal_au", "label_k3_laterite_fe_ni",
                "label_coal", "label_coal_k0_high_cv", "label_coal_k1_med_cv", "label_coal_k2_low_cv",
            )
            if c in df_full.columns
        ]
        is_any_pos = np.zeros(len(df_full), dtype=bool)
        for pc in pos_cols_check:
            is_any_pos |= (df_full[pc].to_numpy() == 1)
        pos_idx_full = np.where(is_any_pos)[0]
        bg_idx_full = np.where(~is_any_pos)[0]
        rng_sub = np.random.default_rng(42)
        n_bg_keep = min(len(bg_idx_full), 80_000)
        bg_keep = np.sort(rng_sub.choice(bg_idx_full, size=n_bg_keep, replace=False))
        keep_idx = np.sort(np.concatenate([pos_idx_full, bg_keep]))
        df = df_full.iloc[keep_idx].reset_index(drop=True)
        logger.info(
            "[%s] Stratified island subsample: %d rows (%d positives + %d background) from %d total land cells",
            region_name, len(df), len(pos_idx_full), len(bg_keep), len(df_full),
        )
        del df_full
    else:
        df = df_full

    NON_FEATURE_COLS = set(EXTENDED_NON_FEATURE_COLS) | set(CANONICAL_NON_FEATURE_COLS) | {
        "alteration_benchmark",
        "commodity_label",
        "is_centroid_artifact",
        "target_class",
        "w_canopy",
        "geology_coal_contact_dist_km",
        "in_coal_bearing_formation",
        "in_holocene_peat_or_karst_mask",
        "in_ophiolite_unit",
    }
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and not c.startswith("label_")
        and not any(pat in c.lower() for pat in _FORBIDDEN_RAW_BAND_PATTERNS)
    ]
    _drop = set(gen["drop_features"] or ())
    if _drop:
        logger.info("[%s] Dropping features by generalisation setting: %s", region_name,
                    sorted(_drop & set(feature_cols)))
        feature_cols = [c for c in feature_cols if c not in _drop]
    # Align feature columns if warm-starting or scanning with a pre-trained model checkpoint.
    # Never silently drop dataset channels the prior model lacks (e.g. newly ingested real SAR):
    # resolve_warm_start_feature_schema logs dropped/missing columns and, under the default
    # "retrain" policy, discards an incompatible warm-start so all channels are learned.
    ref_model_feats = None
    if prior_ensemble is not None and getattr(prior_ensemble, "feature_names_", None):
        ref_model_feats = list(prior_ensemble.feature_names_)
    elif prior_coal_ensemble is not None and getattr(prior_coal_ensemble, "feature_names_", None):
        ref_model_feats = list(prior_coal_ensemble.feature_names_)
    schema_decision = resolve_warm_start_feature_schema(
        candidate_cols=feature_cols,
        prior_feature_names=ref_model_feats,
        available_cols=list(df.columns),
        scan_only=scan_only,
        policy=on_schema_mismatch,
        excluded_cols=NON_FEATURE_COLS,
    )
    if schema_decision["dropped"]:
        logger.warning(
            "[%s] Prior checkpoint lacks %d dataset channel(s): %s",
            region_name, len(schema_decision["dropped"]), schema_decision["dropped"],
        )
    if schema_decision["missing"]:
        logger.warning(
            "[%s] Prior checkpoint requires %d channel(s) absent from dataset: %s",
            region_name, len(schema_decision["missing"]), schema_decision["missing"],
        )
    if schema_decision["use_prior"]:
        if schema_decision["dropped"] or schema_decision["missing"]:
            logger.warning(
                "[%s] Using prior feature schema (%s): %d dropped channels are IGNORED, %d missing channels ZERO-FILLED",
                region_name, schema_decision["reason"], len(schema_decision["dropped"]), len(schema_decision["missing"]),
            )
        for req_f in schema_decision["missing"]:
            df[req_f] = np.float32(0.0)
    elif ref_model_feats:
        logger.warning(
            "[%s] Feature schema mismatch with prior checkpoint -> discarding warm-start and training "
            "from zero on all %d dataset channels (use --on-schema-mismatch prior to keep legacy behaviour)",
            region_name, len(schema_decision["feature_cols"]),
        )
        prior_ensemble = None
        prior_coal_ensemble = None
    feature_cols = list(schema_decision["feature_cols"])

    verify_no_raw_pixel_leakage(feature_cols)
    C = len(feature_cols)

    # Cap distance features at local physical limit (15.0 km / 15000.0 m) to prevent GPS grid triangulation
    for c in feature_cols:
        if c.startswith("dist_") or c.endswith("_dist_km"):
            if c.endswith("_m"):
                df[c] = np.clip(df[c].to_numpy(dtype=np.float32), 0.0, 15000.0)
            else:
                df[c] = np.clip(df[c].to_numpy(dtype=np.float32), 0.0, 15.0)

    # Gate 1: Verify No Raw Pixel Leakage (including strict quarantine of Pass 2 access features)
    verify_no_raw_pixel_leakage(feature_cols)
    logger.info("[%s] Pass 1 verified: %d physical/structural channels", region_name, len(feature_cols))

    X_all = df[feature_cols].to_numpy(dtype=np.float64)
    labels = (
        df["label"].to_numpy(dtype=np.int32)
        if "label" in df.columns
        else (df["label_coal"].to_numpy(dtype=np.int32) if "label_coal" in df.columns else np.zeros(len(df), dtype=np.int32))
    )
    rows = df["row"].to_numpy(dtype=np.int32)
    cols = df["col"].to_numpy(dtype=np.int32)

    # -------------------------------------------------------------------------
    # Step 1.1: Multi-Mineral Targets (K=4), Coal Rank Targets (K=3) & Exploration Footprint
    # -------------------------------------------------------------------------
    k4_col_names = K4_LABEL_COLS
    has_k4 = all(c in df.columns for c in k4_col_names)
    if not has_k4:
        deposit_targets = build_k4_deposit_targets()
        rasters_k4 = rasterize_borneo_k4_labels(deposit_targets, profile, buffer_m=1000.0)
        for c in k4_col_names:
            df[c] = rasters_k4[c][rows, cols]
        df["explored_footprint"] = rasters_k4["explored_footprint"][rows, cols]
        df["alteration_benchmark"] = rasters_k4["alteration_benchmark"][rows, cols]

    y_k4 = np.column_stack([df[c].to_numpy(dtype=np.int32) for c in k4_col_names])

    k3_coal_col_names = [
        "label_coal_k0_high_cv",
        "label_coal_k1_med_cv",
        "label_coal_k2_low_cv",
    ]
    has_k3_coal = all(c in df.columns for c in k3_coal_col_names)
    if not has_k3_coal:
        # Derive K=3 coal targets if running on a pilot tile without precomputed coal rank columns
        ndvi_2d = reconstruct_map(
            df["spec_ndvi"].to_numpy(dtype=np.float64) if "spec_ndvi" in df.columns else np.full(len(df), 0.5),
            rows, cols, H, W, nodata=0.5,
        )
        mine_2d = np.zeros((H, W), dtype=np.int8)
        poly_2d = np.ones((H, W), dtype=np.int32)
        coal_gdf = load_kalimantan_coal_sources()
        coal_res = build_k3_coal_rank_targets(
            concessions=coal_gdf,
            ndvi=ndvi_2d,
            active_mine_mask=mine_2d,
            formation_polygon_id=poly_2d,
            transform=profile.get("transform"),
        )
        for c in k3_coal_col_names:
            df[c] = coal_res[c][rows, cols]
        df["coal_tonnage_weight"] = coal_res["coal_tonnage_weights"][rows, cols]

    y_k3_coal = np.column_stack([df[c].to_numpy(dtype=np.int32) for c in k3_coal_col_names])

    n_coal_pos = int((y_k3_coal == 1).sum())
    if run_coal and n_coal_pos == 0 and not scan_only:
        if domain_clean == "coal":
            if prior_coal_ensemble is not None:
                logger.info(
                    "[%s] Zero ground-truth coal positives inside polygon, switching to scan_only inference using loaded coal ensemble.",
                    region_name,
                )
                scan_only = True
            else:
                raise ValueError(
                    f"[{region_name}] Execution halted for --domain coal: zero ground-truth coal positives found "
                    f"in region (or coal rank columns missing). Cannot train coal prospectivity model without verified coal occurrences."
                )
        elif domain_clean == "both":
            if prior_coal_ensemble is not None:
                logger.info("[%s] Zero coal positives inside polygon; will run pre-trained coal ensemble inference.", region_name)
            else:
                logger.warning(
                    "[%s] Zero ground-truth coal positives found in region (or coal rank columns missing). "
                    "Skipping Domain B (coal) training branch under --domain both.",
                    region_name,
                )
                run_coal = False

    coal_tonnage_weights = (
        df["coal_tonnage_weight"].to_numpy(dtype=np.float64)
        if "coal_tonnage_weight" in df.columns
        else np.ones(len(df), dtype=np.float64)
    )
    w_canopy_arr = (
        df["w_canopy"].to_numpy(dtype=np.float64)
        if "w_canopy" in df.columns
        else (
            compute_continuous_canopy_weight(df["spec_ndvi"].to_numpy(dtype=np.float32)).astype(np.float64)
            if "spec_ndvi" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )
    )

    pos_mask_metals = np.any(y_k4 == 1, axis=1)
    pos_mask_coal = np.any(y_k3_coal == 1, axis=1) if run_coal else np.zeros(len(df), dtype=bool)
    pos_mask_any = pos_mask_metals | pos_mask_coal

    # -------------------------------------------------------------------------
    # Step 1.2: Spatial Block Partitioning & Georeferencing (Prior to Gating & Weights)
    # -------------------------------------------------------------------------
    from rasterio.crs import CRS as RioCRS
    import rasterio.transform
    transform = profile["transform"]
    is_geo = RioCRS.from_user_input(profile["crs"]).is_geographic
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    cell_coords = np.column_stack([xs, ys])

    raw_dx = abs(float(transform.a)) if abs(float(transform.a)) > 0 else 50.0
    px_size_m = raw_dx * 111_320.0 if is_geo else raw_dx
    eff_block_m = block_size_m * 111_320.0 if (is_geo and block_size_m < 50.0) else block_size_m
    block_size_px = min(max(12, int(round(eff_block_m / max(px_size_m, 1.0)))), max(12, min(H, W) // 5))
    block_ids = assign_spatial_blocks(rows, cols, H, W, block_size_px=block_size_px)
    folds = spatial_block_kfold(block_ids, n_folds=5, random_state=42)

    all_idx = np.arange(len(df))

    # Resolve validation fold for metals upfront
    if run_metals:
        total_pos_m = max(1, int(pos_mask_metals.sum()))
        target_val_pos_m = 0.20 * total_pos_m
        metal_valid_folds = [
            (tr_idx_m, va_idx_m)
            for tr_idx_m, va_idx_m in folds
            if pos_mask_metals[tr_idx_m].sum() >= max(2, int(0.5 * total_pos_m))
            and pos_mask_metals[va_idx_m].sum() >= 1
        ]
        if metal_valid_folds:
            chosen_metals_fold = max(
                metal_valid_folds,
                key=lambda pair: (
                    sum(1 for k in range(4) if int((y_k4[pair[1], k] == 1).sum()) >= 1),
                    -abs(int(pos_mask_metals[pair[1]].sum()) - target_val_pos_m),
                ),
            )
        else:
            chosen_metals_fold = folds[0]
        train_metals_idx, val_idx = chosen_metals_fold
    else:
        train_metals_idx = all_idx
        val_idx = np.array([], dtype=int)

    # Resolve validation fold for coal upfront
    if run_coal:
        coal_valid_folds = [
            (tr_idx_c, va_idx_c)
            for tr_idx_c, va_idx_c in folds
            if pos_mask_coal[tr_idx_c].sum() >= 2 and pos_mask_coal[va_idx_c].sum() >= 1
        ]
        if coal_valid_folds:
            chosen_coal_fold = max(
                coal_valid_folds,
                key=lambda pair: sum(1 for c in range(3) if int((y_k3_coal[pair[1], c] == 1).sum()) >= 1),
            )
        else:
            chosen_coal_fold = folds[0]
        tr_coal_idx, val_coal_idx = chosen_coal_fold
    else:
        tr_coal_idx = all_idx
        val_coal_idx = np.array([], dtype=int)

    # Synchronize unified train_idx for dual-domain isolation (no validation cell leakage into normalizers)
    if run_metals and run_coal:
        shared_valid_folds = [
            (tr_i, va_i)
            for tr_i, va_i in folds
            if (pos_mask_metals[tr_i].sum() >= 2 and pos_mask_metals[va_i].sum() >= 1)
            and (pos_mask_coal[tr_i].sum() >= 2 and pos_mask_coal[va_i].sum() >= 1)
        ]
        if shared_valid_folds:
            chosen_fold = max(
                shared_valid_folds,
                key=lambda pair: (
                    sum(1 for k in range(4) if int((y_k4[pair[1], k] == 1).sum()) >= 1)
                    + sum(1 for c in range(3) if int((y_k3_coal[pair[1], c] == 1).sum()) >= 1),
                ),
            )
            train_idx, val_idx = chosen_fold
            val_coal_idx = val_idx
            tr_coal_idx = train_idx
        else:
            val_combined = np.union1d(val_idx, val_coal_idx)
            train_idx = np.setdiff1d(all_idx, val_combined)
            tr_coal_idx = train_idx
    elif run_metals:
        train_idx = train_metals_idx
        tr_coal_idx = train_metals_idx
    else:
        train_idx = tr_coal_idx

    # OUTMOGC closed boundary gates per head (Fitted on TRAINING SPLIT ONLY - No Validation Leakage)
    outmogc_gates_list = []
    outmogc_gate_models: list[Optional[OUTMOGCGate]] = []
    for k in range(4):
        pos_k_train_mask = (y_k4[train_idx, k] == 1)
        if np.sum(pos_k_train_mask) >= 5 and gen["gate_mode"] != "off":
            X_k_pos_tr = X_all[train_idx][pos_k_train_mask]
            _std_gate = gen["gate_mode"] == "standardized"
            _metal_keep = [i for i, c in enumerate(feature_cols) if c not in set(gen["drop_features_metals"] or ())]
            gate_k = OUTMOGCGate(
                n_pca_components=min(4, len(_metal_keep)),
                n_gmm_components=min(2, max(1, len(X_k_pos_tr) // 10)),
                random_state=42,
                standardize=_std_gate,
                feature_index=_metal_keep if len(_metal_keep) < len(feature_cols) else None,
            )
            gate_k.fit(X_k_pos_tr, X_reference=X_all[train_idx][~pos_k_train_mask] if _std_gate else None)
            outmogc_gate_models.append(gate_k)
            outmogc_gates_list.append(gate_k.score(X_all))
        else:
            outmogc_gate_models.append(None)
            outmogc_gates_list.append(np.ones(len(X_all), dtype=np.float64))
    outmogc_gates = np.column_stack(outmogc_gates_list)

    # Exclusion masks (Pass 1 Rule 4: buffer around training deposit occurrences per head ONLY)
    positive_coords_k_train = {k: cell_coords[train_idx][y_k4[train_idx, k] == 1] for k in range(4)}
    _xs = float(gen["k4_exclusion_scale"])
    if _xs <= 0:
        exclusion_masks = np.ones((len(df), 4), dtype=np.float64)
    else:
        exclusion_masks = compute_k4_exclusion_mask(
            coords=cell_coords,
            positive_coords_k=positive_coords_k_train,
            labels_k4=y_k4,
            is_geographic=is_geo,
            r_k_star_m=None if _xs == 1.0 else {
                0: 6000.0 * _xs, 1: 3500.0 * _xs, 2: 6000.0 * _xs, 3: (10000.0 * _xs, 2000.0 * _xs, 45.0)},
        )

    # v4.8 coal negatives: the 1 km ESDM disks (label_coal) are known coal, but the K3 heads
    # are trained on strike-line halos, so the production subsample used to feed every disk
    # cell to the heads as a negative. coal_disk_mode="exclude" gives them weight 0, and
    # coal_buffer_m > 0 also removes unlabeled cells near TRAINING known coal from the negatives.
    coal_disk_cells = (
        (df["label_coal"].to_numpy() == 1) if "label_coal" in df.columns else np.zeros(len(df), dtype=bool)
    ) & ~pos_mask_coal
    coal_disk_neg_excl = coal_disk_cells if gen["coal_disk_mode"] == "exclude" else np.zeros(len(df), dtype=bool)
    coal_neg_excl = coal_disk_neg_excl.copy()
    if run_coal and float(gen["coal_buffer_m"]) > 0:
        from scipy.spatial import cKDTree as _cKDTree

        if is_geo:
            _lat0 = float(np.nanmean(cell_coords[:, 1]))
            _mxy = np.column_stack([cell_coords[:, 0] * 111_320.0 * np.cos(np.radians(_lat0)),
                                    cell_coords[:, 1] * 110_570.0])
        else:
            _mxy = cell_coords
        _in_tr = np.zeros(len(df), dtype=bool)
        _in_tr[tr_coal_idx] = True
        _known_tr = (pos_mask_coal | coal_disk_cells) & _in_tr
        if _known_tr.any():
            _d, _ = _cKDTree(_mxy[_known_tr]).query(_mxy, k=1)
            coal_neg_excl |= (_d <= float(gen["coal_buffer_m"])) & ~pos_mask_coal
        logger.info("[%s] Coal negative exclusion: %d disk cells, %d cells total (buffer %.0f m)",
                    region_name, int(coal_disk_neg_excl.sum()), int(coal_neg_excl.sum()), float(gen["coal_buffer_m"]))

    # Normalise cube channels using ONLY training split statistics (no validation leakage)
    from src.models.unet_model import project_spatial_grid

    tr_rows, tr_cols = rows[train_idx], cols[train_idx]
    if stack is not None:
        # Strictly exclude all NON_FEATURE_COLS from U-Net input cube (Gemini Argon Rule 4)
        stack_feats = list(stack.feature_names) if hasattr(stack, "feature_names") else feature_cols
        feat_indices = [i for i, name in enumerate(stack_feats) if name in feature_cols]
        if feat_indices:
            cube_raw = stack.array[feat_indices].astype(np.float32)
        else:
            valid_indices = [i for i, f in enumerate(stack_feats) if f not in NON_FEATURE_COLS and not f.startswith("label_")]
            cube_raw = stack.array[valid_indices].astype(np.float32)
        cube_norm = cube_raw.copy()
        for ch in range(cube_norm.shape[0]):
            band = cube_norm[ch]
            # Cap distance channels at local physical limit to prevent U-Net GPS triangulation
            feat_name = feature_cols[ch] if ch < len(feature_cols) else ""
            if feat_name.startswith("dist_") or feat_name.endswith("_dist_km"):
                if feat_name.endswith("_m"):
                    band = np.clip(band, 0.0, 15000.0)
                else:
                    band = np.clip(band, 0.0, 15.0)
            tr_vals = band[tr_rows, tr_cols]
            valid_tr = tr_vals[np.isfinite(tr_vals) & (tr_vals > -9000.0)]
            m = float(np.mean(valid_tr)) if len(valid_tr) > 0 else 0.0
            s = float(np.std(valid_tr)) + 1e-6 if len(valid_tr) > 0 else 1.0
            norm_b = (band - m) / s
            valid_b = np.isfinite(band) & (band > -9000.0)
            cube_norm[ch] = np.where(valid_b, norm_b, 0.0)
        dae_decimate_factor = max(1, min(H, W) // 360) if min(H, W) <= 400 else max(5, min(H, W) // 360)
    else:
        # Continuous 2D grid loading strictly excluding NON_FEATURE_COLS (float32 memory-safe)
        coarse_scale = max(1, min(H, W) // 360)
        H_c, W_c = max(16, H // coarse_scale), max(16, W // coarse_scale)
        cube_norm = np.zeros((C, H_c, W_c), dtype=np.float32)
        r_c = np.clip(rows // coarse_scale, 0, H_c - 1)
        c_c = np.clip(cols // coarse_scale, 0, W_c - 1)
        for ch, feat_name in enumerate(feature_cols):
            tif_p = data_dir / f"{feat_name}.tif"
            loaded = False
            if tif_p.exists():
                try:
                    with rasterio.open(tif_p) as src:
                        band_c = src.read(
                            1,
                            out_shape=(H_c, W_c),
                            resampling=rasterio.enums.Resampling.bilinear,
                        ).astype(np.float32)
                        if feat_name.startswith("dist_") or feat_name.endswith("_dist_km"):
                            if feat_name.endswith("_m"):
                                band_c = np.clip(band_c, 0.0, 15000.0)
                            else:
                                band_c = np.clip(band_c, 0.0, 15.0)
                        tr_vals = band_c[r_c[train_idx], c_c[train_idx]]
                        valid_tr = tr_vals[np.isfinite(tr_vals) & (tr_vals > -9000.0)]
                        m = float(np.mean(valid_tr)) if len(valid_tr) > 0 else 0.0
                        s = float(np.std(valid_tr)) + 1e-6 if len(valid_tr) > 0 else 1.0
                        valid_b = np.isfinite(band_c) & (band_c > -9000.0)
                        cube_norm[ch] = np.where(valid_b, (band_c - m) / s, 0.0)
                        loaded = True
                except Exception as exc:
                    logger.debug("Could not read %s as continuous GeoTIFF: %s", tif_p, exc)
            if not loaded:
                vals = X_all[:, ch].astype(np.float32)
                tr_vals = vals[train_idx]
                m = float(np.nanmean(tr_vals)) if np.isfinite(tr_vals).any() else 0.0
                s = float(np.nanstd(tr_vals)) + 1e-6
                norm_vals = np.nan_to_num((vals - m) / s, nan=0.0)
                cube_norm[ch, r_c, c_c] = norm_vals
        dae_decimate_factor = 1

    # -------------------------------------------------------------------------
    # Pass 2: Pooled Selection-Bias Weighting (Shared Trunk for Metals & Coal)
    # -------------------------------------------------------------------------
    _emit_telemetry(
        phase="pass2",
        pass_idx=2,
        round_idx=round_idx,
        total_rounds=total_rounds,
        progress_pct=18.0,
        message=f"[{region_name}] Pass 2: Fitting logistics exploration propensity & Hájek weights",
    )
    explicit_access_cols = [
        c for c in (
            "dist_major_rivers_km",
            "dist_haul_roads_km",
            "dist_transport_km",
            "gfw_forest_disturbance",
            "forest_disturbance_rate",
            "dem_elevation",
            "dem_slope",
            "outcrop_exposure_index",
            "regolith_cover_m",
            "coastal_shelf_dist_km",
        )
        if c in df.columns
    ]
    access_col_indices: list[int] = []
    if explicit_access_cols:
        X_access = df[explicit_access_cols].to_numpy(dtype=np.float64)
    else:
        ACCESS_KEYS = (
            "outcrop", "regolith", "cover", "coastal", "slope", "elev", "dem", "ndvi", "tri"
        )
        access_col_indices = [
            i for i, col in enumerate(feature_cols)
            if any(k in col.lower() for k in ACCESS_KEYS)
        ] or [0]
        X_access = X_all[:, access_col_indices]

    explored_footprint_col = df.get("explored_footprint", None)
    if explored_footprint_col is not None:
        footprint_tr = (explored_footprint_col.to_numpy()[train_idx] > 0)
        explored_mask_tr = footprint_tr | pos_mask_any[train_idx]
    else:
        explored_mask_tr = pos_mask_any[train_idx]

    y_exp_tr = explored_mask_tr.astype(int)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if 0 < y_exp_tr.sum() < len(y_exp_tr):
        prop_pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=500, random_state=42),
        )
        prop_pipe.fit(X_access[train_idx], y_exp_tr)
        e_x = np.clip(prop_pipe.predict_proba(X_access)[:, 1], 0.05, 1.0)
    else:
        prop_pipe = None
        e_x = np.full(len(X_all), 0.20, dtype=np.float64)

    # Asymmetric sample weights for all 4 metallic heads
    sample_weights_k4 = np.column_stack([
        compute_asymmetric_head_weights(
            labels_k=y_k4[:, k],
            e_logistics=e_x,
            exclusion_mask_k=exclusion_masks[:, k],
            hajek_normalize=gen["weight_scheme"] == "hajek_neutral",
        )
        for k in range(4)
    ])

    train_blocks = np.unique(block_ids[train_idx])
    val_blocks = np.unique(block_ids[val_idx])
    overlap = set(train_blocks).intersection(set(val_blocks))
    if overlap:
        val_mask_disjoint = ~np.isin(block_ids[val_idx], list(overlap))
        val_idx = val_idx[val_mask_disjoint]
        val_blocks = np.unique(block_ids[val_idx])

    eff_n_estimators = n_estimators if n_estimators is not None else (500 if train_10x else 100)
    eff_n_bag = n_bag if n_bag is not None else (10 if train_10x else 5)
    eff_dae_epochs = 0 if (disable_unet or scan_only) else (dae_epochs if dae_epochs is not None else (15 if train_10x else 3))
    eff_round = int(round_idx) if round_idx is not None else 1
    eff_total_rounds = max(1, int(total_rounds)) if total_rounds is not None else 1
    eff_lr = float(learning_rate) if learning_rate is not None else 0.05
    eff_max_depth = int(max_depth) if max_depth is not None else 6
    eff_base_models_dir = src_models_dir
    is_final_round = (total_rounds is None) or (eff_round >= eff_total_rounds)
    round_str = f" [Round {round_idx}/{total_rounds}]" if round_idx and total_rounds else ""

    fold_assignments = np.full(len(rows), fill_value=-1, dtype=int)
    for f_idx, (_, va_idx) in enumerate(folds):
        fold_assignments[va_idx] = f_idx

    if stack is not None:
        fold_ids_2d = reconstruct_map(fold_assignments.astype(np.float64), rows, cols, H, W, nodata=-1.0).astype(int)
    else:
        H_c, W_c = cube_norm.shape[1], cube_norm.shape[2]
        coarse_scale = max(1, min(H, W) // 360)
        fold_ids_2d = np.full((H_c, W_c), fill_value=-1, dtype=int)
        r_c = np.clip(rows // coarse_scale, 0, H_c - 1)
        c_c = np.clip(cols // coarse_scale, 0, W_c - 1)
        fold_ids_2d[r_c, c_c] = fold_assignments

    # Defaults if a domain is skipped or for cross-domain reporting
    pr_auc_xgb_heads: list[float] = [float("nan")] * 4
    pr_auc_unet_heads: list[float] = [float("nan")] * 4
    pr_auc_fused_heads: list[float] = [float("nan")] * 4
    w_xgb_k4: list[float] = [0.70] * 4
    w_unet_k4: list[float] = [0.30] * 4
    c_hat_k4: list[float] = [0.5] * 4
    q_hat_k4_list: list[float] = [0.5] * 4
    ranked_gdf = pd.DataFrame()
    cards: list[dict] = []
    cv_rows: list[dict] = []
    oof_metrics: dict = {}
    k4_ensemble: Optional[TwoStageK4Ensemble] = None
    dae_out: Optional[dict[str, np.ndarray]] = None

    coal_summary: dict = {}
    pr_auc_xgb_coal: list[float] = [float("nan")] * 3
    pr_auc_unet_coal: list[float] = [float("nan")] * 3
    pr_auc_fused_coal: list[float] = [float("nan")] * 3
    w_xgb_coal: list[float] = [0.75] * 3
    w_unet_coal: list[float] = [0.25] * 3
    c_hat_k3: list[float] = [0.5] * 3
    q_hat_k3_list: list[float] = [0.5] * 3
    ranked_gdf_coal = pd.DataFrame()
    cards_coal: list[dict] = []
    cv_rows_coal: list[dict] = []
    oof_metrics_coal: dict = {}
    k3_coal_ensemble: Optional[TwoStageK3CoalEnsemble] = None
    dae_coal_out: Optional[dict[str, np.ndarray]] = None

    # =========================================================================
    # DOMAIN A: SIFT K=4 METALLIC MINERAL SYSTEMS (Passes 3A - 7A)
    # =========================================================================
    if run_metals:
        _emit_telemetry(
            phase="pass3",
            pass_idx=3,
            round_idx=round_idx,
            total_rounds=total_rounds,
            progress_pct=30.0,
            message=f"[{region_name}] Pass 3A: {'Executing K=4 Metals inference scan' if scan_only else 'Training K=4 Metals XGBoost + DAE U-Net'}",
        )
        if scan_only and prior_ensemble is not None:
            logger.info("[%s] Scan-only mode: Using loaded TwoStageK4Ensemble without retraining.", region_name)
            k4_ensemble = prior_ensemble
        else:
            logger.info(
                "[%s]%s Running Passes 3A-7A (K=4 Metals): n_estimators=%d, n_bag=%d, dae_epochs=%d, lr=%.3f, depth=%s",
                region_name, round_str, eff_n_estimators, eff_n_bag, eff_dae_epochs,
                float(learning_rate), str(max_depth or 6),
            )
            k4_ensemble = TwoStageK4Ensemble(
                n_estimators=eff_n_estimators,
                n_bag=eff_n_bag,
                learning_rate=float(learning_rate),
                max_depth=int(max_depth) if max_depth is not None else 6,
                random_state=42,
                head0_stacking=bool(gen["head0_stacking"]),
                min_child_weight=(gen["min_child_weight_metals"] or None),
                exclude_features=tuple(gen["drop_features_metals"] or ()),
            )
            k4_ensemble.fit(
                X=X_all[train_idx],
                y_k4=y_k4[train_idx],
                sample_weights_k4=sample_weights_k4[train_idx],
                e_logistics=e_x[train_idx],
                exclusion_masks=exclusion_masks[train_idx],
                spatial_block_ids=block_ids[train_idx],
                prior_ensemble=prior_ensemble,
                round_idx=eff_round,
                fold_id=0,
                X_val=X_all[val_idx],
                y_val_k4=y_k4[val_idx],
                feature_names=feature_cols,
            )
            k4_ensemble.save(ensemble_checkpoint)
            logger.info("[%s] Saved updated TwoStageK4Ensemble weights to %s", region_name, ensemble_checkpoint)

        # Ensure feature alignment for k4_ensemble prediction
        if getattr(k4_ensemble, "feature_names_", None) and list(k4_ensemble.feature_names_) != feature_cols:
            for rf in k4_ensemble.feature_names_:
                if rf not in df.columns:
                    df[rf] = np.float32(0.0)
            X_metals_pred = df[list(k4_ensemble.feature_names_)].to_numpy(dtype=np.float64)
        else:
            X_metals_pred = X_all

        s_xgb_k4_full = k4_ensemble.predict_proba(X_metals_pred)
        std_xgb_k4_full = k4_ensemble.predict_std(X_metals_pred)

        # Mask inactive heads if user selected a subset of active_heads
        if active_heads:
            active_set = {h.strip().lower() for h in active_heads}
            for k, hname in enumerate(K4_HEAD_NAMES):
                if hname.lower() not in active_set and not any(hname.lower().startswith(a) for a in active_set):
                    if any(k_other in active_set for k_other in [h.lower() for h in K4_HEAD_NAMES]):
                        s_xgb_k4_full[:, k] = 0.0

        # Pass 3A.2: District DAE-U-Net (K=4 Metals)
        if stack is not None:
            y_k4_2d = np.stack([
                reconstruct_map(np.where(y_k4[:, k] == 1, 1.0, 0.0), rows, cols, H, W, nodata=0.0)
                for k in range(4)
            ], axis=0)
            r_lookup, c_lookup = rows, cols
        else:
            H_c, W_c = cube_norm.shape[1], cube_norm.shape[2]
            coarse_scale = max(1, min(H, W) // 360)
            r_lookup = np.clip(rows // coarse_scale, 0, H_c - 1)
            c_lookup = np.clip(cols // coarse_scale, 0, W_c - 1)
            y_k4_2d = np.zeros((4, H_c, W_c), dtype=np.float32)
            for k in range(4):
                pos_k = y_k4[:, k] == 1
                y_k4_2d[k, r_lookup[pos_k], c_lookup[pos_k]] = 1.0

        if eff_dae_epochs > 0 and not scan_only and not disable_unet:
            propensity_weights_2d_m = np.ones((cube_norm.shape[1], cube_norm.shape[2]), dtype=np.float32)
            propensity_weights_2d_m[r_lookup, c_lookup] = (1.0 / np.clip(e_x, 0.05, 1.0)).astype(np.float32)

            dae_unet = MPMDistrictDAEUNet(
                in_channels=cube_norm.shape[0],
                n_classes=4,
                window_px=32,
                batch_size=16,
                dropout_p=0.3,
                channel_drop_p=0.15,
                recon_weight=0.5,
                px_size_m=float(px_size_m),
            )
            dae_out = dae_unet.cross_validate_spatial_blocks(
                X_grid_50m=cube_norm,
                y_grid_k4=y_k4_2d,
                block_ids_50m=fold_ids_2d,
                decimate_factor=dae_decimate_factor,
                epochs=eff_dae_epochs,
                checkpoints_dir=models_dir,
                checkpoint_prefix="dae_metals",
                propensity_weights_2d=propensity_weights_2d_m,
            )

            s_unet_k4_full = np.column_stack([
                dae_out[f"oof_prob_k{k}"][r_lookup, c_lookup]
                for k in range(4)
            ])
            dae_recon_err_full = dae_out["dae_recon_err"][r_lookup, c_lookup]
        else:
            dae_out = None
            s_unet_k4_full = s_xgb_k4_full.copy()
            dae_recon_err_full = np.zeros(len(X_all), dtype=np.float32)

        epistemic_k4_full = np.sqrt(
            0.5 * (std_xgb_k4_full ** 2) + 0.5 * ((s_xgb_k4_full - s_unet_k4_full) ** 2) + 1e-4
        )

        # Pass 3A.3: Validation PR-AUC Ensembling Per Head
        fused_k4_full = np.zeros_like(s_xgb_k4_full)
        pr_auc_xgb_heads = []
        pr_auc_unet_heads = []
        pr_auc_fused_heads = []
        w_xgb_k4 = []
        w_unet_k4 = []

        for k in range(4):
            y_k_val = (y_k4[val_idx, k] == 1).astype(int) if len(val_idx) > 0 else np.array([], dtype=int)
            pos_k_val = int(np.sum(y_k_val))
            s_xgb_val_k = s_xgb_k4_full[val_idx, k] if len(val_idx) > 0 else np.array([], dtype=float)
            s_unet_val_k = s_unet_k4_full[val_idx, k] if len(val_idx) > 0 else np.array([], dtype=float)

            if pos_k_val >= 1 and (len(y_k_val) - pos_k_val) >= 1 and dae_out is not None:
                auc_xgb_k = float(pr_auc(y_k_val, s_xgb_val_k))
                auc_unet_k = float(pr_auc(y_k_val, s_unet_val_k))
                _, w_xgb_k, w_unet_k = blend_predictions_by_pr_auc(
                    s_xgb_val_k, s_unet_val_k, auc_xgb_k, auc_unet_k
                )
                fused_val_k = w_xgb_k * s_xgb_val_k + w_unet_k * s_unet_val_k
                auc_fused_k = float(pr_auc(y_k_val, fused_val_k))
                fused_k4_full[:, k] = w_xgb_k * s_xgb_k4_full[:, k] + w_unet_k * s_unet_k4_full[:, k]
            elif pos_k_val >= 1 and (len(y_k_val) - pos_k_val) >= 1:
                auc_xgb_k = float(pr_auc(y_k_val, s_xgb_val_k))
                auc_unet_k = float("nan")
                auc_fused_k = auc_xgb_k
                w_xgb_k = 1.0
                w_unet_k = 0.0
                fused_k4_full[:, k] = s_xgb_k4_full[:, k]
            else:
                auc_xgb_k = float(k4_ensemble.best_val_pr_auc_.get(k, float("nan"))) if hasattr(k4_ensemble, "best_val_pr_auc_") else float("nan")
                auc_unet_k = float("nan")
                auc_fused_k = auc_xgb_k
                w_xgb_k = 1.0
                w_unet_k = 0.0
                fused_k4_full[:, k] = s_xgb_k4_full[:, k]

            w_xgb_k4.append(float(w_xgb_k))
            w_unet_k4.append(float(w_unet_k))
            pr_auc_xgb_heads.append(auc_xgb_k)
            pr_auc_unet_heads.append(auc_unet_k)
            pr_auc_fused_heads.append(auc_fused_k)

        logger.info(
            "[%s] Pass 3A K=4 Metals | H0 (Porphyry)=%s | H1 (Skarn)=%s | H2 (Epithermal)=%s | H3 (Laterite)=%s",
            region_name,
            f"{pr_auc_fused_heads[0]:.4f}" if np.isfinite(pr_auc_fused_heads[0]) else "NaN",
            f"{pr_auc_fused_heads[1]:.4f}" if np.isfinite(pr_auc_fused_heads[1]) else "NaN",
            f"{pr_auc_fused_heads[2]:.4f}" if np.isfinite(pr_auc_fused_heads[2]) else "NaN",
            f"{pr_auc_fused_heads[3]:.4f}" if np.isfinite(pr_auc_fused_heads[3]) else "NaN",
        )

        # Pass 4A: Closed-Boundary Elkan-Noto Scaling
        _emit_telemetry(
            phase="pass4",
            pass_idx=4,
            round_idx=round_idx,
            total_rounds=total_rounds,
            progress_pct=52.0,
            metrics={
                "pr_auc_xgb": round(float(np.nanmean([x for x in pr_auc_xgb_heads if np.isfinite(x)])), 4) if any(np.isfinite(x) for x in pr_auc_xgb_heads) else None,
                "pr_auc_unet": round(float(np.nanmean([x for x in pr_auc_unet_heads if np.isfinite(x)])), 4) if any(np.isfinite(x) for x in pr_auc_unet_heads) else None,
                "pr_auc_fused": round(float(np.nanmean([x for x in pr_auc_fused_heads if np.isfinite(x)])), 4) if any(np.isfinite(x) for x in pr_auc_fused_heads) else None,
            },
            message=f"[{region_name}] Pass 4A: Calibrating Elkan-Noto discovery priors",
        )
        if len(val_idx) > 0 and np.any(y_k4[val_idx] == 1):
            c_hat_k4 = estimate_k4_discovery_priors(
                val_probs_k4=fused_k4_full[val_idx],
                val_targets_k4=y_k4[val_idx],
                min_c=0.10,
                train_block_ids=train_blocks,
                val_block_ids=block_ids[val_idx],
            )
        else:
            c_hat_k4 = [
                float(np.clip(k4_ensemble.c_hat_.get(k, 0.35), 0.10, 1.0))
                if hasattr(k4_ensemble, "c_hat_") else 0.35
                for k in range(4)
            ]

        cal_k4_full = np.zeros_like(fused_k4_full)
        for k in range(4):
            # Guard against unconstrained scaling on heads with zero regional deposit occurrences (unless scan_only)
            if not scan_only and int(np.sum(y_k4[:, k] == 1)) == 0:
                cal_k4_full[:, k] = 0.0
                continue
            cal_k4_full[:, k] = calibrate_closed_boundary_elkan_noto(
                p_raw_k=fused_k4_full[:, k],
                c_hat_k=c_hat_k4[k],
                gamma_closed_k=outmogc_gates[:, k],
                min_c=0.10,
            )

        # Pass 5A: Multi-Mineral Conformal Bounds & Prediction Sets
        _emit_telemetry(
            phase="pass5",
            pass_idx=5,
            round_idx=round_idx,
            total_rounds=total_rounds,
            progress_pct=62.0,
            metrics={"c_hat": round(float(np.nanmean(c_hat_k4)), 4)},
            message=f"[{region_name}] Pass 5A: Computing 90% Studentized Conformal Bounds L(x), U(x)",
        )
        lower_k4_list = []
        q_hat_k4_list = []
        v_scores_k4_list = []

        for k in range(4):
            p_cal_k = cal_k4_full[:, k]
            sig_k = epistemic_k4_full[:, k]
            pos_k_val_mask = (y_k4[val_idx, k] == 1) if len(val_idx) > 0 else np.zeros(0, dtype=bool)
            if np.sum(pos_k_val_mask) >= 1:
                cal_pos_p = p_cal_k[val_idx][pos_k_val_mask]
                cal_pos_sig = sig_k[val_idx][pos_k_val_mask]
            else:
                cal_pos_p = np.array([float(np.percentile(p_cal_k, 95))])
                cal_pos_sig = np.array([float(np.mean(sig_k))])

            l_k, v_k, q_k = compute_multi_mineral_conformal_bounds(
                p_cal_k=p_cal_k,
                sigma_ens_k=sig_k,
                cal_pos_p_cal=cal_pos_p,
                cal_pos_sigma=cal_pos_sig,
                alpha=0.10,
            )
            lower_k4_list.append(l_k)
            v_scores_k4_list.append(v_k)
            q_hat_k4_list.append(q_k)

        lower_k4_full = np.column_stack(lower_k4_list)
        v_scores_k4 = np.column_stack(v_scores_k4_list)
        conformal_sets = [
            [K4_HEAD_NAMES[k] for k in range(4) if v_scores_k4[i, k] <= q_hat_k4_list[k]]
            for i in range(len(cal_k4_full))
        ]

        cal_prob_composite = np.max(cal_k4_full, axis=1)
        lower_L_composite = np.max(lower_k4_full, axis=1)
        upper_U_composite = np.clip(cal_prob_composite + 0.5 * np.max(epistemic_k4_full, axis=1), 0.0, 1.0)
        fused_composite = np.max(fused_k4_full, axis=1)

    # =========================================================================
    # DOMAIN B: SIFT K=3 COAL RANK PROSPECTIVITY ENGINE (Passes 3B - 5B)
    # =========================================================================
    if run_coal:
        _emit_telemetry(
            phase="pass3",
            pass_idx=3,
            round_idx=round_idx,
            total_rounds=total_rounds,
            progress_pct=68.0,
            message=f"[{region_name}] Pass 3B: {'Executing K=3 Coal inference scan' if (scan_only or n_coal_pos == 0) else 'Training K=3 Coal Rank Ensemble'}",
        )
        if (scan_only or n_coal_pos == 0) and prior_coal_ensemble is not None:
            logger.info("[%s] Scan-only mode: Using loaded TwoStageK3CoalEnsemble without retraining.", region_name)
            k3_coal_ensemble = prior_coal_ensemble
        else:
            logger.info(
                "[%s]%s Running Passes 3B-5B (K=3 Coal Rank): n_estimators=%d, n_bag=%d, dae_epochs=%d, lr=%.3f, depth=%s",
                region_name, round_str, eff_n_estimators, eff_n_bag, eff_dae_epochs,
                float(learning_rate), str(max_depth or 5),
            )
            logger.info(
                "[%s] Synchronized coal validation split: %d train cells (%d coal pos), %d val cells (%d coal pos)",
                region_name, len(tr_coal_idx), int(pos_mask_coal[tr_coal_idx].sum()),
                len(val_coal_idx), int(pos_mask_coal[val_coal_idx].sum()),
            )

            k3_coal_ensemble = TwoStageK3CoalEnsemble(
                n_estimators=eff_n_estimators,
                n_bag=eff_n_bag,
                learning_rate=float(learning_rate),
                max_depth=int(max_depth) if max_depth is not None else 5,
                random_state=42,
                unlabeled_weighting="neutral" if gen["weight_scheme"] == "hajek_neutral" else "inverse_propensity",
            )
            k3_coal_ensemble.fit(
                X=X_all[tr_coal_idx],
                y_k3=y_k3_coal[tr_coal_idx],
                tonnage_weights=coal_tonnage_weights[tr_coal_idx],
                e_pooled=e_x[tr_coal_idx],
                w_canopy=w_canopy_arr[tr_coal_idx],
                spatial_block_ids=block_ids[tr_coal_idx],
                prior_ensemble=prior_coal_ensemble,
                round_idx=eff_round,
                fold_id=0,
                X_val=X_all[val_coal_idx],
                y_val_k3=y_k3_coal[val_coal_idx],
                feature_names=feature_cols,
                negative_exclusion_mask=coal_neg_excl[tr_coal_idx],
            )
            k3_coal_ensemble.save(coal_ensemble_checkpoint)
            logger.info("[%s] Saved updated TwoStageK3CoalEnsemble weights to %s", region_name, coal_ensemble_checkpoint)

        if getattr(k3_coal_ensemble, "feature_names_", None) and list(k3_coal_ensemble.feature_names_) != feature_cols:
            for rf in k3_coal_ensemble.feature_names_:
                if rf not in df.columns:
                    df[rf] = np.float32(0.0)
            X_coal_pred = df[list(k3_coal_ensemble.feature_names_)].to_numpy(dtype=np.float64)
            coal_feat_names = list(k3_coal_ensemble.feature_names_)
        else:
            X_coal_pred = X_all
            coal_feat_names = feature_cols

        coal_out_full = k3_coal_ensemble.predict_full_outputs(X_coal_pred, w_canopy=w_canopy_arr)
        # R4.2: Use uncalibrated observation probabilities P(S=1|X) for blending with U-Net
        s_xgb_k3_full = coal_out_full["probs_raw_k3"]
        std_xgb_k3_full = coal_out_full["std_k3"]

        # Pass 3B.2: District DAE-U-Net (K=3 Coal Basin Context)
        if stack is not None:
            y_k3_2d = np.stack([
                reconstruct_map(np.where(y_k3_coal[:, c] == 1, 1.0, 0.0), rows, cols, H, W, nodata=0.0)
                for c in range(3)
            ], axis=0)
            r_lookup_c, c_lookup_c = rows, cols
        else:
            H_c, W_c = cube_norm.shape[1], cube_norm.shape[2]
            coarse_scale = max(1, min(H, W) // 360)
            r_lookup_c = np.clip(rows // coarse_scale, 0, H_c - 1)
            c_lookup_c = np.clip(cols // coarse_scale, 0, W_c - 1)
            y_k3_2d = np.zeros((3, H_c, W_c), dtype=np.float32)
            for c in range(3):
                pos_c = y_k3_coal[:, c] == 1
                y_k3_2d[c, r_lookup_c[pos_c], c_lookup_c[pos_c]] = 1.0

        if eff_dae_epochs > 0 and not scan_only and not disable_unet and n_coal_pos > 0:
            propensity_weights_2d_c = np.ones((cube_norm.shape[1], cube_norm.shape[2]), dtype=np.float32)
            propensity_weights_2d_c[r_lookup_c, c_lookup_c] = (1.0 / np.clip(e_x, 0.05, 1.0)).astype(np.float32)

            dae_unet_coal = MPMDistrictDAEUNet(
                in_channels=cube_norm.shape[0],
                n_classes=3,
                window_px=32,
                batch_size=16,
                dropout_p=0.3,
                channel_drop_p=0.15,
                recon_weight=0.5,
                px_size_m=float(px_size_m),
            )
            dae_coal_out = dae_unet_coal.cross_validate_spatial_blocks(
                X_grid_50m=cube_norm,
                y_grid_k4=y_k3_2d,
                block_ids_50m=fold_ids_2d,
                decimate_factor=dae_decimate_factor,
                epochs=eff_dae_epochs,
                checkpoints_dir=models_dir,
                checkpoint_prefix="dae_coal",
                propensity_weights_2d=propensity_weights_2d_c,
            )

            s_unet_k3_full = np.column_stack([
                dae_coal_out[f"oof_prob_k{c}"][r_lookup_c, c_lookup_c]
                for c in range(3)
            ])
        else:
            dae_coal_out = None
            s_unet_k3_full = s_xgb_k3_full.copy()

        epistemic_k3_full = np.sqrt(
            0.5 * (std_xgb_k3_full ** 2) + 0.5 * ((s_xgb_k3_full - s_unet_k3_full) ** 2) + 1e-4
        )

        # Pass 3B.3: Validation PR-AUC Ensembling Per Coal Rank Head
        fused_k3_full = np.zeros_like(s_xgb_k3_full)
        pr_auc_xgb_coal = []
        pr_auc_unet_coal = []
        pr_auc_fused_coal = []
        w_xgb_coal = []
        w_unet_coal = []

        for c in range(3):
            y_c_val = (y_k3_coal[val_coal_idx, c] == 1).astype(int) if len(val_coal_idx) > 0 else np.array([], dtype=int)
            pos_c_val = int(np.sum(y_c_val))
            s_xgb_val_c = s_xgb_k3_full[val_coal_idx, c] if len(val_coal_idx) > 0 else np.array([], dtype=float)
            s_unet_val_c = s_unet_k3_full[val_coal_idx, c] if len(val_coal_idx) > 0 else np.array([], dtype=float)

            if pos_c_val >= 1 and (len(y_c_val) - pos_c_val) >= 1 and dae_coal_out is not None:
                auc_xgb_c = float(pr_auc(y_c_val, s_xgb_val_c))
                auc_unet_c = float(pr_auc(y_c_val, s_unet_val_c))
                _, w_xgb_c, w_unet_c = blend_predictions_by_pr_auc(
                    s_xgb_val_c, s_unet_val_c, auc_xgb_c, auc_unet_c
                )
                fused_val_c = w_xgb_c * s_xgb_val_c + w_unet_c * s_unet_val_c
                auc_fused_c = float(pr_auc(y_c_val, fused_val_c))
                fused_k3_full[:, c] = w_xgb_c * s_xgb_k3_full[:, c] + w_unet_c * s_unet_k3_full[:, c]
            elif pos_c_val >= 1 and (len(y_c_val) - pos_c_val) >= 1:
                auc_xgb_c = float(pr_auc(y_c_val, s_xgb_val_c))
                auc_unet_c = float("nan")
                auc_fused_c = auc_xgb_c
                w_xgb_c = 1.0
                w_unet_c = 0.0
                fused_k3_full[:, c] = s_xgb_k3_full[:, c]
            else:
                auc_xgb_c = float(k3_coal_ensemble.best_val_pr_auc_.get(c, float("nan"))) if hasattr(k3_coal_ensemble, "best_val_pr_auc_") else float("nan")
                auc_unet_c = float("nan")
                auc_fused_c = auc_xgb_c
                w_xgb_c = 1.0 if dae_coal_out is None else 0.75
                w_unet_c = 0.0 if dae_coal_out is None else 0.25
                fused_k3_full[:, c] = w_xgb_c * s_xgb_k3_full[:, c] + w_unet_c * s_unet_k3_full[:, c]

            w_xgb_coal.append(float(w_xgb_c))
            w_unet_coal.append(float(w_unet_c))
            pr_auc_xgb_coal.append(auc_xgb_c)
            pr_auc_unet_coal.append(auc_unet_c)
            pr_auc_fused_coal.append(auc_fused_c)

        logger.info(
            "[%s] Pass 3B K=3 Coal | H0 (High-CV >6100)=%s | H1 (Med-CV 5100-6100)=%s | H2 (Low-CV <5100)=%s",
            region_name,
            f"{pr_auc_fused_coal[0]:.4f}" if np.isfinite(pr_auc_fused_coal[0]) else "NaN",
            f"{pr_auc_fused_coal[1]:.4f}" if np.isfinite(pr_auc_fused_coal[1]) else "NaN",
            f"{pr_auc_fused_coal[2]:.4f}" if np.isfinite(pr_auc_fused_coal[2]) else "NaN",
        )

        # Pass 4B: Joint Post-Fusion Elkan-Noto Posterior Calibration (R4.2)
        c_hat_k3 = []
        for c in range(3):
            val_pos_mask_c = (y_k3_coal[val_coal_idx, c] == 1) if len(val_coal_idx) > 0 else np.zeros(0, dtype=bool)
            if np.sum(val_pos_mask_c) >= 1:
                c_val = float(np.mean(fused_k3_full[val_coal_idx][val_pos_mask_c, c]))
                c_hat_k3.append(float(np.clip(c_val, 0.10, 1.0)))
            else:
                ens_c = float(k3_coal_ensemble.c_hat_.get(c, 0.50)) if hasattr(k3_coal_ensemble, "c_hat_") else 0.50
                c_hat_k3.append(float(np.clip(ens_c, 0.10, 1.0)))

        cal_k3_raw = np.zeros_like(fused_k3_full)
        for c in range(3):
            cal_k3_raw[:, c] = calibrate_closed_boundary_elkan_noto(
                p_raw_k=fused_k3_full[:, c],
                c_hat_k=c_hat_k3[c],
                min_c=0.10,
            )
        p_cal_coal_any_unmod = 1.0 - np.prod(1.0 - np.clip(cal_k3_raw, 0.0, 1.0), axis=1)

        # Move MacroBasinSequenceGate to post-U-Net fusion in Pass 4B with hierarchical rank softmax (Bug 4, D5)
        if hasattr(k3_coal_ensemble, "basin_gate_") and k3_coal_ensemble.basin_gate_ is not None:
            basin_gate_arr = k3_coal_ensemble.basin_gate_.compute_gate(X_coal_pred, feature_names=coal_feat_names)
        else:
            basin_gate_arr = np.ones(len(X_all), dtype=np.float64)

        cal_k3_full, p_coal_any_full = compute_hierarchical_coal_probabilities(
            p_cal_coal_any=p_cal_coal_any_unmod,
            rank_logits=_prob_to_logit(fused_k3_full),
            basin_gate=basin_gate_arr,
            temperature=1.0,
        )
        expected_cv_class_full = np.argmax(cal_k3_full, axis=1).astype(np.float32)

        # Pass 5B: Multi-Rank Conformal Bounds & Composite Rank Probability
        lower_k3_list = []
        q_hat_k3_list = []
        for c in range(3):
            p_cal_c = cal_k3_full[:, c]
            sig_c = epistemic_k3_full[:, c]
            pos_c_val_mask = (y_k3_coal[val_coal_idx, c] == 1) if len(val_coal_idx) > 0 else np.zeros(0, dtype=bool)
            if np.sum(pos_c_val_mask) >= 1:
                cal_pos_p = p_cal_c[val_coal_idx][pos_c_val_mask]
                cal_pos_sig = sig_c[val_coal_idx][pos_c_val_mask]
            else:
                cal_pos_p = np.array([float(np.percentile(p_cal_c, 95))])
                cal_pos_sig = np.array([float(np.mean(sig_c))])
            l_c, _, q_c = compute_multi_mineral_conformal_bounds(
                p_cal_k=p_cal_c,
                sigma_ens_k=sig_c,
                cal_pos_p_cal=cal_pos_p,
                cal_pos_sigma=cal_pos_sig,
                alpha=0.10,
            )
            lower_k3_list.append(l_c)
            q_hat_k3_list.append(q_c)

        lower_k3_full = np.column_stack(lower_k3_list)

    # =========================================================================
    # RECONSTRUCT 2D CONTINUOUS PROSPECTIVITY & PHYSICAL FIELDS (0% NaN Holes)
    # =========================================================================
    is_final_round = (round_idx is None or total_rounds is None or int(round_idx) >= int(total_rounds))

    if not is_final_round:
        logger.info(
            "[%s] [Round %s/%s] Intermediate tree update complete. Skipping heavy 2.92M raster & polygon generation until final round.",
            region_name, eff_round, total_rounds,
        )
        active_coal_xgb = [x for x in pr_auc_xgb_coal if np.isfinite(x)] if run_coal else []
        active_coal_unet = [x for x in pr_auc_unet_coal if np.isfinite(x)] if run_coal else []
        active_coal_fused = [x for x in pr_auc_fused_coal if np.isfinite(x)] if run_coal else []

        coal_summary = {
            "n_coal_positives_total": int(pos_mask_coal.sum()),
            "n_coal_positives_per_head": {
                K3_COAL_HEAD_NAMES[c]: int((y_k3_coal[:, c] == 1).sum()) for c in range(3)
            },
            "pr_auc_coal_xgb": round(float(np.nanmean(active_coal_xgb)), 4) if active_coal_xgb else None,
            "pr_auc_coal_unet": round(float(np.nanmean(active_coal_unet)), 4) if active_coal_unet else None,
            "pr_auc_coal_fused": round(float(np.nanmean(active_coal_fused)), 4) if active_coal_fused else None,
            "pr_auc_coal_heads": {
                K3_COAL_HEAD_NAMES[c]: (
                    round(float(pr_auc_fused_coal[c]), 4) if np.isfinite(pr_auc_fused_coal[c]) else None
                )
                for c in range(3)
            },
            "pr_auc_coal_xgb_heads": {
                K3_COAL_HEAD_NAMES[c]: (
                    round(float(pr_auc_xgb_coal[c]), 4) if np.isfinite(pr_auc_xgb_coal[c]) else None
                )
                for c in range(3)
            },
            "elkan_noto_c_hat_coal_per_head": [round(float(x), 4) for x in c_hat_k3],
            "conformal_q_hat_90_coal_per_head": [round(float(x), 4) for x in q_hat_k3_list],
            "p_coal_any_mean": round(float(np.nanmean(p_coal_any_full)), 4),
            "p_coal_any_max": round(float(np.nanmax(p_coal_any_full)), 4),
        } if run_coal else None

        active_xgb = [x for x in pr_auc_xgb_heads if np.isfinite(x)]
        active_unet = [x for x in pr_auc_unet_heads if np.isfinite(x)]
        active_fused = [x for x in pr_auc_fused_heads if np.isfinite(x)]

        summary = {
            "region": region_name,
            "domain": domain_clean,
            "n_cells": len(df),
            "n_positives": int(pos_mask_any.sum()),
            "n_metal_positives": int(pos_mask_metals.sum()),
            "n_coal_positives": int(pos_mask_coal.sum()) if run_coal else 0,
            "pr_auc_xgb": round(float(np.nanmean(active_xgb)), 4) if active_xgb else (coal_summary.get("pr_auc_coal_xgb") if coal_summary else None),
            "pr_auc_unet": round(float(np.nanmean(active_unet)), 4) if active_unet else (coal_summary.get("pr_auc_coal_unet") if coal_summary else None),
            "pr_auc_fused": round(float(np.nanmean(active_fused)), 4) if active_fused else (coal_summary.get("pr_auc_coal_fused") if coal_summary else None),
            "pr_auc_heads": {
                K4_HEAD_NAMES[k]: (
                    round(float(pr_auc_fused_heads[k]), 4)
                    if np.isfinite(pr_auc_fused_heads[k])
                    else None
                )
                for k in range(4)
            },
            "pr_auc_xgb_heads": {
                K4_HEAD_NAMES[k]: (
                    round(float(pr_auc_xgb_heads[k]), 4)
                    if np.isfinite(pr_auc_xgb_heads[k])
                    else None
                )
                for k in range(4)
            },
            "coal_k3_metrics": coal_summary if coal_summary else None,
            "w_xgb": round(float(np.nanmean(w_xgb_k4)) if (run_metals and w_xgb_k4 and len(w_xgb_k4) > 0) else (float(np.nanmean(w_xgb_coal)) if (run_coal and w_xgb_coal and len(w_xgb_coal) > 0) else 0.70), 4),
            "w_unet": round(float(np.nanmean(w_unet_k4)) if (run_metals and w_unet_k4 and len(w_unet_k4) > 0) else (float(np.nanmean(w_unet_coal)) if (run_coal and w_unet_coal and len(w_unet_coal) > 0) else 0.30), 4),
            "elkan_noto_c_hat": (
                round(float(np.nanmean(c_hat_k3)), 4)
                if (run_coal and not run_metals and c_hat_k3 and len(c_hat_k3) > 0)
                else (round(float(np.nanmean(c_hat_k4)), 4) if (c_hat_k4 is not None and len(c_hat_k4) > 0) else 0.5)
            ),
            "rejected_warm_start_heads": getattr(k4_ensemble, "rejected_heads_", {}) if k4_ensemble else {},
            "round_idx": round_idx,
            "total_rounds": total_rounds,
            "n_estimators": eff_n_estimators,
            "n_bag": eff_n_bag,
            "dae_epochs": eff_dae_epochs,
            "resolution_m": round(float(px_size_m), 1),
        }

        history_path = out_dir / "ten_rounds_history.json"
        existing_history = []
        if eff_round > 1 and history_path.exists():
            try:
                existing_history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                existing_history = []
        existing_history.append({
            "round": eff_round,
            "cumulative_trees": eff_round * eff_n_estimators,
            "pr_auc_xgb": summary.get("pr_auc_xgb"),
            "pr_auc_unet": summary.get("pr_auc_unet"),
            "pr_auc_fused": summary.get("pr_auc_fused"),
            "pr_auc_heads": summary.get("pr_auc_heads", {}),
            "coal_pr_auc_fused": coal_summary.get("pr_auc_coal_fused") if coal_summary else None,
            "elkan_noto_c_hat": summary.get("elkan_noto_c_hat"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        history_path.write_text(json.dumps(existing_history, indent=2) + "\n", encoding="utf-8")
        _emit_telemetry(
            phase="pass5",
            pass_idx=5,
            round_idx=round_idx,
            total_rounds=total_rounds,
            progress_pct=100.0,
            metrics={
                "pr_auc_xgb": summary.get("pr_auc_xgb"),
                "pr_auc_unet": summary.get("pr_auc_unet"),
                "pr_auc_fused": summary.get("pr_auc_fused"),
                "c_hat": summary.get("elkan_noto_c_hat"),
            },
            message=f"[{region_name}] Round {eff_round}/{total_rounds} complete (Fused PR-AUC={summary.get('pr_auc_fused')})",
        )
        logger.info(
            "[%s] --- [ROUND %s/%s COMPLETED] Fused PR-AUC: %s (Metals) | %s (Coal) ---",
            region_name, eff_round, total_rounds, summary.get("pr_auc_fused"), coal_summary.get("pr_auc_coal_fused") if coal_summary else "N/A",
        )
        return summary

    _emit_telemetry(
        phase="pass6",
        pass_idx=6,
        round_idx=round_idx,
        total_rounds=total_rounds,
        progress_pct=78.0,
        message=f"[{region_name}] Pass 6: Reconstructing 2D prospectivity grids & clipping to polygonal boundary",
    )
    parquet_path = data_dir / "dataset.parquet"
    if is_subsampled and parquet_path.exists():
        logger.info("[%s] Executing streaming chunked inference across 100%% of onshore cells...", region_name)
        raster_out = execute_full_raster_chunked_inference(
            parquet_path=parquet_path,
            feature_cols=feature_cols,
            H=H,
            W=W,
            profile=profile,
            k4_ensemble=k4_ensemble,
            dae_out=dae_out,
            c_hat_k4=c_hat_k4,
            q_hat_k4_list=q_hat_k4_list,
            w_xgb_k4=w_xgb_k4,
            w_unet_k4=w_unet_k4,
            outmogc_gate_models=outmogc_gate_models,
            k3_coal_ensemble=k3_coal_ensemble,
            dae_coal_out=dae_coal_out,
            c_hat_k3=c_hat_k3,
            q_hat_k3_list=q_hat_k3_list,
            w_xgb_coal=w_xgb_coal,
            w_unet_coal=w_unet_coal,
            prop_pipe=prop_pipe,
            explicit_access_cols=explicit_access_cols,
            access_col_indices=access_col_indices,
            chunk_size=250_000,
        )
        ex_2d = raster_out["ex_2d"]
        cal_metals_2d = raster_out["cal_metals_composite_2d"]
        fused_metals_2d = raster_out["fused_metals_composite_2d"]
        lower_metals_2d = raster_out["lower_metals_composite_2d"]
        upper_metals_2d = raster_out["upper_metals_composite_2d"]
        err_metals_2d = raster_out["dae_recon_err_metals_2d"]
        metal_bands_2d = raster_out["cal_k4_bands_2d"] + [cal_metals_2d]

        cal_coal_2d = raster_out["cal_coal_composite_2d"]
        fused_coal_2d = raster_out["fused_coal_composite_2d"]
        lower_coal_2d = raster_out["lower_coal_composite_2d"]
        upper_coal_2d = raster_out["upper_coal_composite_2d"]
        err_coal_2d = raster_out["dae_recon_err_coal_2d"]
        expected_cv_coal_2d = raster_out["expected_cv_class_coal_2d"]
        coal_bands_2d = raster_out["cal_k3_bands_2d"] + [cal_coal_2d, expected_cv_coal_2d]
        basin_gate_2d = (
            reconstruct_map(basin_gate_arr, rows, cols, H, W, nodata=1.0)
            if run_coal
            else np.ones((H, W), dtype=np.float32)
        )
    else:
        # Reconstruct directly when full grid is in memory
        ex_2d = reconstruct_map(e_x, rows, cols, H, W, nodata=np.nan)

        if run_metals:
            cal_metals_2d = reconstruct_map(cal_prob_composite, rows, cols, H, W, nodata=np.nan)
            lower_metals_2d = reconstruct_map(lower_L_composite, rows, cols, H, W, nodata=np.nan)
            upper_metals_2d = reconstruct_map(upper_U_composite, rows, cols, H, W, nodata=np.nan)
            fused_metals_2d = reconstruct_map(fused_composite, rows, cols, H, W, nodata=np.nan)
            err_metals_2d = reconstruct_map(dae_recon_err_full, rows, cols, H, W, nodata=np.nan)
            metal_bands_2d = [
                reconstruct_map(cal_k4_full[:, k], rows, cols, H, W, nodata=np.nan)
                for k in range(4)
            ] + [cal_metals_2d]
        else:
            cal_metals_2d = np.full((H, W), np.nan, dtype=np.float32)
            lower_metals_2d = np.full((H, W), np.nan, dtype=np.float32)
            upper_metals_2d = np.full((H, W), np.nan, dtype=np.float32)
            fused_metals_2d = np.full((H, W), np.nan, dtype=np.float32)
            err_metals_2d = np.full((H, W), np.nan, dtype=np.float32)
            metal_bands_2d = [np.full((H, W), np.nan, dtype=np.float32) for _ in range(5)]

        if run_coal:
            cal_coal_2d = reconstruct_map(p_coal_any_full, rows, cols, H, W, nodata=np.nan)
            lower_coal_2d = reconstruct_map(np.max(lower_k3_full, axis=1), rows, cols, H, W, nodata=np.nan)
            upper_coal_2d = reconstruct_map(
                np.clip(p_coal_any_full + 0.5 * np.max(epistemic_k3_full, axis=1), 0.0, 1.0),
                rows, cols, H, W, nodata=np.nan,
            )
            fused_coal_2d = reconstruct_map(np.max(fused_k3_full, axis=1), rows, cols, H, W, nodata=np.nan)
            if dae_coal_out is not None:
                err_coal_2d = reconstruct_map(dae_coal_out["dae_recon_err"][r_lookup_c, c_lookup_c], rows, cols, H, W, nodata=np.nan)
            else:
                err_coal_2d = np.zeros((H, W), dtype=np.float32)
            expected_cv_coal_2d = reconstruct_map(expected_cv_class_full, rows, cols, H, W, nodata=np.nan)
            coal_bands_2d = [
                reconstruct_map(cal_k3_full[:, c], rows, cols, H, W, nodata=np.nan)
                for c in range(3)
            ] + [cal_coal_2d, expected_cv_coal_2d]
            basin_gate_2d = reconstruct_map(basin_gate_arr, rows, cols, H, W, nodata=1.0)
        else:
            cal_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
            lower_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
            upper_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
            fused_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
            err_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
            expected_cv_coal_2d = np.full((H, W), np.nan, dtype=np.float32)
            basin_gate_2d = np.ones((H, W), dtype=np.float32)

    # Strictly mask all 2D maps to the polygonal boundary if scanned_boundary.geojson is present
    poly_boundary_file = out_dir / "scanned_boundary.geojson"
    if poly_boundary_file.exists():
        try:
            from src.features.polygon_rasterizer import compute_polygon_mask_2d
            inside_poly_2d = compute_polygon_mask_2d(
                poly_boundary_file, height=H, width=W, transform=profile["transform"]
            )
            outside_poly_2d = ~inside_poly_2d
            for arr2d in (
                ex_2d,
                cal_metals_2d, lower_metals_2d, upper_metals_2d, fused_metals_2d, err_metals_2d,
                cal_coal_2d, lower_coal_2d, upper_coal_2d, fused_coal_2d, err_coal_2d, expected_cv_coal_2d,
            ):
                if isinstance(arr2d, np.ndarray) and arr2d.shape == (H, W):
                    arr2d[outside_poly_2d] = np.nan
            if run_metals and metal_bands_2d:
                for b_arr in metal_bands_2d:
                    if isinstance(b_arr, np.ndarray) and b_arr.shape == (H, W):
                        b_arr[outside_poly_2d] = np.nan
            if run_coal and locals().get("coal_bands_2d"):
                for b_arr in coal_bands_2d:
                    if isinstance(b_arr, np.ndarray) and b_arr.shape == (H, W):
                        b_arr[outside_poly_2d] = np.nan
            logger.info("[%s] Applied strict polygonal mask (%d interior cells on %dx%d grid).", region_name, int(inside_poly_2d.sum()), H, W)
        except Exception as poly_exc:
            logger.warning("[%s] Could not apply polygonal boundary mask: %s", region_name, poly_exc)

    # Save Exploration Propensity Map
    _write_singleband_cog(
        ex_2d, profile, maps_dir / "propensity_e_x.tif",
        description="Exploration Logistics Propensity e(x)",
    )

    # Save Metals GeoTIFFs (Domain-Suffixed and Canonical)
    if run_metals:
        if min(H, W) <= 500:
            cal_metals_2d = _apply_border_taper(cal_metals_2d)
            fused_metals_2d = _apply_border_taper(fused_metals_2d)
            lower_metals_2d = _apply_border_taper(lower_metals_2d)
            upper_metals_2d = _apply_border_taper(upper_metals_2d)

        _write_singleband_cog(
            cal_metals_2d, profile, maps_dir / "calibrated_prospectivity_metals.tif",
            description="Calibrated Metals Prospectivity P(Deposit|x)",
        )
        _write_singleband_cog(
            fused_metals_2d, profile, maps_dir / "fused_raw_score_metals.tif",
            description="Fused Raw Ensemble Metals Prospectivity",
        )
        _write_conformal_bounds_cog(
            lower_metals_2d, upper_metals_2d, profile, maps_dir / "conformal_bounds_90ci_metals.tif",
        )
        _write_singleband_cog(
            err_metals_2d, profile, maps_dir / "dae_reconstruction_error_metals.tif",
            description="Metals DAE Reconstruction Error",
        )
        metal_descs = list(K4_HEAD_NAMES) + ["metal_composite"]
        _save_multiband_geotiff(metal_bands_2d, metal_descs, profile, maps_dir / "prospectivity_k4_metals.tif")
        if data_dir.resolve() == Path("data/kalimantan/processed").resolve():
            _save_multiband_geotiff(metal_bands_2d, metal_descs, profile, data_dir / "prospectivity_k4_metals.tif")

        # Canonical files for metals
        _write_singleband_cog(
            cal_metals_2d, profile, maps_dir / "calibrated_prospectivity.tif",
            description="Calibrated Prospectivity P(Deposit|x)",
        )
        _write_singleband_cog(
            fused_metals_2d, profile, maps_dir / "fused_raw_score.tif",
            description="Fused Raw Ensemble Prospectivity",
        )
        _write_conformal_bounds_cog(
            lower_metals_2d, upper_metals_2d, profile, maps_dir / "conformal_bounds_90ci.tif",
        )
        _write_singleband_cog(
            err_metals_2d, profile, maps_dir / "dae_reconstruction_error.tif",
            description="DAE Reconstruction Error",
        )

    # Save Coal GeoTIFFs (Domain-Suffixed and Canonical)
    if run_coal:
        _write_singleband_cog(
            cal_coal_2d, profile, maps_dir / "calibrated_prospectivity_coal.tif",
            description="Calibrated Coal Occurrence Probability P(Coal|x)",
        )
        _write_singleband_cog(
            fused_coal_2d, profile, maps_dir / "fused_raw_score_coal.tif",
            description="Fused Raw Ensemble Coal Prospectivity",
        )
        _write_conformal_bounds_cog(
            lower_coal_2d, upper_coal_2d, profile, maps_dir / "conformal_bounds_90ci_coal.tif",
        )
        _write_singleband_cog(
            err_coal_2d, profile, maps_dir / "dae_reconstruction_error_coal.tif",
            description="Coal DAE Reconstruction Error",
        )
        coal_descs = list(K3_COAL_HEAD_NAMES) + ["p_coal_any", "expected_calorific_class"]
        _save_multiband_geotiff(coal_bands_2d, coal_descs, profile, maps_dir / "prospectivity_k3_coal.tif")
        if data_dir.resolve() == Path("data/kalimantan/processed").resolve():
            _save_multiband_geotiff(coal_bands_2d, coal_descs, profile, data_dir / "prospectivity_k3_coal.tif")

        # If running exclusively coal, coal provides canonical files
        if not run_metals:
            _write_singleband_cog(
                cal_coal_2d, profile, maps_dir / "calibrated_prospectivity.tif",
                description="Calibrated Coal Prospectivity P(Coal|x)",
            )
            _write_singleband_cog(
                fused_coal_2d, profile, maps_dir / "fused_raw_score.tif",
                description="Fused Raw Ensemble Coal Prospectivity",
            )
            _write_conformal_bounds_cog(
                lower_coal_2d, upper_coal_2d, profile, maps_dir / "conformal_bounds_90ci.tif",
            )
            _write_singleband_cog(
                err_coal_2d, profile, maps_dir / "dae_reconstruction_error.tif",
                description="Coal DAE Reconstruction Error",
            )

    # Resolve physical layers for target ranking (Bug 7: fallback to GeoTIFF or exact df columns)
    phys = resolve_physical_layers(stack) if stack is not None else {}
    ref_grid = cal_metals_2d if run_metals else cal_coal_2d

    def _norm_01(arr: np.ndarray) -> np.ndarray:
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return np.zeros_like(arr, dtype=np.float32)
        p2, p98 = np.percentile(valid, [2.0, 98.0])
        if p98 - p2 < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - p2) / (p98 - p2), 0.0, 1.0).astype(np.float32)

    def _fallback_2d_layer(col_name: str, default_val: float = 0.0) -> Optional[np.ndarray]:
        tif_p = data_dir / f"{col_name}.tif"
        if tif_p.exists():
            try:
                with rasterio.open(tif_p) as src:
                    arr = src.read(1, out_shape=(H, W)).astype(np.float32)
                    return np.where(np.isfinite(arr) & (arr > -9000.0), arr, default_val)
            except Exception:
                pass
        if col_name in df.columns:
            return reconstruct_map(df[col_name].to_numpy(dtype=np.float32), rows, cols, H, W, nodata=default_val)
        return None

    td_2d = phys.get("dilation_tendency_Td")
    if td_2d is None:
        td_raw = _fallback_2d_layer("dilation_tendency_Td", 0.0)
        td_2d = _norm_01(td_raw) if td_raw is not None else np.zeros_like(ref_grid)
    alt_2d = phys.get("alteration_index")
    if alt_2d is None:
        alt_raw = _fallback_2d_layer("spec_clay_hydroxyl_gated", 0.0)
        alt_2d = _norm_01(alt_raw) if alt_raw is not None else np.zeros_like(ref_grid)
    cover_2d = phys.get("regolith_cover_depth_m")
    if cover_2d is None:
        cover_raw = _fallback_2d_layer("regolith_cover_m", 0.0)
        if cover_raw is not None:
            cover_2d = cover_raw
        else:
            cover_2d = np.clip((1.0 - np.nan_to_num(ex_2d, nan=0.5)) * 65.0, 2.0, 120.0)
    slope_2d = phys.get("dem_slope")
    if slope_2d is None:
        slope_2d = _fallback_2d_layer("dem_slope", 0.0)

    wx_metals_2d = reconstruct_map(sample_weights_k4[:, 0], rows, cols, H, W, nodata=np.nan)
    wx_coal_2d = reconstruct_map(coal_tonnage_weights, rows, cols, H, W, nodata=np.nan)

    # Pass 6A & Pass 7A: Polygonization, Multi-Criteria Ranking & LLM Site Cards (Metals)
    if run_metals:
        if scan_only or abs(float(l_threshold) - 0.50) > 1e-4:
            eff_l_thr = float(np.clip(float(l_threshold), 0.15, 0.95))
        else:
            eff_l_thr = float(np.clip(np.nanpercentile(lower_metals_2d, 96), 0.35, 0.88))
        eff_closing = 2
        eff_min_area = float(min_area_ha)

        ranked_gdf = polygonize_and_rank_targets(
            calibrated_prob=cal_metals_2d,
            lower_bound_L=lower_metals_2d,
            upper_bound_U=upper_metals_2d,
            dilation_tendency_Td=td_2d,
            alteration_index=alt_2d,
            cover_depth_m=cover_2d,
            profile=profile,
            l_threshold=eff_l_thr,
            closing_size=eff_closing,
            min_area_ha=eff_min_area,
            region_prefix=region_name.upper(),
            domain="metals",
            extra_layers={
                "fused_raw_score": fused_metals_2d,
                "exploration_propensity_e_x": ex_2d,
                "propensity_weight_w": wx_metals_2d,
            },
        )
        if len(ranked_gdf) == 0 and np.isfinite(cal_metals_2d).any():
            fallback_thr = float(np.nanpercentile(cal_metals_2d, 96))
            eff_l_thr = fallback_thr
            ranked_gdf = polygonize_and_rank_targets(
                calibrated_prob=cal_metals_2d,
                lower_bound_L=cal_metals_2d,
                upper_bound_U=upper_metals_2d,
                dilation_tendency_Td=td_2d,
                alteration_index=alt_2d,
                cover_depth_m=cover_2d,
                profile=profile,
                l_threshold=fallback_thr,
                closing_size=eff_closing,
                min_area_ha=eff_min_area,
                region_prefix=region_name.upper(),
                domain="metals",
                extra_layers={
                    "fused_raw_score": fused_metals_2d,
                    "exploration_propensity_e_x": ex_2d,
                    "propensity_weight_w": wx_metals_2d,
                },
            )

        conformal_target_sets = []
        if len(ranked_gdf) > 0:
            target_classes = []
            for p_geom in ranked_gdf.geometry:
                c_proj = p_geom.centroid
                inv_tr = ~profile["transform"]
                col_f, row_f = inv_tr * (c_proj.x, c_proj.y)
                r_idx = int(np.clip(round(row_f), 0, H - 1))
                c_idx = int(np.clip(round(col_f), 0, W - 1))

                cell_head_scores = [metal_bands_2d[k][r_idx, c_idx] for k in range(4)]
                if not any(np.isfinite(cell_head_scores)):
                    cell_head_scores = [0.0, 0.0, 0.0, 0.0]
                dom_head_idx = int(np.nanargmax(cell_head_scores))
                target_classes.append(K4_HEAD_NAMES[dom_head_idx])

                active_heads_cell = [
                    K4_HEAD_NAMES[k] for k in range(4)
                    if np.isfinite(cell_head_scores[k]) and cell_head_scores[k] >= eff_l_thr
                ]
                conformal_target_sets.append(active_heads_cell or [K4_HEAD_NAMES[dom_head_idx]])

            ranked_gdf["target_class"] = target_classes

        # v4.8: greenfield / brownfield lists (top 15 of each) instead of the overall top 15,
        # so targets on known deposits cannot crowd new ground out of the list.
        known_metals_2d = np.zeros((H, W), dtype=bool)
        _pm = np.any(y_k4 == 1, axis=1)
        known_metals_2d[rows[_pm], cols[_pm]] = True
        ranked_gdf = split_targets_by_known_ground(
            ranked_gdf, known_metals_2d, profile, greenfield_km=GREENFIELD_KM, top_n_each=15)
        if conformal_target_sets:
            conformal_target_sets = [conformal_target_sets[i] for i in ranked_gdf["_source_pos"]]
        ranked_gdf = ranked_gdf.drop(columns="_source_pos")

        if len(ranked_gdf) > 0:
            ranked_gdf.to_file(targets_dir / f"ranked_targets_{region_name.lower()}_metals.gpkg", driver="GPKG")
            ranked_gdf.to_file(targets_dir / f"ranked_targets_{region_name.lower()}.gpkg", driver="GPKG")
            for _cls in ("greenfield", "brownfield"):
                _sub = ranked_gdf[ranked_gdf["exploration_class"] == _cls]
                if len(_sub) > 0:
                    _sub.to_file(targets_dir / f"ranked_targets_{region_name.lower()}_metals_{_cls}.gpkg", driver="GPKG")

        _emit_telemetry(
            phase="pass7",
            pass_idx=7,
            round_idx=round_idx,
            total_rounds=total_rounds,
            progress_pct=min(96.0, ((eff_round - 1 + 0.90) / eff_total_rounds) * 100.0),
            metrics={"n_targets": int(len(ranked_gdf))},
        )
        cards_metals_path = targets_dir / "site_cards_metals.json"
        cards = emit_site_cards_json(
            ranked_gdf,
            out_path=str(cards_metals_path),
            elkan_noto_c_hat=float(np.mean(c_hat_k4)),
            sensor_source="EMIT_SWIR",
            window_footprint_km=3.2,
            model=k4_ensemble,
            feature_names=feature_cols,
            feature_matrix=X_all,
            rows=rows,
            cols=cols,
            profile=profile,
            tau_drill=eff_l_thr,
            conformal_target_sets=conformal_target_sets,
            outmogc_gates=outmogc_gates,
        )
        emit_site_cards_json(
            ranked_gdf,
            out_path=str(targets_dir / "site_cards.json"),
            elkan_noto_c_hat=float(np.mean(c_hat_k4)),
            sensor_source="EMIT_SWIR",
            window_footprint_km=3.2,
            model=k4_ensemble,
            feature_names=feature_cols,
            feature_matrix=X_all,
            rows=rows,
            cols=cols,
            profile=profile,
            tau_drill=eff_l_thr,
            conformal_target_sets=conformal_target_sets,
            outmogc_gates=outmogc_gates,
        )

        # True out-of-fold 5-fold spatial block CV metrics for metals (Leakage-Free)
        cv_rows = []
        oof_y_true = []
        oof_y_pred = []

        if not scan_only:
            for f_idx, (tr_f_idx, va_f_idx) in enumerate(folds):
                va_f_pos = pos_mask_metals[va_f_idx]
                p_va_f = va_f_idx[va_f_pos]
                u_va_f = va_f_idx[~va_f_pos]
                if len(p_va_f) >= 1 and pos_mask_metals[tr_f_idx].sum() >= 1 and len(u_va_f) >= 5:
                    fold_ens = TwoStageK4Ensemble(
                        n_estimators=min(30, eff_n_estimators),
                        n_bag=2,
                        random_state=42 + f_idx,
                        learning_rate=eff_lr,
                        max_depth=eff_max_depth,
                        head0_stacking=bool(gen["head0_stacking"]),
                        min_child_weight=(gen["min_child_weight_metals"] or None),
                        exclude_features=tuple(gen["drop_features_metals"] or ()),
                    )
                    fold_ens.fit(
                        X=X_all[tr_f_idx],
                        y_k4=y_k4[tr_f_idx],
                        sample_weights_k4=sample_weights_k4[tr_f_idx],
                        e_logistics=e_x[tr_f_idx],
                        exclusion_masks=exclusion_masks[tr_f_idx],
                        spatial_block_ids=block_ids[tr_f_idx],
                        feature_names=feature_cols,
                    )
                    sub_u = u_va_f
                    ev_f = np.concatenate([p_va_f, sub_u])
                    y_f = np.concatenate([np.ones(len(p_va_f), dtype=int), np.zeros(len(sub_u), dtype=int)])
                    fold_probs = np.max(fold_ens.predict_proba(X_all[ev_f]), axis=1)
                    m_f = score_metrics(y_f, fold_probs)
                    m_f["fold"] = f_idx
                    m_f["domain"] = "metals"
                    m_f["n_val_pos"] = len(p_va_f)
                    m_f["n_val_neg"] = len(sub_u)
                    cv_rows.append(m_f)
                    oof_y_true.extend(y_f.tolist())
                    oof_y_pred.extend(fold_probs.tolist())

        if not cv_rows:
            va_pos = pos_mask_metals[val_idx]
            p_va = val_idx[va_pos]
            u_va = val_idx[~va_pos]
            if len(p_va) >= 1:
                ev_va = np.concatenate([p_va, u_va])
                y_va = np.concatenate([np.ones(len(p_va), dtype=int), np.zeros(len(u_va), dtype=int)])
                val_probs = np.max(fused_k4_full[ev_va], axis=1)
                m_f = score_metrics(y_va, val_probs)
                m_f["fold"] = 0
                m_f["domain"] = "metals"
                m_f["n_val_pos"] = len(p_va)
                m_f["n_val_neg"] = len(u_va)
                cv_rows.append(m_f)
                oof_y_true = y_va.tolist()
                oof_y_pred = val_probs.tolist()

        if cv_rows:
            pd.DataFrame(cv_rows).to_csv(out_dir / "spatial_cv_metrics_metals.csv", index=False)
        if oof_y_true and oof_y_pred:
            oof_metrics = score_metrics(np.array(oof_y_true), np.array(oof_y_pred))
            (out_dir / "spatial_oof_metrics_metals.json").write_text(json.dumps(oof_metrics, indent=2) + "\n")
            if not run_coal:
                (out_dir / "spatial_oof_metrics.json").write_text(json.dumps(oof_metrics, indent=2) + "\n")
                (out_dir / "blind_holdout.json").write_text(json.dumps(oof_metrics, indent=2) + "\n")

    # Pass 6B & Pass 7B: Polygonization, Multi-Criteria Ranking & LLM Site Cards (Coal)
    if run_coal:
        if scan_only or abs(float(l_threshold) - 0.50) > 1e-4:
            eff_l_thr_coal = float(np.clip(float(l_threshold), 0.15, 0.95))
        else:
            eff_l_thr_coal = float(np.clip(np.nanpercentile(lower_coal_2d, 96), 0.35, 0.88))
        eff_closing = 2
        eff_min_area = float(min_area_ha)

        ranked_gdf_coal = polygonize_and_rank_targets(
            calibrated_prob=cal_coal_2d,
            lower_bound_L=lower_coal_2d,
            upper_bound_U=upper_coal_2d,
            dilation_tendency_Td=td_2d,
            alteration_index=alt_2d,
            cover_depth_m=cover_2d,
            profile=profile,
            l_threshold=eff_l_thr_coal,
            closing_size=eff_closing,
            min_area_ha=eff_min_area,
            region_prefix=f"{region_name.upper()}_COAL",
            domain="coal",
            extra_layers={
                "fused_raw_score": fused_coal_2d,
                "exploration_propensity_e_x": ex_2d,
                "propensity_weight_w": wx_coal_2d,
            },
        )
        if len(ranked_gdf_coal) == 0 and np.isfinite(cal_coal_2d).any():
            fallback_thr_coal = float(np.nanpercentile(cal_coal_2d, 96))
            eff_l_thr_coal = fallback_thr_coal
            ranked_gdf_coal = polygonize_and_rank_targets(
                calibrated_prob=cal_coal_2d,
                lower_bound_L=cal_coal_2d,
                upper_bound_U=upper_coal_2d,
                dilation_tendency_Td=td_2d,
                alteration_index=alt_2d,
                cover_depth_m=cover_2d,
                profile=profile,
                l_threshold=fallback_thr_coal,
                closing_size=eff_closing,
                min_area_ha=eff_min_area,
                region_prefix=f"{region_name.upper()}_COAL",
                domain="coal",
                extra_layers={
                    "fused_raw_score": fused_coal_2d,
                    "exploration_propensity_e_x": ex_2d,
                    "propensity_weight_w": wx_coal_2d,
                },
            )

        conformal_target_sets_coal = []
        if len(ranked_gdf_coal) > 0:
            target_classes_coal = []
            for p_geom in ranked_gdf_coal.geometry:
                c_proj = p_geom.centroid
                inv_tr = ~profile["transform"]
                col_f, row_f = inv_tr * (c_proj.x, c_proj.y)
                r_idx = int(np.clip(round(row_f), 0, H - 1))
                c_idx = int(np.clip(round(col_f), 0, W - 1))

                cell_coal_scores = [coal_bands_2d[c][r_idx, c_idx] for c in range(3)]
                if not any(np.isfinite(cell_coal_scores)):
                    cell_coal_scores = [0.0, 0.0, 0.0]
                dom_coal_idx = int(np.nanargmax(cell_coal_scores))
                target_classes_coal.append(K3_COAL_HEAD_NAMES[dom_coal_idx])

                active_coal_heads = [
                    K3_COAL_HEAD_NAMES[c] for c in range(3)
                    if np.isfinite(cell_coal_scores[c]) and cell_coal_scores[c] >= eff_l_thr_coal
                ]
                conformal_target_sets_coal.append(active_coal_heads or [K3_COAL_HEAD_NAMES[dom_coal_idx]])

            ranked_gdf_coal["target_class"] = target_classes_coal

        known_coal_2d = np.zeros((H, W), dtype=bool)
        _pc = pos_mask_coal | (
            (df["label_coal"].to_numpy() == 1) if "label_coal" in df.columns else np.zeros(len(df), dtype=bool))
        known_coal_2d[rows[_pc], cols[_pc]] = True
        ranked_gdf_coal = split_targets_by_known_ground(
            ranked_gdf_coal, known_coal_2d, profile, greenfield_km=GREENFIELD_KM, top_n_each=15)
        if conformal_target_sets_coal:
            conformal_target_sets_coal = [conformal_target_sets_coal[i] for i in ranked_gdf_coal["_source_pos"]]
        ranked_gdf_coal = ranked_gdf_coal.drop(columns="_source_pos")

        if len(ranked_gdf_coal) > 0:
            ranked_gdf_coal.to_file(targets_dir / f"ranked_targets_{region_name.lower()}_coal.gpkg", driver="GPKG")
            for _cls in ("greenfield", "brownfield"):
                _sub = ranked_gdf_coal[ranked_gdf_coal["exploration_class"] == _cls]
                if len(_sub) > 0:
                    _sub.to_file(targets_dir / f"ranked_targets_{region_name.lower()}_coal_{_cls}.gpkg", driver="GPKG")
            if not run_metals:
                ranked_gdf_coal.to_file(targets_dir / f"ranked_targets_{region_name.lower()}.gpkg", driver="GPKG")

        _emit_telemetry(
            phase="pass7",
            pass_idx=7,
            round_idx=round_idx,
            total_rounds=total_rounds,
            progress_pct=min(98.0, ((eff_round - 1 + 0.95) / eff_total_rounds) * 100.0),
            metrics={"n_targets": int(len(ranked_gdf) if len(ranked_gdf) > 0 else len(ranked_gdf_coal))},
        )
        cards_coal_path = targets_dir / "site_cards_coal.json"
        cards_coal = emit_site_cards_json(
            ranked_gdf_coal,
            out_path=str(cards_coal_path),
            elkan_noto_c_hat=float(np.mean(c_hat_k3)),
            sensor_source="SENTINEL2_SWIR",
            window_footprint_km=3.2,
            model=k3_coal_ensemble,
            feature_names=feature_cols,
            feature_matrix=X_all,
            rows=rows,
            cols=cols,
            profile=profile,
            tau_drill=eff_l_thr_coal,
            conformal_target_sets=conformal_target_sets_coal,
            outmogc_gates=None,
            basin_gate=basin_gate_2d,
        )
        if not run_metals:
            emit_site_cards_json(
                ranked_gdf_coal,
                out_path=str(targets_dir / "site_cards.json"),
                elkan_noto_c_hat=float(np.mean(c_hat_k3)),
                sensor_source="SENTINEL2_SWIR",
                window_footprint_km=3.2,
                model=k3_coal_ensemble,
                feature_names=feature_cols,
                feature_matrix=X_all,
                rows=rows,
                cols=cols,
                profile=profile,
                tau_drill=eff_l_thr_coal,
                conformal_target_sets=conformal_target_sets_coal,
                outmogc_gates=None,
                basin_gate=basin_gate_2d,
            )

        # True out-of-fold 5-fold spatial block CV metrics for coal (Leakage-Free)
        cv_rows_coal = []
        oof_y_true_coal = []
        oof_y_pred_coal = []

        if not scan_only:
            for f_idx, (tr_f_idx, va_f_idx) in enumerate(folds):
                va_f_pos = pos_mask_coal[va_f_idx]
                p_va_f = va_f_idx[va_f_pos]
                u_va_f = va_f_idx[~va_f_pos]
                if len(p_va_f) >= 1 and pos_mask_coal[tr_f_idx].sum() >= 1 and len(u_va_f) >= 5:
                    fold_coal_ens = TwoStageK3CoalEnsemble(
                        n_estimators=min(30, eff_n_estimators),
                        n_bag=2,
                        random_state=42 + f_idx,
                        learning_rate=eff_lr,
                        max_depth=max(3, eff_max_depth - 1),
                        unlabeled_weighting="neutral" if gen["weight_scheme"] == "hajek_neutral" else "inverse_propensity",
                    )
                    fold_coal_ens.fit(
                        X=X_all[tr_f_idx],
                        y_k3=y_k3_coal[tr_f_idx],
                        tonnage_weights=coal_tonnage_weights[tr_f_idx],
                        e_pooled=e_x[tr_f_idx],
                        w_canopy=w_canopy_arr[tr_f_idx],
                        spatial_block_ids=block_ids[tr_f_idx],
                        feature_names=feature_cols,
                        negative_exclusion_mask=coal_disk_neg_excl[tr_f_idx],
                    )
                    sub_u = u_va_f
                    ev_f = np.concatenate([p_va_f, sub_u])
                    y_f = np.concatenate([np.ones(len(p_va_f), dtype=int), np.zeros(len(sub_u), dtype=int)])
                    fold_probs = fold_coal_ens.predict_composite_proba(X_all[ev_f], w_canopy=w_canopy_arr[ev_f])
                    m_f = score_metrics(y_f, fold_probs)
                    m_f["fold"] = f_idx
                    m_f["domain"] = "coal"
                    m_f["n_val_pos"] = len(p_va_f)
                    m_f["n_val_neg"] = len(sub_u)
                    cv_rows_coal.append(m_f)
                    oof_y_true_coal.extend(y_f.tolist())
                    oof_y_pred_coal.extend(fold_probs.tolist())

        if not cv_rows_coal:
            va_pos = pos_mask_coal[val_coal_idx]
            p_va = val_coal_idx[va_pos]
            u_va = val_coal_idx[~va_pos]
            if len(p_va) >= 1:
                ev_va = np.concatenate([p_va, u_va])
                y_va = np.concatenate([np.ones(len(p_va), dtype=int), np.zeros(len(u_va), dtype=int)])
                val_probs = p_coal_any_full[ev_va]
                m_f = score_metrics(y_va, val_probs)
                m_f["fold"] = 0
                m_f["domain"] = "coal"
                m_f["n_val_pos"] = len(p_va)
                m_f["n_val_neg"] = len(u_va)
                cv_rows_coal.append(m_f)
                oof_y_true_coal = y_va.tolist()
                oof_y_pred_coal = val_probs.tolist()

        if cv_rows_coal:
            pd.DataFrame(cv_rows_coal).to_csv(out_dir / "spatial_cv_metrics_coal.csv", index=False)
        if oof_y_true_coal and oof_y_pred_coal:
            oof_metrics_coal = score_metrics(np.array(oof_y_true_coal), np.array(oof_y_pred_coal))
            (out_dir / "spatial_oof_metrics_coal.json").write_text(json.dumps(oof_metrics_coal, indent=2) + "\n")
            if not run_metals:
                (out_dir / "spatial_oof_metrics.json").write_text(json.dumps(oof_metrics_coal, indent=2) + "\n")
                (out_dir / "blind_holdout.json").write_text(json.dumps(oof_metrics_coal, indent=2) + "\n")

        active_coal_xgb = [x for x in pr_auc_xgb_coal if np.isfinite(x)]
        active_coal_unet = [x for x in pr_auc_unet_coal if np.isfinite(x)]
        active_coal_fused = [x for x in pr_auc_fused_coal if np.isfinite(x)]

        coal_summary = {
            "n_coal_positives_total": int(pos_mask_coal.sum()),
            "n_coal_positives_per_head": {
                K3_COAL_HEAD_NAMES[c]: int((y_k3_coal[:, c] == 1).sum()) for c in range(3)
            },
            "pr_auc_coal_xgb": round(float(np.nanmean(active_coal_xgb)), 4) if active_coal_xgb else None,
            "pr_auc_coal_unet": round(float(np.nanmean(active_coal_unet)), 4) if active_coal_unet else None,
            "pr_auc_coal_fused": round(float(np.nanmean(active_coal_fused)), 4) if active_coal_fused else None,
            "pr_auc_coal_heads": {
                K3_COAL_HEAD_NAMES[c]: (
                    round(float(pr_auc_fused_coal[c]), 4) if np.isfinite(pr_auc_fused_coal[c]) else None
                )
                for c in range(3)
            },
            "pr_auc_coal_xgb_heads": {
                K3_COAL_HEAD_NAMES[c]: (
                    round(float(pr_auc_xgb_coal[c]), 4) if np.isfinite(pr_auc_xgb_coal[c]) else None
                )
                for c in range(3)
            },
            "elkan_noto_c_hat_coal_per_head": [round(float(x), 4) for x in c_hat_k3],
            "conformal_q_hat_90_coal_per_head": [round(float(x), 4) for x in q_hat_k3_list],
            "p_coal_any_mean": round(float(np.nanmean(cal_coal_2d)), 4),
            "p_coal_any_max": round(float(np.nanmax(cal_coal_2d)), 4),
            "prospectivity_k3_coal_tif": str(maps_dir / "prospectivity_k3_coal.tif"),
        }

    # Canonical spatial_cv_metrics.csv resolution
    all_cv_records = []
    if run_metals and cv_rows:
        all_cv_records.extend(cv_rows)
    if run_coal and cv_rows_coal:
        all_cv_records.extend(cv_rows_coal)
    if all_cv_records:
        pd.DataFrame(all_cv_records).to_csv(out_dir / "spatial_cv_metrics.csv", index=False)

    active_xgb = [x for x in pr_auc_xgb_heads if np.isfinite(x)]
    active_unet = [x for x in pr_auc_unet_heads if np.isfinite(x)]
    active_fused = [x for x in pr_auc_fused_heads if np.isfinite(x)]

    summary = {
        "region": region_name,
        "domain": domain_clean,
        "n_cells": len(df),
        "n_positives": int(pos_mask_any.sum()),
        "n_metal_positives": int(pos_mask_metals.sum()),
        "n_coal_positives": int(pos_mask_coal.sum()),
        "pr_auc_xgb": round(float(np.nanmean(active_xgb)), 4) if active_xgb else coal_summary.get("pr_auc_coal_xgb"),
        "pr_auc_unet": round(float(np.nanmean(active_unet)), 4) if active_unet else coal_summary.get("pr_auc_coal_unet"),
        "pr_auc_fused": round(float(np.nanmean(active_fused)), 4) if active_fused else coal_summary.get("pr_auc_coal_fused"),
        "pr_auc_heads": {
            K4_HEAD_NAMES[k]: (
                round(float(pr_auc_fused_heads[k]), 4)
                if np.isfinite(pr_auc_fused_heads[k])
                else None
            )
            for k in range(4)
        },
        "pr_auc_xgb_heads": {
            K4_HEAD_NAMES[k]: (
                round(float(pr_auc_xgb_heads[k]), 4)
                if np.isfinite(pr_auc_xgb_heads[k])
                else None
            )
            for k in range(4)
        },
        "coal_k3_metrics": coal_summary if coal_summary else None,
        "w_xgb": round(float(np.nanmean(w_xgb_k4)) if (run_metals and w_xgb_k4 and len(w_xgb_k4) > 0) else (float(np.nanmean(w_xgb_coal)) if (run_coal and w_xgb_coal and len(w_xgb_coal) > 0) else 0.70), 4),
        "w_unet": round(float(np.nanmean(w_unet_k4)) if (run_metals and w_unet_k4 and len(w_unet_k4) > 0) else (float(np.nanmean(w_unet_coal)) if (run_coal and w_unet_coal and len(w_unet_coal) > 0) else 0.30), 4),
        "elkan_noto_c_hat": round(float(np.nanmean(c_hat_k4)), 4) if (c_hat_k4 is not None and len(c_hat_k4) > 0) else (round(float(np.nanmean(c_hat_k3)), 4) if (c_hat_k3 is not None and len(c_hat_k3) > 0) else 0.5),
        "elkan_noto_c_hat_mean": round(float(np.nanmean(c_hat_k4)), 4) if (c_hat_k4 is not None and len(c_hat_k4) > 0) else (round(float(np.nanmean(c_hat_k3)), 4) if (c_hat_k3 is not None and len(c_hat_k3) > 0) else 0.5),
        "elkan_noto_c_hat_per_head": [round(float(x), 4) for x in c_hat_k4] if (c_hat_k4 is not None and len(c_hat_k4) > 0) else None,
        "conformal_q_hat_90": round(float(np.nanmean(q_hat_k4_list)), 4) if (q_hat_k4_list is not None and len(q_hat_k4_list) > 0) else (round(float(np.nanmean(q_hat_k3_list)), 4) if (q_hat_k3_list is not None and len(q_hat_k3_list) > 0) else 0.5),
        "conformal_q_hat_90_per_head": [round(float(x), 4) for x in q_hat_k4_list] if (q_hat_k4_list is not None and len(q_hat_k4_list) > 0) else None,
        "rejected_warm_start_heads": getattr(k4_ensemble, "rejected_heads_", {}) if k4_ensemble else {},
        "propensity_min": round(float(e_x.min()), 4),
        "propensity_median": round(float(np.median(e_x)), 4),
        "n_ranked_targets": len(ranked_gdf) if len(ranked_gdf) > 0 else (len(ranked_gdf_coal) if len(ranked_gdf_coal) > 0 else 0),
        "top_target_card": cards[0] if cards else (cards_coal[0] if cards_coal else None),
        "training_mode": (
            "scan_only_inference" if scan_only else (
                f"{'10x_compute_' if train_10x else ''}"
                f"{'incremental_warm_start' if prior_ensemble is not None else ('learned_from_zero' if reset_from_zero else 'standard')}"
            )
        ),
        "is_warm_started": bool(prior_ensemble is not None or prior_coal_ensemble is not None),
        "scan_only": bool(scan_only),
        "warm_start_schema": {
            "policy": on_schema_mismatch,
            "reason": schema_decision["reason"],
            "dropped_channels": list(schema_decision["dropped"]),
            "missing_channels": list(schema_decision["missing"]),
        },
        "n_feature_channels": len(feature_cols),
        "sar_channels_used": [c for c in feature_cols if c.startswith("sar_")],
        "base_model_dir": str(eff_base_models_dir) if (prior_ensemble is not None or prior_coal_ensemble is not None) else None,
        "round_idx": round_idx,
        "total_rounds": total_rounds,
        "n_estimators": eff_n_estimators,
        "n_bag": eff_n_bag,
        "dae_epochs": eff_dae_epochs,
        "learning_rate": round(eff_lr, 4),
        "max_depth": eff_max_depth,
        "l_threshold": round(float(l_threshold), 4),
        "min_area_ha": round(float(min_area_ha), 2),
        "block_size_m": round(float(block_size_m), 1),
        "resolution_m": round(float(px_size_m), 1),
    }
    (out_dir / "seven_pass_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _emit_telemetry(
        phase="complete" if is_final_round else "pass7",
        pass_idx=7,
        round_idx=round_idx,
        total_rounds=total_rounds,
        progress_pct=100.0 if is_final_round else min(99.0, (eff_round / eff_total_rounds) * 100.0),
        metrics={
            "pr_auc_xgb": summary.get("pr_auc_xgb"),
            "pr_auc_unet": summary.get("pr_auc_unet"),
            "pr_auc_fused": summary.get("pr_auc_fused"),
            "c_hat": summary.get("elkan_noto_c_hat"),
            "n_targets": summary.get("n_ranked_targets", 0),
        },
    )
    if total_rounds and total_rounds > 1:
        history_path = out_dir / "ten_rounds_history.json"
        existing_history = []
        if eff_round > 1 and history_path.exists():
            try:
                existing_history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                existing_history = []
        existing_history.append({
            "round": eff_round,
            "cumulative_trees": eff_round * eff_n_estimators,
            "pr_auc_xgb": summary.get("pr_auc_xgb"),
            "pr_auc_unet": summary.get("pr_auc_unet"),
            "pr_auc_fused": summary.get("pr_auc_fused"),
            "pr_auc_heads": summary.get("pr_auc_heads", {}),
            "coal_pr_auc_fused": coal_summary.get("pr_auc_coal_fused") if (run_coal and coal_summary) else None,
            "elkan_noto_c_hat": summary.get("elkan_noto_c_hat"),
            "n_ranked_targets": summary.get("n_ranked_targets", 0),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        history_path.write_text(json.dumps(existing_history, indent=2) + "\n", encoding="utf-8")
    if is_final_round:
        try:
            register_completed_run(out_dir, region_name, summary, run_label=run_label)
        except Exception as e:
            logger.warning("Could not register completed run in catalog: %s", e)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalimantan Dual-Domain (K=4 Metals + K=3 Coal) 7-Pass Training Engine")
    parser.add_argument(
        "--domain",
        type=str,
        choices=["metals", "coal", "both"],
        default="both",
        help="Domain branch to execute: 'metals' (K=4), 'coal' (K=3), or 'both' (default: both)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="KALIMANTAN",
        help="Region to train (must be strictly in Kalimantan)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional override path for processed dataset directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional override path for output directory",
    )
    parser.add_argument(
        "--base-model-dir",
        type=str,
        default=None,
        help="Optional path to an existing models/ checkpoint directory for warm-start or scan-only inference",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Enable continuous incremental weight updates from previous checkpoints (default: True)",
    )
    parser.add_argument(
        "--reset-from-zero",
        "--reset",
        dest="reset_from_zero",
        action="store_true",
        default=False,
        help="Reset models and clear cached checkpoints to learn from zero (default: False)",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        default=False,
        help="Run inference-only scan (Pass 1 + Pass 4-7) using existing checkpoints without retraining",
    )
    parser.add_argument(
        "--on-schema-mismatch",
        type=str,
        choices=list(WARM_START_SCHEMA_POLICIES),
        default="retrain",
        help=(
            "When a warm-start checkpoint's feature schema differs from the dataset (e.g. new SAR channels): "
            "'retrain' (default) discards the warm-start and trains on all dataset channels; "
            "'prior' keeps the checkpoint schema and ignores new channels (legacy)."
        ),
    )
    parser.add_argument(
        "--disable-unet",
        action="store_true",
        default=False,
        help="Disable spatial DAE U-Net pass for fast tabular-only training or scan",
    )
    parser.add_argument(
        "--train-10x",
        action="store_true",
        default=False,
        help="Run 10x compute training run (500 estimators x 10 bags XGBoost, 15 epochs DAE-U-Net)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of sequential incremental training rounds (default: 1)",
    )
    parser.add_argument(
        "--trees-per-round",
        type=int,
        default=50,
        help="Number of incremental XGBoost trees per round (default: 50)",
    )
    parser.add_argument(
        "--bags",
        type=int,
        default=3,
        help="Number of bagging ensembles per round (default: 3)",
    )
    parser.add_argument(
        "--dae-epochs",
        type=int,
        default=2,
        help="Number of DAE U-Net spatial epochs per round (default: 2)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="XGBoost learning rate eta (default: 0.05)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="XGBoost maximum tree depth (default: 6)",
    )
    parser.add_argument(
        "--l-threshold",
        type=float,
        default=0.50,
        help="Conformal lower bound L(x) drill target threshold (default: 0.50)",
    )
    parser.add_argument(
        "--min-area-ha",
        type=float,
        default=5.0,
        help="Minimum drill target polygon area in hectares (default: 5.0)",
    )
    parser.add_argument(
        "--block-size-m",
        type=float,
        default=None,
        help="Optional override for spatial CV block size in meters",
    )
    parser.add_argument(
        "--active-heads",
        type=str,
        default=None,
        help="Optional comma-separated list of active deposit heads",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Custom model run name or descriptive tag",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional training hypothesis notes",
    )
    gen_grp = parser.add_argument_group("v4.8 generalisation (defaults: GENERALISATION_DEFAULTS)")
    gen_grp.add_argument("--weight-scheme", choices=("production", "hajek_neutral"), default=None,
                         help="positive/unlabeled sample weighting for both domains")
    gen_grp.add_argument("--gate", dest="gate_mode", choices=("production", "standardized", "off"), default=None,
                         help="metals OUTMOGC envelope gate")
    gen_grp.add_argument("--head0-stacking", choices=("on", "off"), default=None,
                         help="porphyry head uses skarn/epithermal meta-features")
    gen_grp.add_argument("--min-child-weight-metals", type=float, default=None,
                         help="XGBoost min_child_weight for the metals heads")
    gen_grp.add_argument("--coal-disk-mode", choices=("production_keep", "exclude"), default=None,
                         help="whether ESDM 1 km coal disk cells may act as K3 negatives")
    gen_grp.add_argument("--coal-buffer-m", type=float, default=None,
                         help="unlabeled cells within this distance of training known coal are not negatives")
    gen_grp.add_argument("--drop-features", default=None, help="comma list of feature columns to drop (both domains)")
    gen_grp.add_argument("--drop-features-metals", default=None,
                         help="comma list of feature columns the metals heads must not use ('none' keeps all)")
    gen_grp.add_argument("--legacy-v47", action="store_true",
                         help="reproduce the v4.7 behaviour of every generalisation switch")
    gen_grp.add_argument("--k4-exclusion-scale", type=float, default=None,
                         help="scale the metals exclusion halo around training positives (0 = none)")
    args = parser.parse_args()
    generalisation = {
        "weight_scheme": args.weight_scheme,
        "gate_mode": args.gate_mode,
        "head0_stacking": None if args.head0_stacking is None else args.head0_stacking == "on",
        "min_child_weight_metals": args.min_child_weight_metals,
        "coal_disk_mode": args.coal_disk_mode,
        "coal_buffer_m": args.coal_buffer_m,
        "drop_features": None if args.drop_features is None else tuple(
            c.strip() for c in args.drop_features.split(",") if c.strip()),
        "k4_exclusion_scale": args.k4_exclusion_scale,
        "drop_features_metals": None if args.drop_features_metals is None else (
            () if args.drop_features_metals.strip().lower() == "none" else tuple(
                c.strip() for c in args.drop_features_metals.split(",") if c.strip())),
    }
    if args.legacy_v47:
        legacy = dict(GENERALISATION_LEGACY_V47)
        legacy.update({k: v for k, v in generalisation.items() if v is not None})
        generalisation = legacy

    reset_from_zero = args.reset_from_zero and not args.scan_only
    incremental = (args.incremental and not reset_from_zero) or args.scan_only
    total_rounds = 1 if args.scan_only else max(1, args.rounds)
    base_model_dir = Path(args.base_model_dir) if args.base_model_dir else None
    active_heads_list = [h.strip() for h in args.active_heads.split(",") if h.strip()] if args.active_heads else None

    reg_up = args.region.upper().strip()
    VALID_KALIMANTAN_REGIONS = {
        "KALIMANTAN", "BORNEO", "KALIMANTAN_50M", "KALIMANTAN_500M",
        "EAST_KALIMANTAN", "KALTIM", "KALIMANTAN_TIMUR",
        "HOTSPOT_SCHWANER", "HOTSPOT_KAYAN_NORTH", "HOTSPOT_KAHAYAN",
        "HOTSPOT_SCHWANER_50M", "HOTSPOT_KAYAN_50M", "ALL_KALIMANTAN",
        "KUTAI_BASIN", "KUTAI_BASIN_100M",
        "MAHAKAM_SUTURE", "SANGATTA_TREND", "KELIAN_CORRIDOR",
        "BERAU_FOLD_BELT", "ADANG_FAULT", "MALINAU_CORRIDOR", "CUSTOM_POLYGON",
    }
    if (
        reg_up not in VALID_KALIMANTAN_REGIONS
        and not reg_up.startswith("HOTSPOT_")
        and not reg_up.startswith("KUTAI_")
        and not reg_up.startswith("SCAN_")
        and not reg_up.startswith("POLY_")
        and args.data_dir is None
    ):
        raise ValueError(
            f"Strict Kalimantan-only policy enforced: Training outside Kalimantan is prohibited. "
            f"Requested region: '{args.region}'. Specialized exclusively for Kalimantan deposit mechanics."
        )

    summaries = []
    if args.data_dir is not None:
        custom_data = Path(args.data_dir)
        target_out_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/seven_pass_{reg_up.lower()}")
        eff_block_m = float(args.block_size_m) if args.block_size_m else 5_000.0
        for r in range(1, total_rounds + 1):
            is_reset = reset_from_zero and (r == 1)
            is_inc = incremental or (r > 1)
            s = run_region_seven_pass(
                region_name=reg_up if reg_up != "ALL_KALIMANTAN" else "KALIMANTAN",
                data_dir=custom_data,
                out_dir=target_out_dir,
                block_size_m=eff_block_m,
                l_threshold=args.l_threshold,
                min_area_ha=args.min_area_ha,
                reset_from_zero=is_reset,
                train_10x=args.train_10x,
                incremental=is_inc,
                n_estimators=args.trees_per_round,
                n_bag=args.bags,
                dae_epochs=args.dae_epochs,
                round_idx=r,
                total_rounds=total_rounds,
                run_label=args.run_label,
                domain=args.domain,
                base_model_dir=base_model_dir,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                disable_unet=args.disable_unet,
                active_heads=active_heads_list,
                scan_only=args.scan_only,
                on_schema_mismatch=args.on_schema_mismatch,
                generalisation=generalisation,
            )
            if r == total_rounds:
                summaries.append(s)
    else:
        if reg_up in ("EAST_KALIMANTAN", "KALTIM", "KALIMANTAN_TIMUR", "ALL_KALIMANTAN"):
            ek_data = Path("data/east_kalimantan/processed")
            if (ek_data / "dataset.parquet").exists():
                target_out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/seven_pass_east_kalimantan")
                eff_block_m = float(args.block_size_m) if args.block_size_m else 25_000.0
                for r in range(1, total_rounds + 1):
                    is_reset = reset_from_zero and (r == 1)
                    is_inc = incremental or (r > 1)
                    print(f"\n>>> [EAST KALIMANTAN // ROUND {r}/{total_rounds}] Initiating 7-Pass Training (domain={args.domain})...")
                    s = run_region_seven_pass(
                        region_name="EAST_KALIMANTAN",
                        data_dir=ek_data,
                        out_dir=target_out_dir,
                        block_size_m=eff_block_m,
                        l_threshold=args.l_threshold,
                        min_area_ha=args.min_area_ha,
                        reset_from_zero=is_reset,
                        train_10x=args.train_10x,
                        incremental=is_inc,
                        n_estimators=args.trees_per_round,
                        n_bag=args.bags,
                        dae_epochs=args.dae_epochs,
                        round_idx=r,
                        total_rounds=total_rounds,
                        run_label=args.run_label,
                        domain=args.domain,
                        base_model_dir=base_model_dir,
                        learning_rate=args.learning_rate,
                        max_depth=args.max_depth,
                        disable_unet=args.disable_unet,
                        active_heads=active_heads_list,
                        scan_only=args.scan_only,
                        on_schema_mismatch=args.on_schema_mismatch,
                        generalisation=generalisation,
                    )
                    if r == total_rounds:
                        summaries.append(s)

        if reg_up in ("BORNEO", "ALL_KALIMANTAN"):
            borneo_data = Path("data/borneo/processed")
            if (borneo_data / "dataset.parquet").exists():
                target_out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/seven_pass_borneo")
                eff_block_m = float(args.block_size_m) if args.block_size_m else 2_000.0
                for r in range(1, total_rounds + 1):
                    is_reset = reset_from_zero and (r == 1)
                    is_inc = incremental or (r > 1)
                    print(f"\n>>> [BORNEO // ROUND {r}/{total_rounds}] Initiating 7-Pass Training (domain={args.domain})...")
                    s = run_region_seven_pass(
                        region_name="BORNEO",
                        data_dir=borneo_data,
                        out_dir=target_out_dir,
                        block_size_m=eff_block_m,
                        l_threshold=args.l_threshold,
                        min_area_ha=args.min_area_ha,
                        reset_from_zero=is_reset,
                        train_10x=args.train_10x,
                        incremental=is_inc,
                        n_estimators=args.trees_per_round,
                        n_bag=args.bags,
                        dae_epochs=args.dae_epochs,
                        round_idx=r,
                        total_rounds=total_rounds,
                        run_label=args.run_label,
                        domain=args.domain,
                        base_model_dir=base_model_dir,
                        learning_rate=args.learning_rate,
                        max_depth=args.max_depth,
                        disable_unet=args.disable_unet,
                        active_heads=active_heads_list,
                        scan_only=args.scan_only,
                        on_schema_mismatch=args.on_schema_mismatch,
                        generalisation=generalisation,
                    )
                    if r == total_rounds:
                        summaries.append(s)

        if reg_up in ("KALIMANTAN", "KALIMANTAN_50M", "ALL_KALIMANTAN"):
            kal_data = Path("data/kalimantan/processed")
            if (kal_data / "dataset.parquet").exists():
                target_out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/seven_pass_kalimantan_50m")
                eff_block_m = float(args.block_size_m) if args.block_size_m else 5_000.0
                for r in range(1, total_rounds + 1):
                    is_reset = reset_from_zero and (r == 1)
                    is_inc = incremental or (r > 1)
                    print(f"\n>>> [KALIMANTAN // ROUND {r}/{total_rounds}] Initiating 7-Pass Training (domain={args.domain})...")
                    s = run_region_seven_pass(
                        region_name="KALIMANTAN",
                        data_dir=kal_data,
                        out_dir=target_out_dir,
                        block_size_m=eff_block_m,
                        l_threshold=args.l_threshold,
                        min_area_ha=args.min_area_ha,
                        reset_from_zero=is_reset,
                        train_10x=args.train_10x,
                        incremental=is_inc,
                        n_estimators=args.trees_per_round,
                        n_bag=args.bags,
                        dae_epochs=args.dae_epochs,
                        round_idx=r,
                        total_rounds=total_rounds,
                        run_label=args.run_label,
                        domain=args.domain,
                        base_model_dir=base_model_dir,
                        learning_rate=args.learning_rate,
                        max_depth=args.max_depth,
                        disable_unet=args.disable_unet,
                        active_heads=active_heads_list,
                        scan_only=args.scan_only,
                        on_schema_mismatch=args.on_schema_mismatch,
                        generalisation=generalisation,
                    )
                    if r == total_rounds:
                        summaries.append(s)

    # Persist unified dual-domain metrics JSON
    if summaries:
        audits_dir = Path("audits")
        audits_dir.mkdir(parents=True, exist_ok=True)
        dual_metrics_payload = {
            "domain": args.domain,
            "runs": summaries,
        }
        (audits_dir / "sift_dual_domain_metrics.json").write_text(
            json.dumps(dual_metrics_payload, indent=2) + "\n"
        )
        kal_proc_dir = Path("data/kalimantan/processed")
        if kal_proc_dir.exists():
            (kal_proc_dir / "sift_dual_domain_metrics.json").write_text(
                json.dumps(dual_metrics_payload, indent=2) + "\n"
            )

    print("\n================ KALIMANTAN 7-PASS TRAINING SUMMARY ================")
    print(json.dumps(summaries, indent=2))

    # Auto-refresh UI bundle unless skipped by background API server
    if os.environ.get("MPM_SKIP_UI_REBUILD") != "1":
        try:
            from scripts import build_purged_ui
            build_purged_ui.main()
            logger.info("Successfully refreshed dashboard UI bundle with latest training run.")
        except Exception as e:
            logger.warning("Could not refresh UI bundle: %s", e)


if __name__ == "__main__":
    main()

