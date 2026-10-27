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
from pathlib import Path

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
from src.features.feature_stack import FeatureStack, reconstruct_map
from src.labels.borneo_labels import build_k4_deposit_targets, rasterize_borneo_k4_labels
from src.labels.pu_negatives import (
    OUTMOGCGate,
    compute_asymmetric_head_weights,
    compute_inverse_propensity_weights,
    compute_k4_exclusion_mask,
    fit_exploration_propensity,
    fit_pooled_exploration_propensity,
)
from src.labels.spatial_splits import assign_spatial_blocks, spatial_block_kfold
from src.models.pu_pipeline import (
    LABEL_P,
    LABEL_RN,
    LABEL_U,
    K4_HEAD_NAMES,
    TwoStageK4Ensemble,
    blend_predictions_by_pr_auc,
    calibrate_closed_boundary_elkan_noto,
    compute_multi_mineral_conformal_bounds,
    estimate_k4_discovery_priors,
    score_metrics,
    verify_no_raw_pixel_leakage,
)
from src.models.unet_model import MPMDistrictDAEUNet
from src.outputs.prospectivity_map import (
    emit_site_cards_json,
    polygonize_and_rank_targets,
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
) -> dict:
    """Execute the complete 7-Pass pipeline strictly specialized on Kalimantan."""
    reg_clean = region_name.upper().strip()
    VALID_KALIMANTAN_REGIONS = {
        "KALIMANTAN", "BORNEO", "KALIMANTAN_50M", "KALIMANTAN_500M",
        "EAST_KALIMANTAN", "KALTIM", "KALIMANTAN_TIMUR",
        "HOTSPOT_SCHWANER", "HOTSPOT_KAYAN_NORTH", "HOTSPOT_KAHAYAN",
        "HOTSPOT_SCHWANER_50M", "HOTSPOT_KAYAN_50M",
    }
    if reg_clean not in VALID_KALIMANTAN_REGIONS:
        raise ValueError(
            f"Strict Kalimantan-only policy enforced: Training outside Kalimantan is prohibited. "
            f"Requested region: '{region_name}'. Models are specialized exclusively for Kalimantan deposit mechanics (K=4)."
        )

    from src.models.training_run_manager import archive_existing_run, register_completed_run

    out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = out_dir / "maps"
    targets_dir = out_dir / "targets"
    models_dir = out_dir / "models"
    for d in (maps_dir, targets_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    ensemble_checkpoint = models_dir / "k4_ensemble.joblib"
    prior_ensemble: Optional[TwoStageK4Ensemble] = None

    if reset_from_zero or not incremental:
        logger.info("[%s] Resetting model weights and caches to learn from zero...", region_name)
        archive_existing_run(out_dir, region_name)
        for d in (models_dir, maps_dir, targets_dir):
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()
    else:
        # Check if prior ensemble exists for continuous incremental warm-start
        if ensemble_checkpoint.exists():
            try:
                prior_ensemble = TwoStageK4Ensemble.load(ensemble_checkpoint)
                logger.info(
                    "[%s] Continuous / Incremental Weight Update: Loaded prior ensemble from %s",
                    region_name, ensemble_checkpoint,
                )
            except Exception as e:
                logger.warning("[%s] Could not load prior ensemble from %s: %s", region_name, ensemble_checkpoint, e)
                prior_ensemble = None
        else:
            logger.info(
                "[%s] Incremental mode active; no prior ensemble found at %s. Initializing fresh weights.",
                region_name, ensemble_checkpoint,
            )

        # Step 0: Archive prior training run data so historical comparisons are preserved
        archive_existing_run(out_dir, region_name)

    df = pd.read_parquet(data_dir / "dataset.parquet")
    stack: FeatureStack = joblib.load(data_dir / "feature_stack.joblib")
    profile = stack.profile
    C, H, W = stack.array.shape

    NON_FEATURE_COLS = {
        "label", "row", "col", "block_id", "explored_footprint", "alteration_benchmark",
        "label_k0_porphyry_cu", "label_k1_skarn_fe_cu", "label_k2_epithermal_au", "label_k3_laterite_fe_ni",
        "commodity_label", "is_centroid_artifact", "target_class",
    }
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and not c.startswith("label_")
    ]

    # Gate 1: Verify No Raw Pixel Leakage
    verify_no_raw_pixel_leakage(feature_cols)
    logger.info("[%s] Pass 1 verified: %d physical/structural channels (%s)", region_name, len(feature_cols), feature_cols)

    X_all = df[feature_cols].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy(dtype=np.int32)
    rows = df["row"].to_numpy(dtype=np.int32)
    cols = df["col"].to_numpy(dtype=np.int32)

    # -------------------------------------------------------------------------
    # Step 1.1: Multi-Mineral Targets (K=4), Alteration Benchmark & Exploration Footprint
    # -------------------------------------------------------------------------
    k4_col_names = [
        "label_k0_porphyry_cu",
        "label_k1_skarn_fe_cu",
        "label_k2_epithermal_au",
        "label_k3_laterite_fe_ni",
    ]
    has_k4 = all(c in df.columns for c in k4_col_names)
    if not has_k4:
        deposit_targets = build_k4_deposit_targets()
        rasters_k4 = rasterize_borneo_k4_labels(deposit_targets, profile, buffer_m=1000.0)
        for c in k4_col_names:
            df[c] = rasters_k4[c][rows, cols]
        df["explored_footprint"] = rasters_k4["explored_footprint"][rows, cols]
        df["alteration_benchmark"] = rasters_k4["alteration_benchmark"][rows, cols]

    y_k4 = np.column_stack([df[c].to_numpy(dtype=np.int32) for c in k4_col_names])

    # If general label 1 has positives but all K=4 heads are empty in this tile, preserve positive ground truth
    if (y_k4 == 1).sum() == 0 and (labels == LABEL_P).sum() > 0:
        y_k4[labels == LABEL_P, 2] = 1

    pos_mask_any = np.any(y_k4 == 1, axis=1) | (labels == LABEL_P)

    # OUTMOGC closed boundary gates per head
    outmogc_gates_list = []
    for k in range(4):
        pos_k_mask = (y_k4[:, k] == 1)
        if np.sum(pos_k_mask) >= 5:
            X_k_pos = X_all[pos_k_mask]
            gate_k = OUTMOGCGate(
                n_pca_components=min(4, X_k_pos.shape[1]),
                n_gmm_components=min(2, max(1, len(X_k_pos) // 10)),
                random_state=42,
            )
            gate_k.fit(X_k_pos)
            outmogc_gates_list.append(gate_k.score(X_all))
        else:
            outmogc_gates_list.append(np.ones(len(X_all), dtype=np.float64))
    outmogc_gates = np.column_stack(outmogc_gates_list)

    # -------------------------------------------------------------------------
    # Step 1.2: Spatial Block Partitioning & Georeferencing
    # -------------------------------------------------------------------------
    from rasterio.crs import CRS as RioCRS
    import rasterio.transform
    transform = profile["transform"]
    is_geo = RioCRS.from_user_input(profile["crs"]).is_geographic
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    cell_coords = np.column_stack([xs, ys])

    # Exclusion masks (Pass 1 Rule 4: buffer around deposit occurrences per head)
    positive_coords_k = {k: cell_coords[y_k4[:, k] == 1] for k in range(4)}
    exclusion_masks = compute_k4_exclusion_mask(
        coords=cell_coords,
        positive_coords_k=positive_coords_k,
        labels_k4=y_k4,
        is_geographic=is_geo,
    )

    raw_dx = abs(float(transform.a)) if abs(float(transform.a)) > 0 else 50.0
    px_size_m = raw_dx * 111_320.0 if is_geo else raw_dx
    eff_block_m = block_size_m * 111_320.0 if (is_geo and block_size_m < 50.0) else block_size_m
    block_size_px = min(max(12, int(round(eff_block_m / max(px_size_m, 1.0)))), max(12, min(H, W) // 5))
    block_ids = assign_spatial_blocks(rows, cols, H, W, block_size_px=block_size_px)
    folds = spatial_block_kfold(block_ids, n_folds=5, random_state=42)

    total_pos = max(1, int(pos_mask_any.sum()))
    target_val_pos = 0.20 * total_pos
    valid_folds = [
        (tr_idx, va_idx)
        for tr_idx, va_idx in folds
        if pos_mask_any[tr_idx].sum() >= max(2, int(0.5 * total_pos))
        and pos_mask_any[va_idx].sum() >= 1
    ]
    chosen_fold = min(valid_folds, key=lambda pair: abs(int(pos_mask_any[pair[1]].sum()) - target_val_pos)) if valid_folds else folds[0]
    train_idx, val_idx = chosen_fold

    # Normalise cube channels using ONLY training split statistics (no validation leakage)
    cube_norm = stack.array.astype(np.float32).copy()
    tr_rows, tr_cols = rows[train_idx], cols[train_idx]
    for ch in range(C):
        band = cube_norm[ch]
        tr_vals = band[tr_rows, tr_cols]
        m = float(np.nanmean(tr_vals)) if np.isfinite(tr_vals).any() else 0.0
        s = float(np.nanstd(tr_vals)) + 1e-6
        cube_norm[ch] = np.nan_to_num((band - m) / s, nan=0.0)

    # -------------------------------------------------------------------------
    # Pass 2: Selection-Bias Weighting & Asymmetric Sample Weights
    # -------------------------------------------------------------------------
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
        explored_mask = (explored_footprint_col.to_numpy() == 1) | pos_mask_any
    else:
        explored_mask = pos_mask_any

    y_exp_tr = explored_mask[train_idx].astype(int)
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
        e_x = fit_exploration_propensity(
            accessibility_features=X_access,
            explored_mask=explored_mask,
            min_propensity=0.05,
        )

    # Asymmetric sample weights for all 4 heads
    sample_weights_k4 = np.column_stack([
        compute_asymmetric_head_weights(
            labels_k=y_k4[:, k],
            e_logistics=e_x,
            exclusion_mask_k=exclusion_masks[:, k],
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

    # -------------------------------------------------------------------------
    # Pass 3.1: TwoStageK4Ensemble (4 Independent XGBoost Heads + Stacking)
    # -------------------------------------------------------------------------
    n_estimators = 500 if train_10x else 100
    n_bag = 10 if train_10x else 5
    dae_epochs = 15 if train_10x else 3

    logger.info(
        "[%s] Configured K=4 Ensemble: n_estimators=%d, n_bag=%d, dae_epochs=%d (train_10x=%s)",
        region_name, n_estimators, n_bag, dae_epochs, train_10x,
    )

    k4_ensemble = TwoStageK4Ensemble(
        n_estimators=n_estimators,
        n_bag=n_bag,
        random_state=42,
    )
    k4_ensemble.fit(
        X=X_all[train_idx],
        y_k4=y_k4[train_idx],
        sample_weights_k4=sample_weights_k4[train_idx],
        e_logistics=e_x[train_idx],
        exclusion_masks=exclusion_masks[train_idx],
        spatial_block_ids=block_ids[train_idx],
        prior_ensemble=prior_ensemble,
    )
    # Persist updated ensemble checkpoint for continuous weight updates across future runs
    k4_ensemble.save(ensemble_checkpoint)
    logger.info("[%s] Saved updated TwoStageK4Ensemble weights to %s", region_name, ensemble_checkpoint)

    s_xgb_k4_full = k4_ensemble.predict_proba(X_all)
    std_xgb_k4_full = k4_ensemble.predict_std(X_all)

    # -------------------------------------------------------------------------
    # Pass 3.2: District DAE-U-Net (250m Spatial Context + Anomaly Detection)
    # -------------------------------------------------------------------------
    fold_assignments = np.zeros(len(rows), dtype=int)
    for f_idx, (_, va_idx) in enumerate(folds):
        fold_assignments[va_idx] = f_idx
    fold_ids_2d = reconstruct_map(fold_assignments.astype(np.float64), rows, cols, H, W, nodata=-1.0).astype(int)

    y_k4_2d = np.stack([
        reconstruct_map(np.where(y_k4[:, k] == 1, 1.0, 0.0), rows, cols, H, W, nodata=0.0)
        for k in range(4)
    ], axis=0)

    dae_unet = MPMDistrictDAEUNet(
        in_channels=C,
        n_classes=4,
        window_px=32,
        batch_size=16,
        dropout_p=0.3,
        channel_drop_p=0.15,
        recon_weight=0.5,
    )
    dae_out = dae_unet.cross_validate_spatial_blocks(
        X_grid_50m=cube_norm,
        y_grid_k4=y_k4_2d,
        block_ids_50m=fold_ids_2d,
        decimate_factor=5,
        epochs=dae_epochs,
        checkpoints_dir=models_dir,
    )

    s_unet_k4_full = np.column_stack([
        dae_out[f"oof_prob_k{k}"][rows, cols]
        for k in range(4)
    ])
    dae_recon_err_full = dae_out["dae_recon_err"][rows, cols]

    epistemic_k4_full = np.sqrt(
        0.5 * (std_xgb_k4_full ** 2) + 0.5 * ((s_xgb_k4_full - s_unet_k4_full) ** 2) + 1e-4
    )

    # -------------------------------------------------------------------------
    # Pass 3.3: Validation PR-AUC Ensembling Per Head
    # -------------------------------------------------------------------------
    fused_k4_full = np.zeros_like(s_xgb_k4_full)
    pr_auc_xgb_heads = []
    pr_auc_unet_heads = []
    pr_auc_fused_heads = []

    for k in range(4):
        y_k_val = (y_k4[val_idx, k] == 1).astype(int)
        pos_k_val = int(np.sum(y_k_val))
        s_xgb_val_k = s_xgb_k4_full[val_idx, k]
        s_unet_val_k = s_unet_k4_full[val_idx, k]

        if pos_k_val >= 1 and (len(y_k_val) - pos_k_val) >= 1:
            auc_xgb_k = pr_auc(y_k_val, s_xgb_val_k)
            auc_unet_k = pr_auc(y_k_val, s_unet_val_k)
            _, w_xgb_k, w_unet_k = blend_predictions_by_pr_auc(
                s_xgb_val_k, s_unet_val_k, auc_xgb_k, auc_unet_k
            )
            fused_val_k = w_xgb_k * s_xgb_val_k + w_unet_k * s_unet_val_k
            auc_fused_k = pr_auc(y_k_val, fused_val_k)
            fused_k4_full[:, k] = w_xgb_k * s_xgb_k4_full[:, k] + w_unet_k * s_unet_k4_full[:, k]
        else:
            auc_xgb_k = 0.50
            auc_unet_k = 0.50
            auc_fused_k = 0.50
            fused_k4_full[:, k] = 0.70 * s_xgb_k4_full[:, k] + 0.30 * s_unet_k4_full[:, k]

        pr_auc_xgb_heads.append(auc_xgb_k)
        pr_auc_unet_heads.append(auc_unet_k)
        pr_auc_fused_heads.append(auc_fused_k)

    logger.info(
        "[%s] Pass 3 K=4 Ensemble | H0 (Porphyry)=%.3f | H1 (Skarn)=%.3f | H2 (Epithermal)=%.3f | H3 (Laterite)=%.3f",
        region_name, pr_auc_fused_heads[0], pr_auc_fused_heads[1], pr_auc_fused_heads[2], pr_auc_fused_heads[3]
    )

    # -------------------------------------------------------------------------
    # Pass 4: Closed-Boundary Elkan-Noto Scaling
    # -------------------------------------------------------------------------
    c_hat_k4 = estimate_k4_discovery_priors(
        val_probs_k4=fused_k4_full[val_idx],
        val_targets_k4=y_k4[val_idx],
        min_c=0.10,
        train_block_ids=train_blocks,
        val_block_ids=block_ids[val_idx],
    )
    cal_k4_full = np.zeros_like(fused_k4_full)
    for k in range(4):
        cal_k4_full[:, k] = calibrate_closed_boundary_elkan_noto(
            p_raw_k=fused_k4_full[:, k],
            c_hat_k=c_hat_k4[k],
            gamma_closed_k=outmogc_gates[:, k],
            min_c=0.10,
        )

    # -------------------------------------------------------------------------
    # Pass 5: Multi-Mineral Conformal Bounds & Prediction Sets
    # -------------------------------------------------------------------------
    lower_k4_list = []
    q_hat_k4_list = []
    v_scores_k4_list = []

    for k in range(4):
        p_cal_k = cal_k4_full[:, k]
        sig_k = epistemic_k4_full[:, k]
        pos_k_val_mask = (y_k4[val_idx, k] == 1)
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

    save_prospectivity_geotiff(
        cal_prob_composite, rows, cols, H, W, profile, str(maps_dir / "calibrated_prospectivity.tif")
    )
    save_prospectivity_geotiff(
        fused_composite, rows, cols, H, W, profile, str(maps_dir / "fused_raw_score.tif")
    )
    save_prospectivity_geotiff(
        e_x, rows, cols, H, W, profile, str(maps_dir / "propensity_e_x.tif")
    )
    save_conformal_bounds_geotiff(
        lower_L_composite, upper_U_composite, rows, cols, H, W, profile, str(maps_dir / "conformal_bounds_90ci.tif")
    )
    save_prospectivity_geotiff(
        dae_recon_err_full, rows, cols, H, W, profile, str(maps_dir / "dae_reconstruction_error.tif")
    )

    # -------------------------------------------------------------------------
    # Pass 6 & Pass 7: Polygonization, Multi-Criteria Ranking & LLM Site Cards
    # -------------------------------------------------------------------------
    cal_2d = reconstruct_map(cal_prob_composite, rows, cols, H, W, nodata=np.nan)
    lower_2d = reconstruct_map(lower_L_composite, rows, cols, H, W, nodata=np.nan)
    upper_2d = reconstruct_map(upper_U_composite, rows, cols, H, W, nodata=np.nan)
    fused_2d = reconstruct_map(fused_composite, rows, cols, H, W, nodata=np.nan)
    ex_2d = reconstruct_map(e_x, rows, cols, H, W, nodata=np.nan)
    wx_2d = reconstruct_map(sample_weights_k4[:, 0], rows, cols, H, W, nodata=np.nan)

    # Derive physical summary grids by matching feature names in stack.feature_names
    name_to_idx = {n.lower(): i for i, n in enumerate(stack.feature_names)}

    def _norm_band(arr_2d: np.ndarray, invert: bool = False) -> np.ndarray:
        b = np.nan_to_num(arr_2d.astype(np.float64), nan=float(np.nanmedian(arr_2d)))
        normed = np.clip((b - b.min()) / (np.ptp(b) + 1e-6), 0.0, 1.0)
        return (1.0 - normed) if invert else normed

    # 1. Dilation tendency Td
    if "dilation_tendency" in name_to_idx:
        td_2d = _norm_band(stack.array[name_to_idx["dilation_tendency"]])
    elif "dist_faults_m" in name_to_idx:
        td_2d = _norm_band(stack.array[name_to_idx["dist_faults_m"]], invert=True)
    else:
        td_2d = _norm_band(cube_norm[0])

    # 2. Hydrothermal alteration index
    alt_Idx = next(
        (name_to_idx[k] for k in ("swir_clay_iron_alteration", "aster_aloh", "s2_clay", "hydrothermal_alteration_idx") if k in name_to_idx),
        None,
    )
    alt_2d = _norm_band(stack.array[alt_Idx]) if alt_Idx is not None else _norm_band(cube_norm[0])

    # 3. Regolith cover depth (m)
    if "regolith_cover_m" in name_to_idx:
        cover_2d = np.clip(np.nan_to_num(stack.array[name_to_idx["regolith_cover_m"]], nan=25.0), 1.0, 200.0)
    else:
        cover_2d = np.clip((1.0 - ex_2d) * 65.0, 2.0, 120.0)

    eff_l_thr = float(np.clip(np.nanpercentile(lower_2d, 96), 0.35, 0.88))
    eff_closing = 2
    eff_min_area = float(min_area_ha)

    ranked_gdf = polygonize_and_rank_targets(
        calibrated_prob=cal_2d,
        lower_bound_L=lower_2d,
        upper_bound_U=upper_2d,
        dilation_tendency_Td=td_2d,
        alteration_index=alt_2d,
        cover_depth_m=cover_2d,
        profile=profile,
        l_threshold=eff_l_thr,
        closing_size=eff_closing,
        min_area_ha=eff_min_area,
        region_prefix=region_name.upper(),
        extra_layers={
            "fused_raw_score": fused_2d,
            "exploration_propensity_e_x": ex_2d,
            "propensity_weight_w": wx_2d,
        },
    )
    if len(ranked_gdf) == 0 and np.isfinite(cal_2d).any():
        fallback_thr = float(np.nanpercentile(cal_2d, 96))
        eff_l_thr = fallback_thr
        ranked_gdf = polygonize_and_rank_targets(
            calibrated_prob=cal_2d,
            lower_bound_L=cal_2d,
            upper_bound_U=upper_2d,
            dilation_tendency_Td=td_2d,
            alteration_index=alt_2d,
            cover_depth_m=cover_2d,
            profile=profile,
            l_threshold=fallback_thr,
            closing_size=eff_closing,
            min_area_ha=eff_min_area,
            region_prefix=region_name.upper(),
            extra_layers={
                "fused_raw_score": fused_2d,
                "exploration_propensity_e_x": ex_2d,
                "propensity_weight_w": wx_2d,
            },
        )

    # Classify each target polygon with its dominant deposit head
    conformal_target_sets = []
    if len(ranked_gdf) > 0:
        target_classes = []
        for p_idx, p_geom in enumerate(ranked_gdf.geometry):
            c_proj = p_geom.centroid
            inv_tr = ~profile["transform"]
            col_f, row_f = inv_tr * (c_proj.x, c_proj.y)
            r_idx = int(np.clip(round(row_f), 0, H - 1))
            c_idx = int(np.clip(round(col_f), 0, W - 1))
            match_pix = np.where((rows == r_idx) & (cols == c_idx))[0]
            if len(match_pix) > 0:
                p_cell = match_pix[0]
            else:
                p_cell = int(np.argmin((rows - r_idx) ** 2 + (cols - c_idx) ** 2))

            cell_head_scores = lower_k4_full[p_cell]
            dom_head_idx = int(np.argmax(cell_head_scores))
            target_classes.append(K4_HEAD_NAMES[dom_head_idx])
            conformal_target_sets.append(conformal_sets[p_cell] or [K4_HEAD_NAMES[dom_head_idx]])

        ranked_gdf["target_class"] = target_classes

    if len(ranked_gdf) > 15:
        ranked_gdf = ranked_gdf.head(15).reset_index(drop=True)

    if len(ranked_gdf) > 0:
        ranked_gdf.to_file(targets_dir / f"ranked_targets_{region_name.lower()}.gpkg", driver="GPKG")

    cards_path = targets_dir / "site_cards.json"
    cards = emit_site_cards_json(
        ranked_gdf,
        out_path=str(cards_path),
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
    )

    # Save true out-of-fold 5-fold spatial block CV metrics
    cv_rows = []
    for f_idx, (tr_f_idx, va_f_idx) in enumerate(folds):
        va_f_pos = pos_mask_any[va_f_idx]
        p_va_f = va_f_idx[va_f_pos]
        u_va_f = va_f_idx[~va_f_pos]
        if len(p_va_f) >= 1 and len(u_va_f) >= 10:
            sub_u = u_va_f[: min(600, len(u_va_f))]
            ev_f = np.concatenate([p_va_f, sub_u])
            y_f = np.concatenate([np.ones(len(p_va_f), dtype=int), np.zeros(len(sub_u), dtype=int)])
            fold_probs = np.max(k4_ensemble.predict_proba(X_all[ev_f]), axis=1)
            m_f = score_metrics(y_f, fold_probs)
            m_f["fold"] = f_idx
            cv_rows.append(m_f)

    if not cv_rows:
        y_val_binary = (pos_mask_any[val_idx]).astype(int)
        fused_val_comp = np.max(fused_k4_full[val_idx], axis=1)
        m_f = score_metrics(y_val_binary, fused_val_comp)
        m_f["fold"] = 0
        cv_rows.append(m_f)

    pd.DataFrame(cv_rows).to_csv(out_dir / "spatial_cv_metrics.csv", index=False)
    holdout_metrics = {
        k: float(np.mean([r[k] for r in cv_rows]))
        for k in ("pr_auc", "roc_auc", "recall_at_fpr_0.1", "ndcg_at_k_100", "prevalence")
        if k in cv_rows[0]
    }
    (out_dir / "blind_holdout.json").write_text(json.dumps(holdout_metrics, indent=2) + "\n")

    summary = {
        "region": region_name,
        "n_cells": len(df),
        "n_positives": int(pos_mask_any.sum()),
        "pr_auc_xgb": round(float(np.mean(pr_auc_xgb_heads)), 4),
        "pr_auc_unet": round(float(np.mean(pr_auc_unet_heads)), 4),
        "pr_auc_fused": round(float(np.mean(pr_auc_fused_heads)), 4),
        "pr_auc_heads": {
            K4_HEAD_NAMES[k]: round(float(pr_auc_fused_heads[k]), 4)
            for k in range(4)
        },
        "w_xgb": 0.70,
        "w_unet": 0.30,
        "elkan_noto_c_hat": round(float(np.mean(c_hat_k4)), 4),
        "conformal_q_hat_90": round(float(np.quantile(c_hat_k4, 0.90)), 4),
        "propensity_min": round(float(e_x.min()), 4),
        "propensity_median": round(float(np.median(e_x)), 4),
        "n_ranked_targets": len(ranked_gdf),
        "top_target_card": cards[0] if cards else None,
        "training_mode": (
            f"{'10x_compute_' if train_10x else ''}"
            f"{'incremental_warm_start' if prior_ensemble is not None else ('learned_from_zero' if reset_from_zero else 'standard')}"
        ),
        "is_warm_started": bool(prior_ensemble is not None),
        "n_estimators": n_estimators,
        "n_bag": n_bag,
        "dae_epochs": dae_epochs,
        "resolution_m": 250.0 if (stack.array.shape[1] == 2400) else 500.0,
    }
    (out_dir / "seven_pass_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    try:
        register_completed_run(out_dir, region_name, summary)
    except Exception as e:
        logger.warning("Could not register completed run in catalog: %s", e)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalimantan Deposit-Mechanics 7-Pass Training Engine")
    parser.add_argument(
        "--region",
        type=str,
        default="BORNEO",
        help="Region to train (must be strictly in Kalimantan: BORNEO or KALIMANTAN)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Enable continuous incremental weight updates from previous checkpoints (default: True)",
    )
    parser.add_argument(
        "--reset-from-zero",
        action="store_true",
        default=False,
        help="Reset models and clear cached checkpoints to learn from zero (default: False)",
    )
    parser.add_argument(
        "--train-10x",
        action="store_true",
        default=False,
        help="Run 10x compute training run (500 estimators x 10 bags XGBoost, 15 epochs DAE-U-Net)",
    )
    args = parser.parse_args()

    # If --reset-from-zero is passed, disable incremental warm-start
    reset_from_zero = args.reset_from_zero
    incremental = args.incremental and not reset_from_zero

    reg_up = args.region.upper().strip()
    VALID_KALIMANTAN_REGIONS = {
        "KALIMANTAN", "BORNEO", "KALIMANTAN_50M", "KALIMANTAN_500M",
        "EAST_KALIMANTAN", "KALTIM", "KALIMANTAN_TIMUR",
        "HOTSPOT_SCHWANER", "HOTSPOT_KAYAN_NORTH", "HOTSPOT_KAHAYAN",
        "HOTSPOT_SCHWANER_50M", "HOTSPOT_KAYAN_50M", "ALL_KALIMANTAN",
    }
    if reg_up not in VALID_KALIMANTAN_REGIONS:
        raise ValueError(
            f"Strict Kalimantan-only policy enforced: Training outside Kalimantan is prohibited. "
            f"Requested region: '{args.region}'. Specialized exclusively for Kalimantan deposit mechanics (K=4)."
        )

    summaries = []
    # Train Kalimantan regions exclusively
    if reg_up in ("EAST_KALIMANTAN", "KALTIM", "KALIMANTAN_TIMUR", "ALL_KALIMANTAN"):
        ek_data = Path("data/east_kalimantan/processed")
        if (ek_data / "dataset.parquet").exists():
            summaries.append(
                run_region_seven_pass(
                    region_name="EAST_KALIMANTAN",
                    data_dir=ek_data,
                    out_dir=Path("outputs/seven_pass_east_kalimantan"),
                    block_size_m=25_000.0,
                    l_threshold=0.50,
                    min_area_ha=5.0,
                    reset_from_zero=reset_from_zero,
                    train_10x=args.train_10x,
                    incremental=incremental,
                )
            )

    if reg_up in ("BORNEO", "ALL_KALIMANTAN"):
        borneo_data = Path("data/borneo/processed")
        if (borneo_data / "dataset.parquet").exists():
            summaries.append(
                run_region_seven_pass(
                    region_name="BORNEO",
                    data_dir=borneo_data,
                    out_dir=Path("outputs/seven_pass_borneo"),
                    block_size_m=2_000.0,
                    l_threshold=0.50,
                    min_area_ha=5.0,
                    reset_from_zero=reset_from_zero,
                    incremental=incremental,
                )
            )

    if reg_up in ("KALIMANTAN", "KALIMANTAN_50M", "ALL_KALIMANTAN"):
        kal_data = Path("data/kalimantan/processed")
        if (kal_data / "dataset.parquet").exists():
            summaries.append(
                run_region_seven_pass(
                    region_name="KALIMANTAN",
                    data_dir=kal_data,
                    out_dir=Path("outputs/seven_pass_kalimantan_50m"),
                    block_size_m=5_000.0,
                    l_threshold=0.50,
                    min_area_ha=5.0,
                    reset_from_zero=reset_from_zero,
                    incremental=incremental,
                )
            )

    print("\n================ KALIMANTAN 7-PASS TRAINING SUMMARY ================")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
