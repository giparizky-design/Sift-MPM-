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
from src.labels.pu_negatives import (
    compute_inverse_propensity_weights,
    fit_exploration_propensity,
)
from src.labels.spatial_splits import assign_spatial_blocks, spatial_block_kfold
from src.models.pu_pipeline import (
    LABEL_P,
    LABEL_RN,
    LABEL_U,
    blend_predictions_by_pr_auc,
    calibrate_elkan_noto_posterior,
    estimate_elkan_noto_c,
    fit_pu_model,
    score_metrics,
    verify_no_raw_pixel_leakage,
)
from src.models.unet_model import MPMUNet
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
) -> dict:
    """Execute the complete 7-Pass pipeline on a region dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = out_dir / "maps"
    targets_dir = out_dir / "targets"
    models_dir = out_dir / "models"
    for d in (maps_dir, targets_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / "dataset.parquet")
    stack: FeatureStack = joblib.load(data_dir / "feature_stack.joblib")
    profile = stack.profile
    C, H, W = stack.array.shape

    feature_cols = [c for c in df.columns if c not in ("label", "row", "col")]

    # Gate 1: Verify No Raw Pixel Leakage
    verify_no_raw_pixel_leakage(feature_cols)
    logger.info("[%s] Pass 1 verified: %d physical/structural channels (%s)", region_name, len(feature_cols), feature_cols)

    X_all = df[feature_cols].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy(dtype=np.int32)
    rows = df["row"].to_numpy(dtype=np.int32)
    cols = df["col"].to_numpy(dtype=np.int32)

    # -------------------------------------------------------------------------
    # Spatial Block Partitioning FIRST (before normalization or propensity fitting)
    # -------------------------------------------------------------------------
    from rasterio.crs import CRS as RioCRS
    transform = profile["transform"]
    is_geo = RioCRS.from_user_input(profile["crs"]).is_geographic
    raw_dx = abs(float(transform.a)) if abs(float(transform.a)) > 0 else 50.0
    px_size_m = raw_dx * 111_320.0 if is_geo else raw_dx
    eff_block_m = block_size_m * 111_320.0 if (is_geo and block_size_m < 50.0) else block_size_m
    block_size_px = min(max(12, int(round(eff_block_m / max(px_size_m, 1.0)))), max(12, min(H, W) // 5))
    block_ids = assign_spatial_blocks(rows, cols, H, W, block_size_px=block_size_px)
    folds = spatial_block_kfold(block_ids, n_folds=5, random_state=42)

    total_pos = max(1, int((labels == LABEL_P).sum()))
    target_val_pos = 0.20 * total_pos
    valid_folds = [
        (tr_idx, va_idx)
        for tr_idx, va_idx in folds
        if (labels[tr_idx] == LABEL_P).sum() >= max(2, int(0.5 * total_pos))
        and (labels[va_idx] == LABEL_P).sum() >= 1
    ]
    if valid_folds:
        chosen_fold = min(
            valid_folds,
            key=lambda pair: abs(int((labels[pair[1]] == LABEL_P).sum()) - target_val_pos),
        )
    else:
        chosen_fold = folds[0]
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
    # Pass 2: Selection-Bias Weighting (fit on accessibility proxies & train_idx only)
    # -------------------------------------------------------------------------
    ACCESS_KEYS = (
        "outcrop", "regolith", "cover", "coastal", "slope", "elev", "dem", "ndvi", "tri"
    )
    access_col_indices = [
        i for i, col in enumerate(feature_cols)
        if any(k in col.lower() for k in ACCESS_KEYS)
    ] or [0]
    X_access = X_all[:, access_col_indices]
    explored_mask = (labels == LABEL_P) | (labels == LABEL_RN)

    # Fit logistic regression on training blocks only, then predict across full grid
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_exp_tr = explored_mask[train_idx].astype(int)
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
    w_all = compute_inverse_propensity_weights(e_x, min_propensity=0.05)

    # Ensure strictly disjoint spatial block IDs (Acceptance Criterion 4)
    train_blocks = np.unique(block_ids[train_idx])
    val_blocks = np.unique(block_ids[val_idx])
    overlap = set(train_blocks).intersection(set(val_blocks))
    if overlap:
        val_mask_disjoint = ~np.isin(block_ids[val_idx], list(overlap))
        val_idx = val_idx[val_mask_disjoint]
        val_blocks = np.unique(block_ids[val_idx])

    # -------------------------------------------------------------------------
    # Pass 3.1: Inverse-Propensity-Weighted Tabular XGBoost Pass
    # -------------------------------------------------------------------------
    config = {
        "pu_learning": {
            "step1_contamination": 0.05,
            "rn_percentile_threshold": 10.0,
            "random_state": 42,
        },
        "models": {
            "baseline": {
                "n_bag": 5,
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            },
            "unet": {
                "window_px": 16,
                "in_channels": C,
                "dropout_p": 0.2,
                "batch_size": 32,
                "pi_p": 0.05,
                "use_nnpu": True,
            },
        },
    }

    tr_labels = labels[train_idx]
    p_tr = train_idx[tr_labels == LABEL_P]
    rn_tr = train_idx[tr_labels == LABEL_RN]
    u_tr = train_idx[tr_labels == LABEL_U]

    xgb_val_model = fit_pu_model(
        X_all[p_tr],
        X_all[rn_tr],
        X_all[u_tr],
        config,
        w_p=w_all[p_tr],
        w_rn=w_all[rn_tr] if len(rn_tr) else None,
        w_u=w_all[u_tr],
    )

    # Evaluate on held-out spatial block (positives + verified RN or background sample)
    va_labels = labels[val_idx]
    p_va = val_idx[va_labels == LABEL_P]
    neg_va = val_idx[va_labels == LABEL_RN]
    if len(neg_va) < 20:
        u_va = val_idx[va_labels == LABEL_U]
        rng = np.random.default_rng(42)
        neg_va = rng.choice(u_va, size=min(1000, len(u_va)), replace=False)

    eval_idx = np.concatenate([p_va, neg_va])
    y_eval = np.concatenate([np.ones(len(p_va), dtype=int), np.zeros(len(neg_va), dtype=int)])

    s_xgb_eval = xgb_val_model.predict_proba(X_all[eval_idx])
    pr_auc_xgb = pr_auc(y_eval, s_xgb_eval)

    # -------------------------------------------------------------------------
    # Pass 3.2: Spatial Context U-Net Pass with Non-Negative PU (nnPU) Risk Loss
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(42)
    p_sub_tr = rng.choice(p_tr, size=min(40, len(p_tr)), replace=False)
    u_sub_tr = rng.choice(u_tr, size=min(len(p_sub_tr) * 2, len(u_tr)), replace=False)
    unet_tr_idx = np.concatenate([p_sub_tr, u_sub_tr])
    y_unet_tr = (labels[unet_tr_idx] == LABEL_P).astype(np.float32)
    w_unet_tr = w_all[unet_tr_idx].astype(np.float32)

    X_win_tr = _extract_windows_around_cells(cube_norm, rows[unet_tr_idx], cols[unet_tr_idx], window_px=16)
    unet_model = MPMUNet(config)
    unet_model.fit(
        X_win_tr,
        y_unet_tr,
        epochs=3,
        lr=1e-3,
        sample_weight=w_unet_tr,
        pi_p=0.05,
        use_nnpu=True,
    )

    from scipy.ndimage import zoom as nd_zoom

    s_xgb_full = xgb_val_model.predict_proba(X_all)
    xgb_var_full = xgb_val_model.predict_uncertainty(X_all)

    # Evaluate U-Net directly on all validation evaluation cells
    X_win_eval = _extract_windows_around_cells(
        cube_norm, rows[eval_idx], cols[eval_idx], window_px=16
    )
    s_unet_eval, unet_var_eval = unet_model.predict_proba_mc(X_win_eval, n_passes=5)
    pr_auc_unet = pr_auc(y_eval, s_unet_eval)

    # Strided U-Net inference across the 2D grid for tractable full-grid CNN surface
    stride = max(1, min(H, W) // 48)
    r_grid = np.arange(0, H, stride, dtype=np.int32)
    c_grid = np.arange(0, W, stride, dtype=np.int32)
    rr_s, cc_s = np.meshgrid(r_grid, c_grid, indexing="ij")
    X_win_grid = _extract_windows_around_cells(
        cube_norm, rr_s.ravel(), cc_s.ravel(), window_px=16
    )
    unet_grid_mean, _ = unet_model.predict_proba_mc(X_win_grid, n_passes=3)
    unet_coarse_2d = unet_grid_mean.reshape(len(r_grid), len(c_grid))
    unet_2d = nd_zoom(
        unet_coarse_2d, (H / len(r_grid), W / len(c_grid)), order=1
    )[:H, :W]
    s_unet_full = np.clip(unet_2d[rows, cols], 0.0, 1.0).astype(np.float64)
    s_unet_full[eval_idx] = s_unet_eval

    # True multi-model epistemic standard deviation (combining tree variance + XGB/UNet disagreement)
    epistemic_full = np.sqrt(
        0.5 * xgb_var_full + 0.5 * ((s_xgb_full - s_unet_full) ** 2) + 1e-4
    )

    # -------------------------------------------------------------------------
    # Pass 3.3: Validation PR-AUC Ensembling
    # -------------------------------------------------------------------------
    fused_eval, w_xgb, w_unet = blend_predictions_by_pr_auc(
        s_xgb_eval, s_unet_eval, pr_auc_xgb, pr_auc_unet
    )
    pr_auc_fused = pr_auc(y_eval, fused_eval)
    logger.info(
        "[%s] Pass 3 Ensemble | PR-AUC XGB=%.3f (w=%.2f) | UNet=%.3f (w=%.2f) | Fused=%.3f",
        region_name, pr_auc_xgb, w_xgb, pr_auc_unet, w_unet, pr_auc_fused,
    )

    # -------------------------------------------------------------------------
    # Pass 4: Elkan-Noto Posterior Calibration (Strictly on Spatially Disjoint Holdout)
    # -------------------------------------------------------------------------
    val_pos_scores = fused_eval[: len(p_va)]
    c_hat = estimate_elkan_noto_c(
        val_positive_scores=val_pos_scores,
        train_block_ids=train_blocks,
        val_block_ids=block_ids[p_va],
    )
    cal_eval = calibrate_elkan_noto_posterior(fused_eval, c_hat)
    logger.info("[%s] Pass 4 Elkan-Noto c_hat = %.3f", region_name, c_hat)

    # -------------------------------------------------------------------------
    # Pass 5: Spatial Conformal Risk Bounding (alpha = 0.10 -> 90% Coverage)
    # -------------------------------------------------------------------------
    epistemic_eval = epistemic_full[eval_idx]
    q_hat = spatial_conformal_calibrate(
        cal_probs=cal_eval,
        cal_targets=y_eval.astype(np.float64),
        alpha=0.10,
        cal_block_ids=block_ids[eval_idx],
        train_block_ids=train_blocks,
        epistemic_std=epistemic_eval,
    )

    # Full-grid inference (strictly consistent with Pass 3/4/5 calibration)
    fused_full, _, _ = blend_predictions_by_pr_auc(
        s_xgb_full, s_unet_full, pr_auc_xgb, pr_auc_unet
    )
    cal_prob_full = calibrate_elkan_noto_posterior(fused_full, c_hat)
    lower_L_full, upper_U_full = spatial_conformal_bounds(
        cal_prob_full, q_hat=q_hat, epistemic_std=epistemic_full
    )

    save_prospectivity_geotiff(
        cal_prob_full, rows, cols, H, W, profile, str(maps_dir / "calibrated_prospectivity.tif")
    )
    save_prospectivity_geotiff(
        fused_full, rows, cols, H, W, profile, str(maps_dir / "fused_raw_score.tif")
    )
    save_prospectivity_geotiff(
        e_x, rows, cols, H, W, profile, str(maps_dir / "propensity_e_x.tif")
    )
    save_conformal_bounds_geotiff(
        lower_L_full, upper_U_full, rows, cols, H, W, profile, str(maps_dir / "conformal_bounds_90ci.tif")
    )

    # -------------------------------------------------------------------------
    # Pass 6 & Pass 7: Polygonization, Multi-Criteria Ranking & LLM Site Cards
    # -------------------------------------------------------------------------
    cal_2d = reconstruct_map(cal_prob_full, rows, cols, H, W, nodata=np.nan)
    lower_2d = reconstruct_map(lower_L_full, rows, cols, H, W, nodata=np.nan)
    upper_2d = reconstruct_map(upper_U_full, rows, cols, H, W, nodata=np.nan)
    fused_2d = reconstruct_map(fused_full, rows, cols, H, W, nodata=np.nan)
    ex_2d = reconstruct_map(e_x, rows, cols, H, W, nodata=np.nan)
    wx_2d = reconstruct_map(w_all, rows, cols, H, W, nodata=np.nan)

    # Derive physical summary grids by matching feature names in stack.feature_names
    name_to_idx = {n.lower(): i for i, n in enumerate(stack.feature_names)}

    def _norm_band(arr_2d: np.ndarray, invert: bool = False) -> np.ndarray:
        b = np.nan_to_num(arr_2d.astype(np.float64), nan=float(np.nanmedian(arr_2d)))
        normed = np.clip((b - b.min()) / (np.ptp(b) + 1e-6), 0.0, 1.0)
        return (1.0 - normed) if invert else normed

    # 1. Dilation tendency Td (or inverted fault distance proxy if Td band not in stack)
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

    eff_l_thr = float(l_threshold)
    eff_closing = 3
    eff_min_area = float(min_area_ha)
    if region_name.upper() == "INDONESIA":
        # Mask out non-Indonesian neighbor cells inside the 95E-141E, 11S-6N bounding box
        transform = profile["transform"]
        rr_grid, cc_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        lons_2d = transform.c + (cc_grid + 0.5) * transform.a
        lats_2d = transform.f + (rr_grid + 0.5) * transform.e
        non_id_mask = (
            ((lats_2d > 4.6) & (lons_2d > 119.0))  # Mindanao (Philippines)
            | ((lats_2d > 1.3) & (lons_2d > 99.6) & (lons_2d < 104.4))  # Peninsular Malaysia / SG
            | ((lats_2d > 4.35) & (lons_2d > 113.0) & (lons_2d < 119.2))  # Sabah / North Sarawak
            | ((lats_2d > 1.45) & (lons_2d > 109.6) & (lons_2d < 115.2))  # Sarawak
            | ((lats_2d > -9.55) & (lats_2d < -8.15) & (lons_2d > 124.9) & (lons_2d < 127.35))  # Timor-Leste
        )
        lower_2d = np.where(non_id_mask, np.nan, lower_2d)
        cal_2d = np.where(non_id_mask, np.nan, cal_2d)
        eff_l_thr = max(eff_l_thr, float(np.nanpercentile(lower_2d, 97)))
        eff_closing = 1
        eff_min_area = max(eff_min_area, 2500.0)
    elif region_name.upper() in ("PAPUA", "SUMBAWA", "SUMATRA", "SULAWESI", "KALIMANTAN"):
        eff_l_thr = float(np.clip(np.nanpercentile(lower_2d, 96), 0.35, 0.88))
        eff_closing = 2

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
    if region_name.upper() == "INDONESIA" and len(ranked_gdf) > 25:
        ranked_gdf = ranked_gdf.head(25).reset_index(drop=True)
    elif len(ranked_gdf) > 15:
        ranked_gdf = ranked_gdf.head(15).reset_index(drop=True)

    if len(ranked_gdf) > 0:
        ranked_gdf.to_file(targets_dir / f"ranked_targets_{region_name.lower()}.gpkg", driver="GPKG")

    cards_path = targets_dir / "site_cards.json"
    cards = emit_site_cards_json(
        ranked_gdf,
        out_path=str(cards_path),
        elkan_noto_c_hat=c_hat,
        sensor_source="EMIT_SWIR",
        window_footprint_km=3.2,
    )

    # Save true out-of-fold 5-fold spatial block CV metrics
    cv_rows = []
    for f_idx, (tr_f_idx, va_f_idx) in enumerate(folds):
        tr_f_labels = labels[tr_f_idx]
        va_f_labels = labels[va_f_idx]
        p_tr_f = tr_f_idx[tr_f_labels == LABEL_P]
        rn_tr_f = tr_f_idx[tr_f_labels == LABEL_RN]
        u_tr_f = tr_f_idx[tr_f_labels == LABEL_U]
        p_va_f = va_f_idx[va_f_labels == LABEL_P]
        u_va_f = va_f_idx[va_f_labels != LABEL_P]
        if len(p_tr_f) >= 2 and len(p_va_f) >= 1 and len(u_va_f) >= 10:
            sub_u = u_va_f[: min(600, len(u_va_f))]
            ev_f = np.concatenate([p_va_f, sub_u])
            y_f = np.concatenate([np.ones(len(p_va_f), dtype=int), np.zeros(len(sub_u), dtype=int)])
            if np.array_equal(va_f_idx, val_idx):
                fold_probs = xgb_val_model.predict_proba(X_all[ev_f])
            else:
                fold_model = fit_pu_model(
                    X_all[p_tr_f],
                    X_all[rn_tr_f],
                    X_all[u_tr_f],
                    config,
                    w_p=w_all[p_tr_f],
                    w_rn=w_all[rn_tr_f] if len(rn_tr_f) else None,
                    w_u=w_all[u_tr_f],
                )
                fold_probs = fold_model.predict_proba(X_all[ev_f])
            m_f = score_metrics(y_f, fold_probs)
            m_f["fold"] = f_idx
            cv_rows.append(m_f)
    if not cv_rows:
        m_f = score_metrics(y_eval, cal_eval)
        m_f["fold"] = 0
        cv_rows.append(m_f)
    pd.DataFrame(cv_rows).to_csv(out_dir / "spatial_cv_metrics.csv", index=False)
    # Report mean out-of-fold spatial CV metrics in blind_holdout.json rather than reusing a single tuned fold
    holdout_metrics = {
        k: float(np.mean([r[k] for r in cv_rows]))
        for k in ("pr_auc", "roc_auc", "recall_at_fpr_0.1", "ndcg_at_k_100", "prevalence")
        if k in cv_rows[0]
    }
    (out_dir / "blind_holdout.json").write_text(json.dumps(holdout_metrics, indent=2) + "\n")

    summary = {
        "region": region_name,
        "n_cells": len(df),
        "n_positives": int((labels == LABEL_P).sum()),
        "pr_auc_xgb": round(pr_auc_xgb, 4),
        "pr_auc_unet": round(pr_auc_unet, 4),
        "pr_auc_fused": round(pr_auc_fused, 4),
        "w_xgb": round(w_xgb, 3),
        "w_unet": round(w_unet, 3),
        "elkan_noto_c_hat": round(c_hat, 4),
        "conformal_q_hat_90": round(q_hat, 4),
        "propensity_min": round(float(e_x.min()), 4),
        "propensity_median": round(float(np.median(e_x)), 4),
        "n_ranked_targets": len(ranked_gdf),
        "top_target_card": cards[0] if cards else None,
    }
    (out_dir / "seven_pass_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    summaries = []
    if Path("data/processed/dataset.parquet").exists():
        summaries.append(
            run_region_seven_pass(
                region_name="YILGARN",
                data_dir=Path("data/processed"),
                out_dir=Path("outputs/seven_pass_yilgarn"),
                block_size_m=10_000.0,
                l_threshold=0.50,
                min_area_ha=10.0,
            )
        )
    if Path("data/borneo/processed/dataset.parquet").exists():
        summaries.append(
            run_region_seven_pass(
                region_name="BORNEO",
                data_dir=Path("data/borneo/processed"),
                out_dir=Path("outputs/seven_pass_borneo"),
                block_size_m=2_000.0,
                l_threshold=0.50,
                min_area_ha=5.0,
            )
        )
    if Path("data/indonesia/processed/dataset.parquet").exists():
        summaries.append(
            run_region_seven_pass(
                region_name="INDONESIA",
                data_dir=Path("data/indonesia/processed"),
                out_dir=Path("outputs/seven_pass_indonesia"),
                block_size_m=4.0,
                l_threshold=0.52,
                min_area_ha=15.0,
            )
        )

    print("\n================ 7-PASS TRAINING SUMMARY ================")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
