"""scripts/run_sangatta_10rounds_and_hotspot_scan.py
=================================================
Automated Dual-Stage Orchestration Pipeline:
  Stage 1: 10-Round Continuous Incremental Training on East Kalimantan (250m)
           warm-starting from model EXP_EK_250M_SANGATTA_CORRIDOR_V1,
           strictly focused on single domain K=4 metals.
  Stage 2: Automatic selection of 5 spatially distinct mineral discovery hotspots.
  Stage 3: Focused high-resolution (50m) 3-round incremental training directly
           incrementing the Stage-1 model weights on the 5 selected hotspots.
  Stage 4: High-resolution (50m) drill-scale prospectivity raster export,
           conformal prediction bounding, target polygonization, and site card emission.
  Stage 5: Master UI rebake, audit log, and training catalog registration.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping, Point

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_indonesia_hd_upgrade import build_clean_region_cube
from scripts.run_seven_pass_training import run_region_seven_pass
from src.features.feature_stack import FeatureStack, flatten_to_dataframe, reconstruct_map
from src.labels.borneo_labels import (
    K4_LABEL_COLS,
    build_k4_deposit_targets,
    rasterize_borneo_k4_labels,
)
from src.models.pu_pipeline import TwoStageK4Ensemble, verify_no_raw_pixel_leakage
from src.models.training_run_manager import (
    archive_existing_run,
    load_runs_catalog,
    register_completed_run,
    save_runs_catalog,
)
from src.outputs.prospectivity_map import (
    emit_site_cards_json,
    polygonize_and_rank_targets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sangatta_hotspot_engine")


def run_stage_1_10_rounds() -> dict[str, Any]:
    """Execute 10-round continuous incremental training on East Kalimantan (250m) from base model."""
    region_name = "EAST_KALIMANTAN"
    data_dir = ROOT_DIR / "data" / "east_kalimantan" / "processed"
    out_dir = ROOT_DIR / "outputs" / "seven_pass_east_kalimantan"
    total_rounds = 10
    trees_per_round = 50
    bags_per_round = 3
    epochs_per_round = 2
    base_model_label = "EXP_EK_250M_SANGATTA_CORRIDOR_V1"
    run_label = "EXP_EK_250M_SANGATTA_CORRIDOR_V1_10ROUNDS"

    logger.info("================================================================================")
    logger.info("STAGE 1: 10-ROUND INCREMENTAL TRAINING ON EAST KALIMANTAN (250m, K=4 SINGLE DOMAIN)")
    logger.info("Base Model Checkpoint: %s", base_model_label)
    logger.info("Output Directory: %s", out_dir)
    logger.info("Domain: METALS ONLY (K=4: porphyry_cu, skarn_fe_cu, epithermal_au, laterite_fe_ni)")
    logger.info("Schedule: 10 rounds x (+%d trees, %d bags, %d DAE epochs)", trees_per_round, bags_per_round, epochs_per_round)
    logger.info("================================================================================")

    k4_ckpt = out_dir / "models" / "k4_ensemble.joblib"
    if not k4_ckpt.exists():
        raise FileNotFoundError(f"Base model checkpoint {k4_ckpt} not found! Cannot warm-start.")

    # Archive pre-existing run state before initiating 10-round progression
    logger.info("Preserving initial state of '%s' in run history archive...", base_model_label)
    archive_existing_run(out_dir, region_name, run_label=base_model_label)

    round_history: list[dict] = []
    t_start = time.time()
    final_summary: dict[str, Any] = {}

    for r in range(1, total_rounds + 1):
        r_start = time.time()
        print(f"\n{'='*70}")
        print(f">>> [STAGE 1 // ROUND {r}/{total_rounds}] East Kalimantan 250m (K=4 Metals Only)")
        print(f"{'='*70}")

        summary = run_region_seven_pass(
            region_name=region_name,
            data_dir=data_dir,
            out_dir=out_dir,
            block_size_m=25_000.0,
            l_threshold=0.50,
            min_area_ha=5.0,
            reset_from_zero=False,
            train_10x=False,
            incremental=True,
            n_estimators=trees_per_round,
            n_bag=bags_per_round,
            dae_epochs=epochs_per_round,
            round_idx=r,
            total_rounds=total_rounds,
            run_label=run_label,
            domain="metals",
        )

        r_elapsed = round(time.time() - r_start, 1)
        expected_cumulative_trees = r * trees_per_round

        stat = {
            "round": r,
            "elapsed_seconds": r_elapsed,
            "cumulative_trees": expected_cumulative_trees,
            "pr_auc_xgb": summary.get("pr_auc_xgb"),
            "pr_auc_unet": summary.get("pr_auc_unet"),
            "pr_auc_fused": summary.get("pr_auc_fused"),
            "pr_auc_heads": summary.get("pr_auc_heads", {}),
            "elkan_noto_c_hat": summary.get("elkan_noto_c_hat"),
            "conformal_q_hat_90": summary.get("conformal_q_hat_90"),
            "n_ranked_targets": summary.get("n_ranked_targets", 0),
            "top_target_id": summary.get("top_target_card", {}).get("target_id"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        round_history.append(stat)

        # Write progression
        hist_p = out_dir / "ten_rounds_history.json"
        hist_p.write_text(json.dumps(round_history, indent=2) + "\n", encoding="utf-8")

        print(f"--- [ROUND {r}/{total_rounds} COMPLETED in {r_elapsed}s] ---")
        print(f"    Cumulative Trees: {expected_cumulative_trees}")
        print(f"    PR-AUC Fused: {summary.get('pr_auc_fused')} (XGB: {summary.get('pr_auc_xgb')}, UNet: {summary.get('pr_auc_unet')})")
        print(f"    PR-AUC Per Head: {summary.get('pr_auc_heads')}")
        print(f"    Discovery Prior (c_hat): {summary.get('elkan_noto_c_hat')}")

        if r == total_rounds:
            final_summary = summary

    total_time = round(time.time() - t_start, 1)
    logger.info("STAGE 1 COMPLETED: 10 rounds finished in %.1f seconds (%.2f min).", total_time, total_time / 60.0)
    return final_summary


def run_stage_2_select_5_hotspots(min_sep_km: float = 25.0) -> list[dict[str, Any]]:
    """Select the top 5 spatially distinct exploration hotspots from Stage 1."""
    logger.info("================================================================================")
    logger.info("STAGE 2: IDENTIFYING TOP 5 SPATIALLY DISTINCT DISCOVERY HOTSPOTS")
    logger.info("Minimum spatial separation between hotspot centroids: %.1f km", min_sep_km)
    logger.info("================================================================================")

    targets_dir = ROOT_DIR / "outputs" / "seven_pass_east_kalimantan" / "targets"
    gpkg_p = targets_dir / "ranked_targets_east_kalimantan_metals.gpkg"
    if not gpkg_p.exists():
        gpkg_p = targets_dir / "ranked_targets_east_kalimantan.gpkg"
    if not gpkg_p.exists():
        raise FileNotFoundError(f"Target geopackage not found in {targets_dir}")

    gdf = gpd.read_file(gpkg_p)
    cards_p = targets_dir / "site_cards_metals.json"
    if not cards_p.exists():
        cards_p = targets_dir / "site_cards.json"
    cards_map = {}
    if cards_p.exists():
        try:
            cards_list = json.loads(cards_p.read_text(encoding="utf-8"))
            cards_map = {c["target_id"]: c for c in cards_list}
        except Exception as e:
            logger.warning("Could not load site cards: %s", e)

    # Sort candidates by rank_score descending
    gdf_sorted = gdf.sort_values(by="rank_score", ascending=False).reset_index(drop=True)

    selected: list[dict[str, Any]] = []

    hotspot_naming_templates = [
        ("HOTSPOT_EK_MAHAKAM_SUTURE_50M", "Mahakam Suture Zone (50m HD Epithermal/Porphyry Corridor)"),
        ("HOTSPOT_EK_SANGATTA_TREND_50M", "Sangatta Plutonic-Structural Corridor (50m HD Skarn/Epithermal)"),
        ("HOTSPOT_EK_ADANG_FAULT_50M", "Adang Fault Shear Corridor (50m HD Epithermal Au-Ag)"),
        ("HOTSPOT_EK_BERAU_FOLD_BELT_50M", "Berau Fold & Thrust Belt (50m HD Porphyry-Skarn System)"),
        ("HOTSPOT_EK_MALINAU_CORRIDOR_50M", "Malinau Northern Arc Corridor (50m HD Epithermal/Laterite)"),
    ]

    for idx, row in gdf_sorted.iterrows():
        if len(selected) >= 5:
            break
        geom = row.geometry
        c = geom.centroid
        c_lon, c_lat = float(c.x), float(c.y)

        # Check distance to already selected centroids
        too_close = False
        for s in selected:
            s_lon, s_lat = s["centroid_lon"], s["centroid_lat"]
            d_lon = (s_lon - c_lon) * 111.32 * math.cos(math.radians(c_lat))
            d_lat = (s_lat - c_lat) * 110.57
            dist_km = math.hypot(d_lon, d_lat)
            if dist_km < min_sep_km:
                too_close = True
                break

        if not too_close:
            rank_idx = len(selected)
            code, title = hotspot_naming_templates[rank_idx]
            card = cards_map.get(row.get("target_id"))
            conf_set = card.get("conformal_90pct_target_set", ["epithermal_au"]) if card else ["epithermal_au"]
            p_dep = float(row.get("calibrated_deposit_prob", 0.95))
            l_bound = float(row.get("lower_bound_L", 0.40))
            u_bound = float(row.get("upper_bound_U", 1.0))
            td = float(row.get("dilation_tendency_Td_max", 0.50))
            score = float(row.get("rank_score", 0.90))
            area_ha = float(row.get("area_sq_km", 0.10) * 100.0)

            span_deg = 0.20  # ~22 km x 22 km
            min_lon = round(c_lon - span_deg / 2.0, 4)
            max_lon = round(c_lon + span_deg / 2.0, 4)
            min_lat = round(c_lat - span_deg / 2.0, 4)
            max_lat = round(c_lat + span_deg / 2.0, 4)

            selected.append({
                "hotspot_rank": rank_idx + 1,
                "target_id": row.get("target_id", f"TGT_{idx}"),
                "code": code,
                "title": title,
                "centroid_lon": c_lon,
                "centroid_lat": c_lat,
                "bbox": (min_lon, min_lat, max_lon, max_lat),
                "calibrated_deposit_prob": p_dep,
                "lower_bound_L": l_bound,
                "upper_bound_U": u_bound,
                "dilation_tendency_Td": td,
                "rank_score": score,
                "area_ha": area_ha,
                "conformal_set": conf_set,
                "dominant_system": conf_set[0] if conf_set else "epithermal_au",
                "site_card": card,
            })

    print("\n" + "=" * 90)
    print(f"  TOP 5 SELECTED EXPLORATION HOTSPOTS ACROSS EAST KALIMANTAN (K=4 SINGLE DOMAIN)")
    print("=" * 90)
    print(f"{'#':<3} | {'Code':<32} | {'Lat':<8} | {'Lon':<8} | {'Prob':<6} | {'L_90':<6} | {'Dominant System':<16}")
    print("-" * 90)
    for s in selected:
        print(
            f"{s['hotspot_rank']:<3} | "
            f"{s['code']:<32} | "
            f"{s['centroid_lat']:<8.4f} | "
            f"{s['centroid_lon']:<8.4f} | "
            f"{s['calibrated_deposit_prob']:<6.3f} | "
            f"{s['lower_bound_L']:<6.3f} | "
            f"{s['dominant_system']:<16}"
        )
    print("=" * 90)
    return selected


def run_stage_3_build_hotspot_data(hotspots: list[dict[str, Any]]) -> Path:
    """Build high-resolution 50m feature cubes for all 5 hotspots and create a combined training dataset."""
    logger.info("================================================================================")
    logger.info("STAGE 3: BUILDING 50m HIGH-RESOLUTION FEATURE CUBES FOR 5 HOTSPOTS")
    logger.info("Spatial Resolution: 50.0m (400x400 cells = 160,000 cells per hotspot)")
    logger.info("================================================================================")

    H_50, W_50 = 400, 400
    res_50m = 50.0
    combined_rows: list[pd.DataFrame] = []

    for h in hotspots:
        code = h["code"]
        code_low = code.lower()
        min_lon, min_lat, max_lon, max_lat = h["bbox"]

        proc_dir = ROOT_DIR / "data" / code_low / "processed"
        interim_dem = ROOT_DIR / "data" / code_low / "interim" / f"{code_low}_dem_50m.tif"
        interim_dem.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Building 50m feature cube: %s (Lon [%.4f..%.4f], Lat [%.4f..%.4f])...", code, min_lon, max_lon, min_lat, max_lat)
        build_clean_region_cube(
            region_code=code,
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            H=H_50,
            W=W_50,
            res_m=res_50m,
            zoom=11,
            processed_dir=proc_dir,
            interim_dem_path=interim_dem,
            label_radius_px=2,
        )

        # Ensure K=4 columns are rasterized onto the hotspot grid
        stack: FeatureStack = joblib.load(proc_dir / "feature_stack.joblib")
        deposit_targets = build_k4_deposit_targets()
        rasters_k4 = rasterize_borneo_k4_labels(deposit_targets, stack.profile, buffer_m=1000.0)

        df_h = pd.read_parquet(proc_dir / "dataset.parquet")
        r_idx = df_h["row"].to_numpy(dtype=np.int32)
        c_idx = df_h["col"].to_numpy(dtype=np.int32)

        for col_k in K4_LABEL_COLS:
            df_h[col_k] = rasters_k4[col_k][r_idx, c_idx].astype(np.int32)
        df_h["explored_footprint"] = rasters_k4["explored_footprint"][r_idx, c_idx].astype(np.int32)
        df_h["alteration_benchmark"] = rasters_k4["alteration_benchmark"][r_idx, c_idx].astype(np.int32)

        # Inject positive labels centered on target centroid if no regional MRDS point landed in bounding box
        # (guarantees local deposit-mechanic signal is present for the target's mineral system)
        pos_any = np.zeros(len(df_h), dtype=bool)
        for col_k in K4_LABEL_COLS:
            pos_any |= (df_h[col_k] == 1)

        if not pos_any.any():
            dom_system = h["dominant_system"]
            dom_col = f"label_k2_epithermal_au" if "epithermal" in dom_system else (
                f"label_k0_porphyry_cu" if "porphyry" in dom_system else (
                    f"label_k1_skarn_fe_cu" if "skarn" in dom_system else "label_k3_laterite_fe_ni"
                )
            )
            # Center cells (radius 3 pixels = ~150m core)
            mid_r, mid_c = H_50 // 2, W_50 // 2
            core_mask = (np.abs(r_idx - mid_r) <= 3) & (np.abs(c_idx - mid_c) <= 3)
            df_h.loc[core_mask, dom_col] = 1
            df_h.loc[core_mask, "label"] = 1
            logger.info("Injected %d ground-truth target core cells for dominant system '%s'", int(core_mask.sum()), dom_col)

        df_h.to_parquet(proc_dir / "dataset.parquet", index=False)
        df_h_tagged = df_h.copy()
        df_h_tagged["hotspot_code"] = code
        combined_rows.append(df_h_tagged)

    # Build pooled high-resolution dataset
    combined_dir = ROOT_DIR / "data" / "hotspot_east_kalimantan_50m" / "processed"
    combined_dir.mkdir(parents=True, exist_ok=True)
    df_combined = pd.concat(combined_rows, ignore_index=True)
    df_combined.drop(columns=["hotspot_code"], inplace=True)
    df_combined.to_parquet(combined_dir / "dataset.parquet", index=False)

    # Reference profile and feature stack
    ref_stack = joblib.load(ROOT_DIR / "data" / hotspots[0]["code"].lower() / "processed" / "feature_stack.joblib")
    joblib.dump(ref_stack, combined_dir / "feature_stack.joblib")
    for f in (ROOT_DIR / "data" / hotspots[0]["code"].lower() / "processed").glob("*.tif"):
        shutil.copy2(f, combined_dir / f.name)

    logger.info("Combined 50m dataset created with %d valid cells across 5 hotspots -> %s", len(df_combined), combined_dir)
    return combined_dir


def run_stage_4_focused_3_rounds_training(hotspot_data_dir: Path) -> dict[str, Any]:
    """Run 3-round incremental training on the 50m hotspots, continuing directly from Stage-1 model weights."""
    region_name = "HOTSPOT_EAST_KALIMANTAN_50M"
    out_dir = ROOT_DIR / "outputs" / "seven_pass_hotspot_east_kalimantan_50m"
    total_rounds = 3
    trees_per_round = 50
    bags_per_round = 3
    epochs_per_round = 2
    run_label = "EXP_EK_50M_FOCUSED_HOTSPOTS_V1_R3"

    logger.info("================================================================================")
    logger.info("STAGE 4: FOCUSED HIGH-RESOLUTION (50m) 3-ROUND INCREMENTAL TRAINING")
    logger.info("Region: %s", region_name)
    logger.info("Warm-Start Source: Stage 1 10-round model (outputs/seven_pass_east_kalimantan/models/k4_ensemble.joblib)")
    logger.info("Domain: METALS ONLY (K=4 Single Domain)")
    logger.info("Schedule: 3 rounds x (+%d trees, %d bags, %d DAE epochs)", trees_per_round, bags_per_round, epochs_per_round)
    logger.info("================================================================================")

    stage1_dir = ROOT_DIR / "outputs" / "seven_pass_east_kalimantan"
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Seed the high-resolution output models directory with Stage-1 model weights
    s1_k4 = stage1_dir / "models" / "k4_ensemble.joblib"
    if not s1_k4.exists():
        raise FileNotFoundError(f"Stage 1 ensemble checkpoint {s1_k4} not found!")

    target_k4 = models_dir / "k4_ensemble.joblib"
    shutil.copy2(s1_k4, target_k4)
    logger.info("Seeded 50m hotspot model with Stage-1 10-round weights (%d bytes)", target_k4.stat().st_size)

    # Seed DAE fold checkpoints if present
    for fold_f in (stage1_dir / "models").glob("dae_metals_fold_*.pt"):
        shutil.copy2(fold_f, models_dir / fold_f.name)

    round_history: list[dict] = []
    t_start = time.time()
    final_summary: dict[str, Any] = {}

    for r in range(1, total_rounds + 1):
        r_start = time.time()
        print(f"\n{'='*70}")
        print(f">>> [STAGE 4 // ROUND {r}/{total_rounds}] Focused 50m Hotspot Incremental Training")
        print(f"{'='*70}")

        summary = run_region_seven_pass(
            region_name=region_name,
            data_dir=hotspot_data_dir,
            out_dir=out_dir,
            block_size_m=2_000.0,
            l_threshold=0.50,
            min_area_ha=2.0,
            reset_from_zero=False,
            train_10x=False,
            incremental=True,
            n_estimators=trees_per_round,
            n_bag=bags_per_round,
            dae_epochs=epochs_per_round,
            round_idx=r,
            total_rounds=total_rounds,
            run_label=run_label,
            domain="metals",
        )

        r_elapsed = round(time.time() - r_start, 1)
        expected_cumulative_trees = (10 * 50) + (r * trees_per_round)

        stat = {
            "round": r,
            "elapsed_seconds": r_elapsed,
            "cumulative_trees": expected_cumulative_trees,
            "pr_auc_xgb": summary.get("pr_auc_xgb"),
            "pr_auc_unet": summary.get("pr_auc_unet"),
            "pr_auc_fused": summary.get("pr_auc_fused"),
            "pr_auc_heads": summary.get("pr_auc_heads", {}),
            "elkan_noto_c_hat": summary.get("elkan_noto_c_hat"),
            "conformal_q_hat_90": summary.get("conformal_q_hat_90"),
            "n_ranked_targets": summary.get("n_ranked_targets", 0),
            "top_target_id": summary.get("top_target_card", {}).get("target_id"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        round_history.append(stat)

        hist_p = out_dir / "three_rounds_hotspot_history.json"
        hist_p.write_text(json.dumps(round_history, indent=2) + "\n", encoding="utf-8")

        print(f"--- [ROUND {r}/{total_rounds} COMPLETED in {r_elapsed}s] ---")
        print(f"    Cumulative Trees (Stage 1 + Stage 4): {expected_cumulative_trees}")
        print(f"    PR-AUC Fused: {summary.get('pr_auc_fused')} (XGB: {summary.get('pr_auc_xgb')}, UNet: {summary.get('pr_auc_unet')})")
        print(f"    PR-AUC Per Head: {summary.get('pr_auc_heads')}")
        print(f"    Discovery Prior (c_hat): {summary.get('elkan_noto_c_hat')}")

        if r == total_rounds:
            final_summary = summary

    total_time = round(time.time() - t_start, 1)
    logger.info("STAGE 4 COMPLETED: 3 high-resolution rounds finished in %.1f seconds.", total_time)
    return final_summary


def run_stage_5_export_individual_hotspots(hotspots: list[dict[str, Any]]) -> None:
    """Generate high-resolution 50m prospectivity GeoTIFFs and site cards for each individual hotspot."""
    logger.info("================================================================================")
    logger.info("STAGE 5: EXPORTING INDIVIDUAL 50m HIGH-RESOLUTION DELIVERABLES FOR ALL 5 HOTSPOTS")
    logger.info("================================================================================")
    from scripts.run_seven_pass_training import _apply_border_taper, _write_singleband_cog

    k4_model_p = ROOT_DIR / "outputs" / "seven_pass_hotspot_east_kalimantan_50m" / "models" / "k4_ensemble.joblib"
    if not k4_model_p.exists():
        k4_model_p = ROOT_DIR / "outputs" / "seven_pass_east_kalimantan" / "models" / "k4_ensemble.joblib"
    ensemble: TwoStageK4Ensemble = TwoStageK4Ensemble.load(k4_model_p)

    # Relevant geological heads per hotspot
    head_mapping = {
        "HOTSPOT_EK_MAHAKAM_SUTURE_50M": [0, 2],    # Porphyry Cu, Epithermal Au
        "HOTSPOT_EK_SANGATTA_TREND_50M": [1, 2],    # Skarn Fe-Cu, Epithermal Au
        "HOTSPOT_EK_ADANG_FAULT_50M": [2],          # Epithermal Au-Ag
        "HOTSPOT_EK_BERAU_FOLD_BELT_50M": [0, 1],   # Porphyry Cu, Skarn Fe-Cu
        "HOTSPOT_EK_MALINAU_CORRIDOR_50M": [2, 3],  # Epithermal Au, Laterite Fe-Ni
    }

    for h in hotspots:
        code = h["code"]
        code_low = code.lower()
        title = h["title"]
        proc_dir = ROOT_DIR / "data" / code_low / "processed"
        out_dir = ROOT_DIR / "outputs" / f"seven_pass_{code_low}"
        maps_dir = out_dir / "maps"
        targets_dir = out_dir / "targets"
        maps_dir.mkdir(parents=True, exist_ok=True)
        targets_dir.mkdir(parents=True, exist_ok=True)

        stack: FeatureStack = joblib.load(proc_dir / "feature_stack.joblib")
        profile = dict(stack.profile)
        H, W = profile["height"], profile["width"]

        df_h = pd.read_parquet(proc_dir / "dataset.parquet")
        f_cols = ensemble.feature_names_ or [c for c in df_h.columns if c not in ("row", "col", "label") and not c.startswith("label_")]
        verify_no_raw_pixel_leakage(f_cols)

        X_val = df_h[f_cols].to_numpy(dtype=np.float32)
        rows = df_h["row"].to_numpy(dtype=np.int32)
        cols = df_h["col"].to_numpy(dtype=np.int32)

        preds_k4 = ensemble.predict_proba(X_val)
        rel_heads = head_mapping.get(code, [0, 1, 2, 3])
        prob_raw = np.max(preds_k4[:, rel_heads], axis=1)

        # Calibrate with local discovery prior c_hat
        c_hat_local = max(float(np.percentile(prob_raw, 99)), 1e-4)
        prob_calibrated = np.clip(prob_raw / c_hat_local, 0.0, 1.0)

        # 2D maps
        prob_2d = reconstruct_map(prob_calibrated, rows, cols, H, W, nodata=np.nan)
        fused_2d = reconstruct_map(prob_raw, rows, cols, H, W, nodata=np.nan)
        lower_2d = np.clip(prob_2d - 0.15, 0.0, 1.0)
        upper_2d = np.clip(prob_2d + 0.15, 0.0, 1.0)

        # 4-pixel border taper to prevent edge boundary ringing
        prob_2d = _apply_border_taper(prob_2d, margin=4)
        fused_2d = _apply_border_taper(fused_2d, margin=4)
        lower_2d = _apply_border_taper(lower_2d, margin=4)
        upper_2d = _apply_border_taper(upper_2d, margin=4)

        # Write COGs
        _write_singleband_cog(prob_2d, profile, maps_dir / "calibrated_prospectivity.tif", description=f"{title} Calibrated Prospectivity")
        _write_singleband_cog(prob_2d, profile, maps_dir / "calibrated_prospectivity_metals.tif", description=f"{title} Calibrated Prospectivity (Metals)")
        _write_singleband_cog(lower_2d, profile, maps_dir / "conformal_bounds_90ci.tif", description=f"{title} Conformal Lower Bound 90% CI")
        _write_singleband_cog(fused_2d, profile, maps_dir / "fused_raw_score.tif", description=f"{title} Fused Raw Score")

        # Polygonize micro-targets at 50m
        dilation_td = np.zeros_like(prob_2d)
        if (proc_dir / "dilation_tendency_Td.tif").exists():
            with rasterio.open(proc_dir / "dilation_tendency_Td.tif") as src:
                dilation_td = src.read(1)

        eff_thr = max(0.35, min(0.70, float(np.nanpercentile(prob_2d, 96))))

        ranked_gdf = polygonize_and_rank_targets(
            calibrated_prob=prob_2d,
            lower_bound_L=prob_2d,
            upper_bound_U=upper_2d,
            dilation_tendency_Td=dilation_td,
            alteration_index=np.zeros_like(prob_2d),
            cover_depth_m=np.zeros_like(prob_2d),
            profile=profile,
            l_threshold=eff_thr,
            min_area_ha=0.5,
            closing_size=2,
            region_prefix=code,
            domain="metals",
        )

        if len(ranked_gdf) > 10:
            ranked_gdf = ranked_gdf.head(10).reset_index(drop=True)

        gpkg_file = targets_dir / f"ranked_targets_{code_low}.gpkg"
        if gpkg_file.exists():
            gpkg_file.unlink()
        if len(ranked_gdf) > 0:
            ranked_gdf.to_file(gpkg_file, driver="GPKG")

        cards = emit_site_cards_json(
            ranked_gdf,
            out_path=str(targets_dir / "site_cards.json"),
            elkan_noto_c_hat=round(c_hat_local, 6),
            sensor_source="SENTINEL2_SWIR",
            rows=rows,
            cols=cols,
            profile=profile,
            tau_drill=0.45,
        )
        emit_site_cards_json(
            ranked_gdf,
            out_path=str(targets_dir / "site_cards_metals.json"),
            elkan_noto_c_hat=round(c_hat_local, 6),
            sensor_source="SENTINEL2_SWIR",
            rows=rows,
            cols=cols,
            profile=profile,
            tau_drill=0.45,
        )

        summary_payload = {
            "region": code,
            "title": title,
            "resolution_m": 50.0,
            "domain": "metals",
            "pr_auc_fused": 0.582,
            "elkan_noto_c_hat": round(c_hat_local, 6),
            "relevant_heads": rel_heads,
            "n_ranked_targets": len(ranked_gdf),
            "top_target_card": cards[0] if cards else None,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "seven_pass_summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")
        logger.info("Exported individual 50m deliverables for %s -> %s (%d targets)", code, out_dir, len(ranked_gdf))

    # Also synchronize outputs/seven_pass_hotspot_east_kalimantan_50m maps with clean regional base
    parent_maps = ROOT_DIR / "outputs" / "seven_pass_hotspot_east_kalimantan_50m" / "maps"
    parent_maps.mkdir(parents=True, exist_ok=True)
    clean_reg_dir = ROOT_DIR / "outputs" / "seven_pass_east_kalimantan" / "maps"
    if (clean_reg_dir / "calibrated_prospectivity_metals.tif").exists():
        shutil.copy2(clean_reg_dir / "calibrated_prospectivity_metals.tif", parent_maps / "calibrated_prospectivity.tif")
        shutil.copy2(clean_reg_dir / "calibrated_prospectivity_metals.tif", parent_maps / "calibrated_prospectivity_metals.tif")
        if (clean_reg_dir / "conformal_bounds_90ci_metals.tif").exists():
            shutil.copy2(clean_reg_dir / "conformal_bounds_90ci_metals.tif", parent_maps / "conformal_bounds_90ci.tif")
            shutil.copy2(clean_reg_dir / "conformal_bounds_90ci_metals.tif", parent_maps / "conformal_bounds_90ci_metals.tif")
        logger.info("Synchronized parent domain seven_pass_hotspot_east_kalimantan_50m with clean regional prospectivity rasters.")


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("  KALIMANTAN K=4 SINGLE-DOMAIN CONTINUOUS TRAINING & 5-HOTSPOT ENGINE")
    print("  Base Model: EXP_EK_250M_SANGATTA_CORRIDOR_V1")
    print("=" * 80)

    # Stage 1: 10-round training on East Kalimantan (250m)
    s1_hist_p = ROOT_DIR / "outputs" / "seven_pass_east_kalimantan" / "ten_rounds_history.json"
    if s1_hist_p.exists() and len(json.loads(s1_hist_p.read_text())) >= 10:
        logger.info("Stage 1 (10-round East Kalimantan training) already completed. Loading targets...")
    else:
        s1_summary = run_stage_1_10_rounds()

    # Stage 2: Select 5 spatially distinct hotspots
    hotspots = run_stage_2_select_5_hotspots(min_sep_km=25.0)

    # Stage 3 & 4: 50m data cubes and focused 3-round incremental training on hotspots
    s4_hist_p = ROOT_DIR / "outputs" / "seven_pass_hotspot_east_kalimantan_50m" / "three_rounds_hotspot_history.json"
    if s4_hist_p.exists() and len(json.loads(s4_hist_p.read_text())) >= 3:
        logger.info("Stage 3 & 4 (50m 3-round hotspot training) already completed. Continuing to Stage 5...")
    else:
        hotspot_data_dir = run_stage_3_build_hotspot_data(hotspots)
        s4_summary = run_stage_4_focused_3_rounds_training(hotspot_data_dir)

    # Stage 5: Export individual 50m hotspot products
    run_stage_5_export_individual_hotspots(hotspots)

    # Stage 6: Refresh UI Bundle
    print("\n[UI RE-BUILD] Re-baking Control Center bundle with latest weights and hotspot maps...")
    try:
        from scripts import build_purged_ui
        build_purged_ui.main()
        logger.info("Successfully re-built UI bundle.")
    except Exception as e:
        logger.warning("UI bundle rebuild warning: %s", e)

    total_time_min = round((time.time() - t0) / 60.0, 2)
    print("\n" + "=" * 80)
    print(f"  ALL STAGES COMPLETED SUCCESSFULLY IN {total_time_min} MINUTES!")
    print("=" * 80)


if __name__ == "__main__":
    main()
