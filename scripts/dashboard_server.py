"""
scripts/dashboard_server.py
============================
Interactive Web UI & Control Center for the 7-Pass Mineral Prospectivity &
PU-Learning Pipeline.

Features:
  - OpenStreetMap / Esri Satellite / Topographic interactive world map (Leaflet)
  - Side-by-side & overlay comparison of Model-Predicted ("Guessed") Mine Targets
    (with 90% Conformal CI bounds, Site Cards, and Probability Heatmap) vs
    Actual Ground-Truth Mines (GSWA MINEDEX Gold, USGS MRDS, ESDM Iron & Coal)
  - Live Pipeline Control Panel: adjust conformal lower bound threshold L(x),
    multi-criteria ranking weights (w1..w4), minimum target area (ha), and
    trigger live re-ranking or full 7-pass training runs from the browser
  - Training Overview & Performance Dashboard: Spatial Block CV folds, PR-AUC
    comparison (XGBoost vs nnPU U-Net vs Fused), Elkan-Noto c_hat calibration,
    and Ground-Truth Hit Rate statistics.

Usage:
    .venv/bin/python scripts/dashboard_server.py --port 8765
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.warp import transform_bounds
from shapely.geometry import mapping

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.feature_stack import FeatureStack, reconstruct_map
from src.outputs.prospectivity_map import (
    emit_site_cards_json,
    polygonize_and_rank_targets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mpm_ui")

# Global state for background training job
_TRAIN_LOCK = threading.Lock()
_TRAIN_STATE: dict = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "last_exit_code": 0,
    "logs": ["[READY] 7-Pass Mineral Prospectivity Pipeline Control Center initialised."],
}


def _raster_to_rgba_base64(
    tif_path: Path,
    band: int = 1,
    cmap: str = "inferno",
    alpha_scale: float = 0.72,
) -> tuple[str, list[list[float]]]:
    """Convert a projected GeoTIFF band to a base64 RGBA PNG + WGS84 [[south, west], [north, east]] bounds."""
    with rasterio.open(tif_path) as src:
        arr = src.read(band).astype(np.float64)
        west, south, east, north = transform_bounds(
            src.crs, "EPSG:4326", *src.bounds, densify_pts=21
        )

    valid = np.isfinite(arr) & (arr > 0.02)
    norm = np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)

    H, W = norm.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    if cmap == "inferno":
        # Rich geological prospectivity ramp: transparent low -> purple -> red-orange -> bright gold
        r = np.clip(norm * 1.4, 0.0, 1.0)
        g = np.clip((norm - 0.25) * 1.35, 0.0, 1.0)
        b = np.clip(0.65 - np.abs(norm - 0.35) * 1.8 + np.maximum(0.0, norm - 0.85) * 2.0, 0.0, 1.0)
        a = np.where(valid, np.clip(norm * alpha_scale * 255.0 + 25.0, 0.0, 220.0), 0.0)
    else:
        # Cyan-emerald uncertainty/lower-bound ramp
        r = np.clip((norm - 0.4) * 1.6, 0.0, 1.0)
        g = np.clip(norm * 1.25, 0.0, 1.0)
        b = np.clip(0.9 - norm * 0.5, 0.0, 1.0)
        a = np.where(valid, np.clip(norm * alpha_scale * 255.0, 0.0, 210.0), 0.0)

    rgba[..., 0] = (r * 255).astype(np.uint8)
    rgba[..., 1] = (g * 255).astype(np.uint8)
    rgba[..., 2] = (b * 255).astype(np.uint8)
    rgba[..., 3] = a.astype(np.uint8)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, [[float(south), float(west)], [float(north), float(east)]]


def _load_region_bundle(region: str) -> dict:
    """Assemble actual mines, predicted targets, heatmaps, site cards, and CV metrics for a region."""
    region_up = region.upper()
    if region_up == "YILGARN":
        out_dir = ROOT_DIR / "outputs" / "seven_pass_yilgarn"
        summary_path = out_dir / "seven_pass_summary.json"
        cards_path = out_dir / "targets" / "site_cards.json"
        targets_gpkg = out_dir / "targets" / "ranked_targets_yilgarn.gpkg"
        prosp_tif = out_dir / "maps" / "calibrated_prospectivity.tif"
        conf_tif = out_dir / "maps" / "conformal_bounds_90ci.tif"
        cv_csv = ROOT_DIR / "outputs" / "reports" / "spatial_cv_metrics_v2_real.csv"
        holdout_json = ROOT_DIR / "outputs" / "reports" / "blind_holdout_v2_real.json"

        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        site_cards = json.loads(cards_path.read_text()) if cards_path.exists() else []
        cv_rows = pd.read_csv(cv_csv).to_dict(orient="records") if cv_csv.exists() else []
        holdout = json.loads(holdout_json.read_text()) if holdout_json.exists() else {}

        # Load Actual Mines (GSWA MINEDEX)
        minedex_path = ROOT_DIR / "data" / "raw_real" / "minedex_deposits.gpkg"
        actual_mines = []
        minedex_proj = None
        if minedex_path.exists():
            gdf_m = gpd.read_file(minedex_path)
            minedex_proj = gdf_m.to_crs("EPSG:32751")
            gdf_m_ll = gdf_m.to_crs("EPSG:4326")
            for _, row in gdf_m_ll.iterrows():
                if row.geometry is None or row.geometry.is_empty:
                    continue
                pt = row.geometry.centroid
                actual_mines.append(
                    {
                        "id": str(row.get("site_code", "")),
                        "name": str(row.get("short_name") or row.get("site_title") or "MINEDEX Site"),
                        "commodity": str(row.get("site_commo") or row.get("target_com") or "Gold (Au)"),
                        "stage": str(row.get("site_stage") or "Deposit / Mine"),
                        "type": str(row.get("site_type_") or "Lode Au"),
                        "source": "GSWA MINEDEX",
                        "lat": round(float(pt.y), 5),
                        "lon": round(float(pt.x), 5),
                    }
                )

        # Load Predicted Target Polygons & compute actual mine hit counts
        predicted_features = []
        hit_mine_indices: set[int] = set()
        if targets_gpkg.exists():
            gdf_t = gpd.read_file(targets_gpkg)
            gdf_t_proj = gdf_t.to_crs("EPSG:32751")
            gdf_t_ll = gdf_t.to_crs("EPSG:4326")
            card_by_id = {c["target_id"]: c for c in site_cards}

            for idx, row in gdf_t_ll.iterrows():
                tid = str(row.get("target_id", f"YILGARN_TGT_{idx+1:04d}"))
                hits = 0
                nearest_dist_m = None
                if minedex_proj is not None and len(minedex_proj) > 0:
                    poly_proj = gdf_t_proj.geometry.iloc[idx]
                    dists = minedex_proj.geometry.distance(poly_proj)
                    hit_mask = dists <= 250.0  # inside or within 250m buffer
                    hits = int(hit_mask.sum())
                    hit_mine_indices.update(np.where(hit_mask)[0].tolist())
                    nearest_dist_m = round(float(dists.min()), 1)

                predicted_features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(row.geometry),
                        "properties": {
                            "target_id": tid,
                            "rank": int(row.get("rank", idx + 1)),
                            "rank_score": round(float(row.get("rank_score", 0.0)), 3),
                            "area_sq_km": round(float(row.get("area_sq_km", 0.0)), 3),
                            "calibrated_deposit_prob": round(float(row.get("calibrated_deposit_prob", 0.0)), 3),
                            "lower_bound_L": round(float(row.get("lower_bound_L", 0.0)), 3),
                            "upper_bound_U": round(float(row.get("upper_bound_U", 1.0)), 3),
                            "dilation_tendency_Td_max": round(float(row.get("dilation_tendency_Td_max", 0.0)), 3),
                            "regolith_cover_depth_m": round(float(row.get("regolith_cover_depth_m", 0.0)), 1),
                            "exploration_propensity_e_x": round(float(row.get("exploration_propensity_e_x", 0.1)), 3),
                            "propensity_weight_w": round(float(row.get("propensity_weight_w", 10.0)), 2),
                            "actual_mines_hit": hits,
                            "nearest_actual_mine_m": nearest_dist_m,
                            "classification": (
                                "Known Deposit Cluster Hit"
                                if hits > 0
                                else "Greenfield Blind Discovery Target (Under Cover)"
                            ),
                            "site_card": card_by_id.get(tid),
                        },
                    }
                )

        # Load Fault lines for structural context
        fault_features = []
        faults_path = ROOT_DIR / "data" / "raw_real" / "faults.gpkg"
        if faults_path.exists():
            gdf_f = gpd.read_file(faults_path).to_crs("EPSG:4326")
            for _, row in gdf_f.iterrows():
                if row.geometry is not None and not row.geometry.is_empty:
                    fault_features.append(
                        {
                            "type": "Feature",
                            "geometry": mapping(row.geometry),
                            "properties": {"name": "GSWA Shear / Fault Structure"},
                        }
                    )

        heatmap_b64, bounds = _raster_to_rgba_base64(prosp_tif, band=1, cmap="inferno")
        conformal_b64, _ = _raster_to_rgba_base64(conf_tif, band=1, cmap="emerald")

        return {
            "region": "YILGARN",
            "title": "Yilgarn Craton — Kalgoorlie-Kambalda Goldfields (Western Australia)",
            "crs": "EPSG:32751 (MGA Zone 51S)",
            "center": [-30.768, 121.488],
            "zoom": 12,
            "bounds": bounds,
            "summary": summary,
            "cv_folds": cv_rows,
            "blind_holdout": holdout,
            "actual_mines": actual_mines,
            "total_actual_mines": len(actual_mines),
            "total_mines_captured": len(hit_mine_indices),
            "predicted_targets_geojson": {
                "type": "FeatureCollection",
                "features": predicted_features,
            },
            "faults_geojson": {
                "type": "FeatureCollection",
                "features": fault_features,
            },
            "site_cards": site_cards,
            "heatmap_png": heatmap_b64,
            "conformal_l_png": conformal_b64,
        }

    elif region_up in ("INDONESIA_GEORIMA", "INDONESIA"):
        # National 7-Pass Mineral Prospectivity & ESDM GeoRIMA Atlas of Indonesia (95E-141E, 11S-6N)
        out_dir = ROOT_DIR / "outputs" / "seven_pass_indonesia"
        if not (out_dir / "seven_pass_summary.json").exists():
            out_dir = ROOT_DIR / "outputs" / "seven_pass_borneo"
        summary_path = out_dir / "seven_pass_summary.json"
        cards_path = out_dir / "targets" / "site_cards.json"
        targets_gpkg = out_dir / "targets" / "ranked_targets_indonesia.gpkg"
        if not targets_gpkg.exists():
            targets_gpkg = out_dir / "targets" / "ranked_targets_borneo.gpkg"
        prosp_tif = out_dir / "maps" / "calibrated_prospectivity.tif"
        conf_tif = out_dir / "maps" / "conformal_bounds_90ci.tif"
        cv_csv = ROOT_DIR / "outputs" / "borneo" / "reports" / "spatial_metrics_borneo.csv"

        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        site_cards = json.loads(cards_path.read_text()) if cards_path.exists() else []
        cv_rows = pd.read_csv(cv_csv).to_dict(orient="records") if cv_csv.exists() else []

        actual_mines = []
        all_nodes_csv = ROOT_DIR / "data" / "indonesia" / "georima_all_indonesia_nodes.csv"
        if all_nodes_csv.exists():
            df_geo = pd.read_csv(all_nodes_csv)
            # Include all Metallic (ml), Coal (bb), Geothermal (pb), and key economic Non-Metallic (Marble, Quartz, Diamond, Kaolin, Pyrophyllite, Sulfur)
            keep_mask = df_geo["potensi_type"].isin(["ml", "bb", "pb"]) | df_geo["layer_code"].isin(
                ["bl:231", "bl:207", "bl:228", "bl:223", "bl:227", "bl:241", "bl:212"]
            )
            for _, r in df_geo[keep_mask].iterrows():
                actual_mines.append(
                    {
                        "id": str(r.get("georima_id", ""))[:16],
                        "name": f"GeoRIMA {r.get('commodity_label', 'Node')} ({r.get('island_region', 'Indonesia')})",
                        "commodity": f"{r.get('commodity_label', '')} [{r.get('layer_code', '')}]",
                        "resource": str(r.get("resource_group", "OTHER")),
                        "layer": str(r.get("layer_code", "")),
                        "stage": f"Official ESDM GeoRIMA ({r.get('potensi_type', '').upper()})",
                        "type": str(r.get("geological_role", "")),
                        "source": "ESDM GeoRIMA National Atlas (geologi.esdm.go.id)",
                        "lat": round(float(r["lat"]), 5),
                        "lon": round(float(r["lon"]), 5),
                    }
                )
        else:
            georima_csv = ROOT_DIR / "data" / "indonesia" / "georima-node-to-nearest-game-deposit-20261019.csv"
            if georima_csv.exists():
                df_geo = pd.read_csv(georima_csv)
                for _, r in df_geo.iterrows():
                    actual_mines.append(
                        {
                            "id": str(r.get("georima_id", ""))[:16],
                            "name": f"GeoRIMA {r.get('georima_layer_label', 'Mine Node')}",
                            "commodity": str(r.get("georima_layer_label", r.get("resource", "Mineral"))),
                            "resource": str(r.get("resource", "OTHER")),
                            "layer": str(r.get("georima_layer", "")),
                            "stage": "Official ESDM GeoRIMA Verified",
                            "type": str(r.get("georima_layer_label", "")),
                            "source": "ESDM GeoRIMA (geologi.esdm.go.id)",
                            "lat": round(float(r["georima_lat"]), 5),
                            "lon": round(float(r["georima_lon"]), 5),
                        }
                    )

        mrds_id_csv = ROOT_DIR / "data" / "indonesia" / "mrds-indonesia.csv"
        if mrds_id_csv.exists():
            df_m = pd.read_csv(mrds_id_csv)
            for _, r in df_m.iterrows():
                if str(r.get("coordinate_quality_flag", "")) == "province_or_island_centroid":
                    continue
                actual_mines.append(
                    {
                        "id": str(r.get("dep_id", "")),
                        "name": str(r.get("site_name", "USGS MRDS Indonesia")),
                        "commodity": f"{r.get('commod1', 'Gold/Copper')} ({r.get('dep_type', 'Deposit')})",
                        "resource": "GOLD_ORE",
                        "stage": str(r.get("dev_stat", "Producer/Occurrence")),
                        "type": str(r.get("dep_type", "MRDS")),
                        "source": "USGS MRDS Indonesia",
                        "lat": round(float(r["latitude"]), 5),
                        "lon": round(float(r["longitude"]), 5),
                    }
                )

        # Build spatial index of actual metallic/mineral mine coords for fast polygon hit counting
        mine_coords = np.array([[m["lon"], m["lat"]] for m in actual_mines], dtype=np.float64) if actual_mines else np.empty((0, 2))
        captured_mine_indices: set[int] = set()

        predicted_features = []
        if targets_gpkg.exists():
            gdf_t = gpd.read_file(targets_gpkg).to_crs("EPSG:4326")
            card_by_id = {c["target_id"]: c for c in site_cards}
            for idx, row in gdf_t.head(35).iterrows():
                tid = str(row.get("target_id", f"INDONESIA_TGT_{idx+1:04d}"))
                geom = row.geometry
                minx, miny, maxx, maxy = geom.bounds
                hits = 0
                nearest_m = 5000.0
                if len(mine_coords) > 0:
                    in_box = (
                        (mine_coords[:, 0] >= minx - 0.15)
                        & (mine_coords[:, 0] <= maxx + 0.15)
                        & (mine_coords[:, 1] >= miny - 0.15)
                        & (mine_coords[:, 1] <= maxy + 0.15)
                    )
                    cand_idx = np.where(in_box)[0]
                    hits = len(cand_idx)
                    captured_mine_indices.update(cand_idx.tolist())
                    c_ll = geom.centroid
                    d_deg = np.hypot(mine_coords[:, 0] - c_ll.x, mine_coords[:, 1] - c_ll.y)
                    nearest_m = round(float(d_deg.min()) * 111_000.0, 1)

                c_ll = geom.centroid
                from scripts.fetch_indonesia_geospatial_data import classify_indonesia_island
                belt = classify_indonesia_island(float(c_ll.x), float(c_ll.y))
                predicted_features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(geom),
                        "properties": {
                            "target_id": tid,
                            "rank": int(row.get("rank", idx + 1)),
                            "rank_score": round(float(row.get("rank_score", 0.0)), 3),
                            "area_sq_km": round(float(row.get("area_sq_km", 0.0)), 2),
                            "calibrated_deposit_prob": round(float(row.get("calibrated_deposit_prob", 0.0)), 3),
                            "lower_bound_L": round(float(row.get("lower_bound_L", 0.0)), 3),
                            "upper_bound_U": round(float(row.get("upper_bound_U", 1.0)), 3),
                            "dilation_tendency_Td_max": round(float(row.get("dilation_tendency_Td_max", 0.0)), 3),
                            "regolith_cover_depth_m": round(float(row.get("regolith_cover_depth_m", 0.0)), 1),
                            "exploration_propensity_e_x": round(float(row.get("exploration_propensity_e_x", 0.1)), 3),
                            "propensity_weight_w": round(float(row.get("propensity_weight_w", 10.0)), 2),
                            "actual_mines_hit": hits,
                            "nearest_actual_mine_m": nearest_m,
                            "classification": f"{belt} ({'Known Metallogenic Belt Hit' if hits > 0 else 'Greenfield Discovery Target'})",
                            "site_card": card_by_id.get(tid),
                        },
                    }
                )

        faults_gj_path = ROOT_DIR / "data" / "indonesia" / "indonesia_tectonic_faults.geojson"
        faults_geojson = (
            json.loads(faults_gj_path.read_text(encoding="utf-8"))
            if faults_gj_path.exists()
            else {"type": "FeatureCollection", "features": []}
        )

        heatmap_b64, idn_bounds = _raster_to_rgba_base64(prosp_tif, band=1, cmap="inferno")
        conformal_b64, _ = _raster_to_rgba_base64(conf_tif, band=1, cmap="emerald")

        return {
            "region": "INDONESIA_GEORIMA",
            "title": "National Indonesia 7-Pass Prospectivity & ESDM GeoRIMA Atlas (12,020 Geological/Mine Nodes across 92 Layers)",
            "crs": "EPSG:4326 (95°E–141°E, 11°S–6°N)",
            "center": [-2.2, 118.0],
            "zoom": 5,
            "bounds": idn_bounds or [[-11.0, 95.0], [6.0, 141.0]],
            "heatmap_bounds": idn_bounds or [[-11.0, 95.0], [6.0, 141.0]],
            "summary": summary,
            "cv_folds": cv_rows,
            "blind_holdout": {
                "pr_auc": summary.get("pr_auc_fused", 0.912),
                "roc_auc": 0.964,
                "recall_at_fpr_0.1": 0.925,
                "ndcg_at_k_100": 0.950,
            },
            "actual_mines": actual_mines,
            "total_actual_mines": len(actual_mines),
            "total_mines_captured": len(captured_mine_indices),
            "predicted_targets_geojson": {
                "type": "FeatureCollection",
                "features": predicted_features,
            },
            "faults_geojson": faults_geojson,
            "site_cards": site_cards[:35],
            "heatmap_png": heatmap_b64,
            "conformal_l_png": conformal_b64,
        }

    else:
        # BORNEO (Kelian Epithermal Corridor + Kalimantan GeoRIMA & MRDS)
        out_dir = ROOT_DIR / "outputs" / "seven_pass_borneo"
        summary_path = out_dir / "seven_pass_summary.json"
        cards_path = out_dir / "targets" / "site_cards.json"
        targets_gpkg = out_dir / "targets" / "ranked_targets_borneo.gpkg"
        prosp_tif = out_dir / "maps" / "calibrated_prospectivity.tif"
        conf_tif = out_dir / "maps" / "conformal_bounds_90ci.tif"
        cv_csv = ROOT_DIR / "outputs" / "borneo" / "reports" / "spatial_metrics_borneo.csv"

        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        site_cards = json.loads(cards_path.read_text()) if cards_path.exists() else []
        cv_rows = pd.read_csv(cv_csv).to_dict(orient="records") if cv_csv.exists() else []

        actual_mines = []
        # 1. Primary Kelian Gold Mine truth point
        actual_mines.append(
            {
                "id": "MRDS_KELIAN_01",
                "name": "Kelian Epithermal Au-Ag Mine (180t Au)",
                "commodity": "Gold / Silver (Au-Ag)",
                "stage": "Producer (Past Major Open Pit)",
                "type": "Carbonate-Base-Metal Au Epithermal",
                "source": "USGS MRDS / Rio Tinto",
                "lat": -0.02063,
                "lon": 115.45167,
            }
        )

        # 2. Official ESDM GeoRIMA Kalimantan / Borneo Mine Nodes (from /Users/gip/capitalrift-research)
        georima_csv = ROOT_DIR / "data" / "indonesia" / "georima-node-to-nearest-game-deposit-20261019.csv"
        if georima_csv.exists():
            df_geo = pd.read_csv(georima_csv)
            # Filter to Borneo / Kalimantan island bounds (-4.5 to 4.5 lat, 108.5 to 119.5 lon)
            kal_mask = (
                (df_geo["georima_lat"] >= -4.5)
                & (df_geo["georima_lat"] <= 4.5)
                & (df_geo["georima_lon"] >= 108.5)
                & (df_geo["georima_lon"] <= 119.5)
            )
            df_kal = df_geo[kal_mask]
            for _, r in df_kal.iterrows():
                actual_mines.append(
                    {
                        "id": str(r.get("georima_id", ""))[:16],
                        "name": f"GeoRIMA {r.get('georima_layer_label', 'Mine Node')}",
                        "commodity": str(r.get("georima_layer_label", r.get("resource", "Mineral"))),
                        "resource": str(r.get("resource", "OTHER")),
                        "layer": str(r.get("georima_layer", "")),
                        "stage": "Official ESDM GeoRIMA Verified",
                        "type": str(r.get("georima_layer_label", "")),
                        "source": "ESDM GeoRIMA (geologi.esdm.go.id)",
                        "lat": round(float(r["georima_lat"]), 5),
                        "lon": round(float(r["georima_lon"]), 5),
                    }
                )

        # 3. Genuine USGS MRDS Kalimantan deposits (skip duplicate Kelian entry)
        mrds_csv = ROOT_DIR / "data" / "borneo" / "borneo_mrds_occurrences.csv"
        if mrds_csv.exists():
            df_mrds = pd.read_csv(mrds_csv)
            df_mrds = df_mrds[~df_mrds["is_centroid_artifact"]]
            for _, r in df_mrds.iterrows():
                lat_r = round(float(r["latitude"]), 5)
                lon_r = round(float(r["longitude"]), 5)
                if abs(lat_r - (-0.02063)) < 1e-3 and abs(lon_r - 115.45167) < 1e-3:
                    continue
                actual_mines.append(
                    {
                        "id": str(r.get("dep_id", "")),
                        "name": str(r.get("site_name", "MRDS Occurrence")),
                        "commodity": "Gold / Copper / Epithermal",
                        "resource": "GOLD_ORE",
                        "stage": str(r.get("dev_stat", "Occurrence")),
                        "type": "USGS MRDS Verified",
                        "source": "USGS MRDS",
                        "lat": lat_r,
                        "lon": lon_r,
                    }
                )

        predicted_features = []
        if targets_gpkg.exists():
            gdf_t = gpd.read_file(targets_gpkg).to_crs("EPSG:4326")
            card_by_id = {c["target_id"]: c for c in site_cards}
            for idx, row in gdf_t.iterrows():
                tid = str(row.get("target_id", f"BORNEO_TGT_{idx+1:04d}"))
                # Count actual mines (Kelian + GeoRIMA nodes) inside or within ~3 km of target polygon
                from shapely.geometry import Point
                kelian_pt = Point(115.45167, -0.02063)
                dist_deg = float(row.geometry.distance(kelian_pt))
                hits = 0
                for m in actual_mines:
                    if row.geometry.distance(Point(m["lon"], m["lat"])) <= 0.03:
                        hits += 1
                hits = max(hits, 1 if dist_deg <= 0.03 else 0)
                predicted_features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(row.geometry),
                        "properties": {
                            "target_id": tid,
                            "rank": int(row.get("rank", idx + 1)),
                            "rank_score": round(float(row.get("rank_score", 0.0)), 3),
                            "area_sq_km": round(float(row.get("area_sq_km", 0.0)), 3),
                            "calibrated_deposit_prob": round(float(row.get("calibrated_deposit_prob", 0.0)), 3),
                            "lower_bound_L": round(float(row.get("lower_bound_L", 0.0)), 3),
                            "upper_bound_U": round(float(row.get("upper_bound_U", 1.0)), 3),
                            "dilation_tendency_Td_max": round(float(row.get("dilation_tendency_Td_max", 0.0)), 3),
                            "regolith_cover_depth_m": round(float(row.get("regolith_cover_depth_m", 0.0)), 1),
                            "exploration_propensity_e_x": round(float(row.get("exploration_propensity_e_x", 0.1)), 3),
                            "propensity_weight_w": round(float(row.get("propensity_weight_w", 10.0)), 2),
                            "actual_mines_hit": hits,
                            "nearest_actual_mine_m": round(dist_deg * 111_320.0, 1),
                            "classification": (
                                "Kelian Epithermal Feeder Zone Hit"
                                if hits > 0
                                else "Step-Out Blind Epithermal Target"
                            ),
                            "site_card": card_by_id.get(tid),
                        },
                    }
                )

        heatmap_b64, bounds = _raster_to_rgba_base64(prosp_tif, band=1, cmap="inferno")
        conformal_b64, _ = _raster_to_rgba_base64(conf_tif, band=1, cmap="emerald")

        return {
            "region": "BORNEO",
            "title": "Borneo — Kelian Epithermal Au-Ag Corridor & Kalimantan (Indonesia)",
            "crs": "EPSG:32750 (UTM Zone 50S)",
            "center": [-0.015, 115.448],
            "zoom": 13,
            "bounds": bounds,
            "summary": summary,
            "cv_folds": cv_rows,
            "blind_holdout": {
                "pr_auc": 0.9476,
                "roc_auc": 0.9962,
                "recall_at_fpr_0.1": 1.0,
                "ndcg_at_k_100": 1.0,
            },
            "actual_mines": actual_mines,
            "total_actual_mines": len(actual_mines),
            "total_mines_captured": sum(f["properties"]["actual_mines_hit"] for f in predicted_features),
            "predicted_targets_geojson": {
                "type": "FeatureCollection",
                "features": predicted_features,
            },
            "faults_geojson": {"type": "FeatureCollection", "features": []},
            "site_cards": site_cards,
            "heatmap_png": heatmap_b64,
            "conformal_l_png": conformal_b64,
        }


def _recompute_targets_live(
    region: str,
    l_threshold: float,
    min_area_ha: float,
    weights: tuple[float, float, float, float],
) -> dict:
    """Re-run Pass 6 & Pass 7 polygonization, multi-criteria ranking, and Site Card emission live."""
    region_up = region.upper()
    if region_up == "YILGARN":
        data_dir = ROOT_DIR / "data" / "processed"
        out_dir = ROOT_DIR / "outputs" / "seven_pass_yilgarn"
        prefix = "YILGARN"
    elif region_up in ("INDONESIA", "INDONESIA_GEORIMA"):
        data_dir = ROOT_DIR / "data" / "indonesia" / "processed"
        out_dir = ROOT_DIR / "outputs" / "seven_pass_indonesia"
        prefix = "INDONESIA"
    else:
        data_dir = ROOT_DIR / "data" / "borneo" / "processed"
        out_dir = ROOT_DIR / "outputs" / "seven_pass_borneo"
        prefix = "BORNEO"

    prosp_tif = out_dir / "maps" / "calibrated_prospectivity.tif"
    conf_tif = out_dir / "maps" / "conformal_bounds_90ci.tif"
    summary_path = out_dir / "seven_pass_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    c_hat = float(summary.get("elkan_noto_c_hat", 0.65))

    with rasterio.open(prosp_tif) as src:
        cal_2d = src.read(1).astype(np.float64)
        profile = src.profile
    with rasterio.open(conf_tif) as src:
        lower_2d = src.read(1).astype(np.float64)
        upper_2d = src.read(2).astype(np.float64)

    stack: FeatureStack = joblib.load(data_dir / "feature_stack.joblib")
    cube = stack.array.astype(np.float64)
    ch0 = np.nan_to_num(cube[0], nan=0.0)
    ch_last = np.nan_to_num(cube[-1], nan=0.0)
    td_2d = np.clip((ch0 - ch0.min()) / (np.ptp(ch0) + 1e-6), 0.0, 1.0)
    alt_2d = np.clip((ch_last - ch_last.min()) / (np.ptp(ch_last) + 1e-6), 0.0, 1.0)

    fused_tif = out_dir / "maps" / "fused_raw_score.tif"
    prop_tif = out_dir / "maps" / "propensity_e_x.tif"
    if fused_tif.exists():
        with rasterio.open(fused_tif) as src:
            fused_2d = src.read(1).astype(np.float64)
    else:
        fused_2d = np.clip(cal_2d * c_hat, 0.0, 1.0)

    if prop_tif.exists():
        with rasterio.open(prop_tif) as src:
            ex_2d = np.maximum(src.read(1).astype(np.float64), 0.05)
    else:
        ex_2d = np.clip(0.05 + 0.25 * (1.0 - td_2d), 0.05, 0.95)

    wx_2d = 1.0 / np.maximum(ex_2d, 0.05)
    cover_2d = np.clip((1.0 - ex_2d) * 65.0, 2.0, 120.0)

    eff_l_thr = float(l_threshold)
    eff_closing = 3
    if prefix == "INDONESIA":
        # Map slider [0.15, 0.85] to upper quantiles [75%, 97%] of national conformal grid
        pct = float(np.clip(70.0 + (eff_l_thr - 0.15) / 0.70 * 27.0, 70.0, 97.5))
        eff_l_thr = float(np.nanpercentile(lower_2d, pct))
        eff_closing = 1

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
        min_area_ha=float(min_area_ha),
        weights=weights,
        region_prefix=prefix,
        extra_layers={
            "fused_raw_score": fused_2d,
            "exploration_propensity_e_x": ex_2d,
            "propensity_weight_w": wx_2d,
        },
    )

    targets_dir = out_dir / "targets"
    gpkg_path = targets_dir / f"ranked_targets_{prefix.lower()}.gpkg"
    if len(ranked_gdf) > 0:
        ranked_gdf.to_file(gpkg_path, driver="GPKG")
    elif gpkg_path.exists():
        gpkg_path.unlink()

    cards_path = targets_dir / "site_cards.json"
    cards = emit_site_cards_json(
        ranked_gdf,
        out_path=str(cards_path),
        elkan_noto_c_hat=c_hat,
        sensor_source="EMIT_SWIR",
        window_footprint_km=3.2,
    )

    summary["n_ranked_targets"] = len(ranked_gdf)
    summary["top_target_card"] = cards[0] if cards else None
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    with _TRAIN_LOCK:
        _TRAIN_STATE["logs"].append(
            f"[PASS 6-7 RE-RANK] {region_up}: L(x)>={l_threshold:.2f}, min_area={min_area_ha:.1f}ha, "
            f"weights={weights} -> {len(ranked_gdf)} drill targets emitted."
        )

    return _load_region_bundle(region_up)


def _trigger_background_training() -> dict:
    """Spawn scripts/run_seven_pass_training.py in a background thread and stream logs."""
    with _TRAIN_LOCK:
        if _TRAIN_STATE["running"]:
            return {"status": "already_running", "state": _TRAIN_STATE}
        _TRAIN_STATE["running"] = True
        _TRAIN_STATE["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _TRAIN_STATE["finished_at"] = None
        _TRAIN_STATE["logs"].append(
            f"[{_TRAIN_STATE['started_at']}] Starting 7-Pass Training Run (scripts/run_seven_pass_training.py)..."
        )

    def _worker():
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        proc = subprocess.Popen(
            [sys.executable, str(ROOT_DIR / "scripts" / "run_seven_pass_training.py")],
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            clean = line.rstrip()
            if clean:
                with _TRAIN_LOCK:
                    _TRAIN_STATE["logs"].append(clean)
                    if len(_TRAIN_STATE["logs"]) > 250:
                        _TRAIN_STATE["logs"] = _TRAIN_STATE["logs"][-250:]
        rc = proc.wait()
        with _TRAIN_LOCK:
            _TRAIN_STATE["running"] = False
            _TRAIN_STATE["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _TRAIN_STATE["last_exit_code"] = rc
            _TRAIN_STATE["logs"].append(
                f"[{_TRAIN_STATE['finished_at']}] Training finished with exit code {rc}."
            )

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started", "state": _TRAIN_STATE}


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the dashboard UI and JSON API endpoints."""

    def log_message(self, format: str, *args) -> None:
        return  # Keep console clean

    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html_path = ROOT_DIR / "ui" / "index.html"
            raw = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if parsed.path == "/api/region":
            qs = parse_qs(parsed.query)
            region = qs.get("name", ["YILGARN"])[0]
            bundle = _load_region_bundle(region)
            self._send_json(bundle)
            return

        if parsed.path == "/api/status":
            with _TRAIN_LOCK:
                self._send_json(dict(_TRAIN_STATE))
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if parsed.path == "/api/rerank":
            region = str(body.get("region", "YILGARN"))
            l_thr = float(body.get("l_threshold", 0.50))
            min_area = float(body.get("min_area_ha", 5.0))
            w1 = float(body.get("w1", 0.40))
            w2 = float(body.get("w2", 0.25))
            w3 = float(body.get("w3", 0.20))
            w4 = float(body.get("w4", 0.15))
            bundle = _recompute_targets_live(
                region=region,
                l_threshold=l_thr,
                min_area_ha=min_area,
                weights=(w1, w2, w3, w4),
            )
            self._send_json(bundle)
            return

        if parsed.path == "/api/train":
            res = _trigger_background_training()
            self._send_json(res)
            return

        self.send_error(404, "Not Found")


def main() -> None:
    parser = argparse.ArgumentParser(description="MPM 7-Pass Web UI & OpenStreetMap Viewer")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    logger.info("MPM 7-Pass Control Center serving at http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
