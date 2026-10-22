"""
scripts/run_kalimantan_full_scan.py
===================================
100% Leak-Free Regional Scan of the Entire Region of Kalimantan (Indonesian Borneo)
Bounding Box: 108.5°E - 119.5°E, -4.5°S - 4.5°N (~1,220 km x 1,000 km, ~544,000 km² land)
Resolution: ~1.1 km (800 x 960 grid, ~480,000 valid land/shelf cells)

Features (X, 21 channels):
  - AWS Open Data Terrarium SRTM elevation & bathymetry (z=7)
  - Slope, aspect, local TPI, regional TPI, plan curvature, terrain ruggedness (TRI)
  - Topographic lineament distance & density
  - 247 GEM Active Faults (Adang Fault, Meratus Suture, Lupar Line, Sangkulirang)
  - 3D Cauchy stress resolution: Fault Dilation Tendency (Td) & Slip Tendency (Ts)
  - Airy-Heiskanen isostatic crustal thickness & Bouguer gravity gradient
  - Regolith cover depth (m) & bedrock outcrop exposure index

Validation Labels (y=1 ONLY, strictly excluded from X):
  - Official ESDM GeoRIMA Kalimantan nodes (Gold, Bauxite, Iron, Diamonds, Coal, etc.)
  - USGS MRDS Borneo deposit catalog

Outputs:
  - outputs/seven_pass_kalimantan/maps/calibrated_prospectivity.tif
  - outputs/seven_pass_kalimantan/maps/conformal_bounds_90ci.tif
  - outputs/seven_pass_kalimantan/targets/ranked_targets_kalimantan.gpkg
  - outputs/seven_pass_kalimantan/targets/site_cards.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_indonesia_hd_upgrade import build_clean_region_cube
from scripts.run_seven_pass_training import run_region_seven_pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kalimantan_scan")

KALIMANTAN_DIR = ROOT / "data" / "kalimantan"
PROCESSED_DIR = KALIMANTAN_DIR / "processed"
INTERIM_DIR = KALIMANTAN_DIR / "interim"
OUT_DIR = ROOT / "outputs" / "seven_pass_kalimantan"

# Entire Island of Borneo / Kalimantan bounding coordinates
MIN_LON, MAX_LON = 108.5, 119.5
MIN_LAT, MAX_LAT = -4.5, 4.5
H, W = 800, 960  # ~1.15 km pixel resolution
RES_M = 1150.0


def main() -> None:
    logger.info("=====================================================================")
    logger.info("STARTING ENTIRE KALIMANTAN REGIONAL PROSPECTIVITY SCAN (100%% LEAK-FREE)")
    logger.info("Bounding Box: [%.1fE..%.1fE, %.1fS..%.1fN] | Grid: %dx%d @ ~%.0fm", MIN_LON, MAX_LON, MIN_LAT, MAX_LAT, H, W, RES_M)
    logger.info("=====================================================================")

    # Step 1: Build 21-channel feature cube and labels from SRTM + GEM Faults + GeoRIMA
    build_clean_region_cube(
        region_code="KALIMANTAN",
        min_lon=MIN_LON,
        min_lat=MIN_LAT,
        max_lon=MAX_LON,
        max_lat=MAX_LAT,
        H=H,
        W=W,
        res_m=RES_M,
        zoom=7,
        processed_dir=PROCESSED_DIR,
        interim_dem_path=INTERIM_DIR / "kalimantan_srtm_1km_dem.tif",
        label_radius_px=1,
    )

    # Step 2: Run 7-Pass PU-Learning Pipeline across Kalimantan
    # Block size 0.4° (~45 km spatial blocks) for spatial block CV
    summary = run_region_seven_pass(
        region_name="KALIMANTAN",
        data_dir=PROCESSED_DIR,
        out_dir=OUT_DIR,
        block_size_m=0.40,
        l_threshold=0.52,
        min_area_ha=150.0,
    )

    # Step 3: Update global report
    report_path = ROOT / "outputs" / "seven_pass_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["kalimantan"] = summary
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    logger.info("=====================================================================")
    logger.info("KALIMANTAN REGIONAL SCAN COMPLETE!")
    logger.info("PR-AUC XGB=%.4f | UNet=%.4f | Fused=%.4f | c_hat=%.4f | Targets=%d",
                summary["pr_auc_xgb"], summary["pr_auc_unet"], summary["pr_auc_fused"],
                summary["elkan_noto_c_hat"], summary["n_ranked_targets"])
    logger.info("=====================================================================")


if __name__ == "__main__":
    main()
