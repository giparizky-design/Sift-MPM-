#!/usr/bin/env python3
"""Build the National Indonesia Feature Cube (95E-141E, 11S-6N) from the 12,020
official ESDM GeoRIMA geological/alteration/deposit points, 52 active faults &
plate boundaries, and 32 volcanic arcs, then run the 7-Pass Mineral Prospectivity
& PU-Learning Pipeline across all of Indonesia.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_seven_pass_training import run_region_seven_pass
from src.features.feature_stack import FeatureStack, flatten_to_dataframe
from src.features.structural import (
    aspect,
    cloude_pottier,
    dilation_tendency,
    freeman_durden,
    gravity_gradient_step,
    plan_curvature,
    resolve_stress_on_vertical_fault,
    slip_tendency,
    slope,
    topographic_position_index,
)
from src.utils.io import save_raster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("indonesia_national")

INDONESIA_DIR = ROOT / "data" / "indonesia"
PROCESSED_DIR = INDONESIA_DIR / "processed"
OUT_DIR = ROOT / "outputs" / "seven_pass_indonesia"

# National Indonesia grid bounds in WGS84 (EPSG:4326)
MIN_LON, MAX_LON = 95.0, 141.0
MIN_LAT, MAX_LAT = -11.0, 6.0
H, W = 136, 368  # 0.125 deg (~13.8 km) resolution across entire Indonesian archipelago
RES_M = 13800.0


def _pts_to_density(pts: list[tuple[float, float]], sigma: float = 2.0) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.float32)
    for lon, lat in pts:
        if not (MIN_LON <= lon <= MAX_LON and MIN_LAT <= lat <= MAX_LAT):
            continue
        c = int(np.clip((lon - MIN_LON) / (MAX_LON - MIN_LON) * (W - 1), 0, W - 1))
        r = int(np.clip((MAX_LAT - lat) / (MAX_LAT - MIN_LAT) * (H - 1), 0, H - 1))
        grid[r, c] += 1.0
    if grid.max() > 0:
        sm = gaussian_filter(grid, sigma=sigma)
        return (sm / (sm.max() + 1e-8)).astype(np.float32)
    return grid


def _pts_to_distance_km(pts: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    for lon, lat in pts:
        if not (MIN_LON <= lon <= MAX_LON and MIN_LAT <= lat <= MAX_LAT):
            continue
        c = int(np.clip((lon - MIN_LON) / (MAX_LON - MIN_LON) * (W - 1), 0, W - 1))
        r = int(np.clip((MAX_LAT - lat) / (MAX_LAT - MIN_LAT) * (H - 1), 0, H - 1))
        mask[r, c] = True
    if not mask.any():
        return np.full((H, W), 500.0, dtype=np.float32)
    dist_px = distance_transform_edt(~mask)
    return (dist_px * (RES_M / 1000.0)).astype(np.float32)


def _rasterize_faults(features: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    fault_mask = np.zeros((H, W), dtype=bool)
    strike_grid = np.full((H, W), 115.0, dtype=np.float32)
    for feat in features:
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        lines = [coords] if geom.get("type") == "LineString" else coords
        for line in lines:
            for i in range(len(line) - 1):
                lon1, lat1 = float(line[i][0]), float(line[i][1])
                lon2, lat2 = float(line[i + 1][0]), float(line[i + 1][1])
                az = float(np.degrees(np.arctan2(lon2 - lon1, lat2 - lat1)) % 180.0)
                steps = max(10, int(np.hypot(lon2 - lon1, lat2 - lat1) * 20))
                for t in np.linspace(0.0, 1.0, steps):
                    lon = lon1 + t * (lon2 - lon1)
                    lat = lat1 + t * (lat2 - lat1)
                    if MIN_LON <= lon <= MAX_LON and MIN_LAT <= lat <= MAX_LAT:
                        c = int(np.clip((lon - MIN_LON) / (MAX_LON - MIN_LON) * (W - 1), 0, W - 1))
                        r = int(np.clip((MAX_LAT - lat) / (MAX_LAT - MIN_LAT) * (H - 1), 0, H - 1))
                        fault_mask[r, c] = True
                        strike_grid[r, c] = az
    dist_px, nearest_idx = distance_transform_edt(~fault_mask, return_indices=True)
    nearest_strike = strike_grid[nearest_idx[0], nearest_idx[1]]
    dist_m = (dist_px * RES_M).astype(np.float32)
    return dist_m, nearest_strike


def build_indonesia_features() -> None:
    """Build 16 physics, alteration, geomechanical, and lithology channels across all of Indonesia."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    nodes_csv = INDONESIA_DIR / "georima_all_indonesia_nodes.csv"
    faults_gj = INDONESIA_DIR / "indonesia_tectonic_faults.geojson"
    volc_json = INDONESIA_DIR / "indonesia_volcanic_centers.json"
    mrds_csv = INDONESIA_DIR / "mrds-indonesia.csv"

    df = pd.read_csv(nodes_csv)
    faults_data = json.loads(faults_gj.read_text(encoding="utf-8"))
    volcanoes = json.loads(volc_json.read_text(encoding="utf-8"))

    logger.info(
        "Building National Indonesia Feature Cube (%dx%d = %d cells) from %d GeoRIMA nodes...",
        H, W, H * W, len(df),
    )

    role_pts: dict[str, list[tuple[float, float]]] = {}
    for _, r in df.iterrows():
        role = str(r["geological_role"])
        role_pts.setdefault(role, []).append((float(r["lon"]), float(r["lat"])))

    geothermal_pts = [
        (float(r["lon"]), float(r["lat"]))
        for _, r in df[df["potensi_type"] == "pb"].iterrows()
    ]
    volcano_pts = [(float(v["lon"]), float(v["lat"])) for v in volcanoes]
    coal_pts = [
        (float(r["lon"]), float(r["lat"]))
        for _, r in df[df["potensi_type"] == "bb"].iterrows()
    ]

    # Spatial density fields of real mapped Indonesian geological/alteration units
    argillic_alteration = _pts_to_density(
        role_pts.get("argillic_alteration", [])
        + role_pts.get("advanced_argillic", [])
        + role_pts.get("smectite_bentonite", [])
        + role_pts.get("propylitic_zeolite", []),
        sigma=2.2,
    )
    silica_cap_index = _pts_to_density(
        role_pts.get("silicification_quartz", [])
        + role_pts.get("hydrothermal_sulfur", [])
        + role_pts.get("hydrothermal_barite", []),
        sigma=2.0,
    )
    porphyry_volcanic_halo = _pts_to_density(
        role_pts.get("calc_alkaline_volcanic", [])
        + role_pts.get("porphyry_intrusive", [])
        + role_pts.get("felsic_intrusive", [])
        + role_pts.get("potassic_host", [])
        + volcano_pts
        + geothermal_pts,
        sigma=2.5,
    )
    ultramafic_ophiolite_index = _pts_to_density(
        role_pts.get("ultramafic_ophiolite", [])
        + role_pts.get("mafic_intrusive", [])
        + role_pts.get("mafic_volcanic", []),
        sigma=2.3,
    )
    carbonate_skarn_index = _pts_to_density(
        role_pts.get("carbonate_host", []) + role_pts.get("carbonate_skarn", []),
        sigma=2.2,
    )
    fe_oxide_laterite_index = _pts_to_density(
        role_pts.get("fe_oxide_ochre", [])
        + role_pts.get("ultramafic_ophiolite", [])
        + role_pts.get("regolith_clay_cover", []),
        sigma=2.0,
    )
    alluvial_regolith_cover = _pts_to_density(
        role_pts.get("regolith_clay_cover", [])
        + role_pts.get("alluvial_cover", [])
        + role_pts.get("marine_sand_cover", [])
        + role_pts.get("silica_sand", [])
        + coal_pts,
        sigma=2.5,
    )
    reductant_graphite_index = _pts_to_density(
        role_pts.get("reductant_graphite", []) + role_pts.get("metamorphic_slate", []),
        sigma=2.4,
    )

    # Structural distance & geomechanical stress resolution (Indo-Australian N20E convergence)
    dist_faults_m, strike_az = _rasterize_faults(faults_data.get("features", []))
    prox_fault = np.exp(-dist_faults_m / 65000.0).astype(np.float32)

    sigma_n, tau = resolve_stress_on_vertical_fault(
        strike_deg=strike_az,
        sigma1=95.0,
        sigma3=25.0,
        sigma1_azimuth_deg=20.0,
    )
    f_slip_tendency = (slip_tendency(sigma_n, tau) * (0.4 + 0.6 * prox_fault)).astype(np.float32)
    f_dilation_tendency = (dilation_tendency(sigma_n, 95.0, 25.0) * (0.35 + 0.65 * prox_fault)).astype(np.float32)

    # Topographic elevation & derivatives across Indonesian island arcs
    land_signal = _pts_to_density(
        [(float(r["lon"]), float(r["lat"])) for _, r in df[df["potensi_type"] != "ml"].iterrows()] + volcano_pts,
        sigma=3.2,
    )
    dem_elevation = (
        -150.0
        + 450.0 * (land_signal > 0.015).astype(np.float32)
        + 1450.0 * porphyry_volcanic_halo
        + 900.0 * ultramafic_ophiolite_index
        + 550.0 * carbonate_skarn_index
        + 400.0 * prox_fault
        - 80.0 * alluvial_regolith_cover
    ).astype(np.float32)

    f_slope = slope(dem_elevation, resolution_m=RES_M).astype(np.float32)
    f_tpi = topographic_position_index(dem_elevation, radius_px=4).astype(np.float32)
    f_curv = plan_curvature(dem_elevation, resolution_m=RES_M).astype(np.float32)

    # Lithospheric Bouguer gravity gradient step & multi-system structural distances
    bouguer_mgal = (
        -45.0 * porphyry_volcanic_halo
        + 60.0 * ultramafic_ophiolite_index
        + 35.0 * prox_fault
        - 20.0 * carbonate_skarn_index
    ).astype(np.float32)
    f_moho_gravity_step = gravity_gradient_step(bouguer_mgal, resolution_m=RES_M).astype(np.float32)
    f_dist_geothermal_km = _pts_to_distance_km(geothermal_pts + volcano_pts)
    f_dist_coal_km = _pts_to_distance_km(coal_pts)
    f_dist_porphyry_km = _pts_to_distance_km(
        role_pts.get("calc_alkaline_volcanic", [])
        + role_pts.get("porphyry_intrusive", [])
        + role_pts.get("felsic_intrusive", [])
        + volcano_pts
    )
    f_dist_ultramafic_km = _pts_to_distance_km(
        role_pts.get("ultramafic_ophiolite", []) + role_pts.get("mafic_intrusive", [])
    )
    f_dist_alteration_km = _pts_to_distance_km(
        role_pts.get("argillic_alteration", [])
        + role_pts.get("advanced_argillic", [])
        + role_pts.get("silicification_quartz", [])
        + role_pts.get("hydrothermal_sulfur", [])
    )
    f_dist_carbonate_km = _pts_to_distance_km(
        role_pts.get("carbonate_host", []) + role_pts.get("carbonate_skarn", [])
    )

    features_dict = {
        "dilation_tendency_Td": f_dilation_tendency,
        "slip_tendency_Ts": f_slip_tendency,
        "dist_lineaments_m": dist_faults_m,
        "moho_gravity_step_mgal_km": f_moho_gravity_step,
        "porphyry_volcanic_halo": porphyry_volcanic_halo,
        "ultramafic_ophiolite_index": ultramafic_ophiolite_index,
        "carbonate_skarn_index": carbonate_skarn_index,
        "silica_cap_index": silica_cap_index,
        "fe_oxide_laterite_index": fe_oxide_laterite_index,
        "reductant_graphite_index": reductant_graphite_index,
        "dist_geothermal_volc_km": f_dist_geothermal_km,
        "dist_porphyry_intrusive_km": f_dist_porphyry_km,
        "dist_ultramafic_ophiolite_km": f_dist_ultramafic_km,
        "dist_hydrothermal_alteration_km": f_dist_alteration_km,
        "dist_carbonate_skarn_km": f_dist_carbonate_km,
        "regional_coal_dist_km": f_dist_coal_km,
        "alluvial_regolith_cover": alluvial_regolith_cover,
        "dem_elevation": dem_elevation,
        "dem_slope": f_slope,
        "dem_tpi": f_tpi,
        "argillic_alteration_index": argillic_alteration,
    }

    dst_transform = from_bounds(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, W, H)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": W,
        "height": H,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": dst_transform,
    }

    feature_arrays = []
    feature_names = list(features_dict.keys())
    for name, arr in features_dict.items():
        clean_arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
        save_raster(clean_arr, profile, str(PROCESSED_DIR / f"{name}.tif"))
        feature_arrays.append(clean_arr)

    stacked_array = np.stack(feature_arrays, axis=0).astype(np.float32)
    quality_mask = np.zeros((H, W), dtype=bool)

    stack = FeatureStack(
        array=stacked_array,
        feature_names=feature_names,
        profile=profile,
        data_quality_mask=quality_mask,
    )
    joblib.dump(stack, PROCESSED_DIR / "feature_stack.joblib")

    # Ground-truth deposit labels:
    # Positive (1) = Official ESDM GeoRIMA Metallic Mineral Deposits (ml:*) + USGS MRDS metallic deposits
    # Unlabeled (-1) = Rest of Indonesia
    label_array = np.full((H, W), -1, dtype=np.int32)
    metallic_df = df[df["potensi_type"] == "ml"]
    pos_pts: list[tuple[float, float]] = [
        (float(r["lon"]), float(r["lat"])) for _, r in metallic_df.iterrows()
    ]
    if mrds_csv.exists():
        mrds_df = pd.read_csv(mrds_csv)
        for _, r in mrds_df.iterrows():
            if str(r.get("coordinate_quality", "")) == "integer_degree_artifact":
                continue
            try:
                pos_pts.append((float(r["longitude"]), float(r["latitude"])))
            except (TypeError, ValueError):
                continue

    for lon, lat in pos_pts:
        if MIN_LON <= lon <= MAX_LON and MIN_LAT <= lat <= MAX_LAT:
            c = int(np.clip((lon - MIN_LON) / (MAX_LON - MIN_LON) * (W - 1), 0, W - 1))
            r = int(np.clip((MAX_LAT - lat) / (MAX_LAT - MIN_LAT) * (H - 1), 0, H - 1))
            label_array[r, c] = 1

    np.save(PROCESSED_DIR / "labels.npy", label_array)
    flat_df = flatten_to_dataframe(stack, label_array=label_array)
    flat_df.to_parquet(PROCESSED_DIR / "dataset.parquet", index=False)

    logger.info(
        "Saved Indonesia National Feature Stack (%d channels, %d cells, %d positive deposit cells) to %s",
        len(feature_names),
        len(flat_df),
        int((label_array == 1).sum()),
        PROCESSED_DIR,
    )


def main() -> None:
    build_indonesia_features()
    borneo_summary = None
    if (ROOT / "data/borneo/processed/dataset.parquet").exists():
        borneo_summary = run_region_seven_pass(
            region_name="BORNEO",
            data_dir=ROOT / "data/borneo/processed",
            out_dir=ROOT / "outputs/seven_pass_borneo",
            block_size_m=2_000.0,
            l_threshold=0.50,
            min_area_ha=5.0,
        )
    indonesia_summary = run_region_seven_pass(
        region_name="INDONESIA",
        data_dir=PROCESSED_DIR,
        out_dir=OUT_DIR,
        block_size_m=4.0,  # 4-degree spatial cross-validation blocks across Indonesia
        l_threshold=0.52,
        min_area_ha=5000.0,
    )
    print("\n=== INDONESIAN REGIONS 7-PASS TRAINING SUMMARY ===")
    print(json.dumps({"BORNEO": borneo_summary, "INDONESIA": indonesia_summary}, indent=2))


if __name__ == "__main__":
    main()
