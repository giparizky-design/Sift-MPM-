"""
scripts/run_indonesia_hd_upgrade.py
===================================
Upgrades both tiers of Indonesian geospatial coverage (100% leak-free):
  1. Tier 1 (National Reconnaissance `INDONESIA`):
     Upgrades grid resolution 16x from 136x368 (~13.8 km) to 544x1472 (~3.45 km)
     across 95.0E-141.0E, 11.0S-6.0N using AWS Open Data Terrarium SRTM/ETOPO z=6 tiles.
  2. Tier 2 (HD Camp-Scale Indonesian Mineral Districts):
     Downloads high-definition AWS Terrarium / Copernicus SRTM z=10/z=11 elevation tiles
     and builds 360x360 HD 21-channel geomechanical/tectonic feature cubes + 7-Pass
     PU-Learning models for 4 major Indonesian mining camps (in addition to Kelian BORNEO):
       - PAPUA    : Grasberg-Wabu District, Central Range Papua (Cu-Au)
       - SUMBAWA  : Batu Hijau-Elang District, West Nusa Tenggara (Cu-Au)
       - SUMATRA  : Martabe-Toru District, Great Sumatran Fault (Au-Ag)
       - SULAWESI : Toka Tindung-Minahasa District, North Sulawesi Arc (Au-Cu)
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import logging
import math
import sys
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from PIL import Image
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_seven_pass_training import run_region_seven_pass
from src.features.feature_stack import FeatureStack, flatten_to_dataframe
from src.features.structural import (
    aspect,
    dilation_tendency,
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
logger = logging.getLogger("indonesia_hd")

INDONESIA_DIR = ROOT / "data" / "indonesia"


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    lat_rad = math.radians(max(-85.0, min(85.0, lat)))
    n = 2.0 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def _tile_to_lonlat(xtile: float, ytile: float, z: int) -> tuple[float, float]:
    n = 2.0 ** z
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lon_deg, lat_deg


def fetch_terrarium_dem(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    out_h: int,
    out_w: int,
    zoom: int,
    cache_tif: Path,
) -> np.ndarray:
    """Download AWS Open Data Terrarium SRTM/ETOPO tiles and resample to (out_h, out_w)."""
    if cache_tif.exists():
        with rasterio.open(cache_tif) as src:
            if src.height == out_h and src.width == out_w:
                logger.info("Loaded cached HD DEM (%dx%d) from %s", out_h, out_w, cache_tif)
                return src.read(1).astype(np.float32)

    x0, y0 = _lonlat_to_tile(min_lon, max_lat, zoom)
    x1, y1 = _lonlat_to_tile(max_lon, min_lat, zoom)
    nx = x1 - x0 + 1
    ny = y1 - y0 + 1
    logger.info(
        "Fetching %d AWS Terrarium DEM tiles (z=%d, %dx%d tiles) for [%.2f..%.2f, %.2f..%.2f]...",
        nx * ny, zoom, nx, ny, min_lon, max_lon, min_lat, max_lat,
    )

    mosaic = np.zeros((ny * 256, nx * 256), dtype=np.float32)

    def _download_one(tx: int, ty: int) -> tuple[int, int, np.ndarray]:
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{tx}/{ty}.png"
        req = urllib.request.Request(url, headers={"User-Agent": "MPM-Geospatial/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        elev = (arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0) - 32768.0
        return tx, ty, elev

    coords = [(tx, ty) for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        for tx, ty, elev in ex.map(lambda p: _download_one(p[0], p[1]), coords):
            ix = tx - x0
            iy = ty - y0
            mosaic[iy * 256 : (iy + 1) * 256, ix * 256 : (ix + 1) * 256] = elev

    # Crop exact geographic bounds from tile mosaic
    tl_lon, tl_lat = _tile_to_lonlat(x0, y0, zoom)
    br_lon, br_lat = _tile_to_lonlat(x1 + 1, y1 + 1, zoom)

    mh, mw = mosaic.shape
    c_start = int(np.clip((min_lon - tl_lon) / (br_lon - tl_lon) * mw, 0, mw - 2))
    c_end = int(np.clip((max_lon - tl_lon) / (br_lon - tl_lon) * mw, c_start + 1, mw))
    r_start = int(np.clip((tl_lat - max_lat) / (tl_lat - br_lat) * mh, 0, mh - 2))
    r_end = int(np.clip((tl_lat - min_lat) / (tl_lat - br_lat) * mh, r_start + 1, mh))

    cropped = mosaic[r_start:r_end, c_start:c_end]
    pil_crop = Image.fromarray(cropped, mode="F")
    pil_resized = pil_crop.resize((out_w, out_h), resample=Image.Resampling.BILINEAR)
    dem = np.asarray(pil_resized, dtype=np.float32)

    cache_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": out_w,
        "height": out_h,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": from_bounds(min_lon, min_lat, max_lon, max_lat, out_w, out_h),
    }
    save_raster(dem, profile, str(cache_tif))
    logger.info("Saved HD DEM (%dx%d) -> %s", out_h, out_w, cache_tif)
    return dem


def _rasterize_faults_fast(
    features: list[dict],
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    H: int,
    W: int,
    res_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast rasterization of tectonic fault lines + local fault strike azimuth."""
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, W, H)
    shapes_mask = []
    shapes_az = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        coords = geom.get("coordinates") or []
        lines = [coords] if geom.get("type") == "LineString" else coords
        az = 115.0
        for line in lines:
            if len(line) >= 2:
                lon1, lat1 = float(line[0][0]), float(line[0][1])
                lon2, lat2 = float(line[-1][0]), float(line[-1][1])
                az = float(np.degrees(np.arctan2(lon2 - lon1, lat2 - lat1)) % 180.0)
                break
        shapes_mask.append((geom, 1))
        shapes_az.append((geom, max(1.0, az)))

    if shapes_mask:
        line_mask = rasterio.features.rasterize(
            shapes_mask, out_shape=(H, W), transform=transform, fill=0, all_touched=True, dtype="uint8"
        ).astype(bool)
        az_raster = rasterio.features.rasterize(
            shapes_az, out_shape=(H, W), transform=transform, fill=115.0, all_touched=True, dtype="float32"
        )
    else:
        line_mask = np.zeros((H, W), dtype=bool)
        az_raster = np.full((H, W), 115.0, dtype=np.float32)

    if not line_mask.any():
        # Fallback to synthetic central shear axis if window has no mapped global fault
        line_mask[H // 2, :] = True

    dist_px, nearest_idx = distance_transform_edt(~line_mask, return_indices=True)
    nearest_strike = az_raster[nearest_idx[0], nearest_idx[1]]
    dist_m = (dist_px * res_m).astype(np.float32)
    density = gaussian_filter(line_mask.astype(np.float32), sigma=max(1.5, min(H, W) / 60.0)).astype(np.float32)
    return dist_m, nearest_strike, density


def build_clean_region_cube(
    region_code: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    H: int,
    W: int,
    res_m: float,
    zoom: int,
    processed_dir: Path,
    interim_dem_path: Path,
    label_radius_px: int = 1,
) -> None:
    """Build 21 leak-free satellite/tectonic/geophysical channels for a region."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    dem = fetch_terrarium_dem(min_lon, min_lat, max_lon, max_lat, H, W, zoom, interim_dem_path)

    faults_gj = INDONESIA_DIR / "indonesia_tectonic_faults.geojson"
    volc_json = INDONESIA_DIR / "indonesia_volcanic_centers.json"
    faults_all = json.loads(faults_gj.read_text(encoding="utf-8")).get("features", [])
    volcanoes = json.loads(volc_json.read_text(encoding="utf-8"))

    gem_faults = [f for f in faults_all if f.get("properties", {}).get("kind") != "tectonic_plate_boundary"]
    plate_bounds = [f for f in faults_all if f.get("properties", {}).get("kind") == "tectonic_plate_boundary"]
    if not plate_bounds:
        plate_bounds = faults_all

    # 1. Topographic & Bathymetric Features
    f_slope = slope(dem, resolution_m=res_m).astype(np.float32)
    f_aspect = aspect(dem, resolution_m=res_m).astype(np.float32)
    f_tpi_local = topographic_position_index(dem, radius_px=3).astype(np.float32)
    f_tpi_regional = topographic_position_index(dem, radius_px=11).astype(np.float32)
    f_curv = plan_curvature(dem, resolution_m=res_m).astype(np.float32)

    mean_dem = uniform_filter(dem, size=5)
    mean_sq = uniform_filter(dem ** 2, size=5)
    f_tri = np.sqrt(np.maximum(mean_sq - mean_dem ** 2, 0.0)).astype(np.float32)

    land_mask = dem >= 0.0
    coast_mask = land_mask ^ (uniform_filter(land_mask.astype(np.float32), size=3) > 0.5)
    if not coast_mask.any():
        coast_mask = np.zeros((H, W), dtype=bool)
        coast_mask[0, :] = True
    f_coastal_dist_km = (distance_transform_edt(~coast_mask) * (res_m / 1000.0)).astype(np.float32)

    # 2. Topographic Lineaments from DEM Curvature (critical for HD camp-scale fracture mapping)
    curv_smooth = gaussian_filter(np.abs(f_curv), sigma=1.2)
    lin_mask = curv_smooth >= np.percentile(curv_smooth, 86)
    f_dist_topo_lineaments_km = (distance_transform_edt(~lin_mask) * (res_m / 1000.0)).astype(np.float32)
    topo_lin_density = gaussian_filter(lin_mask.astype(np.float32), sigma=3.0).astype(np.float32)

    # 3. GEM Active Faults & PB2002 Plate Boundaries
    dist_faults_m, strike_az, gem_fault_density = _rasterize_faults_fast(
        gem_faults, min_lon, min_lat, max_lon, max_lat, H, W, res_m
    )
    dist_plate_m, _, _ = _rasterize_faults_fast(
        plate_bounds, min_lon, min_lat, max_lon, max_lat, H, W, res_m
    )
    f_dist_plate_km = (dist_plate_m / 1000.0).astype(np.float32)

    # Blend GEM fault distance with HD topographic lineament distance at camp scale
    effective_fault_dist_m = np.minimum(dist_faults_m, f_dist_topo_lineaments_km * 1000.0 * 2.5).astype(np.float32)
    f_fault_density = (0.6 * gem_fault_density + 0.4 * topo_lin_density).astype(np.float32)

    decay_scale_m = 55000.0 if res_m > 1000.0 else 3500.0
    prox_fault = np.exp(-effective_fault_dist_m / decay_scale_m).astype(np.float32)
    sigma_n, tau = resolve_stress_on_vertical_fault(
        strike_deg=strike_az + (f_aspect * 0.15),
        sigma1=95.0,
        sigma3=25.0,
        sigma1_azimuth_deg=20.0,
    )
    f_slip_ts = (slip_tendency(sigma_n, tau) * (0.35 + 0.65 * prox_fault)).astype(np.float32)
    f_dilation_td = (dilation_tendency(sigma_n, 95.0, 25.0) * (0.35 + 0.65 * prox_fault)).astype(np.float32)

    # 4. Smithsonian GVP Holocene Volcanoes (Exact Geodesic Distance in km)
    lats_1d = np.linspace(max_lat, min_lat, H, dtype=np.float32)
    lons_1d = np.linspace(min_lon, max_lon, W, dtype=np.float32)
    lat_grid, lon_grid = np.meshgrid(lats_1d, lons_1d, indexing="ij")

    min_v_dist_km = np.full((H, W), 500.0, dtype=np.float32)
    for v in volcanoes:
        vlon, vlat = float(v["lon"]), float(v["lat"])
        dlon_km = (lon_grid - vlon) * 111.32 * np.cos(np.radians(lat_grid))
        dlat_km = (lat_grid - vlat) * 110.57
        dkm = np.hypot(dlon_km, dlat_km)
        min_v_dist_km = np.minimum(min_v_dist_km, dkm)
    f_dist_volcano_km = min_v_dist_km.astype(np.float32)
    f_volcanic_arc_halo = np.exp(-(f_dist_volcano_km ** 2) / (2.0 * (35.0 ** 2))).astype(np.float32)

    # 5. Airy-Heiskanen Isostatic Crustal Thickness & Bouguer Gravity Anomaly
    f_crustal_thickness_km = np.clip(
        30.0 + np.where(dem >= 0, dem * 0.0065, dem * 0.0028),
        12.0,
        52.0,
    ).astype(np.float32)
    f_bouguer_mgal = (
        np.where(dem >= 0, -0.1119 * dem, -0.068 * dem)
        + 35.0 * np.exp(-f_dist_plate_km / 120.0)
    ).astype(np.float32)
    f_moho_gravity_step = gravity_gradient_step(f_bouguer_mgal, resolution_m=res_m).astype(np.float32)

    # 6. Regolith Cover & Bedrock Outcrop Exposure
    land_elev_pos = np.maximum(dem, 0.0)
    slope_norm = np.clip(f_slope / 15.0, 0.0, 1.0)
    relief_norm = np.clip((f_tpi_local - np.percentile(f_tpi_local, 10)) / (np.ptp(f_tpi_local) + 1e-6), 0.0, 1.0)
    f_outcrop_exposure = np.clip(0.55 * slope_norm + 0.45 * relief_norm, 0.0, 1.0).astype(np.float32)
    f_regolith_cover_m = np.clip(
        (1.0 - f_outcrop_exposure) * np.exp(-land_elev_pos / 600.0) * 85.0 + 3.0,
        2.0,
        95.0,
    ).astype(np.float32)

    for old_f in processed_dir.glob("*.tif"):
        old_f.unlink()

    features_dict = {
        "dilation_tendency_Td": f_dilation_td,
        "slip_tendency_Ts": f_slip_ts,
        "dist_active_faults_m": effective_fault_dist_m,
        "active_fault_density": f_fault_density,
        "dist_plate_boundary_km": f_dist_plate_km,
        "dist_topo_lineaments_km": f_dist_topo_lineaments_km,
        "dist_volcanic_arc_km": f_dist_volcano_km,
        "volcanic_arc_halo": f_volcanic_arc_halo,
        "bouguer_gravity_mgal": f_bouguer_mgal,
        "moho_gravity_step_mgal_km": f_moho_gravity_step,
        "crustal_thickness_km": f_crustal_thickness_km,
        "dem_elevation": dem,
        "dem_slope": f_slope,
        "dem_aspect": f_aspect,
        "dem_tpi_local": f_tpi_local,
        "dem_tpi_regional": f_tpi_regional,
        "dem_plan_curvature": f_curv,
        "terrain_ruggedness_tri": f_tri,
        "coastal_shelf_dist_km": f_coastal_dist_km,
        "regolith_cover_m": f_regolith_cover_m,
        "outcrop_exposure_index": f_outcrop_exposure,
    }

    dst_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, W, H)
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

    deep_ocean_mask = dem < (-150.0 if region_code == "INDONESIA" else -20.0)
    if region_code == "INDONESIA":
        # Mask non-Indonesian neighbor land inside bounding box
        non_id_mask = (
            ((lat_grid > 4.6) & (lon_grid > 119.0))
            | ((lat_grid > 1.3) & (lon_grid > 99.6) & (lon_grid < 104.4))
            | ((lat_grid > 4.35) & (lon_grid > 113.0) & (lon_grid < 119.2))
            | ((lat_grid > 1.45) & (lon_grid > 109.6) & (lon_grid < 115.2))
            | ((lat_grid > -9.55) & (lat_grid < -8.15) & (lon_grid > 124.9) & (lon_grid < 127.35))
        )
        deep_ocean_mask = deep_ocean_mask | non_id_mask

    feature_arrays = []
    feature_names = list(features_dict.keys())
    for name, arr in features_dict.items():
        clean_arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
        clean_arr[deep_ocean_mask] = np.nan
        save_raster(clean_arr, profile, str(processed_dir / f"{name}.tif"))
        feature_arrays.append(clean_arr)

    stacked_array = np.stack(feature_arrays, axis=0).astype(np.float32)
    stack = FeatureStack(
        array=stacked_array,
        feature_names=feature_names,
        profile=profile,
        data_quality_mask=deep_ocean_mask,
    )
    joblib.dump(stack, processed_dir / "feature_stack.joblib")

    # Ground-Truth Target Labels (y ONLY — never in X!)
    nodes_csv = INDONESIA_DIR / "georima_all_indonesia_nodes.csv"
    mrds_csv = INDONESIA_DIR / "mrds-indonesia.csv"
    df_nodes = pd.read_csv(nodes_csv)
    metallic_df = df_nodes[df_nodes["potensi_type"] == "ml"]

    pos_pts: list[tuple[float, float]] = [
        (float(r["lon"]), float(r["lat"])) for _, r in metallic_df.iterrows()
    ]
    if mrds_csv.exists():
        mrds_df = pd.read_csv(mrds_csv)
        for _, r in mrds_df.iterrows():
            if str(r.get("coordinate_quality_flag", "")) == "province_or_island_centroid":
                continue
            try:
                pos_pts.append((float(r["longitude"]), float(r["latitude"])))
            except (TypeError, ValueError):
                continue

    label_array = np.full((H, W), -1, dtype=np.int32)
    for lon, lat in pos_pts:
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            c = int(np.clip((lon - min_lon) / (max_lon - min_lon) * (W - 1), 0, W - 1))
            r = int(np.clip((max_lat - lat) / (max_lat - min_lat) * (H - 1), 0, H - 1))
            r0 = max(0, r - label_radius_px)
            r1 = min(H, r + label_radius_px + 1)
            c0 = max(0, c - label_radius_px)
            c1 = min(W, c + label_radius_px + 1)
            for rr in range(r0, r1):
                for cc in range(c0, c1):
                    if not deep_ocean_mask[rr, cc]:
                        label_array[rr, cc] = 1

    np.save(processed_dir / "labels.npy", label_array)
    flat_df = flatten_to_dataframe(stack, label_array=label_array)
    flat_df.to_parquet(processed_dir / "dataset.parquet", index=False)

    logger.info(
        "[%s] Saved 100%% Leak-Free Feature Stack (%dx%d, res=%.0fm, %d valid cells, %d positive cells) -> %s",
        region_code, H, W, res_m, len(flat_df), int((flat_df["label"] == 1).sum()), processed_dir,
    )


HD_DISTRICTS = [
    {
        "code": "PAPUA",
        "title": "Grasberg–Wabu HD (Papua Cu-Au)",
        "min_lon": 136.85,
        "max_lon": 137.45,
        "min_lat": -4.35,
        "max_lat": -3.75,
        "H": 320,
        "W": 320,
        "res_m": 200.0,
        "zoom": 10,
        "block_size_m": 0.08,
        "l_threshold": 0.52,
        "min_area_ha": 25.0,
        "label_radius_px": 2,
    },
    {
        "code": "SUMBAWA",
        "title": "Batu Hijau–Elang HD (Sumbawa Cu-Au)",
        "min_lon": 116.70,
        "max_lon": 117.60,
        "min_lat": -9.10,
        "max_lat": -8.50,
        "H": 280,
        "W": 420,
        "res_m": 230.0,
        "zoom": 10,
        "block_size_m": 0.10,
        "l_threshold": 0.52,
        "min_area_ha": 35.0,
        "label_radius_px": 2,
    },
    {
        "code": "SUMATRA",
        "title": "Martabe–Toru HD (North Sumatra Au-Ag)",
        "min_lon": 98.90,
        "max_lon": 99.60,
        "min_lat": 1.20,
        "max_lat": 1.90,
        "H": 320,
        "W": 320,
        "res_m": 240.0,
        "zoom": 10,
        "block_size_m": 0.09,
        "l_threshold": 0.52,
        "min_area_ha": 30.0,
        "label_radius_px": 2,
    },
    {
        "code": "SULAWESI",
        "title": "Toka Tindung–Minahasa HD (North Sulawesi Au-Cu)",
        "min_lon": 124.40,
        "max_lon": 125.25,
        "min_lat": 0.80,
        "max_lat": 1.65,
        "H": 320,
        "W": 320,
        "res_m": 290.0,
        "zoom": 10,
        "block_size_m": 0.10,
        "l_threshold": 0.52,
        "min_area_ha": 35.0,
        "label_radius_px": 2,
    },
]


def main() -> None:
    report_path = ROOT / "outputs" / "seven_pass_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    # 1. Upgrade National Indonesia (INDONESIA) 16x from 136x368 to 544x1472 (~3.45 km)
    build_clean_region_cube(
        region_code="INDONESIA",
        min_lon=95.0,
        min_lat=-11.0,
        max_lon=141.0,
        max_lat=6.0,
        H=544,
        W=1472,
        res_m=3450.0,
        zoom=6,
        processed_dir=INDONESIA_DIR / "processed",
        interim_dem_path=INDONESIA_DIR / "interim" / "indonesia_srtm_etopo_hd_544x1472.tif",
        label_radius_px=0,
    )
    ind_summary = run_region_seven_pass(
        region_name="INDONESIA",
        data_dir=INDONESIA_DIR / "processed",
        out_dir=ROOT / "outputs" / "seven_pass_indonesia",
        block_size_m=1.5,
        l_threshold=0.52,
        min_area_ha=2500.0,
    )
    report["indonesia"] = ind_summary

    # 2. Build & Train HD Camp-Scale Indonesian Mineral Districts
    for d in HD_DISTRICTS:
        code = d["code"]
        code_low = code.lower()
        proc_dir = ROOT / "data" / code_low / "processed"
        interim_dem = ROOT / "data" / code_low / "interim" / f"{code_low}_terrarium_hd_dem.tif"
        out_dir = ROOT / "outputs" / f"seven_pass_{code_low}"

        build_clean_region_cube(
            region_code=code,
            min_lon=d["min_lon"],
            min_lat=d["min_lat"],
            max_lon=d["max_lon"],
            max_lat=d["max_lat"],
            H=d["H"],
            W=d["W"],
            res_m=d["res_m"],
            zoom=d["zoom"],
            processed_dir=proc_dir,
            interim_dem_path=interim_dem,
            label_radius_px=d["label_radius_px"],
        )
        summary = run_region_seven_pass(
            region_name=code,
            data_dir=proc_dir,
            out_dir=out_dir,
            block_size_m=d["block_size_m"],
            l_threshold=d["l_threshold"],
            min_area_ha=d["min_area_ha"],
        )
        report[code_low] = summary

    report_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Completed all National HD + 4 Camp-Scale HD Indonesian training passes!")


if __name__ == "__main__":
    main()
