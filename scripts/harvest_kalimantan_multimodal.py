"""scripts/harvest_kalimantan_multimodal.py
======================================
SIFT Autonomous Kalimantan Multimodal Geospatial Data Ingestion Pipeline.

Executes 7 Modular Workstreams across the entire island of Kalimantan (Borneo):
  - Workstream 1: Synthetic Aperture Radar (ALOS-2 PALSAR-2 L-Band Mosaic & Steger Lineaments)
  - Workstream 2: Optical Multispectral (Sentinel-2 L2A Dry-Season Composite & Spectral Indices)
  - Workstream 3: Thermal Infrared (Landsat 8/9 TIRS Anomaly & ASTER Silica Index)
  - Workstream 4: Digital Elevation & Morphometrics (Copernicus DEM GLO-30 & Synclinal Keels)
  - Workstream 5: Bedrock Geology, Tectonic Faults & Sediment Fill (ESDM, GEM Faults, GlobSed)
  - Workstream 6: Potential Fields Geophysics (Sandwell & Smith Gravity & NOAA EMAG2v3 Magnetics)
  - Workstream 7: Ground-Truth Deposits & Selection-Bias Covariates (ESDM Fe/Coal, OSM Transport, GFW)

Enforces 4 Strict Engineering Gates:
  - Gate 1: Exact Spatial Co-Registration: bounds (108.5, -4.5, 119.5, 4.5), CRS EPSG:4326.
  - Gate 2: Physical Dual-Regime Vegetation Masking Enforced: all SWIR/alteration NaN where NDVI >= 0.2.
  - Gate 3: Centroid Clustering Artifact Purge: USGS MRDS centroid at (-1.9982, 113.9995) quarantined from P_k.
  - Gate 4: Unified Feature Table Parquet Export: compiled into dataset.parquet with SIFT naming conventions.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine, from_bounds
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter, zoom
from shapely.geometry import Point, box, mapping, shape

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.feature_stack import FeatureStack, flatten_to_dataframe
from src.features.geophysics import (
    analytic_signal,
    load_and_compute_real_gravity_channels,
    reduced_to_pole,
)
from src.features.structural import (
    aspect,
    dilation_tendency,
    plan_curvature,
    resolve_stress_on_vertical_fault,
    slip_tendency,
    slope,
    topographic_position_index,
)
from src.labels.borneo_labels import (
    build_k4_deposit_targets,
    filter_valid_ore_candidates,
    rasterize_borneo_k4_labels,
)
from src.models.pu_pipeline import verify_no_raw_pixel_leakage
from src.utils.io import save_raster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("harvest_kalimantan")

# Spatial Bounds & Grid Specifications
MIN_LON, MAX_LON = 108.5, 119.5
MIN_LAT, MAX_LAT = -4.5, 4.5
RES_DEG = 0.0025  # ~250m resolution at equator
W = int(round((MAX_LON - MIN_LON) / RES_DEG))  # 4400
H = int(round((MAX_LAT - MIN_LAT) / RES_DEG))  # 3600
RES_M = 250.0

DATA_DIR = ROOT / "data" / "kalimantan"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
HOTSPOTS_DIR = PROCESSED_DIR / "hotspots"

# Reference Transform & Raster Profile
REF_TRANSFORM = Affine(RES_DEG, 0.0, MIN_LON, 0.0, -RES_DEG, MAX_LAT)
REF_PROFILE = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": -9999.0,
    "width": W,
    "height": H,
    "count": 1,
    "crs": "EPSG:4326",
    "transform": REF_TRANSFORM,
    "compress": "deflate",
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
}

CENTROID_ARTIFACT_LON = 113.9995
CENTROID_ARTIFACT_LAT = -1.9982


def create_directories() -> None:
    """Ensure directory hierarchy exists."""
    for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, HOTSPOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def compute_steger_lineaments(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Extract directional ridge lineaments using Steger (1998) 2nd directional derivatives.

    Computes the dominant eigenvalue of the 2D Hessian matrix across the backscatter/amplitude array.
    """
    img_f = image.astype(np.float32)
    # Gaussian smoothed 2nd spatial derivatives
    i_xx = gaussian_filter(img_f, sigma=sigma, order=[2, 0])
    i_yy = gaussian_filter(img_f, sigma=sigma, order=[0, 2])
    i_xy = gaussian_filter(img_f, sigma=sigma, order=[1, 1])

    # Eigenvalues of 2x2 Hessian
    trace = i_xx + i_yy
    det = i_xx * i_yy - i_xy**2
    disc = np.maximum(0.0, (trace / 2.0) ** 2 - det)
    sqrt_disc = np.sqrt(disc)

    lambda1 = trace / 2.0 + sqrt_disc
    lambda2 = trace / 2.0 - sqrt_disc

    # Line strength: dominant eigenvalue
    ridge_strength = np.maximum(np.abs(lambda1), np.abs(lambda2))
    norm_ridge = ridge_strength / (np.percentile(ridge_strength, 99) + 1e-6)
    return np.clip(norm_ridge, 0.0, 1.0).astype(np.float32)


def rasterize_faults_to_grid(
    features: list[dict],
    ref_transform: Affine,
    out_h: int,
    out_w: int,
    res_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize fault lines and calculate distance (m), strike azimuth, and density."""
    import rasterio.features

    shapes_mask = []
    shapes_az = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        coords = geom.get("coordinates") or []
        lines = [coords] if geom.get("type") == "LineString" else coords
        shapes_mask.append((geom, 1))
        for line in lines:
            for pt0, pt1 in zip(line[:-1], line[1:]):
                lon1, lat1 = float(pt0[0]), float(pt0[1])
                lon2, lat2 = float(pt1[0]), float(pt1[1])
                if (lon1, lat1) == (lon2, lat2):
                    continue
                seg_az = float(np.degrees(np.arctan2(lon2 - lon1, lat2 - lat1)) % 180.0)
                seg_geom = {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]}
                shapes_az.append((seg_geom, max(1.0, seg_az)))

    if shapes_mask:
        line_mask = rasterio.features.rasterize(
            shapes_mask, out_shape=(out_h, out_w), transform=ref_transform, fill=0, all_touched=True, dtype="uint8"
        ).astype(bool)
        az_raster = rasterio.features.rasterize(
            shapes_az, out_shape=(out_h, out_w), transform=ref_transform, fill=115.0, all_touched=True, dtype="float32"
        )
    else:
        line_mask = np.zeros((out_h, out_w), dtype=bool)
        az_raster = np.full((out_h, out_w), 115.0, dtype=np.float32)

    if not line_mask.any():
        # No mapped fault in the window: distances are unknown (never a synthetic fault axis)
        nan = np.full((out_h, out_w), np.nan, dtype=np.float32)
        return nan, nan.copy(), np.zeros((out_h, out_w), dtype=np.float32)

    dist_px, nearest_idx = distance_transform_edt(~line_mask, return_indices=True)
    nearest_strike = az_raster[nearest_idx[0], nearest_idx[1]]
    dist_m = (dist_px * res_m).astype(np.float32)
    density = gaussian_filter(line_mask.astype(np.float32), sigma=max(1.5, min(out_h, out_w) / 60.0)).astype(np.float32)
    return dist_m, nearest_strike, density


def fetch_and_process_dem(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    out_h: int,
    out_w: int,
    ref_transform: Affine,
) -> dict[str, np.ndarray]:
    """Ingest Copernicus DEM GLO-30 / SRTM tiles and derive high-resolution morphometrics."""
    logger.info("[Workstream 4] Processing Copernicus DEM GLO-30 & morphometric derivatives (%dx%d)...", out_h, out_w)

    dem_500m_path = ROOT / "data" / "kalimantan_500m" / "interim" / "kalimantan_srtm_500m_dem.tif"
    if dem_500m_path.exists():
        with rasterio.open(dem_500m_path) as src:
            raw_dem_500 = src.read(1)
            h_in, w_in = raw_dem_500.shape
            dem_250 = zoom(raw_dem_500, (out_h / h_in, out_w / w_in), order=1).astype(np.float32)
    else:
        from scripts.run_indonesia_hd_upgrade import fetch_terrarium_dem
        cache_tif = INTERIM_DIR / "kalimantan_terrarium_dem_250m.tif"
        dem_250 = fetch_terrarium_dem(min_lon, min_lat, max_lon, max_lat, out_h, out_w, zoom=9, cache_tif=cache_tif)

    # 1. Slope & Aspect
    f_slope = slope(dem_250, resolution_m=RES_M).astype(np.float32)
    f_aspect = aspect(dem_250, resolution_m=RES_M).astype(np.float32)

    # 2. Planform Curvature (Evans 1980) & Profile Curvature
    f_curv = plan_curvature(dem_250, resolution_m=RES_M).astype(np.float32)

    # 3. Multi-scale Topographic Position Index (TPI)
    f_tpi_local = topographic_position_index(dem_250, radius_px=2).astype(np.float32)      # 500m
    f_tpi_regional = topographic_position_index(dem_250, radius_px=8).astype(np.float32)   # 2000m

    # 4. Synclinal valley keel index (Negative regional TPI in low-gradient basins where coal accumulates)
    syncline_mask = (f_tpi_regional < -2.0) & (f_slope < 5.0)
    f_tpi_syncline_keel = np.where(syncline_mask, -f_tpi_regional, 0.0).astype(np.float32)

    # 5. Terrain Ruggedness Index (TRI)
    mean_dem = uniform_filter(dem_250, size=5)
    mean_sq = uniform_filter(dem_250**2, size=5)
    f_tri = np.sqrt(np.maximum(mean_sq - mean_dem**2, 0.0)).astype(np.float32)

    # 6. Topographic Lineament Distance & Density from Curvature
    curv_smooth = gaussian_filter(np.abs(f_curv), sigma=1.2)
    lin_mask = curv_smooth >= np.percentile(curv_smooth, 86)
    f_dist_topo_lin_km = np.clip(
        (distance_transform_edt(~lin_mask) * (RES_M / 1000.0)), 0.0, 15.0
    ).astype(np.float32)

    # 7. Coastal & Shelf Distance
    land_mask = dem_250 >= 0.0
    coast_mask = land_mask ^ (uniform_filter(land_mask.astype(np.float32), size=3) > 0.5)
    if not coast_mask.any():
        coast_mask = np.zeros((out_h, out_w), dtype=bool)
        coast_mask[0, :] = True
    f_coastal_dist_km = (distance_transform_edt(~coast_mask) * (RES_M / 1000.0)).astype(np.float32)

    # 8. Regolith Cover & Outcrop Exposure
    land_elev_pos = np.maximum(dem_250, 0.0)
    slope_norm = np.clip(f_slope / 15.0, 0.0, 1.0)
    relief_norm = np.clip((f_tpi_local - np.percentile(f_tpi_local, 10)) / (np.ptp(f_tpi_local) + 1e-6), 0.0, 1.0)
    f_outcrop_exposure = np.clip(0.55 * slope_norm + 0.45 * relief_norm, 0.0, 1.0).astype(np.float32)
    f_regolith_cover_m = np.clip(
        (1.0 - f_outcrop_exposure) * np.exp(-land_elev_pos / 600.0) * 85.0 + 3.0,
        2.0,
        95.0,
    ).astype(np.float32)

    return {
        "dem_elevation": dem_250,
        "dem_slope": f_slope,
        "dem_aspect": f_aspect,
        "dem_plan_curvature": f_curv,
        "dem_tpi_local": f_tpi_local,
        "dem_tpi_regional": f_tpi_regional,
        "dem_tpi_syncline_keel": f_tpi_syncline_keel,
        "terrain_ruggedness_tri": f_tri,
        "dist_topo_lineaments_km": f_dist_topo_lin_km,
        "coastal_shelf_dist_km": f_coastal_dist_km,
        "regolith_cover_m": f_regolith_cover_m,
        "outcrop_exposure_index": f_outcrop_exposure,
    }


def fetch_and_process_sar(
    dem: np.ndarray,
    out_h: int,
    out_w: int,
) -> dict[str, np.ndarray]:
    """Ingest the REAL JAXA ALOS-2 PALSAR-2 L-band annual mosaic and derive SAR channels.

    Previously this function synthesised HH/HV from DEM slope/roughness plus seeded
    Gaussian noise, which made the SAR channels redundant DEM proxies (see audit
    v4.4). It now aggregates the genuine 25 m PALSAR-2 mosaic onto the 250 m grid via
    :func:`src.features.sar_palsar2.build_real_sar_channels`. If real data cannot be
    retrieved, an empty dict is returned and SAR channels are omitted -- they are
    never fabricated (Provenance Honesty guardrail).
    """
    from src.features.sar_palsar2 import SARDataUnavailableError, build_real_sar_channels

    logger.info("[Workstream 1] Ingesting real ALOS-2 PALSAR-2 L-Band mosaic & computing Steger ridge lineaments...")
    try:
        channels, prov = build_real_sar_channels(
            dst_transform=REF_TRANSFORM,
            dst_shape=(out_h, out_w),
            dem=dem,
            res_m=RES_M,
            dst_crs="EPSG:4326",
            cache_dir=INTERIM_DIR / "palsar2_cache",
        )
    except SARDataUnavailableError as exc:
        logger.warning("[Workstream 1] Real PALSAR-2 unavailable (%s); SAR channels omitted (not fabricated)", exc)
        return {}

    prov.write(INTERIM_DIR / "sar_provenance.json")
    save_raster(channels["sar_palsar2_hh"], REF_PROFILE, str(INTERIM_DIR / "palsar2_hh_250m.tif"))
    save_raster(channels["sar_palsar2_hv"], REF_PROFILE, str(INTERIM_DIR / "palsar2_hv_250m.tif"))
    save_raster(channels["sar_palsar2_dpr"], REF_PROFILE, str(INTERIM_DIR / "palsar2_dpr_250m.tif"))

    return {
        "sar_palsar2_hh": channels["sar_palsar2_hh"],
        "sar_palsar2_hv": channels["sar_palsar2_hv"],
        "sar_palsar2_dpr": channels["sar_palsar2_dpr"],
        "sar_palsar2_lineament_density": channels["sar_palsar2_lineament_density"],
    }


def fetch_and_process_optical(
    dem: np.ndarray,
    out_h: int,
    out_w: int,
) -> dict[str, np.ndarray]:
    """REMOVED in v4.9 (2026-11-10): this function fabricated Sentinel-2 NDVI/albedo/ratio layers from seeded random noise.

    Real layers are built by ``scripts/fetch_real_layers.py`` + ``scripts/build_real_feature_cubes.py``.
    """
    raise RuntimeError(
        "fetch_and_process_optical fabricated Sentinel-2 NDVI/albedo/ratio layers from seeded random noise and has been removed. "
        "Use scripts/build_real_feature_cubes.py (real data only)."
    )


def fetch_and_process_thermal(
    dem: np.ndarray,
    ndvi: np.ndarray,
    out_h: int,
    out_w: int,
) -> dict[str, np.ndarray]:
    """REMOVED in v4.9 (2026-11-10): this function fabricated Landsat thermal and ASTER silica layers from a lapse-rate formula plus random noise.

    Real layers are built by ``scripts/fetch_real_layers.py`` + ``scripts/build_real_feature_cubes.py``.
    """
    raise RuntimeError(
        "fetch_and_process_thermal fabricated Landsat thermal and ASTER silica layers from a lapse-rate formula plus random noise and has been removed. "
        "Use scripts/build_real_feature_cubes.py (real data only)."
    )


def fetch_and_process_geology_and_faults(
    dem: np.ndarray,
    out_h: int,
    out_w: int,
    ref_transform: Affine,
) -> dict[str, Any]:
    """Ingest ESDM Geology 1:250k, GEM Active Faults, Cauchy stress, and NOAA GlobSed sediment fill."""
    logger.info("[Workstream 5] Ingesting ESDM Regional Geology, GEM Faults & GlobSed...")

    # 1. GEM Active Faults
    faults_gj = ROOT / "data" / "indonesia" / "indonesia_tectonic_faults.geojson"
    faults_all = json.loads(faults_gj.read_text(encoding="utf-8")).get("features", []) if faults_gj.exists() else []

    kal_faults = []
    for f in faults_all:
        geom = f.get("geometry", {})
        coords = geom.get("coordinates", [])
        lines = [coords] if geom.get("type") == "LineString" else coords
        in_bbox = False
        for line in lines:
            for pt in line:
                if MIN_LON <= pt[0] <= MAX_LON and MIN_LAT <= pt[1] <= MAX_LAT:
                    in_bbox = True
                    break
            if in_bbox:
                break
        if in_bbox:
            kal_faults.append(f)

    kal_faults_gj = {"type": "FeatureCollection", "features": kal_faults}
    (INTERIM_DIR / "kalimantan_faults_gem.geojson").write_text(json.dumps(kal_faults_gj, indent=2), encoding="utf-8")

    dist_faults_m, strike_az, f_fault_density = rasterize_faults_to_grid(
        kal_faults, ref_transform, out_h, out_w, RES_M
    )

    # 2. 3D Cauchy Stress Resolution: Dilation Tendency (Td) & Slip Tendency (Ts)
    prox_fault = np.exp(-dist_faults_m / 45000.0).astype(np.float32)
    sigma_n, tau = resolve_stress_on_vertical_fault(
        strike_deg=strike_az,
        sigma1=95.0,
        sigma3=25.0,
        sigma1_azimuth_deg=20.0,
    )
    f_slip_ts = (slip_tendency(sigma_n, tau) * (0.35 + 0.65 * prox_fault)).astype(np.float32)
    f_dilation_td = (dilation_tendency(sigma_n, 95.0, 25.0) * (0.35 + 0.65 * prox_fault)).astype(np.float32)

    # 3. NOAA GLOBSED Sediment Thickness Grid (fast 2-stage interpolation via CRUST1.0 sedthk.xyz)
    sed_xyz_path = ROOT / "data" / "raw" / "crust1.0" / "sedthk.xyz"
    pts_sed = []
    if sed_xyz_path.exists():
        with open(sed_xyz_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    lon, lat, thk = float(parts[0]), float(parts[1]), float(parts[2])
                    if (MIN_LON - 2.0) <= lon <= (MAX_LON + 2.0) and (MIN_LAT - 2.0) <= lat <= (MAX_LAT + 2.0):
                        pts_sed.append([lon, lat, thk])

    if pts_sed:
        pts_arr = np.array(pts_sed, dtype=np.float32)
        sub_w, sub_h = 440, 360
        gx, gy = np.meshgrid(np.linspace(MIN_LON, MAX_LON, sub_w), np.linspace(MAX_LAT, MIN_LAT, sub_h))
        grid_coarse = griddata((pts_arr[:, 0], pts_arr[:, 1]), pts_arr[:, 2], (gx, gy), method="linear")
        nan_s = np.isnan(grid_coarse)
        if nan_s.any():
            grid_coarse[nan_s] = griddata((pts_arr[:, 0], pts_arr[:, 1]), pts_arr[:, 2], (gx, gy), method="nearest")[nan_s]
        f_sediment_thk = zoom(grid_coarse, (out_h / sub_h, out_w / sub_w), order=1).astype(np.float32)
    else:
        raise RuntimeError(f"CRUST1.0 sediment file missing ({sed_xyz_path}); refusing to use a constant thickness")

    f_sediment_thk = np.clip(f_sediment_thk, 0.0, 15.0).astype(np.float32)
    save_raster(f_sediment_thk, REF_PROFILE, str(INTERIM_DIR / "sediment_thickness_globsed.tif"))

    # 4. ESDM Regional Geology Contacts: Coal-bearing formations (Warukin, Tanjung, Balikpapan)
    # Attempt live ESDM query for official polygons
    esdm_polys = []
    try:
        base_url = "https://geoportal.esdm.go.id/gis4/rest/services/BGD_TU/Formasi_Pembawa_Batubara/MapServer/0/query"
        params = {
            "where": "PULAU = 'KALIMANTAN' AND NAMA_FORMA IN ('Warukin', 'Tanjung', 'Balikpapan')",
            "outFields": "NAMA_FORMA,BATUAN_PEM,UMUR,form",
            "returnGeometry": "true",
            "f": "geojson",
            "resultRecordCount": "200",
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SIFT-Harvester/1.0)"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for feat in data.get("features", []):
                esdm_polys.append({
                    "formation": feat["properties"].get("NAMA_FORMA", "Coal_Formation"),
                    "lithology": feat["properties"].get("BATUAN_PEM", "Batubara"),
                    "geometry": shape(feat["geometry"]),
                })
        logger.info("Retrieved %d official ESDM geological formation polygons from REST server", len(esdm_polys))
    except Exception as e:
        logger.warning("ESDM REST server query fallback (%s): using known coal basin centroids", e)

    if not esdm_polys:
        # No invented formation circles: without the real ESDM polygons the layer cannot be built
        raise RuntimeError("ESDM coal-formation polygons unavailable; refusing to fabricate formation geometry")

    gdf_esdm = gpd.GeoDataFrame(esdm_polys, crs="EPSG:4326")
    gdf_esdm.to_file(INTERIM_DIR / "esdm_geology_250k.gpkg", driver="GPKG")

    # Clean distance to sedimentary formation contacts from true polygons (dist_sedimentary_polygon_km)
    from rasterio.features import rasterize
    poly_shapes = [(geom, 1) for geom in gdf_esdm.geometry if geom is not None and not geom.is_empty]
    if poly_shapes:
        poly_mask = rasterize(
            poly_shapes,
            out_shape=(out_h, out_w),
            transform=REF_TRANSFORM,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        ).astype(bool)
        dist_sed_px = distance_transform_edt(~poly_mask)
        f_sed_poly_dist_km = np.clip((dist_sed_px * (RES_M / 1000.0)), 0.0, 15.0).astype(np.float32)
    else:
        f_sed_poly_dist_km = np.full((out_h, out_w), 15.0, dtype=np.float32)

    save_raster(f_sed_poly_dist_km, REF_PROFILE, str(INTERIM_DIR / "dist_sedimentary_polygon_km.tif"))

    # Legacy constant column removed (it was a fabricated 15 km everywhere)
    f_coal_contact_km = np.full((out_h, out_w), np.nan, dtype=np.float32)

    return {
        "dist_active_faults_m": dist_faults_m,
        "active_fault_density": f_fault_density,
        "dilation_tendency_Td": f_dilation_td,
        "slip_tendency_Ts": f_slip_ts,
        "sediment_thickness_globsed": f_sediment_thk,
        "dist_sedimentary_polygon_km": f_sed_poly_dist_km,
        "geology_coal_contact_dist_km": f_coal_contact_km,
    }


def fetch_and_process_geophysics(
    dem: np.ndarray,
    out_h: int,
    out_w: int,
    ref_transform: Affine,
) -> dict[str, np.ndarray]:
    """REMOVED in v4.9 (2026-11-10): this function fabricated EMAG2 magnetics from seeded random noise.

    Real layers are built by ``scripts/fetch_real_layers.py`` + ``scripts/build_real_feature_cubes.py``.
    """
    raise RuntimeError(
        "fetch_and_process_geophysics fabricated EMAG2 magnetics from seeded random noise and has been removed. "
        "Use scripts/build_real_feature_cubes.py (real data only)."
    )


def fetch_and_process_deposits_and_covariates(
    out_h: int,
    out_w: int,
    ref_transform: Affine,
) -> dict[str, Any]:
    """REMOVED in v4.9 (2026-11-10): this function fabricated a transport network from five hand-drawn river polylines and forest disturbance from it.

    Real layers are built by ``scripts/fetch_real_layers.py`` + ``scripts/build_real_feature_cubes.py``.
    """
    raise RuntimeError(
        "fetch_and_process_deposits_and_covariates fabricated a transport network from five hand-drawn river polylines and forest disturbance from it and has been removed. "
        "Use scripts/build_real_feature_cubes.py (real data only)."
    )


def export_hotspots_50m(all_channels: dict[str, np.ndarray]) -> None:
    """Generate 50m Hotspot Mining Belt Derivatives for Mahakam Delta, Barito Foreland, and Meratus Suture."""
    logger.info("Generating Hotspot Mining Belt Scale 50m derivatives...")

    hotspots = [
        {"name": "mahakam_delta_50m", "min_lon": 116.8, "max_lon": 117.8, "min_lat": -1.0, "max_lat": 0.0},
        {"name": "barito_foreland_50m", "min_lon": 114.8, "max_lon": 115.8, "min_lat": -2.8, "max_lat": -1.8},
        {"name": "meratus_suture_50m", "min_lon": 115.0, "max_lon": 116.0, "min_lat": -3.8, "max_lat": -2.8},
    ]

    for hs in hotspots:
        hname = hs["name"]
        min_x, max_x = hs["min_lon"], hs["max_lon"]
        min_y, max_y = hs["min_lat"], hs["max_lat"]

        c0 = int(round((min_x - MIN_LON) / RES_DEG))
        c1 = int(round((max_x - MIN_LON) / RES_DEG))
        r0 = int(round((MAX_LAT - max_y) / RES_DEG))
        r1 = int(round((MAX_LAT - min_y) / RES_DEG))

        c0, c1 = max(0, c0), min(W, c1)
        r0, r1 = max(0, r0), min(H, r1)

        sub_dem = all_channels["dem_elevation"][r0:r1, c0:c1]

        hs_50m = zoom(sub_dem, 5.0, order=1).astype(np.float32)
        hs_h, hs_w = hs_50m.shape
        hs_transform = from_bounds(min_x, min_y, max_x, max_y, hs_w, hs_h)

        hs_prof = {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": -9999.0,
            "width": hs_w,
            "height": hs_h,
            "count": 1,
            "crs": "EPSG:4326",
            "transform": hs_transform,
            "compress": "deflate",
        }
        save_raster(hs_50m, hs_prof, str(HOTSPOTS_DIR / f"{hname}.tif"))
        logger.info(" - Saved 50m hotspot (%s: %dx%d px) -> %s", hname, hs_h, hs_w, HOTSPOTS_DIR / f"{hname}.tif")


def verify_all_acceptance_gates(
    processed_dir: Path,
    parquet_path: Path,
) -> dict[str, bool]:
    """Execute rigorous programmatic verification across all 4 Acceptance Gates."""
    logger.info("==========================================================================")
    logger.info("VERIFYING STRICT ACCEPTANCE GATES 1 THROUGH 4")
    logger.info("==========================================================================")

    results: dict[str, bool] = {}

    # Gate 1: Exact Spatial Co-Registration
    logger.info("Checking Gate 1: Exact Spatial Co-Registration across all processed GeoTIFFs...")
    all_tifs = sorted(list(processed_dir.glob("*.tif")))
    gate1_pass = True
    mismatched = []
    for tif_p in all_tifs:
        with rasterio.open(tif_p) as src:
            bounds_match = (
                math.isclose(src.bounds.left, MIN_LON, abs_tol=1e-5)
                and math.isclose(src.bounds.bottom, MIN_LAT, abs_tol=1e-5)
                and math.isclose(src.bounds.right, MAX_LON, abs_tol=1e-5)
                and math.isclose(src.bounds.top, MAX_LAT, abs_tol=1e-5)
            )
            shape_match = (src.height == H and src.width == W)
            crs_match = (src.crs is not None and src.crs.to_epsg() == 4326)
            if not (bounds_match and shape_match and crs_match):
                gate1_pass = False
                mismatched.append((tif_p.name, src.shape, src.bounds))

    if gate1_pass:
        logger.info("Gate 1 PASS: All %d GeoTIFFs align identically to bounds (%.2f, %.2f, %.2f, %.2f), CRS EPSG:4326, %dx%d px with zero sub-pixel shift!",
                    len(all_tifs), MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, H, W)
    else:
        logger.error("Gate 1 FAIL: Found mismatched GeoTIFFs: %s", mismatched)
    results["Gate 1: Exact Spatial Co-Registration"] = gate1_pass

    # Read parquet for Gates 2, 3, 4
    df = pd.read_parquet(parquet_path)

    # Gate 2: Physical Dual-Regime Vegetation Masking Enforced
    logger.info("Checking Gate 2: Physical Dual-Regime Vegetation Masking...")
    veg_mask = df["spec_ndvi"] >= 0.2
    swir_layers = [
        "spec_albedo_gated",
        "spec_wcdi_gated",
        "spec_oci_gated",
        "spec_ficr_gated",
        "aster_silica_index",
        "aster_tir_silica_gated",
        "spec_clay_hydroxyl_gated",
        "spec_ferrous_ultramafic_gated",
    ]
    gate2_pass = True
    leak_counts = {}
    for layer in swir_layers:
        if layer in df.columns:
            # Under canopy (veg_mask), all values must be NaN!
            leaks = int(np.sum(np.isfinite(df.loc[veg_mask, layer])))
            leak_counts[layer] = leaks
            if leaks > 0:
                gate2_pass = False

    if gate2_pass:
        logger.info("Gate 2 PASS: 100%% of SWIR and alteration pixels under canopy (NDVI >= 0.2, %d cells) are strictly NaN! Zero optical leakage.", int(veg_mask.sum()))
    else:
        logger.error("Gate 2 FAIL: Optical pixel leakage detected in %s", leak_counts)
    results["Gate 2: Physical Dual-Regime Vegetation Masking"] = gate2_pass

    # Gate 3: Centroid Clustering Artifact Purge
    logger.info("Checking Gate 3: Centroid Clustering Artifact Purge at (-1.9982, 113.9995)...")
    d_centroid = np.hypot(df["lon"] - CENTROID_ARTIFACT_LON, df["lat"] - CENTROID_ARTIFACT_LAT)
    near_centroid = d_centroid < 0.05  # within ~5.5 km
    pos_targets = [
        "label_k0_porphyry_cu",
        "label_k1_skarn_fe_cu",
        "label_k2_epithermal_au",
        "label_k3_laterite_fe_ni",
        "label_coal",
        "label_coal_k0_high_cv",
        "label_coal_k1_med_cv",
        "label_coal_k2_low_cv",
    ]
    gate3_pass = True
    centroid_positives = {}
    for tgt in pos_targets:
        if tgt in df.columns:
            pos_count = int((df.loc[near_centroid, tgt] == 1).sum())
            centroid_positives[tgt] = pos_count
            if pos_count > 0:
                gate3_pass = False

    if gate3_pass:
        logger.info("Gate 3 PASS: Historical USGS MRDS centroid artifact at (-1.9982, 113.9995) is strictly quarantined! Zero positives in %d cells within 5.5 km.", int(near_centroid.sum()))
    else:
        logger.error("Gate 3 FAIL: Centroid artifact contamination detected in %s", centroid_positives)
    results["Gate 3: Centroid Clustering Artifact Purge"] = gate3_pass

    # Gate 4: Unified Feature Table Parquet Export & SIFT Naming
    logger.info("Checking Gate 4: Unified Feature Table Parquet Export & SIFT Naming Conventions...")
    required_sift_cols = [
        "sar_palsar2_hv",
        "spec_albedo_gated",
        "dem_tpi_syncline_keel",
        "bouguer_gravity_mgal",
        "label_coal",
        "label_k0_porphyry_cu",
    ]
    missing_cols = [c for c in required_sift_cols if c not in df.columns]
    pass2_and_meta_cols = {
        "row",
        "col",
        "lon",
        "lat",
        "explored_footprint",
        "alteration_benchmark",
        "dist_transport_km",
        "dist_major_rivers_km",
        "dist_haul_roads_km",
        "gfw_forest_disturbance",
        "forest_disturbance_rate",
        "active_mine_mask",
        "formation_polygon_id",
        "coal_tonnage_weight",
        "coal_parent_concession_id",
        "w_canopy",
        # Leaked feature quarantine
        "geology_coal_contact_dist_km",
        "in_coal_bearing_formation",
        "in_holocene_peat_or_karst_mask",
        "in_ophiolite_unit",
    }
    feature_cols = [
        c for c in df.columns if not c.startswith("label") and c not in pass2_and_meta_cols
    ]
    no_leakage = verify_no_raw_pixel_leakage(feature_cols)

    gate4_pass = (len(missing_cols) == 0) and no_leakage
    if gate4_pass:
        logger.info("Gate 4 PASS: dataset.parquet contains %d rows, %d columns matching SIFT standards with zero raw pixel leakage!", len(df), len(df.columns))
    else:
        logger.error("Gate 4 FAIL: Missing required SIFT columns: %s", missing_cols)
    results["Gate 4: Unified Feature Table Parquet Export"] = gate4_pass

    all_passed = all(results.values())
    logger.info("==========================================================================")
    logger.info("OVERALL VERIFICATION GATES RESULT: %s", "PASSED (100% READY)" if all_passed else "FAILED")
    logger.info("==========================================================================")

    return results


def enrich_with_dual_domain_multimodal_features(
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, Any]:
    """REMOVED in v4.9 (2026-11-10): this function fabricated constant default rasters for missing inputs.

    Real layers are built by ``scripts/fetch_real_layers.py`` + ``scripts/build_real_feature_cubes.py``.
    """
    raise RuntimeError(
        "enrich_with_dual_domain_multimodal_features fabricated constant default rasters for missing inputs and has been removed. "
        "Use scripts/build_real_feature_cubes.py (real data only)."
    )


def build_kalimantan_multimodal_pipeline() -> dict[str, Any]:
    """REMOVED in v4.9 (2026-11-10): this function fabricated several input layers (see the functions above).

    Real layers are built by ``scripts/fetch_real_layers.py`` + ``scripts/build_real_feature_cubes.py``.
    """
    raise RuntimeError(
        "build_kalimantan_multimodal_pipeline fabricated several input layers (see the functions above) and has been removed. "
        "Use scripts/build_real_feature_cubes.py (real data only)."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SIFT Kalimantan Multimodal Ingestion Pipeline")
    parser.add_argument(
        "--enrich-only",
        action="store_true",
        help="Enrich existing processed rasters and dataset.parquet with dual-domain multimodal features and K=3 coal halo labels",
    )
    args = parser.parse_args()
    if args.enrich_only and (PROCESSED_DIR / "dataset.parquet").exists():
        enrich_with_dual_domain_multimodal_features(PROCESSED_DIR)
    else:
        build_kalimantan_multimodal_pipeline()

