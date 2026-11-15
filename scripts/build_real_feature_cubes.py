#!/usr/bin/env python3
"""Build the Kalimantan and East Kalimantan feature cubes from real data only, with one recipe.

Inputs are the caches filled by ``scripts/fetch_real_layers.py`` plus real files already in
the repo (GEM faults, GVP volcanoes, CRUST1.0, satellite gravity, ESDM coal formations,
GeoRIMA rock observations, deposit catalogues). Every layer is registered in
``layer_provenance.json`` as *measured* (a resampled real product) or *derived* (a
documented formula of registered real layers). Missing data stay NaN: nothing is filled
with constants, random fields or hand-drawn geometry.

Outputs (non-destructive, new directories):

* ``data/kalimantan_real/processed``       - island master grid, 0.0025 deg (~250 m), EPSG:4326
* ``data/east_kalimantan_real/processed``  - exact window of the master (114.5-119.5E, -2.8-3.2N)

Usage:
    .venv/bin/python scripts/build_real_feature_cubes.py [--skip-ek]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import rasterio.features
from rasterio.transform import Affine
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter, zoom

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_real_layers import CACHE, MASTER_GRID, RAW_REAL  # noqa: E402
from src.features.real_sources.core import (  # noqa: E402
    MEASURED,
    GridSpec,
    LayerProvenance,
    ProvenanceManifest,
    RealDataUnavailableError,
    load_layer,
    save_layer,
)
from src.features.real_sources.landsat_thermal import nan_gaussian  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("build_real_cubes")

OUT_MASTER = ROOT / "data" / "kalimantan_real" / "processed"
OUT_EK = ROOT / "data" / "east_kalimantan_real" / "processed"
EK_WINDOW = (520, 2400, 2400, 2000)  # row_off, col_off, height, width in the master grid
CENTROID_ARTIFACT = (113.9995, -1.9982)

# Unified structural recipe (the one the East Kalimantan production models were built with)
STRESS_SIGMA1, STRESS_SIGMA3, STRESS_SIGMA1_AZ = 95.0, 25.0, 20.0
FAULT_DECAY_M = 3500.0
# v4.15 (audit v4.13 section 2.1): Td/Ts inherit the strike of the NEAREST GEM fault, so beyond a few km they are a
# Voronoi partition of the fault map (14 distinct values over 3,000 km2 patches), i.e. a location code. They are
# therefore only defined within this distance of a mapped fault and are NaN elsewhere.
FAULT_MAX_DIST_M = 5000.0
TPI_LOCAL_RADIUS_PX, TPI_REGIONAL_RADIUS_PX = 3, 11

# Columns written to the parquet that are not model features (labels are excluded by prefix)
META_COLS = {
    "row", "col", "lon", "lat", "label",
    "dist_transport_km", "dist_major_rivers_km", "dist_haul_roads_km",
    "gfw_forest_disturbance", "forest_disturbance_rate", "tree_cover_2000",
    "active_mine_mask", "w_canopy", "explored_footprint", "alteration_benchmark",
    "formation_polygon_id", "in_coal_bearing_formation", "in_holocene_peat_or_karst_mask", "in_ophiolite_unit",
    "s2_clear_obs", "emag2_error_nt", "emag2_source_code", "thermal_lst_c",
    "coal_tonnage_weight", "coal_parent_concession_id",
    "water_mask", "osm_water_mask", "osm_quarry_mask", "dist_roads_km", "dist_ports_km",
    "dist_calc_alkaline_intrusive_km", "dist_pluton_carbonate_contact_km",
}


class LayerStore:
    """Writes layers to GeoTIFF as they are produced and keeps provenance."""

    def __init__(self, grid: GridSpec, out_dir: Path, manifest: ProvenanceManifest) -> None:
        self.grid, self.out_dir, self.m = grid, out_dir, manifest
        self.names: list[str] = []

    def put(self, name: str, arr: np.ndarray) -> None:
        if name not in self.m.layers:
            raise KeyError(f"Layer {name!r} has no provenance record; refusing to write it")
        arr = np.asarray(arr, dtype=np.float32)
        save_layer(arr, self.grid, self.out_dir / f"{name}.tif")
        self.m.update_valid_fraction(name, arr)
        if name not in self.names:
            self.names.append(name)

    def get(self, name: str) -> np.ndarray:
        return load_layer(self.out_dir / f"{name}.tif")


def _cache(sub: str, name: str) -> np.ndarray:
    p = CACHE / sub / f"{name}.tif"
    if not p.exists():
        raise RealDataUnavailableError(f"Missing cache {p}; run scripts/fetch_real_layers.py --source {sub}")
    return load_layer(p)


def _lonlat(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    lons = grid.transform.c + grid.transform.a * (np.arange(grid.width) + 0.5)
    lats = grid.transform.f + grid.transform.e * (np.arange(grid.height) + 0.5)
    return lons.astype(np.float64), lats.astype(np.float64)


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------
def build_terrain(S: LayerStore) -> dict[str, np.ndarray]:
    from src.features.structural import aspect, plan_curvature, slope, topographic_position_index

    g, m = S.grid, S.m
    dem = load_layer(CACHE / "dem" / "terrarium_z10_master_250m.tif")
    res = g.res_m
    m.add(LayerProvenance(
        name="dem_elevation", kind=MEASURED,
        source="Mapzen/AWS Terrain Tiles (terrarium; SRTM, GMTED2010, ETOPO1 composite)",
        url="https://registry.opendata.aws/terrain-tiles/", method="zoom-10 tiles mosaicked and resampled to grid",
        native_resolution_m=150.0))
    S.put("dem_elevation", dem)
    f_slope = slope(dem, resolution_m=res).astype(np.float32)
    f_aspect = aspect(dem, resolution_m=res).astype(np.float32)
    f_curv = plan_curvature(dem, resolution_m=res).astype(np.float32)
    f_tpi_l = topographic_position_index(dem, radius_px=TPI_LOCAL_RADIUS_PX).astype(np.float32)
    f_tpi_r = topographic_position_index(dem, radius_px=TPI_REGIONAL_RADIUS_PX).astype(np.float32)
    keel = np.where((f_tpi_r < -2.0) & (f_slope < 5.0), -f_tpi_r, 0.0).astype(np.float32)
    mean_dem = uniform_filter(dem, size=5)
    f_tri = np.sqrt(np.maximum(uniform_filter(dem ** 2, size=5) - mean_dem ** 2, 0.0)).astype(np.float32)
    curv_s = gaussian_filter(np.abs(f_curv), sigma=1.2)
    lin_mask = curv_s >= np.percentile(curv_s, 86)
    f_lin_km = (distance_transform_edt(~lin_mask) * res / 1000.0).astype(np.float32)
    land = dem >= 0.0
    coast = land ^ (uniform_filter(land.astype(np.float32), size=3) > 0.5)
    f_coast_km = (distance_transform_edt(~coast) * res / 1000.0).astype(np.float32)
    slope_n = np.clip(f_slope / 15.0, 0.0, 1.0)
    relief_n = np.clip((f_tpi_l - np.percentile(f_tpi_l, 10)) / (np.ptp(f_tpi_l) + 1e-6), 0.0, 1.0)
    f_outcrop = np.clip(0.55 * slope_n + 0.45 * relief_n, 0.0, 1.0).astype(np.float32)

    for name, arr, method in (
        ("dem_slope", f_slope, "Horn slope (deg)"),
        ("dem_aspect", f_aspect, "aspect (deg)"),
        ("dem_plan_curvature", f_curv, "Evans (1980) plan curvature"),
        ("dem_tpi_local", f_tpi_l, f"TPI, radius {TPI_LOCAL_RADIUS_PX} px"),
        ("dem_tpi_regional", f_tpi_r, f"TPI, radius {TPI_REGIONAL_RADIUS_PX} px"),
        ("dem_tpi_syncline_keel", keel, "-TPI_regional where TPI_regional<-2 and slope<5, else 0"),
        ("terrain_ruggedness_tri", f_tri, "5x5 local elevation standard deviation"),
        ("dist_topo_lineaments_km", f_lin_km, "distance to top-14% smoothed |plan curvature| cells"),
        ("coastal_shelf_dist_km", f_coast_km, "distance to the DEM 0 m coastline"),
        ("outcrop_exposure_index", f_outcrop, "0.55*slope/15 + 0.45*normalised local TPI (terrain index, not a mapped outcrop)"),
    ):
        m.derive(name, ["dem_elevation"], method)
        S.put(name, arr)
    return {"dem": dem, "slope": f_slope, "lin_km": f_lin_km, "lin_mask": lin_mask, "coast_km": f_coast_km}


# ---------------------------------------------------------------------------
# Structure (GEM faults + DEM lineaments; PB2002 plate boundaries; GVP volcanoes)
# ---------------------------------------------------------------------------
def _rasterize_lines(features: list[dict], g: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    shapes_mask, shapes_az = [], []
    for feat in features:
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        lines = [coords] if geom.get("type") == "LineString" else coords
        shapes_mask.append((geom, 1))
        for line in lines:
            for p0, p1 in zip(line[:-1], line[1:]):
                if tuple(p0[:2]) == tuple(p1[:2]):
                    continue
                az = float(np.degrees(np.arctan2(p1[0] - p0[0], p1[1] - p0[1])) % 180.0)
                shapes_az.append(({"type": "LineString", "coordinates": [p0[:2], p1[:2]]}, max(1.0, az)))
    if not shapes_mask:
        return np.zeros(g.shape, bool), np.full(g.shape, np.nan, np.float32)
    mask = rasterio.features.rasterize(shapes_mask, out_shape=g.shape, transform=g.transform, fill=0, all_touched=True, dtype="uint8").astype(bool)
    az = rasterio.features.rasterize(shapes_az, out_shape=g.shape, transform=g.transform, fill=0, all_touched=True, dtype="float32")
    az[az == 0] = np.nan
    return mask, az


def build_structure(S: LayerStore, terr: dict) -> None:
    from src.features.structural import dilation_tendency, resolve_stress_on_vertical_fault, slip_tendency

    g, m = S.grid, S.m
    res = g.res_m
    faults_gj = ROOT / "data" / "indonesia" / "indonesia_tectonic_faults.geojson"
    feats = json.loads(faults_gj.read_text(encoding="utf-8")).get("features", [])
    gem = [f for f in feats if f.get("properties", {}).get("kind") != "tectonic_plate_boundary"]
    plates = [f for f in feats if f.get("properties", {}).get("kind") == "tectonic_plate_boundary"]
    m.add(LayerProvenance(name="gem_active_faults", kind=MEASURED, source="GEM Global Active Faults Database (+ PB2002 plate boundaries)",
                          url="https://github.com/GEMScienceTools/gem-global-active-faults", method=f"{faults_gj.relative_to(ROOT)}"))
    fmask, faz = _rasterize_lines(gem, g)
    if not fmask.any():
        raise RealDataUnavailableError("No GEM faults intersect the grid")
    dist_px, idx = distance_transform_edt(~fmask, return_indices=True)
    dist_gem_m = (dist_px * res).astype(np.float32)
    strike = faz[idx[0], idx[1]]
    eff_dist_m = np.minimum(dist_gem_m, terr["lin_km"] * 1000.0 * 2.5).astype(np.float32)
    lin_density = gaussian_filter(terr["lin_mask"].astype(np.float32), sigma=3.0)
    gem_density = gaussian_filter(fmask.astype(np.float32), sigma=max(1.5, min(g.shape) / 60.0))
    density = (0.6 * gem_density + 0.4 * lin_density).astype(np.float32)
    # v4.15: proximity from the mapped GEM fault only (the 2.5 x DEM-lineament term made it a terrain-curvature distance)
    prox = np.exp(-dist_gem_m / FAULT_DECAY_M).astype(np.float32)
    near_fault = dist_gem_m <= FAULT_MAX_DIST_M
    sn, tau = resolve_stress_on_vertical_fault(strike_deg=strike, sigma1=STRESS_SIGMA1, sigma3=STRESS_SIGMA3, sigma1_azimuth_deg=STRESS_SIGMA1_AZ)
    ts = np.where(near_fault, slip_tendency(sn, tau) * (0.35 + 0.65 * prox), np.nan).astype(np.float32)
    td = np.where(near_fault, dilation_tendency(sn, STRESS_SIGMA1, STRESS_SIGMA3) * (0.35 + 0.65 * prox), np.nan).astype(np.float32)
    stress_note = (f"Andersonian 2-D stress model sigma1={STRESS_SIGMA1}, sigma3={STRESS_SIGMA3} MPa, sigma1 azimuth {STRESS_SIGMA1_AZ} deg "
                   "(assumed regional field, not measured) resolved on the nearest GEM fault strike")
    m.derive("dist_active_faults_m", ["gem_active_faults", "dist_topo_lineaments_km"], "min(distance to GEM fault, 2.5 x distance to DEM lineament)")
    m.derive("active_fault_density", ["gem_active_faults", "dist_topo_lineaments_km"], "0.6 * Gaussian GEM fault density + 0.4 * Gaussian DEM lineament density")
    m.derive("slip_tendency_Ts", ["gem_active_faults"], f"Ts * (0.35+0.65*exp(-d_gem/{FAULT_DECAY_M:.0f} m)); NaN beyond {FAULT_MAX_DIST_M:.0f} m of a mapped GEM fault", notes=stress_note)
    m.derive("dilation_tendency_Td", ["gem_active_faults"], f"Td * (0.35+0.65*exp(-d_gem/{FAULT_DECAY_M:.0f} m)); NaN beyond {FAULT_MAX_DIST_M:.0f} m of a mapped GEM fault", notes=stress_note)
    for n, a in (("dist_active_faults_m", eff_dist_m), ("active_fault_density", density), ("slip_tendency_Ts", ts), ("dilation_tendency_Td", td)):
        S.put(n, a)

    pmask, _ = _rasterize_lines(plates, g)
    if pmask.any():
        m.derive("dist_plate_boundary_km", ["gem_active_faults"], "distance to PB2002 plate boundary lines")
        S.put("dist_plate_boundary_km", (distance_transform_edt(~pmask) * res / 1000.0).astype(np.float32))

    volc = json.loads((ROOT / "data" / "indonesia" / "indonesia_volcanic_centers.json").read_text(encoding="utf-8"))
    lons, lats = _lonlat(g)
    lon2, lat2 = np.meshgrid(lons.astype(np.float32), lats.astype(np.float32))
    dmin = np.full(g.shape, np.inf, dtype=np.float32)
    for v in volc:
        dk = np.hypot((lon2 - float(v["lon"])) * 111.32 * np.cos(np.radians(lat2)), (lat2 - float(v["lat"])) * 110.57)
        np.minimum(dmin, dk, out=dmin)
    m.add(LayerProvenance(name="gvp_volcanoes", kind=MEASURED, source="Smithsonian Global Volcanism Program Holocene volcanoes",
                          url="https://volcano.si.edu", method="data/indonesia/indonesia_volcanic_centers.json"))
    m.derive("dist_volcanic_arc_km", ["gvp_volcanoes"], "great-circle-approximate distance to nearest Holocene volcano")
    S.put("dist_volcanic_arc_km", dmin)
    # No volcanic_arc_halo: the nearest Holocene volcano is >= 245 km from every land cell, so a
    # 35 km halo is identically zero over Kalimantan and carries no information.


# ---------------------------------------------------------------------------
# Potential fields, sediment, cover
# ---------------------------------------------------------------------------
def build_potential_fields(S: LayerStore, dem: np.ndarray) -> None:
    from src.features.coal_and_multimodal_features import compute_potential_field_derivatives
    from src.features.geophysics import analytic_signal, load_and_compute_real_gravity_channels, reduced_to_pole
    from src.features.real_sources.global_grids import build_emag2_layers

    g, m = S.grid, S.m
    bg, crust, moho = load_and_compute_real_gravity_channels(dem=dem, dst_crs=g.crs, dst_transform=g.transform, res_m=g.res_m)
    m.add(LayerProvenance(name="bouguer_gravity_mgal", kind=MEASURED, source="Sandwell & Smith satellite free-air gravity with Bouguer slab correction",
                          method="src.features.geophysics.load_and_compute_real_gravity_channels", native_resolution_m=1850.0))
    m.derive("crustal_thickness_km", ["bouguer_gravity_mgal"], "Parker-Oldenburg Moho inversion")
    m.derive("moho_gravity_step_mgal_km", ["bouguer_gravity_mgal"], "horizontal gradient of the regional Bouguer field")
    m.derive("gravity_residual_30km_mgal", ["bouguer_gravity_mgal"], "Bouguer minus 30 km NaN-aware Gaussian low-pass")
    S.put("bouguer_gravity_mgal", bg)
    S.put("crustal_thickness_km", crust)
    S.put("moho_gravity_step_mgal_km", moho)
    S.put("gravity_residual_30km_mgal", (bg - nan_gaussian(bg, 30000.0 / g.res_m)).astype(np.float32))

    # CRUST1.0 sediment thickness (1 degree)
    sed_xyz = ROOT / "data" / "raw" / "crust1.0" / "sedthk.xyz"
    if not sed_xyz.exists():
        raise RealDataUnavailableError(f"CRUST1.0 sediment file missing: {sed_xyz}")
    b = g.bounds_4326()
    pts = []
    for line in sed_xyz.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) >= 3:
            lo, la, t = float(p[0]), float(p[1]), float(p[2])
            if b[0] - 2 <= lo <= b[2] + 2 and b[1] - 2 <= la <= b[3] + 2:
                pts.append((lo, la, t))
    pts = np.array(pts)
    sub_h, sub_w = g.height // 10, g.width // 10
    gx, gy = np.meshgrid(np.linspace(b[0], b[2], sub_w), np.linspace(b[3], b[1], sub_h))
    coarse = griddata(pts[:, :2], pts[:, 2], (gx, gy), method="linear")
    sed_km = zoom(coarse, (g.height / sub_h, g.width / sub_w), order=1).astype(np.float32)[: g.height, : g.width]
    m.add(LayerProvenance(name="sediment_thickness_m", kind=MEASURED, source="CRUST1.0 sediment thickness (Laske et al. 2013)",
                          url="https://igppweb.ucsd.edu/~gabi/crust1.html", method="1-degree nodes, linear interpolation, km -> m", native_resolution_m=111000.0))
    sed_m = (sed_km * 1000.0).astype(np.float32)
    S.put("sediment_thickness_m", sed_m)

    emag = build_emag2_layers(g, m, RAW_REAL / "emag2")
    tmi = emag["emag2_tmi_nt"]
    for k in ("emag2_tmi_nt", "emag2_error_nt", "emag2_source_code"):
        if k in emag:
            S.put(k, emag[k])
    ok = np.isfinite(tmi)
    tmi0 = np.where(ok, tmi - np.nanmean(tmi), 0.0)  # FFT needs a full array; masked back afterwards
    rtp = reduced_to_pole(tmi=tmi0, inclination_deg=-14.0, declination_deg=1.5, dx=g.res_m, dy=g.res_m, min_inc_deg=20.0).astype(np.float32)
    asig = analytic_signal(tmi=tmi0, dx=g.res_m, dy=g.res_m).astype(np.float32)
    pot = compute_potential_field_derivatives(bouguer_mgal=bg, sed_thickness_m=sed_m, tmi=tmi0, rtp=rtp, analytic_sig=asig, res_m=g.res_m)
    m.derive("emag2_rtp_magnetics", ["emag2_tmi_nt"], "MacLeod (1993) reduced-to-pole, I=-14, D=1.5, |I|>=20 stabilisation")
    m.derive("emag2_analytic_signal", ["emag2_tmi_nt"], "3-D analytic signal amplitude")
    m.derive("mag_rtp_tilt_derivative", ["emag2_rtp_magnetics"], "Miller & Singh (1994) tilt derivative of RTP")
    m.derive("gravity_residual_basin_mgal", ["bouguer_gravity_mgal", "sediment_thickness_m"],
             "Bouguer high-pass + (-3.5 mGal per km of CRUST1.0 sediment), clipped +-45")
    S.put("emag2_rtp_magnetics", np.where(ok, rtp, np.nan))
    S.put("emag2_analytic_signal", np.where(ok, asig, np.nan))
    S.put("mag_rtp_tilt_derivative", np.where(ok, pot["mag_rtp_tilt_derivative"], np.nan))
    S.put("gravity_residual_basin_mgal", pot["gravity_residual_basin_mgal"])

    from src.features.real_sources.global_grids import build_depth_to_bedrock

    S.put("depth_to_bedrock_m", build_depth_to_bedrock(g, m, RAW_REAL / "soilgrids")["depth_to_bedrock_m"])


POTENTIAL_FIELD_LAYERS = (
    "bouguer_gravity_mgal", "crustal_thickness_km", "moho_gravity_step_mgal_km", "gravity_residual_30km_mgal",
    "sediment_thickness_m", "emag2_tmi_nt", "emag2_error_nt", "emag2_source_code", "emag2_rtp_magnetics",
    "emag2_analytic_signal", "mag_rtp_tilt_derivative", "gravity_residual_basin_mgal", "depth_to_bedrock_m",
)


def reuse_potential_fields(S: LayerStore, previous: ProvenanceManifest) -> None:
    """Re-register the potential-field layers of a previous build of this script (skips ~40 min of FFTs).

    Only layers written by this script with a provenance record are reused; anything missing raises.
    """
    for name in POTENTIAL_FIELD_LAYERS:
        p = S.out_dir / f"{name}.tif"
        if name not in previous.layers or not p.exists():
            if name in ("emag2_error_nt", "emag2_source_code"):
                continue  # optional quality layers
            raise RealDataUnavailableError(f"--reuse-potential-fields: {name} missing from the previous build; run without it")
        S.m.add(previous.layers[name])
        arr = load_layer(p)
        if name == "emag2_error_nt":
            arr[arr < 0] = np.nan  # fill codes (see global_grids.build_emag2_layers)
        S.put(name, arr)


# ---------------------------------------------------------------------------
# Optical, thermal, SAR, land cover, transport
# ---------------------------------------------------------------------------
def build_remote_sensing(S: LayerStore, dem: np.ndarray) -> dict[str, np.ndarray]:
    from src.features.real_sources.aster_tir import build_aster_tir_layers
    from src.features.real_sources.global_grids import build_hansen_gfc_layers
    from src.features.real_sources.landsat_thermal import build_landsat_thermal_layers
    from src.features.real_sources.osm_transport import build_osm_transport_layers
    from src.features.real_sources.sentinel2 import build_sentinel2_layers

    g, m = S.grid, S.m
    keep: dict[str, np.ndarray] = {}
    s2 = build_sentinel2_layers(g, m, CACHE / "sentinel2")
    for k in ("spec_ndvi", "spec_ndre", "s2_clear_obs"):
        S.put(k, s2[k])
    keep.update({k: s2[k] for k in ("spec_ndvi", "spec_ndre", "spec_albedo", "spec_ficr", "spec_clay_hydroxyl", "spec_ferrous_ultramafic")})

    th = build_landsat_thermal_layers(g, m, CACHE / "landsat", dem=dem)
    S.put("thermal_lst_c", th["thermal_lst_c"])
    S.put("thermal_anomaly_st", th["thermal_anomaly_st"])
    keep["thermal_anomaly_st"] = th["thermal_anomaly_st"]

    aster = build_aster_tir_layers(g, m, CACHE / "aster")
    keep.update(aster)

    sar_dir = CACHE / "palsar2"
    sar_prov = json.loads((sar_dir / "sar_provenance.json").read_text(encoding="utf-8"))
    for k in ("sar_palsar2_hh", "sar_palsar2_hv", "sar_palsar2_dpr", "sar_palsar2_lineament_density"):
        m.add(LayerProvenance(name=k, kind=MEASURED, source=sar_prov.get("sensor_source", "JAXA ALOS-2 PALSAR-2 annual 25 m mosaic"),
                              url="https://planetarycomputer.microsoft.com/dataset/alos-palsar-mosaic",
                              method="src.features.sar_palsar2.build_real_sar_channels (gamma0 dB, area-averaged)",
                              native_resolution_m=25.0, temporal_coverage=str(sar_prov.get("year", "")),
                              item_ids=list(sar_prov.get("item_ids", []))))
        a = load_layer(sar_dir / f"{k}.tif")
        S.put(k, a)
        keep[k] = a

    gfc = build_hansen_gfc_layers(g, m, RAW_REAL / "hansen_gfc") if not (CACHE / "hansen" / "gfw_forest_disturbance.tif").exists() else None
    if gfc is None:
        gfc = {k: _cache("hansen", k) for k in ("gfw_forest_disturbance", "tree_cover_2000", "forest_disturbance_rate")}
        from src.features.real_sources.global_grids import GFC_BASE, GFC_VERSION

        for k in ("gfw_forest_disturbance", "tree_cover_2000"):
            m.add(LayerProvenance(name=k, kind=MEASURED, source=f"Hansen et al. (2013) Global Forest Change {GFC_VERSION}", url=GFC_BASE,
                                  native_resolution_m=30.0, temporal_coverage="2000-2023", method="block-averaged 30 m pixels"))
        m.derive("forest_disturbance_rate", ["gfw_forest_disturbance", "tree_cover_2000"], "loss fraction / year-2000 tree-cover fraction")
    for k, v in gfc.items():
        S.put(k, v)
    keep["gfw"] = gfc["gfw_forest_disturbance"]

    osm = build_osm_transport_layers(g, m, RAW_REAL / "osm")
    for k, v in osm.items():
        S.put(k, v)
    keep["dist_major_rivers_km"] = osm["dist_major_rivers_km"]
    keep["osm_quarry_mask"] = osm["osm_quarry_mask"]

    # v4.15: surface water, removed from every gated optical layer (audit v4.13 section 2.6)
    from src.features.real_sources.water import compute_mndwi, compute_water_mask

    mndwi = compute_mndwi(s2["s2_b03"], s2["s2_b11"])
    water = compute_water_mask(s2["spec_ndvi"], mndwi=mndwi, dem=dem, osm_water=osm["osm_water_mask"])
    m.derive("water_mask", ["spec_ndvi", "s2_b03", "s2_b11", "osm_water_mask", "dem_elevation"],
             "NDVI<0 | MNDWI>0 (B03,B11) | OSM water polygons | DEM<0; 1 = water. Context mask, not a feature")
    S.put("water_mask", water.astype(np.float32))
    keep["water_mask"] = water
    return keep


# ---------------------------------------------------------------------------
# Geology, hydrology, gated indices
# ---------------------------------------------------------------------------
def build_geology_and_gated(S: LayerStore, terr: dict, rs: dict) -> dict[str, np.ndarray]:
    from src.features.coal_and_multimodal_features import (
        compute_continuous_canopy_weight,
        compute_hydro_twi_and_censored_upslope_amd,
        compute_lithologic_and_stratigraphic_contacts,
        compute_metallic_and_coal_gated_optical,
        compute_sar_bedding_strike_and_anisotropy,
    )
    from src.features.sar_palsar2 import _fill_nearest

    g, m = S.grid, S.m
    lons, lats = _lonlat(g)

    # Coal-bearing formations retrieved earlier from the ESDM geoportal (REST service now intermittently 503)
    esdm_src = ROOT / "data" / "kalimantan" / "interim" / "esdm_geology_250k.gpkg"
    esdm = RAW_REAL / "esdm" / "esdm_coal_bearing_formations.gpkg"
    if not esdm.exists():
        import geopandas as gpd

        gdf = gpd.read_file(esdm_src)
        if len(gdf) < 10:
            raise RealDataUnavailableError(f"{esdm_src} has {len(gdf)} polygons; looks like the old 3-circle fallback, refusing")
        esdm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(esdm_src, esdm)
    m.add(LayerProvenance(name="esdm_coal_formations", kind=MEASURED,
                          source="ESDM Badan Geologi geoportal, Formasi Pembawa Batubara (Warukin, Tanjung, Balikpapan) polygons",
                          url="https://geoportal.esdm.go.id/gis4/rest/services/BGD_TU/Formasi_Pembawa_Batubara/MapServer/0",
                          method=str(esdm.relative_to(ROOT)), notes="Coal-bearing formations only; not a full geological map."))
    m.add(LayerProvenance(name="georima_rock_nodes", kind=MEASURED, source="ESDM GeoRIMA lithology observation nodes (deposit/commodity records excluded)",
                          method="data/indonesia/georima_all_indonesia_nodes.csv",
                          notes="Point observations cluster where geologists have worked; distance layers inherit that sampling bias."))
    litho = compute_lithologic_and_stratigraphic_contacts(
        geology_gpkg_path=esdm, grid_lon=lons, grid_lat=lats, dem=terr["dem"], slope=terr["slope"],
        coastal_dist_km=terr["coast_km"], georima_csv_path=ROOT / "data" / "indonesia" / "georima_all_indonesia_nodes.csv", res_m=g.res_m)
    litho_inputs = {
        "formation_polygon_id": ["esdm_coal_formations"],
        "dist_sedimentary_polygon_km": ["esdm_coal_formations"],
        "in_coal_bearing_formation": ["esdm_coal_formations"],
        "dist_marine_deltaic_contact_km": ["esdm_coal_formations", "georima_rock_nodes"],
        "dist_calc_alkaline_intrusive_km": ["georima_rock_nodes"],
        "dist_pluton_carbonate_contact_km": ["georima_rock_nodes"],
        "in_ophiolite_unit": ["georima_rock_nodes"],
        "in_holocene_peat_or_karst_mask": ["esdm_coal_formations", "georima_rock_nodes", "dem_elevation", "coastal_shelf_dist_km"],
    }
    for k, v in litho.items():
        if k in litho_inputs:
            m.derive(k, litho_inputs[k], "src.features.coal_and_multimodal_features.compute_lithologic_and_stratigraphic_contacts")
            S.put(k, v)

    ndvi = rs["spec_ndvi"]
    w_can = compute_continuous_canopy_weight(ndvi=ndvi)
    m.derive("w_canopy", ["spec_ndvi"], "clip((0.30-NDVI)/0.20, 0, 1); NaN NDVI -> 0")
    S.put("w_canopy", w_can)
    gfw = rs["gfw"]
    hh = rs["sar_palsar2_hh"]
    # v4.15 (audit v4.13 section 2.7): the old mask was forest loss (13.2% of EK land, mostly plantation, logging and
    # fire) plus an HH > -2.5 dB clause that selects no bare ground. It is now the mapped open-pit workings only.
    mine = np.asarray(rs["osm_quarry_mask"], dtype=np.float32)
    m.derive("active_mine_mask", ["osm_quarry_mask"], "OSM landuse=quarry polygons (mapped open-pit workings); 1 = quarry. Forest loss and SAR are not used")
    S.put("active_mine_mask", mine)

    hv = rs["sar_palsar2_hv"]
    sar_bad = ~(np.isfinite(hh) & np.isfinite(hv))
    sar_mm = compute_sar_bedding_strike_and_anisotropy(hh_db=_fill_nearest(hh), hv_db=_fill_nearest(hv), dem=terr["dem"], res_m=g.res_m)
    for k in ("sar_bedding_strike_deg", "sar_glcm_anisotropy"):
        m.derive(k, ["sar_palsar2_hh", "sar_palsar2_hv", "dem_elevation"], "src.features.coal_and_multimodal_features.compute_sar_bedding_strike_and_anisotropy (NaN where SAR invalid)")
        S.put(k, np.where(sar_bad, np.nan, sar_mm[k]))

    hydro = compute_hydro_twi_and_censored_upslope_amd(dem=terr["dem"], slope=terr["slope"], ndre=rs["spec_ndre"], ficr=rs["spec_ficr"], active_mine_mask=mine, res_m=g.res_m)
    m.derive("dem_twi", ["dem_elevation"], "ln(a / tan(beta)) with D8 flow accumulation")
    m.derive("amd_upslope_anomaly_censored", ["spec_ficr", "spec_ndre", "dem_elevation", "active_mine_mask"], "ferric/red-edge drainage anomaly, mine-catchment censored, NaN where S2 missing")
    S.put("dem_twi", hydro["dem_twi"])
    S.put("amd_upslope_anomaly_censored", hydro["amd_upslope_anomaly_censored"])

    river_bin = (rs["dist_major_rivers_km"] <= 0.0).astype(np.uint8)
    gated = compute_metallic_and_coal_gated_optical(
        s2_stack={"ndvi": ndvi, "albedo": rs["spec_albedo"], "ficr": rs["spec_ficr"], "clay_hydroxyl": rs["spec_clay_hydroxyl"],
                  "ferrous_ultramafic": rs["spec_ferrous_ultramafic"]},
        aster_tir_silica=rs["aster_silica_index"], thermal_st=rs["thermal_anomaly_st"], w_canopy=w_can,
        river_mask=river_bin, disturbance_mask=gfw, in_coal_bearing_formation=litho["in_coal_bearing_formation"], res_m=g.res_m,
        water_mask=rs["water_mask"])
    gate_src = {"spec_albedo_gated": "spec_albedo", "spec_ficr_gated": "spec_ficr", "spec_clay_hydroxyl_gated": "spec_clay_hydroxyl",
                "spec_ferrous_ultramafic_gated": "spec_ferrous_ultramafic", "aster_tir_silica_gated": "aster_silica_index"}
    for k, src_name in gate_src.items():
        if k in gated:
            m.derive(k, [src_name, "w_canopy", "water_mask"], "NaN where canopy is opaque (w_canopy=0 or NDVI>=0.20) or the cell is surface water")
            S.put(k, gated[k])
    if "thermal_anomaly_st_filtered" in gated:
        m.derive("thermal_anomaly_st_filtered", ["thermal_anomaly_st", "gfw_forest_disturbance", "spec_ficr", "in_coal_bearing_formation"],
                 "thermal anomaly censored (NaN) on recent clearing unless ferric clinker or mapped coal formation")
        S.put("thermal_anomaly_st_filtered", gated["thermal_anomaly_st_filtered"])
    qi_g = np.where((w_can <= 0.0) | (ndvi >= 0.20) | ~np.isfinite(ndvi) | rs["water_mask"], np.nan, rs["aster_quartz_index"]).astype(np.float32)
    m.derive("aster_quartz_index_gated", ["aster_quartz_index", "w_canopy", "water_mask"], "NaN where canopy is opaque or the cell is surface water")
    S.put("aster_quartz_index_gated", qi_g)
    return {"formation_polygon_id": litho["formation_polygon_id"], "active_mine_mask": mine, "sar_bedding": S.get("sar_bedding_strike_deg")}


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def build_labels(S: LayerStore, rs: dict, geo: dict) -> list[str]:
    import geopandas as gpd
    from shapely.geometry import Point

    from src.labels.borneo_labels import (
        build_k3_coal_rank_targets,
        build_k4_deposit_targets,
        build_pooled_exploration_footprint,
        load_kalimantan_coal_sources,
        rasterize_borneo_k4_labels,
    )

    g = S.grid
    prof = g.profile()
    tg = build_k4_deposit_targets()
    for key in ("positives_k0", "positives_k1", "positives_k2", "positives_k3"):
        gdf = tg[key]
        if len(gdf):
            bad = np.hypot(gdf.geometry.x - CENTROID_ARTIFACT[0], gdf.geometry.y - CENTROID_ARTIFACT[1]) < 0.08
            tg[key] = gdf[~bad].copy()
    k4 = rasterize_borneo_k4_labels(tg, prof, buffer_m=1000.0)
    labels: dict[str, np.ndarray] = {k: np.asarray(k4[k]) for k in ("label_k0_porphyry_cu", "label_k1_skarn_fe_cu", "label_k2_epithermal_au", "label_k3_laterite_fe_ni")}

    coal_csv = ROOT / "data" / "borneo" / "kalimantan_coal_deposits.csv"
    coal = np.full(g.shape, -1, dtype=np.int8)
    if coal_csv.exists():
        df = pd.read_csv(coal_csv)
        df = df[np.hypot(df["longitude"] - CENTROID_ARTIFACT[0], df["latitude"] - CENTROID_ARTIFACT[1]) >= 0.08]
        shapes = [(Point(float(r.longitude), float(r.latitude)).buffer(1000.0 / 111320.0), 1) for r in df.itertuples()]
        if shapes:
            rasterio.features.rasterize(shapes, out=coal, transform=g.transform, all_touched=True, dtype=np.int8)

    coal_gdf = load_kalimantan_coal_sources()
    ct = build_k3_coal_rank_targets(
        concessions=coal_gdf, ndvi=rs["spec_ndvi"], active_mine_mask=geo["active_mine_mask"], formation_polygon_id=geo["formation_polygon_id"],
        sar_bedding_strike_deg=geo["sar_bedding"], gfw_forest_disturbance=rs["gfw"], sar_hh_db=rs["sar_palsar2_hh"], bounds=g.bounds_4326())
    for k in ("label_coal_k0_high_cv", "label_coal_k1_med_cv", "label_coal_k2_low_cv"):
        labels[k] = np.asarray(ct[k])
    halo_any = (labels["label_coal_k0_high_cv"] == 1) | (labels["label_coal_k1_med_cv"] == 1) | (labels["label_coal_k2_low_cv"] == 1)
    labels["label_coal"] = np.where((coal == 1) | halo_any, 1, -1).astype(np.int8)
    labels["explored_footprint"] = build_pooled_exploration_footprint(
        metallic_explored_footprint=np.asarray(k4["explored_footprint"], dtype=np.float32), coal_concessions=coal_gdf,
        active_mine_mask=geo["active_mine_mask"], gfw_forest_disturbance=rs["gfw"], bounds=g.bounds_4326())
    labels["alteration_benchmark"] = np.asarray(k4["alteration_benchmark"])
    labels["coal_tonnage_weight"] = np.asarray(ct["coal_tonnage_weights"], dtype=np.float32)
    labels["coal_parent_concession_id"] = np.asarray(ct["coal_parent_concession_id"], dtype=np.float32)

    m = S.m
    m.add(LayerProvenance(name="deposit_catalogue", kind=MEASURED, source="Kalimantan deposit catalogue (src.labels.borneo_labels.build_k4_deposit_targets) and ESDM coal sources",
                          notes="Labels, not features. USGS MRDS island-centroid artefact purged."))
    for k, a in labels.items():
        m.derive(k, ["deposit_catalogue"], "label rasterisation (1 km buffer) / K3 coal rank halo targets")
        S.put(k, a.astype(np.float32))
    any_metal = np.zeros(g.shape, bool)
    for k in ("label_k0_porphyry_cu", "label_k1_skarn_fe_cu", "label_k2_epithermal_au", "label_k3_laterite_fe_ni"):
        any_metal |= labels[k] == 1
    np.save(S.out_dir / "labels.npy", np.where(any_metal, 1, -1).astype(np.int32))
    return list(labels)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def valid_land_mask(g: GridSpec, dem: np.ndarray, coast_km: np.ndarray) -> np.ndarray:
    lons, lats = _lonlat(g)
    lon2, lat2 = np.meshgrid(lons, lats)
    non_id = ((lat2 > 4.35) & (lon2 > 113.0) & (lon2 < 119.2)) | ((lat2 > 1.45) & (lon2 > 109.6) & (lon2 < 115.2)) | ((lat2 > 4.10) & (lon2 > 117.5))
    return np.isfinite(dem) & (dem >= 0.0) & (coast_km <= 150.0) & ~non_id


def write_parquet(out_dir: Path, names: list[str], label_names: list[str], valid: np.ndarray, g: GridSpec,
                  window: tuple[int, int, int, int] | None = None) -> int:
    """Assemble dataset.parquet column by column from the GeoTIFFs in ``out_dir``."""
    rows, cols = np.where(valid)
    lons, lats = _lonlat(g)
    cols_out = {"row": rows.astype(np.int32), "col": cols.astype(np.int32), "lon": lons[cols].astype(np.float32), "lat": lats[rows].astype(np.float32)}
    for n in names:
        a = load_layer(out_dir / f"{n}.tif")
        v = a[rows, cols]
        if n.startswith("label_") or n in ("active_mine_mask", "formation_polygon_id", "in_ophiolite_unit", "in_holocene_peat_or_karst_mask",
                                            "explored_footprint", "alteration_benchmark", "in_coal_bearing_formation", "coal_parent_concession_id"):
            v = np.nan_to_num(v, nan=-1 if n.startswith("label_") else 0).astype(np.int32)
        else:
            v = v.astype(np.float32)
        cols_out[n] = v
    lab = np.load(out_dir / "labels.npy")
    cols_out["label"] = lab[rows, cols].astype(np.int32)
    table = pa.table(cols_out)
    pq.write_table(table, out_dir / "dataset.parquet", compression="zstd")
    return len(rows)


def crop_to_ek(master_dir: Path, ek_dir: Path, names: list[str], label_names: list[str], manifest: ProvenanceManifest) -> int:
    r0, c0, h, w = EK_WINDOW
    tr = MASTER_GRID.transform * Affine.translation(c0, r0)
    ek_grid = GridSpec(crs=MASTER_GRID.crs, transform=tr, height=h, width=w)
    assert abs(tr.c - 114.5) < 1e-9 and abs(tr.f - 3.2) < 1e-9, tr
    ek_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        a = load_layer(master_dir / f"{n}.tif")[r0:r0 + h, c0:c0 + w]
        save_layer(a, ek_grid, ek_dir / f"{n}.tif")
    lab = np.load(master_dir / "labels.npy")[r0:r0 + h, c0:c0 + w]
    np.save(ek_dir / "labels.npy", lab)
    dem = load_layer(ek_dir / "dem_elevation.tif")
    coast = load_layer(ek_dir / "coastal_shelf_dist_km.tif")
    valid = valid_land_mask(ek_grid, dem, coast)
    n = write_parquet(ek_dir, names, label_names, valid, ek_grid)
    ek_m = ProvenanceManifest(ek_grid)
    ek_m.layers = dict(manifest.layers)
    for name in names:
        ek_m.update_valid_fraction(name, load_layer(ek_dir / f"{name}.tif"), valid)
    ek_m.write(ek_dir)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ek", action="store_true")
    ap.add_argument("--reuse-potential-fields", action="store_true",
                    help="reuse gravity/magnetics/sediment/bedrock layers from the previous build in the output directory")
    args = ap.parse_args()
    t0 = time.time()
    g = MASTER_GRID
    OUT_MASTER.mkdir(parents=True, exist_ok=True)
    previous = ProvenanceManifest.read(OUT_MASTER) if args.reuse_potential_fields else None
    if args.reuse_potential_fields and previous is None:
        raise RealDataUnavailableError(f"--reuse-potential-fields: no {ProvenanceManifest.FILENAME} in {OUT_MASTER}")
    m = ProvenanceManifest(g)
    S = LayerStore(g, OUT_MASTER, m)

    terr = build_terrain(S)
    logger.info("terrain done (%.1f min)", (time.time() - t0) / 60)
    build_structure(S, terr)
    logger.info("structure done (%.1f min)", (time.time() - t0) / 60)
    if previous is not None:
        reuse_potential_fields(S, previous)
    else:
        build_potential_fields(S, terr["dem"])
    logger.info("potential fields done (%.1f min)", (time.time() - t0) / 60)
    rs = build_remote_sensing(S, terr["dem"])
    logger.info("remote sensing done (%.1f min)", (time.time() - t0) / 60)
    geo = build_geology_and_gated(S, terr, rs)
    logger.info("geology/gated done (%.1f min)", (time.time() - t0) / 60)
    label_names = build_labels(S, rs, geo)
    del rs, geo

    valid = valid_land_mask(g, terr["dem"], terr["coast_km"])
    names = list(S.names)
    n_rows = write_parquet(OUT_MASTER, names, label_names, valid, g)
    for name in names:
        m.update_valid_fraction(name, load_layer(OUT_MASTER / f"{name}.tif"), valid)
    m.write(OUT_MASTER)
    logger.info("Master cube: %d valid cells, %d layers -> %s", n_rows, len(names), OUT_MASTER)

    if not args.skip_ek:
        n_ek = crop_to_ek(OUT_MASTER, OUT_EK, names, label_names, m)
        logger.info("East Kalimantan cube: %d valid cells -> %s", n_ek, OUT_EK)

    feats = [n for n in names if n not in META_COLS and not n.startswith("label_")]
    summary = {"layers": names, "feature_candidates": feats, "master_rows": n_rows, "minutes": round((time.time() - t0) / 60, 1)}
    (OUT_MASTER / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
