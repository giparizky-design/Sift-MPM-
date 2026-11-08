"""Polygonal (non-rectangular) region rasterization and multi-resolution feature cube builder.

Strictly masks all feature channels, dataset.parquet rows, and output prospectivity maps
to the exact interior of a user-selected or predefined polygon using
`rasterio.features.geometry_mask(..., invert=True)` at configurable physical resolutions
(50m, 100m, 250m, 500m).
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_bounds
from rasterio.windows import from_bounds as win_from_bounds
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union

from src.features.feature_stack import FeatureStack
from src.features.structural import plan_curvature
from src.labels.borneo_labels import (
    K4_LABEL_COLS,
    build_k3_coal_rank_targets,
    build_k4_deposit_targets,
    load_kalimantan_coal_sources,
    rasterize_borneo_k4_labels,
)
from src.schema.canonical import (
    EXTENDED_NON_FEATURE_COLS,
    NON_FEATURE_COLS,
    _FORBIDDEN_RAW_BAND_PATTERNS,
    verify_no_raw_pixel_leakage,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

VALID_RESOLUTIONS_M = (50, 100, 250, 500)

# Maximum recommended polygon area (km^2) per resolution to prevent OOM
MAX_AREA_KM2_BY_RES: dict[int, float] = {
    50: 5_000.0,
    100: 25_000.0,
    250: 250_000.0,
    500: 650_000.0,
}

CATEGORICAL_OR_INT_LAYERS = {
    "active_mine_mask",
    "formation_polygon_id",
    "in_coal_bearing_formation",
    "in_holocene_peat_or_karst_mask",
    "in_ophiolite_unit",
    "in_carbonate_host",
    "in_ultramafic_belt",
    "label",
    "label_coal",
    "label_coal_k0_high_cv",
    "label_coal_k1_med_cv",
    "label_coal_k2_low_cv",
    "label_k0_porphyry_cu",
    "label_k1_skarn_fe_cu",
    "label_k2_epithermal_au",
    "label_k3_laterite_fe_ni",
    "coal_parent_concession_id",
    "explored_footprint",
    "alteration_benchmark",
}


def compute_polygon_area_km2(geom: Union[Polygon, MultiPolygon, dict, str, Path]) -> float:
    """Compute approximate geodesic area in km^2 for a WGS84 Polygon/MultiPolygon or GeoJSON."""
    if isinstance(geom, (Polygon, MultiPolygon)):
        shp = geom
    elif isinstance(geom, dict) and geom.get("type") in ("Polygon", "MultiPolygon"):
        shp = shape(geom)
    else:
        geoms = extract_shapely_geometries(geom)
        shp = unary_union(geoms)
    if shp.is_empty:
        return 0.0
    minx, miny, maxx, maxy = shp.bounds
    mid_lat_rad = math.radians(0.5 * (miny + maxy))
    deg_lat_km = 110.574
    deg_lon_km = 111.320 * max(0.1, math.cos(mid_lat_rad))
    return float(shp.area * deg_lat_km * deg_lon_km)


def extract_shapely_geometries(geojson_obj: Union[dict, str, Path]) -> list[Union[Polygon, MultiPolygon]]:
    """Extract Shapely Polygon/MultiPolygon objects from a GeoJSON dict, string, or file path."""
    if isinstance(geojson_obj, Path) or (isinstance(geojson_obj, str) and not geojson_obj.lstrip().startswith("{")):
        p = Path(geojson_obj)
        data = json.loads(p.read_text(encoding="utf-8"))
    elif isinstance(geojson_obj, str):
        data = json.loads(geojson_obj)
    elif isinstance(geojson_obj, dict):
        data = geojson_obj
    else:
        raise ValueError(f"Unsupported polygon input type: {type(geojson_obj)}")

    gtype = data.get("type")
    geoms: list[Union[Polygon, MultiPolygon]] = []
    if gtype == "FeatureCollection":
        for feat in data.get("features", []):
            geom_dict = feat.get("geometry")
            if geom_dict and geom_dict.get("type") in ("Polygon", "MultiPolygon"):
                s = shape(geom_dict)
                if not s.is_empty:
                    geoms.append(s if s.is_valid else s.buffer(0))
    elif gtype == "Feature":
        geom_dict = data.get("geometry")
        if geom_dict and geom_dict.get("type") in ("Polygon", "MultiPolygon"):
            s = shape(geom_dict)
            if not s.is_empty:
                geoms.append(s if s.is_valid else s.buffer(0))
    elif gtype in ("Polygon", "MultiPolygon"):
        s = shape(data)
        if not s.is_empty:
            geoms.append(s if s.is_valid else s.buffer(0))
    elif "coordinates" in data:
        s = shape({"type": "Polygon", "coordinates": data["coordinates"]})
        if not s.is_empty:
            geoms.append(s if s.is_valid else s.buffer(0))

    if not geoms:
        raise ValueError("No valid Polygon or MultiPolygon geometry found in input GeoJSON.")
    return geoms


def normalize_polygon_geojson(
    geojson_obj: Union[dict, str, Path],
    name: str = "Custom Exploration Polygon",
    region_id: str = "CUSTOM_POLYGON",
    resolution_m: int = 250,
) -> dict[str, Any]:
    """Normalize any Polygon/Feature/FeatureCollection into a canonical FeatureCollection with metadata."""
    geoms = extract_shapely_geometries(geojson_obj)
    merged = unary_union(geoms)
    if not merged.is_valid:
        merged = merged.buffer(0)
    area_km2 = compute_polygon_area_km2(merged)
    minx, miny, maxx, maxy = merged.bounds
    c = merged.centroid

    return {
        "type": "FeatureCollection",
        "name": name,
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "region_id": region_id,
                    "name": name,
                    "boundary_type": "polygonal_domain",
                    "resolution_m": int(resolution_m),
                    "area_sq_km": round(area_km2, 2),
                    "bbox": [round(minx, 5), round(miny, 5), round(maxx, 5), round(maxy, 5)],
                    "centroid_lat_lon": [round(float(c.y), 5), round(float(c.x), 5)],
                },
                "geometry": mapping(merged),
            }
        ],
    }


def get_predefined_polygons() -> dict[str, dict[str, Any]]:
    """Return predefined non-rectangular geological/provincial polygons for Kalimantan."""
    polygons: dict[str, dict[str, Any]] = {}

    # 1. East Kalimantan Provincial Polygon (exact boundary from GeoJSON)
    ek_path = ROOT_DIR / "data" / "east_kalimantan" / "east_kalimantan_boundary.geojson"
    if ek_path.exists():
        try:
            ek_fc = normalize_polygon_geojson(
                ek_path,
                name="East Kalimantan Province (Kalimantan Timur)",
                region_id="EAST_KALIMANTAN",
                resolution_m=250,
            )
            ek_props = ek_fc["features"][0]["properties"]
            ek_props["recommended_resolutions_m"] = [250, 500]
            ek_props["domain"] = "both"
            ek_clean_fc = json.loads(json.dumps(ek_fc))
            ek_fc["id"] = "EAST_KALIMANTAN"
            ek_fc["title"] = "East Kalimantan Province (Kalimantan Timur)"
            ek_fc["default_resolution_m"] = 250
            ek_fc["area_km2"] = ek_props["area_sq_km"]
            ek_fc["domain"] = "both"
            ek_fc["is_predefined"] = True
            ek_fc["geojson"] = ek_clean_fc
            polygons["EAST_KALIMANTAN"] = ek_fc
        except Exception as exc:
            logger.warning("Could not load east_kalimantan_boundary.geojson: %s", exc)

    # 2. Predefined non-rectangular geological & structural polygons across East Kalimantan
    structural_defs = [
        {
            "region_id": "KUTAI_BASIN",
            "name": "Kutai Sedimentary Basin (Non-Rectangular Structural Outline)",
            "default_res": 100,
            "recommended_resolutions_m": [100, 250, 500],
            "domain": "both",
            "coords": [
                [115.45, -1.35],
                [116.15, -1.65],
                [116.85, -1.28],
                [117.42, -0.82],
                [117.62, -0.15],
                [117.65, 0.58],
                [117.52, 1.12],
                [116.95, 1.28],
                [116.22, 0.96],
                [115.65, 0.35],
                [115.32, -0.38],
                [115.45, -1.35],
            ],
        },
        {
            "region_id": "MAHAKAM_SUTURE",
            "name": "Mahakam Fold-Thrust & Anticlinorium Corridor",
            "default_res": 100,
            "recommended_resolutions_m": [50, 100, 250],
            "domain": "both",
            "coords": [
                [116.68, -0.85],
                [117.12, -0.92],
                [117.42, -0.62],
                [117.46, -0.22],
                [117.18, -0.08],
                [116.78, -0.24],
                [116.62, -0.55],
                [116.68, -0.85],
            ],
        },
        {
            "region_id": "SANGATTA_TREND",
            "name": "Sangatta–Bengalon High-CV Coal & Intrusive Dome",
            "default_res": 100,
            "recommended_resolutions_m": [50, 100, 250],
            "domain": "both",
            "coords": [
                [117.18, 0.36],
                [117.58, 0.32],
                [117.78, 0.62],
                [117.74, 1.02],
                [117.38, 1.08],
                [117.12, 0.78],
                [117.18, 0.36],
            ],
        },
        {
            "region_id": "KELIAN_CORRIDOR",
            "name": "Kelian–Muyup Volcanic Epithermal Au-Ag Corridor",
            "default_res": 50,
            "recommended_resolutions_m": [50, 100, 250],
            "domain": "metals",
            "coords": [
                [115.28, -0.22],
                [115.58, -0.26],
                [115.74, -0.04],
                [115.68, 0.22],
                [115.38, 0.25],
                [115.22, 0.04],
                [115.28, -0.22],
            ],
        },
        {
            "region_id": "BERAU_FOLD_BELT",
            "name": "Berau Sub-Basin & Carbonate Skarn Belt",
            "default_res": 100,
            "recommended_resolutions_m": [50, 100, 250],
            "domain": "both",
            "coords": [
                [117.08, 1.82],
                [117.58, 1.76],
                [117.85, 2.08],
                [117.76, 2.42],
                [117.28, 2.46],
                [117.02, 2.15],
                [117.08, 1.82],
            ],
        },
        {
            "region_id": "ADANG_FAULT",
            "name": "Adang–Paternoster Deep Cross-Basin Fault Zone",
            "default_res": 100,
            "recommended_resolutions_m": [50, 100, 250],
            "domain": "both",
            "coords": [
                [115.82, -1.78],
                [116.42, -1.85],
                [116.68, -1.48],
                [116.52, -1.12],
                [115.95, -1.08],
                [115.74, -1.42],
                [115.82, -1.78],
            ],
        },
        {
            "region_id": "MALINAU_CORRIDOR",
            "name": "Malinau–Mangkalihat Structural Corridor",
            "default_res": 100,
            "recommended_resolutions_m": [50, 100, 250],
            "domain": "both",
            "coords": [
                [116.22, 1.38],
                [116.72, 1.32],
                [116.92, 1.68],
                [116.78, 1.98],
                [116.32, 2.02],
                [116.14, 1.68],
                [116.22, 1.38],
            ],
        },
    ]

    for sdef in structural_defs:
        rid = str(sdef["region_id"])
        fc = normalize_polygon_geojson(
            {"type": "Polygon", "coordinates": [sdef["coords"]]},
            name=str(sdef["name"]),
            region_id=rid,
            resolution_m=int(sdef["default_res"]),
        )
        props = fc["features"][0]["properties"]
        props["recommended_resolutions_m"] = sdef["recommended_resolutions_m"]
        props["domain"] = sdef["domain"]
        clean_fc = json.loads(json.dumps(fc))
        fc["id"] = rid
        fc["title"] = str(sdef["name"])
        fc["default_resolution_m"] = int(sdef["default_res"])
        fc["area_km2"] = props["area_sq_km"]
        fc["domain"] = str(sdef["domain"])
        fc["is_predefined"] = True
        fc["geojson"] = clean_fc
        polygons[rid] = fc

    return polygons


def validate_polygon_and_resolution(
    geojson_obj: Union[dict, str, Path],
    resolution_m: Union[int, float],
    allow_auto_coarsen: bool = True,
) -> dict[str, Any]:
    """Validate requested resolution against polygon geodesic area (km^2).

    Returns:
        Dictionary with `valid`, `resolution_m`, `effective_resolution_m`,
        `recommended_resolution_m`, `area_km2`, `est_valid_cells`, `fill_ratio`, `message`, `warning`.
    """
    res = int(round(float(resolution_m)))
    if res not in VALID_RESOLUTIONS_M:
        res = min(VALID_RESOLUTIONS_M, key=lambda x: abs(x - res))

    geoms = extract_shapely_geometries(geojson_obj)
    merged = unary_union(geoms)
    area_km2 = compute_polygon_area_km2(merged)
    minx, miny, maxx, maxy = merged.bounds
    mid_lat_rad = math.radians(0.5 * (miny + maxy))
    bbox_area_km2 = max(
        1e-3,
        float((maxx - minx) * 111.320 * max(0.1, math.cos(mid_lat_rad)) * (maxy - miny) * 110.574),
    )
    fill_ratio = float(np.clip(area_km2 / bbox_area_km2, 0.01, 1.0))
    max_area = MAX_AREA_KM2_BY_RES[res]

    rec_res = res
    for candidate_res in sorted(VALID_RESOLUTIONS_M):
        if area_km2 <= MAX_AREA_KM2_BY_RES[candidate_res]:
            rec_res = candidate_res
            break
    else:
        rec_res = 500

    is_within_limit = area_km2 <= max_area
    if not is_within_limit and not allow_auto_coarsen:
        raise ValueError(
            f"Polygon area ({area_km2:,.1f} km²) exceeds maximum safe limit ({max_area:,.0f} km²) "
            f"for {res}m resolution. Recommended resolution: {rec_res}m."
        )

    eff_res = res if is_within_limit else rec_res
    cell_area_km2 = (float(eff_res) / 1000.0) ** 2
    est_valid_cells = max(1, int(round(area_km2 / cell_area_km2)))
    msg = (
        f"Validated {res}m resolution on {area_km2:,.1f} km² polygon (~{est_valid_cells:,} interior cells)."
        if is_within_limit
        else f"Auto-adjusted resolution from {res}m to {eff_res}m for {area_km2:,.1f} km² polygon."
    )

    return {
        "valid": bool(is_within_limit),
        "resolution_m": int(res),
        "effective_resolution_m": int(eff_res),
        "recommended_resolution_m": int(rec_res),
        "area_km2": round(float(area_km2), 2),
        "est_valid_cells": int(est_valid_cells),
        "fill_ratio": round(fill_ratio, 4),
        "message": msg,
        "warning": "" if is_within_limit else msg,
    }


def compute_polygon_mask_2d(
    geojson_obj: Union[dict, str, Path],
    height: int,
    width: int,
    transform: Affine,
    crs: str = "EPSG:4326",
) -> np.ndarray:
    """Return a 2D boolean mask (H, W) where True = inside polygon and False = outside."""
    _ = crs
    geoms = extract_shapely_geometries(geojson_obj)
    return geometry_mask(
        [mapping(g) for g in geoms],
        out_shape=(int(height), int(width)),
        transform=transform,
        invert=True,
        all_touched=True,
    )


def apply_polygon_mask_to_array(
    arr: np.ndarray,
    inside_mask: np.ndarray,
    outside_value: float = np.nan,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """Mask any 2D (H, W) or 3D (C, H, W) array outside `inside_mask` to `outside_value`."""
    eff_fill = outside_value if fill_value is None else fill_value
    if arr.ndim == 2:
        out = arr.copy()
        out[~inside_mask] = eff_fill
        return out
    if arr.ndim == 3:
        out = arr.copy()
        out[:, ~inside_mask] = eff_fill
        return out
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def build_polygonal_feature_cube(
    polygon_geojson: Union[dict, str, Path],
    resolution_m: int = 250,
    region_id: str = "CUSTOM_POLYGON",
    out_data_dir: Optional[Union[Path, str]] = None,
    out_dir: Optional[Union[Path, str]] = None,
    region_name: Optional[str] = None,
    source_dirs: Optional[Sequence[Union[Path, str]]] = None,
    target_feature_names: Optional[Sequence[str]] = None,
    allow_auto_coarsen: bool = True,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> dict[str, Any]:
    """Rasterize and clip multimodal physical channels strictly inside a polygon at `resolution_m`."""
    t0 = time.time()
    if region_name:
        region_id = str(region_name)
    if out_dir is not None and out_data_dir is None:
        out_data_dir = out_dir

    val_info = validate_polygon_and_resolution(
        polygon_geojson, resolution_m=resolution_m, allow_auto_coarsen=allow_auto_coarsen
    )
    eff_res_m = int(val_info["effective_resolution_m"])
    area_km2 = float(val_info["area_km2"])
    val_msg = str(val_info["message"])
    logger.info("[%s] %s", region_id, val_msg)

    norm_fc = normalize_polygon_geojson(
        polygon_geojson,
        name=region_id.replace("_", " ").title(),
        region_id=region_id,
        resolution_m=eff_res_m,
    )
    geoms = extract_shapely_geometries(norm_fc)
    merged_poly = unary_union(geoms)
    min_lon, min_lat, max_lon, max_lat = merged_poly.bounds

    # Pad bounding box slightly by 2 pixels so polygon edges are cleanly sampled
    res_deg = float(eff_res_m) / 111_320.0
    pad_deg = 2.0 * res_deg
    min_lon = max(108.5, min_lon - pad_deg)
    max_lon = min(119.5, max_lon + pad_deg)
    min_lat = max(-4.5, min_lat - pad_deg)
    max_lat = min(7.0, max_lat + pad_deg)

    W = max(16, int(round((max_lon - min_lon) / res_deg)))
    H = max(16, int(round((max_lat - min_lat) / res_deg)))
    dst_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, W, H)

    out_dir = Path(out_data_dir) if out_data_dir is not None else (
        ROOT_DIR / "data" / "polygons" / f"{region_id.lower()}_{eff_res_m}m" / "processed"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save exact polygonal boundary GeoJSON
    boundary_path = out_dir / "scanned_boundary.geojson"
    boundary_path.write_text(json.dumps(norm_fc, indent=2) + "\n", encoding="utf-8")

    if progress_cb:
        progress_cb(f"Rasterizing polygon mask ({H}x{W} @ {eff_res_m}m, {area_km2:,.1f} km²)", 5.0)

    inside_poly_mask = compute_polygon_mask_2d(norm_fc, height=H, width=W, transform=dst_transform)
    n_inside_poly = int(inside_poly_mask.sum())
    if n_inside_poly == 0:
        raise ValueError(f"Polygon produced 0 interior grid cells on {H}x{W} grid at {eff_res_m}m.")

    # Resolve source directories for physical GeoTIFFs
    if source_dirs is not None:
        src_dirs = [Path(d) for d in source_dirs if Path(d).exists()]
    else:
        # Prefer East Kalimantan 250m stack first, supplemented by Kalimantan 75-channel processed rasters
        candidates = [
            ROOT_DIR / "data" / "east_kalimantan" / "processed",
            ROOT_DIR / "data" / "kalimantan" / "processed",
            ROOT_DIR / "data" / "kutai_basin_100m" / "processed",
        ]
        src_dirs = [d for d in candidates if d.exists()]

    if not src_dirs:
        raise FileNotFoundError("No processed raster source directories found in data/.")

    # Collect available channel GeoTIFFs across source_dirs
    skip_stems = {
        "prospectivity_k3_coal",
        "prospectivity_k4_metals",
        "fused_raw_score",
        "conformal_bounds_90ci",
        "calibrated_prospectivity",
        "calibrated_prospectivity_metals",
        "calibrated_prospectivity_coal",
        "dae_reconstruction_error",
        "geology_coal_contact_dist_km",
    }
    tif_map: dict[str, Path] = {}
    for sdir in reversed(src_dirs):
        for p in sorted(sdir.glob("*.tif")):
            if p.stem not in skip_stems:
                tif_map[p.stem] = p

    # If target_feature_names is specified, make sure we prioritize those channels plus logistics/labels
    essential_aux = {
        "dem_elevation",
        "dem_slope",
        "dem_plan_curvature",
        "spec_ndvi",
        "w_canopy",
        "regolith_cover_m",
        "outcrop_exposure_index",
        "dist_major_rivers_km",
        "dist_haul_roads_km",
        "dist_transport_km",
        "gfw_forest_disturbance",
        "forest_disturbance_rate",
        "coastal_shelf_dist_km",
        "dilation_tendency_Td",
        "spec_clay_hydroxyl_gated",
        "explored_footprint",
        "alteration_benchmark",
        "coal_tonnage_weight",
        "label_coal",
        "label_coal_k0_high_cv",
        "label_coal_k1_med_cv",
        "label_coal_k2_low_cv",
    }
    if target_feature_names:
        desired_stems = set(target_feature_names) | essential_aux
        selected_tifs = {k: v for k, v in tif_map.items() if k in desired_stems}
        # If none matched, fall back to all available tifs
        if len(selected_tifs) < 5:
            selected_tifs = tif_map
    else:
        selected_tifs = tif_map

    out_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": W,
        "height": H,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": dst_transform,
        "compress": "deflate",
    }

    raster_data: dict[str, np.ndarray] = {}
    total_ch = max(1, len(selected_tifs))

    for idx, (name, p) in enumerate(sorted(selected_tifs.items()), 1):
        is_cat = name in CATEGORICAL_OR_INT_LAYERS
        resample_method = Resampling.nearest if is_cat else Resampling.bilinear
        try:
            with rasterio.open(p) as src:
                win = win_from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
                arr = src.read(1, window=win, out_shape=(H, W), resampling=resample_method, boundless=True, fill_value=-9999.0)
        except Exception as exc:
            logger.warning("Could not window-read %s: %s", p, exc)
            continue

        arr_f32 = np.where(np.isnan(arr) | (arr <= -9000.0), np.nan, arr).astype(np.float32)
        # Strictly mask outside polygon to NaN
        arr_f32[~inside_poly_mask] = np.nan
        raster_data[name] = arr_f32

        if progress_cb and (idx % 5 == 0 or idx == total_ch):
            pct = 5.0 + 55.0 * (idx / total_ch)
            progress_cb(f"Resampled & polygon-clipped channel {idx}/{total_ch}: {name}", pct)

    # Establish onshore land mask strictly inside the polygon
    if "dem_elevation" in raster_data:
        dem_arr = raster_data["dem_elevation"]
        valid_mask = inside_poly_mask & np.isfinite(dem_arr) & (dem_arr >= -5.0)
        # If no valid cells have dem >= -5 (e.g. edge coverage), fall back to finite inside_poly_mask
        if int(valid_mask.sum()) == 0:
            valid_mask = inside_poly_mask & np.isfinite(dem_arr)
    else:
        first_arr = next(iter(raster_data.values()))
        valid_mask = inside_poly_mask & np.isfinite(first_arr)

    if int(valid_mask.sum()) == 0:
        valid_mask = inside_poly_mask.copy()

    # Recompute Evans (1980) plan curvature at the exact requested physical resolution_m
    if "dem_elevation" in raster_data:
        dem_clean = np.where(valid_mask, np.nan_to_num(raster_data["dem_elevation"], nan=15.0), 0.0).astype(np.float32)
        f_curv = plan_curvature(dem_clean, resolution_m=float(eff_res_m)).astype(np.float32)
        f_curv[~valid_mask] = np.nan
        raster_data["dem_plan_curvature"] = f_curv

        if "dist_topo_lineaments_km" in raster_data or (target_feature_names and "dist_topo_lineaments_km" in target_feature_names):
            curv_smooth = gaussian_filter(np.abs(np.nan_to_num(f_curv, nan=0.0)), sigma=1.2)
            valid_curv = curv_smooth[valid_mask]
            thr = float(np.percentile(valid_curv, 86)) if len(valid_curv) > 0 else 0.0
            lin_mask = (curv_smooth >= thr) & valid_mask
            dist_km = (distance_transform_edt(~lin_mask) * (float(eff_res_m) / 1000.0)).astype(np.float32)
            dist_km[~valid_mask] = np.nan
            raster_data["dist_topo_lineaments_km"] = dist_km

    if progress_cb:
        progress_cb("Rasterizing K=4 metallic & K=3 coal ground-truth labels inside polygon...", 68.0)

    # Rasterize K=4 metallic targets at requested resolution
    deposit_targets = build_k4_deposit_targets()
    buffer_m = max(300.0, min(1000.0, float(eff_res_m) * 4.0))
    rasters_k4 = rasterize_borneo_k4_labels(deposit_targets, out_profile, buffer_m=buffer_m)
    for k_col in K4_LABEL_COLS:
        arr_k = rasters_k4[k_col].astype(np.float32)
        arr_k[~valid_mask] = np.nan
        raster_data[k_col] = arr_k

    for aux_col in ("explored_footprint", "alteration_benchmark"):
        if aux_col in rasters_k4:
            arr_aux = rasters_k4[aux_col].astype(np.float32)
            arr_aux[~valid_mask] = np.nan
            if aux_col in raster_data:
                raster_data[aux_col] = np.where(
                    valid_mask,
                    np.maximum(np.nan_to_num(raster_data[aux_col], nan=0.0), np.nan_to_num(arr_aux, nan=0.0)),
                    np.nan,
                ).astype(np.float32)
            else:
                raster_data[aux_col] = arr_aux

    # Ensure K=3 coal labels are populated inside the polygon
    k3_cols = ["label_coal_k0_high_cv", "label_coal_k1_med_cv", "label_coal_k2_low_cv"]
    has_coal_pos = any(
        c in raster_data and int(np.sum(np.nan_to_num(raster_data[c], nan=0.0) == 1)) > 0
        for c in k3_cols
    )
    if not has_coal_pos:
        try:
            ndvi_2d = np.nan_to_num(raster_data.get("spec_ndvi", np.full((H, W), 0.5, dtype=np.float32)), nan=0.5).astype(np.float64)
            mine_2d = np.zeros((H, W), dtype=np.int8)
            poly_2d = np.ones((H, W), dtype=np.int32)
            coal_gdf = load_kalimantan_coal_sources()
            coal_res = build_k3_coal_rank_targets(
                concessions=coal_gdf,
                ndvi=ndvi_2d,
                active_mine_mask=mine_2d,
                formation_polygon_id=poly_2d,
                transform=dst_transform,
            )
            for c_col in k3_cols:
                c_arr = coal_res[c_col].astype(np.float32)
                c_arr[~valid_mask] = np.nan
                raster_data[c_col] = c_arr
            ton_arr = coal_res["coal_tonnage_weights"].astype(np.float32)
            ton_arr[~valid_mask] = np.nan
            raster_data["coal_tonnage_weight"] = ton_arr
        except Exception as exc:
            logger.warning("Could not rasterize K=3 coal targets onto polygon grid: %s", exc)
            for c_col in k3_cols:
                if c_col not in raster_data:
                    raster_data[c_col] = np.where(valid_mask, 0.0, np.nan).astype(np.float32)

    # Write per-channel GeoTIFFs masked outside polygon
    for name, arr_2d in raster_data.items():
        tif_path = out_dir / f"{name}.tif"
        arr_write = np.where(np.isfinite(arr_2d) & valid_mask, arr_2d, -9999.0).astype(np.float32)
        with rasterio.open(tif_path, "w", **out_profile) as dst:
            dst.write(arr_write, 1)

    # Extract tabular dataset.parquet strictly for valid cells inside polygon
    rows_valid, cols_valid = np.where(valid_mask)
    rows_valid = rows_valid.astype(np.int32)
    cols_valid = cols_valid.astype(np.int32)
    n_valid = len(rows_valid)

    xs, ys = rasterio.transform.xy(dst_transform, rows_valid, cols_valid)
    df_dict: dict[str, np.ndarray] = {
        "row": rows_valid,
        "col": cols_valid,
        "lon": np.asarray(xs, dtype=np.float32),
        "lat": np.asarray(ys, dtype=np.float32),
    }

    for name, arr_2d in raster_data.items():
        vals = arr_2d[rows_valid, cols_valid]
        if name in CATEGORICAL_OR_INT_LAYERS:
            df_dict[name] = np.nan_to_num(vals, nan=0.0).astype(np.int32)
        else:
            df_dict[name] = vals.astype(np.float32)

    pos_metals_any = np.zeros(n_valid, dtype=bool)
    for k_col in K4_LABEL_COLS:
        if k_col in df_dict:
            pos_metals_any |= (df_dict[k_col] == 1)

    pos_coal_any = np.zeros(n_valid, dtype=bool)
    for c_col in k3_cols:
        if c_col in df_dict:
            pos_coal_any |= (df_dict[c_col] == 1)
    df_dict["label_coal"] = pos_coal_any.astype(np.int32)
    df_dict["label"] = np.where(pos_metals_any | pos_coal_any, 1, -1).astype(np.int32)

    df = pd.DataFrame(df_dict)

    non_feat_set = set(EXTENDED_NON_FEATURE_COLS) | set(NON_FEATURE_COLS) | CATEGORICAL_OR_INT_LAYERS | {
        "row", "col", "lon", "lat", "w_canopy", "coal_tonnage_weight", "geology_coal_contact_dist_km"
    }
    feature_cols = [
        c for c in df.columns
        if c not in non_feat_set
        and not c.startswith("label_")
        and not any(pat in c.lower() for pat in _FORBIDDEN_RAW_BAND_PATTERNS)
    ]
    if target_feature_names:
        # Order matching target_feature_names first, plus any extra valid features
        ordered = [c for c in target_feature_names if c in feature_cols]
        extras = [c for c in feature_cols if c not in ordered]
        feature_cols = ordered + extras

    verify_no_raw_pixel_leakage(feature_cols)

    # Cap distance features at physical limit (15.0 km / 15000.0 m)
    for c in feature_cols:
        if c.startswith("dist_") or c.endswith("_dist_km"):
            limit = 15000.0 if c.endswith("_m") else 15.0
            df[c] = np.clip(df[c].to_numpy(dtype=np.float32), 0.0, limit)

    parquet_path = out_dir / "dataset.parquet"
    df.to_parquet(parquet_path, index=False)

    # Build and save FeatureStack with NaN outside polygon
    stack_features = feature_cols[:28] if len(feature_cols) > 28 else list(feature_cols)
    stack_layers = [
        np.where(valid_mask, raster_data[f], np.nan).astype(np.float32)
        for f in stack_features
    ]
    stack_arr = np.stack(stack_layers, axis=0).astype(np.float32)
    stack_profile = dict(out_profile)
    stack_profile["count"] = len(stack_features)

    f_stack = FeatureStack(
        array=stack_arr,
        feature_names=stack_features,
        profile=stack_profile,
    )
    stack_path = out_dir / "feature_stack.joblib"
    joblib.dump(f_stack, stack_path)

    if progress_cb:
        progress_cb(
            f"Polygonal cube ready: {n_valid:,} valid cells inside polygon ({int(pos_metals_any.sum())} metal pos, {int(pos_coal_any.sum())} coal pos)",
            100.0,
        )

    elapsed = time.time() - t0
    meta_dict = {
        "region_id": region_id,
        "data_dir": str(out_dir),
        "boundary_geojson_path": str(boundary_path),
        "boundary_geojson": norm_fc,
        "resolution_m": eff_res_m,
        "area_sq_km": round(area_km2, 2),
        "area_km2": round(area_km2, 2),
        "height": H,
        "width": W,
        "n_valid_cells": n_valid,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "n_metal_positives": int(pos_metals_any.sum()),
        "n_coal_positives": int(pos_coal_any.sum()),
        "elapsed_sec": round(elapsed, 2),
    }
    meta_path = out_dir / "feature_cube_meta.json"
    meta_path.write_text(json.dumps(meta_dict, indent=2) + "\n", encoding="utf-8")
    return meta_dict
