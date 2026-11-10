"""Real Sentinel-2 L2A dry-season composite (Microsoft Planetary Computer, ``sentinel-2-l2a``).

Per MGRS tile, the lowest-cloud dry-season scenes are read at an overview level close
to the target resolution, cloud/shadow/cirrus pixels are removed with the L2A Scene
Classification Layer (SCL), and the per-pixel median surface reflectance is taken.
Tiles are then mean-mosaicked onto the target grid. Pixels with no clear observation
stay NaN.

Indices (all from the composited surface reflectance, documented formulas):

* ``spec_ndvi``  = (B08 - B04) / (B08 + B04)
* ``spec_ndre``  = (B08 - B05) / (B08 + B05)
* ``spec_albedo`` = Liang (2001) shortwave broadband albedo with Sentinel-2 equivalents
  0.356 B02 + 0.130 B04 + 0.373 B08 + 0.085 B11 + 0.072 B12 - 0.0018
* ``spec_ficr``  = B04 / B02 (ferric iron ratio)
* ``spec_clay_hydroxyl`` = B11 / B12 (Al/Mg-OH clay & hydroxyl ratio)
* ``spec_ferrous_ultramafic`` = B12 / B8A + B03 / B04 (ferrous iron index)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rasterio.enums import Resampling

from .core import (
    MEASURED,
    GridSpec,
    LayerProvenance,
    ProvenanceManifest,
    RealDataUnavailableError,
    cached_layers,
    composite_items_by_group,
    read_cog_decimated,
    reproject_into,
    save_layer,
    sign_href,
    stac_search_chunked,
)

logger = logging.getLogger(__name__)

COLLECTION = "sentinel-2-l2a"
BANDS = ("B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12")
# SCL classes kept: 4 vegetation, 5 bare soil, 6 water, 7 unclassified (low-prob cloud), 2 dark area
SCL_CLEAR = (2, 4, 5, 6, 7)
BAND_NAMES = tuple(f"s2_{b.lower()}" for b in BANDS)


def _reflectance_offset(item: Any) -> float:
    """L2A processing baseline >= 04.00 (25 Jan 2022) adds a +1000 DN offset."""
    pb = item.properties.get("s2:processing_baseline")
    try:
        if pb is not None and float(pb) >= 4.0:
            return -1000.0
    except (TypeError, ValueError):
        pass
    if item.datetime is not None and item.datetime.strftime("%Y-%m-%d") >= "2022-01-25":
        return -1000.0
    return 0.0


def _make_reader(read_res_m: float):
    def read_item(item: Any, grid: GridSpec, win) -> Optional[dict[str, np.ndarray]]:
        scl, scl_tr, scl_crs, _ = read_cog_decimated(sign_href(item.assets["SCL"].href), read_res_m, resampling=Resampling.nearest)
        clear = np.isin(scl[0], SCL_CLEAR)
        if clear.mean() < 0.02:
            return None
        offset = _reflectance_offset(item)
        out: dict[str, np.ndarray] = {}
        for b, name in zip(BANDS, BAND_NAMES):
            arr, tr, crs, _ = read_cog_decimated(sign_href(item.assets[b].href), read_res_m)
            a = arr[0]
            if a.shape != clear.shape:
                # resample the clear mask onto this band's decimated grid
                m = reproject_mask(clear, scl_tr, scl_crs, a.shape, tr)
            else:
                m = clear
            refl = (a + offset) / 10000.0
            refl = np.where(m & (a > 0), refl, np.nan).astype(np.float32)
            out[name] = reproject_into(refl, tr, crs, grid, win, Resampling.average)
        return out

    return read_item


def reproject_mask(mask: np.ndarray, src_tr, crs, dst_shape, dst_tr) -> np.ndarray:
    from rasterio.warp import reproject as _rp

    dst = np.zeros(dst_shape, dtype=np.uint8)
    _rp(mask.astype(np.uint8), dst, src_transform=src_tr, src_crs=crs, dst_transform=dst_tr, dst_crs=crs, resampling=Resampling.nearest)
    return dst.astype(bool)


MIN_REFLECTANCE = 0.005  # composite cells darker than 0.5% in any used band (deep water/shadow/offset residue) are invalid


def compute_indices(b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Ratios of near-zero or negative surface reflectance are not physical (NDVI reached -5e5);
    # such cells are masked to NaN, never replaced.
    b = {k: np.where(np.isfinite(v) & (v >= MIN_REFLECTANCE), v, np.nan).astype(np.float32) for k, v in b.items()}

    def nd(x, y):
        with np.errstate(invalid="ignore", divide="ignore"):
            return ((x - y) / (x + y)).astype(np.float32)

    def ratio(x, y):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = x / y
        return np.where(np.isfinite(r) & (y > 1e-4), r, np.nan).astype(np.float32)

    B02, B03, B04, B05 = b["s2_b02"], b["s2_b03"], b["s2_b04"], b["s2_b05"]
    B08, B8A, B11, B12 = b["s2_b08"], b["s2_b8a"], b["s2_b11"], b["s2_b12"]
    return {
        "spec_ndvi": nd(B08, B04),
        "spec_ndre": nd(B08, B05),
        "spec_albedo": (0.356 * B02 + 0.130 * B04 + 0.373 * B08 + 0.085 * B11 + 0.072 * B12 - 0.0018).astype(np.float32),
        "spec_ficr": ratio(B04, B02),
        "spec_clay_hydroxyl": ratio(B11, B12),
        "spec_ferrous_ultramafic": (ratio(B12, B8A) + ratio(B03, B04)).astype(np.float32),
    }


INDEX_METHODS = {
    "spec_ndvi": "(B08-B04)/(B08+B04)",
    "spec_ndre": "(B08-B05)/(B08+B05)",
    "spec_albedo": "Liang (2001) broadband albedo: 0.356*B02+0.130*B04+0.373*B08+0.085*B11+0.072*B12-0.0018",
    "spec_ficr": "B04/B02 ferric iron ratio",
    "spec_clay_hydroxyl": "B11/B12 clay/hydroxyl ratio",
    "spec_ferrous_ultramafic": "B12/B8A + B03/B04 ferrous iron index",
}


def build_sentinel2_layers(
    grid: GridSpec,
    manifest: ProvenanceManifest,
    cache_dir: Path,
    years: tuple[int, int] = (2019, 2024),
    months: tuple[int, ...] = (6, 7, 8, 9),
    max_cloud: float = 40.0,
    scenes_per_tile: int = 16,
    read_res_m: float = 160.0,
    max_workers: int = 16,
) -> dict[str, np.ndarray]:
    """Build the real S2 dry-season composite indices on ``grid`` (cached in ``cache_dir``)."""
    names = list(BAND_NAMES)
    cached = cached_layers(cache_dir, names + ["s2_clear_obs"], grid)
    if cached is None:
        bbox = grid.bounds_4326()
        items = stac_search_chunked(
            COLLECTION, bbox, f"{years[0]}-01-01/{years[1]}-12-31", query={"eo:cloud_cover": {"lt": max_cloud}}
        )
        items = [it for it in items if it.datetime is not None and it.datetime.month in months]
        if not items:
            raise RealDataUnavailableError("No Sentinel-2 L2A scenes found for the grid")
        groups: dict[str, list[Any]] = defaultdict(list)
        for it in items:
            groups[it.properties.get("s2:mgrs_tile", it.id[:6])].append(it)
        groups = {k: sorted(v, key=lambda i: i.properties.get("eo:cloud_cover", 100.0))[:scenes_per_tile] for k, v in groups.items()}
        logger.info("Sentinel-2: %d MGRS tiles, %d scenes selected", len(groups), sum(len(v) for v in groups.values()))
        bands, obs, used = composite_items_by_group(grid, groups, _make_reader(read_res_m), names, max_workers, "S2")
        for n, a in bands.items():
            save_layer(a, grid, cache_dir / f"{n}.tif")
        save_layer(obs.astype(np.float32), grid, cache_dir / "s2_clear_obs.tif")
        (cache_dir / "s2_item_ids.txt").write_text("\n".join(used), encoding="utf-8")
        cached = dict(bands)
        cached["s2_clear_obs"] = obs.astype(np.float32)
    used_ids = (cache_dir / "s2_item_ids.txt").read_text(encoding="utf-8").split() if (cache_dir / "s2_item_ids.txt").exists() else []

    idx = compute_indices(cached)
    temporal = f"{years[0]}-{years[1]}, months {','.join(map(str, months))}"
    for n in names:
        manifest.add(
            LayerProvenance(
                name=n,
                kind=MEASURED,
                source="ESA Copernicus Sentinel-2 L2A surface reflectance (Microsoft Planetary Computer sentinel-2-l2a)",
                url="https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a",
                method=f"SCL-masked (keep {SCL_CLEAR}) per-MGRS-tile median of <= {scenes_per_tile} lowest-cloud scenes "
                f"(cloud < {max_cloud}%), read at ~{read_res_m:.0f} m overview, area-averaged to grid",
                native_resolution_m=10.0 if n in ("s2_b02", "s2_b03", "s2_b04", "s2_b08") else 20.0,
                temporal_coverage=temporal,
                item_ids=used_ids,
                notes="Raw band; excluded from model features by the raw-band leakage guard.",
            )
        )
    manifest.add(
        LayerProvenance(
            name="s2_clear_obs",
            kind=MEASURED,
            source="Sentinel-2 L2A SCL clear-observation count",
            method="max per-tile count of clear SCL observations behind the composite",
            temporal_coverage=temporal,
        )
    )
    for n, m in INDEX_METHODS.items():
        inputs = {
            "spec_ndvi": ["s2_b08", "s2_b04"],
            "spec_ndre": ["s2_b08", "s2_b05"],
            "spec_albedo": ["s2_b02", "s2_b04", "s2_b08", "s2_b11", "s2_b12"],
            "spec_ficr": ["s2_b04", "s2_b02"],
            "spec_clay_hydroxyl": ["s2_b11", "s2_b12"],
            "spec_ferrous_ultramafic": ["s2_b12", "s2_b8a", "s2_b03", "s2_b04"],
        }[n]
        manifest.derive(n, inputs, m)
    out = dict(idx)
    out.update({n: cached[n] for n in names})
    out["s2_clear_obs"] = cached["s2_clear_obs"]
    return out
