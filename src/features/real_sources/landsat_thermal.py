"""Real Landsat 8/9 Collection-2 Level-2 surface temperature (Planetary Computer ``landsat-c2-l2``).

Per WRS-2 path/row, dry-season scenes with low land cloud cover are read (ST_B10
``lwir11`` and ``qa_pixel``), cloud / dilated cloud / shadow / fill pixels are removed
with the QA bits, and the per-pixel median land surface temperature (deg C) is taken.

Derived layer ``thermal_anomaly_st`` (deg C): the median LST corrected to sea level with
the standard 6.5 K/km environmental lapse rate (real DEM), minus its 25 km NaN-aware
Gaussian background. It isolates local warm/cool features from the regional climate.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter

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
from .sentinel2 import reproject_mask

logger = logging.getLogger(__name__)

COLLECTION = "landsat-c2-l2"
ST_SCALE, ST_OFFSET = 0.00341802, 149.0  # Kelvin
QA_BAD_BITS = (0, 1, 3, 4)  # fill, dilated cloud, cloud, cloud shadow
MIN_SEA_LEVEL_LST_C = 18.0  # 0.1st percentile of lapse-corrected land LST over Kalimantan is ~20 C


def _reader(read_res_m: float):
    def read_item(item: Any, grid: GridSpec, win) -> Optional[dict[str, np.ndarray]]:
        if "lwir11" not in item.assets or "qa_pixel" not in item.assets:
            return None
        qa, qtr, qcrs, _ = read_cog_decimated(sign_href(item.assets["qa_pixel"].href), read_res_m, resampling=Resampling.nearest)
        q = np.nan_to_num(qa[0], nan=1.0).astype(np.uint16)
        bad = np.zeros(q.shape, dtype=bool)
        for bit in QA_BAD_BITS:
            bad |= ((q >> bit) & 1).astype(bool)
        if (~bad).mean() < 0.02:
            return None
        st, tr, crs, _ = read_cog_decimated(sign_href(item.assets["lwir11"].href), read_res_m)
        a = st[0]
        m = ~bad if a.shape == bad.shape else reproject_mask(~bad, qtr, qcrs, a.shape, tr)
        lst_c = np.where(m & (a > 0), a * ST_SCALE + ST_OFFSET - 273.15, np.nan).astype(np.float32)
        lst_c[(lst_c < -20) | (lst_c > 80)] = np.nan
        return {"thermal_lst_c": reproject_into(lst_c, tr, crs, grid, win, Resampling.average)}

    return read_item


def nan_gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    ok = np.isfinite(a)
    num = gaussian_filter(np.where(ok, a, 0.0).astype(np.float64), sigma)
    den = gaussian_filter(ok.astype(np.float64), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 1e-3, out, np.nan).astype(np.float32)


def build_landsat_thermal_layers(
    grid: GridSpec,
    manifest: ProvenanceManifest,
    cache_dir: Path,
    dem: np.ndarray,
    years: tuple[int, int] = (2019, 2024),
    months: tuple[int, ...] = (6, 7, 8, 9),
    max_cloud_land: float = 50.0,
    scenes_per_pathrow: int = 12,
    read_res_m: float = 240.0,
    background_km: float = 25.0,
    max_workers: int = 16,
) -> dict[str, np.ndarray]:
    cached = cached_layers(cache_dir, ["thermal_lst_c"], grid)
    if cached is None:
        items = stac_search_chunked(
            COLLECTION,
            grid.bounds_4326(),
            f"{years[0]}-01-01/{years[1]}-12-31",
            query={"platform": {"in": ["landsat-8", "landsat-9"]}, "landsat:cloud_cover_land": {"lt": max_cloud_land}},
        )
        items = [it for it in items if it.datetime is not None and it.datetime.month in months]
        if not items:
            raise RealDataUnavailableError("No Landsat 8/9 L2 scenes found for the grid")
        groups: dict[str, list[Any]] = defaultdict(list)
        for it in items:
            key = f"{it.properties.get('landsat:wrs_path')}_{it.properties.get('landsat:wrs_row')}"
            groups[key].append(it)
        groups = {k: sorted(v, key=lambda i: i.properties.get("landsat:cloud_cover_land", 100.0))[:scenes_per_pathrow] for k, v in groups.items()}
        logger.info("Landsat ST: %d path/rows, %d scenes", len(groups), sum(len(v) for v in groups.values()))
        res, obs, used = composite_items_by_group(grid, groups, _reader(read_res_m), ["thermal_lst_c"], max_workers, "LandsatST")
        save_layer(res["thermal_lst_c"], grid, cache_dir / "thermal_lst_c.tif")
        (cache_dir / "landsat_item_ids.txt").write_text("\n".join(used), encoding="utf-8")
        cached = res
    used_ids = (cache_dir / "landsat_item_ids.txt").read_text(encoding="utf-8").split() if (cache_dir / "landsat_item_ids.txt").exists() else []
    lst = cached["thermal_lst_c"]
    lst_sl = lst + 0.0065 * np.maximum(np.nan_to_num(dem, nan=0.0), 0.0)
    # Cloud tops that slipped through QA_PIXEL leave cold cells (down to -20 C). A ~10:30 daytime
    # tropical land surface, lapse-corrected to sea level, is not below MIN_SEA_LEVEL_LST_C: mask them.
    cold = np.isfinite(lst_sl) & (lst_sl < MIN_SEA_LEVEL_LST_C)
    lst = np.where(cold, np.nan, lst).astype(np.float32)
    lst_sl = np.where(cold, np.nan, lst_sl)
    bg = nan_gaussian(lst_sl, sigma=background_km * 1000.0 / grid.res_m)
    anomaly = (lst_sl - bg).astype(np.float32)

    manifest.add(
        LayerProvenance(
            name="thermal_lst_c",
            kind=MEASURED,
            source="USGS Landsat 8/9 Collection 2 Level-2 Surface Temperature ST_B10 (Planetary Computer landsat-c2-l2)",
            url="https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2",
            method=f"QA_PIXEL-masked (bits {QA_BAD_BITS}) per-path/row median of <= {scenes_per_pathrow} scenes "
            f"(land cloud < {max_cloud_land}%), ST = DN*{ST_SCALE}+{ST_OFFSET} K -> deg C, area-averaged to grid",
            native_resolution_m=100.0,
            temporal_coverage=f"{years[0]}-{years[1]}, months {','.join(map(str, months))}, ~10:30 local overpass",
            item_ids=used_ids,
            notes=f"TIRS native 100 m, delivered resampled to 30 m. Cells whose lapse-corrected LST < {MIN_SEA_LEVEL_LST_C:.0f} C "
            f"(residual cloud; {int(cold.sum())} cells) are NaN.",
        )
    )
    manifest.derive(
        "thermal_anomaly_st",
        ["thermal_lst_c", "dem_elevation"],
        f"(LST + 6.5 K/km * elevation) minus {background_km:.0f} km NaN-aware Gaussian background",
    )
    return {"thermal_lst_c": lst, "thermal_anomaly_st": anomaly}
