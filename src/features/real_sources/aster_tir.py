"""Real ASTER L1T thermal-infrared silica indices (Planetary Computer ``aster-l1t``).

ASTER TIR bands 10-14 (90 m) are converted from DN to at-sensor radiance with the
published unit-conversion coefficients, L = (DN - 1) * UCC. Cold pixels (band-13
brightness temperature below ``min_bt13_k``) are treated as cloud and removed. Indices
follow Ninomiya et al. (2005):

* ``aster_silica_index`` (SiO2 index) = L13 / L10
* ``aster_quartz_index`` (QI) = L11^2 / (L10 * L12)

The Planetary Computer archive covers 2000-2006 only, and humid-tropical cloud leaves
large gaps. Uncovered pixels stay NaN.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

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

COLLECTION = "aster-l1t"
UCC = {10: 6.822e-3, 11: 6.780e-3, 12: 6.590e-3, 13: 5.693e-3, 14: 5.225e-3}
K1_B13, K2_B13 = 866.468, 1350.069


def _reader(read_res_m: float, min_bt13_k: float):
    def read_item(item: Any, grid: GridSpec, win) -> Optional[dict[str, np.ndarray]]:
        if "TIR" not in item.assets:
            return None
        dn, tr, crs, _ = read_cog_decimated(sign_href(item.assets["TIR"].href), read_res_m, bands=(1, 2, 3, 4, 5))
        valid = np.all(np.isfinite(dn) & (dn > 1), axis=0)
        L = {b: (dn[i] - 1.0) * UCC[b] for i, b in enumerate((10, 11, 12, 13, 14))}
        with np.errstate(invalid="ignore", divide="ignore"):
            bt13 = K2_B13 / np.log(K1_B13 / L[13] + 1.0)
            si = L[13] / L[10]
            qi = (L[11] ** 2) / (L[10] * L[12])
        ok = valid & np.isfinite(bt13) & (bt13 >= min_bt13_k)
        if ok.mean() < 0.02:
            return None
        return {
            "aster_silica_index": reproject_into(np.where(ok, si, np.nan), tr, crs, grid, win),
            "aster_quartz_index": reproject_into(np.where(ok, qi, np.nan), tr, crs, grid, win),
        }

    return read_item


def build_aster_tir_layers(
    grid: GridSpec,
    manifest: ProvenanceManifest,
    cache_dir: Path,
    max_cloud: float = 20.0,
    min_bt13_k: float = 288.0,
    read_res_m: float = 180.0,
    max_workers: int = 16,
) -> dict[str, np.ndarray]:
    names = ["aster_silica_index", "aster_quartz_index"]
    cached = cached_layers(cache_dir, names, grid)
    if cached is None:
        items = stac_search_chunked(COLLECTION, grid.bounds_4326(), "2000-01-01/2008-12-31", query={"eo:cloud_cover": {"lte": max_cloud}})
        if not items:
            raise RealDataUnavailableError("No ASTER L1T scenes found for the grid")
        groups: dict[str, list[Any]] = defaultdict(list)
        for it in items:
            cx = 0.5 * (it.bbox[0] + it.bbox[2])
            cy = 0.5 * (it.bbox[1] + it.bbox[3])
            groups[f"{math.floor(cx)}_{math.floor(cy)}"].append(it)
        logger.info("ASTER TIR: %d scenes in %d 1-degree groups", len(items), len(groups))
        res, _, used = composite_items_by_group(grid, groups, _reader(read_res_m, min_bt13_k), names, max_workers, "ASTER")
        for n in names:
            save_layer(res[n], grid, cache_dir / f"{n}.tif")
        (cache_dir / "aster_item_ids.txt").write_text("\n".join(used), encoding="utf-8")
        cached = res
    used_ids = (cache_dir / "aster_item_ids.txt").read_text(encoding="utf-8").split() if (cache_dir / "aster_item_ids.txt").exists() else []
    common = dict(
        kind=MEASURED,
        source="NASA/METI ASTER L1T TIR at-sensor radiance (Planetary Computer aster-l1t)",
        url="https://planetarycomputer.microsoft.com/dataset/aster-l1t",
        native_resolution_m=90.0,
        temporal_coverage="2000-2006 archive on Planetary Computer",
        item_ids=used_ids,
        notes=f"Scene cloud <= {max_cloud}%; pixels with band-13 brightness temperature < {min_bt13_k} K removed as cloud. "
        "Radiance-based (no emissivity separation).",
    )
    manifest.add(LayerProvenance(name="aster_silica_index", method="Ninomiya SiO2 index L13/L10, median per 1-degree scene group", **common))
    manifest.add(LayerProvenance(name="aster_quartz_index", method="Ninomiya quartz index L11^2/(L10*L12), median per 1-degree scene group", **common))
    return {n: cached[n] for n in names}
