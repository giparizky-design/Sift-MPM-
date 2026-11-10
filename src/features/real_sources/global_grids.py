"""Real global gridded products: EMAG2v3 magnetics, Hansen GFC forest loss, SoilGrids depth to bedrock."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds as window_from_bounds

from .core import (
    MEASURED,
    GDAL_REMOTE_ENV,
    GridSpec,
    LayerProvenance,
    ProvenanceManifest,
    RealDataUnavailableError,
    download,
    reproject_into,
    warp_file_to_grid,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EMAG2v3 (Meyer, Saltus & Chulliat 2017), NOAA NCEI
# ---------------------------------------------------------------------------
EMAG2_BASE = "https://www.ngdc.noaa.gov/geomag/data/EMAG2/EMAG2_V3_20170530"
EMAG2_SEALEVEL = "EMAG2_V3_20170530_Sealevel.tif"
EMAG2_ERROR = "EMAG2_V3_20170530_Error.tif"
EMAG2_CODE = "EMAG2_V3_20170530_Code.tif"


def build_emag2_layers(grid: GridSpec, manifest: ProvenanceManifest, raw_dir: Path) -> dict[str, np.ndarray]:
    """Total-field magnetic anomaly at sea level (nT) and its source code / error grids."""
    out: dict[str, np.ndarray] = {}
    sea = download(f"{EMAG2_BASE}/{EMAG2_SEALEVEL}", raw_dir / EMAG2_SEALEVEL, min_bytes=10_000_000)
    tmi = warp_file_to_grid(sea, grid, resampling=Resampling.cubic_spline)
    # EMAG2 uses a large negative fill value for no data
    tmi[(tmi < -5e4) | (tmi > 5e4)] = np.nan
    if not np.isfinite(tmi).any():
        raise RealDataUnavailableError("EMAG2v3 has no valid cells over the grid")
    out["emag2_tmi_nt"] = tmi
    manifest.add(
        LayerProvenance(
            name="emag2_tmi_nt",
            kind=MEASURED,
            source="NOAA NCEI EMAG2v3 Earth Magnetic Anomaly Grid, sea-level version (Meyer et al. 2017)",
            url=f"{EMAG2_BASE}/{EMAG2_SEALEVEL}",
            method="2 arc-minute grid warped to target grid with cubic spline",
            native_resolution_m=3700.0,
            notes="Regional-scale field (~3.7 km). EMAG2v3 leaves unsurveyed cells empty (no model fill): "
            "over Kalimantan only ~27% of land has data; the rest is NaN. RTP/analytic-signal FFTs near data edges ring.",
        )
    )
    for fname, name, desc in ((EMAG2_ERROR, "emag2_error_nt", "EMAG2v3 error estimate (nT)"), (EMAG2_CODE, "emag2_source_code", "EMAG2v3 data source code")):
        try:
            # Read only the grid window over HTTP range requests (the NOAA server is slow for full downloads)
            local = raw_dir / fname
            p = local if local.exists() else "/vsicurl/" + f"{EMAG2_BASE}/{fname}"
            arr = warp_file_to_grid(p, grid, resampling=Resampling.nearest)
            arr[(arr < -5e4) | (arr > 5e4) | (arr <= -999.0)] = np.nan
            if name == "emag2_error_nt":
                arr[arr < 0] = np.nan  # an error estimate cannot be negative; negatives are fill codes
            out[name] = arr
            manifest.add(
                LayerProvenance(name=name, kind=MEASURED, source=desc, url=f"{EMAG2_BASE}/{fname}", method="nearest-neighbour warp", native_resolution_m=3700.0,
                                notes="Quality/provenance layer; not a model feature.")
            )
        except Exception as exc:
            logger.warning("EMAG2 auxiliary grid %s unavailable (quality layer omitted): %s", fname, exc)
    return out


# ---------------------------------------------------------------------------
# Hansen et al. (2013) Global Forest Change v1.11 (2000-2023)
# ---------------------------------------------------------------------------
GFC_VERSION = "GFC-2023-v1.11"
GFC_BASE = f"https://storage.googleapis.com/earthenginepartners-hansen/{GFC_VERSION}"


def _gfc_tiles(bounds: tuple[float, float, float, float]) -> list[str]:
    west, south, east, north = bounds
    tiles = []
    for top in range(int(math.ceil(north / 10.0) * 10), int(math.floor(south / 10.0) * 10), -10):
        for left in range(int(math.floor(west / 10.0) * 10), int(math.ceil(east / 10.0) * 10), 10):
            lat = f"{abs(top):02d}{'N' if top >= 0 else 'S'}"
            lon = f"{abs(left):03d}{'E' if left >= 0 else 'W'}"
            tiles.append(f"{lat}_{lon}")
    return tiles


def _block_mean_into(path: Path, grid: GridSpec, fn, acc_sum: np.ndarray, acc_cnt: np.ndarray) -> None:
    """Aggregate a fine 0.00025-deg tile into the grid by exact block means of fn(values)."""
    with rasterio.open(path) as src:
        gb = grid.bounds_4326()
        tb = src.bounds
        west, south = max(gb[0], tb.left), max(gb[1], tb.bottom)
        east, north = min(gb[2], tb.right), min(gb[3], tb.top)
        if east <= west or north <= south:
            return
        win = window_from_bounds(west, south, east, north, transform=src.transform).round_offsets().round_lengths()
        # process in row strips
        strip = 2000
        for r in range(0, int(win.height), strip):
            h = min(strip, int(win.height) - r)
            w = Window(win.col_off, win.row_off + r, win.width, h)
            vals = src.read(1, window=w)
            v = fn(vals).astype(np.float32)
            tr = src.window_transform(w)
            part = reproject_into(v, tr, src.crs, grid, None, Resampling.average)
            ok = np.isfinite(part)
            acc_sum[ok] += part[ok]
            acc_cnt[ok] += 1


def build_hansen_gfc_layers(grid: GridSpec, manifest: ProvenanceManifest, raw_dir: Path) -> dict[str, np.ndarray]:
    """Forest-loss fraction (2001-2023) and year-2000 tree cover, aggregated to the grid."""
    loss_sum = np.zeros(grid.shape, np.float64)
    loss_cnt = np.zeros(grid.shape, np.float64)
    tc_sum = np.zeros(grid.shape, np.float64)
    tc_cnt = np.zeros(grid.shape, np.float64)
    tiles = _gfc_tiles(grid.bounds_4326())
    for t in tiles:
        ly = download(f"{GFC_BASE}/Hansen_{GFC_VERSION}_lossyear_{t}.tif", raw_dir / f"Hansen_{GFC_VERSION}_lossyear_{t}.tif", 100_000)
        tc = download(f"{GFC_BASE}/Hansen_{GFC_VERSION}_treecover2000_{t}.tif", raw_dir / f"Hansen_{GFC_VERSION}_treecover2000_{t}.tif", 100_000)
        _block_mean_into(ly, grid, lambda a: (a > 0).astype(np.float32), loss_sum, loss_cnt)
        _block_mean_into(tc, grid, lambda a: a.astype(np.float32) / 100.0, tc_sum, tc_cnt)
    with np.errstate(invalid="ignore", divide="ignore"):
        loss = np.where(loss_cnt > 0, loss_sum / np.maximum(loss_cnt, 1), np.nan).astype(np.float32)
        tc2000 = np.where(tc_cnt > 0, tc_sum / np.maximum(tc_cnt, 1), np.nan).astype(np.float32)
        rate = np.where(tc2000 > 0.05, np.clip(loss / tc2000, 0.0, 1.0), np.where(np.isfinite(tc2000), 0.0, np.nan)).astype(np.float32)
    common = dict(
        kind=MEASURED,
        source=f"Hansen et al. (2013) Global Forest Change {GFC_VERSION}",
        url=GFC_BASE,
        native_resolution_m=30.0,
        temporal_coverage="2000-2023",
        item_ids=tiles,
    )
    manifest.add(LayerProvenance(name="gfw_forest_disturbance", method="fraction of 30 m pixels with any loss year 2001-2023, block-averaged", **common))
    manifest.add(LayerProvenance(name="tree_cover_2000", method="mean year-2000 percent tree cover / 100, block-averaged", **common))
    manifest.derive("forest_disturbance_rate", ["gfw_forest_disturbance", "tree_cover_2000"], "loss fraction / year-2000 tree-cover fraction (0 where tree cover <= 5%)")
    return {"gfw_forest_disturbance": loss, "tree_cover_2000": tc2000, "forest_disturbance_rate": rate}


# ---------------------------------------------------------------------------
# SoilGrids 2017 absolute depth to bedrock (Shangguan et al. 2017)
# ---------------------------------------------------------------------------
BDTICM_URL = "https://files.isric.org/soilgrids/former/2017-03-10/data/BDTICM_M_250m_ll.tif"


def build_depth_to_bedrock(grid: GridSpec, manifest: ProvenanceManifest, raw_dir: Path, overview_factor: int = 2) -> dict[str, np.ndarray]:
    """Depth to bedrock (m), read from the global COG at a coarse overview and cached as a regional clip."""
    clip = raw_dir / "BDTICM_M_250m_ll_kalimantan_clip.tif"
    if not clip.exists():
        gb = grid.bounds_4326()
        pad = 0.25
        try:
            with rasterio.Env(**GDAL_REMOTE_ENV):
                with rasterio.open("/vsicurl/" + BDTICM_URL) as src:
                    win = window_from_bounds(gb[0] - pad, gb[1] - pad, gb[2] + pad, gb[3] + pad, transform=src.transform).round_offsets().round_lengths()
                    oh = int(math.ceil(win.height / overview_factor))
                    ow = int(math.ceil(win.width / overview_factor))
                    arr = src.read(1, window=win, out_shape=(oh, ow), resampling=Resampling.average).astype(np.float32)
                    nod = src.nodata
                    tr = src.window_transform(win) * Affine.scale(win.width / ow, win.height / oh)
                    crs = src.crs
        except Exception as exc:
            raise RealDataUnavailableError(f"SoilGrids BDTICM unavailable: {exc}") from exc
        if nod is not None:
            arr[arr == nod] = np.nan
        arr[arr < 0] = np.nan
        clip.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(clip, "w", driver="GTiff", height=oh, width=ow, count=1, dtype="float32", crs=crs, transform=tr, nodata=np.nan, compress="deflate") as dst:
            dst.write(arr, 1)
    d_cm = warp_file_to_grid(clip, grid, resampling=Resampling.bilinear)
    manifest.add(
        LayerProvenance(
            name="depth_to_bedrock_m",
            kind=MEASURED,
            source="ISRIC SoilGrids 2017 absolute depth to bedrock BDTICM (Shangguan et al. 2017)",
            url=BDTICM_URL,
            method=f"global 250 m product read at 1/{overview_factor} overview, bilinear to grid, cm -> m",
            native_resolution_m=250.0,
            notes="Machine-learning product trained on borehole/soil-profile observations; treat as modelled observation.",
        )
    )
    return {"depth_to_bedrock_m": (d_cm / 100.0).astype(np.float32)}
