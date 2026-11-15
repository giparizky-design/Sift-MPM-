"""Real ALOS-2 PALSAR-2 L-band SAR ingestion and SAR-derived structural channels.

Replaces the former DEM-slope + seeded-noise placeholder in
``scripts/harvest_kalimantan_multimodal.py::fetch_and_process_sar`` with the
genuine JAXA ALOS/ALOS-2 PALSAR/PALSAR-2 annual 25 m global mosaic (HH/HV
amplitude, gamma-naught calibrated DN) served as Cloud-Optimized GeoTIFFs by the
Microsoft Planetary Computer STAC collection ``alos-palsar-mosaic``.

Calibration (Shimada et al. 2009; JAXA mosaic product format):
    gamma0_dB = 10 * log10(DN^2) + CF,   CF = -83.0 dB

Spatial averaging onto the target grid is performed in the *linear power*
domain (not in dB) so that coarse cells represent physically meaningful mean
backscatter. Pixels flagged as no-data, layover, or radar shadow in the JAXA
quality mask are excluded before averaging. Cells with no valid contributing
pixels remain ``NaN`` -- no values are fabricated (Provenance Honesty guardrail).

Emitted canonical channels (same names consumed by ``src/models/coal.py`` and
``src/models/metals.py``):
    - ``sar_palsar2_hh``                HH gamma0 [dB]
    - ``sar_palsar2_hv``                HV gamma0 [dB]
    - ``sar_palsar2_dpr``               Dual-pol ratio HV/HH (linear power)
    - ``sar_palsar2_lineament_density`` Steger (1998) ridge density on HH
    - ``sar_bedding_strike_deg``        Structure-tensor bedding strike [0, 180)
    - ``sar_glcm_anisotropy``           Radar structural anisotropy [0, 1]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

logger = logging.getLogger(__name__)

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PALSAR_COLLECTION = "alos-palsar-mosaic"
PALSAR_CALIBRATION_FACTOR_DB = -83.0
PALSAR_NATIVE_RES_DEG = 1.0 / 4500.0  # 0.8 arc-second (~25 m)

# JAXA mosaic quality mask classes
MASK_NODATA = 0
MASK_WATER = 50
MASK_LAYOVER = 100
MASK_SHADOW = 150
MASK_LAND = 255
VALID_MASK_CLASSES: tuple[int, ...] = (MASK_LAND, MASK_WATER)

SAR_CHANNEL_NAMES: tuple[str, ...] = (
    "sar_palsar2_hh",
    "sar_palsar2_hv",
    "sar_palsar2_dpr",
    "sar_palsar2_lineament_density",
    "sar_bedding_strike_deg",
    "sar_glcm_anisotropy",
)


class SARDataUnavailableError(RuntimeError):
    """Raised when real PALSAR-2 data cannot be retrieved for the requested grid."""


@dataclass
class SARProvenance:
    """Provenance record written alongside SAR channels (``sar_provenance.json``)."""

    sensor_source: str = "JAXA ALOS-2 PALSAR-2 annual 25 m mosaic (HH/HV)"
    stac_endpoint: str = PC_STAC_URL
    collection: str = PALSAR_COLLECTION
    year: int = 2021
    item_ids: list[str] = field(default_factory=list)
    calibration: str = "gamma0_dB = 10*log10(DN^2) - 83.0"
    aggregation: str = "linear-power mean of land/water pixels (layover/shadow/nodata excluded)"
    valid_fraction: float = 0.0
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def write(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def dn_to_linear_power(dn: np.ndarray) -> np.ndarray:
    """Convert JAXA mosaic DN to linear gamma0 power. DN <= 0 becomes NaN."""
    dn_f = np.asarray(dn, dtype=np.float64)
    lin = (dn_f**2) * (10.0 ** (PALSAR_CALIBRATION_FACTOR_DB / 10.0))
    return np.where(dn_f > 0, lin, np.nan).astype(np.float64)


def linear_power_to_db(lin: np.ndarray) -> np.ndarray:
    """Convert linear gamma0 power to dB, NaN where non-positive or non-finite."""
    lin_f = np.asarray(lin, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(lin_f)
    return np.where(np.isfinite(out) & (lin_f > 0), out, np.nan).astype(np.float32)


def dn_to_gamma0_db(dn: np.ndarray) -> np.ndarray:
    """Convert JAXA mosaic DN directly to gamma0 dB."""
    return linear_power_to_db(dn_to_linear_power(dn))


# ---------------------------------------------------------------------------
# Remote retrieval (Planetary Computer STAC, signed COG access)
# ---------------------------------------------------------------------------
def _search_palsar_items(bounds: Sequence[float], year: int) -> list[Any]:
    try:
        import planetary_computer
        import pystac_client
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SARDataUnavailableError(
            "pystac_client and planetary_computer are required for PALSAR-2 ingestion"
        ) from exc
    try:
        catalog = pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)
        search = catalog.search(
            collections=[PALSAR_COLLECTION],
            bbox=list(bounds),
            datetime=f"{year}-01-01/{year}-12-31",
        )
        items = list(search.items())
    except Exception as exc:  # pragma: no cover - network dependent
        raise SARDataUnavailableError(f"PALSAR-2 STAC search failed: {exc}") from exc
    # The collection holds both FNF-free mosaics; keep only items carrying HH/HV/mask
    items = [it for it in items if all(a in it.assets for a in ("HH", "HV", "mask"))]
    if not items:
        raise SARDataUnavailableError(f"No PALSAR-2 mosaic tiles found for bbox={bounds} year={year}")
    return items


def _accumulate_tile(
    item: Any,
    dst_transform: Any,
    dst_shape: tuple[int, int],
    sum_hh: np.ndarray,
    sum_hv: np.ndarray,
    count: np.ndarray,
    total: np.ndarray,
) -> None:
    """Read one 1x1 deg tile at a decimated overview level and accumulate onto the target grid.

    Each decimated source pixel is assigned to the target cell containing its
    centre (box aggregation). This is exact for coarse targets (target cell >=
    decimated pixel) and avoids any interpolation in the dB domain.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds as window_from_bounds

    H, W = dst_shape
    dst_res_x = abs(dst_transform.a)
    dst_res_y = abs(dst_transform.e)
    west = dst_transform.c
    north = dst_transform.f
    east = west + W * dst_res_x
    south = north - H * dst_res_y

    with rasterio.open(item.assets["HH"].href) as s_hh:
        tb = s_hh.bounds
        ix_w, ix_e = max(tb.left, west), min(tb.right, east)
        ix_s, ix_n = max(tb.bottom, south), min(tb.top, north)
        if ix_w >= ix_e or ix_s >= ix_n:
            return
        win = window_from_bounds(ix_w, ix_s, ix_e, ix_n, transform=s_hh.transform).round_offsets().round_lengths()
        if win.width <= 0 or win.height <= 0:
            return
        native = abs(s_hh.transform.a)
        # Decimate so the read pixel is <= half the target cell (>= 2x2 samples per cell)
        decim = max(1, int(np.floor(min(dst_res_x, dst_res_y) / native / 2.0)))
        out_h = max(1, int(win.height // decim))
        out_w = max(1, int(win.width // decim))
        hh = s_hh.read(1, window=win, out_shape=(out_h, out_w), resampling=Resampling.average)
        win_transform = s_hh.window_transform(win)

    with rasterio.open(item.assets["HV"].href) as s_hv:
        hv = s_hv.read(1, window=win, out_shape=(out_h, out_w), resampling=Resampling.average)
    with rasterio.open(item.assets["mask"].href) as s_mk:
        mk = s_mk.read(1, window=win, out_shape=(out_h, out_w), resampling=Resampling.nearest)

    px_w = win_transform.a * (win.width / out_w)
    px_h = win_transform.e * (win.height / out_h)
    cx = win_transform.c + (np.arange(out_w) + 0.5) * px_w
    cy = win_transform.f + (np.arange(out_h) + 0.5) * px_h
    dst_c = np.floor((cx - west) / dst_res_x).astype(np.int64)
    dst_r = np.floor((north - cy) / dst_res_y).astype(np.int64)
    rr, cc = np.meshgrid(dst_r, dst_c, indexing="ij")
    inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)

    valid = inside & np.isin(mk, VALID_MASK_CLASSES) & (hh > 0) & (hv > 0)
    flat_idx_all = (rr[inside] * W + cc[inside]).ravel()
    np.add.at(total.ravel(), flat_idx_all, 1)
    if not valid.any():
        return
    flat_idx = (rr[valid] * W + cc[valid]).ravel()
    np.add.at(sum_hh.ravel(), flat_idx, dn_to_linear_power(hh[valid]).ravel())
    np.add.at(sum_hv.ravel(), flat_idx, dn_to_linear_power(hv[valid]).ravel())
    np.add.at(count.ravel(), flat_idx, 1)


def fetch_palsar2_gamma0(
    dst_transform: Any,
    dst_shape: tuple[int, int],
    dst_crs: str = "EPSG:4326",
    year: int = 2021,
    cache_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, SARProvenance]:
    """Fetch real PALSAR-2 HH/HV gamma0 (dB) aggregated onto a target geographic grid.

    Args:
        dst_transform: Affine transform of the target grid (must be EPSG:4326 degrees).
        dst_shape: ``(H, W)`` of the target grid.
        dst_crs: Target CRS. Only ``EPSG:4326`` is supported (mosaic native CRS).
        year: Mosaic year (2015--2021 available on Planetary Computer).
        cache_dir: Optional directory; if given, results are cached as ``.npz``.

    Returns:
        ``(hh_db, hv_db, provenance)``. Arrays are float32 with ``NaN`` where no valid
        PALSAR-2 pixels fall within a cell.

    Raises:
        SARDataUnavailableError: When the data cannot be retrieved.
    """
    if str(dst_crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
        raise SARDataUnavailableError(f"PALSAR-2 aggregation requires EPSG:4326 target grid, got {dst_crs}")

    H, W = int(dst_shape[0]), int(dst_shape[1])
    west, north = float(dst_transform.c), float(dst_transform.f)
    east = west + W * abs(dst_transform.a)
    south = north - H * abs(dst_transform.e)
    bounds = (west, south, east, north)

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = f"palsar2_{year}_{west:.5f}_{south:.5f}_{east:.5f}_{north:.5f}_{H}x{W}"
        cache_path = cache_dir / f"{key}.npz"
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=False)
            prov = SARProvenance(**json.loads(str(z["provenance"])))
            logger.info("Loaded cached PALSAR-2 gamma0 grid from %s", cache_path)
            return z["hh_db"], z["hv_db"], prov

    items = _search_palsar_items(bounds, year)
    logger.info("Aggregating %d real PALSAR-2 mosaic tiles (year %d) onto %dx%d grid...", len(items), year, H, W)

    sum_hh = np.zeros((H, W), dtype=np.float64)
    sum_hv = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.int64)
    total = np.zeros((H, W), dtype=np.int64)
    used_ids: list[str] = []
    for i, item in enumerate(items, 1):
        try:
            _accumulate_tile(item, dst_transform, (H, W), sum_hh, sum_hv, count, total)
            used_ids.append(item.id)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Skipping PALSAR-2 tile %s: %s", item.id, exc)
        if i % 10 == 0 or i == len(items):
            logger.info("  ... %d/%d tiles aggregated", i, len(items))

    if not used_ids or count.sum() == 0:
        raise SARDataUnavailableError("All PALSAR-2 tile reads failed or produced no valid pixels")

    with np.errstate(invalid="ignore", divide="ignore"):
        hh_db = linear_power_to_db(np.where(count > 0, sum_hh / np.maximum(count, 1), np.nan))
        hv_db = linear_power_to_db(np.where(count > 0, sum_hv / np.maximum(count, 1), np.nan))

    prov = SARProvenance(year=year, item_ids=sorted(used_ids), valid_fraction=float(np.isfinite(hh_db).mean()))
    if cache_path is not None:
        np.savez_compressed(cache_path, hh_db=hh_db, hv_db=hv_db, provenance=json.dumps(prov.to_dict()))
    return hh_db, hv_db, prov


# ---------------------------------------------------------------------------
# Derived channels
# ---------------------------------------------------------------------------
def steger_ridge_strength(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Steger (1998) ridge strength from the dominant Hessian eigenvalue, normalized to [0, 1]."""
    img_f = np.asarray(image, dtype=np.float32)
    i_xx = gaussian_filter(img_f, sigma=sigma, order=[2, 0])
    i_yy = gaussian_filter(img_f, sigma=sigma, order=[0, 2])
    i_xy = gaussian_filter(img_f, sigma=sigma, order=[1, 1])
    trace = i_xx + i_yy
    det = i_xx * i_yy - i_xy**2
    sqrt_disc = np.sqrt(np.maximum(0.0, (trace / 2.0) ** 2 - det))
    ridge = np.maximum(np.abs(trace / 2.0 + sqrt_disc), np.abs(trace / 2.0 - sqrt_disc))
    return np.clip(ridge / (np.percentile(ridge, 99) + 1e-6), 0.0, 1.0).astype(np.float32)


def _fill_nearest(arr: np.ndarray) -> np.ndarray:
    """Nearest-valid fill used ONLY as filter support; results are re-masked afterwards."""
    a = np.asarray(arr, dtype=np.float32)
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    if bad.all():
        return np.zeros_like(a)
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def compute_palsar2_channels(
    hh_db: np.ndarray,
    hv_db: np.ndarray,
    dem: Optional[np.ndarray] = None,
    res_m: float = 250.0,
) -> dict[str, np.ndarray]:
    """Compute the 6 canonical SAR channels from real HH/HV gamma0 (dB).

    NaN cells in the inputs stay NaN in every output channel.
    """
    hh = np.asarray(hh_db, dtype=np.float32)
    hv = np.asarray(hv_db, dtype=np.float32)
    invalid = ~(np.isfinite(hh) & np.isfinite(hv))

    with np.errstate(over="ignore", invalid="ignore"):
        dpr = (10.0 ** (hv / 10.0)) / (10.0 ** (hh / 10.0) + 1e-6)

    hh_f = _fill_nearest(hh)
    hv_f = _fill_nearest(hv)
    lin_density = gaussian_filter(steger_ridge_strength(hh_f, sigma=2.0), sigma=4.0)

    from src.features.coal_and_multimodal_features import compute_sar_bedding_strike_and_anisotropy

    dem_arr = _fill_nearest(dem) if dem is not None else np.zeros_like(hh_f)
    mm = compute_sar_bedding_strike_and_anisotropy(hh_db=hh_f, hv_db=hv_f, dem=dem_arr, res_m=res_m)

    out = {
        "sar_palsar2_hh": hh,
        "sar_palsar2_hv": hv,
        "sar_palsar2_dpr": dpr,
        "sar_palsar2_lineament_density": lin_density,
        "sar_bedding_strike_deg": mm["sar_bedding_strike_deg"],
        "sar_glcm_anisotropy": mm["sar_glcm_anisotropy"],
    }
    return {k: np.where(invalid, np.nan, v).astype(np.float32) for k, v in out.items()}


def build_real_sar_channels(
    dst_transform: Any,
    dst_shape: tuple[int, int],
    dem: Optional[np.ndarray] = None,
    res_m: float = 250.0,
    dst_crs: str = "EPSG:4326",
    year: int = 2021,
    cache_dir: Optional[Path] = None,
) -> tuple[dict[str, np.ndarray], SARProvenance]:
    """Fetch real PALSAR-2 data and compute all canonical SAR channels for a grid."""
    hh_db, hv_db, prov = fetch_palsar2_gamma0(dst_transform, dst_shape, dst_crs=dst_crs, year=year, cache_dir=cache_dir)
    return compute_palsar2_channels(hh_db, hv_db, dem=dem, res_m=res_m), prov
