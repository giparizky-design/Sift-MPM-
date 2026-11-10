"""Shared plumbing for real (observed) input layers.

Every layer that reaches a training cube must come from a named, retrievable data
product. This module provides:

* :class:`GridSpec` - the target grid (CRS, affine transform, shape).
* :class:`LayerProvenance` / :class:`ProvenanceManifest` - a sidecar record of where
  every feature column came from. Training refuses columns without one.
* :class:`RealDataUnavailableError` - raised when a source cannot be retrieved. Callers
  must then omit the layer. They must never substitute generated values.
* Helpers to read Cloud-Optimised GeoTIFFs at a decimated overview level and
  reproject them onto the target grid, and a tile-wise median compositor for STAC
  image collections.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds

logger = logging.getLogger(__name__)

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# GDAL settings for remote COG access
GDAL_REMOTE_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "VSI_CACHE": "TRUE",
}


class RealDataUnavailableError(RuntimeError):
    """A real data source could not be retrieved. The layer must be omitted, never faked."""


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridSpec:
    """Target raster grid."""

    crs: str
    transform: Affine
    height: int
    width: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) in the grid CRS."""
        west, south, east, north = array_bounds(self.height, self.width, self.transform)
        return (west, south, east, north)

    def bounds_4326(self) -> tuple[float, float, float, float]:
        if str(self.crs).upper() in ("EPSG:4326", "OGC:CRS84"):
            return self.bounds
        return tuple(transform_bounds(self.crs, "EPSG:4326", *self.bounds, densify_pts=21))  # type: ignore[return-value]

    @property
    def res_m(self) -> float:
        """Approximate pixel size in metres (uses mid-latitude for geographic grids)."""
        px = abs(self.transform.a)
        if str(self.crs).upper() in ("EPSG:4326", "OGC:CRS84"):
            mid_lat = 0.5 * (self.bounds[1] + self.bounds[3])
            return px * 111_320.0 * math.cos(math.radians(mid_lat))
        return px

    def profile(self, dtype: str = "float32", nodata: float = -9999.0) -> dict[str, Any]:
        return {
            "driver": "GTiff",
            "dtype": dtype,
            "nodata": nodata,
            "width": self.width,
            "height": self.height,
            "count": 1,
            "crs": self.crs,
            "transform": self.transform,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }

    def window_for_bounds_4326(self, bbox: Sequence[float], pad_px: int = 2) -> Optional[Window]:
        """Grid window covering a lon/lat bbox, clipped to the grid, or None if disjoint."""
        if str(self.crs).upper() in ("EPSG:4326", "OGC:CRS84"):
            b = bbox
        else:
            b = transform_bounds("EPSG:4326", self.crs, *bbox, densify_pts=21)
        win = window_from_bounds(*b, transform=self.transform)
        r0 = max(0, int(math.floor(win.row_off)) - pad_px)
        c0 = max(0, int(math.floor(win.col_off)) - pad_px)
        r1 = min(self.height, int(math.ceil(win.row_off + win.height)) + pad_px)
        c1 = min(self.width, int(math.ceil(win.col_off + win.width)) + pad_px)
        if r1 <= r0 or c1 <= c0:
            return None
        return Window(c0, r0, c1 - c0, r1 - r0)

    def window_transform(self, win: Window) -> Affine:
        return rasterio.windows.transform(win, self.transform)

    def to_dict(self) -> dict[str, Any]:
        return {"crs": str(self.crs), "transform": list(self.transform)[:6], "height": self.height, "width": self.width}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
MEASURED = "measured"  # resampled observation from a named product
DERIVED = "derived"  # deterministic function of other real layers, documented formula


@dataclass
class LayerProvenance:
    name: str
    kind: str  # MEASURED or DERIVED
    source: str  # product name / citation
    url: str = ""
    inputs: list[str] = field(default_factory=list)
    method: str = ""
    native_resolution_m: Optional[float] = None
    temporal_coverage: str = ""
    item_ids: list[str] = field(default_factory=list)
    valid_fraction: Optional[float] = None
    notes: str = ""
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if self.kind not in (MEASURED, DERIVED):
            raise ValueError(f"LayerProvenance.kind must be '{MEASURED}' or '{DERIVED}', got {self.kind!r}")
        if not self.source:
            raise ValueError(f"LayerProvenance for {self.name!r} needs a source")


class ProvenanceManifest:
    """Per-dataset record of every layer's origin (``layer_provenance.json``)."""

    FILENAME = "layer_provenance.json"

    def __init__(self, grid: Optional[GridSpec] = None) -> None:
        self.grid = grid
        self.layers: dict[str, LayerProvenance] = {}

    def add(self, prov: LayerProvenance) -> None:
        self.layers[prov.name] = prov

    def derive(self, name: str, inputs: Iterable[str], method: str, source: str = "", notes: str = "") -> None:
        """Register a layer derived from other registered real layers."""
        inputs = list(inputs)
        missing = [i for i in inputs if i not in self.layers]
        if missing:
            raise KeyError(f"Derived layer {name!r} depends on unregistered inputs {missing}")
        src = source or "derived from: " + ", ".join(sorted({self.layers[i].source for i in inputs}))
        self.add(LayerProvenance(name=name, kind=DERIVED, source=src, inputs=inputs, method=method, notes=notes))

    def update_valid_fraction(self, name: str, arr: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        if name in self.layers:
            a = arr if mask is None else arr[mask]
            self.layers[name].valid_fraction = float(np.isfinite(a).mean()) if a.size else 0.0

    def missing(self, columns: Iterable[str]) -> list[str]:
        return [c for c in columns if c not in self.layers]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "geospatials.layer_provenance/1",
            "grid": self.grid.to_dict() if self.grid else None,
            "policy": "Every layer is a real observation or a documented function of real observations. "
            "No generated, random or placeholder values. Missing data are NaN.",
            "layers": {k: asdict(v) for k, v in sorted(self.layers.items())},
        }

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / self.FILENAME
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def read(cls, data_dir: Path) -> Optional["ProvenanceManifest"]:
        p = Path(data_dir) / cls.FILENAME
        if not p.exists():
            return None
        raw = json.loads(p.read_text(encoding="utf-8"))
        m = cls()
        for name, d in raw.get("layers", {}).items():
            m.layers[name] = LayerProvenance(**d)
        return m


# ---------------------------------------------------------------------------
# Remote COG reading
# ---------------------------------------------------------------------------
def decimation_factor(native_res: float, target_res: float) -> int:
    """Largest power of two f such that native_res * f <= target_res (>= 1)."""
    f = 1
    while native_res * f * 2 <= target_res * 1.0001:
        f *= 2
    return f


def read_cog_decimated(
    href: str,
    target_res: float,
    bands: Sequence[int] = (1,),
    resampling: Resampling = Resampling.average,
) -> tuple[np.ndarray, Affine, Any, Optional[float]]:
    """Read a (remote) COG at the coarsest overview finer than ``target_res``.

    Returns (array (B, h, w) float32 with nodata -> NaN, transform, crs, native_res).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(4):
        try:
            with rasterio.Env(**GDAL_REMOTE_ENV):
                with rasterio.open(href) as src:
                    native = abs(src.res[0])
                    f = decimation_factor(native, target_res)
                    out_h = max(1, int(math.ceil(src.height / f)))
                    out_w = max(1, int(math.ceil(src.width / f)))
                    arr = src.read(list(bands), out_shape=(len(bands), out_h, out_w), resampling=resampling).astype(np.float32)
                    nod = src.nodata
                    if nod is not None:
                        arr[arr == nod] = np.nan
                    tr = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
                    return arr, tr, src.crs, native
        except Exception as exc:  # network hiccups / expired signature
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    raise RealDataUnavailableError(f"Failed to read {href.split('?')[0]}: {last_exc}")


def reproject_into(
    src_arr: np.ndarray,
    src_transform: Affine,
    src_crs: Any,
    grid: GridSpec,
    win: Optional[Window] = None,
    resampling: Resampling = Resampling.average,
) -> np.ndarray:
    """Reproject a 2-D float array (NaN = nodata) onto the grid (or a window of it)."""
    if win is None:
        dst_shape = grid.shape
        dst_tr = grid.transform
    else:
        dst_shape = (int(win.height), int(win.width))
        dst_tr = grid.window_transform(win)
    dst = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=np.asarray(src_arr, dtype=np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_tr,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


def warp_file_to_grid(path: str | Path, grid: GridSpec, band: int = 1, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Warp a local/remote raster band onto the grid (nodata -> NaN)."""
    with rasterio.Env(**GDAL_REMOTE_ENV):
        with rasterio.open(path) as src:
            dst = np.full(grid.shape, np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, band),
                destination=dst,
                src_nodata=src.nodata,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )
    return dst


# ---------------------------------------------------------------------------
# STAC search + tile-wise median compositing
# ---------------------------------------------------------------------------
def stac_search_chunked(
    collection: str,
    bbox: Sequence[float],
    datetime_range: str,
    query: Optional[dict] = None,
    chunk_deg: float = 2.0,
    max_items_per_chunk: int = 5000,
) -> list[Any]:
    """Search a Planetary Computer collection over a large bbox in chunks (unsigned items)."""
    import pystac_client

    cat = pystac_client.Client.open(PC_STAC_URL)
    west, south, east, north = bbox
    seen: dict[str, Any] = {}
    lon = west
    while lon < east:
        lat = south
        while lat < north:
            sub = [lon, lat, min(east, lon + chunk_deg), min(north, lat + chunk_deg)]
            for attempt in range(4):
                try:
                    res = cat.search(collections=[collection], bbox=sub, datetime=datetime_range, query=query or {}, max_items=max_items_per_chunk)
                    for it in res.items():
                        seen.setdefault(it.id, it)
                    break
                except Exception as exc:
                    logger.warning("STAC search retry %d for %s %s: %s", attempt + 1, collection, sub, exc)
                    time.sleep(5.0 * (attempt + 1))
            else:
                raise RealDataUnavailableError(f"STAC search failed for {collection} {sub}")
            lat += chunk_deg
        lon += chunk_deg
    logger.info("STAC %s: %d unique items over %s (%s)", collection, len(seen), list(bbox), datetime_range)
    return list(seen.values())


def sign_href(href: str) -> str:
    import planetary_computer as pc

    return pc.sign(href)


class MedianMosaic:
    """Accumulates per-group (tile) median composites into a grid-wide mean mosaic."""

    def __init__(self, grid: GridSpec, names: Sequence[str]) -> None:
        self.grid = grid
        self.names = list(names)
        self.sum = {n: np.zeros(grid.shape, dtype=np.float64) for n in self.names}
        self.cnt = {n: np.zeros(grid.shape, dtype=np.uint16) for n in self.names}
        self.obs = np.zeros(grid.shape, dtype=np.uint16)  # clear observations behind each composite pixel

    def add_group(self, win: Window, stacks: dict[str, list[np.ndarray]]) -> None:
        r0, c0 = int(win.row_off), int(win.col_off)
        r1, c1 = r0 + int(win.height), c0 + int(win.width)
        first = self.names[0]
        if not stacks.get(first):
            return
        cube0 = np.stack(stacks[first], axis=0)
        n_obs = np.isfinite(cube0).sum(axis=0).astype(np.uint16)
        self.obs[r0:r1, c0:c1] = np.maximum(self.obs[r0:r1, c0:c1], n_obs)
        for n in self.names:
            if not stacks.get(n):
                continue
            cube = np.stack(stacks[n], axis=0)
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(cube, axis=0)
            ok = np.isfinite(med)
            self.sum[n][r0:r1, c0:c1][ok] += med[ok]
            self.cnt[n][r0:r1, c0:c1][ok] += 1

    def result(self) -> dict[str, np.ndarray]:
        out = {}
        for n in self.names:
            with np.errstate(invalid="ignore", divide="ignore"):
                out[n] = np.where(self.cnt[n] > 0, self.sum[n] / np.maximum(self.cnt[n], 1), np.nan).astype(np.float32)
        return out


def composite_items_by_group(
    grid: GridSpec,
    groups: dict[str, list[Any]],
    read_item: Callable[[Any, GridSpec, Window], Optional[dict[str, np.ndarray]]],
    names: Sequence[str],
    max_workers: int = 16,
    progress_label: str = "composite",
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Median-composite items within each group (e.g. MGRS tile), then mean-mosaic groups.

    ``read_item(item, grid, window)`` returns per-name arrays shaped like ``window``
    (NaN where masked) or None if the item is unusable.
    """
    mosaic = MedianMosaic(grid, names)
    used_ids: list[str] = []
    n_groups = len(groups)
    t0 = time.time()
    for gi, (gkey, items) in enumerate(sorted(groups.items())):
        bboxes = np.array([it.bbox for it in items], dtype=np.float64)
        gb = (bboxes[:, 0].min(), bboxes[:, 1].min(), bboxes[:, 2].max(), bboxes[:, 3].max())
        win = grid.window_for_bounds_4326(gb)
        if win is None:
            continue
        stacks: dict[str, list[np.ndarray]] = {n: [] for n in names}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(read_item, it, grid, win): it for it in items}
            for fut in as_completed(futs):
                it = futs[fut]
                try:
                    res = fut.result()
                except RealDataUnavailableError as exc:
                    logger.warning("[%s] skipping %s: %s", progress_label, it.id, exc)
                    continue
                if res is None:
                    continue
                for n in names:
                    if n in res:
                        stacks[n].append(res[n])
                used_ids.append(it.id)
        mosaic.add_group(win, stacks)
        if (gi + 1) % 5 == 0 or gi + 1 == n_groups:
            logger.info("[%s] %d/%d groups composited (%.1f min)", progress_label, gi + 1, n_groups, (time.time() - t0) / 60.0)
    return mosaic.result(), mosaic.obs, sorted(used_ids)


def save_layer(arr: np.ndarray, grid: GridSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = grid.profile()
    out = np.where(np.isfinite(arr), arr, prof["nodata"]).astype(np.float32)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


def load_layer(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    return a


def cached_layers(cache_dir: Path, names: Sequence[str], grid: GridSpec) -> Optional[dict[str, np.ndarray]]:
    """Return cached grid layers if all exist and match the grid shape."""
    out = {}
    for n in names:
        p = cache_dir / f"{n}.tif"
        if not p.exists():
            return None
        with rasterio.open(p) as src:
            if (src.height, src.width) != grid.shape or not np.allclose(list(src.transform)[:6], list(grid.transform)[:6]):
                return None
        out[n] = load_layer(p)
    return out


def download(url: str, dest: Path, min_bytes: int = 1024) -> Path:
    """Download a public file once (resumable-safe via temp file)."""
    import urllib.request

    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s -> %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "geospatials-research-pipeline"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as exc:
        raise RealDataUnavailableError(f"Download failed for {url}: {exc}") from exc
    os.replace(tmp, dest)
    return dest


# ---------------------------------------------------------------------------
# Training-time gate
# ---------------------------------------------------------------------------
ALLOW_UNVERIFIED_ENV = "MPM_ALLOW_UNVERIFIED_LAYERS"


def assert_feature_provenance(data_dir: Path, feature_cols: Iterable[str]) -> None:
    """Refuse to train or evaluate on columns that have no real-data provenance record.

    A dataset directory must carry ``layer_provenance.json`` covering every feature column.
    Setting ``MPM_ALLOW_UNVERIFIED_LAYERS=1`` downgrades this to a warning (legacy datasets
    such as the pre-v4.9 ``data/kalimantan/processed`` contain fabricated layers).
    """
    feature_cols = list(feature_cols)
    allow = os.environ.get(ALLOW_UNVERIFIED_ENV, "").strip() in ("1", "true", "yes")
    man = ProvenanceManifest.read(Path(data_dir))
    if man is None:
        msg = (f"{data_dir} has no {ProvenanceManifest.FILENAME}: its layers cannot be shown to be real. "
               f"Build cubes with scripts/build_real_feature_cubes.py, or set {ALLOW_UNVERIFIED_ENV}=1 to override.")
        if allow:
            logger.warning("UNVERIFIED LAYERS: %s", msg)
            return
        raise RealDataUnavailableError(msg)
    missing = man.missing(feature_cols)
    if missing:
        msg = f"{len(missing)} feature column(s) in {data_dir} lack provenance records: {missing}"
        if allow:
            logger.warning("UNVERIFIED LAYERS: %s", msg)
            return
        raise RealDataUnavailableError(msg)
    logger.info("Provenance gate passed: all %d feature columns in %s are real or derived from real layers", len(feature_cols), data_dir)
