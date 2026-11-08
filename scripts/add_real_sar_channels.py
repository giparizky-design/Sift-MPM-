"""Add (or replace) real ALOS-2 PALSAR-2 SAR channels in a processed region dataset.

Fetches the genuine JAXA PALSAR-2 annual 25 m mosaic (Planetary Computer STAC),
aggregates it onto the region's grid (taken from ``dem_elevation.tif`` or
``feature_stack.joblib``), derives the 6 canonical SAR channels, and writes:

  - ``<data_dir>/sar_*.tif``               (6 float32 GeoTIFFs, NaN = no valid SAR)
  - ``<data_dir>/dataset.parquet``         (6 ``sar_*`` columns added/overwritten)
  - ``<data_dir>/feature_stack.joblib``    (channels appended/overwritten, if present)
  - ``<data_dir>/sar_provenance.json``     (tile IDs, year, calibration, valid fraction)

Any previously synthetic ``sar_*`` channels are overwritten. If real data cannot be
retrieved the script exits non-zero and leaves the dataset untouched -- it never
fabricates SAR values.

Usage:
    .venv/bin/python scripts/add_real_sar_channels.py --data-dir data/east_kalimantan/processed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.sar_palsar2 import (  # noqa: E402
    SAR_CHANNEL_NAMES,
    SARDataUnavailableError,
    build_real_sar_channels,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("add_real_sar_channels")


def _resolve_grid(data_dir: Path) -> tuple[Affine, tuple[int, int], str, np.ndarray | None]:
    """Return (transform, (H, W), crs, dem) for the dataset grid."""
    dem_tif = data_dir / "dem_elevation.tif"
    if dem_tif.exists():
        with rasterio.open(dem_tif) as src:
            dem = src.read(1).astype(np.float32)
            if src.nodata is not None:
                dem = np.where(dem == src.nodata, np.nan, dem)
            return src.transform, (src.height, src.width), str(src.crs), dem
    fs_path = data_dir / "feature_stack.joblib"
    if fs_path.exists():
        fs = joblib.load(fs_path)
        prof = dict(fs.profile)
        dem = None
        if "dem_elevation" in fs.feature_names:
            dem = fs.array[list(fs.feature_names).index("dem_elevation")].astype(np.float32)
        return prof["transform"], tuple(fs.array.shape[1:]), str(prof.get("crs", "EPSG:4326")), dem
    raise FileNotFoundError(f"No dem_elevation.tif or feature_stack.joblib in {data_dir}")


def _pixel_size_m(transform: Affine, crs: str, height: int) -> float:
    if "4326" in crs:
        lat_c = transform.f - 0.5 * height * abs(transform.e)
        return float(abs(transform.a) * 111_320.0 * np.cos(np.radians(lat_c)))
    return float(abs(transform.a))


def add_real_sar_channels(data_dir: Path, year: int = 2021, cache_dir: Path | None = None) -> dict:
    data_dir = Path(data_dir)
    transform, (H, W), crs, dem = _resolve_grid(data_dir)
    res_m = _pixel_size_m(transform, crs, H)
    logger.info("Grid %s: %dx%d, CRS=%s, ~%.0f m/px", data_dir, H, W, crs, res_m)

    channels, prov = build_real_sar_channels(
        dst_transform=transform,
        dst_shape=(H, W),
        dem=dem,
        res_m=res_m,
        dst_crs=crs,
        year=year,
        cache_dir=cache_dir or (data_dir.parent / "interim" / "palsar2_cache"),
    )

    # 1. GeoTIFFs
    profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": np.nan, "width": W, "height": H,
        "count": 1, "crs": crs, "transform": transform, "compress": "deflate",
    }
    for name in SAR_CHANNEL_NAMES:
        with rasterio.open(data_dir / f"{name}.tif", "w", **profile) as dst:
            dst.write(channels[name].astype(np.float32), 1)

    # 2. dataset.parquet
    pq_path = data_dir / "dataset.parquet"
    stats: dict[str, dict] = {}
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        rows = df["row"].to_numpy(dtype=np.int64)
        cols = df["col"].to_numpy(dtype=np.int64)
        replaced = [n for n in SAR_CHANNEL_NAMES if n in df.columns]
        for name in SAR_CHANNEL_NAMES:
            vals = channels[name][rows, cols].astype(np.float32)
            df[name] = vals
            stats[name] = {
                "nan_frac": float(np.isnan(vals).mean()),
                "median": float(np.nanmedian(vals)) if np.isfinite(vals).any() else None,
            }
        df.to_parquet(pq_path, index=False)
        logger.info(
            "Updated %s (%d rows): added %d SAR columns, overwrote %d previous (synthetic) columns",
            pq_path, len(df), len(SAR_CHANNEL_NAMES) - len(replaced), len(replaced),
        )

    # 3. feature_stack.joblib (keeps rebuild scripts such as build_east_kalimantan_250m.py consistent)
    fs_path = data_dir / "feature_stack.joblib"
    if fs_path.exists():
        fs = joblib.load(fs_path)
        names = list(fs.feature_names)
        arr = fs.array
        if tuple(arr.shape[1:]) == (H, W):
            extra = []
            for name in SAR_CHANNEL_NAMES:
                if name in names:
                    arr[names.index(name)] = channels[name]
                else:
                    extra.append(name)
            if extra:
                arr = np.concatenate([arr, np.stack([channels[n] for n in extra]).astype(arr.dtype)], axis=0)
                names.extend(extra)
            fs.array = arr
            fs.feature_names = names
            if isinstance(getattr(fs, "profile", None), dict):
                fs.profile["count"] = len(names)
            joblib.dump(fs, fs_path)
            logger.info("Updated %s -> %d channels", fs_path, len(names))
        else:
            logger.warning("feature_stack.joblib grid %s != raster grid %s; not updated", arr.shape[1:], (H, W))

    prov.write(data_dir / "sar_provenance.json")
    logger.info("Real PALSAR-2 channels written (%d tiles, valid fraction %.4f)", len(prov.item_ids), prov.valid_fraction)
    return {"provenance": prov.to_dict(), "parquet_stats": stats}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, nargs="+", help="processed dataset directory(ies)")
    ap.add_argument("--year", type=int, default=2021, help="PALSAR-2 mosaic year (2015-2021)")
    args = ap.parse_args()
    rc = 0
    for d in args.data_dir:
        try:
            add_real_sar_channels(d, year=args.year)
        except SARDataUnavailableError as exc:
            logger.error("Real SAR unavailable for %s: %s (dataset left unchanged)", d, exc)
            rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
