"""
src/utils/io.py
===============
Low-level I/O utilities for the Mineral Prospectivity Mapping (MPM) pipeline.

Role in the pipeline
--------------------
Every processing stage — harmonisation, feature engineering, model training,
and output map generation — reads from and writes to geospatial files.
This module provides a single, consistent interface so that the rest of the
codebase never calls rasterio or geopandas directly.  All rasters are written
as Cloud-Optimised GeoTIFFs (COGs) to allow efficient partial reads by
downstream consumers (e.g. tile servers, QGIS, Google Earth Engine ingestion).

Design decisions
----------------
- COG creation uses the standard two-step: write to an in-memory VRT / temp
  file then copy with ``copy_files`` using the COG driver options.  This
  guarantees valid block ordering without requiring GDAL ≥ 3.1's native COG
  driver, which may not be present in all environments.
- ``stack_rasters`` reprojects every input to the CRS / transform / shape of
  the *first* raster in the list before stacking, so the caller does not need
  to pre-align datasets.  A fill value (default NaN) is used for pixels that
  fall outside the extent of any source raster.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.shutil
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raster I/O
# ---------------------------------------------------------------------------

def load_raster(path: str | os.PathLike) -> tuple[np.ndarray, dict]:
    """Load a GeoTIFF (or any GDAL-readable raster) into memory.

    Args:
        path: Absolute or relative path to the raster file.

    Returns:
        A two-element tuple:

        - **data** (``np.ndarray``, shape ``(bands, height, width)``):
          All bands read into a float32 array.  Nodata pixels are replaced
          with ``np.nan`` when the dtype allows it; otherwise they are left
          at the rasterio nodata value.
        - **profile** (``dict``):
          The full rasterio dataset profile (crs, transform, dtype, nodata,
          width, height, count, driver …) that can be passed directly to
          ``save_raster`` or ``rasterio.open(..., **profile)``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        rasterio.errors.RasterioIOError: If the file cannot be opened.

    Example:
        >>> data, profile = load_raster("dem_50m.tif")
        >>> print(data.shape, profile["crs"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as src:
        profile = src.profile.copy()
        data = src.read()  # shape: (bands, height, width)
        nodata = src.nodata

    # Convert integer types to float32 so we can store NaN for nodata.
    if np.issubdtype(data.dtype, np.integer) or data.dtype == np.float16:
        data = data.astype(np.float32)

    if nodata is not None and np.issubdtype(data.dtype, np.floating):
        data[data == nodata] = np.nan

    logger.debug("Loaded raster %s  shape=%s  crs=%s", path, data.shape, profile.get("crs"))
    return data, profile


def save_raster(
    data: np.ndarray,
    profile: dict,
    path: str | os.PathLike,
) -> None:
    """Save a numpy array as a Cloud-Optimised GeoTIFF (COG).

    The output uses DEFLATE compression with 512×512 tiles — a widely
    accepted balance between compression ratio, random-access efficiency, and
    compatibility with web-map tile servers.

    Internally the function writes to a temporary GeoTIFF first, then uses
    ``rasterio.shutil.copy`` with ``copy_src_overviews=True`` to produce the
    final COG with correct internal block ordering.  Overview levels
    (2, 4, 8, 16, 32) are built automatically so the file is immediately
    usable in multi-resolution viewers.

    Args:
        data: Array of shape ``(bands, height, width)`` **or**
              ``(height, width)`` for single-band output.
        profile: Rasterio profile dict describing CRS, transform, nodata,
                 dtype, etc.  The driver, compression, tiling, and interleave
                 keys are overridden by this function.
        path: Destination path (parent directories are created if absent).

    Raises:
        ValueError: If *data* has more than 3 dimensions or the band count
                    does not match the profile.

    Example:
        >>> data, profile = load_raster("dem_50m.tif")
        >>> save_raster(data * 2, profile, "dem_50m_doubled.tif")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalise shape to (bands, height, width)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim != 3:
        raise ValueError(
            f"data must be 2-D or 3-D, got shape {data.shape}"
        )

    bands, height, width = data.shape

    # Replace NaN with nodata value for integer output dtypes
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=bands,
        height=height,
        width=width,
        dtype=str(data.dtype),
    )

    # Ensure nodata is compatible with output dtype
    nodata = out_profile.get("nodata")
    if nodata is not None:
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            if nodata < info.min or nodata > info.max:
                out_profile["nodata"] = None
                nodata = None
            else:
                out_profile["nodata"] = int(nodata)
                nodata = int(nodata)
        elif data.dtype == bool:
            out_profile["nodata"] = None
            nodata = None

    # Handle NaN → nodata substitution
    write_data = data.copy()
    if np.issubdtype(data.dtype, np.floating) and nodata is not None:
        write_data[np.isnan(write_data)] = nodata

    # Step 1: Write to a temporary unoptimised GeoTIFF
    tmp_profile = out_profile.copy()
    tmp_profile.update(tiled=False, compress=None, interleave="band")
    tmp_profile.pop("blockxsize", None)
    tmp_profile.pop("blockysize", None)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        with rasterio.open(tmp_path, "w", **tmp_profile) as tmp_ds:
            tmp_ds.write(write_data)

            # Build overviews (required for valid COG) if grid is large enough
            if width >= 64 and height >= 64:
                overview_levels = [2, 4, 8]
                tmp_ds.build_overviews(overview_levels, Resampling.average)
                tmp_ds.update_tags(ns="rio_overview", resampling="average")

        # Step 2: Copy to final COG with correct block ordering
        block_size = min(256, width, height)
        block_size = max(16, (block_size // 16) * 16)
        use_tiling = (width >= 64 and height >= 64)

        cog_profile = {
            "driver": "GTiff",
            "compress": "deflate",
            "tiled": use_tiling,
            "copy_src_overviews": use_tiling,
        }
        if use_tiling:
            cog_profile["blockxsize"] = block_size
            cog_profile["blockysize"] = block_size

        with rasterio.open(tmp_path) as tmp_ds:
            rasterio.shutil.copy(tmp_ds, str(path), **cog_profile)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    logger.info("Saved COG raster → %s  shape=%s", path, data.shape)


# ---------------------------------------------------------------------------
# Vector I/O
# ---------------------------------------------------------------------------

def load_vector(
    path: str | os.PathLike,
    crs: str | CRS | None = None,
) -> gpd.GeoDataFrame:
    """Load any OGR-supported vector file into a GeoDataFrame.

    Args:
        path: Path to the vector file (GeoPackage, Shapefile, GeoJSON, etc.).
        crs: Optional target CRS.  If provided and different from the file's
             native CRS, the GeoDataFrame is reprojected before returning.
             Accepts any form understood by ``pyproj.CRS.from_user_input``,
             e.g. ``"EPSG:3577"`` (GDA94 / Australian Albers).

    Returns:
        A ``gpd.GeoDataFrame`` in the requested CRS (or original CRS if
        *crs* is ``None``).

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file has no associated CRS and *crs* is not
                    provided (cannot reproject from unknown CRS).

    Example:
        >>> gdf = load_vector("deposits.gpkg", crs="EPSG:3577")
        >>> gdf.crs.to_epsg()
        3577
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    gdf = gpd.read_file(path)
    logger.debug("Loaded vector %s  rows=%d  crs=%s", path, len(gdf), gdf.crs)

    if crs is not None:
        target_crs = CRS.from_user_input(crs)
        if gdf.crs is None:
            raise ValueError(
                f"Vector file {path} has no CRS; cannot reproject to {crs}."
            )
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
            logger.debug("Reprojected vector to %s", target_crs)

    return gdf


def save_vector(gdf: gpd.GeoDataFrame, path: str | os.PathLike) -> None:
    """Save a GeoDataFrame as a GeoPackage (.gpkg).

    GeoPackage is the OGC standard container format: it is a single-file
    SQLite database with no per-column length limits (unlike Shapefile) and
    supports mixed geometry types and UTF-8 attribute names.

    Args:
        gdf: GeoDataFrame to save.
        path: Destination path.  The ``.gpkg`` extension is appended if
              absent.  Parent directories are created as needed.

    Example:
        >>> save_vector(predictions_gdf, "outputs/predictions.gpkg")
    """
    path = Path(path)
    if path.suffix.lower() != ".gpkg":
        path = path.with_suffix(".gpkg")
    path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(str(path), driver="GPKG")
    logger.info("Saved vector → %s  rows=%d", path, len(gdf))


# ---------------------------------------------------------------------------
# Raster stacking
# ---------------------------------------------------------------------------

def stack_rasters(
    paths: Sequence[str | os.PathLike],
    fill_value: float = np.nan,
) -> tuple[np.ndarray, dict]:
    """Align and stack multiple GeoTIFFs into a single C×H×W array.

    In MPM workflows, predictor layers (geophysics, geochemistry, remote
    sensing) arrive in different projections, resolutions, and extents.
    This function reprojects every raster to the CRS, transform, and shape
    of the *first* path in the list (the "reference" grid) before stacking.
    This design matches the ``reproject_raster`` philosophy in
    ``harmonization.reproject``: the reference grid should be the 50 m
    pilot grid profile built by ``build_pilot_grid``.

    Multi-band source rasters are fully included (each band becomes a
    slice in the output stack), so the output band count equals the sum of
    all bands across all input files.

    Args:
        paths: Ordered sequence of paths to GeoTIFF files.  The first path
               defines the reference CRS, resolution, and extent.
        fill_value: Value used for pixels that fall outside the extent of a
                    source raster after reprojection.  Defaults to ``np.nan``
                    for easy masking with ``np.nanmean`` etc.

    Returns:
        A two-element tuple:

        - **stack** (``np.ndarray``, shape ``(C, H, W)``, dtype float32):
          Stacked predictor cube in reference-grid coordinates.
        - **profile** (``dict``):
          Rasterio profile of the reference grid (suitable for
          ``save_raster``).

    Raises:
        ValueError: If *paths* is empty.
        FileNotFoundError: If any path does not exist.

    Example:
        >>> stack, profile = stack_rasters(["dem.tif", "aeromag.tif", "ndvi.tif"])
        >>> print(stack.shape)  # e.g. (3, 2000, 2000)
    """
    if not paths:
        raise ValueError("paths must contain at least one raster path.")

    paths = [Path(p) for p in paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Raster not found: {p}")

    # ----- Reference grid from first raster --------------------------------
    with rasterio.open(paths[0]) as ref_ds:
        ref_profile = ref_ds.profile.copy()
        ref_crs = ref_ds.crs
        ref_transform = ref_ds.transform
        ref_height = ref_ds.height
        ref_width = ref_ds.width
        ref_nodata = ref_ds.nodata if ref_ds.nodata is not None else fill_value

    bands_list: list[np.ndarray] = []

    for p in paths:
        with rasterio.open(p) as src:
            src_count = src.count

            if (
                src.crs == ref_crs
                and src.transform == ref_transform
                and src.height == ref_height
                and src.width == ref_width
            ):
                # Already on the reference grid — read directly
                layer = src.read().astype(np.float32)
                # Replace nodata with fill_value
                if src.nodata is not None:
                    layer[layer == src.nodata] = fill_value
            else:
                # Reproject to reference grid
                logger.debug("Reprojecting %s → reference grid", p)
                layer = np.full(
                    (src_count, ref_height, ref_width),
                    fill_value,
                    dtype=np.float32,
                )
                for band_idx in range(1, src_count + 1):
                    src_data = src.read(band_idx).astype(np.float32)
                    if src.nodata is not None:
                        src_data[src_data == src.nodata] = fill_value

                    reproject(
                        source=src_data,
                        destination=layer[band_idx - 1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=fill_value,
                        dst_nodata=fill_value,
                    )

            bands_list.append(layer)

    stack = np.concatenate(bands_list, axis=0)  # (C, H, W)

    # Build output profile
    out_profile = ref_profile.copy()
    out_profile.update(
        count=stack.shape[0],
        dtype="float32",
        nodata=fill_value,
    )

    logger.info(
        "Stacked %d files → shape %s  fill_value=%s",
        len(paths),
        stack.shape,
        fill_value,
    )
    return stack, out_profile
