"""
src/harmonization/reproject.py
================================
Reprojection and resampling helpers that bring every predictor layer onto
the 50 m equal-area pilot grid.

Geoscientific rationale
-----------------------
Mineral prospectivity models require all predictor layers to share an
identical spatial reference: same coordinate reference system (CRS), pixel
size, extent, and grid alignment.  Even a sub-pixel shift between layers
can introduce spurious spatial correlations or mask genuine ones — especially
important for structural geology proxies derived from subtle magnetic or
gravity gradients.

For Australian MPM work the canonical equal-area projection is **GDA2020 /
Australian Albers (EPSG:9473)** or its GDA94 predecessor **(EPSG:3577)**,
both using a pixel size of 50 m.  Fifty metres is fine enough to resolve
most lithostructural contacts visible in aeromagnetics yet coarse enough to
keep the predictor cube manageable for model training.

This module:

1. Computes a rasterio profile (the "pilot grid") from a config dict.
2. Reprojects any raster file to that profile.
3. Rasterises vector attributes onto the pilot grid.
4. Validates that a set of raster files all share the same grid.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject as warp_reproject, calculate_default_transform

from src.utils.io import load_raster, save_raster

logger = logging.getLogger(__name__)

# Mapping from user-friendly string names to rasterio Resampling members.
_RESAMPLING_MAP: dict[str, Resampling] = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "gauss": Resampling.gauss,
    "rms": Resampling.rms,
}


# ---------------------------------------------------------------------------
# 1. Build pilot grid profile
# ---------------------------------------------------------------------------

def build_pilot_grid(config: dict) -> dict:
    """Compute the rasterio profile for the 50 m equal-area pilot grid.

    Geoscientific rationale
    -----------------------
    All predictor rasters must share a common spatial reference.  This
    function translates a human-readable area-of-interest (AOI) config into
    a ``rasterio``-compatible profile dict that can be passed directly to
    ``rasterio.open`` or used as the ``target_profile`` in
    :func:`reproject_raster`.

    The pixel size is fixed at 50 m in projected (linear) units, which
    matches GA's standard aeromagnetics and radiometric grids and allows
    meaningful structural interpretation without excessive data volume.

    Args:
        config: Dictionary with the following keys:

            - ``aoi_bounds_geo`` (``tuple[float, float, float, float]``):
              AOI bounding box **in the target CRS** (left, bottom, right,
              top), in projected metres.
            - ``crs`` (``str | int``):
              Target CRS, e.g. ``"EPSG:3577"`` or ``3577``.
            - ``pixel_size`` (``float``, optional):
              Pixel size in metres.  Defaults to ``50.0`` if absent.
            - ``nodata`` (``float``, optional):
              NoData sentinel value.  Defaults to ``-9999.0``.

    Returns:
        A ``rasterio`` profile dict with keys: ``driver``, ``dtype``,
        ``nodata``, ``crs``, ``transform``, ``width``, ``height``,
        ``count``, ``compress``, ``tiled``, ``blockxsize``,
        ``blockysize``.

    Raises:
        KeyError: If ``aoi_bounds_geo`` or ``crs`` are missing from config.

    Example:
        >>> config = {
        ...     "aoi_bounds_geo": (1_200_000, -3_900_000, 1_600_000, -3_500_000),
        ...     "crs": "EPSG:3577",
        ... }
        >>> profile = build_pilot_grid(config)
        >>> profile["width"], profile["height"]
        (8000, 8000)
    """
    bounds = config.get("aoi_bounds_proj") or config.get("aoi_bounds_geo")
    if bounds is None:
        raise KeyError("config must contain 'aoi_bounds_proj' or 'aoi_bounds_geo'")

    crs_input = config["crs"]
    pixel_size: float = float(config.get("pixel_size", config.get("cell_size_m", 50.0)))
    nodata: float = float(config.get("nodata", -9999.0))

    left, bottom, right, top = bounds
    # Auto-project if bounds appear to be in geographic degrees (WGS84/GDA2020 geographic)
    if abs(left) <= 360 and abs(right) <= 360 and abs(bottom) <= 90 and abs(top) <= 90:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", crs_input, always_xy=True)
        left, bottom = transformer.transform(left, bottom)
        right, top = transformer.transform(right, top)

    if left >= right:
        raise ValueError(f"AOI left ({left}) must be < right ({right}).")
    if bottom >= top:
        raise ValueError(f"AOI bottom ({bottom}) must be < top ({top}).")

    width = int(round((right - left) / pixel_size))
    height = int(round((top - bottom) / pixel_size))

    # rasterio transforms follow the convention: origin at top-left corner,
    # y increases downward → negative y pixel size.
    transform = from_bounds(left, bottom, right, top, width, height)

    crs = CRS.from_user_input(crs_input)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": nodata,
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "count": 1,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    logger.info(
        "Pilot grid: CRS=%s  pixel=%.1fm  size=%dx%d  bounds=%s",
        crs.to_epsg() or crs.to_string(),
        pixel_size,
        width,
        height,
        bounds,
    )
    return profile


# ---------------------------------------------------------------------------
# 2. Reproject a raster file
# ---------------------------------------------------------------------------

def reproject_raster(
    src_path: str | os.PathLike,
    dst_path: str | os.PathLike,
    target_profile: dict,
    resampling_method: str = "bilinear",
) -> None:
    """Reproject a raster file to the target CRS and resolution.

    Geoscientific rationale
    -----------------------
    Continuous geophysical fields (magnetics, radiometrics, gravity) are
    best resampled with **bilinear** interpolation, which preserves the
    smooth character of potential-field data without introducing the ringing
    artefacts of higher-order methods.

    Categorical or classified rasters (geology, regolith class) must use
    **nearest-neighbour** to avoid interpolating across category boundaries
    — a bilinear blend of granite (1) and basalt (2) is geologically
    meaningless.

    Args:
        src_path: Path to the source GeoTIFF.
        dst_path: Destination path for the reprojected output COG.
        target_profile: Rasterio profile defining the target CRS, transform,
                        width, and height (as returned by :func:`build_pilot_grid`).
        resampling_method: Resampling algorithm name.  Must be one of:
                           ``"nearest"``, ``"bilinear"``, ``"cubic"``,
                           ``"cubic_spline"``, ``"lanczos"``, ``"average"``,
                           ``"mode"``, ``"gauss"``, ``"rms"``.

    Raises:
        FileNotFoundError: If *src_path* does not exist.
        ValueError: If *resampling_method* is not recognised.
        rasterio.errors.RasterioIOError: If the source file cannot be opened.

    Example:
        >>> reproject_raster(
        ...     "raw/dem_gda94.tif",
        ...     "harmonised/dem_50m.tif",
        ...     target_profile,
        ...     resampling_method="bilinear",
        ... )
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    if not src_path.exists():
        raise FileNotFoundError(f"Source raster not found: {src_path}")

    if resampling_method not in _RESAMPLING_MAP:
        raise ValueError(
            f"Unknown resampling method '{resampling_method}'. "
            f"Choose from: {sorted(_RESAMPLING_MAP)}"
        )
    resampling = _RESAMPLING_MAP[resampling_method]

    dst_crs: CRS = target_profile["crs"]
    dst_transform = target_profile["transform"]
    dst_width: int = target_profile["width"]
    dst_height: int = target_profile["height"]
    dst_nodata = target_profile.get("nodata", -9999.0)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        band_count = src.count

        # Allocate output array
        dst_data = np.full(
            (band_count, dst_height, dst_width),
            dst_nodata,
            dtype=np.float32,
        )

        for band_idx in range(1, band_count + 1):
            src_band = src.read(band_idx).astype(np.float32)

            warp_reproject(
                source=src_band,
                destination=dst_data[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                src_nodata=src_nodata,
                dst_nodata=dst_nodata,
            )

    out_profile = target_profile.copy()
    out_profile.update(count=band_count, dtype="float32", nodata=dst_nodata)

    # Replace sentinel nodata with NaN for save_raster (COG)
    dst_data_nan = dst_data.astype(np.float32)
    dst_data_nan[dst_data_nan == dst_nodata] = np.nan

    save_raster(dst_data_nan, out_profile, dst_path)

    logger.info(
        "Reprojected %s → %s  method=%s  shape=(%d, %d)",
        src_path.name,
        dst_path.name,
        resampling_method,
        dst_height,
        dst_width,
    )


# ---------------------------------------------------------------------------
# 3. Rasterise a vector attribute
# ---------------------------------------------------------------------------

def reproject_vector_to_raster(
    gdf: gpd.GeoDataFrame,
    attribute: str,
    target_profile: dict,
    dtype: str = "float32",
    all_touched: bool = False,
) -> np.ndarray:
    """Rasterise a GeoDataFrame attribute onto the pilot grid.

    Geoscientific rationale
    -----------------------
    Many prospectivity inputs are delivered as vector polygons or lines —
    geological mapping (lithostratigraphy, faults), stream-sediment anomaly
    catchments, and geochemical sample locations.  Converting them to raster
    grids creates spatially continuous predictor layers that can be fed into
    the model alongside remote-sensing and geophysical rasters.

    The ``all_touched`` parameter is particularly important for **fault/
    lineament** vectors: setting it to ``True`` ensures that every pixel the
    fault line crosses receives the attribute value, preventing thin structures
    from disappearing at 50 m resolution.

    Args:
        gdf: Source GeoDataFrame.  Will be reprojected to the target CRS
             if its CRS differs.
        attribute: Column name in *gdf* whose values will be burned into the
                   raster.  Must be numeric.
        target_profile: Rasterio profile defining the target grid
                        (CRS, transform, width, height).
        dtype: NumPy/GDAL dtype string for the output array, e.g.
               ``"float32"``, ``"int16"``, ``"uint8"``.  Defaults to
               ``"float32"``.
        all_touched: If ``True``, all pixels touched by a geometry receive
                     its value.  If ``False`` (default), only pixels whose
                     centres fall within the geometry are set.  Use ``True``
                     for line features.

    Returns:
        A 2-D ``np.ndarray`` of shape ``(height, width)`` with ``dtype``
        dtype.  Pixels not covered by any geometry are filled with 0.

    Raises:
        KeyError: If *attribute* is not a column in *gdf*.
        ValueError: If *gdf* is empty.

    Example:
        >>> arr = reproject_vector_to_raster(
        ...     geology_gdf, "rock_code", target_profile, dtype="int16"
        ... )
    """
    if attribute not in gdf.columns:
        raise KeyError(f"Attribute '{attribute}' not found in GeoDataFrame columns: {list(gdf.columns)}")
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty; nothing to rasterise.")

    target_crs: CRS = target_profile["crs"]

    # Reproject vector to target CRS if necessary
    if gdf.crs is None:
        logger.warning("GeoDataFrame has no CRS; assuming it matches the target CRS.")
    elif gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    transform = target_profile["transform"]
    width: int = target_profile["width"]
    height: int = target_profile["height"]

    # Build (geometry, value) pairs for rasterize
    shapes = (
        (geom, value)
        for geom, value in zip(gdf.geometry, gdf[attribute])
        if geom is not None and not geom.is_empty
    )

    out_array = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype=dtype,
    )

    logger.debug(
        "Rasterised attribute '%s'  shape=%s  dtype=%s  non-zero=%.2f%%",
        attribute,
        out_array.shape,
        dtype,
        100.0 * np.count_nonzero(out_array) / out_array.size,
    )
    return out_array


# ---------------------------------------------------------------------------
# 4. Validate alignment
# ---------------------------------------------------------------------------

def validate_alignment(raster_paths: list[str]) -> bool:
    """Check that all rasters share the same CRS, transform, and shape.

    Geoscientific rationale
    -----------------------
    A common source of silent errors in MPM workflows is stacking layers that
    are *almost* aligned but differ by a half-pixel offset, a small CRS
    datum shift, or a rounding discrepancy in the transform.  These
    misalignments produce systematic spatial offsets in the model predictions
    that are very difficult to detect post-hoc.  Running this check after the
    harmonisation step — and again before model training — catches problems
    early.

    Args:
        raster_paths: List of paths to raster files to compare.  All are
                      compared against the first entry (reference).

    Returns:
        ``True`` if all rasters are aligned.

    Raises:
        ValueError: Describing the first misalignment found (CRS, transform,
                    or shape mismatch), including the path of the offending
                    raster and the reference values.
        FileNotFoundError: If any path does not exist.

    Example:
        >>> validate_alignment(["dem_50m.tif", "aeromag_50m.tif", "ndvi_50m.tif"])
        True
    """
    if not raster_paths:
        raise ValueError("raster_paths must contain at least one path.")

    paths = [Path(p) for p in raster_paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Raster not found: {p}")

    # Read reference metadata
    with rasterio.open(paths[0]) as ref_ds:
        ref_crs = ref_ds.crs
        ref_transform = ref_ds.transform
        ref_shape = (ref_ds.height, ref_ds.width)

    logger.debug(
        "Alignment reference: %s  CRS=%s  shape=%s  transform=%s",
        paths[0].name,
        ref_crs,
        ref_shape,
        ref_transform,
    )

    for p in paths[1:]:
        with rasterio.open(p) as ds:
            candidate_crs = ds.crs
            candidate_transform = ds.transform
            candidate_shape = (ds.height, ds.width)

        # --- CRS check ---
        if candidate_crs != ref_crs:
            raise ValueError(
                f"CRS mismatch for '{p.name}':\n"
                f"  Reference ({paths[0].name}): {ref_crs.to_string()}\n"
                f"  Candidate ({p.name}):        {candidate_crs.to_string()}\n"
                "Run reproject_raster() to harmonise before stacking."
            )

        # --- Shape check ---
        if candidate_shape != ref_shape:
            raise ValueError(
                f"Shape mismatch for '{p.name}':\n"
                f"  Reference ({paths[0].name}): height={ref_shape[0]}, width={ref_shape[1]}\n"
                f"  Candidate ({p.name}):        height={candidate_shape[0]}, width={candidate_shape[1]}\n"
                "Ensure both rasters were projected to the same pilot grid."
            )

        # --- Transform check (element-wise with tolerance) ---
        ref_t = list(ref_transform)[:6]
        cand_t = list(candidate_transform)[:6]
        if not all(abs(a - b) < 1e-3 for a, b in zip(ref_t, cand_t)):
            raise ValueError(
                f"Transform mismatch for '{p.name}':\n"
                f"  Reference ({paths[0].name}): {ref_transform}\n"
                f"  Candidate ({p.name}):        {candidate_transform}\n"
                "Sub-pixel offsets will cause systematic prediction errors. "
                "Re-run reproject_raster() snapping to the same pilot grid."
            )

        logger.debug("✓ Aligned: %s", p.name)

    logger.info("validate_alignment: all %d rasters are aligned.", len(paths))
    return True
