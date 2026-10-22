"""
src/harmonization/cover_mask.py
================================
Build the data-quality / cover-mask layer for the MPM predictor cube.

Geoscientific rationale
-----------------------
Surface cover is the single greatest source of spectral noise in
mineral-system mapping.  Three cover types are particularly problematic:

1. **Dense vegetation** (trees, shrubs, spinifex):
   ASTER SWIR bands (4–9, 1.6–2.5 µm) that are crucial for mapping
   hydrothermal alteration minerals (kaolinite, alunite, carbonate) are
   overwhelmed by the chlorophyll "red-edge" and cellulose absorption
   features.  The NDVI threshold of 0.2 is conservative for semi-arid
   terrains; it may need lowering to 0.15 in hyper-arid zones.

2. **Deep regolith** (laterite, calcrete, silcrete):
   Thick regolith (> 30 m) masks the primary lithological signal of the
   bedrock.  Geochemical dispersion patterns in regolith can still guide
   prospectivity, but spectral proxies for primary mineralogy become
   unreliable.  The regolith thickness threshold is configurable.

3. **Water bodies and playas**:
   Water and salt lake surfaces produce strong, broad absorption features
   that contaminate both the spectral library matching and geochemical
   interpolation.  The DEM-based heuristic (low slope + high upstream flow
   accumulation) is a lightweight proxy when a dedicated water-body mask is
   unavailable.

The provenance-band GeoTIFF produced by :func:`save_provenance_band` allows
model training runs to selectively down-weight or exclude masked pixels and
is also invaluable for auditing prediction confidence in post-modelling
review.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectral index utilities
# ---------------------------------------------------------------------------

def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute the Normalised Difference Vegetation Index (NDVI).

    Standard formula: ``NDVI = (NIR − RED) / (NIR + RED)``

    NDVI saturates above ~0.8 in dense tropical canopies but is reliable
    in the sparse-to-moderate vegetation typical of Australian target
    terrains (0.05–0.5).  Division-by-zero (pixels where NIR + RED == 0,
    i.e. fully dark pixels or missing data) is handled by returning 0.0
    for those pixels — a convention that places them below any practical
    vegetation threshold.

    Args:
        red: 2-D array of RED band reflectance values (ASTER band 2,
             0.63–0.69 µm, or Landsat band 4 / Sentinel-2 band 4).
             Values should be in [0, 1] or [0, 10000] — only the *ratio*
             matters, so the absolute scale is irrelevant.
        nir: 2-D array of NIR band reflectance values (ASTER band 3N,
             0.78–0.86 µm, or Landsat band 5 / Sentinel-2 band 8).
             Must be the same shape as *red*.

    Returns:
        2-D ``np.ndarray`` of dtype ``float32`` with NDVI values in
        ``[-1.0, 1.0]``.  Pixels where ``NIR + RED == 0`` are set to
        ``0.0``.

    Raises:
        ValueError: If *red* and *nir* have different shapes.

    Example:
        >>> ndvi = compute_ndvi(red_band, nir_band)
        >>> print(ndvi.min(), ndvi.max())
        -0.15 0.72
    """
    if red.shape != nir.shape:
        raise ValueError(
            f"red and nir must have the same shape; got {red.shape} vs {nir.shape}."
        )

    red = red.astype(np.float32)
    nir = nir.astype(np.float32)

    denominator = nir + red
    # Where denominator is zero or NaN, use 1.0 to avoid division; result is
    # overridden to 0.0 below.
    safe_denom = np.where(denominator == 0, 1.0, denominator)
    ndvi = (nir - red) / safe_denom
    ndvi[denominator == 0] = 0.0
    ndvi = np.clip(ndvi, -1.0, 1.0)

    logger.debug(
        "compute_ndvi: min=%.3f  max=%.3f  mean=%.3f",
        float(np.nanmin(ndvi)),
        float(np.nanmax(ndvi)),
        float(np.nanmean(ndvi)),
    )
    return ndvi.astype(np.float32)


# ---------------------------------------------------------------------------
# Individual mask builders
# ---------------------------------------------------------------------------

def build_vegetation_mask(
    ndvi: np.ndarray,
    threshold: float = 0.2,
) -> np.ndarray:
    """Build a boolean mask identifying vegetated pixels.

    Vegetated pixels are flagged as ``True`` (masked out / unreliable for
    ASTER SWIR-based alteration mapping).  Pixels with NDVI below the
    threshold are considered bare ground, exposed rock, or dry litter —
    suitable for spectral analysis.

    Args:
        ndvi: 2-D NDVI array (output of :func:`compute_ndvi`), values in
              ``[-1, 1]``.
        threshold: NDVI threshold above which pixels are considered
                   vegetated.  Defaults to ``0.2``.  Increase to ``0.3`` in
                   wetter climates; lower to ``0.1`` in hyper-arid deserts
                   where even sparse scrub suppresses SWIR signal.

    Returns:
        2-D boolean ``np.ndarray``: ``True`` = vegetation present (unreliable
        for SWIR mineral mapping).

    Example:
        >>> veg_mask = build_vegetation_mask(ndvi, threshold=0.2)
        >>> print(veg_mask.sum(), "vegetated pixels")
    """
    mask = ndvi >= float(threshold)
    logger.debug(
        "build_vegetation_mask(threshold=%.2f): %.1f%% pixels masked",
        threshold,
        100.0 * mask.sum() / mask.size,
    )
    return mask.astype(bool)


def build_cover_mask(
    regolith_thickness: np.ndarray,
    threshold_m: float = 30.0,
) -> np.ndarray:
    """Build a boolean mask for pixels with deep regolith cover.

    Deep weathering profiles (lateritic, ferruginous, calcrete) blanket the
    bedrock and destroy primary lithological spectral signatures.  At depths
    > 30 m, even geochemical anomalies in stream sediments become diffuse
    and ambiguous.

    Args:
        regolith_thickness: 2-D array of regolith thickness in **metres**.
                            Typically derived from the Geoscience Australia
                            national regolith depth model or a borehole
                            interpolation.  NaN pixels are treated as
                            *not* masked (i.e. unknown depth → assume shallow).
        threshold_m: Regolith depth threshold in metres above which pixels
                     are flagged as unreliable.  Defaults to ``30.0`` m.

    Returns:
        2-D boolean ``np.ndarray``: ``True`` = regolith too deep (unreliable
        bedrock signal).

    Example:
        >>> regolith_mask = build_cover_mask(regolith_arr, threshold_m=30.0)
    """
    # NaN → treat as zero (shallow / unknown); mask only definite deep pixels.
    thickness = np.where(np.isnan(regolith_thickness), 0.0, regolith_thickness)
    mask = thickness > float(threshold_m)
    logger.debug(
        "build_cover_mask(threshold=%.1fm): %.1f%% pixels masked",
        threshold_m,
        100.0 * mask.sum() / mask.size,
    )
    return mask.astype(bool)


def build_water_mask(
    dem: np.ndarray,
    flow_acc: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    """Build a heuristic boolean mask for water bodies and playas.

    Geoscientific rationale
    -----------------------
    Lakes, ephemeral playas, and drainage channel floors are identified by
    combining two topographic proxies:

    - **Very low slope** (< 0.5°): Flat surfaces in arid Australia are
      overwhelmingly salt lakes, playas, or alluvial floodplains.  Exposed
      rock rarely has slopes < 0.5°.
    - **High upstream flow accumulation** (> 1 000 cells): A large
      contributing area means the location is topographically depressed
      relative to its surroundings — a necessary condition for water
      ponding.

    Together the two conditions greatly reduce false positives (e.g. flat
    mesas or wind-deflated plains with low drainage area).

    Args:
        dem: 2-D array of elevation in metres (any datum).
        flow_acc: 2-D array of upstream flow accumulation in cells
                  (output of standard D8 or D-infinity routing, e.g. via
                  WhiteboxTools ``FlowAccumulationFullWorkflow`` or SAGA GIS
                  ``Flow Accumulation (Top-Down)``).
        slope: 2-D array of surface slope in **degrees**.  Typically
               computed from the DEM using ``gdal_DEMprocessing`` or
               ``richdem``.

    Returns:
        2-D boolean ``np.ndarray``: ``True`` = probable water body / playa
        (exclude from spectral and geochemical analysis).

    Raises:
        ValueError: If the three arrays do not share the same shape.

    Example:
        >>> water_mask = build_water_mask(dem, flow_acc, slope)
    """
    if not (dem.shape == flow_acc.shape == slope.shape):
        raise ValueError(
            "dem, flow_acc, and slope must have identical shapes; "
            f"got {dem.shape}, {flow_acc.shape}, {slope.shape}."
        )

    flat = slope < 0.5           # degrees
    # structural.flow_accumulation() outputs log10(acc + 1), where 1000 cells = log10(1001) ≈ 3.00043
    finite_max = float(np.nanmax(flow_acc)) if np.isfinite(flow_acc).any() else 0.0
    flow_threshold = np.log10(1000.0 + 1.0) if finite_max <= 25.0 else 1000.0
    high_flow = flow_acc > flow_threshold
    mask = flat & high_flow

    logger.debug(
        "build_water_mask: %.1f%% pixels masked (slope<0.5° AND flow_acc>1000)",
        100.0 * mask.sum() / mask.size,
    )
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Mask combination
# ---------------------------------------------------------------------------

def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """OR-combine multiple boolean masks into a single composite mask.

    Any pixel flagged by *at least one* constituent mask is masked out in
    the combined result.  This is the conservative approach: it is preferable
    to exclude a pixel that is borderline-vegetated than to include a
    contaminated pixel in model training.

    Args:
        *masks: Any number of 2-D boolean arrays.  All must have identical
                shapes.

    Returns:
        2-D boolean ``np.ndarray``: ``True`` wherever at least one input
        mask is ``True``.

    Raises:
        ValueError: If no masks are provided, or if masks have different shapes.

    Example:
        >>> combined = combine_masks(veg_mask, water_mask, regolith_mask)
        >>> print(f"{combined.sum()} pixels excluded")
    """
    if not masks:
        raise ValueError("At least one mask must be supplied.")

    reference_shape = masks[0].shape
    for i, m in enumerate(masks[1:], start=1):
        if m.shape != reference_shape:
            raise ValueError(
                f"Mask {i} has shape {m.shape} but mask 0 has shape "
                f"{reference_shape}; all masks must be the same shape."
            )

    combined = masks[0].astype(bool).copy()
    for m in masks[1:]:
        combined |= m.astype(bool)

    n_masked = int(combined.sum())
    pct = 100.0 * n_masked / combined.size
    logger.info(
        "combine_masks: %d input masks → %d pixels (%.1f%%) masked total",
        len(masks),
        n_masked,
        pct,
    )
    return combined


# ---------------------------------------------------------------------------
# Provenance band output
# ---------------------------------------------------------------------------

def save_provenance_band(
    masks: dict[str, np.ndarray],
    profile: dict,
    out_path: str | os.PathLike,
) -> None:
    """Save a multi-band provenance GeoTIFF, one band per named mask.

    Each band stores a single boolean mask as uint8 (0 = valid, 1 = masked).
    Band descriptions are set to the mask names so the file is self-describing
    in any GIS viewer.

    Geoscientific rationale
    -----------------------
    Storing individual mask components — rather than only the combined mask —
    preserves the ability to audit which cover type drove exclusion in any
    particular area.  This is especially important in peer-review and
    regulatory contexts where the model's spatial confidence must be
    documented.  The file also enables sensitivity analysis: "How does model
    performance change if we include vegetated pixels?"

    Args:
        masks: Ordered dictionary mapping mask name (e.g. ``"vegetation"``,
               ``"regolith"``, ``"water"``) to a 2-D boolean array.
        profile: Rasterio profile of the pilot grid (CRS, transform, width,
                 height).  ``count`` and ``dtype`` are overridden internally.
        out_path: Destination path for the provenance GeoTIFF (COG).
                  Parent directories are created if absent.

    Raises:
        ValueError: If *masks* is empty or if mask shapes are inconsistent
                    with the profile dimensions.

    Example:
        >>> save_provenance_band(
        ...     masks={"vegetation": veg_mask, "regolith": reg_mask, "water": water_mask},
        ...     profile=pilot_profile,
        ...     out_path="outputs/cover_mask_provenance.tif",
        ... )
    """
    if not masks:
        raise ValueError("masks dict must not be empty.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    expected_h = profile["height"]
    expected_w = profile["width"]

    band_names = list(masks.keys())
    n_bands = len(band_names)

    # Stack all masks into a uint8 array
    data = np.zeros((n_bands, expected_h, expected_w), dtype=np.uint8)
    for i, (name, mask) in enumerate(masks.items()):
        if mask.shape != (expected_h, expected_w):
            raise ValueError(
                f"Mask '{name}' has shape {mask.shape} but profile specifies "
                f"({expected_h}, {expected_w})."
            )
        data[i] = mask.astype(np.uint8)

    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype="uint8",
        count=n_bands,
        nodata=255,           # 255 = no data (distinct from 0=valid, 1=masked)
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    with rasterio.open(str(out_path), "w", **out_profile) as dst:
        dst.write(data)
        for i, name in enumerate(band_names, start=1):
            dst.update_tags(i, name=name, description=f"Cover mask: {name} (0=valid, 1=masked)")
            # Set band description for display in GIS tools
            dst.set_band_description(i, name)

    logger.info(
        "Saved provenance band → %s  bands=%s",
        out_path,
        band_names,
    )


def update_cover_mask_from_aem(
    existing_mask: np.ndarray,
    regolith_depth_m: np.ndarray,
    max_penetration_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Update the harmonization cover mask with AEM-inverted regolith thickness.

    Combines an existing boolean surface cover mask (e.g. vegetation + water)
    with continuous AEM-inverted regolith thickness, and returns both the
    updated boolean mask (for optical/geochemical masking) and the continuous
    regolith thickness array (preserved for Pass 2 propensity modeling and
    Pass 6 multi-criteria ranking).

    Args:
        existing_mask: 2-D boolean mask of shape ``(H, W)``.
        regolith_depth_m: 2-D float array of AEM-inverted regolith cover depth
            in metres, shape ``(H, W)``.
        max_penetration_m: Depth threshold (m) beyond which surface optical and
            soil geochemical signals are masked as covered. Defaults to ``30.0``.

    Returns:
        Tuple ``(updated_mask, continuous_regolith_depth_m)`` where
        ``updated_mask`` is a 2-D boolean array and
        ``continuous_regolith_depth_m`` is a 2-D float64 array with NaNs
        filled to 0.0 for downstream numerical stability.
    """
    existing = np.asarray(existing_mask, dtype=bool)
    depth = np.asarray(regolith_depth_m, dtype=np.float64)
    if existing.shape != depth.shape:
        raise ValueError(
            f"existing_mask shape {existing.shape} != regolith_depth_m shape {depth.shape}"
        )
    aem_deep_mask = build_cover_mask(depth, threshold_m=max_penetration_m)
    updated_mask = combine_masks(existing, aem_deep_mask)
    clean_depth = np.where(np.isnan(depth), 0.0, np.clip(depth, 0.0, None))
    return updated_mask, clean_depth

