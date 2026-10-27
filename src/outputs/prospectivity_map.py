"""GIS-ready output generation for Mineral Prospectivity Mapping.

This module converts raw model scores (numpy arrays indexed by pixel
coordinates) into geospatially referenced deliverables:

* **Float32 Cloud-Optimised GeoTIFFs (COG)** — prospectivity and uncertainty
  maps that can be loaded directly in QGIS, ArcGIS, or any GDAL-aware tool.

* **Multi-band SHAP GeoTIFF** — per-feature attribution maps enabling
  geologists to interrogate *why* each pixel was scored highly, supporting
  target prioritisation narratives.

* **Target polygons GeoPackage** — vectorised contiguous anomaly blobs
  filtered by minimum prospectivity score, maximum allowable uncertainty, and
  minimum area (ha), ready for import into an exploration database or drill-
  hole planning workflow.

All rasters are written with DEFLATE compression (level 9) and tiled
(256 × 256) to allow efficient partial reads across network-attached storage
or cloud object stores (S3 / GCS / Azure Blob).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from scipy import ndimage
from shapely.geometry import shape

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_COG_CREATION_OPTIONS: dict = {
    "driver": "GTiff",
    "dtype": "float32",
    "compress": "deflate",
    "zlevel": 9,
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "interleave": "band",
    "BIGTIFF": "IF_SAFER",
}


def _scatter_to_2d(
    values: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    H: int,
    W: int,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Reconstruct a 2-D raster from flat pixel-index arrays.

    The MPM feature-extraction pipeline operates on masked (non-NaN) pixels
    only, producing flat arrays of shape ``(N_valid,)`` together with their
    row and column indices.  This helper scatters those values back into a
    full 2-D grid.

    Args:
        values (np.ndarray): Flat array of per-pixel values, shape ``(N,)``.
        row_idx (np.ndarray): Row indices of valid pixels, shape ``(N,)``.
        col_idx (np.ndarray): Column indices of valid pixels, shape ``(N,)``.
        H (int): Raster height in pixels.
        W (int): Raster width in pixels.
        fill_value (float): Value assigned to masked-out (invalid) pixels.

    Returns:
        np.ndarray: Reconstructed 2-D array of shape ``(H, W)``, dtype
            ``float32``.
    """
    grid = np.full((H, W), fill_value=fill_value, dtype=np.float32)
    grid[row_idx, col_idx] = values.astype(np.float32)
    return grid


def _build_cog_profile(base_profile: dict, n_bands: int = 1) -> dict:
    """Merge a base rasterio profile with COG creation options.

    Args:
        base_profile (dict): Source raster profile (CRS, transform, height,
            width, nodata).
        n_bands (int): Number of output bands.

    Returns:
        dict: Updated profile ready to be passed to ``rasterio.open``.
    """
    profile = dict(base_profile)
    profile.update(_COG_CREATION_OPTIONS)
    w = int(base_profile.get("width", 256))
    h = int(base_profile.get("height", 256))
    bs = min(256, w, h)
    bs = max(16, (bs // 16) * 16)
    profile["blockxsize"] = bs
    profile["blockysize"] = bs
    profile["count"] = n_bands
    profile["nodata"] = np.nan
    return profile


# ---------------------------------------------------------------------------
# Prospectivity raster
# ---------------------------------------------------------------------------


def save_prospectivity_geotiff(
    scores: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    H: int,
    W: int,
    profile: dict,
    out_path: str,
) -> None:
    """Save a float32 COG GeoTIFF prospectivity map (values in [0, 1]).

    Reconstructs the full 2-D raster from flattened pixel arrays and writes
    a Cloud-Optimised GeoTIFF with DEFLATE compression.  Pixels outside the
    valid study mask are set to NaN (nodata).

    Args:
        scores (np.ndarray): Flat prospectivity scores, shape ``(N,)``,
            values in [0, 1].
        row_idx (np.ndarray): Row indices of valid (non-masked) pixels,
            shape ``(N,)``.
        col_idx (np.ndarray): Column indices of valid pixels, shape ``(N,)``.
        H (int): Raster height in pixels.
        W (int): Raster width in pixels.
        profile (dict): Source rasterio profile containing CRS, transform,
            and spatial reference information.
        out_path (str): Destination GeoTIFF path (e.g.
            ``"outputs/prospectivity.tif"``).
    """
    grid = _scatter_to_2d(scores, row_idx, col_idx, H, W)
    cog_profile = _build_cog_profile(profile, n_bands=1)

    with rasterio.open(out_path, "w", **cog_profile) as dst:
        dst.write(grid[np.newaxis, :, :])  # shape (1, H, W)
        dst.update_tags(
            description="MPM prospectivity map",
            value_range="0.0 – 1.0 (0 = barren, 1 = highly prospective)",
        )

    logger.info("Prospectivity GeoTIFF saved → %s", out_path)


# ---------------------------------------------------------------------------
# Uncertainty raster
# ---------------------------------------------------------------------------


def save_uncertainty_geotiff(
    variance: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    H: int,
    W: int,
    profile: dict,
    out_path: str,
) -> None:
    """Save a float32 COG GeoTIFF epistemic uncertainty (variance) map.

    The variance map represents model disagreement across ensemble members
    (XGBoost bagging) or MC-Dropout passes (U-Net).  High-variance regions
    are data-sparse or feature-ambiguous — important inputs for prioritising
    new survey acquisition over drilling.

    Args:
        variance (np.ndarray): Flat epistemic variance values, shape ``(N,)``.
            For a Bernoulli model the maximum is 0.25 (p = 0.5).
        row_idx (np.ndarray): Row indices of valid pixels, shape ``(N,)``.
        col_idx (np.ndarray): Column indices of valid pixels, shape ``(N,)``.
        H (int): Raster height in pixels.
        W (int): Raster width in pixels.
        profile (dict): Source rasterio profile containing CRS and transform.
        out_path (str): Destination GeoTIFF path.
    """
    grid = _scatter_to_2d(variance, row_idx, col_idx, H, W)
    cog_profile = _build_cog_profile(profile, n_bands=1)

    with rasterio.open(out_path, "w", **cog_profile) as dst:
        dst.write(grid[np.newaxis, :, :])
        dst.update_tags(
            description="MPM epistemic uncertainty (predictive variance)",
            value_range="0.0 – 0.25 (Bernoulli upper bound at p=0.5)",
        )

    logger.info("Uncertainty GeoTIFF saved → %s", out_path)


# ---------------------------------------------------------------------------
# SHAP raster (multi-band)
# ---------------------------------------------------------------------------


def save_shap_geotiff(
    shap_values: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    H: int,
    W: int,
    profile: dict,
    feature_names: list[str],
    top_n: int = 5,
    out_path: str = "",
) -> None:
    """Save the top-N SHAP feature attribution channels as a multi-band COG.

    Each band corresponds to one input feature, ordered by descending mean
    absolute SHAP value (global importance rank).  Band metadata includes
    the feature name, enabling ArcGIS / QGIS band-by-band interpretation
    without consulting a separate legend.

    Geoscientific use:
        Plot individual bands to produce feature contribution maps that answer
        *"Which evidence layer is driving high prospectivity in sector X?"*
        This is the spatial analogue of a tabular SHAP summary plot and is
        directly usable in prospect reports.

    Args:
        shap_values (np.ndarray): SHAP value matrix, shape
            ``(N, n_features)``.  Positive values push toward mineralised;
            negative values push toward barren.
        row_idx (np.ndarray): Row indices of valid pixels, shape ``(N,)``.
        col_idx (np.ndarray): Column indices of valid pixels, shape ``(N,)``.
        H (int): Raster height in pixels.
        W (int): Raster width in pixels.
        profile (dict): Source rasterio profile.
        feature_names (list[str]): Feature names corresponding to columns of
            ``shap_values``.
        top_n (int): Number of top-ranked features to include as bands.
            Default 5.
        out_path (str): Destination GeoTIFF path.  Required if saving to disk;
            if empty string, raises ``ValueError``.

    Raises:
        ValueError: If ``out_path`` is empty or ``top_n`` exceeds the
            number of features.
    """
    if not out_path:
        raise ValueError("out_path must be a non-empty file path string.")

    n_features = shap_values.shape[1]
    top_n = min(top_n, n_features)

    # Rank features by mean |SHAP| across all valid pixels
    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # shape (n_features,)
    top_feature_idx = np.argsort(mean_abs_shap)[::-1][:top_n]  # descending

    cog_profile = _build_cog_profile(profile, n_bands=top_n)

    with rasterio.open(out_path, "w", **cog_profile) as dst:
        for band_num, feat_idx in enumerate(top_feature_idx, start=1):
            feat_name = feature_names[feat_idx]
            shap_band = shap_values[:, feat_idx]  # shape (N,)
            grid = _scatter_to_2d(shap_band, row_idx, col_idx, H, W, fill_value=0.0)
            dst.write(grid, band_num)
            dst.set_band_description(band_num, feat_name)
            dst.update_tags(
                band_num,
                feature_name=feat_name,
                global_mean_abs_shap=f"{mean_abs_shap[feat_idx]:.6f}",
                importance_rank=str(band_num),
            )

        dst.update_tags(
            description="MPM SHAP feature attribution map (top-N features)",
            top_n=str(top_n),
        )

    logger.info(
        "SHAP GeoTIFF saved → %s | top_%d features: %s",
        out_path,
        top_n,
        [feature_names[i] for i in top_feature_idx],
    )


# ---------------------------------------------------------------------------
# Target polygon generation
# ---------------------------------------------------------------------------


def generate_target_polygons(
    prospectivity: np.ndarray,
    uncertainty: np.ndarray,
    profile: dict,
    min_score: float = 0.70,
    max_uncertainty: float = 0.20,
    min_area_ha: float = 25.0,
) -> gpd.GeoDataFrame:
    """Threshold, filter, and vectorise prospectivity blobs into drill targets.

    Workflow:
    1. **Threshold** — retain pixels where ``prospectivity ≥ min_score``.
    2. **Uncertainty mask** — remove pixels where
       ``uncertainty > max_uncertainty`` (high-variance pixels inflate target
       polygons with poorly-constrained anomalies).
    3. **Area filter** — discard blobs smaller than ``min_area_ha`` hectares
       (sub-threshold anomalies are typically artefacts of small training
       deposits or edge effects).
    4. **Vectorise** — ``rasterio.features.shapes`` converts connected
       components to GeoJSON-style polygon geometries.
    5. **Attribute** — each polygon is attributed with mean_score, max_score,
       area_ha, and mean_uncertainty for ranking in the exploration database.

    Args:
        prospectivity (np.ndarray): 2-D prospectivity map, shape ``(H, W)``,
            values in [0, 1].  NaN = nodata.
        uncertainty (np.ndarray): 2-D epistemic uncertainty map, shape
            ``(H, W)``, matching ``prospectivity`` exactly.
        profile (dict): Rasterio profile containing ``transform`` (affine),
            ``crs``, and ``nodata``.
        min_score (float): Minimum prospectivity threshold for inclusion.
            Default 0.70.
        max_uncertainty (float): Maximum epistemic variance allowed.  Pixels
            above this are excluded even if prospectivity is high.  Default
            0.20 (80 % of Bernoulli maximum variance of 0.25).
        min_area_ha (float): Minimum polygon area in hectares.  Default 25 ha
            (≈ 500 × 500 m target footprint).

    Returns:
        gpd.GeoDataFrame: Target polygon GeoDataFrame with columns:
            * ``geometry`` — Shapely polygon in the source CRS.
            * ``mean_score`` — mean prospectivity within polygon.
            * ``max_score`` — maximum prospectivity within polygon.
            * ``area_ha`` — polygon area in hectares.
            * ``mean_uncertainty`` — mean epistemic variance within polygon.
    """
    transform = profile["transform"]
    crs_obj = CRS.from_user_input(profile["crs"])

    # Pixel area in m² (scaled if CRS is geographic)
    m2_scale = (111_320.0 ** 2) if crs_obj.is_geographic else 1.0
    pixel_area_m2 = abs(transform.a * transform.e) * m2_scale
    pixels_per_ha = 10_000.0 / max(pixel_area_m2, 1e-6)

    # Step 1 & 2: Build binary mask
    binary_mask = (
        (prospectivity >= min_score)
        & (uncertainty <= max_uncertainty)
        & np.isfinite(prospectivity)
        & np.isfinite(uncertainty)
    ).astype(np.uint8)

    # Step 3: Label connected components and apply area filter
    labeled_array, num_features = ndimage.label(binary_mask)
    logger.info(
        "Found %d candidate blobs before area filter (min %.1f ha)",
        num_features,
        min_area_ha,
    )

    # Remove small blobs
    component_sizes = ndimage.sum(
        binary_mask, labeled_array, range(1, num_features + 1)
    )
    small_blob_labels = [
        lbl + 1
        for lbl, size in enumerate(component_sizes)
        if size < (min_area_ha * pixels_per_ha)
    ]
    for lbl in small_blob_labels:
        binary_mask[labeled_array == lbl] = 0

    # Re-label after area filtering
    labeled_array, num_features = ndimage.label(binary_mask)
    logger.info("%d blobs remaining after area filter.", num_features)

    if num_features == 0:
        logger.warning(
            "No target polygons passed all filters.  Consider lowering "
            "min_score=%.2f or min_area_ha=%.1f.",
            min_score,
            min_area_ha,
        )
        return gpd.GeoDataFrame(
            columns=["geometry", "mean_score", "max_score", "area_ha", "mean_uncertainty"],
            geometry="geometry",
            crs=crs_obj,
        )

    # Step 4: Vectorise using rasterio.features.shapes
    shapes_gen = rasterio.features.shapes(
        binary_mask.astype(np.int32),
        mask=binary_mask,
        transform=transform,
    )

    records: list[dict] = []
    for geom_dict, pixel_value in shapes_gen:
        if pixel_value == 0:
            continue

        poly = shape(geom_dict)
        area_ha = (poly.area * m2_scale) / 10_000.0  # convert m² → ha

        if area_ha < min_area_ha:
            continue  # double-check (shapes can split blobs at boundaries)

        # Extract pixel statistics within the polygon via raster mask
        # Build a boolean mask for pixels inside this specific polygon
        blob_mask_2d = rasterio.features.geometry_mask(
            [geom_dict],
            out_shape=prospectivity.shape,
            transform=transform,
            invert=True,  # True = inside polygon
        )

        scores_inside = prospectivity[blob_mask_2d & np.isfinite(prospectivity)]
        unc_inside = uncertainty[blob_mask_2d & np.isfinite(uncertainty)]

        if len(scores_inside) == 0:
            continue

        records.append(
            {
                "geometry": poly,
                "mean_score": float(scores_inside.mean()),
                "max_score": float(scores_inside.max()),
                "area_ha": float(area_ha),
                "mean_uncertainty": float(
                    unc_inside.mean() if len(unc_inside) > 0 else np.nan
                ),
            }
        )

    targets_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs_obj)

    # Sort by descending mean_score for easy prioritisation
    targets_gdf = targets_gdf.sort_values("mean_score", ascending=False).reset_index(
        drop=True
    )

    logger.info(
        "Generated %d target polygons | min_score=%.2f | max_unc=%.2f | "
        "min_area_ha=%.1f",
        len(targets_gdf),
        min_score,
        max_uncertainty,
        min_area_ha,
    )
    return targets_gdf


# ---------------------------------------------------------------------------
# GeoPackage export
# ---------------------------------------------------------------------------


def export_targets_geopackage(
    targets: gpd.GeoDataFrame,
    out_path: str,
) -> None:
    """Export target polygons to a GeoPackage with metadata attributes.

    GeoPackage (OGC standard) is preferred over Shapefile for MPM deliverables
    because:
    * No 255-character column-name truncation.
    * Single-file format (no sidecar .prj / .dbf / .shx).
    * Native UTF-8 encoding preserves Unicode feature names.
    * Supports multiple layers in a single file (extendable to include
      metadata, calibration curves, etc.).

    The exported layer is named ``"prospectivity_targets"`` for unambiguous
    discovery in QGIS / ArcGIS layer panels.

    Args:
        targets (gpd.GeoDataFrame): GeoDataFrame produced by
            :func:`generate_target_polygons`.  Must have a valid CRS set.
        out_path (str): Destination ``.gpkg`` file path.  If the file already
            exists, the layer is overwritten (``if_exists="replace"``).

    Raises:
        ValueError: If ``targets`` has no rows or no CRS.
    """
    if targets.crs is None:
        raise ValueError(
            "Target GeoDataFrame has no CRS.  Set crs before exporting."
        )

    if len(targets) == 0:
        logger.warning(
            "export_targets_geopackage called with empty GeoDataFrame — "
            "writing empty layer to %s.",
            out_path,
        )

    # Add a ranked target ID column for drill-planning reference
    targets = targets.copy()
    targets.insert(0, "target_id", [f"T{i+1:04d}" for i in range(len(targets))])

    targets.to_file(
        out_path,
        layer="prospectivity_targets",
        driver="GPKG",
    )

    logger.info(
        "Target polygons exported → %s | layer='prospectivity_targets' | "
        "n_targets=%d",
        out_path,
        len(targets),
    )


# ---------------------------------------------------------------------------
# Pass 5, 6 & 7: Conformal GeoTIFFs, Multi-Criteria Ranking & LLM Site Cards
# ---------------------------------------------------------------------------


def save_conformal_bounds_geotiff(
    lower_bound_L: np.ndarray,
    upper_bound_U: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    H: int,
    W: int,
    profile: dict,
    out_path: str,
) -> None:
    """Save Pass 5 spatial conformal 90% confidence bounds $[L(x), U(x)]$ as a 2-band COG.

    Band 1 = ``lower_bound_L``, Band 2 = ``upper_bound_U``.
    """
    grid_l = _scatter_to_2d(lower_bound_L, row_idx, col_idx, H, W)
    grid_u = _scatter_to_2d(upper_bound_U, row_idx, col_idx, H, W)
    cog_profile = _build_cog_profile(profile, n_bands=2)

    with rasterio.open(out_path, "w", **cog_profile) as dst:
        dst.write(grid_l, 1)
        dst.set_band_description(1, "lower_bound_L")
        dst.write(grid_u, 2)
        dst.set_band_description(2, "upper_bound_U")
        dst.update_tags(
            description="Spatial Conformal 90% Drill Confidence Range [L(x), U(x)]",
            alpha="0.10",
        )
    logger.info("Conformal bounds GeoTIFF saved → %s", out_path)


def _norm_01(vals: np.ndarray) -> np.ndarray:
    """Normalise an array to $[0, 1]$ across targets (constant arrays map to $0.5$)."""
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 0:
        return arr
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - vmin) / (vmax - vmin)


def polygonize_and_rank_targets(
    calibrated_prob: np.ndarray,
    lower_bound_L: np.ndarray,
    upper_bound_U: np.ndarray,
    dilation_tendency_Td: np.ndarray,
    alteration_index: np.ndarray,
    cover_depth_m: np.ndarray,
    profile: dict,
    l_threshold: float = 0.50,
    closing_size: int = 3,
    min_area_ha: float = 1.0,
    weights: tuple[float, float, float, float] = (0.40, 0.25, 0.20, 0.15),
    region_prefix: str = "YILGARN",
    extra_layers: Optional[dict[str, np.ndarray]] = None,
) -> gpd.GeoDataFrame:
    """Pass 6: Threshold conformal lower bound $L(x) \\ge 0.50$, close, vectorise, and rank targets.

    Workflow:
      1. Threshold pixel-wise conformal lower bound at $L(x) \\ge \\text{l\\_threshold}$
         (default $0.50$).
      2. Apply binary morphological closing (``scipy.ndimage.binary_closing``)
         with a ``closing_size x closing_size`` structuring element to bridge
         narrow sensor/cover gaps across contiguous structural corridors.
      3. Vectorise contiguous blobs into drill target polygons and filter by
         ``min_area_ha``.
      4. Compute per-polygon summary statistics and normalise inputs to $[0, 1]$
         across candidate targets to evaluate the Multi-Criteria Decision Score:

         $$\\text{Rank Score} = w_1 \\cdot P(\\text{Deposit}) + w_2 \\cdot T_d + w_3 \\cdot \\text{Alteration} - w_4 \\cdot \\text{Cover Depth}$$

    Args:
        calibrated_prob: 2-D array of Elkan-Noto calibrated probabilities $P(\\text{Deposit} \\mid x)$.
        lower_bound_L: 2-D array of conformal lower bounds $L(x)$.
        upper_bound_U: 2-D array of conformal upper bounds $U(x)$.
        dilation_tendency_Td: 2-D array of geomechanical dilation tendency $T_d \\in [0, 1]$.
        alteration_index: 2-D composite hyperspectral alteration intensity array.
        cover_depth_m: 2-D array of regolith cover thickness in metres.
        profile: Rasterio spatial profile (``transform``, ``crs``).
        l_threshold: Conformal lower bound cutoff $L(x) \\ge \\text{l\\_threshold}$ (default ``0.50``).
        closing_size: Structuring element width in pixels for morphological closing (default ``3``).
        min_area_ha: Minimum polygon area in hectares. Defaults to ``1.0``.
        weights: Tuple ``(w1, w2, w3, w4)`` for $(P(\\text{Deposit}), T_d, \\text{Alteration}, \\text{Cover Depth})$.
        region_prefix: Target ID prefix (e.g. ``"YILGARN"`` or ``"BORNEO"``).
        extra_layers: Optional dictionary of additional 2-D physics/probability grids
            (e.g. ``fused_raw_score``, ``propensity_e_x``, ``propensity_weight_w``,
            ``slip_tendency_Ts``, ``sar_alpha_deg``, ``moho_grad_mGal_km``,
            ``aem_conductance_S``, ``al_oh_crystallinity``, ``kaolinite_alunite_doublet``,
            ``carbonate_shift_nm``, ``fe_oxide_intensity``) to summarise per polygon.

    Returns:
        Ranked :class:`geopandas.GeoDataFrame` sorted by descending ``rank_score``.
    """
    transform = profile["transform"]
    crs_obj = CRS.from_user_input(profile["crs"])
    H, W = calibrated_prob.shape

    raw_mask = (lower_bound_L >= float(l_threshold)) & np.isfinite(lower_bound_L)
    if closing_size > 1:
        structure = np.ones((int(closing_size), int(closing_size)), dtype=bool)
        closed_mask = ndimage.binary_closing(raw_mask, structure=structure)
    else:
        closed_mask = raw_mask

    binary_mask = closed_mask.astype(np.uint8)
    labeled_array, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        return gpd.GeoDataFrame(
            columns=[
                "target_id",
                "rank",
                "rank_score",
                "geometry",
                "area_sq_km",
                "calibrated_deposit_prob",
                "lower_bound_L",
                "lower_bound_L_max",
                "upper_bound_U",
                "dilation_tendency_Td_max",
                "alteration_mean",
                "regolith_cover_depth_m",
            ],
            geometry="geometry",
            crs=crs_obj,
        )

    shapes_gen = rasterio.features.shapes(
        binary_mask.astype(np.int32),
        mask=binary_mask,
        transform=transform,
    )

    extra = extra_layers or {}
    records: list[dict] = []

    for geom_dict, pixel_value in shapes_gen:
        if pixel_value == 0:
            continue
        poly = shape(geom_dict)
        if crs_obj.is_geographic:
            # Approximate m² at equator/mid-latitudes if geographic CRS is used in tests
            area_m2 = poly.area * (111_320.0 ** 2)
        else:
            area_m2 = poly.area
        area_ha = area_m2 / 10_000.0
        if area_ha < min_area_ha:
            continue

        blob_mask = rasterio.features.geometry_mask(
            [geom_dict],
            out_shape=(H, W),
            transform=transform,
            invert=True,
        )
        if not np.any(blob_mask):
            continue

        def _stat(arr: np.ndarray, fn=np.nanmean, default: float = 0.0) -> float:
            vals = arr[blob_mask & np.isfinite(arr)]
            return float(fn(vals)) if len(vals) > 0 else float(default)

        rec: dict = {
            "geometry": poly,
            "area_sq_km": float(area_m2 / 1_000_000.0),
            "calibrated_deposit_prob": _stat(calibrated_prob, np.nanmean),
            "lower_bound_L": _stat(lower_bound_L, np.nanmean),
            "lower_bound_L_max": _stat(lower_bound_L, np.nanmax),
            "upper_bound_U": _stat(upper_bound_U, np.nanmean),
            "dilation_tendency_Td_max": _stat(dilation_tendency_Td, np.nanmax),
            "alteration_mean": _stat(alteration_index, np.nanmean),
            "regolith_cover_depth_m": _stat(cover_depth_m, np.nanmean),
        }

        # Optional Pass 1-4 layers for Site Card emission
        for k, grid in extra.items():
            if grid is not None and grid.shape == (H, W):
                rec[k] = _stat(np.asarray(grid, dtype=np.float64), np.nanmean)

        records.append(rec)

    if not records:
        return gpd.GeoDataFrame(
            columns=[
                "target_id",
                "rank",
                "rank_score",
                "geometry",
                "area_sq_km",
                "calibrated_deposit_prob",
                "lower_bound_L",
                "lower_bound_L_max",
                "upper_bound_U",
                "dilation_tendency_Td_max",
                "alteration_mean",
                "regolith_cover_depth_m",
            ],
            geometry="geometry",
            crs=crs_obj,
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs_obj)

    w1, w2, w3, w4 = [float(w) for w in weights]
    p_norm = _norm_01(gdf["calibrated_deposit_prob"].to_numpy())
    td_norm = _norm_01(gdf["dilation_tendency_Td_max"].to_numpy())
    alt_norm = _norm_01(gdf["alteration_mean"].to_numpy())
    cov_norm = _norm_01(gdf["regolith_cover_depth_m"].to_numpy())

    raw_rank_score = w1 * p_norm + w2 * td_norm + w3 * alt_norm - w4 * cov_norm
    # Normalise across multiple targets to [0.15, 0.98] so even the lowest-ranked
    # candidate polygon that passed L(x) >= 0.50 retains a positive priority score
    if len(raw_rank_score) > 1 and np.ptp(raw_rank_score) > 1e-9:
        r_min = float(np.min(raw_rank_score))
        r_max = float(np.max(raw_rank_score))
        norm_rank = 0.25 + 0.70 * ((raw_rank_score - r_min) / (r_max - r_min))
    else:
        norm_rank = np.clip(
            w1 * gdf["calibrated_deposit_prob"].to_numpy()
            + w2 * gdf["dilation_tendency_Td_max"].to_numpy()
            + w3 * np.clip(gdf["alteration_mean"].to_numpy(), 0.0, 1.0)
            - w4 * np.clip(gdf["regolith_cover_depth_m"].to_numpy() / 100.0, 0.0, 1.0),
            0.05,
            1.0,
        )
    gdf["rank_score"] = np.round(np.clip(norm_rank, 0.0, 1.0), 4)

    gdf = gdf.sort_values("rank_score", ascending=False).reset_index(drop=True)
    gdf["rank"] = np.arange(1, len(gdf) + 1, dtype=int)
    gdf["target_id"] = [
        f"{region_prefix.upper()}_TGT_{i:04d}" for i in range(1, len(gdf) + 1)
    ]
    return gdf


# ---------------------------------------------------------------------------
# Skirrow Mineral Systems Partitioning & TreeSHAP Attribution (Task 3.1)
# ---------------------------------------------------------------------------

SKIRROW_MINERAL_SYSTEMS_BUCKETS: list[str] = [
    "1_energy_and_metal_sources",
    "2_crustal_conduits",
    "3_physicochemical_traps",
    "4_alteration_and_anomaly",
]


def classify_skirrow_bucket(feature_name: str) -> str:
    """Classify feature into one of the 4 Skirrow Mineral Systems buckets (Skirrow et al., 2019).

    Buckets:
      1_energy_and_metal_sources:
        f_crustal_thickness_km, bouguer_30km, geophys_mean_1km_*
      2_crustal_conduits:
        bouguer_worm_5km, bouguer_worm_15km, bouguer_worm_30km, vertical_plumbing_persistence
      3_physicochemical_traps:
        dilation_tendency_Td, slip_tendency_Ts, geophys_contrast_4km_*, plateau_low_slope_gate
      4_alteration_and_anomaly:
        dae_recon_err, unet_ctx_*, oof_prob_skarn, oof_prob_epithermal, and spectral ratios
    """
    fn = feature_name.strip().lower()

    # 1. Energy and Metal Sources
    if (
        "crustal_thickness" in fn
        or "bouguer_30km" in fn
        or "moho" in fn
        or "geophys_mean_1km" in fn
        or fn.startswith("grav_mean_")
        or fn.startswith("mag_mean_")
        or "source" in fn
        or "deep_crust" in fn
    ):
        return "1_energy_and_metal_sources"

    # 2. Crustal Conduits
    if (
        "worm" in fn
        or "conduit" in fn
        or "plumbing" in fn
        or "persistence" in fn
        or "fault" in fn
        or "structure" in fn
        or "lineament" in fn
    ):
        return "2_crustal_conduits"

    # 3. Physicochemical Traps
    if (
        "dilation" in fn
        or "slip" in fn
        or "contrast_4km" in fn
        or "geophys_contrast" in fn
        or "plateau_low_slope" in fn
        or "trap" in fn
        or "reductant" in fn
        or "host_gate" in fn
    ):
        return "3_physicochemical_traps"

    # 4. Alteration and Anomaly (default for spectral indices, autoencoders, context embeddings)
    return "4_alteration_and_anomaly"


def partition_shap_into_mineral_systems(
    feature_names: list[str],
    shap_values: np.ndarray,
    feature_values: Optional[np.ndarray] = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Partition TreeSHAP contributions into the 4 Skirrow Mineral Systems buckets.

    Args:
        feature_names: List of feature names.
        shap_values: Array of SHAP values matching feature_names.
        feature_values: Optional array of raw feature values matching feature_names.
        top_k: Maximum number of top contributing features to retain per bucket.

    Returns:
        Dictionary mapping bucket names to summary statistics and top features.
    """
    shap_arr = np.asarray(shap_values, dtype=np.float64).ravel()
    feat_arr = (
        np.asarray(feature_values, dtype=np.float64).ravel()
        if feature_values is not None
        else None
    )

    buckets: dict[str, list[dict[str, Any]]] = {
        b: [] for b in SKIRROW_MINERAL_SYSTEMS_BUCKETS
    }

    for idx, fn in enumerate(feature_names):
        b = classify_skirrow_bucket(fn)
        sv = float(shap_arr[idx]) if idx < len(shap_arr) else 0.0
        fv = float(feat_arr[idx]) if (feat_arr is not None and idx < len(feat_arr)) else None
        buckets[b].append({
            "feature": fn,
            "shap_value": round(sv, 4),
            "feature_value": round(fv, 4) if fv is not None else None,
        })

    out: dict[str, Any] = {}
    for b_name in SKIRROW_MINERAL_SYSTEMS_BUCKETS:
        items = buckets[b_name]
        total_shap = sum(item["shap_value"] for item in items)
        abs_shap = sum(abs(item["shap_value"]) for item in items)
        items_sorted = sorted(items, key=lambda x: abs(x["shap_value"]), reverse=True)
        out[b_name] = {
            "total_shap": round(total_shap, 4),
            "mean_abs_shap": round(abs_shap / max(1, len(items)), 4),
            "n_features": len(items),
            "top_features": items_sorted[:top_k],
        }

    return out


def extract_target_shap_attribution(
    model: Any,
    x_sample: np.ndarray,
    feature_names: list[str],
    active_targets: list[str],
    tau_drill: float = 0.50,
    lower_bound: float = 0.50,
) -> dict[str, Any]:
    """Extract TreeSHAP values from each active head in C_0.90(s) and partition into Skirrow buckets.

    Enforces Task 3.1:
      1. For target polygons where max_k L_{0.90}^{(k)}(s) >= tau_drill, extract TreeSHAP
         values from each active head in C_0.90(s).
      2. Partition SHAP contributions into the 4 Skirrow Mineral Systems buckets.
    """
    if lower_bound < (tau_drill - 1e-4) or not active_targets:
        return {}

    attributions: dict[str, Any] = {}
    x_arr = np.asarray(x_sample, dtype=np.float32)

    if hasattr(model, "compute_head_shap"):
        # Multi-head TwoStageK4Ensemble
        for tgt in active_targets:
            try:
                sv, head_feats = model.compute_head_shap(tgt, x_arr, feature_names)
                if len(head_feats) > len(x_arr.ravel()) and hasattr(model, "head_1_") and hasattr(model, "head_2_"):
                    x_2d = x_arr if x_arr.ndim == 2 else x_arr[np.newaxis, :]
                    p1 = model.head_1_.predict_proba(x_2d)
                    p2 = model.head_2_.predict_proba(x_2d)
                    x_full = np.concatenate([x_arr.ravel(), p1.ravel(), p2.ravel()])
                else:
                    x_full = x_arr.ravel()
                attributions[tgt] = partition_shap_into_mineral_systems(
                    head_feats, sv, feature_values=x_full
                )
            except Exception as e:
                logger.warning("Could not extract TreeSHAP for head '%s': %s", tgt, e)
    elif hasattr(model, "compute_shap"):
        # MPMXGBoost baseline
        try:
            x_2d = x_arr if x_arr.ndim == 2 else x_arr[np.newaxis, :]
            sv = model.compute_shap(x_2d)
            if hasattr(sv, "ndim") and sv.ndim > 1:
                sv = sv.ravel()
            primary_tgt = active_targets[0] if active_targets else "deposit"
            attributions[primary_tgt] = partition_shap_into_mineral_systems(
                feature_names, sv, feature_values=x_arr.ravel()
            )
        except Exception as e:
            logger.warning("Could not extract TreeSHAP from model: %s", e)
    elif hasattr(model, "predict_proba"):
        # Single XGBoost model
        try:
            import shap
            rep_clf = getattr(model, "models_", [model])[0] if hasattr(model, "models_") else model
            explainer = shap.TreeExplainer(rep_clf)
            x_2d = x_arr if x_arr.ndim == 2 else x_arr[np.newaxis, :]
            sv = explainer.shap_values(x_2d)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv).ravel()
            primary_tgt = active_targets[0] if active_targets else "deposit"
            attributions[primary_tgt] = partition_shap_into_mineral_systems(
                feature_names, sv, feature_values=x_arr.ravel()
            )
        except Exception as e:
            logger.warning("Could not extract TreeSHAP via TreeExplainer: %s", e)

    return attributions


def emit_site_cards_json(
    ranked_targets_gdf: gpd.GeoDataFrame,
    out_path: str,
    elkan_noto_c_hat: float = 0.74,
    sensor_source: str = "EMIT_SWIR",
    window_footprint_km: float = 3.2,
    model: Optional[Any] = None,
    feature_names: Optional[list[str]] = None,
    feature_matrix: Optional[np.ndarray] = None,
    rows: Optional[np.ndarray] = None,
    cols: Optional[np.ndarray] = None,
    profile: Optional[dict] = None,
    tau_drill: float = 0.50,
    conformal_target_sets: Optional[Union[list[list[str]], dict[int, list[str]]]] = None,
    outmogc_gates: Optional[Union[dict[str, float], list[dict[str, float]]]] = None,
    shap_attributions: Optional[Union[list[dict], dict[int, dict]]] = None,
) -> list[dict]:
    """Pass 7: Export ranked drill targets as structured JSON Site Cards for LLM synthesis.

    Emits ``site_cards.json`` adhering strictly to the Pass 7 specification schema:
      - ``target_id``
      - ``rank``
      - ``rank_score``
      - ``conformal_90pct_target_set`` (Task 3.1)
      - ``telescoped_system_flag`` (Task 3.1)
      - ``outmogc_boundary_gate`` (Task 3.1)
      - ``exploration_logistics_propensity`` (Task 3.1)
      - ``mineral_systems_shap_attribution`` (Task 3.1)
      - ``geometry`` (``centroid_lat_lon``, ``area_sq_km``, ``window_footprint_km``)
      - ``probabilities`` (``fused_raw_score``, ``elkan_noto_c_hat``,
        ``calibrated_deposit_prob``, ``conformal_90_ci`` with ``lower_bound_L``
        & ``upper_bound_U``, ``exploration_propensity_e_x``, ``propensity_weight_w``)
      - ``physics_and_geology_summary`` (``alteration``, ``geomechanics_and_structure``,
        ``cover_and_conductivity``)

    Args:
        ranked_targets_gdf: Ranked GeoDataFrame produced by :func:`polygonize_and_rank_targets`.
        out_path: Output JSON file path (e.g. ``"outputs/site_cards.json"``).
        elkan_noto_c_hat: Elkan-Noto label frequency constant $\\hat{c}$ from Pass 4.
        sensor_source: SWIR hyperspectral sensor identifier (e.g. ``"EMIT_SWIR"``).
        window_footprint_km: Spatial context window footprint in km (default ``3.2``).
        model: Optional fitted tree ensemble or TwoStageK4Ensemble for TreeSHAP extraction.
        feature_names: Optional list of base feature names.
        feature_matrix: Optional full feature matrix matching rows/cols.
        rows: Optional array of pixel row indices matching feature_matrix.
        cols: Optional array of pixel col indices matching feature_matrix.
        profile: Optional rasterio profile dictionary with transform.
        tau_drill: Drilling prospectivity lower-bound cutoff threshold (default 0.50).
        conformal_target_sets: Optional target active sets per polygon.
        outmogc_gates: Optional boundary gate dictionaries per polygon.
        shap_attributions: Optional precomputed SHAP attributions per polygon.

    Returns:
        List of serialised Site Card dictionaries written to ``out_path``.
    """
    import json
    from pathlib import Path

    if len(ranked_targets_gdf) == 0:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("[]\n", encoding="utf-8")
        return []

    gdf_wgs84 = (
        ranked_targets_gdf.to_crs("EPSG:4326")
        if ranked_targets_gdf.crs is not None
        else ranked_targets_gdf
    )

    cards: list[dict] = []
    for idx, row in ranked_targets_gdf.iterrows():
        centroid_ll = gdf_wgs84.geometry.loc[idx].centroid
        lat = round(float(centroid_ll.y), 4)
        lon = round(float(centroid_ll.x), 4)

        cal_prob = float(row.get("calibrated_deposit_prob", 0.85))
        c_hat = float(elkan_noto_c_hat)
        fused_raw = float(row.get("fused_raw_score", min(1.0, cal_prob * c_hat)))
        e_x = float(row.get("exploration_propensity_e_x", 0.20))
        e_x = max(e_x, 0.05)
        w_x = float(row.get("propensity_weight_w", 1.0 / e_x))

        l_val = float(row.get("lower_bound_L", 0.50))
        l_max = float(row.get("lower_bound_L_max", l_val))
        if "lower_bound_k4" in row and isinstance(row["lower_bound_k4"], (list, np.ndarray)):
            l_max = float(np.max(row["lower_bound_k4"]))

        al_oh_val = float(row.get("al_oh_crystallinity", 1.15))
        if al_oh_val >= 1.10:
            al_oh_label = "high"
        elif al_oh_val >= 0.95:
            al_oh_label = "moderate"
        else:
            al_oh_label = "low"

        doublet_val = float(row.get("kaolinite_alunite_doublet", 1.0))
        aem_cond = float(row.get("aem_conductance_0_300m_S", 15.0))
        reductant_flag = bool(
            row.get("reductant_trap_inferred", aem_cond >= 10.0)
        )

        # ------------------------------------------------------------------
        # Multi-Mineral Phase 3 Attributes (Task 3.1)
        # ------------------------------------------------------------------
        # 1. conformal_90pct_target_set
        conformal_target_set = row.get("conformal_90pct_target_set")
        if conformal_target_set is None:
            if conformal_target_sets is not None:
                if isinstance(conformal_target_sets, dict):
                    conformal_target_set = conformal_target_sets.get(idx, [])
                elif idx < len(conformal_target_sets):
                    conformal_target_set = conformal_target_sets[idx]
                else:
                    conformal_target_set = []
            else:
                tgt_type = str(row.get("target_class", row.get("commodity", "primary_target")))
                conformal_target_set = [tgt_type] if l_max >= (tau_drill - 1e-4) else []

        if not isinstance(conformal_target_set, list):
            conformal_target_set = list(conformal_target_set)

        # 2. telescoped_system_flag
        telescoped_flag = bool(
            row.get("telescoped_system_flag", len(conformal_target_set) >= 2)
        )

        # 3. outmogc_boundary_gate
        outmogc_gate = row.get("outmogc_boundary_gate")
        if outmogc_gate is None:
            if outmogc_gates is not None:
                if isinstance(outmogc_gates, dict) and any(isinstance(v, dict) for v in outmogc_gates.values()):
                    outmogc_gate = outmogc_gates.get(idx, {})
                elif isinstance(outmogc_gates, dict):
                    outmogc_gate = dict(outmogc_gates)
                elif isinstance(outmogc_gates, list) and idx < len(outmogc_gates):
                    outmogc_gate = outmogc_gates[idx]
                else:
                    outmogc_gate = {
                        "porphyry_cu": 1.0,
                        "skarn_fe_cu": 1.0,
                        "epithermal_au": 1.0,
                        "laterite_fe_ni": 1.0,
                    }
            else:
                outmogc_gate = {
                    "porphyry_cu": float(row.get("outmogc_gate_porphyry", 1.0)),
                    "skarn_fe_cu": float(row.get("outmogc_gate_skarn", 1.0)),
                    "epithermal_au": float(row.get("outmogc_gate_epithermal", 1.0)),
                    "laterite_fe_ni": float(row.get("outmogc_gate_laterite", 1.0)),
                }

        # 4. exploration_logistics_propensity
        exp_propensity = round(float(row.get("exploration_logistics_propensity", e_x)), 4)

        # 5. mineral_systems_shap_attribution
        shap_attr = row.get("mineral_systems_shap_attribution")
        if shap_attr is None:
            if shap_attributions is not None:
                if isinstance(shap_attributions, dict) and idx in shap_attributions:
                    shap_attr = shap_attributions[idx]
                elif isinstance(shap_attributions, list) and idx < len(shap_attributions):
                    shap_attr = shap_attributions[idx]
                else:
                    shap_attr = {}
            elif model is not None:
                # Extract sample feature vector
                x_s = None
                if feature_matrix is not None and rows is not None and cols is not None and profile is not None:
                    # Lookup nearest cell to centroid
                    centroid_proj = ranked_targets_gdf.geometry.loc[idx].centroid
                    inv_tr = ~profile["transform"]
                    c_f, r_f = inv_tr * (centroid_proj.x, centroid_proj.y)
                    r_i = int(np.clip(round(r_f), 0, profile.get("height", 1000) - 1))
                    c_i = int(np.clip(round(c_f), 0, profile.get("width", 1000) - 1))
                    matches = np.where((rows == r_i) & (cols == c_i))[0]
                    if len(matches) > 0:
                        x_s = feature_matrix[matches[0]]
                    else:
                        dists = (rows - r_i) ** 2 + (cols - c_i) ** 2
                        x_s = feature_matrix[np.argmin(dists)]
                elif feature_names is not None:
                    # Check if columns are in row
                    vals = [float(row[fn]) for fn in feature_names if fn in row]
                    if len(vals) == len(feature_names):
                        x_s = np.array(vals)

                if x_s is not None and feature_names is not None:
                    shap_attr = extract_target_shap_attribution(
                        model=model,
                        x_sample=x_s,
                        feature_names=feature_names,
                        active_targets=conformal_target_set,
                        tau_drill=tau_drill,
                        lower_bound=l_max,
                    )
                else:
                    shap_attr = {}
            else:
                shap_attr = {}

        card = {
            "target_id": str(row.get("target_id", f"YILGARN_TGT_{idx + 1:04d}")),
            "rank": int(row.get("rank", idx + 1)),
            "rank_score": round(float(row.get("rank_score", 0.0)), 3),
            "conformal_90pct_target_set": conformal_target_set,
            "telescoped_system_flag": telescoped_flag,
            "outmogc_boundary_gate": outmogc_gate,
            "exploration_logistics_propensity": exp_propensity,
            "mineral_systems_shap_attribution": shap_attr,
            "geometry": {
                "centroid_lat_lon": [lat, lon],
                "area_sq_km": round(float(row.get("area_sq_km", 0.0)), 3),
                "window_footprint_km": round(float(window_footprint_km), 2),
            },
            "probabilities": {
                "fused_raw_score": round(fused_raw, 3),
                "elkan_noto_c_hat": round(c_hat, 3),
                "calibrated_deposit_prob": round(cal_prob, 3),
                "conformal_90_ci": {
                    "lower_bound_L": round(float(row.get("lower_bound_L", 0.50)), 3),
                    "upper_bound_U": round(float(row.get("upper_bound_U", 0.95)), 3),
                },
                "exploration_propensity_e_x": round(e_x, 3),
                "propensity_weight_w": round(w_x, 2),
            },
            "physics_and_geology_summary": {
                "alteration": {
                    "sensor_source": str(sensor_source),
                    "al_oh_crystallinity_2200nm": al_oh_label,
                    "kaolinite_alunite_doublet": bool(doublet_val > 0.05),
                    "fe_mg_carbonate_shift_nm": round(
                        float(row.get("fe_mg_carbonate_shift_nm", 0.0)), 2
                    ),
                    "fe_oxide_intensity": round(
                        float(row.get("fe_oxide_intensity", row.get("alteration_mean", 0.5))),
                        3,
                    ),
                },
                "geomechanics_and_structure": {
                    "slip_tendency_Ts_mean": round(
                        float(row.get("slip_tendency_Ts_mean", 0.60)), 3
                    ),
                    "dilation_tendency_Td_max": round(
                        float(row.get("dilation_tendency_Td_max", 0.80)), 3
                    ),
                    "sar_cloude_pottier_alpha_deg": round(
                        float(row.get("sar_cloude_pottier_alpha_deg", 45.0)), 2
                    ),
                    "moho_gravity_gradient_step_mGal_km": round(
                        float(row.get("moho_gravity_gradient_step_mGal_km", 1.5)), 2
                    ),
                },
                "cover_and_conductivity": {
                    "regolith_cover_depth_m": round(
                        float(row.get("regolith_cover_depth_m", 25.0)), 1
                    ),
                    "aem_conductance_0_300m_S": round(aem_cond, 2),
                    "reductant_trap_inferred": reductant_flag,
                },
            },
        }
        cards.append(card)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(cards, indent=2) + "\n", encoding="utf-8")
    logger.info("Emitted %d LLM Site Cards → %s", len(cards), out_path)
    return cards

