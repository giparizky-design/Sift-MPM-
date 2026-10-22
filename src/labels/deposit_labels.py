"""Deposit label preparation for Positive-Unlabeled (PU) learning.

PU Framing
----------
In Mineral Prospectivity Mapping the training signal is inherently incomplete:

  P  (Positive)          – Confirmed economic Au deposits from the MINEDEX
                           registry (contained metal ≥ threshold).  These cells
                           are *known* mineral occurrences and labelled 1.

  RN (Reliable Negative) – Drillhole collars from WAMEX where deep, complete
                           intersections returned low-grade or barren results.
                           These cells are *geologically tested* and labelled 0.

  U  (Unlabeled)         – Everything else.  May contain undiscovered deposits
                           (false negatives in a standard supervised framing).
                           Labelled -1 and treated as unlabeled in PU learning.

The U set is **not** treated as negative in model training.  PU learning
algorithms (e.g., Mordelet & Vert 2014; Liu et al. 2003) or semi-supervised
classifiers are applied downstream to handle the unknown status of U cells.

References
----------
Mordelet & Vert (2014) "A bagging SVM to learn from positive and unlabeled
    examples." Pattern Recognition Letters.
Liu et al. (2003) "Building text classifiers using positive and unlabeled
    examples." ICDM.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.transform
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-wide target CRS (GDA2020 / MGA zone 51, EPSG:7851 – Eastern Goldfields, ~120-126E)
# Override via environment or pass profile explicitly.
# ---------------------------------------------------------------------------
_TARGET_EPSG: int = 7851  # GDA2020 / MGA zone 51


def _target_crs() -> CRS:
    """Return the project-level target CRS."""
    return CRS.from_epsg(_TARGET_EPSG)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def load_minedex(
    path: str,
    min_au_koz: float = 100.0,
) -> gpd.GeoDataFrame:
    """Load and filter MINEDEX mineral occurrences to economic Au deposits.

    MINEDEX is the Western Australian mineral occurrences database maintained
    by DMIRS.  Only deposits with contained gold ≥ *min_au_koz* thousand
    troy ounces are retained; sub-economic occurrences and prospects are
    excluded because they introduce ambiguous positives into the P set.

    Args:
        path: Absolute or relative path to a MINEDEX CSV or Shapefile.
              CSV files must contain columns 'LONGITUDE', 'LATITUDE' (WGS-84)
              or 'EASTING', 'NORTHING' plus 'ZONE' (GDA94 MGA).
              Shapefiles are read directly.
        min_au_koz: Minimum contained gold in thousand troy ounces (koz).
                    Defaults to 100 koz — a common "significant deposit"
                    threshold in WA exploration literature.

    Returns:
        GeoDataFrame with columns:
            - geometry        : Point in target CRS
            - deposit_name    : Name of the deposit
            - au_koz          : Contained Au (koz)
            - classification  : always "P"

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required columns are missing from the source file.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"MINEDEX source not found: {src}")

    suffix = src.suffix.lower()

    # ------------------------------------------------------------------
    # 1.  Read raw data
    # ------------------------------------------------------------------
    if suffix in {".shp", ".gpkg", ".geojson", ".json"}:
        gdf: gpd.GeoDataFrame = gpd.read_file(str(src))
    elif suffix == ".csv":
        df = pd.read_csv(str(src))
        gdf = _csv_to_geodataframe(df, src)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # ------------------------------------------------------------------
    # 2.  Normalise column names (case-insensitive)
    # ------------------------------------------------------------------
    gdf.columns = [c.strip().lower() for c in gdf.columns]

    # Attempt to find the Au koz column from common MINEDEX naming variants
    au_col_candidates = [
        "au_koz", "au_contained_koz", "au_koz_contained", "gold_koz", "contained_au_koz",
        "au_resource_koz", "resource_au_koz", "gold_contained_koz",
    ]
    au_col: Optional[str] = None
    for cand in au_col_candidates:
        if cand in gdf.columns:
            au_col = cand
            break

    # If official DMIRS layer without direct contained koz column, filter by site stage/type
    if au_col is None and ("site_stage" in gdf.columns or "site_type_" in gdf.columns):
        logger.info("MINEDEX source contains DMIRS site attributes; applying stage/type filter.")
        if "site_type_" in gdf.columns:
            gdf = gdf[~gdf["site_type_"].str.lower().isin(["infrastructure", "other"])].copy()
        if "site_stage" in gdf.columns:
            valid_stages = {"operating", "shut", "care and maintenance", "under development", "undeveloped"}
            gdf = gdf[gdf["site_stage"].str.lower().isin(valid_stages)].copy()
        gdf["au_koz"] = 100.0  # Assign baseline significant deposit proxy
        au_col = "au_koz"

    if au_col is None:
        raise ValueError(
            f"Cannot find Au koz column in MINEDEX data.  "
            f"Expected one of: {au_col_candidates}.  "
            f"Found: {list(gdf.columns)}"
        )

    # Attempt to find deposit name column
    name_col_candidates = [
        "deposit_name", "name", "site_name", "occurrence_name", "deposit", "site_title", "short_name",
    ]
    name_col: Optional[str] = None
    for cand in name_col_candidates:
        if cand in gdf.columns:
            name_col = cand
            break

    # ------------------------------------------------------------------
    # 3.  Filter to economic Au deposits
    # ------------------------------------------------------------------
    gdf[au_col] = pd.to_numeric(gdf[au_col], errors="coerce")
    before = len(gdf)
    gdf = gdf[gdf[au_col] >= min_au_koz].copy()
    after = len(gdf)
    logger.info(
        "MINEDEX filter: retained %d / %d deposits with Au ≥ %.1f koz",
        after, before, min_au_koz,
    )

    # ------------------------------------------------------------------
    # 4.  Reproject to target CRS
    # ------------------------------------------------------------------
    if gdf.crs is None:
        warnings.warn(
            "MINEDEX GeoDataFrame has no CRS set.  Assuming WGS-84 (EPSG:4326).",
            stacklevel=2,
        )
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs(_target_crs())

    # ------------------------------------------------------------------
    # 5.  Build clean output GeoDataFrame
    # ------------------------------------------------------------------
    result = gpd.GeoDataFrame(
        {
            "geometry": gdf.geometry,
            "deposit_name": gdf[name_col] if name_col else "unknown",
            "au_koz": gdf[au_col].values,
            "classification": "P",
        },
        crs=_target_crs(),
    )

    # Warn about coordinate precision before returning
    check_coordinate_precision(result)
    return result


def _csv_to_geodataframe(df: pd.DataFrame, src: Path) -> gpd.GeoDataFrame:
    """Convert a raw MINEDEX CSV DataFrame to a GeoDataFrame.

    Args:
        df:  Raw DataFrame from ``pd.read_csv``.
        src: Source path (used for error messages only).

    Returns:
        GeoDataFrame with geometry set from coordinate columns.

    Raises:
        ValueError: If no recognisable coordinate columns are found.
    """
    cols_lower = {c.strip().lower(): c for c in df.columns}
    df.columns = [c.strip().lower() for c in df.columns]

    # WGS-84 lon/lat
    if "longitude" in cols_lower and "latitude" in cols_lower:
        df = df.copy()
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df = df.dropna(subset=["longitude", "latitude"])
        geometry = [Point(x, y) for x, y in zip(df["longitude"], df["latitude"])]
        return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # MGA easting / northing
    if "easting" in cols_lower and "northing" in cols_lower:
        df = df.copy()
        df["easting"] = pd.to_numeric(df["easting"], errors="coerce")
        df["northing"] = pd.to_numeric(df["northing"], errors="coerce")
        df = df.dropna(subset=["easting", "northing"])
        geometry = [Point(x, y) for x, y in zip(df["easting"], df["northing"])]
        # Determine zone; if absent assume the project zone but warn loudly, because a wrong
        # zone silently shifts coordinates by hundreds of km.
        if "zone" in df.columns:
            zone = int(df["zone"].iloc[0])
        else:
            zone = _TARGET_EPSG - 7800
            warnings.warn(
                f"No 'zone' column in {src}; assuming MGA zone {zone} (EPSG:{_TARGET_EPSG}).",
                stacklevel=2,
            )
        # GDA2020 MGA zones are EPSG:7846-7859 (zone 46 -> 7846 ... zone 59 -> 7859).
        # Legacy GDA94 collars (EPSG:283xx) can be requested with datum='gda94' in a future loader.
        epsg = 7800 + zone
        return gpd.GeoDataFrame(df, geometry=geometry, crs=CRS.from_epsg(epsg))

    raise ValueError(
        f"Cannot identify coordinate columns in {src}.  "
        "Expected 'LONGITUDE'/'LATITUDE' or 'EASTING'/'NORTHING'."
    )


# ---------------------------------------------------------------------------


def load_wamex_drillholes(path: str) -> gpd.GeoDataFrame:
    """Load WAMEX drillhole collar locations from a CSV export.

    WAMEX (Western Australian Mineral Exploration) open-file reports are
    released by DMIRS.  Collar CSVs record the surface position of each
    drillhole and are the spatial anchor for joining assay data.

    The function accepts both:
      * Geographic coordinates – columns 'LAT' / 'LONG' (decimal degrees,
        GDA94/WGS-84).
      * MGA projected coordinates – columns 'EASTING', 'NORTHING', 'ZONE'.

    Args:
        path: Path to the WAMEX collar CSV file.

    Returns:
        GeoDataFrame in target CRS with columns preserved from the source
        plus a normalised 'hole_id' column (lower-cased 'holeid' or 'hole_id').

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If coordinate columns cannot be found.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"WAMEX collar file not found: {src}")

    suffix = src.suffix.lower()
    if suffix in {".shp", ".gpkg", ".geojson", ".json"}:
        gdf_raw = gpd.read_file(str(src))
        df = pd.DataFrame(gdf_raw.drop(columns="geometry"))
        df.columns = [c.strip().lower() for c in df.columns]
        for candidate in ("holeid", "hole_id", "dh_id", "drillhole_id", "collar_id"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "hole_id"})
                break
        else:
            df["hole_id"] = [f"DH_{i:06d}" for i in range(len(df))]
        gdf = gpd.GeoDataFrame(df, geometry=gdf_raw.geometry, crs=gdf_raw.crs)
        gdf = gdf.to_crs(_target_crs())
        logger.info("Loaded %d WAMEX drillhole collars from %s", len(gdf), src)
        return gdf

    df = pd.read_csv(str(src))
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise hole_id column name
    for candidate in ("holeid", "hole_id", "dh_id", "drillhole_id", "collar_id"):
        if candidate in df.columns:
            df = df.rename(columns={candidate: "hole_id"})
            break
    else:
        # If no recognised ID column, fabricate sequential IDs
        warnings.warn(
            "No 'hole_id' column found in WAMEX collar CSV – generating sequential IDs.",
            stacklevel=2,
        )
        df["hole_id"] = [f"DH_{i:06d}" for i in range(len(df))]

    # Build GeoDataFrame from coordinates
    gdf = _csv_to_geodataframe(df, src)

    # Reproject to target CRS
    gdf = gdf.to_crs(_target_crs())

    logger.info("Loaded %d WAMEX drillhole collars from %s", len(gdf), src)
    return gdf


# ---------------------------------------------------------------------------


def classify_drillhole_cells(
    drillholes: gpd.GeoDataFrame,
    assay_path: str,
    min_depth_m: float = 50.0,
    barren_threshold_gt: float = 0.05,
) -> gpd.GeoDataFrame:
    """Identify reliable negative (RN) drillhole collars from assay data.

    A drillhole qualifies as a Reliable Negative when:

    1. ``max_depth ≥ min_depth_m`` – shallow holes (<50 m) may have missed
       deeper mineralisation and are excluded to avoid false negatives.
    2. ``best_au ≤ barren_threshold_gt`` – the peak Au grade (g/t) in any
       assay interval must be below the barren threshold.  The default 0.05 g/t
       is comfortably below the ~0.5–1.0 g/t cut-off for economic ore in WA
       open-pit operations, providing a safety margin.

    Args:
        drillholes:         GeoDataFrame of collar locations (from
                            :func:`load_wamex_drillholes`).
        assay_path:         Path to the assay CSV.  Required columns:
                            ``hole_id``, ``depth_from`` (m), ``depth_to`` (m),
                            ``au_gt`` (Au grade in g/t).
        min_depth_m:        Minimum total drilled depth to accept a hole as
                            adequately tested.  Defaults to 50 m.
        barren_threshold_gt: Maximum peak Au grade (g/t) for a hole to be
                            considered barren.  Defaults to 0.05 g/t.

    Returns:
        GeoDataFrame subset of *drillholes* that satisfy both criteria, with
        an added ``classification`` column set to "RN".

    Raises:
        FileNotFoundError: If *assay_path* does not exist.
        ValueError: If required columns are missing from the assay table.
    """
    assay_src = Path(assay_path)
    if not assay_src.exists():
        raise FileNotFoundError(f"Assay file not found: {assay_src}")

    assay = pd.read_csv(str(assay_src))
    assay.columns = [c.strip().lower() for c in assay.columns]

    # Normalise column names in assay table
    col_map: dict[str, list[str]] = {
        "hole_id":    ["hole_id", "holeid", "dh_id", "drillhole_id"],
        "depth_from": ["depth_from", "from", "from_m", "depth_from_m", "frm"],
        "depth_to":   ["depth_to", "to", "to_m", "depth_to_m"],
        "au_gt":      ["au_gt", "au_g_t", "au", "gold_gt", "gold_g_t", "au_ppm"],
    }
    renamed: dict[str, str] = {}
    for std_name, candidates in col_map.items():
        for cand in candidates:
            if cand in assay.columns:
                renamed[cand] = std_name
                break
        else:
            raise ValueError(
                f"Assay table is missing column '{std_name}'.  "
                f"Tried: {candidates}.  Found: {list(assay.columns)}"
            )
    assay = assay.rename(columns=renamed)

    # Coerce numeric
    for col in ("depth_from", "depth_to", "au_gt"):
        assay[col] = pd.to_numeric(assay[col], errors="coerce")

    # Compute per-hole summary statistics
    summary = (
        assay.groupby("hole_id")
        .agg(
            max_depth=("depth_to", "max"),
            best_au=("au_gt", "max"),
        )
        .reset_index()
    )

    # Apply qualification criteria
    qualified = summary[
        (summary["max_depth"] >= min_depth_m)
        & (summary["best_au"] <= barren_threshold_gt)
    ]

    logger.info(
        "RN qualification: %d / %d holes pass depth ≥ %.1f m AND Au ≤ %.3f g/t",
        len(qualified),
        len(summary),
        min_depth_m,
        barren_threshold_gt,
    )

    # Join back to spatial collars
    # Normalise hole_id in collars too
    drillholes = drillholes.copy()
    drillholes["hole_id"] = drillholes["hole_id"].astype(str).str.strip().str.lower()
    qualified["hole_id"] = qualified["hole_id"].astype(str).str.strip().str.lower()

    rn_gdf = drillholes[drillholes["hole_id"].isin(qualified["hole_id"])].copy()
    rn_gdf["classification"] = "RN"

    return rn_gdf


# ---------------------------------------------------------------------------


def rasterise_labels(
    positives: gpd.GeoDataFrame,
    reliable_negatives: gpd.GeoDataFrame,
    target_profile: dict,
    buffer_p_m: float = 500.0,
    exclusion_buffer_m: float = 2000.0,
) -> np.ndarray:
    """Rasterise P and RN labels to a 2-D label array.

    Spatial decisions
    -----------------
    * **Positive buffer** (``buffer_p_m``): A circular buffer of radius
      *buffer_p_m* is applied to each P centroid before rasterisation.
      This reflects the spatial extent of the alteration halo expected
      around an economic deposit (hydrothermal, skarn, etc.) and ensures
      that cells immediately adjacent to a known deposit are not
      misclassified as negative.

    * **Exclusion buffer** (``exclusion_buffer_m``): Any RN cell that falls
      within *exclusion_buffer_m* of a P point is demoted to U (−1).
      The rationale is that the distal halo of a deposit may extend several
      km; drillholes within this radius may have intersected the halo but
      missed the ore shoot, making them unreliable negatives.

    Array encoding
    --------------
    ::

         1  → P  (confirmed deposit cell)
         0  → RN (verified barren cell)
        -1  → U  (unexplored / unknown)

    Args:
        positives:          GeoDataFrame of positive deposits (P).
        reliable_negatives: GeoDataFrame of reliable negative drill sites (RN).
        target_profile:     Rasterio-style profile dict containing at minimum
                            ``width``, ``height``, ``transform``, and ``crs``.
        buffer_p_m:         Radius in metres to buffer positive centroids.
        exclusion_buffer_m: Exclusion radius in metres around P centroids; RN
                            cells within this radius are set to U.

    Returns:
        Integer NumPy array of shape ``(height, width)`` with values in
        {-1, 0, 1}.
    """
    height: int = target_profile["height"]
    width: int = target_profile["width"]
    transform: rasterio.transform.Affine = target_profile["transform"]
    crs: CRS = CRS.from_user_input(target_profile["crs"])

    # Ensure inputs are in target CRS
    if positives.crs != crs:
        positives = positives.to_crs(crs)
    if reliable_negatives.crs != crs:
        reliable_negatives = reliable_negatives.to_crs(crs)

    # ------------------------------------------------------------------ #
    # Initialise everything to U = -1
    # ------------------------------------------------------------------ #
    label_array = np.full((height, width), -1, dtype=np.int8)

    # ------------------------------------------------------------------ #
    # Rasterise RN (0) first so P overwrites where there is overlap
    # ------------------------------------------------------------------ #
    if len(reliable_negatives) > 0:
        rn_buffered = reliable_negatives.copy()
        rn_radius_m = max(25.0, abs(transform.a) * 0.75)
        rn_buffered.geometry = reliable_negatives.geometry.buffer(rn_radius_m)
        rn_shapes = [
            (geom, 0)
            for geom in rn_buffered.geometry
            if geom is not None and not geom.is_empty
        ]
        if rn_shapes:
            rn_raster = rasterio.features.rasterize(
                shapes=rn_shapes,
                out_shape=(height, width),
                transform=transform,
                all_touched=True,
                fill=-1,
                dtype=np.int8,
            )
            # Only overwrite U cells
            label_array[rn_raster == 0] = 0

    # ------------------------------------------------------------------ #
    # Buffer positive centroids and rasterise as 1
    # ------------------------------------------------------------------ #
    if len(positives) > 0:
        p_buffered = positives.copy()
        p_buffered.geometry = positives.geometry.buffer(buffer_p_m)
        p_shapes = [
            (geom, 1)
            for geom in p_buffered.geometry
            if geom is not None and not geom.is_empty
        ]
        if p_shapes:
            p_raster = rasterio.features.rasterize(
                shapes=p_shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.int8,
            )
            label_array[p_raster == 1] = 1

    # ------------------------------------------------------------------ #
    # Exclusion buffer: demote RN → U within exclusion_buffer_m of P
    # ------------------------------------------------------------------ #
    if len(positives) > 0:
        # Pixel resolution in metres (average of x/y pixel size)
        pixel_size_m = abs(transform.a)  # assumes square pixels

        # Build a binary mask of P cells (after buffering) and compute EDT
        p_mask = (label_array == 1).astype(np.uint8)
        # Distance transform gives distance from nearest P=1 cell in pixels
        dist_from_p_px = distance_transform_edt(1 - p_mask)
        dist_from_p_m = dist_from_p_px * pixel_size_m

        # RN cells too close to P → U
        rn_mask = label_array == 0
        exclusion_mask = (dist_from_p_m <= exclusion_buffer_m) & rn_mask
        label_array[exclusion_mask] = -1

        n_excluded = int(exclusion_mask.sum())
        logger.info(
            "Exclusion buffer: demoted %d RN cells to U (within %.0f m of P)",
            n_excluded, exclusion_buffer_m,
        )

    summary = label_summary(label_array)
    logger.info("Label raster summary: %s", summary)
    return label_array


# ---------------------------------------------------------------------------


def label_summary(label_array: np.ndarray) -> dict:
    """Compute count and fraction of each label class in the label array.

    Args:
        label_array: Integer array with values in {1 (P), 0 (RN), -1 (U)}.

    Returns:
        Dictionary with keys:
            - ``n_positive``   : count of P cells
            - ``n_reliable_negative``: count of RN cells
            - ``n_unlabeled``  : count of U cells
            - ``frac_positive``: fraction of all cells that are P
            - ``frac_reliable_negative``: fraction that are RN
            - ``frac_unlabeled``: fraction that are U
            - ``total_cells``  : total cell count
    """
    total = label_array.size
    n_p = int(np.sum(label_array == 1))
    n_rn = int(np.sum(label_array == 0))
    n_u = int(np.sum(label_array == -1))

    return {
        "n_positive": n_p,
        "n_reliable_negative": n_rn,
        "n_unlabeled": n_u,
        "frac_positive": n_p / total,
        "frac_reliable_negative": n_rn / total,
        "frac_unlabeled": n_u / total,
        "total_cells": total,
    }


# ---------------------------------------------------------------------------


def check_coordinate_precision(
    gdf: gpd.GeoDataFrame,
    expected_precision_m: float = 100.0,
) -> None:
    """Warn if geometries appear to have been rounded to low coordinate precision.

    In a projected CRS, coordinates are in metres.  If any coordinate value is
    an integer (e.g., 6 650 000 instead of 6 650 234.71), it strongly suggests
    the data was rounded – possibly after a lossy conversion through a GIS
    that stored decimal-degree coordinates with insufficient significant figures.

    This is a known data quality issue with older MINEDEX and WAMEX releases
    where deposit locations were digitised from paper 1:100 000 maps.

    No exception is raised; only :mod:`logging` warnings are emitted so that
    downstream processing is not interrupted.

    Args:
        gdf:                   GeoDataFrame to inspect.
        expected_precision_m:  The precision threshold in metres.  Coordinates
                               whose fractional part is zero (integer metres)
                               trigger the warning, as rounding to the nearest
                               metre already exceeds expected_precision_m=100.
    """
    if gdf is None or len(gdf) == 0:
        return

    if gdf.crs is None:
        logger.warning(
            "check_coordinate_precision: GeoDataFrame has no CRS – skipping check."
        )
        return

    low_precision_ids: list[int] = []
    for idx, geom in gdf.geometry.items():
        if geom is None or geom.is_empty:
            continue
        # Use the centroid for simplicity (works for Point and Polygon)
        cx, cy = geom.centroid.x, geom.centroid.y

        if gdf.crs.is_projected:
            # Projected CRS: check if coordinate is integer or near-integer
            # (rounded to nearest metre or coarser)
            x_frac = abs(cx - round(cx, 0))
            y_frac = abs(cy - round(cy, 0))
            if x_frac == 0.0 or y_frac == 0.0:
                low_precision_ids.append(idx)
        else:
            # Geographic CRS (degrees): check decimal places
            # 1 arc-second ≈ 30 m, so < 4 decimal places → > 10 m uncertainty
            # For expected_precision_m=100 m we need ≥ 3 decimal places
            x_decimals = len(str(cx).rstrip("0").split(".")[-1]) if "." in str(cx) else 0
            y_decimals = len(str(cy).rstrip("0").split(".")[-1]) if "." in str(cy) else 0
            # Approximate: 1 degree ≈ 111 km, so precision_degrees ≈ expected_precision_m / 111000
            min_decimals = max(1, int(np.ceil(-np.log10(expected_precision_m / 111_000))))
            if x_decimals < min_decimals or y_decimals < min_decimals:
                low_precision_ids.append(idx)

    if low_precision_ids:
        logger.warning(
            "check_coordinate_precision: %d / %d features appear to have "
            "coordinate precision worse than %.0f m.  Feature indices: %s%s",
            len(low_precision_ids),
            len(gdf),
            expected_precision_m,
            str(low_precision_ids[:10]),
            " ..." if len(low_precision_ids) > 10 else "",
        )
    else:
        logger.debug(
            "check_coordinate_precision: all %d features pass precision check (≥ %.0f m).",
            len(gdf),
            expected_precision_m,
        )
