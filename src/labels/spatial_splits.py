"""Leakage-resistant spatial block cross-validation for MPM.

Spatial Autocorrelation Leakage
--------------------------------
Standard random k-fold cross-validation assigns geographically proximate
cells to both training and validation sets.  Because geophysical features
(magnetics, gravity, geochemistry) are spatially autocorrelated at distances
of kilometres to tens of kilometres, a model trained on cells immediately
adjacent to its validation set leaks spatial structure through the features
rather than the labels.  The result is optimistic performance estimates that
do not generalise to unseen geological terranes.

Spatial Block Cross-Validation (Roberts et al. 2017) partitions the study
area into contiguous rectangular blocks that are larger than the range of
spatial autocorrelation, then assigns whole blocks to either train or
validation.  This ensures the minimum distance between training and
validation observations is approximately one block width, suppressing
autocorrelation leakage.

Block size guidance: the block should be at least as large as the
variogram range of the most correlated predictor.  For W.A. regional
geophysics this is typically 5–20 km; a safe default is ~10 km.

Blind Holdout
-------------
In addition to CV folds, a geographically isolated "blind holdout" area
is reserved completely from training.  This area – typically a poorly-
explored sub-region or a district discovered after the training period –
provides an unbiased estimate of true generalisation performance.

References
----------
Roberts, D.R., Bahn, V., Ciuti, S., Boyce, M.S., Elith, J., Guillera-
    Arroita, G., ... & Dormann, C.F. (2017). Cross-validation strategies
    for data with temporal, spatial, hierarchical, or phylogenetic
    structure. Ecography, 40(8), 913-929.
"""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.transform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block assignment
# ---------------------------------------------------------------------------


def assign_spatial_blocks(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    H: int,
    W: int,
    block_size_px: int,
) -> np.ndarray:
    """Assign each raster cell to a rectangular spatial block.

    Blocks are defined by integer division of the (row, col) indices by
    ``block_size_px``.  The block ID is encoded as a single integer:

        block_id = (row // block_size_px) * n_col_blocks + (col // block_size_px)

    This gives a unique, stable ID for each block regardless of array shape.

    Args:
        row_idx:        1-D array of row indices for each sample, shape (n,).
        col_idx:        1-D array of column indices for each sample, shape (n,).
        H:              Raster height (number of rows).
        W:              Raster width (number of columns).
        block_size_px:  Side length of each square block in pixels.
                        Should correspond to the autocorrelation range in
                        pixels (e.g., if pixel size = 100 m and range = 5 km,
                        use block_size_px = 50).

    Returns:
        1-D integer array of shape (n,) containing the block ID for each
        input cell.

    Raises:
        ValueError: If ``block_size_px`` ≤ 0 or arrays have mismatched lengths.
    """
    if block_size_px <= 0:
        raise ValueError(f"block_size_px must be > 0, got {block_size_px}")
    if row_idx.shape != col_idx.shape:
        raise ValueError(
            f"row_idx and col_idx must have the same shape, "
            f"got {row_idx.shape} and {col_idx.shape}"
        )

    n_col_blocks = max(1, int(np.ceil(W / block_size_px)))

    block_row = row_idx // block_size_px
    block_col = col_idx // block_size_px
    block_ids = block_row * n_col_blocks + block_col

    n_unique = len(np.unique(block_ids))
    logger.info(
        "assign_spatial_blocks: %d unique blocks from %d samples "
        "(block_size_px=%d, H=%d, W=%d)",
        n_unique, len(row_idx), block_size_px, H, W,
    )
    return block_ids.astype(np.int64)


# ---------------------------------------------------------------------------
# Spatial block k-fold
# ---------------------------------------------------------------------------


def spatial_block_kfold(
    block_ids: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group spatial blocks into k cross-validation folds.

    Each block is assigned to exactly one fold.  All samples within a block
    go to train or validation together, ensuring no block is split.  Blocks
    are shuffled before assignment so that geographically adjacent blocks are
    not always in the same fold (which would bias the spatial coverage of
    validation sets).

    This is equivalent to ``GroupKFold`` from scikit-learn but with explicit
    shuffling of block order.

    Args:
        block_ids:    1-D integer array of block IDs per sample, shape (n,).
                      From :func:`assign_spatial_blocks`.
        n_folds:      Number of cross-validation folds.  Must be ≤ number of
                      unique blocks.
        random_state: Random seed for block shuffling.

    Returns:
        List of ``n_folds`` tuples ``(train_indices, val_indices)`` where both
        are 1-D integer arrays of sample indices (not block IDs).

    Raises:
        ValueError: If n_folds > number of unique blocks.
    """
    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)

    if n_folds > n_blocks:
        raise ValueError(
            f"n_folds ({n_folds}) must be ≤ number of unique blocks ({n_blocks})."
        )

    rng = np.random.default_rng(random_state)
    shuffled_blocks = unique_blocks.copy()
    rng.shuffle(shuffled_blocks)

    # Assign each block to a fold (round-robin)
    block_to_fold: dict[int, int] = {
        int(b): int(i % n_folds)
        for i, b in enumerate(shuffled_blocks)
    }

    fold_assignments = np.array([block_to_fold[int(bid)] for bid in block_ids])

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(block_ids))

    for fold in range(n_folds):
        val_mask = fold_assignments == fold
        train_indices = all_indices[~val_mask]
        val_indices = all_indices[val_mask]

        logger.info(
            "Fold %d/%d: train=%d samples (%d blocks), val=%d samples (%d blocks)",
            fold + 1, n_folds,
            len(train_indices),
            int(np.sum([block_to_fold[int(b)] != fold for b in unique_blocks])),
            len(val_indices),
            int(np.sum([block_to_fold[int(b)] == fold for b in unique_blocks])),
        )
        splits.append((train_indices, val_indices))

    return splits


# ---------------------------------------------------------------------------
# Mining camp integrity enforcement
# ---------------------------------------------------------------------------


def ensure_camp_integrity(
    block_ids: np.ndarray,
    camp_polygons: gpd.GeoDataFrame,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    profile: dict,
) -> np.ndarray:
    """Override block IDs so cells within a mining camp polygon share one block.

    Mining camps (e.g., Kalgoorlie–Boulder, Meekatharra, Laverton) represent
    dense clusters of historical workings and modern production areas.  Splitting
    a camp across training and validation folds would allow the model to learn
    camp-specific spatial patterns in training that directly predict camp cells
    in validation, inflating performance metrics.

    This function assigns all cells inside each camp polygon the *smallest*
    existing block ID that occurs within that polygon, ensuring the entire
    camp lands in one fold.

    Args:
        block_ids:      1-D integer array of current block IDs, shape (n,).
                        Modified in-place and returned.
        camp_polygons:  GeoDataFrame of mining camp polygons.  Must be in the
                        same CRS as *profile['crs']*.
        row_idx:        1-D array of raster row indices, shape (n,).
        col_idx:        1-D array of raster column indices, shape (n,).
        profile:        Rasterio-style profile dict with 'transform' and 'crs'.

    Returns:
        Modified ``block_ids`` array with camp cells sharing a block ID.
    """
    if camp_polygons is None or len(camp_polygons) == 0:
        logger.info("ensure_camp_integrity: no camp polygons provided – skipping.")
        return block_ids

    transform: rasterio.transform.Affine = profile["transform"]
    block_ids = block_ids.copy()

    # Convert pixel (row, col) → projected coordinates
    cols_arr = col_idx.astype(float)
    rows_arr = row_idx.astype(float)
    xs, ys = rasterio.transform.xy(transform, rows_arr, cols_arr)
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    # Build a GeoDataFrame of sample points for spatial join
    from shapely.geometry import Point as ShapelyPoint

    sample_points = gpd.GeoDataFrame(
        {"sample_idx": np.arange(len(block_ids))},
        geometry=[ShapelyPoint(x, y) for x, y in zip(xs, ys)],
        crs=profile["crs"],
    )

    # Ensure camp polygons are in the same CRS as samples
    if camp_polygons.crs != sample_points.crs:
        camp_polygons = camp_polygons.to_crs(sample_points.crs)

    # Spatial join: find which camp (if any) each sample falls in
    joined = gpd.sjoin(
        sample_points,
        camp_polygons[["geometry"]].reset_index(drop=True).assign(
            camp_id=np.arange(len(camp_polygons))
        ),
        how="left",
        predicate="within",
    )

    # For each camp, assign all its cells the same block ID (min block ID in camp)
    for camp_id, group in joined[joined["camp_id"].notna()].groupby("camp_id"):
        sample_indices = group["sample_idx"].values.astype(int)
        dominant_block = int(np.min(block_ids[sample_indices]))
        block_ids[sample_indices] = dominant_block
        logger.info(
            "ensure_camp_integrity: camp %d – assigned %d cells → block %d",
            int(camp_id), len(sample_indices), dominant_block,
        )

    return block_ids


# ---------------------------------------------------------------------------
# Blind holdout reservation
# ---------------------------------------------------------------------------


def reserve_blind_holdout(
    df: pd.DataFrame,
    holdout_polygon: gpd.GeoDataFrame,
    profile: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a sample DataFrame into training and blind holdout sets.

    The holdout set contains all rows whose spatial location falls inside
    the *holdout_polygon*.  The polygon should represent a geographically
    isolated sub-region that was not used during model development – e.g.,
    a district where significant discoveries were made after the training
    data cutoff date.

    The function expects ``df`` to have columns ``row`` and ``col`` giving
    the raster coordinates of each sample.  These are converted to projected
    coordinates using *profile['transform']*.

    Args:
        df:               DataFrame of labelled samples.  Must contain integer
                          columns ``'row'`` and ``'col'`` (raster indices).
        holdout_polygon:  GeoDataFrame with a single (or union of) polygon(s)
                          defining the blind holdout area.
        profile:          Rasterio-style profile dict with ``'transform'`` and
                          ``'crs'``.

    Returns:
        Tuple ``(training_set, blind_holdout)`` where both are DataFrames
        with the same columns as *df*.  Indices are reset on both outputs.

    Raises:
        KeyError: If ``'row'`` or ``'col'`` columns are missing from *df*.
    """
    for required_col in ("row", "col"):
        if required_col not in df.columns:
            raise KeyError(
                f"reserve_blind_holdout: DataFrame missing required column '{required_col}'."
            )

    transform: rasterio.transform.Affine = profile["transform"]

    # Convert raster indices → projected coordinates
    xs, ys = rasterio.transform.xy(
        transform,
        df["row"].values.astype(float),
        df["col"].values.astype(float),
    )
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    from shapely.geometry import Point as ShapelyPoint

    sample_gdf = gpd.GeoDataFrame(
        {"df_index": np.arange(len(df))},
        geometry=[ShapelyPoint(x, y) for x, y in zip(xs, ys)],
        crs=profile["crs"],
    )

    # Reproject holdout polygon to match sample CRS if needed
    if holdout_polygon.crs != sample_gdf.crs:
        holdout_polygon = holdout_polygon.to_crs(sample_gdf.crs)

    # Dissolve all holdout polygons into a single geometry for the sjoin
    holdout_union = holdout_polygon.dissolve()

    # Spatial join: identify points inside holdout
    joined = gpd.sjoin(
        sample_gdf,
        holdout_union[["geometry"]].assign(in_holdout=True),
        how="left",
        predicate="within",
    )

    in_holdout_mask = joined["in_holdout"].fillna(False).values.astype(bool)

    holdout_df_idx = np.arange(len(df))[in_holdout_mask]
    train_df_idx = np.arange(len(df))[~in_holdout_mask]

    training_set = df.iloc[train_df_idx].reset_index(drop=True)
    blind_holdout = df.iloc[holdout_df_idx].reset_index(drop=True)

    logger.info(
        "reserve_blind_holdout: training=%d samples, blind holdout=%d samples "
        "(%.1f %% of total)",
        len(training_set),
        len(blind_holdout),
        100.0 * len(blind_holdout) / max(1, len(df)),
    )

    return training_set, blind_holdout


# ---------------------------------------------------------------------------
# Buffered blind holdout (mask-based, leakage-safe)
# ---------------------------------------------------------------------------


def blind_holdout_masks(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    profile: dict,
    holdout_polygon: gpd.GeoDataFrame,
    buffer_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for a blind-holdout district and the training pool.

    Cells inside ``holdout_polygon`` form the blind test set. Cells within
    ``buffer_m`` of the polygon (but outside it) are excluded from BOTH sets: they
    are spatially autocorrelated with the holdout, so letting them into training
    would leak holdout geology into the model.

    Args:
        row_idx: Raster row index per sample, shape (n,).
        col_idx: Raster column index per sample, shape (n,).
        profile: Rasterio-style profile with ``transform``.
        holdout_polygon: GeoDataFrame whose union defines the holdout district,
            in the same CRS as the raster.
        buffer_m: Width of the exclusion ring in metres.

    Returns:
        ``(holdout_mask, train_mask)`` boolean arrays of shape (n,).

    Raises:
        ValueError: If ``buffer_m`` is negative.
    """
    import shapely

    if buffer_m < 0:
        raise ValueError(f"buffer_m must be >= 0, got {buffer_m}")

    xs, ys = rasterio.transform.xy(
        profile["transform"],
        np.asarray(row_idx, dtype=float),
        np.asarray(col_idx, dtype=float),
    )
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    union = holdout_polygon.geometry.union_all()
    holdout_mask = np.asarray(shapely.contains_xy(union, xs, ys), dtype=bool)

    if buffer_m > 0:
        buffered_mask = np.asarray(
            shapely.contains_xy(union.buffer(buffer_m), xs, ys), dtype=bool
        )
    else:
        buffered_mask = holdout_mask

    train_mask = ~buffered_mask
    logger.info(
        "blind_holdout_masks: holdout=%d, buffer ring dropped=%d, train pool=%d",
        int(holdout_mask.sum()),
        int((buffered_mask & ~holdout_mask).sum()),
        int(train_mask.sum()),
    )
    return holdout_mask, train_mask


# ---------------------------------------------------------------------------
# Positive-cluster integrity (keep each deposit inside one CV block)
# ---------------------------------------------------------------------------


def enforce_cluster_integrity(
    block_ids: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    positive_raster: np.ndarray,
) -> np.ndarray:
    """Force every connected cluster of positive cells into a single spatial block.

    A deposit's alteration halo is many contiguous 50 m cells. If a square block
    boundary cuts through it, some cells land in training and some in validation,
    which leaks the deposit's signature. Each 8-connected positive cluster is
    reassigned the block ID of its centroid cell.

    Args:
        block_ids: Block ID per sample, shape (n,). Not modified.
        row_idx: Raster row per sample, shape (n,).
        col_idx: Raster column per sample, shape (n,).
        positive_raster: (H, W) array, non-zero where a cell is positive.

    Returns:
        New block ID array of shape (n,).
    """
    from scipy import ndimage

    out = np.asarray(block_ids).copy()
    cluster_raster, n_clusters = ndimage.label(
        np.asarray(positive_raster) != 0, structure=np.ones((3, 3), dtype=int)
    )
    if n_clusters == 0:
        return out

    sample_cluster = cluster_raster[row_idx, col_idx]
    for cid in range(1, n_clusters + 1):
        members = np.where(sample_cluster == cid)[0]
        if len(members) == 0:
            continue
        c_row = int(round(row_idx[members].mean()))
        c_col = int(round(col_idx[members].mean()))
        # Use the member closest to the centroid so the chosen block really exists.
        nearest = members[
            np.argmin((row_idx[members] - c_row) ** 2 + (col_idx[members] - c_col) ** 2)
        ]
        out[members] = out[nearest]

    logger.info("enforce_cluster_integrity: unified %d positive clusters", n_clusters)
    return out
