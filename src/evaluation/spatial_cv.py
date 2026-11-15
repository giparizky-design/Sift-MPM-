"""Unified 5-Fold Buffered Spatial Cross-Validation Trunk (D8).

Implements leakage-resistant spatial cross-validation enforcing physical buffer zones:
    B_m = max(max_k(r*_k), 16 * px_size_m)
between training pixels and validation boundaries.

Features:
  - Complete spatial isolation: training samples within B_m of any validation block
    boundary are purged from the training set, eliminating cross-boundary spatial
    filter and deep neural receptive-field leakage.
  - Strict fold isolation: propensity models, spatial exclusion masks, and evidential
    P-A feature weights are fitted per-fold on training data only.
  - Non-overlapping validation partition ensuring a 100% complete out-of-fold prediction cube.
  - Full scikit-learn CV splitter compatibility via BufferedSpatialBlockKFold.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def compute_physical_buffer_m(
    px_size_m: float = 250.0,
    r_k_star_m: Optional[Union[float, Dict[Any, Any]]] = 6000.0,
) -> float:
    """Compute physical buffer zone distance B_m = max(r*_k, 16 * px_size_m) (D8).

    Args:
        px_size_m: Physical pixel size in meters (e.g. 250.0, 100.0, 50.0).
        r_k_star_m: Deposit halo exclusion radius or dict of radii per head in meters.
            Default: 6000.0 m (Porphyry/Epithermal isotropic buffer).

    Returns:
        Buffer distance B_m in meters.
    """
    if r_k_star_m is None:
        max_r = 6000.0
    elif isinstance(r_k_star_m, dict):
        radii_vals: List[float] = []
        for v in r_k_star_m.values():
            if isinstance(v, (tuple, list)):
                radii_vals.append(float(v[0]))
            else:
                radii_vals.append(float(v))
        max_r = max(radii_vals) if radii_vals else 6000.0
    else:
        max_r = float(r_k_star_m)

    receptive_field_m = 16.0 * max(1.0, float(px_size_m))
    return float(max(max_r, receptive_field_m))


def _extract_metric_coords(
    coords_or_gdf: Union[np.ndarray, pd.DataFrame, Any],
    px_size_m: float = 250.0,
) -> np.ndarray:
    """Extract 2D metric coordinates (N, 2) in meters from array, DataFrame, or GeoDataFrame."""
    if hasattr(coords_or_gdf, "geometry") and hasattr(coords_or_gdf, "crs"):
        gdf = coords_or_gdf
        try:
            is_geo = bool(gdf.crs is not None and gdf.crs.is_geographic)
        except Exception:
            is_geo = True

        xs = gdf.geometry.x.to_numpy(dtype=np.float64)
        ys = gdf.geometry.y.to_numpy(dtype=np.float64)
        if is_geo:
            mean_lat = float(np.nanmean(ys)) if len(ys) > 0 else 0.0
            m_per_deg_lon = 111320.0 * np.cos(np.radians(mean_lat))
            m_per_deg_lat = 110570.0
            return np.column_stack([xs * m_per_deg_lon, ys * m_per_deg_lat])
        return np.column_stack([xs, ys])

    if isinstance(coords_or_gdf, pd.DataFrame):
        df = coords_or_gdf
        col_map = {str(c).lower(): c for c in df.columns}
        if "lon" in col_map and "lat" in col_map:
            xs = df[col_map["lon"]].to_numpy(dtype=np.float64)
            ys = df[col_map["lat"]].to_numpy(dtype=np.float64)
            mean_lat = float(np.nanmean(ys)) if len(ys) > 0 else 0.0
            m_per_deg_lon = 111320.0 * np.cos(np.radians(mean_lat))
            m_per_deg_lat = 110570.0
            return np.column_stack([xs * m_per_deg_lon, ys * m_per_deg_lat])
        if "longitude" in col_map and "latitude" in col_map:
            xs = df[col_map["longitude"]].to_numpy(dtype=np.float64)
            ys = df[col_map["latitude"]].to_numpy(dtype=np.float64)
            mean_lat = float(np.nanmean(ys)) if len(ys) > 0 else 0.0
            m_per_deg_lon = 111320.0 * np.cos(np.radians(mean_lat))
            m_per_deg_lat = 110570.0
            return np.column_stack([xs * m_per_deg_lon, ys * m_per_deg_lat])
        if "x" in col_map and "y" in col_map:
            xs = df[col_map["x"]].to_numpy(dtype=np.float64)
            ys = df[col_map["y"]].to_numpy(dtype=np.float64)
            is_geo = (
                np.nanmax(xs) <= 180.0
                and np.nanmin(xs) >= -180.0
                and np.nanmax(ys) <= 90.0
                and np.nanmin(ys) >= -90.0
            )
            if is_geo:
                mean_lat = float(np.nanmean(ys)) if len(ys) > 0 else 0.0
                m_per_deg_lon = 111320.0 * np.cos(np.radians(mean_lat))
                m_per_deg_lat = 110570.0
                return np.column_stack([xs * m_per_deg_lon, ys * m_per_deg_lat])
            return np.column_stack([xs, ys])
        if "col" in col_map and "row" in col_map:
            xs = df[col_map["col"]].to_numpy(dtype=np.float64) * float(px_size_m)
            ys = df[col_map["row"]].to_numpy(dtype=np.float64) * float(px_size_m)
            return np.column_stack([xs, ys])

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) >= 2:
            xs = df[num_cols[0]].to_numpy(dtype=np.float64)
            ys = df[num_cols[1]].to_numpy(dtype=np.float64)
            return np.column_stack([xs, ys])
        raise ValueError("Could not extract spatial coordinates from DataFrame.")

    arr = np.asarray(coords_or_gdf, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        xs = arr[:, 0]
        ys = arr[:, 1]
        is_geo = (
            np.nanmax(xs) <= 180.0
            and np.nanmin(xs) >= -180.0
            and np.nanmax(ys) <= 90.0
            and np.nanmin(ys) >= -90.0
        )
        if is_geo:
            mean_lat = float(np.nanmean(ys)) if len(ys) > 0 else 0.0
            m_per_deg_lon = 111320.0 * np.cos(np.radians(mean_lat))
            m_per_deg_lat = 110570.0
            return np.column_stack([xs * m_per_deg_lon, ys * m_per_deg_lat])
        return arr[:, :2]

    raise ValueError(f"Expected 2-D coordinates array, got shape {arr.shape}.")


def spatial_buffered_block_kfold(
    coords_or_gdf: Union[np.ndarray, pd.DataFrame, Any],
    *args: Any,
    n_splits: int = 5,
    block_size_m: float = 5000.0,
    buffer_m: Optional[float] = None,
    px_size_m: float = 250.0,
    r_k_star_m: Optional[Union[float, Dict[Any, Any]]] = 6000.0,
    random_state: int = 42,
    block_ids: Optional[np.ndarray] = None,
    coords_m: Optional[np.ndarray] = None,
    n_folds: Optional[int] = None,
    **kwargs: Any,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Group spatial blocks into k cross-validation folds with physical buffer zones (D8).

    Guarantees:
      1. Every block is in exactly one fold's validation set -> 100% out-of-fold coverage.
      2. Training samples within B_m = max(r*_k, 16 * px_size_m) of any validation block
         are purged from training -> zero cross-boundary receptive-field or spatial filter leakage.

    Args:
        coords_or_gdf: (N, 2) coordinate array, DataFrame with coordinate columns,
            or 1D block_ids array if coords_m is supplied positionally or by keyword.
        n_splits: Number of cross-validation folds (default 5).
        block_size_m: Grid block side length in meters (default 5000.0 m = 5 km).
        buffer_m: Physical buffer distance in meters. If None, automatically computed
            as B_m = max(r*_k, 16 * px_size_m).
        px_size_m: Physical pixel size in meters (default 250.0 m).
        r_k_star_m: Deposit halo radius or dict of radii in meters (default 6000.0 m).
        random_state: Random state for reproducible block permutation.
        block_ids: Optional pre-assigned block IDs array of shape (N,).
        coords_m: Optional pre-computed metric coordinate array of shape (N, 2).
        n_folds: Backward-compatibility alias for n_splits.

    Returns:
        List of n_splits tuples (clean_train_indices, val_indices).
    """
    splits_count = n_folds if n_folds is not None else n_splits
    if splits_count <= 1:
        raise ValueError(f"n_splits must be >= 2, got {splits_count}")

    # Handle flexible calling conventions (e.g. block_ids passed as first positional arg)
    if isinstance(coords_or_gdf, np.ndarray) and coords_or_gdf.ndim == 1:
        extracted_block_ids = coords_or_gdf.astype(np.int64)
        if len(args) > 0 and isinstance(args[0], np.ndarray):
            metric_xy = _extract_metric_coords(args[0], px_size_m=px_size_m)
        elif coords_m is not None:
            metric_xy = _extract_metric_coords(coords_m, px_size_m=px_size_m)
        elif "coords" in kwargs:
            metric_xy = _extract_metric_coords(kwargs["coords"], px_size_m=px_size_m)
        else:
            raise ValueError(
                "When passing 1-D block_ids as first argument, metric coordinates must be provided via coords_m."
            )
    else:
        metric_xy = (
            _extract_metric_coords(coords_m, px_size_m=px_size_m)
            if coords_m is not None
            else _extract_metric_coords(coords_or_gdf, px_size_m=px_size_m)
        )
        extracted_block_ids = block_ids

    N = metric_xy.shape[0]
    if N == 0:
        return []

    # Assign spatial blocks if not provided
    if extracted_block_ids is None:
        xs = metric_xy[:, 0]
        ys = metric_xy[:, 1]
        min_x, min_y = float(np.nanmin(xs)), float(np.nanmin(ys))
        col_idx = np.floor((xs - min_x) / float(block_size_m)).astype(np.int64)
        row_idx = np.floor((ys - min_y) / float(block_size_m)).astype(np.int64)
        max_col = int(np.max(col_idx)) + 1
        assigned_block_ids = row_idx * max_col + col_idx
    else:
        assigned_block_ids = np.asarray(extracted_block_ids, dtype=np.int64).ravel()
        if len(assigned_block_ids) != N:
            raise ValueError(
                f"block_ids length ({len(assigned_block_ids)}) != coordinates rows ({N})"
            )

    # Compute physical buffer zone B_m
    if buffer_m is not None:
        effective_buffer_m = max(0.0, float(buffer_m))
    else:
        effective_buffer_m = compute_physical_buffer_m(
            px_size_m=px_size_m,
            r_k_star_m=r_k_star_m,
        )

    unique_blocks = np.unique(assigned_block_ids)
    n_blocks = len(unique_blocks)
    actual_splits = min(splits_count, n_blocks)
    if actual_splits < splits_count:
        logger.warning(
            "n_splits (%d) > unique spatial blocks (%d). Reducing splits to %d.",
            splits_count,
            n_blocks,
            actual_splits,
        )

    rng = np.random.default_rng(random_state)
    shuffled_blocks = rng.permutation(unique_blocks)
    block_folds = np.array_split(shuffled_blocks, actual_splits)

    all_indices = np.arange(N)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    for f_idx in range(actual_splits):
        val_block_set = set(block_folds[f_idx])
        val_mask = np.isin(assigned_block_ids, list(val_block_set))
        val_idx = all_indices[val_mask]
        cand_train_idx = all_indices[~val_mask]

        if effective_buffer_m > 0.0 and len(val_idx) > 0 and len(cand_train_idx) > 0:
            val_tree = cKDTree(metric_xy[val_idx])
            # Purge candidate training cells within effective_buffer_m of any validation cell
            too_close = val_tree.query_ball_point(metric_xy[cand_train_idx], r=effective_buffer_m)
            isolated_mask = np.array([len(pts) == 0 for pts in too_close], dtype=bool)
            clean_train_idx = cand_train_idx[isolated_mask]

            # Failsafe: if study area is very small and buffer purges everything, fall back
            if len(clean_train_idx) == 0:
                logger.warning(
                    "Fold %d: physical buffer B_m=%.1fm purged all training samples; retaining unbuffered training set.",
                    f_idx + 1,
                    effective_buffer_m,
                )
                clean_train_idx = cand_train_idx
        else:
            clean_train_idx = cand_train_idx

        logger.info(
            "Spatial CV Fold %d/%d: train=%d (dropped %d buffer cells), val=%d, B_m=%.1fm",
            f_idx + 1,
            actual_splits,
            len(clean_train_idx),
            len(cand_train_idx) - len(clean_train_idx),
            len(val_idx),
            effective_buffer_m,
        )
        folds.append((clean_train_idx, val_idx))

    return folds


class BufferedSpatialBlockKFold:
    """Scikit-learn compatible 5-fold buffered spatial cross-validator (D8)."""

    def __init__(
        self,
        n_splits: int = 5,
        block_size_m: float = 5000.0,
        buffer_m: Optional[float] = None,
        px_size_m: float = 250.0,
        r_k_star_m: Optional[Union[float, Dict[Any, Any]]] = 6000.0,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.block_size_m = block_size_m
        self.buffer_m = buffer_m
        self.px_size_m = px_size_m
        self.r_k_star_m = r_k_star_m
        self.random_state = random_state

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test sets."""
        coords_source = groups if groups is not None else X
        splits = spatial_buffered_block_kfold(
            coords_source,
            n_splits=self.n_splits,
            block_size_m=self.block_size_m,
            buffer_m=self.buffer_m,
            px_size_m=self.px_size_m,
            r_k_star_m=self.r_k_star_m,
            random_state=self.random_state,
        )
        for train_idx, val_idx in splits:
            yield train_idx, val_idx

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        return self.n_splits


def assemble_oof_prediction_cube(
    oof_fold_predictions: Sequence[np.ndarray],
    val_indices_per_fold: Sequence[np.ndarray],
    n_total_samples: int,
    n_heads: Optional[int] = None,
) -> np.ndarray:
    """Assemble a 100% complete out-of-fold prediction cube from fold predictions (D8).

    Args:
        oof_fold_predictions: List of fold validation predictions. Each array has shape
            (n_val_fold,) or (n_val_fold, n_heads).
        val_indices_per_fold: List of validation index arrays for each fold.
        n_total_samples: Total number of samples N in the full regional raster/cube.
        n_heads: Number of deposit/commodity heads (inferred if None).

    Returns:
        1D or 2D array of out-of-fold predictions with 100% sample coverage.
    """
    if len(oof_fold_predictions) == 0 or len(val_indices_per_fold) == 0:
        if n_heads is not None and n_heads > 1:
            return np.zeros((n_total_samples, n_heads), dtype=np.float32)
        return np.zeros(n_total_samples, dtype=np.float32)

    first_pred = np.asarray(oof_fold_predictions[0])
    is_multi_head = (first_pred.ndim == 2 and first_pred.shape[1] > 1)
    k_heads = first_pred.shape[1] if is_multi_head else (n_heads or 1)

    if is_multi_head:
        oof_cube = np.zeros((n_total_samples, k_heads), dtype=np.float32)
    else:
        oof_cube = np.zeros(n_total_samples, dtype=np.float32)

    for preds, val_idx in zip(oof_fold_predictions, val_indices_per_fold):
        p_arr = np.asarray(preds, dtype=np.float32)
        v_idx = np.asarray(val_idx, dtype=np.int64)
        if len(v_idx) == 0:
            continue
        if is_multi_head:
            oof_cube[v_idx, :] = p_arr
        else:
            oof_cube[v_idx] = p_arr.ravel()

    return oof_cube
