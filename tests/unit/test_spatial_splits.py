"""Unit tests for leakage-resistant spatial block cross-validation.

Geoscientific context:
    Spatial autocorrelation is one of the most critical sources of optimistic bias
    in mineral prospectivity mapping. Regional geophysical and alteration signatures
    (e.g., magnetic anomalies, hydrothermal phyllic alteration halos, regional fault systems)
    often span kilometres to tens of kilometres.

    If random train-test splitting is used, test cells adjacent to training cells
    share near-identical signatures, causing models to memorize local geography
    rather than learning transferable geological rules.

    Spatial block partitioning enforces that validation cells belong to entirely separate
    spatial blocks or mining camps, simulating deployment to unseen terranes.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from shapely.geometry import box, Point

from src.labels.spatial_splits import (
    assign_spatial_blocks,
    ensure_camp_integrity,
    reserve_blind_holdout,
    spatial_block_kfold,
)


def test_assign_spatial_blocks_properties():
    """Test assign_spatial_blocks generates valid block IDs and handles dimensions."""
    H, W = 100, 100
    block_size = 20

    # Grid of 25 coordinate points
    rows = np.array([0, 19, 20, 39, 99])
    cols = np.array([0, 19, 20, 39, 99])
    r_grid, c_grid = np.meshgrid(rows, cols)
    r_flat = r_grid.flatten()
    c_flat = c_grid.flatten()

    block_ids = assign_spatial_blocks(r_flat, c_flat, H, W, block_size)

    assert len(block_ids) == len(r_flat)
    assert block_ids.dtype == np.int64
    # Points in the same block should receive the same block ID
    # (0, 0) and (19, 19) are in the top-left block (block_row=0, block_col=0)
    assert block_ids[0] == block_ids[6]  # both in block (0, 0)


def test_assign_spatial_blocks_invalid_args():
    """Test validation errors for invalid block size or mismatched input shapes."""
    with pytest.raises(ValueError, match="block_size_px must be > 0"):
        assign_spatial_blocks(np.array([0]), np.array([0]), 100, 100, block_size_px=0)

    with pytest.raises(ValueError, match="must have the same shape"):
        assign_spatial_blocks(np.array([0, 1]), np.array([0]), 100, 100, block_size_px=10)


def test_no_block_straddles_train_val():
    """Verify that no block_id appears in both train and validation sets in any fold."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    n_blocks = 20
    block_ids = rng.integers(0, n_blocks, size=n_samples)

    n_folds = 5
    splits = spatial_block_kfold(block_ids, n_folds=n_folds, random_state=42)

    assert len(splits) == n_folds

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_blocks = set(block_ids[train_idx])
        val_blocks = set(block_ids[val_idx])

        # Core leakage prevention assertion: intersection must be empty
        intersection = train_blocks.intersection(val_blocks)
        assert len(intersection) == 0, (
            f"Fold {fold_idx} leaked spatial blocks across train and validation: {intersection}"
        )


def test_all_cells_assigned():
    """Verify that all input indices appear in exactly one fold's validation set."""
    rng = np.random.default_rng(123)
    n_samples = 500
    block_ids = rng.integers(0, 15, size=n_samples)

    n_folds = 5
    splits = spatial_block_kfold(block_ids, n_folds=n_folds, random_state=123)

    all_val_indices = []
    for train_idx, val_idx in splits:
        # Train and val within a fold must be mutually exclusive
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        all_val_indices.extend(val_idx)

    # All samples must be evaluated exactly once across all folds
    assert sorted(all_val_indices) == list(range(n_samples)), (
        "Validation indices across folds do not form a complete, non-overlapping partition of all samples."
    )


def test_ensure_camp_integrity():
    """Verify that cells falling within a mining camp are consolidated into a single block."""
    # Define a 1000m x 1000m transform (pixel size 50m)
    # Affine(a, b, c, d, e, f) -> x = a*col + b*row + c, y = d*col + e*row + f
    transform = Affine(50.0, 0.0, 500000.0, 0.0, -50.0, 6800000.0)
    profile = {"crs": "EPSG:7851", "transform": transform}

    # Samples at various row, col
    row_idx = np.array([2, 3, 4, 18, 19])
    col_idx = np.array([2, 3, 4, 18, 19])
    # Initially assign distinct block IDs
    initial_block_ids = np.array([1, 2, 3, 8, 9])

    # Camp polygon covering the first 3 samples (around x=500100..500250, y=6799750..6799900)
    camp_box = box(500050.0, 6799700.0, 500300.0, 6799950.0)
    camp_gdf = gpd.GeoDataFrame(
        {"camp_name": ["Kalgoorlie_Camp"], "geometry": [camp_box]},
        crs="EPSG:7851",
    )

    updated_block_ids = ensure_camp_integrity(
        initial_block_ids, camp_gdf, row_idx, col_idx, profile
    )

    # First three samples should now share the minimum block ID among them (min(1, 2, 3) = 1)
    assert updated_block_ids[0] == updated_block_ids[1] == updated_block_ids[2] == 1
    # Samples outside the camp should retain their original block IDs
    assert updated_block_ids[3] == 8
    assert updated_block_ids[4] == 9


def test_blind_holdout_spatial_separation():
    """Verify that samples falling inside the blind holdout polygon are excluded from training."""
    transform = Affine(50.0, 0.0, 500000.0, 0.0, -50.0, 6800000.0)
    profile = {"crs": "EPSG:7851", "transform": transform}

    df = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(5)],
        "row": [2, 3, 4, 15, 16],
        "col": [2, 3, 4, 15, 16],
        "au_grade": [1.5, 0.8, 3.2, 0.01, 0.02],
    })

    # Holdout polygon covers samples 0, 1, 2
    holdout_box = box(500050.0, 6799700.0, 500300.0, 6799950.0)
    holdout_gdf = gpd.GeoDataFrame(
        {"region": ["Laverton_Holdout"], "geometry": [holdout_box]},
        crs="EPSG:7851",
    )

    train_df, holdout_df = reserve_blind_holdout(df, holdout_gdf, profile)

    assert len(holdout_df) == 3
    assert len(train_df) == 2
    assert set(holdout_df["sample_id"]) == {"S0", "S1", "S2"}
    assert set(train_df["sample_id"]) == {"S3", "S4"}
    # Disjoint check
    assert len(set(train_df["sample_id"]).intersection(set(holdout_df["sample_id"]))) == 0
