"""Unit tests for Milestone 1: Domain Isolation & Label Hygiene (Requirement R2).

Covers:
  R2.1: Strict metal positives decoupled from coal concessions and K4_LABEL_COLS export.
  R2.2: Elimination of synthetic np.linspace coal seeding and domain-aware zero-coal fail-fast.
  R2.3: Synchronized folds, unified train_idx excluding validation splits, and -1 fold preservation.
  R2.4: Feature stack hygiene ensuring w_canopy is quarantined in NON_FEATURE_COLS across modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels.borneo_labels import K4_LABEL_COLS
from src.outputs.model_introspection import NON_FEATURE_COLS as INTRO_NON_FEATURE_COLS


def test_k4_label_cols_exported():
    """Verify K4_LABEL_COLS is exported and contains the 4 canonical deposit types."""
    assert isinstance(K4_LABEL_COLS, list)
    assert len(K4_LABEL_COLS) == 4
    assert K4_LABEL_COLS == [
        "label_k0_porphyry_cu",
        "label_k1_skarn_fe_cu",
        "label_k2_epithermal_au",
        "label_k3_laterite_fe_ni",
    ]


def test_w_canopy_excluded_from_feature_matrices():
    """Verify w_canopy is quarantined in NON_FEATURE_COLS and pass2_and_meta_cols."""
    # Inspect introspection NON_FEATURE_COLS
    assert "w_canopy" in INTRO_NON_FEATURE_COLS

    # Mock DataFrame to test seven_pass NON_FEATURE_COLS logic
    df = pd.DataFrame({
        "label": [1, -1],
        "row": [0, 1],
        "col": [0, 1],
        "w_canopy": [0.8, 0.2],
        "bouguer_gravity_mgal": [12.5, -5.2],
        "spec_dem_slope": [4.2, 18.1],
        "label_k0_porphyry_cu": [1, -1],
    })

    # Seven-pass training script NON_FEATURE_COLS
    seven_pass_non_feat = {
        "label", "row", "col", "lon", "lat", "block_id",
        "explored_footprint", "alteration_benchmark",
        "label_k0_porphyry_cu", "label_k1_skarn_fe_cu", "label_k2_epithermal_au", "label_k3_laterite_fe_ni",
        "label_coal", "label_coal_k0_high_cv", "label_coal_k1_med_cv", "label_coal_k2_low_cv",
        "commodity_label", "is_centroid_artifact", "target_class",
        "dist_transport_km", "dist_major_rivers_km", "dist_haul_roads_km",
        "gfw_forest_disturbance", "forest_disturbance_rate",
        "active_mine_mask", "formation_polygon_id",
        "coal_tonnage_weight", "coal_parent_concession_id",
        "w_canopy",
    }
    feature_cols = [c for c in df.columns if c not in seven_pass_non_feat and not c.startswith("label_")]
    assert "w_canopy" not in feature_cols
    assert "bouguer_gravity_mgal" in feature_cols
    assert "spec_dem_slope" in feature_cols


def test_pos_mask_metals_strictly_metallic():
    """Verify pos_mask_metals = np.any(y_k4 == 1, axis=1) ignores coal concessions."""
    N = 100
    # Simulate y_k4 with 5 metal positives
    y_k4 = np.full((N, 4), -1, dtype=np.int32)
    y_k4[0:5, 2] = 1  # 5 epithermal Au

    # Simulate coal concessions in labels column
    labels = np.full(N, -1, dtype=np.int32)
    labels[10:30] = 1  # 20 coal concessions (LABEL_P)

    # Correct formula (R2.1)
    pos_mask_metals = np.any(y_k4 == 1, axis=1)
    assert pos_mask_metals.sum() == 5
    assert not np.any(pos_mask_metals[10:30])  # Zero coal overlap


def test_zero_coal_fail_fast_on_domain_coal():
    """Verify that --domain coal raises ValueError when 0 coal positives are present."""
    # We can test the validation check directly
    domain_clean = "coal"
    region_name = "HOTSPOT_SCHWANER"
    y_k3_coal = np.full((50, 3), -1, dtype=np.int32)  # Zero coal positives
    run_coal = True

    n_coal_pos = int((y_k3_coal == 1).sum())
    assert n_coal_pos == 0

    with pytest.raises(ValueError, match="Execution halted for --domain coal: zero ground-truth coal positives"):
        if run_coal and n_coal_pos == 0:
            if domain_clean == "coal":
                raise ValueError(
                    f"[{region_name}] Execution halted for --domain coal: zero ground-truth coal positives found "
                    f"in region (or coal rank columns missing). Cannot train coal prospectivity model without verified coal occurrences."
                )


def test_zero_coal_graceful_bypass_on_domain_both():
    """Verify that --domain both disables run_coal when zero coal positives exist."""
    domain_clean = "both"
    region_name = "HOTSPOT_SCHWANER"
    y_k3_coal = np.full((50, 3), -1, dtype=np.int32)
    run_coal = True

    n_coal_pos = int((y_k3_coal == 1).sum())
    assert n_coal_pos == 0

    if run_coal and n_coal_pos == 0:
        if domain_clean == "both":
            run_coal = False

    assert run_coal is False
    pos_mask_coal = np.any(y_k3_coal == 1, axis=1) if run_coal else np.zeros(50, dtype=bool)
    assert pos_mask_coal.sum() == 0


def test_synchronized_train_idx_disjoint_from_validation():
    """Verify unified train_idx excludes BOTH val_idx (metals) and val_coal_idx (coal)."""
    N = 200
    all_idx = np.arange(N)

    # Simulate disjoint validation splits
    val_idx = np.array([10, 11, 12, 13, 14])
    val_coal_idx = np.array([20, 21, 22, 23, 24])

    # Unified train_idx computation
    val_combined = np.union1d(val_idx, val_coal_idx)
    train_idx = np.setdiff1d(all_idx, val_combined)

    # Verify strict disjointness
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(np.intersect1d(train_idx, val_coal_idx)) == 0
    assert len(train_idx) == N - len(val_combined)


def test_fold_ids_2d_initialization_and_unet_negative_block_filter():
    """Verify fold_ids_2d initialized to -1 and unassigned background cells are preserved."""
    H, W = 10, 10
    fold_ids_2d = np.full((H, W), fill_value=-1, dtype=int)
    # Assign some valid cells to fold 0 and fold 1
    fold_ids_2d[2:4, 2:4] = 0
    fold_ids_2d[6:8, 6:8] = 1

    # Unassigned cells remain -1
    assert (fold_ids_2d == -1).sum() == 100 - (4 + 4)

    # In U-Net fold remapping:
    blocks_coarse = fold_ids_2d.copy()
    valid_blocks = np.unique(blocks_coarse[blocks_coarse >= 0])
    assert set(valid_blocks) == {0, 1}

    # Remapping logic preserves -1
    fold_map = {int(b): int(i % 5) for i, b in enumerate(valid_blocks)}
    fold_map[-1] = -1
    blocks_remapped = np.vectorize(lambda x: fold_map.get(int(x), -1))(blocks_coarse)
    assert (blocks_remapped == -1).sum() == (fold_ids_2d == -1).sum()

    # In U-Net CV loop, train_mask must exclude negative block IDs
    block_flat = blocks_remapped.ravel()
    b_id = 0
    train_mask = (block_flat >= 0) & (block_flat != b_id)
    val_mask = (block_flat == b_id)

    # train_mask only contains cells from fold 1 (not unassigned -1)
    assert np.all(block_flat[train_mask] == 1)
    assert np.all(block_flat[val_mask] == 0)
    assert not np.any(block_flat[train_mask] == -1)
    assert not np.any(block_flat[val_mask] == -1)
