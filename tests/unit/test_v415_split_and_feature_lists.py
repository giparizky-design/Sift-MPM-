"""v4.15 (audit v4.14 T19/T21): deposit-cluster integrity, training buffer, pinned feature lists."""

import numpy as np
import pytest

from src.labels.spatial_splits import (
    assign_spatial_blocks,
    count_clusters_spanning_folds,
    enforce_cluster_integrity,
    positive_cluster_raster,
    spatial_block_kfold,
    training_buffer_exclusion,
)


def _grid(H=120, W=120):
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return H, W, rr.ravel().astype(np.int32), cc.ravel().astype(np.int32)


def _straddling_positives(H, W, block=20):
    """A 3x3 positive cluster centred on every interior vertical block boundary."""
    pos = np.zeros((H, W), bool)
    for r in range(10, H - 10, block):
        for c in range(block, W - block + 1, block):
            pos[r - 1:r + 2, c - 1:c + 2] = True
    return pos


def test_cluster_integrity_puts_every_deposit_in_one_fold():
    H, W, rows, cols = _grid()
    pos_2d = _straddling_positives(H, W)
    pos = pos_2d[rows, cols]
    blocks = assign_spatial_blocks(rows, cols, H, W, block_size_px=20)
    folds = spatial_block_kfold(blocks, n_folds=5, random_state=42)
    cl = positive_cluster_raster(rows, cols, pos, H, W)
    before = count_clusters_spanning_folds(cl, rows, cols, folds)
    assert before > 0, "fixture must produce clusters cut across folds"
    blocks2 = enforce_cluster_integrity(blocks, rows, cols, pos_2d)
    folds2 = spatial_block_kfold(blocks2, n_folds=5, random_state=42)
    assert count_clusters_spanning_folds(cl, rows, cols, folds2) == 0
    # folds still partition all samples
    assert sorted(np.concatenate([va for _, va in folds2]).tolist()) == list(range(len(rows)))


def test_training_buffer_removes_cells_near_held_out_deposits_only():
    H, W, rows, cols = _grid(60, 60)
    pos_2d = np.zeros((H, W), bool)
    pos_2d[10:13, 10:13] = True      # held-out deposit
    pos_2d[50:53, 50:53] = True      # training deposit, far away
    pos = pos_2d[rows, cols]
    cl = positive_cluster_raster(rows, cols, pos, H, W)
    val_idx = np.where((rows < 30) & (cols < 30))[0]
    excl = training_buffer_exclusion(val_idx, rows, cols, pos, cl, buffer_px=8.0)
    # held-out cluster itself and cells within 8 px are excluded
    assert excl[(rows == 11) & (cols == 11)].all()
    assert excl[(rows == 11) & (cols == 12 + 8)].all()
    assert not excl[(rows == 11) & (cols == 12 + 9)].any()
    # the far training deposit is untouched
    assert not excl[(rows == 51) & (cols == 51)].any()
    # no buffer, or no held-out positive -> nothing excluded
    assert not training_buffer_exclusion(val_idx, rows, cols, pos, cl, buffer_px=0.0).any()
    assert not training_buffer_exclusion(np.where((rows > 30))[0], rows, cols, pos & (rows < 30), cl, buffer_px=8.0).any()


def test_train_cells_keep_min_distance_to_validation_deposits():
    H, W, rows, cols = _grid(80, 80)
    rng = np.random.default_rng(1)
    pos_2d = np.zeros((H, W), bool)
    for r, c in rng.integers(5, 75, size=(12, 2)):
        pos_2d[r:r + 2, c:c + 2] = True
    pos = pos_2d[rows, cols]
    cl = positive_cluster_raster(rows, cols, pos, H, W)
    blocks = enforce_cluster_integrity(assign_spatial_blocks(rows, cols, H, W, block_size_px=16), rows, cols, pos_2d)
    folds = spatial_block_kfold(blocks, n_folds=5, random_state=42)
    from scipy.spatial import cKDTree

    for tr, va in folds:
        va_pos = va[pos[va]]
        if len(va_pos) == 0:
            continue
        tr = tr[~training_buffer_exclusion(va, rows, cols, pos, cl, buffer_px=6.0)[tr]]
        if len(tr) == 0:
            continue
        d, _ = cKDTree(np.column_stack([rows[va_pos], cols[va_pos]])).query(np.column_stack([rows[tr], cols[tr]]))
        assert d.min() > 6.0


# ---------------------------------------------------------------- pinned feature lists
def _write(tmp_path, name, cols):
    p = tmp_path / name
    p.write_text("\n".join(cols) + "\n", encoding="utf-8")
    return str(p)


def test_pinned_lists_union_and_metals_exclusion(tmp_path):
    from scripts.run_seven_pass_training import resolve_pinned_feature_lists

    avail = ["a", "b", "c", "d", "label_k0"]
    m = _write(tmp_path, "m.txt", ["a", "b"])
    c = _write(tmp_path, "c.txt", ["a", "b", "c"])
    out = resolve_pinned_feature_lists(m, c, True, True, avail, {"d"})
    assert out["feature_cols"] == ["a", "b", "c"]
    assert out["metals_exclude"] == ("c",)
    only_m = resolve_pinned_feature_lists(m, c, True, False, avail, {"d"})
    assert only_m["feature_cols"] == ["a", "b"] and only_m["metals_exclude"] == ()
    assert resolve_pinned_feature_lists("none", "none", True, True, avail, {"d"}) is None


def test_pinned_list_missing_or_banned_columns_raise(tmp_path):
    from scripts.run_seven_pass_training import resolve_pinned_feature_lists

    avail = ["a", "b", "d", "label_k0"]
    with pytest.raises(ValueError, match="absent"):
        resolve_pinned_feature_lists(_write(tmp_path, "x.txt", ["a", "zzz"]), "none", True, False, avail, {"d"})
    with pytest.raises(ValueError, match="non-feature"):
        resolve_pinned_feature_lists(_write(tmp_path, "y.txt", ["a", "d"]), "none", True, False, avail, {"d"})
    with pytest.raises(ValueError, match="non-feature"):
        resolve_pinned_feature_lists(_write(tmp_path, "z.txt", ["a", "label_k0"]), "none", True, False, avail, {"d"})
