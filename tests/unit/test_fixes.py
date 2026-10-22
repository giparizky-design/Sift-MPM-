"""Regression tests for the code fixes: CRS, PU score direction, leakage-safe CV."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from shapely.geometry import box

from src.labels.deposit_labels import _TARGET_EPSG, load_wamex_drillholes
from src.labels.pu_negatives import step1_isolation_forest, step2_select_reliable_negatives
from src.labels.spatial_splits import blind_holdout_masks, enforce_cluster_integrity
from src.models.pu_pipeline import fit_pu_model, spatial_cv_pu

CONFIG = {
    "models": {"baseline": {"n_bag": 2, "hyperparams": {"n_estimators": 10, "max_depth": 3}}},
    "pu_learning": {"step1_contamination": 0.05, "rn_percentile_threshold": 10.0},
}


# ---------------------------- CRS ------------------------------------------
def test_target_crs_is_mga_zone_51():
    assert _TARGET_EPSG == 7851


def test_wamex_zone_51_collar_stays_in_goldfields(tmp_path):
    """A zone-51 collar must land near Kalgoorlie (~121.4E), not be shifted to zone 55."""
    csv = tmp_path / "collars.csv"
    pd.DataFrame(
        {"hole_id": ["a"], "easting": [350000.0], "northing": [6590000.0], "zone": [51]}
    ).to_csv(csv, index=False)
    gdf = load_wamex_drillholes(str(csv))
    assert gdf.crs.to_epsg() == 7851
    assert gdf.geometry.iloc[0].x == pytest.approx(350000.0, abs=1e-3)
    lon = gdf.to_crs(4326).geometry.iloc[0].x
    assert 121.0 < lon < 122.0


# ---------------------- PU score direction ---------------------------------
def test_isolation_scores_higher_means_more_anomalous():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, (500, 4))
    outliers = rng.normal(8, 0.5, (10, 4))
    X_u = np.vstack([normal, outliers])
    scores = step1_isolation_forest(normal[:5], X_u, contamination=0.05, random_state=0)
    assert scores[-10:].mean() > scores[:500].mean()


def test_reliable_negatives_never_include_planted_outliers():
    rng = np.random.default_rng(1)
    normal = rng.normal(0, 1, (500, 4))
    outliers = rng.normal(8, 0.5, (10, 4))
    X_u = np.vstack([normal, outliers])
    scores = step1_isolation_forest(normal[:5], X_u, contamination=0.05, random_state=1)
    mask = step2_select_reliable_negatives(X_u, scores, percentile_threshold=20.0)
    assert not mask[-10:].any(), "deposit-like cells were chosen as reliable negatives"


# --------------------------- holdout ---------------------------------------
def _profile():
    return {"transform": Affine(50.0, 0, 500000.0, 0, -50.0, 6800000.0), "crs": "EPSG:7851"}


def test_holdout_buffer_ring_excluded_from_train_and_holdout():
    rows = np.zeros(21, dtype=int)
    cols = np.arange(21) * 10  # x = 500025 + 500*k
    poly = gpd.GeoDataFrame(geometry=[box(502000, 6799000, 503000, 6800000)], crs="EPSG:7851")
    hold, train = blind_holdout_masks(rows, cols, _profile(), poly, buffer_m=600)
    assert hold.any() and train.any()
    assert not (hold & train).any()
    ring = ~hold & ~train
    assert ring.any(), "buffer ring should drop cells from both sets"


def test_holdout_negative_buffer_rejected():
    poly = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:7851")
    with pytest.raises(ValueError):
        blind_holdout_masks(np.array([0]), np.array([0]), _profile(), poly, buffer_m=-1)


# ------------------------ cluster integrity --------------------------------
def test_positive_cluster_is_never_split_across_blocks():
    H = W = 40
    pos = np.zeros((H, W), dtype=np.uint8)
    pos[18:23, 18:23] = 1  # one deposit straddling the 20 px block boundary
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rows, cols = rr.ravel(), cc.ravel()
    blocks = (rows // 20) * 2 + (cols // 20)
    fixed = enforce_cluster_integrity(blocks, rows, cols, pos)
    members = pos[rows, cols] == 1
    assert len(np.unique(blocks[members])) > 1  # the problem existed
    assert len(np.unique(fixed[members])) == 1  # and is fixed


# --------------------------- PU CV logic -----------------------------------
def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 3))
    label = np.full(n, -1)
    label[:40] = 1
    X[:40] += 3.0
    label[40:80] = 0
    return pd.DataFrame({"f0": X[:, 0], "f1": X[:, 1], "f2": X[:, 2], "label": label})


def test_fit_pu_model_requires_positives():
    X = np.random.default_rng(0).normal(size=(50, 3))
    with pytest.raises(ValueError):
        fit_pu_model(X[:0], X[:10], X[10:], CONFIG)


def test_cv_skips_folds_missing_a_class_and_scores_valid_ones():
    df = _frame()
    idx = np.arange(len(df))
    cols = ["f0", "f1", "f2"]
    # fold 0: test has only P (skipped); fold 1: test has both P and RN (scored)
    only_p = idx[:20]
    both = np.concatenate([idx[20:40], idx[40:60]])
    folds = [
        (np.setdiff1d(idx, only_p), only_p),
        (np.setdiff1d(idx, both), both),
    ]
    out = spatial_cv_pu(df, cols, folds, CONFIG)
    assert list(out["fold"]) == [1]
    assert out.loc[0, "n_test_pos"] == 20 and out.loc[0, "n_test_rn"] == 20
    assert out.loc[0, "pr_auc"] > out.loc[0, "prevalence"]
