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


# ---------------- Audit v2.0 Remediation Verification Tests -----------------


def test_plan_curvature_gaussian_hill_radial_symmetry():
    """Verify Evans (1980) plan_curvature is radially symmetric on a 2D Gaussian hill."""
    from src.features.structural import plan_curvature

    H = W = 101
    res_m = 100.0
    cy = cx = 50
    y, x = np.meshgrid(np.arange(H) - cy, np.arange(W) - cx, indexing="ij")
    r2 = (x * res_m) ** 2 + (y * res_m) ** 2
    sigma_m = 1500.0
    dem = 800.0 * np.exp(-r2 / (2.0 * sigma_m**2))

    curv = plan_curvature(dem, resolution_m=res_m)
    assert curv.shape == (H, W)
    # At equal radial distances along N, S, E, W and diagonals, plan curvature must match
    d = 12
    c_north = curv[cy - d, cx]
    c_south = curv[cy + d, cx]
    c_east = curv[cy, cx + d]
    c_west = curv[cy, cx - d]
    assert c_north > 0.0  # convex hillslope has positive divergent plan curvature
    np.testing.assert_allclose([c_north, c_south, c_east, c_west], c_north, rtol=1e-4)

    # Flat terrain must be exactly zeroed
    flat = np.full((32, 32), 120.0, dtype=np.float64)
    assert np.all(plan_curvature(flat, resolution_m=res_m) == 0.0)


def test_equatorial_rtp_macleod_stabilization_and_ring_taper():
    """Verify equatorial dipole (I=-14 deg, D=1.5 deg) RTP yields a centered positive peak with zero N-S blowup."""
    from src.features.geophysics import _pad_and_taper, reduced_to_pole

    H = W = 64
    dx = dy = 250.0
    cy = cx = H // 2
    y, x = np.meshgrid((np.arange(H) - cy) * dy, (np.arange(W) - cx) * dx, indexing="ij")
    r2 = x**2 + y**2
    base_pole = 200.0 * np.exp(-r2 / (2.0 * (1500.0**2)))

    # Verify _pad_and_taper keeps interior demeaned values un-attenuated (weight = 1.0)
    padded, r, c, pr, pc, crop_slices, g_mean = _pad_and_taper(base_pole + 500.0)
    np.testing.assert_allclose(padded[crop_slices] + g_mean, base_pole + 500.0, atol=1e-10)

    # Forward-synthesize equatorial TMI anomaly at I=-14 deg, D=1.5 deg in Fourier domain
    inc_rad = np.deg2rad(-14.0)
    dec_rad = np.deg2rad(1.5)
    kx = 2.0 * np.pi * np.fft.fftfreq(W, d=dx)
    ky = -2.0 * np.pi * np.fft.fftfreq(H, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    theta = np.arctan2(KX, KY)
    p_phase = np.sin(inc_rad) + 1j * np.cos(inc_rad) * np.cos(theta - dec_rad)
    tmi_eq = np.real(np.fft.ifft2(np.fft.fft2(base_pole) * (p_phase**2))) + 40_000.0

    rtp = reduced_to_pole(tmi_eq, inclination_deg=-14.0, declination_deg=1.5, dx=dx, dy=dy)
    peak_r, peak_c = np.unravel_index(np.argmax(rtp), rtp.shape)
    assert abs(peak_r - cy) <= 1 and abs(peak_c - cx) <= 1
    assert rtp[cy, cx] > 0.0
    # Border ringing must be negligible relative to center peak
    border_std = float(np.std(np.concatenate([rtp[0, :], rtp[-1, :], rtp[:, 0], rtp[:, -1]])))
    assert border_std < 0.05 * float(rtp[cy, cx])


def test_east_kalimantan_gravity_not_collinear_with_dem():
    """Verify bouguer_gravity_mgal on disk comes from real satellite gravimetry, not -0.1119 * dem."""
    from pathlib import Path
    import rasterio

    ek_dir = Path("data/east_kalimantan/processed")
    dem_tif = ek_dir / "dem_elevation.tif"
    bg_tif = ek_dir / "bouguer_gravity_mgal.tif"
    if dem_tif.exists() and bg_tif.exists():
        with rasterio.open(dem_tif) as s1, rasterio.open(bg_tif) as s2:
            dem = s1.read(1)
            bg = s2.read(1)
        valid = np.isfinite(dem) & np.isfinite(bg) & (dem > -9000.0) & (bg > -9000.0)
        corr = float(np.corrcoef(dem[valid], bg[valid])[0, 1])
        assert abs(corr) < 0.90, f"Bouguer gravity is still collinear with DEM (corr={corr:.4f})!"


def test_decimate_spatial_grid_preserves_categorical_fold_ids():
    """Verify order=0 nearest-neighbor decimation never interpolates intermediate fold IDs."""
    from src.models.unet_model import decimate_spatial_grid

    block_ids = np.zeros((100, 100), dtype=np.int32)
    block_ids[:, 50:] = 4  # sharp boundary between Fold 0 and Fold 4
    block_ids_ds = decimate_spatial_grid(block_ids, target_res_m=250.0, native_res_m=50.0, order=0)
    assert set(np.unique(block_ids_ds)).issubset(set(np.unique(block_ids)))
    assert 2 not in set(np.unique(block_ids_ds))


def test_model_introspection_xgb_tree_and_activation_summary():
    """Verify XGBoost JSON tree enrichment, root-to-leaf path tracing, and leaf evaluation."""
    from src.outputs.model_introspection import (
        UNET_LAYER_CATALOG,
        _enrich_xgb_tree_node,
        _eval_tree_leaf,
        trace_xgb_decision_path,
    )

    feature_names = ["dilation_tendency_Td", "slip_tendency_Ts", "bouguer_gravity_mgal"]
    raw_tree = {
        "nodeid": 0,
        "depth": 0,
        "split": "f0",
        "split_condition": 0.75,
        "yes": 1,
        "no": 2,
        "missing": 1,
        "gain": 42.5,
        "cover": 120.0,
        "children": [
            {"nodeid": 1, "leaf": -0.15, "cover": 80.0},
            {"nodeid": 2, "leaf": 0.35, "cover": 40.0},
        ],
    }
    parsed = _enrich_xgb_tree_node(raw_tree, feature_names)
    assert parsed["is_leaf"] is False
    assert parsed["feature_name"] == "dilation_tendency_Td"
    assert len(parsed["children"]) == 2

    feat_vec = np.array([0.85, 0.4, -12.0], dtype=np.float32)  # >= 0.75 -> goes right (node 2, leaf=0.35)
    assert np.isclose(_eval_tree_leaf(raw_tree, feat_vec), 0.35)

    traced = trace_xgb_decision_path(parsed, feat_vec.tolist())
    assert traced["active_nodes"] == [0, 2]
    assert traced["leaf_nodeid"] == 2
    assert np.isclose(traced["leaf_weight"], 0.35)
    assert len(UNET_LAYER_CATALOG) == 17


