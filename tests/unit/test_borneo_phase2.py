"""Unit and verification tests for the Borneo K=4 Phase 2 Engine.

Covers:
  - Task 2.1: District DAE-U-Net with 15% channel dropout, spatial attention,
    bottleneck DAE reconstruction head, grid decimation, and 50m zoom projection.
  - Task 2.2: Yousefi & Carranza P-A intersection feature weights, independent
    XGBoost heads with 1D sample weights, and two-stage stacking for Head 0.
  - Task 2.3: Pass 4 Closed-Boundary Elkan-Noto Scaling and Pass 5 Multi-Mineral
    Conformal Prediction Sets & 90% Confidence Lower Bounds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.pu_pipeline import (
    K4_HEAD_NAMES,
    TwoStageK4Ensemble,
    build_conformal_target_sets,
    calibrate_closed_boundary_elkan_noto,
    compute_multi_mineral_conformal_bounds,
    compute_pa_intersection_weights,
    estimate_k4_discovery_priors,
    fit_k4_independent_xgboost_heads,
)
from src.models.unet_model import (
    DistrictDAEUNet,
    MPMDistrictDAEUNet,
    SpatialAttention,
    decimate_spatial_grid,
    project_spatial_grid,
)


# ---------------------------------------------------------------------------
# Task 2.1 Tests: District DAE-U-Net
# ---------------------------------------------------------------------------


def test_spatial_attention_forward():
    """Verify SpatialAttention module computes valid spatial attention weights."""
    att = SpatialAttention(kernel_size=7)
    x = torch.randn(2, 16, 32, 32)
    out = att(x)
    assert out.shape == x.shape
    # Attention values are bounded in [0, 1] applied multiplicatively
    assert not torch.isnan(out).any()


def test_district_dae_unet_architecture_and_forward():
    """Verify DistrictDAEUNet forward pass with 4 heads and bottleneck DAE reconstruction."""
    in_channels = 8
    model = DistrictDAEUNet(
        in_channels=in_channels,
        n_classes=4,
        channel_drop_p=0.15,
    )
    model.train()
    x = torch.randn(2, in_channels, 64, 64)
    preds, recon, recon_err, ctx = model(x)

    # Supervised predictions shape: (B, 4, H, W)
    assert preds.shape == (2, 4, 64, 64)
    # Bottleneck DAE reconstruction shape: (B, in_channels, H, W)
    assert recon.shape == (2, in_channels, 64, 64)
    # Pixel-wise reconstruction error shape: (B, 1, H, W)
    assert recon_err.shape == (2, 1, 64, 64)
    # Context embeddings shape: (B, 64, H, W)
    assert ctx.shape == (2, 64, 64, 64)

    assert not torch.isnan(preds).any()
    assert not torch.isnan(recon).any()
    assert not torch.isnan(recon_err).any()

    # In eval mode, channel dropout is inactive
    model.eval()
    with torch.no_grad():
        preds_eval, recon_eval, _, _ = model(x)
        assert preds_eval.shape == (2, 4, 64, 64)
        assert recon_eval.shape == (2, in_channels, 64, 64)


def test_spatial_grid_decimation_and_projection():
    """Verify decimation from 50m to 250m/500m and smooth zoom projection back."""
    # Create 50m resolution grid (e.g. 100 x 100 cells = 5 km x 5 km)
    H, W = 100, 100
    grid_50m = np.random.randn(H, W).astype(np.float32)

    # Decimate from 50m to 250m (factor 5)
    decimated_250m = decimate_spatial_grid(grid_50m, target_res_m=250.0, native_res_m=50.0)
    assert decimated_250m.shape == (20, 20)

    # Project back to 50m
    projected_50m = project_spatial_grid(decimated_250m, target_shape=(H, W), order=1)
    assert projected_50m.shape == (H, W)
    assert not np.isnan(projected_50m).any()


def test_mpm_district_dae_unet_wrapper():
    """Verify MPMDistrictDAEUNet high-level training and prediction interface."""
    in_channels = 6
    N_samples = 4
    H, W = 32, 32
    X_windows = np.random.randn(N_samples, in_channels, H, W).astype(np.float32)
    # y_targets: shape (N_samples, 4) with {1, -1}
    y_targets = np.full((N_samples, 4), -1, dtype=np.int32)
    y_targets[0, 0] = 1
    y_targets[1, 1] = 1
    y_targets[2, 2] = 1
    y_targets[3, 3] = 1

    wrapper = MPMDistrictDAEUNet(
        in_channels=in_channels,
        n_classes=4,
        window_px=H,
        batch_size=2,
        recon_weight=0.5,
    )
    wrapper.fit(X_windows, y_targets, epochs=2, lr=1e-3)

    test_patch = np.random.randn(1, in_channels, H, W).astype(np.float32)
    preds, recon_err, ctx = wrapper.predict(test_patch)

    assert preds.shape == (1, 4)
    assert recon_err.shape == (1,)
    assert ctx.shape == (1, 4)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()
    assert (recon_err >= 0.0).all()


# ---------------------------------------------------------------------------
# Task 2.2 Tests: P-A Feature Weights & Two-Stage Stacking
# ---------------------------------------------------------------------------


def test_compute_pa_intersection_weights_predictive_vs_noise():
    """Verify that Yousefi & Carranza P-A intersection weights prioritize predictive features."""
    np.random.seed(42)
    N = 500
    n_pos = 40

    X = np.random.randn(N, 3)
    # Feature 0: strongly predictive (high values at deposits)
    X[:n_pos, 0] += 3.5
    # Feature 1: strongly inverse predictive (low values at deposits)
    X[:n_pos, 1] -= 3.0
    # Feature 2: pure uninformative noise (same distribution)

    y = np.zeros(N, dtype=np.int32)
    y[:n_pos] = 1

    weights = compute_pa_intersection_weights(X, y, n_thresholds=80)
    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0)
    assert (weights >= 0.0).all()

    # Both predictive features must have significantly higher weight than noise
    assert weights[0] > weights[2] * 2.0
    assert weights[1] > weights[2] * 2.0


def test_two_stage_k4_ensemble_fit_and_predict():
    """Verify TwoStageK4Ensemble fits 4 independent heads with out-of-fold stacking for Head 0."""
    np.random.seed(42)
    N = 200
    D = 5

    X = np.random.randn(N, D)
    y_k4 = np.zeros((N, 4), dtype=np.int32)
    # Head 0: 20 positives
    y_k4[:20, 0] = 1
    # Head 1: 30 positives
    y_k4[20:50, 1] = 1
    # Head 2: 30 positives
    y_k4[50:80, 2] = 1
    # Head 3: 25 positives
    y_k4[80:105, 3] = 1

    e_logistics = np.random.uniform(0.1, 0.9, N)
    exclusion_masks = np.ones((N, 4), dtype=np.float64)

    ensemble = fit_k4_independent_xgboost_heads(
        X=X,
        y_k4=y_k4,
        e_logistics=e_logistics,
        exclusion_masks=exclusion_masks,
        n_estimators=15,
        n_bag=3,
        random_state=42,
    )

    # Verify out-of-fold probabilities were computed for Head 1 and Head 2
    assert ensemble.oof_prob_skarn_ is not None
    assert ensemble.oof_prob_epithermal_ is not None
    assert len(ensemble.oof_prob_skarn_) == N
    assert len(ensemble.oof_prob_epithermal_) == N

    # Verify P-A weights exist for all 4 heads
    assert len(ensemble.pa_weights_) == 4
    # Head 0 should have D + 2 features due to telescoping out-of-fold features
    assert len(ensemble.pa_weights_[0]) == D + 2
    assert len(ensemble.pa_weights_[1]) == D

    # Verify prediction shapes and ranges
    X_test = np.random.randn(30, D)
    probs = ensemble.predict_proba(X_test)
    assert probs.shape == (30, 4)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    stds = ensemble.predict_std(X_test)
    assert stds.shape == (30, 4)
    assert (stds >= 0.0).all()

    vars_ = ensemble.predict_uncertainty(X_test)
    assert vars_.shape == (30, 4)
    assert np.allclose(vars_, stds ** 2)


# ---------------------------------------------------------------------------
# Task 2.3 Tests: Pass 4 Scaling & Pass 5 Conformal Risk Sets
# ---------------------------------------------------------------------------


def test_pass4_closed_boundary_elkan_noto_scaling():
    """Verify Pass 4 Elkan-Noto scaling with closed-boundary gate and no propensity division."""
    p_raw = np.array([0.05, 0.20, 0.50, 0.80])
    c_hat = 0.25
    gamma_closed = np.array([0.0, 0.5, 1.0, 1.0])

    p_cal = calibrate_closed_boundary_elkan_noto(
        p_raw_k=p_raw,
        c_hat_k=c_hat,
        gamma_closed_k=gamma_closed,
        min_c=0.10,
    )

    # When gamma_closed == 0, p_cal must be exactly 0
    assert p_cal[0] == 0.0
    # Scaled: 0.20 / 0.25 = 0.80; with gamma=0.5 -> 0.40
    assert np.isclose(p_cal[1], 0.40)
    # Scaled: 0.50 / 0.25 = 2.0 -> capped at 1.0
    assert np.isclose(p_cal[2], 1.0)
    assert np.isclose(p_cal[3], 1.0)


def test_estimate_k4_discovery_priors_spatial_independence():
    """Verify discovery prior estimation enforces spatial block disjointness."""
    val_probs = np.full((100, 4), 0.4)
    val_targets = np.zeros((100, 4), dtype=np.int32)
    val_targets[:10, :] = 1

    train_blocks = np.array([1, 2, 3])
    val_blocks_leak = np.array([3, 4, 5])
    val_blocks_disjoint = np.array([4, 5, 6])

    # Leaking blocks must raise ValueError (Acceptance Criterion 4)
    with pytest.raises(ValueError, match="leakage"):
        estimate_k4_discovery_priors(
            val_probs_k4=val_probs,
            val_targets_k4=val_targets,
            train_block_ids=train_blocks,
            val_block_ids=val_blocks_leak,
        )

    # Disjoint blocks succeed
    c_priors = estimate_k4_discovery_priors(
        val_probs_k4=val_probs,
        val_targets_k4=val_targets,
        train_block_ids=train_blocks,
        val_block_ids=val_blocks_disjoint,
    )
    assert len(c_priors) == 4
    assert np.allclose(c_priors, 0.4)


def test_pass5_multi_mineral_conformal_bounds_and_sets():
    """Verify Pass 5 nonconformity scores, finite-sample quantiles, lower bounds, and target sets."""
    # Held-out calibration positives
    n_cal_pos = 20
    cal_pos_p = np.full(n_cal_pos, 0.85)
    cal_pos_sig = np.full(n_cal_pos, 0.06)

    # Query cells
    p_cal = np.array([0.95, 0.80, 0.40, 0.10])
    sigma = np.array([0.04, 0.08, 0.10, 0.02])

    l_bound, v_scores, q_hat = compute_multi_mineral_conformal_bounds(
        p_cal_k=p_cal,
        sigma_ens_k=sigma,
        cal_pos_p_cal=cal_pos_p,
        cal_pos_sigma=cal_pos_sig,
        alpha=0.10,
    )

    assert len(l_bound) == 4
    assert len(v_scores) == 4
    assert 0.0 < q_hat < 1.0

    # Key conformal invariant: L > 0 if and only if V <= q_hat
    is_conformal = (v_scores <= q_hat)
    assert np.all((l_bound > 0.0) == is_conformal)
    # Lower bound must be <= p_cal
    assert np.all(l_bound <= p_cal + 1e-6)

    # Target sets
    # Mock nonconformity matrix for 4 heads across 2 cells
    v_matrix = np.array([
        [0.10, 0.15, 0.90, 0.85],  # cell 0: confident in heads 0 and 1
        [0.85, 0.90, 0.12, 0.88],  # cell 1: confident in head 2
    ])
    q_vec = [0.30, 0.30, 0.30, 0.30]
    target_sets = build_conformal_target_sets(v_matrix, q_vec)

    assert target_sets[0] == ["porphyry_cu", "skarn_fe_cu"]
    assert target_sets[1] == ["epithermal_au"]
