"""Unit tests for Milestone 2: Algorithmic & Model Integrity (Requirements R3.3, R3.4, R4.1, R4.2, R4.3).

Covers:
  R4.1: Unbiased PU Prior in XGBoost (scale_pos_weight = 1.0 in _BaggedXGBoostHead).
  R4.2: Consistent Pre-Calibration Fusion for Coal (probs_raw_k3 and predict_proba_raw).
  R3.3: Optical Warm-Start & Epistemic Variance combination in TwoStageK3CoalEnsemble.
  R3.4: Sedimentary Coal Target Ranking (domain="coal" formula and domain="metals" preservation).
  R4.3: Class-Decoupled nnPULoss & Propensity Weight slicing in _OnTheFlyPatchDataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

from src.models.pu_pipeline import _BaggedXGBoostHead, TwoStageK3CoalEnsemble
from src.models.unet_model import nnPULoss, MPMDistrictDAEUNet
from src.outputs.prospectivity_map import polygonize_and_rank_targets


def test_r4_1_unbiased_pu_prior_in_xgboost():
    """Verify R4.1: _BaggedXGBoostHead sets scale_pos_weight = 1.0."""
    rng = np.random.default_rng(42)
    N = 100
    D = 5
    X = rng.normal(size=(N, D)).astype(np.float32)
    # Imbalanced labels (e.g. 5 positives, 95 negatives)
    y = np.zeros(N, dtype=np.int32)
    y[:5] = 1

    head = _BaggedXGBoostHead(n_estimators=5, max_depth=2, n_bag=2, random_state=42)
    head.fit(X, y)

    assert len(head.models_) == 2
    for clf in head.models_:
        assert clf.scale_pos_weight == 1.0


def test_r4_2_coal_predict_proba_raw_and_full_outputs():
    """Verify R4.2: TwoStageK3CoalEnsemble exposes probs_raw_k3 and predict_proba_raw."""
    rng = np.random.default_rng(101)
    N = 120
    # Include features matching subcanopy indices and optical indices
    cols = [
        "fault_density_2km",
        "dem_slope",
        "bouguer_gravity_mgal",
        "sentinel2_ferric_oxide",
        "kaolinite_crystallinity",
    ]
    X = pd.DataFrame(rng.uniform(0.1, 1.0, size=(N, len(cols))), columns=cols)
    y_k3 = np.zeros((N, 3), dtype=np.int32)
    y_k3[:6, 0] = 1
    y_k3[6:12, 1] = 1
    y_k3[12:18, 2] = 1

    ensemble = TwoStageK3CoalEnsemble(n_estimators=10, n_bag=2, random_state=42)
    ensemble.fit(X, y_k3)

    outputs = ensemble.predict_full_outputs(X)
    assert "probs_raw_k3" in outputs
    assert "probs_k3" in outputs
    assert outputs["probs_raw_k3"].shape == (N, 3)
    assert outputs["probs_k3"].shape == (N, 3)
    # Verify outputs are in [0, 1]
    assert np.all(outputs["probs_raw_k3"] >= 0.0) and np.all(outputs["probs_raw_k3"] <= 1.0)
    assert np.all(outputs["probs_k3"] >= 0.0) and np.all(outputs["probs_k3"] <= 1.0)

    # Verify predict_proba_raw matches outputs["probs_raw_k3"]
    p_raw = ensemble.predict_proba_raw(X)
    assert p_raw.shape == (N, 3)
    np.testing.assert_allclose(p_raw, outputs["probs_raw_k3"], rtol=1e-5)


def test_r3_3_optical_warm_start_and_epistemic_variance():
    """Verify R3.3: Optical heads receive prior estimators and predict_std fuses variance."""
    rng = np.random.default_rng(202)
    N = 80
    cols = [
        "fault_density_2km",
        "dem_slope",
        "sentinel2_ferric_oxide",
    ]
    X = pd.DataFrame(rng.uniform(0.1, 1.0, size=(N, len(cols))), columns=cols)
    y_k3 = np.zeros((N, 3), dtype=np.int32)
    y_k3[:5, 0] = 1
    y_k3[5:10, 1] = 1
    y_k3[10:15, 2] = 1

    # Round 1
    ens_r1 = TwoStageK3CoalEnsemble(n_estimators=10, n_bag=2, random_state=42)
    ens_r1.fit(X, y_k3, round_idx=1)

    # Round 2 warm-started from Round 1
    ens_r2 = TwoStageK3CoalEnsemble(n_estimators=10, n_bag=2, random_state=42)
    ens_r2.fit(X, y_k3, prior_ensemble=ens_r1, round_idx=2)

    # Check predict_std with default w_canopy=None (backward compatibility)
    std_default = ens_r2.predict_std(X)
    assert std_default.shape == (N, 3)
    assert np.all(std_default >= 0.0)

    # Check predict_std with w_canopy provided
    w_canopy = np.full(N, 0.5, dtype=np.float32)
    std_fused = ens_r2.predict_std(X, w_canopy=w_canopy)
    assert std_fused.shape == (N, 3)
    assert np.all(std_fused >= 0.0)

    # When w_canopy is 0, fused variance must equal default subcanopy variance
    w_zero = np.zeros(N, dtype=np.float32)
    std_zero = ens_r2.predict_std(X, w_canopy=w_zero)
    np.testing.assert_allclose(std_zero, std_default, rtol=1e-5)


def test_r3_4_sedimentary_coal_target_ranking():
    """Verify R3.4: domain='coal' scores polygons using flatness, continuity, and certainty."""
    H, W = 40, 40
    transform = from_origin(115.0, 0.0, 0.005, 0.005)
    profile = {
        "transform": transform,
        "crs": "EPSG:4326",
    }

    # Two distinct target blobs
    cal_prob = np.zeros((H, W), dtype=np.float32)
    lower_L = np.zeros((H, W), dtype=np.float32)
    upper_U = np.zeros((H, W), dtype=np.float32)
    dilation_Td = np.zeros((H, W), dtype=np.float32)
    alteration = np.zeros((H, W), dtype=np.float32)
    cover_m = np.zeros((H, W), dtype=np.float32)

    # Blob 1: High prob, flat slope, low dilation, low uncertainty (IDEAL COAL)
    cal_prob[5:15, 5:15] = 0.85
    lower_L[5:15, 5:15] = 0.70
    upper_U[5:15, 5:15] = 0.90
    dilation_Td[5:15, 5:15] = 0.10

    # Blob 2: High prob, steep slope, high dilation, high uncertainty (DISRUPTED / STEEP)
    cal_prob[25:35, 25:35] = 0.85
    lower_L[25:35, 25:35] = 0.55
    upper_U[25:35, 25:35] = 0.95
    dilation_Td[25:35, 25:35] = 0.90

    dem_slope = np.zeros((H, W), dtype=np.float32)
    dem_slope[5:15, 5:15] = 2.0   # Flat 2 degrees
    dem_slope[25:35, 25:35] = 25.0 # Steep 25 degrees

    extra_layers = {
        "dem_slope": dem_slope,
        "epistemic_uncertainty": (upper_U - lower_L) / 2.0,
    }

    # 1. Test domain="coal"
    gdf_coal = polygonize_and_rank_targets(
        calibrated_prob=cal_prob,
        lower_bound_L=lower_L,
        upper_bound_U=upper_U,
        dilation_tendency_Td=dilation_Td,
        alteration_index=alteration,
        cover_depth_m=cover_m,
        profile=profile,
        l_threshold=0.50,
        domain="coal",
        extra_layers=extra_layers,
    )
    assert len(gdf_coal) == 2
    # Blob 1 must rank #1 over Blob 2 due to lower slope, lower dilation, and lower sigma
    assert gdf_coal.iloc[0]["rank"] == 1
    assert gdf_coal.iloc[0]["rank_score"] > gdf_coal.iloc[1]["rank_score"]

    # 2. Test domain="metals" default preservation
    gdf_metals = polygonize_and_rank_targets(
        calibrated_prob=cal_prob,
        lower_bound_L=lower_L,
        upper_bound_U=upper_U,
        dilation_tendency_Td=dilation_Td,
        alteration_index=alteration,
        cover_depth_m=cover_m,
        profile=profile,
        l_threshold=0.50,
        domain="metals",
    )
    assert len(gdf_metals) == 2
    # In metals, higher dilation T_d gives a higher bonus
    assert gdf_metals.iloc[0]["dilation_tendency_Td_max"] > gdf_metals.iloc[1]["dilation_tendency_Td_max"]


def test_r4_3_class_decoupled_nnpu_loss_and_propensity_weights():
    """Verify R4.3: nnPULoss supports per-head priors and computes independent risks."""
    # 1. Test class-decoupled nnPULoss with multi-head priors
    pi_p = [0.01, 0.05, 0.10, 0.02]
    loss_fn = nnPULoss(pi_p=pi_p, gamma=1.0, beta=0.0)

    B = 8
    K = 4
    # Multi-head logits (B, K)
    logits = torch.randn(B, K, requires_grad=True)
    preds = torch.sigmoid(logits)
    targets = (torch.rand(B, K) > 0.8).float()
    weights = torch.rand(B, K) + 0.1

    loss = loss_fn(preds, targets, sample_weights=weights)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == (B, K)

    # 2. Test backward compatibility with scalar pi_p and 1D inputs
    loss_scalar = nnPULoss(pi_p=0.05)
    logits_1d = torch.tensor([2.0, -2.0, 1.0, -1.0], requires_grad=True)
    preds_1d = torch.sigmoid(logits_1d)
    targets_1d = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss_1d = loss_scalar(preds_1d, targets_1d)
    assert loss_1d.ndim == 0
    loss_1d.backward()
    assert logits_1d.grad is not None

    # 3. Test MPMDistrictDAEUNet.cross_validate_spatial_blocks with propensity_weights_2d
    H, W = 32, 32
    C = 6
    X_cube = np.random.randn(C, H, W).astype(np.float32)
    y_grid = np.zeros((4, H, W), dtype=np.float32)
    y_grid[:, 5:10, 5:10] = 1.0
    block_ids = np.zeros((H, W), dtype=np.int32)
    block_ids[:16, :16] = 0
    block_ids[:16, 16:] = 1
    block_ids[16:, :16] = 2
    block_ids[16:, 16:] = 3

    propensity_2d = np.random.uniform(0.5, 2.0, size=(H, W)).astype(np.float32)

    dae = MPMDistrictDAEUNet(in_channels=C, n_classes=4, window_px=16, batch_size=8, pi_p=pi_p)
    out = dae.cross_validate_spatial_blocks(
        X_grid_50m=X_cube,
        y_grid_k4=y_grid,
        block_ids_50m=block_ids,
        decimate_factor=1,
        epochs=1,
        propensity_weights_2d=propensity_2d,
    )
    assert "dae_recon_err" in out
    assert out["dae_recon_err"].shape == (H, W)
    for k in range(4):
        assert f"oof_prob_k{k}" in out
        assert out[f"oof_prob_k{k}"].shape == (H, W)
