"""Dedicated unit tests for MPM Pipeline v4.0 mathematical corrections and bug fixes.

Validates:
- D1, D9, Bug 1: Decoupled conformal prediction set gates and studentized lower bounds under exploration shift.
- D2, Bug 5: Prior-corrected Bayesian optical Weight of Evidence in TwoStageK3CoalEnsemble.
- D3, D4: Hájek self-normalized positive weights, neutral unlabeled weights, and monotonic logit scaling.
- D4, D6, Bug 3, Bug 8: MPMDistrictDAEUNet vector priors, bias init, ASPP bottleneck, and 0-positive masking.
- D5, Bug 4: Hierarchical coal presence and conditional rank softmax with continuous MacroBasinSequenceGate.
- D6, Bug 10: Spatial geometry scaling in physical meters (px_size_m), coal rim halos, and telemetry.
- D7: Lowell-Guilbert Porphyry Cu concentric lateral zonation features and monotonic constraints.
- D8: Unified 5-fold buffered spatial cross-validation trunk with physical buffer B_m.
- D10: Causal propensity covariates restriction (roads, rivers, ports) and slope exclusion.
- Bug 6: Composite chain-rule TreeSHAP for coal and site card physical gates integration.
- Bug 7: Feature cube physical layer dataframe fallback reconstruction and 2nd-98th percentile clipping.
- Bug 9: Canonical schema centralization in src/schema/canonical.py and leakage prevention.
- R3: Backward-compatible facade module re-exports in src/models/pu_pipeline.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.spatial_cv import (
    BufferedSpatialBlockKFold,
    assemble_oof_prediction_cube,
    compute_physical_buffer_m,
    spatial_buffered_block_kfold,
)
from src.labels.borneo_labels import (
    COAL_RIM_HALO_METERS,
    get_rank_adaptive_offsets_cells,
)
from src.labels.pu_negatives import (
    CANONICAL_PROPENSITY_COLS,
    FORBIDDEN_PROPENSITY_PATTERNS,
    compute_asymmetric_pu_weights,
    fit_pooled_exploration_propensity,
)
from src.models.calibration import (
    calibrate_elkan_noto_posterior,
    calibrate_monotonic_logit_scaling,
    estimate_elkan_noto_c,
)
from src.models.coal import (
    BARREN_OPTICAL_COAL_FEATURES,
    SUBCANOPY_COAL_FEATURES,
    MacroBasinSequenceGate,
    TwoStageK3CoalEnsemble,
    compute_hierarchical_coal_probabilities,
    fuse_canopy_log_odds,
)
from src.models.conformal import (
    ConformalQuantile,
    build_conformal_target_sets,
    compute_multi_mineral_conformal_bounds,
    compute_studentized_conformal_lower_bound,
    evaluate_conformal_set_gate,
    weighted_conformal_quantile,
)
from src.models.fusion import (
    blend_predictions_by_pr_auc,
    compute_ensemble_weights,
    precision_weighted_log_odds_fusion,
)
from src.models.metals import (
    ZONATION_META_FEATURE_NAMES,
    _BaggedXGBoostHead,
    TwoStageK4Ensemble,
    compute_concentric_zonation_features,
    compute_hajek_asymmetric_head_weights,
    compute_pa_intersection_weights,
)
from src.models.unet_model import (
    ASPPBottleneck,
    MPMDistrictDAEUNet,
    nnPULoss,
)
from src.outputs.prospectivity_map import (
    _norm_01,
    emit_site_cards_json,
    resolve_physical_layers,
)
from src.schema.canonical import (
    FORBIDDEN_RAW_BAND_REGEXES,
    K3_COAL_HEAD_NAMES,
    K4_HEAD_NAMES,
    NON_FEATURE_COLS,
    NON_FEATURE_COLS_SET,
    verify_no_raw_pixel_leakage,
)


# ---------------------------------------------------------------------------
# D1, D9, Bug 1: Decoupled Conformal Bounds & Importance Weighting
# ---------------------------------------------------------------------------


def test_d1_d9_bug1_decoupled_conformal_bounds():
    """Verify D1, D9, and Bug 1: decoupled prediction set gate and studentized lower bound."""
    # 1. Test weighted_conformal_quantile with exploration shift likelihood weights w = 1 / clip(e, 0.05, 1.0)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    propensity = np.array([0.1, 0.2, 0.5, 0.8, 0.9, 0.3, 0.2, 0.1, 0.05, 0.05])
    w_conf = 1.0 / np.clip(propensity, 0.05, 1.0)
    q_weighted = weighted_conformal_quantile(scores, weights=w_conf, alpha=0.10)
    assert 0.0 < q_weighted <= 1.0

    # 2. Test ConformalQuantile class properties
    cq = ConformalQuantile(q_set=0.85, q_res=1.25)
    assert isinstance(cq, float)
    assert cq.q_set == 0.85
    assert cq.q_res == 1.25
    assert float(cq) == 0.85

    # 3. Test studentized lower bound: p_low90 = clip(p_cal - q_res * sigma, 0, 1)
    p_cal = np.array([0.80, 0.50, 0.20, 0.05])
    sigma_low = np.array([0.02, 0.02, 0.02, 0.02])
    sigma_high = np.array([0.20, 0.20, 0.20, 0.20])
    q_res = 1.5

    low_bound_low_unc = compute_studentized_conformal_lower_bound(p_cal, sigma_low, q_res)
    low_bound_high_unc = compute_studentized_conformal_lower_bound(p_cal, sigma_high, q_res)

    # Inversely scaling with uncertainty without negative distortion
    assert np.all(low_bound_low_unc >= low_bound_high_unc)
    assert np.all(low_bound_high_unc >= 0.0)
    assert np.all(low_bound_low_unc <= 1.0)
    assert np.isclose(low_bound_low_unc[0], np.clip(0.80 - 1.5 * 0.02, 0, 1))

    # 4. Test evaluate_conformal_set_gate: p_cal - 0.5 * sigma >= 1 - q_set
    set_gate = evaluate_conformal_set_gate(p_cal, sigma_low, q_set_90=0.85)
    assert set_gate.dtype == bool
    assert set_gate[0] == True  # 0.80 - 0.5*0.02 = 0.79 >= 1 - 0.85 = 0.15

    # 5. Test compute_multi_mineral_conformal_bounds in 2D multi-head mode
    N, K = 50, 4
    rng = np.random.default_rng(42)
    probs_2d = rng.uniform(0.1, 0.9, size=(N, K))
    sigma_2d = rng.uniform(0.01, 0.1, size=(N, K))
    y_2d = np.zeros((N, K), dtype=np.int32)
    y_2d[:10, :] = 1  # 10 positives per head

    p_gates, p_low, q_set_dict, q_res_dict = compute_multi_mineral_conformal_bounds(
        p_cal_k=probs_2d,
        sigma_ens_k=sigma_2d,
        y_k=y_2d,
        alpha=0.10,
    )
    assert p_gates.shape == (N, K)
    assert p_low.shape == (N, K)
    assert len(q_set_dict) == K
    assert len(q_res_dict) == K
    assert np.all(p_low >= 0.0) and np.all(p_low <= 1.0)

    # 6. Test build_conformal_target_sets
    v_scores = 1.0 - probs_2d + 0.5 * sigma_2d
    target_sets = build_conformal_target_sets(v_scores, [q_set_dict[k] for k in range(K)])
    assert len(target_sets) == N
    assert isinstance(target_sets[0], list)


# ---------------------------------------------------------------------------
# D2, Bug 5: Prior-Corrected Bayesian Optical WoE
# ---------------------------------------------------------------------------


def test_d2_bug5_prior_corrected_optical_woe():
    """Verify D2 and Bug 5: Bayesian optical WoE log-odds fusion with prior offset."""
    # 1. Test low-probability regime positive lift
    # Suppose p_opt = 0.20, pi_opt = 0.04. Without prior offset, logit(0.20) < 0,
    # which would penalize despite p_opt being 5x above geological prevalence.
    # Prior-corrected WoE: logit(0.20) - logit(0.04) > 0 -> positive evidential lift!
    p_sub = np.array([0.10])
    p_opt = np.array([0.20])
    pi_opt = 0.04
    w_canopy = np.array([0.80])  # high optical confidence (e.g. open pit / clear canopy)

    from src.models.coal import _logit, _sigmoid

    lo_sub = _logit(p_sub)
    ll_opt = _logit(p_opt)
    lo_pi_opt = _logit(np.array([pi_opt]))[0]

    fused_lo = fuse_canopy_log_odds(
        log_odds_basin=lo_sub,
        log_lambda_optical=ll_opt,
        w_canopy=w_canopy,
        logit_pi_opt=lo_pi_opt,
    )
    fused_p = _sigmoid(fused_lo)
    assert fused_p[0] > p_sub[0], "Prior-corrected optical evidence must boost prospectivity when p_opt > pi_opt"

    # 2. Test below-prior suppression (p_opt <= pi_opt)
    p_opt_low = np.array([0.02])
    ll_opt_low = _logit(p_opt_low)
    fused_lo_low = fuse_canopy_log_odds(
        log_odds_basin=lo_sub,
        log_lambda_optical=ll_opt_low,
        w_canopy=w_canopy,
        logit_pi_opt=lo_pi_opt,
    )
    # When p_opt <= pi_opt, diff is negative, clipped to 0 by max(0, .), so fused == basin
    assert np.isclose(fused_lo_low[0], lo_sub[0]), "Below-prior optical scores must not penalize sub-canopy baseline"

    # 3. Test feature inventory sets
    assert len(SUBCANOPY_COAL_FEATURES) > 30
    assert "sediment_thickness_m" in SUBCANOPY_COAL_FEATURES
    assert "gravity_residual_basin_mgal" in SUBCANOPY_COAL_FEATURES
    assert "spec_albedo_gated" in BARREN_OPTICAL_COAL_FEATURES


# ---------------------------------------------------------------------------
# D3, D4: Hájek Normalization & Neutral Unlabeled Weights
# ---------------------------------------------------------------------------


def test_d3_d4_hajek_positive_normalization():
    """Verify D3 and D4: Hájek self-normalized positive weights and neutral unlabeled weights."""
    N = 100
    labels = np.full(N, -1, dtype=np.int32)
    labels[:20] = 1  # 20 positives

    rng = np.random.default_rng(123)
    propensities = rng.uniform(0.05, 0.95, size=N)
    exclusion_mask = np.ones(N, dtype=np.float64)
    exclusion_mask[80:] = 0.0  # 20 unlabeled inside exclusion zone

    # Test compute_hajek_asymmetric_head_weights
    w_metals = compute_hajek_asymmetric_head_weights(
        labels_k=labels,
        e_logistics=propensities,
        exclusion_mask_k=exclusion_mask,
    )
    # Positive weights self-normalized to mean 1.0 over P_k
    pos_weights = w_metals[labels == 1]
    assert np.isclose(np.mean(pos_weights), 1.0, atol=1e-6)

    # Unlabeled weights outside exclusion zone are neutral 1.0
    unl_outside = w_metals[(labels == -1) & (exclusion_mask == 1.0)]
    assert np.all(np.isclose(unl_outside, 1.0))

    # Unlabeled weights inside exclusion zone are 0.0
    unl_inside = w_metals[(labels == -1) & (exclusion_mask == 0.0)]
    assert np.all(np.isclose(unl_inside, 0.0))

    # Test compute_asymmetric_pu_weights in pu_negatives
    w_pu = compute_asymmetric_pu_weights(
        labels_k=labels,
        e_logistics=propensities,
        exclusion_mask_k=exclusion_mask,
        hajek_normalize=True,
    )
    assert np.isclose(np.mean(w_pu[labels == 1]), 1.0, atol=1e-6)

    # Test monotonic logit scaling with domain prior offset
    raw_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    c_hat = 0.50
    pi_geol = 0.05
    cal_scores = calibrate_monotonic_logit_scaling(raw_scores, c_hat=c_hat, pi_k=pi_geol)
    assert np.all(cal_scores >= 0.0) and np.all(cal_scores <= 1.0)
    assert np.all(np.diff(cal_scores) > 0)  # strictly monotonic


# ---------------------------------------------------------------------------
# D4, D6, Bug 3, Bug 8: U-Net Upgrades & Zero-Positive Masking
# ---------------------------------------------------------------------------


def test_d4_d6_bug3_bug8_unet_upgrades():
    """Verify D4, D6, Bug 3, Bug 8: ASPPBottleneck, nnPULoss, and zero-positive head masking."""
    # 1. Test ASPPBottleneck with physical meters parameterization (D6, Bug 8)
    aspp_250 = ASPPBottleneck(in_channels=1024, out_channels=1024, px_size_m=250.0)
    x_feat = torch.randn(2, 1024, 8, 8)
    out_aspp = aspp_250(x_feat)
    assert out_aspp.shape == (2, 1024, 8, 8)

    # Test resolution adaptation for 100m
    aspp_100 = ASPPBottleneck(in_channels=1024, out_channels=1024, px_size_m=100.0)
    out_aspp_100 = aspp_100(x_feat)
    assert out_aspp_100.shape == (2, 1024, 8, 8)

    # 2. Test nnPULoss with vector priors and zero-positive head masking (D4, Bug 3)
    priors_k4 = [0.005, 0.010, 0.015, 0.008]
    loss_fn = nnPULoss(pi_p=priors_k4, gamma=1.0, beta=0.0)

    B, K, H, W = 2, 4, 16, 16
    preds = torch.sigmoid(torch.randn(B, K, H, W))
    targets = torch.zeros(B, K, H, W)
    # Head 0, 1, 2 have positive labels; Head 3 has ZERO positives (unobserved head)
    targets[:, 0, :2, :2] = 1.0
    targets[:, 1, :2, :2] = 1.0
    targets[:, 2, :2, :2] = 1.0
    # targets[:, 3] remains 0.0 (Bug 3 zero-positive head)

    sample_weights = torch.ones(B, K, H, W)
    loss = loss_fn(preds, targets, sample_weights=sample_weights)

    assert not torch.isnan(loss)
    assert loss.item() >= 0.0, "nnPU risk loss must be non-negative"

    # 3. Test bias initialization to logit(pi_k) in DistrictDAEUNet
    from src.models.unet_model import DistrictDAEUNet
    dae_unet = DistrictDAEUNet(
        in_channels=12,
        n_classes=4,
        px_size_m=250.0,
        pi_p=priors_k4,
    )
    bias = dae_unet.out_heads.bias.detach().cpu().numpy()
    expected_bias = np.log(np.array(priors_k4) / (1.0 - np.array(priors_k4)))
    assert np.allclose(bias, expected_bias, atol=1e-3)


# ---------------------------------------------------------------------------
# D5, Bug 4: Hierarchical Coal Softmax & Macro Basin Gate
# ---------------------------------------------------------------------------


def test_d5_bug4_hierarchical_coal_softmax():
    """Verify D5 and Bug 4: Hierarchical coal presence and conditional rank softmax."""
    # 1. Test MacroBasinSequenceGate with continuous physical features (Bug 4)
    gate = MacroBasinSequenceGate()
    feature_names = [
        "dist_sedimentary_polygon_km",
        "sar_moisture_index",
        "mag_analytic_signal",
        "sediment_thickness_m",
        "gravity_residual_basin_mgal",
        "dem_slope",
    ]
    # Inside basin: thick sediment, low gravity, close to polygon, flat slope
    X_inside = np.array([[1.0, 0.7, 50.0, 2000.0, -15.0, 3.0]])
    # Outside basin: thin sediment, high gravity, far from polygon, steep slope
    X_outside = np.array([[45.0, 0.1, 350.0, 50.0, 40.0, 35.0]])

    g_in = gate.compute_gate(X_inside, feature_names=feature_names)
    g_out = gate.compute_gate(X_outside, feature_names=feature_names)
    assert g_in[0] > g_out[0], "Basin gate must be higher inside sedimentary sequence"
    assert 0.0 <= g_in[0] <= 1.0
    assert 0.0 <= g_out[0] <= 1.0

    # 2. Test compute_hierarchical_coal_probabilities (D5)
    p_cal_any = np.array([0.80, 0.40, 0.10])
    basin_gate_vals = np.array([0.90, 0.50, 0.00])
    rank_logits = np.array([
        [2.0, 1.0, 0.0],
        [0.5, 1.5, 0.5],
        [0.0, 0.0, 2.0],
    ])

    p_k3_final, p_coal_total = compute_hierarchical_coal_probabilities(
        p_cal_coal_any=p_cal_any,
        rank_logits=rank_logits,
        basin_gate=basin_gate_vals,
        temperature=1.0,
    )
    assert p_k3_final.shape == (3, 3)
    assert p_coal_total.shape == (3,)

    # Total coal presence is G_basin * p_any
    assert np.isclose(p_coal_total[0], 0.90 * 0.80)
    assert np.isclose(p_coal_total[2], 0.0)  # outside basin -> 0.0

    # Mutually exclusive rank partition: sum of rank probabilities equals p_coal_total
    rank_sums = np.sum(p_k3_final, axis=1)
    assert np.allclose(rank_sums, p_coal_total, atol=1e-5)


# ---------------------------------------------------------------------------
# D6, Bug 10: Spatial Geometry Scaling in Physical Meters
# ---------------------------------------------------------------------------


def test_d6_bug10_spatial_geometry_meters():
    """Verify D6 and Bug 10: coal rim halo distances and physical pixel scaling."""
    # Physical meter invariants
    assert COAL_RIM_HALO_METERS[0] == (1000.0, 2000.0)
    assert COAL_RIM_HALO_METERS[1] == (1500.0, 3000.0)
    assert COAL_RIM_HALO_METERS[2] == (1500.0, 3500.0)

    # 250m grid scaling
    offsets_250 = get_rank_adaptive_offsets_cells(250.0)
    assert offsets_250[0] == (4, 8)
    assert offsets_250[1] == (6, 12)
    assert offsets_250[2] == (6, 14)

    # 100m grid scaling
    offsets_100 = get_rank_adaptive_offsets_cells(100.0)
    assert offsets_100[0] == (10, 20)
    assert offsets_100[1] == (15, 30)
    assert offsets_100[2] == (15, 35)

    # 50m grid scaling
    offsets_50 = get_rank_adaptive_offsets_cells(50.0)
    assert offsets_50[0] == (20, 40)
    assert offsets_50[1] == (30, 60)
    assert offsets_50[2] == (30, 70)


# ---------------------------------------------------------------------------
# D7: Porphyry Cu Concentric Zonation & Monotonic Constraints
# ---------------------------------------------------------------------------


def test_d7_porphyry_cu_zonation_and_monotonic():
    """Verify D7: Lowell-Guilbert concentric lateral zonation features and monotonic constraints."""
    N = 100
    p_skarn = np.linspace(0.1, 0.9, N, dtype=np.float32)
    p_epithermal = np.linspace(0.9, 0.1, N, dtype=np.float32)
    p_laterite = np.full(N, 0.05, dtype=np.float32)
    coords_m = np.column_stack([np.linspace(0, 10000, N), np.linspace(0, 10000, N)])

    zonation_matrix = compute_concentric_zonation_features(
        p_skarn=p_skarn,
        p_epithermal=p_epithermal,
        p_laterite=p_laterite,
        coords_m=coords_m,
    )
    assert zonation_matrix.shape == (N, 7)
    assert len(ZONATION_META_FEATURE_NAMES) == 7

    # Verify monotonic constraints in _BaggedXGBoostHead
    # (+1 on skarn & epithermal, -1 on laterite)
    monotone = (1, 1, 0, 0, 0, 0, -1)
    head = _BaggedXGBoostHead(
        n_estimators=10,
        max_depth=4,
        n_bag=2,
        monotone_constraints=monotone,
        random_state=42,
    )
    X_synth = zonation_matrix[:40]
    y_synth = np.zeros(40, dtype=np.int32)
    y_synth[:10] = 1

    head.fit(X_synth, y_synth)
    preds = head.predict_proba(X_synth)
    assert preds.shape == (40,)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


# ---------------------------------------------------------------------------
# D8: Unified 5-Fold Buffered Spatial Cross-Validation
# ---------------------------------------------------------------------------


def test_d8_unified_buffered_spatial_cv():
    """Verify D8: 5-fold buffered spatial cross-validation trunk with physical buffer B_m."""
    # 1. Test physical buffer calculation B_m = max(r*_k, 16 * px_size_m)
    assert compute_physical_buffer_m(px_size_m=250.0, r_k_star_m=6000.0) == 6000.0
    assert compute_physical_buffer_m(px_size_m=500.0, r_k_star_m=6000.0) == 8000.0  # 16 * 500 = 8000 > 6000

    # 2. Test spatial_buffered_block_kfold
    # Synthetic grid of 400 points across 20 km x 20 km
    xs = np.tile(np.linspace(0, 20000, 20), 20)
    ys = np.repeat(np.linspace(0, 20000, 20), 20)
    coords = np.column_stack([xs, ys])

    splits = spatial_buffered_block_kfold(
        coords,
        n_splits=5,
        block_size_m=5000.0,
        buffer_m=2000.0,
        px_size_m=250.0,
        random_state=42,
    )
    assert len(splits) == 5

    all_val_indices = []
    for f_idx, (train_idx, val_idx) in enumerate(splits):
        assert len(val_idx) > 0
        assert len(train_idx) > 0
        # No overlap between train and val
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        all_val_indices.extend(val_idx)

        # Check physical buffer zone: no training cell within buffer_m of any val cell
        from scipy.spatial import cKDTree
        val_tree = cKDTree(coords[val_idx])
        dists, _ = val_tree.query(coords[train_idx], k=1)
        assert np.all(dists >= 2000.0 - 1e-3), f"Fold {f_idx}: training sample found inside buffer zone!"

    # 100% out-of-fold coverage
    assert set(all_val_indices) == set(range(len(coords)))

    # 3. Test assemble_oof_prediction_cube
    fold_preds = [np.ones(len(val_idx), dtype=np.float32) * (i + 1) for i, (_, val_idx) in enumerate(splits)]
    val_indices = [val_idx for _, val_idx in splits]
    oof_cube = assemble_oof_prediction_cube(fold_preds, val_indices, n_total_samples=len(coords))
    assert oof_cube.shape == (len(coords),)
    assert np.all(oof_cube > 0.0)


# ---------------------------------------------------------------------------
# D10: Causal Propensity Covariates Restriction
# ---------------------------------------------------------------------------


def test_d10_causal_propensity_covariates():
    """Verify D10: Propensity model covariates restricted to pre-mining regional infrastructure."""
    # Verify canonical columns
    assert "dist_roads_km" in CANONICAL_PROPENSITY_COLS
    assert "dist_major_rivers_km" in CANONICAL_PROPENSITY_COLS
    assert "dist_ports_km" in CANONICAL_PROPENSITY_COLS

    # Verify forbidden patterns
    assert "slope" in FORBIDDEN_PROPENSITY_PATTERNS
    assert "dem" in FORBIDDEN_PROPENSITY_PATTERNS
    assert "elevation" in FORBIDDEN_PROPENSITY_PATTERNS
    assert "haul_roads" in FORBIDDEN_PROPENSITY_PATTERNS

    # Test fit_pooled_exploration_propensity column filtering
    N = 50
    df_features = pd.DataFrame({
        "dist_roads": np.linspace(1, 20, N),
        "dist_rivers": np.linspace(2, 30, N),
        "dist_ports": np.linspace(5, 50, N),
        "dem_slope": np.linspace(0, 45, N),          # forbidden
        "dem_elevation": np.linspace(10, 1000, N),    # forbidden
        "dist_haul_roads": np.linspace(0.1, 5, N),   # forbidden
    })
    explored = np.zeros(N, dtype=np.int32)
    explored[:15] = 1

    e_hat = fit_pooled_exploration_propensity(
        accessibility_features=df_features,
        explored_footprint=explored,
        min_propensity=0.05,
        max_propensity=1.0,
    )
    assert len(e_hat) == N
    assert np.all(e_hat >= 0.05) and np.all(e_hat <= 1.0)


# ---------------------------------------------------------------------------
# Bug 6: Coal TreeSHAP & Site Card Physical Gates
# ---------------------------------------------------------------------------


def test_bug6_coal_treeshap_and_site_cards(tmp_path):
    """Verify Bug 6: Composite chain-rule TreeSHAP in TwoStageK3CoalEnsemble and site card gates."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    # 1. Test TwoStageK3CoalEnsemble composite TreeSHAP
    ensemble = TwoStageK3CoalEnsemble(n_estimators=10, n_bag=2, random_state=42)
    feature_cols = [
        "sediment_thickness_m",
        "gravity_residual_basin_mgal",
        "dem_slope",
        "spec_albedo_gated",
        "spec_wcdi_gated",
    ]
    N = 40
    rng = np.random.default_rng(99)
    X = pd.DataFrame(rng.uniform(0.1, 10.0, size=(N, len(feature_cols))), columns=feature_cols)
    y_k3 = np.zeros((N, 3), dtype=np.int32)
    y_k3[:5, 0] = 1
    y_k3[5:10, 1] = 1
    y_k3[10:15, 2] = 1
    w_canopy = rng.uniform(0.2, 0.9, size=N)

    ensemble.fit(X, y_k3, w_canopy=w_canopy)
    shap_vals, feat_names = ensemble.compute_head_shap(head=0, X=X, w_canopy=w_canopy)
    assert shap_vals.shape[0] == N
    assert len(feat_names) > 0
    assert not np.isnan(shap_vals).any()

    # 2. Test emit_site_cards_json accepts outmogc_gates and basin_gate
    targets_gdf = gpd.GeoDataFrame({
        "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        "mean_score": [0.85],
        "max_score": [0.92],
        "area_ha": [25.0],
        "mean_uncertainty": [0.04],
        "target_id": ["TARGET_001"],
    }, crs="EPSG:4326")

    out_json = str(tmp_path / "site_cards.json")
    cards = emit_site_cards_json(
        ranked_targets_gdf=targets_gdf,
        out_path=out_json,
        outmogc_gates=np.array([[1.0, 1.0, 0.8, 0.5]]),
        basin_gate=np.array([0.95]),
    )
    assert len(cards) == 1
    # Check that basin_gate is recorded in site card
    assert "basin_gate" in cards[0] or "macro_basin_sequence_gate" in cards[0]


# ---------------------------------------------------------------------------
# Bug 7: Feature Cube Missing Physical Layers Fallback
# ---------------------------------------------------------------------------


def test_bug7_physical_layer_fallback_norm():
    """Verify Bug 7: 2nd-98th percentile clipping in _norm_01 and physical layer resolver."""
    # Outlier test: extreme spike in 100 samples must not compress normal range
    vals = np.linspace(1.0, 10.0, 100)
    vals[-1] = 1000.0  # extreme outlier at 99th percentile
    normed = _norm_01(vals)
    assert normed.shape == vals.shape
    assert np.all(normed >= 0.0) and np.all(normed <= 1.0)
    assert normed[-1] == 1.0  # clipped to p98 and scaled to 1.0
    # Middle values are preserved across the range
    assert normed[50] > 0.3

    # Constant array test
    const_vals = np.full(10, 42.0)
    normed_const = _norm_01(const_vals)
    assert np.all(normed_const == 0.5)

    # Empty array test
    assert len(_norm_01(np.array([]))) == 0

    # Test resolve_physical_layers fallback when stack is None
    layers = resolve_physical_layers(None)
    assert isinstance(layers, dict)
    assert layers["dilation_tendency_Td"] is None


# ---------------------------------------------------------------------------
# Bug 9: Canonical Schema Centralization
# ---------------------------------------------------------------------------


def test_bug9_canonical_schema_centralization():
    """Verify Bug 9: Centralized schemas and leak-prevention validation."""
    assert len(NON_FEATURE_COLS) == 24
    assert "geology_coal_contact_dist_km" in NON_FEATURE_COLS
    assert "dist_kelian_center_m" in NON_FEATURE_COLS
    assert "in_coal_bearing_formation" in NON_FEATURE_COLS

    # Test verify_no_raw_pixel_leakage
    clean_features = ["dem_slope", "gravity_bouguer_mgal", "sar_vv_vh_ratio", "spec_albedo_gated"]
    assert verify_no_raw_pixel_leakage(clean_features) == True

    # Raw band leak must raise ValueError
    with pytest.raises(ValueError, match="Raw sensor pixel leakage"):
        verify_no_raw_pixel_leakage(["b01", "dem_slope"])

    with pytest.raises(ValueError, match="Raw sensor pixel leakage"):
        verify_no_raw_pixel_leakage(["aster_b4", "dem_slope"])

    with pytest.raises(ValueError, match="Raw sensor pixel leakage"):
        verify_no_raw_pixel_leakage(["dist_kelian_center_m", "dem_slope"])


# ---------------------------------------------------------------------------
# R3: Backward-Compatible Facade Module Re-exports
# ---------------------------------------------------------------------------


def test_r3_facade_backward_compatibility():
    """Verify R3: 100% backward-compatible symbol re-exports in src.models.pu_pipeline."""
    import src.models.pu_pipeline as facade

    # Model classes
    assert hasattr(facade, "TwoStageK4Ensemble")
    assert hasattr(facade, "TwoStageK3CoalEnsemble")
    assert hasattr(facade, "MacroBasinSequenceGate")
    assert hasattr(facade, "_BaggedXGBoostHead")
    assert hasattr(facade, "OUTMOGCGate")
    assert hasattr(facade, "MPMXGBoost")

    # Functions
    assert hasattr(facade, "compute_multi_mineral_conformal_bounds")
    assert hasattr(facade, "compute_studentized_conformal_lower_bound")
    assert hasattr(facade, "evaluate_conformal_set_gate")
    assert hasattr(facade, "fuse_canopy_log_odds")
    assert hasattr(facade, "compute_hierarchical_coal_probabilities")
    assert hasattr(facade, "blend_predictions_by_pr_auc")
    assert hasattr(facade, "precision_weighted_log_odds_fusion")
    assert hasattr(facade, "estimate_elkan_noto_c")
    assert hasattr(facade, "calibrate_monotonic_logit_scaling")
    assert hasattr(facade, "compute_hajek_asymmetric_head_weights")
    assert hasattr(facade, "compute_asymmetric_head_weights")
    assert hasattr(facade, "compute_concentric_zonation_features")
    assert hasattr(facade, "verify_no_raw_pixel_leakage")

    # Legacy PU routines
    assert hasattr(facade, "pu_settings")
    assert hasattr(facade, "fit_pu_model")

    # Canonical schema constants
    assert hasattr(facade, "NON_FEATURE_COLS")
    assert hasattr(facade, "K4_HEAD_NAMES")
    assert hasattr(facade, "K3_COAL_HEAD_NAMES")
    assert hasattr(facade, "DISCOVERY_PRIORS_K4")
    assert hasattr(facade, "COAL_PRIORS_K3")
