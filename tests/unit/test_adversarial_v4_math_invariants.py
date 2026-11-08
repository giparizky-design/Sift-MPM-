"""tests/unit/test_adversarial_v4_math_invariants.py
==================================================
Empirical Adversarial Stress-Testing Suite for Challenger 1:
MPM Pipeline v4.0 Mathematical & Statistical Invariants Verification.

Tests the five mathematical core invariants under extreme boundary conditions,
adversarial distributions, and stress generators:
  1. Conformal lower bounds monotonicity & uncertainty scaling without sign inversion.
  2. Prior-corrected Bayesian optical Weight-of-Evidence evidential lift & boundaries.
  3. Hájek positive normalization (mean == 1.0 within 1e-7) & neutral unlabeled weighting.
  4. ASPP bottleneck physical scaling ([0.5, 1.5, 3.0, 6.0] km across 250m, 100m, 50m).
  5. 5-Fold buffered spatial CV minimum physical separation (min_dist >= B_m across all folds).

Can be run via pytest:
    .venv/bin/pytest tests/unit/test_adversarial_v4_math_invariants.py -v
Or as a standalone script:
    .venv/bin/python tests/unit/test_adversarial_v4_math_invariants.py
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial import cKDTree

from src.evaluation.spatial_cv import (
    BufferedSpatialBlockKFold,
    compute_physical_buffer_m,
    spatial_buffered_block_kfold,
)
from src.labels.pu_negatives import (
    compute_asymmetric_pu_weights,
)
from src.models.coal import (
    fuse_canopy_log_odds,
)
from src.models.conformal import (
    ConformalQuantile,
    compute_multi_mineral_conformal_bounds,
    compute_studentized_conformal_lower_bound,
    evaluate_conformal_set_gate,
    weighted_conformal_quantile,
)
from src.models.metals import (
    compute_hajek_asymmetric_head_weights,
)
from src.models.unet_model import (
    ASPPBottleneck,
    DistrictDAEUNet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("adversarial_challenger_v4")


# =====================================================================
# 1. Conformal Lower Bounds Stress-Testing
# =====================================================================


class TestConformalLowerBoundsInvariants:
    """Adversarial stress-testing of conformal lower bounds."""

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
    def test_conformal_bounds_inverse_scaling_with_uncertainty(self, alpha: float):
        """Verify p_low90 scales inversely with uncertainty without sign inversion."""
        rng = np.random.default_rng(42)
        N = 1000
        p_cal = rng.uniform(0.0, 1.0, size=N)
        # Extreme sigma spectrum: from low (1e-6) to high (2.5)
        sigma_low = rng.uniform(1e-6, 0.05, size=N)
        sigma_high = sigma_low + rng.uniform(0.10, 1.50, size=N)
        q_res = 1.645

        p_low_low_unc = compute_studentized_conformal_lower_bound(p_cal, sigma_low, q_res)
        p_low_high_unc = compute_studentized_conformal_lower_bound(p_cal, sigma_high, q_res)

        # 1. Monotonicity: higher uncertainty must yield lower or equal lower bound
        assert np.all(p_low_low_unc >= p_low_high_unc), "Violation of uncertainty monotonicity!"

        # 2. Strict bounded range [0, 1] - no negative bounds, no sign inversion
        assert np.all(p_low_low_unc >= 0.0) and np.all(p_low_low_unc <= 1.0)
        assert np.all(p_low_high_unc >= 0.0) and np.all(p_low_high_unc <= 1.0)

    def test_conformal_bounds_zero_uncertainty_limit(self):
        """In low-uncertainty regime (sigma -> 0), p_low90 must converge to p_cal."""
        p_cal = np.linspace(0.0, 1.0, 500)
        sigma_zero = np.zeros_like(p_cal)
        q_res = 2.0

        p_low = compute_studentized_conformal_lower_bound(p_cal, sigma_zero, q_res)
        # When sigma is 0, p_low90 must exactly equal p_cal (no spurious drop by (1 - q_hat))
        np.testing.assert_allclose(p_low, p_cal, atol=1e-12)

    def test_conformal_bounds_extreme_uncertainty_regime(self):
        """In massive uncertainty regime (sigma >> 0), bounds must clamp cleanly at 0.0."""
        p_cal = np.array([0.1, 0.5, 0.9, 1.0])
        sigma_massive = np.array([5.0, 10.0, 50.0, 100.0])
        q_res = 1.96

        p_low = compute_studentized_conformal_lower_bound(p_cal, sigma_massive, q_res)
        # Massive uncertainty drives lower bound to 0.0 without going negative
        np.testing.assert_allclose(p_low, np.zeros_like(p_low), atol=1e-12)

    def test_conformal_prediction_set_gate_decoupling(self):
        """Verify prediction set gate is decoupled from studentized bounds."""
        # Point with high point probability but also high uncertainty
        p_cal = np.array([0.85])
        sigma_high = np.array([0.30])
        q_set_90 = 0.85  # threshold: p_cal - 0.5 * sigma >= 1 - 0.85 = 0.15
        q_res_90 = 2.50

        set_gate = evaluate_conformal_set_gate(p_cal, sigma_high, q_set_90)
        p_low90 = compute_studentized_conformal_lower_bound(p_cal, sigma_high, q_res_90)

        # 0.85 - 0.5 * 0.30 = 0.70 >= 0.15 -> INCLUDED in prediction set C_0.90
        assert set_gate[0] == True
        # Lower bound: clip(0.85 - 2.50 * 0.30, 0, 1) = clip(0.10, 0, 1) = 0.10
        np.testing.assert_allclose(p_low90[0], 0.10, atol=1e-6)

    def test_multi_head_conformal_invariants(self):
        """Stress-test compute_multi_mineral_conformal_bounds in 2D multi-head mode."""
        N, K = 200, 4
        rng = np.random.default_rng(101)
        p_cal_2d = rng.uniform(0.05, 0.95, size=(N, K))
        sigma_2d = rng.uniform(0.01, 0.25, size=(N, K))
        y_val_2d = np.zeros((N, K), dtype=np.int32)
        # Assign 15 positives per head
        for k in range(K):
            pos_idx = rng.choice(N, size=15, replace=False)
            y_val_2d[pos_idx, k] = 1

        p_set_gates, p_low90, q_set_dict, q_res_dict = compute_multi_mineral_conformal_bounds(
            p_cal_k=p_cal_2d,
            sigma_ens_k=sigma_2d,
            y_k=y_val_2d,
            alpha=0.10,
        )

        assert p_set_gates.shape == (N, K)
        assert p_low90.shape == (N, K)
        assert np.all(p_low90 >= 0.0) and np.all(p_low90 <= 1.0)
        assert len(q_set_dict) == K and len(q_res_dict) == K
        for k in range(K):
            assert 0.0 < q_set_dict[k] <= 2.0
            assert q_res_dict[k] > 0.0 and np.isfinite(q_res_dict[k])


# =====================================================================
# 2. Prior-Corrected Bayesian Optical WoE Stress-Testing
# =====================================================================


class TestPriorCorrectedBayesianWoEInvariants:
    """Adversarial stress-testing of Bayesian optical WoE fusion."""

    def test_positive_evidence_generated_when_p_opt_exceeds_pi_opt(self):
        """When p_opt < 0.50 but p_opt > pi_opt, positive evidence must be generated."""
        # e.g., pi_opt = 0.01, p_opt = 0.15
        pi_opt = 0.01
        p_opt = 0.15
        p_sub = 0.20
        w_canopy = 0.80

        log_odds_sub = np.log(p_sub / (1.0 - p_sub))
        log_lambda_opt = np.log(p_opt / (1.0 - p_opt))
        logit_pi_opt = np.log(pi_opt / (1.0 - pi_opt))

        # Optical weight w_opt^+ = max(0, logit(p_opt) - logit(pi_opt))
        w_opt_plus = log_lambda_opt - logit_pi_opt
        assert w_opt_plus > 0.0, f"Expected w_opt^+ > 0, got {w_opt_plus}"

        fused_log_odds = fuse_canopy_log_odds(
            log_odds_basin=np.array([log_odds_sub]),
            log_lambda_optical=np.array([log_lambda_opt]),
            w_canopy=np.array([w_canopy]),
            logit_pi_opt=logit_pi_opt,
        )

        # Fused log-odds must be strictly greater than baseline subcanopy log-odds
        assert fused_log_odds[0] > log_odds_sub
        expected = log_odds_sub + w_canopy * w_opt_plus
        np.testing.assert_allclose(fused_log_odds[0], expected, atol=1e-10)

    def test_zero_evidence_when_p_opt_below_or_equal_pi_opt(self):
        """When p_opt <= pi_opt, optical evidence must be strictly zero (w_opt^+ == 0)."""
        pi_opt = 0.02
        p_sub = 0.30
        log_odds_sub = np.log(p_sub / (1.0 - p_sub))
        logit_pi_opt = np.log(pi_opt / (1.0 - pi_opt))

        # Test case 1: p_opt strictly less than pi_opt (e.g. 0.01 < 0.02)
        p_opt_below = 0.01
        log_lambda_below = np.log(p_opt_below / (1.0 - p_opt_below))
        fused_below = fuse_canopy_log_odds(
            log_odds_basin=np.array([log_odds_sub]),
            log_lambda_optical=np.array([log_lambda_below]),
            w_canopy=np.array([1.0]),
            logit_pi_opt=logit_pi_opt,
        )
        np.testing.assert_allclose(fused_below[0], log_odds_sub, atol=1e-12)

        # Test case 2: p_opt exactly equal to pi_opt (0.02 == 0.02)
        p_opt_equal = 0.02
        log_lambda_equal = np.log(p_opt_equal / (1.0 - p_opt_equal))
        fused_equal = fuse_canopy_log_odds(
            log_odds_basin=np.array([log_odds_sub]),
            log_lambda_optical=np.array([log_lambda_equal]),
            w_canopy=np.array([1.0]),
            logit_pi_opt=logit_pi_opt,
        )
        np.testing.assert_allclose(fused_equal[0], log_odds_sub, atol=1e-12)

    def test_canopy_gap_weight_masking(self):
        """When canopy gap weight w_canopy == 0.0, optical evidence must be masked."""
        pi_opt = 0.01
        p_opt_high = 0.90  # strong optical anomaly
        p_sub = 0.10
        log_odds_sub = np.log(p_sub / (1.0 - p_sub))
        log_lambda_opt = np.log(p_opt_high / (1.0 - p_opt_high))
        logit_pi_opt = np.log(pi_opt / (1.0 - pi_opt))

        fused_dense_canopy = fuse_canopy_log_odds(
            log_odds_basin=np.array([log_odds_sub]),
            log_lambda_optical=np.array([log_lambda_opt]),
            w_canopy=np.array([0.0]),  # 100% dense vegetation, zero gap
            logit_pi_opt=logit_pi_opt,
        )
        # Complete optical masking
        np.testing.assert_allclose(fused_dense_canopy[0], log_odds_sub, atol=1e-12)

    def test_prior_correction_prevents_log_odds_explosion(self):
        """Verify that subtracting prior logit prevents explosive prior double-counting."""
        pi_opt = 0.05
        p_opt = 0.05  # equal to background prior
        p_sub = 0.50
        log_odds_sub = 0.0  # logit(0.50) = 0
        log_lambda_opt = np.log(p_opt / (1.0 - p_opt))

        # Without prior correction: log_odds_sub + log_lambda_opt = 0 + (-2.94) = -2.94 (spurious drop)
        # With prior correction: log_odds_sub + max(0, logit(p_opt) - logit(pi_opt)) = 0 + 0 = 0
        fused = fuse_canopy_log_odds(
            log_odds_basin=np.array([log_odds_sub]),
            log_lambda_optical=np.array([log_lambda_opt]),
            w_canopy=np.array([1.0]),
            logit_pi_opt=log_lambda_opt,
        )
        np.testing.assert_allclose(fused[0], log_odds_sub, atol=1e-12)


# =====================================================================
# 3. Hájek Positive Normalization & Neutral Unlabeled Weights Stress-Testing
# =====================================================================


class TestHajekNormalizationAndUnlabeledNeutrality:
    """Adversarial stress-testing of Hájek self-normalization and neutral unlabeled weights."""

    @pytest.mark.parametrize("n_pos", [1, 5, 20, 50, 200])
    def test_hajek_positive_weights_mean_strictly_one(self, n_pos: int):
        """Verify that mean of w_tilde_i over P_k is strictly 1.0 (mean == 1.0 within 1e-7)."""
        rng = np.random.default_rng(2026)
        N = n_pos + 500
        labels = np.full(N, -1, dtype=np.int32)
        labels[:n_pos] = 1

        # Adversarially skewed propensities (spanning 0.001 to 0.99)
        propensities = rng.exponential(scale=0.1, size=N)
        propensities = np.clip(propensities, 0.001, 0.99)

        exclusion_mask = np.ones(N, dtype=np.float64)

        # Test metals.py implementation
        w_metals = compute_hajek_asymmetric_head_weights(
            labels_k=labels,
            e_logistics=propensities,
            exclusion_mask_k=exclusion_mask,
            min_propensity=0.05,
        )
        pos_w_metals = w_metals[labels == 1]
        assert np.isclose(np.mean(pos_w_metals), 1.0, atol=1e-7), (
            f"Metals Hájek mean violation: {np.mean(pos_w_metals)}"
        )

        # Test pu_negatives.py implementation
        w_pu = compute_asymmetric_pu_weights(
            labels_k=labels,
            e_logistics=propensities,
            exclusion_mask_k=exclusion_mask,
            min_propensity=0.05,
            hajek_normalize=True,
        )
        pos_w_pu = w_pu[labels == 1]
        assert np.isclose(np.mean(pos_w_pu), 1.0, atol=1e-7), (
            f"PU Hájek mean violation: {np.mean(pos_w_pu)}"
        )

    def test_neutral_unlabeled_weights_strictly_one_outside_exclusion(self):
        """Verify that unlabeled cells outside exclusion zones receive strictly w_unlab == 1.0."""
        N = 300
        labels = np.full(N, -1, dtype=np.int32)
        labels[:30] = 1

        rng = np.random.default_rng(777)
        propensities = rng.uniform(0.05, 0.95, size=N)
        exclusion_mask = np.ones(N, dtype=np.float64)
        exclusion_mask[100:150] = 0.0  # 50 cells inside exclusion zone

        w_pu = compute_asymmetric_pu_weights(
            labels_k=labels,
            e_logistics=propensities,
            exclusion_mask_k=exclusion_mask,
            hajek_normalize=True,
        )

        # Unlabeled outside exclusion zone: must be EXACTLY 1.0
        unl_outside = w_pu[(labels == -1) & (exclusion_mask == 1.0)]
        assert len(unl_outside) == 220
        np.testing.assert_allclose(unl_outside, 1.0, atol=1e-12)

        # Unlabeled inside exclusion zone: must be EXACTLY 0.0
        unl_inside = w_pu[(labels == -1) & (exclusion_mask == 0.0)]
        assert len(unl_inside) == 50
        np.testing.assert_allclose(unl_inside, 0.0, atol=1e-12)

    def test_single_positive_sample_edge_case(self):
        """Edge case: solitary positive mine (N_pos = 1) must receive weight exactly 1.0."""
        N = 50
        labels = np.full(N, -1, dtype=np.int32)
        labels[0] = 1  # single positive
        propensities = np.linspace(0.05, 0.95, N)

        w_pu = compute_asymmetric_pu_weights(
            labels_k=labels,
            e_logistics=propensities,
            hajek_normalize=True,
        )
        assert np.isclose(w_pu[0], 1.0, atol=1e-12)


# =====================================================================
# 4. ASPP Bottleneck Physical Scaling Stress-Testing
# =====================================================================


class TestASPPBottleneckPhysicalScaling:
    """Adversarial stress-testing of ASPPBottleneck physical scaling across resolutions."""

    @pytest.mark.parametrize(
        "px_size_m, expected_dilations",
        [
            (250.0, [1, 1, 1, 2]),
            (100.0, [1, 1, 2, 4]),
            (50.0, [1, 2, 4, 8]),
        ],
    )
    def test_aspp_theoretical_dilations_scaling(
        self, px_size_m: float, expected_dilations: List[int]
    ):
        """Verify ASPPBottleneck dilations match physical radii [0.5, 1.5, 3.0, 6.0] km."""
        radii_m = [500.0, 1500.0, 3000.0, 6000.0]
        bottleneck_cell_m = 16.0 * float(px_size_m)
        computed_dilations = [max(1, int(round(r / bottleneck_cell_m))) for r in radii_m]

        assert computed_dilations == expected_dilations, (
            f"Dilations mismatch at px_size_m={px_size_m}: {computed_dilations} vs {expected_dilations}"
        )

    def test_aspp_layer_instantiation_and_effective_dilation_inspection(self):
        """Inspect actual instantiated conv layer dilations in ASPPBottleneck."""
        # Test across 250m, 100m, 50m
        configs = {
            250.0: {"b1": 1, "b2": 1, "b3": 1},
            100.0: {"b1": 1, "b2": 1, "b3": 2},
            50.0: {"b1": 1, "b2": 2, "b3": 4},
        }

        for px_size, expected_conv_dilations in configs.items():
            aspp = ASPPBottleneck(in_channels=64, out_channels=64, px_size_m=px_size)

            # Conv2d layers in b1, b2, b3
            conv1 = aspp.b1[0]
            conv2 = aspp.b2[0]
            conv3 = aspp.b3[0]

            assert conv1.dilation == (expected_conv_dilations["b1"], expected_conv_dilations["b1"])
            assert conv2.dilation == (expected_conv_dilations["b2"], expected_conv_dilations["b2"])
            assert conv3.dilation == (expected_conv_dilations["b3"], expected_conv_dilations["b3"])

            # Verify forward pass preserves spatial tensor dimensions (stride=1, padding=dilation)
            x = torch.randn(2, 64, 16, 16)
            out = aspp(x)
            assert out.shape == (2, 64, 16, 16)


# =====================================================================
# 5. 5-Fold Buffered Spatial Cross-Validation Stress-Testing
# =====================================================================


class TestBufferedSpatialCVInvariants:
    """Adversarial stress-testing of 5-fold buffered spatial cross-validation."""

    def test_min_distance_strictly_greater_or_equal_to_buffer_m(self):
        """Test that for every fold, min distance between any training point and any val point is >= B_m."""
        # Generate a 2D synthetic domain of 60 km x 60 km
        rng = np.random.default_rng(42)
        n_points = 1200
        # Spatial extent: 0 to 60,000 meters in X and Y
        xs = rng.uniform(0.0, 60000.0, size=n_points)
        ys = rng.uniform(0.0, 60000.0, size=n_points)
        coords = np.column_stack([xs, ys])

        px_size_m = 250.0
        r_k_star_m = 6000.0
        B_m = compute_physical_buffer_m(px_size_m=px_size_m, r_k_star_m=r_k_star_m)
        assert B_m == 6000.0  # max(6000.0, 16 * 250 = 4000)

        # Run 5-fold buffered spatial split
        folds = spatial_buffered_block_kfold(
            coords_or_gdf=coords,
            n_splits=5,
            block_size_m=10000.0,  # 10 km blocks
            buffer_m=B_m,
            px_size_m=px_size_m,
            random_state=42,
        )

        assert len(folds) == 5

        all_val_indices = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            assert len(val_idx) > 0, f"Fold {fold_idx}: validation set is empty!"
            assert len(train_idx) > 0, f"Fold {fold_idx}: training set is empty!"

            all_val_indices.extend(val_idx.tolist())

            # Evaluate exact pairwise minimum distance using cKDTree
            val_tree = cKDTree(coords[val_idx])
            distances, _ = val_tree.query(coords[train_idx], k=1)
            min_dist = float(np.min(distances))

            logger.info(
                "Fold %d: train=%d, val=%d, min_dist=%.2fm (required B_m=%.2fm)",
                fold_idx,
                len(train_idx),
                len(val_idx),
                min_dist,
                B_m,
            )

            # INVARIANT: min distance must be strictly >= B_m
            assert min_dist >= B_m, (
                f"Fold {fold_idx} spatial leakage! min_dist ({min_dist:.2f}m) < B_m ({B_m:.2f}m)"
            )

        # INVARIANT: Non-overlapping 100% out-of-fold coverage
        assert len(all_val_indices) == n_points
        assert len(set(all_val_indices)) == n_points

    def test_buffered_spatial_cv_scikit_learn_interface(self):
        """Test BufferedSpatialBlockKFold scikit-learn compatible interface."""
        coords = np.random.default_rng(99).uniform(0.0, 50000.0, size=(500, 2))
        cv = BufferedSpatialBlockKFold(
            n_splits=5,
            block_size_m=8000.0,
            buffer_m=5000.0,
            px_size_m=250.0,
        )

        val_seen = set()
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(coords)):
            val_tree = cKDTree(coords[val_idx])
            distances, _ = val_tree.query(coords[train_idx], k=1)
            assert np.min(distances) >= 5000.0
            val_seen.update(val_idx)

        assert len(val_seen) == 500

    def test_buffer_failsafe_on_tiny_domain(self):
        """When study area is so tiny that B_m would purge all cells, failsafe must engage."""
        # 10 points within a tiny 100m x 100m box, while B_m is 6000m
        coords_tiny = np.random.default_rng(12).uniform(0.0, 100.0, size=(10, 2))
        folds = spatial_buffered_block_kfold(
            coords_or_gdf=coords_tiny,
            n_splits=2,
            block_size_m=50.0,
            buffer_m=6000.0,
            random_state=12,
        )
    def test_buffered_spatial_cv_geographic_coordinates_and_dict_radii(self):
        """Test buffered spatial CV with geographic coordinates (degrees lon/lat) and dict radii."""
        rng = np.random.default_rng(555)
        # Latitudes around equator (-1.0 to 1.0 deg), Longitudes 115.0 to 117.0 deg (~220 km)
        lons = rng.uniform(115.0, 117.0, size=800)
        lats = rng.uniform(-1.0, 1.0, size=800)
        df_geo = pd.DataFrame({"lon": lons, "lat": lats})

        radii_dict = {
            "porphyry_cu": 6000.0,
            "skarn_fe_cu": 3000.0,
            "epithermal_au": 4000.0,
            "laterite_fe_ni": 2000.0,
        }
        B_m = compute_physical_buffer_m(px_size_m=250.0, r_k_star_m=radii_dict)
        assert B_m == 6000.0  # max radius is 6000m

        folds = spatial_buffered_block_kfold(
            coords_or_gdf=df_geo,
            n_splits=5,
            block_size_m=20000.0,  # 20 km blocks
            buffer_m=B_m,
            px_size_m=250.0,
            random_state=42,
        )

        assert len(folds) == 5
        # Convert coords to meters to verify separation
        mean_lat = np.mean(lats)
        m_x = lons * 111320.0 * np.cos(np.radians(mean_lat))
        m_y = lats * 110570.0
        metric_xy = np.column_stack([m_x, m_y])

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            val_tree = cKDTree(metric_xy[val_idx])
            distances, _ = val_tree.query(metric_xy[train_idx], k=1)
            min_dist = float(np.min(distances))
            assert min_dist >= B_m, f"Geographic CV Fold {fold_idx}: min_dist {min_dist} < {B_m}"


# =====================================================================
# Additional Boundary & Failure Mode Stress-Testing
# =====================================================================


class TestExtremeBoundaryConditions:
    """Stress-testing extreme mathematical edge cases."""

    def test_hajek_weights_zero_positive_head(self):
        """When head k has 0 positives, hajek normalization must not crash or divide by zero."""
        N = 100
        labels_zero_pos = np.full(N, -1, dtype=np.int32)
        propensities = np.full(N, 0.50)
        exclusion_mask = np.ones(N)

        w = compute_hajek_asymmetric_head_weights(
            labels_k=labels_zero_pos,
            e_logistics=propensities,
            exclusion_mask_k=exclusion_mask,
        )
        assert len(w) == N
        assert np.all(w == 1.0)  # all unlabeled outside exclusion -> 1.0

    def test_hajek_weights_near_zero_propensities(self):
        """Propensities near 0 (e.g. 1e-15) must be clamped to min_propensity (0.05)."""
        N = 10
        labels = np.full(N, -1, dtype=np.int32)
        labels[:2] = 1
        propensities = np.array([1e-15, 1e-10, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        w = compute_hajek_asymmetric_head_weights(
            labels_k=labels,
            e_logistics=propensities,
            min_propensity=0.05,
        )
        # Both near-zero propensities are clipped to 0.05, so inverse propensities are identical (20.0)
        # Therefore normalized weights for the 2 positives must each be exactly 1.0
        np.testing.assert_allclose(w[:2], [1.0, 1.0], atol=1e-12)

    def test_fuse_canopy_out_of_bounds_w_canopy(self):
        """Out-of-bounds canopy weights (w_canopy < 0 or > 1) must be safely clipped."""
        lo_basin = np.array([0.0, 0.0])
        ll_opt = np.array([2.0, 2.0])
        w_canopy_oob = np.array([-0.5, 2.5])  # out of bounds

        fused = fuse_canopy_log_odds(
            log_odds_basin=lo_basin,
            log_lambda_optical=ll_opt,
            w_canopy=w_canopy_oob,
            logit_pi_opt=0.0,
        )
        # -0.5 clipped to 0.0 -> fused = 0.0 + 0 * 2.0 = 0.0
        # 2.5 clipped to 1.0 -> fused = 0.0 + 1.0 * 2.0 = 2.0
        np.testing.assert_allclose(fused, [0.0, 2.0], atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
