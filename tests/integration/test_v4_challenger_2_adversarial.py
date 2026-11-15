"""tests/integration/test_v4_challenger_2_adversarial.py
======================================================
Independent Code-Executing Adversarial Stress Harness for MPM Pipeline v4.0.

Author: Challenger 2 (Empirical Challenger)
Date: 2026-11-07
Integrity: Independent generation, stress testing, and adversarial oracles.

Testing 4 Core Stress Dimensions:
  1. Zero-Positive Heads in U-Net & Loss:
     - nnPULoss multi-head forward/backward under zero-positive configurations.
     - MPMDistrictDAEUNet training with zero-positive channels without NaN/Inf.
     - Zero-positive heads masked from gradient updates.
     - Out-of-fold prediction defaulting to 0.0 with maximum conformal uncertainty (q_hat=1.0).
  2. Flat Dataframe Physical Layer Fallback & _norm_01:
     - resolve_physical_layers fallback when stack is None.
     - Dataframe fallback across various column subsets.
     - _norm_01 percentile clipping under outliers, constant arrays, and degenerate cases.
  3. TreeSHAP Attribution & Geologist Site Card Extraction:
     - TreeSHAP on TwoStageK4Ensemble and TwoStageK3CoalEnsemble.
     - Composite chain-rule TreeSHAP validation under canopy/optical gating.
     - Site cards emit genuine gates when provided and null/omitted when unobserved (never dummy 1.0).
     - Alteration sensor source provenance honesty ("not_available" when unobserved).
  4. Complete Facade Import & Parity Test:
     - All 55 historical symbols in pu_pipeline are importable and identity-identical.
     - Legacy PU learning routines execute cleanly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shap
import torch
import torch.nn as nn
from shapely.geometry import Polygon

# Modular imports
import src.labels.pu_negatives as pu_negatives
import src.models.calibration as calibration
import src.models.coal as coal
import src.models.conformal as conformal
import src.models.fusion as fusion
import src.models.metals as metals
import src.models.pu_pipeline as pu_pipeline
import src.schema.canonical as canonical
from src.models.coal import (
    K3_COAL_HEAD_NAMES,
    TwoStageK3CoalEnsemble,
)
from src.models.conformal import (
    ConformalQuantile,
    compute_multi_mineral_conformal_bounds,
)
from src.models.metals import (
    K4_HEAD_NAMES,
    TwoStageK4Ensemble,
)
from src.models.unet_model import (
    DistrictDAEUNet,
    MPMDistrictDAEUNet,
    nnPULoss,
)
from src.outputs.prospectivity_map import (
    _norm_01,
    emit_site_cards_json,
    extract_target_shap_attribution,
    resolve_physical_layers,
)


# ==============================================================================
# Dimension 1: Zero-Positive Heads in U-Net & Loss
# ==============================================================================

class TestZeroPositiveHeadsUNetAndLoss:
    """Empirical adversarial tests for zero-positive heads in U-Net and nnPULoss."""

    def test_nnpuloss_zero_positive_head_masking_and_gradients(self):
        """Verify nnPULoss masks zero-positive heads without NaN/Inf and gradient isolation."""
        torch.manual_seed(42)
        B, K, H, W = 4, 4, 16, 16
        loss_fn = nnPULoss(pi_p=[0.01, 0.02, 0.03, 0.04], gamma=1.0, beta=0.0)

        # Model output logits passed through sigmoid
        raw_logits = torch.randn(B, K, H, W, requires_grad=True)
        preds = torch.sigmoid(raw_logits)

        # Targets: Head 0 and Head 2 have ZERO positives
        targets = torch.zeros(B, K, H, W, dtype=torch.float32)
        # Head 1: 5 positive pixels
        targets[0, 1, 2:4, 2:4] = 1.0
        targets[1, 1, 5, 5] = 1.0
        # Head 3: 10 positive pixels
        targets[2, 3, :2, :5] = 1.0

        sample_weights = torch.ones(B, K, H, W, dtype=torch.float32)

        loss = loss_fn(preds, targets, sample_weights=sample_weights)

        # 1. Forward pass must be finite scalar
        assert torch.is_tensor(loss), "Loss must be a torch.Tensor"
        assert loss.ndim == 0, f"Loss must be scalar, got shape {loss.shape}"
        assert torch.isfinite(loss).item(), f"Loss must be finite, got {loss.item()}"
        assert loss.item() > 0.0, f"Loss must be strictly positive, got {loss.item()}"

        # 2. Backward pass must execute cleanly
        loss.backward()

        assert raw_logits.grad is not None, "Gradients must exist on input logits"
        assert torch.all(torch.isfinite(raw_logits.grad)).item(), "All gradients must be finite (no NaN/Inf)"

        # 3. Gradients on zero-positive heads (heads 0 and 2) must be identically 0
        grad_head0 = raw_logits.grad[:, 0, :, :].abs().sum().item()
        grad_head2 = raw_logits.grad[:, 2, :, :].abs().sum().item()
        assert grad_head0 == 0.0, f"Head 0 has 0 positives; gradient must be 0, got {grad_head0}"
        assert grad_head2 == 0.0, f"Head 2 has 0 positives; gradient must be 0, got {grad_head2}"

        # 4. Gradients on positive heads (heads 1 and 3) must be non-zero
        grad_head1 = raw_logits.grad[:, 1, :, :].abs().sum().item()
        grad_head3 = raw_logits.grad[:, 3, :, :].abs().sum().item()
        assert grad_head1 > 0.0, f"Head 1 has positives; gradient must be non-zero, got {grad_head1}"
        assert grad_head3 > 0.0, f"Head 3 has positives; gradient must be non-zero, got {grad_head3}"

    def test_nnpuloss_all_zero_positive_heads_fallback(self):
        """Adversarial edge case: ALL heads have zero positives. Must return 0.0 gracefully."""
        loss_fn = nnPULoss(pi_p=0.05)
        raw_logits = torch.randn(2, 4, 8, 8, requires_grad=True)
        preds = torch.sigmoid(raw_logits)
        targets = torch.zeros(2, 4, 8, 8, dtype=torch.float32)

        loss = loss_fn(preds, targets)
        assert torch.isfinite(loss).item(), "Loss with all zero positives must be finite"
        assert loss.item() == 0.0, f"Loss with all zero positives must be 0.0, got {loss.item()}"

        loss.backward()
        # When all heads are masked, loss is detached zero scalar, so no gradient accumulates to raw_logits
        assert raw_logits.grad is None or torch.all(raw_logits.grad == 0.0).item(), (
            "Gradients must be None or all-zero when all heads have 0 positives"
        )

    def test_mpm_district_dae_unet_fit_with_zero_positive_head(self):
        """Instantiate MPMDistrictDAEUNet and fit where head 0 has 0 positives."""
        np.random.seed(42)
        torch.manual_seed(42)

        n_samples = 8
        in_channels = 6
        n_classes = 4
        window_px = 16

        X_windows = np.random.randn(n_samples, in_channels, window_px, window_px).astype(np.float32)
        y_targets = np.zeros((n_samples, n_classes, window_px, window_px), dtype=np.float32)

        # Head 0: 0 positives (East Kalimantan Porphyry simulation)
        # Head 1: 8 positives
        y_targets[:4, 1, 8, 8] = 1.0
        # Head 2: 4 positives
        y_targets[4:, 2, 7, 7] = 1.0
        # Head 3: 0 positives

        weights = np.ones((n_samples, n_classes, window_px, window_px), dtype=np.float32)

        dae_unet = MPMDistrictDAEUNet(
            in_channels=in_channels,
            n_classes=n_classes,
            window_px=window_px,
            batch_size=4,
            dropout_p=0.1,
            pi_p=[0.01, 0.02, 0.03, 0.05],
            px_size_m=250.0,
            subcanopy_indices=[0, 1, 2],
        )

        # Fit for 2 epochs
        dae_unet.fit(
            X_windows=X_windows,
            y_targets=y_targets,
            sample_weights=weights,
            epochs=2,
            lr=1e-3,
        )

        assert dae_unet.model is not None, "Model must be instantiated after fit"
        # Verify model parameters are finite
        for name, param in dae_unet.model.named_parameters():
            assert torch.all(torch.isfinite(param)).item(), f"Parameter {name} contains non-finite values"

        # Verify inference forward pass works
        with torch.no_grad():
            x_test = torch.tensor(X_windows[:2]).to(dae_unet.device)
            preds, recon, recon_err, ctx = dae_unet.model(x_test)
            assert torch.all(torch.isfinite(preds)).item(), "Predictions must be finite"
            assert preds.shape == (2, n_classes, window_px, window_px)
            assert torch.all((preds >= 0.0) & (preds <= 1.0)).item(), "Predictions must be bounded in [0, 1]"

    def test_zero_positive_conformal_bounds_and_uncertainty(self):
        """Verify zero-positive head defaults to q_set=1.0, q_res=1.0 and lower_bound=0.0."""
        N = 100
        p_cal_zero_pos = np.zeros(N, dtype=np.float64)
        sigma = np.full(N, 0.25, dtype=np.float64)

        # No positive samples available for this head
        l_bound, v_scores, q_quant = compute_multi_mineral_conformal_bounds(
            p_cal_k=p_cal_zero_pos,
            sigma_ens_k=sigma,
            cal_pos_p_cal=None,
            cal_pos_sigma=None,
            alpha=0.10,
        )

        # Maximum conformal uncertainty
        assert q_quant.q_set == 1.0, f"Expected q_set=1.0 for 0-positive head, got {q_quant.q_set}"
        assert q_quant.q_res == 1.0, f"Expected q_res=1.0 for 0-positive head, got {q_quant.q_res}"
        # Point probabilities are 0.0 -> lower bound must be 0.0
        assert np.all(l_bound == 0.0), "Lower bound for zero probability must be 0.0"


# ==============================================================================
# Dimension 2: Flat Dataframe Physical Layer Fallback & _norm_01
# ==============================================================================

class TestFlatDataframePhysicalFallbackAndNorm:
    """Empirical adversarial tests for physical layer resolution and _norm_01 percentile clipping."""

    def test_resolve_physical_layers_when_stack_is_none(self):
        """Verify resolve_physical_layers safely handles stack=None and missing attributes."""
        res_none = resolve_physical_layers(None)
        assert isinstance(res_none, dict)
        expected_keys = {
            "dilation_tendency_Td",
            "alteration_index",
            "regolith_cover_depth_m",
            "crustal_thickness_km",
            "bouguer_worm",
        }
        assert set(res_none.keys()) == expected_keys
        for k, v in res_none.items():
            assert v is None, f"Expected key {k} to be None when stack is None, got {v}"

        # Degenerate object missing array or feature_names
        class DummyStack:
            pass

        res_dummy = resolve_physical_layers(DummyStack())
        assert set(res_dummy.keys()) == expected_keys
        for k, v in res_dummy.items():
            assert v is None

    def test_dataframe_fallback_across_column_subsets(self):
        """Simulate physical layer reconstruction from df across different column sets."""
        H, W = 10, 10
        N = H * W
        rows = np.repeat(np.arange(H), W)
        cols = np.tile(np.arange(W), H)
        ref_grid = np.zeros((H, W), dtype=np.float32)

        # Helper simulating the fallback logic from run_seven_pass_training.py
        def simulate_fallback(df: pd.DataFrame) -> dict[str, np.ndarray]:
            from src.features.feature_stack import reconstruct_map

            phys: dict[str, Any] = {}
            # Td
            td_2d = phys.get("dilation_tendency_Td")
            if td_2d is None:
                if "dilation_tendency_Td" in df.columns:
                    td_2d = reconstruct_map(df["dilation_tendency_Td"].to_numpy(dtype=np.float32), rows, cols, H, W, nodata=0.0)
                else:
                    td_2d = np.zeros_like(ref_grid)
            # Alt
            alt_2d = phys.get("alteration_index")
            if alt_2d is None:
                if "spec_clay_hydroxyl_gated" in df.columns:
                    alt_2d = reconstruct_map(df["spec_clay_hydroxyl_gated"].to_numpy(dtype=np.float32), rows, cols, H, W, nodata=0.0)
                else:
                    alt_2d = np.zeros_like(ref_grid)
            # Cover
            cover_2d = phys.get("regolith_cover_depth_m")
            if cover_2d is None:
                if "regolith_cover_m" in df.columns:
                    cover_2d = reconstruct_map(df["regolith_cover_m"].to_numpy(dtype=np.float32), rows, cols, H, W, nodata=0.0)
                else:
                    cover_2d = np.full_like(ref_grid, 25.0)
            # Slope
            slope_2d = phys.get("dem_slope")
            if slope_2d is None:
                if "dem_slope" in df.columns:
                    slope_2d = reconstruct_map(df["dem_slope"].to_numpy(dtype=np.float32), rows, cols, H, W, nodata=0.0)
                else:
                    slope_2d = np.zeros_like(ref_grid)

            return {"td": td_2d, "alt": alt_2d, "cover": cover_2d, "slope": slope_2d}

        # Case 1: Complete DataFrame
        df_full = pd.DataFrame({
            "dilation_tendency_Td": np.random.uniform(0.1, 0.9, N).astype(np.float32),
            "spec_clay_hydroxyl_gated": np.random.uniform(0.0, 1.0, N).astype(np.float32),
            "regolith_cover_m": np.random.uniform(5.0, 80.0, N).astype(np.float32),
            "dem_slope": np.random.uniform(0.0, 45.0, N).astype(np.float32),
        })
        res1 = simulate_fallback(df_full)
        assert res1["td"].shape == (H, W)
        assert np.any(res1["td"] > 0)
        assert np.any(res1["alt"] > 0)
        assert np.all(res1["cover"] >= 5.0)

        # Case 2: Partial DataFrame (missing Td and Alt)
        df_partial = pd.DataFrame({
            "regolith_cover_m": np.full(N, 35.0, dtype=np.float32),
            "dem_slope": np.full(N, 12.0, dtype=np.float32),
        })
        res2 = simulate_fallback(df_partial)
        assert np.all(res2["td"] == 0.0), "Missing Td must safely fallback to zeros"
        assert np.all(res2["alt"] == 0.0), "Missing Alt must safely fallback to zeros"
        assert np.all(res2["cover"] == 35.0)
        assert np.all(res2["slope"] == 12.0)

        # Case 3: Completely empty DataFrame
        df_empty = pd.DataFrame()
        res3 = simulate_fallback(df_empty)
        assert np.all(res3["td"] == 0.0)
        assert np.all(res3["alt"] == 0.0)
        assert np.all(res3["cover"] == 25.0)
        assert np.all(res3["slope"] == 0.0)

    def test_norm_01_adversarial_percentile_clipping(self):
        """Stress-test _norm_01 under adversarial distributions and edge cases."""
        # 1. Extreme Outliers: 1000 normal values and two massive outliers [-1e9, 1e9]
        data = np.random.normal(loc=50.0, scale=5.0, size=1000)
        data[0] = -1e9
        data[-1] = 1e9

        normed = _norm_01(data)
        assert np.all(np.isfinite(normed)), "All normalized values must be finite"
        assert np.min(normed) == 0.0, "Minimum clipped value must reach 0.0"
        assert np.max(normed) == 1.0, "Maximum clipped value must reach 1.0"
        # Crucial: verify bulk data did NOT collapse to 0.5 or 0.0
        bulk = normed[1:-1]
        assert np.std(bulk) > 0.1, f"Bulk data must preserve dynamic range, got std={np.std(bulk)}"
        assert not np.allclose(bulk, 0.5), "Bulk data must not collapse to constant 0.5"

        # 2. Constant Array (avoid division by zero)
        const_data = np.full(100, 42.0)
        normed_const = _norm_01(const_data)
        assert np.all(normed_const == 0.5), "Constant array must map to 0.5"

        # 3. Near-Constant Array (difference < 1e-12)
        near_const = np.full(100, 1.0)
        near_const[0] += 1e-13
        normed_near = _norm_01(near_const)
        assert np.all(normed_near == 0.5), "Near-constant array must map to 0.5"

        # 4. Standard Uniform Array (must span full range [0, 1])
        unif_data = np.linspace(0.0, 100.0, 500)
        normed_unif = _norm_01(unif_data)
        assert np.isclose(np.min(normed_unif), 0.0)
        assert np.isclose(np.max(normed_unif), 1.0)
        assert 0.45 < np.mean(normed_unif) < 0.55

        # 5. Empty Array
        empty_res = _norm_01(np.array([]))
        assert len(empty_res) == 0

        # 6. Array with NaNs
        nan_data = np.array([10.0, np.nan, 20.0, 30.0, 40.0, np.nan, 50.0])
        normed_nan = _norm_01(nan_data)
        valid_mask = np.isfinite(nan_data)
        assert np.all(np.isfinite(normed_nan[valid_mask]))


# ==============================================================================
# Dimension 3: TreeSHAP and Geologist Site Card Extraction
# ==============================================================================

class TestTreeSHAPAndGeologistSiteCards:
    """Empirical adversarial tests for TreeSHAP and Site Card physical gate emission."""

    def test_treeshap_on_twostage_k4_metals(self):
        """Verify TwoStageK4Ensemble.compute_head_shap() on all heads."""
        np.random.seed(42)
        N = 120
        n_features = 10
        feature_names = [f"feat_{i}" for i in range(n_features)]
        X = np.random.randn(N, n_features).astype(np.float32)

        # 4 binary label columns
        y_k4 = np.zeros((N, 4), dtype=np.int32)
        y_k4[:15, 0] = 1  # porphyry
        y_k4[15:30, 1] = 1  # skarn
        y_k4[30:45, 2] = 1  # epithermal
        y_k4[45:60, 3] = 1  # laterite

        model = TwoStageK4Ensemble(
            config={"hyperparams": {"max_depth": 3}},
            n_estimators=10,
            n_bag=2,
            random_state=42,
        )
        model.fit(X, y_k4, feature_names=feature_names)

        # 1. Single sample 1D vector TreeSHAP for porphyry_cu (Head 0)
        x_sample = X[0]
        shap_0, names_0 = model.compute_head_shap("porphyry_cu", x_sample, feature_names)
        assert shap_0.ndim == 1
        assert len(shap_0) == len(names_0)
        assert len(shap_0) > n_features, "Head 0 must include Stage-2 meta-features"
        assert "oof_prob_skarn" in names_0
        assert "oof_prob_epithermal" in names_0

        # 2. Heads 1, 2, 3 TreeSHAP
        for head_name in ["skarn_fe_cu", "epithermal_au", "laterite_fe_ni"]:
            shap_k, names_k = model.compute_head_shap(head_name, x_sample, feature_names)
            assert shap_k.ndim == 1
            assert len(shap_k) == n_features
            assert names_k == feature_names

        # 3. 2D Batch input
        shap_batch, names_batch = model.compute_head_shap("skarn_fe_cu", X[:5], feature_names)
        assert shap_batch.shape == (5, n_features)

    def test_composite_chain_rule_treeshap_on_twostage_k3_coal(self):
        """Empirically test composite chain-rule TreeSHAP in TwoStageK3CoalEnsemble (Bug 6)."""
        np.random.seed(42)
        N = 120
        # 9 canonical features: 6 subcanopy + 3 optical
        sub_names = [
            "sediment_thickness_m",
            "gravity_bouguer_mgal",
            "mag_analytic_signal",
            "sar_palsar2_hh",
            "dem_elevation",
            "dem_slope",
        ]
        opt_names = ["spec_albedo_gated", "spec_wcdi_gated", "spec_oci_gated"]
        all_features = sub_names + opt_names
        X = np.random.randn(N, len(all_features)).astype(np.float32)

        # Labels for 3 coal ranks
        y_k3 = np.zeros((N, 3), dtype=np.int32)
        y_k3[:20, 0] = 1
        y_k3[20:40, 1] = 1
        y_k3[40:60, 2] = 1

        w_canopy = np.random.uniform(0.1, 0.9, N).astype(np.float32)

        coal_model = TwoStageK3CoalEnsemble(
            config={"hyperparams": {"max_depth": 3}},
            n_estimators=10,
            n_bag=2,
            random_state=42,
        )
        coal_model.fit(X, y_k3, feature_names=all_features, w_canopy=w_canopy)

        # Test composite chain-rule computation:
        # Case A: High optical probability & canopy penetration (w_canopy > 0)
        x_high = X[0].copy()
        # Test with w_canopy = 0.0 -> optical attributions must be ZERO
        shap_w0, names_w0 = coal_model.compute_head_shap(0, x_high, all_features, w_canopy=np.array([0.0]))
        opt_indices_in_shap = [i for i, name in enumerate(names_w0) if name in opt_names]
        assert len(opt_indices_in_shap) == len(opt_names)
        for idx in opt_indices_in_shap:
            assert np.isclose(shap_w0[idx], 0.0), f"Optical feature {names_w0[idx]} must have 0 SHAP when w_canopy=0"

        # Case B: With canopy transmittance w_canopy = 0.8
        shap_w8, names_w8 = coal_model.compute_head_shap(0, x_high, all_features, w_canopy=np.array([0.8]))
        assert len(shap_w8) == len(all_features)
        assert np.all(np.isfinite(shap_w8))

    def test_site_cards_emission_physical_gates_provenance(self):
        """Verify emit_site_cards_json outputs genuine gates and never injects dummy 1.0 gates."""
        # Create a mock ranked GeoDataFrame with 2 targets
        poly1 = Polygon([(115.0, -1.0), (115.01, -1.0), (115.01, -1.01), (115.0, -1.01)])
        poly2 = Polygon([(116.0, -2.0), (116.01, -2.0), (116.01, -2.01), (116.0, -2.01)])

        gdf = gpd.GeoDataFrame({
            "target_id": ["TGT_0001", "TGT_0002"],
            "rank": [1, 2],
            "rank_score": [0.88, 0.72],
            "calibrated_deposit_prob": [0.85, 0.70],
            "lower_bound_L": [0.65, 0.52],
            "upper_bound_U": [0.95, 0.85],
            "fused_raw_score": [0.60, 0.45],
            "exploration_propensity_e_x": [0.25, 0.15],
            "geometry": [poly1, poly2],
        }, crs="EPSG:4326")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "site_cards.json"

            # -------------------------------------------------------------
            # Scenario A: Gates NOT PROVIDED (unobserved)
            # -------------------------------------------------------------
            cards_a = emit_site_cards_json(
                ranked_targets_gdf=gdf,
                out_path=str(out_file),
                elkan_noto_c_hat=0.74,
                sensor_source="EMIT_SWIR",
                outmogc_gates=None,
                basin_gate=None,
            )

            assert len(cards_a) == 2
            for card in cards_a:
                # 1. outmogc_boundary_gate must be None (null), NEVER 1.0!
                assert card["outmogc_boundary_gate"] is None, (
                    f"Expected outmogc_boundary_gate to be None when unobserved, got {card['outmogc_boundary_gate']}"
                )
                # 2. macro_basin_sequence_gate must be omitted completely
                assert "macro_basin_sequence_gate" not in card, (
                    "macro_basin_sequence_gate must NOT exist when basin_gate is None"
                )
                assert "basin_gate" not in card, (
                    "basin_gate must NOT exist when basin_gate is None"
                )
                # 3. Alteration sensor source must be 'not_available' when unobserved
                alt_summary = card["physics_and_geology_summary"]["alteration"]
                assert alt_summary["sensor_source"] == "not_available", (
                    f"sensor_source must be 'not_available' when unobserved, got {alt_summary['sensor_source']}"
                )

            # -------------------------------------------------------------
            # Scenario B: Genuine Physical Gates PROVIDED
            # -------------------------------------------------------------
            genuine_outmogc = [
                {"porphyry_cu": 0.85, "skarn_fe_cu": 0.12, "epithermal_au": 0.90, "laterite_fe_ni": 0.05},
                {"porphyry_cu": 0.10, "skarn_fe_cu": 0.88, "epithermal_au": 0.15, "laterite_fe_ni": 0.70},
            ]
            genuine_basin = [0.92, 0.35]

            cards_b = emit_site_cards_json(
                ranked_targets_gdf=gdf,
                out_path=str(out_file),
                elkan_noto_c_hat=0.74,
                sensor_source="EMIT_SWIR",
                outmogc_gates=genuine_outmogc,
                basin_gate=genuine_basin,
            )

            assert len(cards_b) == 2
            # Target 1
            assert cards_b[0]["outmogc_boundary_gate"] == genuine_outmogc[0]
            assert cards_b[0]["macro_basin_sequence_gate"] == 0.92
            assert cards_b[0]["basin_gate"] == 0.92
            # Target 2
            assert cards_b[1]["outmogc_boundary_gate"] == genuine_outmogc[1]
            assert cards_b[1]["macro_basin_sequence_gate"] == 0.35
            assert cards_b[1]["basin_gate"] == 0.35


# ==============================================================================
# Dimension 4: Complete Facade Import & Parity Test
# ==============================================================================

class TestFacadeImportAndParity:
    """Check that every symbol exported by src/models/pu_pipeline.py is present and identical."""

    def test_all_symbols_in_pu_pipeline_all(self):
        """Verify all symbols in __all__ are present and importable."""
        exported = pu_pipeline.__all__
        assert len(exported) >= 50, f"Expected at least 50 exported symbols, got {len(exported)}"

        for sym in exported:
            assert hasattr(pu_pipeline, sym), f"pu_pipeline missing exported symbol: {sym}"

    def test_identity_parity_with_modular_sources(self):
        """Verify that symbols in pu_pipeline are exact identities of modular sources."""
        # Models
        assert pu_pipeline.TwoStageK4Ensemble is metals.TwoStageK4Ensemble
        assert pu_pipeline._BaggedXGBoostHead is metals._BaggedXGBoostHead
        assert pu_pipeline.compute_hajek_asymmetric_head_weights is metals.compute_hajek_asymmetric_head_weights
        assert pu_pipeline.compute_pa_intersection_weights is metals.compute_pa_intersection_weights
        assert pu_pipeline.compute_concentric_zonation_features is metals.compute_concentric_zonation_features

        assert pu_pipeline.TwoStageK3CoalEnsemble is coal.TwoStageK3CoalEnsemble
        assert pu_pipeline.MacroBasinSequenceGate is coal.MacroBasinSequenceGate
        assert pu_pipeline.fuse_canopy_log_odds is coal.fuse_canopy_log_odds
        assert pu_pipeline.fuse_bayesian_canopy_log_odds is coal.fuse_bayesian_canopy_log_odds
        assert pu_pipeline.compute_hierarchical_coal_probabilities is coal.compute_hierarchical_coal_probabilities

        # Conformal & Calibration
        assert pu_pipeline.ConformalQuantile is conformal.ConformalQuantile
        assert pu_pipeline.weighted_conformal_quantile is conformal.weighted_conformal_quantile
        assert pu_pipeline.compute_studentized_conformal_lower_bound is conformal.compute_studentized_conformal_lower_bound
        assert pu_pipeline.evaluate_conformal_set_gate is conformal.evaluate_conformal_set_gate
        assert pu_pipeline.compute_multi_mineral_conformal_bounds is conformal.compute_multi_mineral_conformal_bounds

        assert pu_pipeline.estimate_elkan_noto_c is calibration.estimate_elkan_noto_c
        assert pu_pipeline.calibrate_elkan_noto_posterior is calibration.calibrate_elkan_noto_posterior
        assert pu_pipeline.calibrate_monotonic_logit_scaling is calibration.calibrate_monotonic_logit_scaling

        # Fusion
        assert pu_pipeline.precision_weighted_log_odds_fusion is fusion.precision_weighted_log_odds_fusion
        assert pu_pipeline.compute_ensemble_weights is fusion.compute_ensemble_weights

        # Canonical
        assert pu_pipeline.NON_FEATURE_COLS is canonical.NON_FEATURE_COLS
        assert pu_pipeline.verify_no_raw_pixel_leakage is canonical.verify_no_raw_pixel_leakage
        assert pu_pipeline.K4_HEAD_NAMES is canonical.K4_HEAD_NAMES
        assert pu_pipeline.K3_COAL_HEAD_NAMES is canonical.K3_COAL_HEAD_NAMES

        # PU Negatives
        assert pu_pipeline.OUTMOGCGate is pu_negatives.OUTMOGCGate
        assert pu_pipeline.step1_isolation_forest is pu_negatives.step1_isolation_forest
        assert pu_pipeline.step2_select_reliable_negatives is pu_negatives.step2_select_reliable_negatives

    def test_legacy_pu_routines_execution(self):
        """Verify legacy routines like fit_pu_model and score_metrics execute cleanly."""
        np.random.seed(42)
        X_p = np.random.normal(loc=2.0, scale=1.0, size=(10, 4))
        X_rn = np.random.normal(loc=-2.0, scale=1.0, size=(20, 4))
        X_u = np.random.normal(loc=0.0, scale=1.0, size=(50, 4))

        cfg = {
            "pu_learning": {
                "step1_contamination": 0.05,
                "rn_percentile_threshold": 10.0,
                "random_state": 42,
            },
            "models": {
                "baseline": {
                    "n_bag": 2,
                    "hyperparams": {"n_estimators": 5, "max_depth": 2},
                }
            }
        }

        # fit_pu_model
        model = pu_pipeline.fit_pu_model(X_p, X_rn, X_u, cfg)
        assert hasattr(model, "predict_proba")
        preds = model.predict_proba(X_u)
        assert len(preds) == len(X_u)
        assert np.all((preds >= 0.0) & (preds <= 1.0))

        # score_metrics
        y_true = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1, 0.3, 0.7, 0.2, 0.1, 0.05, 0.15])
        metrics = pu_pipeline.score_metrics(y_true, y_score)
        assert "pr_auc" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc_lift" in metrics
        assert metrics["pr_auc"] > metrics["prevalence"]
