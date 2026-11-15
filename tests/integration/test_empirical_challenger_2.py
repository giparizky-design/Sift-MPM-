"""tests/integration/test_empirical_challenger_2.py
================================================
Empirical verification and adversarial stress harness for the
Kalimantan Prospectivity Modeling Pipeline Stabilization Gate.

Author: Challenger 2 (Empirical Challenger)
Date: 2026-10-29

5 Empirical Challenges:
  1. XGBoost PU Prior: _BaggedXGBoostHead fit on imbalanced PU data (N_neg/N_pos = 100).
     Assert scale_pos_weight == 1.0, verify raw positive validation scores do not explode to >0.85+.
  2. Optical Warm-Start & Variance Combination: TwoStageK3CoalEnsemble.predict_std(X, w_canopy=w).
     Verify variance matches sqrt((1 - w)^2 * sigma_sub^2 + w^2 * sigma_opt^2).
  3. Class-Decoupled nnPULoss: Multi-head tensor (B, 4, H, W) with heterogeneous priors [0.001, 0.01, 0.05, 0.02].
     Verify backward gradient pass executes cleanly without negative risk contamination across heads.
  4. Full-Raster Streaming Chunked Inference: execute_full_raster_chunked_inference on a test dataset.
     Verify 100% of valid onshore cells have finite predictions (0% NaN on land) and peak memory < 600 MB.
  5. Disk Governance: Mock historical run directories; run prune_historical_runs("TEST", max_keep_runs=2).
     Verify exactly 2 directories remain. Verify check_disk_space raises DiskSpaceError below threshold.
     Assert *.pt files are excluded by archive_existing_run().
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from affine import Affine

from scripts.run_seven_pass_training import (
    execute_full_raster_chunked_inference,
)
from src.models.pu_pipeline import (
    _BaggedXGBoostHead,
    TwoStageK3CoalEnsemble,
    TwoStageK4Ensemble,
)
from src.models.training_run_manager import (
    DiskSpaceError,
    archive_existing_run,
    check_disk_space,
    prune_historical_runs,
    save_runs_catalog,
)
from src.models.unet_model import nnPULoss


# ==============================================================================
# Challenge 1: XGBoost PU Prior Stress Test
# ==============================================================================
def challenge_xgboost_pu_prior() -> dict:
    """Empirically fit _BaggedXGBoostHead on imbalanced positive/unlabeled data (N_neg / N_pos = 100).

    Assert scale_pos_weight == 1.0 and verify raw positive validation scores do not
    artificially explode to >0.85+.
    """
    rng = np.random.default_rng(42)
    N_pos = 50
    N_neg = 5000  # Imbalance ratio = 100:1
    D = 8

    # Training data: realistic feature separation with noise
    X_pos_tr = rng.normal(loc=0.75, scale=1.0, size=(N_pos, D)).astype(np.float32)
    X_neg_tr = rng.normal(loc=0.0, scale=1.0, size=(N_neg, D)).astype(np.float32)
    X_tr = np.vstack([X_pos_tr, X_neg_tr])
    y_tr = np.concatenate([np.ones(N_pos, dtype=np.int32), np.zeros(N_neg, dtype=np.int32)])

    # Validation data
    N_val_pos = 100
    N_val_neg = 5000
    X_pos_va = rng.normal(loc=0.75, scale=1.0, size=(N_val_pos, D)).astype(np.float32)
    X_neg_va = rng.normal(loc=0.0, scale=1.0, size=(N_val_neg, D)).astype(np.float32)

    head = _BaggedXGBoostHead(
        n_estimators=30,
        max_depth=3,
        n_bag=3,
        random_state=42,
    )
    head.fit(X_tr, y_tr)

    # 1. Assert scale_pos_weight == 1.0 on all bagged models
    assert len(head.models_) == 3, f"Expected 3 bagged models, got {len(head.models_)}"
    for idx, clf in enumerate(head.models_):
        assert clf.scale_pos_weight == 1.0, (
            f"Bag {idx} scale_pos_weight is {clf.scale_pos_weight}, expected 1.0"
        )

    # 2. Evaluate raw predicted probabilities on validation positives
    probs_val_pos = head.predict_proba(X_pos_va)
    mean_pos_prob = float(np.mean(probs_val_pos))
    median_pos_prob = float(np.median(probs_val_pos))
    max_pos_prob = float(np.max(probs_val_pos))
    p90_pos_prob = float(np.percentile(probs_val_pos, 90))

    # Empirical assertion: mean raw positive probability must not explode to >0.85
    # (typically sits in 0.05-0.45 reflecting true labeled prevalence)
    assert mean_pos_prob < 0.85, (
        f"Mean positive probability exploded to {mean_pos_prob:.4f} > 0.85! Prior is biased."
    )

    # 3. Adversarial counter-check: What if scale_pos_weight = 100?
    import xgboost as xgb
    clf_biased = xgb.XGBClassifier(
        n_estimators=30,
        max_depth=3,
        scale_pos_weight=100.0,
        random_state=42,
        eval_metric="aucpr",
    )
    clf_biased.fit(X_tr, y_tr)
    biased_pos_probs = clf_biased.predict_proba(X_pos_va)[:, 1]
    biased_mean = float(np.mean(biased_pos_probs))

    results = {
        "scale_pos_weight_asserted": 1.0,
        "n_pos": N_pos,
        "n_neg": N_neg,
        "imbalance_ratio": N_neg / N_pos,
        "mean_raw_pos_prob": mean_pos_prob,
        "median_raw_pos_prob": median_pos_prob,
        "p90_raw_pos_prob": p90_pos_prob,
        "max_raw_pos_prob": max_pos_prob,
        "counter_biased_scale_pos_weight_100_mean": biased_mean,
        "verdict": "PASS",
    }
    return results


# ==============================================================================
# Challenge 2: Optical Warm-Start & Epistemic Variance Combination
# ==============================================================================
def challenge_optical_warmstart_and_variance() -> dict:
    """Empirically test TwoStageK3CoalEnsemble.predict_std(X, w_canopy=w).

    Verify variance matches sqrt((1 - w)^2 * sigma_sub^2 + w^2 * sigma_opt^2).
    """
    rng = np.random.default_rng(202)
    N = 150
    cols = [
        "dist_tanjung_fm_km",
        "dist_balikpapan_fm_km",
        "dist_warukin_fm_km",
        "sediment_thickness_m",
        "gravity_residual_basin_mgal",
        "dem_slope",
        "spec_albedo_gated",
        "spec_wcdi_gated",
        "spec_oci_gated",
        "thermal_anomaly_st_filtered",
    ]
    X_mat = rng.uniform(0.1, 2.0, size=(N, len(cols))).astype(np.float32)
    X = pd.DataFrame(X_mat, columns=cols)

    y_k3 = np.zeros((N, 3), dtype=np.int32)
    y_k3[:10, 0] = 1
    y_k3[10:20, 1] = 1
    y_k3[20:30, 2] = 1

    # Round 1 initial fit
    ens_r1 = TwoStageK3CoalEnsemble(n_estimators=10, n_bag=4, random_state=42)
    ens_r1.fit(X, y_k3, round_idx=1)

    # Round 2 incremental warm-start
    ens_r2 = TwoStageK3CoalEnsemble(n_estimators=10, n_bag=4, random_state=42)
    ens_r2.fit(X, y_k3, prior_ensemble=ens_r1, round_idx=2)

    # 1. Assert optical heads were warm-started from Round 1
    assert hasattr(ens_r2, "optical_heads_"), "ens_r2 missing optical_heads_"
    for c in range(3):
        assert c in ens_r2.optical_heads_, f"Optical head {c} missing"
        assert len(ens_r2.optical_heads_[c].models_) > 0, f"Optical head {c} has no models"

    # 2. Extract subcanopy and optical feature arrays
    X_sub = X.values[:, ens_r2.subcanopy_indices_]
    X_opt = X.values[:, ens_r2.optical_indices_]

    # Baseline subcanopy std
    std_sub = np.column_stack([
        ens_r2.subcanopy_heads_[c].predict_std(X_sub) for c in range(3)
    ])
    # Baseline optical std
    std_opt = np.column_stack([
        ens_r2.optical_heads_[c].predict_std(X_opt) for c in range(3)
    ])

    test_weights = [0.0, 0.15, 0.50, 0.85, 1.0]
    max_discrepancies = []

    for w_val in test_weights:
        w_arr = np.full(N, w_val, dtype=np.float32)
        std_fused = ens_r2.predict_std(X, w_canopy=w_arr)

        # Expected: sqrt((1 - w)^2 * sigma_sub^2 + w^2 * sigma_opt^2)
        w_2d = w_arr[:, np.newaxis]
        var_expected = ((1.0 - w_2d) ** 2) * (std_sub ** 2) + (w_2d ** 2) * (std_opt ** 2)
        std_expected = np.sqrt(np.maximum(0.0, var_expected))

        discrepancy = float(np.max(np.abs(std_fused - std_expected)))
        max_discrepancies.append((w_val, discrepancy))
        assert discrepancy < 1e-5, (
            f"Variance formula failed at w_canopy={w_val}: max diff = {discrepancy:.2e}"
        )

        # Boundary checks
        if w_val == 0.0:
            np.testing.assert_allclose(std_fused, std_sub, rtol=1e-5, atol=1e-5)
        elif w_val == 1.0:
            np.testing.assert_allclose(std_fused, std_opt, rtol=1e-5, atol=1e-5)

    # Also check backward compatibility with w_canopy=None
    std_default = ens_r2.predict_std(X)
    assert std_default.shape == (N, 3)
    assert np.all(std_default >= 0.0)

    results = {
        "optical_heads_warmstarted": True,
        "test_canopy_weights": test_weights,
        "max_discrepancies": max_discrepancies,
        "boundary_w0_matches_subcanopy": True,
        "boundary_w1_matches_optical": True,
        "backward_compatibility_none_supported": True,
        "verdict": "PASS",
    }
    return results


# ==============================================================================
# Challenge 3: Class-Decoupled nnPULoss Multi-Head Stress Test
# ==============================================================================
def challenge_class_decoupled_nnpu_loss() -> dict:
    """Test nnPULoss with multi-head tensor (B, 4, H, W) and heterogeneous class priors.

    Verify backward gradient pass executes cleanly without negative risk contamination across heads.
    """
    torch.manual_seed(42)
    B = 8
    K = 4
    H, W = 16, 16

    heterogeneous_priors = [0.001, 0.01, 0.05, 0.02]
    loss_fn = nnPULoss(pi_p=heterogeneous_priors, gamma=1.0, beta=0.0)

    # 1. Multi-head logits (B, K, H, W)
    logits = torch.randn(B, K, H, W, requires_grad=True)
    preds = torch.sigmoid(logits)

    # Targets: sparse positives
    targets = (torch.rand(B, K, H, W) > 0.95).float()
    # Propensity weights: 2D weights expanded
    weights = torch.rand(B, K, H, W) * 1.5 + 0.5

    # Compute loss and backward pass
    loss = loss_fn(preds, targets, sample_weights=weights)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() >= 0.0, f"Loss is negative: {loss.item()}"

    loss.backward()
    assert logits.grad is not None, "Logits gradient is None"
    assert logits.grad.shape == (B, K, H, W), f"Grad shape mismatch: {logits.grad.shape}"
    assert torch.all(torch.isfinite(logits.grad)), "NaN or Inf detected in gradients"

    # 2. Adversarial Challenge: Negative Risk Decoupling Verification
    # Head 0 has high false positive predictions on unlabeled instances -> forces negative raw risk
    # Head 1 has standard predictions where raw risk is strictly positive.
    # We assert that the non-negative clamping on Head 0 does NOT zero out or mutate Head 1 risk!
    logits_adv = torch.zeros(B, K, H, W, requires_grad=True)
    with torch.no_grad():
        # Force Head 0 to have extreme unlabeled activation (p ~ 0.99) while targets are 0
        logits_adv[:, 0, :, :] = 5.0
        # Force Head 1 to have modest activation
        logits_adv[:, 1, :, :] = -1.0
        # Ensure targets on head 0 are 0 for all pixels except 1
        targets_adv = torch.zeros(B, K, H, W)
        targets_adv[:, 0, 0, 0] = 1.0  # minimal positive
        targets_adv[:, 1, 0, 0] = 1.0

    preds_adv = torch.sigmoid(logits_adv)
    loss_adv = loss_fn(preds_adv, targets_adv)
    assert torch.isfinite(loss_adv)

    # Compute separate single-head losses for verification of decoupling
    loss_fn_h0 = nnPULoss(pi_p=heterogeneous_priors[0], gamma=1.0, beta=0.0)
    loss_fn_h1 = nnPULoss(pi_p=heterogeneous_priors[1], gamma=1.0, beta=0.0)

    loss_h0_isolated = loss_fn_h0(preds_adv[:, 0:1], targets_adv[:, 0:1])
    loss_h1_isolated = loss_fn_h1(preds_adv[:, 1:2], targets_adv[:, 1:2])

    loss_fn_all = nnPULoss(pi_p=heterogeneous_priors[:2], gamma=1.0, beta=0.0)
    loss_combined_01 = loss_fn_all(preds_adv[:, :2], targets_adv[:, :2])

    # In a class-decoupled loss, loss_combined_01 must equal mean(loss_h0 + loss_h1)
    expected_mean_loss = 0.5 * (loss_h0_isolated.item() + loss_h1_isolated.item())
    actual_loss = loss_combined_01.item()
    diff = abs(actual_loss - expected_mean_loss)
    assert diff < 1e-5, (
        f"Cross-head contamination detected! Combined loss {actual_loss:.6f} != expected {expected_mean_loss:.6f}"
    )

    results = {
        "multi_head_shape": (B, K, H, W),
        "heterogeneous_priors": heterogeneous_priors,
        "loss_finite": True,
        "loss_value": float(loss.item()),
        "gradients_finite": True,
        "gradient_shape": list(logits.grad.shape),
        "decoupling_verified": True,
        "cross_head_contamination_diff": diff,
        "verdict": "PASS",
    }
    return results


# ==============================================================================
# Challenge 4: Full-Raster Streaming Chunked Inference Stress Test
# ==============================================================================
def challenge_full_raster_streaming_inference() -> dict:
    """Test execute_full_raster_chunked_inference() on a realistic raster grid.

    Verify that 100% of valid onshore cells have finite predictions (0% NaN on land)
    and that peak memory stays under 600 MB RAM.
    """
    H, W = 120, 160  # 19,200 grid cells
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": np.nan,
        "width": W,
        "height": H,
        "count": 1,
        "crs": "EPSG:32750",
        "transform": Affine(250.0, 0.0, 500000.0, 0.0, -250.0, 9900000.0),
    }

    # Island geography: rows 10..109, cols 15..144 are valid onshore land
    rows_list = []
    cols_list = []
    for r in range(10, 110):
        for c in range(15, 145):
            rows_list.append(r)
            cols_list.append(c)
    n_land_cells = len(rows_list)  # Exactly 100 * 130 = 13,000 land cells

    rng = np.random.default_rng(42)
    feature_cols = [
        "phys_grav_hgm",
        "phys_mag_rtp",
        "phys_ternary_rad",
        "phys_dem_slope",
    ]

    data = {
        "row": np.array(rows_list, dtype=np.int32),
        "col": np.array(cols_list, dtype=np.int32),
        "label": rng.choice([0, 1], size=n_land_cells, p=[0.95, 0.05]).astype(np.int32),
        "w_canopy": rng.uniform(0.0, 1.0, size=n_land_cells).astype(np.float32),
    }
    for fc in feature_cols:
        data[fc] = rng.normal(0.0, 1.0, size=n_land_cells).astype(np.float32)

    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        parquet_path = Path(tmp_dir) / "test_dataset.parquet"
        df.to_parquet(parquet_path, index=False)

        # Mock K=4 Metals ensemble
        mock_k4 = MagicMock(spec=TwoStageK4Ensemble)
        mock_k4.predict_proba.side_effect = lambda X: np.full((len(X), 4), 0.65, dtype=np.float32)
        mock_k4.predict_std.side_effect = lambda X: np.full((len(X), 4), 0.05, dtype=np.float32)

        # Mock K=3 Coal ensemble
        mock_k3_coal = MagicMock(spec=TwoStageK3CoalEnsemble)
        mock_k3_coal.predict_full_outputs.side_effect = lambda X, w_canopy=None: {
            "probs_k3": np.full((len(X), 3), 0.70, dtype=np.float32),
            "probs_raw_k3": np.full((len(X), 3), 0.55, dtype=np.float32),
            "std_k3": np.full((len(X), 3), 0.04, dtype=np.float32),
        }

        # DAE context
        dae_metals = {
            f"oof_prob_k{k}": np.full((H, W), 0.60, dtype=np.float32) for k in range(4)
        }
        dae_metals["dae_recon_err"] = np.full((H, W), 0.02, dtype=np.float32)

        dae_coal = {
            f"oof_prob_k{k}": np.full((H, W), 0.50, dtype=np.float32) for k in range(3)
        }
        dae_coal["dae_recon_err"] = np.full((H, W), 0.03, dtype=np.float32)

        # Memory profile tracking
        tracemalloc.start()
        mem_before, _ = tracemalloc.get_traced_memory()

        # Run chunked streaming inference with small chunk size to force multiple streaming passes
        out = execute_full_raster_chunked_inference(
            parquet_path=parquet_path,
            feature_cols=feature_cols,
            H=H,
            W=W,
            profile=profile,
            k4_ensemble=mock_k4,
            dae_out=dae_metals,
            c_hat_k4=[0.5, 0.5, 0.5, 0.5],
            q_hat_k4_list=[0.4, 0.4, 0.4, 0.4],
            w_xgb_k4=[0.7, 0.7, 0.7, 0.7],
            w_unet_k4=[0.3, 0.3, 0.3, 0.3],
            k3_coal_ensemble=mock_k3_coal,
            dae_coal_out=dae_coal,
            c_hat_k3=[0.5, 0.5, 0.5],
            q_hat_k3_list=[0.4, 0.4, 0.4],
            w_xgb_coal=[0.75, 0.75, 0.75],
            w_unet_coal=[0.25, 0.25, 0.25],
            chunk_size=2000,  # 7 chunks
        )

        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mem_mb = peak_mem_bytes / (1024 * 1024)

    # 1. Assert peak memory < 600 MB
    assert peak_mem_mb < 600.0, f"Peak memory exceeded 600 MB: {peak_mem_mb:.2f} MB"

    cal_m = out["cal_metals_composite_2d"]
    cal_c = out["cal_coal_composite_2d"]
    lower_m = out["lower_metals_composite_2d"]
    lower_c = out["lower_coal_composite_2d"]

    # 2. Assert 100% of onshore valid land cells have finite predictions
    finite_m_count = int(np.sum(np.isfinite(cal_m)))
    finite_c_count = int(np.sum(np.isfinite(cal_c)))
    nan_m_on_land = 0
    nan_c_on_land = 0

    for r, c in zip(rows_list, cols_list):
        if not np.isfinite(cal_m[r, c]):
            nan_m_on_land += 1
        if not np.isfinite(cal_c[r, c]):
            nan_c_on_land += 1

    assert nan_m_on_land == 0, f"Found {nan_m_on_land} NaN holes on valid land in metals!"
    assert nan_c_on_land == 0, f"Found {nan_c_on_land} NaN holes on valid land in coal!"
    assert finite_m_count == n_land_cells, (
        f"Finite cells {finite_m_count} != expected land cells {n_land_cells}"
    )
    assert finite_c_count == n_land_cells, (
        f"Finite cells {finite_c_count} != expected land cells {n_land_cells}"
    )

    # 3. Assert ocean background cells remain cleanly NaN
    assert np.isnan(cal_m[0, 0]), "Ocean background pixel (0, 0) should be NaN"
    assert np.isnan(cal_c[0, 0]), "Ocean background pixel (0, 0) should be NaN"

    results = {
        "grid_shape": (H, W),
        "total_pixels": H * W,
        "onshore_valid_cells": n_land_cells,
        "finite_metals_cells": finite_m_count,
        "finite_coal_cells": finite_c_count,
        "nan_on_land_percentage": 0.0,
        "peak_memory_mb": peak_mem_mb,
        "memory_bound_asserted_mb": 600.0,
        "verdict": "PASS",
    }
    return results


# ==============================================================================
# Challenge 5: Disk Governance & Checkpoint Exclusion Stress Test
# ==============================================================================
def challenge_disk_governance() -> dict:
    """Stress test disk space checks, historical run pruning, and checkpoint exclusion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Test check_disk_space
        # Adequate threshold succeeds
        assert check_disk_space(target_path=tmp_path, min_free_gb=0.0001) is True

        # High threshold raises DiskSpaceError
        raised = False
        try:
            check_disk_space(target_path=tmp_path, min_free_gb=10_000_000.0)
        except DiskSpaceError as e:
            raised = True
            assert isinstance(e, OSError), "DiskSpaceError must subclass OSError"
        assert raised, "check_disk_space failed to raise DiskSpaceError on high threshold"

        # Mock disk usage simulating 0.35 GB free
        with patch("shutil.disk_usage", return_value=(100 * (1024**3), 99.65 * (1024**3), int(0.35 * (1024**3)))):
            mock_raised = False
            try:
                check_disk_space(target_path=tmp_path, min_free_gb=1.0)
            except DiskSpaceError as e:
                mock_raised = True
                assert "0.35 GB free" in str(e)
            assert mock_raised, "check_disk_space did not raise on simulated 0.35 GB free space"

        # 2. Test prune_historical_runs
        history_dir = tmp_path / "run_history"
        region_dir = history_dir / "test_region"
        region_dir.mkdir(parents=True, exist_ok=True)

        catalog_path = history_dir / "training_runs_catalog.json"
        catalog = {"version": "1.0", "runs": []}

        run_names = [f"run_20261029_0{i}0000_00000{i}" for i in range(1, 6)]
        for rname in run_names:
            rdir = region_dir / rname
            rdir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "run_id": f"RUN_TEST_{rname}",
                "region": "TEST_REGION",
                "run_dir": str(rdir.resolve()),
            }
            (rdir / "manifest.json").write_text(json.dumps(manifest))
            catalog["runs"].append(manifest)

        catalog_path.write_text(json.dumps(catalog))

        # Prune with max_keep_runs=2
        with patch("src.models.training_run_manager.CATALOG_PATH", catalog_path):
            pruned = prune_historical_runs("TEST_REGION", max_keep_runs=2, base_dir=history_dir)

        assert len(pruned) == 3, f"Expected 3 pruned runs, got {len(pruned)}"
        remaining_dirs = sorted([d.name for d in region_dir.iterdir() if d.is_dir()])
        assert len(remaining_dirs) == 2, f"Expected exactly 2 remaining directories, got {len(remaining_dirs)}"
        assert remaining_dirs == ["run_20261029_040000_000004", "run_20261029_050000_000005"]

        # Catalog must have exactly 2 runs
        updated_catalog = json.loads(catalog_path.read_text())
        assert len(updated_catalog["runs"]) == 2

        # 3. Test archive_existing_run *.pt Checkpoint Exclusion
        outputs_dir = tmp_path / "outputs"
        active_out = outputs_dir / "seven_pass_test"
        active_out.mkdir(parents=True, exist_ok=True)
        models_dir = active_out / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        maps_dir = active_out / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        targets_dir = active_out / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)

        # Create mock checkpoints
        (models_dir / "k4_ensemble.joblib").write_bytes(b"lightweight_joblib_data")
        (models_dir / "k3_coal_ensemble.joblib").write_bytes(b"lightweight_coal_data")
        (models_dir / "calibration_priors.json").write_text('{"c_hat": 0.5}')
        (models_dir / "dae_metals_fold_0.pt").write_bytes(b"heavy_pytorch_weights_0")
        (models_dir / "dae_metals_fold_1.pt").write_bytes(b"heavy_pytorch_weights_1")
        (models_dir / "dae_coal_fold_0.pt").write_bytes(b"heavy_coal_weights_0")

        # Nested checkpoint
        nested_dir = models_dir / "checkpoints"
        nested_dir.mkdir(parents=True, exist_ok=True)
        (nested_dir / "sub_checkpoint.pt").write_bytes(b"nested_heavy_weights")
        (nested_dir / "hyperparams.json").write_text('{"lr": 0.001}')

        (maps_dir / "calibrated_prospectivity.tif").write_bytes(b"mock_geotiff")
        (targets_dir / "site_cards.json").write_text("[]")
        (active_out / "seven_pass_summary.json").write_text(json.dumps({"pr_auc_fused": 0.91}))

        with (
            patch("src.models.training_run_manager.OUTPUTS_DIR", outputs_dir),
            patch("src.models.training_run_manager.RUN_HISTORY_DIR", history_dir),
            patch("src.models.training_run_manager.CATALOG_PATH", catalog_path),
            patch("src.models.training_run_manager.ROOT_DIR", tmp_path),
        ):
            manifest = archive_existing_run(active_out, "TEST_REGION", run_label="Test Archival")

        assert manifest is not None
        archived_run_dir = Path(manifest["run_dir"])
        archived_models = archived_run_dir / "models"

        # Assert *.pt files are completely excluded
        pt_files_in_archive = list(archived_run_dir.rglob("*.pt"))
        assert len(pt_files_in_archive) == 0, (
            f"Found *.pt files in archive directory: {[str(p) for p in pt_files_in_archive]}"
        )

        # Assert .joblib, .json, .tif are present
        assert (archived_models / "k4_ensemble.joblib").exists()
        assert (archived_models / "k3_coal_ensemble.joblib").exists()
        assert (archived_models / "calibration_priors.json").exists()
        assert (archived_models / "checkpoints" / "hyperparams.json").exists()
        assert (archived_run_dir / "maps" / "calibrated_prospectivity.tif").exists()

        # Assert active directory was NOT mutated (original .pt still exists for continuity)
        assert (models_dir / "dae_metals_fold_0.pt").exists()
        assert (models_dir / "dae_metals_fold_1.pt").exists()

    results = {
        "check_disk_space_success": True,
        "check_disk_space_raises_disk_space_error": True,
        "disk_space_error_subclasses_oserror": True,
        "prune_historical_runs_retains_exactly_2": True,
        "pt_files_excluded_from_archive": True,
        "joblib_and_metadata_preserved_in_archive": True,
        "active_models_retained_on_disk": True,
        "verdict": "PASS",
    }
    return results


# ==============================================================================
# PyTest Integration Tests
# ==============================================================================
def test_challenge_1_xgboost_pu_prior():
    res = challenge_xgboost_pu_prior()
    assert res["verdict"] == "PASS"


def test_challenge_2_optical_warmstart_and_variance():
    res = challenge_optical_warmstart_and_variance()
    assert res["verdict"] == "PASS"


def test_challenge_3_class_decoupled_nnpu_loss():
    res = challenge_class_decoupled_nnpu_loss()
    assert res["verdict"] == "PASS"


def test_challenge_4_full_raster_streaming_inference():
    res = challenge_full_raster_streaming_inference()
    assert res["verdict"] == "PASS"


def test_challenge_5_disk_governance():
    res = challenge_disk_governance()
    assert res["verdict"] == "PASS"


# ==============================================================================
# Direct Command-Line Execution Harness
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING ADVERSARIAL EMPIRICAL CHALLENGE HARNESS — CHALLENGER 2")
    print("=" * 80)

    print("\n[Challenge 1/5] Testing XGBoost PU Prior (scale_pos_weight == 1.0, uninflated raw scores)...")
    res1 = challenge_xgboost_pu_prior()
    print(f"  -> scale_pos_weight asserted: {res1['scale_pos_weight_asserted']}")
    print(f"  -> mean raw pos prob: {res1['mean_raw_pos_prob']:.4f} (must be < 0.85)")
    print(f"  -> median raw pos prob: {res1['median_raw_pos_prob']:.4f}")
    print(f"  -> counter biased (scale_pos_weight=100) mean pos prob: {res1['counter_biased_scale_pos_weight_100_mean']:.4f}")
    print(f"  -> Verdict: {res1['verdict']}")

    print("\n[Challenge 2/5] Testing Optical Warm-Start & Epistemic Variance Combination...")
    res2 = challenge_optical_warmstart_and_variance()
    print(f"  -> Optical heads warm-started: {res2['optical_heads_warmstarted']}")
    for w, diff in res2["max_discrepancies"]:
        print(f"  -> w_canopy={w:.2f}: max discrepancy = {diff:.2e}")
    print(f"  -> Verdict: {res2['verdict']}")

    print("\n[Challenge 3/5] Testing Class-Decoupled nnPULoss Multi-Head Stress...")
    res3 = challenge_class_decoupled_nnpu_loss()
    print(f"  -> Multi-head shape: {res3['multi_head_shape']}")
    print(f"  -> Heterogeneous priors: {res3['heterogeneous_priors']}")
    print(f"  -> Decoupling verified (diff = {res3['cross_head_contamination_diff']:.2e})")
    print(f"  -> Verdict: {res3['verdict']}")

    print("\n[Challenge 4/5] Testing Full-Raster Streaming Chunked Inference (0% NaN, Peak RAM)...")
    res4 = challenge_full_raster_streaming_inference()
    print(f"  -> Total pixels: {res4['total_pixels']}")
    print(f"  -> Onshore valid cells: {res4['onshore_valid_cells']}")
    print(f"  -> Finite metals cells: {res4['finite_metals_cells']}")
    print(f"  -> NaN on land percentage: {res4['nan_on_land_percentage']:.2f}%")
    print(f"  -> Peak memory: {res4['peak_memory_mb']:.2f} MB (bound: < 600.0 MB)")
    print(f"  -> Verdict: {res4['verdict']}")

    print("\n[Challenge 5/5] Testing Disk Governance & Checkpoint Exclusion...")
    res5 = challenge_disk_governance()
    print(f"  -> check_disk_space success: {res5['check_disk_space_success']}")
    print(f"  -> check_disk_space raises DiskSpaceError: {res5['check_disk_space_raises_disk_space_error']}")
    print(f"  -> prune_historical_runs retains exactly 2: {res5['prune_historical_runs_retains_exactly_2']}")
    print(f"  -> *.pt files excluded from archive: {res5['pt_files_excluded_from_archive']}")
    print(f"  -> Verdict: {res5['verdict']}")

    print("\n" + "=" * 80)
    print("ALL 5 EMPIRICAL CHALLENGES PASSED EMPIRICALLY!")
    print("=" * 80)
