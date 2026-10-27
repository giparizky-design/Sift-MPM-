"""
tests/unit/test_incremental_weights.py
======================================
Unit tests for continuous / incremental weight updates:
  1. _BaggedXGBoostHead save/load and incremental warm-starting.
  2. TwoStageK4Ensemble save/load and continuous warm-starting.
  3. MPMDistrictDAEUNet spatial CV checkpointing and weight continuity.
  4. Training run manager archival of model checkpoints.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import xgboost as xgb

from src.models.pu_pipeline import (
    _BaggedXGBoostHead,
    TwoStageK4Ensemble,
    fit_k4_independent_xgboost_heads,
)
from src.models.unet_model import MPMDistrictDAEUNet, DistrictDAEUNet
from src.models.training_run_manager import archive_existing_run, load_runs_catalog


def test_bagged_xgboost_head_save_load_and_incremental_fit(tmp_path):
    """Verify _BaggedXGBoostHead saves, loads, and accumulates trees during warm-start."""
    np.random.seed(42)
    N, D = 100, 6
    X = np.random.randn(N, D).astype(np.float32)
    y = np.random.randint(0, 2, size=N)
    y[:10] = 1  # ensure positives

    # Run 1: initial fit with 10 trees
    head_1 = _BaggedXGBoostHead(n_estimators=10, n_bag=2, max_depth=3, random_state=42)
    head_1.fit(X, y)
    assert len(head_1.models_) == 2
    for m in head_1.models_:
        assert m.get_booster().num_boosted_rounds() == 10

    # Save and reload
    save_path = tmp_path / "head_1.joblib"
    head_1.save(save_path)
    loaded_head_1 = _BaggedXGBoostHead.load(save_path)
    assert len(loaded_head_1.models_) == 2

    # Run 2: warm-start from loaded head with 15 additional trees
    head_2 = _BaggedXGBoostHead(n_estimators=15, n_bag=2, max_depth=3, random_state=42)
    head_2.fit(X, y, prior_models=loaded_head_1.models_)
    assert len(head_2.models_) == 2
    # Boosted rounds must accumulate: 10 + 15 = 25 trees
    for m in head_2.models_:
        assert m.get_booster().num_boosted_rounds() == 25

    # Predictions remain valid probabilities in [0, 1]
    probs = head_2.predict_proba(X)
    assert probs.shape == (N,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_two_stage_k4_ensemble_continuous_warm_start(tmp_path):
    """Verify TwoStageK4Ensemble warm-starts all 4 deposit heads from prior checkpoint."""
    np.random.seed(42)
    N, D = 150, 5
    X = np.random.randn(N, D).astype(np.float32)
    y_k4 = np.zeros((N, 4), dtype=np.int32)
    for k in range(4):
        y_k4[k * 15 : (k + 1) * 15, k] = 1

    # Run 1: Initial training run
    ens_1 = TwoStageK4Ensemble(n_estimators=10, n_bag=2, random_state=42)
    ens_1.fit(X=X, y_k4=y_k4)

    # Verify initial tree counts
    assert ens_1.head_1_.models_[0].get_booster().num_boosted_rounds() == 10
    assert ens_1.head_2_.models_[0].get_booster().num_boosted_rounds() == 10
    assert ens_1.head_3_.models_[0].get_booster().num_boosted_rounds() == 10
    assert ens_1.head_0_.models_[0].get_booster().num_boosted_rounds() == 10

    # Save ensemble checkpoint
    ens_path = tmp_path / "models" / "k4_ensemble.joblib"
    ens_1.save(ens_path)
    assert ens_path.exists()

    # Load ensemble checkpoint
    loaded_ens = TwoStageK4Ensemble.load(ens_path)

    # Run 2: Incremental warm-start run adding 10 additional trees
    ens_2 = TwoStageK4Ensemble(n_estimators=10, n_bag=2, random_state=42)
    ens_2.fit(X=X, y_k4=y_k4, prior_ensemble=loaded_ens)

    # Verify all 4 heads accumulated trees: 10 + 10 = 20 trees
    assert ens_2.head_1_.models_[0].get_booster().num_boosted_rounds() == 20
    assert ens_2.head_2_.models_[0].get_booster().num_boosted_rounds() == 20
    assert ens_2.head_3_.models_[0].get_booster().num_boosted_rounds() == 20
    assert ens_2.head_0_.models_[0].get_booster().num_boosted_rounds() == 20

    # Output predictions and uncertainties are valid
    preds = ens_2.predict_proba(X)
    stds = ens_2.predict_std(X)
    assert preds.shape == (N, 4)
    assert stds.shape == (N, 4)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()
    assert (stds >= 0.0).all()


def test_mpm_district_dae_unet_checkpoint_persistence(tmp_path):
    """Verify MPMDistrictDAEUNet saves and loads fold checkpoints across training sessions."""
    torch.manual_seed(42)
    np.random.seed(42)
    C, H, W = 4, 32, 32
    X_cube = np.random.randn(C, H, W).astype(np.float32)
    y_grid = np.zeros((4, H, W), dtype=np.float32)
    y_grid[:, 10:20, 10:20] = 1.0
    block_ids = np.zeros((H, W), dtype=np.int32)
    block_ids[:16, :] = 0
    block_ids[16:, :] = 1

    checkpoints_dir = tmp_path / "models"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    dae = MPMDistrictDAEUNet(in_channels=C, n_classes=4, window_px=16, batch_size=8)

    # Run 1: initial 1 epoch per fold with checkpoint saving
    out_1 = dae.cross_validate_spatial_blocks(
        X_grid_50m=X_cube,
        y_grid_k4=y_grid,
        block_ids_50m=block_ids,
        decimate_factor=1,
        epochs=1,
        checkpoints_dir=checkpoints_dir,
    )
    assert "dae_recon_err" in out_1
    # Check that fold weights were saved to disk
    ckpt_files = list(checkpoints_dir.glob("dae_unet_fold_*.pt"))
    assert len(ckpt_files) >= 1

    # Run 2: warm-start from saved fold weights
    out_2 = dae.cross_validate_spatial_blocks(
        X_grid_50m=X_cube,
        y_grid_k4=y_grid,
        block_ids_50m=block_ids,
        decimate_factor=1,
        epochs=1,
        checkpoints_dir=checkpoints_dir,
    )
    assert "dae_recon_err" in out_2
    assert out_2["dae_recon_err"].shape == (H, W)


def test_archival_preserves_models_directory(tmp_path, monkeypatch):
    """Verify archive_existing_run archives the models directory with all weights."""
    mock_outputs = tmp_path / "outputs"
    mock_history = mock_outputs / "run_history"
    mock_catalog = mock_history / "training_runs_catalog.json"
    mock_outputs.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("src.models.training_run_manager.OUTPUTS_DIR", mock_outputs)
    monkeypatch.setattr("src.models.training_run_manager.RUN_HISTORY_DIR", mock_history)
    monkeypatch.setattr("src.models.training_run_manager.CATALOG_PATH", mock_catalog)
    monkeypatch.setattr("src.models.training_run_manager.ROOT_DIR", tmp_path)

    out_dir = mock_outputs / "seven_pass_east_kalimantan"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    targets_dir = out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    # Create mock model checkpoint
    (models_dir / "k4_ensemble.joblib").write_bytes(b"mock_ensemble_weights")
    (models_dir / "dae_unet_fold_0.pt").write_bytes(b"mock_pytorch_weights")
    (maps_dir / "calibrated_prospectivity.tif").write_bytes(b"mock_map")
    (targets_dir / "site_cards.json").write_text("[]")
    (out_dir / "seven_pass_summary.json").write_text(
        json.dumps({"pr_auc_fused": 0.89, "training_mode": "incremental_warm_start", "is_warm_started": True})
    )

    manifest = archive_existing_run(out_dir, "EAST_KALIMANTAN")
    assert manifest is not None
    archived_run_dir = Path(manifest["run_dir"])

    # Verify models were copied to the archive directory
    assert (archived_run_dir / "models" / "k4_ensemble.joblib").exists()
    assert (archived_run_dir / "models" / "dae_unet_fold_0.pt").exists()
    assert (archived_run_dir / "models" / "k4_ensemble.joblib").read_bytes() == b"mock_ensemble_weights"
    assert manifest["training_mode"] == "incremental_warm_start"
    assert manifest["is_warm_started"] is True
