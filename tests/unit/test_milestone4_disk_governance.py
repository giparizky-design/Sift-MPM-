"""tests/unit/test_milestone4_disk_governance.py
=============================================
Regression unit tests for Milestone 4: Disk Governance & Crash-Safe Run Archival (Requirement R5).

Tests cover:
  1. R5.1: Historical run pruning (prune_historical_runs retains exactly max_keep_runs
     and cleans up purged records from training_runs_catalog.json).
  2. R5.2: Checkpoint deep-copy exclusion (archive_existing_run copies .joblib models,
     metadata, and configs, but strictly excludes heavy .pt fold checkpoints).
  3. R5.3: Pre-flight disk safety check (check_disk_space checks available storage,
     succeeds when adequate, and raises DiskSpaceError subclassing OSError when below threshold).
  4. Integration: Automatic pruning and pre-flight check execution inside archive_existing_run.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
import pytest

from src.models.training_run_manager import (
    DiskSpaceError,
    check_disk_space,
    prune_historical_runs,
    archive_existing_run,
    load_runs_catalog,
    save_runs_catalog,
)


@pytest.fixture
def temp_run_env(tmp_path, monkeypatch):
    """Setup an isolated sandboxed outputs environment for training run tests."""
    mock_outputs = tmp_path / "outputs"
    mock_history = mock_outputs / "run_history"
    mock_catalog = mock_history / "training_runs_catalog.json"
    mock_outputs.mkdir(parents=True, exist_ok=True)
    mock_history.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("src.models.training_run_manager.OUTPUTS_DIR", mock_outputs)
    monkeypatch.setattr("src.models.training_run_manager.RUN_HISTORY_DIR", mock_history)
    monkeypatch.setattr("src.models.training_run_manager.CATALOG_PATH", mock_catalog)
    monkeypatch.setattr("src.models.training_run_manager.ROOT_DIR", tmp_path)

    return {
        "root": tmp_path,
        "outputs": mock_outputs,
        "history": mock_history,
        "catalog": mock_catalog,
    }


# =====================================================================
# R5.3: Pre-Flight Disk Safety Check Tests
# =====================================================================


def test_check_disk_space_success(tmp_path):
    """Verify check_disk_space returns True when free space is above the threshold."""
    # A tiny threshold like 0.0001 GB (~100 KB) should succeed on any working machine
    result = check_disk_space(target_path=tmp_path, min_free_gb=0.0001)
    assert result is True


def test_check_disk_space_raises_disk_space_error(tmp_path):
    """Verify check_disk_space raises DiskSpaceError when free space is below threshold."""
    # A huge threshold like 10,000,000 GB must fail
    with pytest.raises(DiskSpaceError) as exc_info:
        check_disk_space(target_path=tmp_path, min_free_gb=10_000_000.0)

    # Verify exception is subclass of OSError
    assert isinstance(exc_info.value, OSError)
    err_msg = str(exc_info.value)
    assert "Pre-flight disk space check failed" in err_msg
    assert "minimum required is 10000000.00 GB" in err_msg


def test_check_disk_space_with_mocked_disk_usage(monkeypatch, tmp_path):
    """Verify check_disk_space behavior with simulated low and adequate disk space."""
    # Simulate 500 MB free (0.465 GB)
    simulated_low = (100 * (1024**3), 99.5 * (1024**3), int(0.5 * (1024**3)))
    monkeypatch.setattr(shutil, "disk_usage", lambda path: simulated_low)

    with pytest.raises(DiskSpaceError) as exc_info:
        check_disk_space(target_path=tmp_path, min_free_gb=1.0)
    assert "0.50 GB free" in str(exc_info.value)

    # Simulate 2.5 GB free
    simulated_ok = (100 * (1024**3), 97.5 * (1024**3), int(2.5 * (1024**3)))
    monkeypatch.setattr(shutil, "disk_usage", lambda path: simulated_ok)

    assert check_disk_space(target_path=tmp_path, min_free_gb=1.0) is True


def test_check_disk_space_nonexistent_path(tmp_path):
    """Verify check_disk_space resolves to existing ancestor if target path does not exist."""
    nested_nonexistent = tmp_path / "deeply" / "nested" / "nonexistent" / "output_dir"
    # Should resolve to tmp_path and succeed
    result = check_disk_space(target_path=nested_nonexistent, min_free_gb=0.0001)
    assert result is True


# =====================================================================
# R5.1: Historical Run Pruning Tests
# =====================================================================


def test_prune_historical_runs_enforces_max_keep_runs(temp_run_env):
    """Verify prune_historical_runs retains exactly max_keep_runs and removes older ones."""
    history_dir = temp_run_env["history"]
    region_dir = history_dir / "borneo"
    region_dir.mkdir(parents=True, exist_ok=True)

    catalog_data = {"version": "1.0", "runs": []}
    run_names = [
        "run_20261029_010000_000001",
        "run_20261029_020000_000002",
        "run_20261029_030000_000003",
        "run_20261029_040000_000004",
        "run_20261029_050000_000005",
    ]

    # Create 5 historical run directories with manifests and catalog records
    for i, rname in enumerate(run_names):
        rdir = region_dir / rname
        rdir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": f"RUN_BORNEO_{rname}",
            "region": "BORNEO",
            "run_dir": str(rdir.resolve()),
            "relative_path": str(rdir.relative_to(temp_run_env["root"])),
        }
        (rdir / "manifest.json").write_text(json.dumps(manifest))
        catalog_data["runs"].append(manifest)

    temp_run_env["catalog"].write_text(json.dumps(catalog_data))

    # Prune keeping at most 2 runs
    pruned = prune_historical_runs("BORNEO", max_keep_runs=2, base_dir=history_dir)

    # Must return 3 pruned paths
    assert len(pruned) == 3
    pruned_names = {p.name for p in pruned}
    assert pruned_names == {
        "run_20261029_010000_000001",
        "run_20261029_020000_000002",
        "run_20261029_030000_000003",
    }

    # Verify pruned directories no longer exist on disk
    for p in pruned:
        assert not p.exists()

    # Exactly 2 newest runs remain on disk
    remaining_dirs = sorted([d.name for d in region_dir.iterdir() if d.is_dir()])
    assert remaining_dirs == [
        "run_20261029_040000_000004",
        "run_20261029_050000_000005",
    ]

    # Verify catalog was cleaned up
    updated_catalog = json.loads(temp_run_env["catalog"].read_text())
    assert len(updated_catalog["runs"]) == 2
    surviving_ids = {r["run_id"] for r in updated_catalog["runs"]}
    assert surviving_ids == {
        "RUN_BORNEO_run_20261029_040000_000004",
        "RUN_BORNEO_run_20261029_050000_000005",
    }


def test_prune_historical_runs_no_op_when_fewer_than_max(temp_run_env):
    """Verify prune_historical_runs does nothing when run count <= max_keep_runs."""
    history_dir = temp_run_env["history"]
    region_dir = history_dir / "east_kalimantan"
    region_dir.mkdir(parents=True, exist_ok=True)

    r1 = region_dir / "run_20261029_100000_000001"
    r2 = region_dir / "run_20261029_110000_000002"
    r1.mkdir(parents=True, exist_ok=True)
    r2.mkdir(parents=True, exist_ok=True)

    pruned = prune_historical_runs("EAST_KALIMANTAN", max_keep_runs=2, base_dir=history_dir)
    assert pruned == []
    assert r1.exists()
    assert r2.exists()


def test_prune_historical_runs_nonexistent_region(temp_run_env):
    """Verify prune_historical_runs returns empty list gracefully for nonexistent region."""
    pruned = prune_historical_runs("NONEXISTENT_REGION", max_keep_runs=2)
    assert pruned == []


# =====================================================================
# R5.2: Checkpoint Deep-Copy Exclusion Tests
# =====================================================================


def test_archive_existing_run_excludes_pt_checkpoints(temp_run_env):
    """Verify archive_existing_run copies .joblib and metadata but excludes heavy .pt fold checkpoints."""
    out_dir = temp_run_env["outputs"] / "seven_pass_borneo"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    targets_dir = out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    # Populate active output with lightweight models and heavy .pt checkpoints
    (models_dir / "k4_ensemble.joblib").write_bytes(b"joblib_xgboost_ensemble")
    (models_dir / "k3_coal_ensemble.joblib").write_bytes(b"joblib_coal_ensemble")
    (models_dir / "calibration_priors.json").write_text(json.dumps({"c_hat": 0.25}))
    (models_dir / "dae_metals_fold_0.pt").write_bytes(b"pytorch_weights_fold_0")
    (models_dir / "dae_metals_fold_1.pt").write_bytes(b"pytorch_weights_fold_1")
    (models_dir / "unet_checkpoint.pt").write_bytes(b"pytorch_checkpoint")

    # Nested subfolder with a .pt file
    nested_dir = models_dir / "checkpoints"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "nested_fold.pt").write_bytes(b"nested_pytorch")
    (nested_dir / "config.json").write_text('{"lr": 0.001}')

    (maps_dir / "calibrated_prospectivity.tif").write_bytes(b"geotiff_data")
    (targets_dir / "site_cards.json").write_text(json.dumps([{"target_id": "TGT_001"}]))
    (out_dir / "seven_pass_summary.json").write_text(json.dumps({"pr_auc_fused": 0.88}))

    manifest = archive_existing_run(out_dir, "BORNEO", run_label="Test Checkpoint Exclusion")
    assert manifest is not None
    archived_dir = Path(manifest["run_dir"])
    archived_models = archived_dir / "models"

    # Lightweight files MUST exist in archived models
    assert (archived_models / "k4_ensemble.joblib").exists()
    assert (archived_models / "k4_ensemble.joblib").read_bytes() == b"joblib_xgboost_ensemble"
    assert (archived_models / "k3_coal_ensemble.joblib").exists()
    assert (archived_models / "calibration_priors.json").exists()
    assert (archived_models / "checkpoints" / "config.json").exists()

    # Heavy .pt files MUST NOT exist in archived models
    assert not (archived_models / "dae_metals_fold_0.pt").exists()
    assert not (archived_models / "dae_metals_fold_1.pt").exists()
    assert not (archived_models / "unet_checkpoint.pt").exists()
    assert not (archived_models / "checkpoints" / "nested_fold.pt").exists()

    # Active output directory MUST still have all original files intact
    assert (models_dir / "dae_metals_fold_0.pt").exists()
    assert (models_dir / "dae_metals_fold_1.pt").exists()
    assert (models_dir / "k4_ensemble.joblib").exists()


# =====================================================================
# Integration: Automatic Pruning & Disk Safety in archive_existing_run
# =====================================================================


def test_archive_existing_run_triggers_automatic_pruning(temp_run_env):
    """Verify archive_existing_run automatically prunes older runs to keep max 2."""
    history_region_dir = temp_run_env["history"] / "borneo"
    history_region_dir.mkdir(parents=True, exist_ok=True)

    # Seed 2 prior archived runs
    r1 = history_region_dir / "run_20261029_010000_000001"
    r2 = history_region_dir / "run_20261029_020000_000002"
    r1.mkdir(parents=True, exist_ok=True)
    r2.mkdir(parents=True, exist_ok=True)

    catalog = load_runs_catalog()
    catalog["runs"] = [
        {"run_id": "RUN_BORNEO_001", "region": "BORNEO", "run_dir": str(r1.resolve())},
        {"run_id": "RUN_BORNEO_002", "region": "BORNEO", "run_dir": str(r2.resolve())},
    ]
    save_runs_catalog(catalog)

    # Create active directory to archive
    out_dir = temp_run_env["outputs"] / "seven_pass_borneo"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "targets").mkdir(parents=True, exist_ok=True)
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)
    (out_dir / "maps" / "calibrated_prospectivity.tif").write_bytes(b"map")
    (out_dir / "targets" / "site_cards.json").write_text("[]")
    (out_dir / "seven_pass_summary.json").write_text(json.dumps({"pr_auc_fused": 0.90}))

    # Archive the new run - this should trigger automatic pruning
    manifest = archive_existing_run(out_dir, "BORNEO")
    assert manifest is not None

    # After archiving 3rd run, only 2 should remain in run_history/borneo
    remaining_runs = sorted([d.name for d in history_region_dir.iterdir() if d.is_dir()])
    assert len(remaining_runs) == 2
    # Oldest run r1 must have been pruned
    assert not r1.exists()
    assert r2.exists()
    assert Path(manifest["run_dir"]).exists()

    # Catalog must have exactly 2 runs
    updated_catalog = load_runs_catalog()
    assert len(updated_catalog["runs"]) == 2
    remaining_run_ids = {r["run_id"] for r in updated_catalog["runs"]}
    assert "RUN_BORNEO_001" not in remaining_run_ids
    assert "RUN_BORNEO_002" in remaining_run_ids
    assert manifest["run_id"] in remaining_run_ids


def test_archive_fails_fast_on_low_disk_space(temp_run_env, monkeypatch):
    """Verify archive_existing_run fails fast with DiskSpaceError when free space < 1.0 GB."""
    out_dir = temp_run_env["outputs"] / "seven_pass_borneo"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "targets").mkdir(parents=True, exist_ok=True)
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)
    (out_dir / "maps" / "calibrated_prospectivity.tif").write_bytes(b"map")
    (out_dir / "targets" / "site_cards.json").write_text("[]")
    (out_dir / "seven_pass_summary.json").write_text(json.dumps({"pr_auc_fused": 0.90}))

    # Simulate 0.3 GB free disk space
    simulated_low = (100 * (1024**3), 99.7 * (1024**3), int(0.3 * (1024**3)))
    monkeypatch.setattr(shutil, "disk_usage", lambda path: simulated_low)

    with pytest.raises(DiskSpaceError) as exc_info:
        archive_existing_run(out_dir, "BORNEO")

    assert "0.30 GB free" in str(exc_info.value)

    # Verify no archive directory was created in run_history
    history_region = temp_run_env["history"] / "borneo"
    assert not history_region.exists()
