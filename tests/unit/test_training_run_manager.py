"""
tests/unit/test_training_run_manager.py
======================================
Unit tests for the Training Run Versioning, Comparison, and Archival Manager.
"""

import json
import shutil
from pathlib import Path
import pytest

from src.models.training_run_manager import (
    archive_existing_run,
    register_completed_run,
    list_training_runs,
    compare_training_runs,
    clean_old_training_data,
    load_runs_catalog,
    save_runs_catalog,
    CATALOG_PATH,
    RUN_HISTORY_DIR,
)


@pytest.fixture
def temp_run_env(tmp_path, monkeypatch):
    """Setup a sandboxed outputs directory for training run tests."""
    mock_outputs = tmp_path / "outputs"
    mock_history = mock_outputs / "run_history"
    mock_catalog = mock_history / "training_runs_catalog.json"
    mock_outputs.mkdir(parents=True, exist_ok=True)

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


def test_archive_existing_run_success(temp_run_env):
    """Verify that an existing run directory is archived with its manifest and cataloged."""
    out_dir = temp_run_env["outputs"] / "seven_pass_borneo"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets_dir = out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "region": "BORNEO",
        "pr_auc_xgb": 0.86,
        "pr_auc_unet": 0.81,
        "pr_auc_fused": 0.88,
        "elkan_noto_c_hat": 0.25,
        "conformal_q_hat_90": 0.65,
        "n_ranked_targets": 12,
    }
    (out_dir / "seven_pass_summary.json").write_text(json.dumps(summary_data))
    (targets_dir / "site_cards.json").write_text(
        json.dumps([{"target_id": "TGT_0001", "conformal_90pct_target_set": ["epithermal_au"]}])
    )
    (maps_dir / "calibrated_prospectivity.tif").write_bytes(b"mock_tif_bytes")

    manifest = archive_existing_run(out_dir, "BORNEO", run_label="Test Borneo Archival")

    assert manifest is not None
    assert manifest["region"] == "BORNEO"
    assert manifest["is_active"] is False
    assert manifest["metrics"]["pr_auc_fused"] == 0.88
    assert manifest["n_ranked_targets"] == 12
    assert "epithermal_au" in manifest["ore_categories"]

    # Verify catalog contains this run
    catalog = load_runs_catalog()
    assert len(catalog["runs"]) == 1
    assert catalog["runs"][0]["run_id"] == manifest["run_id"]


def test_archive_empty_directory_returns_none(temp_run_env):
    """Archiving a directory with no model artifacts returns None."""
    out_dir = temp_run_env["outputs"] / "empty_dir"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = archive_existing_run(out_dir, "BORNEO")
    assert manifest is None


def test_register_completed_run_and_switch_active(temp_run_env):
    """Verify that registering a new run marks previous runs as inactive."""
    out_dir = temp_run_env["outputs"] / "seven_pass_borneo"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seven_pass_summary.json").write_text(json.dumps({"pr_auc_fused": 0.85}))

    run1 = register_completed_run(
        out_dir, "BORNEO", {"pr_auc_fused": 0.85, "n_ranked_targets": 5}, run_label="Run 1"
    )
    assert run1["is_active"] is True

    run2 = register_completed_run(
        out_dir, "BORNEO", {"pr_auc_fused": 0.90, "n_ranked_targets": 8}, run_label="Run 2"
    )
    assert run2["is_active"] is True

    catalog = load_runs_catalog()
    runs = catalog["runs"]
    assert len(runs) == 2
    # run1 should now be marked inactive
    assert runs[0]["is_active"] is False
    assert runs[1]["is_active"] is True


def test_list_training_runs_filtering(temp_run_env):
    """Verify listing runs with optional regional filter."""
    out_b = temp_run_env["outputs"] / "seven_pass_borneo"
    out_y = temp_run_env["outputs"] / "seven_pass_yilgarn"
    out_b.mkdir(parents=True, exist_ok=True)
    out_y.mkdir(parents=True, exist_ok=True)

    register_completed_run(out_b, "BORNEO", {"pr_auc_fused": 0.88})
    register_completed_run(out_y, "YILGARN", {"pr_auc_fused": 0.87})

    all_runs = list_training_runs()
    assert len(all_runs) == 2

    borneo_runs = list_training_runs("BORNEO")
    assert len(borneo_runs) == 1
    assert borneo_runs[0]["region"] == "BORNEO"

    yilgarn_runs = list_training_runs("YILGARN")
    assert len(yilgarn_runs) == 1
    assert yilgarn_runs[0]["region"] == "YILGARN"


def test_compare_training_runs(temp_run_env):
    """Verify side-by-side metric delta computation between two training runs."""
    out_b = temp_run_env["outputs"] / "seven_pass_borneo"
    out_b.mkdir(parents=True, exist_ok=True)

    run_a = register_completed_run(
        out_b,
        "BORNEO",
        {
            "pr_auc_xgb": 0.80,
            "pr_auc_unet": 0.78,
            "pr_auc_fused": 0.82,
            "elkan_noto_c_hat": 0.20,
            "conformal_q_hat_90": 0.60,
            "n_ranked_targets": 10,
        },
    )
    run_b = register_completed_run(
        out_b,
        "BORNEO",
        {
            "pr_auc_xgb": 0.85,
            "pr_auc_unet": 0.82,
            "pr_auc_fused": 0.89,
            "elkan_noto_c_hat": 0.26,
            "conformal_q_hat_90": 0.68,
            "n_ranked_targets": 15,
        },
    )

    comp = compare_training_runs(run_a["run_id"], run_b["run_id"])
    assert comp["target_count_delta"] == 5
    deltas = comp["metric_deltas_b_minus_a"]
    assert deltas["pr_auc_fused"] == pytest.approx(0.07, abs=1e-4)
    assert deltas["pr_auc_xgb"] == pytest.approx(0.05, abs=1e-4)
    assert deltas["elkan_noto_c_hat"] == pytest.approx(0.06, abs=1e-4)


def test_clean_old_training_data_safety(temp_run_env):
    """Verify that cleaner removes temporary test runs while safeguarding production outputs."""
    outputs = temp_run_env["outputs"]

    # Production directory that MUST NEVER be removed
    prod_dir = outputs / "seven_pass_borneo"
    prod_dir.mkdir(parents=True, exist_ok=True)
    (prod_dir / "seven_pass_summary.json").write_text("{}")

    # Temporary directory that SHOULD be removed
    temp_dir = outputs / "test_run_yilgarn"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "dummy.txt").write_text("temp data")

    # Obsolete legacy map file that SHOULD be removed
    maps_dir = outputs / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    legacy_map = maps_dir / "prospectivity_v1.tif"
    legacy_map.write_bytes(b"dummy")

    cleaned = clean_old_training_data(keep_active_only=True, dry_run=False, remove_stale_legacy=True)

    assert any("test_run_yilgarn" in p for p in cleaned)
    assert any("prospectivity_v1.tif" in p for p in cleaned)
    assert not temp_dir.exists()
    assert not legacy_map.exists()

    # Verify production directory is 100% intact
    assert prod_dir.exists()
    assert (prod_dir / "seven_pass_summary.json").exists()
