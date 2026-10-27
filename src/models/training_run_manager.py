"""
src/models/training_run_manager.py
==================================
Training Run Versioning, Comparison, and Archival Manager.

Prevents overwriting or fusing together distinct training runs:
  1. Archives previous training artifacts into ``outputs/run_history/<region>/run_<timestamp>``
     before any new training run overwrites the active directory.
  2. Maintains a master catalog at ``outputs/run_history/training_runs_catalog.json``
     tracking run metadata, PR-AUC metrics, discovery priors, and ore categories.
  3. Provides side-by-side run comparison (metric deltas, target list changes, score shifts).
  4. Provides safe pruning of obsolete or temporary training runs.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("training_run_manager")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
RUN_HISTORY_DIR = OUTPUTS_DIR / "run_history"
CATALOG_PATH = RUN_HISTORY_DIR / "training_runs_catalog.json"


def _get_utc_timestamp() -> str:
    """Return compact ISO-like UTC timestamp string with microsecond resolution."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def load_runs_catalog() -> dict[str, Any]:
    """Load master training runs catalog or create a fresh structure."""
    if CATALOG_PATH.exists():
        try:
            return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Error reading runs catalog, recreating: %s", e)
    return {
        "version": "1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }


def save_runs_catalog(catalog: dict[str, Any]) -> None:
    """Save master training runs catalog atomically."""
    RUN_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    catalog["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp_path = CATALOG_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    tmp_path.replace(CATALOG_PATH)


def archive_existing_run(
    out_dir: Union[Path, str],
    region_name: str,
    run_label: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Archive an existing completed training run from ``out_dir`` to ``outputs/run_history``.

    Called BEFORE a new training execution begins to ensure previous model weights,
    maps, and target site cards are preserved independently for comparison.

    Args:
        out_dir: Current active output directory (e.g. ``outputs/seven_pass_yilgarn``).
        region_name: Name of the region (e.g. ``"YILGARN"``, ``"BORNEO"``).
        run_label: Optional descriptive tag for the archived run.

    Returns:
        Archived run manifest dictionary if run was archived, or None if out_dir had no prior run.
    """
    out_p = Path(out_dir)
    if not out_p.exists():
        return None

    summary_file = out_p / "seven_pass_summary.json"
    cards_file = out_p / "targets" / "site_cards.json"
    maps_dir = out_p / "maps"
    prosp_tif = maps_dir / "calibrated_prospectivity.tif"
    prosp_vrt = maps_dir / "calibrated_prospectivity.vrt"

    # Only archive if the prior run actually produced maps and targets
    has_valid_maps = (prosp_tif.exists() and prosp_tif.stat().st_size > 0) or (prosp_vrt.exists() and prosp_vrt.stat().st_size > 0)
    if not has_valid_maps or not summary_file.exists():
        logger.info("Skipping archival for '%s': prior run directory lacks completed maps or summary.", region_name)
        return None

    timestamp = _get_utc_timestamp()
    reg_clean = region_name.lower().strip()
    run_id = f"RUN_{region_name.upper()}_{timestamp}"
    archive_dir = RUN_HISTORY_DIR / reg_clean / f"run_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Archiving prior training run for '%s' → %s", region_name, archive_dir)

    # Copy summary and holdout metrics
    summary_data = {}
    if summary_file.exists():
        try:
            summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
        except Exception:
            pass
        shutil.copy2(summary_file, archive_dir / "seven_pass_summary.json")

    holdout_file = out_p / "blind_holdout.json"
    if holdout_file.exists():
        shutil.copy2(holdout_file, archive_dir / "blind_holdout.json")

    # Copy targets directory
    if (out_p / "targets").exists():
        shutil.copytree(out_p / "targets", archive_dir / "targets", dirs_exist_ok=True)

    # Copy maps directory
    if maps_dir.exists():
        shutil.copytree(maps_dir, archive_dir / "maps", dirs_exist_ok=True)

    # Extract detected ore categories from site cards if available
    ore_categories: list[str] = []
    archived_cards_file = archive_dir / "targets" / "site_cards.json"
    if archived_cards_file.exists():
        try:
            cards = json.loads(archived_cards_file.read_text(encoding="utf-8"))
            cats = set()
            for c in cards:
                for tg in c.get("conformal_90pct_target_set", []):
                    cats.add(str(tg))
            ore_categories = sorted(cats)
        except Exception:
            pass

    manifest = {
        "run_id": run_id,
        "region": region_name.upper(),
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "label": run_label or f"Archived prior run of {region_name.upper()}",
        "relative_path": str(archive_dir.relative_to(ROOT_DIR)),
        "run_dir": str(archive_dir.resolve()),
        "is_active": False,
        "metrics": {
            "pr_auc_xgb": summary_data.get("pr_auc_xgb"),
            "pr_auc_unet": summary_data.get("pr_auc_unet"),
            "pr_auc_fused": summary_data.get("pr_auc_fused"),
            "elkan_noto_c_hat": summary_data.get("elkan_noto_c_hat"),
            "conformal_q_hat_90": summary_data.get("conformal_q_hat_90"),
        },
        "n_ranked_targets": summary_data.get("n_ranked_targets", 0),
        "ore_categories": ore_categories,
    }

    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Update master catalog
    catalog = load_runs_catalog()
    catalog["runs"].append(manifest)
    save_runs_catalog(catalog)

    return manifest


def register_completed_run(
    out_dir: Union[Path, str],
    region_name: str,
    summary: dict[str, Any],
    run_label: Optional[str] = None,
) -> dict[str, Any]:
    """Register a newly finished training run in the master catalog as active.

    Args:
        out_dir: Active output directory.
        region_name: Region name.
        summary: Output summary dictionary from ``run_region_seven_pass``.
        run_label: Optional descriptive label.

    Returns:
        Run manifest dictionary.
    """
    out_p = Path(out_dir)
    timestamp = _get_utc_timestamp()
    run_id = f"RUN_{region_name.upper()}_{timestamp}_ACTIVE"

    ore_categories: list[str] = []
    cards_file = out_p / "targets" / "site_cards.json"
    if cards_file.exists():
        try:
            cards = json.loads(cards_file.read_text(encoding="utf-8"))
            cats = set()
            for c in cards:
                for tg in c.get("conformal_90pct_target_set", []):
                    cats.add(str(tg))
            ore_categories = sorted(cats)
        except Exception:
            pass

    manifest = {
        "run_id": run_id,
        "region": region_name.upper(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": run_label or f"Active run {region_name.upper()}",
        "relative_path": str(out_p.relative_to(ROOT_DIR)) if out_p.is_relative_to(ROOT_DIR) else str(out_p),
        "run_dir": str(out_p.resolve()),
        "is_active": True,
        "metrics": {
            "pr_auc_xgb": summary.get("pr_auc_xgb"),
            "pr_auc_unet": summary.get("pr_auc_unet"),
            "pr_auc_fused": summary.get("pr_auc_fused"),
            "elkan_noto_c_hat": summary.get("elkan_noto_c_hat"),
            "conformal_q_hat_90": summary.get("conformal_q_hat_90"),
        },
        "n_ranked_targets": summary.get("n_ranked_targets", 0),
        "ore_categories": ore_categories,
    }

    catalog = load_runs_catalog()
    reg_up = region_name.upper()
    # Mark any previously active run for this region as inactive
    for r in catalog["runs"]:
        r_reg = (r.get("region") or "").upper()
        kal_regions = ("BORNEO", "KALIMANTAN", "EAST_KALIMANTAN", "KALTIM", "KALIMANTAN_TIMUR")
        if (r_reg == reg_up or (reg_up in kal_regions and r_reg in kal_regions)) and r.get("is_active"):
            r["is_active"] = False

    catalog["runs"].append(manifest)
    save_runs_catalog(catalog)

    logger.info("Registered active training run: %s (%s)", run_id, region_name)
    return manifest


def list_training_runs(region_name: Optional[str] = None) -> list[dict[str, Any]]:
    """Return all catalogued training runs, optionally filtered by region."""
    catalog = load_runs_catalog()
    runs = catalog.get("runs", [])
    if region_name:
        reg_up = region_name.upper()
        kal_regions = ("BORNEO", "KALIMANTAN", "EAST_KALIMANTAN", "KALTIM", "KALIMANTAN_TIMUR")
        if reg_up in kal_regions:
            runs = [r for r in runs if r.get("region") in kal_regions]
        else:
            runs = [r for r in runs if r.get("region") == reg_up]
    return sorted(runs, key=lambda r: r.get("created_at") or r.get("archived_at") or "", reverse=True)


def compare_training_runs(run_id_a: str, run_id_b: str) -> dict[str, Any]:
    """Compare two training runs side-by-side.

    Computes:
      - Metric deltas (PR-AUC XGB, UNet, Fused, c_hat, q_hat)
      - Target count delta
      - Ore categories comparison
    """
    catalog = load_runs_catalog()
    runs_by_id = {r["run_id"]: r for r in catalog.get("runs", [])}

    run_a = runs_by_id.get(run_id_a)
    run_b = runs_by_id.get(run_id_b)

    if not run_a or not run_b:
        raise ValueError(f"One or both run IDs not found: '{run_id_a}', '{run_id_b}'")

    m_a = run_a.get("metrics", {})
    m_b = run_b.get("metrics", {})

    deltas = {}
    for k in ("pr_auc_xgb", "pr_auc_unet", "pr_auc_fused", "elkan_noto_c_hat", "conformal_q_hat_90"):
        val_a = m_a.get(k)
        val_b = m_b.get(k)
        if val_a is not None and val_b is not None:
            deltas[k] = round(float(val_b) - float(val_a), 4)

    return {
        "run_a": run_a,
        "run_b": run_b,
        "metric_deltas_b_minus_a": deltas,
        "target_count_delta": run_b.get("n_ranked_targets", 0) - run_a.get("n_ranked_targets", 0),
        "ore_categories_a": run_a.get("ore_categories", []),
        "ore_categories_b": run_b.get("ore_categories", []),
    }


def clean_old_training_data(
    keep_active_only: bool = True,
    dry_run: bool = False,
    remove_stale_legacy: bool = True,
) -> list[str]:
    """Remove obsolete temporary test runs and stale unversioned legacy outputs.

    Safeguards:
      - NEVER deletes the live active region outputs needed by the application
        (e.g. outputs/seven_pass_yilgarn, outputs/seven_pass_borneo, outputs/seven_pass_kalimantan_50m).
      - NEVER deletes outputs/run_history.
      - Removes temporary folders such as ``outputs/test_run_yilgarn``, old v1/v2_real
        targets/maps that are superseded, and transient debug logs.

    Args:
        keep_active_only: If True, preserves active seven_pass_* directories.
        dry_run: If True, returns items that would be removed without deleting.
        remove_stale_legacy: If True, removes legacy outputs/targets and outputs/maps files.

    Returns:
        List of paths cleaned up.
    """
    removed: list[str] = []

    # 1. Temporary test run directories
    temp_dirs = [
        OUTPUTS_DIR / "test_run_yilgarn",
    ]
    for td in temp_dirs:
        if td.exists():
            removed.append(str(td))
            if not dry_run:
                shutil.rmtree(td, ignore_errors=True)

    # 2. Obsolete legacy map/target rasters and gpkgs from Oct 21
    if remove_stale_legacy:
        legacy_files = [
            OUTPUTS_DIR / "targets" / "targets_v1.gpkg",
            OUTPUTS_DIR / "targets" / "targets_v2_real.gpkg",
            OUTPUTS_DIR / "maps" / "prospectivity_v1.tif",
            OUTPUTS_DIR / "maps" / "prospectivity_v2_real.tif",
            OUTPUTS_DIR / "maps" / "shap_top5_v1.tif",
            OUTPUTS_DIR / "maps" / "shap_top5_v2_real.tif",
            OUTPUTS_DIR / "maps" / "uncertainty_v1.tif",
            OUTPUTS_DIR / "maps" / "uncertainty_v2_real.tif",
            OUTPUTS_DIR / "models" / "v1",
            OUTPUTS_DIR / "models" / "v2_real",
        ]
        for lf in legacy_files:
            if lf.exists():
                removed.append(str(lf))
                if not dry_run:
                    if lf.is_dir():
                        shutil.rmtree(lf, ignore_errors=True)
                    else:
                        lf.unlink(missing_ok=True)

    logger.info("Cleaned %d old training items (dry_run=%s)", len(removed), dry_run)
    return removed
