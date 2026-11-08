"""
scripts/sagemaker_entry.py
==========================
SageMaker PyTorch Container Entrypoint for Incremental MPM Training.

Execution workflow:
  1. Resolves SageMaker channel directories (/opt/ml/input/data/data, /opt/ml/input/data/models).
  2. Sets up workspace directory structure and links pre-processed rasters & parquet datasets.
  3. Pre-populates models/ directory with incoming checkpoint weights from S3 models channel.
  4. Runs the incremental 7-pass training engine (XGBoost + DAE UNet + Conformal Bounding).
  5. Exports updated model checkpoints, history metrics, and reports to /opt/ml/model for S3 upload.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sagemaker_entry")


def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker MPM Training Launcher")
    parser.add_argument("--region", type=str, default="EAST_KALIMANTAN", help="Region name")
    parser.add_argument("--rounds", type=int, default=1, help="Number of incremental rounds to run")
    parser.add_argument("--trees-per-round", type=int, default=50, help="XGB trees added per round")
    parser.add_argument("--bags-per-round", type=int, default=3, help="Ensemble bags per round")
    parser.add_argument("--dae-epochs", type=int, default=2, help="DAE UNet fine-tuning epochs per round")
    parser.add_argument("--block-size-m", type=float, default=25000.0, help="Spatial block CV size in meters")
    parser.add_argument("--l-threshold", type=float, default=0.50, help="Prospectivity threshold")
    parser.add_argument("--min-area-ha", type=float, default=5.0, help="Minimum target cluster area in ha")

    # SageMaker standard arguments
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--data-channel", type=str, default=os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data"))
    parser.add_argument("--models-channel", type=str, default=os.environ.get("SM_CHANNEL_MODELS", "/opt/ml/input/data/models"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    return parser.parse_args()


def setup_workspace(args: argparse.Namespace) -> tuple[Path, Path]:
    """Map SageMaker channel inputs into project directory layout."""
    logger.info("Setting up SageMaker training workspace...")

    sm_data = Path(args.data_channel)
    sm_models = Path(args.models_channel)
    local_data_dir = ROOT_DIR / "data" / "east_kalimantan" / "processed"
    local_out_dir = ROOT_DIR / "outputs" / "seven_pass_east_kalimantan"
    local_models_dir = local_out_dir / "models"

    local_data_dir.mkdir(parents=True, exist_ok=True)
    local_models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate dataset
    if (sm_data / "east_kalimantan" / "processed" / "dataset.parquet").exists():
        src_data = sm_data / "east_kalimantan" / "processed"
    elif (sm_data / "dataset.parquet").exists():
        src_data = sm_data
    elif local_data_dir.exists() and (local_data_dir / "dataset.parquet").exists():
        src_data = local_data_dir
    else:
        # Search anywhere under sm_data
        matches = list(sm_data.glob("**/dataset.parquet"))
        if matches:
            src_data = matches[0].parent
        else:
            raise FileNotFoundError(f"dataset.parquet not found in {sm_data} or {local_data_dir}")

    logger.info("Using dataset from: %s", src_data)
    if src_data != local_data_dir:
        for item in src_data.iterdir():
            dest = local_data_dir / item.name
            if not dest.exists():
                try:
                    os.symlink(item, dest)
                except OSError:
                    shutil.copy2(item, dest)

    # 2. Check for incoming checkpoint weights from S3 models channel
    if sm_models.exists():
        model_files = list(sm_models.glob("**/*.pt")) + list(sm_models.glob("**/*.joblib"))
        logger.info("Found %d pre-existing model weights in %s", len(model_files), sm_models)
        for mf in model_files:
            dest = local_models_dir / mf.name
            shutil.copy2(mf, dest)
            logger.info("  Loaded checkpoint: %s -> %s", mf.name, dest)

    # 3. Check for existing history
    history_src = None
    if (sm_models / "ten_rounds_history.json").exists():
        history_src = sm_models / "ten_rounds_history.json"
    elif (local_out_dir / "ten_rounds_history.json").exists():
        history_src = local_out_dir / "ten_rounds_history.json"

    if history_src and history_src != (local_out_dir / "ten_rounds_history.json"):
        shutil.copy2(history_src, local_out_dir / "ten_rounds_history.json")
        logger.info("Loaded previous round history from %s", history_src)

    return local_data_dir, local_out_dir


def main():
    args = parse_args()
    logger.info("================================================================================")
    logger.info("  SAGEMAKER TRAINING INSTANCE: %s", args.region)
    logger.info("  Rounds to run: %d | Trees/round: %d | Bags: %d | DAE epochs: %d",
                args.rounds, args.trees_per_round, args.bags_per_round, args.dae_epochs)
    logger.info("================================================================================")

    data_dir, out_dir = setup_workspace(args)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    from scripts.run_seven_pass_training import run_region_seven_pass

    history_path = out_dir / "ten_rounds_history.json"
    history = []
    start_round = 1
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            start_round = len(history) + 1
            logger.info("Resuming from Round %d (found %d completed rounds in history)", start_round, len(history))
        except Exception as e:
            logger.warning("Could not read existing history: %s", e)

    total_rounds_target = start_round + args.rounds - 1

    for r in range(start_round, total_rounds_target + 1):
        round_start = time.time()
        logger.info(">>> [ROUND %d/%d] Starting incremental training round on AWS instance...", r, total_rounds_target)

        summary = run_region_seven_pass(
            region_name=args.region,
            data_dir=data_dir,
            out_dir=out_dir,
            block_size_m=args.block_size_m,
            l_threshold=args.l_threshold,
            min_area_ha=args.min_area_ha,
            reset_from_zero=False,
            train_10x=False,
            incremental=True,
            n_estimators=args.trees_per_round,
            n_bag=args.bags_per_round,
            dae_epochs=args.dae_epochs,
            round_idx=r,
            total_rounds=total_rounds_target,
        )

        round_elapsed = round(time.time() - round_start, 1)
        cumulative_trees = r * args.trees_per_round

        stat = {
            "round": r,
            "elapsed_seconds": round_elapsed,
            "cumulative_trees": cumulative_trees,
            "pr_auc_xgb": summary.get("pr_auc_xgb"),
            "pr_auc_unet": summary.get("pr_auc_unet"),
            "pr_auc_fused": summary.get("pr_auc_fused"),
            "pr_auc_heads": summary.get("pr_auc_heads", {}),
            "elkan_noto_c_hat": summary.get("elkan_noto_c_hat"),
            "conformal_q_hat_90": summary.get("conformal_q_hat_90"),
            "n_ranked_targets": summary.get("n_ranked_targets", 0),
            "top_target_id": summary.get("top_target_card", {}).get("target_id"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "compute_environment": "AWS_SageMaker_Instance",
        }
        history.append(stat)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

        logger.info("--- [ROUND %d COMPLETED in %.1fs] ---", r, round_elapsed)
        logger.info("    PR-AUC Fused: %s (XGB: %s, UNet: %s)",
                    summary.get("pr_auc_fused"), summary.get("pr_auc_xgb"), summary.get("pr_auc_unet"))
        logger.info("    Discovery Prior (c_hat): %s | Targets Found: %s",
                    summary.get("elkan_noto_c_hat"), summary.get("n_ranked_targets"))

    # Copy output artifacts to SageMaker model-dir for export to S3
    model_export_dir = Path(args.model_dir)
    model_export_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting trained models and reports to %s for S3 sync...", model_export_dir)

    for item in models_dir.glob("*"):
        if item.is_file():
            shutil.copy2(item, model_export_dir / item.name)
            logger.info("  Exported model: %s (%d bytes)", item.name, item.stat().st_size)

    if history_path.exists():
        shutil.copy2(history_path, model_export_dir / "ten_rounds_history.json")

    reports_dir = out_dir / "reports"
    if reports_dir.exists():
        export_reports = model_export_dir / "reports"
        export_reports.mkdir(exist_ok=True)
        for r_file in reports_dir.glob("*"):
            if r_file.is_file():
                shutil.copy2(r_file, export_reports / r_file.name)

    logger.info("All training rounds and exports finished successfully.")


if __name__ == "__main__":
    main()
