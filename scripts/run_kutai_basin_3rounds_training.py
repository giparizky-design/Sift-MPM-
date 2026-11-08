"""scripts/run_kutai_basin_3rounds_training.py
===========================================
Single-Domain K=3 Multi-Rank Coal Prospectivity Mapping across the entire Kutai Basin.
Spatial Resolution: 100m Ultra-HD Grid (3,778 rows x 3,333 cols = 12.59M pixels)
Domain: SIFT K=3 Coal (Bituminous / Sub-Bituminous / Lignite)
Schedule: 3 Sequential Incremental Training Rounds (Warm-Started Ensembles)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.build_kutai_basin_100m import build_kutai_basin_100m_cube
from scripts.run_seven_pass_training import run_region_seven_pass
from src.models.training_run_manager import register_completed_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kutai_basin_3rounds")


def run_kutai_basin_training(rebuild_cube: bool = False) -> dict:
    data_dir = ROOT_DIR / "data" / "kutai_basin_100m" / "processed"
    if rebuild_cube or not (data_dir / "dataset.parquet").exists():
        logger.info("Kutai Basin 100m dataset rebuild requested or missing; initiating cube builder...")
        build_kutai_basin_100m_cube()

    region_name = "KUTAI_BASIN_100M"
    out_dir = ROOT_DIR / "outputs" / "seven_pass_kutai_basin_100m"
    total_rounds = 3
    trees_per_round = 40
    bags_per_round = 3
    dae_epochs = 2
    block_size_m = 10_000.0  # 10 km spatial cross-validation blocks
    run_label = "EXP_KUTAI_BASIN_100M_K3_COAL_V1_3ROUNDS"

    logger.info("================================================================================")
    logger.info("STARTING 3-ROUND INCREMENTAL K=3 COAL TRAINING ON KUTAI BASIN (100m)")
    logger.info("Region: %s", region_name)
    logger.info("Domain: COAL ONLY (SIFT K=3 Multi-Rank Sedimentary Systems)")
    logger.info("Resolution: 100m Grid (12.59M cells)")
    logger.info("Schedule: 3 Rounds x (+%d trees, %d bags, %d DAE epochs)", trees_per_round, bags_per_round, dae_epochs)
    logger.info("Run Label: %s", run_label)
    logger.info("================================================================================")

    start_time = time.time()
    round_summaries = []

    for r in range(1, total_rounds + 1):
        is_reset = (r == 1)
        is_inc = (r > 1)
        logger.info("\n>>> [KUTAI BASIN 100m // ROUND %d/%d] Executing 7-Pass Coal Training Pipeline...", r, total_rounds)
        
        summ = run_region_seven_pass(
            region_name=region_name,
            data_dir=data_dir,
            out_dir=out_dir,
            block_size_m=block_size_m,
            l_threshold=0.50,
            min_area_ha=10.0,
            reset_from_zero=is_reset,
            train_10x=False,
            incremental=is_inc,
            n_estimators=trees_per_round,
            n_bag=bags_per_round,
            dae_epochs=dae_epochs,
            round_idx=r,
            total_rounds=total_rounds,
            domain="coal",
        )
        round_summaries.append(summ)

    total_elapsed = time.time() - start_time
    final_summary = round_summaries[-1]
    coal_metrics = final_summary.get("coal_k3_metrics", {})

    logger.info("================================================================================")
    logger.info("3-ROUND KUTAI BASIN 100m COAL TRAINING COMPLETED in %.1fs (%.1f min)", total_elapsed, total_elapsed / 60.0)
    logger.info("Final Fused Coal PR-AUC: %s", coal_metrics.get("pr_auc_coal_fused"))
    logger.info("  - Head 0 (High-CV > 6,100 kcal/kg): PR-AUC %s", coal_metrics.get("pr_auc_head_0"))
    logger.info("  - Head 1 (Med-CV 5,100-6,100 kcal/kg): PR-AUC %s", coal_metrics.get("pr_auc_head_1"))
    logger.info("  - Head 2 (Low-CV < 5,100 kcal/kg):  PR-AUC %s", coal_metrics.get("pr_auc_head_2"))
    logger.info("XGBoost PR-AUC: %s | DAE U-Net PR-AUC: %s", coal_metrics.get("pr_auc_coal_xgb"), coal_metrics.get("pr_auc_coal_unet"))
    logger.info("================================================================================")

    # Ensure run is registered in master catalog
    try:
        register_completed_run(out_dir, region_name, final_summary, run_label=run_label)
        logger.info("Registered active run in training catalog: %s", run_label)
    except Exception as e:
        logger.warning("Could not re-register completed run: %s", e)

    return final_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kutai Basin 100m 3-Round Coal Training")
    parser.add_argument("--rebuild-cube", action="store_true", help="Force rebuild of 100m feature cube and dataset.parquet")
    parser.add_argument("--reset", action="store_true", help="Alias for --rebuild-cube")
    args = parser.parse_args()
    run_kutai_basin_training(rebuild_cube=(args.rebuild_cube or args.reset))
