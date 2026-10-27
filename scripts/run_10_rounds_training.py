"""
scripts/run_10_rounds_training.py
==================================
Automated 10-Round Continuous / Incremental Training Engine.
Executes 10 consecutive training rounds on East Kalimantan (250m resolution),
accumulating model weights, refining neural spatial context embeddings,
and tracking metric evolution across all 10 rounds.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_seven_pass_training import run_region_seven_pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ten_round_engine")


def main() -> None:
    region_name = "EAST_KALIMANTAN"
    data_dir = ROOT_DIR / "data" / "east_kalimantan" / "processed"
    out_dir = ROOT_DIR / "outputs" / "seven_pass_east_kalimantan"
    total_rounds = 10
    trees_per_round = 50
    bags_per_round = 3
    epochs_per_round = 2

    if not (data_dir / "dataset.parquet").exists():
        logger.error("Dataset not found at %s", data_dir / "dataset.parquet")
        sys.exit(1)

    print("=" * 80)
    print(f"  STARTING 10-ROUND CONTINUOUS INCREMENTAL TRAINING: {region_name}")
    print(f"  Resolution: 250m | Target: 2,920,344 cells")
    print(f"  Schedule: {total_rounds} rounds | +{trees_per_round} trees/round | {bags_per_round} bags | {epochs_per_round} DAE epochs/round")
    print(f"  Accumulated Booster Target: {total_rounds * trees_per_round} trees by Round {total_rounds}")
    print("=" * 80)

    round_history: list[dict] = []
    start_total_time = time.time()

    for r in range(1, total_rounds + 1):
        round_start = time.time()
        print(f"\n>>> [ROUND {r}/{total_rounds}] Initiating incremental training round...")

        summary = run_region_seven_pass(
            region_name=region_name,
            data_dir=data_dir,
            out_dir=out_dir,
            block_size_m=25_000.0,
            l_threshold=0.50,
            min_area_ha=5.0,
            reset_from_zero=False,
            train_10x=False,
            incremental=True,
            n_estimators=trees_per_round,
            n_bag=bags_per_round,
            dae_epochs=epochs_per_round,
            round_idx=r,
            total_rounds=total_rounds,
        )

        round_elapsed = round(time.time() - round_start, 1)
        expected_cumulative_trees = r * trees_per_round

        round_stat = {
            "round": r,
            "elapsed_seconds": round_elapsed,
            "cumulative_trees": expected_cumulative_trees,
            "pr_auc_xgb": summary.get("pr_auc_xgb"),
            "pr_auc_unet": summary.get("pr_auc_unet"),
            "pr_auc_fused": summary.get("pr_auc_fused"),
            "pr_auc_heads": summary.get("pr_auc_heads", {}),
            "elkan_noto_c_hat": summary.get("elkan_noto_c_hat"),
            "conformal_q_hat_90": summary.get("conformal_q_hat_90"),
            "n_ranked_targets": summary.get("n_ranked_targets", 0),
            "top_target_id": summary.get("top_target_card", {}).get("target_id"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        round_history.append(round_stat)

        # Save round history incrementally to disk
        history_path = out_dir / "ten_rounds_history.json"
        history_path.write_text(json.dumps(round_history, indent=2) + "\n", encoding="utf-8")

        print(f"--- [ROUND {r}/{total_rounds} COMPLETED in {round_elapsed}s] ---")
        print(f"    Cumulative Trees: {expected_cumulative_trees}")
        print(f"    PR-AUC Fused: {summary.get('pr_auc_fused')} (XGB: {summary.get('pr_auc_xgb')}, UNet: {summary.get('pr_auc_unet')})")
        print(f"    Discovery Prior (c_hat): {summary.get('elkan_noto_c_hat')}")
        print(f"    Targets Found: {summary.get('n_ranked_targets')} | Top: {round_stat['top_target_id']}")

    total_elapsed = round(time.time() - start_total_time, 1)
    print("\n" + "=" * 80)
    print(f"  ALL 10 TRAINING ROUNDS COMPLETED SUCCESSFULLY in {total_elapsed}s ({round(total_elapsed/60, 1)} min)!")
    print("=" * 80)

    # Re-build UI bundle with latest round outputs and archived checkpoints
    print("\n[UI RE-BUILD] Re-baking dashboard bundle with final 10-round weights...")
    try:
        from scripts import build_purged_ui
        build_purged_ui.main()
        print("[UI RE-BUILD] Successfully refreshed dashboard bundle!")
    except Exception as e:
        print(f"[UI RE-BUILD] Warning during UI re-build: {e}")

    # Summary table
    print("\n" + "=" * 80)
    print("  10-ROUND TRAINING PROGRESSION SUMMARY")
    print("=" * 80)
    print(f"{'Round':<6} | {'Cumulative Trees':<16} | {'PR-AUC Fused':<12} | {'c_hat':<8} | {'Targets':<8} | {'Time (s)':<8}")
    print("-" * 70)
    for stat in round_history:
        print(
            f"{stat['round']:<6} | "
            f"{stat['cumulative_trees']:<16} | "
            f"{stat['pr_auc_fused']:<12.4f} | "
            f"{stat['elkan_noto_c_hat']:<8.4f} | "
            f"{stat['n_ranked_targets']:<8} | "
            f"{stat['elapsed_seconds']:<8.1f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
