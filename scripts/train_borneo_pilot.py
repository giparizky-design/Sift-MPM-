"""
scripts/train_borneo_pilot.py
===============================
Train, evaluate, and generate GIS deliverables for the Borneo (Kelian Epithermal
Au-Ag corridor & Kalimantan mineral systems) pilot.

Pipeline:
1. Load Borneo feature dataset (193,600 cells, 9 features in EPSG:32750).
2. PU Learning: Isolation Forest step-1 anomaly scoring on background, followed
   by 10-bagging XGBoost ensemble on (P U RN).
3. 5-fold Spatial Block Cross-Validation with cluster integrity enforcement.
4. Export GIS maps (prospectivity COG, uncertainty COG) and vector drill targets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio

from src.features.feature_stack import FeatureStack, reconstruct_map
from src.labels.pu_negatives import build_pu_training_set, step1_isolation_forest, step2_select_reliable_negatives
from src.labels.spatial_splits import assign_spatial_blocks, enforce_cluster_integrity, spatial_block_kfold
from src.models.pu_pipeline import fit_pu_model, score_metrics, verify_no_raw_pixel_leakage
from src.models.xgboost_model import MPMXGBoost
from src.outputs.prospectivity_map import generate_target_polygons
from src.utils.io import save_raster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_borneo")


def main() -> None:
    data_dir = Path("data/borneo/processed")
    models_dir = Path("outputs/borneo/models")
    maps_dir = Path("outputs/borneo/maps")
    targets_dir = Path("outputs/borneo/targets")
    reports_dir = Path("outputs/borneo/reports")

    for d in (models_dir, maps_dir, targets_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print(" TRAINING & EVALUATING BORNEO PROSPECTIVITY (KELIAN CORRIDOR)    ")
    print("==================================================================")

    df = pd.read_parquet(data_dir / "dataset.parquet")
    stack: FeatureStack = joblib.load(data_dir / "feature_stack.joblib")
    profile = stack.profile
    H, W = stack.array.shape[1], stack.array.shape[2]

    feature_cols = [c for c in df.columns if c not in ("label", "row", "col")]
    logger.info("Loaded Borneo dataset: %d cells | %d features: %s", len(df), len(feature_cols), feature_cols)

    # Acceptance Criterion 1: Verify strictly no raw sensor pixel leakage
    verify_no_raw_pixel_leakage(feature_cols)

    # -------------------------------------------------------------------------
    # 1. Full PU Model Training
    # -------------------------------------------------------------------------
    pu_config = {
        "pu_learning": {
            "step1_contamination": 0.05,
            "rn_percentile_threshold": 10.0,
            "random_state": 42,
        },
        "models": {
            "baseline": {
                "n_bag": 10,
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            },
            "xgboost": {
                "n_bagging_iterations": 10,
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            },
        },
    }

    logger.info("Step 1: Training production PU XGBoost model on full Borneo dataset...")
    p_all = df[df["label"] == 1][feature_cols].to_numpy()
    u_all = df[df["label"] == -1][feature_cols].to_numpy()
    rn_empty = np.empty((0, len(feature_cols)), dtype=np.float32)

    model = fit_pu_model(p_all, rn_empty, u_all, pu_config)
    model.save(str(models_dir / "xgboost_ensemble_borneo.joblib"))
    logger.info("Saved production Borneo PU model to %s", models_dir / "xgboost_ensemble_borneo.joblib")

    # -------------------------------------------------------------------------
    # 2. Leakage-Resistant Spatial Cross-Validation & Out-of-Fold Evaluation
    # -------------------------------------------------------------------------
    logger.info("Step 2: Leakage-safe 5-fold spatial cross-validation with cluster integrity...")
    X_full = df[feature_cols].to_numpy()
    prosp_probs = model.predict_proba(X_full)
    uncertainty = model.predict_uncertainty(X_full)

    # Assign spatial blocks across the full grid and enforce deposit cluster integrity
    block_ids = assign_spatial_blocks(
        df["row"].to_numpy(), df["col"].to_numpy(), H, W, block_size_px=44
    )
    pos_mask = (df["label"] == 1).to_numpy()
    pos_raster = np.zeros((H, W), dtype=np.uint8)
    pos_raster[df.loc[pos_mask, "row"].to_numpy(), df.loc[pos_mask, "col"].to_numpy()] = 1
    block_ids = enforce_cluster_integrity(
        block_ids, df["row"].to_numpy(), df["col"].to_numpy(), pos_raster
    )

    folds = spatial_block_kfold(block_ids, n_folds=5, random_state=42)
    oof_probs = np.full(len(df), np.nan, dtype=np.float64)
    fold_records: list[dict] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        tr_df = df.iloc[tr_idx]
        va_df = df.iloc[va_idx]

        p_tr = tr_df[tr_df["label"] == 1][feature_cols].to_numpy()
        u_tr = tr_df[tr_df["label"] == -1][feature_cols].to_numpy()
        rn_tr = np.empty((0, len(feature_cols)), dtype=np.float32)

        if len(p_tr) == 0:
            logger.warning("Fold %d has no training positives; skipping.", fold_idx)
            continue

        # Subsample unlabeled background to 25,000 for efficient fold-level Isolation Forest
        if len(u_tr) > 25000:
            rng = np.random.default_rng(42 + fold_idx)
            u_tr = u_tr[rng.choice(len(u_tr), 25000, replace=False)]

        fold_model = fit_pu_model(p_tr, rn_tr, u_tr, pu_config)

        va_X = va_df[feature_cols].to_numpy()
        va_preds = fold_model.predict_proba(va_X)
        oof_probs[va_idx] = va_preds

        va_y = (va_df["label"] == 1).astype(int).to_numpy()
        fold_m = score_metrics(va_y, va_preds)
        fold_m["fold"] = fold_idx
        fold_m["n_train_pos"] = len(p_tr)
        fold_m["n_val_pos"] = int(va_y.sum())
        fold_m["n_val_cells"] = len(va_y)
        fold_records.append(fold_m)

        logger.info(
            "Fold %d: PR-AUC=%.4f (prev=%.4f, lift=%.1fx) | ROC-AUC=%.4f | Recall@FPR10=%.4f",
            fold_idx,
            fold_m["pr_auc"],
            fold_m["prevalence"],
            fold_m.get("pr_auc_lift", float("nan")),
            fold_m["roc_auc"],
            fold_m["recall_at_fpr_0.1"],
        )

    y_all = (df["label"] == 1).astype(int).to_numpy()
    eval_metrics = score_metrics(y_all, oof_probs)
    eval_metrics["fold"] = "overall_oof"
    eval_metrics["n_train_pos"] = int(y_all.sum())
    eval_metrics["n_val_pos"] = int(y_all.sum())
    eval_metrics["n_val_cells"] = len(y_all)
    fold_records.append(eval_metrics)

    logger.info(
        "Kelian Prospectivity Ranking (Leakage-Safe Spatial OOF): PR-AUC=%.4f (Prevalence=%.4f) | ROC-AUC=%.4f | Recall@FPR10=%.4f",
        eval_metrics["pr_auc"],
        eval_metrics["prevalence"],
        eval_metrics["roc_auc"],
        eval_metrics["recall_at_fpr_0.1"],
    )

    pd.DataFrame(fold_records).to_csv(reports_dir / "spatial_metrics_borneo.csv", index=False)

    # -------------------------------------------------------------------------
    # 3. Generate Maps & Target Polygons
    # -------------------------------------------------------------------------
    logger.info("Reconstructing prospectivity and uncertainty maps across 193,600 cells...")

    prosp_2d = reconstruct_map(prosp_probs, df["row"].to_numpy(), df["col"].to_numpy(), H, W, nodata=0.0)
    unc_2d = reconstruct_map(uncertainty, df["row"].to_numpy(), df["col"].to_numpy(), H, W, nodata=1.0)

    save_raster(prosp_2d, profile, str(maps_dir / "prospectivity_kelian.tif"))
    save_raster(unc_2d, profile, str(maps_dir / "uncertainty_kelian.tif"))
    logger.info("Saved prospectivity and uncertainty GeoTIFFs to %s", maps_dir)

    # Vector targets
    logger.info("Extracting exploration target polygons (score >= 0.70, unc <= 0.20, area >= 15 ha)...")
    targets_gdf = generate_target_polygons(
        prospectivity=prosp_2d,
        uncertainty=unc_2d,
        profile=profile,
        min_score=0.70,
        max_uncertainty=0.20,
        min_area_ha=15.0,
    )
    targets_gdf.to_file(targets_dir / "targets_kelian.gpkg", driver="GPKG")
    logger.info("Generated %d target polygons in %s", len(targets_gdf), targets_dir / "targets_kelian.gpkg")

    print("\n[SUCCESS] Borneo Kelian Prospectivity Modeling Pipeline Finished.")


if __name__ == "__main__":
    main()
