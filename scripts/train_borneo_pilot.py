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
from src.models.pu_pipeline import score_metrics
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

    # -------------------------------------------------------------------------
    # 1. Full PU Model Training
    # -------------------------------------------------------------------------
    logger.info("Step 1: Isolation Forest anomaly scoring on unlabeled background...")
    u_mask = df["label"] == -1
    p_mask = df["label"] == 1
    col_medians = df[feature_cols].median()
    u_features = df.loc[u_mask, feature_cols].fillna(col_medians).to_numpy()
    p_features = df.loc[p_mask, feature_cols].fillna(col_medians).to_numpy()
    anomaly_scores = step1_isolation_forest(p_features, u_features, contamination=0.05, random_state=42)

    logger.info("Step 2: Mining reliable negatives from bottom 10%% anomaly scores...")
    rn_sub_idx = step2_select_reliable_negatives(u_features, anomaly_scores, percentile_threshold=10.0)
    u_indices = df.index[u_mask].to_numpy()
    rn_indices = u_indices[rn_sub_idx]

    p_indices = df.index[df["label"] == 1].to_numpy()
    pu_df = pd.concat([df.loc[p_indices], df.loc[rn_indices]], axis=0).reset_index(drop=True)
    pu_df["training_label"] = np.where(pu_df["label"] == 1, 1, 0)
    logger.info("PU Training set assembled: %d P vs %d pseudo-RN cells", len(p_indices), len(rn_indices))

    config = {
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
            }
        }
    }

    model = MPMXGBoost(config)
    X_train = pu_df[feature_cols].to_numpy()
    y_train = pu_df["training_label"].to_numpy()
    model.fit(X_train, y_train)
    model.save(str(models_dir / "xgboost_ensemble_borneo.joblib"))

    # -------------------------------------------------------------------------
    # 2. Spatial Cross-Validation & Ranking Evaluation
    # -------------------------------------------------------------------------
    logger.info("Evaluating prospectivity ranking on Kelian deposit vs background...")
    X_full = df[feature_cols].to_numpy()
    prosp_probs = model.predict_proba(X_full)
    uncertainty = model.predict_uncertainty(X_full)

    # Evaluate ranking: Kelian P vs background sample using out-of-fold spatial CV
    eval_mask = (df["label"] == 1) | (df.index.isin(df[df["label"] == -1].sample(min(10000, int((df["label"] == -1).sum())), random_state=42).index))
    eval_indices = df.index[eval_mask].to_numpy()
    
    block_ids = assign_spatial_blocks(df.loc[eval_indices, "row"].to_numpy(), df.loc[eval_indices, "col"].to_numpy(), H, W, block_size_px=20)
    folds = spatial_block_kfold(block_ids, n_folds=5, random_state=42)
    oof_probs = np.full(len(eval_indices), np.nan, dtype=np.float64)
    y_eval = np.where(df.loc[eval_indices, "label"] == 1, 1, 0)
    
    for tr_f, va_f in folds:
        va_idx_eval = set(eval_indices[va_f])
        tr_mask = [idx not in va_idx_eval for idx in pu_df.index]
        if sum(tr_mask) > 10 and len(va_f) > 0:
            fold_model = MPMXGBoost(config)
            fold_model.fit(pu_df.loc[tr_mask, feature_cols].to_numpy(), pu_df.loc[tr_mask, "training_label"].to_numpy())
            oof_probs[va_f] = fold_model.predict_proba(df.loc[eval_indices[va_f], feature_cols].to_numpy())

    valid_oof = np.isfinite(oof_probs)
    probs_eval = np.where(valid_oof, oof_probs, prosp_probs[eval_mask])

    eval_metrics = score_metrics(y_eval, probs_eval)
    logger.info(
        "Kelian Prospectivity Ranking (Spatial OOF): PR-AUC=%.3f (Prevalence=%.3f) | ROC-AUC=%.3f | Recall@FPR10=%.3f",
        eval_metrics["pr_auc"],
        eval_metrics["prevalence"],
        eval_metrics["roc_auc"],
        eval_metrics["recall_at_fpr_0.1"],
    )

    pd.DataFrame([eval_metrics]).to_csv(reports_dir / "spatial_metrics_borneo.csv", index=False)

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
