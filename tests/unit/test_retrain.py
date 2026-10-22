"""Unit tests for controlled retraining and governance pipeline in MPM.

Geoscientific context:
    Continuous learning or uncontrolled feedback loops where AI predictions
    are ingested as training ground-truth destroy the validity of prospectivity models.
    Every training label must trace back to physical drilling/assays with human
    expert sign-off.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.retrain import (
    FeedbackLoopError,
    MissingReviewerError,
    RetrainingPipeline,
)


@pytest.fixture
def temp_lake_and_model_dirs(tmp_path):
    """Fixture providing clean directories for label lake and model outputs."""
    lake_dir = tmp_path / "label_lake"
    lake_dir.mkdir(parents=True)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    return lake_dir, model_dir


def test_feedback_loop_guard_raises_error(temp_lake_and_model_dirs):
    """Verify that any label shard containing ai_prediction raises FeedbackLoopError."""
    lake_dir, model_dir = temp_lake_and_model_dirs

    # Create a contaminated shard containing ai_prediction
    contaminated_df = pd.DataFrame({
        "assay_id": ["A001", "A002", "A003"],
        "easting": [500000.0, 500100.0, 500200.0],
        "northing": [6800000.0, 6800100.0, 6800200.0],
        "label": ["P", "RN", "P"],
        "label_source": ["field_verified", "ai_prediction", "field_verified"],
        "reviewer_id": ["GEO_001", "AUTO_AI", "GEO_001"],
    })
    contaminated_df.to_parquet(lake_dir / "shard_contaminated.parquet")

    config = {"model": {"type": "xgboost"}, "cv": {"n_folds": 5}}
    pipeline = RetrainingPipeline(config, str(lake_dir), str(model_dir))

    with pytest.raises(FeedbackLoopError, match="CONTAMINATION DETECTED"):
        pipeline.load_label_lake()


def test_missing_reviewer_raises_error(temp_lake_and_model_dirs, tmp_path):
    """Verify that new drillhole data without a human reviewer is rejected."""
    lake_dir, model_dir = temp_lake_and_model_dirs

    new_csv = tmp_path / "new_drilling.csv"
    pd.DataFrame({
        "assay_id": ["A001"],
        "easting": [500000.0],
        "northing": [6800000.0],
        "label": ["P"],
        "au_ppm": [2.5],
        "depth_m": [120.0],
    }).to_csv(new_csv, index=False)

    config = {"model": {"type": "xgboost"}, "cv": {"n_folds": 5}}
    pipeline = RetrainingPipeline(config, str(lake_dir), str(model_dir))

    with pytest.raises(MissingReviewerError, match="reviewer"):
        pipeline.prepare_new_labels(str(new_csv), reviewer_id="")


def test_prepare_new_labels_success(temp_lake_and_model_dirs, tmp_path):
    """Verify that verified drillholes with valid reviewer ID are appended to the lake."""
    lake_dir, model_dir = temp_lake_and_model_dirs

    new_csv = tmp_path / "new_drilling.csv"
    pd.DataFrame({
        "assay_id": ["A001", "A002"],
        "easting": [500000.0, 500100.0],
        "northing": [6800000.0, 6800100.0],
        "label": ["P", "RN"],
        "au_ppm": [1.8, 0.01],
        "depth_m": [100.0, 80.0],
    }).to_csv(new_csv, index=False)

    config = {"model": {"type": "xgboost"}, "cv": {"n_folds": 5}}
    pipeline = RetrainingPipeline(config, str(lake_dir), str(model_dir))

    pipeline.prepare_new_labels(str(new_csv), reviewer_id="GEO_WA_4429")

    # Load and verify shard contents
    lake_df = pipeline.load_label_lake()
    assert len(lake_df) == 2
    assert (lake_df["reviewer_id"] == "GEO_WA_4429").all()
    assert (lake_df["label_source"] == "field_verified").all()
    assert (lake_df["label"].isin(["P", "RN"])).all()
