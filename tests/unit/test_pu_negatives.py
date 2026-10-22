"""Unit tests for Positive-Unlabeled (PU) learning reliable negative selection.

Geoscientific context:
    In mineral prospectivity mapping, exploration records are heavily biased toward
    known deposits. An unlabeled pixel (U) is merely unexplored or undrilled, not
    barren. Two-step PU learning identifies reliable negatives (RN) by filtering
    out pixels with deposit-like anomalous geophysical signatures, leaving only
    verifiable background for model training.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.labels.pu_negatives import (
    build_pu_training_set,
    pu_bagging_wrapper,
    step1_isolation_forest,
    step2_select_reliable_negatives,
)


def test_step1_isolation_forest_scoring():
    """Verify IsolationForest returns decision scores for unlabeled cells."""
    rng = np.random.default_rng(42)
    n_features = 8
    X_pos = rng.normal(loc=5.0, scale=1.0, size=(20, n_features))
    X_unlabeled = rng.normal(loc=0.0, scale=1.0, size=(200, n_features))

    scores = step1_isolation_forest(
        X_positive=X_pos,
        X_unlabeled=X_unlabeled,
        contamination=0.05,
        random_state=42,
    )

    assert len(scores) == len(X_unlabeled)
    assert scores.ndim == 1
    assert not np.isnan(scores).any()


def test_step2_select_reliable_negatives():
    """Verify that Step 2 selects the lowest score percentile as reliable negatives."""
    rng = np.random.default_rng(42)
    n_u = 100
    X_unlabeled = rng.normal(size=(n_u, 4))
    # Synthetic scores from -1.0 (anomalous) to +1.0 (normal)
    scores = np.linspace(-1.0, 1.0, n_u)

    rn_mask = step2_select_reliable_negatives(
        X_unlabeled, scores, percentile_threshold=20.0
    )

    assert len(rn_mask) == n_u
    assert rn_mask.dtype == bool
    # 20th percentile of 100 elements should flag ~20 elements
    selected_count = np.sum(rn_mask)
    assert 19 <= selected_count <= 21


def test_build_pu_training_set():
    """Verify assembly of positive + mined reliable negative training set."""
    rng = np.random.default_rng(42)
    n_features = 4
    n_p = 10
    n_u = 100

    X_p = rng.normal(loc=2.0, scale=0.5, size=(n_p, n_features))
    X_u = rng.normal(loc=0.0, scale=0.5, size=(n_u, n_features))
    y_labels = np.array([1] * n_p + [-1] * n_u)

    config = {
        "pu_learning": {
            "step1_contamination": 0.05,
            "rn_percentile_threshold": 15.0,
            "random_state": 42,
        }
    }

    X_train, y_train = build_pu_training_set(X_p, X_u, y_labels, config)

    # Output labels should be exclusively binary 1 (P) and 0 (RN)
    assert set(np.unique(y_train)) == {0, 1}
    assert (y_train == 1).sum() == n_p
    assert (y_train == 0).sum() > 0
    assert X_train.shape[1] == n_features
    assert len(X_train) == len(y_train)


def test_pu_bagging_wrapper():
    """Verify PU bagging ensemble trains multiple base estimators."""
    rng = np.random.default_rng(42)
    n_p = 15
    n_u = 85
    X_p = rng.normal(loc=2.0, scale=1.0, size=(n_p, 3))
    X_u = rng.normal(loc=-1.0, scale=1.0, size=(n_u, 3))

    X = np.vstack([X_p, X_u])
    # 1 for positive, -1 for unlabeled
    y = np.array([1] * n_p + [-1] * n_u)

    base_clf = LogisticRegression(solver="lbfgs")
    n_models = 5

    models = pu_bagging_wrapper(
        base_model=base_clf,
        X=X,
        y=y,
        n_estimators=n_models,
        subsample_negative_ratio=1.0,
        random_state=42,
    )

    assert len(models) == n_models
    for m in models:
        # Each fitted model can predict probability for test inputs
        probs = m.predict_proba(X[:5])
        assert probs.shape == (5, 2)
