"""Unit tests for leakage-resistant evaluation metrics in MPM.

Geoscientific context:
    In Mineral Prospectivity Mapping, economic deposits are rare (<0.5% of pixels).
    Standard evaluation metrics like ROC-AUC can be dangerously misleading due to
    the massive true-negative pool. Precision-Recall AUC (PR-AUC), Recall@FPR,
    and ranking metrics (NDCG@K) directly reflect exploration utility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compare_models,
    ndcg_at_k,
    pr_auc,
    recall_at_fpr,
    roc_auc,
)


def test_pr_auc_perfect_vs_random():
    """Verify PR-AUC behaviour on perfect vs random rankings."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_perfect = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    y_worst = np.array([0.1, 0.1, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6])

    perfect_score = pr_auc(y_true, y_perfect)
    worst_score = pr_auc(y_true, y_worst)

    assert perfect_score == 1.0
    assert worst_score < 0.25
    assert perfect_score > worst_score


def test_roc_auc_optimism():
    """Check that ROC-AUC yields higher apparent performance than PR-AUC on imbalanced labels."""
    # 1 positive out of 100 samples (1% prevalence)
    y_true = np.zeros(100)
    y_true[0] = 1
    # Model assigns positive sample rank 5 out of 100
    y_score = np.linspace(0, 1, 100)
    y_score[0] = 0.95

    roc_val = roc_auc(y_true, y_score)
    pr_val = pr_auc(y_true, y_score)

    # ROC-AUC will be ~0.95, while PR-AUC is much lower due to false positives ranked above it
    assert roc_val > 0.90
    assert pr_val < roc_val


def test_recall_at_fpr():
    """Verify recall achieved at a specified false positive rate threshold."""
    # 4 positives, 6 negatives
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    # 3 positives score above all negatives, 1 positive scores low
    y_score = np.array([0.9, 0.85, 0.8, 0.2, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1])

    # At FPR = 0.0, we capture 3 out of 4 positives = 0.75
    rec_fpr0 = recall_at_fpr(y_true, y_score, target_fpr=0.0)
    assert rec_fpr0 == 0.75

    # At FPR = 1.0, we capture all positives = 1.0
    rec_fpr1 = recall_at_fpr(y_true, y_score, target_fpr=1.0)
    assert rec_fpr1 == 1.0


def test_ndcg_at_k():
    """Verify NDCG@k ranking metric."""
    y_true = np.array([1, 0, 0, 0, 0])
    y_best = np.array([0.9, 0.4, 0.3, 0.2, 0.1])
    y_suboptimal = np.array([0.1, 0.4, 0.3, 0.2, 0.9])

    score_best = ndcg_at_k(y_true, y_best, k=3)
    score_sub = ndcg_at_k(y_true, y_suboptimal, k=3)

    assert score_best == 1.0
    assert score_sub == 0.0  # target was placed outside top-3


def test_compare_models_adoption_rule():
    """Verify U-Net adoption rule: requires both >5% PR-AUC gain AND statistical significance."""
    # Case 1: Substantial improvement (>0.05) and significant across 5 folds
    xgb_df = pd.DataFrame({"pr_auc": [0.20, 0.22, 0.21, 0.19, 0.23]})
    unet_df_win = pd.DataFrame({"pr_auc": [0.30, 0.32, 0.31, 0.29, 0.33]})  # +0.10 gain

    res_win = compare_models(xgb_df, unet_df_win)
    assert res_win["adopt_unet"] is True
    assert res_win["improvement"] > 0.05
    assert res_win["p_value"] < 0.05

    # Case 2: Insufficient improvement (<0.05)
    unet_df_marginal = pd.DataFrame({"pr_auc": [0.22, 0.24, 0.23, 0.21, 0.25]})  # +0.02 gain
    res_marginal = compare_models(xgb_df, unet_df_marginal)
    assert res_marginal["adopt_unet"] is False

    # Case 3: Fold mismatch raises ValueError
    with pytest.raises(ValueError, match="Fold count mismatch"):
        compare_models(xgb_df, unet_df_win.iloc[:3])
