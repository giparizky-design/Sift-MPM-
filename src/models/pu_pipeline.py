"""Facade module for Positive-Unlabeled (PU) learning and multi-domain ensembles.

Re-exports 100% of symbols from modular domain submodules for full backward compatibility:
  - `src.schema.canonical`: Canonical schemas, priors, band patterns, labels.
  - `src.models.metals`: TwoStageK4Ensemble, _BaggedXGBoostHead, P-A weights, zonation.
  - `src.models.coal`: TwoStageK3CoalEnsemble, MacroBasinSequenceGate, optical WoE, coal softmax.
  - `src.models.calibration`: Elkan-Noto c_hat prior, closed-boundary scaling, monotonic logit scaling.
  - `src.models.conformal`: ConformalQuantile, weighted conformal quantiles, set gates, lower bounds.
  - `src.models.fusion`: PR-AUC ensembling and precision-weighted log-odds fusion.
  - `src.labels.pu_negatives`: OUTMOGCGate, asymmetric head weights, isolation forest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.evaluation.metrics import ndcg_at_k, pr_auc, recall_at_fpr, roc_auc
from src.labels.pu_negatives import (
    OUTMOGCGate,
    compute_asymmetric_head_weights,
    step1_isolation_forest,
    step2_select_reliable_negatives,
)
from src.models.calibration import (
    _logit,
    _sigmoid,
    calibrate_closed_boundary_elkan_noto,
    calibrate_elkan_noto_posterior,
    calibrate_monotonic_logit_scaling,
    estimate_elkan_noto_c,
    estimate_k4_discovery_priors,
)
from src.models.coal import (
    BARREN_OPTICAL_COAL_FEATURES,
    SUBCANOPY_COAL_FEATURES,
    MacroBasinSequenceGate,
    TwoStageK3CoalEnsemble,
    compute_hierarchical_coal_probabilities,
    fuse_bayesian_canopy_log_odds,
    fuse_canopy_log_odds,
)
from src.models.conformal import (
    ConformalQuantile,
    build_conformal_target_sets,
    compute_multi_mineral_conformal_bounds,
    compute_studentized_conformal_lower_bound,
    evaluate_conformal_set_gate,
    weighted_conformal_quantile,
)
from src.models.fusion import (
    blend_predictions_by_pr_auc,
    compute_ensemble_weights,
    precision_weighted_log_odds_fusion,
)
from src.models.metals import (
    ZONATION_META_FEATURE_NAMES,
    _BaggedXGBoostHead,
    TwoStageK4Ensemble,
    compute_concentric_zonation_features,
    compute_hajek_asymmetric_head_weights,
    compute_pa_intersection_weights,
    fit_k4_independent_xgboost_heads,
)
from src.models.xgboost_model import MPMXGBoost
from src.schema.canonical import (
    _FORBIDDEN_RAW_BAND_PATTERNS,
    COAL_PREVALENCE_PRIORS,
    COAL_PRIORS_K3,
    DISCOVERY_PRIORS_K4,
    EXTENDED_NON_FEATURE_COLS,
    FORBIDDEN_RAW_BAND_REGEXES,
    K3_COAL_HEAD_NAMES,
    K4_HEAD_NAMES,
    LABEL_P,
    LABEL_RN,
    LABEL_U,
    METALS_PREVALENCE_PRIORS,
    NON_FEATURE_COLS,
    NON_FEATURE_COLS_SET,
    verify_no_raw_pixel_leakage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy Single-Commodity PU Learning Routines
# ---------------------------------------------------------------------------


def pu_settings(config: dict) -> dict:
    """Extract PU hyper-parameters from pilot config with safe defaults."""
    pu = config.get("pu_learning", {})
    return {
        "contamination": float(pu.get("step1_contamination", 0.05)),
        "percentile_threshold": float(pu.get("rn_percentile_threshold", 10.0)),
        "random_state": int(pu.get("random_state", 42)),
    }


def _median_fill(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Fill NaNs with training medians for IsolationForest only."""
    filled = X.copy()
    nan_r, nan_c = np.where(np.isnan(filled))
    filled[nan_r, nan_c] = medians[nan_c]
    return filled


def fit_pu_model(
    X_p: np.ndarray,
    X_rn: np.ndarray,
    X_u: np.ndarray,
    config: dict,
    w_p: Optional[np.ndarray] = None,
    w_rn: Optional[np.ndarray] = None,
    w_u: Optional[np.ndarray] = None,
) -> MPMXGBoost:
    """Fit two-step PU model on supplied training cells only."""
    if len(X_p) == 0:
        raise ValueError("fit_pu_model: no positive cells in the training split.")

    pu = pu_settings(config)
    pseudo_rn = np.empty((0, X_p.shape[1]), dtype=X_p.dtype)
    w_pseudo = np.empty((0,), dtype=np.float64)

    if len(X_u) > 0:
        medians = np.nanmedian(X_u, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        scores = step1_isolation_forest(
            X_positive=_median_fill(X_p, medians),
            X_unlabeled=_median_fill(X_u, medians),
            contamination=pu["contamination"],
            random_state=pu["random_state"],
        )
        mask = step2_select_reliable_negatives(
            X_unlabeled=X_u,
            scores=scores,
            percentile_threshold=pu["percentile_threshold"],
        )
        pseudo_rn = X_u[mask]
        if w_u is not None:
            w_pseudo = np.asarray(w_u, dtype=np.float64)[mask]

    X_neg = np.vstack([X_rn, pseudo_rn]) if len(X_rn) else pseudo_rn
    if len(X_neg) == 0:
        raise ValueError("fit_pu_model: no negative cells (verified or mined).")

    X_train = np.vstack([X_p, X_neg])
    y_train = np.concatenate(
        [np.ones(len(X_p), dtype=np.int32), np.zeros(len(X_neg), dtype=np.int32)]
    )

    sample_weight: Optional[np.ndarray] = None
    if w_p is not None or w_rn is not None or w_u is not None:
        wp_arr = (
            np.asarray(w_p, dtype=np.float64)
            if w_p is not None
            else np.ones(len(X_p), dtype=np.float64)
        )
        wrn_arr = (
            np.asarray(w_rn, dtype=np.float64)
            if w_rn is not None
            else np.ones(len(X_rn), dtype=np.float64)
        )
        w_ps_arr = (
            w_pseudo
            if w_u is not None
            else np.ones(len(pseudo_rn), dtype=np.float64)
        )
        w_neg = np.concatenate([wrn_arr, w_ps_arr]) if len(X_rn) else w_ps_arr
        sample_weight = np.concatenate([wp_arr, w_neg])

    logger.info(
        "fit_pu_model: %d P, %d verified RN, %d pseudo-RN from %d U",
        len(X_p), len(X_rn), len(pseudo_rn), len(X_u),
    )
    model = MPMXGBoost(config)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def fit_pu_model_from_frame(df: pd.DataFrame, feature_cols: list[str], config: dict) -> MPMXGBoost:
    """Convenience wrapper: split labelled frame into P / RN / U and fit."""
    labels = df["label"].to_numpy()
    X = df[feature_cols].to_numpy()
    return fit_pu_model(
        X[labels == LABEL_P], X[labels == LABEL_RN], X[labels == LABEL_U], config
    )


def score_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """PR-AUC (with prevalence baseline), ROC-AUC, recall@FPR10 and NDCG@100."""
    prevalence = float(np.mean(y_true))
    pr = pr_auc(y_true, y_score)
    return {
        "prevalence": prevalence,
        "pr_auc": pr,
        "pr_auc_lift": pr / prevalence if prevalence > 0 else float("nan"),
        "roc_auc": roc_auc(y_true, y_score),
        "recall_at_fpr_0.1": recall_at_fpr(y_true, y_score, target_fpr=0.1),
        "ndcg_at_k_100": ndcg_at_k(y_true, y_score, k=100),
    }


def spatial_cv_pu(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: dict,
) -> pd.DataFrame:
    """Leakage-safe spatial CV for single-commodity PU pipeline."""
    labels = df["label"].to_numpy()
    X = df[feature_cols].to_numpy()
    records: list[dict] = []

    for i, (train_idx, test_idx) in enumerate(folds):
        eval_idx = test_idx[np.isin(labels[test_idx], (LABEL_P, LABEL_RN))]
        y_eval = (labels[eval_idx] == LABEL_P).astype(int)
        n_pos, n_rn = int(y_eval.sum()), int((1 - y_eval).sum())
        if n_pos == 0 or n_rn == 0:
            logger.warning(
                "Fold %d skipped: test blocks hold %d P and %d verified RN (need both).",
                i, n_pos, n_rn,
            )
            continue

        tr_labels = labels[train_idx]
        if not (tr_labels == LABEL_P).any():
            logger.warning("Fold %d skipped: no positives in training blocks.", i)
            continue

        model = fit_pu_model(
            X[train_idx][tr_labels == LABEL_P],
            X[train_idx][tr_labels == LABEL_RN],
            X[train_idx][tr_labels == LABEL_U],
            config,
        )
        metrics = score_metrics(y_eval, model.predict_proba(X[eval_idx]))
        records.append(
            {
                "fold": i,
                "n_train_pos": int((tr_labels == LABEL_P).sum()),
                "n_test_pos": n_pos,
                "n_test_rn": n_rn,
                **metrics,
            }
        )
        logger.info(
            "Fold %d | test P=%d RN=%d | PR-AUC=%.3f (prevalence %.3f)",
            i, n_pos, n_rn, metrics["pr_auc"], metrics["prevalence"],
        )

    return pd.DataFrame(records)


__all__ = [
    # Schema & Canonical constants
    "NON_FEATURE_COLS",
    "NON_FEATURE_COLS_SET",
    "EXTENDED_NON_FEATURE_COLS",
    "_FORBIDDEN_RAW_BAND_PATTERNS",
    "FORBIDDEN_RAW_BAND_REGEXES",
    "K4_HEAD_NAMES",
    "K3_COAL_HEAD_NAMES",
    "DISCOVERY_PRIORS_K4",
    "COAL_PRIORS_K3",
    "METALS_PREVALENCE_PRIORS",
    "COAL_PREVALENCE_PRIORS",
    "LABEL_P",
    "LABEL_RN",
    "LABEL_U",
    "verify_no_raw_pixel_leakage",
    # Calibration
    "estimate_elkan_noto_c",
    "calibrate_elkan_noto_posterior",
    "estimate_k4_discovery_priors",
    "calibrate_closed_boundary_elkan_noto",
    "calibrate_monotonic_logit_scaling",
    "_logit",
    "_sigmoid",
    # Conformal
    "ConformalQuantile",
    "weighted_conformal_quantile",
    "compute_studentized_conformal_lower_bound",
    "evaluate_conformal_set_gate",
    "compute_multi_mineral_conformal_bounds",
    "build_conformal_target_sets",
    # Fusion
    "compute_ensemble_weights",
    "blend_predictions_by_pr_auc",
    "precision_weighted_log_odds_fusion",
    # Metals (Domain A)
    "compute_hajek_asymmetric_head_weights",
    "compute_pa_intersection_weights",
    "compute_concentric_zonation_features",
    "ZONATION_META_FEATURE_NAMES",
    "_BaggedXGBoostHead",
    "TwoStageK4Ensemble",
    "fit_k4_independent_xgboost_heads",
    # Coal (Domain B)
    "SUBCANOPY_COAL_FEATURES",
    "BARREN_OPTICAL_COAL_FEATURES",
    "fuse_canopy_log_odds",
    "fuse_bayesian_canopy_log_odds",
    "MacroBasinSequenceGate",
    "TwoStageK3CoalEnsemble",
    "compute_hierarchical_coal_probabilities",
    # PU labels & OutMOGC
    "OUTMOGCGate",
    "compute_asymmetric_head_weights",
    "step1_isolation_forest",
    "step2_select_reliable_negatives",
    # Legacy Single-Commodity PU
    "pu_settings",
    "_median_fill",
    "fit_pu_model",
    "fit_pu_model_from_frame",
    "score_metrics",
    "spatial_cv_pu",
]
