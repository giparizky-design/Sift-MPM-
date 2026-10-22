"""Shared PU-learning fit and leakage-safe spatial CV for the MPM pipeline.

Both the production training stage and the evaluation stage call
:func:`fit_pu_model`, so a CV fold is trained by exactly the same procedure as the
final model. Every data-dependent step (isolation-forest scoring, reliable-negative
mining, XGBoost fitting) happens *inside* the training split, never on data that
belongs to the validation block.

Evaluation uses only verified labels: positives (P) versus drill-verified reliable
negatives (RN). Pseudo-negatives mined from unlabeled ground (U) are used for
training only, never for scoring, because they are chosen to look like background
and would make any model appear skilful.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import ndcg_at_k, pr_auc, recall_at_fpr, roc_auc
from src.labels.pu_negatives import step1_isolation_forest, step2_select_reliable_negatives
from src.models.xgboost_model import MPMXGBoost

logger = logging.getLogger(__name__)

LABEL_P, LABEL_RN, LABEL_U = 1, 0, -1


def pu_settings(config: dict) -> dict:
    """Extract PU hyper-parameters from the pilot config with safe defaults."""
    pu = config.get("pu_learning", {})
    return {
        "contamination": float(pu.get("step1_contamination", 0.05)),
        "percentile_threshold": float(pu.get("rn_percentile_threshold", 10.0)),
        "random_state": int(pu.get("random_state", 42)),
    }


def _median_fill(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Fill NaNs with training medians. Used ONLY for the isolation forest.

    IsolationForest cannot ingest NaN. XGBoost receives the original NaNs so its
    native missing-value handling is preserved (no imputation of model inputs).
    """
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
    """Fit the two-step PU model on the supplied training cells only.

    Step 1 scores U cells with an isolation forest fitted on U; step 2 keeps the
    most background-like fraction as pseudo reliable negatives. The XGBoost
    ensemble is then trained on P (1) versus verified RN + pseudo-RN (0),
    optionally weighted by Pass 2 inverse propensity sample weights $w_i = 1 / e(x_i)$.

    Args:
        X_p: Positive cell features, shape (n_p, f).
        X_rn: Drill-verified reliable-negative features, shape (n_rn, f).
        X_u: Unlabeled cell features, shape (n_u, f).
        config: Pilot configuration dictionary.
        w_p: Optional inverse propensity weights for positive cells, shape (n_p,).
        w_rn: Optional inverse propensity weights for verified RN cells, shape (n_rn,).
        w_u: Optional inverse propensity weights for unlabeled cells, shape (n_u,).

    Returns:
        Fitted :class:`MPMXGBoost`.

    Raises:
        ValueError: If there are no positive cells or no negative cells at all.
    """
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
    """Convenience wrapper: split a labelled frame into P / RN / U and fit."""
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
    """Leakage-safe spatial CV for the PU pipeline.

    For each fold the model (including isolation-forest mining) is fitted on the
    training blocks only. It is scored on the *verified* P and RN cells of the
    held-out blocks. Folds whose test blocks lack either class, or whose training
    blocks lack positives, are skipped and logged.

    Args:
        df: Frame with feature columns and ``label`` (1 / 0 / -1), one row per cell.
        feature_cols: Feature column names.
        folds: ``(train_idx, test_idx)`` positional index pairs from
            :func:`src.labels.spatial_splits.spatial_block_kfold`.
        config: Pilot configuration dictionary.

    Returns:
        One row per evaluated fold with counts and metrics. May be empty.
    """
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


# ---------------------------------------------------------------------------
# Acceptance Criterion 1: No Raw Pixel Leakage Gate
# ---------------------------------------------------------------------------

_FORBIDDEN_RAW_BAND_PATTERNS: tuple[str, ...] = (
    "raw_band",
    "aster_b",
    "landsat_b",
    "sentinel_b",
    "emit_raw",
    "prisma_raw",
    "enmap_raw",
    "dn_",
    "radiance_",
    "reflectance_band",
)


def verify_no_raw_pixel_leakage(feature_names: list[str]) -> bool:
    """Verify feature columns contain only Pass 1 physical/geological indices.

    Enforces **Acceptance Criterion 1 (No Raw Pixel Leakage)**: raw sensor
    reflectance/radiance bands (e.g. ``aster_b4``, ``raw_band_1``, ``dn_3``)
    must never be passed directly to tabular (`pu_pipeline.py`) or spatial
    (`unet_model.py`) learners; all inputs must be physical, structural, or
    geomechanical indices from Pass 1.

    Args:
        feature_names: List of feature channel/column names.

    Returns:
        ``True`` if all feature names pass verification.

    Raises:
        ValueError: If any feature name matches a raw sensor band pattern.
    """
    violating: list[str] = []
    for name in feature_names:
        low = name.strip().lower()
        if any(pat in low for pat in _FORBIDDEN_RAW_BAND_PATTERNS):
            violating.append(name)
    if violating:
        raise ValueError(
            f"Raw sensor pixel leakage detected in feature matrix: {violating}. "
            "Only Pass 1 physical, structural, and geomechanical indices are allowed."
        )
    return True


# ---------------------------------------------------------------------------
# Pass 3.3: Validation PR-AUC Ensembling
# ---------------------------------------------------------------------------


def compute_ensemble_weights(
    pr_auc_xgb: float,
    pr_auc_unet: float,
) -> tuple[float, float]:
    """Compute normalised ensemble weights $(w_{\\text{xgb}}, w_{\\text{unet}})$ from spatial-CV PR-AUC.

    Satisfies $w_{\\text{xgb}} + w_{\\text{unet}} = 1$ with $w \\propto \\text{PR-AUC}$.

    Args:
        pr_auc_xgb: Spatial cross-validation PR-AUC of the tabular XGBoost pass.
        pr_auc_unet: Spatial cross-validation PR-AUC of the spatial U-Net pass.

    Returns:
        Tuple ``(w_xgb, w_unet)`` summing to ``1.0``.
    """
    a_xgb = max(0.0, float(pr_auc_xgb))
    a_unet = max(0.0, float(pr_auc_unet))
    total = a_xgb + a_unet
    if total <= 0.0:
        return 0.5, 0.5
    return a_xgb / total, a_unet / total


def blend_predictions_by_pr_auc(
    s_xgb: np.ndarray,
    s_unet: np.ndarray,
    pr_auc_xgb: float,
    pr_auc_unet: float,
) -> tuple[np.ndarray, float, float]:
    """Blend tabular ($S_{\\text{xgb}}$) and spatial ($S_{\\text{unet}}$) scores weighted by spatial-CV PR-AUC.

    $$\\text{Fused Score}(x) = w_{\\text{xgb}} \\cdot S_{\\text{xgb}}(x) + w_{\\text{unet}} \\cdot S_{\\text{unet}}(x)$$

    where $w_{\\text{xgb}} + w_{\\text{unet}} = 1$.

    Args:
        s_xgb: 1-D or 2-D array of XGBoost scores $S_{\\text{xgb}}(x) \\in [0, 1]$.
        s_unet: Array of U-Net scores $S_{\\text{unet}}(x) \\in [0, 1]$, same shape as ``s_xgb``.
        pr_auc_xgb: Mean spatial-CV PR-AUC of the XGBoost model.
        pr_auc_unet: Mean spatial-CV PR-AUC of the U-Net model.

    Returns:
        Tuple ``(fused_score, w_xgb, w_unet)``.
    """
    arr_xgb = np.asarray(s_xgb, dtype=np.float64)
    arr_unet = np.asarray(s_unet, dtype=np.float64)
    if arr_xgb.shape != arr_unet.shape:
        raise ValueError(
            f"s_xgb shape {arr_xgb.shape} != s_unet shape {arr_unet.shape}"
        )
    w_xgb, w_unet = compute_ensemble_weights(pr_auc_xgb, pr_auc_unet)
    fused = np.clip(w_xgb * arr_xgb + w_unet * arr_unet, 0.0, 1.0)
    return fused, w_xgb, w_unet


# ---------------------------------------------------------------------------
# Pass 4: Elkan-Noto Posterior Calibration
# ---------------------------------------------------------------------------


def estimate_elkan_noto_c(
    val_positive_scores: np.ndarray,
    train_block_ids: Optional[np.ndarray] = None,
    val_block_ids: Optional[np.ndarray] = None,
    min_c: float = 0.05,
) -> float:
    """Estimate the Elkan & Noto (2008) label frequency constant $\\hat{c} = P(s=1 \\mid y=1)$.

    Uses the e1 estimator from Elkan & Noto (2008): the average predicted
    probability on held-out validation positives:

    $$\\hat{c} = \\frac{1}{|P_{\\text{val}}|} \\sum_{x \\in P_{\\text{val}}} g(x)$$

    To enforce **Acceptance Criterion 4 (Spatial Block Independence)**, if
    ``train_block_ids`` and ``val_block_ids`` are supplied, this function
    verifies that the validation positives reside in spatially disjoint holdout
    blocks with zero overlap with training blocks.

    Args:
        val_positive_scores: 1-D array of fused ensemble scores $g(x)$ evaluated
            strictly on held-out validation positive instances ($s = 1$).
        train_block_ids: Optional 1-D array of spatial block IDs used in training.
        val_block_ids: Optional 1-D array of spatial block IDs for the validation
            positives. Raises ``ValueError`` if any block ID overlaps with
            ``train_block_ids``.
        min_c: Safety floor on $\\hat{c}$ to avoid division by zero. Defaults to ``0.05``.

    Returns:
        Estimated label frequency constant $\\hat{c} \\in [\\text{min\\_c}, 1.0]$.
    """
    if train_block_ids is not None and val_block_ids is not None:
        overlap = set(np.asarray(train_block_ids).ravel()).intersection(
            set(np.asarray(val_block_ids).ravel())
        )
        if overlap:
            raise ValueError(
                f"Spatial block leakage detected in Elkan-Noto c_hat estimation! "
                f"Overlapping block IDs between train and calibration: {sorted(overlap)}"
            )

    scores = np.asarray(val_positive_scores, dtype=np.float64).ravel()
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        raise ValueError("val_positive_scores contains no finite positive scores.")

    c_hat = float(np.mean(scores))
    return float(np.clip(c_hat, min_c, 1.0))


def calibrate_elkan_noto_posterior(
    fused_scores: np.ndarray,
    c_hat: float,
) -> np.ndarray:
    """Convert relative ensemble scores into calibrated deposit probabilities (Elkan & Noto, 2008).

    $$P(\\text{Deposit} \\mid x) = \\min\\left(1.0,\\; \\frac{\\text{Fused Score}(x)}{\\hat{c}}\\right)$$

    Args:
        fused_scores: Array of fused ensemble scores in $[0, 1]$.
        c_hat: Exploration discovery rate / label frequency constant $\\hat{c} \\in (0, 1]$.

    Returns:
        Calibrated deposit posterior probabilities in $[0, 1]$, same shape as ``fused_scores``.
    """
    if c_hat <= 0.0 or c_hat > 1.0:
        raise ValueError(f"c_hat must be in (0, 1], got {c_hat}")
    scores = np.asarray(fused_scores, dtype=np.float64)
    calibrated = np.minimum(1.0, np.clip(scores, 0.0, 1.0) / float(c_hat))
    return np.where(np.isnan(scores), np.nan, calibrated)

