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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb

from src.evaluation.metrics import ndcg_at_k, pr_auc, recall_at_fpr, roc_auc
from src.labels.pu_negatives import (
    compute_asymmetric_head_weights,
    step1_isolation_forest,
    step2_select_reliable_negatives,
)
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
    "sentinel_b",
    "s2_b",
    "b11",
    "b13",
    "landsat_b",
    "emit_raw",
    "prisma_raw",
    "enmap_raw",
    "dn_",
    "radiance_",
    "reflectance_band",
    "kelian",
    "georima",
    "mrds",
    "minedex",
    "mine_dist",
    "deposit_dist",
    "regional_fe",
    "regional_coal",
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


# ---------------------------------------------------------------------------
# Borneo K=4 Multi-Mineral Engine (Passes 3, 4 & 5)
# ---------------------------------------------------------------------------

K4_HEAD_NAMES: list[str] = [
    "porphyry_cu",
    "skarn_fe_cu",
    "epithermal_au",
    "laterite_fe_ni",
]


def compute_pa_intersection_weights(
    X: np.ndarray,
    y_k: np.ndarray,
    n_thresholds: int = 100,
) -> np.ndarray:
    """Compute Yousefi & Carranza (2015) Prediction-Area (P-A) intersection weights.

    For each feature $j \\in \\{1, \\dots, D\\}$, evaluates thresholds $t$ to find
    the intersection point $t^*$ where:
    $$P_r(t^*) = 1 - Q_a(t^*)$$
    (i.e. $P_r(t^*) + Q_a(t^*) = 1$), where $P_r(t)$ is the proportion of positive
    deposits enclosed, and $Q_a(t)$ is the proportion of total study area occupied.

    Evaluates both feature directions ($x \\ge t$ and $x \\le t$) to extract the
    maximum contrast evidential weight:
    $$W_E^{(j,k)} = \\max\\left(0,\\; \\ln\\left(\\frac{P_r^*}{Q_a^*}\\right)\\right)$$
    and returns normalized feature weights $\\mathbf{w}_E^{(k)}$ summing to 1.0.

    Args:
        X: Feature matrix of shape (N, D).
        y_k: Binary target vector of shape (N,) with positives marked as 1.
        n_thresholds: Number of candidate thresholds to test (default 100).

    Returns:
        1-D array of shape (D,) with normalized feature weights summing to 1.0.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y_k, dtype=np.int32).ravel()
    pos_mask = (y_arr == 1)
    n_pos = int(np.sum(pos_mask))
    N, D = X_arr.shape

    if n_pos == 0 or n_pos == N or D == 0:
        return np.ones(D, dtype=np.float64) / max(D, 1)

    raw_weights = np.zeros(D, dtype=np.float64)

    for j in range(D):
        feat = X_arr[:, j]
        valid = np.isfinite(feat)
        pos_valid = pos_mask & valid
        if not np.any(pos_valid):
            continue

        feat_pos = feat[pos_valid]
        feat_all = feat[valid]
        n_all_valid = len(feat_all)
        n_pos_valid = len(feat_pos)
        if n_pos_valid == 0 or n_all_valid == 0:
            continue

        best_w = 0.0
        # Evaluate both directions: greater and less
        for direction in ("greater", "less"):
            t_candidates = np.linspace(np.min(feat_all), np.max(feat_all), n_thresholds)
            if direction == "greater":
                pr = np.array([np.sum(feat_pos >= t) / n_pos_valid for t in t_candidates])
                qa = np.array([np.sum(feat_all >= t) / n_all_valid for t in t_candidates])
            else:
                pr = np.array([np.sum(feat_pos <= t) / n_pos_valid for t in t_candidates])
                qa = np.array([np.sum(feat_all <= t) / n_all_valid for t in t_candidates])

            # Condition: P_r(t) = 1 - Q_a(t) <=> diff = P_r(t) + Q_a(t) - 1.0 = 0
            diff = pr + qa - 1.0
            crossing_indices = np.where(np.diff(np.sign(diff)) != 0)[0]

            if len(crossing_indices) > 0:
                idx = crossing_indices[0]
                d0, d1 = diff[idx], diff[idx + 1]
                denom = d0 - d1
                alpha = d0 / denom if np.abs(denom) > 1e-12 else 0.5
                pr_star = (1.0 - alpha) * pr[idx] + alpha * pr[idx + 1]
                qa_star = (1.0 - alpha) * qa[idx] + alpha * qa[idx + 1]
            else:
                idx = int(np.argmin(np.abs(diff)))
                pr_star = pr[idx]
                qa_star = qa[idx]

            if qa_star > 1e-6 and pr_star > 1e-6:
                ratio = pr_star / qa_star
                w = max(0.0, float(np.log(ratio)))
                if w > best_w:
                    best_w = w

        raw_weights[j] = best_w

    total_w = float(np.sum(raw_weights))
    if total_w > 0.0:
        return raw_weights / total_w
    return np.ones(D, dtype=np.float64) / max(D, 1)


class _BaggedXGBoostHead:
    """Specialized bagged XGBoost estimator for a single deposit head.

    Supports tree_method='hist', max_delta_step=1.0, bootstrap bagging for
    epistemic standard deviation estimation, and Yousefi & Carranza P-A feature weights.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_child_weight: float = 1.0,
        max_delta_step: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        tree_method: str = "hist",
        n_bag: int = 5,
        feature_weights: Optional[np.ndarray] = None,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        self.n_bag = max(1, int(n_bag))
        self.feature_weights = (
            np.asarray(feature_weights, dtype=np.float64)
            if feature_weights is not None
            else None
        )
        self.random_state = int(random_state)
        self.models_: list[xgb.XGBClassifier] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "_BaggedXGBoostHead":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32).ravel()
        n_samples = len(y_arr)

        w_arr = (
            np.asarray(sample_weight, dtype=np.float32).ravel()
            if sample_weight is not None
            else None
        )

        rng = np.random.default_rng(self.random_state)
        self.models_ = []

        if np.sum(y_arr == 1) == 0:
            self.is_constant_zero_ = True
            return self

        self.is_constant_zero_ = False
        for b in range(self.n_bag):
            if self.n_bag == 1:
                idx = np.arange(n_samples)
            else:
                idx = rng.choice(n_samples, size=int(0.80 * n_samples), replace=True)

            pos_bag = (y_arr[idx] == 1)
            neg_bag = (y_arr[idx] == 0)

            # Ensure positive occurrences are not omitted in bootstrap subsampling
            if not np.any(pos_bag) and np.any(y_arr == 1):
                idx = np.unique(np.concatenate([idx, np.where(y_arr == 1)[0]]))
                pos_bag = (y_arr[idx] == 1)
                neg_bag = (y_arr[idx] == 0)

            sw_bag = w_arr[idx] if w_arr is not None else None

            # Dynamic class-balance scaling: balance weighted negative-to-positive ratio
            if sw_bag is not None:
                w_pos_sum = float(np.sum(sw_bag[pos_bag]))
                w_neg_sum = float(np.sum(sw_bag[neg_bag]))
                spw = max(1.0, w_neg_sum / max(w_pos_sum, 1e-6))
            else:
                spw = max(1.0, float(np.sum(neg_bag)) / max(float(np.sum(pos_bag)), 1.0))

            clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight,
                max_delta_step=self.max_delta_step,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                tree_method=self.tree_method,
                eval_metric="aucpr",
                scale_pos_weight=spw,
                random_state=self.random_state + b,
                feature_weights=self.feature_weights,
            )

            clf.fit(X_arr[idx], y_arr[idx], sample_weight=sw_bag)
            self.models_.append(clf)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "is_constant_zero_", False):
            return np.zeros(len(X), dtype=np.float32)
        if not self.models_:
            raise RuntimeError("Head is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        preds = np.stack([m.predict_proba(X_arr)[:, 1] for m in self.models_], axis=0)
        return np.mean(preds, axis=0)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "is_constant_zero_", False):
            return np.zeros(len(X), dtype=np.float64)
        if not self.models_:
            raise RuntimeError("Head is not fitted yet.")
        if len(self.models_) <= 1:
            return np.zeros(len(X), dtype=np.float64)
        X_arr = np.asarray(X, dtype=np.float32)
        preds = np.stack([m.predict_proba(X_arr)[:, 1] for m in self.models_], axis=0)
        return np.std(preds, axis=0)

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        return self.predict_std(X) ** 2

    def compute_shap(self, X: np.ndarray) -> np.ndarray:
        """Compute TreeSHAP values using the first ensemble member as representative.

        Args:
            X: Input feature array of shape (N, D) or (D,).

        Returns:
            SHAP value array of shape (N, D) or (D,).
        """
        if getattr(self, "is_constant_zero_", False):
            X_arr = np.asarray(X, dtype=np.float32)
            return np.zeros_like(X_arr, dtype=np.float64)
        if not self.models_:
            raise RuntimeError("Head is not fitted yet.")
        import shap

        X_arr = np.asarray(X, dtype=np.float32)
        single = (X_arr.ndim == 1)
        if single:
            X_arr = X_arr[np.newaxis, :]

        rep_clf = self.models_[0]
        explainer = shap.TreeExplainer(rep_clf)
        sv = explainer.shap_values(X_arr)
        if isinstance(sv, list):
            sv = sv[1]
        sv_arr = np.asarray(sv, dtype=np.float64)
        if single and sv_arr.ndim == 2:
            return sv_arr[0]
        return sv_arr


class TwoStageK4Ensemble:
    """Two-Stage Stacking Multi-Mineral Deposit-Mechanics Engine (Passes 3, 4 & 5).

    Architecture:
      - Head 1 (Skarn Fe-Cu): Independent tree ensemble with 1D sample weights w_1
        and P-A evidential feature weights W_E^{(1)}.
      - Head 2 (Epithermal Au-Ag): Independent tree ensemble with 1D sample weights w_2
        and P-A evidential feature weights W_E^{(2)}.
      - Head 3 (Laterite Fe-Ni): Independent tree ensemble with 1D sample weights w_3
        and P-A evidential feature weights W_E^{(3)}.
      - Head 0 (Porphyry Cu-Au Rescue): Telescoping tree ensemble ingesting out-of-fold
        calibrated probabilities [oof_prob_skarn, oof_prob_epithermal] appended to X,
        fitted with shallow regularization (max_depth=3, min_child_weight=10) to prevent
        overfitting on small positive counts (N ~ 35-45).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        n_estimators: int = 100,
        n_bag: int = 5,
        random_state: int = 42,
    ) -> None:
        self.config = config or {}
        self.n_estimators = n_estimators
        self.n_bag = n_bag
        self.random_state = random_state

        self.head_0_: Optional[_BaggedXGBoostHead] = None
        self.head_1_: Optional[_BaggedXGBoostHead] = None
        self.head_2_: Optional[_BaggedXGBoostHead] = None
        self.head_3_: Optional[_BaggedXGBoostHead] = None

        self.pa_weights_: dict[int, np.ndarray] = {}
        self.oof_prob_skarn_: Optional[np.ndarray] = None
        self.oof_prob_epithermal_: Optional[np.ndarray] = None

    def _extract_head_labels_and_weights(
        self,
        y_k4: Union[np.ndarray, dict, list],
        sample_weights_k4: Optional[Union[np.ndarray, dict, list]],
        e_logistics: Optional[np.ndarray],
        exclusion_masks: Optional[np.ndarray],
        N: int,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Normalize y and sample weights into standard 1D binary vectors for all 4 heads."""
        targets: dict[int, np.ndarray] = {}
        weights: dict[int, np.ndarray] = {}

        for k in range(4):
            # Target extraction
            if isinstance(y_k4, dict):
                k_name = K4_HEAD_NAMES[k]
                raw_y = y_k4.get(k, y_k4.get(k_name, np.zeros(N, dtype=np.int32)))
            elif isinstance(y_k4, (list, tuple)):
                raw_y = y_k4[k]
            else:
                raw_y = np.asarray(y_k4)[:, k]
            raw_y = np.asarray(raw_y, dtype=np.int32).ravel()
            # Convert {1, -1} or {1, 0} to binary {1, 0}
            bin_y = (raw_y == 1).astype(np.int32)
            targets[k] = bin_y

            # Weight extraction
            if sample_weights_k4 is not None:
                if isinstance(sample_weights_k4, dict):
                    k_name = K4_HEAD_NAMES[k]
                    w_k = sample_weights_k4.get(k, sample_weights_k4.get(k_name, np.ones(N)))
                elif isinstance(sample_weights_k4, (list, tuple)):
                    w_k = sample_weights_k4[k]
                else:
                    w_k = np.asarray(sample_weights_k4)[:, k]
                weights[k] = np.asarray(w_k, dtype=np.float64).ravel()
            elif e_logistics is not None and exclusion_masks is not None:
                # Use Task 1.3 compute_asymmetric_head_weights
                # labels for PU: 1 for positives, -1 for unlabeled
                pu_labels = np.where(bin_y == 1, 1, -1)
                m_k = (
                    exclusion_masks[:, k]
                    if exclusion_masks.ndim == 2
                    else exclusion_masks
                )
                weights[k] = compute_asymmetric_head_weights(
                    labels_k=pu_labels,
                    e_logistics=e_logistics,
                    exclusion_mask_k=m_k,
                )
            else:
                weights[k] = np.ones(N, dtype=np.float64)

        return targets, weights

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_k4: Union[np.ndarray, dict, list],
        sample_weights_k4: Optional[Union[np.ndarray, dict, list]] = None,
        e_logistics: Optional[np.ndarray] = None,
        exclusion_masks: Optional[np.ndarray] = None,
        spatial_block_ids: Optional[np.ndarray] = None,
        n_cv_folds: int = 5,
    ) -> "TwoStageK4Ensemble":
        """Fit all 4 deposit heads with P-A feature weighting and two-stage stacking.

        Args:
            X: Feature matrix of shape (N, D).
            y_k4: Targets for all 4 heads (shape (N, 4) or dict/list).
            sample_weights_k4: Optional 1D sample weights for each head.
            e_logistics: Optional observation propensity e(s) (shape (N,)).
            exclusion_masks: Optional buffer exclusion mask M (shape (N, 4)).
            spatial_block_ids: Optional spatial block IDs for out-of-fold CV.
            n_cv_folds: Number of folds for out-of-fold probability stacking.

        Returns:
            Fitted TwoStageK4Ensemble instance.
        """
        if isinstance(X, pd.DataFrame):
            verify_no_raw_pixel_leakage(list(X.columns))
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        N, D = X_arr.shape
        targets, weights = self._extract_head_labels_and_weights(
            y_k4=y_k4,
            sample_weights_k4=sample_weights_k4,
            e_logistics=e_logistics,
            exclusion_masks=exclusion_masks,
            N=N,
        )

        # ------------------------------------------------------------------
        # Step 1: P-A Feature Weights for Stage 1 (Heads 1, 2, 3)
        # ------------------------------------------------------------------
        self.pa_weights_[1] = compute_pa_intersection_weights(X_arr, targets[1])
        self.pa_weights_[2] = compute_pa_intersection_weights(X_arr, targets[2])
        self.pa_weights_[3] = compute_pa_intersection_weights(X_arr, targets[3])

        # ------------------------------------------------------------------
        # Step 2: Out-of-Fold Calibrated Probabilities for Skarn and Epithermal
        # ------------------------------------------------------------------
        oof_skarn = np.zeros(N, dtype=np.float32)
        oof_epithermal = np.zeros(N, dtype=np.float32)

        from sklearn.model_selection import GroupKFold, KFold
        if spatial_block_ids is not None and len(np.unique(spatial_block_ids)) >= 2:
            n_splits = min(n_cv_folds, len(np.unique(spatial_block_ids)))
            splitter = GroupKFold(n_splits=n_splits).split(X_arr, groups=spatial_block_ids)
        else:
            n_splits = max(2, min(n_cv_folds, max(2, N // 10)))
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state).split(X_arr)

        for tr_idx, val_idx in splitter:
            # Fit fold Skarn head
            h1_fold = _BaggedXGBoostHead(
                n_estimators=max(20, self.n_estimators // 2),
                max_depth=5,
                max_delta_step=1.0,
                tree_method="hist",
                n_bag=1,
                feature_weights=self.pa_weights_[1],
                random_state=self.random_state,
            )
            h1_fold.fit(X_arr[tr_idx], targets[1][tr_idx], sample_weight=weights[1][tr_idx])
            oof_skarn[val_idx] = h1_fold.predict_proba(X_arr[val_idx])

            # Fit fold Epithermal head
            h2_fold = _BaggedXGBoostHead(
                n_estimators=max(20, self.n_estimators // 2),
                max_depth=5,
                max_delta_step=1.0,
                tree_method="hist",
                n_bag=1,
                feature_weights=self.pa_weights_[2],
                random_state=self.random_state,
            )
            h2_fold.fit(X_arr[tr_idx], targets[2][tr_idx], sample_weight=weights[2][tr_idx])
            oof_epithermal[val_idx] = h2_fold.predict_proba(X_arr[val_idx])

        self.oof_prob_skarn_ = oof_skarn
        self.oof_prob_epithermal_ = oof_epithermal

        # ------------------------------------------------------------------
        # Step 3: Fit Final Stage 1 Heads (Heads 1, 2, 3)
        # ------------------------------------------------------------------
        self.head_1_ = _BaggedXGBoostHead(
            n_estimators=self.n_estimators,
            max_depth=6,
            max_delta_step=1.0,
            tree_method="hist",
            n_bag=self.n_bag,
            feature_weights=self.pa_weights_[1],
            random_state=self.random_state + 10,
        )
        self.head_1_.fit(X_arr, targets[1], sample_weight=weights[1])

        self.head_2_ = _BaggedXGBoostHead(
            n_estimators=self.n_estimators,
            max_depth=6,
            max_delta_step=1.0,
            tree_method="hist",
            n_bag=self.n_bag,
            feature_weights=self.pa_weights_[2],
            random_state=self.random_state + 20,
        )
        self.head_2_.fit(X_arr, targets[2], sample_weight=weights[2])

        self.head_3_ = _BaggedXGBoostHead(
            n_estimators=self.n_estimators,
            max_depth=6,
            max_delta_step=1.0,
            tree_method="hist",
            n_bag=self.n_bag,
            feature_weights=self.pa_weights_[3],
            random_state=self.random_state + 30,
        )
        self.head_3_.fit(X_arr, targets[3], sample_weight=weights[3])

        # ------------------------------------------------------------------
        # Step 4: Rescue Head 0 (Porphyry Cu-Au) with Two-Stage Telescoping
        # ------------------------------------------------------------------
        X_head0 = np.column_stack([X_arr, oof_skarn, oof_epithermal])
        self.pa_weights_[0] = compute_pa_intersection_weights(X_head0, targets[0])

        self.head_0_ = _BaggedXGBoostHead(
            n_estimators=self.n_estimators,
            max_depth=3,  # shallow regularization
            min_child_weight=10.0,  # robust leaf support
            max_delta_step=1.0,
            tree_method="hist",
            n_bag=self.n_bag,
            feature_weights=self.pa_weights_[0],
            random_state=self.random_state + 40,
        )
        self.head_0_.fit(X_head0, targets[0], sample_weight=weights[0])

        logger.info(
            "TwoStageK4Ensemble fitted successfully: 4 independent heads with "
            "P-A weights and two-stage stacking for Head 0."
        )
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities for all 4 heads: [p_0, p_1, p_2, p_3].

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            2-D float64 array of shape (N, 4) with probabilities in [0, 1].
        """
        if self.head_0_ is None or self.head_1_ is None:
            raise RuntimeError("TwoStageK4Ensemble must be fitted before predict_proba().")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        p1 = self.head_1_.predict_proba(X_arr)
        p2 = self.head_2_.predict_proba(X_arr)
        p3 = self.head_3_.predict_proba(X_arr)

        X_head0 = np.column_stack([X_arr, p1, p2])
        p0 = self.head_0_.predict_proba(X_head0)

        return np.column_stack([p0, p1, p2, p3])

    def predict_std(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict inter-model epistemic standard deviation for all 4 heads: [sigma_0, sigma_1, sigma_2, sigma_3].

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            2-D float64 array of shape (N, 4) with standard deviations in [0, 0.5].
        """
        if self.head_0_ is None or self.head_1_ is None:
            raise RuntimeError("TwoStageK4Ensemble must be fitted before predict_std().")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        p1 = self.head_1_.predict_proba(X_arr)
        p2 = self.head_2_.predict_proba(X_arr)

        std1 = self.head_1_.predict_std(X_arr)
        std2 = self.head_2_.predict_std(X_arr)
        std3 = self.head_3_.predict_std(X_arr)

        X_head0 = np.column_stack([X_arr, p1, p2])
        std0 = self.head_0_.predict_std(X_head0)

        return np.column_stack([std0, std1, std2, std3])

    def predict_uncertainty(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict epistemic variance for all 4 heads."""
        return self.predict_std(X) ** 2

    def get_head(self, head: Union[int, str]) -> _BaggedXGBoostHead:
        """Retrieve a specific deposit head by index (0-3) or name.

        Args:
            head: Head index (0: porphyry, 1: skarn, 2: epithermal, 3: laterite)
                or exact head name string in K4_HEAD_NAMES.

        Returns:
            Fitted _BaggedXGBoostHead estimator.
        """
        if isinstance(head, str):
            k_map = {name: i for i, name in enumerate(K4_HEAD_NAMES)}
            if head not in k_map:
                raise ValueError(
                    f"Unknown head name '{head}'. Expected one of {K4_HEAD_NAMES}"
                )
            k = k_map[head]
        else:
            k = int(head)

        heads = {
            0: self.head_0_,
            1: self.head_1_,
            2: self.head_2_,
            3: self.head_3_,
        }
        h = heads.get(k)
        if h is None:
            raise RuntimeError(f"Head {k} ({K4_HEAD_NAMES[k]}) is not fitted yet.")
        return h

    def compute_head_shap(
        self,
        head: Union[int, str],
        X: Union[np.ndarray, pd.DataFrame],
        base_feature_names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Compute TreeSHAP values for a specific head, returning values and feature names.

        For Head 0 (porphyry_cu), automatically evaluates the out-of-fold probability
        telescoping features [oof_prob_skarn, oof_prob_epithermal] and appends them
        to the feature matrix.

        Args:
            head: Head index or name string in K4_HEAD_NAMES.
            X: Input feature array or DataFrame of shape (N, D) or (D,).
            base_feature_names: List of D base feature names.

        Returns:
            Tuple (shap_values, feature_names) where shap_values is (N, D_head) or (D_head,)
            and feature_names has length D_head.
        """
        k_idx = head if isinstance(head, int) else (
            K4_HEAD_NAMES.index(head) if head in K4_HEAD_NAMES else 0
        )
        h = self.get_head(k_idx)

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        single = (X_arr.ndim == 1)
        if single:
            X_arr = X_arr[np.newaxis, :]

        if k_idx == 0:
            p1 = self.head_1_.predict_proba(X_arr)
            p2 = self.head_2_.predict_proba(X_arr)
            X_head0 = np.column_stack([X_arr, p1, p2])
            feat_names = list(base_feature_names) + ["oof_prob_skarn", "oof_prob_epithermal"]
            sv = h.compute_shap(X_head0)
        else:
            feat_names = list(base_feature_names)
            sv = h.compute_shap(X_arr)

        if single and sv.ndim == 2:
            sv = sv[0]
        return sv, feat_names


def fit_k4_independent_xgboost_heads(
    X: Union[np.ndarray, pd.DataFrame],
    y_k4: Union[np.ndarray, dict, list],
    sample_weights_k4: Optional[Union[np.ndarray, dict, list]] = None,
    e_logistics: Optional[np.ndarray] = None,
    exclusion_masks: Optional[np.ndarray] = None,
    spatial_block_ids: Optional[np.ndarray] = None,
    config: Optional[dict] = None,
    n_estimators: int = 100,
    n_bag: int = 5,
    random_state: int = 42,
) -> TwoStageK4Ensemble:
    """Convenience functional interface to fit the TwoStageK4Ensemble."""
    ensemble = TwoStageK4Ensemble(
        config=config,
        n_estimators=n_estimators,
        n_bag=n_bag,
        random_state=random_state,
    )
    ensemble.fit(
        X=X,
        y_k4=y_k4,
        sample_weights_k4=sample_weights_k4,
        e_logistics=e_logistics,
        exclusion_masks=exclusion_masks,
        spatial_block_ids=spatial_block_ids,
    )
    return ensemble


# ---------------------------------------------------------------------------
# Pass 4 & Pass 5: Closed-Boundary Scaling & Conformal Risk Sets
# ---------------------------------------------------------------------------


def calibrate_closed_boundary_elkan_noto(
    p_raw_k: np.ndarray,
    c_hat_k: float,
    gamma_closed_k: Optional[np.ndarray] = None,
    min_c: float = 0.10,
) -> np.ndarray:
    """Pass 4: Closed-Boundary Elkan-Noto Scaling for Head k.

    $$p_{\\text{cal}, k} = \\Gamma_{\\text{closed}}^{(k)}(\\mathbf{s}) \\cdot \\min\\left(1.0,\\; \\frac{p_{\\text{raw}, k}}{\\max(\\hat{c}_k, 0.10)}\right)$$

    **Mandatory Guard (Rule 5):** Never divide by $e_{\\text{logistics}}$ in Pass 4.
    Inverse-propensity weighting was already applied in Pass 3 to deposit samples.

    Args:
        p_raw_k: Array of raw model probability scores in [0, 1].
        c_hat_k: Holdout scalar discovery prior constant $\\hat{c}_k \\in (0, 1]$.
        gamma_closed_k: Optional closed-boundary geomechanical/geological host gate
            $\\Gamma_{\\text{closed}}^{(k)}(\\mathbf{s}) \\in [0, 1]$.
        min_c: Safety floor on $\\hat{c}_k$ (default 0.10).

    Returns:
        Calibrated deposit posterior probabilities in [0, 1].
    """
    c_floor = max(float(c_hat_k), float(min_c))
    p_raw = np.asarray(p_raw_k, dtype=np.float64)
    scaled = np.minimum(1.0, np.clip(p_raw, 0.0, 1.0) / c_floor)

    if gamma_closed_k is not None:
        gamma = np.clip(np.asarray(gamma_closed_k, dtype=np.float64), 0.0, 1.0)
        p_cal = gamma * scaled
    else:
        p_cal = scaled

    return np.clip(p_cal, 0.0, 1.0)


def estimate_k4_discovery_priors(
    val_probs_k4: np.ndarray,
    val_targets_k4: np.ndarray,
    min_c: float = 0.10,
    train_block_ids: Optional[np.ndarray] = None,
    val_block_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate discovery prior constants $\\hat{c}_k = \\text{mean}(\\hat{p}_{\\text{raw}, k}[P_k])$ for all 4 heads.

    Enforces Acceptance Criterion 4 (Spatial Block Independence).

    Args:
        val_probs_k4: Validation prediction matrix of shape (N, 4).
        val_targets_k4: Validation target matrix of shape (N, 4) with 1 for positives.
        min_c: Minimum floor for each discovery prior (default 0.10).
        train_block_ids: Optional spatial block IDs used in training.
        val_block_ids: Optional spatial block IDs for validation samples.

    Returns:
        1-D array of shape (4,) containing $(\\hat{c}_0, \\hat{c}_1, \\hat{c}_2, \\hat{c}_3)$.
    """
    if train_block_ids is not None and val_block_ids is not None:
        overlap = set(np.asarray(train_block_ids).ravel()).intersection(
            set(np.asarray(val_block_ids).ravel())
        )
        if overlap:
            raise ValueError(
                f"Spatial block leakage detected in discovery prior estimation: {sorted(overlap)}"
            )

    probs = np.asarray(val_probs_k4, dtype=np.float64)
    targets = np.asarray(val_targets_k4, dtype=np.int32)
    c_priors = np.zeros(4, dtype=np.float64)

    for k in range(4):
        pos_mask = (targets[:, k] == 1)
        if np.any(pos_mask):
            scores = probs[pos_mask, k]
            scores = scores[np.isfinite(scores)]
            if len(scores) > 0:
                c_priors[k] = float(np.clip(np.mean(scores), min_c, 1.0))
            else:
                c_priors[k] = float(min_c)
        else:
            c_priors[k] = float(min_c)

    return c_priors


def compute_multi_mineral_conformal_bounds(
    p_cal_k: np.ndarray,
    sigma_ens_k: Optional[np.ndarray] = None,
    cal_pos_p_cal: Optional[np.ndarray] = None,
    cal_pos_sigma: Optional[np.ndarray] = None,
    alpha: float = 0.10,
    q_hat_k: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pass 5: Multi-Mineral Conformal Prediction Sets & 90% Confidence Lower Bounds.

    Computes nonconformity scores:
    $$V_{\\text{score}, k} = 1.0 - p_{\\text{cal}, k} + (0.5 \\cdot \\sigma_{\\text{ens}, k})$$
    the finite-sample conformal quantile $\\hat{q}_{1-\\alpha}^{(k)}$ on held-out positives,
    and the conservative 90% Conformal Lower Bound:
    $$L_{0.90}^{(k)}(\\mathbf{s}) = \\max\\left(0.0,\\; p_{\\text{cal}, k}(\\mathbf{s}) - 0.5\\,\\sigma_{\\text{ens}, k}(\\mathbf{s}) - (1.0 - \\hat{q}_{0.90}^{(k)})\\right)$$

    Args:
        p_cal_k: Calibrated probability array for head k.
        sigma_ens_k: Inter-model standard deviation $\\sigma_{\\text{ens}, k}(\\mathbf{s})$.
        cal_pos_p_cal: Optional calibrated probabilities on held-out positive validation blocks.
        cal_pos_sigma: Optional standard deviations on held-out positive validation blocks.
        alpha: Miscoverage rate (default 0.10 for 90% conformal guarantee).
        q_hat_k: Pre-calibrated conformal quantile $\\hat{q}_{0.90}^{(k)}$ if already computed.

    Returns:
        Tuple (lower_bound_k, v_scores_k, q_hat_k).
    """
    p_cal = np.asarray(p_cal_k, dtype=np.float64)
    sigma = (
        np.asarray(sigma_ens_k, dtype=np.float64)
        if sigma_ens_k is not None
        else np.zeros_like(p_cal)
    )

    v_scores = 1.0 - p_cal + (0.5 * sigma)

    if q_hat_k is None:
        if cal_pos_p_cal is not None and len(cal_pos_p_cal) > 0:
            cp_cal = np.asarray(cal_pos_p_cal, dtype=np.float64)
            cp_sig = (
                np.asarray(cal_pos_sigma, dtype=np.float64)
                if cal_pos_sigma is not None
                else np.zeros_like(cp_cal)
            )
            cal_v = 1.0 - cp_cal + (0.5 * cp_sig)
            n_pos = len(cal_v)
            level = min(1.0, np.ceil((n_pos + 1) * (1.0 - alpha)) / n_pos)
            q_hat = float(np.quantile(cal_v, level))
        else:
            q_hat = 1.0
    else:
        q_hat = float(q_hat_k)

    # 90% Conformal Lower Bound
    # When V <= q_hat, L > 0; when V > q_hat, L = 0
    l_bound = np.clip(p_cal - (0.5 * sigma) - (1.0 - q_hat), 0.0, 1.0)

    return l_bound, v_scores, q_hat


def build_conformal_target_sets(
    v_scores_k4: np.ndarray,
    q_hat_k4: Union[np.ndarray, list[float]],
    target_names: Optional[list[str]] = None,
) -> list[list[Union[str, int]]]:
    """Construct multi-mineral prediction sets $\\mathcal{C}_{0.90}(\\mathbf{s}) = \\{k : V_{i,k} \\le \\hat{q}_{0.90}^{(k)}\\}$.

    Args:
        v_scores_k4: Nonconformity score matrix of shape (N, 4).
        q_hat_k4: 1-D array or list of conformal quantiles for the 4 heads.
        target_names: Optional list of class names (defaults to K4_HEAD_NAMES).

    Returns:
        List of length N, where each element is a list of target classes included in $\\mathcal{C}_{0.90}$.
    """
    v_arr = np.asarray(v_scores_k4, dtype=np.float64)
    q_arr = np.asarray(q_hat_k4, dtype=np.float64).ravel()
    names = target_names if target_names is not None else K4_HEAD_NAMES

    target_sets: list[list[Union[str, int]]] = []
    n_samples = v_arr.shape[0]

    for i in range(n_samples):
        included = [names[k] for k in range(4) if v_arr[i, k] <= q_arr[k]]
        target_sets.append(included)

    return target_sets


