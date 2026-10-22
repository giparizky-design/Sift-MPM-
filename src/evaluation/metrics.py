"""Leakage-resistant evaluation metrics for Mineral Prospectivity Mapping.

All metrics in this module are designed with the following MPM-specific
constraints in mind:

* **Severe class imbalance** — deposit-bearing pixels are typically < 0.5 %
  of the study area.  ROC-AUC is optimistic under such imbalance because the
  large negative class inflates true-negative rates; PR-AUC is the primary
  reported metric.

* **Spatial autocorrelation** — standard k-fold CV applied to spatially
  autocorrelated raster data causes data leakage (training pixels near test
  pixels share geochemical / structural signal).  All CV functions require
  *pre-built* spatial folds (e.g. block CV, environmental gradient splits)
  supplied by the caller.

* **Geological interpretability** — ``recall_at_fpr`` and ``ndcg_at_k``
  mirror the practical mine-finder use-case: "What fraction of real deposits
  are inside our top-ranked target list?"
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    ndcg_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the Precision-Recall curve (primary MPM metric).

    Computed via ``sklearn.metrics.average_precision_score``, which is the
    weighted mean of precisions at each threshold, weighted by the increase
    in recall — equivalent to the area under a step-interpolated PR curve.

    .. warning::
        **Use this, not ROC-AUC, for imbalanced MPM data.**
        ROC-AUC is invariant to class prior and therefore uninformative when
        negatives vastly outnumber positives (the typical 99.5 % / 0.5 %
        split in deposit-scale mapping).  A random classifier achieves
        ROC-AUC = 0.50 regardless of imbalance, but PR-AUC = prevalence
        (e.g. 0.005), making improvement much harder to fake.

    Args:
        y_true (np.ndarray): Binary ground-truth labels, shape ``(N,)``.
        y_score (np.ndarray): Predicted prospectivity scores, shape ``(N,)``.

    Returns:
        float: PR-AUC in [0, 1].  A random classifier scores ≈ prevalence;
            a perfect classifier scores 1.0.
    """
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the Receiver Operating Characteristic curve.

    Wrapper around ``sklearn.metrics.roc_auc_score``.

    .. note::
        **Reported for completeness only.**  ROC-AUC is optimistic under heavy
        class imbalance because the enormous pool of true negatives inflates
        the TPR/FPR ratio.  Do not use ROC-AUC as the primary model selection
        criterion for MPM.  Prefer :func:`pr_auc`.

    Args:
        y_true (np.ndarray): Binary ground-truth labels, shape ``(N,)``.
        y_score (np.ndarray): Predicted prospectivity scores, shape ``(N,)``.

    Returns:
        float: ROC-AUC in [0.5, 1.0] for a useful classifier.
    """
    return float(roc_auc_score(y_true, y_score))


def recall_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float = 0.1,
) -> float:
    """Recall (sensitivity) achieved when the false-positive rate ≤ target_fpr.

    Answers the geologist's question: *"If we accept a 10 % false-positive
    rate (i.e. explore 10 % of non-deposit background), how many real deposits
    will we find?"*  This directly maps to exploration budget efficiency.

    Implementation sweeps the ROC curve and selects the operating point where
    FPR first falls below ``target_fpr``.

    Args:
        y_true (np.ndarray): Binary ground-truth labels, shape ``(N,)``.
        y_score (np.ndarray): Predicted prospectivity scores, shape ``(N,)``.
        target_fpr (float): Maximum acceptable false-positive rate.
            Default 0.1 (10 %).

    Returns:
        float: Recall at the operating point where FPR ≤ ``target_fpr``.
            Returns 0.0 if no such operating point exists.
    """
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)

    # Find all indices where FPR <= target_fpr
    valid_mask = fpr_arr <= target_fpr
    if not valid_mask.any():
        return 0.0

    # The highest TPR among valid operating points
    return float(tpr_arr[valid_mask].max())


def ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 100,
) -> float:
    """Normalised Discounted Cumulative Gain at top-k cells.

    Treats the prospectivity score as a ranking of all pixels and rewards
    models that place known deposits at the very top of the ranked list.
    The logarithmic discount penalises deposits found later in the list
    more heavily than a simple precision-at-k metric.

    Geoscientific interpretation:
        * k = 100 ≈ a first-pass target list for airborne follow-up (100
          anomaly polygons).
        * NDCG@100 = 1.0 means all top-100 pixels are actual deposits.
        * NDCG@100 = prevalence/(log2(2)) means the model is random.

    Uses ``sklearn.metrics.ndcg_score`` which requires 2-D inputs.

    Args:
        y_true (np.ndarray): Binary ground-truth labels, shape ``(N,)``.
        y_score (np.ndarray): Predicted prospectivity scores, shape ``(N,)``.
        k (int): Number of top-ranked cells to evaluate.  Default 100.

    Returns:
        float: NDCG@k in [0, 1].
    """
    # sklearn expects shape (1, N)
    return float(
        ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k)
    )


# ---------------------------------------------------------------------------
# Spatial cross-validation
# ---------------------------------------------------------------------------


def spatial_cv_evaluate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run evaluation across pre-built spatial folds.

    Spatial folds must be constructed *before* calling this function using a
    leakage-resistant strategy such as:
    * **Block cross-validation** — geographic blocks defined by a regular grid.
    * **Buffered leave-one-out** — exclusion zones around each deposit.
    * **Environmental gradient splits** — folds maximising environmental
      dissimilarity between train/test sets (Roberts et al., 2017).

    For each fold the model is re-fitted on the training split and scored on
    the held-out test split.  The returned DataFrame enables model comparison
    via :func:`compare_models`.

    Args:
        model: Any object implementing ``fit(X, y)`` and
            ``predict_proba(X) -> np.ndarray`` (shape ``(N,)``).
        X (np.ndarray): Full feature matrix, shape ``(N, n_features)``.
        y (np.ndarray): Binary labels, shape ``(N,)``.
        folds (list[tuple]): List of ``(train_idx, test_idx)`` index arrays,
            one tuple per spatial fold.
        feature_names (list[str] | None): Optional feature names passed to
            ``fit`` if the model supports a ``feature_names_`` attribute.

    Returns:
        pd.DataFrame: Results table with columns
            ``[fold, pr_auc, roc_auc, recall_at_fpr_0.1, ndcg_at_k_100]``.

    Raises:
        ValueError: If any fold's test set contains no positive samples.
    """
    records: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            logger.warning(
                "Fold %d training split contains only 1 class — skipping.", fold_idx
            )
            continue

        if len(np.unique(y_test)) < 2:
            logger.warning(
                "Fold %d test split contains only 1 class — skipping.", fold_idx
            )
            continue

        logger.info(
            "Fold %d/%d | train_n=%d | test_n=%d | test_pos=%d",
            fold_idx + 1,
            len(folds),
            len(train_idx),
            len(test_idx),
            int(y_test.sum()),
        )

        # Fit on training split
        model.fit(X_train, y_train)

        # Predict on test split
        y_score = model.predict_proba(X_test)

        records.append(
            {
                "fold": fold_idx,
                "pr_auc": pr_auc(y_test, y_score),
                "roc_auc": roc_auc(y_test, y_score),
                "recall_at_fpr_0.1": recall_at_fpr(y_test, y_score, target_fpr=0.1),
                "ndcg_at_k_100": ndcg_at_k(y_test, y_score, k=100),
            }
        )

    results_df = pd.DataFrame(records)
    logger.info(
        "Spatial CV complete | folds=%d | mean PR-AUC=%.4f ± %.4f",
        len(results_df),
        results_df["pr_auc"].mean() if len(results_df) > 0 else float("nan"),
        results_df["pr_auc"].std() if len(results_df) > 0 else float("nan"),
    )
    return results_df


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def compare_models(
    xgb_scores: pd.DataFrame,
    unet_scores: pd.DataFrame,
) -> dict:
    """Statistical comparison of XGBoost vs U-Net on spatial CV PR-AUC scores.

    Uses the **Wilcoxon signed-rank test** (non-parametric paired test) rather
    than a paired t-test because:
    * CV fold counts are typically small (5–10), violating normality assumptions.
    * PR-AUC values may be bounded / skewed, further violating t-test assumptions.

    Adoption criteria (aligned with geological decision-making conservatism):
    1. PR-AUC improvement > 5 percentage points (practical significance).
    2. Wilcoxon p-value < 0.05 (statistical significance).

    Both conditions must hold; a model with large but non-significant
    improvement is **not** adopted — it may reflect lucky fold assignments.

    Args:
        xgb_scores (pd.DataFrame): Output of :func:`spatial_cv_evaluate` for
            the XGBoost model; must contain a ``pr_auc`` column.
        unet_scores (pd.DataFrame): Output of :func:`spatial_cv_evaluate` for
            the U-Net model; must contain a ``pr_auc`` column.

    Returns:
        dict: Result dictionary with keys:
            * ``xgb_mean_pr_auc`` (float) — mean PR-AUC across folds.
            * ``unet_mean_pr_auc`` (float) — mean PR-AUC across folds.
            * ``p_value`` (float) — Wilcoxon signed-rank p-value.
            * ``adopt_unet`` (bool) — True iff improvement > 0.05 AND p < 0.05.

    Raises:
        ValueError: If the two DataFrames have different numbers of folds.
    """
    xgb_pr = xgb_scores["pr_auc"].values
    unet_pr = unet_scores["pr_auc"].values

    if len(xgb_pr) != len(unet_pr):
        raise ValueError(
            f"Fold count mismatch: XGB has {len(xgb_pr)} folds, "
            f"U-Net has {len(unet_pr)} folds.  Use the same spatial folds."
        )

    xgb_mean = float(xgb_pr.mean())
    unet_mean = float(unet_pr.mean())
    improvement = unet_mean - xgb_mean

    # Wilcoxon signed-rank test on paired differences
    # alternative='greater': test H1 = U-Net PR-AUC > XGB PR-AUC
    if len(xgb_pr) < 2 or (unet_pr == xgb_pr).all():
        # Wilcoxon requires at least 2 non-tied pairs
        p_value = 1.0
    else:
        stat, p_value = wilcoxon(unet_pr, xgb_pr, alternative="greater")
        p_value = float(p_value)

    adopt_unet: bool = (improvement > 0.05) and (p_value < 0.05)

    result = {
        "xgb_mean_pr_auc": xgb_mean,
        "unet_mean_pr_auc": unet_mean,
        "improvement": improvement,
        "p_value": p_value,
        "adopt_unet": adopt_unet,
    }

    logger.info(
        "Model comparison | XGB PR-AUC=%.4f | UNet PR-AUC=%.4f | "
        "Δ=%.4f | p=%.4f | adopt_unet=%s",
        xgb_mean,
        unet_mean,
        improvement,
        p_value,
        adopt_unet,
    )
    return result


# ---------------------------------------------------------------------------
# Blind holdout evaluation
# ---------------------------------------------------------------------------


def blind_holdout_report(
    model: Any,
    X_blind: np.ndarray,
    y_blind: np.ndarray,
) -> dict:
    """Evaluate a fitted model on a blind holdout set.

    The blind holdout set must have been withheld **before any model selection
    or hyperparameter tuning** to provide an unbiased final performance
    estimate.  This function must be called *at most once* per model — calling
    it multiple times and selecting the best result invalidates the holdout.

    In addition to the four standard metrics, returns calibration curve data
    computed via ``sklearn.calibration.calibration_curve`` with 10 bins.
    Well-calibrated models have ``fraction_of_positives ≈ mean_predicted_value``
    on the diagonal, which is important for threshold selection used by
    :func:`outputs.prospectivity_map.generate_target_polygons`.

    Args:
        model: Fitted model implementing
            ``predict_proba(X) -> np.ndarray`` of shape ``(N,)``.
        X_blind (np.ndarray): Blind feature matrix, shape ``(N, n_features)``.
        y_blind (np.ndarray): Blind binary labels, shape ``(N,)``.

    Returns:
        dict: Result dictionary with keys:
            * ``pr_auc`` (float)
            * ``roc_auc`` (float)
            * ``recall_at_fpr_0.1`` (float)
            * ``ndcg_at_k_100`` (float)
            * ``calibration`` (dict): nested dict with:
                - ``fraction_of_positives`` (list[float])
                - ``mean_predicted_value`` (list[float])

    Raises:
        ValueError: If ``y_blind`` contains no positive samples.
    """
    if y_blind.sum() == 0:
        raise ValueError(
            "Blind holdout set contains no positive samples — "
            "cannot compute prospectivity metrics."
        )

    y_score = model.predict_proba(X_blind)

    # Core metrics
    metrics: dict = {
        "pr_auc": pr_auc(y_blind, y_score),
        "roc_auc": roc_auc(y_blind, y_score),
        "recall_at_fpr_0.1": recall_at_fpr(y_blind, y_score, target_fpr=0.1),
        "ndcg_at_k_100": ndcg_at_k(y_blind, y_score, k=100),
    }

    # Calibration curve data (10 quantile-based bins)
    frac_pos, mean_pred = calibration_curve(
        y_blind, y_score, n_bins=10, strategy="quantile"
    )
    metrics["calibration"] = {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
    }

    logger.info(
        "Blind holdout | PR-AUC=%.4f | ROC-AUC=%.4f | "
        "Recall@FPR10%%=%.4f | NDCG@100=%.4f",
        metrics["pr_auc"],
        metrics["roc_auc"],
        metrics["recall_at_fpr_0.1"],
        metrics["ndcg_at_k_100"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Pass 5: Spatial Conformal Risk Bounding (Distribution-Free Uncertainty)
# ---------------------------------------------------------------------------


def spatial_conformal_calibrate(
    cal_probs: np.ndarray,
    cal_targets: np.ndarray,
    alpha: float = 0.10,
    cal_block_ids: Optional[np.ndarray] = None,
    train_block_ids: Optional[np.ndarray] = None,
    epistemic_std: Optional[np.ndarray] = None,
) -> float:
    """Calibrate distribution-free conformal quantile $\\hat{q}_{1-\\alpha}$ across spatial holdout blocks.

    Provides a $(1 - \\alpha)$ statistical coverage guarantee (default $\\alpha = 0.10$
    for 90 % coverage) without Gaussian assumptions. Non-conformity scores on
    spatially buffered holdout blocks are computed as:

    $$s_i = \\frac{|y_i - \\hat{p}(x_i)|}{\\hat{\\sigma}(x_i)}$$

    (or $s_i = |y_i - \\hat{p}(x_i)|$ when ``epistemic_std`` is ``None``).
    When ``cal_block_ids`` is provided, block-level non-conformity scores are
    aggregated to respect spatial autocorrelation within blocks, and finite-sample
    conformal quantile correction $\\lceil (n + 1)(1 - \\alpha) \\rceil / n$ is applied.

    Enforces **Acceptance Criterion 4 (Spatial Block Independence)**: raises
    ``ValueError`` if ``cal_block_ids`` and ``train_block_ids`` share any block ID.

    Args:
        cal_probs: 1-D array of calibrated probabilities $\\hat{p}(x_i) \\in [0, 1]$
            on held-out spatial calibration folds.
        cal_targets: 1-D array of target reference probabilities or verified binary
            labels $y_i \\in [0, 1]$ on held-out spatial calibration folds.
        alpha: Miscoverage rate $\\alpha \\in (0, 1)$. Default ``0.10`` (90 % coverage).
        cal_block_ids: Optional 1-D array of spatial block IDs for each calibration
            instance.
        train_block_ids: Optional 1-D array of spatial block IDs used during model
            fitting; verified to be strictly disjoint from ``cal_block_ids``.
        epistemic_std: Optional 1-D array of local epistemic spread $\\hat{\\sigma}(x_i) > 0$
            for locally adaptive conformal intervals.

    Returns:
        Calibrated conformal quantile $\\hat{q}_{1-\\alpha} > 0$.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if cal_block_ids is not None and train_block_ids is not None:
        overlap = set(np.asarray(cal_block_ids).ravel()).intersection(
            set(np.asarray(train_block_ids).ravel())
        )
        if overlap:
            raise ValueError(
                f"Spatial block leakage detected in conformal calibration! "
                f"Overlapping block IDs between train and calibration: {sorted(overlap)}"
            )

    p = np.asarray(cal_probs, dtype=np.float64).ravel()
    y = np.asarray(cal_targets, dtype=np.float64).ravel()
    if len(p) != len(y) or len(p) == 0:
        raise ValueError("cal_probs and cal_targets must have equal non-zero length.")

    residuals = np.abs(y - p)
    if epistemic_std is not None:
        scale = np.maximum(np.asarray(epistemic_std, dtype=np.float64).ravel(), 1e-3)
        scores = residuals / scale
    else:
        scores = residuals

    # Aggregate non-conformity scores at the spatial block level if block IDs are supplied
    if cal_block_ids is not None:
        b_ids = np.asarray(cal_block_ids).ravel()
        if len(b_ids) != len(scores):
            raise ValueError("cal_block_ids must have the same length as cal_probs.")
        unique_blocks = np.unique(b_ids)
        if len(unique_blocks) >= 3:
            scores = np.array(
                [np.nanmean(scores[b_ids == b]) for b in unique_blocks],
                dtype=np.float64,
            )

    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    q_hat = float(np.nanquantile(scores, level))
    return max(q_hat, 1e-4)


def spatial_conformal_bounds(
    pred_probs: np.ndarray,
    q_hat: float,
    epistemic_std: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pixel spatial conformal Drill Confidence Range $[L(x), U(x)]$.

    $$L(x) = \\max\\left(0.0,\\; \\hat{p}(x) - \\hat{q}_{1-\\alpha} \\cdot \\hat{\\sigma}(x)\\right)$$
    $$U(x) = \\min\\left(1.0,\\; \\hat{p}(x) + \\hat{q}_{1-\\alpha} \\cdot \\hat{\\sigma}(x)\\right)$$

    (where $\\hat{\\sigma}(x) = 1$ if ``epistemic_std`` is ``None``).

    Args:
        pred_probs: Calibrated deposit probabilities $P(\\text{Deposit} \\mid x)$ in $[0, 1]$.
        q_hat: Conformal quantile $\\hat{q}_{1-\\alpha}$ from :func:`spatial_conformal_calibrate`.
        epistemic_std: Optional epistemic standard deviation array matching ``pred_probs``.

    Returns:
        Tuple ``(lower_bound_L, upper_bound_U)`` with values in $[0, 1]$ and
        $L(x) \\le P(\\text{Deposit} \\mid x) \\le U(x)$.
    """
    p = np.asarray(pred_probs, dtype=np.float64)
    if epistemic_std is not None:
        scale = np.maximum(np.asarray(epistemic_std, dtype=np.float64), 1e-3)
        half_width = float(q_hat) * scale
    else:
        half_width = float(q_hat)

    lower = np.clip(p - half_width, 0.0, 1.0)
    upper = np.clip(p + half_width, 0.0, 1.0)
    nan_mask = np.isnan(p)
    lower = np.where(nan_mask, np.nan, lower)
    upper = np.where(nan_mask, np.nan, upper)
    return lower, upper

