"""Multi-modal and cross-architecture prospectivity score fusion routines.

Implements spatial cross-validation PR-AUC weighted ensembling and precision-weighted
log-odds fusion between XGBoost and U-Net learners.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p_clip = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def compute_ensemble_weights(
    pr_auc_xgb: float,
    pr_auc_unet: float,
) -> Tuple[float, float]:
    """Compute normalized ensemble weights (w_xgb, w_unet) from spatial-CV PR-AUC.

    Satisfies w_xgb + w_unet = 1 with w proportional to PR-AUC.

    Args:
        pr_auc_xgb: Spatial cross-validation PR-AUC of the tabular XGBoost pass.
        pr_auc_unet: Spatial cross-validation PR-AUC of the spatial U-Net pass.

    Returns:
        Tuple (w_xgb, w_unet) summing to 1.0.
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
) -> Tuple[np.ndarray, float, float]:
    """Blend tabular (s_xgb) and spatial (s_unet) scores weighted by validation PR-AUC.

    Fused Score(x) = w_xgb * s_xgb(x) + w_unet * s_unet(x)

    Args:
        s_xgb: Array of XGBoost scores in [0, 1].
        s_unet: Array of U-Net scores in [0, 1], same shape as s_xgb.
        pr_auc_xgb: Mean validation PR-AUC of the XGBoost model.
        pr_auc_unet: Mean validation PR-AUC of the U-Net model.

    Returns:
        Tuple (fused_score, w_xgb, w_unet).
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


def precision_weighted_log_odds_fusion(
    p_xgb: np.ndarray,
    p_unet: np.ndarray,
    sigma_xgb: Optional[np.ndarray] = None,
    sigma_unet: Optional[np.ndarray] = None,
    default_sigma: float = 0.10,
    eps_var: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Fuse XGBoost and U-Net predictions in log-odds space weighted by inverse variance (Bug 8).

    $$l_{\text{fused}}(s) = \frac{l_{\text{xgb}}(s) / \sigma_{\text{xgb}}^2(s) + l_{\text{unet}}(s) / \sigma_{\text{unet}}^2(s)}{1/\sigma_{\text{xgb}}^2(s) + 1/\sigma_{\text{unet}}^2(s)}$$
    $$\sigma_{\text{fused}}(s) = \sqrt{\frac{1}{1/\sigma_{\text{xgb}}^2(s) + 1/\sigma_{\text{unet}}^2(s)}}$$
    $$p_{\text{fused}}(s) = \sigma(l_{\text{fused}}(s))$$

    Args:
        p_xgb: XGBoost predicted probabilities in [0, 1].
        p_unet: U-Net predicted probabilities in [0, 1].
        sigma_xgb: Epistemic standard deviation of XGBoost predictions.
        sigma_unet: Epistemic standard deviation of U-Net predictions.
        default_sigma: Fallback uncertainty if sigma is missing or zero.
        eps_var: Variance floor to prevent division by zero.

    Returns:
        Tuple (p_fused, sigma_fused).
    """
    px = np.asarray(p_xgb, dtype=np.float64)
    pu = np.asarray(p_unet, dtype=np.float64)

    if px.shape != pu.shape:
        raise ValueError(f"p_xgb shape {px.shape} != p_unet shape {pu.shape}")

    sx = np.asarray(sigma_xgb, dtype=np.float64) if sigma_xgb is not None else np.full_like(px, default_sigma)
    su = np.asarray(sigma_unet, dtype=np.float64) if sigma_unet is not None else np.full_like(pu, default_sigma)

    var_x = np.maximum(sx ** 2, eps_var)
    var_u = np.maximum(su ** 2, eps_var)

    prec_x = 1.0 / var_x
    prec_u = 1.0 / var_u
    prec_total = prec_x + prec_u

    lx = _logit(px)
    lu = _logit(pu)

    l_fused = (lx * prec_x + lu * prec_u) / prec_total
    p_fused = _sigmoid(l_fused)
    sigma_fused = np.sqrt(1.0 / prec_total)

    return np.clip(p_fused, 0.0, 1.0), sigma_fused
