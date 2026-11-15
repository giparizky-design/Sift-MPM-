"""Calibration and posterior estimation routines for the MPM Pipeline.

Implements Elkan & Noto (2008) label frequency constant estimation (c_hat),
closed-boundary posterior scaling, multi-head discovery prior estimation,
and monotonic logit scaling calibration with prior offsets.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Safe element-wise logit transform."""
    p_clip = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Safe element-wise sigmoid transform with exponential overflow guard."""
    x_clip = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def estimate_elkan_noto_c(
    val_positive_scores: np.ndarray,
    train_block_ids: Optional[np.ndarray] = None,
    val_block_ids: Optional[np.ndarray] = None,
    min_c: float = 0.05,
) -> float:
    """Estimate the Elkan & Noto (2008) label frequency constant c_hat = P(s=1 | y=1).

    Uses the e1 estimator from Elkan & Noto (2008): the average predicted
    probability on held-out validation positives:
        c_hat = (1 / |P_val|) * sum_{x in P_val} g(x)

    Enforces spatial block independence: verifies that validation positives
    reside in spatially disjoint holdout blocks with zero overlap with training blocks.

    Args:
        val_positive_scores: 1-D array of model scores on held-out positive instances.
        train_block_ids: Optional 1-D array of spatial block IDs used in training.
        val_block_ids: Optional 1-D array of spatial block IDs for validation samples.
        min_c: Safety floor on c_hat to avoid division by zero. Defaults to 0.05.

    Returns:
        Estimated label frequency constant c_hat in [min_c, 1.0].
    """
    if train_block_ids is not None and val_block_ids is not None:
        overlap = set(np.asarray(train_block_ids).ravel()).intersection(
            set(np.asarray(val_block_ids).ravel())
        )
        if overlap:
            raise ValueError(
                "Spatial block leakage detected in Elkan-Noto c_hat estimation! "
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

    P(Deposit | x) = min(1.0, Fused_Score(x) / c_hat)

    Args:
        fused_scores: Array of fused ensemble scores in [0, 1].
        c_hat: Exploration discovery rate / label frequency constant in (0, 1].

    Returns:
        Calibrated deposit posterior probabilities in [0, 1], preserving NaNs.
    """
    if c_hat <= 0.0 or c_hat > 1.0:
        raise ValueError(f"c_hat must be in (0, 1], got {c_hat}")
    scores = np.asarray(fused_scores, dtype=np.float64)
    calibrated = np.minimum(1.0, np.clip(scores, 0.0, 1.0) / float(c_hat))
    return np.where(np.isnan(scores), np.nan, calibrated)


def estimate_k4_discovery_priors(
    val_probs_k4: np.ndarray,
    val_targets_k4: np.ndarray,
    min_c: float = 0.10,
    train_block_ids: Optional[np.ndarray] = None,
    val_block_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate discovery prior constants c_hat_k = mean(p_raw,k[P_k]) for all 4 heads.

    Enforces spatial block independence.

    Args:
        val_probs_k4: Validation prediction matrix of shape (N, 4).
        val_targets_k4: Validation target matrix of shape (N, 4) with 1 for positives.
        min_c: Minimum floor for each discovery prior (default 0.10).
        train_block_ids: Optional spatial block IDs used in training.
        val_block_ids: Optional spatial block IDs for validation samples.

    Returns:
        1-D array of shape (4,) containing (c_hat_0, c_hat_1, c_hat_2, c_hat_3).
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
        pos_mask = targets[:, k] == 1
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


def calibrate_closed_boundary_elkan_noto(
    p_raw_k: np.ndarray,
    c_hat_k: float,
    gamma_closed_k: Optional[np.ndarray] = None,
    min_c: float = 0.10,
) -> np.ndarray:
    """Pass 4: Closed-Boundary Elkan-Noto Scaling for Head k.

    p_cal,k = Gamma_closed^{(k)}(s) * min(1.0, p_raw,k / max(c_hat_k, min_c))

    Args:
        p_raw_k: Array of raw model probability scores in [0, 1].
        c_hat_k: Holdout scalar discovery prior constant in (0, 1].
        gamma_closed_k: Optional closed-boundary geomechanical/geological host gate in [0, 1].
        min_c: Safety floor on c_hat_k (default 0.10).

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


def calibrate_monotonic_logit_scaling(
    s: np.ndarray,
    c_hat: float,
    pi_k: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Calibrate out-of-fold scores via monotonic logit scaling with prior offset (D3).

    Adjusts raw model scores in log-odds space using the discovery frequency c_hat
    and geological domain class prior pi_k:
        logit(P(Y=1 | x)) = logit(s(x)) - logit(c_hat) + logit(pi_k)
        P(Y=1 | x) = sigmoid(logit(P(Y=1 | x)))

    Args:
        s: Raw or fused model scores in [0, 1].
        c_hat: Holdout discovery prior constant P(S=1 | Y=1) in (0, 1].
        pi_k: Domain geological class prior prevalence P(Y=1) in (0, 1].
        eps: Small numerical constant for logit clipping.

    Returns:
        Calibrated prospectivity probabilities in [0, 1].
    """
    s_arr = np.asarray(s, dtype=np.float64)
    c_val = np.clip(np.asarray(c_hat, dtype=np.float64), eps, 1.0 - eps)
    pi_val = np.clip(np.asarray(pi_k, dtype=np.float64), eps, 1.0 - eps)

    logit_s = _logit(s_arr, eps=eps)
    logit_c = np.log(c_val / (1.0 - c_val))
    logit_pi = np.log(pi_val / (1.0 - pi_val))

    calibrated_logits = logit_s - logit_c + logit_pi
    calibrated_probs = _sigmoid(calibrated_logits)

    return np.where(np.isnan(s_arr), np.nan, np.clip(calibrated_probs, 0.0, 1.0))
