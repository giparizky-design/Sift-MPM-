"""Conformal prediction and statistical risk calibration for the MPM Pipeline.

Implements decoupled 90% conformal prediction set gates and studentized lower bounds
under exploration covariate shift with importance weighting (D1, D9, Bug 1).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.schema.canonical import K3_COAL_HEAD_NAMES, K4_HEAD_NAMES

logger = logging.getLogger(__name__)


class ConformalQuantile(float):
    """Float subclass storing both the set gate quantile and studentized residual quantile."""

    q_set: float
    q_res: float

    def __new__(cls, q_set: float, q_res: float = 1.0) -> "ConformalQuantile":
        obj = super().__new__(cls, float(q_set))
        obj.q_set = float(q_set)
        obj.q_res = float(q_res)
        return obj


def weighted_conformal_quantile(
    scores: np.ndarray,
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.10,
) -> float:
    """Compute the (1 - alpha) conformal quantile under optional importance weights.

    Under covariate shift (Tibshirani et al., 2019), weights represent the likelihood ratio
        w_conf(x) = 1.0 / clip(e(x), 0.05, 1.0)
    between the target exploration domain and the calibration discovery distribution.

    Args:
        scores: 1-D array of nonconformity or residual scores.
        weights: Optional 1-D array of non-negative importance weights matching scores.
        alpha: Miscoverage rate (default 0.10 for 90% coverage).

    Returns:
        Weighted (1 - alpha) quantile threshold.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return 1.0

    if weights is None:
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        return float(np.quantile(s, level))

    w = np.asarray(weights, dtype=np.float64).ravel()
    if len(w) != len(scores):
        w = np.ones(n, dtype=np.float64)
    else:
        w = w[np.isfinite(scores)]

    w = np.clip(w, 1e-8, None)
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]

    cum_w = np.cumsum(w_sorted) / np.sum(w_sorted)
    target = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n) if n > 0 else (1.0 - alpha)
    target = min(target, 1.0 - alpha) if target > 1.0 else (1.0 - alpha)

    idx = int(np.searchsorted(cum_w, target))
    idx = min(idx, n - 1)
    return float(s_sorted[idx])


def compute_studentized_conformal_lower_bound(
    p_cal: np.ndarray,
    sigma: np.ndarray,
    q_res_90: float,
) -> np.ndarray:
    """Compute the conservative 90% studentized prospectivity lower bound (D1).

    p_low90(s) = clip(p_cal(s) - q_res_90 * sigma_epistemic(s), 0.0, 1.0)

    Scales inversely with local epistemic uncertainty sigma without additive distortion
    when sigma -> 0.

    Args:
        p_cal: Array of calibrated probabilities in [0, 1].
        sigma: Array of epistemic standard deviations.
        q_res_90: Studentized residual quantile.

    Returns:
        Lower bound array clipped to [0, 1].
    """
    p_arr = np.asarray(p_cal, dtype=np.float64)
    s_arr = np.asarray(sigma, dtype=np.float64) if sigma is not None else np.zeros_like(p_arr)
    return np.clip(p_arr - float(q_res_90) * s_arr, 0.0, 1.0)


def evaluate_conformal_set_gate(
    p_cal: np.ndarray,
    sigma: np.ndarray,
    q_set_90: float,
) -> np.ndarray:
    """Evaluate membership in the 90% conformal prediction set (D1).

    Condition: p_cal(s) - 0.5 * sigma(s) >= 1.0 - q_set_90
    Equivalent to: V(s) = 1.0 - p_cal(s) + 0.5 * sigma(s) <= q_set_90

    Args:
        p_cal: Calibrated prospectivity probabilities.
        sigma: Epistemic standard deviations.
        q_set_90: Conformal set gate quantile.

    Returns:
        Boolean mask of candidate inclusion.
    """
    p_arr = np.asarray(p_cal, dtype=np.float64)
    s_arr = np.asarray(sigma, dtype=np.float64) if sigma is not None else np.zeros_like(p_arr)
    return (p_arr - 0.5 * s_arr) >= (1.0 - float(q_set_90))


def compute_multi_mineral_conformal_bounds(
    p_cal_k: np.ndarray,
    sigma_ens_k: Optional[np.ndarray] = None,
    cal_pos_p_cal: Optional[np.ndarray] = None,
    cal_pos_sigma: Optional[np.ndarray] = None,
    alpha: float = 0.10,
    q_hat_k: Optional[Union[float, ConformalQuantile]] = None,
    cal_pos_weights: Optional[np.ndarray] = None,
    q_res_k: Optional[float] = None,
    y_k: Optional[np.ndarray] = None,
    target_names: Optional[Sequence[str]] = None,
    y_val: Optional[np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, ConformalQuantile],
    Tuple[np.ndarray, np.ndarray, Dict[int, float], Dict[int, float]],
]:
    """Pass 5: Multi-Mineral Conformal Prediction Sets & 90% Confidence Lower Bounds.

    Decouples the 90% prediction set gate from the studentized lower bound (D1, D9, Bug 1):
      - Prediction set gate: p_cal - 0.5 * sigma >= 1.0 - q_set_90
      - Studentized lower bound: p_low90 = clip(p_cal - q_res_90 * sigma, 0.0, 1.0)
      - Importance weights under exploration shift: w_conf = 1 / clip(e(x), 0.05, 1.0)

    Supports both:
      1. Single-head invocation (1D p_cal_k): Returns (l_bound, v_scores, q_quantile)
      2. Multi-head invocation (2D p_cal_k with y_k or 2D matrix):
         Returns (p_set_gates, p_low90, q_set_dict, q_res_dict)

    Args:
        p_cal_k: Calibrated probabilities for head k (1-D) or all K heads (2-D).
        sigma_ens_k: Inter-model epistemic standard deviation.
        cal_pos_p_cal: Held-out positive validation calibrated probabilities.
        cal_pos_sigma: Held-out positive validation standard deviations.
        alpha: Miscoverage rate (default 0.10 for 90% confidence).
        q_hat_k: Pre-calibrated set quantile or ConformalQuantile.
        cal_pos_weights: Importance weights for calibration positives.
        q_res_k: Pre-calibrated studentized residual quantile.
        y_k: Optional target labels for multi-head calibration.
        target_names: Optional sequence of commodity head names.
        y_val: Optional alias for y_k.

    Returns:
        Single-head mode: Tuple (l_bound, v_scores, q_quantile).
        Multi-head mode: Tuple (p_set_gates, p_low90, q_set_dict, q_res_dict).
    """
    if y_k is None and y_val is not None:
        y_k = y_val

    p_arr = np.asarray(p_cal_k, dtype=np.float64)
    sig_arr = (
        np.asarray(sigma_ens_k, dtype=np.float64)
        if sigma_ens_k is not None
        else np.zeros_like(p_arr)
    )

    if cal_pos_p_cal is None and y_k is not None and p_arr.ndim <= 1:
        y_arr = np.asarray(y_k).ravel()
        pos_mask = (y_arr > 0)
        if np.any(pos_mask):
            cal_pos_p_cal = p_arr[pos_mask]
            cal_pos_sigma = sig_arr[pos_mask]

    # Multi-head mode branch: (N, K) input with y_k or when cal_pos_p_cal is omitted on 2D input
    if p_arr.ndim == 2 and (y_k is not None or (cal_pos_p_cal is None and cal_pos_sigma is None and q_hat_k is None)):
        N, K = p_arr.shape
        p_set_gates = np.zeros((N, K), dtype=bool)
        p_low90 = np.zeros((N, K), dtype=np.float64)
        q_set_dict: Dict[int, float] = {}
        q_res_dict: Dict[int, float] = {}

        y_mat = np.asarray(y_k, dtype=np.int32) if y_k is not None else None

        for k in range(K):
            pk = p_arr[:, k]
            sk = sig_arr[:, k]

            if y_mat is not None and np.any(y_mat[:, k] == 1):
                pos_mask = (y_mat[:, k] == 1)
                cp_p = pk[pos_mask]
                cp_s = sk[pos_mask]
                cp_w = cal_pos_weights[pos_mask] if cal_pos_weights is not None else None
            else:
                cp_p = None
                cp_s = None
                cp_w = None

            l_k, v_k, q_k = _compute_single_head_conformal(
                p_cal=pk,
                sigma=sk,
                cal_pos_p=cp_p,
                cal_pos_sig=cp_s,
                cal_pos_w=cp_w,
                alpha=alpha,
                q_set_in=q_hat_k,
                q_res_in=q_res_k,
            )
            p_set_gates[:, k] = (v_k <= q_k.q_set)
            p_low90[:, k] = l_k
            q_set_dict[k] = q_k.q_set
            q_res_dict[k] = q_k.q_res

        return p_set_gates, p_low90, q_set_dict, q_res_dict

    # Single-head mode branch: (N,) or (N, 1)
    l_bound, v_scores, q_quant = _compute_single_head_conformal(
        p_cal=p_arr,
        sigma=sig_arr,
        cal_pos_p=cal_pos_p_cal,
        cal_pos_sig=cal_pos_sigma,
        cal_pos_w=cal_pos_weights,
        alpha=alpha,
        q_set_in=q_hat_k,
        q_res_in=q_res_k,
    )
    return l_bound, v_scores, q_quant


def _compute_single_head_conformal(
    p_cal: np.ndarray,
    sigma: np.ndarray,
    cal_pos_p: Optional[np.ndarray],
    cal_pos_sig: Optional[np.ndarray],
    cal_pos_w: Optional[np.ndarray],
    alpha: float,
    q_set_in: Optional[Union[float, ConformalQuantile]],
    q_res_in: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, ConformalQuantile]:
    """Helper computing decoupled set quantile, residual quantile, and lower bound."""
    p_vec = np.asarray(p_cal, dtype=np.float64).ravel()
    s_vec = np.asarray(sigma, dtype=np.float64).ravel() if sigma is not None else np.zeros_like(p_vec)
    v_scores = 1.0 - p_vec + 0.5 * s_vec

    # 1. Determine Prediction Set Quantile q_set
    if q_set_in is not None:
        q_set = float(q_set_in)
    elif cal_pos_p is not None and len(cal_pos_p) > 0:
        cp_p = np.asarray(cal_pos_p, dtype=np.float64).ravel()
        cp_s = (
            np.asarray(cal_pos_sig, dtype=np.float64).ravel()
            if cal_pos_sig is not None
            else np.zeros_like(cp_p)
        )
        cp_v = 1.0 - cp_p + 0.5 * cp_s
        q_set = weighted_conformal_quantile(scores=cp_v, weights=cal_pos_w, alpha=alpha)
    else:
        q_set = 1.0

    # 2. Determine Studentized Residual Quantile q_res
    if q_res_in is not None:
        q_res = float(q_res_in)
    elif isinstance(q_set_in, ConformalQuantile):
        q_res = float(q_set_in.q_res)
    elif cal_pos_p is not None and len(cal_pos_p) > 0:
        cp_p = np.asarray(cal_pos_p, dtype=np.float64).ravel()
        cp_s = (
            np.asarray(cal_pos_sig, dtype=np.float64).ravel()
            if cal_pos_sig is not None
            else np.zeros_like(cp_p)
        )
        # Studentized nonconformity: |y - p| / sigma on positives y=1
        residuals = (1.0 - cp_p) / np.maximum(cp_s, 1e-4)
        q_res = weighted_conformal_quantile(scores=residuals, weights=cal_pos_w, alpha=alpha)
    else:
        q_res = 1.0

    q_quant = ConformalQuantile(q_set=q_set, q_res=q_res)

    # 3. Compute lower bound:
    # If studentized residual quantile was explicitly specified (q_res_in is not None),
    # compute studentized lower bound: clip(p_cal - q_res * sigma, 0.0, 1.0) (D1, Bug 1).
    # Otherwise, maintain exact backward compatibility with legacy single-head callers:
    # clip(p_cal - 0.5 * sigma - (1.0 - q_set), 0.0, 1.0).
    if q_res_in is not None:
        l_bound = np.clip(p_vec - q_res * s_vec, 0.0, 1.0)
    else:
        l_bound = np.clip(p_vec - (0.5 * s_vec) - (1.0 - q_set), 0.0, 1.0)

    if p_cal.ndim > 1:
        l_bound = l_bound.reshape(p_cal.shape)
        v_scores = v_scores.reshape(p_cal.shape)

    return l_bound, v_scores, q_quant


def build_conformal_target_sets(
    v_scores_k4: np.ndarray,
    q_hat_k4: Union[np.ndarray, Sequence[Union[float, ConformalQuantile]]],
    target_names: Optional[Sequence[str]] = None,
) -> List[List[str]]:
    """Construct multi-mineral/rank prediction candidate sets C_{0.90}(s).

    Includes class k if V_{i,k} <= q_hat^{(k)} (i.e. p_cal - 0.5 * sigma >= 1 - q_hat).

    Args:
        v_scores_k4: Nonconformity score matrix of shape (N, K).
        q_hat_k4: Sequence of conformal quantiles for the K heads.
        target_names: Optional sequence of class names.

    Returns:
        List of length N of lists of target classes included in C_{0.90}.
    """
    v_arr = np.asarray(v_scores_k4, dtype=np.float64)
    q_arr = np.array([float(q) for q in q_hat_k4], dtype=np.float64).ravel()

    if target_names is not None:
        names = list(target_names)
    elif v_arr.shape[1] == 3:
        names = list(K3_COAL_HEAD_NAMES)
    else:
        names = list(K4_HEAD_NAMES)

    n_heads = min(len(names), v_arr.shape[1], len(q_arr))
    target_sets: List[List[str]] = []

    for i in range(v_arr.shape[0]):
        included = [names[k] for k in range(n_heads) if v_arr[i, k] <= q_arr[k]]
        target_sets.append(included)

    return target_sets
