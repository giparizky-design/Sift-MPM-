"""Kalimantan K=4 Multi-Mineral Deposit-Mechanics Engine (Domain A).

Implements TwoStageK4Ensemble, _BaggedXGBoostHead, Yousefi & Carranza (2015) P-A
feature weighting, Hájek self-normalized positive sample weights, and concentric
multi-ring lateral zonation meta-features for Porphyry Cu meta-stacking (D3, D7).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
import xgboost as xgb

from src.models.calibration import calibrate_monotonic_logit_scaling
from src.schema.canonical import (
    K4_HEAD_NAMES,
    METALS_PREVALENCE_PRIORS,
    verify_no_raw_pixel_leakage,
)

logger = logging.getLogger(__name__)


def compute_hajek_asymmetric_head_weights(
    labels_k: np.ndarray,
    e_logistics: np.ndarray,
    exclusion_mask_k: Optional[np.ndarray] = None,
    min_propensity: float = 0.05,
) -> np.ndarray:
    """Compute Hájek self-normalized positive weights and neutral unlabeled weights (D3).

    Eliminates SAR/SCAR double-correction:
      - Positive cells (label == 1): w_tilde_i = (1 / e(x_i)) / mean_{j in P_k}(1 / e(x_j)),
        guaranteeing mean positive weight of 1.0 over P_k.
      - Unlabeled cells (label == -1 or 0): w_unlab = exclusion_mask * 1.0 (neutral).

    Args:
        labels_k: 1-D array of labels where 1 = positive, -1 or 0 = unlabeled.
        e_logistics: 1-D array of observation propensities e(s).
        exclusion_mask_k: Optional 1-D array of spatial exclusion buffer mask in {0.0, 1.0}.
        min_propensity: Minimum observation propensity floor (default 0.05).

    Returns:
        1-D float64 array of sample weights.
    """
    labels = np.asarray(labels_k).ravel()
    e_prop = np.asarray(e_logistics, dtype=np.float64).ravel()
    m_mask = (
        np.asarray(exclusion_mask_k, dtype=np.float64).ravel()
        if exclusion_mask_k is not None
        else np.ones_like(labels, dtype=np.float64)
    )

    if len(labels) != len(e_prop) or len(labels) != len(m_mask):
        raise ValueError(
            f"Dimension mismatch: labels ({len(labels)}), e_logistics ({len(e_prop)}), exclusion_mask ({len(m_mask)})"
        )

    w_k = np.zeros(len(labels), dtype=np.float64)
    pos_mask = (labels == 1)
    unl_mask = (labels != 1)

    # 1. Hájek self-normalized positive weights (mean 1.0 over P_k)
    inv_e_pos = 1.0 / np.clip(e_prop[pos_mask], float(min_propensity), 1.0)
    mean_inv_e = float(np.mean(inv_e_pos)) if len(inv_e_pos) > 0 else 1.0
    w_k[pos_mask] = inv_e_pos / max(mean_inv_e, 1e-6)

    # 2. Neutral unlabeled weights outside exclusion buffer
    w_k[unl_mask] = m_mask[unl_mask] * 1.0

    return w_k


def compute_pa_intersection_weights(
    X: np.ndarray,
    y_k: np.ndarray,
    n_thresholds: int = 100,
) -> np.ndarray:
    """Compute Yousefi & Carranza (2015) Prediction-Area (P-A) intersection weights.

    For each feature j, evaluates candidate thresholds to find intersection point t*
    where P_r(t*) = 1 - Q_a(t*). Returns normalized evidential feature weights.

    Args:
        X: Feature matrix of shape (N, D).
        y_k: Binary target vector of shape (N,) with positives marked as 1.
        n_thresholds: Number of candidate thresholds to test (default 100).

    Returns:
        1-D array of shape (D,) with normalized weights summing to 1.0.
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
        for direction in ("greater", "less"):
            t_candidates = np.linspace(np.min(feat_all), np.max(feat_all), n_thresholds)
            if direction == "greater":
                pr = np.array([np.sum(feat_pos >= t) / n_pos_valid for t in t_candidates])
                qa = np.array([np.sum(feat_all >= t) / n_all_valid for t in t_candidates])
            else:
                pr = np.array([np.sum(feat_pos <= t) / n_pos_valid for t in t_candidates])
                qa = np.array([np.sum(feat_all <= t) / n_all_valid for t in t_candidates])

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


def compute_concentric_zonation_features(
    p_skarn: np.ndarray,
    p_epithermal: np.ndarray,
    p_laterite: np.ndarray,
    coords_m: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute Lowell & Guilbert (1970) concentric lateral zonation meta-features (D7).

    Features generated:
      1. oof_prob_skarn: p1 (contact metasomatism)
      2. oof_prob_epithermal: p2 (distal vein halo)
      3. zonation_inner_core: 0-1.5 km potassic alteration core mean(p1, p2)
      4. zonation_proximal_skarn: 1.5-4.0 km proximal contact ring mean(p1)
      5. zonation_distal_epithermal: 3.0-7.0 km distal halo ring mean(p2)
      6. zonation_radial_gradient: proximal_skarn - distal_epithermal
      7. oof_prob_laterite: p3 (ultramafic laterite, negative indicator)

    Args:
        p_skarn: 1-D array of skarn probabilities.
        p_epithermal: 1-D array of epithermal probabilities.
        p_laterite: 1-D array of laterite probabilities.
        coords_m: Optional (N, 2) array of spatial coordinates in meters.

    Returns:
        (N, 7) array of concentric meta-features.
    """
    p1 = np.asarray(p_skarn, dtype=np.float32).ravel()
    p2 = np.asarray(p_epithermal, dtype=np.float32).ravel()
    p3 = np.asarray(p_laterite, dtype=np.float32).ravel()
    N = len(p1)

    if coords_m is not None and len(coords_m) == N and N <= 50_000:
        try:
            from scipy.spatial import cKDTree

            xy = np.asarray(coords_m, dtype=np.float64)
            tree = cKDTree(xy)

            # Query multi-ring radius neighborhoods
            idx_1500 = tree.query_ball_point(xy, r=1500.0)
            idx_4000 = tree.query_ball_point(xy, r=4000.0)
            idx_3000 = tree.query_ball_point(xy, r=3000.0)
            idx_7000 = tree.query_ball_point(xy, r=7000.0)

            f_core = np.array([np.mean(0.5 * (p1[pts] + p2[pts])) if len(pts) > 0 else 0.5 * (p1[i] + p2[i]) for i, pts in enumerate(idx_1500)], dtype=np.float32)

            f_prox = np.zeros(N, dtype=np.float32)
            for i in range(N):
                ring_pts = list(set(idx_4000[i]) - set(idx_1500[i]))
                f_prox[i] = np.mean(p1[ring_pts]) if len(ring_pts) > 0 else p1[i]

            f_dist = np.zeros(N, dtype=np.float32)
            for i in range(N):
                ring_pts = list(set(idx_7000[i]) - set(idx_3000[i]))
                f_dist[i] = np.mean(p2[ring_pts]) if len(ring_pts) > 0 else p2[i]

            f_grad = f_prox - f_dist
        except Exception as e:
            logger.debug("Falling back to point zonation features: %s", e)
            f_core = 0.5 * (p1 + p2)
            f_prox = p1.copy()
            f_dist = p2.copy()
            f_grad = f_prox - f_dist
    else:
        # Fast fallback when spatial coordinates are absent or grid is huge
        f_core = 0.5 * (p1 + p2)
        f_prox = p1.copy()
        f_dist = p2.copy()
        f_grad = f_prox - f_dist

    return np.column_stack([p1, p2, f_core, f_prox, f_dist, f_grad, p3])


ZONATION_META_FEATURE_NAMES: List[str] = [
    "oof_prob_skarn",
    "oof_prob_epithermal",
    "zonation_inner_core",
    "zonation_proximal_skarn",
    "zonation_distal_epithermal",
    "zonation_radial_gradient",
    "oof_prob_laterite",
]


class _BaggedXGBoostHead:
    """Specialized bagged XGBoost estimator for a single deposit head.

    Supports tree_method='hist', max_delta_step=1.0, bootstrap bagging for
    epistemic standard deviation estimation, monotone_constraints, and P-A feature weights.
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
        n_bag: int = 10,
        feature_weights: Optional[np.ndarray] = None,
        monotone_constraints: Optional[Union[Sequence[int], Tuple[int, ...], Dict[str, int]]] = None,
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
        self.monotone_constraints = monotone_constraints
        self.random_state = int(random_state)
        self.models_: list[xgb.XGBClassifier] = []

    def save(self, path: Union[Path, str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "_BaggedXGBoostHead":
        return joblib.load(Path(path))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        prior_models: Optional[list[xgb.XGBClassifier]] = None,
        round_idx: int = 1,
        fold_id: int = 0,
        max_neg_ratio: int = 250,
    ) -> "_BaggedXGBoostHead":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32).ravel()
        n_samples = len(y_arr)

        w_arr = (
            np.asarray(sample_weight, dtype=np.float32).ravel()
            if sample_weight is not None
            else None
        )

        r_eff = max(1, int(round_idx))
        eta_r = max(0.01, float(self.learning_rate) * (0.85 ** (r_eff - 1)))
        self.effective_learning_rate_ = eta_r
        self.models_ = []

        pos_indices = np.where(y_arr == 1)[0]
        if len(pos_indices) == 0:
            self.is_constant_zero_ = True
            return self

        self.is_constant_zero_ = False
        if w_arr is not None:
            neg_candidates = np.where((y_arr == 0) & (w_arr > 0.0))[0]
            if len(neg_candidates) == 0:
                neg_candidates = np.where(y_arr == 0)[0]
        else:
            neg_candidates = np.where(y_arr == 0)[0]

        for b in range(self.n_bag):
            rng = np.random.default_rng(
                seed=42 + r_eff * 1009 + int(fold_id) + b * 131 + (self.random_state - 42)
            )
            if len(neg_candidates) > max(50_000, len(pos_indices) * max_neg_ratio):
                n_neg_sub = min(len(neg_candidates), max(50_000, len(pos_indices) * max_neg_ratio))
                neg_sub = rng.choice(neg_candidates, size=n_neg_sub, replace=False)
                idx = np.concatenate([pos_indices, neg_sub])
            elif self.n_bag == 1:
                idx = np.arange(n_samples)
            else:
                idx = rng.choice(n_samples, size=int(0.80 * n_samples), replace=True)

            pos_bag = (y_arr[idx] == 1)
            if not np.any(pos_bag) and len(pos_indices) > 0:
                idx = np.unique(np.concatenate([idx, pos_indices]))

            sw_bag = w_arr[idx] if w_arr is not None else None

            clf_kwargs: Dict[str, Any] = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": eta_r,
                "min_child_weight": self.min_child_weight,
                "max_delta_step": self.max_delta_step,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "tree_method": self.tree_method,
                "eval_metric": "aucpr",
                "scale_pos_weight": 1.0,
                "random_state": 42 + r_eff * 1009 + int(fold_id) + b,
                "feature_weights": self.feature_weights,
            }
            if self.monotone_constraints is not None:
                clf_kwargs["monotone_constraints"] = self.monotone_constraints

            clf = xgb.XGBClassifier(**clf_kwargs)

            xgb_model = None
            if prior_models and len(prior_models) > 0:
                p_clf = prior_models[b % len(prior_models)]
                try:
                    p_booster = p_clf.get_booster()
                    if p_booster is not None and p_booster.num_features() == X_arr.shape[1]:
                        xgb_model = p_booster
                except Exception as e:
                    logger.debug("Could not extract prior booster for bag %d: %s", b, e)

            clf.fit(X_arr[idx], y_arr[idx], sample_weight=sw_bag, xgb_model=xgb_model)
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

    Upgraded in v4.0 (D3, D7):
      - Stage-1 Heads (Skarn 1, Epithermal 2, Laterite 3):
        Fitted with max_depth=6, n_bag=10, P-A evidential feature weights,
        and Hájek self-normalized positive weights (D3).
      - Stage-2 Head (Porphyry Cu 0):
        Aligned tree parameters (max_depth=6, n_bag=10), fold-internal P-A weighting,
        concentric multi-ring lateral zonation features (inner core 0-1.5 km,
        proximal contact ring 1.5-4.0 km, distal halo ring 3.0-7.0 km, radial gradient),
        and monotonic direction constraints (+1 on proximal skarn/distal epithermal,
        -1 on laterite) (D7).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        n_estimators: int = 100,
        n_bag: int = 10,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42,
        head0_stacking: bool = True,
        min_child_weight: Optional[float] = None,
        exclude_features: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = config or {}
        self.n_estimators = int(n_estimators)
        self.n_bag = max(1, int(n_bag))
        self.learning_rate = float(learning_rate)
        self.max_depth = max(2, int(max_depth))
        self.random_state = int(random_state)
        # v4.8: head0_stacking=False fits porphyry on the base features only (no skarn /
        # epithermal meta-features); min_child_weight overrides the per-head defaults.
        self.head0_stacking = bool(head0_stacking)
        self.min_child_weight = None if min_child_weight is None else float(min_child_weight)
        # v4.8: columns of the input schema the metals heads must not use (e.g. smooth regional
        # gravity/crust fields that act as location surrogates). The ensemble keeps the full
        # input schema in feature_names_ and selects its own columns, so callers pass full rows.
        self.exclude_features = tuple(exclude_features or ())
        self.keep_idx_: Optional[list[int]] = None
        self.model_feature_names_: list[str] = []

        self.head_0_: Optional[_BaggedXGBoostHead] = None
        self.head_1_: Optional[_BaggedXGBoostHead] = None
        self.head_2_: Optional[_BaggedXGBoostHead] = None
        self.head_3_: Optional[_BaggedXGBoostHead] = None

        self.pa_weights_: dict[int, np.ndarray] = {}
        self.oof_prob_skarn_: Optional[np.ndarray] = None
        self.oof_prob_epithermal_: Optional[np.ndarray] = None
        self.oof_prob_laterite_: Optional[np.ndarray] = None
        eta0 = float(self.learning_rate)
        self.head_base_eta_: dict[int, float] = {0: eta0, 1: eta0, 2: eta0, 3: eta0}
        self.best_val_pr_auc_: dict[int, float] = {}
        self.rejected_heads_: dict[int, bool] = {0: False, 1: False, 2: False, 3: False}
        self.c_hat_: dict[int, float] = {0: 0.10, 1: 0.10, 2: 0.10, 3: 0.10}
        self.feature_names_: list[str] = []

    def save(self, path: Union[Path, str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "TwoStageK4Ensemble":
        return joblib.load(Path(path))

    def select_model_inputs(self, X_arr: np.ndarray) -> np.ndarray:
        """Columns the heads were trained on, from a full-schema (or already selected) matrix."""
        keep = getattr(self, "keep_idx_", None)
        X_arr = np.asarray(X_arr)
        if keep is None:
            return X_arr
        n_in = len(getattr(self, "feature_names_", []) or [])
        if X_arr.shape[-1] == n_in and n_in != len(keep):
            return X_arr[..., keep]
        return X_arr

    def _mcw(self, default: float) -> float:
        v = getattr(self, "min_child_weight", None)
        return float(default) if v is None else float(v)

    def _head0_input(
        self,
        X_arr: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        coords_m: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Head-0 design matrix and the names of the columns appended to the base features."""
        if not getattr(self, "head0_stacking", True):
            return X_arr, []
        if getattr(self, "zonation_enabled_", False):
            zonation_features = compute_concentric_zonation_features(
                p_skarn=p1, p_epithermal=p2, p_laterite=p3, coords_m=coords_m
            )
            return np.column_stack([X_arr, zonation_features]), list(ZONATION_META_FEATURE_NAMES)
        return np.column_stack([X_arr, p1, p2]), ["oof_prob_skarn", "oof_prob_epithermal"]

    def _extract_head_labels_and_weights(
        self,
        y_k4: Union[np.ndarray, dict, list],
        sample_weights_k4: Optional[Union[np.ndarray, dict, list]],
        e_logistics: Optional[np.ndarray],
        exclusion_masks: Optional[np.ndarray],
        N: int,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        targets: dict[int, np.ndarray] = {}
        weights: dict[int, np.ndarray] = {}

        for k in range(4):
            if isinstance(y_k4, dict):
                k_name = K4_HEAD_NAMES[k]
                raw_y = y_k4.get(k, y_k4.get(k_name, np.zeros(N, dtype=np.int32)))
            elif isinstance(y_k4, (list, tuple)):
                raw_y = y_k4[k]
            else:
                raw_y = np.asarray(y_k4)[:, k]
            raw_y = np.asarray(raw_y, dtype=np.int32).ravel()
            bin_y = (raw_y == 1).astype(np.int32)
            targets[k] = bin_y

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
                # Hájek self-normalized positive weights (D3)
                pu_labels = np.where(bin_y == 1, 1, -1)
                m_k = (
                    exclusion_masks[:, k]
                    if exclusion_masks.ndim == 2
                    else exclusion_masks
                )
                weights[k] = compute_hajek_asymmetric_head_weights(
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
        prior_ensemble: Optional["TwoStageK4Ensemble"] = None,
        round_idx: int = 1,
        fold_id: int = 0,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val_k4: Optional[Union[np.ndarray, dict, list]] = None,
        feature_names: Optional[list[str]] = None,
        coords_m: Optional[np.ndarray] = None,
    ) -> "TwoStageK4Ensemble":
        """Fit all 4 deposit heads with P-A weights, Hájek sample weights, and concentric zonation."""
        if isinstance(X, pd.DataFrame):
            col_names = list(X.columns)
            verify_no_raw_pixel_leakage(col_names)
            X_arr = X.values.astype(np.float32)
        else:
            col_names = list(feature_names) if feature_names is not None else []
            if col_names:
                verify_no_raw_pixel_leakage(col_names)
            X_arr = np.asarray(X, dtype=np.float32)

        self.feature_names_ = col_names
        excl = set(getattr(self, "exclude_features", ()) or ())
        if excl and col_names:
            self.keep_idx_ = [i for i, n in enumerate(col_names) if n not in excl]
            X_arr = X_arr[:, self.keep_idx_]
            col_names = [col_names[i] for i in self.keep_idx_]
        else:
            self.keep_idx_ = None
        self.model_feature_names_ = list(col_names)
        N, D = X_arr.shape

        targets, weights = self._extract_head_labels_and_weights(
            y_k4=y_k4,
            sample_weights_k4=sample_weights_k4,
            e_logistics=e_logistics,
            exclusion_masks=exclusion_masks,
            N=N,
        )

        eta0 = float(getattr(self, "learning_rate", 0.05))
        depth0 = int(getattr(self, "max_depth", 6))
        if prior_ensemble is not None:
            self.head_base_eta_ = dict(
                getattr(prior_ensemble, "head_base_eta_", {0: eta0, 1: eta0, 2: eta0, 3: eta0})
            )
            self.best_val_pr_auc_ = dict(getattr(prior_ensemble, "best_val_pr_auc_", {}))
        else:
            self.head_base_eta_ = {0: eta0, 1: eta0, 2: eta0, 3: eta0}
            self.best_val_pr_auc_ = {}
        self.rejected_heads_ = {0: False, 1: False, 2: False, 3: False}

        # ------------------------------------------------------------------
        # Step 1: P-A Feature Weights for Stage 1 (Heads 1, 2, 3) + Priors
        # ------------------------------------------------------------------
        self.pa_weights_[1] = compute_pa_intersection_weights(X_arr, targets[1])
        self.pa_weights_[2] = compute_pa_intersection_weights(X_arr, targets[2])
        self.pa_weights_[3] = compute_pa_intersection_weights(X_arr, targets[3])

        if col_names and len(col_names) == D:
            skarn_priors = {
                "dist_pluton_carbonate_contact_km",
                "mag_analytic_signal",
                "mag_rtp_tilt_derivative",
                "in_carbonate_host",
                "dist_carbonate_km",
            }
            epithermal_priors = {
                "dist_calc_alkaline_intrusive_km",
                "sar_palsar2_lineament_density",
                "aster_tir_silica_gated",
                "spec_clay_hydroxyl_gated",
                "mag_rtp_tilt_derivative",
            }
            laterite_priors = {
                "spec_ferrous_ultramafic_gated",
                "dem_twi",
                "dem_slope",
                "in_ultramafic_belt",
            }
            for head_k, prior_set in ((1, skarn_priors), (2, epithermal_priors), (3, laterite_priors)):
                w_k = self.pa_weights_[head_k].copy()
                for j, cname in enumerate(col_names):
                    if cname in prior_set:
                        w_k[j] = max(w_k[j], 1.5 / max(D, 1)) * 1.25
                s_k = float(np.sum(w_k))
                if s_k > 0:
                    self.pa_weights_[head_k] = w_k / s_k

        # ------------------------------------------------------------------
        # Step 2: Out-of-Fold Predictions for Stage 1 Heads (1, 2, 3)
        # ------------------------------------------------------------------
        oof_skarn = np.zeros(N, dtype=np.float32)
        oof_epithermal = np.zeros(N, dtype=np.float32)
        oof_laterite = np.zeros(N, dtype=np.float32)

        if not getattr(self, "head0_stacking", True):
            splitter = iter(())  # porphyry uses base features only; no OOF meta-features needed
        elif spatial_block_ids is not None and len(np.unique(spatial_block_ids)) >= 2:
            n_splits = min(n_cv_folds, len(np.unique(spatial_block_ids)))
            splitter = GroupKFold(n_splits=n_splits).split(X_arr, groups=spatial_block_ids)
        else:
            n_splits = max(2, min(n_cv_folds, max(2, N // 10)))
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state).split(X_arr)

        for f_i, (tr_idx, val_idx) in enumerate(splitter):
            # Internal fold fitting with matching max_depth
            h1_fold = _BaggedXGBoostHead(
                n_estimators=max(20, self.n_estimators // 2),
                max_depth=depth0,
                learning_rate=self.head_base_eta_.get(1, eta0),
                min_child_weight=self._mcw(1.0),
                max_delta_step=1.0,
                tree_method="hist",
                n_bag=1,
                feature_weights=self.pa_weights_[1],
                random_state=self.random_state + f_i,
            )
            h1_fold.fit(X_arr[tr_idx], targets[1][tr_idx], sample_weight=weights[1][tr_idx], round_idx=round_idx, fold_id=f_i)
            oof_skarn[val_idx] = h1_fold.predict_proba(X_arr[val_idx])

            h2_fold = _BaggedXGBoostHead(
                n_estimators=max(20, self.n_estimators // 2),
                max_depth=depth0,
                learning_rate=self.head_base_eta_.get(2, eta0),
                min_child_weight=self._mcw(1.0),
                max_delta_step=1.0,
                tree_method="hist",
                n_bag=1,
                feature_weights=self.pa_weights_[2],
                random_state=self.random_state + f_i + 10,
            )
            h2_fold.fit(X_arr[tr_idx], targets[2][tr_idx], sample_weight=weights[2][tr_idx], round_idx=round_idx, fold_id=f_i)
            oof_epithermal[val_idx] = h2_fold.predict_proba(X_arr[val_idx])

            h3_fold = _BaggedXGBoostHead(
                n_estimators=max(20, self.n_estimators // 2),
                max_depth=depth0,
                learning_rate=self.head_base_eta_.get(3, eta0),
                min_child_weight=self._mcw(1.0),
                max_delta_step=1.0,
                tree_method="hist",
                n_bag=1,
                feature_weights=self.pa_weights_[3],
                random_state=self.random_state + f_i + 20,
            )
            h3_fold.fit(X_arr[tr_idx], targets[3][tr_idx], sample_weight=weights[3][tr_idx], round_idx=round_idx, fold_id=f_i)
            oof_laterite[val_idx] = h3_fold.predict_proba(X_arr[val_idx])

        self.oof_prob_skarn_ = oof_skarn
        self.oof_prob_epithermal_ = oof_epithermal
        self.oof_prob_laterite_ = oof_laterite

        # ------------------------------------------------------------------
        # Step 3: Fit Final Stage 1 Heads (Heads 1, 2, 3)
        # ------------------------------------------------------------------
        prior_h1 = getattr(prior_ensemble, "head_1_", None).models_ if getattr(prior_ensemble, "head_1_", None) else None
        prior_h2 = getattr(prior_ensemble, "head_2_", None).models_ if getattr(prior_ensemble, "head_2_", None) else None
        prior_h3 = getattr(prior_ensemble, "head_3_", None).models_ if getattr(prior_ensemble, "head_3_", None) else None
        prior_h0 = getattr(prior_ensemble, "head_0_", None).models_ if getattr(prior_ensemble, "head_0_", None) else None

        self.head_1_ = _BaggedXGBoostHead(
            n_estimators=self.n_estimators,
            max_depth=depth0,
            learning_rate=self.head_base_eta_.get(1, eta0),
            min_child_weight=self._mcw(1.0),
            max_delta_step=1.0,
            tree_method="hist",
            n_bag=self.n_bag,
            feature_weights=self.pa_weights_[1],
            random_state=self.random_state + 10,
        )
        self.head_1_.fit(X_arr, targets[1], sample_weight=weights[1], prior_models=prior_h1, round_idx=round_idx, fold_id=fold_id)

        self.head_2_ = _BaggedXGBoostHead(
            n_estimators=self.n_estimators,
            max_depth=depth0,
            learning_rate=self.head_base_eta_.get(2, eta0),
            min_child_weight=self._mcw(1.0),
            max_delta_step=1.0,
            tree_method="hist",
            n_bag=self.n_bag,
            feature_weights=self.pa_weights_[2],
            random_state=self.random_state + 20,
        )
        self.head_2_.fit(X_arr, targets[2], sample_weight=weights[2], prior_models=prior_h2, round_idx=round_idx, fold_id=fold_id)

        self.head_3_ = _BaggedXGBoostHead(
            n_estimators=self.n_estimators,
            max_depth=depth0,
            learning_rate=self.head_base_eta_.get(3, eta0),
            min_child_weight=self._mcw(1.0),
            max_delta_step=1.0,
            tree_method="hist",
            n_bag=self.n_bag,
            feature_weights=self.pa_weights_[3],
            random_state=self.random_state + 30,
        )
        self.head_3_.fit(X_arr, targets[3], sample_weight=weights[3], prior_models=prior_h3, round_idx=round_idx, fold_id=fold_id)

        # ------------------------------------------------------------------
        # Step 4: Fit Stage-2 Head 0 (Porphyry Cu) with Concentric Zonation (D7)
        # ------------------------------------------------------------------
        if coords_m is not None:
            self.zonation_enabled_ = True
            zonation_features = compute_concentric_zonation_features(
                p_skarn=oof_skarn,
                p_epithermal=oof_epithermal,
                p_laterite=oof_laterite,
                coords_m=coords_m,
            )
            X_head0 = np.column_stack([X_arr, zonation_features])
            # Monotonic direction constraints (D7):
            # +1 on proximal skarn ring (col D + 3) and distal epithermal ring (col D + 4),
            # -1 on laterite (col D + 6). 0 on all base features.
            monotone_tuple = tuple([0] * D + [0, 0, 0, 1, 1, 0, -1])
        elif not getattr(self, "head0_stacking", True):
            self.zonation_enabled_ = False
            X_head0 = X_arr
            monotone_tuple = None
        else:
            self.zonation_enabled_ = False
            X_head0 = np.column_stack([X_arr, oof_skarn, oof_epithermal])
            monotone_tuple = tuple([0] * D + [1, 1])

        self.pa_weights_[0] = compute_pa_intersection_weights(X_head0, targets[0])

        self.head_0_ = _BaggedXGBoostHead(
            n_estimators=self.n_estimators,
            max_depth=depth0,  # Aligned to Stage-1 depth
            learning_rate=self.head_base_eta_.get(0, eta0),
            min_child_weight=self._mcw(3.0),
            max_delta_step=1.0,
            tree_method="hist",
            n_bag=self.n_bag,
            feature_weights=self.pa_weights_[0],
            monotone_constraints=monotone_tuple,
            random_state=self.random_state + 40,
        )
        self.head_0_.fit(
            X_head0, targets[0], sample_weight=weights[0], prior_models=prior_h0, round_idx=round_idx, fold_id=fold_id
        )

        # ------------------------------------------------------------------
        # Step 5: Estimate c_hat on Positive Training Occurrences
        # ------------------------------------------------------------------
        for k in range(4):
            pos_idx = np.where(targets[k] == 1)[0]
            if len(pos_idx) > 0:
                if k == 0:
                    p_pos = self.head_0_.predict_proba(X_head0[pos_idx])
                elif k == 1:
                    p_pos = self.head_1_.predict_proba(X_arr[pos_idx])
                elif k == 2:
                    p_pos = self.head_2_.predict_proba(X_arr[pos_idx])
                else:
                    p_pos = self.head_3_.predict_proba(X_arr[pos_idx])
                self.c_hat_[k] = float(np.clip(np.mean(p_pos), 0.05, 1.0))
            else:
                self.c_hat_[k] = 0.50

        # ------------------------------------------------------------------
        # Step 6: Per-Head Validation Checkpoint Gate
        # ------------------------------------------------------------------
        if X_val is not None and y_val_k4 is not None:
            from src.evaluation.metrics import pr_auc as _eval_pr_auc

            X_val_arr = X_val.values.astype(np.float32) if isinstance(X_val, pd.DataFrame) else np.asarray(X_val, dtype=np.float32)
            X_val_arr = self.select_model_inputs(X_val_arr)
            val_targets, _ = self._extract_head_labels_and_weights(
                y_k4=y_val_k4,
                sample_weights_k4=None,
                e_logistics=None,
                exclusion_masks=None,
                N=len(X_val_arr),
            )
            # Gate Stage-1 heads (1, 2, 3)
            for k in (1, 2, 3):
                y_vk = val_targets[k]
                if int(np.sum(y_vk == 1)) >= 1 and int(np.sum(y_vk == 0)) >= 1:
                    head_curr = getattr(self, f"head_{k}_")
                    p_curr = head_curr.predict_proba(X_val_arr)
                    auc_curr = float(_eval_pr_auc(y_vk, p_curr))
                    best_prev = self.best_val_pr_auc_.get(k)
                    if prior_ensemble is not None and best_prev is None:
                        head_prev = getattr(prior_ensemble, f"head_{k}_", None)
                        if head_prev is not None:
                            best_prev = float(_eval_pr_auc(y_vk, head_prev.predict_proba(X_val_arr)))
                    if prior_ensemble is not None and best_prev is not None and auc_curr < (best_prev - 1e-4):
                        setattr(self, f"head_{k}_", getattr(prior_ensemble, f"head_{k}_"))
                        self.head_base_eta_[k] = max(0.01, self.head_base_eta_.get(k, 0.05) * 0.5)
                        self.rejected_heads_[k] = True
                        self.best_val_pr_auc_[k] = best_prev
                    else:
                        self.best_val_pr_auc_[k] = max(auc_curr, best_prev if best_prev is not None else 0.0)

            # Gate Stage-2 Head 0
            y_v0 = val_targets[0]
            if int(np.sum(y_v0 == 1)) >= 1 and int(np.sum(y_v0 == 0)) >= 1:
                p1_val = self.head_1_.predict_proba(X_val_arr)
                p2_val = self.head_2_.predict_proba(X_val_arr)
                p3_val = self.head_3_.predict_proba(X_val_arr)
                if getattr(self, "zonation_enabled_", False):
                    val_zonation = compute_concentric_zonation_features(
                        p_skarn=p1_val, p_epithermal=p2_val, p_laterite=p3_val, coords_m=None
                    )
                    X_val_h0 = np.column_stack([X_val_arr, val_zonation])
                else:
                    X_val_h0 = np.column_stack([X_val_arr, p1_val, p2_val])
                p0_curr = self.head_0_.predict_proba(X_val_h0)
                auc0_curr = float(_eval_pr_auc(y_v0, p0_curr))
                best0_prev = self.best_val_pr_auc_.get(0)
                if prior_ensemble is not None and best0_prev is None and getattr(prior_ensemble, "head_0_", None) is not None:
                    best0_prev = float(_eval_pr_auc(y_v0, prior_ensemble.head_0_.predict_proba(X_val_h0)))
                if prior_ensemble is not None and best0_prev is not None and auc0_curr < (best0_prev - 1e-4):
                    self.head_0_ = prior_ensemble.head_0_
                    self.head_base_eta_[0] = max(0.01, self.head_base_eta_.get(0, 0.05) * 0.5)
                    self.rejected_heads_[0] = True
                    self.best_val_pr_auc_[0] = best0_prev
                else:
                    self.best_val_pr_auc_[0] = max(auc0_curr, best0_prev if best0_prev is not None else 0.0)

        return self

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords_m: Optional[np.ndarray] = None,
        calibrate: bool = False,
    ) -> np.ndarray:
        """Predict probabilities for all 4 heads: [p_0, p_1, p_2, p_3].

        Args:
            X: Feature matrix of shape (N, D).
            coords_m: Optional spatial coordinates in meters for concentric zonation.
            calibrate: If True, applies monotonic logit scaling calibration with prior offset (D3).

        Returns:
            2-D float64 array of shape (N, 4) with probabilities in [0, 1].
        """
        if self.head_0_ is None or self.head_1_ is None:
            raise RuntimeError("TwoStageK4Ensemble must be fitted before predict_proba().")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
        X_arr = self.select_model_inputs(X_arr)

        p1 = self.head_1_.predict_proba(X_arr)
        p2 = self.head_2_.predict_proba(X_arr)
        p3 = self.head_3_.predict_proba(X_arr)

        X_head0, _ = self._head0_input(X_arr, p1, p2, p3, coords_m)
        p0 = self.head_0_.predict_proba(X_head0)

        raw_probs = np.column_stack([p0, p1, p2, p3])

        if not calibrate:
            return raw_probs

        cal_probs = np.zeros_like(raw_probs)
        for k in range(4):
            k_name = K4_HEAD_NAMES[k]
            pi_k = METALS_PREVALENCE_PRIORS.get(k_name, 0.01)
            c_hat_k = self.c_hat_.get(k, 0.10)
            cal_probs[:, k] = calibrate_monotonic_logit_scaling(raw_probs[:, k], c_hat=c_hat_k, pi_k=pi_k)

        return cal_probs

    def predict_std(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords_m: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict inter-model epistemic standard deviation for all 4 heads: [sig_0, sig_1, sig_2, sig_3]."""
        if self.head_0_ is None or self.head_1_ is None:
            raise RuntimeError("TwoStageK4Ensemble must be fitted before predict_std().")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
        X_arr = self.select_model_inputs(X_arr)

        p1 = self.head_1_.predict_proba(X_arr)
        p2 = self.head_2_.predict_proba(X_arr)
        p3 = self.head_3_.predict_proba(X_arr)

        std1 = self.head_1_.predict_std(X_arr)
        std2 = self.head_2_.predict_std(X_arr)
        std3 = self.head_3_.predict_std(X_arr)

        X_head0, _ = self._head0_input(X_arr, p1, p2, p3, coords_m)
        std0 = self.head_0_.predict_std(X_head0)

        return np.column_stack([std0, std1, std2, std3])

    def predict_uncertainty(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        coords_m: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.predict_std(X, coords_m=coords_m) ** 2

    def get_head(self, head: Union[int, str]) -> _BaggedXGBoostHead:
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
        coords_m: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Compute TreeSHAP values for a specific head, returning values and feature names."""
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
        if getattr(self, "keep_idx_", None) is not None and len(base_feature_names) == len(self.feature_names_):
            X_arr = self.select_model_inputs(X_arr)
            base_feature_names = [base_feature_names[i] for i in self.keep_idx_]

        if k_idx == 0:
            p1 = self.head_1_.predict_proba(X_arr)
            p2 = self.head_2_.predict_proba(X_arr)
            p3 = self.head_3_.predict_proba(X_arr)
            X_head0, extra_names = self._head0_input(X_arr, p1, p2, p3, coords_m)
            feat_names = list(base_feature_names) + extra_names
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
    n_bag: int = 10,
    random_state: int = 42,
    prior_ensemble: Optional[TwoStageK4Ensemble] = None,
    coords_m: Optional[np.ndarray] = None,
) -> TwoStageK4Ensemble:
    """Convenience functional interface to fit TwoStageK4Ensemble."""
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
        prior_ensemble=prior_ensemble,
        coords_m=coords_m,
    )
    return ensemble
