"""Kutai Basin K=3 Rank-Stratified Coal Prospectivity Engine (Domain B).

Implements TwoStageK3CoalEnsemble, MacroBasinSequenceGate, prior-corrected Bayesian
optical Weight of Evidence, hierarchical coal-presence conditional rank softmax,
and composite chain-rule TreeSHAP (D2, D5, Bug 4, Bug 5, Bug 6).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.models.metals import _BaggedXGBoostHead, compute_pa_intersection_weights
from src.schema.canonical import (
    COAL_PREVALENCE_PRIORS,
    K3_COAL_HEAD_NAMES,
    verify_no_raw_pixel_leakage,
)

logger = logging.getLogger(__name__)

SUBCANOPY_COAL_FEATURES: List[str] = [
    "sediment_thickness_m",
    "sediment_thickness_globsed",
    "gravity_residual_basin_mgal",
    "gravity_residual_30km_mgal",
    "gravity_bouguer_mgal",
    "bouguer_gravity_mgal",
    "gravity_isostatic_residual",
    "gravity_vertical_gradient",
    "gravity_horizontal_gradient",
    "mag_tilt_derivative",
    "mag_rtp_tilt_derivative",
    "mag_analytic_signal",
    "emag2_analytic_signal",
    "emag2_rtp_magnetics",
    "mag_tmi_nt",
    "mag_first_vertical_derivative",
    "sar_palsar2_hh",
    "sar_palsar2_hv",
    "sar_palsar2_dpr",
    "sar_palsar2_hh_db",
    "sar_palsar2_hv_db",
    "sar_palsar2_hh_hv_ratio",
    "sar_palsar2_texture_glcm",
    "sar_palsar2_lineament_density",
    "sar_bedding_strike_deg",
    "sar_bedding_anisotropy",
    "sar_glcm_anisotropy",
    "dem_strike_persistence",
    "sar_vv",
    "sar_vh",
    "sar_vv_vh_ratio",
    "sar_coherence",
    "sar_texture_energy",
    "dem_elevation",
    "dem_slope",
    "dem_tpi_local",
    "dem_tpi_regional",
    "dem_tpi_syncline_keel",
    "dem_twi",
    "dem_plan_curvature",
    "dem_profile_curvature",
    "dem_Terrain_Ruggedness_Index",
    "terrain_ruggedness_tri",
    "dem_valley_depth",
    "dist_tanjung_fm_km",
    "dist_balikpapan_fm_km",
    "dist_warukin_fm_km",
    "dist_tertiary_coal_boundary_km",
    "dist_marine_deltaic_contact_km",
    "dist_sedimentary_polygon_km",
    "dist_calc_alkaline_intrusive_km",
    "dist_pluton_carbonate_contact_km",
    "dist_major_fault_km",
    "dist_active_faults_m",
    "fault_intersection_density",
    "active_fault_density",
    "structural_corridor_score",
    "upslope_amd_index",
    "amd_upslope_anomaly_censored",
]

BARREN_OPTICAL_COAL_FEATURES: List[str] = [
    "spec_albedo_gated",
    "spec_wcdi_gated",
    "spec_oci_gated",
    "spec_ficr_gated",
    "thermal_anomaly_st_filtered",
    "thermal_anomaly_st",
    "upslope_amd_index",
    "amd_upslope_anomaly_censored",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p_clip = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def fuse_canopy_log_odds(
    log_odds_basin: np.ndarray,
    log_lambda_optical: np.ndarray,
    w_canopy: np.ndarray,
    logit_pi_opt: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """Prior-corrected Bayesian optical Weight of Evidence fusion (D2, Bug 5).

    Evaluates:
        w_opt,c^+(s) = max(0.0, logit(p_opt,c(s)) - logit(pi_opt,c))
        l_fused,c(s) = logit(p_sub,c(s)) + w_canopy(s) * w_opt,c^+(s)

    Subtracting logit(pi_opt,c) ensures that optical anomalies above background
    (even if p_opt < 0.50) provide valid positive evidential lift.
    """
    lo_basin = np.asarray(log_odds_basin, dtype=np.float64)
    ll_opt = np.asarray(log_lambda_optical, dtype=np.float64)
    w_c = np.clip(np.asarray(w_canopy, dtype=np.float64), 0.0, 1.0)
    if w_c.ndim == 1 and lo_basin.ndim == 2:
        w_c = w_c[:, np.newaxis]

    lo_basin_clean = np.where(np.isfinite(lo_basin), lo_basin, 0.0)

    if logit_pi_opt is not None:
        pi_offset = np.asarray(logit_pi_opt, dtype=np.float64)
        ll_adj = ll_opt - pi_offset
    else:
        ll_adj = ll_opt

    ll_opt_pos = np.maximum(0.0, np.where(np.isfinite(ll_adj), ll_adj, 0.0))
    return lo_basin_clean + (w_c * ll_opt_pos)


fuse_bayesian_canopy_log_odds = fuse_canopy_log_odds


class MacroBasinSequenceGate:
    """Regional Sedimentary Basin & Stratigraphic Sequence Gate G_basin(s) in [0, 1].

    Upgraded in v4.0 (Bug 4):
    Replaces dead binary polygon masks with continuous physical features
    (sediment_thickness_m, gravity_residual_basin_mgal, dist_sedimentary_polygon_km,
    sar_moisture_index, mag_analytic_signal).
    """

    def __init__(
        self,
        min_sediment_thickness_m: float = 300.0,
        opt_sediment_thickness_m: float = 1500.0,
        max_basement_slope_deg: float = 25.0,
    ) -> None:
        self.min_sediment_thickness_m = float(min_sediment_thickness_m)
        self.opt_sediment_thickness_m = float(opt_sediment_thickness_m)
        self.max_basement_slope_deg = float(max_basement_slope_deg)

    @staticmethod
    def _get_col(
        X_arr: np.ndarray,
        col_map: dict[str, int],
        name: str,
        default_val: float = 0.0,
    ) -> np.ndarray:
        if name in col_map:
            col = X_arr[:, col_map[name]].astype(np.float64)
            return np.where(np.isfinite(col), col, default_val)
        return np.full(X_arr.shape[0], default_val, dtype=np.float64)

    def compute_gate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Compute continuous basin gate G_basin(s) in [0.02, 1.0]."""
        if isinstance(X, pd.DataFrame):
            names = list(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            names = list(feature_names) if feature_names is not None else []
        col_map = {n: i for i, n in enumerate(names)}

        sed = self._get_col(X_arr, col_map, "sediment_thickness_m", default_val=1200.0)
        grav_res = self._get_col(X_arr, col_map, "gravity_residual_basin_mgal", default_val=-5.0)
        tpi_reg = self._get_col(X_arr, col_map, "dem_tpi_regional", default_val=0.0)
        slope = self._get_col(X_arr, col_map, "dem_slope", default_val=5.0)

        # Continuous physical features replacing dead binary masks (Bug 4)
        dist_sed = self._get_col(X_arr, col_map, "dist_sedimentary_polygon_km", default_val=0.0)
        sar_moist = self._get_col(X_arr, col_map, "sar_moisture_index", default_val=0.0)
        mag_as = self._get_col(X_arr, col_map, "mag_analytic_signal", default_val=0.02)

        sed_score = _sigmoid((sed - self.min_sediment_thickness_m) / 250.0)
        grav_score = _sigmoid((-grav_res + 5.0) / 8.0)
        topo_score = _sigmoid((self.max_basement_slope_deg - slope) / 5.0) * _sigmoid((30.0 - tpi_reg) / 25.0)
        sed_prox_score = _sigmoid((10.0 - dist_sed) / 3.0)

        gate = np.clip(0.40 * sed_score + 0.25 * grav_score + 0.15 * topo_score + 0.20 * sed_prox_score, 0.02, 1.0)

        # Suppress magnetic highs (ophiolite/intrusive basement) and extreme SAR moisture (peat domes)
        gate = np.where(mag_as > 0.08, gate * 0.20, gate)
        gate = np.where(sar_moist > 0.85, gate * 0.25, gate)

        return gate.astype(np.float32)

    def select_reliable_negatives(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[list[str]] = None,
        positive_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Select geological Reliable Negatives (RN_coal) using continuous physical features."""
        if isinstance(X, pd.DataFrame):
            names = list(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            names = list(feature_names) if feature_names is not None else []
        col_map = {n: i for i, n in enumerate(names)}

        sed = self._get_col(X_arr, col_map, "sediment_thickness_m", default_val=1200.0)
        slope = self._get_col(X_arr, col_map, "dem_slope", default_val=5.0)
        dist_intr = self._get_col(X_arr, col_map, "dist_calc_alkaline_intrusive_km", default_val=50.0)
        mag_as = self._get_col(X_arr, col_map, "mag_analytic_signal", default_val=0.02)
        dist_sed = self._get_col(X_arr, col_map, "dist_sedimentary_polygon_km", default_val=0.0)

        rn_mask = (
            (sed < self.min_sediment_thickness_m)
            | (slope > self.max_basement_slope_deg)
            | (mag_as > 0.10)
            | (dist_sed > 15.0)
            | ((dist_intr < 1.5) & (sed < 600.0))
        )
        if positive_mask is not None:
            rn_mask &= ~np.asarray(positive_mask, dtype=bool)
        return rn_mask


class TwoStageK3CoalEnsemble:
    """Two-Stage Sub-Canopy + Barren-Window Multi-Rank Coal Prospectivity Engine (K=3).

    Upgrades in v4.0 (D2, D5, Bug 4, Bug 5, Bug 6):
      - Prior-corrected Bayesian optical Weight of Evidence in log-odds fusion (D2).
      - Continuous feature filtering against SUBCANOPY_COAL_FEATURES (Bug 4).
      - Passing w_canopy_val and warm-start evaluation in fit() (Bug 5).
      - Composite chain-rule TreeSHAP in compute_head_shap() (Bug 6).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        n_estimators: int = 100,
        n_bag: int = 5,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_state: int = 42,
    ) -> None:
        self.config = config or {}
        self.n_estimators = int(n_estimators)
        self.n_bag = max(1, int(n_bag))
        self.learning_rate = float(learning_rate)
        self.max_depth = max(2, int(max_depth))
        self.random_state = int(random_state)

        self.basin_gate_ = MacroBasinSequenceGate()
        self.subcanopy_heads_: dict[int, _BaggedXGBoostHead] = {}
        self.optical_heads_: dict[int, _BaggedXGBoostHead] = {}
        self.pa_weights_: dict[int, np.ndarray] = {}
        self.c_hat_: dict[int, float] = {0: 0.5, 1: 0.5, 2: 0.5}
        self.pi_opt_: dict[int, float] = {0: 0.02, 1: 0.04, 2: 0.05}
        self.feature_names_: list[str] = []
        self.subcanopy_indices_: list[int] = []
        self.optical_indices_: list[int] = []
        self.w_canopy_col_idx_: Optional[int] = None
        eta0 = float(self.learning_rate)
        self.head_base_eta_: dict[int, float] = {0: eta0, 1: eta0, 2: eta0}
        self.best_val_pr_auc_: dict[int, float] = {}
        self.rejected_heads_: dict[int, bool] = {0: False, 1: False, 2: False}

    def save(self, path: Union[Path, str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "TwoStageK3CoalEnsemble":
        return joblib.load(Path(path))

    def _extract_coal_labels_and_weights(
        self,
        y_k3: Union[np.ndarray, dict, list],
        tonnage_weights: Optional[np.ndarray],
        e_pooled: Optional[np.ndarray],
        N: int,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        targets: dict[int, np.ndarray] = {}
        weights: dict[int, np.ndarray] = {}

        if e_pooled is not None:
            e_clip = np.clip(np.asarray(e_pooled, dtype=np.float64).ravel(), 0.05, 1.0)
            inv_prop = 1.0 / e_clip
            inv_prop = inv_prop / max(float(np.mean(inv_prop)), 1e-6)
        else:
            inv_prop = np.ones(N, dtype=np.float64)

        if tonnage_weights is not None:
            w_ton = np.clip(np.asarray(tonnage_weights, dtype=np.float64).ravel(), 1.0, 20.0)
        else:
            w_ton = np.ones(N, dtype=np.float64)

        for c in range(3):
            c_name = K3_COAL_HEAD_NAMES[c]
            if isinstance(y_k3, dict):
                raw_y = y_k3.get(
                    c,
                    y_k3.get(
                        c_name,
                        y_k3.get(f"label_{c_name}", np.zeros(N, dtype=np.int32)),
                    ),
                )
            elif isinstance(y_k3, list):
                raw_y = y_k3[c]
            else:
                arr_y = np.asarray(y_k3)
                if arr_y.ndim == 2:
                    raw_y = arr_y[:, c] if arr_y.shape[1] == 3 else arr_y[c, :]
                else:
                    raw_y = arr_y

            y_bin = (np.asarray(raw_y).ravel() == 1).astype(np.int32)
            targets[c] = y_bin

            w_c = inv_prop.copy()
            pos_mask = (y_bin == 1)
            if np.any(pos_mask):
                w_c[pos_mask] *= w_ton[pos_mask]
            weights[c] = w_c

        return targets, weights

    def _resolve_feature_subsets(self, col_names: list[str], D: int) -> None:
        """Resolve feature indices enforcing SUBCANOPY_COAL_FEATURES (Bug 4)."""
        self.feature_names_ = list(col_names)
        if not col_names:
            self.subcanopy_indices_ = list(range(D))
            self.optical_indices_ = list(range(min(6, D)))
            self.w_canopy_col_idx_ = None
            return

        # Enforce SUBCANOPY_COAL_FEATURES intersection
        sub_idx = [i for i, n in enumerate(col_names) if n in SUBCANOPY_COAL_FEATURES]
        if len(sub_idx) < 5:
            optical_set = set(BARREN_OPTICAL_COAL_FEATURES) | {"w_canopy"}
            sub_idx = [i for i, n in enumerate(col_names) if n not in optical_set]
            if not sub_idx:
                sub_idx = list(range(D))

        opt_idx = [i for i, n in enumerate(col_names) if n in BARREN_OPTICAL_COAL_FEATURES]
        self.subcanopy_indices_ = sub_idx
        self.optical_indices_ = opt_idx
        self.w_canopy_col_idx_ = col_names.index("w_canopy") if "w_canopy" in col_names else None

    def _extract_w_canopy(
        self,
        X_arr: np.ndarray,
        w_canopy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        N = X_arr.shape[0]
        if w_canopy is not None:
            w_c = np.asarray(w_canopy, dtype=np.float64).ravel()
            return np.clip(np.where(np.isfinite(w_c), w_c, 0.0), 0.0, 1.0)
        if self.w_canopy_col_idx_ is not None and self.w_canopy_col_idx_ < X_arr.shape[1]:
            w_c = X_arr[:, self.w_canopy_col_idx_].astype(np.float64)
            return np.clip(np.where(np.isfinite(w_c), w_c, 0.0), 0.0, 1.0)
        return np.zeros(N, dtype=np.float64)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_k3: Union[np.ndarray, dict, list],
        tonnage_weights: Optional[np.ndarray] = None,
        e_pooled: Optional[np.ndarray] = None,
        w_canopy: Optional[np.ndarray] = None,
        w_canopy_val: Optional[np.ndarray] = None,
        spatial_block_ids: Optional[np.ndarray] = None,
        prior_ensemble: Optional["TwoStageK3CoalEnsemble"] = None,
        round_idx: int = 1,
        fold_id: int = 0,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val_k3: Optional[Union[np.ndarray, dict, list]] = None,
        feature_names: Optional[list[str]] = None,
    ) -> "TwoStageK3CoalEnsemble":
        """Fit Stage 1 Sub-Canopy Heads and Stage 2 Barren-Window Optical Calibrator (Bug 5)."""
        if isinstance(X, pd.DataFrame):
            col_names = list(X.columns)
            verify_no_raw_pixel_leakage(col_names)
            X_arr = X.values.astype(np.float32)
        else:
            col_names = list(feature_names) if feature_names is not None else []
            if col_names:
                verify_no_raw_pixel_leakage(col_names)
            X_arr = np.asarray(X, dtype=np.float32)

        N, D = X_arr.shape
        self._resolve_feature_subsets(col_names, D)
        targets, weights = self._extract_coal_labels_and_weights(
            y_k3=y_k3,
            tonnage_weights=tonnage_weights,
            e_pooled=e_pooled,
            N=N,
        )

        eta0 = float(getattr(self, "learning_rate", 0.05))
        depth0 = int(getattr(self, "max_depth", 5))
        if prior_ensemble is not None:
            self.head_base_eta_ = dict(
                getattr(prior_ensemble, "head_base_eta_", {0: eta0, 1: eta0, 2: eta0})
            )
            self.best_val_pr_auc_ = dict(getattr(prior_ensemble, "best_val_pr_auc_", {}))
            self.pi_opt_ = dict(getattr(prior_ensemble, "pi_opt_", {0: 0.02, 1: 0.04, 2: 0.05}))
        else:
            self.head_base_eta_ = {0: eta0, 1: eta0, 2: eta0}
            self.best_val_pr_auc_ = {}
            self.pi_opt_ = {0: 0.02, 1: 0.04, 2: 0.05}
        self.rejected_heads_ = {0: False, 1: False, 2: False}

        any_pos = (targets[0] == 1) | (targets[1] == 1) | (targets[2] == 1)
        rn_mask = self.basin_gate_.select_reliable_negatives(
            X_arr, feature_names=col_names, positive_mask=any_pos
        )

        X_sub = X_arr[:, self.subcanopy_indices_]
        sub_names = [col_names[i] for i in self.subcanopy_indices_] if col_names else []

        rank_priors: dict[int, set[str]] = {
            0: {
                "dist_tanjung_fm_km",
                "dist_calc_alkaline_intrusive_km",
                "mag_rtp_tilt_derivative",
                "sar_bedding_anisotropy",
                "sediment_thickness_m",
                "gravity_residual_basin_mgal",
            },
            1: {
                "dist_balikpapan_fm_km",
                "dist_tertiary_coal_boundary_km",
                "sar_bedding_anisotropy",
                "sediment_thickness_m",
                "gravity_residual_basin_mgal",
                "dem_tpi_regional",
            },
            2: {
                "dist_warukin_fm_km",
                "dist_tertiary_coal_boundary_km",
                "sediment_thickness_m",
                "gravity_residual_basin_mgal",
                "dem_tpi_regional",
                "dem_twi",
            },
        }

        w_c_train = self._extract_w_canopy(X_arr, w_canopy=w_canopy)

        for c in range(3):
            y_c = targets[c]
            w_c = weights[c].copy()
            if np.any(rn_mask):
                w_c[rn_mask & (y_c == 0)] *= 1.35

            pa_w = compute_pa_intersection_weights(X_sub, y_c)
            if sub_names and len(sub_names) == X_sub.shape[1]:
                for j, sname in enumerate(sub_names):
                    if sname in rank_priors[c]:
                        pa_w[j] = max(pa_w[j], 1.5 / max(len(sub_names), 1)) * 1.30
                pa_sum = float(np.sum(pa_w))
                if pa_sum > 0:
                    pa_w = pa_w / pa_sum
            self.pa_weights_[c] = pa_w

            prior_sub_models = None
            if prior_ensemble is not None and c in getattr(prior_ensemble, "subcanopy_heads_", {}):
                prior_sub_models = prior_ensemble.subcanopy_heads_[c].models_

            head_sub = _BaggedXGBoostHead(
                n_estimators=self.n_estimators,
                max_depth=depth0,
                learning_rate=self.head_base_eta_.get(c, eta0),
                min_child_weight=3.0,
                max_delta_step=1.0,
                tree_method="hist",
                n_bag=self.n_bag,
                feature_weights=pa_w,
                random_state=self.random_state + 100 + c * 17,
            )
            head_sub.fit(
                X_sub,
                y_c,
                sample_weight=w_c,
                prior_models=prior_sub_models,
                round_idx=round_idx,
                fold_id=fold_id,
            )
            self.subcanopy_heads_[c] = head_sub

            # Stage 2: Barren-Window Optical Calibration Head
            if len(self.optical_indices_) > 0:
                X_opt = X_arr[:, self.optical_indices_]
                barren_mask = w_c_train > 0.05
                if int(np.sum(barren_mask & (y_c == 1))) >= 3 and int(np.sum(barren_mask & (y_c == 0))) >= 10:
                    fit_mask = barren_mask
                else:
                    fit_mask = np.ones(N, dtype=bool)

                # Prior probability in optical subset (D2)
                n_opt_pos = int(np.sum(y_c[fit_mask] == 1))
                n_opt_tot = int(np.sum(fit_mask))
                pi_opt_c = float(np.clip(n_opt_pos / max(n_opt_tot, 1), 1e-4, 0.50))
                if n_opt_pos == 0:
                    c_name = K3_COAL_HEAD_NAMES[c]
                    pi_opt_c = COAL_PREVALENCE_PRIORS.get(c_name, 0.03)
                self.pi_opt_[c] = pi_opt_c

                prior_opt_models = None
                if prior_ensemble is not None and c in getattr(prior_ensemble, "optical_heads_", {}):
                    prior_opt_models = prior_ensemble.optical_heads_[c].models_

                head_opt = _BaggedXGBoostHead(
                    n_estimators=max(25, self.n_estimators // 2),
                    max_depth=3,
                    learning_rate=0.04,
                    min_child_weight=5.0,
                    max_delta_step=1.0,
                    tree_method="hist",
                    n_bag=max(1, self.n_bag // 2),
                    random_state=self.random_state + 200 + c * 17,
                )
                head_opt.fit(
                    X_opt[fit_mask],
                    y_c[fit_mask],
                    sample_weight=w_c[fit_mask],
                    prior_models=prior_opt_models,
                    round_idx=round_idx,
                    fold_id=fold_id,
                )
                self.optical_heads_[c] = head_opt

            pos_idx = np.where(y_c == 1)[0]
            if len(pos_idx) > 0:
                p_sub_pos = head_sub.predict_proba(X_sub[pos_idx])
                self.c_hat_[c] = float(np.clip(np.mean(p_sub_pos), 0.15, 1.0))
            else:
                self.c_hat_[c] = 1.0

        # Validation checkpoint gate (Bug 5: pass w_canopy_val and evaluate prior_ensemble)
        if X_val is not None and y_val_k3 is not None:
            from src.evaluation.metrics import pr_auc as _eval_pr_auc

            val_N = len(X_val)
            val_targets, _ = self._extract_coal_labels_and_weights(
                y_k3=y_val_k3, tonnage_weights=None, e_pooled=None, N=val_N
            )
            val_preds = self.predict_proba(X_val, w_canopy=w_canopy_val)

            for c in range(3):
                if np.sum(val_targets[c] == 1) > 0 and np.sum(val_targets[c] == 0) > 0:
                    curr_auc = float(_eval_pr_auc(val_targets[c], val_preds[:, c]))
                    prev_best = self.best_val_pr_auc_.get(c)
                    if prior_ensemble is not None and prev_best is None:
                        prev_val_preds = prior_ensemble.predict_proba(X_val, w_canopy=w_canopy_val)
                        prev_best = float(_eval_pr_auc(val_targets[c], prev_val_preds[:, c]))

                    if prior_ensemble is not None and prev_best is not None and curr_auc < prev_best - 1e-4:
                        self.rejected_heads_[c] = True
                        self.subcanopy_heads_[c] = prior_ensemble.subcanopy_heads_[c]
                        if c in prior_ensemble.optical_heads_:
                            self.optical_heads_[c] = prior_ensemble.optical_heads_[c]
                        self.head_base_eta_[c] = max(0.01, self.head_base_eta_.get(c, 0.05) * 0.5)
                        self.best_val_pr_auc_[c] = prev_best
                    else:
                        self.best_val_pr_auc_[c] = max(prev_best if prev_best is not None else 0.0, curr_auc)

        return self

    def predict_full_outputs(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        w_canopy: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Compute sub-canopy log-odds, prior-corrected optical WoE, and fused outputs."""
        if not self.subcanopy_heads_:
            raise RuntimeError("TwoStageK3CoalEnsemble is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            verify_no_raw_pixel_leakage(list(X.columns))
            X_arr = X.values.astype(np.float32)
            col_names = list(X.columns)
        else:
            if self.feature_names_:
                verify_no_raw_pixel_leakage(self.feature_names_)
            X_arr = np.asarray(X, dtype=np.float32)
            col_names = self.feature_names_

        N = X_arr.shape[0]
        X_sub = X_arr[:, self.subcanopy_indices_]
        w_c = self._extract_w_canopy(X_arr, w_canopy=w_canopy)
        basin_gate = self.basin_gate_.compute_gate(X_arr, feature_names=col_names)

        log_odds_basin = np.zeros((N, 3), dtype=np.float64)
        log_lambda_opt = np.zeros((N, 3), dtype=np.float64)
        log_odds_final = np.zeros((N, 3), dtype=np.float64)
        probs_k3 = np.zeros((N, 3), dtype=np.float64)
        probs_raw_k3 = np.zeros((N, 3), dtype=np.float64)
        std_k3 = np.zeros((N, 3), dtype=np.float64)

        has_opt = len(self.optical_indices_) > 0
        X_opt = X_arr[:, self.optical_indices_] if has_opt else None

        for c in range(3):
            head_sub = self.subcanopy_heads_[c]
            p_sub = np.clip(head_sub.predict_proba(X_sub), 1e-5, 1.0 - 1e-5)
            # D5: Sub-canopy log-odds without premature basin gate attenuation
            lo_basin_c = _logit(p_sub)
            log_odds_basin[:, c] = lo_basin_c
            std_sub_c = head_sub.predict_std(X_sub)

            pi_opt_c = self.pi_opt_.get(c, 0.03)
            logit_pi_c = _logit(np.array([pi_opt_c]))[0]

            if has_opt and c in self.optical_heads_ and X_opt is not None:
                p_opt = np.clip(self.optical_heads_[c].predict_proba(X_opt), 1e-5, 1.0 - 1e-5)
                # D2: Prior-corrected optical Weight of Evidence
                ll_opt_c = _logit(p_opt)
                std_opt_c = self.optical_heads_[c].predict_std(X_opt)
                var_fused_c = ((1.0 - w_c) ** 2) * (std_sub_c ** 2) + (w_c ** 2) * (std_opt_c ** 2)
                std_k3[:, c] = np.sqrt(np.maximum(0.0, var_fused_c))
            else:
                ll_opt_c = np.zeros(N, dtype=np.float64)
                std_k3[:, c] = std_sub_c

            log_lambda_opt[:, c] = ll_opt_c
            lo_final_c = fuse_canopy_log_odds(lo_basin_c, ll_opt_c, w_c, logit_pi_opt=logit_pi_c)
            log_odds_final[:, c] = lo_final_c

            p_raw_c = _sigmoid(lo_final_c)
            probs_raw_k3[:, c] = np.clip(np.where(np.isfinite(p_raw_c), p_raw_c, 0.0), 0.0, 1.0)
            c_hat = max(0.15, float(self.c_hat_.get(c, 0.5)))
            p_cal_c = 0.65 * p_raw_c + 0.35 * np.minimum(1.0, p_raw_c / c_hat)
            probs_k3[:, c] = np.clip(np.where(np.isfinite(p_cal_c), p_cal_c, 0.0), 0.0, 1.0)

        p_coal_any = 1.0 - np.prod(1.0 - probs_k3, axis=1)
        p_coal_any = np.clip(np.where(np.isfinite(p_coal_any), p_coal_any, 0.0), 0.0, 1.0)
        expected_class = np.argmax(probs_k3, axis=1).astype(np.int32)

        return {
            "probs_k3": probs_k3.astype(np.float32),
            "probs_raw_k3": probs_raw_k3.astype(np.float32),
            "p_coal_any": p_coal_any.astype(np.float32),
            "expected_calorific_class": expected_class,
            "std_k3": np.where(np.isfinite(std_k3), std_k3, 0.0).astype(np.float32),
            "log_odds_basin_k3": log_odds_basin.astype(np.float32),
            "log_lambda_optical_k3": log_lambda_opt.astype(np.float32),
            "log_odds_final_k3": log_odds_final.astype(np.float32),
            "basin_gate": basin_gate.astype(np.float32),
        }

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        w_canopy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.predict_full_outputs(X, w_canopy=w_canopy)["probs_k3"]

    def predict_proba_raw(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        w_canopy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.predict_full_outputs(X, w_canopy=w_canopy)["probs_raw_k3"]

    def predict_composite_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        w_canopy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.predict_full_outputs(X, w_canopy=w_canopy)["p_coal_any"]

    def predict_expected_calorific_class(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        w_canopy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.predict_full_outputs(X, w_canopy=w_canopy)["expected_calorific_class"]

    def predict_std(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        w_canopy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.predict_full_outputs(X, w_canopy=w_canopy)["std_k3"]

    def compute_head_shap(
        self,
        head: Union[int, str],
        X: Union[np.ndarray, pd.DataFrame],
        base_feature_names: Optional[list[str]] = None,
        w_canopy: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Compute composite chain-rule TreeSHAP values for head c (Bug 6).

        phi_j^{(c)}(s) = phi_j^{(sub, c)}(s) + w_canopy(s) * I(p_opt,c(s) > pi_opt,c) * phi_j^{(opt, c)}(s)

        Args:
            head: Head index (0, 1, 2) or name in K3_COAL_HEAD_NAMES.
            X: Input feature matrix or DataFrame.
            base_feature_names: Optional base feature names.
            w_canopy: Optional canopy transmittance weights.

        Returns:
            Tuple (shap_values, feature_names).
        """
        c = head if isinstance(head, int) else (
            K3_COAL_HEAD_NAMES.index(head) if head in K3_COAL_HEAD_NAMES else 0
        )
        if c not in self.subcanopy_heads_:
            raise RuntimeError(f"Coal head {c} is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            col_names = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            col_names = list(base_feature_names) if base_feature_names is not None else self.feature_names_
            X_arr = np.asarray(X, dtype=np.float32)

        single = (X_arr.ndim == 1)
        if single:
            X_arr = X_arr[np.newaxis, :]

        N = X_arr.shape[0]
        w_c = self._extract_w_canopy(X_arr, w_canopy=w_canopy)

        X_sub = X_arr[:, self.subcanopy_indices_]
        sub_shap = self.subcanopy_heads_[c].compute_shap(X_sub)
        sub_names = [col_names[i] for i in self.subcanopy_indices_] if col_names else [f"sub_{i}" for i in range(X_sub.shape[1])]

        if len(self.optical_indices_) > 0 and c in self.optical_heads_:
            X_opt = X_arr[:, self.optical_indices_]
            opt_shap = self.optical_heads_[c].compute_shap(X_opt)
            opt_names = [col_names[i] for i in self.optical_indices_] if col_names else [f"opt_{i}" for i in range(X_opt.shape[1])]
            p_opt = np.clip(self.optical_heads_[c].predict_proba(X_opt), 1e-5, 1.0 - 1e-5)
            pi_opt = self.pi_opt_.get(c, 0.03)

            # Indicator I(p_opt > pi_opt)
            active_mask = (p_opt > pi_opt).astype(np.float64)[:, np.newaxis]
            w_c_mat = w_c[:, np.newaxis]
            weighted_opt_shap = w_c_mat * active_mask * opt_shap

            all_names = list(sub_names) + list(opt_names)
            combined_shap = np.column_stack([sub_shap, weighted_opt_shap])
        else:
            all_names = list(sub_names)
            combined_shap = sub_shap

        if single and combined_shap.ndim == 2:
            combined_shap = combined_shap[0]

        return combined_shap, all_names


def compute_hierarchical_coal_probabilities(
    p_cal_coal_any: np.ndarray,
    rank_logits: np.ndarray,
    basin_gate: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute hierarchical coal prospectivity with conditional rank softmax (D5).

    Evaluates:
        P_coal_total(s) = G_basin(s) * P_cal(coal_any | x_s)
        P(rank = c | coal, s) = exp(z_c / T) / sum_c' exp(z_c' / T)
        P_c(s) = P_coal_total(s) * P(rank = c | coal, s)

    Args:
        p_cal_coal_any: 1-D array of calibrated composite coal probabilities.
        rank_logits: (N, 3) array of uncalibrated rank logits or fused log-odds.
        basin_gate: 1-D array of regional sedimentary sequence gate G_basin(s) in [0, 1].
        temperature: Softmax temperature parameter T (default 1.0).

    Returns:
        Tuple (p_k3_final, p_coal_total) where p_k3_final is (N, 3) and p_coal_total is (N,).
    """
    p_any = np.asarray(p_cal_coal_any, dtype=np.float64).ravel()
    g_basin = np.clip(np.asarray(basin_gate, dtype=np.float64).ravel(), 0.0, 1.0)
    logits = np.asarray(rank_logits, dtype=np.float64)

    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError(f"rank_logits must have shape (N, 3), got {logits.shape}")

    # Total coal occurrence prospectivity gated by regional basin sequence
    p_coal_total = np.clip(g_basin * p_any, 0.0, 1.0)

    # Temperature-scaled softmax across rank logits for mutually exclusive rank partitioning
    t_val = max(float(temperature), 0.05)
    max_logit = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp((logits - max_logit) / t_val)
    rank_softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    p_k3_final = p_coal_total[:, np.newaxis] * rank_softmax
    return np.clip(p_k3_final, 0.0, 1.0), p_coal_total
