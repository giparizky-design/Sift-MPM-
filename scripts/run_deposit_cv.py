#!/usr/bin/env python3
"""Deposit-level, buffered spatial cross-validation of the production ensembles.

For each fold, held-out deposit clusters and every cell within ``--buffer-m``
of them are removed from training. The model is retrained from zero with the
production preprocessing (feature selection, 15 km distance caps, logistics
propensity, exclusion masks, Hájek head weights, OUTMOGC gates, Elkan-Noto
scaling) fitted on the training cells only. Each held-out deposit is then
ranked against the background cells of the map. See ``src/evaluation/deposit_cv.py``.

v4.8 additions (finding new ground):
  * novelty metrics: share of the top 1/5/10% of the map lying more than 5 km / 15 km
    from ANY known label, against the base rate of the map;
  * frontier AUC: success rate over held-out deposits more than ``--frontier-m`` from
    the nearest training deposit;
  * switches mirroring each candidate fix (``--weight-scheme``, ``--gate``,
    ``--head0-stacking``, ``--min-child-weight``, ``--coal-disk-mode``, ``--coal-buffer-m``).

v4.9 addition: ``--models unet`` trains the production DAE U-Net (``MPMDistrictDAEUNet``
architecture, loss, subsampling, decimation and input cube) once per fold on that fold's
training cells only, next to the production XGBoost, and reports the U-Net alone and the
production blend ``w_xgb*s_xgb + w_unet*s_unet``. ``w_unet`` is chosen on the training folds
only, by an inner deposit-level CV over the remaining folds (see ``--unet-inner``).

What this does NOT reproduce from ``run_seven_pass_training.py``:
  * the validation-fold Elkan-Noto ĉ (the ensemble's own in-sample ĉ is used),
  * warm-starting from a prior checkpoint (every fold trains from zero).

Example:
    .venv/bin/python scripts/run_deposit_cv.py \
        --data-dir data/east_kalimantan/processed --domain metals \
        --out-dir outputs/deposit_cv/east_kalimantan_metals
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# U-Net runs load torch next to XGBoost/sklearn, which ship their own libomp. On this machine,
# xgboost-then-torch deadlocks in the first CPU torch op and torch-then-xgboost segfaults in a
# multithreaded XGBoost fit (seen in v4.9). run_seven_pass_training.py avoids both with
# OMP_NUM_THREADS=1 and importing torch first; do the same, but only when the U-Net is requested,
# so other harness runs keep multithreaded XGBoost.
if any("unet" in a for a in sys.argv[1:]):
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch  # noqa: F401,E402

    torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.deposit_cv import (  # noqa: E402
    DEFAULT_CAPTURE_LEVELS,
    assign_cluster_folds,
    deposit_area_fractions,
    label_deposit_clusters,
    success_rate_curve,
    summarize_clusters,
    summarize_success_rate,
    training_exclusion_mask,
)
from src.labels.pu_negatives import (  # noqa: E402
    CANONICAL_PROPENSITY_COLS,
    ExplorationPropensity,
    OUTMOGCGate,
    compute_asymmetric_head_weights,
    compute_k4_exclusion_mask,
)
from src.labels.spatial_splits import assign_spatial_blocks  # noqa: E402
from src.models.calibration import calibrate_closed_boundary_elkan_noto  # noqa: E402
from src.schema.canonical import (  # noqa: E402
    EXTENDED_NON_FEATURE_COLS,
    K3_COAL_HEAD_NAMES,
    K4_HEAD_NAMES,
    NON_FEATURE_COLS as CANONICAL_NON_FEATURE_COLS,
    _FORBIDDEN_RAW_BAND_PATTERNS,
    verify_no_raw_pixel_leakage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("deposit_cv")

K4_LABEL_COLS = ["label_k0_porphyry_cu", "label_k1_skarn_fe_cu", "label_k2_epithermal_au", "label_k3_laterite_fe_ni"]
K3_LABEL_COLS = ["label_coal_k0_high_cv", "label_coal_k1_med_cv", "label_coal_k2_low_cv"]

# Mirrors the local NON_FEATURE_COLS set in run_seven_pass_training.run_region_seven_pass (L692-L702).
ORCHESTRATOR_EXTRA_NON_FEATURE = {
    "alteration_benchmark", "commodity_label", "is_centroid_artifact", "target_class", "w_canopy",
    "geology_coal_contact_dist_km", "in_coal_bearing_formation", "in_holocene_peat_or_karst_mask",
    "in_ophiolite_unit",
}
# v4.15: the same propensity covariates as the orchestrator (roads, rivers, ports), in model order.
ACCESS_COLS = tuple(CANONICAL_PROPENSITY_COLS)


# ---------------------------------------------------------------------------
# Data preparation (mirrors Pass 1 of the orchestrator)
# ---------------------------------------------------------------------------

def select_feature_cols(columns: list[str]) -> list[str]:
    non_feature = set(EXTENDED_NON_FEATURE_COLS) | set(CANONICAL_NON_FEATURE_COLS) | ORCHESTRATOR_EXTRA_NON_FEATURE
    return [
        c for c in columns
        if c not in non_feature
        and not c.startswith("label_")
        and not any(pat in c.lower() for pat in _FORBIDDEN_RAW_BAND_PATTERNS)
    ]


def clip_distance_cols(df: pd.DataFrame, cols: list[str]) -> None:
    """Cap distance channels at 15 km, as the orchestrator does before training."""
    for c in cols:
        if c in df.columns and (c.startswith("dist_") or c.endswith("_dist_km")):
            hi = 15000.0 if c.endswith("_m") else 15.0
            df[c] = np.clip(df[c].to_numpy(dtype=np.float32), 0.0, hi)


def load_grid(data_dir: Path) -> tuple[dict, int, int, float, bool]:
    import rasterio
    from rasterio.crs import CRS

    ref = data_dir / "dem_elevation.tif"
    if not ref.exists():
        raise FileNotFoundError(f"{ref} is needed for the grid profile")
    with rasterio.open(ref) as src:
        profile = dict(src.profile)
        H, W = int(src.height), int(src.width)
    is_geo = CRS.from_user_input(profile["crs"]).is_geographic
    dx = abs(float(profile["transform"].a))
    px_size_m = dx * 111_320.0 if is_geo else dx
    return profile, H, W, px_size_m, is_geo


# ---------------------------------------------------------------------------
# Model adapters. Each returns {variant_name: (N, K) head scores, plus an "any" column}.
# ---------------------------------------------------------------------------

class FoldData:
    """Training sample for one fold, plus what adapters need to rebuild production inputs."""

    def __init__(self, X: np.ndarray, y_heads: np.ndarray, pos_any_other: np.ndarray, rows: np.ndarray,
                 cols: np.ndarray, explored: np.ndarray, access: np.ndarray, w_canopy: np.ndarray,
                 tonnage: np.ndarray, coords: np.ndarray, block_ids: np.ndarray,
                 neg_excl: Optional[np.ndarray] = None):
        self.X, self.y_heads, self.pos_any_other = X, y_heads, pos_any_other
        self.rows, self.cols, self.explored, self.access = rows, cols, explored, access
        self.w_canopy, self.tonnage, self.coords, self.block_ids = w_canopy, tonnage, coords, block_ids
        # Unlabeled training cells that must not act as negatives (coal disks / buffer).
        self.neg_excl = np.zeros(len(X), bool) if neg_excl is None else neg_excl

    @property
    def pos_domain(self) -> np.ndarray:
        return np.any(self.y_heads == 1, axis=1)


def _propensity_model(fd: FoldData) -> ExplorationPropensity:
    y = fd.explored | fd.pos_domain | fd.pos_any_other
    return ExplorationPropensity(columns=ACCESS_COLS, random_state=42).fit(fd.access, y)


def fit_propensity(fd: FoldData, access_other: Optional[np.ndarray] = None):
    """Logistics exploration propensity e(x) on training cells (orchestrator Pass 2).

    With ``access_other``, also returns e(x) for those rows (same fitted model). ``fd.access`` columns
    are ``ACCESS_COLS`` in order.
    """
    model = _propensity_model(fd)
    e_tr = model.score(fd.access)
    e_o = None if access_other is None else model.score(access_other)
    return e_tr if access_other is None else (e_tr, e_o)


class ProductionMetals:
    name = "xgb_prod"
    variants = ("raw", "calibrated")

    def __init__(self, args, feature_cols, is_geo):
        self.args, self.feature_cols, self.is_geo = args, feature_cols, is_geo

    def fit(self, fd: FoldData) -> None:
        from src.models.pu_pipeline import TwoStageK4Ensemble

        e_x = fit_propensity(fd)
        y = fd.y_heads
        pos_coords = {k: fd.coords[y[:, k] == 1] for k in range(4)}
        sc = float(self.args.k4_exclusion_scale)
        if sc <= 0:
            excl = np.ones((len(y), 4))
        else:
            radii = None if sc == 1.0 else {0: 6000.0 * sc, 1: 3500.0 * sc, 2: 6000.0 * sc,
                                            3: (10000.0 * sc, 2000.0 * sc, 45.0)}
            excl = compute_k4_exclusion_mask(coords=fd.coords, positive_coords_k=pos_coords, labels_k4=y,
                                             is_geographic=self.is_geo, r_k_star_m=radii)
        hajek = self.args.weight_scheme == "hajek_neutral"
        w = np.column_stack([
            compute_asymmetric_head_weights(labels_k=y[:, k], e_logistics=e_x, exclusion_mask_k=excl[:, k],
                                            hajek_normalize=hajek)
            for k in range(4)
        ])
        self.ens = TwoStageK4Ensemble(n_estimators=self.args.n_estimators, n_bag=self.args.n_bag,
                                      learning_rate=self.args.lr, max_depth=self.args.max_depth or 6,
                                      random_state=42, head0_stacking=self.args.head0_stacking == "on",
                                      min_child_weight=self.args.min_child_weight)
        self.ens.fit(X=fd.X, y_k4=y, sample_weights_k4=w, e_logistics=e_x, exclusion_masks=excl,
                     spatial_block_ids=fd.block_ids, feature_names=self.feature_cols)
        self.gates = []
        for k in range(4):
            Xp = fd.X[y[:, k] == 1]
            if len(Xp) >= 5 and self.args.gate != "off":
                g = OUTMOGCGate(n_pca_components=min(4, Xp.shape[1]),
                                n_gmm_components=min(2, max(1, len(Xp) // 10)), random_state=42,
                                standardize=self.args.gate == "standardized")
                g.fit(Xp, X_reference=fd.X[y[:, k] != 1] if self.args.gate == "standardized" else None)
                self.gates.append(g)
            else:
                self.gates.append(None)
        self.n_train_pos = (y == 1).sum(axis=0)

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        raw = np.asarray(self.ens.predict_proba(X), dtype=np.float64)
        cal = np.zeros_like(raw)
        c_hat = getattr(self.ens, "c_hat_", {}) or {}
        for k in range(4):
            if self.n_train_pos[k] == 0:
                continue
            gate = self.gates[k].score(X) if self.gates[k] is not None else None
            cal[:, k] = calibrate_closed_boundary_elkan_noto(
                p_raw_k=raw[:, k], c_hat_k=float(np.clip(c_hat.get(k, 0.35), 0.10, 1.0)),
                gamma_closed_k=gate, min_c=0.10)
        return {"raw": np.column_stack([raw, raw.max(axis=1)]),
                "calibrated": np.column_stack([cal, cal.max(axis=1)])}


class ProductionCoal:
    name = "xgb_prod"
    variants = ("raw", "hierarchical", "gated")

    def __init__(self, args, feature_cols, is_geo):
        self.args, self.feature_cols = args, feature_cols

    def fit(self, fd: FoldData) -> None:
        from src.models.pu_pipeline import TwoStageK3CoalEnsemble

        e_x = fit_propensity(fd)
        self.ens = TwoStageK3CoalEnsemble(
            n_estimators=self.args.n_estimators, n_bag=self.args.n_bag, learning_rate=self.args.lr,
            max_depth=self.args.max_depth or 5, random_state=42,
            unlabeled_weighting="neutral" if self.args.weight_scheme == "hajek_neutral" else "inverse_propensity",
            min_child_weight=self.args.min_child_weight if self.args.min_child_weight is not None else 3.0)
        self.ens.fit(X=fd.X, y_k3=fd.y_heads, tonnage_weights=fd.tonnage, e_pooled=e_x, w_canopy=fd.w_canopy,
                     spatial_block_ids=fd.block_ids, feature_names=self.feature_cols,
                     negative_exclusion_mask=fd.neg_excl)

    def predict(self, X: np.ndarray, w_canopy: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
        out = self.ens.predict_full_outputs(X, w_canopy=w_canopy)
        raw = np.asarray(out["probs_raw_k3"], dtype=np.float64)
        hier = np.asarray(out["probs_k3"], dtype=np.float64)
        p_any = np.asarray(out["p_coal_any"], dtype=np.float64)
        gate = np.asarray(out["basin_gate"], dtype=np.float64)
        # "gated" mirrors production Pass 4B: P_coal_total = G_basin * P(coal_any)
        return {"raw": np.column_stack([raw, raw.max(axis=1)]),
                "hierarchical": np.column_stack([hier, p_any]),
                "gated": np.column_stack([hier * gate[:, None], p_any * gate])}


class LogRegBaseline:
    """Standardized, L2-regularized logistic regression per head and for any-positive."""
    name = "logreg"
    variants = ("default",)

    def __init__(self, args, feature_cols, is_geo):
        pass

    def fit(self, fd: FoldData) -> None:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        targets = [fd.y_heads[:, k] == 1 for k in range(fd.y_heads.shape[1])] + [fd.pos_domain]
        self.models = []
        for t in targets:
            if t.sum() < 2:
                self.models.append(None)
                continue
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42))
            m.fit(fd.X, t.astype(int))
            self.models.append(m)

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        cols = [m.predict_proba(X)[:, 1] if m is not None else np.zeros(len(X)) for m in self.models]
        return {"default": np.column_stack(cols)}


class SingleFeatureBaseline:
    """Best single feature (and sign) chosen by training-set ROC-AUC, per target."""
    name = "single_feature"
    variants = ("default",)

    def __init__(self, args, feature_cols, is_geo):
        self.feature_cols = feature_cols

    def fit(self, fd: FoldData) -> None:
        from scipy.stats import rankdata

        self.med = np.nanmedian(fd.X, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Xf = np.where(np.isfinite(fd.X), fd.X, self.med)
        ranks = np.column_stack([rankdata(Xf[:, j]) for j in range(Xf.shape[1])])
        targets = [fd.y_heads[:, k] == 1 for k in range(fd.y_heads.shape[1])] + [fd.pos_domain]
        self.choice = []
        for t in targets:
            n1, n0 = int(t.sum()), int((~t).sum())
            if n1 < 2:
                self.choice.append(None)
                continue
            auc = (ranks[t].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
            j = int(np.argmax(np.abs(auc - 0.5)))
            self.choice.append((j, 1.0 if auc[j] >= 0.5 else -1.0, float(auc[j])))

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        cols = []
        for ch in self.choice:
            if ch is None:
                cols.append(np.zeros(len(X)))
            else:
                j, s, _ = ch
                cols.append(s * np.where(np.isfinite(X[:, j]), X[:, j], self.med[j]))
        return {"default": np.column_stack(cols)}

    def describe(self) -> list:
        return [None if c is None else {"feature": self.feature_cols[c[0]], "sign": c[1], "train_auc": c[2]}
                for c in self.choice]


# ---------------------------------------------------------------------------
# DAE U-Net adapter (v4.9). Mirrors Pass 3A.2 / 3B.2 of run_seven_pass_training.py and
# MPMDistrictDAEUNet.cross_validate_spatial_blocks, but trains ONE network per fold on that
# fold's training cells (held-out deposits, their buffer and withheld deposits removed),
# instead of the production block-OOF scheme that ignores the deposit buffer.
# ---------------------------------------------------------------------------

UNET_W_GRID: tuple[float, ...] = tuple(round(float(x), 2) for x in np.linspace(0.0, 1.0, 11))


def production_dae_decimate_factor(H: int, W: int) -> int:
    """Orchestrator rule (run_seven_pass_training.py, U-Net cube normalisation block)."""
    return max(1, min(H, W) // 360) if min(H, W) <= 400 else max(5, min(H, W) // 360)


class UNetCube:
    """Production U-Net input cube.

    Channels are the feature-stack layers that are model features, in stack order, as in the
    orchestrator. Distance channels are capped at 15 km. Falls back to the parquet values
    (invalid cells NaN) when a feature is missing from the stack.
    """

    def __init__(self, data_dir: Path, feature_cols: list[str], H: int, W: int,
                 rows: np.ndarray, cols: np.ndarray, X_all: np.ndarray):
        import joblib

        self.H, self.W = H, W
        self.source = "parquet"
        names = list(feature_cols)
        cube = None
        sp = data_dir / "feature_stack.joblib"
        if sp.exists():
            st = joblib.load(sp)
            sn = list(getattr(st, "feature_names", []) or [])
            arr = getattr(st, "array", None)
            if arr is not None and tuple(arr.shape[1:]) == (H, W) and all(c in sn for c in feature_cols):
                keep = set(feature_cols)
                names = [c for c in sn if c in keep]
                cube = np.stack([np.asarray(arr[sn.index(c)], dtype=np.float32) for c in names])
                self.source = "feature_stack"
            del st, arr
        if cube is None:
            cube = np.full((len(names), H, W), np.nan, dtype=np.float32)
            for j, c in enumerate(names):
                cube[j, rows, cols] = X_all[:, feature_cols.index(c)]
        for j, c in enumerate(names):
            # Orchestrator caps by channel index into feature_cols; identical when the stack order
            # equals the feature order (true for the production lists). Capped by name here.
            if c.startswith("dist_") or c.endswith("_dist_km"):
                cube[j] = np.clip(cube[j], 0.0, 15000.0 if c.endswith("_m") else 15.0)
        self.names, self.cube = names, cube

    def fit_predict(self, *, n_classes: int, y_2d: np.ndarray, elig_2d: np.ndarray, norm_rows: np.ndarray,
                    norm_cols: np.ndarray, w_2d: np.ndarray, decimate: int, epochs: int, px_size_m: float,
                    window_px: int, seed: int, batch_size: int = 64, lr: float = 1e-4) -> tuple[np.ndarray, dict]:
        """Train one DistrictDAEUNet on eligible cells and return native-grid head probabilities (K, H, W)."""
        import torch
        from torch import nn

        from src.models.unet_model import (DistrictDAEUNet, decimate_spatial_grid, get_best_device, nnPULoss,
                                           project_spatial_grid)

        t0 = time.time()
        # cross_validate_spatial_blocks hard-codes native_res_m=50, so the zoom is 1/decimate.
        tres = 50.0 * decimate
        coarse = []
        for j in range(len(self.names)):
            band = self.cube[j]
            v = band[norm_rows, norm_cols]
            v = v[np.isfinite(v) & (v > -9000.0)]
            m = float(v.mean()) if v.size else 0.0
            s = float(v.std()) + 1e-6 if v.size else 1.0
            valid = np.isfinite(band) & (band > -9000.0)
            coarse.append(decimate_spatial_grid(np.where(valid, (band - m) / s, 0.0).astype(np.float32),
                                                target_res_m=tres, native_res_m=50.0, order=1))
        Xc = np.stack(coarse).astype(np.float32)
        del coarse
        t_prep = time.time() - t0
        yc = decimate_spatial_grid(y_2d.astype(np.float32), target_res_m=tres, native_res_m=50.0, order=0)
        ec = decimate_spatial_grid(np.where(elig_2d, 0, -1).astype(np.int32), target_res_m=tres,
                                   native_res_m=50.0, order=0)
        wc = decimate_spatial_grid(w_2d.astype(np.float32), target_res_m=tres, native_res_m=50.0, order=1)
        C, Hc, Wc = Xc.shape
        y_flat = yc.reshape(n_classes, -1).T
        w_flat = np.repeat(wc.ravel()[:, None], n_classes, axis=1)

        # Production subsample (cross_validate_spatial_blocks): <=1500 positives, 2x unlabeled (200..3000)
        train_idx = np.where(ec.ravel() >= 0)[0]
        is_pos = np.any(y_flat[train_idx] > 0, axis=1)
        rng = np.random.default_rng(seed)
        pos_idx, unl_idx = train_idx[is_pos], train_idx[~is_pos]
        sub_pos = rng.choice(pos_idx, size=1500, replace=False) if len(pos_idx) > 1500 else pos_idx
        n_u = min(3000, 2 * max(len(sub_pos), 200))
        sub_u = rng.choice(unl_idx, size=n_u, replace=False) if len(unl_idx) > n_u else unl_idx
        sub = np.concatenate([sub_pos, sub_u])

        win = min(max(16, window_px), Hc, Wc)
        pad = win // 2
        Xp = torch.from_numpy(np.pad(Xc, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")).float()
        torch.manual_seed(seed)
        device = get_best_device()
        model = DistrictDAEUNet(in_channels=C, n_classes=n_classes, dropout_p=0.3, channel_drop_p=0.15,
                                px_size_m=float(px_size_m), pi_p=0.05).to(device)
        crit = nnPULoss(pi_p=0.05).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        yt = torch.from_numpy(y_flat[sub]).float()
        wt = torch.from_numpy(w_flat[sub]).float()
        n = len(sub)
        bs = max(int(batch_size), 64)
        n_batches = n // bs if n > bs else 1
        model.train()
        for _ in range(epochs):
            perm = rng.permutation(n)
            for b in range(n_batches):
                ii = perm[b * bs:(b + 1) * bs]
                cells = sub[ii]
                rr, cc = cells // Wc, cells % Wc
                xb = torch.stack([Xp[:, r:r + win, c:c + win] for r, c in zip(rr, cc)]).to(device)
                yb = yt[ii][:, :, None, None].to(device)
                wb = wt[ii][:, :, None, None].to(device)
                opt.zero_grad()
                p, recon, _, _ = model(xb)
                hc, wc_ = p.shape[-2] // 2, p.shape[-1] // 2
                p = p[:, :, hc:hc + 1, wc_:wc_ + 1]
                loss = crit(p, yb, sample_weights=wb) + 0.5 * nn.functional.mse_loss(recon, xb)
                loss.backward()
                opt.step()

        t_train = time.time() - t0 - t_prep
        model.eval()
        ph, pw = (16 - Hc % 16) % 16, (16 - Wc % 16) % 16
        Xd = np.pad(Xc, ((0, 0), (0, ph), (0, pw)), mode="reflect")
        with torch.no_grad():
            p_full = model(torch.from_numpy(Xd).unsqueeze(0).float().to(device))[0]
            p_full = p_full[0, :n_classes, :Hc, :Wc].detach().cpu().numpy()
        zero_heads = [k for k in range(n_classes) if not y_2d[k].any()]
        for k in zero_heads:
            p_full[k] = 0.0
        out = project_spatial_grid(p_full, target_shape=(self.H, self.W), order=1).astype(np.float32)
        for k in zero_heads:
            out[k] = 0.0
        del model, opt, Xp
        if device.type == "mps":
            torch.mps.empty_cache()
        info = {"coarse_grid": [int(Hc), int(Wc)], "decimate": int(decimate), "n_coarse_train_cells": int(len(train_idx)),
                "n_coarse_pos": int(len(pos_idx)), "n_sub_pos": int(len(sub_pos)), "n_sub_unlabeled": int(len(sub_u)),
                "epochs": int(epochs), "window_px": int(win), "device": device.type,
                "seconds": round(time.time() - t0, 1), "seconds_prep": round(t_prep, 1),
                "seconds_train": round(t_train, 1)}
        logger.info("U-Net fit: %s", info)
        return out, info


def xgb_components(model, X: np.ndarray, w_canopy: Optional[np.ndarray] = None) -> dict:
    """Uncalibrated XGBoost head scores plus what the post-processing needs (production blends these)."""
    if isinstance(model, ProductionMetals):
        c_hat = getattr(model.ens, "c_hat_", {}) or {}
        return {"domain": "metals", "raw": np.asarray(model.ens.predict_proba(X), dtype=np.float64),
                "gates": [g.score(X) if g is not None else None for g in model.gates],
                "c_hat": {k: float(np.clip(c_hat.get(k, 0.35), 0.10, 1.0)) for k in range(4)},
                "n_train_pos": np.asarray(model.n_train_pos)}
    out = model.ens.predict_full_outputs(X, w_canopy=w_canopy)
    return {"domain": "coal", "raw": np.asarray(out["probs_raw_k3"], dtype=np.float64),
            "basin_gate": np.asarray(out["basin_gate"], dtype=np.float64),
            "c_hat": {c: max(0.15, float(model.ens.c_hat_.get(c, 0.5))) for c in range(3)}}


def blend_outputs(comp: dict, s_unet: np.ndarray, w_unet) -> dict[str, np.ndarray]:
    """Production blend ``(1 - w_unet)*s_xgb + w_unet*s_unet`` per head, clipped to [0, 1]
    (``blend_predictions_by_pr_auc``), then the harness's XGBoost post-processing.
    ``w_unet = 0`` reproduces the XGBoost-only variants."""
    raw = comp["raw"]
    w = np.broadcast_to(np.asarray(w_unet, dtype=np.float64), (raw.shape[1],))
    fused = np.clip((1.0 - w) * raw + w * s_unet, 0.0, 1.0)
    if comp["domain"] == "metals":
        cal = np.zeros_like(fused)
        for k in range(4):
            if comp["n_train_pos"][k] == 0:
                continue
            cal[:, k] = calibrate_closed_boundary_elkan_noto(p_raw_k=fused[:, k], c_hat_k=comp["c_hat"][k],
                                                             gamma_closed_k=comp["gates"][k], min_c=0.10)
        return {"raw": np.column_stack([fused, fused.max(axis=1)]),
                "calibrated": np.column_stack([cal, cal.max(axis=1)])}
    # Coal: same transform as TwoStageK3CoalEnsemble.predict_full_outputs, then the Pass 4B basin gate
    cal = np.column_stack([np.clip(0.65 * fused[:, c] + 0.35 * np.minimum(1.0, fused[:, c] / comp["c_hat"][c]), 0, 1)
                           for c in range(3)])
    p_any = np.clip(1.0 - np.prod(1.0 - cal, axis=1), 0.0, 1.0)
    g = comp["basin_gate"]
    return {"raw": np.column_stack([fused, fused.max(axis=1)]),
            "gated": np.column_stack([cal * g[:, None], p_any * g])}


def _cluster_max(scores: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    return pd.Series(scores).groupby(cluster_ids).max().to_numpy()


MODEL_REGISTRY = {"xgb_prod": None, "logreg": LogRegBaseline, "single_feature": SingleFeatureBaseline,
                  "unet": None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--domain", choices=("metals", "coal"), required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", default="xgb_prod,logreg,single_feature")
    p.add_argument("--n-folds", type=int, default=5, help="0 = leave-one-block-out")
    p.add_argument("--block-m", type=float, default=25_000.0, help="block size for grouping deposits into folds")
    p.add_argument("--buffer-m", type=float, default=5_000.0, help="training exclusion buffer around held-out deposits")
    p.add_argument("--train-bg-sample", type=int, default=80_000, help="background cells per training fold (production: 80k)")
    p.add_argument("--eval-bg-sample", type=int, default=500_000, help="background cells to rank against (0 = all)")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--n-bag", type=int, default=5)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--features-from-checkpoint", type=Path, default=None,
                   help="use the feature list stored in a saved ensemble (.joblib)")
    p.add_argument("--features-file", type=Path, default=None,
                   help="pinned feature list, one column per line (same files the orchestrator reads: config/features/*.txt)")
    p.add_argument("--drop", default="", help="comma-separated feature columns to remove")
    p.add_argument("--features", default="", help="explicit comma-separated feature list (overrides automatic selection)")
    p.add_argument("--eval-bounds", default="", help="x0,y0,x1,y1 in the grid CRS: only deposits whose centroid "
                   "lies inside are held out and scored, and they are ranked against background cells inside")
    p.add_argument("--train-pos-bounds", default="", help="x0,y0,x1,y1: deposits whose centroid lies outside are "
                   "withheld from training (their cells and buffer are excluded)")
    p.add_argument("--train-bg-bounds", default="", help="x0,y0,x1,y1: non-deposit cells outside are excluded from training")
    p.add_argument("--region-col", default="", help="parquet column with a region key per cell (excluded from features)")
    p.add_argument("--eval-regions", default="", help="comma list: only deposits in these regions are held out and "
                   "scored, ranked against background cells in these regions (AND with --eval-bounds)")
    p.add_argument("--train-pos-regions", default="", help="comma list: deposits in other regions are withheld from training")
    p.add_argument("--train-bg-regions", default="", help="comma list: non-deposit training cells only from these regions")
    p.add_argument("--seed", type=int, default=42)
    # v4.8 switches (defaults reproduce production as of v4.7)
    p.add_argument("--weight-scheme", choices=("production", "hajek_neutral"), default="production",
                   help="production: metals un-normalised 1/e positives + (0.25+0.75e) unlabeled; coal 1/e on every "
                   "cell. hajek_neutral: Hájek-normalised 1/e on positives, weight 1 on unlabeled cells")
    p.add_argument("--gate", choices=("production", "standardized", "off"), default="production",
                   help="metals OUTMOGC gate: raw-feature PCA (production), z-scored PCA, or no gate")
    p.add_argument("--head0-stacking", choices=("on", "off"), default="on",
                   help="porphyry head uses skarn/epithermal meta-features (on) or base features only (off)")
    p.add_argument("--min-child-weight", type=float, default=None, help="override XGBoost min_child_weight")
    p.add_argument("--k4-exclusion-scale", type=float, default=1.0,
                   help="metals: scale the per-head exclusion radii around training positives (6/3.5/6 km and a "
                   "10x2 km ellipse); 0 = no exclusion, every unlabeled cell is a negative")
    p.add_argument("--coal-disk-mode", choices=("random", "production_keep", "exclude"), default="random",
                   help="label_coal 1 km disk cells at known mines: sampled like any background (random), all kept "
                   "as K3 negatives as the production subsample does (production_keep), or never negatives (exclude)")
    p.add_argument("--coal-buffer-m", type=float, default=0.0,
                   help="coal: unlabeled cells within this distance of known TRAINING coal are not negatives")
    p.add_argument("--novelty-dists-m", default="5000,15000",
                   help="distances from any known label used for the novelty metrics")
    p.add_argument("--frontier-m", type=float, default=25_000.0,
                   help="held-out deposits farther than this from every training deposit form the frontier subset")
    # v4.9 DAE U-Net adapter (--models unet)
    p.add_argument("--unet-epochs", type=int, default=3, help="U-Net epochs per fit (production default 3)")
    p.add_argument("--unet-decimate", type=int, default=0, help="U-Net decimation factor (0 = production rule)")
    p.add_argument("--unet-window", type=int, default=32, help="U-Net training window in coarse pixels")
    p.add_argument("--unet-inner", choices=("on", "off"), default="on",
                   help="choose w_unet by an inner deposit-level CV over the training folds (on), or skip the "
                   "selected blends (off)")
    p.add_argument("--unet-inner-bg-sample", type=int, default=100_000,
                   help="background cells per inner fold for the inner success rate")
    p.add_argument("--unet-fixed-w", default="", help="comma list of fixed w_unet values to report as well")
    p.add_argument("--unet-neg-excl", choices=("off", "on"), default="off",
                   help="coal: also keep coal disks / the --coal-buffer-m zone out of the U-Net's unlabeled "
                   "pool (off = production U-Net, which has no such exclusion)")
    return p.parse_args(argv)


def parse_regions(text: str) -> Optional[set[str]]:
    vals = {t.strip() for t in text.split(",") if t.strip()}
    return vals or None


def in_regions(region: Optional[np.ndarray], allowed: Optional[set[str]], n: int) -> np.ndarray:
    if allowed is None:
        return np.ones(n, dtype=bool)
    if region is None:
        raise SystemExit("region filters need --region-col")
    return np.isin(region, sorted(allowed))


def parse_bounds(text: str) -> Optional[tuple[float, float, float, float]]:
    if not text.strip():
        return None
    v = [float(t) for t in text.split(",")]
    if len(v) != 4:
        raise SystemExit(f"bounds need 4 numbers x0,y0,x1,y1, got {text!r}")
    return (min(v[0], v[2]), min(v[1], v[3]), max(v[0], v[2]), max(v[1], v[3]))


def in_bounds(x: np.ndarray, y: np.ndarray, b: Optional[tuple]) -> np.ndarray:
    if b is None:
        return np.ones(np.shape(x), dtype=bool)
    return (x >= b[0]) & (x <= b[2]) & (y >= b[1]) & (y <= b[3])


def main(argv=None) -> dict:
    args = parse_args(argv)
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    label_cols = K4_LABEL_COLS if args.domain == "metals" else K3_LABEL_COLS
    head_names = list(K4_HEAD_NAMES) if args.domain == "metals" else list(K3_COAL_HEAD_NAMES)
    other_cols = K3_LABEL_COLS if args.domain == "metals" else K4_LABEL_COLS

    import pyarrow.parquet as pq

    all_cols = pq.read_schema(args.data_dir / "dataset.parquet").names
    profile, H, W, px_size_m, is_geo = load_grid(args.data_dir)
    missing = [c for c in label_cols if c not in all_cols]
    if missing:
        raise SystemExit(f"dataset.parquet lacks label columns {missing}")

    region_col = args.region_col.strip() or None
    if region_col and region_col not in all_cols:
        raise SystemExit(f"--region-col {region_col} not in dataset")
    feature_cols = [c for c in select_feature_cols(list(all_cols)) if c != region_col]
    if args.features_from_checkpoint:
        import joblib
        ck = joblib.load(args.features_from_checkpoint)
        feature_cols = list(getattr(ck, "feature_names_", []) or [])
    if args.features_file is not None:
        from src.schema.canonical import load_feature_list

        feature_cols = load_feature_list(args.features_file)
    if args.features.strip():
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    absent = [c for c in feature_cols if c not in all_cols]
    if absent:
        raise SystemExit(f"requested features absent from dataset: {absent}")
    drop = [c.strip() for c in args.drop.split(",") if c.strip()]
    feature_cols = [c for c in feature_cols if c not in drop]
    from src.features.real_sources.core import assert_feature_provenance

    assert_feature_provenance(args.data_dir, feature_cols)
    verify_no_raw_pixel_leakage(feature_cols)
    eval_b, pos_b, bg_b = (parse_bounds(args.eval_bounds), parse_bounds(args.train_pos_bounds),
                           parse_bounds(args.train_bg_bounds))

    if region_col and region_col in feature_cols:
        raise SystemExit("the region column cannot be a feature")
    eval_r, pos_r, bg_r = (parse_regions(args.eval_regions), parse_regions(args.train_pos_regions),
                           parse_regions(args.train_bg_regions))
    aux = ["row", "col", "explored_footprint", "w_canopy", "coal_tonnage_weight", "label_coal",
           *label_cols, *other_cols, *ACCESS_COLS]
    if region_col:
        aux.append(region_col)
    df = pd.read_parquet(args.data_dir / "dataset.parquet",
                         columns=list(dict.fromkeys([c for c in aux + feature_cols if c in all_cols])))
    # v4.15: access distances are not capped (the orchestrator does not cap them either; ports reach >100 km)
    clip_distance_cols(df, feature_cols)
    logger.info("%d cells, grid %dx%d, px %.1f m, %d features", len(df), H, W, px_size_m, len(feature_cols))

    rows = df["row"].to_numpy(np.int32)
    cols = df["col"].to_numpy(np.int32)
    X_all = df[feature_cols].to_numpy(np.float32)
    y_heads = np.column_stack([df[c].to_numpy(np.int32) for c in label_cols])
    y_heads = np.where(y_heads == 1, 1, -1).astype(np.int32)
    present_other = [c for c in other_cols if c in df.columns]
    pos_other = (np.column_stack([df[c].to_numpy() == 1 for c in present_other]).any(axis=1)
                 if present_other else np.zeros(len(df), bool))
    explored = (df["explored_footprint"].to_numpy() > 0) if "explored_footprint" in df.columns else np.zeros(len(df), bool)
    missing_access = [c for c in ACCESS_COLS if c not in df.columns]
    if missing_access:
        raise SystemExit(f"propensity covariates {missing_access} are not in {args.data_dir}; rebuild the cube (v4.15)")
    access = df[list(ACCESS_COLS)].to_numpy(np.float64)
    w_canopy = df["w_canopy"].to_numpy(np.float64) if "w_canopy" in df.columns else np.zeros(len(df))
    tonnage = df["coal_tonnage_weight"].to_numpy(np.float64) if "coal_tonnage_weight" in df.columns else np.ones(len(df))
    region = df[region_col].astype(str).to_numpy() if region_col else None
    coal_disk = (df["label_coal"].to_numpy() == 1) if "label_coal" in df.columns else np.zeros(len(df), bool)
    del df

    # Deposit clusters on the full grid
    pos_dom = np.any(y_heads == 1, axis=1)
    pos_2d = np.zeros((H, W), bool)
    pos_2d[rows[pos_dom], cols[pos_dom]] = True
    head_2d = np.zeros((len(label_cols), H, W), bool)
    for k in range(len(label_cols)):
        m = y_heads[:, k] == 1
        head_2d[k, rows[m], cols[m]] = True
    import rasterio.transform
    tr = profile["transform"]
    cell_x = tr.c + (cols + 0.5) * tr.a
    cell_y = tr.f + (rows + 0.5) * tr.e

    cluster_2d = label_deposit_clusters(pos_2d)

    # Distance (m) from every cell to ANY known label of this domain. Coal also counts the
    # 1 km ESDM disks (label_coal), which are known coal but not K3 halo positives.
    from scipy import ndimage as _ndi
    known_2d = pos_2d.copy()
    if args.domain == "coal":
        known_2d[rows[coal_disk], cols[coal_disk]] = True
    dist_known_m = (_ndi.distance_transform_edt(~known_2d) * px_size_m)[rows, cols].astype(np.float32)
    novelty_dists = [float(t) for t in args.novelty_dists_m.split(",") if t.strip()]
    coal_disk_only = coal_disk & ~pos_dom if args.domain == "coal" else np.zeros(len(rows), bool)
    clusters = summarize_clusters(cluster_2d, head_2d, head_names)
    clusters["centroid_x"] = tr.c + (clusters["centroid_col"].to_numpy() + 0.5) * tr.a
    clusters["centroid_y"] = tr.f + (clusters["centroid_row"].to_numpy() + 0.5) * tr.e
    clusters["evaluated"] = in_bounds(clusters["centroid_x"].to_numpy(), clusters["centroid_y"].to_numpy(), eval_b)
    clusters["trainable"] = in_bounds(clusters["centroid_x"].to_numpy(), clusters["centroid_y"].to_numpy(), pos_b)
    if region is not None:
        pos_cells = np.where(pos_dom)[0]
        cl_reg = (pd.DataFrame({"cid": cluster_2d[rows[pos_cells], cols[pos_cells]], "reg": region[pos_cells]})
                  .groupby("cid")["reg"].agg(lambda s: s.value_counts().index[0]))
        clusters["region"] = cl_reg.reindex(clusters["cluster_id"]).to_numpy()
        clusters["evaluated"] &= in_regions(clusters["region"].to_numpy(), eval_r, len(clusters))
        clusters["trainable"] &= in_regions(clusters["region"].to_numpy(), pos_r, len(clusters))
    block_px = max(1, int(round(args.block_m / px_size_m)))
    ev_mask = clusters["evaluated"].to_numpy()
    clusters["fold"] = -1
    clusters.loc[ev_mask, "fold"] = assign_cluster_folds(clusters[ev_mask], block_px=block_px,
                                                         n_folds=args.n_folds, random_state=args.seed)
    folds = sorted(clusters.loc[ev_mask, "fold"].unique().tolist())
    cell_cluster = cluster_2d[rows, cols]
    buffer_px = args.buffer_m / px_size_m
    withheld_ids = clusters.loc[~clusters["trainable"], "cluster_id"].to_numpy()
    withheld_excl = training_exclusion_mask(cluster_2d, withheld_ids, buffer_px)[rows, cols]
    in_bg_train = in_bounds(cell_x, cell_y, bg_b) & in_regions(region, bg_r, len(rows))
    in_eval = in_bounds(cell_x, cell_y, eval_b) & in_regions(region, eval_r, len(rows))
    logger.info("%d deposit clusters (%d evaluated, %d withheld), %d folds (block %d px, buffer %.1f px)",
                len(clusters), int(ev_mask.sum()), len(withheld_ids), len(folds), block_px, buffer_px)
    clusters.to_csv(args.out_dir / "folds.csv", index=False)

    # Fixed background sample to rank deposits against (same for every fold and model)
    rng = np.random.default_rng(args.seed)
    bg_all = np.where(~pos_dom & ~coal_disk_only & in_eval)[0]
    bg_eval = bg_all if args.eval_bg_sample <= 0 or args.eval_bg_sample >= len(bg_all) else \
        np.sort(rng.choice(bg_all, size=args.eval_bg_sample, replace=False))

    # Production training uses the orchestrator's block grid (default 25 km) for spatial_block_ids
    prod_block_px = min(max(12, int(round(args.block_m / max(px_size_m, 1.0)))), max(12, min(H, W) // 5))

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    adapters = {}
    for mn in model_names:
        if mn not in MODEL_REGISTRY:
            raise SystemExit(f"unknown model {mn}; choose from {list(MODEL_REGISTRY)}")
        cls = (ProductionMetals if args.domain == "metals" else ProductionCoal) if mn == "xgb_prod" else MODEL_REGISTRY[mn]
        adapters[mn] = cls

    records = []
    novelty_records = []
    fold_log = []
    targets = head_names + ["any"]

    def coal_negative_exclusion(excl: np.ndarray) -> np.ndarray:
        """Per-cell: unlabeled cells that must not act as coal negatives (disks / buffer)."""
        out = np.zeros(len(rows), bool)
        if args.domain != "coal":
            return out
        if args.coal_disk_mode == "exclude":
            out |= coal_disk_only
        if args.coal_buffer_m > 0:
            known_tr_2d = np.zeros((H, W), bool)
            kt = np.where(known_2d[rows, cols] & ~excl)[0]
            known_tr_2d[rows[kt], cols[kt]] = True
            d_tr = _ndi.distance_transform_edt(~known_tr_2d) * px_size_m
            out |= (d_tr[rows, cols] <= args.coal_buffer_m) & ~pos_dom
        return out

    def make_fold(excl: np.ndarray, rng_seed: int) -> tuple[FoldData, np.ndarray]:
        """Training sample when the cells in ``excl`` are held out (production subsampling)."""
        cand = np.where(~excl)[0]
        cand_pos = cand[pos_dom[cand] | (pos_other[cand] & in_bg_train[cand])]
        cand_bg = cand[~(pos_dom[cand] | pos_other[cand]) & in_bg_train[cand]]
        disk_keep = np.zeros(0, dtype=cand.dtype)
        if args.domain == "coal" and args.coal_disk_mode != "random":
            disk_keep = cand_bg[coal_disk_only[cand_bg]]
            cand_bg = cand_bg[~coal_disk_only[cand_bg]]
        frng = np.random.default_rng(rng_seed)
        bg_take = cand_bg if len(cand_bg) <= args.train_bg_sample else frng.choice(cand_bg, args.train_bg_sample, replace=False)
        tr_idx = np.unique(np.concatenate([cand_pos, bg_take, disk_keep]))
        neg_excl = coal_negative_exclusion(excl)[tr_idx]
        xs, ys = rasterio.transform.xy(tr, rows[tr_idx], cols[tr_idx])
        fd = FoldData(
            X=X_all[tr_idx].astype(np.float64), y_heads=y_heads[tr_idx], pos_any_other=pos_other[tr_idx],
            rows=rows[tr_idx], cols=cols[tr_idx], explored=explored[tr_idx], access=access[tr_idx],
            w_canopy=w_canopy[tr_idx], tonnage=tonnage[tr_idx], coords=np.column_stack([xs, ys]),
            block_ids=assign_spatial_blocks(rows[tr_idx], cols[tr_idx], H, W, block_size_px=prod_block_px),
            neg_excl=neg_excl,
        )
        return fd, tr_idx

    # ---- DAE U-Net (v4.9) -------------------------------------------------
    unet_cube = None
    xgb_cls = ProductionMetals if args.domain == "metals" else ProductionCoal
    primary_variant = "calibrated" if args.domain == "metals" else "gated"
    fixed_w = [float(t) for t in args.unet_fixed_w.split(",") if t.strip()]
    if "unet" in adapters:
        unet_cube = UNetCube(args.data_dir, feature_cols, H, W, rows, cols, X_all)
        unet_decimate = args.unet_decimate or production_dae_decimate_factor(H, W)
        logger.info("U-Net cube: %d channels from %s, decimate %d", len(unet_cube.names), unet_cube.source,
                    unet_decimate)

    def fit_unet(excl: np.ndarray, fd: FoldData, seed: int) -> tuple[np.ndarray, dict]:
        """U-Net on the cells not in ``excl``: production eligibility, labels and 1/e(x) weights."""
        elig = ~excl & (pos_dom | in_bg_train)
        if args.unet_neg_excl == "on":
            elig &= ~coal_negative_exclusion(excl)
        elig_2d = np.zeros((H, W), bool)
        elig_2d[rows[elig], cols[elig]] = True
        y_2d = np.zeros((len(label_cols), H, W), np.float32)
        for k in range(len(label_cols)):
            m = elig & (y_heads[:, k] == 1)
            y_2d[k, rows[m], cols[m]] = 1.0
        _, e_all = fit_propensity(fd, access)  # production: 1/e(x) on every cell
        w_2d = np.ones((H, W), np.float32)
        w_2d[rows, cols] = (1.0 / e_all).astype(np.float32)
        return unet_cube.fit_predict(n_classes=len(label_cols), y_2d=y_2d, elig_2d=elig_2d, norm_rows=fd.rows,
                                     norm_cols=fd.cols, w_2d=w_2d, decimate=unet_decimate,
                                     epochs=args.unet_epochs, px_size_m=px_size_m, window_px=args.unet_window,
                                     seed=seed)

    def select_unet_weights(f: int, excl_f: np.ndarray) -> dict:
        """Choose w_unet using only fold f's training folds: inner deposit-level CV over the other folds."""
        from src.evaluation.metrics import pr_auc
        from src.models.fusion import compute_ensemble_weights

        irng = np.random.default_rng(args.seed + 7000 + f)
        bg_in_all = bg_eval[~excl_f[bg_eval]]
        inner = []
        for g in folds:
            if g == f:
                continue
            ids_g = clusters.loc[clusters["fold"] == g, "cluster_id"].to_numpy()
            excl_in = excl_f | training_exclusion_mask(cluster_2d, ids_g, buffer_px)[rows, cols]
            fd_in, _ = make_fold(excl_in, args.seed + 5000 + 10 * f + g)
            ts = time.time()
            xgb_in = xgb_cls(args, feature_cols, is_geo)
            xgb_in.fit(fd_in)
            grid_in, _ = fit_unet(excl_in, fd_in, seed=args.seed + 100 + 10 * f + g)
            test_g = np.where(np.isin(cell_cluster, ids_g))[0]
            bg_in = bg_in_all if len(bg_in_all) <= args.unet_inner_bg_sample else \
                np.sort(irng.choice(bg_in_all, args.unet_inner_bg_sample, replace=False))
            idx = np.concatenate([test_g, bg_in])
            inner.append({"comp": xgb_components(xgb_in, X_all[idx].astype(np.float64), w_canopy[idx]),
                          "su": grid_in[:, rows[idx], cols[idx]].T.astype(np.float64),
                          "n_test": len(test_g), "cl": cell_cluster[test_g], "yk": y_heads[test_g] == 1})
            del grid_in
            logger.info("fold %d inner fold %d (U-Net + XGB) in %.1fs", f, g, time.time() - ts)
        # (a) deposit-level: grid search on the inner success-rate AUC of the primary "any" score
        sr = {}
        for w in UNET_W_GRID:
            af = []
            for it in inner:
                S = blend_outputs(it["comp"], it["su"], w)[primary_variant][:, -1]
                af.append(deposit_area_fractions(S[it["n_test"]:], _cluster_max(S[:it["n_test"]], it["cl"])))
            sr[w] = float(1.0 - np.concatenate(af).mean())
        best = max(sr.values())
        w_sr = min(w for w in UNET_W_GRID if sr[w] >= best - 1e-12)  # ties -> less U-Net
        # (b) production rule: per head, weights proportional to pixel PR-AUC on held-out cells
        w_pr, ap = [], []
        for k in range(len(label_cols)):
            y = np.concatenate([np.concatenate([it["yk"][:, k], np.zeros(len(it["su"]) - it["n_test"], bool)])
                                for it in inner]).astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                w_pr.append(0.0 if args.domain == "metals" else 0.25)  # production fallbacks
                ap.append(None)
                continue
            sx = np.concatenate([it["comp"]["raw"][:, k] for it in inner])
            su = np.concatenate([it["su"][:, k] for it in inner])
            a_x, a_u = float(pr_auc(y, sx)), float(pr_auc(y, su))
            w_pr.append(float(compute_ensemble_weights(a_x, a_u)[1]))
            ap.append({"xgb": a_x, "unet": a_u})
        return {"w_srauc": w_sr, "inner_srauc_by_w": {str(k): v for k, v in sr.items()},
                "w_prauc": w_pr, "inner_pixel_pr_auc": ap,
                "n_inner_deposits": int(sum(len(np.unique(it["cl"])) for it in inner))}

    for f in folds:
        test_ids = clusters.loc[clusters["fold"] == f, "cluster_id"].to_numpy()
        excl_2d = training_exclusion_mask(cluster_2d, test_ids, buffer_px)
        excl = excl_2d[rows, cols] | withheld_excl
        fd, tr_idx = make_fold(excl, args.seed + 1000 + f)
        neg_excl = fd.neg_excl
        # Distance from each held-out cluster to the nearest TRAINING deposit (frontier subset)
        train_dep_2d = (cluster_2d >= 0) & ~excl_2d
        if len(withheld_ids):
            train_dep_2d &= ~np.isin(cluster_2d, withheld_ids)
        d_train_dep = _ndi.distance_transform_edt(~train_dep_2d) * px_size_m
        # Novelty decomposition: distance to labels the fold model was TRAINED on vs to the
        # held-out deposits. Top cells near training labels = re-finding known ground; near
        # held-out deposits = genuine discovery of unseen deposits.
        known_tr_fold_2d = np.zeros((H, W), bool)
        _kt = np.where(known_2d[rows, cols] & ~excl)[0]
        known_tr_fold_2d[rows[_kt], cols[_kt]] = True
        d_known_train_bg = (_ndi.distance_transform_edt(~known_tr_fold_2d) * px_size_m)[rows[bg_eval], cols[bg_eval]]
        test_2d = np.isin(cluster_2d, test_ids)
        d_test_bg = (_ndi.distance_transform_edt(~test_2d) * px_size_m)[rows[bg_eval], cols[bg_eval]]
        test_cells = np.where(np.isin(cell_cluster, test_ids))[0]
        # Rank against the whole map's background, as a geologist would see it
        eval_idx = np.concatenate([test_cells, bg_eval])
        n_test = len(test_cells)
        entry = {"fold": int(f), "n_test_clusters": int(len(test_ids)), "n_train_cells": int(len(tr_idx)),
                 "n_train_pos_per_head": {h: int((fd.y_heads[:, k] == 1).sum()) for k, h in enumerate(head_names)},
                 "n_excluded_cells": int(excl.sum()), "n_negative_excluded_cells": int(neg_excl.sum()),
                 "models": {}}
        for mn, cls in adapters.items():
            ts = time.time()
            Xe = X_all[eval_idx].astype(np.float64)
            if mn == "unet":
                model = xgb_cls(args, feature_cols, is_geo)
                model.fit(fd)
                logger.info("fold %d XGBoost fit in %.1fs", f, time.time() - ts)
                grid, uinfo = fit_unet(excl, fd, seed=args.seed + f)
                s_u = grid[:, rows[eval_idx], cols[eval_idx]].T.astype(np.float64)
                del grid
                xo = model.predict(Xe, w_canopy=w_canopy[eval_idx]) if isinstance(model, ProductionCoal) else model.predict(Xe)
                out = {"alone": np.column_stack([s_u, s_u.max(axis=1)])}
                out.update({f"xgb_{v}": S for v, S in xo.items()})
                comp = xgb_components(model, Xe, w_canopy[eval_idx])
                minfo = {"unet": uinfo}
                if args.unet_inner == "on":
                    sel = select_unet_weights(f, excl)
                    minfo.update(sel)
                    for name, wv in (("srauc", sel["w_srauc"]), ("prauc", sel["w_prauc"])):
                        out.update({f"{name}_{v}": S for v, S in blend_outputs(comp, s_u, wv).items()})
                for wv in fixed_w:
                    out.update({f"w{wv:.2f}_{v}": S for v, S in blend_outputs(comp, s_u, wv).items()})
                entry["models"][mn] = {"seconds": round(time.time() - ts, 1), **minfo}
                logger.info("fold %d: w_unet srauc=%s prauc=%s", f, minfo.get("w_srauc"), minfo.get("w_prauc"))
            else:
                model = cls(args, feature_cols, is_geo)
                model.fit(fd)
                out = model.predict(Xe, w_canopy=w_canopy[eval_idx]) if isinstance(model, ProductionCoal) else model.predict(Xe)
                entry["models"][mn] = {"seconds": round(time.time() - ts, 1)}
            if hasattr(model, "describe"):
                entry["models"][mn]["chosen"] = dict(zip(targets, model.describe()))
            for variant, S in out.items():
                s_test, s_bg = S[:n_test], S[n_test:]
                cl_test = cell_cluster[test_cells]
                for ti, tname in enumerate(targets):
                    for cid in test_ids:
                        row = clusters.loc[cid]
                        if tname != "any" and not bool(row[f"has_{tname}"]):
                            continue
                        in_c = cl_test == cid
                        sc = s_test[in_c, ti]
                        dscore = float(np.max(sc))
                        tc = test_cells[in_c]
                        records.append({
                            "model": mn, "variant": variant, "target": tname, "fold": int(f),
                            "cluster_id": int(cid), "n_cells": int(row["n_cells"]),
                            "centroid_x": float(row["centroid_x"]), "centroid_y": float(row["centroid_y"]),
                            "region": row["region"] if "region" in row.index else None,
                            "score_max": dscore, "score_median": float(np.median(sc)),
                            "area_fraction": float(deposit_area_fractions(s_bg[:, ti], [dscore])[0]),
                            "area_fraction_median_cell": float(deposit_area_fractions(s_bg[:, ti], [np.median(sc)])[0]),
                            "dist_train_deposit_km": float(np.min(d_train_dep[rows[tc], cols[tc]]) / 1000.0),
                        })
                # Novelty: where does the top of the map sit relative to ALL known labels?
                d_bg = dist_known_m[bg_eval]
                for ti, tname in enumerate(targets):
                    sb = s_bg[:, ti]
                    for q in (0.01, 0.05, 0.10):
                        thr = np.quantile(sb, 1.0 - q)
                        top = sb >= thr
                        rec = {"model": mn, "variant": variant, "target": tname, "fold": int(f), "top_q": q,
                               "n_top": int(top.sum())}
                        for dm in novelty_dists:
                            rec[f"far_{int(dm)}m"] = float(np.mean(d_bg[top] > dm))
                            rec[f"base_far_{int(dm)}m"] = float(np.mean(d_bg > dm))
                            rec[f"near_train_{int(dm)}m"] = float(np.mean(d_known_train_bg[top] <= dm))
                            rec[f"base_near_train_{int(dm)}m"] = float(np.mean(d_known_train_bg <= dm))
                            rec[f"near_test_{int(dm)}m"] = float(np.mean(d_test_bg[top] <= dm))
                            rec[f"base_near_test_{int(dm)}m"] = float(np.mean(d_test_bg <= dm))
                        novelty_records.append(rec)
            logger.info("fold %d model %s done in %.1fs", f, mn, time.time() - ts)
        fold_log.append(entry)

    per_dep = pd.DataFrame(records)
    per_dep.to_csv(args.out_dir / "per_deposit.csv", index=False)
    nov = pd.DataFrame(novelty_records)
    nov.to_csv(args.out_dir / "novelty_per_fold.csv", index=False)

    results = {}
    curves = []
    for (mn, variant, tname), g in per_dep.groupby(["model", "variant", "target"], sort=False):
        key = f"{mn}/{variant}"
        a = g["area_fraction"].to_numpy()
        summ = summarize_success_rate(a, DEFAULT_CAPTURE_LEVELS).to_dict()
        summ["success_rate_auc_median_cell"] = (
            float(1.0 - g["area_fraction_median_cell"].mean()) if not summ["insufficient_data"] else None)
        results.setdefault(key, {})[tname] = summ
        if tname == "any":
            fr = g[g["dist_train_deposit_km"] * 1000.0 > args.frontier_m]["area_fraction"].to_numpy()
            fsum = summarize_success_rate(fr, DEFAULT_CAPTURE_LEVELS).to_dict()
            fsum["frontier_m"] = args.frontier_m
            results[key]["any_frontier"] = fsum
        c = success_rate_curve(a)
        c["model"], c["variant"], c["target"] = mn, variant, tname
        curves.append(c)
    if curves:
        pd.concat(curves).to_csv(args.out_dir / "success_rate_curves.csv", index=False)
    novelty = {}
    if not nov.empty:
        far_cols = [c for c in nov.columns if c.startswith(("far_", "base_far_", "near_", "base_near_"))]
        agg = nov.groupby(["model", "variant", "target", "top_q"])[far_cols].mean().reset_index()
        for _, r in agg.iterrows():
            d = novelty.setdefault(f"{r['model']}/{r['variant']}", {}).setdefault(r["target"], {})
            entry = {c: float(r[c]) for c in far_cols}
            for dm in novelty_dists:
                inside_top = 1.0 - entry[f"far_{int(dm)}m"]
                inside_base = 1.0 - entry[f"base_far_{int(dm)}m"]
                entry[f"enrichment_within_{int(dm)}m"] = inside_top / max(inside_base, 1e-9)
                if f"near_train_{int(dm)}m" in entry:
                    entry[f"enrichment_near_train_{int(dm)}m"] = (
                        entry[f"near_train_{int(dm)}m"] / max(entry[f"base_near_train_{int(dm)}m"], 1e-9))
                    entry[f"enrichment_near_test_{int(dm)}m"] = (
                        entry[f"near_test_{int(dm)}m"] / max(entry[f"base_near_test_{int(dm)}m"], 1e-9))
            d[f"top_{r['top_q']:.2f}"] = entry

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir), "domain": args.domain,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "px_size_m": px_size_m, "grid": [H, W], "n_cells": int(len(rows)),
        "n_eval_background_cells": int(len(bg_eval)),
        "features": feature_cols,
        "n_deposit_clusters": int(len(clusters)),
        "n_evaluated_clusters": int(ev_mask.sum()),
        "n_withheld_clusters": int(len(withheld_ids)),
        "n_clusters_per_head": {h: int(clusters[f"has_{h}"].sum()) for h in head_names},
        "n_evaluated_clusters_per_head": {h: int(clusters.loc[ev_mask, f"has_{h}"].sum()) for h in head_names},
        "bounds": {"eval": eval_b, "train_pos": pos_b, "train_bg": bg_b},
        "regions": {"column": region_col, "eval": sorted(eval_r) if eval_r else None,
                    "train_pos": sorted(pos_r) if pos_r else None, "train_bg": sorted(bg_r) if bg_r else None},
        "n_evaluated_clusters_by_region": (clusters.loc[ev_mask, "region"].value_counts().to_dict()
                                           if "region" in clusters else None),
        "folds": fold_log,
        "results": results,
        "novelty": novelty,
        "method_notes": [
            "Deposit = 8-connected cluster of positive cells; score = max over its cells (median-cell variant also reported).",
            "area_fraction = share of background (non-deposit) cells scoring above the deposit; ties count half.",
            "success_rate_auc = 1 - mean(area_fraction); 0.5 = random map.",
            "Held-out clusters and all cells within buffer_m are excluded from training; preprocessing is refit per fold.",
            "Not reproduced: validation-fold c_hat, warm start.",
            "unet (v4.9): one production DAE U-Net per fold on the fold's training cells; 'alone' = U-Net heads "
            "(any = max); 'xgb_*' = the XGBoost trained alongside; 'srauc_*' = blend with w_unet chosen by inner "
            "deposit-level CV on the training folds (grid 0..1, primary any score); 'prauc_*' = blend with the "
            "production rule (per-head w proportional to pixel PR-AUC) on the same inner held-out cells; "
            "'wX.XX_*' = fixed w_unet.",
            "novelty: share of the top-q background cells farther than d from ANY known label (all folds' labels; "
            "coal also counts label_coal disks), averaged over folds; enrichment_within_d = share within d in the "
            "top-q over the share within d in the whole background. Lower enrichment = the map reaches further "
            "into unlabeled ground.",
            "any_frontier: success rate over held-out deposits farther than frontier_m from every training deposit.",
            "Coal evaluation background excludes label_coal disk cells (known coal) since v4.8.",
        ],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (args.out_dir / "deposit_cv_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(per_dep, targets, args.out_dir)
    _print_table(results, targets + ["any_frontier"])
    _print_novelty(novelty)
    logger.info("done in %.1fs -> %s", time.time() - t0, args.out_dir)
    return summary


def _plot(per_dep: pd.DataFrame, targets: list[str], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return
    for tname in targets:
        g = per_dep[per_dep["target"] == tname]
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1, label="random")
        for (mn, variant), gg in g.groupby(["model", "variant"]):
            a = np.sort(gg["area_fraction"].to_numpy())
            yv = np.arange(1, len(a) + 1) / len(a)
            ax.step(np.concatenate([[0], a, [1]]), np.concatenate([[0], yv, [1]]), where="post",
                    label=f"{mn}/{variant} (AUC {1 - a.mean():.2f})")
        ax.set_xlabel("Share of background area flagged")
        ax.set_ylabel("Share of held-out deposits captured")
        ax.set_title(f"Deposit-level success rate: {tname} (n={g['cluster_id'].nunique()})")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / f"success_rate_{tname}.png", dpi=130)
        plt.close(fig)


def _print_table(results: dict, targets: list[str]) -> None:
    print(f"\n{'model/variant':28s} {'target':18s} {'n':>4s} {'AUC':>6s} {'95% CI':>13s} {'p':>8s} {'cap@10%':>8s}")
    for key, res in results.items():
        for tname in targets:
            r = res.get(tname)
            if r is None:
                continue
            if r["insufficient_data"]:
                print(f"{key:28s} {tname:18s} {r['n_deposits']:4d}  insufficient_data")
                continue
            ci = r["success_rate_auc_ci95"]
            print(f"{key:28s} {tname:18s} {r['n_deposits']:4d} {r['success_rate_auc']:6.3f} "
                  f"[{ci[0]:.2f},{ci[1]:.2f}] {r['p_value_vs_random']:8.4f} {r['capture']['0.10']['rate']:8.2f}")


def _print_novelty(novelty: dict) -> None:
    print(f"\n{'model/variant':28s} {'top':>5s} {'within5km':>10s} {'base':>6s} {'enrich':>7s} "
          f"{'within15km':>11s} {'base':>6s} {'enrich':>7s} {'xTrain5':>8s} {'xTest5':>7s} {'xTest15':>8s}")
    for key, by_t in novelty.items():
        for tq, e in by_t.get("any", {}).items():
            try:
                print(f"{key:28s} {tq[4:]:>5s} {1 - e['far_5000m']:10.3f} {1 - e['base_far_5000m']:6.3f} "
                      f"{e['enrichment_within_5000m']:7.2f} {1 - e['far_15000m']:11.3f} "
                      f"{1 - e['base_far_15000m']:6.3f} {e['enrichment_within_15000m']:7.2f} "
                      f"{e.get('enrichment_near_train_5000m', float('nan')):8.2f} "
                      f"{e.get('enrichment_near_test_5000m', float('nan')):7.2f} "
                      f"{e.get('enrichment_near_test_15000m', float('nan')):8.2f}")
            except KeyError:
                pass


if __name__ == "__main__":
    main()
