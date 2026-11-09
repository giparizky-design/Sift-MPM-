#!/usr/bin/env python3
"""Deposit-level, buffered spatial cross-validation of the production ensembles.

For each fold, held-out deposit clusters and every cell within ``--buffer-m``
of them are removed from training. The model is retrained from zero with the
production preprocessing (feature selection, 15 km distance caps, logistics
propensity, exclusion masks, Hájek head weights, OUTMOGC gates, Elkan-Noto
scaling) fitted on the training cells only. Each held-out deposit is then
ranked against the background cells of the map. See ``src/evaluation/deposit_cv.py``.

What this does NOT reproduce from ``run_seven_pass_training.py``:
  * the DAE U-Net and its PR-AUC blend with XGBoost,
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
# Mirrors explicit_access_cols in run_seven_pass_training (Pass 2).
ACCESS_COLS = (
    "dist_major_rivers_km", "dist_haul_roads_km", "dist_transport_km", "gfw_forest_disturbance",
    "forest_disturbance_rate", "dem_elevation", "dem_slope", "outcrop_exposure_index",
    "regolith_cover_m", "coastal_shelf_dist_km",
)


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
                 tonnage: np.ndarray, coords: np.ndarray, block_ids: np.ndarray):
        self.X, self.y_heads, self.pos_any_other = X, y_heads, pos_any_other
        self.rows, self.cols, self.explored, self.access = rows, cols, explored, access
        self.w_canopy, self.tonnage, self.coords, self.block_ids = w_canopy, tonnage, coords, block_ids

    @property
    def pos_domain(self) -> np.ndarray:
        return np.any(self.y_heads == 1, axis=1)


def fit_propensity(fd: FoldData) -> np.ndarray:
    """Logistics exploration propensity e(x) on training cells (orchestrator Pass 2)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = (fd.explored | fd.pos_domain | fd.pos_any_other).astype(int)
    if not (0 < y.sum() < len(y)):
        return np.full(len(y), 0.20)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=500, random_state=42))
    pipe.fit(fd.access, y)
    return np.clip(pipe.predict_proba(fd.access)[:, 1], 0.05, 1.0)


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
        excl = compute_k4_exclusion_mask(coords=fd.coords, positive_coords_k=pos_coords, labels_k4=y,
                                         is_geographic=self.is_geo)
        w = np.column_stack([
            compute_asymmetric_head_weights(labels_k=y[:, k], e_logistics=e_x, exclusion_mask_k=excl[:, k])
            for k in range(4)
        ])
        self.ens = TwoStageK4Ensemble(n_estimators=self.args.n_estimators, n_bag=self.args.n_bag,
                                      learning_rate=self.args.lr, max_depth=self.args.max_depth or 6,
                                      random_state=42)
        self.ens.fit(X=fd.X, y_k4=y, sample_weights_k4=w, e_logistics=e_x, exclusion_masks=excl,
                     spatial_block_ids=fd.block_ids, feature_names=self.feature_cols)
        self.gates = []
        for k in range(4):
            Xp = fd.X[y[:, k] == 1]
            if len(Xp) >= 5:
                g = OUTMOGCGate(n_pca_components=min(4, Xp.shape[1]),
                                n_gmm_components=min(2, max(1, len(Xp) // 10)), random_state=42)
                g.fit(Xp)
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
    variants = ("raw", "hierarchical")

    def __init__(self, args, feature_cols, is_geo):
        self.args, self.feature_cols = args, feature_cols

    def fit(self, fd: FoldData) -> None:
        from src.models.pu_pipeline import TwoStageK3CoalEnsemble

        e_x = fit_propensity(fd)
        self.ens = TwoStageK3CoalEnsemble(n_estimators=self.args.n_estimators, n_bag=self.args.n_bag,
                                          learning_rate=self.args.lr, max_depth=self.args.max_depth or 5,
                                          random_state=42)
        self.ens.fit(X=fd.X, y_k3=fd.y_heads, tonnage_weights=fd.tonnage, e_pooled=e_x, w_canopy=fd.w_canopy,
                     spatial_block_ids=fd.block_ids, feature_names=self.feature_cols)

    def predict(self, X: np.ndarray, w_canopy: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
        out = self.ens.predict_full_outputs(X, w_canopy=w_canopy)
        raw = np.asarray(out["probs_raw_k3"], dtype=np.float64)
        hier = np.asarray(out["probs_k3"], dtype=np.float64)
        return {"raw": np.column_stack([raw, raw.max(axis=1)]),
                "hierarchical": np.column_stack([hier, np.asarray(out["p_coal_any"], dtype=np.float64)])}


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


MODEL_REGISTRY = {"xgb_prod": None, "logreg": LogRegBaseline, "single_feature": SingleFeatureBaseline}


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
    if args.features.strip():
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    absent = [c for c in feature_cols if c not in all_cols]
    if absent:
        raise SystemExit(f"requested features absent from dataset: {absent}")
    drop = [c.strip() for c in args.drop.split(",") if c.strip()]
    feature_cols = [c for c in feature_cols if c not in drop]
    verify_no_raw_pixel_leakage(feature_cols)
    eval_b, pos_b, bg_b = (parse_bounds(args.eval_bounds), parse_bounds(args.train_pos_bounds),
                           parse_bounds(args.train_bg_bounds))

    if region_col and region_col in feature_cols:
        raise SystemExit("the region column cannot be a feature")
    eval_r, pos_r, bg_r = (parse_regions(args.eval_regions), parse_regions(args.train_pos_regions),
                           parse_regions(args.train_bg_regions))
    aux = ["row", "col", "explored_footprint", "w_canopy", "coal_tonnage_weight", *label_cols, *other_cols, *ACCESS_COLS]
    if region_col:
        aux.append(region_col)
    df = pd.read_parquet(args.data_dir / "dataset.parquet",
                         columns=list(dict.fromkeys([c for c in aux + feature_cols if c in all_cols])))
    clip_distance_cols(df, feature_cols + [c for c in ACCESS_COLS if c in df.columns])
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
    access_cols = [c for c in ACCESS_COLS if c in df.columns]
    access = df[access_cols].to_numpy(np.float64) if access_cols else X_all[:, :1].astype(np.float64)
    w_canopy = df["w_canopy"].to_numpy(np.float64) if "w_canopy" in df.columns else np.zeros(len(df))
    tonnage = df["coal_tonnage_weight"].to_numpy(np.float64) if "coal_tonnage_weight" in df.columns else np.ones(len(df))
    region = df[region_col].astype(str).to_numpy() if region_col else None
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
    bg_all = np.where(~pos_dom & in_eval)[0]
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
    fold_log = []
    targets = head_names + ["any"]
    for f in folds:
        test_ids = clusters.loc[clusters["fold"] == f, "cluster_id"].to_numpy()
        excl_2d = training_exclusion_mask(cluster_2d, test_ids, buffer_px)
        excl = excl_2d[rows, cols] | withheld_excl
        cand = np.where(~excl)[0]
        cand_pos = cand[pos_dom[cand] | (pos_other[cand] & in_bg_train[cand])]
        cand_bg = cand[~(pos_dom[cand] | pos_other[cand]) & in_bg_train[cand]]
        frng = np.random.default_rng(args.seed + 1000 + f)
        bg_take = cand_bg if len(cand_bg) <= args.train_bg_sample else frng.choice(cand_bg, args.train_bg_sample, replace=False)
        tr_idx = np.sort(np.concatenate([cand_pos, bg_take]))
        xs, ys = rasterio.transform.xy(tr, rows[tr_idx], cols[tr_idx])
        fd = FoldData(
            X=X_all[tr_idx].astype(np.float64), y_heads=y_heads[tr_idx], pos_any_other=pos_other[tr_idx],
            rows=rows[tr_idx], cols=cols[tr_idx], explored=explored[tr_idx], access=access[tr_idx],
            w_canopy=w_canopy[tr_idx], tonnage=tonnage[tr_idx], coords=np.column_stack([xs, ys]),
            block_ids=assign_spatial_blocks(rows[tr_idx], cols[tr_idx], H, W, block_size_px=prod_block_px),
        )
        test_cells = np.where(np.isin(cell_cluster, test_ids))[0]
        # Rank against the whole map's background, as a geologist would see it
        eval_idx = np.concatenate([test_cells, bg_eval])
        n_test = len(test_cells)
        entry = {"fold": int(f), "n_test_clusters": int(len(test_ids)), "n_train_cells": int(len(tr_idx)),
                 "n_train_pos_per_head": {h: int((fd.y_heads[:, k] == 1).sum()) for k, h in enumerate(head_names)},
                 "n_excluded_cells": int(excl.sum()), "models": {}}
        for mn, cls in adapters.items():
            ts = time.time()
            model = cls(args, feature_cols, is_geo)
            model.fit(fd)
            Xe = X_all[eval_idx].astype(np.float64)
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
                        sc = s_test[cl_test == cid, ti]
                        dscore = float(np.max(sc))
                        records.append({
                            "model": mn, "variant": variant, "target": tname, "fold": int(f),
                            "cluster_id": int(cid), "n_cells": int(row["n_cells"]),
                            "centroid_x": float(row["centroid_x"]), "centroid_y": float(row["centroid_y"]),
                            "region": row["region"] if "region" in row.index else None,
                            "score_max": dscore, "score_median": float(np.median(sc)),
                            "area_fraction": float(deposit_area_fractions(s_bg[:, ti], [dscore])[0]),
                            "area_fraction_median_cell": float(deposit_area_fractions(s_bg[:, ti], [np.median(sc)])[0]),
                        })
            logger.info("fold %d model %s done in %.1fs", f, mn, time.time() - ts)
        fold_log.append(entry)

    per_dep = pd.DataFrame(records)
    per_dep.to_csv(args.out_dir / "per_deposit.csv", index=False)

    results = {}
    curves = []
    for (mn, variant, tname), g in per_dep.groupby(["model", "variant", "target"], sort=False):
        key = f"{mn}/{variant}"
        a = g["area_fraction"].to_numpy()
        summ = summarize_success_rate(a, DEFAULT_CAPTURE_LEVELS).to_dict()
        summ["success_rate_auc_median_cell"] = (
            float(1.0 - g["area_fraction_median_cell"].mean()) if not summ["insufficient_data"] else None)
        results.setdefault(key, {})[tname] = summ
        c = success_rate_curve(a)
        c["model"], c["variant"], c["target"] = mn, variant, tname
        curves.append(c)
    if curves:
        pd.concat(curves).to_csv(args.out_dir / "success_rate_curves.csv", index=False)

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
        "method_notes": [
            "Deposit = 8-connected cluster of positive cells; score = max over its cells (median-cell variant also reported).",
            "area_fraction = share of background (non-deposit) cells scoring above the deposit; ties count half.",
            "success_rate_auc = 1 - mean(area_fraction); 0.5 = random map.",
            "Held-out clusters and all cells within buffer_m are excluded from training; preprocessing is refit per fold.",
            "Not reproduced: DAE U-Net blend, validation-fold c_hat, warm start.",
        ],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (args.out_dir / "deposit_cv_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(per_dep, targets, args.out_dir)
    _print_table(results, targets)
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


if __name__ == "__main__":
    main()
