"""Deposit-level spatial cross-validation and success-rate metrics.

Pixel PR-AUC treats each labelled cell as an independent sample. In this
repository every deposit point is rasterized as a ~1 km disk, so a single
deposit becomes 50 or more "positives" and the pixel metric mostly rewards
hitting large disks. The unit of evidence in prospectivity mapping is the
deposit, so this module evaluates at that level:

1. Connected positive cells are grouped into deposit clusters.
2. Clusters are grouped by spatial block and whole blocks are assigned to
   folds, so neighbouring deposits are held out together.
3. While a fold is held out, its clusters and every cell within
   ``buffer_m`` of them are removed from training.
4. Each held-out deposit is scored by the fold model. Its *area fraction*
   is the share of background (non-deposit) cells that score higher, that is
   the share of the map a geologist would have to flag to capture it.
5. The success-rate curve is the share of held-out deposits captured as a
   function of flagged area. A random map gives the diagonal (AUC 0.5).

The functions here are model-agnostic. ``scripts/run_deposit_cv.py`` wires
them to the production ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage, stats

DEFAULT_CAPTURE_LEVELS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
MIN_DEPOSITS_FOR_METRICS = 5


def label_deposit_clusters(pos_2d: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Label connected positive cells as deposit clusters.

    Args:
        pos_2d: Boolean (H, W) raster of positive cells.
        connectivity: 4 or 8 neighbour connectivity.

    Returns:
        int32 (H, W) raster with cluster ids 0..n-1 and -1 for non-positive cells.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    structure = np.ones((3, 3), dtype=bool) if connectivity == 8 else ndimage.generate_binary_structure(2, 1)
    lab, _ = ndimage.label(np.asarray(pos_2d, dtype=bool), structure=structure)
    return (lab.astype(np.int32) - 1)


def summarize_clusters(cluster_2d: np.ndarray, head_masks_2d: Optional[np.ndarray] = None,
                       head_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One row per cluster: size, centroid (row, col) and which heads it contains."""
    ids = cluster_2d[cluster_2d >= 0]
    n = int(ids.max()) + 1 if ids.size else 0
    rr, cc = np.nonzero(cluster_2d >= 0)
    cid = cluster_2d[rr, cc]
    n_cells = np.bincount(cid, minlength=n)
    df = pd.DataFrame({
        "cluster_id": np.arange(n, dtype=int),
        "n_cells": n_cells,
        "centroid_row": np.bincount(cid, weights=rr, minlength=n) / np.maximum(n_cells, 1),
        "centroid_col": np.bincount(cid, weights=cc, minlength=n) / np.maximum(n_cells, 1),
    })
    if head_masks_2d is not None:
        names = list(head_names) if head_names is not None else [f"head_{k}" for k in range(head_masks_2d.shape[0])]
        for k, name in enumerate(names):
            hk = head_masks_2d[k][rr, cc].astype(bool)
            df[f"has_{name}"] = np.bincount(cid, weights=hk, minlength=n) > 0
    return df


def assign_cluster_folds(clusters: pd.DataFrame, block_px: int, n_folds: int = 5,
                         random_state: int = 42) -> np.ndarray:
    """Assign clusters to folds by the spatial block holding their centroid.

    Whole blocks go to one fold, so clusters within the same block are always
    held out together. ``n_folds=0`` means leave-one-block-out.

    Blocks are shuffled, then assigned greedily (largest block first) to the
    fold with the fewest clusters so far, which keeps folds balanced.
    """
    if len(clusters) == 0:
        return np.zeros(0, dtype=int)
    if block_px < 1:
        raise ValueError("block_px must be >= 1")
    br = (clusters["centroid_row"].to_numpy() // block_px).astype(np.int64)
    bc = (clusters["centroid_col"].to_numpy() // block_px).astype(np.int64)
    block_key = br * 1_000_003 + bc
    uniq, inv = np.unique(block_key, return_inverse=True)
    n_blocks = len(uniq)
    if n_folds == 0 or n_folds >= n_blocks:
        return inv.astype(int)
    rng = np.random.default_rng(random_state)
    counts = np.bincount(inv, minlength=n_blocks)
    order = rng.permutation(n_blocks)
    order = order[np.argsort(-counts[order], kind="stable")]
    fold_of_block = np.empty(n_blocks, dtype=int)
    load = np.zeros(n_folds, dtype=int)
    for b in order:
        f = int(np.argmin(load))
        fold_of_block[b] = f
        load[f] += counts[b]
    return fold_of_block[inv]


def training_exclusion_mask(cluster_2d: np.ndarray, test_cluster_ids: Sequence[int],
                            buffer_px: float) -> np.ndarray:
    """Cells that must not be used for training while ``test_cluster_ids`` are held out.

    Returns True for the held-out clusters' cells and every cell within
    ``buffer_px`` (Euclidean, in pixels) of them.
    """
    test_2d = np.isin(cluster_2d, np.asarray(list(test_cluster_ids), dtype=np.int32))
    if not test_2d.any():
        return test_2d
    if buffer_px <= 0:
        return test_2d
    dist = ndimage.distance_transform_edt(~test_2d)
    return dist <= float(buffer_px)


def deposit_area_fractions(background_scores: np.ndarray, deposit_scores: np.ndarray) -> np.ndarray:
    """Share of background cells scoring above each deposit (ties count half).

    0 means the deposit is the top-ranked location; 0.5 is what a random map gives.
    """
    bg = np.sort(np.asarray(background_scores, dtype=np.float64))
    bg = bg[np.isfinite(bg)]
    if bg.size == 0:
        raise ValueError("background_scores is empty")
    d = np.asarray(deposit_scores, dtype=np.float64)
    below = np.searchsorted(bg, d, side="left")
    below_or_equal = np.searchsorted(bg, d, side="right")
    above = bg.size - below_or_equal
    ties = below_or_equal - below
    frac = (above + 0.5 * ties) / bg.size
    return np.where(np.isfinite(d), frac, 1.0)


def success_rate_curve(area_fracs: np.ndarray, grid: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Share of deposits captured (area fraction <= f) for each flagged-area level f."""
    a = np.asarray(area_fracs, dtype=np.float64)
    g = np.linspace(0.0, 1.0, 201) if grid is None else np.asarray(grid, dtype=np.float64)
    captured = (a[None, :] <= g[:, None]).mean(axis=1) if a.size else np.full(g.shape, np.nan)
    return pd.DataFrame({"area_fraction": g, "deposits_captured": captured})


@dataclass
class SuccessRateSummary:
    n_deposits: int
    insufficient_data: bool
    auc: Optional[float]
    auc_ci95: Optional[tuple[float, float]]
    p_value_vs_random: Optional[float]
    median_area_fraction: Optional[float]
    capture: dict

    def to_dict(self) -> dict:
        return {
            "n_deposits": self.n_deposits,
            "insufficient_data": self.insufficient_data,
            "success_rate_auc": self.auc,
            "success_rate_auc_ci95": list(self.auc_ci95) if self.auc_ci95 else None,
            "p_value_vs_random": self.p_value_vs_random,
            "median_area_fraction": self.median_area_fraction,
            "capture": self.capture,
        }


def summarize_success_rate(area_fracs: np.ndarray,
                           capture_levels: Sequence[float] = DEFAULT_CAPTURE_LEVELS,
                           n_boot: int = 2000, n_null: int = 200_000,
                           min_deposits: int = MIN_DEPOSITS_FOR_METRICS,
                           random_state: int = 0) -> SuccessRateSummary:
    """Success-rate AUC, bootstrap CI, one-sided p-value against a random map, and capture rates.

    AUC = 1 - mean(area fraction). Under a random map each area fraction is
    Uniform(0, 1), so the p-value is P(mean of n uniforms <= observed mean),
    estimated by Monte Carlo. Capture p-values are one-sided binomial tests
    with success probability equal to the flagged-area level.

    With fewer than ``min_deposits`` deposits nothing is computed and the
    summary is flagged ``insufficient_data``.
    """
    a = np.asarray(area_fracs, dtype=np.float64)
    n = int(a.size)
    if n < min_deposits:
        return SuccessRateSummary(n, True, None, None, None, None, {})
    rng = np.random.default_rng(random_state)
    auc = float(1.0 - a.mean())
    boot = 1.0 - a[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    ci = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
    null_means = rng.random((n_null, n), dtype=np.float32).mean(axis=1) if n * n_null <= 50_000_000 else \
        rng.normal(0.5, np.sqrt(1.0 / (12.0 * n)), size=n_null)
    p = float((np.sum(null_means <= a.mean()) + 1) / (n_null + 1))
    capture = {}
    for f in capture_levels:
        k = int(np.sum(a <= f))
        capture[f"{f:.2f}"] = {
            "captured": k,
            "rate": k / n,
            "lift": (k / n) / f,
            "p_value": float(stats.binom.sf(k - 1, n, f)),
        }
    return SuccessRateSummary(n, False, auc, ci, p, float(np.median(a)), capture)
