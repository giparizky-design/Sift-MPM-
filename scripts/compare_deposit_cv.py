#!/usr/bin/env python3
"""Paired, per-deposit comparison of two ``run_deposit_cv.py`` runs.

Both runs must hold out the same deposit clusters (same data, folds and buffer).
For every model/variant/target present in both, it reports:

* ΔAUC = AUC(candidate) − AUC(base), with a bootstrap 95% CI over deposits,
* the Wilcoxon signed-rank p-value on per-deposit area fractions,
* how many deposits rank better / worse,
* the change in novelty (share of the top 5% of area within 5 km / 15 km of known labels).

Different models or variant names (e.g. the v4.9 U-Net blends against xgb_prod) are paired with
``--cand-model`` and ``--pairs base_variant:cand_variant``.

Acceptance rule used in audit v4.8: a change is accepted when the ΔAUC CI for
"any" includes or exceeds zero (no significant paired drop).

Example:
    .venv/bin/python scripts/compare_deposit_cv.py \
        outputs/deposit_cv/v48/metals_M0_baseline outputs/deposit_cv/v48/metals_M1_hajek
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired(base: pd.DataFrame, cand: pd.DataFrame, n_boot: int = 4000, seed: int = 0) -> dict:
    m = base.merge(cand, on="cluster_id", suffixes=("_b", "_c"))
    if m.empty:
        return {}
    a_b, a_c = m["area_fraction_b"].to_numpy(), m["area_fraction_c"].to_numpy()
    d = a_b - a_c  # positive = candidate ranks the deposit higher
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    nz = d[np.abs(d) > 1e-12]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 2 else 1.0
    return {
        "n": int(len(d)),
        "auc_base": float(1 - a_b.mean()),
        "auc_cand": float(1 - a_c.mean()),
        "delta_auc": float(d.mean()),
        "delta_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "wilcoxon_p": p,
        "better": int((d > 1e-12).sum()),
        "worse": int((d < -1e-12).sum()),
    }


def novelty_delta(sb: dict, sc: dict, key_b: str, key_c: str) -> dict:
    out = {}
    nb = sb.get("novelty", {}).get(key_b, {}).get("any", {})
    nc = sc.get("novelty", {}).get(key_c, {}).get("any", {})
    for tq in ("top_0.01", "top_0.05"):
        if tq in nb and tq in nc:
            for dm in (5000, 15000):
                k = f"enrichment_within_{dm}m"
                if k in nb[tq] and k in nc[tq]:
                    out[f"{tq}_{k}"] = {"base": nb[tq][k], "cand": nc[tq][k]}
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("cand", type=Path)
    ap.add_argument("--model", default="xgb_prod")
    ap.add_argument("--cand-model", default=None, help="model name in the candidate run (default: --model)")
    ap.add_argument("--pairs", default="", help="comma list of base_variant:cand_variant pairs to compare "
                    "(default: every variant present in both runs, matched by name)")
    ap.add_argument("--targets", default="any", help="comma list of targets ('all' for every target)")
    ap.add_argument("--metric", choices=("max", "median_cell"), default="max",
                    help="deposit score: max over the cluster's cells (default) or its median cell, which is "
                    "free of the max-over-cluster null inflation found in audit v4.10")
    ap.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = ap.parse_args(argv)
    cand_model = args.cand_model or args.model
    col = "area_fraction" if args.metric == "max" else "area_fraction_median_cell"

    pb = pd.read_csv(args.base / "per_deposit.csv")
    pc = pd.read_csv(args.cand / "per_deposit.csv")
    sb = json.loads((args.base / "deposit_cv_summary.json").read_text())
    sc = json.loads((args.cand / "deposit_cv_summary.json").read_text())
    pb, pc = pb[pb["model"] == args.model], pc[pc["model"] == cand_model]
    targets = sorted(set(pb["target"]) & set(pc["target"])) if args.targets == "all" else \
        [t.strip() for t in args.targets.split(",") if t.strip()]
    if args.pairs.strip():
        pairs = [tuple(p.split(":", 1)) for p in args.pairs.split(",") if p.strip()]
    else:
        pairs = [(v, v) for v in sorted(set(pb["variant"]) & set(pc["variant"]))]

    res = {}
    print(f"{'variant':14s} {'target':16s} {'n':>4s} {'AUC base':>9s} {'AUC cand':>9s} {'ΔAUC':>7s} "
          f"{'95% CI':>17s} {'p':>7s} {'better/worse':>13s}")
    for vb, vc in pairs:
        v = vb if vb == vc else f"{vb}->{vc}"
        for t in targets:
            gb = pb[(pb["variant"] == vb) & (pb["target"] == t)][["cluster_id", col]].rename(
                columns={col: "area_fraction"})
            gc = pc[(pc["variant"] == vc) & (pc["target"] == t)][["cluster_id", col]].rename(
                columns={col: "area_fraction"})
            r = paired(gb, gc)
            if not r:
                continue
            key_b, key_c = f"{args.model}/{vb}", f"{cand_model}/{vc}"
            key = key_b if key_b == key_c else f"{key_b}->{key_c}"
            r["novelty"] = novelty_delta(sb, sc, key_b, key_c) if t == "any" else {}
            res[f"{key}/{t}"] = r
            print(f"{v:14s} {t:16s} {r['n']:4d} {r['auc_base']:9.3f} {r['auc_cand']:9.3f} {r['delta_auc']:+7.3f} "
                  f"[{r['delta_ci95'][0]:+.3f},{r['delta_ci95'][1]:+.3f}] {r['wilcoxon_p']:7.3f} "
                  f"{r['better']:6d}/{r['worse']:<6d}")
            for nk, nv in r["novelty"].items():
                print(f"{'':14s}   {nk:44s} {nv['base']:.2f} -> {nv['cand']:.2f}")
    if args.out:
        args.out.write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    main()
