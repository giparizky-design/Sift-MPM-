#!/usr/bin/env python3
"""How close does the top of a prospectivity map sit to known labelled deposits?

For a map band, report the share of the top 1% / 5% of valid cells that lies within
2, 5 and 15 km of any known label, next to the share of ALL valid cells within those
distances (the base rate). The ratio is the near-known enrichment: ~1 means the top of
the map is spread like the map as a whole; >>1 means it re-finds known ground.

Known labels: metals = any K=4 head label; coal = K=3 halo labels plus the 1 km
ESDM ``label_coal`` disks.

Example:
    .venv/bin/python scripts/measure_map_novelty.py \
        --data-dir data/east_kalimantan/processed \
        --map outputs/seven_pass_east_kalimantan/maps/prospectivity_k4_metals.tif --band 5 --domain metals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

K4 = ["label_k0_porphyry_cu", "label_k1_skarn_fe_cu", "label_k2_epithermal_au", "label_k3_laterite_fe_ni"]
K3 = ["label_coal_k0_high_cv", "label_coal_k1_med_cv", "label_coal_k2_low_cv", "label_coal"]


def measure(data_dir: Path, map_path: Path, band: int, domain: str,
            dists_km=(2.0, 5.0, 15.0), tops=(0.01, 0.05), mask_map: Path | None = None) -> dict:
    import pyarrow.parquet as pq

    pq_path = data_dir / "dataset.parquet"
    label_cols = [c for c in (K4 if domain == "metals" else K3) if c in pq.read_schema(pq_path).names]
    df = pd.read_parquet(pq_path, columns=["row", "col", *label_cols])
    with rasterio.open(map_path) as src:
        a = src.read(band).astype(np.float64)
        tr = src.transform
        geo = src.crs.is_geographic
    px_m = abs(tr.a) * (111_320.0 if geo else 1.0)
    H, W = a.shape
    known = np.zeros((H, W), bool)
    lab = np.zeros(len(df), bool)
    for c in label_cols:
        lab |= df[c].to_numpy() == 1
    known[df["row"].to_numpy()[lab], df["col"].to_numpy()[lab]] = True
    dist_km = ndimage.distance_transform_edt(~known) * px_m / 1000.0
    r, c = df["row"].to_numpy(), df["col"].to_numpy()
    v, d = a[r, c], dist_km[r, c]
    ok = np.isfinite(v) & ~lab
    if mask_map is not None:  # compare two maps on the same cells
        with rasterio.open(mask_map) as msrc:
            ok &= np.isfinite(msrc.read(1)[r, c])
    v, d, rr, cc = v[ok], d[ok], r[ok], c[ok]
    out = {"map": str(map_path), "band": band, "domain": domain, "n_valid_unlabelled_cells": int(ok.sum()),
           "base": {f"within_{k:g}km": float(np.mean(d <= k)) for k in dists_km}}
    for q in tops:
        top = v >= np.quantile(v, 1 - q)
        e = {}
        for k in dists_km:
            share = float(np.mean(d[top] <= k))
            e[f"within_{k:g}km"] = share
            e[f"enrichment_{k:g}km"] = share / max(out["base"][f"within_{k:g}km"], 1e-12)
        e["median_dist_km"] = float(np.median(d[top]))
        e["centroid_row_col"] = [float(np.median(rr[top])), float(np.median(cc[top]))]
        out[f"top_{q:.2f}"] = e
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument("--domain", choices=("metals", "coal"), required=True)
    ap.add_argument("--mask-map", type=Path, default=None,
                    help="only score cells that are finite in this raster (band 1)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    res = measure(args.data_dir, args.map, args.band, args.domain, mask_map=args.mask_map)
    res["mask_map"] = str(args.mask_map) if args.mask_map else None
    print(json.dumps(res, indent=2))
    if args.out:
        args.out.write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    main()
