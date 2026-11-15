#!/usr/bin/env python3
"""Attach archipelago-wide K=4 labels and island regions to the national 3.5 km feature cube.

Reads ``data/indonesia/processed`` (features, grid) and writes
``data/indonesia_k4/processed/dataset.parquet`` with:
  * the national feature columns (unchanged),
  * ``label_k0_porphyry_cu`` .. ``label_k3_laterite_fe_ni`` and ``label`` (any), in {1, -1},
  * ``region``: island group per cell (kalimantan, sulawesi, sumatra, java_bali,
    nusa_tenggara, maluku, papua).

Regions are assigned per connected landmass (DEM > 0, 8-connected) by majority
vote of the GeoRIMA ``island_region`` of nodes on it; remaining cells (shelf,
landmasses without nodes) take the region of the nearest node.

Label rules are in ``src/labels/archipelago_labels.py``. A provenance JSON
records counts by head, source and region.

Feature provenance warning: the national cube's ``bouguer_gravity_mgal``,
``crustal_thickness_km`` and ``moho_gravity_step_mgal_km`` come from a
Kalimantan-only gravity file, nearest-filled elsewhere (audit v4.7). They are
kept in the parquet for traceability but must not be used as features.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labels.archipelago_labels import (  # noqa: E402
    K4_TARGETS,
    build_archipelago_k4_targets,
    load_archipelago_sources,
    points_to_cells,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("build_archipelago_k4")

SRC_DIR = ROOT / "data" / "indonesia" / "processed"
DEST_DIR = ROOT / "data" / "indonesia_k4" / "processed"
REGION_KEYS = {
    "Kalimantan (Borneo)": "kalimantan",
    "Sulawesi & Sangihe": "sulawesi",
    "Sumatra & Tin Belt": "sumatra",
    "Java & Bali": "java_bali",
    "Nusa Tenggara & Banda Arc": "nusa_tenggara",
    "Maluku & Halmahera": "maluku",
    "Papua & Bird's Head": "papua",
}
UNRELIABLE_FEATURES = ["bouguer_gravity_mgal", "crustal_thickness_km", "moho_gravity_step_mgal_km"]
LABEL_COLS = {
    "k0_porphyry_cu": "label_k0_porphyry_cu",
    "k1_skarn_fe_cu": "label_k1_skarn_fe_cu",
    "k2_epithermal_au": "label_k2_epithermal_au",
    "k3_laterite_fe_ni": "label_k3_laterite_fe_ni",
}


def assign_regions(dem: np.ndarray, transform, nodes: pd.DataFrame) -> np.ndarray:
    """Per-cell region key for the whole grid."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    H, W = dem.shape
    nodes = nodes[nodes["island_region"].isin(REGION_KEYS)].copy()
    nodes["key"] = nodes["island_region"].map(REGION_KEYS)
    r, c, ok = points_to_cells(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), transform, H, W)
    nodes, r, c = nodes[ok], r[ok], c[ok]

    land = np.isfinite(dem) & (dem > 0)
    comp, n_comp = ndimage.label(land, structure=np.ones((3, 3), bool))
    region = np.full((H, W), "", dtype=object)
    node_comp = comp[r, c]
    vote = pd.DataFrame({"comp": node_comp, "key": nodes["key"].to_numpy()})
    vote = vote[vote["comp"] > 0]
    if len(vote):
        winner = vote.groupby("comp")["key"].agg(lambda s: s.value_counts().index[0])
        lut = np.full(n_comp + 1, "", dtype=object)
        lut[winner.index.to_numpy()] = winner.to_numpy()
        region = lut[comp]
    # Remaining cells: nearest node
    rr, cc = np.nonzero(region == "")
    tree = cKDTree(np.column_stack([r, c]))
    _, idx = tree.query(np.column_stack([rr, cc]))
    region[rr, cc] = nodes["key"].to_numpy()[idx]
    return region


def main() -> None:
    import rasterio

    df = pd.read_parquet(SRC_DIR / "dataset.parquet")
    with rasterio.open(SRC_DIR / "dem_elevation.tif") as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        H, W = src.height, src.width
        nodata = src.nodata
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    logger.info("national cube: %d cells, grid %dx%d", len(df), H, W)

    cell_index = np.full((H, W), -1, dtype=np.int64)
    cell_index[df["row"].to_numpy(), df["col"].to_numpy()] = np.arange(len(df))

    src_tabs = load_archipelago_sources()
    targets = build_archipelago_k4_targets(src_tabs["georima"], src_tabs["mrds_indonesia"],
                                           src_tabs["iron"], src_tabs["borneo_mrds"])
    region_grid = assign_regions(dem, transform, src_tabs["georima"])
    df["region"] = region_grid[df["row"].to_numpy(), df["col"].to_numpy()].astype(str)

    prov = {"generated_at": datetime.now(timezone.utc).isoformat(), "source_dir": str(SRC_DIR),
            "unreliable_features_do_not_use": UNRELIABLE_FEATURES, "heads": {}}
    any_pos = np.zeros(len(df), dtype=bool)
    for t in K4_TARGETS:
        pts = targets[t]
        r, c, ok = points_to_cells(pts["lon"].to_numpy(), pts["lat"].to_numpy(), transform, H, W)
        idx = np.full(len(pts), -1, dtype=np.int64)
        idx[ok] = cell_index[r[ok], c[ok]]
        kept = idx >= 0
        lab = np.full(len(df), -1, dtype=np.int32)
        lab[idx[kept]] = 1
        df[LABEL_COLS[t]] = lab
        any_pos |= lab == 1
        pts = pts.assign(kept=kept, region=np.where(kept, df["region"].to_numpy()[np.maximum(idx, 0)], ""))
        prov["heads"][t] = {
            "n_points": int(len(pts)),
            "n_points_on_grid_cells": int(kept.sum()),
            "n_points_dropped_offgrid_or_deep_ocean": int((~kept).sum()),
            "n_positive_cells": int((lab == 1).sum()),
            "points_by_source": pts["source"].value_counts().to_dict(),
            "points_by_region": pts.loc[pts["kept"], "region"].value_counts().to_dict(),
        }
        logger.info("%s: %d points -> %d cells (%d dropped)", t, len(pts), int((lab == 1).sum()), int((~kept).sum()))
    df["label"] = np.where(any_pos, 1, -1).astype(np.int32)
    prov["cells_by_region"] = df["region"].value_counts().to_dict()

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DEST_DIR / "dataset.parquet", index=False)
    link = DEST_DIR / "dem_elevation.tif"
    if not link.exists():
        os.symlink(os.path.relpath(SRC_DIR / "dem_elevation.tif", DEST_DIR), link)
    (DEST_DIR / "label_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    logger.info("wrote %s (%d cells, %d positive cells)", DEST_DIR, len(df), int(any_pos.sum()))


if __name__ == "__main__":
    main()
