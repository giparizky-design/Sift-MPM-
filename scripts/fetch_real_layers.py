#!/usr/bin/env python3
"""Download / composite the real source layers for the Kalimantan 250 m master grid into caches.

Run one source per process so slow network sources proceed in parallel:

    .venv/bin/python scripts/fetch_real_layers.py --source s2
    .venv/bin/python scripts/fetch_real_layers.py --source landsat
    ...

Sources: s2, landsat, aster, emag2, hansen, bedrock, osm, palsar, dem.
Nothing here generates values; a source that cannot be fetched fails loudly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.real_sources.core import GridSpec, ProvenanceManifest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fetch_real_layers")

# Kalimantan master grid (identical to scripts/harvest_kalimantan_multimodal.py). The East
# Kalimantan grid (114.5..119.5E, -2.8..3.2N) is an exact window of it.
MASTER_GRID = GridSpec(crs="EPSG:4326", transform=Affine(0.0025, 0.0, 108.5, 0.0, -0.0025, 4.5), height=3600, width=4400)
REAL_ROOT = ROOT / "data" / "kalimantan_real"
CACHE = REAL_ROOT / "cache"
RAW_REAL = ROOT / "data" / "raw_real"


def fetch(source: str, workers: int) -> None:
    m = ProvenanceManifest(MASTER_GRID)
    g = MASTER_GRID
    if source == "s2":
        from src.features.real_sources.sentinel2 import build_sentinel2_layers

        build_sentinel2_layers(g, m, CACHE / "sentinel2", max_workers=workers)
    elif source == "landsat":
        from src.features.real_sources.landsat_thermal import build_landsat_thermal_layers

        # dem only affects the derived anomaly, which is recomputed at cube build time
        from src.features.real_sources.core import LayerProvenance, MEASURED

        m.add(LayerProvenance(name="dem_elevation", kind=MEASURED, source="(fetch stage only)"))
        build_landsat_thermal_layers(g, m, CACHE / "landsat", dem=np.zeros(g.shape, np.float32), max_workers=workers)
    elif source == "aster":
        from src.features.real_sources.aster_tir import build_aster_tir_layers

        build_aster_tir_layers(g, m, CACHE / "aster", max_workers=workers)
    elif source == "emag2":
        from src.features.real_sources.global_grids import build_emag2_layers

        build_emag2_layers(g, m, RAW_REAL / "emag2")
    elif source == "hansen":
        from src.features.real_sources.global_grids import build_hansen_gfc_layers
        from src.features.real_sources.core import save_layer

        out = build_hansen_gfc_layers(g, m, RAW_REAL / "hansen_gfc")
        for k, v in out.items():
            save_layer(v, g, CACHE / "hansen" / f"{k}.tif")
    elif source == "bedrock":
        from src.features.real_sources.global_grids import build_depth_to_bedrock

        build_depth_to_bedrock(g, m, RAW_REAL / "soilgrids")
    elif source == "osm":
        from src.features.real_sources.osm_transport import build_osm_transport_layers

        build_osm_transport_layers(g, m, RAW_REAL / "osm")
    elif source == "palsar":
        from src.features.sar_palsar2 import build_real_sar_channels

        dem = _dem()
        chans, prov = build_real_sar_channels(
            dst_transform=g.transform, dst_shape=g.shape, dem=dem, res_m=250.0, dst_crs="EPSG:4326", cache_dir=CACHE / "palsar2"
        )
        prov.write(CACHE / "palsar2" / "sar_provenance.json")
        from src.features.real_sources.core import save_layer

        for k, v in chans.items():
            save_layer(v, g, CACHE / "palsar2" / f"{k}.tif")
    elif source == "dem":
        _dem()
    else:
        raise SystemExit(f"unknown source {source}")
    logger.info("Fetch of %s complete", source)


def _dem() -> np.ndarray:
    from scripts.run_indonesia_hd_upgrade import fetch_terrarium_dem

    b = MASTER_GRID.bounds
    cache_tif = CACHE / "dem" / "terrarium_z10_master_250m.tif"
    cache_tif.parent.mkdir(parents=True, exist_ok=True)
    return fetch_terrarium_dem(b[0], b[1], b[2], b[3], MASTER_GRID.height, MASTER_GRID.width, zoom=10, cache_tif=cache_tif)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    fetch(a.source, a.workers)
