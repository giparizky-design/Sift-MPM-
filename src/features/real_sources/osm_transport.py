"""Real OpenStreetMap transport network from Geofabrik shapefile extracts (ODbL).

Extracts: ``asia/indonesia/kalimantan`` and ``asia/malaysia-singapore-brunei`` (Sabah,
Sarawak and Brunei, so distances near the border are not inflated).

Layers (km, distance transform on the grid):

* ``dist_transport_km``    - nearest public road (motorway..tertiary incl. links) or OSM river
* ``dist_major_rivers_km`` - nearest OSM ``waterway=river``
* ``dist_haul_roads_km``   - nearest OSM ``track`` (any grade) / ``unclassified`` road (logging and haul tracks)
* ``dist_roads_km``        - nearest public road (motorway..tertiary incl. links), roads only (v4.15)
* ``dist_ports_km``        - nearest OSM ``ferry_terminal`` (point or area); the best port proxy in OSM (v4.15)

Masks (1 = inside, 0 = outside; not model features):

* ``osm_water_mask``       - OSM ``water`` / ``riverbank`` / ``reservoir`` / ``dock`` polygons (wetlands excluded)
* ``osm_quarry_mask``      - OSM ``landuse=quarry`` polygons (open-pit workings; v4.15 ``active_mine_mask``)

OSM completeness in interior Borneo is uneven; these are distances to mapped features,
not ground truth of accessibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rasterio.features
from scipy.ndimage import distance_transform_edt

from .core import MEASURED, GridSpec, LayerProvenance, ProvenanceManifest, RealDataUnavailableError, download

logger = logging.getLogger(__name__)

GEOFABRIK = {
    "kalimantan": "https://download.geofabrik.de/asia/indonesia/kalimantan-latest-free.shp.zip",
    "malaysia-singapore-brunei": "https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest-free.shp.zip",
}
ROAD_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
TRACK_PREFIX = ("track",)
TRACK_CLASSES = {"unclassified"}
WATER_CLASSES = {"water", "riverbank", "reservoir", "dock"}
QUARRY_CLASSES = {"quarry"}
PORT_CLASSES = {"ferry_terminal"}


def _read_layer(zip_path: Path, layer: str, bbox: tuple[float, float, float, float]):
    import geopandas as gpd

    return gpd.read_file(f"zip://{zip_path}!{layer}", bbox=bbox, columns=["fclass"])


def _load_lines(raw_dir: Path, bbox: tuple[float, float, float, float]) -> dict[str, list]:
    roads, rivers, tracks, ports, water, quarries = [], [], [], [], [], []
    meta = {}
    for name, url in GEOFABRIK.items():
        z = download(url, raw_dir / f"{name}-latest-free.shp.zip", min_bytes=10_000_000)
        r = _read_layer(z, "gis_osm_roads_free_1.shp", bbox)
        w = _read_layer(z, "gis_osm_waterways_free_1.shp", bbox)
        fc = r["fclass"].astype(str)
        roads.extend(r.geometry[fc.isin(ROAD_CLASSES)].tolist())
        tracks.extend(r.geometry[fc.str.startswith(TRACK_PREFIX) | fc.isin(TRACK_CLASSES)].tolist())
        rivers.extend(w.geometry[w["fclass"].astype(str) == "river"].tolist())
        for layer in ("gis_osm_transport_free_1.shp", "gis_osm_transport_a_free_1.shp"):
            t = _read_layer(z, layer, bbox)
            ports.extend(t.geometry[t["fclass"].astype(str).isin(PORT_CLASSES)].tolist())
        wa = _read_layer(z, "gis_osm_water_a_free_1.shp", bbox)
        water.extend(wa.geometry[wa["fclass"].astype(str).isin(WATER_CLASSES)].tolist())
        lu = _read_layer(z, "gis_osm_landuse_a_free_1.shp", bbox)
        quarries.extend(lu.geometry[lu["fclass"].astype(str).isin(QUARRY_CLASSES)].tolist())
        meta[name] = {"file": z.name, "bytes": z.stat().st_size, "mtime": z.stat().st_mtime}
        logger.info("OSM %s: %d road, %d track, %d river lines in bbox", name, int(fc.isin(ROAD_CLASSES).sum()),
                    int((fc.str.startswith(TRACK_PREFIX) | fc.isin(TRACK_CLASSES)).sum()), int((w["fclass"] == "river").sum()))
    if not roads and not rivers:
        raise RealDataUnavailableError("Geofabrik extracts contain no roads or rivers in the grid bbox")
    (raw_dir / "osm_extract_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"roads": roads, "rivers": rivers, "tracks": tracks, "ports": ports, "water": water, "quarries": quarries}


def _poly_mask(geoms: list, grid: GridSpec, all_touched: bool) -> np.ndarray:
    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(grid.shape, dtype=np.float32)
    return rasterio.features.rasterize(shapes, out_shape=grid.shape, transform=grid.transform, fill=0, all_touched=all_touched, dtype="uint8").astype(np.float32)


def _dist_km(geoms: list, grid: GridSpec) -> np.ndarray:
    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.full(grid.shape, np.nan, dtype=np.float32)
    mask = rasterio.features.rasterize(shapes, out_shape=grid.shape, transform=grid.transform, fill=0, all_touched=True, dtype="uint8").astype(bool)
    return (distance_transform_edt(~mask) * grid.res_m / 1000.0).astype(np.float32)


def build_osm_transport_layers(grid: GridSpec, manifest: ProvenanceManifest, raw_dir: Path) -> dict[str, np.ndarray]:
    lines = _load_lines(raw_dir, grid.bounds_4326())
    out = {
        "dist_transport_km": _dist_km(lines["roads"] + lines["rivers"], grid),
        "dist_major_rivers_km": _dist_km(lines["rivers"], grid),
        "dist_haul_roads_km": _dist_km(lines["tracks"], grid),
        "dist_roads_km": _dist_km(lines["roads"], grid),
        "dist_ports_km": _dist_km(lines["ports"], grid),
        "osm_water_mask": _poly_mask(lines["water"], grid, all_touched=False),
        "osm_quarry_mask": _poly_mask(lines["quarries"], grid, all_touched=True),
    }
    common = dict(kind=MEASURED, source="OpenStreetMap (ODbL) via Geofabrik free shapefile extracts (kalimantan, malaysia-singapore-brunei)",
                  url="https://download.geofabrik.de/asia/indonesia/kalimantan.html")
    manifest.add(LayerProvenance(name="dist_transport_km", method="distance to OSM motorway..tertiary (incl. links) roads or rivers", **common))
    manifest.add(LayerProvenance(name="dist_major_rivers_km", method="distance to OSM waterway=river", **common))
    manifest.add(LayerProvenance(name="dist_haul_roads_km", method="distance to OSM track (all grades) / unclassified roads", **common))
    manifest.add(LayerProvenance(name="dist_roads_km", method="distance to OSM motorway..tertiary (incl. links) roads only (no rivers)", **common))
    manifest.add(LayerProvenance(name="dist_ports_km", method="distance to OSM amenity=ferry_terminal (points and areas); proxy for ports, OSM has no port class", **common))
    manifest.add(LayerProvenance(name="osm_water_mask", method="rasterised OSM water/riverbank/reservoir/dock polygons (wetland classes excluded); 1 = water", **common))
    manifest.add(LayerProvenance(name="osm_quarry_mask", method="rasterised OSM landuse=quarry polygons (all-touched); 1 = quarry", **common))
    return out
