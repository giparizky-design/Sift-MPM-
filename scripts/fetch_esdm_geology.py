#!/usr/bin/env python3
"""Fetch open Kalimantan geology / fault / mineral-occurrence vector layers from ArcGIS REST services.

Record-only fetcher. Nothing is digitised, filled, merged or reclassified: every feature is saved
with the attributes the service returned, reprojected server-side to EPSG:4326 (``outSR=4326``).

Sources are kept in separate output folders so one publisher's data is never presented as another's:

* ``data/raw_real/esdm_geology/``      ESDM / Badan Geologi geoportal (geoportal.esdm.go.id/gis4).
* ``data/raw_real/big_atlas_geologi/`` Badan Informasi Geospasial (BIG) "Atlas 250K" map services.
* ``data/raw_real/bnpb_inarisk_faults/`` BNPB inaRISK fault services (PuSGeN source-model faults,
  PVMBG large-scale active faults).

Usage (from repo root)::

    .venv/bin/python scripts/fetch_esdm_geology.py                      # all sources, Kalimantan bbox
    .venv/bin/python scripts/fetch_esdm_geology.py --bbox 114.5 -2.8 119.5 3.2
    .venv/bin/python scripts/fetch_esdm_geology.py --only esdm_geology  # one source group
    .venv/bin/python scripts/fetch_esdm_geology.py --force               # ignore completed pages

Resumable: every page is written to ``<group>/raw/<layer>/page_NNNN.geojson`` and skipped on rerun;
a layer whose GeoPackage already exists with the expected feature count is skipped.

Paging: features are selected with ``returnIdsOnly`` inside the bbox, then fetched in object-ID
chunks no larger than the layer's ``maxRecordCount``. This works for services that do not support
``resultOffset`` pagination (several shapefile-backed MapServer layers here return
"Pagination is not supported") and gives stable, resumable page boundaries.

Politeness: one request at a time, exponential backoff on 429/5xx (5 tries per request), and a
service is abandoned after persistent failure (recorded in provenance.json with the HTTP status).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_REAL = ROOT / "data" / "raw_real"
KALIMANTAN_BBOX = (108.5, -4.5, 119.5, 4.5)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fetch_esdm_geology")

UA = "Geospatials-MPM-research-fetcher/1.0 (record-only; contact via repo owner)"
TIMEOUT = 120
MAX_TRIES = 5
RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass
class LayerSpec:
    group: str                 # output folder under data/raw_real/
    name: str                  # output gpkg stem
    service: str               # MapServer / FeatureServer URL (no layer id)
    layer_id: int
    kind: str                  # what the layer is, in plain words
    publisher: str             # who publishes the service (as stated by the service / portal item)
    use_bbox: bool = True      # False = take the whole layer (tiny national layers)
    notes: str = ""


ESDM = "https://geoportal.esdm.go.id/gis4/rest/services"
BIG = "https://geoservices.big.go.id/gis/rest/services/PTRA"
BNPB = "https://gis.bnpb.go.id/server/rest/services/inarisk"

LAYERS: list[LayerSpec] = [
    # --- ESDM / Badan Geologi geoportal (primary target; down with HTTP 503 on 2026-11-10) ---
    LayerSpec("esdm_geology", "geology_units", f"{ESDM}/bgl_bgs_pm/BGL_BGS_PM_Geologi_Litologi/MapServer", 0,
              "Regional geological units (lithology), Badan Geologi Pusat Survei Geologi", "ESDM / Badan Geologi",
              notes="URL from ArcGIS Online item 97a1c92a930045a2860fcb821fc23efe ('Geologi _ Litologi Regional Indonesia')."),
    LayerSpec("esdm_geology", "faults", f"{ESDM}/bgl_bgs_pm/BGL_BGS_PM_Geologi_Geostruktur/MapServer", 0,
              "Regional geological structures (faults, folds)", "ESDM / Badan Geologi",
              notes="URL from ArcGIS Online item ded22d0f08e84c8ba698e490515174b1 ('Geologi _ Struktur Geologi Regional')."),
    LayerSpec("esdm_geology", "coal_bearing_formations", f"{ESDM}/BGD_TU/Formasi_Pembawa_Batubara/MapServer", 0,
              "Coal-bearing formations (already cached elsewhere in data/raw_real/esdm/)", "ESDM / Badan Geologi",
              notes="Endpoint used by scripts/build_real_feature_cubes.py."),
    # --- BIG Atlas 250K (separate publisher; derived national atlas, not the ESDM sheets themselves) ---
    LayerSpec("big_atlas_geologi", "geology_units", f"{BIG}/Atlas_250K_Geologi/MapServer", 15,
              "Geologi Regional polygons (FORMASI, SANDI, PENYUSUN, ages)", "Badan Informasi Geospasial (BIG)"),
    LayerSpec("big_atlas_geologi", "faults", f"{BIG}/Atlas_250K_Geologi/MapServer", 7,
              "Jenis Sesar (fault type / name) lines", "Badan Informasi Geospasial (BIG)"),
    LayerSpec("big_atlas_geologi", "tectonics_orogen", f"{BIG}/Atlas_250K_Geologi/MapServer", 13,
              "Tektonika Orogen polygons", "Badan Informasi Geospasial (BIG)"),
    LayerSpec("big_atlas_geologi", "tectonics_neogen", f"{BIG}/Atlas_250K_Geologi/MapServer", 14,
              "Tektonika Neogen crustal-element polygons", "Badan Informasi Geospasial (BIG)"),
    LayerSpec("big_atlas_geologi", "metal_mineral_points", f"{BIG}/Atlas_250K_Logam/MapServer", 4,
              "Mineral Logam points (commodity group only; atlas symbol points)", "Badan Informasi Geospasial (BIG)"),
    LayerSpec("big_atlas_geologi", "metal_bearing_rocks", f"{BIG}/Atlas_250K_Logam/MapServer", 9,
              "Batuan Pembawa Logam (metal-bearing rock unit) polygons", "Badan Informasi Geospasial (BIG)"),
    LayerSpec("big_atlas_geologi", "nonmetal_resource_points", f"{BIG}/Atlas_250K_NonLogam/MapServer", 4,
              "Sumber Daya Geologi Non Logam points (category only)", "Badan Informasi Geospasial (BIG)"),
    # --- BNPB inaRISK fault services (separate publisher; redistributes PuSGeN / PVMBG fault models) ---
    LayerSpec("bnpb_inarisk_faults", "pusgen_kalimantan_faults", f"{BNPB}/Faults/MapServer", 2,
              "Kalimantan_Fault: seismic source fault segments (Mmax, slip rate, type)", "BNPB inaRISK", use_bbox=False,
              notes="Layer naming and attributes (Mmax_d, Sliprate_1, CLASSSTR) match the PuSGeN 2017 source model; "
                    "the service itself carries no description or citation."),
    LayerSpec("bnpb_inarisk_faults", "pusgen_2024_shallow_crustal", f"{BNPB}/Faults_new/MapServer", 1,
              "Pusgen_2024_Shallow_Crustal_v6 fault sources", "BNPB inaRISK",
              notes="Layer name says PuSGeN 2024 v6, not the 2017 model."),
    LayerSpec("bnpb_inarisk_faults", "pvmbg_active_faults_large_scale", f"{BNPB}/Faults_new/MapServer", 0,
              "patahan_aktif_skalabesar_pvmbg (large-scale active faults, PVMBG)", "BNPB inaRISK"),
]


class ServiceFailed(RuntimeError):
    def __init__(self, msg: str, status: int | None):
        super().__init__(msg)
        self.status = status


@dataclass
class Fetcher:
    session: requests.Session = field(default_factory=requests.Session)
    dead_hosts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session.headers["User-Agent"] = UA

    def request(self, url: str, params: dict, method: str = "GET") -> dict:
        host = url.split("/")[2]
        if host in self.dead_hosts:
            raise ServiceFailed(f"host abandoned earlier: {self.dead_hosts[host]}", None)
        last_status: int | None = None
        last_err = ""
        for attempt in range(MAX_TRIES):
            try:
                if method == "POST":
                    r = self.session.post(url, data=params, timeout=TIMEOUT)
                else:
                    r = self.session.get(url, params=params, timeout=TIMEOUT)
                last_status = r.status_code
                if r.status_code in RETRY_STATUS:
                    last_err = f"HTTP {r.status_code}: {r.text[:200].strip()}"
                elif r.status_code != 200:
                    raise ServiceFailed(f"HTTP {r.status_code}: {r.text[:300].strip()}", r.status_code)
                else:
                    try:
                        d = r.json()
                    except ValueError:
                        raise ServiceFailed(f"non-JSON 200 response: {r.text[:300].strip()}", 200)
                    if isinstance(d, dict) and "error" in d:
                        code = d["error"].get("code")
                        last_err = f"ArcGIS error {code}: {d['error'].get('message')} {d['error'].get('details')}"
                        last_status = code
                        if code not in RETRY_STATUS:
                            raise ServiceFailed(last_err, code)
                    else:
                        return d
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
                last_status = None
            wait = 2 ** attempt * 2
            log.warning("  %s (try %d/%d) -> retry in %ds", last_err[:160], attempt + 1, MAX_TRIES, wait)
            time.sleep(wait)
        self.dead_hosts[host] = last_err
        raise ServiceFailed(f"persistent failure after {MAX_TRIES} tries: {last_err}", last_status)


def bbox_params(bbox: tuple[float, float, float, float]) -> dict:
    return {"geometry": ",".join(str(v) for v in bbox), "geometryType": "esriGeometryEnvelope",
            "inSR": "4326", "spatialRel": "esriSpatialRelIntersects"}


def fetch_layer(f: Fetcher, spec: LayerSpec, bbox, force: bool) -> dict:
    gdir = RAW_REAL / spec.group
    rdir = gdir / "raw" / spec.name
    rdir.mkdir(parents=True, exist_ok=True)
    layer_url = f"{spec.service}/{spec.layer_id}"
    rec: dict = {"layer": spec.name, "kind": spec.kind, "publisher": spec.publisher, "service_url": spec.service,
                 "layer_id": spec.layer_id, "layer_url": layer_url, "notes": spec.notes,
                 "retrieval_date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}

    svc_meta = f.request(spec.service, {"f": "json"})
    (gdir / "raw" / f"{spec.name}__service.json").write_text(json.dumps(svc_meta, indent=1, ensure_ascii=False))
    meta = f.request(layer_url, {"f": "json"})
    (gdir / "raw" / f"{spec.name}__layer.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    rec["terms_as_stated"] = {
        "service.serviceDescription": svc_meta.get("serviceDescription"),
        "service.description": svc_meta.get("description"),
        "service.copyrightText": svc_meta.get("copyrightText"),
        "layer.description": meta.get("description"),
        "layer.copyrightText": meta.get("copyrightText"),
    }
    rec["layer_name_on_service"] = meta.get("name")
    rec["geometry_type"] = meta.get("geometryType")
    rec["max_record_count"] = meta.get("maxRecordCount") or svc_meta.get("maxRecordCount") or 1000
    rec["supports_pagination"] = meta.get("advancedQueryCapabilities", {}).get("supportsPagination")
    rec["attributes"] = [{"name": x["name"], "alias": x.get("alias"), "type": x["type"]} for x in meta.get("fields", [])]
    oid_field = meta.get("objectIdField") or next((x["name"] for x in meta.get("fields", []) if x["type"] == "esriFieldTypeOID"), None)
    rec["object_id_field"] = oid_field

    id_params = {"where": "1=1", "returnIdsOnly": "true", "f": "json"}
    if spec.use_bbox:
        id_params.update(bbox_params(bbox))
    ids = f.request(f"{layer_url}/query", id_params).get("objectIds") or []
    ids = sorted(ids)
    chunk = min(int(rec["max_record_count"]), 500)
    rec["query"] = {
        "step1_ids": {"url": f"{layer_url}/query", "params": id_params},
        "step2_pages": {"url": f"{layer_url}/query", "method": "POST",
                        "params": {"objectIds": f"<chunk of <= {chunk} ids>", "outFields": "*", "outSR": "4326",
                                   "returnGeometry": "true", "f": "geojson"}},
        "bbox_filter": list(bbox) if spec.use_bbox else None,
    }
    rec["n_object_ids"] = len(ids)
    log.info("  %s: %d features to fetch in pages of %d", spec.name, len(ids), chunk)

    pages = [ids[i:i + chunk] for i in range(0, len(ids), chunk)]
    for i, page_ids in enumerate(pages):
        pf = rdir / f"page_{i:04d}.geojson"
        if pf.exists() and not force:
            continue
        d = f.request(f"{layer_url}/query", {"objectIds": ",".join(map(str, page_ids)), "outFields": "*",
                                             "outSR": "4326", "returnGeometry": "true", "f": "geojson"}, method="POST")
        got = len(d.get("features", []))
        if got != len(page_ids):
            log.warning("  page %d: asked %d ids, got %d features", i, len(page_ids), got)
        pf.write_text(json.dumps(d, ensure_ascii=False))
        time.sleep(0.3)
    # drop stale pages beyond current paging (e.g. after a bbox change)
    for extra in sorted(rdir.glob("page_*.geojson"))[len(pages):]:
        extra.unlink()
    rec["n_pages"] = len(pages)
    return rec


def assemble(spec: LayerSpec, rec: dict) -> dict:
    import geopandas as gpd
    import pandas as pd

    rdir = RAW_REAL / spec.group / "raw" / spec.name
    frames = []
    for pf in sorted(rdir.glob("page_*.geojson")):
        g = gpd.read_file(pf)
        if len(g):
            frames.append(g)
    out = RAW_REAL / spec.group / f"{spec.name}.gpkg"
    if not frames:
        rec["feature_count"] = 0
        return rec
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf.to_file(out, layer=spec.name, driver="GPKG")
    rec["output"] = str(out.relative_to(ROOT))
    rec["feature_count"] = int(len(gdf))
    rec["null_geometries"] = int(gdf.geometry.isna().sum() + gdf.geometry.is_empty.sum())
    rec["invalid_geometries"] = int((~gdf.geometry.is_valid).sum())
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"), default=KALIMANTAN_BBOX)
    ap.add_argument("--only", nargs="*", help="source groups or layer names to fetch")
    ap.add_argument("--force", action="store_true", help="re-download pages that already exist")
    args = ap.parse_args()
    bbox = tuple(args.bbox)

    f = Fetcher()
    prov: dict[str, dict] = {}
    for spec in LAYERS:
        if args.only and spec.group not in args.only and spec.name not in args.only:
            continue
        log.info("[%s] %s  (%s/%d)", spec.group, spec.name, spec.service, spec.layer_id)
        pfile = RAW_REAL / spec.group / "provenance.json"
        if spec.group not in prov:
            prov[spec.group] = json.loads(pfile.read_text()) if pfile.exists() else {"layers": {}, "attempts": []}
        try:
            rec = fetch_layer(f, spec, bbox, args.force)
            rec = assemble(spec, rec)
            rec["status"] = "ok"
        except ServiceFailed as e:
            log.error("  FAILED: %s", e)
            rec = {"layer": spec.name, "service_url": spec.service, "layer_id": spec.layer_id, "publisher": spec.publisher,
                   "status": "failed", "http_status": e.status, "error": str(e), "notes": spec.notes,
                   "retrieval_date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        prov[spec.group]["attempts"].append({k: rec.get(k) for k in ("layer", "status", "http_status", "error", "retrieval_date_utc")})
        if rec["status"] == "ok" or prov[spec.group]["layers"].get(spec.name, {}).get("status") != "ok":
            prov[spec.group]["layers"][spec.name] = rec
        prov[spec.group]["bbox_requested"] = list(bbox)
        prov[spec.group]["fetcher"] = "scripts/fetch_esdm_geology.py"
        pfile.parent.mkdir(parents=True, exist_ok=True)
        pfile.write_text(json.dumps(prov[spec.group], indent=2, ensure_ascii=False))
    for g, p in prov.items():
        for name, rec in p["layers"].items():
            log.info("%-22s %-34s %-7s %s", g, name, rec.get("status"), rec.get("feature_count", rec.get("error", ""))[:90]
                     if isinstance(rec.get("feature_count", rec.get("error", "")), str) else rec.get("feature_count"))


if __name__ == "__main__":
    main()
