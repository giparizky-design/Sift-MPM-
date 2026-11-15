#!/usr/bin/env python
"""Search (and optionally download) NASA EMIT L2B mineral granules over Kalimantan / Borneo.

Part 1 (no login): query NASA CMR's public search API for EMITL2BMIN (all versions),
EMITL2ARFL and EMITL2AMASK (metadata only), write footprints + attributes to
``data/raw_real/emit/granules.gpkg`` / ``granules.csv``, compute land coverage against
the real DEM land mask, and write ``provenance.json``.

Part 2 (``--download``): download EMITL2BMIN mineral-ID (``EMIT_L2B_MIN_*.nc``) and
uncertainty/band-depth-uncertainty (``EMIT_L2B_MINUNCERT_*.nc``) files, plus the matching
L2A mask file, for granules with cloud cover < ``--max-cloud``. Credentials are read only
from ``~/.netrc`` (machine urs.earthdata.nasa.gov) or the ``EARTHDATA_TOKEN`` environment
variable; they are never accepted as arguments and never written anywhere.

Resumable: the CMR search is cheap and re-run each time; downloads skip files whose size
matches CMR and whose SHA-512 (when CMR provides one) verifies; partial downloads are kept
as ``*.part`` and resumed with HTTP Range requests.

Usage (from the repo root):
    .venv/bin/python scripts/fetch_emit.py                      # search + coverage only
    .venv/bin/python scripts/fetch_emit.py --download           # also download (needs login)
    .venv/bin/python scripts/fetch_emit.py --bbox 114.5 -2.8 119.5 3.2 --download
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import netrc
import os
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "raw_real" / "emit"
LAND_MASK = REPO / "data" / "kalimantan_real" / "processed" / "dem_elevation.tif"
CMR = "https://cmr.earthdata.nasa.gov/search"
URS_HOST = "urs.earthdata.nasa.gov"
DEFAULT_BBOX = (108.5, -4.5, 119.5, 7.5)  # W, S, E, N
EAST_KAL_BBOX = (114.5, -2.8, 119.5, 3.2)
SIZE_CAP_BYTES = 40 * 1024**3
UA = {"User-Agent": "geospatials-fetch_emit/1.0 (research; requests)"}

PRIMARY = "EMITL2BMIN"
REFERENCE = ["EMITL2ARFL", "EMITL2AMASK"]


# ----------------------------------------------------------------------------- CMR search
def find_collections(short_name: str) -> list[dict]:
    r = requests.get(f"{CMR}/collections.umm_json",
                     params={"short_name": short_name, "provider": "LPCLOUD", "page_size": 50},
                     headers=UA, timeout=60)
    r.raise_for_status()
    out = []
    for it in r.json().get("items", []):
        u = it["umm"]
        out.append({
            "short_name": u.get("ShortName"),
            "version": u.get("Version"),
            "concept_id": it["meta"]["concept-id"],
            "revision_date": it["meta"].get("revision-date"),
            "entry_title": u.get("EntryTitle"),
            "doi": (u.get("DOI") or {}).get("DOI"),
            "collection_progress": u.get("CollectionProgress"),
            "use_constraints": u.get("UseConstraints"),
            "access_constraints": u.get("AccessConstraints"),
        })
    return sorted(out, key=lambda c: c["version"])


def search_granules(concept_id: str, bbox: tuple[float, float, float, float]) -> tuple[list[dict], dict]:
    params = {"collection_concept_id": concept_id,
              "bounding_box": ",".join(str(v) for v in bbox),
              "page_size": 2000, "sort_key": "start_date"}
    headers = dict(UA)
    items, hits = [], None
    while True:
        for attempt in range(5):
            try:
                r = requests.get(f"{CMR}/granules.umm_json", params=params, headers=headers, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        js = r.json()
        hits = js.get("hits", hits)
        batch = js.get("items", [])
        items.extend(batch)
        sa = r.headers.get("CMR-Search-After")
        if not batch or not sa or len(items) >= (hits or 0):
            break
        headers["CMR-Search-After"] = sa
    query = {"url": f"{CMR}/granules.umm_json", "params": params,
             "paging": "CMR-Search-After header", "hits": hits, "retrieved": len(items)}
    return items, query


def _attr(umm: dict, name: str):
    for a in umm.get("AdditionalAttributes", []) or []:
        if a.get("Name") == name:
            v = a.get("Values") or [None]
            return v[0]
    return None


def _footprint(umm: dict):
    from shapely.geometry import MultiPolygon, Polygon
    polys = []
    geom = umm.get("SpatialExtent", {}).get("HorizontalSpatialDomain", {}).get("Geometry", {})
    for gp in geom.get("GPolygons", []) or []:
        pts = [(p["Longitude"], p["Latitude"]) for p in gp["Boundary"]["Points"]]
        if len(pts) >= 4:
            polys.append(Polygon(pts))
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def parse_granule(item: dict, coll: dict) -> dict:
    umm, meta = item["umm"], item["meta"]
    t = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
    files = {a["Name"]: a for a in (umm.get("DataGranule", {}) or {}).get("ArchiveAndDistributionInformation", []) or []}
    https = [u["URL"] for u in umm.get("RelatedUrls", []) if u.get("URL", "").startswith("https://data.lpdaac")
             and u.get("Type") == "GET DATA"]
    browse = [u["URL"] for u in umm.get("RelatedUrls", []) if u.get("Type") == "GET RELATED VISUALIZATION"
              and u.get("URL", "").startswith("https://")]

    def pick(substr):
        return [u for u in https if substr in u.rsplit("/", 1)[-1]]

    def size_of(urls):
        return sum(int(files.get(u.rsplit("/", 1)[-1], {}).get("SizeInBytes", 0) or 0) for u in urls)

    begin = t.get("BeginningDateTime")
    rec = {
        "granule_id": umm.get("GranuleUR"),
        "concept_id": meta["concept-id"],
        "short_name": coll["short_name"],
        "version": coll["version"],
        "collection_concept_id": coll["concept_id"],
        "acq_start": begin,
        "acq_end": t.get("EndingDateTime"),
        "acq_key": (begin or "").replace("-", "").replace(":", "").replace("Z", "")[:15],
        "cloud_cover": umm.get("CloudCover"),
        "day_night": (umm.get("DataGranule", {}) or {}).get("DayNightFlag"),
        "orbit": _attr(umm, "ORBIT"),
        "orbit_segment": _attr(umm, "ORBIT_SEGMENT"),
        "scene": _attr(umm, "SCENE"),
        "solar_zenith": _attr(umm, "SOLAR_ZENITH"),
        "solar_azimuth": _attr(umm, "SOLAR_AZIMUTH"),
        "revision_date": meta.get("revision-date"),
        "browse_url": browse[0] if browse else None,
        "all_data_urls": ";".join(https),
        "file_checksums": json.dumps({k: (v.get("Checksum") or {}).get("Value") for k, v in files.items()}),
        "file_checksum_alg": next(((v.get("Checksum") or {}).get("Algorithm") for v in files.values()
                                   if v.get("Checksum")), None),
        "file_sizes": json.dumps({k: v.get("SizeInBytes") for k, v in files.items()}),
        "geometry": _footprint(umm),
    }
    sn = coll["short_name"]
    if sn == "EMITL2BMIN":
        d, u = pick("_L2B_MIN_"), pick("_L2B_MINUNCERT_")
        rec.update(min_url=";".join(d), minuncert_url=";".join(u),
                   data_bytes=size_of(d), uncert_bytes=size_of(u))
    elif sn == "EMITL2ARFL":
        d, u, m = pick("_L2A_RFL_"), pick("_L2A_RFLUNCERT_"), pick("_L2A_MASK_")
        rec.update(rfl_url=";".join(d), rfluncert_url=";".join(u), mask_url=";".join(m),
                   data_bytes=size_of(d), uncert_bytes=size_of(u), mask_bytes=size_of(m))
    elif sn == "EMITL2AMASK":
        m = [x for x in https if x.endswith(".nc")]
        rec.update(mask_url=";".join(m), mask_bytes=size_of(m))
    return rec


# ----------------------------------------------------------------------------- coverage
def coverage(gdf, land_path: Path, label: str) -> dict:
    import numpy as np
    import rasterio
    from rasterio import features
    from shapely.ops import unary_union

    # NOTE: dem_elevation.tif is finite everywhere (it carries bathymetry offshore), so "finite"
    # is not a land mask. Land = finite & elevation >= 0 m, the same rule
    # scripts/build_real_feature_cubes.py uses (`land = dem >= 0.0`). A second region, the
    # master cube's valid cells (rows of dataset.parquet; Indonesian Kalimantan only, dem >= -20 m,
    # <= 150 km from coast), is reported too because that is what EMIT layers would be joined to.
    with rasterio.open(land_path) as ds:
        dem = ds.read(1).astype("float32")
        land = np.isfinite(dem) & (dem >= 0.0)
        if ds.nodata is not None:
            land &= dem != ds.nodata
        tr, shape, crs, bounds = ds.transform, (ds.height, ds.width), ds.crs, ds.bounds
    cube_valid = None
    pq_path = land_path.parent / "dataset.parquet"
    if pq_path.exists():
        import pyarrow.parquet as pq
        rc = pq.read_table(pq_path, columns=["row", "col"]).to_pandas()
        cube_valid = np.zeros(shape, bool)
        cube_valid[rc["row"].to_numpy(), rc["col"].to_numpy()] = True
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    out = {"label": label, "n_granules": len(gdf), "land_mask": str(land_path.relative_to(REPO)),
           "land_rule": "finite & dem_elevation >= 0 m",
           "land_mask_crs": str(crs), "land_mask_bounds": list(bounds)}
    if geoms:
        u = unary_union([g.buffer(0) for g in geoms])
        cov = features.rasterize([(u, 1)], out_shape=shape, transform=tr, fill=0,
                                 all_touched=False, dtype="uint8").astype(bool)
    else:
        cov = np.zeros(shape, bool)
    rows, cols = np.indices(shape)
    lon = tr.c + (cols + 0.5) * tr.a
    lat = tr.f + (rows + 0.5) * tr.e
    ek = (lon >= EAST_KAL_BBOX[0]) & (lon <= EAST_KAL_BBOX[2]) & (lat >= EAST_KAL_BBOX[1]) & (lat <= EAST_KAL_BBOX[3])
    regions = [("kalimantan_land_mask", land), ("east_kalimantan", land & ek)]
    if cube_valid is not None:
        regions += [("master_cube_valid_cells", cube_valid), ("east_kalimantan_cube_valid", cube_valid & ek)]
    for name, region in regions:
        n_land = int(region.sum())
        n_cov = int((region & cov).sum())
        out[name] = {"land_cells": n_land, "covered_land_cells": n_cov,
                     "share": round(n_cov / n_land, 4) if n_land else None}
    return out


# ----------------------------------------------------------------------------- download
def credentials_present() -> str | None:
    if os.environ.get("EARTHDATA_TOKEN"):
        return "token"
    try:
        if netrc.netrc().authenticators(URS_HOST):
            return "netrc"
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    return None


def make_session(mode: str) -> requests.Session:
    s = requests.Session()
    s.trust_env = True  # netrc honoured for the URS redirect
    s.headers.update(UA)
    if mode == "token":
        s.headers["Authorization"] = f"Bearer {os.environ['EARTHDATA_TOKEN']}"
    return s


def _verify(path: Path, expected_size: int | None, sha512: str | None) -> bool:
    if not path.exists():
        return False
    if expected_size and path.stat().st_size != expected_size:
        return False
    if sha512:
        h = hashlib.sha512()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest().lower() == sha512.lower()
    return True


def download_file(s: requests.Session, url: str, dest: Path, size: int | None, sha512: str | None) -> str:
    if _verify(dest, size, sha512):
        return "exists"
    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with s.get(url, headers=headers, stream=True, timeout=300, allow_redirects=True) as r:
        if r.status_code in (401, 403):
            raise PermissionError(f"HTTP {r.status_code} for {url} (final URL host: "
                                  f"{requests.utils.urlparse(r.url).netloc})")
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype:
            raise PermissionError(f"Got an HTML page instead of data for {url} (login/EULA page?) "
                                  f"at {requests.utils.urlparse(r.url).netloc}")
        r.raise_for_status()
        mode = "ab" if (start and r.status_code == 206) else "wb"
        with open(part, mode) as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    part.rename(dest)
    if not _verify(dest, size, sha512):
        bad = dest.with_suffix(dest.suffix + ".bad")
        dest.rename(bad)
        return "checksum_failed"
    return "downloaded"


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), default=DEFAULT_BBOX)
    ap.add_argument("--download", action="store_true", help="download L2B MIN/MINUNCERT + L2A mask files")
    ap.add_argument("--max-cloud", type=float, default=50.0, help="download only cloud_cover < this (percent)")
    ap.add_argument("--download-version", default="prefer-latest",
                    help="'prefer-latest' (one version per acquisition, newest wins), or an explicit version e.g. 002")
    ap.add_argument("--no-mask", action="store_true", help="skip L2A mask download")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    import geopandas as gpd
    import pandas as pd

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    bbox = tuple(args.bbox)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    collections, queries, frames = {}, [], []
    for sn in [PRIMARY] + REFERENCE:
        colls = find_collections(sn)
        collections[sn] = colls
        for c in colls:
            items, q = search_granules(c["concept_id"], bbox)
            q.update(short_name=sn, version=c["version"])
            queries.append(q)
            recs = [parse_granule(it, c) for it in items]
            print(f"{sn} v{c['version']} ({c['concept_id']}): {len(recs)} granules (hits={q['hits']})")
            if recs:
                frames.append(gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:4326"))

    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    l2b = gdf[gdf.short_name == PRIMARY].copy()

    # Link L2B granules to an L2A mask with the same acquisition start time (prefer same version,
    # then newest mask version). EMITL2ARFL v001 bundles the MASK file; v002 moved it to EMITL2AMASK.
    masks = gdf[(gdf.get("mask_url").notna()) & (gdf.get("mask_url") != "")] if "mask_url" in gdf else gdf.iloc[0:0]
    mask_by_key: dict[tuple[str, str], tuple[str, str]] = {}
    for _, m in masks.sort_values("version").iterrows():
        mask_by_key[(m.acq_key, m.version)] = (m.mask_url, m.granule_id)
        mask_by_key[(m.acq_key, "*")] = (m.mask_url, m.granule_id)
    def _mask(row):
        return mask_by_key.get((row.acq_key, row.version)) or mask_by_key.get((row.acq_key, "*")) or (None, None)
    mm = l2b.apply(_mask, axis=1, result_type="expand")
    l2b["mask_url"], l2b["mask_granule_id"] = mm[0], mm[1]

    # One version per acquisition for download selection (newest wins). V002 re-segmented the
    # scenes, so start times shift by ~10-25 s between versions: an older-version granule is
    # treated as superseded when a newer-version granule starts within +-120 s and covers
    # >= 50 % of its footprint.
    l2b["_t"] = pd.to_datetime(l2b.acq_start, utc=True)
    l2b["superseded_by"] = None
    versions = sorted(l2b.version.unique())
    for i, v in enumerate(versions[:-1]):
        newer = l2b[l2b.version.isin(versions[i + 1:])]
        for idx, r in l2b[l2b.version == v].iterrows():
            cand = newer[(newer._t - r._t).abs() <= pd.Timedelta(seconds=120)]
            if r.geometry is None or cand.empty:
                continue
            area = r.geometry.area
            for _, c in cand.iterrows():
                if c.geometry is not None and area > 0 and r.geometry.intersection(c.geometry).area / area >= 0.5:
                    l2b.at[idx, "superseded_by"] = c.granule_id
                    break
    l2b["is_latest_version_for_acq"] = l2b.superseded_by.isna()
    l2b = l2b.drop(columns="_t")
    gdf["superseded_by"] = None
    gdf.loc[l2b.index, "superseded_by"] = l2b["superseded_by"]
    gdf.loc[l2b.index, ["mask_url", "mask_granule_id", "is_latest_version_for_acq"]] = \
        l2b[["mask_url", "mask_granule_id", "is_latest_version_for_acq"]]

    gpkg = out / "granules.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    for col in ("is_latest_version_for_acq",):
        gdf[col] = gdf[col].astype("object")
    gdf.to_file(gpkg, layer="emit_granules", driver="GPKG")
    gdf.drop(columns="geometry").assign(footprint_wkt=gdf.geometry.to_wkt()).to_csv(out / "granules.csv", index=False)

    # ---- statistics
    def cc_counts(df):
        cc = pd.to_numeric(df.cloud_cover, errors="coerce")
        return {"total": int(len(df)), "cloud_cover_reported": int(cc.notna().sum()),
                "cloud_lt_20": int((cc < 20).sum()), "cloud_lt_50": int((cc < 50).sum())}

    stats = {"by_collection_version": {f"{sn}.{v}": cc_counts(d) for (sn, v), d in gdf.groupby(["short_name", "version"])},
             "l2bmin_unique_acquisitions": int(l2b.is_latest_version_for_acq.sum()),
             "l2bmin_superseded_older_version": int((~l2b.is_latest_version_for_acq).sum()),
             "l2bmin_latest_version_per_acq": cc_counts(l2b[l2b.is_latest_version_for_acq]),
             "l2bmin_without_mask_match": int(l2b.mask_url.isna().sum())}
    lat_all = l2b[l2b.is_latest_version_for_acq]
    mbytes = dict(zip(gdf.granule_id, pd.to_numeric(gdf.get("mask_bytes"), errors="coerce").fillna(0)))
    est = {}
    for thr in (20, 50, 101):
        s_ = lat_all[pd.to_numeric(lat_all.cloud_cover, errors="coerce") < thr]
        est[f"cloud_lt_{thr}" if thr <= 100 else "all"] = {
            "granules": int(len(s_)),
            "l2b_min_plus_uncert_bytes": int((s_.data_bytes + s_.uncert_bytes).sum()),
            "l2a_mask_bytes": int(sum(mbytes.get(g, 0) for g in s_.mask_granule_id)),
        }
    stats["download_size_estimate_newest_version"] = est

    cov = {}
    if LAND_MASK.exists():
        cc = pd.to_numeric(l2b.cloud_cover, errors="coerce")
        lat = l2b[l2b.is_latest_version_for_acq]
        ccl = pd.to_numeric(lat.cloud_cover, errors="coerce")
        cov = {
            "l2bmin_all_versions": coverage(l2b, LAND_MASK, "EMITL2BMIN any version"),
            **{f"l2bmin_v{v}": coverage(d, LAND_MASK, f"EMITL2BMIN v{v}") for v, d in l2b.groupby("version")},
            "l2bmin_any_cloud_lt_50": coverage(l2b[cc < 50], LAND_MASK, "EMITL2BMIN any version, cloud<50 (as reported by that version)"),
            "l2bmin_latest_cloud_lt_50": coverage(lat[ccl < 50], LAND_MASK, "EMITL2BMIN newest version per acquisition, cloud<50"),
            "l2bmin_latest_cloud_lt_20": coverage(lat[ccl < 20], LAND_MASK, "EMITL2BMIN newest version per acquisition, cloud<20"),
        }
    else:
        print(f"WARNING: land mask missing at {LAND_MASK}; coverage not computed")

    # ---- download
    dl = {"requested": bool(args.download), "ran": False}
    if args.download:
        mode = credentials_present()
        if not mode:
            dl["status"] = ("skipped: no Earthdata credentials (no ~/.netrc entry for "
                            f"{URS_HOST} and EARTHDATA_TOKEN unset)")
            print(dl["status"])
        else:
            dl.update(ran=True, credential_source=mode)
            sel = l2b if args.download_version != "prefer-latest" else l2b[l2b.is_latest_version_for_acq]
            if args.download_version != "prefer-latest":
                sel = sel[sel.version == args.download_version]
            cc = pd.to_numeric(sel.cloud_cover, errors="coerce")
            if cc.notna().any():
                sel = sel[cc < args.max_cloud]
                dl["cloud_filter"] = f"cloud_cover < {args.max_cloud}"
            else:
                dl["cloud_filter"] = "cloud cover not reported; all granules selected"
            mask_meta = {r.granule_id: r for _, r in gdf[gdf.short_name != PRIMARY].iterrows()}
            jobs = []
            for _, r in sel.iterrows():
                sizes, sums = json.loads(r.file_sizes), json.loads(r.file_checksums)
                for u in filter(None, (r.min_url or "").split(";") + (r.minuncert_url or "").split(";")):
                    n = u.rsplit("/", 1)[-1]
                    jobs.append((u, out / "l2bmin" / n, sizes.get(n), sums.get(n)))
                if not args.no_mask and r.mask_url:
                    mr = mask_meta.get(r.mask_granule_id)
                    ms = json.loads(mr.file_sizes) if mr is not None else {}
                    mc = json.loads(mr.file_checksums) if mr is not None else {}
                    for u in str(r.mask_url).split(";"):
                        n = u.rsplit("/", 1)[-1]
                        jobs.append((u, out / "l2bmin" / "masks" / n, ms.get(n), mc.get(n)))
            total = sum(j[2] or 0 for j in jobs)
            dl.update(n_files=len(jobs), n_granules=int(len(sel)), planned_bytes=int(total))
            print(f"download plan: {len(sel)} granules, {len(jobs)} files, {total/1e9:.2f} GB")
            if total > SIZE_CAP_BYTES:
                dl["status"] = f"aborted: planned {total/1e9:.1f} GB exceeds 40 GiB cap"
                print(dl["status"])
            else:
                s = make_session(mode)
                results = {}
                for i, (u, dest, size, sha) in enumerate(jobs, 1):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        st = download_file(s, u, dest, size, sha)
                    except PermissionError as e:
                        dl["status"] = f"stopped: {e}"
                        print(dl["status"])
                        break
                    except requests.RequestException as e:
                        st = f"error: {e.__class__.__name__}"
                    results[dest.name] = st
                    print(f"[{i}/{len(jobs)}] {st}: {dest.name}")
                else:
                    dl["status"] = "complete"
                dl["results"] = {k: sum(1 for v in results.values() if v == k)
                                 for k in sorted(set(results.values()))}
                dl["bytes_on_disk"] = int(sum(p.stat().st_size for p in (out / "l2bmin").rglob("*.nc")))

    prov = {
        "retrieved_utc": now,
        "script": "scripts/fetch_emit.py",
        "bbox_wsen": list(bbox),
        "cmr_endpoint": CMR,
        "collections": collections,
        "granule_queries": queries,
        "counts": stats,
        "coverage": cov,
        "coverage_method": ("Union of CMR footprint polygons (GPolygons, EPSG:4326) rasterised (cell-centre rule) "
                            "onto land cells (dem_elevation finite and >= 0 m; the DEM is finite offshore because it "
                            "carries bathymetry) and onto the master cube's valid cells (dataset.parquet rows); East Kalimantan = cell centres within "
                            f"{list(EAST_KAL_BBOX)} (W,S,E,N). Footprints are scene boundaries, not valid-pixel "
                            "masks: cloud, water and nodata within a scene are NOT removed."),
        "licence": {
            "cmr_use_constraints": {f"{c['short_name']}.{c['version']}": c["use_constraints"]
                                    for v in collections.values() for c in v},
            "policy": ("NASA Earth Science Data and Information Policy: NASA promotes the full and open sharing "
                       "of Earth science data; no period of exclusive access. Free Earthdata Login required for "
                       "download from LP DAAC. Cite the product DOI (e.g. 10.5067/EMIT/EMITL2BMIN.002)."),
            "policy_url": "https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-information-policy",
        },
        "download": dl,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(json.dumps({"counts": stats, "coverage": {k: {r: v[r]["share"] for r in ("kalimantan_land_mask", "east_kalimantan")}
                                                     for k, v in cov.items()}, "download": dl}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
