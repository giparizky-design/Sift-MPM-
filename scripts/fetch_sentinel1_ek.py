#!/usr/bin/env python
"""Dry-season Sentinel-1 RTC (gamma0) VV/VH median composite for East Kalimantan.

Source: Microsoft Planetary Computer STAC collection ``sentinel-1-rtc`` (Sentinel-1 GRD,
radiometrically terrain corrected to gamma-naught by Catalyst/PCI, 10 m UTM COGs, linear power).

Method
  * scenes: June 1 - September 30 of each year in --years (default 2021, 2022, 2023),
    one orbit direction (--orbit, default descending; chosen for coverage, see FETCH_REPORT.md);
  * output grid: EPSG:4326 at 0.0001 deg (~11 m), 1 deg x 1 deg tiles aligned to whole degrees and
    clipped to --bbox; each scene is warped on the fly (GDAL WarpedVRT, bilinear, in linear power)
    and read windowed from the remote COG -- no scene is downloaded whole;
  * per-acquisition: slices of one pass (same date + relative orbit) are merged by nanmean
    (they only overlap at slice edges), so one pass counts as one observation;
  * per-pixel nanmedian over acquisitions of linear gamma0, then dB = 10*log10(median);
  * invalid input (nodata -32768, NaN, <= 0) is ignored; pixels with zero valid acquisitions stay
    NaN. No gap filling, no interpolation across missing scenes;
  * 0.1 deg blocks with no land cell (DEM elevation > 0 on the 250 m project grid, 1-cell buffer)
    are not read (left NaN) to save transfer; this is recorded in provenance.json.

Outputs per tile <out>/tiles/S1_<orbit>_<lat>_<lon>_{VV_db,VH_db,count}.tif and <out>/provenance.json.
Idempotent and resumable: finished tiles that open correctly are skipped; inside a tile, finished
0.1 deg blocks are cached under <out>/_work/ and reused after an interruption.

Usage (from the repo root):
    .venv/bin/python scripts/fetch_sentinel1_ek.py --bbox 114.5 -2.8 119.5 3.2 --out data/raw_real/sentinel1
    .venv/bin/python scripts/fetch_sentinel1_ek.py --tiles S01E116 --max-minutes 9   # chunked run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import planetary_computer
import pystac
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import box, shape

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
RES = 0.0001
BLOCK = 1000  # pixels (0.1 deg)
LAND_DEM = Path("data/east_kalimantan_real/processed/dem_elevation.tif")
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="6",
    GDAL_HTTP_RETRY_DELAY="5",
    GDAL_HTTP_TIMEOUT="120",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE=str(64 << 20),
    GDAL_CACHEMAX=1024,
)


def tile_name(lat0: int, lon0: int) -> str:
    """Name by the tile's south-west corner, e.g. S01E116 = lat -1..0, lon 116..117."""
    ns = "N" if lat0 >= 0 else "S"
    return f"{ns}{abs(lat0):02d}E{lon0:03d}"


def tiles_for_bbox(bbox):
    w, s, e, n = bbox
    out = []
    for lat0 in range(math.floor(s), math.ceil(n)):
        for lon0 in range(math.floor(w), math.ceil(e)):
            tb = (max(lon0, w), max(lat0, s), min(lon0 + 1, e), min(lat0 + 1, n))
            if tb[2] > tb[0] and tb[3] > tb[1]:
                out.append((tile_name(lat0, lon0), tb))
    return out


def _ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as src:
            src.read(1, window=((src.height - 1, src.height), (src.width - 1, src.width)))
        return True
    except Exception:
        return False


def search_items(bbox, years, orbit, cache: Path):
    if cache.exists():
        items = [pystac.Item.from_dict(d) for d in json.loads(cache.read_text())]
    else:
        cat = pystac_client.Client.open(STAC_URL)
        items = []
        for y in years:
            items += list(cat.search(collections=[COLLECTION], bbox=bbox,
                                     datetime=f"{y}-06-01/{y}-09-30").items())
        cache.write_text(json.dumps([i.to_dict() for i in items]))
    return [i for i in items if i.properties.get("sat:orbit_state") == orbit]


def land_lookup():
    with rasterio.open(LAND_DEM) as s:
        dem = s.read(1)
        land = np.isfinite(dem) & (dem > 0)
        if s.nodata is not None:
            land &= dem != s.nodata
        return land, s.transform


def block_has_land(land, ltr, bounds) -> bool:
    w, s, e, n = bounds
    c0 = int(math.floor((w - ltr.c) / ltr.a)) - 1
    c1 = int(math.ceil((e - ltr.c) / ltr.a)) + 1
    r0 = int(math.floor((ltr.f - n) / -ltr.e)) - 1
    r1 = int(math.ceil((ltr.f - s) / -ltr.e)) + 1
    sub = land[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)]
    return bool(sub.any())


def read_scene(item, pol, bounds, shp):
    """Warp one scene's polarisation onto the block grid; returns float32 linear power, NaN invalid."""
    w, s, e, n = bounds
    h, wd = shp
    tr = from_origin(w, n, RES, RES)
    href = planetary_computer.sign(item).assets[pol].href
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs="EPSG:4326", transform=tr, width=wd, height=h,
                           resampling=Resampling.bilinear, src_nodata=src.nodata,
                           nodata=np.nan, dtype="float32") as vrt:
                a = vrt.read(1)
    a = a.astype(np.float32)
    a[~np.isfinite(a) | (a <= 0)] = np.nan
    return a


def process_block(groups, bounds, shp, workers):
    """Return (vv_med, vh_med, count, acquisitions_used) for one block."""
    gb = box(*bounds)
    jobs = []
    for key, its in groups.items():
        its = [i for i in its if shape(i.geometry).intersects(gb)]
        if its:
            jobs.append((key, its))
    if not jobs:
        nan = np.full(shp, np.nan, np.float32)
        return nan, nan.copy(), np.zeros(shp, np.uint8), []

    def one_acq(job):
        key, its = job
        out = {}
        for pol in ("vv", "vh"):
            arrs = []
            for it in its:
                for attempt in range(4):
                    try:
                        arrs.append(read_scene(it, pol, bounds, shp))
                        break
                    except Exception as exc:  # transient network/sign errors
                        if attempt == 3:
                            raise RuntimeError(f"{it.id} {pol}: {exc}") from exc
                        time.sleep(10 * (attempt + 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[pol] = np.nanmean(np.stack(arrs), axis=0) if len(arrs) > 1 else arrs[0]
        return key, out

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one_acq, jobs))
    vv = np.stack([r[1]["vv"] for r in res])
    vh = np.stack([r[1]["vh"] for r in res])
    both = np.isfinite(vv) & np.isfinite(vh)
    vv[~both] = np.nan
    vh[~both] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vv_m = np.nanmedian(vv, axis=0).astype(np.float32)
        vh_m = np.nanmedian(vh, axis=0).astype(np.float32)
    cnt = both.sum(axis=0).astype(np.uint8)
    used = [r[0] for r in res]
    return vv_m, vh_m, cnt, used


def to_db(lin):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(lin > 0, 10.0 * np.log10(lin), np.nan).astype(np.float32)


def process_tile(name, tb, groups, out: Path, land, ltr, workers, deadline):
    tiles = out / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    paths = {k: tiles / f"S1_{name}_{k}.tif" for k in ("VV_db", "VH_db", "count")}
    if all(_ok(p) for p in paths.values()):
        return "exists", None
    w, s, e, n = tb
    W = int(round((e - w) / RES))
    H = int(round((n - s) / RES))
    work = out / "_work" / name
    work.mkdir(parents=True, exist_ok=True)
    meta = {"blocks_total": 0, "blocks_skipped_no_land": 0, "blocks_no_scene": 0, "acquisitions": set()}
    for r0 in range(0, H, BLOCK):
        for c0 in range(0, W, BLOCK):
            meta["blocks_total"] += 1
            h, wd = min(BLOCK, H - r0), min(BLOCK, W - c0)
            bb = (w + c0 * RES, n - (r0 + h) * RES, w + (c0 + wd) * RES, n - r0 * RES)
            f = work / f"b_{r0:05d}_{c0:05d}.npz"
            if f.exists():
                try:
                    z = np.load(f, allow_pickle=False)
                    meta["acquisitions"].update(json.loads(str(z["used"])))
                    meta["blocks_skipped_no_land"] += int(z["skipped"])
                    continue
                except Exception:
                    f.unlink()
            if time.time() > deadline:
                return "partial", None
            if not block_has_land(land, ltr, bb):
                np.savez(f, skipped=1, used="[]")
                meta["blocks_skipped_no_land"] += 1
                continue
            t0 = time.time()
            vv, vh, cnt, used = process_block(groups, bb, (h, wd), workers)
            np.savez_compressed(f, vv=vv, vh=vh, cnt=cnt, skipped=0, used=json.dumps(used))
            meta["acquisitions"].update(used)
            print(f"  {name} block r{r0} c{c0}: {len(used)} acq, valid={np.isfinite(vv).mean():.3f}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    # assemble
    vv_t = np.full((H, W), np.nan, np.float32)
    vh_t = np.full((H, W), np.nan, np.float32)
    cnt_t = np.zeros((H, W), np.uint8)
    for r0 in range(0, H, BLOCK):
        for c0 in range(0, W, BLOCK):
            z = np.load(work / f"b_{r0:05d}_{c0:05d}.npz", allow_pickle=False)
            if int(z["skipped"]):
                continue
            h, wd = z["vv"].shape
            vv_t[r0:r0 + h, c0:c0 + wd] = z["vv"]
            vh_t[r0:r0 + h, c0:c0 + wd] = z["vh"]
            cnt_t[r0:r0 + h, c0:c0 + wd] = z["cnt"]
    prof = dict(driver="GTiff", width=W, height=H, count=1, crs="EPSG:4326",
                transform=from_origin(w, n, RES, RES), tiled=True, blockxsize=512, blockysize=512,
                compress="deflate", BIGTIFF="IF_SAFER")
    for key, arr, dtype, nod, extra in (
        ("VV_db", to_db(vv_t), "float32", np.nan, dict(predictor=3)),
        ("VH_db", to_db(vh_t), "float32", np.nan, dict(predictor=3)),
        ("count", cnt_t, "uint8", None, dict(predictor=2)),
    ):
        tmp = paths[key].with_suffix(".tmp.tif")
        with rasterio.open(tmp, "w", dtype=dtype, nodata=nod, **prof, **extra) as dst:
            dst.write(arr, 1)
            dst.update_tags(product=f"Sentinel-1 RTC gamma0 dry-season (Jun-Sep) median, {key}",
                            units="dB" if key != "count" else "valid acquisitions")
            dst.build_overviews([2, 4, 8, 16], Resampling.average if key != "count" else Resampling.nearest)
        tmp.rename(paths[key])
    stats = {"valid_frac": float(np.isfinite(vv_t).mean()), "blocks_total": meta["blocks_total"],
             "blocks_skipped_no_land": meta["blocks_skipped_no_land"],
             "acquisitions": sorted(meta["acquisitions"]),
             "count_max": int(cnt_t.max()),
             "count_median_where_valid": float(np.median(cnt_t[cnt_t > 0])) if (cnt_t > 0).any() else 0.0}
    (work / "stats.json").write_text(json.dumps(stats))
    # block cache is no longer needed once the tile is written
    for p in work.glob("b_*.npz"):
        p.unlink()
    return "done", stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, default=[114.5, -2.8, 119.5, 3.2],
                    metavar=("W", "S", "E", "N"))
    ap.add_argument("--out", type=Path, default=Path("data/raw_real/sentinel1"))
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023])
    ap.add_argument("--orbit", choices=["ascending", "descending"], default="descending")
    ap.add_argument("--tiles", nargs="*", help="only these tile names (e.g. S01E116)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop cleanly after N minutes (0 = no limit)")
    ap.add_argument("--max-gb", type=float, default=25.0)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else float("inf")

    items = search_items(args.bbox, args.years, args.orbit, out / "_stac_items.json")
    groups = defaultdict(list)
    for it in items:
        groups[f"{it.datetime.date().isoformat()}_R{it.properties['sat:relative_orbit']:03d}"].append(it)
    print(f"{len(items)} {args.orbit} scenes, {len(groups)} acquisitions (date+relative orbit)", flush=True)
    land, ltr = land_lookup()

    status = {}
    stats_path = out / "_tile_status.json"
    if stats_path.exists():
        status = json.loads(stats_path.read_text())
    for name, tb in tiles_for_bbox(args.bbox):
        if args.tiles and name not in args.tiles:
            continue
        t0 = time.time()
        st, stats = process_tile(name, tb, groups, out, land, ltr, args.workers, deadline)
        if st == "done":
            status[name] = {"bounds": tb, **stats}
        stats_path.write_text(json.dumps(status, indent=1))
        print(f"tile {name}: {st} ({time.time() - t0:.0f}s)", flush=True)
        size_gb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
        if size_gb > args.max_gb:
            print(f"STOP: output {size_gb:.1f} GB exceeds --max-gb", file=sys.stderr)
            return 2
        if st == "partial":
            print("deadline reached; rerun to resume", flush=True)
            return 3

    all_tiles = [n for n, _ in tiles_for_bbox(args.bbox)]
    if all(n in status for n in all_tiles):
        used = sorted({a for s in status.values() for a in s["acquisitions"]})
        ids = sorted(i.id for a in used for i in groups[a])
        prov = {
            "product": "Sentinel-1 C-band RTC gamma0, dry-season (Jun-Sep) per-pixel median, VV and VH",
            "stac_api": STAC_URL, "collection": COLLECTION, "licence": "CC-BY-4.0 (STAC collection licence); "
            "contains modified Copernicus Sentinel data [2021-2023]",
            "access": "anonymous; hrefs signed with planetary_computer.sign, no subscription key",
            "bbox": args.bbox, "years": args.years, "months": "06-01..09-30", "orbit_direction": args.orbit,
            "grid": f"EPSG:4326, {RES} deg (~11 m), 1x1 deg tiles aligned to whole degrees, clipped to bbox",
            "processing": [
                "per scene: GDAL WarpedVRT from native 10 m UTM to output grid, bilinear, linear gamma0, src nodata -32768",
                "values <= 0 or non-finite -> invalid",
                "slices of one pass (same date + relative orbit) merged by nanmean -> one acquisition",
                "pixel valid for an acquisition only if both VV and VH valid",
                "per-pixel nanmedian over acquisitions in linear power, then dB = 10*log10(median)",
                "no gap filling; pixels with 0 valid acquisitions are NaN; count band = acquisitions used",
                "0.1 deg blocks without any land cell (DEM>0 on 250 m grid, 1-cell buffer) not read -> NaN",
            ],
            "n_scenes_searched": len(items), "acquisitions_used": used, "scene_ids": ids,
            "tiles": status,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "script": "scripts/fetch_sentinel1_ek.py",
        }
        (out / "provenance.json").write_text(json.dumps(prov, indent=1))
        print("all tiles done; provenance written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
