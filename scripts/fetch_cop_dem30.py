#!/usr/bin/env python
"""Download Copernicus GLO-30 DEM (1x1 degree COG tiles) for one or more bboxes.

Source: AWS open-data bucket ``copernicus-dem-30m`` over plain HTTPS (no login).
The bucket's ``tileList.txt`` is used to select only existing (land) tiles, so
ocean cells are never probed. Downloads are idempotent and resumable: a tile
already on disk with the remote Content-Length and that opens with rasterio is
skipped. Files are written to ``<name>.tif.part`` then renamed.

Tiles are kept byte-for-byte as served. Nothing is filled, resampled or
substituted. Afterwards a GDAL VRT mosaic (``cop_dem30.vrt``) is built with
``rasterio``'s bundled GDAL (``rasterio.vrt`` / ``gdal.BuildVRT`` are not
needed: the VRT XML is written directly from tile metadata).

Usage:
    .venv/bin/python scripts/fetch_cop_dem30.py \
        --bbox 94 -11.5 142 6.5 --bbox 108 -5 120 7.5 \
        --out data/raw_real/cop_dem30
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
TILE_LIST_URL = f"{BASE}/tileList.txt"
UA = {"User-Agent": "geospatials-fetch-cop-dem30/1.0"}
VRT_NODATA = -32767.0


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def cells_for_bbox(w: float, s: float, e: float, n: float) -> set[tuple[int, int]]:
    """1-degree cells (SW-corner lat, lon) intersecting the bbox."""
    out = set()
    for lat in range(math.floor(s), math.ceil(n)):
        for lon in range(math.floor(w), math.ceil(e)):
            out.add((lat, lon))
    return out


def http_get(url: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Length"])
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return None
        raise


def opens_ok(path: Path) -> bool:
    import rasterio

    try:
        with rasterio.open(path) as ds:
            ds.read(1, window=((0, 1), (0, 1)))
        return True
    except Exception:
        return False


def fetch_tile(name: str, tiles_dir: Path, retries: int = 4) -> dict:
    url = f"{BASE}/{name}/{name}.tif"
    dest = tiles_dir / f"{name}.tif"
    last_err = None
    for attempt in range(retries):
        try:
            size = remote_size(url)
            if size is None:
                return {"name": name, "status": "missing_remote", "url": url}
            if dest.exists() and dest.stat().st_size == size and opens_ok(dest):
                return {"name": name, "status": "skipped", "bytes": size}
            part = dest.with_suffix(".tif.part")
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            if part.stat().st_size != size:
                raise IOError(f"size mismatch {part.stat().st_size} != {size}")
            part.replace(dest)
            if not opens_ok(dest):
                dest.unlink()
                raise IOError("downloaded file does not open with rasterio")
            return {"name": name, "status": "downloaded", "bytes": size}
        except Exception as exc:  # network hiccups -> retry
            last_err = repr(exc)
            time.sleep(2 * (attempt + 1))
    return {"name": name, "status": "failed", "error": last_err}


def build_vrt(tiles: list[Path], vrt_path: Path) -> dict:
    """Write a simple-source GDAL VRT mosaic referencing tiles by relative path.

    Copernicus GLO-30 tiles have 1 arc-second latitude spacing everywhere and
    longitude spacing that depends on latitude band (1" for |lat| < 50). All
    tiles in this project's area are within |lat| < 50, so they share one grid.
    """
    import rasterio

    infos = []
    for p in sorted(tiles):
        with rasterio.open(p) as ds:
            infos.append((p, ds.transform, ds.width, ds.height, ds.nodata,
                          ds.dtypes[0], ds.crs.to_wkt(), ds.bounds))
    resx = {round(t.a, 12) for _, t, *_ in infos}
    resy = {round(t.e, 12) for _, t, *_ in infos}
    if len(resx) != 1 or len(resy) != 1:
        raise RuntimeError(f"mixed resolutions, cannot build simple VRT: {resx} {resy}")
    rx, ry = infos[0][1].a, infos[0][1].e
    west = min(b.left for *_, b in infos)
    north = max(b.top for *_, b in infos)
    east = max(b.right for *_, b in infos)
    south = min(b.bottom for *_, b in infos)
    W = int(round((east - west) / rx))
    H = int(round((north - south) / -ry))
    dtype_map = {"float32": "Float32", "int16": "Int16", "float64": "Float64"}
    dt = dtype_map[infos[0][5]]
    nodata = infos[0][4]
    if nodata is None:
        # Tiles carry no nodata (ocean has no tile at all). Without a VRT-level
        # nodata GDAL would read uncovered areas as 0, i.e. silently "fill"
        # missing tiles. -32767 is outside GLO-30's value range.
        nodata = VRT_NODATA
    lines = [
        f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">',
        f"  <SRS dataAxisToSRSAxisMapping=\"2,1\">{infos[0][6]}</SRS>",
        f"  <GeoTransform>{west!r}, {rx!r}, 0.0, {north!r}, 0.0, {ry!r}</GeoTransform>",
        f'  <VRTRasterBand dataType="{dt}" band="1">',
    ]
    if nodata is not None:
        lines.append(f"    <NoDataValue>{nodata!r}</NoDataValue>")
    for p, t, w, h, *_ in infos:
        xoff = int(round((t.c - west) / rx))
        yoff = int(round((t.f - north) / ry))
        rel = p.relative_to(vrt_path.parent).as_posix()
        lines += [
            "    <SimpleSource>",
            f'      <SourceFilename relativeToVRT="1">{rel}</SourceFilename>',
            "      <SourceBand>1</SourceBand>",
            f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}" />',
            f'      <DstRect xOff="{xoff}" yOff="{yoff}" xSize="{w}" ySize="{h}" />',
            "    </SimpleSource>",
        ]
    lines += ["  </VRTRasterBand>", "</VRTDataset>", ""]
    vrt_path.write_text("\n".join(lines))
    return {"width": W, "height": H, "bounds": [west, south, east, north],
            "res_deg": [rx, ry], "dtype": infos[0][5], "nodata": nodata}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, action="append", required=True,
                    metavar=("W", "S", "E", "N"), help="lon/lat bbox; repeat for union")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-gb", type=float, default=40.0,
                    help="abort before downloading if the selected tiles exceed this")
    ap.add_argument("--no-vrt", action="store_true")
    args = ap.parse_args(argv)
    if not 8 <= args.workers <= 16:
        ap.error("--workers must be between 8 and 16")

    out = args.out
    tiles_dir = out / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    cells: set[tuple[int, int]] = set()
    for b in args.bbox:
        cells |= cells_for_bbox(*b)
    available = set(http_get(TILE_LIST_URL).decode().split())
    wanted = {tile_name(lat, lon): (lat, lon) for lat, lon in cells}
    selected = sorted(n for n in wanted if n in available)
    not_listed = sorted(wanted.keys() - available)
    print(f"cells in bbox union: {len(cells)}; tiles listed: {len(selected)}; "
          f"cells without tile (ocean/no data): {len(not_listed)}", flush=True)

    t0 = time.time()
    results = []
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_tile, n, tiles_dir): n for n in selected}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            total_bytes += r.get("bytes", 0)
            if total_bytes > args.max_gb * 1e9:
                print(f"ABORT: exceeded {args.max_gb} GB", flush=True)
                pool.shutdown(cancel_futures=True)
                break
            if i % 25 == 0 or r["status"] in ("failed", "missing_remote"):
                print(f"[{i}/{len(selected)}] {r['name']} {r['status']} "
                      f"total={total_bytes/1e9:.2f} GB", flush=True)

    status = {}
    for r in results:
        status[r["status"]] = status.get(r["status"], 0) + 1
    print("status counts:", status, f"elapsed {time.time()-t0:.0f}s", flush=True)
    summary = {
        "bboxes": args.bbox, "cells_in_union": len(cells),
        "tiles_listed": len(selected), "status_counts": status,
        "failed": [r for r in results if r["status"] in ("failed", "missing_remote")],
        "cells_not_in_tile_list": len(not_listed),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "fetch_log.json").write_text(json.dumps(summary, indent=2))

    if not args.no_vrt and not summary["failed"]:
        present = sorted(tiles_dir.glob("*.tif"))
        info = build_vrt(present, out / "cop_dem30.vrt")
        print("VRT:", json.dumps(info), flush=True)
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
