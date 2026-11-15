#!/usr/bin/env python
"""Download JAXA ALOS-2 PALSAR-2 annual mosaic tiles (native 25 m) from Planetary Computer.

Collection: ``alos-palsar-mosaic`` (Microsoft Planetary Computer STAC), year 2021 by default.

For every 1 deg x 1 deg mosaic tile intersecting ``--bbox`` this script:
  1. downloads the original COG assets HH, HV (uint16 DN), mask and date byte-for-byte
     into ``<out>/dn/`` (no clipping, no resampling, no value change);
  2. writes HH and HV in dB into ``<out>/db/`` on the identical 25 m grid using JAXA's
     calibration  dB = 10*log10(DN^2) - 83.0  (DN == 0 -> NaN; nothing else is masked,
     the JAXA mask tile is kept alongside so users can drop layover/shadow pixels);
  3. writes ``<out>/provenance.json``.

Idempotent and resumable: an output that already exists and opens/reads correctly is skipped;
partial downloads are written to ``*.part`` and only renamed when complete.

Usage (from the repo root):
    .venv/bin/python scripts/fetch_palsar2_25m.py --bbox 114.5 -2.8 119.5 3.2 --out data/raw_real/palsar2_25m
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import requests

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "alos-palsar-mosaic"
CF_DB = -83.0
DN_ASSETS = ("HH", "HV", "mask", "date")
DB_POLS = ("HH", "HV")


def _ok(path: Path, full_read: bool = False) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as src:
            if full_read:
                src.read(1)
            else:
                src.read(1, window=((src.height - 1, src.height), (src.width - 1, src.width)))
        return True
    except Exception:
        return False


def _download(href: str, dst: Path, retries: int = 5) -> int:
    part = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(href, stream=True, timeout=120) as r:
                if r.status_code in (401, 403):
                    raise PermissionError(f"HTTP {r.status_code} for {dst.name}: {r.text[:300]}")
                r.raise_for_status()
                expected = int(r.headers.get("Content-Length", -1))
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            got = part.stat().st_size
            if expected >= 0 and got != expected:
                raise IOError(f"size mismatch {got} != {expected}")
            part.rename(dst)
            return got
        except PermissionError:
            raise
        except Exception as exc:  # network hiccup: retry
            print(f"  retry {attempt}/{retries} {dst.name}: {exc}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"failed to download {dst.name}")


def _write_db(dn_path: Path, db_path: Path) -> dict:
    with rasterio.open(dn_path) as src:
        dn = src.read(1).astype(np.float64)
        prof = src.profile.copy()
        dn_nodata = src.nodata  # JAXA tiles declare nodata = 1 (DN 1 occurs only where mask == 0)
    mask_path = dn_path.with_name(dn_path.name.rsplit("_", 1)[0] + "_mask.tif")
    with rasterio.open(mask_path) as msrc:
        mk = msrc.read(1)
    invalid = (dn <= 0) | (mk == 0)
    if dn_nodata is not None:
        invalid |= dn == dn_nodata
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(dn**2) + CF_DB
    db = np.where(invalid, np.nan, db).astype(np.float32)
    prof.update(dtype="float32", nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    prof.pop("photometric", None)
    tmp = db_path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(db, 1)
        dst.update_tags(1, units="dB", formula="dB = 10*log10(DN^2) - 83.0", source=dn_path.name)
    tmp.rename(db_path)
    return {"valid_px": int(np.isfinite(db).sum()), "total_px": int(db.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, default=[114.5, -2.8, 119.5, 3.2],
                    metavar=("W", "S", "E", "N"), help="EPSG:4326 bounding box")
    ap.add_argument("--out", type=Path, default=Path("data/raw_real/palsar2_25m"))
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--max-gb", type=float, default=25.0, help="abort if output dir grows beyond this")
    args = ap.parse_args()

    out: Path = args.out
    (out / "dn").mkdir(parents=True, exist_ok=True)
    (out / "db").mkdir(parents=True, exist_ok=True)

    cat = pystac_client.Client.open(STAC_URL)
    items = list(cat.search(collections=[COLLECTION], bbox=args.bbox,
                            datetime=f"{args.year}-01-01/{args.year}-12-31").items())
    items = [it for it in items if all(a in it.assets for a in DN_ASSETS)]
    items.sort(key=lambda it: it.id)
    print(f"{len(items)} tiles for bbox={args.bbox} year={args.year}", flush=True)
    if not items:
        print("no items found", file=sys.stderr)
        return 1

    records = []
    for n, item in enumerate(items, 1):
        signed = planetary_computer.sign(item)
        rec = {"id": item.id, "bbox": item.bbox, "properties": {k: item.properties.get(k) for k in (
            "cf", "gsd", "start_datetime", "end_datetime", "sat:orbit_state", "palsar:beam_number",
            "sar:product_type", "sar:polarizations", "proj:code", "proj:shape", "proj:transform")},
            "files": {}}
        for a in DN_ASSETS:
            dst = out / "dn" / f"{item.id}_{a}.tif"
            if not _ok(dst):
                nbytes = _download(signed.assets[a].href, dst)
                if not _ok(dst, full_read=True):
                    raise RuntimeError(f"downloaded file does not open: {dst}")
                print(f"[{n}/{len(items)}] {dst.name} {nbytes/1e6:.1f} MB", flush=True)
            rec["files"][a] = str(dst.relative_to(out))
        for pol in DB_POLS:
            src = out / "dn" / f"{item.id}_{pol}.tif"
            dst = out / "db" / f"{item.id}_{pol}_db.tif"
            if not _ok(dst):
                stats = _write_db(src, dst)
                print(f"[{n}/{len(items)}] {dst.name} valid={stats['valid_px']/stats['total_px']:.3f}", flush=True)
            rec["files"][f"{pol}_db"] = str(dst.relative_to(out))
        records.append(rec)
        size_gb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
        if size_gb > args.max_gb:
            print(f"STOP: output {size_gb:.1f} GB exceeds --max-gb {args.max_gb}", file=sys.stderr)
            return 2

    coll = cat.get_collection(COLLECTION)
    prov = {
        "product": f"JAXA ALOS-2 PALSAR-2 annual 25 m mosaic, year {args.year}",
        "stac_api": STAC_URL,
        "collection": COLLECTION,
        "collection_license_field": coll.license,
        "licence_note": ("JAXA terms of use for the Global PALSAR-2/PALSAR/JERS-1 mosaic: free use with "
                         "attribution, e.g. '(c) JAXA' (see https://earth.jaxa.jp/policy/en.html and the "
                         "Planetary Computer dataset page for alos-palsar-mosaic). STAC licence field is 'other'."),
        "bbox_requested": args.bbox,
        "year": args.year,
        "n_tiles": len(records),
        "access": "anonymous; hrefs signed with planetary_computer.sign (no subscription key)",
        "native_grid": "EPSG:4326, 1/4500 deg (0.8 arcsec, ~25 m), 4500x4500 per 1x1 deg tile",
        "dn_files": "original COG assets copied byte-for-byte (HH, HV uint16 DN; mask; date). Tiles are NOT clipped to the bbox.",
        "db_files": "float32, same grid as DN tile, NaN where DN == declared nodata (1), DN <= 0 or mask == 0; layover (100) / shadow (150) NOT masked, see mask tile",
        "calibration_formula": "dB = 10*log10(DN^2) - 83.0  (gamma-naught backscatter, dB)",
        "calibration_source": ("JAXA EORC 'Global 25 m Resolution PALSAR-2/PALSAR Mosaic and Forest/Non-Forest Map (FNF) "
                               "Dataset Description' (calibration factor CF = -83.0 dB for PALSAR-2 mosaics); "
                               "also carried per item as STAC property 'cf' = '83.0 dB'."),
        "mask_classes": {"0": "no data", "50": "ocean/water", "100": "layover", "150": "shadowing", "255": "land"},
        "date_layer": "days since launch of ALOS-2 (2014-05-24) per JAXA dataset description",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/fetch_palsar2_25m.py",
        "tiles": records,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
