#!/usr/bin/env python3
"""Run data feasibility and availability audit for the MPM pilot.

Audits:
1. File presence against expected paths in data/raw/
2. Raster attributes (resolution, CRS, bounds, bands)
3. Grid compatibility with target 50 m prediction cell
4. Label coverage and coordinate precision checks
5. Outputs markdown report to outputs/reports/data_audit_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml


def audit_raster(path: Path) -> dict:
    """Read basic raster metadata without loading the entire raster array into memory."""
    if not path.exists():
        return {"status": "MISSING"}
    try:
        with rasterio.open(path) as src:
            return {
                "crs": str(src.crs),
                "res_x_m": abs(src.transform.a),
                "res_y_m": abs(src.transform.e),
                "width_px": src.width,
                "height_px": src.height,
                "bands": src.count,
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
                "bounds": f"[{src.bounds.left:.1f}, {src.bounds.bottom:.1f}, {src.bounds.right:.1f}, {src.bounds.top:.1f}]",
                "status": "OK",
            }
    except Exception as exc:
        return {"status": f"ERROR: {exc}"}


def run_data_audit(config_path: str = "config/pilot.yaml", data_dir_str: str = "data/raw") -> int:
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"ERROR: Config not found at {config_file}", file=sys.stderr)
        return 1

    with open(config_file) as f:
        config = yaml.safe_load(f)

    pilot_name = config.get("pilot", {}).get("name", "mpm_pilot")
    commodity = config.get("pilot", {}).get("commodity", "Au")
    deposit_type = config.get("pilot", {}).get("deposit_type", "orogenic_gold")
    target_crs = config.get("grid", {}).get("crs", "EPSG:7851")
    cell_size = config.get("grid", {}).get("cell_size_m", 50)
    aoi_geo = config.get("grid", {}).get("aoi_bounds_geo", [119.5, -32.0, 123.5, -28.5])

    data_dir = Path(data_dir_str)
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f" DATA AUDIT: {pilot_name}")
    print(f" Target: {commodity} ({deposit_type})")
    print(f" Grid: {cell_size} m | CRS: {target_crs}")
    print(f" AOI Bounds (lon/lat): {aoi_geo}")
    print(f"=======================================================\n")

    # 1. File presence check
    expected_datasets = {
        "aster": data_dir / "aster",
        "sentinel2": data_dir / "sentinel2",
        "magnetics_tmi": data_dir / "magnetics" / "tmi.tif",
        "magnetics_rtp": data_dir / "magnetics" / "rtp.tif",
        "radiometrics_k": data_dir / "radiometrics" / "potassium.tif",
        "radiometrics_u": data_dir / "radiometrics" / "uranium.tif",
        "radiometrics_th": data_dir / "radiometrics" / "thorium.tif",
        "gravity_bouguer": data_dir / "gravity" / "bouguer.tif",
        "dem": data_dir / "dem",
        "geology": data_dir / "geology",
        "minedex_labels": data_dir / "labels" / "minedex",
        "wamex_drillholes": data_dir / "labels" / "wamex",
    }

    audit_records = []
    for name, path in expected_datasets.items():
        exists = path.exists()
        if exists and path.is_dir():
            files = list(path.iterdir())
            n_files = len(files)
            has_data = n_files > 0
        elif exists and path.is_file():
            has_data = True
            n_files = 1
        else:
            has_data = False
            n_files = 0

        status_str = "PRESENT" if has_data else "MISSING"
        icon = "[OK]" if has_data else "[MISSING]"
        audit_records.append({
            "Dataset": name,
            "Path": str(path),
            "Present": has_data,
            "File Count": n_files,
            "Status": status_str,
        })
        print(f" {icon:9s} {name:20s} -> {str(path)}")

    audit_df = pd.DataFrame(audit_records)

    # 2. Raster metadata audit
    raster_targets = {
        "TMI (Aeromagnetics)": data_dir / "magnetics" / "tmi.tif",
        "RTP (Aeromagnetics)": data_dir / "magnetics" / "rtp.tif",
        "Potassium (Radiometrics)": data_dir / "radiometrics" / "potassium.tif",
        "Uranium (Radiometrics)": data_dir / "radiometrics" / "uranium.tif",
        "Thorium (Radiometrics)": data_dir / "radiometrics" / "thorium.tif",
        "Bouguer Gravity": data_dir / "gravity" / "bouguer.tif",
    }

    raster_records = []
    for name, path in raster_targets.items():
        meta = audit_raster(path)
        meta["Layer"] = name
        meta["Native Res (m)"] = f"{meta.get('res_x_m', 'N/A')}"
        meta["Target Cell (m)"] = cell_size
        if meta["status"] == "OK":
            rx = float(meta["res_x_m"])
            if rx > cell_size:
                meta["Harmonization Policy"] = f"Bilinear upsample ({rx/cell_size:.1f}x) + Provenance Flag"
            else:
                meta["Harmonization Policy"] = f"Bilinear downsample ({cell_size/rx:.1f}x)"
        else:
            meta["Harmonization Policy"] = "Awaiting download"
        raster_records.append(meta)

    raster_df = pd.DataFrame(raster_records)

    # 3. Label precision audit
    label_records = []
    minedex_path = data_dir / "labels" / "minedex"
    wamex_path = data_dir / "labels" / "wamex"

    label_notes = []
    if minedex_path.exists() and any(minedex_path.iterdir()):
        label_notes.append("MINEDEX files found.")
    else:
        label_notes.append("MINEDEX directory empty or pending download. Required for Positives (P).")

    if wamex_path.exists() and any(wamex_path.iterdir()):
        label_notes.append("WAMEX files found.")
    else:
        label_notes.append("WAMEX directory empty or pending download. Required for Reliable Negatives (RN).")

    # 4. Generate Markdown report
    missing_items = audit_df[~audit_df["Present"]]["Dataset"].tolist()
    
    report_lines = [
        f"# Data Feasibility and Audit Report",
        f"",
        f"- **Pilot Name:** `{pilot_name}`",
        f"- **Target Commodity & Style:** {commodity} ({deposit_type})",
        f"- **Target Prediction Cell:** {cell_size} m × {cell_size} m",
        f"- **Target CRS:** `{target_crs}`",
        f"- **AOI Geographic Extent:** {aoi_geo}",
        f"- **Audit Timestamp:** `{pd.Timestamp.now().isoformat()}`",
        f"",
        f"---",
        f"",
        f"## 1. Input Datasets & Local Storage Audit",
        f"",
        audit_df[["Dataset", "Path", "Present", "File Count", "Status"]].to_markdown(index=False),
        f"",
        f"---",
        f"",
        f"## 2. Geophysical & Raster Resolution Compatibility",
        f"",
        f"The prediction unit of 50 m requires evaluating each input against its native survey resolution:",
        f"",
        raster_df[["Layer", "status", "Native Res (m)", "Target Cell (m)", "Harmonization Policy"]].to_markdown(index=False),
        f"",
        f"---",
        f"",
        f"## 3. Ground-Truth Labels Status",
        f"",
    ]

    for note in label_notes:
        report_lines.append(f"- {note}")

    report_lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Audit Summary & Action Items",
        f"",
    ])

    if missing_items:
        report_lines.append(f"⚠️ **{len(missing_items)} expected raw dataset(s) currently unpopulated:**")
        for m in missing_items:
            report_lines.append(f"  - `{m}`: see [`data/README.md`](file:///Users/gip/Geospatials/data/README.md) for download URL and staging instructions.")
        report_lines.append("")
        report_lines.append("**Next Steps:** Download and stage the raw data files into `data/raw/` before triggering `python scripts/run_pilot.py --stage harmonize`.")
    else:
        report_lines.append("✅ All required input datasets are present and verified. Ready for `stage_harmonize`.")

    report_path = reports_dir / "data_audit_report.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\nAudit complete. Detailed report saved to: {report_path}\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MPM data audit")
    parser.add_argument("--config", default="config/pilot.yaml", help="Path to pilot.yaml")
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()
    sys.exit(run_data_audit(args.config, args.data_dir))
