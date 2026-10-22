#!/usr/bin/env python3
"""Generate a realistic benchmark dataset for the Yilgarn Orogenic Gold pilot.

Geoscientific context:
    This script populates data/raw/ with realistic geoscientific layers over a 10 km × 10 km
    test window in the Eastern Goldfields (GDA2020 / MGA Zone 51, EPSG:7851).
    It simulates:
    1. ASTER multispectral reflectance (VNIR, SWIR, TIR) with hydrothermal alteration
       (Al-OH / sericite enrichment) over gold deposits.
    2. Sentinel-2 dry-season reflectance with low background NDVI (<0.2).
    3. Aeromagnetic TMI & RTP (80 m native resolution) exhibiting north-northwest
       striking magnetic shears and magnetite-destructive alteration halos.
    4. Radiometrics (100 m native): K %, eU ppm, eTh ppm showing potassic alteration.
    5. Bouguer Gravity (400 m native): density contrast at granite-greenstone margins.
    6. DEM (30 m native): topography with drainage lineaments.
    7. Geology: Greenstone belt corridor flanked by granitic domes.
    8. Labels: MINEDEX gold deposits (P) + WAMEX drillholes and assays (RN + U).
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter
from shapely.geometry import box, Point, Polygon


def generate_benchmark_data(dest_dir: str = "data/raw") -> None:
    raw_root = Path(dest_dir)
    raw_root.mkdir(parents=True, exist_ok=True)

    # 10 km × 10 km area in Eastern Goldfields (MGA Zone 51, EPSG:7851)
    # Origin near Kalgoorlie corridor: easting 350,000 to 360,000; northing 6,590,000 to 6,600,000
    xmin, xmax = 350000.0, 360000.0
    ymin, ymax = 6590000.0, 6600000.0
    crs = "EPSG:7851"

    rng = np.random.default_rng(2026)

    print("Generating benchmark geoscientific datasets in data/raw/ ...")

    # Define 3 known mineralized centres (gold deposits)
    deposit_coords = [
        (352900.0, 6597100.0, "Golden Mile North", 850.0),   # 850 koz Au, CV block (1,1)
        (357200.0, 6593200.0, "Hannan South", 320.0),       # 320 koz Au, inside blind district
        (352200.0, 6593100.0, "Paddy's Lode", 150.0),       # 150 koz Au, CV block (0,2)
    ]

    # --- 1. Topography / DEM (30 m resolution) ---
    dem_dir = raw_root / "dem"
    dem_dir.mkdir(exist_ok=True)
    res_dem = 30.0
    w_dem = int((xmax - xmin) / res_dem)
    h_dem = int((ymax - ymin) / res_dem)
    t_dem = from_bounds(xmin, ymin, xmax, ymax, w_dem, h_dem)

    # Smooth rolling topography with regional NNW ridge
    xx, yy = np.meshgrid(np.linspace(0, 10, w_dem), np.linspace(0, 10, h_dem))
    dem_arr = (
        320.0
        + 15.0 * np.sin(xx * 0.8)
        + 25.0 * np.cos(yy * 0.5)
        + gaussian_filter(rng.normal(0, 5, (h_dem, w_dem)), sigma=3)
    ).astype(np.float32)

    with rasterio.open(
        dem_dir / "dem.tif", "w",
        driver="GTiff", height=h_dem, width=w_dem, count=1,
        dtype="float32", crs=crs, transform=t_dem, nodata=-9999.0
    ) as dst:
        dst.write(dem_arr, 1)

    # --- 2. Aeromagnetics (80 m resolution: TMI and RTP) ---
    mag_dir = raw_root / "magnetics"
    mag_dir.mkdir(exist_ok=True)
    res_mag = 80.0
    w_mag = int((xmax - xmin) / res_mag)
    h_mag = int((ymax - ymin) / res_mag)
    t_mag = from_bounds(xmin, ymin, xmax, ymax, w_mag, h_mag)

    # Background regional magnetic gradient + NNW magnetic dykes/shears
    xm, ym = np.meshgrid(np.linspace(0, 10, w_mag), np.linspace(0, 10, h_mag))
    regional_mag = 50000.0 + 80.0 * xm - 50.0 * ym
    # NNW shear zone anomaly (striking ~340 deg)
    shear_anomaly = 250.0 * np.exp(-((xm - ym * 0.3 - 3.5) ** 2) / 0.8)
    # Demagnetization halos at deposits
    rtp_arr = regional_mag + shear_anomaly + rng.normal(0, 20, (h_mag, w_mag))
    for dep_x, dep_y, _, _ in deposit_coords:
        col = int((dep_x - xmin) / (xmax - xmin) * w_mag)
        row = int((ymax - dep_y) / (ymax - ymin) * h_mag)
        rr, cc = np.ogrid[:h_mag, :w_mag]
        dist2 = (rr - row) ** 2 + (cc - col) ** 2
        # Magnetite destruction (alteration halo) creates local TMI low
        rtp_arr -= 180.0 * np.exp(-dist2 / 12.0)

    rtp_arr = gaussian_filter(rtp_arr, sigma=1.0).astype(np.float32)
    tmi_arr = (rtp_arr - 40.0 * np.sin(ym)).astype(np.float32)

    for fname, arr in [("tmi.tif", tmi_arr), ("rtp.tif", rtp_arr)]:
        with rasterio.open(
            mag_dir / fname, "w",
            driver="GTiff", height=h_mag, width=w_mag, count=1,
            dtype="float32", crs=crs, transform=t_mag, nodata=-9999.0
        ) as dst:
            dst.write(arr, 1)

    # --- 3. Radiometrics (100 m resolution: K, U, Th) ---
    rad_dir = raw_root / "radiometrics"
    rad_dir.mkdir(exist_ok=True)
    res_rad = 100.0
    w_rad = int((xmax - xmin) / res_rad)
    h_rad = int((ymax - ymin) / res_rad)
    t_rad = from_bounds(xmin, ymin, xmax, ymax, w_rad, h_rad)

    # Potassium: enriched along alteration zones
    xr, yr = np.meshgrid(np.linspace(0, 10, w_rad), np.linspace(0, 10, h_rad))
    k_arr = (1.2 + 0.3 * np.sin(xr * 0.5) + rng.normal(0, 0.1, (h_rad, w_rad))).astype(np.float32)
    # Add potassic alteration anomaly around deposits
    for dep_x, dep_y, _, _ in deposit_coords:
        col = int((dep_x - xmin) / (xmax - xmin) * w_rad)
        row = int((ymax - dep_y) / (ymax - ymin) * h_rad)
        rr, cc = np.ogrid[:h_rad, :w_rad]
        dist2 = (rr - row) ** 2 + (cc - col) ** 2
        k_arr += 0.8 * np.exp(-dist2 / 6.0).astype(np.float32)

    u_arr = (2.5 + rng.normal(0, 0.2, (h_rad, w_rad))).astype(np.float32)
    th_arr = (8.0 + rng.normal(0, 0.5, (h_rad, w_rad))).astype(np.float32)

    for fname, arr in [("potassium.tif", k_arr), ("uranium.tif", u_arr), ("thorium.tif", th_arr)]:
        with rasterio.open(
            rad_dir / fname, "w",
            driver="GTiff", height=h_rad, width=w_rad, count=1,
            dtype="float32", crs=crs, transform=t_rad, nodata=-9999.0
        ) as dst:
            dst.write(arr, 1)

    # --- 4. Bouguer Gravity (400 m resolution) ---
    grav_dir = raw_root / "gravity"
    grav_dir.mkdir(exist_ok=True)
    res_grav = 400.0
    w_grav = int((xmax - xmin) / res_grav)
    h_grav = int((ymax - ymin) / res_grav)
    t_grav = from_bounds(xmin, ymin, xmax, ymax, w_grav, h_grav)

    xg, yg = np.meshgrid(np.linspace(0, 10, w_grav), np.linspace(0, 10, h_grav))
    grav_arr = (-350.0 + 15.0 * np.cos(xg * 0.4) - 10.0 * yg + rng.normal(0, 1, (h_grav, w_grav))).astype(np.float32)

    with rasterio.open(
        grav_dir / "bouguer.tif", "w",
        driver="GTiff", height=h_grav, width=w_grav, count=1,
        dtype="float32", crs=crs, transform=t_grav, nodata=-9999.0
    ) as dst:
        dst.write(grav_arr, 1)

    # --- 5. ASTER Multispectral (30 m SWIR, 15 m VNIR) ---
    aster_dir = raw_root / "aster"
    aster_dir.mkdir(exist_ok=True)
    res_aster = 30.0
    w_ast = int((xmax - xmin) / res_aster)
    h_ast = int((ymax - ymin) / res_aster)
    t_ast = from_bounds(xmin, ymin, xmax, ymax, w_ast, h_ast)

    # Standard reflectance background for semi-arid regolith (0.05 to 0.45)
    b1 = rng.uniform(0.10, 0.15, (h_ast, w_ast)).astype(np.float32)
    b2 = rng.uniform(0.12, 0.18, (h_ast, w_ast)).astype(np.float32)
    b3 = rng.uniform(0.15, 0.22, (h_ast, w_ast)).astype(np.float32)
    b4 = rng.uniform(0.20, 0.28, (h_ast, w_ast)).astype(np.float32)
    b5 = rng.uniform(0.22, 0.30, (h_ast, w_ast)).astype(np.float32)
    b6 = rng.uniform(0.20, 0.26, (h_ast, w_ast)).astype(np.float32)
    b7 = rng.uniform(0.21, 0.29, (h_ast, w_ast)).astype(np.float32)
    b8 = rng.uniform(0.18, 0.25, (h_ast, w_ast)).astype(np.float32)
    b9 = rng.uniform(0.16, 0.22, (h_ast, w_ast)).astype(np.float32)
    b11 = rng.uniform(8.5, 9.5, (h_ast, w_ast)).astype(np.float32)
    b13 = rng.uniform(8.8, 9.8, (h_ast, w_ast)).astype(np.float32)

    # Add phyllic alteration signature: high Al-OH = (B5+B7)/B6
    # By depressing B6 (absorption at 2.20 um) and boosting B5, B7 over deposits
    for dep_x, dep_y, _, _ in deposit_coords:
        col = int((dep_x - xmin) / (xmax - xmin) * w_ast)
        row = int((ymax - dep_y) / (ymax - ymin) * h_ast)
        rr, cc = np.ogrid[:h_ast, :w_ast]
        dist2 = (rr - row) ** 2 + (cc - col) ** 2
        anomaly = np.exp(-dist2 / 10.0).astype(np.float32)
        b6 -= 0.08 * anomaly
        b5 += 0.04 * anomaly
        b7 += 0.04 * anomaly

    aster_bands = {
        "B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5,
        "B6": b6, "B7": b7, "B8": b8, "B9": b9, "B11": b11, "B13": b13,
    }
    for bname, barr in aster_bands.items():
        with rasterio.open(
            aster_dir / f"aster_{bname.lower()}.tif", "w",
            driver="GTiff", height=h_ast, width=w_ast, count=1,
            dtype="float32", crs=crs, transform=t_ast, nodata=-9999.0
        ) as dst:
            dst.write(barr, 1)

    # --- 6. Sentinel-2 (20 m resolution) ---
    s2_dir = raw_root / "sentinel2"
    s2_dir.mkdir(exist_ok=True)
    res_s2 = 20.0
    w_s2 = int((xmax - xmin) / res_s2)
    h_s2 = int((ymax - ymin) / res_s2)
    t_s2 = from_bounds(xmin, ymin, xmax, ymax, w_s2, h_s2)

    # Dry season: low NDVI (~0.12)
    s2_b4 = rng.uniform(0.14, 0.18, (h_s2, w_s2)).astype(np.float32)
    s2_b8 = s2_b4 * rng.uniform(1.10, 1.25, (h_s2, w_s2)).astype(np.float32)
    s2_b11 = rng.uniform(0.25, 0.32, (h_s2, w_s2)).astype(np.float32)
    s2_b12 = rng.uniform(0.20, 0.28, (h_s2, w_s2)).astype(np.float32)

    for bname, barr in [("B4", s2_b4), ("B8", s2_b8), ("B11", s2_b11), ("B12", s2_b12)]:
        with rasterio.open(
            s2_dir / f"s2_{bname.lower()}.tif", "w",
            driver="GTiff", height=h_s2, width=w_s2, count=1,
            dtype="float32", crs=crs, transform=t_s2, nodata=-9999.0
        ) as dst:
            dst.write(barr, 1)

    # --- 7. Geology Vectors ---
    geo_dir = raw_root / "geology"
    geo_dir.mkdir(exist_ok=True)
    # Greenstone belt corridor (prospective) and Granitoid batholith (background)
    greenstone_poly = box(351000.0, 6590000.0, 358500.0, 6600000.0)
    granite_poly_west = box(350000.0, 6590000.0, 351000.0, 6600000.0)
    granite_poly_east = box(358500.0, 6590000.0, 360000.0, 6600000.0)

    geology_gdf = gpd.GeoDataFrame({
        "unit_id": [1, 2, 3],
        "rock_type": ["Greenstone_Basalt", "Granite_Batholith_West", "Granite_Batholith_East"],
        "is_prospective": [1, 0, 0],
        "geometry": [greenstone_poly, granite_poly_west, granite_poly_east]
    }, crs=crs)
    geology_gdf.to_file(geo_dir / "geology.gpkg", driver="GPKG")

    # --- 8. MINEDEX Labels (Positives P) ---
    minedex_dir = raw_root / "labels" / "minedex"
    minedex_dir.mkdir(parents=True, exist_ok=True)
    minedex_records = []
    for dep_x, dep_y, name, au_oz in deposit_coords:
        minedex_records.append({
            "deposit_name": name,
            "commodity": "Au",
            "deposit_type": "orogenic_gold",
            "au_contained_koz": au_oz,
            "geometry": Point(dep_x, dep_y),
            "classification": "P",
        })
    minedex_gdf = gpd.GeoDataFrame(minedex_records, crs=crs)
    minedex_gdf.to_file(minedex_dir / "deposits.gpkg", driver="GPKG")

    # --- 9. WAMEX Drillholes (Reliable Negatives RN + Unlabeled U) ---
    wamex_dir = raw_root / "labels" / "wamex"
    wamex_dir.mkdir(parents=True, exist_ok=True)

    drillhole_rows = []
    assay_rows = []
    hole_counter = 1

    # Place 12 confirmed barren drillholes (>50m depth, assay < 0.05 g/t Au, outside deposit halos)
    # Every hole is >1 km from any deposit (so the P exclusion buffer keeps it as RN).
    # Holes are placed in the same 2.5 km CV blocks as the deposits and inside the blind
    # district so every held-out fold / the blind test has BOTH verified classes.
    rn_positions = [
        (354500.0, 6595500.0), (354000.0, 6596000.0),   # block with Golden Mile North
        (351000.0, 6594500.0), (351500.0, 6594800.0),   # block with Paddy's Lode
        (358800.0, 6594600.0), (358600.0, 6592000.0),   # inside blind district
        (351500.0, 6591000.0), (351700.0, 6591200.0), (351900.0, 6591400.0),
        (353500.0, 6599000.0), (353700.0, 6599200.0), (353900.0, 6599400.0),
        (357800.0, 6598000.0), (358000.0, 6598200.0), (358200.0, 6598400.0),
        (354700.0, 6591500.0), (354900.0, 6591700.0), (355100.0, 6591900.0),
    ]

    for hx, hy in rn_positions:
        hid = f"WAMEX_DH_{hole_counter:04d}"
        drillhole_rows.append({
            "hole_id": hid,
            "easting": hx,
            "northing": hy,
            "zone": 51,
            "max_depth_m": 85.0,
            "collar_survey_accuracy_m": 5.0,
        })
        assay_rows.append({
            "hole_id": hid,
            "from_m": 0.0,
            "to_m": 85.0,
            "au_ppm": float(rng.uniform(0.005, 0.025)),  # strictly < 0.05 g/t
        })
        hole_counter += 1

    # Add positive verification drillholes at known deposit locations
    for dep_x, dep_y, _, _ in deposit_coords:
        hid = f"WAMEX_DH_{hole_counter:04d}"
        drillhole_rows.append({
            "hole_id": hid,
            "easting": dep_x,
            "northing": dep_y,
            "zone": 51,
            "max_depth_m": 140.0,
            "collar_survey_accuracy_m": 2.0,
        })
        assay_rows.append({
            "hole_id": hid,
            "from_m": 50.0,
            "to_m": 75.0,
            "au_ppm": float(rng.uniform(2.5, 8.2)),  # economic mineralization
        })
        hole_counter += 1

    pd.DataFrame(drillhole_rows).to_csv(wamex_dir / "collars.csv", index=False)
    pd.DataFrame(assay_rows).to_csv(wamex_dir / "assays.csv", index=False)

    print("✅ Benchmark geoscientific datasets successfully generated:")
    print(f"   Extent: [{xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f}] in {crs}")
    print(f"   Deposits: {len(deposit_coords)} confirmed orogenic gold deposits")
    print(f"   Drillholes: {len(drillhole_rows)} collars with depth & multi-element assays")


if __name__ == "__main__":
    generate_benchmark_data()
