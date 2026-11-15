"""Tests for archipelago-wide K=4 label construction (src/labels/archipelago_labels.py)."""

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

from src.labels.archipelago_labels import (
    build_archipelago_k4_targets,
    clean_mrds_indonesia,
    points_to_cells,
)


def _georima():
    rows = [
        # lon, lat, resource_group, commodity_label, geological_role, island_region
        (121.0, -2.0, "COPPER_ORE", "Tembaga", "ml", "Sulawesi & Sangihe"),
        (121.1, -2.1, "GOLD_ORE", "Emas Primer", "ml", "Sulawesi & Sangihe"),
        (121.2, -2.2, "GOLD_ORE", "Emas Alluvial", "ml", "Sulawesi & Sangihe"),   # placer: dropped
        (121.3, -2.3, "NICKEL_COBALT", "Nikel", "ml", "Sulawesi & Sangihe"),
        (121.4, -2.4, "IRON_ORE", "Besi Primer", "ml", "Sulawesi & Sangihe"),     # skarn outside Borneo
        (121.5, -2.5, "IRON_ORE", "Besi Laterit", "ml", "Sulawesi & Sangihe"),    # laterite outside Borneo
        (121.6, -2.6, "MARBLE", "Marmer", "carbonate_skarn", "Sulawesi & Sangihe"),
        (115.0, -1.0, "IRON_ORE", "Besi Primer", "ml", "Kalimantan (Borneo)"),    # Borneo: ESDM source instead
        (121.0, -2.0, "BASE_METALS", "Seng", "ml", "Sulawesi & Sangihe"),         # duplicate coordinate of Cu
    ]
    df = pd.DataFrame(rows, columns=["lon", "lat", "resource_group", "commodity_label", "geological_role", "island_region"])
    df["is_centroid_artifact"] = False
    return df


def _mrds():
    return pd.DataFrame({
        "longitude": [125.0, 126.0, 127.0, 115.5],
        "latitude": [1.0, 1.5, 2.0, 0.5],
        "commod1": ["Gold", "Nickel", "Copper", "Gold"],
        "commod2": ["", "", "", ""],
        "commod3": ["", "", "", ""],
        "coordinate_quality_flag": ["", "shared coordinate; verify site-level precision", "", ""],
    })


def test_targets_apply_hygiene_and_archipelago_substitutes():
    t = build_archipelago_k4_targets(_georima(), _mrds())
    k0, k1, k2, k3 = (t[k] for k in ("k0_porphyry_cu", "k1_skarn_fe_cu", "k2_epithermal_au", "k3_laterite_fe_ni"))
    # Cu and Zn share a coordinate -> one point; MRDS copper added
    assert len(k0[(k0.lon == 121.0) & (k0.lat == -2.0)]) == 1
    assert ((k0.lon == 127.0) & (k0.source == "mrds_indonesia")).any()
    # Placer gold dropped; primary gold and MRDS gold outside Borneo kept; MRDS gold inside Borneo bbox not used
    assert not ((k2.lon == 121.2)).any()
    assert ((k2.lon == 121.1)).any() and ((k2.lon == 125.0)).any()
    assert not ((k2.lon == 115.5)).any()
    # Primary Fe: GeoRIMA Besi Primer outside Borneo only; marble (carbonate_skarn, 121.6) is a host, not a deposit (v4.15)
    assert set(k1.lon) == {121.4}
    # Laterite: nickel + Besi Laterit; flagged MRDS nickel dropped
    assert set(k3.lon) == {121.3, 121.5}


def test_clean_mrds_drops_flagged_and_missing_coordinates():
    m = _mrds()
    m.loc[0, "longitude"] = None
    out = clean_mrds_indonesia(m)
    assert set(out["longitude"]) == {127.0, 115.5}


def test_points_to_cells_uses_affine_transform():
    tr = from_origin(95.0, 6.0, 0.5, 0.5)  # 0.5 deg cells
    r, c, ok = points_to_cells(np.array([95.1, 96.9, 200.0]), np.array([5.9, 4.1, 0.0]), tr, H=10, W=10)
    assert ok.tolist() == [True, True, False]
    assert (r[0], c[0]) == (0, 0)
    assert (r[1], c[1]) == (3, 3)


def test_runner_region_filters():
    from scripts.run_deposit_cv import in_regions, parse_regions

    assert parse_regions("") is None
    allowed = parse_regions("sulawesi, papua")
    reg = np.array(["sulawesi", "kalimantan", "papua"])
    assert in_regions(reg, allowed, 3).tolist() == [True, False, True]
    assert in_regions(None, None, 3).all()
    with pytest.raises(SystemExit):
        in_regions(None, allowed, 3)
