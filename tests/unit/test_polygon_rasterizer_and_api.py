"""Unit tests for polygonal rasterization, custom model hyperparams, and backend API helpers."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from rasterio.transform import from_bounds

from src.features.polygon_rasterizer import (
    apply_polygon_mask_to_array,
    build_polygonal_feature_cube,
    compute_polygon_area_km2,
    compute_polygon_mask_2d,
    get_predefined_polygons,
    normalize_polygon_geojson,
    validate_polygon_and_resolution,
)
from src.models.coal import TwoStageK3CoalEnsemble
from src.models.metals import TwoStageK4Ensemble
from src.schema.canonical import verify_no_raw_pixel_leakage


def _sample_triangle_geojson() -> dict:
    """Return a non-rectangular triangle polygon in East Kalimantan (~30 km across)."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Test Triangle Polygon"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [116.80, -0.60],
                            [117.10, -0.60],
                            [116.95, -0.35],
                            [116.80, -0.60],
                        ]
                    ],
                },
            }
        ],
    }


def test_predefined_polygons_catalog():
    polys = get_predefined_polygons()
    for key in [
        "EAST_KALIMANTAN",
        "KUTAI_BASIN",
        "MAHAKAM_SUTURE",
        "SANGATTA_TREND",
        "KELIAN_CORRIDOR",
        "BERAU_FOLD_BELT",
        "ADANG_FAULT",
        "MALINAU_CORRIDOR",
    ]:
        assert key in polys
        fc = normalize_polygon_geojson(polys[key], name=key)
        assert fc["type"] == "FeatureCollection"
        area = compute_polygon_area_km2(fc)
        assert area > 100.0


def test_validate_polygon_and_resolution():
    tri = _sample_triangle_geojson()
    res = validate_polygon_and_resolution(tri, 100)
    assert res["valid"] is True
    assert res["resolution_m"] == 100
    assert res["area_km2"] > 100.0
    assert res["est_valid_cells"] > 1000
    # Triangle occupies ~50% of its bounding box
    assert 0.35 <= res["fill_ratio"] <= 0.65


def test_compute_polygon_mask_2d_and_nan_clipping():
    tri = _sample_triangle_geojson()
    height, width = 60, 60
    transform = from_bounds(116.75, -0.65, 117.15, -0.30, width, height)
    mask = compute_polygon_mask_2d(
        tri, transform=transform, height=height, width=width, crs="EPSG:4326"
    )
    assert mask.shape == (height, width)
    assert mask.dtype == bool
    # Non-rectangular triangle: both interior (True) and exterior (False) must exist
    n_inside = int(mask.sum())
    n_outside = int((~mask).sum())
    assert n_inside > 200
    assert n_outside > 1000

    arr_3d = np.ones((3, height, width), dtype=np.float32)
    clipped = apply_polygon_mask_to_array(arr_3d, mask, fill_value=np.nan)
    assert np.isnan(clipped[:, ~mask]).all()
    assert np.all(clipped[:, mask] == 1.0)


def test_build_polygonal_feature_cube_strict_interior(tmp_path: Path):
    tri = _sample_triangle_geojson()
    out_dir = tmp_path / "poly_scan_500m"
    meta = build_polygonal_feature_cube(
        polygon_geojson=tri,
        resolution_m=500,
        out_dir=out_dir,
        region_name="TEST_TRIANGLE",
    )
    assert (out_dir / "dataset.parquet").exists()
    assert (out_dir / "scanned_boundary.geojson").exists()
    assert (out_dir / "feature_cube_meta.json").exists()

    df = pd.read_parquet(out_dir / "dataset.parquet")
    assert len(df) == meta["n_valid_cells"]
    # Verify strictly fewer valid cells than bounding box H*W because exterior is excluded
    total_bbox_cells = meta["height"] * meta["width"]
    assert len(df) < int(total_bbox_cells * 0.65)
    assert len(df) > int(total_bbox_cells * 0.20)

    # Verify zero raw sensor pixel leakage
    from src.schema.canonical import EXTENDED_NON_FEATURE_COLS

    verify_no_raw_pixel_leakage(meta["feature_names"])
    feature_cols = [
        c
        for c in df.columns
        if c not in EXTENDED_NON_FEATURE_COLS
        and not c.startswith("y_")
        and not c.startswith("label_")
    ]
    verify_no_raw_pixel_leakage(feature_cols)


def test_custom_hyperparameters_in_ensembles():
    m_k4 = TwoStageK4Ensemble(
        n_estimators=20,
        n_bag=2,
        learning_rate=0.08,
        max_depth=4,
    )
    assert m_k4.learning_rate == pytest.approx(0.08)
    assert m_k4.max_depth == 4

    m_k3 = TwoStageK3CoalEnsemble(
        n_estimators=15,
        n_bag=2,
        learning_rate=0.03,
        max_depth=5,
    )
    assert m_k3.learning_rate == pytest.approx(0.03)
    assert m_k3.max_depth == 5


def test_dashboard_server_polygonal_rgba_and_catalogs(tmp_path: Path):
    import rasterio
    from scripts.dashboard_server import (
        _list_available_models,
        _list_available_polygons,
        _raster_to_rgba_base64,
    )

    models = _list_available_models()
    assert isinstance(models, list)
    assert len(models) >= 1
    assert "model_id" in models[0]

    polygons = _list_available_polygons()
    assert isinstance(polygons, list)
    poly_ids = {p["id"] for p in polygons}
    assert "EAST_KALIMANTAN" in poly_ids
    assert "KUTAI_BASIN" in poly_ids

    # Write a synthetic raster and verify _raster_to_rgba_base64 masks outside triangle to alpha=0
    tri = _sample_triangle_geojson()
    tif_path = tmp_path / "prospectivity_p.tif"
    h, w = 80, 80
    transform = from_bounds(116.75, -0.65, 117.15, -0.30, w, h)
    data = np.full((h, w), 0.75, dtype=np.float32)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    b64_png, bounds = _raster_to_rgba_base64(tif_path, max_dim=80, boundary_geojson=tri)
    assert b64_png is not None
    assert bounds is not None

    b64_payload = b64_png.split(",", 1)[-1] if "," in b64_png else b64_png
    raw_bytes = base64.b64decode(b64_payload)
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
    rgba = np.array(img)
    alpha = rgba[:, :, 3]
    # Top-left corner (outside triangle) must be completely transparent (alpha == 0)
    assert alpha[0, 0] == 0
    assert alpha[0, -1] == 0
    # Interior of triangle must have opaque pixels (alpha > 0)
    assert int((alpha > 0).sum()) > 500
