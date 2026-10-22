"""Unit tests for GIS-ready output generation in MPM.

Geoscientific context:
    Model outputs must interface seamlessly with industry standard GIS tools
    (QGIS, ArcGIS, Leapfrog). Prospectivity scores, epistemic uncertainties,
    feature SHAP attributions, and discrete target polygons must preserve
    spatial georeferencing and strict metadata tagging.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine

from src.outputs.prospectivity_map import (
    _scatter_to_2d,
    generate_target_polygons,
)


def test_scatter_to_2d():
    """Verify that 1D valid pixel values are scattered correctly to 2D grid."""
    H, W = 10, 10
    row_idx = np.array([1, 2, 5])
    col_idx = np.array([3, 4, 8])
    values = np.array([0.75, 0.90, 0.33], dtype=np.float32)

    grid = _scatter_to_2d(values, row_idx, col_idx, H, W, fill_value=np.nan)

    assert grid.shape == (H, W)
    assert grid[1, 3] == pytest.approx(0.75)
    assert grid[2, 4] == pytest.approx(0.90)
    assert grid[5, 8] == pytest.approx(0.33)
    assert np.isnan(grid[0, 0])
    assert np.isnan(grid[9, 9])


def test_generate_target_polygons():
    """Verify thresholding, uncertainty masking, and vectorisation of target polygons."""
    H, W = 100, 100
    # 50m pixel resolution: 1 pixel = 2,500 m² = 0.25 ha
    # 100 pixels = 25 ha
    transform = Affine(50.0, 0.0, 500000.0, 0.0, -50.0, 6800000.0)
    profile = {
        "crs": "EPSG:7851",
        "transform": transform,
        "width": W,
        "height": H,
        "nodata": np.nan,
    }

    prospectivity = np.zeros((H, W), dtype=np.float32)
    uncertainty = np.full((H, W), 0.10, dtype=np.float32)

    # Blob 1: 15x15 pixels = 225 pixels = 56.25 ha (> 25 ha threshold)
    # High prospectivity (0.85), low uncertainty (0.10)
    prospectivity[20:35, 20:35] = 0.85

    # Blob 2: High prospectivity (0.90) but high uncertainty (0.25 > 0.20 max)
    prospectivity[60:75, 60:75] = 0.90
    uncertainty[60:75, 60:75] = 0.25  # should be filtered out

    # Blob 3: High prospectivity (0.80), low uncertainty (0.10), but tiny: 4x4 pixels = 16 pixels = 4 ha (< 25 ha)
    prospectivity[80:84, 80:84] = 0.80  # should be filtered out by min_area_ha

    targets_gdf = generate_target_polygons(
        prospectivity=prospectivity,
        uncertainty=uncertainty,
        profile=profile,
        min_score=0.70,
        max_uncertainty=0.20,
        min_area_ha=25.0,
    )

    assert isinstance(targets_gdf, gpd.GeoDataFrame)
    # Only Blob 1 satisfies all criteria: score >= 0.70, uncert <= 0.20, area >= 25 ha
    assert len(targets_gdf) == 1
    target = targets_gdf.iloc[0]
    assert target["mean_score"] >= 0.80
    assert target["area_ha"] > 25.0
    assert target["mean_uncertainty"] <= 0.20
