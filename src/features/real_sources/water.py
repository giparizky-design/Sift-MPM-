"""Surface-water mask for the real-data cube (v4.15, audit v4.13 section 2.6).

The optical canopy gate keeps cells with NDVI < 0.20. Open water has negative NDVI and
therefore passed the gate: 79% of cells with a valid ``spec_clay_hydroxyl_gated`` value on
the East Kalimantan cube were below sea level. This mask removes water before gating.

A cell is water when ANY of the following holds (NaN inputs never trigger a rule):

* ``NDVI < 0``                 (open water absorbs NIR more than red)
* ``MNDWI > 0``                (Xu 2006: (B03 - B11) / (B03 + B11))
* inside an OSM water polygon  (``water``, ``riverbank``, ``reservoir``, ``dock``; wetland
                                classes are NOT included, swamp forest and mangrove carry canopy)
* ``DEM < 0``                  (below the 0 m coastline)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_mndwi(b03_green: np.ndarray, b11_swir1: np.ndarray, min_reflectance: float = 0.005) -> np.ndarray:
    """Modified NDWI, (B03 - B11) / (B03 + B11); NaN where either band is not physical."""
    g = np.asarray(b03_green, dtype=np.float32)
    s = np.asarray(b11_swir1, dtype=np.float32)
    ok = np.isfinite(g) & np.isfinite(s) & (g >= min_reflectance) & (s >= min_reflectance)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (g - s) / (g + s)
    return np.where(ok, out, np.nan).astype(np.float32)


def compute_water_mask(
    ndvi: np.ndarray,
    mndwi: Optional[np.ndarray] = None,
    dem: Optional[np.ndarray] = None,
    osm_water: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean water mask (True = water). All supplied rasters must share one shape."""
    ndvi = np.asarray(ndvi, dtype=np.float32)
    water = np.zeros(ndvi.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        water |= ndvi < 0.0
        if mndwi is not None:
            water |= np.asarray(mndwi, dtype=np.float32) > 0.0
        if dem is not None:
            water |= np.asarray(dem, dtype=np.float32) < 0.0
    if osm_water is not None:
        water |= np.asarray(osm_water).astype(bool)
    return water
