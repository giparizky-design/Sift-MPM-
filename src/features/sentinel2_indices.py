"""Sentinel-2 Spectral Feature Engineering for Mineral Prospectivity Mapping.

Sentinel-2 MultiSpectral Instrument (MSI) provides 13 bands at 10–60 m
resolution. Compared with ASTER SWIR (30 m, 6 bands), Sentinel-2 SWIR
coverage is limited to two broad bands (B11 at 1.610 µm and B12 at 2.190 µm),
making it impossible to resolve individual clay species. However, its 10 m
VNIR bands (B2, B3, B4, B8) deliver superior spatial detail for:
  - Vegetation masking (NDVI) at finer resolution than ASTER
  - Iron oxide mapping
  - Water body and river channel delineation (NDWI)

All indices use surface reflectance (Level-2A) or, where unavailable,
atmospherically corrected top-of-atmosphere data. Band naming follows the
Sentinel-2 convention (B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR1,
B12=SWIR2).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _safe_ratio(
    a: np.ndarray,
    b: np.ndarray,
    fill: float = np.nan,
) -> np.ndarray:
    """Element-wise ratio a/b; fills with `fill` where b is zero or NaN.

    Args:
        a: Numerator array.
        b: Denominator array.
        fill: Value to use where division is undefined. Defaults to NaN.

    Returns:
        Float64 ratio array of the same shape as inputs.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    bad = (b == 0) | ~np.isfinite(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(bad, fill, a / b)
    return result


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------

def ndvi(
    b4_red: np.ndarray,
    b8_nir: np.ndarray,
) -> np.ndarray:
    """Normalised Difference Vegetation Index (NDVI).

    Formula: (B8 − B4) / (B8 + B4)

    NDVI is the standard proxy for photosynthetically active vegetation.
    Within the MPM workflow it serves two roles:
      1. **Vegetation masking threshold**: pixels with NDVI ≥ 0.2 are flagged
         and excluded from ASTER SWIR-based alteration indices.
      2. **Indirect lithology signal**: very low NDVI (< 0.05) in otherwise
         humid climates can indicate barren, altered, or mineralised ground.

    Values range from −1 (water, shadows) to +1 (dense canopy). Bare rock
    and soil typically produce 0.0–0.15.

    Args:
        b4_red: Band 4 surface reflectance (Red, ~665 nm). Can be raw DN
                if consistently scaled with b8_nir.
        b8_nir: Band 8 surface reflectance (NIR, ~842 nm).

    Returns:
        NDVI array (float64) in the range [−1, 1], NaN where both bands
        sum to zero.
    """
    b4 = np.asarray(b4_red, dtype=np.float64)
    b8 = np.asarray(b8_nir, dtype=np.float64)
    return _safe_ratio(b8 - b4, b8 + b4)


def ndwi(
    b3_green: np.ndarray,
    b8_nir: np.ndarray,
) -> np.ndarray:
    """Normalised Difference Water Index (NDWI) for surface water masking.

    Formula: (B3 − B8) / (B3 + B8)

    Gao (1996) / McFeeters (1996) NDWI. Open water bodies show strongly
    positive values (> 0.2), while vegetation and dry soil are negative.
    Within the MPM workflow, NDWI ≥ 0.1 is used to mask lakes, rivers, and
    wetlands before alteration index computation — water reflectance in the
    SWIR mimics clay-like spectral shapes and produces artefacts.

    Args:
        b3_green: Band 3 surface reflectance (Green, ~560 nm).
        b8_nir: Band 8 surface reflectance (NIR, ~842 nm).

    Returns:
        NDWI array (float64) in the range [−1, 1], NaN where both bands
        sum to zero.

    References:
        McFeeters, S.K. (1996). The use of the Normalised Difference Water
        Index (NDWI) in the delineation of open water features.
        International Journal of Remote Sensing, 17(7), 1425-1432.
    """
    b3 = np.asarray(b3_green, dtype=np.float64)
    b8 = np.asarray(b8_nir, dtype=np.float64)
    return _safe_ratio(b3 - b8, b3 + b8)


def clay_index(
    b11_swir1: np.ndarray,
    b12_swir2: np.ndarray,
) -> np.ndarray:
    """Coarse clay proxy using Sentinel-2 SWIR band ratio B11/B12.

    Formula: B11 / B12

    .. warning::
        **Spectral resolution limitation**: Sentinel-2 SWIR1 (B11, ~1610 nm)
        and SWIR2 (B12, ~2190 nm) are broad bands (90–180 nm FWHM). They
        cannot resolve individual clay species (e.g. kaolinite vs. smectite
        vs. illite) because their diagnostic Al-OH and Mg-OH absorptions at
        2.1–2.4 µm are blended within the single B12 window. Use this ratio
        only as a regional screening layer. For clay species discrimination,
        use the ASTER SWIR indices (B5/B6/B7/B8/B9) which resolve the 2.2 µm
        and 2.35 µm features separately.

    High B11/B12 values indicate enhanced SWIR1 relative to SWIR2, which
    broadly correlates with clay-rich or hydroxyl-bearing assemblages. The
    index is most informative in arid to semi-arid settings where vegetation
    contamination is low.

    Args:
        b11_swir1: Band 11 surface reflectance (SWIR1, ~1610 nm).
        b12_swir2: Band 12 surface reflectance (SWIR2, ~2190 nm). Denominator.

    Returns:
        Clay proxy ratio array (float64), NaN where B12 is zero or NaN.
    """
    b11 = np.asarray(b11_swir1, dtype=np.float64)
    b12 = np.asarray(b12_swir2, dtype=np.float64)
    return _safe_ratio(b11, b12)


def iron_oxide_s2(
    b4_red: np.ndarray,
    b2_blue: np.ndarray,
) -> np.ndarray:
    """Sentinel-2 iron oxide (Fe³⁺) proxy: B4/B2 ratio.

    Formula: B4 / B2

    Analogous to the ASTER B2/B1 ferric oxide index. Haematite and goethite
    exhibit strong absorption in the blue (~490 nm, Band 2) and high
    reflectance in the red (~665 nm, Band 4), yielding elevated B4/B2 ratios
    in gossanous or iron-stained outcrops. While Sentinel-2 has finer spatial
    resolution than ASTER VNIR (10 m vs. 15 m), the band positions are not
    identical, so values are not directly comparable.

    Args:
        b4_red: Band 4 surface reflectance (Red, ~665 nm).
        b2_blue: Band 2 surface reflectance (Blue, ~490 nm). Denominator.

    Returns:
        Iron oxide proxy array (float64), NaN where B2 is zero or NaN.
    """
    b4 = np.asarray(b4_red, dtype=np.float64)
    b2 = np.asarray(b2_blue, dtype=np.float64)
    return _safe_ratio(b4, b2)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_all_s2_features(
    bands: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute all Sentinel-2 spectral features from a band dictionary.

    Processes all implemented Sentinel-2 spectral indices in a single call.
    The returned dictionary is ready for merging into the MPM feature stack
    alongside ASTER indices, geophysical grids, and structural features.

    Expected keys in `bands`:
        ``b2``  – Band 2 Blue reflectance (~490 nm)
        ``b3``  – Band 3 Green reflectance (~560 nm)
        ``b4``  – Band 4 Red reflectance (~665 nm)
        ``b8``  – Band 8 NIR reflectance (~842 nm)
        ``b11`` – Band 11 SWIR1 reflectance (~1610 nm)
        ``b12`` – Band 12 SWIR2 reflectance (~2190 nm)

    Args:
        bands: Dictionary mapping lowercase band name strings to 2-D
               reflectance arrays of identical spatial shape.

    Returns:
        Dictionary with string keys and float64 index arrays:
            - ``"ndvi"``        – Vegetation index; use threshold ≥ 0.2 for
                                  ASTER veg mask generation.
            - ``"ndwi"``        – Water index; threshold ≥ 0.1 for water mask.
            - ``"clay_index"``  – Coarse clay/hydroxyl proxy (B11/B12).
            - ``"iron_oxide"``  – Fe³⁺ proxy (B4/B2).

    Raises:
        KeyError: If any required band key is absent from `bands`.
    """
    required_keys = {"b2", "b3", "b4", "b8", "b11", "b12"}
    missing = required_keys - bands.keys()
    if missing:
        raise KeyError(f"Missing required Sentinel-2 band keys: {sorted(missing)}")

    return {
        "ndvi": ndvi(bands["b4"], bands["b8"]),
        "ndwi": ndwi(bands["b3"], bands["b8"]),
        "clay_index": clay_index(bands["b11"], bands["b12"]),
        "iron_oxide": iron_oxide_s2(bands["b4"], bands["b2"]),
    }
