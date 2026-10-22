"""ASTER Band Ratio Feature Engineering for Mineral Prospectivity Mapping.

ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer) carries
14 spectral bands spanning VNIR, SWIR, and TIR regions. For hydrothermal
alteration mapping, the SWIR bands 4–9 (1.60–2.43 µm) are the primary
fingerprint bands because diagnostic absorption features of phyllosilicates
(Al-OH at ~2.2 µm), carbonates (CO3 at ~2.35 µm), and chlorite/epidote
(Fe-OH at ~2.25 µm) all fall within this window.

**Vegetation masking is mandatory**: SWIR reflectance from green vegetation
overwhelms the mineralogical signal. All spectral indices must be set to NaN
wherever NDVI ≥ 0.2, following the standard practice recommended by
Rowan & Mars (2003) and Crosta et al. (2003).

References:
    - Rowan, L.C. & Mars, J.C. (2003). Lithologic mapping in the Mountain Pass,
      California area using Advanced Spaceborne Thermal Emission and Reflection
      Radiometer (ASTER) data. Remote Sensing of Environment, 84(3), 350-366.
    - Hewson, R.D. et al. (2005). Mapping minerals associated with gold
      mineralisation using ASTER data. Proceedings of IGARSS.
    - Ninomiya, Y. (2002). Mapping quartz, carbonate minerals, and mafic-
      ultramafic rocks using remotely sensed multispectral thermal infrared
      ASTER data. Proc. SPIE, 4710, 191-202.
    - Ninomiya, Y., Fu, B., & Cudahy, T.J. (2005). Detecting lithology with
      Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER)
      multispectral thermal infrared "radiance-at-sensor" data. Remote Sensing
      of Environment, 99(1-2), 127-139.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_ratio(
    a: np.ndarray,
    b: np.ndarray,
    fill: float = np.nan,
) -> np.ndarray:
    """Compute element-wise ratio a/b, guarding against division by zero or NaN.

    Args:
        a: Numerator array.
        b: Denominator array. Any element that is zero or NaN causes the
           corresponding output element to be set to `fill`.
        fill: Sentinel value written where `b` is zero or NaN.
              Defaults to ``np.nan`` so downstream statistics ignore bad pixels.

    Returns:
        Float64 array of the same shape as `a` and `b` with safe ratio values.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Build mask of problematic denominator pixels
    bad = (b == 0) | ~np.isfinite(b)

    # Suppress the actual division warning; we will overwrite bad pixels
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(bad, fill, a / b)

    return result


# ---------------------------------------------------------------------------
# SWIR spectral indices
# ---------------------------------------------------------------------------

def aloh_content(
    b5: np.ndarray,
    b6: np.ndarray,
    b7: np.ndarray,
) -> np.ndarray:
    """Al-OH content index sensitive to muscovite, sericite, and pyrophyllite.

    Formula: (B5 + B7) / B6

    Band B6 (2.185–2.225 µm) straddles the ~2.20 µm Al-OH absorption feature.
    High values of (B5+B7)/B6 indicate strong Al-OH absorptions, which are
    diagnostic of phyllic (sericitic) and advanced argillic alteration zones —
    both of which are critical pathfinders for epithermal and porphyry copper
    systems.

    Args:
        b5: ASTER Band 5 reflectance array (2.145–2.185 µm).
        b6: ASTER Band 6 reflectance array (2.185–2.225 µm). Denominator.
        b7: ASTER Band 7 reflectance array (2.235–2.285 µm).

    Returns:
        Al-OH content index array (float64), same shape as inputs.

    References:
        Rowan, L.C. & Mars, J.C. (2003). Remote Sensing of Environment,
        84(3), 350-366.
    """
    b5 = np.asarray(b5, dtype=np.float64)
    b7 = np.asarray(b7, dtype=np.float64)
    b6 = np.asarray(b6, dtype=np.float64)
    return _safe_ratio(b5 + b7, b6)


def carbonate_chlorite(
    b6: np.ndarray,
    b7: np.ndarray,
    b8: np.ndarray,
    b9: np.ndarray,
) -> np.ndarray:
    """Carbonate / chlorite index for propylitic alteration mapping.

    Formula: (B6 + B9) / (B7 + B8)

    Propylitic alteration (chlorite, epidote, carbonate, calcite) is the
    distal expression of many porphyry and epithermal systems. The CO3
    absorption at ~2.35 µm falls in B8 (2.295–2.365 µm), while chlorite
    Fe-OH absorbs at ~2.25 µm (B7). High ratio values indicate carbonate-
    dominated propylitic assemblages; low values favour chlorite.

    Args:
        b6: ASTER Band 6 reflectance array (2.185–2.225 µm).
        b7: ASTER Band 7 reflectance array (2.235–2.285 µm).
        b8: ASTER Band 8 reflectance array (2.295–2.365 µm).
        b9: ASTER Band 9 reflectance array (2.360–2.430 µm).

    Returns:
        Carbonate/chlorite ratio array (float64), same shape as inputs.

    References:
        Hewson, R.D. et al. (2005). Proceedings of IGARSS 2005.
    """
    b6 = np.asarray(b6, dtype=np.float64)
    b7 = np.asarray(b7, dtype=np.float64)
    b8 = np.asarray(b8, dtype=np.float64)
    b9 = np.asarray(b9, dtype=np.float64)
    return _safe_ratio(b6 + b9, b7 + b8)


def ferric_oxide_content(
    b1: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Ferric iron (Fe³⁺) oxide content index for gossans and iron-cap mapping.

    Formula: B2 / B1

    Ferric iron (haematite, goethite) is a strong absorber in the visible
    blue (~0.52 µm, Band 1) and a strong reflector in green/yellow (~0.66 µm,
    Band 2). Gossans and limonitic crusts that cap sulphide orebodies therefore
    show elevated B2/B1 ratios. This index is useful for locating oxidised
    caps that may indicate buried mineralisation.

    Args:
        b1: ASTER Band 1 reflectance array (0.52–0.60 µm). Denominator.
        b2: ASTER Band 2 reflectance array (0.63–0.69 µm).

    Returns:
        Fe³⁺ oxide index array (float64), same shape as inputs.

    References:
        Ninomiya, Y. (2002). Proc. SPIE, 4710, 191-202.
    """
    b1 = np.asarray(b1, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    return _safe_ratio(b2, b1)


def ferrous_silicate(
    b4: np.ndarray,
    b5: np.ndarray,
) -> np.ndarray:
    """Ferrous iron (Fe²⁺) silicate index sensitive to chlorite and epidote.

    Formula: B5 / B4

    Fe²⁺ in silicate minerals (chlorite, epidote) causes absorption at
    ~1.65 µm (Band 4). High B5/B4 values indicate Fe²⁺ silicate enrichment
    associated with propylitic alteration and mafic lithologies. Useful as a
    complementary layer to the carbonate/chlorite SWIR index.

    Args:
        b4: ASTER Band 4 reflectance array (1.600–1.700 µm). Denominator.
        b5: ASTER Band 5 reflectance array (2.145–2.185 µm).

    Returns:
        Fe²⁺ silicate index array (float64), same shape as inputs.
    """
    b4 = np.asarray(b4, dtype=np.float64)
    b5 = np.asarray(b5, dtype=np.float64)
    return _safe_ratio(b5, b4)


def silica_index(
    b11: np.ndarray,
    b13: np.ndarray,
) -> np.ndarray:
    """TIR silica emissivity index for mapping silicic alteration and quartzite.

    Formula: B13 / B11

    Quartz and silicified rocks show a prominent Reststrahlen emission feature
    (emissivity minimum / reflectance maximum) around 8.6–9.1 µm, captured by
    TIR Band 11 (8.475–8.825 µm). Band 13 (10.25–10.95 µm) falls outside this
    feature and serves as a reference. High B13/B11 ratios indicate silica-rich
    lithologies. This TIR-based index is particularly useful in arid terrain
    where SWIR vegetation masks are minimal but silicification extent is large.

    Args:
        b11: ASTER Band 11 emissivity/radiance array (8.475–8.825 µm).
             Denominator.
        b13: ASTER Band 13 emissivity/radiance array (10.25–10.95 µm).

    Returns:
        Silica index array (float64), same shape as inputs.

    References:
        Ninomiya, Y., Fu, B., & Cudahy, T.J. (2005). Remote Sensing of
        Environment, 99(1-2), 127-139.
    """
    b11 = np.asarray(b11, dtype=np.float64)
    b13 = np.asarray(b13, dtype=np.float64)
    return _safe_ratio(b13, b11)


def opaque_index(
    b1: np.ndarray,
    b4: np.ndarray,
) -> np.ndarray:
    """Crude opaque mineral / sulphide proxy using VNIR-to-SWIR slope.

    Formula: B4 / B1

    Opaque minerals (magnetite, ilmenite, sulphides such as pyrite when
    tarnished) tend to have flat or featureless spectra across VNIR to SWIR,
    yielding B4/B1 ratios near unity. Bright siliceous or clay-rich rocks have
    rising SWIR reflectance and therefore higher ratios. Low B4/B1 in
    conjunction with other alteration indices can help delineate sulphide-
    bearing zones. This proxy is coarse and should be treated as a supporting
    layer rather than a standalone discriminant.

    Args:
        b1: ASTER Band 1 reflectance array (0.52–0.60 µm). Denominator.
        b4: ASTER Band 4 reflectance array (1.600–1.700 µm).

    Returns:
        Opaque index array (float64), same shape as inputs.
    """
    b1 = np.asarray(b1, dtype=np.float64)
    b4 = np.asarray(b4, dtype=np.float64)
    return _safe_ratio(b4, b1)


# ---------------------------------------------------------------------------
# Vegetation masking
# ---------------------------------------------------------------------------

def apply_vegetation_mask(
    index: np.ndarray,
    veg_mask: np.ndarray,
) -> np.ndarray:
    """Set spectral index pixels to NaN wherever a vegetation mask is True.

    Green vegetation reflectance dominates SWIR signals when NDVI ≥ 0.2,
    rendering alteration indices unreliable. This function enforces the mask
    by writing NaN into flagged pixels so they propagate cleanly through
    downstream statistics and ML pipelines.

    Args:
        index: Spectral index array of any shape.
        veg_mask: Boolean array, same shape as `index`. True = vegetated cell
                  that should be masked.

    Returns:
        Copy of `index` with masked pixels set to NaN (float64).

    Raises:
        ValueError: If `index` and `veg_mask` have different shapes.
    """
    index = np.asarray(index, dtype=np.float64).copy()
    veg_mask = np.asarray(veg_mask, dtype=bool)

    if index.shape != veg_mask.shape:
        raise ValueError(
            f"Shape mismatch: index {index.shape} vs veg_mask {veg_mask.shape}"
        )

    index[veg_mask] = np.nan
    return index


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_all_indices(
    bands: dict[str, np.ndarray],
    veg_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute all ASTER spectral indices and apply the vegetation mask.

    Orchestrates the full ASTER alteration index pipeline:
      1. Compute each index from the supplied band arrays.
      2. Apply the vegetation mask (NaN where NDVI ≥ 0.2).
      3. Return a dictionary ready for stacking into the feature matrix.

    Expected keys in `bands`:
        ``b1`` – Band 1 VNIR (0.52–0.60 µm)
        ``b2`` – Band 2 VNIR (0.63–0.69 µm)
        ``b4`` – Band 4 SWIR (1.600–1.700 µm)
        ``b5`` – Band 5 SWIR (2.145–2.185 µm)
        ``b6`` – Band 6 SWIR (2.185–2.225 µm)
        ``b7`` – Band 7 SWIR (2.235–2.285 µm)
        ``b8`` – Band 8 SWIR (2.295–2.365 µm)
        ``b9`` – Band 9 SWIR (2.360–2.430 µm)
        ``b11`` – Band 11 TIR (8.475–8.825 µm)
        ``b13`` – Band 13 TIR (10.25–10.95 µm)

    Args:
        bands: Dictionary mapping band name strings (lowercase, e.g. ``"b1"``)
               to 2-D reflectance/radiance arrays of identical shape.
        veg_mask: Boolean 2-D array (same spatial shape as the band arrays).
                  True where NDVI ≥ 0.2. Passed to ``apply_vegetation_mask``
                  for every index.

    Returns:
        Dictionary with string keys and float64 masked index arrays:
            - ``"aloh_content"``
            - ``"carbonate_chlorite"``
            - ``"ferric_oxide_content"``
            - ``"ferrous_silicate"``
            - ``"silica_index"``
            - ``"opaque_index"``

    Raises:
        KeyError: If any required band key is missing from `bands`.
    """
    required_keys = {"b1", "b2", "b4", "b5", "b6", "b7", "b8", "b9", "b11", "b13"}
    missing = required_keys - bands.keys()
    if missing:
        raise KeyError(f"Missing required ASTER band keys: {sorted(missing)}")

    indices: dict[str, np.ndarray] = {
        "aloh_content": aloh_content(bands["b5"], bands["b6"], bands["b7"]),
        "carbonate_chlorite": carbonate_chlorite(
            bands["b6"], bands["b7"], bands["b8"], bands["b9"]
        ),
        "ferric_oxide_content": ferric_oxide_content(bands["b1"], bands["b2"]),
        "ferrous_silicate": ferrous_silicate(bands["b4"], bands["b5"]),
        "silica_index": silica_index(bands["b11"], bands["b13"]),
        "opaque_index": opaque_index(bands["b1"], bands["b4"]),
    }

    # Apply vegetation mask to every index
    return {
        name: apply_vegetation_mask(arr, veg_mask)
        for name, arr in indices.items()
    }



# =============================================================================
# Pass 1.1 — Sensor-agnostic hyperspectral SWIR/VNIR indices
# =============================================================================
# The functions below operate on an imaging-spectrometer cube of shape
# (bands, H, W) with an explicit wavelength vector in nanometres, so the same
# code serves EMIT (~7.4 nm sampling, 381–2493 nm), PRISMA (~10 nm, 400–2500 nm)
# and EnMAP (~6.5/10 nm, 420–2450 nm).  They are NOT meaningful on ASTER: ASTER
# SWIR has six broad bands, cannot resolve absorption-feature positions, and its
# SWIR detector failed in April 2008, which is why these sensors are needed for
# post-2008 alteration mapping.
#
# All inputs must be surface reflectance (atmospherically corrected, L2A).
# Feature-position and depth conventions follow the continuum-removal approach
# (Clark & Roush 1984) and the AusSpec / CSIRO SWIR mineral-mapping products
# (e.g. Cudahy et al. 2008).
# -----------------------------------------------------------------------------

# Window definitions in nm: (left shoulder, right shoulder)
AL_OH_WINDOW_NM: tuple[float, float] = (2120.0, 2245.0)
WATER_1900_WINDOW_NM: tuple[float, float] = (1850.0, 2040.0)
CARBONATE_WINDOW_NM: tuple[float, float] = (2270.0, 2370.0)
FERRIC_900_WINDOW_NM: tuple[float, float] = (750.0, 1300.0)
CALCITE_REFERENCE_NM: float = 2340.0


def _window_indices(wavelengths_nm: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Return band indices whose centres fall inside ``window`` (inclusive)."""
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    idx = np.where((wl >= window[0]) & (wl <= window[1]))[0]
    if idx.size < 3:
        raise ValueError(
            f"Need >= 3 bands in window {window} nm; sensor provides {idx.size}. "
            "Broad-band sensors (ASTER, Sentinel-2) cannot resolve this feature."
        )
    return idx


def continuum_removed_window(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-continuum removal over a single absorption window.

    The continuum is the straight line joining the reflectance at the first and
    last band of the window (a two-point hull, adequate for isolated features).

    Args:
        cube: Reflectance cube, shape (bands, H, W).
        wavelengths_nm: Band-centre wavelengths, shape (bands,).
        window: (left, right) shoulder wavelengths in nm.

    Returns:
        (cr, wl): continuum-removed reflectance of shape (n_win, H, W) and the
        wavelengths of the window bands, shape (n_win,).
    """
    idx = _window_indices(wavelengths_nm, window)
    wl = np.asarray(wavelengths_nm, dtype=np.float64)[idx]
    r = cube[idx].astype(np.float64)
    r_left, r_right = r[0], r[-1]
    frac = ((wl - wl[0]) / (wl[-1] - wl[0]))[:, None, None]
    continuum = r_left[None] + frac * (r_right - r_left)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = np.where(continuum > 0, r / continuum, np.nan)
    return cr, wl


def absorption_depth_and_position(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Depth (1 − CR_min) and sub-band minimum wavelength of an absorption.

    The minimum position is refined by fitting a parabola through the
    continuum-removed minimum and its two neighbours, which gives wavelength
    precision finer than the band spacing (standard practice for the 2200 nm
    Al-OH and 2340 nm carbonate features).

    Returns:
        (depth, position_nm), each float32 of shape (H, W). NaN where the
        continuum is invalid or the minimum lies on a window shoulder.
    """
    cr, wl = continuum_removed_window(cube, wavelengths_nm, window)
    filled = np.where(np.isfinite(cr), cr, np.inf)
    k = np.argmin(filled, axis=0)                      # (H, W)
    n = cr.shape[0]
    interior = (k > 0) & (k < n - 1)
    k_c = np.clip(k, 1, n - 2)
    take = lambda off: np.take_along_axis(cr, (k_c + off)[None], axis=0)[0]  # noqa: E731
    y0, y1, y2 = take(-1), take(0), take(1)
    x0, x1, x2 = wl[k_c - 1], wl[k_c], wl[k_c + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
        b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
        pos = np.where(a > 0, -b / (2 * a), x1)
    depth = 1.0 - np.nanmin(cr, axis=0)
    pos = np.where(interior & np.isfinite(depth), pos, np.nan)
    depth = np.where(interior, depth, np.nan)
    return depth.astype(np.float32), pos.astype(np.float32)


def al_oh_crystallinity(cube: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """White-mica (illite) crystallinity index: depth(2200 nm) / depth(1900 nm).

    Well-ordered, higher-temperature white mica (proximal to hydrothermal
    upflow) has a deep Al-OH feature relative to the molecular-water feature;
    poorly crystalline illite/smectite has a relatively deeper 1900 nm feature.
    Values > ~1 indicate well-crystallised white mica.
    """
    d2200, _ = absorption_depth_and_position(cube, wavelengths_nm, AL_OH_WINDOW_NM)
    d1900, _ = absorption_depth_and_position(cube, wavelengths_nm, WATER_1900_WINDOW_NM)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.where(d1900 > 1e-4, d2200 / d1900, np.nan)
    return ic.astype(np.float32)


def kaolinite_alunite_doublet(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    min_doublet_depth: float = 0.005,
) -> dict[str, np.ndarray]:
    """Detect the 2160/2165 nm + 2205 nm doublet of kaolinite-group / alunite.

    Kaolinite and alunite show a secondary minimum near 2160–2170 nm on the
    short-wavelength flank of the main ~2205 nm feature; white mica does not.
    The test is a local minimum in the continuum-removed spectrum between
    2150 and 2180 nm that is deeper (relative to its neighbouring local
    maximum toward 2190 nm) than ``min_doublet_depth``.

    Returns:
        dict with ``doublet_present`` (bool, H×W) and ``doublet_depth`` (float32).
    """
    cr, wl = continuum_removed_window(cube, wavelengths_nm, AL_OH_WINDOW_NM)
    sec = (wl >= 2150) & (wl <= 2180)
    shoulder = (wl > 2180) & (wl <= 2195)
    if sec.sum() == 0 or shoulder.sum() == 0:
        raise ValueError("Spectral sampling too coarse to resolve the 2165/2205 nm doublet.")
    sec_min = np.nanmin(cr[sec], axis=0)
    shoulder_max = np.nanmax(cr[shoulder], axis=0)
    depth = (shoulder_max - sec_min).astype(np.float32)
    return {"doublet_present": depth > min_doublet_depth, "doublet_depth": depth}


def carbonate_wavelength_shift(cube: np.ndarray, wavelengths_nm: np.ndarray) -> dict[str, np.ndarray]:
    """Carbonate CO3 feature position and its shift from calcite (2340 nm).

    Mg substitution moves the feature to shorter wavelength (dolomite ~2320 nm,
    magnesite ~2300 nm); Fe-bearing carbonates (ankerite/siderite) sit in
    between, so a positive ``shift_nm`` = 2340 − position indicates Mg/Fe
    carbonate, the typical proximal carbonation halo of orogenic gold systems.
    Note: chlorite/epidote (Mg-OH ~2330–2350 nm) overlap this window; interpret
    together with the 2250 nm Fe-OH feature where available.
    """
    depth, pos = absorption_depth_and_position(cube, wavelengths_nm, CARBONATE_WINDOW_NM)
    return {
        "carbonate_depth": depth,
        "carbonate_position_nm": pos,
        "carbonate_shift_nm": (CALCITE_REFERENCE_NM - pos).astype(np.float32),
    }


def fe_oxide_intensity(cube: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """Ferric-iron crystal-field absorption depth near 900 nm (hematite/goethite).

    Returns continuum-removed depth over 750–1300 nm, float32 (H, W).
    Strongly affected by vegetation red-edge; apply a vegetation mask.
    """
    depth, _ = absorption_depth_and_position(cube, wavelengths_nm, FERRIC_900_WINDOW_NM)
    return depth


def compute_hyperspectral_indices(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    veg_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute all Pass-1.1 hyperspectral alteration channels.

    Args:
        cube: Surface reflectance, shape (bands, H, W).
        wavelengths_nm: Band centres (nm).
        veg_mask: Optional boolean (H, W); True cells are set to NaN.

    Returns:
        Dict of float32 (H, W) layers, plus boolean ``kaolinite_alunite_doublet``.
    """
    d2200, p2200 = absorption_depth_and_position(cube, wavelengths_nm, AL_OH_WINDOW_NM)
    out: dict[str, np.ndarray] = {
        "hs_aloh_depth_2200": d2200,
        "hs_aloh_position_nm": p2200,
        "hs_aloh_crystallinity": al_oh_crystallinity(cube, wavelengths_nm),
        "hs_fe_oxide_intensity": fe_oxide_intensity(cube, wavelengths_nm),
    }
    carb = carbonate_wavelength_shift(cube, wavelengths_nm)
    out["hs_carbonate_depth"] = carb["carbonate_depth"]
    out["hs_carbonate_shift_nm"] = carb["carbonate_shift_nm"]
    dbl = kaolinite_alunite_doublet(cube, wavelengths_nm)
    out["hs_kaol_alun_doublet_depth"] = dbl["doublet_depth"]
    if veg_mask is not None:
        out = {k: apply_vegetation_mask(v, veg_mask) for k, v in out.items()}
    return out
