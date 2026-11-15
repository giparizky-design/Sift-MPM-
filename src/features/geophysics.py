"""Aeromagnetic and Gravity Derivative Features for Mineral Prospectivity Mapping.

Potential-field grids (Total Magnetic Intensity and Bouguer gravity) encode the
geometry, depth, and composition of subsurface geological bodies. Computing
their spatial derivatives reveals structural architecture that controls
mineralising fluid pathways and trap sites.

All frequency-domain filters in this module share the same processing chain:
  1. Zero-pad the input grid to the next power of 2 in both dimensions to
     suppress circular convolution artefacts (wrap-around).
  2. Apply a 2-D Hanning taper on the grid edges to reduce Gibbs ringing
     caused by abrupt truncation of the potential field at survey boundaries.
  3. Compute the 2-D FFT, apply the filter operator in wavenumber space.
  4. Inverse FFT, then crop back to the original grid dimensions.

Coordinate convention:
  - ``dx`` and ``dy`` are the grid cell spacings in **metres** along the
    x (easting) and y (northing) directions, respectively.
  - Wavenumbers are in **radians per metre** (rad m⁻¹).

References:
    - Blakely, R.J. (1995). Potential Theory in Gravity and Magnetic
      Applications. Cambridge University Press.
    - Cooper, G.R.J. & Cowan, D.R. (2006). Enhancing potential field data
      using filters based on the local phase. Computers & Geosciences, 32,
      1585-1591.
    - Miller, H.G. & Singh, V. (1994). Potential field tilt—a new concept
      for location of potential field sources. Journal of Applied Geophysics,
      32, 213-217.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter, percentile_filter, uniform_filter


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is ≥ n.

    Args:
        n: Positive integer.

    Returns:
        Next power of 2 ≥ n.
    """
    p = 1
    while p < n:
        p <<= 1
    return p


def _hanning_taper_2d(rows: int, cols: int) -> np.ndarray:
    """Build a 2-D Hanning taper (outer product of two 1-D Hanning windows).

    The taper smoothly attenuates the grid edges to zero, reducing Gibbs
    ringing artefacts that arise from abrupt truncation of potential-field
    grids at survey boundaries.

    Args:
        rows: Number of rows in the target array.
        cols: Number of columns in the target array.

    Returns:
        2-D float64 Hanning taper array of shape (rows, cols) with values
        in [0, 1].
    """
    win_r = np.hanning(rows)
    win_c = np.hanning(cols)
    return np.outer(win_r, win_c)


def _wavenumber_grids(
    rows: int,
    cols: int,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D wavenumber grids (kx, ky) in rad m⁻¹.

    The grids are arranged to match the output layout of ``numpy.fft.fft2``,
    with ``ky`` sign-inverted so positive ``ky`` points Geographic North
    (decreasing row index in north-up rasters).

    Args:
        rows: Number of rows of the (possibly padded) array.
        cols: Number of columns of the (possibly padded) array.
        dx: Grid cell spacing in the x (easting) direction, in metres.
        dy: Grid cell spacing in the y (northing) direction, in metres.

    Returns:
        Tuple ``(kx, ky)`` where each element is a 2-D float64 array of
        shape (rows, cols) containing angular wavenumbers in rad m⁻¹.
        ``kx`` varies along columns (East positive); ``ky`` varies along
        rows (North positive).
    """
    kx_1d = 2.0 * np.pi * fftfreq(cols, d=dx)   # shape (cols,), East positive
    ky_1d = -2.0 * np.pi * fftfreq(rows, d=dy)  # shape (rows,), North positive
    kx, ky = np.meshgrid(kx_1d, ky_1d)          # both shape (rows, cols)
    return kx, ky


def _ring_cosine_taper_1d(total_len: int, pad_before: int, inner_len: int) -> np.ndarray:
    """Build a 1-D window that is 1.0 across the interior and cosine-tapered in the pad ring."""
    w = np.ones(total_len, dtype=np.float64)
    if pad_before > 0:
        t = np.arange(pad_before, dtype=np.float64) / float(pad_before)
        w[:pad_before] = 0.5 * (1.0 - np.cos(np.pi * t))
    pad_after = total_len - (pad_before + inner_len)
    if pad_after > 0:
        t = np.arange(1, pad_after + 1, dtype=np.float64) / float(pad_after)
        w[pad_before + inner_len :] = 0.5 * (1.0 + np.cos(np.pi * t))
    return w


def _pad_and_taper(
    grid: np.ndarray,
    pad_width: int | None = None,
) -> tuple[np.ndarray, int, int, int, int, tuple[slice, slice], float]:
    """Demean, reflect-pad, and apply a ring-only cosine taper outside the interior grid.

    All interior grid weights remain strictly ``1.0`` so coastal and border
    anomalies are never attenuated, while the reflective pad ring smoothly
    tapers to ``0.0`` at the outer periodic FFT boundary.

    Args:
        grid: 2-D real-valued input array of shape (rows, cols).
        pad_width: Optional minimum padding width on each side.

    Returns:
        Tuple ``(padded, rows, cols, pad_rows, pad_cols, crop_slices, grid_mean)``.
    """
    grid = np.squeeze(np.asarray(grid, dtype=np.float64))
    if grid.ndim != 2:
        raise ValueError(f"grid must be a 2D array, got shape {grid.shape}")
    rows, cols = grid.shape

    finite_mask = np.isfinite(grid)
    grid_mean = float(np.nanmean(grid)) if np.any(finite_mask) else 0.0
    demeaned = np.where(finite_mask, grid - grid_mean, 0.0)

    min_pad_r = pad_width if pad_width is not None else max(8, rows // 4)
    min_pad_c = pad_width if pad_width is not None else max(8, cols // 4)
    pad_rows = _next_power_of_2(rows + 2 * min_pad_r)
    pad_cols = _next_power_of_2(cols + 2 * min_pad_c)

    pad_top = (pad_rows - rows) // 2
    pad_bottom = pad_rows - rows - pad_top
    pad_left = (pad_cols - cols) // 2
    pad_right = pad_cols - cols - pad_left

    padded = np.pad(
        demeaned,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="reflect",
    )

    win_r = _ring_cosine_taper_1d(pad_rows, pad_top, rows)
    win_c = _ring_cosine_taper_1d(pad_cols, pad_left, cols)
    padded = padded * np.outer(win_r, win_c)

    crop_slices = (slice(pad_top, pad_top + rows), slice(pad_left, pad_left + cols))
    return padded, rows, cols, pad_rows, pad_cols, crop_slices, grid_mean


# ---------------------------------------------------------------------------
# Frequency-domain potential-field filters
# ---------------------------------------------------------------------------

def reduced_to_pole(
    tmi: np.ndarray,
    inclination_deg: float,
    declination_deg: float,
    dx: float = 1.0,
    dy: float = 1.0,
    min_inc_deg: float = 20.0,
) -> np.ndarray:
    """Reduce Total Magnetic Intensity (TMI) to the pole (RTP) with MacLeod (1993) stabilization.

    Implements northward wavenumber sign correction (row 0 = Geographic North)
    and MacLeod et al. (1993) low-latitude pseudo-inclination amplitude
    stabilization (``|I_amp| >= min_inc_deg``) so equatorial surveys such as
    Kalimantan (|I| ~ 14°) do not suffer North-South wavenumber blow-ups.

    Args:
        tmi: 2-D Total Magnetic Intensity grid (nT), shape (rows, cols).
        inclination_deg: IGRF geomagnetic inclination in degrees (positive
                         downward; typical range −90° to +90°).
        declination_deg: IGRF geomagnetic declination in degrees (positive
                         eastward of geographic north).
        dx: Grid cell spacing in the x-direction (metres).
        dy: Grid cell spacing in the y-direction (metres).
        min_inc_deg: Minimum amplitude pseudo-inclination (degrees) for
                     MacLeod et al. (1993) equatorial stabilization.

    Returns:
        Zero-mean RTP-transformed magnetic anomaly grid (nT), float64, same
        spatial shape as ``tmi``.
    """
    tmi = np.asarray(tmi, dtype=np.float64)
    padded, rows, cols, pad_rows, pad_cols, crop_slices, _ = _pad_and_taper(tmi)

    # 1. Correct ky sign so positive ky points Geographic North (row 0 is North)
    kx = 2.0 * np.pi * fftfreq(pad_cols, d=dx)
    ky = -2.0 * np.pi * fftfreq(pad_rows, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    theta = np.arctan2(KX, KY)  # Azimuth clockwise from Geographic North

    # 2. MacLeod et al. (1993) low-latitude pseudo-inclination stabilization
    inc_rad = np.deg2rad(inclination_deg)
    dec_rad = np.deg2rad(declination_deg)
    sign_i = -1.0 if inclination_deg < 0 else 1.0
    inc_amp_rad = np.deg2rad(sign_i * max(abs(inclination_deg), min_inc_deg))

    # Complex directional phase & amplitude terms
    p_phase = np.sin(inc_rad) + 1j * np.cos(inc_rad) * np.cos(theta - dec_rad)
    p_amp = np.sin(inc_amp_rad) + 1j * np.cos(inc_amp_rad) * np.cos(theta - dec_rad)

    # MacLeod (1993) pseudo-inclination stabilized RTP filter:
    # Phase matches exact 1 / p_phase^2 = conj(p_phase)^2 / |p_phase|^2,
    # while amplitude is bounded by 1 / |p_amp|^2:
    rtp_filter = (np.conj(p_phase) ** 2) / ((np.abs(p_amp) ** 2) * (np.abs(p_phase) ** 2) + 1e-12)
    rtp_filter[0, 0] = 0.0

    spectrum = fft2(padded)
    rtp_spectrum = spectrum * rtp_filter
    result = np.real(ifft2(rtp_spectrum))

    return result[crop_slices]


def first_vertical_derivative(
    grid: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the First Vertical Derivative (FVD) of a potential-field grid.

    The vertical derivative dF/dz in frequency domain is obtained by
    multiplying the spectrum by the wavenumber magnitude k:

        FVD(k) = k · F(k)

    where k = √(kx² + ky²).

    Args:
        grid: 2-D potential-field grid (any consistent units), shape (rows,
              cols).
        dx: Grid cell spacing in the x-direction (metres).
        dy: Grid cell spacing in the y-direction (metres).

    Returns:
        FVD grid (units per metre), float64, same shape as `grid`.
    """
    grid = np.asarray(grid, dtype=np.float64)
    padded, rows, cols, pad_rows, pad_cols, crop_slices, _ = _pad_and_taper(grid)

    kx, ky = _wavenumber_grids(pad_rows, pad_cols, dx, dy)
    k = np.sqrt(kx**2 + ky**2)

    spectrum = fft2(padded)
    fvd_spectrum = spectrum * k
    result = np.real(ifft2(fvd_spectrum))

    return result[crop_slices]


def analytic_signal(
    tmi: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the Total Analytic Signal (TAS) amplitude of a magnetic grid.

    The analytic signal amplitude is:

        |A(x,y)| = sqrt( (dF/dx)² + (dF/dy)² + (dF/dz)² )

    Args:
        tmi: 2-D Total Magnetic Intensity grid (nT), shape (rows, cols).
        dx: Grid cell spacing in the x-direction (metres).
        dy: Grid cell spacing in the y-direction (metres).

    Returns:
        Analytic signal amplitude grid (nT m⁻¹), float64, same shape as
        `tmi`. All values are non-negative.
    """
    tmi = np.asarray(tmi, dtype=np.float64)
    padded, rows, cols, pad_rows, pad_cols, crop_slices, _ = _pad_and_taper(tmi)

    kx, ky = _wavenumber_grids(pad_rows, pad_cols, dx, dy)
    k = np.sqrt(kx**2 + ky**2)

    spectrum = fft2(padded)

    # Horizontal gradients: multiply by i*kx and i*ky
    dfdx = np.real(ifft2(1j * kx * spectrum))[crop_slices]
    dfdy = np.real(ifft2(1j * ky * spectrum))[crop_slices]
    # Vertical gradient: multiply by k
    dfdz = np.real(ifft2(k * spectrum))[crop_slices]

    return np.sqrt(dfdx**2 + dfdy**2 + dfdz**2)


def tilt_derivative(
    grid: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the Tilt Derivative (TDR) of a potential-field grid.

    TDR = arctan( FVD / HGM )

    Args:
        grid: 2-D potential-field grid (TMI nT or gravity mGal), shape
              (rows, cols).
        dx: Grid cell spacing in the x-direction (metres).
        dy: Grid cell spacing in the y-direction (metres).

    Returns:
        TDR grid (radians) in the range [−π/2, +π/2], float64, same shape
        as `grid`.
    """
    grid = np.asarray(grid, dtype=np.float64)
    padded, rows, cols, pad_rows, pad_cols, crop_slices, _ = _pad_and_taper(grid)

    kx, ky = _wavenumber_grids(pad_rows, pad_cols, dx, dy)
    k = np.sqrt(kx**2 + ky**2)

    spectrum = fft2(padded)

    dfdx = np.real(ifft2(1j * kx * spectrum))[crop_slices]
    dfdy = np.real(ifft2(1j * ky * spectrum))[crop_slices]
    fvd = np.real(ifft2(k * spectrum))[crop_slices]

    hgm = np.sqrt(dfdx**2 + dfdy**2)
    tdr = np.arctan2(fvd, hgm)
    return tdr


def horizontal_gradient_magnitude(
    grid: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the Horizontal Gradient Magnitude (HGM) of a potential-field grid.

    HGM = sqrt( (dG/dx)² + (dG/dy)² )

    Args:
        grid: 2-D potential-field grid (gravity mGal or TMI nT), shape
              (rows, cols).
        dx: Grid cell spacing in the x-direction (metres).
        dy: Grid cell spacing in the y-direction (metres).

    Returns:
        HGM grid (units per metre), float64, non-negative, same shape as
        `grid`.
    """
    grid = np.asarray(grid, dtype=np.float64)
    padded, rows, cols, pad_rows, pad_cols, crop_slices, _ = _pad_and_taper(grid)

    kx, ky = _wavenumber_grids(pad_rows, pad_cols, dx, dy)
    spectrum = fft2(padded)

    dfdx = np.real(ifft2(1j * kx * spectrum))[crop_slices]
    dfdy = np.real(ifft2(1j * ky * spectrum))[crop_slices]

    return np.sqrt(dfdx**2 + dfdy**2)


def upward_continue(
    grid: np.ndarray,
    height_m: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Upward-continue a potential-field grid to a higher observation plane.

    The upward continuation operator in the frequency domain is:

        UC(k, h) = exp(−k · h)

    Args:
        grid: 2-D potential-field grid (nT or mGal), shape (rows, cols).
        height_m: Continuation height in metres. Must be positive.
        dx: Grid cell spacing in the x-direction (metres).
        dy: Grid cell spacing in the y-direction (metres).

    Returns:
        Upward-continued grid (same units as input, with regional DC mean
        restored), float64, same shape as `grid`.

    Raises:
        ValueError: If `height_m` ≤ 0.
    """
    if height_m <= 0:
        raise ValueError(f"height_m must be positive, got {height_m}")

    grid = np.asarray(grid, dtype=np.float64)
    padded, rows, cols, pad_rows, pad_cols, crop_slices, grid_mean = _pad_and_taper(grid)

    kx, ky = _wavenumber_grids(pad_rows, pad_cols, dx, dy)
    k = np.sqrt(kx**2 + ky**2)

    uc_operator = np.exp(-k * height_m)

    spectrum = fft2(padded)
    uc_spectrum = spectrum * uc_operator
    result = np.real(ifft2(uc_spectrum))

    return result[crop_slices] + grid_mean


# ---------------------------------------------------------------------------
# Pass 1.3: Airborne EM (AusAEM / Tempest) & Regolith Cover Inversion
# ---------------------------------------------------------------------------


def aem_depth_slice_conductance(
    conductivity_3d: np.ndarray,
    depth_bounds_m: np.ndarray,
    reductant_trap_threshold_s: float = 10.0,
) -> dict[str, np.ndarray]:
    """Integrate 3-D Airborne EM conductivity slices (0–300 m) to conductance.

    Conductance $S(x,y) = \\int_0^{z_{\\max}} \\sigma(x,y,z)\\,dz$ (Siemens) maps
    graphitic/sulfidic shear zones that act as chemical reductant traps for
    oxidised gold-bearing hydrothermal fluids.

    Args:
        conductivity_3d: 3-D array of conductivity $\\sigma$ in S m⁻¹, shape
            ``(n_layers, H, W)``.
        depth_bounds_m: 1-D monotonically increasing depth boundaries in metres,
            shape ``(n_layers + 1,)``, e.g. ``[0, 10, 30, 60, 100, 180, 300]``.
        reductant_trap_threshold_s: Minimum integrated conductance (S) to infer
            a graphitic/sulfidic reductant trap. Defaults to ``10.0`` S.

    Returns:
        Dictionary with keys:
          - ``"aem_conductance_0_300m_S"``: 2-D float64 conductance map (S).
          - ``"aem_mean_conductivity_s_m"``: 2-D float64 depth-weighted mean
            conductivity (S m⁻¹).
          - ``"reductant_trap_inferred"``: 2-D boolean mask where conductance
            $\\ge$ ``reductant_trap_threshold_s``.
    """
    sigma = np.asarray(conductivity_3d, dtype=np.float64)
    bounds = np.asarray(depth_bounds_m, dtype=np.float64)
    if sigma.ndim != 3:
        raise ValueError(f"conductivity_3d must be 3-D (n_layers, H, W), got {sigma.shape}")
    if bounds.ndim != 1 or len(bounds) != sigma.shape[0] + 1:
        raise ValueError(
            "depth_bounds_m must be 1-D with length n_layers + 1; "
            f"got {bounds.shape} for {sigma.shape[0]} layers."
        )
    if np.any(np.diff(bounds) <= 0):
        raise ValueError("depth_bounds_m must be strictly increasing.")

    dz = np.diff(bounds)[:, np.newaxis, np.newaxis]  # (n_layers, 1, 1)
    safe_sigma = np.where(np.isnan(sigma), 0.0, np.clip(sigma, 0.0, None))
    valid_dz = np.where(np.isnan(sigma), 0.0, dz)

    conductance = np.sum(safe_sigma * dz, axis=0)
    total_depth = np.sum(valid_dz, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_sigma = np.where(total_depth > 0, conductance / total_depth, np.nan)

    all_nan = np.all(np.isnan(sigma), axis=0)
    conductance = np.where(all_nan, np.nan, conductance)
    reductant_trap = (conductance >= float(reductant_trap_threshold_s)) & (~all_nan)

    return {
        "aem_conductance_0_300m_S": conductance,
        "aem_mean_conductivity_s_m": mean_sigma,
        "reductant_trap_inferred": reductant_trap,
    }


def invert_regolith_thickness_from_aem(
    conductivity_3d: np.ndarray,
    depth_bounds_m: np.ndarray,
    bedrock_sigma_threshold_s_m: float = 0.01,
) -> np.ndarray:
    """Invert 3-D AEM conductivity slices for continuous regolith cover depth.

    In deeply weathered terrains (e.g. Yilgarn Craton, central Kalimantan),
    saline/clay-rich saprolite and lateritic regolith exhibit significantly
    higher electrical conductivity ($\\sigma > 0.01$ S m⁻¹) than fresh
    crystalline basement ($\\sigma \\ll 0.01$ S m⁻¹). This 1-D vertical
    conductance inversion estimates the continuous depth to resistive fresh
    basement by integrating the conductive upper layers and linearly
    interpolating the transition across the saprock interface.

    Args:
        conductivity_3d: 3-D conductivity array (S m⁻¹), shape ``(n_layers, H, W)``.
        depth_bounds_m: 1-D array of layer boundary depths (m), shape ``(n_layers + 1,)``.
        bedrock_sigma_threshold_s_m: Conductivity threshold (S m⁻¹) below which
            material is classified as resistive fresh basement. Defaults to ``0.01``.

    Returns:
        2-D float64 array of estimated regolith cover thickness in metres,
        shape ``(H, W)``.
    """
    sigma = np.asarray(conductivity_3d, dtype=np.float64)
    bounds = np.asarray(depth_bounds_m, dtype=np.float64)
    if sigma.ndim != 3 or bounds.ndim != 1 or len(bounds) != sigma.shape[0] + 1:
        raise ValueError("Invalid shapes for conductivity_3d and depth_bounds_m.")

    n_layers, H, W = sigma.shape
    dz = np.diff(bounds)
    thr = float(bedrock_sigma_threshold_s_m)

    regolith_depth = np.zeros((H, W), dtype=np.float64)
    reached_basement = np.zeros((H, W), dtype=bool)

    for k in range(n_layers):
        layer_sigma = sigma[k]
        thick = dz[k]
        is_regolith = (layer_sigma >= thr) & (~reached_basement) & np.isfinite(layer_sigma)

        # Fractional transition in the layer where conductivity drops below threshold
        if k > 0:
            prev_sigma = sigma[k - 1]
            in_transition = (
                (~reached_basement)
                & (layer_sigma < thr)
                & (prev_sigma >= thr)
                & np.isfinite(layer_sigma)
                & np.isfinite(prev_sigma)
            )
            denom = np.maximum(prev_sigma - layer_sigma, 1e-9)
            frac = np.clip((prev_sigma - thr) / denom, 0.0, 1.0)
            regolith_depth = np.where(
                in_transition, regolith_depth + frac * thick, regolith_depth
            )

        regolith_depth = np.where(is_regolith, regolith_depth + thick, regolith_depth)
        reached_basement |= (layer_sigma < thr) & np.isfinite(layer_sigma)

    all_nan = np.all(np.isnan(sigma), axis=0)
    return np.where(all_nan, np.nan, regolith_depth)


# ---------------------------------------------------------------------------
# Pass 1 Physics: Bouguer Reduction, Moho Inversion, Worms & Host Gates
# ---------------------------------------------------------------------------


def bouguer_slab_reduction(
    free_air_mgal: np.ndarray,
    dem_m: np.ndarray,
) -> np.ndarray:
    """Convert Free-Air gravity (Sandwell & Smith v33.1) to Complete Bouguer anomaly.

    Enforces Reconciled Engineering Rule 6 (Bouguer Slab Guard):
    Applies standard Bouguer slab reduction before any derivative or worm filtering:
        Delta g_B = Delta g_FA - (0.1119 * max(0.0, h_DEM))

    Args:
        free_air_mgal: 2-D array of Free-Air gravity anomaly (mGal).
        dem_m: 2-D array of topographic DEM elevation (metres).

    Returns:
        2-D array of Complete Bouguer gravity anomaly (mGal).
    """
    fa = np.asarray(free_air_mgal, dtype=np.float64)
    h = np.asarray(dem_m, dtype=np.float64)
    if fa.shape != h.shape:
        raise ValueError(f"free_air_mgal shape {fa.shape} != dem_m shape {h.shape}")
    return fa - (0.1119 * np.maximum(0.0, h))


def moho_depth_from_bouguer(
    bouguer_mgal: np.ndarray,
    pixel_size_m: float,
) -> np.ndarray:
    """Compute Moho crustal depth from 30 km low-pass Bouguer slab gravity inversion.

    Enforces Reconciled Engineering Rule 6:
    Replaces synthetic Airy-Heiskanen DEM formulas (30 + dem * 0.0065) with
    direct 30 km low-pass Bouguer gravity inversion:
        bouguer_30km = gaussian_filter(bouguer_mgal, sigma=30000.0 / pixel_size_m)
        moho_depth_km = np.clip(32.0 - (bouguer_30km / 16.76), 18.0, 45.0)

    Args:
        bouguer_mgal: 2-D array of Complete Bouguer gravity (mGal).
        pixel_size_m: Grid resolution in metres.

    Returns:
        2-D array of Moho crustal thickness/depth (km), clipped to [18.0, 45.0].
    """
    if pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be positive, got {pixel_size_m}")
    bg = np.asarray(bouguer_mgal, dtype=np.float64)
    sigma = 30000.0 / float(pixel_size_m)
    bg_30km = gaussian_filter(bg, sigma=sigma)
    return np.clip(32.0 - (bg_30km / 16.76), 18.0, 45.0)


def multiscale_crustal_worms(
    field_grid: np.ndarray,
    dx: float,
    dy: float,
    depths_m: Union[list[float], tuple[float, ...]] = (5000.0, 15000.0, 30000.0),
    prefix: str = "bouguer",
) -> dict[str, np.ndarray]:
    """Compute upward-continued horizontal gradient magnitude worms at multiple crustal depths.

    For each upward continuation height h (e.g. 5 km, 15 km, 30 km):
        1. Upward-continues the potential-field grid by h metres.
        2. Computes the horizontal gradient magnitude (HGM).

    Args:
        field_grid: 2-D potential-field grid (Bouguer gravity mGal or RTP nT).
        dx: Cell spacing along easting in metres.
        dy: Cell spacing along northing in metres.
        depths_m: Sequence of continuation heights in metres.
        prefix: Name prefix for dictionary keys (e.g. 'bouguer' or 'rtp').

    Returns:
        Dictionary with keys f'{prefix}_worm_{depth_km}km'.
    """
    grid = np.asarray(field_grid, dtype=np.float64)
    worms: dict[str, np.ndarray] = {}
    for d in depths_m:
        d_val = float(d)
        d_km = int(round(d_val / 1000.0))
        uc = upward_continue(grid, height_m=d_val, dx=dx, dy=dy)
        hgm = horizontal_gradient_magnitude(uc, dx=dx, dy=dy)
        worms[f"{prefix}_worm_{d_km}km"] = hgm
    return worms


def compute_merdith_contrasts(
    field_grid: np.ndarray,
    pixel_size_m: float,
    prefix: str = "mag",
) -> dict[str, np.ndarray]:
    """Compute Merdith multi-scale contrasts: 1 km local mean and 4 km 80th/20th percentile contrast.

    Merdith et al. multi-scale potential-field architecture captures regional
    geophysical gradients:
      - 1 km local mean: geophys_mean_1km_*
      - 4 km 80th/20th percentile contrast ratio: geophys_contrast_4km_*

    Args:
        field_grid: 2-D potential-field grid.
        pixel_size_m: Grid resolution in metres.
        prefix: Name suffix/indicator (e.g. 'mag' or 'bouguer').

    Returns:
        Dictionary with keys:
          - f'geophys_mean_1km_{prefix}'
          - f'geophys_contrast_4km_{prefix}'
    """
    grid = np.asarray(field_grid, dtype=np.float64)
    win_1km = max(3, int(round(1000.0 / float(pixel_size_m))))
    win_4km = max(5, int(round(4000.0 / float(pixel_size_m))))

    # 1 km local mean
    mean_1km = uniform_filter(grid, size=win_1km)

    # 4 km 80th and 20th percentiles
    p80 = percentile_filter(grid, percentile=80, size=win_4km)
    p20 = percentile_filter(grid, percentile=20, size=win_4km)

    # Safe contrast ratio
    denom = np.maximum(np.abs(p20), 1e-6)
    contrast_4km = np.where(
        p20 > 0,
        p80 / denom,
        (p80 - p20) / (np.abs(p20) + 1.0)
    )

    return {
        f"geophys_mean_1km_{prefix}": mean_1km,
        f"geophys_contrast_4km_{prefix}": contrast_4km,
    }


def compute_intrusive_host_gate(
    analytic_signal_grid: np.ndarray,
    geophys_contrast_4km_mag: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Compute intrusive/arc host prior for Heads k=0, 1, 2.

    Computes mag_analytic_signal_texture * geophys_contrast_4km_mag.
    Texture is measured via local standard deviation of analytic signal.

    Args:
        analytic_signal_grid: 2-D Total Analytic Signal amplitude grid.
        geophys_contrast_4km_mag: 4 km 80th/20th percentile magnetic contrast ratio.
        window_size: Filter window size for texture estimation.

    Returns:
        2-D array of intrusive host gate prior.
    """
    as_arr = np.asarray(analytic_signal_grid, dtype=np.float64)
    mean = uniform_filter(as_arr, size=window_size)
    mean_sq = uniform_filter(as_arr ** 2, size=window_size)
    as_texture = np.sqrt(np.maximum(0.0, mean_sq - mean ** 2))

    contrast = np.asarray(geophys_contrast_4km_mag, dtype=np.float64)
    return as_texture * contrast


def compute_ultramafic_plateau_gate(
    slope_deg: np.ndarray,
    bouguer_15km: np.ndarray,
    max_slope_deg: float = 12.0,
) -> np.ndarray:
    """Compute ultramafic plateau prior for Head k=3 (Lateritic Fe-Ni).

    Computes plateau_low_slope_gate * bouguer_15km_high.
    Plateau low slope penalizes steep valleys and favors flat lateritic carapaces.
    Bouguer 15 km high captures dense ultramafic/ophiolite basement.

    Args:
        slope_deg: 2-D array of surface slope in degrees.
        bouguer_15km: 2-D array of 15 km upward continued Bouguer gravity (or 15 km worm).
        max_slope_deg: Maximum slope threshold for plateau gate.

    Returns:
        2-D array of ultramafic plateau gate prior in [0, 1].
    """
    sl = np.asarray(slope_deg, dtype=np.float64)
    low_slope_gate = np.clip(1.0 - (sl / max_slope_deg), 0.0, 1.0)

    b15 = np.asarray(bouguer_15km, dtype=np.float64)
    p10, p90 = np.percentile(b15, [10, 90])
    denom = max(1e-6, p90 - p10)
    bouguer_high = np.clip((b15 - p10) / denom, 0.0, 1.0)

    return low_slope_gate * bouguer_high


def compute_canopy_safe_spectral_indices(
    swir_clay_ratio: np.ndarray,
    swir_alunite_kaolinite_ratio: np.ndarray,
    swir_phyllic_ratio: np.ndarray,
    tir_quartz_ratio: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return strictly leakage-compliant engineered optical/spectral features for pilot blocks.

    Features are strictly named:
      - 'swir_veg_norm_clay_idx'
      - 'swir_argillic_rbd_logical'
      - 'swir_phyllic_rbd_logical'
      - 'tir_quartz_silicification_idx'
    (Never containing raw sensor band substrings 'b11', 'b13', 'aster_b', or 's2_b').
    """
    return {
        "swir_veg_norm_clay_idx": np.asarray(swir_clay_ratio, dtype=np.float32),
        "swir_argillic_rbd_logical": np.asarray(swir_alunite_kaolinite_ratio, dtype=np.float32),
        "swir_phyllic_rbd_logical": np.asarray(swir_phyllic_ratio, dtype=np.float32),
        "tir_quartz_silicification_idx": np.asarray(tir_quartz_ratio, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Pass 1.4: Real Satellite Gravimetry, Bouguer Reduction & Parker-Oldenburg Moho
# ---------------------------------------------------------------------------


def parker_oldenburg_moho_inversion(
    bouguer_mgal: np.ndarray,
    dx_m: float,
    dy_m: float,
    z0_km: float = 32.0,
    delta_rho: float = 400.0,
    k_low_km: float = 100.0,
    k_high_km: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert regional Bouguer gravity for Moho depth (km) and Moho THD step (mGal/km).

    Uses frequency-domain first-order Parker-Oldenburg downward continuation
    with cosine low-pass wavenumber tapering (Oldenburg 1974) to recover
    crustal thickness from the real Bouguer gravity anomaly, and computes the
    Total Horizontal Derivative (THD) of the regional Moho gravity field:

        THD_Moho = sqrt((dg_regional / dx)^2 + (dg_regional / dy)^2)

    Args:
        bouguer_mgal: 2-D Bouguer gravity anomaly grid (mGal).
        dx_m: Grid spacing in x (metres).
        dy_m: Grid spacing in y (metres).
        z0_km: Reference mean Moho depth (km).
        delta_rho: Mantle-crust density contrast (kg/m^3), typically 400 kg/m^3.
        k_low_km: Long-wavelength full-pass cutoff wavelength (km).
        k_high_km: Short-wavelength stop cutoff wavelength (km) to suppress
                   shallow crustal noise during downward continuation.

    Returns:
        Tuple ``(crustal_thickness_km, moho_gravity_step_mgal_km)`` as float32 arrays.
    """
    g = np.asarray(bouguer_mgal, dtype=np.float64)
    padded, rows, cols, pad_rows, pad_cols, crop_slices, g_mean = _pad_and_taper(g)
    kx, ky = _wavenumber_grids(pad_rows, pad_cols, dx_m, dy_m)
    k = np.hypot(kx, ky)  # rad / m

    # Oldenburg (1974) cosine low-pass filter between wavelengths k_low_km and k_high_km
    k_pass = 2.0 * np.pi / (k_low_km * 1000.0)
    k_stop = 2.0 * np.pi / (k_high_km * 1000.0)
    lp_filter = np.zeros_like(k)
    lp_filter[k <= k_pass] = 1.0
    trans = (k > k_pass) & (k < k_stop)
    lp_filter[trans] = 0.5 * (
        1.0 + np.cos(np.pi * (k[trans] - k_pass) / (k_stop - k_pass))
    )

    spec = fft2(padded)
    reg_spec = spec * lp_filter
    g_regional = np.real(ifft2(reg_spec))[crop_slices] + g_mean

    # Total Horizontal Derivative (THD) of the regional Moho gravity field in mGal/km
    dg_dx = np.real(ifft2(1j * kx * reg_spec))[crop_slices] * 1000.0
    dg_dy = np.real(ifft2(1j * ky * reg_spec))[crop_slices] * 1000.0
    moho_thd = np.hypot(dg_dx, dg_dy).astype(np.float32)

    # First-order Parker-Oldenburg Moho relief:
    # Delta_z(k) = - F[Delta_g(x)] * exp(|k| * z0_eff) / (2 * pi * G * delta_rho)
    # With stabilized regional attenuation so long-wavelength Moho root is recovered cleanly
    G_si = 6.67430e-11
    slab_factor_mgal_per_km = 2.0 * np.pi * G_si * delta_rho * 1.0e8  # ~16.77 mGal/km
    z0_eff_m = min(z0_km * 1000.0, 12000.0)  # stabilized continuation scale inside passband
    po_gain = np.exp(np.minimum(k * z0_eff_m, 1.2)) * lp_filter
    delta_g_po = np.real(ifft2(spec * po_gain))[crop_slices] + g_mean
    crustal_thickness_km = np.clip(
        z0_km - (delta_g_po / slab_factor_mgal_per_km),
        18.0,
        48.0,
    ).astype(np.float32)

    return crustal_thickness_km, moho_thd


def load_and_compute_real_gravity_channels(
    dem: np.ndarray,
    dst_crs: str,
    dst_transform,
    res_m: float,
    gravity_tif_candidates: list[str | Path] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real satellite Free-Air gravity, reproject with cubic spline, and derive Bouguer/Moho grids.

    Raises:
        FileNotFoundError: If no real satellite gravity GeoTIFF is found on disk.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    root_dir = Path(__file__).resolve().parent.parent.parent
    if gravity_tif_candidates is None:
        gravity_tif_candidates = [
            root_dir / "data" / "raw" / "geophysics" / "sandwell_smith_gravity_kalimantan.tif",
            root_dir / "data" / "kalimantan" / "processed" / "sandwell_smith_gravity_kalimantan.tif",
            root_dir / "data" / "raw" / "geophysics" / "sandwell_smith_gravity_indonesia.tif",
        ]

    src_path: Path | None = None
    for cand in gravity_tif_candidates:
        p = Path(cand)
        if p.exists():
            src_path = p
            break

    if src_path is None:
        raise FileNotFoundError(
            "Real satellite gravity GeoTIFF is missing on disk. Looked in: "
            f"{[str(c) for c in gravity_tif_candidates]}. "
            "Run `python scripts/fetch_satellite_gravity.py` to ingest authentic "
            "Sandwell & Smith / WGM2012 satellite gravimetry—synthetic `-0.1119 * dem` "
            "fallback is strictly forbidden."
        )

    H, W = dem.shape
    free_air_mgal = np.full((H, W), np.nan, dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=free_air_mgal,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.cubic_spline,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    # Fill any minor boundary NaNs with nearest valid gravity measurement
    nan_mask = ~np.isfinite(free_air_mgal)
    if np.all(nan_mask):
        raise ValueError(
            f"Reprojected gravity grid from {src_path} contains all NaNs for target bounds."
        )
    if np.any(nan_mask):
        from scipy.ndimage import distance_transform_edt

        _, nearest_idx = distance_transform_edt(nan_mask, return_indices=True)
        free_air_mgal = free_air_mgal[tuple(nearest_idx)]

    bouguer_mgal = bouguer_slab_reduction(free_air_mgal, dem).astype(np.float32)
    crustal_thickness_km, moho_gravity_step = parker_oldenburg_moho_inversion(
        bouguer_mgal,
        dx_m=res_m,
        dy_m=res_m,
    )
    return bouguer_mgal, crustal_thickness_km, moho_gravity_step



