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

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


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
    i.e. zero frequency is at the top-left corner and frequencies increase
    positively then wrap through the Nyquist.

    Args:
        rows: Number of rows of the (possibly zero-padded) array.
        cols: Number of columns of the (possibly zero-padded) array.
        dx: Grid cell spacing in the x (easting) direction, in metres.
        dy: Grid cell spacing in the y (northing) direction, in metres.

    Returns:
        Tuple ``(kx, ky)`` where each element is a 2-D float64 array of
        shape (rows, cols) containing angular wavenumbers in rad m⁻¹.
        ``kx`` varies along columns; ``ky`` varies along rows.
    """
    # fftfreq returns cycles per sample; multiply by 2π/spacing to get rad m⁻¹
    kx_1d = fftfreq(cols, d=dx) * 2.0 * np.pi  # shape (cols,)
    ky_1d = fftfreq(rows, d=dy) * 2.0 * np.pi  # shape (rows,)
    kx, ky = np.meshgrid(kx_1d, ky_1d)          # both shape (rows, cols)
    return kx, ky


def _pad_and_taper(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, int, int, int]:
    """Zero-pad a 2-D grid to the next power of 2 and apply a Hanning taper.

    The taper is applied **before** padding so that the padded zeros form a
    smooth boundary rather than encountering a step from the Hanning taper.

    Args:
        grid: 2-D real-valued input array of shape (rows, cols).

    Returns:
        Tuple ``(padded, rows, cols, pad_rows, pad_cols)`` where:
            - ``padded``: Tapered and zero-padded array of shape
              (pad_rows, pad_cols).
            - ``rows``: Original number of rows.
            - ``cols``: Original number of columns.
            - ``pad_rows``: Padded number of rows (next power of 2).
            - ``pad_cols``: Padded number of columns (next power of 2).
    """
    grid = np.squeeze(np.asarray(grid, dtype=np.float64))
    if grid.ndim != 2:
        raise ValueError(f"grid must be a 2D array, got shape {grid.shape}")
    rows, cols = grid.shape

    taper = _hanning_taper_2d(rows, cols)
    tapered = grid * taper

    pad_rows = _next_power_of_2(rows)
    pad_cols = _next_power_of_2(cols)

    padded = np.zeros((pad_rows, pad_cols), dtype=np.float64)
    padded[:rows, :cols] = tapered
    return padded, rows, cols, pad_rows, pad_cols


# ---------------------------------------------------------------------------
# Frequency-domain potential-field filters
# ---------------------------------------------------------------------------

def reduced_to_pole(
    tmi: np.ndarray,
    inclination_deg: float,
    declination_deg: float,
) -> np.ndarray:
    """Reduce Total Magnetic Intensity (TMI) to the pole (RTP).

    RTP transforms the observed TMI — which shows dipolar anomaly shapes
    skewed by the present-day geomagnetic field direction — into the form
    the anomaly would take at the magnetic pole, where induced magnetisation
    is vertical and anomalies appear symmetric and centred over their sources.
    This removes the directional asymmetry that complicates geological
    interpretation in mid-latitude surveys.

    The frequency-domain RTP operator is (Blakely 1995, eq. 12.49):

        RTP(k) = 1 / [ sin(I) + i·cos(I)·cos(D - θ) ]²

    where I = inclination, D = declination, θ = direction of wavenumber
    vector, and i = √−1.

    .. warning::
        **High-latitude instability**: As inclination approaches 0° (near the
        magnetic equator) the RTP operator becomes singular, amplifying noise
        catastrophically in the N-S wavenumber directions. For surveys where
        |inclination| < 20°, use **pseudogravity transformation** instead
        (which remains stable at all latitudes) or apply a low-inclination
        stabilisation factor (e.g. Luo et al. 2010).

    Processing chain: Hanning taper → zero-pad to next power of 2 → FFT →
    apply RTP operator → IFFT → crop to original size.

    Args:
        tmi: 2-D Total Magnetic Intensity grid (nT), shape (rows, cols).
             NaN values must be filled (e.g. with grid mean) before calling.
        inclination_deg: IGRF geomagnetic inclination in degrees (positive
                         downward; typical range −90° to +90°).
        declination_deg: IGRF geomagnetic declination in degrees (positive
                         eastward of geographic north).

    Returns:
        RTP-transformed magnetic grid (nT, approximately), float64, same
        spatial shape as `tmi`.
    """
    tmi = np.asarray(tmi, dtype=np.float64)
    padded, rows, cols, pad_rows, pad_cols = _pad_and_taper(tmi)

    kx, ky = _wavenumber_grids(pad_rows, pad_cols, dx=1.0, dy=1.0)
    # We work with unit grid spacing here; actual cell size cancels in the
    # direction-cosine formulation.

    inc = np.deg2rad(inclination_deg)
    dec = np.deg2rad(declination_deg)

    # Wavenumber magnitude and azimuth
    k = np.sqrt(kx**2 + ky**2)
    # Avoid division by zero at DC (k=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arctan2(kx, ky)  # azimuth of wavenumber vector

    # Direction cosine of magnetisation vector in frequency domain:
    # L(k) = sin(I) + i·cos(I)·cos(D - theta)
    L = np.sin(inc) + 1j * np.cos(inc) * np.cos(dec - theta)

    # At DC (k=0) the RTP is undefined; set to 1 (passes DC unchanged)
    L2 = L**2
    safe_L2 = np.where(k == 0, 1.0 + 0j, L2)

    # RTP operator: 1/L²  (equivalent to removing both the direction of the
    # inducing field and the direction of resultant magnetisation)
    rtp_operator = np.where(k == 0, 1.0 + 0j, 1.0 / safe_L2)

    spectrum = fft2(padded)
    rtp_spectrum = spectrum * rtp_operator
    result = np.real(ifft2(rtp_spectrum))

    return result[:rows, :cols]


def first_vertical_derivative(
    grid: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the First Vertical Derivative (FVD) of a potential-field grid.

    The vertical derivative dF/dz in frequency domain is obtained by
    multiplying the spectrum by the wavenumber magnitude k:

        FVD(k) = k · F(k)

    where k = √(kx² + ky²). FVD sharpens anomaly edges and enhances shallow
    sources over deep ones, making it useful for mapping shallow structural
    contacts. Because differentiation amplifies high-frequency noise, the
    Hanning taper applied during preprocessing is important for stability.

    Args:
        grid: 2-D potential-field grid (any consistent units), shape (rows,
              cols). Fill NaN before calling.
        dx: Grid cell spacing in the x-direction (metres).
        dy: Grid cell spacing in the y-direction (metres).

    Returns:
        FVD grid (units per metre), float64, same shape as `grid`.
    """
    grid = np.asarray(grid, dtype=np.float64)
    padded, rows, cols, pad_rows, pad_cols = _pad_and_taper(grid)

    kx, ky = _wavenumber_grids(pad_rows, pad_cols, dx, dy)
    k = np.sqrt(kx**2 + ky**2)

    spectrum = fft2(padded)
    fvd_spectrum = spectrum * k
    result = np.real(ifft2(fvd_spectrum))

    return result[:rows, :cols]


def analytic_signal(
    tmi: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the Total Analytic Signal (TAS) amplitude of a magnetic grid.

    The analytic signal amplitude is:

        |A(x,y)| = sqrt( (dF/dx)² + (dF/dy)² + (dF/dz)² )

    All three gradients are computed in the frequency domain:
        - dF/dx: multiply spectrum by i·kx
        - dF/dy: multiply spectrum by i·ky
        - dF/dz: multiply spectrum by k (= |k|)

    The analytic signal is independent of the direction of magnetisation,
    making it particularly useful for remanently magnetised bodies or in
    low-latitude surveys where RTP is unstable. Anomaly peaks are centred
    directly above causative bodies.

    Args:
        tmi: 2-D Total Magnetic Intensity grid (nT), shape (rows, cols).
        dx: Grid cell spacing in the x-direction (metres).
        dy: Grid cell spacing in the y-direction (metres).

    Returns:
        Analytic signal amplitude grid (nT m⁻¹), float64, same shape as
        `tmi`. All values are non-negative.
    """
    tmi = np.asarray(tmi, dtype=np.float64)
    padded, rows, cols, pad_rows, pad_cols = _pad_and_taper(tmi)

    kx, ky = _wavenumber_grids(pad_rows, pad_cols, dx, dy)
    k = np.sqrt(kx**2 + ky**2)

    spectrum = fft2(padded)

    # Horizontal gradients: multiply by i*kx and i*ky
    dfdx = np.real(ifft2(1j * kx * spectrum))[:rows, :cols]
    dfdy = np.real(ifft2(1j * ky * spectrum))[:rows, :cols]
    # Vertical gradient: multiply by k
    dfdz = np.real(ifft2(k * spectrum))[:rows, :cols]

    return np.sqrt(dfdx**2 + dfdy**2 + dfdz**2)


def tilt_derivative(
    grid: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the Tilt Derivative (TDR) of a potential-field grid.

    TDR = arctan( FVD / HGM )

    where FVD is the first vertical derivative and HGM is the horizontal
    gradient magnitude:

        HGM = sqrt( (dF/dx)² + (dF/dy)² )

    .. note::
        **Geological contact mapping**: The zero-contour of TDR (where TDR = 0)
        maps directly onto geological contacts and lithological boundaries,
        because the tilt angle changes sign across edges between bodies of
        contrasting physical properties (Verduzco et al. 2004). The
        ±π/4 contours approximate the half-depth to the source.

    All three gradient components are computed in the frequency domain using
    the same padded, tapered FFT framework as the other filters in this module.

    Args:
        grid: 2-D potential-field grid (TMI nT or gravity mGal), shape
              (rows, cols). Fill NaN before calling.
        dx: Grid cell spacing in the x-direction (metres).
        dy: Grid cell spacing in the y-direction (metres).

    Returns:
        TDR grid (radians) in the range [−π/2, +π/2], float64, same shape
        as `grid`.

    References:
        Miller, H.G. & Singh, V. (1994). Journal of Applied Geophysics,
        32, 213-217.
        Verduzco, B. et al. (2004). The Leading Edge, 23(1), 75-79.
    """
    grid = np.asarray(grid, dtype=np.float64)
    padded, rows, cols, pad_rows, pad_cols = _pad_and_taper(grid)

    kx, ky = _wavenumber_grids(pad_rows, pad_cols, dx, dy)
    k = np.sqrt(kx**2 + ky**2)

    spectrum = fft2(padded)

    dfdx = np.real(ifft2(1j * kx * spectrum))[:rows, :cols]
    dfdy = np.real(ifft2(1j * ky * spectrum))[:rows, :cols]
    fvd = np.real(ifft2(k * spectrum))[:rows, :cols]

    hgm = np.sqrt(dfdx**2 + dfdy**2)

    # arctan2 handles the quadrant correctly and avoids division by zero
    tdr = np.arctan2(fvd, hgm)
    return tdr


def horizontal_gradient_magnitude(
    grid: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the Horizontal Gradient Magnitude (HGM) of a potential-field grid.

    HGM = sqrt( (dG/dx)² + (dG/dy)² )

    Applied to a Bouguer gravity grid, edges (ridges) in the HGM map
    correspond to density contrasts — i.e., geological contacts, fault zones,
    and lithological boundaries. In magnetic data, HGM edges mark magnetic
    susceptibility contrasts. HGM is preferred over vertical derivatives for
    gravity data because it is less sensitive to noise.

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
    padded, rows, cols, pad_rows, pad_cols = _pad_and_taper(grid)

    kx, ky = _wavenumber_grids(pad_rows, pad_cols, dx, dy)
    spectrum = fft2(padded)

    dfdx = np.real(ifft2(1j * kx * spectrum))[:rows, :cols]
    dfdy = np.real(ifft2(1j * ky * spectrum))[:rows, :cols]

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

    where k is the wavenumber magnitude (rad m⁻¹) and h is the continuation
    height in metres. This exponential decay attenuates short-wavelength
    components more strongly than long-wavelength ones, effectively separating
    shallow (short-wavelength) anomalies from deep (long-wavelength) sources.

    Common MPM uses:
      - Upward-continue 500–2000 m to remove near-surface cultural noise.
      - Subtract upward-continued field from original to isolate shallow
        sources (equivalent-source residual).

    Args:
        grid: 2-D potential-field grid (nT or mGal), shape (rows, cols).
              Fill NaN before calling.
        height_m: Continuation height in metres. Must be positive.
        dx: Grid cell spacing in the x-direction (metres).
        dy: Grid cell spacing in the y-direction (metres).

    Returns:
        Upward-continued grid (same units as input), float64, same shape
        as `grid`.

    Raises:
        ValueError: If `height_m` ≤ 0.
    """
    if height_m <= 0:
        raise ValueError(f"height_m must be positive, got {height_m}")

    grid = np.asarray(grid, dtype=np.float64)
    padded, rows, cols, pad_rows, pad_cols = _pad_and_taper(grid)

    kx, ky = _wavenumber_grids(pad_rows, pad_cols, dx, dy)
    k = np.sqrt(kx**2 + ky**2)

    # Upward continuation operator; DC term (k=0) is passed unmodified
    uc_operator = np.exp(-k * height_m)

    spectrum = fft2(padded)
    uc_spectrum = spectrum * uc_operator
    result = np.real(ifft2(uc_spectrum))

    return result[:rows, :cols]


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

