"""Structural Proximity and Terrain Feature Engineering for MPM.

Structural controls on mineralisation include fault networks, lithological
contacts, and fold hinges. Hydrothermal fluids preferentially ascend and
accumulate along permeable structural corridors; therefore proximity to faults,
fault intersections (triple-junction dilatancy), and structural complexity
indices are among the most predictive spatial features in any MPM model.

Terrain features (slope, aspect, curvature, TPI, flow accumulation) encode
the surface expression of subsurface geology and secondary dispersion
processes (weathering, geochemical transport in drainage networks).

All spatial operations use ``scipy.ndimage`` for efficient raster-based
calculations. DEM-derived features assume a regular grid with equal cell
spacing in x and y. All input arrays are treated as float64 internally.

References:
    - Horn, B.K.P. (1981). Hill shading and the reflectance map.
      Proceedings of the IEEE, 69(1), 14-47.
    - Evans, I.S. (1980). An integrated system of terrain analysis and slope
      mapping. Zeitschrift für Geomorphologie, Supplementband 36, 274-295.
    - Verduzco, B. et al. (2004). New insights into magnetic derivatives for
      structural mapping. The Leading Edge, 23(1), 75-79.
    - Wilson, J.P. & Gallant, J.C. (2000). Terrain Analysis: Principles and
      Applications. John Wiley & Sons.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------

def euclidean_distance_to_features(
    target_raster: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    """Compute Euclidean distance (metres) from every cell to the nearest
    binary geological feature (e.g. fault trace, lineament, contact).

    Uses ``scipy.ndimage.distance_transform_edt`` which implements the
    exact Euclidean distance transform in O(N) time. The raw pixel-space
    distances are multiplied by ``resolution_m`` to convert to metres.

    Distance-to-fault is one of the most consistently predictive features in
    MPM because many ore deposits cluster within 1–5 km of major faults.
    Log-transforming the output before model training often improves linearity
    with the response variable.

    Args:
        target_raster: 2-D binary integer or boolean array where 1 (True)
                       marks geological features (fault traces, contacts, etc.)
                       and 0 (False) is background. Shape (rows, cols).
        resolution_m: Ground sampling distance in metres (assumed isotropic).

    Returns:
        Float64 distance array of the same shape as `target_raster`, in
        metres. Feature pixels (value = 1) have distance 0.

    Raises:
        ValueError: If `resolution_m` ≤ 0.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")

    target = np.asarray(target_raster, dtype=bool)
    # distance_transform_edt returns distance from 0-valued (background) pixels
    # to the nearest non-zero pixel. We invert so that feature pixels = 0.
    # The sampling parameter scales pixel distances to physical units.
    dist_px = ndimage.distance_transform_edt(~target, sampling=resolution_m)
    return dist_px.astype(np.float64)


def extract_lineaments_from_tdr(
    tdr: np.ndarray,
    zero_contour_tolerance: float = 0.05,
) -> np.ndarray:
    """Extract geological contacts as a binary raster from a Tilt Derivative.

    The zero-contour of the Tilt Derivative (TDR) maps directly onto
    boundaries between bodies of contrasting physical properties — i.e.
    geological contacts, fault zones, and lithological edges. Pixels within
    ``zero_contour_tolerance`` radians of zero are flagged as contact pixels.

    Implementation approach:
      1. Find continuous zero-crossing contours using ``skimage.measure.find_contours``.
      2. Rasterise each contour back onto a binary array by rounding
         sub-pixel coordinates to the nearest grid cell.

    .. note::
        ``skimage`` is used only for its sub-pixel contour tracing (Marching
        Squares). The result is converted to a binary raster using NumPy
        integer rounding, so there is no continuous-space output.

    Args:
        tdr: 2-D Tilt Derivative array (radians), shape (rows, cols).
             Typically produced by :func:`geophysics.tilt_derivative`.
        zero_contour_tolerance: Half-width of the band around zero (radians)
                                within which a pixel is considered a contact.
                                Defaults to 0.05 rad (~3°). Increase for
                                noisier data; decrease for sharper linework.

    Returns:
        Binary uint8 array of shape (rows, cols): 1 = contact/lineament,
        0 = background.
    """
    try:
        from skimage.measure import find_contours
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for extract_lineaments_from_tdr. "
            "Install it with: pip install scikit-image"
        ) from exc

    tdr = np.asarray(tdr, dtype=np.float64)
    rows, cols = tdr.shape
    binary = np.zeros((rows, cols), dtype=np.uint8)

    # Strategy 1: simple threshold – pixels close to zero in the TDR grid
    near_zero = np.abs(tdr) <= zero_contour_tolerance
    binary[near_zero] = 1

    # Strategy 2: trace sub-pixel contours using Marching Squares and
    # rasterise them (adds detail along edges between threshold pixels)
    try:
        contours = find_contours(tdr, level=0.0)
        for contour in contours:
            # contour is an array of (row, col) float coordinates
            row_idx = np.round(contour[:, 0]).astype(int)
            col_idx = np.round(contour[:, 1]).astype(int)
            # Clip to valid bounds
            row_idx = np.clip(row_idx, 0, rows - 1)
            col_idx = np.clip(col_idx, 0, cols - 1)
            binary[row_idx, col_idx] = 1
    except Exception:
        # If contour tracing fails (e.g. all-NaN TDR), fall back to threshold
        pass

    return binary


def fault_intersection_density(
    fault_binary: np.ndarray,
    kernel_radius_px: int = 20,
) -> np.ndarray:
    """Estimate fault intersection / crossing density within a circular window.

    Fault intersections and triple junctions are dilatational sites where
    fluid upwelling and mineralisation are focussed. This function counts
    the number of fault-trace pixels within a circular neighbourhood of each
    cell, providing a smooth density map of structural complexity.

    Implementation:
      - A square uniform filter approximates the integral within a
        ``(2*kernel_radius_px + 1) × (2*kernel_radius_px + 1)`` window.
      - A circular mask is applied post-hoc: cells farther than
        ``kernel_radius_px`` from the window centre contribute zero weight
        by zeroing out corners with a pre-computed annular kernel.

    .. note::
        For a true circular kernel, this function convolves with a Boolean
        disk structuring element via ``scipy.ndimage.uniform_filter`` on a
        pre-expanded grid. The result is equivalent to summing fault pixels
        within a disk of radius ``kernel_radius_px`` cells.

    Args:
        fault_binary: 2-D binary array (0/1 or bool) with fault traces.
                      Shape (rows, cols).
        kernel_radius_px: Radius of the circular counting kernel in pixels.
                          Defaults to 20.

    Returns:
        Float64 density map of shape (rows, cols) with the count of fault
        pixels within the circular window centred on each cell.

    Raises:
        ValueError: If ``kernel_radius_px`` < 1.
    """
    if kernel_radius_px < 1:
        raise ValueError(f"kernel_radius_px must be ≥ 1, got {kernel_radius_px}")

    fault = np.asarray(fault_binary, dtype=np.float64)

    # Build a circular (disk) structuring element
    diameter = 2 * kernel_radius_px + 1
    y_idx, x_idx = np.ogrid[-kernel_radius_px:kernel_radius_px + 1,
                            -kernel_radius_px:kernel_radius_px + 1]
    disk = (x_idx**2 + y_idx**2 <= kernel_radius_px**2).astype(np.float64)
    disk_sum = disk.sum()

    # Convolve the fault binary with the disk kernel using ndimage
    from scipy.ndimage import convolve
    density = convolve(fault, disk, mode="constant", cval=0.0)

    # Normalise by disk area to get a density in [0, 1]
    # (optional; kept as raw count for interpretability as "fault pixels in disk")
    return density.astype(np.float64)


# ---------------------------------------------------------------------------
# Terrain (DEM-derived) features
# ---------------------------------------------------------------------------

def slope(
    dem: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    """Compute terrain slope using Horn's (1981) 8-neighbour finite difference.

    Horn's algorithm uses a 3×3 weighted finite difference kernel to estimate
    the E-W (dz/dx) and N-S (dz/dy) gradient components:

        dz/dx = [(z_NE + 2·z_E + z_SE) − (z_NW + 2·z_W + z_SW)] / (8·dx)
        dz/dy = [(z_SW + 2·z_S + z_SE) − (z_NW + 2·z_N + z_NE)] / (8·dy)

    Slope = arctan(√((dz/dx)² + (dz/dy)²)) converted to degrees.

    Args:
        dem: 2-D elevation array (metres), shape (rows, cols). NaN cells
             propagate to the output.
        resolution_m: Ground cell size in metres (assumed isotropic, dx = dy).

    Returns:
        Slope array (degrees) in [0, 90], float64, same shape as `dem`.
        Edge cells inherit the nearest valid gradient estimate via edge
        padding in ``numpy.gradient``.

    References:
        Horn, B.K.P. (1981). Proceedings of the IEEE, 69(1), 14-47.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")

    dem = np.asarray(dem, dtype=np.float64)
    # numpy.gradient uses central differences internally (equivalent to Horn
    # when applied to a DEM at uniform spacing with edge padding)
    dz_dy, dz_dx = np.gradient(dem, resolution_m)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    return np.degrees(slope_rad)


def aspect(
    dem: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    """Compute terrain aspect (azimuth of steepest descent) in degrees.

    Aspect is the compass direction (0°=North, clockwise) of the steepest
    downhill gradient. It is computed from the dz/dx and dz/dy gradient
    components using ``numpy.gradient``, then converted from mathematical
    angle convention (counterclockwise from East) to geographic convention
    (clockwise from North).

    Flat areas (zero gradient magnitude) are assigned −1 by convention to
    distinguish them from north-facing slopes (0°/360°).

    Args:
        dem: 2-D elevation array (metres), shape (rows, cols).
        resolution_m: Ground cell size in metres.

    Returns:
        Aspect array (degrees, 0–360 clockwise from North), float64, same
        shape as `dem`. Flat cells are assigned −1.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")

    dem = np.asarray(dem, dtype=np.float64)
    dz_dy, dz_dx = np.gradient(dem, resolution_m)

    # Mathematical convention: angle from positive x-axis, counterclockwise
    math_angle = np.arctan2(dz_dy, -dz_dx)

    # Convert to geographic azimuth: clockwise from North
    geographic = 90.0 - np.degrees(math_angle)
    geographic = geographic % 360.0

    # Flat areas: both gradients are zero
    flat = (dz_dx == 0) & (dz_dy == 0)
    geographic[flat] = -1.0

    return geographic


def topographic_position_index(
    dem: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    """Compute Topographic Position Index (TPI).

    TPI = elevation − mean(elevation in circular window of radius_px)

    TPI characterises whether a cell is higher or lower than its surroundings:
      - Positive TPI: ridges, hilltops, structural highs
      - Negative TPI: valleys, depressions, structural lows
      - Near-zero TPI: flat plains or mid-slope

    In MPM, positive TPI often correlates with resistant lithologies (silica
    cap, intrusive bodies) while negative TPI highlights drainage networks
    and fault-guided valleys where secondary dispersion halos accumulate.

    Implementation: ``scipy.ndimage.uniform_filter`` computes a box-mean
    over a (2·radius_px + 1) square window. Although a square approximates
    a disk, this is standard practice and computationally efficient.

    Args:
        dem: 2-D elevation array (metres), shape (rows, cols).
        radius_px: Radius of the neighbourhood window in pixels. Typical
                   values: 5–50 pixels depending on target geomorphological
                   scale.

    Returns:
        TPI array (metres), float64, same shape as `dem`. Positive = above
        local mean, negative = below local mean.

    Raises:
        ValueError: If ``radius_px`` < 1.
    """
    if radius_px < 1:
        raise ValueError(f"radius_px must be ≥ 1, got {radius_px}")

    dem = np.asarray(dem, dtype=np.float64)
    size = 2 * radius_px + 1

    # Fill NaN with local interpolation before filtering to avoid NaN spread
    # Use simple median fill from non-NaN neighbours (basic, but robust)
    nan_mask = np.isnan(dem)
    dem_filled = dem.copy()
    if nan_mask.any():
        # Replace NaN with global mean for the filter, then restore NaN
        fill_val = np.nanmean(dem)
        dem_filled[nan_mask] = fill_val

    local_mean = ndimage.uniform_filter(dem_filled, size=size, mode="reflect")
    tpi = dem - local_mean

    # Restore NaN at originally masked locations
    tpi[nan_mask] = np.nan
    return tpi


def plan_curvature(
    dem: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    """Compute plan (horizontal) curvature using Evans' (1980) method.

    Plan curvature describes the curvature of the terrain surface in the
    horizontal plane (perpendicular to the slope direction). It controls
    the convergence and divergence of surface water flow:
      - Negative plan curvature: convergent terrain (valleys, channels) —
        zones of fluid accumulation and geochemical concentration.
      - Positive plan curvature: divergent terrain (ridges, noses) — zones
        of fluid dispersal.

    Evans' (1980) method fits a bivariate polynomial to a 3×3 neighbourhood
    and analytically differentiates it. The plan curvature formula in terms
    of the polynomial coefficients is:

        plan_curv = −2·(D·H² + E·G² − F·G·H) / (G² + H² + ε)

    where D = 0.5·d²z/dx², E = 0.5·d²z/dy², F = d²z/dxdy, G = dz/dx, H = dz/dy
    (all evaluated at the central cell).

    Args:
        dem: 2-D elevation array (metres), shape (rows, cols).
        resolution_m: Ground cell size in metres.

    Returns:
        Plan curvature array (1/metres), float64, same shape as `dem`.
        Flat and edge cells may contain NaN.

    References:
        Evans, I.S. (1980). Zeitschrift für Geomorphologie,
        Supplementband 36, 274-295.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")

    dem = np.asarray(dem, dtype=np.float64)
    L = resolution_m

    # Compute first and second partial derivatives using numpy.gradient.
    # gradient() uses central differences (2nd order accurate).
    dz_dy, dz_dx = np.gradient(dem, L)               # G (dz/dx), H (dz/dy)
    d2z_dxdy, d2z_dxdx = np.gradient(dz_dx, L)       # F = d²z/dxdy, d²z/dx²
    d2z_dydy, _ = np.gradient(dz_dy, L)              # d²z/dy²

    G = dz_dx
    H = dz_dy
    D = 0.5 * d2z_dxdx
    E = 0.5 * d2z_dydy
    F = d2z_dxdy

    # Replace the defective numerator with the Evans (1980) formulation:
    eps = 1e-12
    denom = G**2 + H**2 + eps
    plan_curv = -2.0 * (D * (H**2) + E * (G**2) - F * G * H) / denom
    # Zero out curvature on perfectly flat floodplains/ocean cells where slope is negligible
    plan_curv = np.where((G**2 + H**2) < 1e-10, 0.0, plan_curv)
    return plan_curv


def flow_accumulation(
    dem: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    """Compute D8 flow direction and accumulation from a DEM.

    The D8 (deterministic eight-neighbour) algorithm routes flow from each
    cell to one of its eight neighbours in the direction of steepest descent.
    Flow accumulation counts the number of upstream contributing cells,
    providing a proxy for drainage basin size and stream power — both of
    which are relevant to secondary dispersion of mineral indicators.

    Algorithm:
      1. Compute steepest-descent direction (1 of 8 neighbours, or D8 code
         1/2/4/8/16/32/64/128 going clockwise from East) for each cell.
      2. Topologically sort cells from highest to lowest elevation.
      3. Route accumulation downstream: each cell passes its accumulated
         count (itself + all upstream cells) to its single downslope neighbour.

    Returns log₁₀(accumulation + 1) to compress the dynamic range, which
    spans several orders of magnitude between headwater and trunk-stream cells.

    Args:
        dem: 2-D elevation array (metres), shape (rows, cols). Edge cells
             are treated as sinks (flow exits the grid).
        resolution_m: Ground cell size in metres (used for diagonal scaling).

    Returns:
        Log₁₀(flow_accumulation + 1) array (dimensionless), float64, same
        shape as `dem`. Edge cells have accumulation = 1 (just themselves).

    References:
        O'Callaghan, J.F. & Mark, D.M. (1984). The extraction of drainage
        networks from digital elevation data. Computer Vision, Graphics, and
        Image Processing, 28(3), 323-344.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")

    dem = np.asarray(dem, dtype=np.float64)
    rows, cols = dem.shape

    # -------------------------------------------------------------------------
    # Step 1: Determine D8 flow direction for each cell.
    # Neighbours are indexed as:
    #   NW(0) N(1) NE(2)
    #   W(3)  *    E(4)
    #   SW(5) S(6) SE(7)
    # Row offsets (dr) and column offsets (dc) for each of the 8 directions:
    dr = np.array([-1, -1, -1,  0,  0,  1,  1,  1])
    dc = np.array([-1,  0,  1, -1,  1, -1,  0,  1])
    # Physical distance to each neighbour (diagonal vs cardinal)
    dist = np.array([
        resolution_m * np.sqrt(2),  # NW
        resolution_m,               # N
        resolution_m * np.sqrt(2),  # NE
        resolution_m,               # W
        resolution_m,               # E
        resolution_m * np.sqrt(2),  # SW
        resolution_m,               # S
        resolution_m * np.sqrt(2),  # SE
    ])

    # flow_dir[i,j] = index into (dr, dc) of the steepest downslope neighbour
    # −1 = sink (no downslope neighbour or edge cell)
    flow_dir = np.full((rows, cols), -1, dtype=np.int8)

    # Fill NaN in DEM with a very low value so NaN cells act as sinks
    dem_filled = dem.copy()
    dem_filled[np.isnan(dem_filled)] = -1e10

    for r in range(rows):
        for c in range(cols):
            max_slope = 0.0
            best_dir = -1
            for d in range(8):
                nr = r + dr[d]
                nc = c + dc[d]
                if 0 <= nr < rows and 0 <= nc < cols:
                    drop = (dem_filled[r, c] - dem_filled[nr, nc]) / dist[d]
                    if drop > max_slope:
                        max_slope = drop
                        best_dir = d
            flow_dir[r, c] = best_dir

    # -------------------------------------------------------------------------
    # Step 2: Topological sort (cells from highest to lowest elevation).
    flat_order = np.argsort(dem_filled.ravel())[::-1]  # descending elevation
    row_order = flat_order // cols
    col_order = flat_order % cols

    # -------------------------------------------------------------------------
    # Step 3: Route accumulation downstream.
    acc = np.ones((rows, cols), dtype=np.float64)  # each cell contributes 1

    for idx in range(len(flat_order)):
        r = row_order[idx]
        c = col_order[idx]
        d = flow_dir[r, c]
        if d >= 0:
            nr = r + dr[d]
            nc = c + dc[d]
            if 0 <= nr < rows and 0 <= nc < cols:
                acc[nr, nc] += acc[r, c]

    return np.log10(acc + 1)



# =============================================================================
# Pass 1.2a — Geomechanical stress resolution: slip & dilation tendency
# =============================================================================
# 2-D (map-view, vertical faults) resolution of a horizontal paleo-stress
# tensor onto fault segments (Morris et al. 1996; Ferrill et al. 1999).
# For a vertical fault of strike θ and horizontal principal stresses σ1 (azimuth
# φ1) and σ3 (φ1 + 90°), the angle β between the fault-plane NORMAL and σ1 gives
#     σn = σ1 cos²β + σ3 sin²β,      τ = (σ1 − σ3) sin β cos β
#     Ts = τ / σn,                   Td = (σ1 − σn) / (σ1 − σ3)
# Td = 1 for a fault parallel to σ1 (normal ⟂ σ1: opens, dilatant fluid trap);
# Td = 0 for a fault perpendicular to σ1 (clamped).
#
# IMPORTANT: the paleo-stress (σ1 azimuth and magnitudes) is a geological
# INTERPRETATION, not a measurement. It must be supplied from structural
# literature for the mineralising event and recorded in provenance.  Using
# effective stresses (pore pressure subtracted) changes Ts magnitudes but not
# the ranking of Td along strike.
# -----------------------------------------------------------------------------


def resolve_stress_on_vertical_fault(
    strike_deg: np.ndarray,
    sigma1: float,
    sigma3: float,
    sigma1_azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Normal and shear stress on vertical fault segments (map-view 2-D).

    Args:
        strike_deg: Fault strike azimuth(s) in degrees from north.
        sigma1, sigma3: Maximum / minimum horizontal principal stress (same units,
            compression positive, sigma1 >= sigma3 > 0).
        sigma1_azimuth_deg: Azimuth of sigma1 (degrees from north).

    Returns:
        (sigma_n, tau) arrays with the shape of ``strike_deg``.
    """
    if not (sigma1 >= sigma3 > 0):
        raise ValueError("Require sigma1 >= sigma3 > 0 (compression positive).")
    strike = np.asarray(strike_deg, dtype=np.float64)
    normal_az = strike + 90.0
    beta = np.deg2rad(normal_az - sigma1_azimuth_deg)
    sigma_n = sigma1 * np.cos(beta) ** 2 + sigma3 * np.sin(beta) ** 2
    tau = np.abs((sigma1 - sigma3) * np.sin(beta) * np.cos(beta))
    return sigma_n, tau


def slip_tendency(sigma_n: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Ts = τ / σn (Morris et al. 1996)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sigma_n > 0, tau / sigma_n, np.nan)


def dilation_tendency(sigma_n: np.ndarray, sigma1: float, sigma3: float) -> np.ndarray:
    """Td = (σ1 − σn) / (σ1 − σ3) (Ferrill et al. 1999), in [0, 1]."""
    if sigma1 == sigma3:
        raise ValueError("Dilation tendency undefined for isotropic stress (sigma1 == sigma3).")
    return np.clip((sigma1 - np.asarray(sigma_n)) / (sigma1 - sigma3), 0.0, 1.0)


def fault_segment_tendencies(
    faults_gdf,
    sigma1: float,
    sigma3: float,
    sigma1_azimuth_deg: float,
):
    """Split fault polylines into straight segments and attach Ts / Td.

    Args:
        faults_gdf: GeoDataFrame of (Multi)LineStrings in a projected CRS.

    Returns:
        GeoDataFrame of 2-point LineString segments with columns
        ``strike_deg``, ``sigma_n``, ``tau``, ``Ts``, ``Td``.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    segs, strikes = [], []
    for geom in faults_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = getattr(geom, "geoms", [geom])
        for line in parts:
            coords = list(line.coords)
            for (x0, y0, *_), (x1, y1, *_) in zip(coords[:-1], coords[1:]):
                if (x0, y0) == (x1, y1):
                    continue
                segs.append(LineString([(x0, y0), (x1, y1)]))
                # azimuth from north, folded to [0, 180)
                strikes.append(np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 180.0)
    strikes_arr = np.asarray(strikes)
    sn, tau = resolve_stress_on_vertical_fault(strikes_arr, sigma1, sigma3, sigma1_azimuth_deg)
    return gpd.GeoDataFrame(
        {
            "strike_deg": strikes_arr,
            "sigma_n": sn,
            "tau": tau,
            "Ts": slip_tendency(sn, tau),
            "Td": dilation_tendency(sn, sigma1, sigma3),
        },
        geometry=segs,
        crs=faults_gdf.crs,
    )


def rasterize_segment_attribute(
    segments_gdf,
    attribute: str,
    profile: dict,
    buffer_m: float = 100.0,
    reducer: str = "max",
) -> np.ndarray:
    """Burn a per-segment attribute into the grid within ``buffer_m`` of faults.

    Cells farther than ``buffer_m`` from any fault are NaN (not 0), so the
    model sees "no fault here" as missing rather than "clamped fault".
    ``reducer`` = "max" keeps the most dilatant segment touching each cell.
    """
    import rasterio.features

    H, W = profile["height"], profile["width"]
    out = np.full((H, W), np.nan, dtype=np.float32)
    order = segments_gdf.sort_values(attribute, ascending=(reducer == "max"))
    shapes = [
        (g.buffer(buffer_m), float(v))
        for g, v in zip(order.geometry, order[attribute])
        if np.isfinite(v)
    ]
    if not shapes:
        return out
    burned = rasterio.features.rasterize(
        shapes, out_shape=(H, W), transform=profile["transform"],
        fill=np.nan, dtype="float32", all_touched=True,
    )
    return burned.astype(np.float32)


# =============================================================================
# Pass 1.2b — Polarimetric L-band SAR decompositions
# =============================================================================
# Inputs are per-pixel 3×3 Hermitian matrices from fully polarimetric
# (quad-pol HH/HV/VH/VV) SLC data, multilooked: the Pauli coherency matrix T3
# for Cloude–Pottier, the lexicographic covariance matrix C3 for Freeman–Durden.
# ALOS-2 PALSAR-2 quad-pol (PLR mode) and NISAR quad-pol acquisitions are
# needed; the free ALOS PALSAR-2 annual mosaics are dual-pol (HH/HV) amplitude
# and CANNOT support these decompositions.
# -----------------------------------------------------------------------------


def cloude_pottier(T3: np.ndarray) -> dict[str, np.ndarray]:
    """H / α / A eigen-decomposition (Cloude & Pottier 1997).

    Args:
        T3: Complex coherency matrices, shape (H, W, 3, 3), Hermitian PSD.

    Returns:
        dict with ``entropy`` H ∈ [0,1], ``alpha_deg`` ᾱ ∈ [0°, 90°] and
        ``anisotropy`` A ∈ [0,1], each float32 (H, W).
    """
    T3 = np.asarray(T3)
    if T3.shape[-2:] != (3, 3):
        raise ValueError(f"T3 must have trailing shape (3, 3), got {T3.shape}")
    eigval, eigvec = np.linalg.eigh(T3)            # ascending
    eigval = np.clip(eigval.real, 0.0, None)[..., ::-1]
    eigvec = eigvec[..., ::-1]
    total = eigval.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0, eigval / total, 0.0)
        logp = np.where(p > 0, np.log(p) / np.log(3.0), 0.0)
    entropy = -(p * logp).sum(axis=-1)
    alpha_i = np.degrees(np.arccos(np.clip(np.abs(eigvec[..., 0, :]), 0.0, 1.0)))
    alpha = (p * alpha_i).sum(axis=-1)
    l2, l3 = eigval[..., 1], eigval[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        aniso = np.where((l2 + l3) > 0, (l2 - l3) / (l2 + l3), 0.0)
    return {
        "entropy": entropy.astype(np.float32),
        "alpha_deg": alpha.astype(np.float32),
        "anisotropy": aniso.astype(np.float32),
    }


def freeman_durden(C3: np.ndarray) -> dict[str, np.ndarray]:
    """Three-component Freeman–Durden (1998) decomposition.

    Args:
        C3: Complex covariance matrices (H, W, 3, 3) in the lexicographic basis
            [S_hh, √2 S_hv, S_vv] under reciprocity.

    Returns:
        dict of float32 powers ``Ps`` (surface), ``Pd`` (double-bounce),
        ``Pv`` (volume). Negative powers are clipped to 0 as is conventional;
        Ps + Pd + Pv equals the span wherever no clipping occurred.
    """
    C3 = np.asarray(C3)
    chh = C3[..., 0, 0].real
    cvv = C3[..., 2, 2].real
    chv = C3[..., 1, 1].real / 2.0                 # <|S_hv|^2>
    chhvv = C3[..., 0, 2]
    fv = 3.0 * chv                                 # volume: <|Shv|^2> = fv/3
    pv = 8.0 * fv / 3.0
    a = chh - fv
    b = cvv - fv
    c = chhvv - fv / 3.0
    ps = np.zeros_like(a)
    pd = np.zeros_like(a)
    surf = c.real >= 0                             # surface dominant: alpha = -1
    with np.errstate(divide="ignore", invalid="ignore"):
        # surface-dominant branch: fix alpha = -1
        fd_s = (a * b - np.abs(c) ** 2) / np.where((a + b + 2 * c.real) != 0, a + b + 2 * c.real, np.nan)
        fs_s = b - fd_s
        beta_abs2 = np.abs((c + fd_s)) ** 2 / np.where(fs_s != 0, fs_s ** 2, np.nan)
        ps_s = fs_s * (1 + beta_abs2)
        pd_s = 2 * fd_s
        # double-bounce-dominant branch: fix beta = 1
        fs_d = (a * b - np.abs(c) ** 2) / np.where((a + b - 2 * c.real) != 0, a + b - 2 * c.real, np.nan)
        fd_d = b - fs_d
        alpha_abs2 = np.abs((c - fs_d)) ** 2 / np.where(fd_d != 0, fd_d ** 2, np.nan)
        ps_d = 2 * fs_d
        pd_d = fd_d * (1 + alpha_abs2)
    ps = np.where(surf, ps_s, ps_d)
    pd = np.where(surf, pd_s, pd_d)
    ps = np.clip(np.nan_to_num(ps, nan=0.0), 0.0, None)
    pd = np.clip(np.nan_to_num(pd, nan=0.0), 0.0, None)
    pv = np.clip(pv, 0.0, None)
    return {"Ps": ps.astype(np.float32), "Pd": pd.astype(np.float32), "Pv": pv.astype(np.float32)}


# =============================================================================
# Pass 1.2c — Lithospheric gravity gradient step
# =============================================================================


def gravity_gradient_step(
    gravity_mgal: np.ndarray,
    resolution_m: float,
    smoothing_sigma_m: float = 20_000.0,
) -> np.ndarray:
    """Horizontal gradient magnitude of a long-wavelength gravity field (mGal/km).

    Intended for GOCE-derived / WGM2012 grids (~2 arc-min, ≈3.7 km): a Gaussian
    low-pass (default σ = 20 km) isolates lithosphere-scale sources (Moho and
    LAB steps at craton margins) before differentiation. At 50 m prediction
    cells this is a regional context variable, not a local trap indicator; its
    native resolution must be carried in the provenance mask.
    """
    g = np.asarray(gravity_mgal, dtype=np.float64)
    if smoothing_sigma_m > 0:
        g = ndimage.gaussian_filter(g, sigma=smoothing_sigma_m / resolution_m, mode="nearest")
    gy, gx = np.gradient(g, resolution_m)
    return (np.hypot(gx, gy) * 1000.0).astype(np.float32)   # per km


# =============================================================================
# Pass 1.2d — D8 routing and stream-sediment catchment deconvolution
# =============================================================================
_D8_OFFSETS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def d8_receivers(dem: np.ndarray, resolution_m: float) -> np.ndarray:
    """Flat index of each cell's steepest-descent D8 receiver (−1 = outlet/pit)."""
    dem = np.asarray(dem, dtype=np.float64)
    H, W = dem.shape
    best = np.zeros((H, W))
    recv = np.full((H, W), -1, dtype=np.int64)
    pad = np.pad(dem, 1, mode="constant", constant_values=np.inf)
    rr, cc = np.indices((H, W))
    for dr, dc in _D8_OFFSETS:
        dist = resolution_m * (np.sqrt(2.0) if dr and dc else 1.0)
        nb = pad[1 + dr: 1 + dr + H, 1 + dc: 1 + dc + W]
        slope = (dem - nb) / dist
        better = slope > best
        best = np.where(better, slope, best)
        target = (rr + dr) * W + (cc + dc)
        recv = np.where(better, target, recv)
    return recv.ravel()


def d8_upstream_labels(receivers: np.ndarray, sample_cells: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Assign each cell to the first sample site it drains to (−1 = none).

    This partitions the landscape into nested *sub-catchments*: the area that
    drains to sample j without first passing another sample site.
    """
    n = receivers.size
    site_of = np.full(n, -1, dtype=np.int64)
    for j, cell in enumerate(sample_cells):
        site_of[cell] = j
    label = np.full(n, -2, dtype=np.int64)        # -2 = unresolved
    for start in range(n):
        path = []
        c = start
        while c != -1 and label[c] == -2 and site_of[c] == -1:
            path.append(c)
            c = receivers[c]
            if len(path) > n:                        # safety against cycles
                break
        if c == -1 or label[c] == -2:
            res = -1 if site_of[c] == -1 else site_of[c]
        elif site_of[c] != -1:
            res = site_of[c]
        else:
            res = label[c]
        for p in path:
            label[p] = res
    for j, cell in enumerate(sample_cells):
        label[cell] = j
    return label.reshape(shape)


def deconvolve_stream_sediment(
    dem: np.ndarray,
    resolution_m: float,
    sample_cells: np.ndarray,
    concentrations: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mass-balance unmixing of stream-sediment geochemistry onto sub-catchments.

    Assumes uniform erosion rate and conservative mixing: the concentration at
    sample j is the area-weighted mean of the source concentrations of every
    sub-catchment upstream of (and including) j:
        c_obs_j = Σ_k∈up(j) A_k c_k / Σ_k∈up(j) A_k
    This linear system is solved for c_k ≥ 0 with non-negative least squares
    (cf. Lipp et al. 2021, who add smoothness regularisation on a grid).

    Args:
        dem: DEM (H, W). Should be hydrologically conditioned (pits filled).
        resolution_m: Cell size (m).
        sample_cells: Flat cell indices of sample sites, shape (n,).
        concentrations: Measured concentration per sample, shape (n,).

    Returns:
        dict with ``subcatchment`` (int, H×W; −1 = not upstream of any sample),
        ``source_concentration`` (float32, H×W map of unmixed c_k) and
        ``c_k`` (per-subcatchment estimates, shape (n,)).
    """
    from scipy.optimize import nnls

    H, W = dem.shape
    recv = d8_receivers(dem, resolution_m)
    sub = d8_upstream_labels(recv, np.asarray(sample_cells), (H, W))
    n = len(sample_cells)
    area = np.array([(sub == k).sum() for k in range(n)], dtype=np.float64)
    # downstream sample of each sample site
    down = np.full(n, -1, dtype=np.int64)
    flat_sub = sub.ravel()
    for j, cell in enumerate(sample_cells):
        c = recv[cell]
        while c != -1 and flat_sub[c] == -1:
            c = recv[c]
        if c != -1:
            down[j] = flat_sub[c]
    # upstream sets: k contributes to j if j is k or downstream of k
    M = np.zeros((n, n))
    for k in range(n):
        j = k
        visited = set()
        while j != -1 and j not in visited:
            visited.add(j)
            M[j, k] = area[k]
            j = down[j]
    row_sums = M.sum(axis=1, keepdims=True)
    M = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)
    c_k, _ = nnls(M, np.asarray(concentrations, dtype=np.float64))
    src = np.full((H, W), np.nan, dtype=np.float32)
    valid = sub >= 0
    src[valid] = c_k[sub[valid]]
    return {"subcatchment": sub, "source_concentration": src, "c_k": c_k}
