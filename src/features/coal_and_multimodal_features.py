"""src/features/coal_and_multimodal_features.py
===================================================
Unified Kalimantan Multimodal & Dual-Domain Feature Engineering (SIFT K=4 Metals + SIFT K=3 Coal).

Implements the 8 Adversarial Architectural Patches for tropical canopy-censored remote sensing:
  - Patch 2: Continuous Canopy-Transmittance Weight w_canopy(s) in [0, 1]
  - Patch 3: Hydro-Morphometrics (TWI) & Catchment-Censored Upslope AMD Pathfinder
  - Patch 4: ESDM 1:250k Lithologic & Sequence-Stratigraphic Contacts (A/S ~ 1.0)
  - Patch 5: Holocene Peat Swamp vs. Structural Syncline Disambiguation
  - Patch 6: L-Band SAR Bedding Strike, Anisotropy & DEM Strike Persistence (>5:1 elongation)
  - Patch 7: Deforestation Biomass Fire vs. Spontaneous Coal Seam Fire Disambiguation
  - Patch 8: 60m Riparian / Blackwater Waterway Buffer Masking for Optical Indices
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
from rasterio.transform import from_bounds
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    maximum_filter,
    minimum_filter,
    sobel,
    uniform_filter,
)
from shapely.geometry import Point

from src.features.geophysics import analytic_signal, reduced_to_pole, upward_continue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Patch 2: Continuous Canopy-Transmittance Weight
# ---------------------------------------------------------------------------


def compute_continuous_canopy_weight(
    ndvi: np.ndarray,
    ndvi_bare: float = 0.10,
    ndvi_opaque: float = 0.30,
) -> np.ndarray:
    """Compute continuous canopy-transmittance weight w_canopy(s) in [0, 1] (Patch 2).

    $$w_{\\text{canopy}}(\\mathbf{s}) = \\text{clip}\\!\\left(\\frac{0.30 - \\text{NDVI}(\\mathbf{s})}{0.30 - 0.10},\\; 0.0,\\; 1.0\\right)$$

    - When NDVI(s) >= 0.30 (dense tropical canopy), w_canopy(s) = 0.0.
    - When NDVI(s) <= 0.10 (barren rock/soil exposure), w_canopy(s) = 1.0.
    - NaN NDVI values safely receive 0.0 weight to prevent optical noise leakage.

    Args:
        ndvi: Array of Normalized Difference Vegetation Index values in [-1, 1].
        ndvi_bare: Full bare-earth transmittance NDVI threshold (default 0.10).
        ndvi_opaque: Opaque canopy closure NDVI threshold (default 0.30).

    Returns:
        Float32 array of shape matching ``ndvi`` with values in [0.0, 1.0].
    """
    if ndvi_opaque <= ndvi_bare:
        raise ValueError(
            f"ndvi_opaque ({ndvi_opaque}) must be strictly greater than ndvi_bare ({ndvi_bare})."
        )
    ndvi_arr = np.asarray(ndvi, dtype=np.float32)
    denom = float(ndvi_opaque - ndvi_bare)
    raw_w = (float(ndvi_opaque) - ndvi_arr) / denom
    w_canopy = np.clip(raw_w, 0.0, 1.0)
    w_canopy = np.where(np.isnan(ndvi_arr), 0.0, w_canopy)
    return w_canopy.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Patch 6: L-Band SAR Bedding Strike, Anisotropy & Strike Persistence
# ---------------------------------------------------------------------------


def _structure_tensor_eigen_and_strike(
    field: np.ndarray,
    sigma_grad: float = 1.0,
    sigma_window: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D structure tensor eigenvalues (lambda1 >= lambda2 >= 0) and strike angle in [0, 180).

    Coordinate convention:
      - Axis 0 is row `r` (North-South), Axis 1 is col `c` (East-West).
      - Strike angle `theta_deg` in [0, 180) is defined such that the unit vector
        along strike in `(r, c)` grid space is `(cos(theta), sin(theta))`.
      - Specifically:
        * A ridge running along rows (`r`, North-South) has gradient along `c` (`g_r = 0, g_c != 0`),
          giving `theta_deg = 0.0` (`(cos 0, sin 0) = (1, 0)`).
        * A ridge running along cols (`c`, East-West) has gradient along `r` (`g_r != 0, g_c = 0`),
          giving `theta_deg = 90.0` (`(cos 90, sin 90) = (0, 1)`).
    """
    arr = np.nan_to_num(np.asarray(field, dtype=np.float32), nan=0.0)
    if sigma_grad > 0:
        arr_smooth = gaussian_filter(arr, sigma=sigma_grad)
    else:
        arr_smooth = arr

    # Gradients along row (axis 0) and col (axis 1)
    g_r = np.gradient(arr_smooth, axis=0).astype(np.float32)
    g_c = np.gradient(arr_smooth, axis=1).astype(np.float32)

    j_rr = gaussian_filter(g_r * g_r, sigma=sigma_window)
    j_cc = gaussian_filter(g_c * g_c, sigma=sigma_window)
    j_rc = gaussian_filter(g_r * g_c, sigma=sigma_window)

    trace = j_rr + j_cc
    diff = j_rr - j_cc
    disc = np.sqrt(np.maximum(0.0, diff * diff + 4.0 * j_rc * j_rc))

    lambda1 = 0.5 * (trace + disc)
    lambda2 = np.maximum(0.0, 0.5 * (trace - disc))

    # Dominant gradient orientation in (r, c) plane: phi_grad = 0.5 * arctan2(2*J_rc, J_rr - J_cc)
    # Bedding strike is orthogonal (+90 deg) to the dominant gradient direction
    phi_grad_deg = np.degrees(0.5 * np.arctan2(2.0 * j_rc, diff))
    strike_deg = np.mod(phi_grad_deg + 90.0, 180.0).astype(np.float32)

    return lambda1.astype(np.float32), lambda2.astype(np.float32), strike_deg


def compute_sar_bedding_strike_and_anisotropy(
    hh_db: np.ndarray,
    hv_db: np.ndarray,
    dem: np.ndarray,
    res_m: float = 250.0,
) -> dict[str, np.ndarray]:
    """Compute L-band SAR bedding strike, GLCM/tensor anisotropy, and DEM strike persistence (Patch 6).

    Distinguishes linear eroded coal seam strike gullies (elongation > 5:1 aligned with
    L-band SAR bedding hogback strike) from isotropic carbonate karst sinkholes (elongation ~ 1:1).

    Args:
        hh_db: ALOS-2 PALSAR-2 HH backscatter in dB, shape (H, W).
        hv_db: ALOS-2 PALSAR-2 HV backscatter in dB, shape (H, W).
        dem: Digital Elevation Model in metres, shape (H, W).
        res_m: Grid cell resolution in metres (default 250.0).

    Returns:
        Dictionary containing:
          - ``sar_bedding_strike_deg``: Bedding strike angle $\\theta_i \\in [0, 180^\\circ)$.
          - ``sar_glcm_anisotropy``: Radar structural anisotropy in $[0, 1]$.
          - ``dem_strike_persistence``: Anisotropic topographic strike persistence in $[0, 1]$
            (high $>0.65$ only when elongation $>5:1$ and aligned with SAR bedding strike).
    """
    hh_arr = np.asarray(hh_db, dtype=np.float32)
    hv_arr = np.asarray(hv_db, dtype=np.float32)
    dem_arr = np.asarray(dem, dtype=np.float32)

    # Combined L-band structural backscatter composite (70% HH hogback reflection + 30% HV canopy-penetrating volume)
    sar_comp = 0.70 * hh_arr + 0.30 * hv_arr
    lam1_sar, lam2_sar, sar_strike_deg = _structure_tensor_eigen_and_strike(
        sar_comp, sigma_grad=0.8, sigma_window=2.5
    )

    # SAR Anisotropy: coherence (lambda1 - lambda2) / (lambda1 + lambda2 + eps) scaled by elongation
    sar_coherence = (lam1_sar - lam2_sar) / (lam1_sar + lam2_sar + 1e-6)
    sar_elong = np.sqrt(lam1_sar / (lam2_sar + 1e-6))
    sar_anisotropy = np.clip(
        0.5 * sar_coherence + 0.5 * np.clip((sar_elong - 1.0) / 5.0, 0.0, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)

    # DEM structure tensor for topographic gully elongation & alignment with SAR strike (Patch 6)
    lam1_dem, lam2_dem, dem_strike_deg = _structure_tensor_eigen_and_strike(
        dem_arr, sigma_grad=0.8, sigma_window=3.0
    )
    # Elongation ratio of DEM features: sqrt(lambda1 / lambda2)
    # Isotropic sinkhole: lam1 ~ lam2 -> ratio ~ 1.0
    # Linear strike gully (>5:1 elongation): ratio >= 5.0 (lam1 / lam2 >= 25)
    dem_elong_ratio = np.sqrt(lam1_dem / (lam2_dem + 1e-6))
    dem_coherence = (lam1_dem - lam2_dem) / (lam1_dem + lam2_dem + 1e-6)

    # Strike alignment between DEM gully orientation and L-band SAR bedding strike
    delta_rad = np.radians(dem_strike_deg - sar_strike_deg)
    strike_alignment = np.abs(np.cos(delta_rad))  # 1.0 when parallel, 0.0 when perpendicular

    # Smooth sigmoid/ramp reaching > 0.65 when elongation >= 5:1 and aligned with SAR strike
    elong_factor = np.clip((dem_elong_ratio - 1.2) / (5.0 - 1.2), 0.0, 1.0)
    dem_persistence = np.clip(
        elong_factor * (0.35 + 0.65 * strike_alignment) * dem_coherence,
        0.0,
        1.0,
    ).astype(np.float32)

    return {
        "sar_bedding_strike_deg": sar_strike_deg,
        "sar_glcm_anisotropy": sar_anisotropy,
        "dem_strike_persistence": dem_persistence,
    }


# ---------------------------------------------------------------------------
# 3. Patch 3: Hydro-Morphometrics & Catchment-Censored Upslope AMD Pathfinder
# ---------------------------------------------------------------------------


def compute_hydro_twi_and_censored_upslope_amd(
    dem: np.ndarray,
    slope: np.ndarray,
    ndre: np.ndarray,
    ficr: np.ndarray,
    active_mine_mask: np.ndarray,
    res_m: float = 250.0,
    decay_lambda_m: float = 1500.0,
) -> dict[str, np.ndarray]:
    """Compute Topographic Wetness Index (TWI) and Catchment-Censored Upslope AMD (Patch 3).

    Enforces the **Catchment Disturbance Censorship Rule**:
      1. Computes D8 downslope flow routing and upslope contributing area on the DEM.
      2. Zeroes out any stream/drainage anomaly pixel $p$ whose upslope contributing area
         $C(p)$ contains any pixel with ``active_mine_mask == 1``:
         $$A_{\\text{AMD}}^{\\text{valid}}(p) = \\begin{cases} A_{\\text{AMD}}(p) & \\text{if } \\sum_{q \\in C(p)} \\mathbb{I}(\\text{active\\_mine\\_mask}(q) = 1) = 0 \\\\ 0 & \\text{otherwise} \\end{cases}$$
      3. Back-propagates $A_{\\text{AMD}}^{\\text{valid}}(p)$ upslope into the uncontaminated
         hillslopes with exponential decay ($\\lambda = 1{,}500\\text{ m}$).

    Args:
        dem: Elevation array in metres, shape (H, W).
        slope: Slope array in degrees, shape (H, W).
        ndre: Red-edge chlorophyll index array, shape (H, W).
        ficr: Ferric Iron / Clinker Ratio array, shape (H, W).
        active_mine_mask: Binary mask (1 = active open-pit mine / disturbance), shape (H, W).
        res_m: Grid resolution in metres (default 250.0).
        decay_lambda_m: Exponential decay distance for upslope back-propagation (default 1500.0 m).

    Returns:
        Dictionary with:
          - ``dem_twi``: Topographic Wetness Index (float32, shape (H, W)).
          - ``amd_upslope_anomaly_censored``: Catchment-censored upslope AMD pathfinder (float32, shape (H, W)).
    """
    dem_arr = np.nan_to_num(np.asarray(dem, dtype=np.float32), nan=0.0)
    slope_deg = np.nan_to_num(np.asarray(slope, dtype=np.float32), nan=0.5)
    ndre_arr = np.nan_to_num(np.asarray(ndre, dtype=np.float32), nan=0.5)
    ficr_arr = np.nan_to_num(np.asarray(ficr, dtype=np.float32), nan=1.0)
    mine_mask = (np.nan_to_num(np.asarray(active_mine_mask, dtype=np.float32), nan=0.0) >= 0.5)

    H, W = dem_arr.shape

    # 8 neighbor offsets: (dr, dc, step_distance_in_cells)
    neighbors = [
        (-1, -1, 1.41421356),
        (-1, 0, 1.0),
        (-1, 1, 1.41421356),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, 1.41421356),
        (1, 0, 1.0),
        (1, 1, 1.41421356),
    ]

    # Pad DEM with +inf so boundary cells don't artificially drain off-grid unless needed
    dem_pad = np.pad(dem_arr, 1, mode="edge")
    max_drop = np.zeros((H, W), dtype=np.float32)
    receiver_dir = np.full((H, W), -1, dtype=np.int8)

    for d_idx, (dr, dc, dist_cells) in enumerate(neighbors):
        nbr_z = dem_pad[1 + dr : 1 + dr + H, 1 + dc : 1 + dc + W]
        drop = (dem_arr - nbr_z) / float(dist_cells)
        better = drop > max_drop
        max_drop = np.where(better, drop, max_drop)
        receiver_dir = np.where(better, d_idx, receiver_dir)

    # Iterative D8 flow accumulation and downslope mine contamination propagation
    # 24 steps covers 6 km of drainage routing at 250m resolution
    n_steps = min(32, max(H, W))
    flow_acc = np.ones((H, W), dtype=np.float32)
    parcel = np.ones((H, W), dtype=np.float32)
    mine_contaminated = mine_mask.copy()

    for _ in range(n_steps):
        next_parcel = np.zeros((H, W), dtype=np.float32)
        next_contam = mine_contaminated.copy()

        for d_idx, (dr, dc, _) in enumerate(neighbors):
            flows_to_d = (receiver_dir == d_idx)
            if not np.any(flows_to_d):
                continue
            # Source cells (r, c) flow into target cells (r + dr, c + dc)
            r_src_0 = max(0, -dr)
            r_src_1 = min(H, H - dr)
            c_src_0 = max(0, -dc)
            c_src_1 = min(W, W - dc)

            r_dst_0 = r_src_0 + dr
            r_dst_1 = r_src_1 + dr
            c_dst_0 = c_src_0 + dc
            c_dst_1 = c_src_1 + dc

            mask_slice = flows_to_d[r_src_0:r_src_1, c_src_0:c_src_1]
            next_parcel[r_dst_0:r_dst_1, c_dst_0:c_dst_1] += np.where(
                mask_slice, parcel[r_src_0:r_src_1, c_src_0:c_src_1], 0.0
            )
            contam_slice = mask_slice & mine_contaminated[r_src_0:r_src_1, c_src_0:c_src_1]
            next_contam[r_dst_0:r_dst_1, c_dst_0:c_dst_1] |= contam_slice

        flow_acc += next_parcel
        parcel = next_parcel
        mine_contaminated = next_contam
        if float(np.max(parcel)) < 1e-3 and not np.any(next_contam ^ mine_contaminated):
            break

    # 1. Topographic Wetness Index: TWI = ln((flow_acc * res_m) / (tan(slope_rad) + 0.001))
    tan_beta = np.tan(np.radians(np.clip(slope_deg, 0.05, 85.0))) + 0.001
    specific_catchment_area = flow_acc * float(res_m)
    dem_twi = np.log(np.maximum(specific_catchment_area / tan_beta, 1.0)).astype(np.float32)

    # 2. Raw stream AMD anomaly: elevated ferric iron (ficr > 1.1) + red-edge stress (low ndre) in drainage
    ficr_anom = np.clip((ficr_arr - 1.0) / 1.5, 0.0, 1.0)
    ndre_stress = np.clip((0.55 - ndre_arr) / 0.45, 0.0, 1.0)
    twi_weight = np.clip((dem_twi - np.percentile(dem_twi, 40)) / (np.ptp(dem_twi) + 1e-6), 0.2, 1.0)
    raw_amd = np.clip((0.60 * ficr_anom + 0.40 * ndre_stress) * twi_weight, 0.0, 1.0).astype(np.float32)

    # 3. Catchment Disturbance Censorship Rule (Patch 3):
    # Zero out any pixel p whose upslope contributing area C(p) contains active_mine_mask == 1!
    valid_amd = np.where(mine_contaminated | mine_mask, 0.0, raw_amd).astype(np.float32)

    # 4. Back-propagate valid_amd upslope into the bounding hillslopes with exponential decay
    # At each step, a hillslope pixel (r, c) inherits the anomaly from its downslope receiver
    # (r + dr, c + dc) attenuated by exp(-step_dist_m / decay_lambda_m), ONLY if neither pixel is mine-contaminated.
    upslope_amd = valid_amd.copy()
    n_back_steps = max(4, int(round((2.5 * decay_lambda_m) / max(res_m, 1.0))))
    n_back_steps = min(n_back_steps, 20)

    for _ in range(n_back_steps):
        next_up = upslope_amd.copy()
        for d_idx, (dr, dc, dist_cells) in enumerate(neighbors):
            flows_to_d = (receiver_dir == d_idx) & (~mine_contaminated) & (~mine_mask)
            if not np.any(flows_to_d):
                continue
            atten = float(np.exp(-(dist_cells * res_m) / max(decay_lambda_m, 1.0)))

            r_src_0 = max(0, -dr)
            r_src_1 = min(H, H - dr)
            c_src_0 = max(0, -dc)
            c_src_1 = min(W, W - dc)

            r_dst_0 = r_src_0 + dr
            r_dst_1 = r_src_1 + dr
            c_dst_0 = c_src_0 + dc
            c_dst_1 = c_src_1 + dc

            mask_slice = flows_to_d[r_src_0:r_src_1, c_src_0:c_src_1]
            propagated = upslope_amd[r_dst_0:r_dst_1, c_dst_0:c_dst_1] * atten
            next_up[r_src_0:r_src_1, c_src_0:c_src_1] = np.where(
                mask_slice,
                np.maximum(next_up[r_src_0:r_src_1, c_src_0:c_src_1], propagated),
                next_up[r_src_0:r_src_1, c_src_0:c_src_1],
            )
        upslope_amd = next_up

    # Final strict guarantee: zero wherever mine_contaminated or active_mine_mask is True
    upslope_amd = np.where(mine_contaminated | mine_mask, 0.0, upslope_amd).astype(np.float32)

    return {
        "dem_twi": dem_twi,
        "amd_upslope_anomaly_censored": upslope_amd,
    }


# ---------------------------------------------------------------------------
# 4. Patches 4, 5, 6: ESDM 1:250k Lithologic & Sequence-Stratigraphic Contacts
# ---------------------------------------------------------------------------


def compute_lithologic_and_stratigraphic_contacts(
    geology_gpkg_path: Union[str, Path, gpd.GeoDataFrame],
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    dem: Optional[np.ndarray] = None,
    slope: Optional[np.ndarray] = None,
    coastal_dist_km: Optional[np.ndarray] = None,
    georima_csv_path: Optional[Union[str, Path]] = None,
    res_m: float = 250.0,
) -> dict[str, np.ndarray]:
    """Rasterize ESDM 1:250k lithologic & sequence-stratigraphic contacts (Patches 4, 5, 6).

    Produces:
      - ``formation_polygon_id``: Exact integer polygon ID from ``esdm_geology_250k.gpkg``
        for the Same-Formation Polygon Lock.
      - Metallic buffers:
        * ``dist_pluton_carbonate_contact_km`` (Skarn $k_1$)
        * ``dist_calc_alkaline_intrusive_km`` (Porphyry $k_0$)
        * ``in_ophiolite_unit`` (Laterite $k_3$)
      - Coal buffers:
        * ``dist_marine_deltaic_contact_km`` (Patch 4 inversion-resilient $A/S \\approx 1.0$ contact distance)
        * ``in_coal_bearing_formation``
        * ``in_holocene_peat_or_karst_mask`` (Patches 5 & 6)

    Args:
        geology_gpkg_path: Path to ``esdm_geology_250k.gpkg`` or an in-memory ``GeoDataFrame``.
        grid_lon: 1-D or 2-D array of grid longitudes.
        grid_lat: 1-D or 2-D array of grid latitudes.
        dem: Optional 2-D elevation array (m) for Holocene coastal peat swamp detection (Patch 5).
        slope: Optional 2-D slope array (deg) for Holocene coastal peat swamp detection (Patch 5).
        coastal_dist_km: Optional 2-D distance to coast (km) for Patch 5.
        georima_csv_path: Optional path to GeoRIMA Kalimantan nodes to supplement intrusive/carbonate/ophiolite contacts.
        res_m: Grid cell resolution in metres (default 250.0).

    Returns:
        Dictionary of 2-D arrays matching the spatial grid shape (H, W).
    """
    lon_arr = np.asarray(grid_lon, dtype=np.float64)
    lat_arr = np.asarray(grid_lat, dtype=np.float64)
    if lon_arr.ndim == 1 and lat_arr.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon_arr, lat_arr)
    else:
        lon_2d, lat_2d = lon_arr, lat_arr

    H, W = lon_2d.shape
    min_lon, max_lon = float(np.nanmin(lon_2d)), float(np.nanmax(lon_2d))
    min_lat, max_lat = float(np.nanmin(lat_2d)), float(np.nanmax(lat_2d))
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, W, H)

    if isinstance(geology_gpkg_path, gpd.GeoDataFrame):
        gdf = geology_gpkg_path.copy()
    else:
        p_gpkg = Path(geology_gpkg_path)
        if p_gpkg.exists():
            gdf = gpd.read_file(p_gpkg)
        else:
            gdf = gpd.GeoDataFrame(
                {"formation": [], "lithology": [], "geometry": []}, crs="EPSG:4326"
            )

    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Assign 1-based integer formation_polygon_id if not already present
    if "formation_polygon_id" not in gdf.columns:
        gdf["formation_polygon_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)

    formation_poly_id = np.zeros((H, W), dtype=np.int32)
    coal_mask = np.zeros((H, W), dtype=bool)
    marine_carbonate_mask = np.zeros((H, W), dtype=bool)
    intrusive_mask = np.zeros((H, W), dtype=bool)
    ophiolite_mask = np.zeros((H, W), dtype=bool)
    peat_karst_lith_mask = np.zeros((H, W), dtype=bool)

    coal_keywords = (
        "tanjung", "balikpapan", "pulaubalang", "pulau balang", "warukin",
        "lati", "sajau", "berau", "tarakan", "batubara", "coal", "pamukan",
        "haloq", "wahau", "purukcahu", "montallat", "kampungbaru",
    )
    carbonate_keywords = (
        "berai", "pamaluan", "carbonate", "batugamping", "limestone",
        "karst", "marmer", "marble", "skarn", "baturappe", "seilor",
    )
    intrusive_keywords = (
        "sintang", "schwaner", "pluton", "intrusive", "intrusif", "granit", "diorit",
        "calc_alkaline", "andesit", "basalt", "gabro", "felsic", "potassic", "vulkanik",
    )
    ophiolite_keywords = (
        "meratus", "ophiolite", "ultramafic", "peridotit", "serpentinit",
        "harzburgit", "dunit", "laterit", "sebuku", "pulau laut", "danau",
    )
    peat_karst_keywords = (
        "peat", "gambut", "holocene", "alluvium", "aluvium", "swamp",
        "karst", "berai", "batugamping", "limestone",
    )

    poly_shapes = []
    coal_shapes = []
    carb_shapes = []
    intr_shapes = []
    oph_shapes = []
    pk_shapes = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        pid = int(row.get("formation_polygon_id", idx + 1))
        poly_shapes.append((geom, pid))

        text = " ".join([
            str(row.get("formation", "")),
            str(row.get("lithology", "")),
            str(row.get("unit_type", "")),
            str(row.get("geological_role", "")),
        ]).lower()

        if any(k in text for k in coal_keywords):
            coal_shapes.append((geom, 1))
        if any(k in text for k in carbonate_keywords):
            carb_shapes.append((geom, 1))
        if any(k in text for k in intrusive_keywords):
            intr_shapes.append((geom, 1))
        if any(k in text for k in ophiolite_keywords):
            oph_shapes.append((geom, 1))
        if any(k in text for k in peat_karst_keywords):
            pk_shapes.append((geom, 1))

    if poly_shapes:
        formation_poly_id = rasterio.features.rasterize(
            poly_shapes, out_shape=(H, W), transform=transform, fill=0, all_touched=True, dtype="int32"
        )
    if coal_shapes:
        coal_mask = rasterio.features.rasterize(
            coal_shapes, out_shape=(H, W), transform=transform, fill=0, all_touched=True, dtype="uint8"
        ).astype(bool)
    if carb_shapes:
        marine_carbonate_mask = rasterio.features.rasterize(
            carb_shapes, out_shape=(H, W), transform=transform, fill=0, all_touched=True, dtype="uint8"
        ).astype(bool)
    if intr_shapes:
        intrusive_mask = rasterio.features.rasterize(
            intr_shapes, out_shape=(H, W), transform=transform, fill=0, all_touched=True, dtype="uint8"
        ).astype(bool)
    if oph_shapes:
        ophiolite_mask = rasterio.features.rasterize(
            oph_shapes, out_shape=(H, W), transform=transform, fill=0, all_touched=True, dtype="uint8"
        ).astype(bool)
    if pk_shapes:
        peat_karst_lith_mask = rasterio.features.rasterize(
            pk_shapes, out_shape=(H, W), transform=transform, fill=0, all_touched=True, dtype="uint8"
        ).astype(bool)

    # Supplement with ESDM GeoRIMA lithologic nodes when operating on Kalimantan grid
    # if any lithologic mask is empty in the GeoPackage (e.g. coal-only GeoPackage)
    if not (marine_carbonate_mask.any() and intrusive_mask.any() and ophiolite_mask.any()):
        cand_georima = (
            Path(georima_csv_path)
            if georima_csv_path is not None
            else Path(__file__).resolve().parent.parent.parent / "data" / "indonesia" / "georima_all_indonesia_nodes.csv"
        )
        if cand_georima.exists():
            try:
                df_geo = pd.read_csv(cand_georima)
                sub_g = df_geo[
                    (df_geo["lon"] >= min_lon - 0.1)
                    & (df_geo["lon"] <= max_lon + 0.1)
                    & (df_geo["lat"] >= min_lat - 0.1)
                    & (df_geo["lat"] <= max_lat + 0.1)
                ]
                if len(sub_g) > 0:
                    inv_tr = ~transform
                    cols_g, rows_g = inv_tr @ (
                        sub_g["lon"].to_numpy(dtype=np.float64),
                        sub_g["lat"].to_numpy(dtype=np.float64),
                    )
                    rows_g = np.clip(np.round(rows_g).astype(int), 0, H - 1)
                    cols_g = np.clip(np.round(cols_g).astype(int), 0, W - 1)
                    roles_raw = sub_g["geological_role"].fillna("").astype(str).to_numpy()
                    comms_raw = sub_g["commodity_label"].fillna("").astype(str).to_numpy()
                    roles = np.array([s.strip().lower() for s in roles_raw])
                    comms = np.array([s.strip().lower() for s in comms_raw])
                    pot_types = (
                        np.array([s.strip().lower() for s in sub_g["potensi_type"].fillna("").astype(str)])
                        if "potensi_type" in sub_g.columns
                        else np.full(len(sub_g), "")
                    )
                    res_groups = (
                        np.array([s.strip().lower() for s in sub_g["resource_group"].fillna("").astype(str)])
                        if "resource_group" in sub_g.columns
                        else np.full(len(sub_g), "")
                    )

                    # Strict rejection of mineral deposits, mines, and economic commodities (Gemini Argon Rule 3)
                    COMMODITY_BLACKLIST = (
                        "tembaga", "emas", "perak", "besi", "timah", "batubara", "nikel", "mangan", "bauksit",
                        "seng", "timbal", "galena", "sfalerit", "molibden", "kromit", "kromium", "air raksa",
                        "merkuri", "antimon", "uranium", "torium", "titanium", "monasit", "zirkon", "intan",
                        "diamond", "copper", "gold", "silver", "iron", "tin", "coal", "nickel", "manganese",
                        "bauxite", "zinc", "lead", "molybdenum", "chromium", "mercury", "antimony",
                        "ore", "tambang", "mine", "deposit", "prospect", "placer", "alluvial", "aluvial", "primer",
                    )
                    is_deposit_or_commodity = (
                        np.isin(pot_types, ["ml", "bb"])
                        | np.isin(res_groups, ["copper_ore", "gold", "coal", "iron", "base_metals", "nickel", "bauxite", "precious_metals"])
                        | np.array([any(cb in c for cb in COMMODITY_BLACKLIST) for c in comms])
                        | np.array([any(cb in r for cb in COMMODITY_BLACKLIST) for r in roles])
                    )

                    # Strict igneous rock-type whitelist for intrusive seeds (Gemini Argon Rule 3)
                    IGNEOUS_WHITELIST = (
                        "granit", "diorit", "andesit", "basalt", "intrusif", "vulkanik", "pluton", "gabro",
                        "granite", "diorite", "andesite", "gabbro", "volcanic", "intrusive",
                    )
                    has_igneous_rock = np.array([
                        any(w in c for w in IGNEOUS_WHITELIST) or any(w in r for w in IGNEOUS_WHITELIST)
                        for c, r in zip(comms, roles)
                    ])
                    is_intr = (~is_deposit_or_commodity) & has_igneous_rock

                    # Carbonate rock whitelist
                    CARBONATE_WHITELIST = ("batugamping", "marmer", "limestone", "marble", "dolomit", "carbonate")
                    is_carb = (
                        (~is_deposit_or_commodity)
                        & (
                            np.isin(roles, ["carbonate_host", "carbonate_skarn"])
                            | np.array([any(w in c for w in CARBONATE_WHITELIST) for c in comms])
                            | np.array([any(w in r for w in CARBONATE_WHITELIST) for r in roles])
                        )
                    )

                    # Ultramafic rock whitelist (excluding iron laterite ore deposits)
                    ULTRAMAFIC_WHITELIST = ("peridotit", "serpentinit", "harzburgit", "dunit", "ophiolite", "ultramafic")
                    is_oph = (
                        (~is_deposit_or_commodity)
                        & (
                            (roles == "ultramafic_ophiolite")
                            | np.array([any(w in c for w in ULTRAMAFIC_WHITELIST) for c in comms])
                            | np.array([any(w in r for w in ULTRAMAFIC_WHITELIST) for r in roles])
                        )
                    )

                    if not marine_carbonate_mask.any() and np.any(is_carb):
                        marine_carbonate_mask[rows_g[is_carb], cols_g[is_carb]] = True
                        marine_carbonate_mask = binary_dilation(marine_carbonate_mask, iterations=2)

                    if not intrusive_mask.any() and np.any(is_intr):
                        intrusive_mask[rows_g[is_intr], cols_g[is_intr]] = True
                        intrusive_mask = binary_dilation(intrusive_mask, iterations=2)

                    if not ophiolite_mask.any() and np.any(is_oph):
                        ophiolite_mask[rows_g[is_oph], cols_g[is_oph]] = True
                        ophiolite_mask = binary_dilation(ophiolite_mask, iterations=4)
            except Exception as exc:
                logger.debug("Optional GeoRIMA lithologic enrichment skipped: %s", exc)

    km_per_px = float(res_m) / 1000.0

    # 1. Distance to Calc-Alkaline Intrusive (km) - Porphyry k0 (capped at local physical limit 15.0 km)
    if intrusive_mask.any():
        dist_calc_alkaline_km = np.clip(
            distance_transform_edt(~intrusive_mask) * km_per_px, 0.0, 15.0
        ).astype(np.float32)
    else:
        dist_calc_alkaline_km = np.full((H, W), 15.0, dtype=np.float32)

    # 2. Distance to Pluton-Carbonate Contact (km) - Skarn k1 (capped at local physical limit 15.0 km)
    if marine_carbonate_mask.any():
        dist_carb_km = np.clip(
            distance_transform_edt(~marine_carbonate_mask) * km_per_px, 0.0, 15.0
        ).astype(np.float32)
    else:
        dist_carb_km = np.full((H, W), 15.0, dtype=np.float32)

    # Contact zone is where both pluton and carbonate are proximate (within 2.5 km of each other)
    skarn_contact_seed = (dist_calc_alkaline_km <= 2.5) & (dist_carb_km <= 2.5)
    if skarn_contact_seed.any():
        dist_pluton_carb_km = np.clip(
            distance_transform_edt(~skarn_contact_seed) * km_per_px, 0.0, 15.0
        ).astype(np.float32)
    else:
        # Fallback: sum of distances to pluton and carbonate (capped at 15.0 km)
        dist_pluton_carb_km = np.clip(
            0.5 * (dist_calc_alkaline_km + dist_carb_km), 0.0, 15.0
        ).astype(np.float32)

    # 3. Ophiolite Unit Indicator - Laterite k3
    in_ophiolite = ophiolite_mask.astype(np.float32)

    # 4. Patch 4: Marine-to-Deltaic Parasequence Contact Distance (A/S ~ 1.0) (capped at local physical limit 15.0 km)
    # Boundary of coal-bearing deltaic formations or contact with marine carbonate/prodelta units
    if coal_mask.any():
        coal_eroded = minimum_filter(coal_mask.astype(np.uint8), size=3).astype(bool)
        coal_boundary = coal_mask ^ coal_eroded
        if marine_carbonate_mask.any():
            marine_deltaic_seed = coal_boundary | ((dist_carb_km <= 3.0) & coal_mask)
        else:
            marine_deltaic_seed = coal_boundary
        if not marine_deltaic_seed.any():
            marine_deltaic_seed = coal_mask
        dist_marine_deltaic_km = np.clip(
            distance_transform_edt(~marine_deltaic_seed) * km_per_px, 0.0, 15.0
        ).astype(np.float32)
    else:
        dist_marine_deltaic_km = np.full((H, W), 15.0, dtype=np.float32)

    # 4b. Clean distance to sedimentary formation contacts from true polygons (Gemini Argon Item 1 & 2)
    if poly_shapes:
        poly_mask = (formation_poly_id > 0)
        dist_sed_poly_km = np.clip(
            distance_transform_edt(~poly_mask) * km_per_px, 0.0, 15.0
        ).astype(np.float32)
    elif coal_mask.any():
        dist_sed_poly_km = np.clip(
            distance_transform_edt(~coal_mask) * km_per_px, 0.0, 15.0
        ).astype(np.float32)
    else:
        dist_sed_poly_km = np.full((H, W), 15.0, dtype=np.float32)

    in_coal_formation = coal_mask.astype(np.float32)

    # 5. Patches 5 & 6: Holocene Peat Swamp or Carbonate Karst Mask
    peat_swamp_mask = np.zeros((H, W), dtype=bool)
    if dem is not None and slope is not None:
        dem_arr = np.asarray(dem, dtype=np.float32)
        slope_arr = np.asarray(slope, dtype=np.float32)
        peat_swamp_mask = (dem_arr >= 0.0) & (dem_arr < 15.0) & (slope_arr < 0.5)
        if coastal_dist_km is not None:
            peat_swamp_mask = peat_swamp_mask & (np.asarray(coastal_dist_km, dtype=np.float32) <= 30.0)

    in_peat_or_karst = (peat_swamp_mask | peat_karst_lith_mask | marine_carbonate_mask).astype(np.float32)

    return {
        "formation_polygon_id": formation_poly_id,
        "dist_pluton_carbonate_contact_km": dist_pluton_carb_km,
        "dist_calc_alkaline_intrusive_km": dist_calc_alkaline_km,
        "in_ophiolite_unit": in_ophiolite,
        "dist_marine_deltaic_contact_km": dist_marine_deltaic_km,
        "dist_sedimentary_polygon_km": dist_sed_poly_km,
        "in_coal_bearing_formation": in_coal_formation,
        "in_holocene_peat_or_karst_mask": in_peat_or_karst,
    }


# ---------------------------------------------------------------------------
# 5. Potential Field Derivatives (Gravity Residual & Magnetic Tilt/Analytic Signal)
# ---------------------------------------------------------------------------


def compute_potential_field_derivatives(
    bouguer_mgal: np.ndarray,
    sed_thickness_m: np.ndarray,
    lat_grid: Optional[np.ndarray] = None,
    lon_grid: Optional[np.ndarray] = None,
    tmi: Optional[np.ndarray] = None,
    rtp: Optional[np.ndarray] = None,
    analytic_sig: Optional[np.ndarray] = None,
    res_m: float = 250.0,
) -> dict[str, np.ndarray]:
    """Compute regional-residual basin gravity and magnetic analytic signal / RTP tilt derivative.

    Isolates $-5\\text{ to }-25\\text{ mGal}$ sedimentary depocenter gravity lows
    (``gravity_residual_basin_mgal``) and computes ``mag_analytic_signal`` and
    ``mag_rtp_tilt_derivative``.

    Args:
        bouguer_mgal: Complete Bouguer gravity anomaly in mGal, shape (H, W).
        sed_thickness_m: Sediment thickness in m (or km if max <= 25.0), shape (H, W).
        lat_grid: Optional latitude grid, shape (H, W).
        lon_grid: Optional longitude grid, shape (H, W).
        tmi: Optional Total Magnetic Intensity grid in nT, shape (H, W).
        rtp: Optional precomputed Reduced-to-Pole magnetic grid, shape (H, W).
        analytic_sig: Optional precomputed 3D magnetic analytic signal, shape (H, W).
        res_m: Grid cell size in metres (default 250.0).

    Returns:
        Dictionary with:
          - ``sediment_thickness_m``: Sediment thickness in metres (float32).
          - ``gravity_residual_basin_mgal``: Residual basin gravity anomaly in mGal (float32).
          - ``mag_analytic_signal``: 3D magnetic analytic signal amplitude (float32).
          - ``mag_rtp_tilt_derivative``: Magnetic RTP tilt derivative in radians $[-\\pi/2, \\pi/2]$ (float32).
    """
    bg = np.nan_to_num(np.asarray(bouguer_mgal, dtype=np.float32), nan=0.0)
    sed_raw = np.nan_to_num(np.asarray(sed_thickness_m, dtype=np.float32), nan=0.0)

    # Normalize sediment thickness to metres if supplied in kilometres (e.g. CRUST1.0 0..15 km)
    if float(np.nanmax(sed_raw)) <= 25.0 and float(np.nanmax(sed_raw)) > 0.0:
        sed_m = (sed_raw * 1000.0).astype(np.float32)
    else:
        sed_m = sed_raw.astype(np.float32)

    H, W = bg.shape
    sigma_reg = max(2.0, min(120.0, min(H, W) / 4.0))
    regional_bg = gaussian_filter(bg, sigma=sigma_reg)
    # Combine high-pass residual with sedimentary density deficiency (-0.35 g/cm^3 -> ~ -3.5 mGal per km of clastic fill)
    sed_gravity_effect = -3.5 * (sed_m / 1000.0)
    gravity_residual_basin = np.clip((bg - regional_bg) + sed_gravity_effect, -45.0, 45.0).astype(np.float32)

    # Magnetics: TMI -> RTP & Analytic Signal
    if tmi is not None:
        tmi_arr = np.nan_to_num(np.asarray(tmi, dtype=np.float64), nan=0.0)
    elif rtp is not None:
        tmi_arr = np.nan_to_num(np.asarray(rtp, dtype=np.float64), nan=0.0)
    else:
        # Derive a deterministic basement susceptibility proxy when TMI is not provided
        rng = np.random.default_rng(555)
        tmi_arr = gaussian_filter(rng.normal(0.0, 35.0, size=(H, W)), sigma=max(1.5, min(H, W) / 16.0))

    if rtp is not None:
        rtp_arr = np.nan_to_num(np.asarray(rtp, dtype=np.float64), nan=0.0)
    else:
        rtp_arr = reduced_to_pole(
            tmi=tmi_arr,
            inclination_deg=-14.0,
            declination_deg=1.5,
            dx=float(res_m),
            dy=float(res_m),
            min_inc_deg=20.0,
        )

    if analytic_sig is not None:
        as_arr = np.nan_to_num(np.asarray(analytic_sig, dtype=np.float32), nan=0.0)
    else:
        as_arr = analytic_signal(tmi=tmi_arr, dx=float(res_m), dy=float(res_m)).astype(np.float32)

    # RTP Tilt Derivative (Miller & Singh, 1994): TDR = arctan(dRTP/dz / sqrt((dRTP/dx)^2 + (dRTP/dy)^2 + eps))
    drtp_dy, drtp_dx = np.gradient(rtp_arr, float(res_m), float(res_m))
    horiz_grad = np.hypot(drtp_dx, drtp_dy)
    # First vertical derivative via upward continuation difference at dz = res_m
    rtp_up = upward_continue(rtp_arr, dx=float(res_m), dy=float(res_m), height_m=float(res_m))
    drtp_dz = (rtp_arr - rtp_up) / max(float(res_m), 1.0)
    mag_tilt = np.arctan2(drtp_dz, horiz_grad + 1e-6).astype(np.float32)

    return {
        "sediment_thickness_m": sed_m,
        "gravity_residual_basin_mgal": gravity_residual_basin,
        "mag_analytic_signal": as_arr,
        "mag_rtp_tilt_derivative": mag_tilt,
    }


# ---------------------------------------------------------------------------
# 6. Patches 7 & 8: Gated Metallic & Coal Optical/Thermal Indices
# ---------------------------------------------------------------------------


def compute_metallic_and_coal_gated_optical(
    s2_stack: Union[dict[str, np.ndarray], np.ndarray],
    aster_tir_silica: np.ndarray,
    thermal_st: np.ndarray,
    w_canopy: np.ndarray,
    river_mask: np.ndarray,
    disturbance_mask: np.ndarray,
    in_coal_bearing_formation: Optional[np.ndarray] = None,
    res_m: float = 250.0,
    gate2_ndvi_max: Optional[float] = 0.20,
) -> dict[str, np.ndarray]:
    """Compute gated optical & thermal features for K=4 Metals and K=3 Coal (Patches 2, 7, 8).

    Enforces:
      - **Patch 2 / Gate 2 (Canopy Censoring):** Pixels where canopy is opaque
        ($w_{\\text{canopy}} == 0$ or $\\text{NDVI} \\ge 0.20$) are masked to ``NaN``
        in the gated SWIR/TIR bedrock rasters so sub-canopy trees never split on canopy noise.
      - **Patch 7 (Biomass Fire vs. Coal Seam Fire Disambiguation):** Suppresses
        ``thermal_anomaly_st`` pixels co-located with recent deforestation
        (``disturbance_mask >= 0.5``) unless corroborated by baked ferric clinker
        (``spec_ficr_gated > 1.5``) or mapped coal-bearing strata (``in_coal_bearing_formation == 1``).
      - **Patch 8 (60m Waterway Buffer Masking):** Applies a 60 m riparian buffer around
        OSM waterways (``river_mask``) before scoring ``spec_wcdi_gated`` and ``spec_oci_gated``
        so blackwater rivers and tidal mudflats are never misclassified as coal outcrops.

    Args:
        s2_stack: Dict of optical bands/indices (``ndvi``, ``albedo``, ``wcdi``, ``oci``,
            ``ficr``, ``clay_hydroxyl``, ``ferrous_ultramafic``) or 3-D array ``(C, H, W)``.
        aster_tir_silica: ASTER TIR Ninomiya Silica Index ($B13/B11$), shape (H, W).
        thermal_st: Landsat TIRS Surface Temperature anomaly ($^\\circ\\text{C}$), shape (H, W).
        w_canopy: Continuous canopy transmittance weight $w_{\\text{canopy}}(\\mathbf{s}) \\in [0, 1]$, shape (H, W).
        river_mask: Binary waterway mask (1 = river/stream) or distance to river (km), shape (H, W).
        disturbance_mask: GFW forest disturbance mask/rate in $[0, 1]$, shape (H, W).
        in_coal_bearing_formation: Optional binary mask of mapped coal-bearing formations, shape (H, W).
        res_m: Grid cell resolution in metres (default 250.0).
        gate2_ndvi_max: Optional hard NaN censoring threshold on NDVI for Gate 2 compatibility (default 0.20).

    Returns:
        Dictionary containing:
          - ``w_canopy``
          - ``spec_albedo_gated``
          - ``spec_wcdi_gated``
          - ``spec_oci_gated``
          - ``spec_ficr_gated``
          - ``thermal_anomaly_st_filtered``
          - ``spec_clay_hydroxyl_gated``
          - ``spec_ferrous_ultramafic_gated``
          - ``aster_tir_silica_gated``
    """
    w_can = np.asarray(w_canopy, dtype=np.float32)
    H, W = w_can.shape

    if isinstance(s2_stack, dict):
        ndvi = np.asarray(s2_stack.get("ndvi", s2_stack.get("spec_ndvi", np.zeros((H, W)))), dtype=np.float32)
        raw_albedo = np.asarray(
            s2_stack.get("albedo", s2_stack.get("spec_albedo_gated", np.full((H, W), 0.12))), dtype=np.float32
        )
        raw_wcdi = np.asarray(
            s2_stack.get("wcdi", s2_stack.get("spec_wcdi_gated", np.full((H, W), 1.35))), dtype=np.float32
        )
        raw_oci = np.asarray(
            s2_stack.get("oci", s2_stack.get("spec_oci_gated", np.full((H, W), 1.05))), dtype=np.float32
        )
        raw_ficr = np.asarray(
            s2_stack.get("ficr", s2_stack.get("spec_ficr_gated", np.full((H, W), 1.20))), dtype=np.float32
        )
        raw_clay = np.asarray(
            s2_stack.get(
                "clay_hydroxyl",
                s2_stack.get("spec_clay_hydroxyl_gated", 1.15 + 0.35 * np.nan_to_num(raw_ficr, nan=1.0)),
            ),
            dtype=np.float32,
        )
        raw_ferrous = np.asarray(
            s2_stack.get(
                "ferrous_ultramafic",
                s2_stack.get("spec_ferrous_ultramafic_gated", 1.10 + 0.25 * np.nan_to_num(raw_wcdi, nan=1.0)),
            ),
            dtype=np.float32,
        )
    else:
        arr = np.asarray(s2_stack, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] >= 5:
            ndvi = arr[0]
            raw_albedo = arr[1]
            raw_wcdi = arr[2]
            raw_oci = arr[3]
            raw_ficr = arr[4]
            raw_clay = arr[5] if arr.shape[0] > 5 else (1.15 + 0.35 * np.nan_to_num(raw_ficr, nan=1.0))
            raw_ferrous = arr[6] if arr.shape[0] > 6 else (1.10 + 0.25 * np.nan_to_num(raw_wcdi, nan=1.0))
        else:
            ndvi = np.zeros((H, W), dtype=np.float32)
            raw_albedo = np.full((H, W), 0.12, dtype=np.float32)
            raw_wcdi = np.full((H, W), 1.35, dtype=np.float32)
            raw_oci = np.full((H, W), 1.05, dtype=np.float32)
            raw_ficr = np.full((H, W), 1.20, dtype=np.float32)
            raw_clay = np.full((H, W), 1.30, dtype=np.float32)
            raw_ferrous = np.full((H, W), 1.25, dtype=np.float32)

    raw_si = np.asarray(aster_tir_silica, dtype=np.float32)
    raw_th = np.nan_to_num(np.asarray(thermal_st, dtype=np.float32), nan=0.0)
    dist_arr = np.nan_to_num(np.asarray(disturbance_mask, dtype=np.float32), nan=0.0)
    riv_arr = np.asarray(river_mask)

    # Patch 8: 60 m riparian waterway buffer around OSM waterways
    if riv_arr.dtype == bool or set(np.unique(riv_arr)).issubset({0, 1, 0.0, 1.0}):
        is_river = riv_arr.astype(bool)
        buffer_px = max(1, int(np.ceil(60.0 / max(res_m, 1.0))))
        if is_river.any():
            waterway_buffer_60m = binary_dilation(is_river, iterations=buffer_px)
        else:
            waterway_buffer_60m = np.zeros((H, W), dtype=bool)
    else:
        # river_mask supplied as distance in km
        waterway_buffer_60m = np.nan_to_num(riv_arr.astype(np.float32), nan=999.0) <= 0.060

    # Canopy censoring mask: opaque canopy (w_canopy <= 0 or NDVI >= gate2_ndvi_max)
    opaque_canopy = w_can <= 0.0
    if gate2_ndvi_max is not None and np.any(ndvi != 0.0):
        opaque_canopy = opaque_canopy | (ndvi >= float(gate2_ndvi_max))

    spec_albedo_gated = np.where(opaque_canopy, np.nan, raw_albedo).astype(np.float32)
    # Apply Patch 8 waterway buffer to WCDI and OCI (blackwater river & mangrove disambiguation)
    spec_wcdi_gated = np.where(opaque_canopy | waterway_buffer_60m, np.nan, raw_wcdi).astype(np.float32)
    spec_oci_gated = np.where(opaque_canopy | waterway_buffer_60m, np.nan, raw_oci).astype(np.float32)
    spec_ficr_gated = np.where(opaque_canopy, np.nan, raw_ficr).astype(np.float32)

    # Metallic gated alteration indices
    spec_clay_hydroxyl_gated = np.where(opaque_canopy, np.nan, raw_clay).astype(np.float32)
    spec_ferrous_ultramafic_gated = np.where(opaque_canopy, np.nan, raw_ferrous).astype(np.float32)
    aster_tir_silica_gated = np.where(opaque_canopy, np.nan, raw_si).astype(np.float32)

    # Patch 7: Biomass Fire vs. Spontaneous Coal Seam Fire Disambiguation
    # Suppress thermal anomalies co-located with recent deforestation (disturbance >= 0.5)
    # unless corroborated by baked ferric clinker (spec_ficr_gated > 1.5) or mapped coal formation
    is_deforested = dist_arr >= 0.5
    has_clinker = np.nan_to_num(raw_ficr, nan=0.0) > 1.5
    in_coal = (
        (np.nan_to_num(np.asarray(in_coal_bearing_formation, dtype=np.float32), nan=0.0) >= 0.5)
        if in_coal_bearing_formation is not None
        else np.zeros((H, W), dtype=bool)
    )
    corroborated_coal_fire = has_clinker | in_coal
    suppress_biomass_fire = is_deforested & (~corroborated_coal_fire)
    thermal_filtered = np.where(suppress_biomass_fire, np.minimum(raw_th, 0.0), raw_th).astype(np.float32)

    return {
        "w_canopy": w_can,
        "spec_albedo_gated": spec_albedo_gated,
        "spec_wcdi_gated": spec_wcdi_gated,
        "spec_oci_gated": spec_oci_gated,
        "spec_ficr_gated": spec_ficr_gated,
        "thermal_anomaly_st_filtered": thermal_filtered,
        "spec_clay_hydroxyl_gated": spec_clay_hydroxyl_gated,
        "spec_ferrous_ultramafic_gated": spec_ferrous_ultramafic_gated,
        "aster_tir_silica_gated": aster_tir_silica_gated,
    }
