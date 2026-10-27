"""Unit tests for Phase 1: Physics, Gravity Reduction & Label Hygiene (Passes 1 & 2).

Verifies the 6 Reconciled Engineering Rules & Phase 1 Execution Spec:
  1. Global Hygiene Guard & Emas Alluvial Exclusion from P_k
  2. K=4 Multi-Mineral Target Definitions ({1, -1}) & Alteration Benchmark
  3. Bouguer Slab Reduction & 30 km Low-Pass Moho Inversion
  4. Multiscale Crustal Worms & Merdith Multi-Scale Contrasts
  5. Petrophysical Host Gates (Intrusive Arc & Ultramafic Plateau) & No Raw Pixel Leakage
  6. Pooled Logistics Propensity & N x 4 Exclusion Mask M (Isotropic & Anisotropic)
  7. Asymmetric Sample Weighting (No Double-Division / Unexplored Jungle Penalization)
  8. OUTMOGC Closed-Boundary Gate with Exponential Mahalanobis Decay
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from src.features.geophysics import (
    bouguer_slab_reduction,
    compute_canopy_safe_spectral_indices,
    compute_intrusive_host_gate,
    compute_merdith_contrasts,
    compute_ultramafic_plateau_gate,
    moho_depth_from_bouguer,
    multiscale_crustal_worms,
)
from src.labels.borneo_labels import (
    build_k4_deposit_targets,
    filter_valid_ore_candidates,
    load_borneo_vector_sources,
    rasterize_borneo_k4_labels,
    spatial_join_borneo_k4,
)
from src.labels.pu_negatives import (
    OUTMOGCGate,
    compute_asymmetric_head_weights,
    compute_k4_exclusion_mask,
    fit_pooled_exploration_propensity,
)
from src.models.pu_pipeline import verify_no_raw_pixel_leakage


# ---------------------------------------------------------------------------
# 1. Global Hygiene Guard & Placer Exclusion Tests
# ---------------------------------------------------------------------------


def test_global_hygiene_guard_excludes_alluvial_and_artifacts():
    """Verify that 'Emas Alluvial' and is_centroid_artifact == True are strictly excluded."""
    df_raw = pd.DataFrame({
        "commodity_label": ["Emas Primer", "Emas Alluvial", "Tembaga", "Kaolin", "Emas Primer"],
        "is_centroid_artifact": [False, False, True, False, False],
        "geological_role": ["ml", "alluvial_cover", "ml", "argillic_alteration", "ml"],
        "lon": [115.0, 114.0, 113.0, 112.0, 115.5],
        "lat": [-0.5, -1.0, 0.0, 1.0, -0.2],
    })

    cleaned = filter_valid_ore_candidates(df_raw)
    assert len(cleaned) == 3
    assert "Emas Alluvial" not in cleaned["commodity_label"].values
    assert not cleaned["is_centroid_artifact"].any()
    assert set(cleaned["commodity_label"].values) == {"Emas Primer", "Kaolin"}


def test_spatial_join_never_promotes_alluvial_placers_to_positive_targets():
    """Verify spatial join never labels cells matching alluvial gold as positive in P_k."""
    # Source vectors with Kelian epithermal gold, an alluvial gold site, and an ESDM concession
    import geopandas as gpd

    gdf_k0 = gpd.GeoDataFrame({"geometry": [Point(111.0, 0.0)]}, crs="EPSG:4326")
    gdf_k1 = gpd.GeoDataFrame({"geometry": [Point(112.0, 0.0)]}, crs="EPSG:4326")
    gdf_k2 = gpd.GeoDataFrame({"geometry": [Point(115.45, -0.02)]}, crs="EPSG:4326")  # Kelian bedrock
    gdf_k3 = gpd.GeoDataFrame({"geometry": [Point(116.4, -3.5)]}, crs="EPSG:4326")
    gdf_alt = gpd.GeoDataFrame({"geometry": [Point(112.5, 0.5)]}, crs="EPSG:4326")

    # Explored footprint includes all above plus an alluvial river placer at (114.5, -1.5)
    gdf_footprint = gpd.GeoDataFrame(
        {"geometry": [Point(115.45, -0.02), Point(114.5, -1.5)]}, crs="EPSG:4326"
    )

    targets = {
        "positives_k0": gdf_k0,
        "positives_k1": gdf_k1,
        "positives_k2": gdf_k2,
        "positives_k3": gdf_k3,
        "alteration_benchmark": gdf_alt,
        "explored_footprint": gdf_footprint,
    }

    grid = pd.DataFrame({
        "lon": [115.45, 114.5, 110.0],
        "lat": [-0.02, -1.5, 0.0],
    })

    joined = spatial_join_borneo_k4(grid, targets, match_radius_m=1000.0)

    # Kelian bedrock cell (index 0) is P2=1 and explored=1
    assert joined.loc[0, "label_k2_epithermal_au"] == 1
    assert joined.loc[0, "explored_footprint"] == 1

    # Alluvial placer cell (index 1) MUST NOT be marked 1 in any K=4 target, but IS explored
    assert joined.loc[1, "label_k0_porphyry_cu"] == -1
    assert joined.loc[1, "label_k1_skarn_fe_cu"] == -1
    assert joined.loc[1, "label_k2_epithermal_au"] == -1
    assert joined.loc[1, "label_k3_laterite_fe_ni"] == -1
    assert joined.loc[1, "explored_footprint"] == 1

    # Remote jungle cell (index 2) is unlabeled (-1) and unexplored (0)
    assert joined.loc[2, "label_k2_epithermal_au"] == -1
    assert joined.loc[2, "explored_footprint"] == 0


# ---------------------------------------------------------------------------
# 2. Borneo Sources & K=4 Target Definitions
# ---------------------------------------------------------------------------


def test_borneo_k4_deposit_targets_integration():
    """Verify loading real Borneo vector sources and building K=4 targets."""
    sources = load_borneo_vector_sources()
    assert len(sources["georima"]) > 2000
    assert len(sources["iron"]) == 79
    assert len(sources["mrds"]) == 50

    targets = build_k4_deposit_targets(
        sources["georima"], sources["iron"], sources["mrds"], sources["coal"]
    )

    # Head k=0: Porphyry Cu-Au (N ≈ 35-45)
    assert 30 <= len(targets["positives_k0"]) <= 55

    # Head k=1: Contact Fe-Cu Skarn (ESDM 57 + GeoRIMA 14 = 71)
    assert 68 <= len(targets["positives_k1"]) <= 75

    # Head k=2: Epithermal Au-Ag (GeoRIMA 87 + MRDS primary)
    assert 85 <= len(targets["positives_k2"]) <= 100

    # Head k=3: Lateritic Fe-Ni (ESDM 21 + GeoRIMA 8 = 29)
    assert 25 <= len(targets["positives_k3"]) <= 35

    # Alteration benchmark (exact strings: argillic, advanced_argillic, propylitic_zeolite, Kaolin)
    assert 40 <= len(targets["alteration_benchmark"]) <= 55

    # Exploration footprint (pooled human footprint N ≈ 1,800 to 2,600)
    assert len(targets["explored_footprint"]) >= 1500


def test_rasterize_borneo_k4_labels():
    """Verify rasterizing K=4 target sets into 2-D arrays."""
    from rasterio.transform import from_bounds

    sources = load_borneo_vector_sources()
    targets = build_k4_deposit_targets(
        sources["georima"], sources["iron"], sources["mrds"], sources["coal"]
    )

    profile = {
        "height": 50,
        "width": 50,
        "transform": from_bounds(115.0, -1.0, 116.0, 0.0, 50, 50),
        "crs": "EPSG:4326",
    }
    rasters = rasterize_borneo_k4_labels(targets, profile, buffer_m=1000.0)

    for k in (0, 1, 2, 3):
        col = f"label_k{k}_" + ["porphyry_cu", "skarn_fe_cu", "epithermal_au", "laterite_fe_ni"][k]
        assert col in rasters
        assert rasters[col].shape == (50, 50)
        assert set(np.unique(rasters[col])).issubset({-1, 1})

    assert "alteration_benchmark" in rasters
    assert set(np.unique(rasters["alteration_benchmark"])).issubset({0, 1})
    assert "explored_footprint" in rasters
    assert set(np.unique(rasters["explored_footprint"])).issubset({0, 1})


# ---------------------------------------------------------------------------
# 3. Bouguer Slab Reduction & Moho Inversion (Rule 6)
# ---------------------------------------------------------------------------


def test_bouguer_slab_reduction_physics():
    """Verify Bouguer slab conversion: Delta g_B = Delta g_FA - (0.1119 * max(0, dem))."""
    free_air = np.array([[50.0, 20.0], [10.0, -5.0]])
    dem = np.array([[1000.0, 500.0], [0.0, -200.0]])  # oceanic bathymetry (-200 m) is not reduced

    bouguer = bouguer_slab_reduction(free_air, dem)
    assert np.isclose(bouguer[0, 0], 50.0 - 0.1119 * 1000.0)  # 50 - 111.9 = -61.9 mGal
    assert np.isclose(bouguer[0, 1], 20.0 - 0.1119 * 500.0)   # 20 - 55.95 = -35.95 mGal
    assert np.isclose(bouguer[1, 0], 10.0)                     # 10 - 0 = 10 mGal
    assert np.isclose(bouguer[1, 1], -5.0)                    # bathymetry untouched -> -5 mGal


def test_moho_depth_from_bouguer_filter():
    """Verify 30 km low-pass Bouguer slab Moho inversion replaces fake DEM elevation formula."""
    H, W = 128, 128
    res_m = 1000.0
    # Center anomaly of +50 mGal (dense oceanic mantle) vs background of -50 mGal (thick continental crust)
    x = np.linspace(-60, 60, W)
    y = np.linspace(-60, 60, H)
    xx, yy = np.meshgrid(x, y)
    r = np.hypot(xx, yy)
    bg = np.where(r < 20.0, 50.0, -50.0)

    moho = moho_depth_from_bouguer(bg, pixel_size_m=res_m)
    assert moho.shape == (H, W)
    # Bounded to [18.0, 45.0] km
    assert np.all((moho >= 18.0) & (moho <= 45.0))
    # High Bouguer (center) -> shallower Moho (thinner crust, e.g. oceanic ~20-25 km)
    # Low Bouguer (periphery) -> deeper Moho (thick continental keel ~35-40 km)
    assert moho[H // 2, W // 2] < moho[0, 0]


# ---------------------------------------------------------------------------
# 4. Multi-Scale Crustal Worms & Merdith Contrasts
# ---------------------------------------------------------------------------


def test_multiscale_crustal_worms():
    """Verify upward continued horizontal gradient worms at 5 km, 15 km, 30 km."""
    H, W = 64, 64
    x = np.linspace(-30000, 30000, W)
    xx, _ = np.meshgrid(x, x)
    bg = 30.0 * np.tanh(xx / 5000.0)

    worms = multiscale_crustal_worms(bg, dx=1000.0, dy=1000.0, depths_m=[5000.0, 15000.0, 30000.0])
    assert "bouguer_worm_5km" in worms
    assert "bouguer_worm_15km" in worms
    assert "bouguer_worm_30km" in worms

    # Deeper upward continuation attenuates gradient amplitude
    assert np.nanmax(worms["bouguer_worm_5km"]) > np.nanmax(worms["bouguer_worm_15km"])
    assert np.nanmax(worms["bouguer_worm_15km"]) > np.nanmax(worms["bouguer_worm_30km"])


def test_merdith_contrasts_1km_mean_and_4km_ratio():
    """Verify 1 km local mean and 4 km 80th/20th percentile contrast ratio."""
    rng = np.random.default_rng(123)
    field = np.abs(rng.normal(loc=100.0, scale=30.0, size=(50, 50)))
    contrasts = compute_merdith_contrasts(field, pixel_size_m=500.0, prefix="mag")

    assert "geophys_mean_1km_mag" in contrasts
    assert "geophys_contrast_4km_mag" in contrasts
    assert contrasts["geophys_mean_1km_mag"].shape == (50, 50)
    assert contrasts["geophys_contrast_4km_mag"].shape == (50, 50)
    assert np.all(contrasts["geophys_contrast_4km_mag"] >= 1.0)


# ---------------------------------------------------------------------------
# 5. Canopy-Safe Host Gates & No Raw Pixel Leakage (Rule 1)
# ---------------------------------------------------------------------------


def test_host_gates_physics():
    """Verify intrusive host prior (k=0,1,2) and ultramafic plateau prior (k=3)."""
    H, W = 32, 32
    # Intrusive prior: analytic signal texture * contrast
    as_grid = np.random.uniform(5.0, 50.0, (H, W))
    contrast_mag = np.random.uniform(1.2, 3.5, (H, W))
    intrusive_prior = compute_intrusive_host_gate(as_grid, contrast_mag)
    assert intrusive_prior.shape == (H, W)
    assert np.all(intrusive_prior >= 0.0)

    # Ultramafic plateau prior: flat plateau (slope 2°) + high Bouguer vs steep mountain (slope 25°)
    slope = np.array([[2.0, 25.0], [1.0, 30.0]])
    bouguer_15km = np.array([[40.0, -10.0], [50.0, -20.0]])
    plateau_prior = compute_ultramafic_plateau_gate(slope, bouguer_15km)
    assert plateau_prior[0, 0] > plateau_prior[0, 1]
    assert np.all((plateau_prior >= 0.0) & (plateau_prior <= 1.0))


def test_verify_no_raw_pixel_leakage_with_b11_b13_s2_b():
    """Verify raw band guard rejects b11, b13, aster_b, s2_b while accepting physics names."""
    safe_features = [
        "swir_veg_norm_clay_idx",
        "swir_argillic_rbd_logical",
        "swir_phyllic_rbd_logical",
        "tir_quartz_silicification_idx",
        "geophys_mean_1km_mag",
        "geophys_contrast_4km_mag",
        "bouguer_worm_15km",
    ]
    assert verify_no_raw_pixel_leakage(safe_features) is True

    # Test rejection of forbidden patterns
    for bad in ["s2_b11", "b11_clay", "aster_b13", "raw_band_3", "s2_b8a"]:
        with pytest.raises(ValueError, match="Raw sensor pixel leakage"):
            verify_no_raw_pixel_leakage(["swir_veg_norm_clay_idx", bad])


# ---------------------------------------------------------------------------
# 6. Pooled Logistics Propensity & Exclusion Mask M (Rule 5)
# ---------------------------------------------------------------------------


def test_pooled_exploration_propensity_and_bounds():
    """Verify pooled propensity bounds e(s) in [0.05, 1.0]."""
    rng = np.random.default_rng(42)
    N = 250
    X_acc = rng.uniform(0.0, 100.0, size=(N, 4))
    explored = (rng.uniform(size=N) > 0.6).astype(int)

    e_s = fit_pooled_exploration_propensity(X_acc, explored, min_propensity=0.05)
    assert len(e_s) == N
    assert np.all(e_s >= 0.05)
    assert np.all(e_s <= 1.0)


def test_k4_exclusion_mask_isotropic_and_anisotropic():
    """Verify N x 4 exclusion mask M: r0,r2=6km, r1=3.5km, r3=10x2km ellipse."""
    # 3 points:
    # 0: At positive point P3 (should be masked for unlabeled)
    # 1: 5 km along-strike of P3 (inside 10 km along-strike -> masked M[1, 3] = 0)
    # 2: 5 km across-strike of P3 (outside 2 km across-strike -> NOT masked M[2, 3] = 1)
    p3_lon, p3_lat = 116.0, -3.0
    mean_lat = -3.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(mean_lat))
    m_per_deg_lat = 110570.0

    # Along strike 45°: dx = dy = 5000 / sqrt(2)
    d_along = 5000.0 / np.sqrt(2.0)
    pt1_lon = p3_lon + (d_along / m_per_deg_lon)
    pt1_lat = p3_lat + (d_along / m_per_deg_lat)

    # Across strike -45°: dx = -5000 / sqrt(2), dy = 5000 / sqrt(2)
    pt2_lon = p3_lon - (d_along / m_per_deg_lon)
    pt2_lat = p3_lat + (d_along / m_per_deg_lat)

    coords = np.array([
        [p3_lon, p3_lat],
        [pt1_lon, pt1_lat],
        [pt2_lon, pt2_lat],
    ])

    pos_k = {
        0: np.array([[110.0, 0.0]]),
        1: np.array([[110.0, 0.0]]),
        2: np.array([[110.0, 0.0]]),
        3: np.array([[p3_lon, p3_lat]]),
    }

    labels_k4 = np.full((3, 4), -1)  # all unlabeled
    M = compute_k4_exclusion_mask(coords, pos_k, labels_k4=labels_k4, is_geographic=True)

    assert M.shape == (3, 4)
    # Point 0 is at P3 -> M[0, 3] = 0
    assert M[0, 3] == 0.0
    # Point 1 is 5 km along-strike (inside 10 km semi-major axis) -> M[1, 3] = 0
    assert M[1, 3] == 0.0
    # Point 2 is 5 km across-strike (outside 2 km semi-minor axis) -> M[2, 3] = 1
    assert M[2, 3] == 1.0


# ---------------------------------------------------------------------------
# 7. Asymmetric Sample Weighting (Rule 5)
# ---------------------------------------------------------------------------


def test_asymmetric_head_weights_no_unexplored_jungle_penalty():
    """Verify asymmetric sample weights: w[pos]=1/e(s), w[unl]=M*(0.25 + 0.75*e(s))."""
    labels_k = np.array([1, 1, -1, -1, -1])
    # Cell 0: Road positive (e=0.8)
    # Cell 1: Deep jungle positive (e=0.05)
    # Cell 2: Road unlabeled (e=0.8, M=1)
    # Cell 3: Deep jungle unlabeled (e=0.05, M=1)
    # Cell 4: Negative halo unlabeled (e=0.8, M=0)
    e_logistics = np.array([0.80, 0.05, 0.80, 0.05, 0.80])
    exclusion_mask_k = np.array([1.0, 1.0, 1.0, 1.0, 0.0])

    w = compute_asymmetric_head_weights(labels_k, e_logistics, exclusion_mask_k)

    # Positives: inverse propensity
    assert np.isclose(w[0], 1.0 / 0.80)   # 1.25
    assert np.isclose(w[1], 1.0 / 0.05)   # 20.0

    # Unlabeled road vs jungle:
    # Road unlabeled: 1.0 * (0.25 + 0.75 * 0.80) = 0.85
    # Jungle unlabeled: 1.0 * (0.25 + 0.75 * 0.05) = 0.2875
    # Jungle unlabeled weight is LOWER than road unlabeled (exploration-confidence weighted!)
    # Proves deep jungle is NEVER made 20x more negative than roads!
    assert np.isclose(w[2], 0.85)
    assert np.isclose(w[3], 0.2875)
    assert w[2] > w[3]

    # Masked negative halo cell (M=0) receives 0 weight
    assert np.isclose(w[4], 0.0)


# ---------------------------------------------------------------------------
# 8. OUTMOGC Closed-Boundary Gate
# ---------------------------------------------------------------------------


def test_outmogc_closed_boundary_gate():
    """Verify OUTMOGC closed-boundary gate: Gamma=1 inside 95% ellipsoid, exp decay outside."""
    rng = np.random.default_rng(42)
    # Positive deposits clustered around (10, 10, ...)
    X_pos = rng.normal(loc=10.0, scale=1.5, size=(40, 8))

    gate = OUTMOGCGate(n_pca_components=4, n_gmm_components=2, random_state=42)
    gate.fit(X_pos)

    # Test instances:
    # Inside points (from positive distribution)
    X_in = rng.normal(loc=10.0, scale=1.0, size=(5, 8))
    # Extreme outside outliers
    X_out = rng.normal(loc=-100.0, scale=1.0, size=(5, 8))

    gamma_in = gate.score(X_in)
    gamma_out = gate.score(X_out)

    assert gamma_in.shape == (5,)
    assert gamma_out.shape == (5,)
    assert np.all((gamma_in >= 0.5) & (gamma_in <= 1.0))
    # Extreme outliers have near-zero Gamma
    assert np.all(gamma_out < 1e-6)
