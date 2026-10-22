"""Unit and verification tests for the 7-Pass Mineral Prospectivity & PU-Learning Pipeline.

Covers all 7 passes and the 4 explicit Acceptance Criteria:
  1. No Raw Pixel Leakage
  2. Propensity Stability (e(x_i) >= 0.05)
  3. nnPU Non-Negativity Clamping & Gradient Reversal
  4. Spatial Block Independence for Elkan-Noto c_hat and Conformal Quantiles
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from rasterio.transform import from_bounds
from shapely.geometry import LineString

from src.evaluation.metrics import (
    spatial_conformal_calibrate,
    spatial_conformal_bounds,
)
from src.features.aster_indices import (
    al_oh_crystallinity,
    carbonate_wavelength_shift,
    compute_hyperspectral_indices,
    fe_oxide_intensity,
    kaolinite_alunite_doublet,
)
from src.features.geophysics import (
    aem_depth_slice_conductance,
    invert_regolith_thickness_from_aem,
)
from src.features.structural import (
    cloude_pottier,
    deconvolve_stream_sediment,
    dilation_tendency,
    fault_segment_tendencies,
    freeman_durden,
    gravity_gradient_step,
    resolve_stress_on_vertical_fault,
    slip_tendency,
)
from src.harmonization.cover_mask import update_cover_mask_from_aem
from src.labels.pu_negatives import (
    compute_inverse_propensity_weights,
    fit_exploration_propensity,
)
from src.models.pu_pipeline import (
    blend_predictions_by_pr_auc,
    calibrate_elkan_noto_posterior,
    compute_ensemble_weights,
    estimate_elkan_noto_c,
    fit_pu_model,
    verify_no_raw_pixel_leakage,
)
from src.models.unet_model import MPMUNet, nnPULoss
from src.outputs.prospectivity_map import (
    emit_site_cards_json,
    polygonize_and_rank_targets,
    save_conformal_bounds_geotiff,
)


# ---------------------------------------------------------------------------
# Pass 1: Sensor & Physics Extraction Tests
# ---------------------------------------------------------------------------


def test_pass1_1_hyperspectral_indices():
    """Verify Pass 1.1 hyperspectral SWIR/VNIR feature extraction."""
    # Construct synthetic hyperspectral cube from 750 nm to 2450 nm (5 nm step = 341 bands)
    wavelengths = np.linspace(750.0, 2450.0, 341)
    H, W = 8, 8
    cube = np.full((len(wavelengths), H, W), 0.35, dtype=np.float64)

    # Inject sharp 2200 nm illite absorption + shallow 1900 nm water feature in top-left
    idx_1900 = int(np.argmin(np.abs(wavelengths - 1900.0)))
    idx_2165 = int(np.argmin(np.abs(wavelengths - 2165.0)))
    idx_2200 = int(np.argmin(np.abs(wavelengths - 2200.0)))
    idx_2320 = int(np.argmin(np.abs(wavelengths - 2320.0)))  # Mg-carbonate shifted from 2340 nm
    idx_900 = int(np.argmin(np.abs(wavelengths - 900.0)))

    cube[idx_1900, :4, :4] = 0.30  # shallow 1900 nm trough (depth ~0.14)
    cube[idx_1900, 4:, 4:] = 0.20  # deeper 1900 nm trough in background (smectitic)
    cube[idx_2200, :4, :4] = 0.12  # deep sharp 2200 nm trough
    cube[idx_2200, 4:, 4:] = 0.30  # weak 2200 nm trough in background
    cube[idx_2165, :4, :4] = 0.18  # secondary 2165 nm doublet trough
    cube[idx_2320, :4, :4] = 0.15  # carbonate trough at ~2320 nm (shift = 2340 - 2320 = +20 nm)
    cube[idx_900, :4, :4] = 0.14   # Fe3+ crystal-field absorption at 900 nm
    cube[idx_900, 4:, 4:] = 0.32

    out = compute_hyperspectral_indices(cube, wavelengths)
    assert "hs_aloh_crystallinity" in out
    assert "hs_kaol_alun_doublet_depth" in out
    assert "hs_carbonate_shift_nm" in out
    assert "hs_fe_oxide_intensity" in out
    assert out["hs_aloh_crystallinity"][0, 0] > out["hs_aloh_crystallinity"][6, 6]
    assert out["hs_kaol_alun_doublet_depth"][0, 0] > 0.05
    assert out["hs_carbonate_shift_nm"][0, 0] > 10.0  # positive shift (2340 - 2320 = +20 nm)
    assert out["hs_fe_oxide_intensity"][0, 0] > out["hs_fe_oxide_intensity"][6, 6]


def test_pass1_2_geomechanics_sar_and_deconvolution():
    """Verify Pass 1.2 stress tensor (Ts, Td), SAR decompositions, gravity step, and D8 deconvolution."""
    import geopandas as gpd

    # 1. Geomechanical stress resolution: sigma1 = 100 MPa at 090° (E-W compression), sigma3 = 40 MPa
    sigma_n, tau = resolve_stress_on_vertical_fault(
        strike_deg=np.array([90.0, 0.0, 60.0]),
        sigma1=100.0,
        sigma3=40.0,
        sigma1_azimuth_deg=90.0,
    )
    td = dilation_tendency(sigma_n, sigma1=100.0, sigma3=40.0)
    ts = slip_tendency(sigma_n, tau)

    assert np.isclose(td[0], 1.0, atol=1e-6)  # strike parallel to sigma1 -> max dilation
    assert np.isclose(td[1], 0.0, atol=1e-6)  # strike perpendicular to sigma1 -> min dilation
    assert ts[2] > 0.3                        # oblique 30° fault has strong shear slip tendency

    # Fault segment tendencies on GeoDataFrame of LineStrings
    faults_gdf = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 0), (0, 100)])],
        crs="EPSG:32751",
    )
    segs = fault_segment_tendencies(faults_gdf, sigma1=100.0, sigma3=40.0, sigma1_azimuth_deg=90.0)
    assert len(segs) == 2

    # 2. Cloude-Pottier and Freeman-Durden L-band SAR decompositions (H, W, 3, 3)
    T3 = np.zeros((6, 6, 3, 3), dtype=np.complex128)
    T3[..., 0, 0] = 0.5
    T3[..., 1, 1] = 0.3
    T3[..., 2, 2] = 0.1
    cp = cloude_pottier(T3)
    fd = freeman_durden(T3)
    assert np.all((cp["entropy"] >= 0.0) & (cp["entropy"] <= 1.0))
    assert np.all((cp["alpha_deg"] >= 0.0) & (cp["alpha_deg"] <= 90.0))
    assert np.all(fd["Pd"] >= 0.0)

    # 3. Gravity gradient step
    x = np.linspace(-5000, 5000, 32)
    xx, _ = np.meshgrid(x, x)
    bouguer = 25.0 * np.tanh(xx / 1000.0)  # sharp craton-margin step across x=0
    grad_mgal_km = gravity_gradient_step(bouguer, resolution_m=float(x[1] - x[0]), smoothing_sigma_m=500.0)
    assert grad_mgal_km[16, 16] > grad_mgal_km[16, 0]

    # 4. D8 catchment mass-balance deconvolution
    dem = np.outer(np.linspace(100.0, 10.0, 10), np.ones(10))  # slopes southward
    sample_cells = np.array([8 * 10 + 5, 4 * 10 + 5])
    grades = np.array([120.0, 30.0])
    deconv = deconvolve_stream_sediment(dem, resolution_m=50.0, sample_cells=sample_cells, concentrations=grades)
    assert deconv["source_concentration"].shape == (10, 10)
    assert np.nanmax(deconv["source_concentration"]) > 0.0


def test_pass1_3_aem_conductance_and_regolith_inversion():
    """Verify Pass 1.3 0-300 m AEM conductance integration and regolith cover inversion."""
    depth_bounds = np.array([0.0, 10.0, 30.0, 60.0, 100.0, 180.0, 300.0])
    # 6 layers, 8x8 grid
    sigma = np.full((6, 8, 8), 0.002, dtype=np.float64)  # resistive basement (0.002 S/m)
    # Top 2 layers (0-30 m) are conductive regolith (0.05 S/m)
    sigma[0, :, :] = 0.05
    sigma[1, :, :] = 0.05
    # Graphitic shear zone at (2, 2) has high conductivity at depth (layers 3-5 = 0.15 S/m)
    sigma[3:, 2, 2] = 0.15

    aem = aem_depth_slice_conductance(sigma, depth_bounds, reductant_trap_threshold_s=10.0)
    assert aem["aem_conductance_0_300m_S"][2, 2] > 30.0
    assert bool(aem["reductant_trap_inferred"][2, 2]) is True
    assert bool(aem["reductant_trap_inferred"][0, 0]) is False

    reg_depth = invert_regolith_thickness_from_aem(
        sigma, depth_bounds, bedrock_sigma_threshold_s_m=0.01
    )
    # Background has conductive layers 0 (10 m) and 1 (20 m) = 30 m + transition in layer 2 (30 m)
    assert 30.0 <= reg_depth[0, 0] <= 60.0

    existing_mask = np.zeros((8, 8), dtype=bool)
    updated_mask, clean_depth = update_cover_mask_from_aem(
        existing_mask, reg_depth, max_penetration_m=25.0
    )
    assert updated_mask.all()
    assert np.all(clean_depth >= 30.0)


# ---------------------------------------------------------------------------
# Pass 2 & Acceptance Criterion 2: Selection-Bias Weighting
# ---------------------------------------------------------------------------


def test_pass2_propensity_stability_and_weights():
    """Verify Pass 2 propensity model bounds e(x_i) >= 0.05 and computes w_i = 1/e(x_i)."""
    rng = np.random.default_rng(42)
    N = 300
    regolith_depth = rng.uniform(0.0, 120.0, size=N)
    outcrop_exposure = (regolith_depth < 5.0).astype(float)
    dist_to_road = rng.uniform(0.0, 50_000.0, size=N)

    X_acc = np.column_stack([regolith_depth, outcrop_exposure, dist_to_road])
    # Historical exploration strongly biased toward shallow cover and near roads
    logit = 2.5 - 0.08 * regolith_depth - 0.0002 * dist_to_road + 1.5 * outcrop_exposure
    prob = 1.0 / (1.0 + np.exp(-logit))
    explored = (rng.uniform(size=N) < prob).astype(int)

    e_x = fit_exploration_propensity(X_acc, explored, min_propensity=0.05)
    w_i = compute_inverse_propensity_weights(e_x, min_propensity=0.05)

    # Acceptance Criterion 2: e(x_i) bounded away from zero (>= 0.05)
    assert np.all(e_x >= 0.05)
    assert np.all(w_i <= 20.0 + 1e-9)
    assert np.allclose(w_i, 1.0 / e_x)


# ---------------------------------------------------------------------------
# Pass 3 & Acceptance Criteria 1 & 3: Tabular, Spatial nnPU, and Ensembling
# ---------------------------------------------------------------------------


def test_acceptance_criterion_1_no_raw_pixel_leakage():
    """Verify raw sensor band names are rejected while Pass 1 physical indices pass."""
    valid_features = [
        "al_oh_crystallinity",
        "kaolinite_alunite_doublet",
        "carbonate_shift_nm",
        "fe_oxide_intensity",
        "slip_tendency_Ts",
        "dilation_tendency_Td",
        "sar_alpha_deg",
        "aem_conductance_0_300m_S",
    ]
    assert verify_no_raw_pixel_leakage(valid_features) is True

    with pytest.raises(ValueError, match="Raw sensor pixel leakage"):
        verify_no_raw_pixel_leakage(["al_oh_crystallinity", "aster_b4", "raw_band_1"])


def test_pass3_2_nnpu_loss_clamping_and_gradient_reversal():
    """Verify Acceptance Criterion 3: nnPU clamps negative unlabeled risk at 0 and reverses gradient."""
    loss_fn = nnPULoss(pi_p=0.20, gamma=1.0, beta=0.0)

    # Construct predictions where unlabeled risk R_u^-(g) - pi_p * R_p^-(g) < 0:
    # Unlabeled examples predicted very close to 0 (so R_u^-(g) = -log(1-p) ~ 0),
    # while positive examples are also predicted close to 0 (so R_p^-(g) = -log(1-p) is small
    # wait: to make R_p^-(g) = -log(1-p) LARGE, positive examples must have p close to 1!).
    # Indeed: when positives have p -> 0.999, -log(1 - 0.999) = 6.91, so pi_p * R_p^- = 1.38.
    # If unlabeled have p -> 0.001, R_u^- = -log(0.999) = 0.001.
    # Then r_neg = 0.001 - 1.38 = -1.379 < 0!
    logits = torch.tensor([6.0, 6.0, -6.0, -6.0], requires_grad=True)
    preds = torch.sigmoid(logits)
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    weights = torch.tensor([1.0, 2.0, 1.0, 5.0])

    loss = loss_fn(preds, targets, sample_weights=weights)
    assert loss_fn.last_clamped is True
    assert loss_fn.last_unlabeled_risk < 0.0
    # Forward loss must be strictly non-negative (equal to pi_p * R_p^+)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    # Unlabeled logits (indices 2, 3) receive reversed gradient (< 0 because d(r_neg)/d(logit_u) > 0)
    assert logits.grad[2].item() < 0.0


def test_pass3_tabular_spatial_and_pr_auc_ensembling():
    """Verify inverse-propensity XGBoost, nnPU U-Net training, and PR-AUC ensembling."""
    rng = np.random.default_rng(7)
    X_p = rng.normal(loc=1.5, scale=0.5, size=(20, 6))
    X_rn = rng.normal(loc=-1.0, scale=0.5, size=(30, 6))
    X_u = rng.normal(loc=-0.5, scale=1.0, size=(60, 6))

    w_p = rng.uniform(1.0, 10.0, size=20)
    w_rn = rng.uniform(1.0, 5.0, size=30)
    w_u = rng.uniform(1.0, 8.0, size=60)

    cfg = {
        "models": {
            "baseline": {"n_estimators": 20, "max_depth": 3, "n_bag": 3, "random_state": 7},
            "unet": {"window_px": 16, "in_channels": 4, "batch_size": 4, "pi_p": 0.1},
        }
    }
    xgb_model = fit_pu_model(X_p, X_rn, X_u, cfg, w_p=w_p, w_rn=w_rn, w_u=w_u)
    s_xgb = xgb_model.predict_proba(np.vstack([X_p[:4], X_rn[:4]]))
    assert s_xgb.shape == (8,)

    # Small U-Net test with nnPU and propensity weights
    X_win = rng.normal(size=(8, 4, 16, 16)).astype(np.float32)
    y_win = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    w_win = rng.uniform(1.0, 5.0, size=8).astype(np.float32)
    torch.set_num_threads(1)
    unet = MPMUNet(cfg)
    unet.fit(X_win, y_win, epochs=1, lr=1e-3, sample_weight=w_win, pi_p=0.15, use_nnpu=True)
    s_unet, _ = unet.predict_proba_mc(X_win, n_passes=2)

    fused, w1, w2 = blend_predictions_by_pr_auc(s_xgb, s_unet, pr_auc_xgb=0.85, pr_auc_unet=0.65)
    assert np.isclose(w1 + w2, 1.0)
    assert w1 > w2
    assert np.all((fused >= 0.0) & (fused <= 1.0))


# ---------------------------------------------------------------------------
# Pass 4, 5 & Acceptance Criterion 4: Elkan-Noto & Conformal Calibration
# ---------------------------------------------------------------------------


def test_pass4_and_pass5_with_spatial_block_independence():
    """Verify Elkan-Noto c_hat, Conformal [L, U], and Acceptance Criterion 4 (block independence)."""
    val_pos_scores = np.array([0.68, 0.75, 0.80, 0.73])
    c_hat = estimate_elkan_noto_c(
        val_pos_scores,
        train_block_ids=np.array([0, 1, 2]),
        val_block_ids=np.array([3, 4]),
    )
    assert np.isclose(c_hat, 0.74, atol=1e-2)

    # Acceptance Criterion 4: overlapping train/val blocks must raise ValueError
    with pytest.raises(ValueError, match="Spatial block leakage"):
        estimate_elkan_noto_c(
            val_pos_scores,
            train_block_ids=np.array([0, 1, 2]),
            val_block_ids=np.array([2, 3]),
        )

    raw_scores = np.array([0.37, 0.68, 0.85])
    cal_probs = calibrate_elkan_noto_posterior(raw_scores, c_hat)
    assert np.isclose(cal_probs[1], 0.68 / c_hat, atol=1e-4)
    assert cal_probs[2] == 1.0

    # Pass 5: Conformal bounds at alpha = 0.10 (90% coverage)
    holdout_probs = np.linspace(0.1, 0.9, 30)
    holdout_targets = np.clip(holdout_probs + np.sin(np.arange(30)) * 0.08, 0.0, 1.0)
    q_hat = spatial_conformal_calibrate(
        holdout_probs,
        holdout_targets,
        alpha=0.10,
        cal_block_ids=np.repeat([3, 4], 15),
        train_block_ids=np.array([0, 1, 2]),
    )
    assert 0.0 < q_hat < 0.25

    with pytest.raises(ValueError, match="Spatial block leakage"):
        spatial_conformal_calibrate(
            holdout_probs,
            holdout_targets,
            alpha=0.10,
            cal_block_ids=np.repeat([2, 4], 15),
            train_block_ids=np.array([0, 1, 2]),
        )

    lower_L, upper_U = spatial_conformal_bounds(cal_probs, q_hat)
    assert np.all(lower_L <= cal_probs)
    assert np.all(cal_probs <= upper_U)
    assert np.all((lower_L >= 0.0) & (upper_U <= 1.0))


# ---------------------------------------------------------------------------
# Pass 6 & Pass 7: Polygonization, Multi-Criteria Ranking & Site Card JSON
# ---------------------------------------------------------------------------


def test_pass6_and_pass7_polygonization_ranking_and_site_cards(tmp_path: Path):
    """Verify L(x) >= 0.50 thresholding, morphological closing, ranking, and JSON schema."""
    H, W = 32, 32
    transform = from_bounds(320_000.0, 6_840_000.0, 323_200.0, 6_843_200.0, W, H)  # 100 m px
    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32751",
        "transform": transform,
    }

    cal_prob = np.full((H, W), 0.20, dtype=np.float64)
    lower_L = np.full((H, W), 0.10, dtype=np.float64)
    upper_U = np.full((H, W), 0.35, dtype=np.float64)
    td_map = np.full((H, W), 0.40, dtype=np.float64)
    alt_map = np.full((H, W), 0.30, dtype=np.float64)
    cover_map = np.full((H, W), 60.0, dtype=np.float64)

    # Target 1 (high probability, high Td, high alteration, shallow cover) with 1-px gap bridged by closing
    cal_prob[4:10, 4:7] = 0.92
    cal_prob[4:10, 8:11] = 0.92
    lower_L[4:10, 4:7] = 0.65
    lower_L[4:10, 8:11] = 0.65
    upper_U[4:10, 4:11] = 0.98
    td_map[4:10, 4:11] = 0.88
    alt_map[4:10, 4:11] = 0.82
    cover_map[4:10, 4:11] = 18.0

    # Target 2 (lower probability, deeper cover)
    cal_prob[20:26, 20:26] = 0.72
    lower_L[20:26, 20:26] = 0.55
    upper_U[20:26, 20:26] = 0.88
    td_map[20:26, 20:26] = 0.50
    alt_map[20:26, 20:26] = 0.45
    cover_map[20:26, 20:26] = 85.0

    ranked_gdf = polygonize_and_rank_targets(
        calibrated_prob=cal_prob,
        lower_bound_L=lower_L,
        upper_bound_U=upper_U,
        dilation_tendency_Td=td_map,
        alteration_index=alt_map,
        cover_depth_m=cover_map,
        profile=profile,
        l_threshold=0.50,
        closing_size=3,
        min_area_ha=5.0,
        region_prefix="YILGARN",
    )

    assert len(ranked_gdf) == 2
    assert ranked_gdf.iloc[0]["rank"] == 1
    assert ranked_gdf.iloc[0]["rank_score"] > ranked_gdf.iloc[1]["rank_score"]

    # Save Conformal GeoTIFF and Site Cards JSON
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    conformal_tif = tmp_path / "conformal_bounds.tif"
    save_conformal_bounds_geotiff(
        lower_L.ravel(), upper_U.ravel(), rows.ravel(), cols.ravel(), H, W, profile, str(conformal_tif)
    )
    assert conformal_tif.exists()

    json_path = tmp_path / "site_cards.json"
    cards = emit_site_cards_json(ranked_gdf, str(json_path), elkan_noto_c_hat=0.74)
    assert len(cards) == 2
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    top = loaded[0]
    assert top["target_id"] == "YILGARN_TGT_0001"
    assert top["rank"] == 1
    assert set(top["geometry"].keys()) == {"centroid_lat_lon", "area_sq_km", "window_footprint_km"}
    assert set(top["probabilities"].keys()) == {
        "fused_raw_score",
        "elkan_noto_c_hat",
        "calibrated_deposit_prob",
        "conformal_90_ci",
        "exploration_propensity_e_x",
        "propensity_weight_w",
    }
    assert set(top["probabilities"]["conformal_90_ci"].keys()) == {"lower_bound_L", "upper_bound_U"}
    assert set(top["physics_and_geology_summary"].keys()) == {
        "alteration",
        "geomechanics_and_structure",
        "cover_and_conductivity",
    }
