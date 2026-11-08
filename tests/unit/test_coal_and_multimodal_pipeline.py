"""Unit tests for the SIFT Dual-Domain Engine Upgrade (K=4 Metals + K=3 Coal).

Covers all 5 mandatory verification areas:
1. Rim-Anchored Rank-Adaptive Sub-Canopy Halo Sampler & 3 Non-Negotiable Spatial Guardrails:
   - Guardrail 1: Rim-Anchored Origin (d=0 at disturbance boundary rather than pit centroid).
   - Guardrail 2: Same-Formation Polygon Lock (formation_polygon_id[r', c'] == formation_polygon_id[r_i, c_i] > 0).
   - Guardrail 3: 3x3 Canopy Integrity Check (every pixel in 3x3 neighborhood has NDVI >= 0.25 and active_mine_mask == 0).
   - Rank-Adaptive Offsets: High-CV [4, 8], Med-CV [6, 12], Low-CV [6, 14] cells from the disturbance rim.
2. Continuous Canopy-Transmittance Weight & Bayesian Log-Odds Fusion:
   - w_canopy(s) == 0.0 for NDVI >= 0.30 and 1.0 for NDVI <= 0.10.
   - LogOdds_final == LogOdds_basin when w_canopy == 0.0, and monotonically non-decreasing in positive optical log-likelihood ratio.
3. Catchment-Censored Upslope AMD Pathfinder:
   - Stream pixels whose upslope contributing area contains an active_mine_mask == 1 pixel have their AMD anomaly strictly zeroed out (0.0) before upslope back-propagation.
4. Feature Leakage & Anthropogenic Quarantine Gate:
   - verify_no_raw_pixel_leakage() raises ValueError on raw bands (B02, B11, B13) and anthropogenic/metadata variables (dist_major_rivers_km, dist_haul_roads_km, gfw_forest_disturbance, explored_footprint, active_mine_mask, formation_polygon_id) for both TwoStageK4Ensemble and TwoStageK3CoalEnsemble.
5. Dual-Domain Ensemble End-to-End Execution:
   - TwoStageK4Ensemble and TwoStageK3CoalEnsemble fit and predict calibrated probabilities in [0.0, 1.0] with zero NaN leakage, valid epistemic uncertainties, and expected calorific class in {0, 1, 2}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.coal_and_multimodal_features import (
    compute_continuous_canopy_weight,
    compute_hydro_twi_and_censored_upslope_amd,
)
from src.labels.borneo_labels import (
    COAL_HEAD_NAMES,
    RANK_ADAPTIVE_OFFSETS_CELLS,
    build_k3_coal_rank_targets,
)
from src.models.pu_pipeline import (
    BARREN_OPTICAL_COAL_FEATURES,
    K3_COAL_HEAD_NAMES,
    SUBCANOPY_COAL_FEATURES,
    TwoStageK3CoalEnsemble,
    TwoStageK4Ensemble,
    fuse_canopy_log_odds,
    verify_no_raw_pixel_leakage,
)


def _grid_bounds_and_center(h: int = 45, w: int = 45, cr: int = 22, cc: int = 22) -> tuple[tuple[float, float, float, float], float, float]:
    """Return (bounds=(west, south, east, north), center_lat, center_lon) for a synthetic 2D grid."""
    west, south = 116.0, -1.0
    east = west + w * 0.0025
    north = south + h * 0.0025
    # Row 0 is North, Row H-1 is South in standard rasterio north-up convention
    center_lon = west + (cc + 0.5) * 0.0025
    center_lat = north - (cr + 0.5) * 0.0025
    return (west, south, east, north), center_lat, center_lon


class TestHaloSamplerAndSpatialGuardrails:
    """Test Area 1: Rim-Anchored Rank-Adaptive Halo Sampler & 3 Non-Negotiable Spatial Guardrails."""

    def test_guardrail_1_rim_anchored_origin_and_rank_adaptive_offsets(self) -> None:
        """Verify a synthetic mine pit of radius R shifts the halo origin to the pit rim (offset_cells in [d_min, d_max])."""
        h, w = 55, 55
        cr, cc = 27, 27
        bounds, center_lat, center_lon = _grid_bounds_and_center(h, w, cr, cc)

        # Create a contiguous mine pit of radius R = 4 cells around (cr, cc)
        rr, cc_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        dist_from_center = np.hypot(rr - cr, cc_grid - cc)
        pit_mask_2d = (dist_from_center <= 4.0).astype(np.int8)

        # Closed forest canopy everywhere outside the pit, barren inside the pit
        ndvi_2d = np.where(pit_mask_2d == 1, 0.12, 0.78).astype(np.float32)

        # Uniform host formation polygon ID = 7 everywhere
        formation_2d = np.full((h, w), 7, dtype=np.int32)

        # Strike = 0 deg (along columns: sin(0)=0 -> dr=0, cos(0)=1 -> dc=+-1)
        strike_2d = np.zeros((h, w), dtype=np.float32)

        for rank_class, (d_min, d_max) in RANK_ADAPTIVE_OFFSETS_CELLS.items():
            coal_df = pd.DataFrame(
                {
                    "latitude": [center_lat],
                    "longitude": [center_lon],
                    "coal_rank_head": [rank_class],
                    "totsdbb": [50.0],
                }
            )
            res = build_k3_coal_rank_targets(
                concessions=coal_df,
                ndvi=ndvi_2d,
                active_mine_mask=pit_mask_2d,
                formation_polygon_id=formation_2d,
                sar_bedding_strike_deg=strike_2d,
                bounds=bounds,
            )

            raster_k = res[COAL_HEAD_NAMES[rank_class]]
            pos_r, pos_c = np.where(raster_k == 1)
            assert len(pos_r) > 0, f"Expected halo positives for rank_class={rank_class}"

            # None of the positive halo pixels may fall inside the pit
            assert np.all(pit_mask_2d[pos_r, pos_c] == 0)

            # Because the pit has radius 4 cells along the ray, rim_step_cells must be >= 3
            halo_df = res["halo_samples_df"]
            assert len(halo_df) == len(pos_r)
            assert np.all(halo_df["rim_step_cells"].to_numpy() >= 3)
            # And offset_cells from the pit rim must lie strictly within [d_min, d_max]
            offsets = halo_df["offset_cells"].to_numpy()
            assert np.all((offsets >= d_min) & (offsets <= d_max))
            # Distance from the pit centroid is rim_step_cells + offset_cells >= 3 + d_min
            d_center = dist_from_center[pos_r, pos_c]
            assert np.all(d_center >= 3.0 + d_min - 0.5)

    def test_guardrail_2_same_formation_polygon_lock(self) -> None:
        """Verify a strike vector crossing a synthetic formation polygon boundary (1 -> 2) is strictly rejected."""
        h, w = 55, 55
        cr, cc = 27, 27
        bounds, center_lat, center_lon = _grid_bounds_and_center(h, w, cr, cc)

        # Strike = 0 deg -> steps along columns (East-West)
        strike_2d = np.zeros((h, w), dtype=np.float32)
        ndvi_2d = np.full((h, w), 0.80, dtype=np.float32)
        mine_2d = np.zeros((h, w), dtype=np.int8)
        mine_2d[cr, cc] = 1
        ndvi_2d[cr, cc] = 0.10

        # Formation polygon ID is 1 on the West side (col <= cc + 2), but changes to 2 on the East side (col > cc + 2)
        _, cc_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        formation_2d = np.where(cc_grid <= cc + 2, 1, 2).astype(np.int32)

        coal_df = pd.DataFrame(
            {
                "latitude": [center_lat],
                "longitude": [center_lon],
                "coal_rank_head": [0],
                "totsdbb": [20.0],
            }
        )
        res = build_k3_coal_rank_targets(
            concessions=coal_df,
            ndvi=ndvi_2d,
            active_mine_mask=mine_2d,
            formation_polygon_id=formation_2d,
            sar_bedding_strike_deg=strike_2d,
            bounds=bounds,
        )

        pos_r, pos_c = np.where(res["coal_k0_high_cv"] == 1)
        assert len(pos_r) > 0, "Westward strike direction (formation 1) should still yield valid positives"
        # Every selected halo pixel MUST belong to formation 1; zero pixels in formation 2 may be selected
        assert np.all(formation_2d[pos_r, pos_c] == 1)
        assert np.all(pos_c < cc), "Eastward strike candidates crossing into formation 2 must be rejected"

    def test_guardrail_3_3x3_canopy_integrity_check(self) -> None:
        """Verify any candidate whose 3x3 neighborhood contains a pixel with NDVI < 0.25 or active_mine_mask == 1 is rejected."""
        h, w = 55, 55
        cr, cc = 27, 27
        bounds, center_lat, center_lon = _grid_bounds_and_center(h, w, cr, cc)

        strike_2d = np.zeros((h, w), dtype=np.float32)  # East-West along columns
        formation_2d = np.full((h, w), 3, dtype=np.int32)
        ndvi_2d = np.full((h, w), 0.78, dtype=np.float32)
        mine_2d = np.zeros((h, w), dtype=np.int8)
        mine_2d[cr, cc] = 1
        ndvi_2d[cr, cc] = 0.08

        # Contaminate the 3x3 neighborhood on the East side (cols cc+3..cc+10) with a low-NDVI pixel at row cr+1
        for c_east in range(cc + 3, cc + 11):
            ndvi_2d[cr + 1, c_east] = 0.20  # Violates 3x3 canopy integrity (< 0.25) even though row cr has NDVI=0.78!

        coal_df = pd.DataFrame(
            {
                "latitude": [center_lat],
                "longitude": [center_lon],
                "coal_rank_head": [0],
                "totsdbb": [15.0],
            }
        )
        res = build_k3_coal_rank_targets(
            concessions=coal_df,
            ndvi=ndvi_2d,
            active_mine_mask=mine_2d,
            formation_polygon_id=formation_2d,
            sar_bedding_strike_deg=strike_2d,
            bounds=bounds,
        )

        pos_r, pos_c = np.where(res["coal_k0_high_cv"] == 1)
        assert len(pos_r) > 0
        # All East-side candidates must be rejected because their 3x3 window touches row cr+1 with NDVI=0.20
        assert np.all(pos_c < cc), "East-side candidates with 3x3 neighbor NDVI < 0.25 must be rejected"


class TestCanopyTransmittanceAndBayesianLogOddsFusion:
    """Test Area 2: Continuous Canopy-Transmittance Weight & Bayesian Log-Odds Fusion."""

    def test_continuous_canopy_weight_boundaries(self) -> None:
        """Verify w_canopy(s) == 0.0 for NDVI >= 0.30 and 1.0 for NDVI <= 0.10."""
        ndvi = np.array([-0.10, 0.05, 0.10, 0.20, 0.30, 0.45, 0.85], dtype=np.float32)
        w = compute_continuous_canopy_weight(ndvi)
        assert w.dtype == np.float32
        # NDVI <= 0.10 -> 1.0
        assert np.allclose(w[:3], 1.0)
        # NDVI == 0.20 -> midpoint 0.5
        assert np.isclose(w[3], 0.5, atol=1e-5)
        # NDVI >= 0.30 -> strictly 0.0
        assert np.all(w[4:] == 0.0)

    def test_bayesian_log_odds_fusion_zero_canopy_and_monotonicity(self) -> None:
        """Verify LogOdds_final == LogOdds_basin when w_canopy == 0.0 and monotonicity in positive LLR."""
        lo_basin = np.array([-2.2, -1.1, 0.0, 1.2], dtype=np.float64)
        ll_optical = np.array([1.5, -0.8, 2.0, -1.5], dtype=np.float64)
        w_zero = np.zeros_like(lo_basin)

        lo_fused_zero = fuse_canopy_log_odds(lo_basin, ll_optical, w_zero)
        assert np.allclose(lo_fused_zero, lo_basin, atol=1e-8)

        # Positive optical log-likelihood ratio (ln Lambda_optical > 0):
        # Increasing w_canopy from 0 -> 0.5 -> 1.0 must monotonically increase LogOdds_final
        ll_pos = np.full(4, 1.25, dtype=np.float64)
        lo_w0 = fuse_canopy_log_odds(lo_basin, ll_pos, np.full(4, 0.0))
        lo_w5 = fuse_canopy_log_odds(lo_basin, ll_pos, np.full(4, 0.5))
        lo_w1 = fuse_canopy_log_odds(lo_basin, ll_pos, np.full(4, 1.0))
        assert np.all(lo_w5 > lo_w0)
        assert np.all(lo_w1 > lo_w5)

        # Negative optical LLR (< 0) is clamped to 0 so shadow/soil never penalizes sub-canopy basin prior
        ll_neg = np.full(4, -2.0, dtype=np.float64)
        lo_neg_fused = fuse_canopy_log_odds(lo_basin, ll_neg, np.full(4, 1.0))
        assert np.allclose(lo_neg_fused, lo_basin, atol=1e-8)


class TestCatchmentCensoredUpslopeAMD:
    """Test Area 3: Catchment-Censored Upslope AMD Pathfinder."""

    def test_mine_contaminated_stream_amd_strictly_zeroed(self) -> None:
        """Verify any stream pixel whose upslope contributing area contains an active_mine_mask == 1 pixel is zeroed."""
        h, w = 24, 24
        rr, cc = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        # Two V-shaped parallel north-to-south drainage valleys centered at col=6 and col=18,
        # separated by a high drainage divide at col=12 so the two catchments never mix
        dist_valley_left = np.abs(cc - 6.0)
        dist_valley_right = np.abs(cc - 18.0)
        valley_profile = np.where(cc < 12.0, dist_valley_left * 12.0, dist_valley_right * 12.0)
        dem = (h - rr) * 15.0 + valley_profile
        slope = np.full((h, w), 4.0, dtype=np.float32)

        # Strong ferric iron (ficr) and red-edge stress (low ndre) along both valley channels (col=6 and col=18)
        ficr = np.full((h, w), 0.8, dtype=np.float32)
        ficr[:, 6] = 3.8
        ficr[:, 18] = 3.8

        ndre = np.full((h, w), 0.65, dtype=np.float32)
        ndre[:, 6] = 0.15
        ndre[:, 18] = 0.15

        # Place an active open-pit mine at the headwaters of the right valley (row=2, col=18)
        active_mine_mask = np.zeros((h, w), dtype=np.uint8)
        active_mine_mask[2, 18] = 1

        res = compute_hydro_twi_and_censored_upslope_amd(
            dem=dem,
            slope=slope,
            ndre=ndre,
            ficr=ficr,
            active_mine_mask=active_mine_mask,
        )
        amd = res["amd_upslope_anomaly_censored"]
        assert amd.shape == (h, w)
        assert np.all(np.isfinite(amd))

        # Clean stream valley at col=6 must have positive AMD anomaly
        assert float(amd[12, 6]) > 0.0, "Unmined natural stream with high FICR + low NDRE should retain positive AMD signal"

        # Mine-contaminated stream valley at col=18 downslope of row=2 must be strictly 0.0
        assert np.all(amd[2:, 18] == 0.0), (
            f"Stream pixels downslope of active mine must be strictly 0.0, got {amd[2:, 18]}"
        )


class TestFeatureLeakageAndAnthropogenicQuarantineGate:
    """Test Area 4: Feature Leakage & Anthropogenic Quarantine Gate."""

    @pytest.mark.parametrize(
        "forbidden_col",
        [
            "B02",
            "B11",
            "B13",
            "dist_major_rivers_km",
            "dist_haul_roads_km",
            "dist_transport_km",
            "gfw_forest_disturbance",
            "forest_disturbance_rate",
            "explored_footprint",
            "active_mine_mask",
            "formation_polygon_id",
        ],
    )
    def test_verify_no_raw_pixel_leakage_blocks_forbidden_features(self, forbidden_col: str) -> None:
        """Verify verify_no_raw_pixel_leakage and both TwoStage ensembles reject raw bands and anthropogenic features."""
        valid_cols = ["sar_bedding_anisotropy", "bouguer_tilt_angle", "dem_twi"]
        with pytest.raises(ValueError, match="Raw sensor pixel leakage detected"):
            verify_no_raw_pixel_leakage(valid_cols + [forbidden_col])

        rng = np.random.default_rng(42)
        X = rng.normal(size=(40, 4)).astype(np.float32)

        k4 = TwoStageK4Ensemble(random_state=42)
        y_k4 = np.zeros((40, 4), dtype=np.int8)
        y_k4[:5, 0] = 1
        with pytest.raises(ValueError, match="Raw sensor pixel leakage detected"):
            k4.fit(X, y_k4, feature_names=valid_cols + [forbidden_col])

        k3 = TwoStageK3CoalEnsemble(random_state=42)
        y_k3 = np.zeros((40, 3), dtype=np.int8)
        y_k3[:5, 0] = 1
        with pytest.raises(ValueError, match="Raw sensor pixel leakage detected"):
            k3.fit(X, y_k3, feature_names=valid_cols + [forbidden_col])


class TestDualDomainEnsembleEndToEnd:
    """Test Area 5: Dual-Domain Ensemble End-to-End Execution."""

    def test_two_stage_k4_and_k3_ensembles_fit_and_predict_valid_probabilities(self) -> None:
        """Verify TwoStageK4Ensemble and TwoStageK3CoalEnsemble produce calibrated probabilities in [0, 1] with zero NaNs."""
        rng = np.random.default_rng(123)
        n = 320
        feature_names = list(dict.fromkeys(list(SUBCANOPY_COAL_FEATURES) + list(BARREN_OPTICAL_COAL_FEATURES)))
        d = len(feature_names)
        X = rng.normal(size=(n, d)).astype(np.float32)

        # Inject a few NaNs in features to verify zero-NaN output robustness
        X[0, 2] = np.nan
        X[15, -1] = np.nan

        blocks = np.repeat(np.arange(4, dtype=np.int32), n // 4)
        propensity = rng.uniform(0.15, 0.85, size=n).astype(np.float64)
        w_canopy = rng.choice([0.0, 0.0, 0.35, 0.85, 1.0], size=n).astype(np.float32)

        # 1. Test TwoStageK4Ensemble
        y_k4 = np.zeros((n, 4), dtype=np.int8)
        for k in range(4):
            pos_idx = rng.choice(n, size=18, replace=False)
            y_k4[pos_idx, k] = 1
            X[pos_idx, k] += 2.0

        k4_model = TwoStageK4Ensemble(n_estimators=30, n_bag=3, random_state=42)
        k4_model.fit(
            X,
            y_k4,
            e_logistics=propensity,
            spatial_block_ids=blocks,
            feature_names=feature_names,
        )
        probs_k4 = k4_model.predict_proba(X)
        std_k4 = k4_model.predict_std(X)
        assert probs_k4.shape == (n, 4)
        assert std_k4.shape == (n, 4)
        assert not np.any(np.isnan(probs_k4))
        assert not np.any(np.isnan(std_k4))
        assert np.all((probs_k4 >= 0.0) & (probs_k4 <= 1.0))
        assert np.all(std_k4 >= 0.0)

        # 2. Test TwoStageK3CoalEnsemble
        y_k3 = np.zeros((n, 3), dtype=np.int8)
        w_ton = np.ones(n, dtype=np.float32)
        for c_idx in range(3):
            pos_idx = rng.choice(n, size=20, replace=False)
            y_k3[pos_idx, c_idx] = 1
            w_ton[pos_idx] = 2.2
            X[pos_idx, c_idx + 4] += 2.2

        k3_model = TwoStageK3CoalEnsemble(n_estimators=30, n_bag=3, random_state=42)
        k3_model.fit(
            X=X,
            y_k3=y_k3,
            tonnage_weights=w_ton,
            e_pooled=propensity,
            w_canopy=w_canopy,
            spatial_block_ids=blocks,
            feature_names=feature_names,
        )
        full_k3 = k3_model.predict_full_outputs(X, w_canopy=w_canopy)

        probs_k3 = full_k3["probs_k3"]
        std_k3 = full_k3["std_k3"]
        p_any = full_k3["p_coal_any"]
        exp_cal = full_k3["expected_calorific_class"]

        assert probs_k3.shape == (n, 3)
        assert std_k3.shape == (n, 3)
        assert p_any.shape == (n,)
        assert exp_cal.shape == (n,)
        assert not np.any(np.isnan(probs_k3))
        assert not np.any(np.isnan(std_k3))
        assert not np.any(np.isnan(p_any))
        assert np.all((probs_k3 >= 0.0) & (probs_k3 <= 1.0))
        assert np.all((p_any >= 0.0) & (p_any <= 1.0))
        assert set(np.unique(exp_cal)).issubset({0, 1, 2})
