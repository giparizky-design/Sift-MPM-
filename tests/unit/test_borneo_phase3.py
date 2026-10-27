"""Unit and verification tests for Phase 3: Mineral Systems SHAP & LLM Site Cards (Passes 6 & 7).

Covers:
  - Task 3.1 Rule 1 & 2: Feature classification and TreeSHAP partitioning into the 4 Skirrow
    Mineral Systems buckets (energy & metal sources, crustal conduits, physicochemical traps,
    alteration & anomaly).
  - Task 3.1 Rule 3: Multi-mineral JSON site card serialization with conformal target sets,
    telescoped system flag, OUTMOGC boundary gates, exploration propensity, and SHAP attribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from src.models.pu_pipeline import (
    K4_HEAD_NAMES,
    TwoStageK4Ensemble,
    fit_k4_independent_xgboost_heads,
)
from src.outputs.prospectivity_map import (
    SKIRROW_MINERAL_SYSTEMS_BUCKETS,
    classify_skirrow_bucket,
    emit_site_cards_json,
    extract_target_shap_attribution,
    partition_shap_into_mineral_systems,
)


def test_classify_skirrow_mineral_systems_buckets():
    """Verify that features are partitioned strictly into the 4 Skirrow Mineral Systems buckets."""
    # 1. Energy & Metal Sources
    assert classify_skirrow_bucket("f_crustal_thickness_km") == "1_energy_and_metal_sources"
    assert classify_skirrow_bucket("bouguer_30km") == "1_energy_and_metal_sources"
    assert classify_skirrow_bucket("geophys_mean_1km_mag") == "1_energy_and_metal_sources"
    assert classify_skirrow_bucket("grav_mean_1km") == "1_energy_and_metal_sources"

    # 2. Crustal Conduits
    assert classify_skirrow_bucket("bouguer_worm_5km") == "2_crustal_conduits"
    assert classify_skirrow_bucket("bouguer_worm_15km") == "2_crustal_conduits"
    assert classify_skirrow_bucket("bouguer_worm_30km") == "2_crustal_conduits"
    assert classify_skirrow_bucket("vertical_plumbing_persistence") == "2_crustal_conduits"

    # 3. Physicochemical Traps
    assert classify_skirrow_bucket("dilation_tendency_Td") == "3_physicochemical_traps"
    assert classify_skirrow_bucket("slip_tendency_Ts") == "3_physicochemical_traps"
    assert classify_skirrow_bucket("geophys_contrast_4km_mag") == "3_physicochemical_traps"
    assert classify_skirrow_bucket("plateau_low_slope_gate") == "3_physicochemical_traps"

    # 4. Alteration & Anomaly
    assert classify_skirrow_bucket("dae_recon_err") == "4_alteration_and_anomaly"
    assert classify_skirrow_bucket("unet_ctx_0") == "4_alteration_and_anomaly"
    assert classify_skirrow_bucket("oof_prob_skarn") == "4_alteration_and_anomaly"
    assert classify_skirrow_bucket("oof_prob_epithermal") == "4_alteration_and_anomaly"
    assert classify_skirrow_bucket("al_oh_crystallinity") == "4_alteration_and_anomaly"
    assert classify_skirrow_bucket("kaolinite_alunite_doublet") == "4_alteration_and_anomaly"


def test_partition_shap_into_mineral_systems():
    """Verify that SHAP contributions are grouped into buckets with correct sums and top features."""
    feature_names = [
        "f_crustal_thickness_km",
        "bouguer_worm_15km",
        "dilation_tendency_Td",
        "dae_recon_err",
        "unet_ctx_0",
    ]
    shap_vals = np.array([0.40, 0.25, -0.30, 0.50, 0.15])
    feat_vals = np.array([32.5, 0.85, 0.70, 0.12, 1.40])

    partitioned = partition_shap_into_mineral_systems(
        feature_names=feature_names,
        shap_values=shap_vals,
        feature_values=feat_vals,
        top_k=2,
    )

    for b in SKIRROW_MINERAL_SYSTEMS_BUCKETS:
        assert b in partitioned
        assert "total_shap" in partitioned[b]
        assert "top_features" in partitioned[b]

    # Verify bucket 1
    assert np.isclose(partitioned["1_energy_and_metal_sources"]["total_shap"], 0.40)
    assert partitioned["1_energy_and_metal_sources"]["n_features"] == 1
    assert partitioned["1_energy_and_metal_sources"]["top_features"][0]["feature"] == "f_crustal_thickness_km"

    # Verify bucket 4 (dae_recon_err + unet_ctx_0 = 0.50 + 0.15 = 0.65)
    assert np.isclose(partitioned["4_alteration_and_anomaly"]["total_shap"], 0.65)
    assert partitioned["4_alteration_and_anomaly"]["n_features"] == 2
    assert partitioned["4_alteration_and_anomaly"]["top_features"][0]["feature"] == "dae_recon_err"


def test_two_stage_k4_ensemble_compute_head_shap():
    """Verify TreeSHAP extraction for active heads from TwoStageK4Ensemble."""
    np.random.seed(42)
    N, D = 80, 4
    base_feature_names = [
        "f_crustal_thickness_km",
        "bouguer_worm_15km",
        "dilation_tendency_Td",
        "dae_recon_err",
    ]
    X = np.random.randn(N, D)
    y_k4 = np.zeros((N, 4), dtype=np.int32)
    y_k4[:15, 0] = 1
    y_k4[15:30, 1] = 1

    ens = TwoStageK4Ensemble(n_estimators=10, n_bag=2, random_state=42)
    ens.fit(X, y_k4)

    # Head 1 (skarn) with 1D input should return (D,)
    sv1, fn1 = ens.compute_head_shap("skarn_fe_cu", X[0], base_feature_names)
    assert len(fn1) == D
    assert sv1.shape == (D,)

    # Head 0 (porphyry) with 1D input should have D + 2 features (telescoping OOF probabilities)
    sv0, fn0 = ens.compute_head_shap("porphyry_cu", X[0], base_feature_names)
    assert len(fn0) == D + 2
    assert "oof_prob_skarn" in fn0
    assert "oof_prob_epithermal" in fn0
    assert sv0.shape == (D + 2,)

    # Head 1 with 2D batch input should return (N, D)
    sv_batch, _ = ens.compute_head_shap("skarn_fe_cu", X[:3], base_feature_names)
    assert sv_batch.shape == (3, D)


def test_extract_target_shap_attribution_drill_gate():
    """Verify that SHAP attribution is extracted only when max_k L_0.90 >= tau_drill."""
    np.random.seed(42)
    N, D = 60, 4
    base_feats = ["f_crustal_thickness_km", "bouguer_worm_15km", "dilation_tendency_Td", "dae_recon_err"]
    X = np.random.randn(N, D)
    y_k4 = np.zeros((N, 4), dtype=np.int32)
    y_k4[:15, 0] = 1

    ens = TwoStageK4Ensemble(n_estimators=10, n_bag=2, random_state=42)
    ens.fit(X, y_k4)

    x_query = X[0]

    # When lower_bound >= tau_drill, attribution is populated for active heads
    attr_confident = extract_target_shap_attribution(
        model=ens,
        x_sample=x_query,
        feature_names=base_feats,
        active_targets=["porphyry_cu"],
        tau_drill=0.50,
        lower_bound=0.72,
    )
    assert "porphyry_cu" in attr_confident
    assert len(attr_confident["porphyry_cu"]) == 4

    # When lower_bound < tau_drill, returns empty attribution
    attr_subthreshold = extract_target_shap_attribution(
        model=ens,
        x_sample=x_query,
        feature_names=base_feats,
        active_targets=["porphyry_cu"],
        tau_drill=0.50,
        lower_bound=0.35,
    )
    assert attr_subthreshold == {}


def test_emit_site_cards_json_multi_mineral_provenance(tmp_path: Path):
    """Verify Phase 3 site card serialization with all 5 mandatory multi-mineral fields."""
    poly1 = Polygon([(115.0, 0.0), (115.1, 0.0), (115.1, 0.1), (115.0, 0.1)])
    poly2 = Polygon([(116.0, 1.0), (116.1, 1.0), (116.1, 1.1), (116.0, 1.1)])
    poly3 = Polygon([(117.0, -1.0), (117.1, -1.0), (117.1, -0.9), (117.0, -0.9)])

    # Target 1: Telescoped system (porphyry + skarn, |C_0.90| = 2)
    # Target 2: Single system (laterite, |C_0.90| = 1)
    # Target 3: Subthreshold anomaly (|C_0.90| = 0)
    gdf = gpd.GeoDataFrame(
        {
            "target_id": ["BORNEO_TGT_0001", "BORNEO_TGT_0002", "BORNEO_TGT_0003"],
            "rank": [1, 2, 3],
            "rank_score": [0.94, 0.81, 0.42],
            "calibrated_deposit_prob": [0.89, 0.78, 0.45],
            "lower_bound_L": [0.75, 0.62, 0.28],
            "upper_bound_U": [0.98, 0.90, 0.65],
            "fused_raw_score": [0.82, 0.70, 0.38],
            "exploration_propensity_e_x": [0.22, 0.35, 0.50],
            "propensity_weight_w": [4.54, 2.85, 2.0],
            "conformal_90pct_target_set": [
                ["porphyry_cu", "skarn_fe_cu"],
                ["laterite_fe_ni"],
                [],
            ],
            "telescoped_system_flag": [True, False, False],
            "outmogc_boundary_gate": [
                {"porphyry_cu": 1.0, "skarn_fe_cu": 1.0, "epithermal_au": 0.8, "laterite_fe_ni": 0.0},
                {"porphyry_cu": 0.0, "skarn_fe_cu": 0.0, "epithermal_au": 0.0, "laterite_fe_ni": 1.0},
                {"porphyry_cu": 0.5, "skarn_fe_cu": 0.5, "epithermal_au": 0.5, "laterite_fe_ni": 0.5},
            ],
            "exploration_logistics_propensity": [0.22, 0.35, 0.50],
            "mineral_systems_shap_attribution": [
                {
                    "porphyry_cu": {
                        "1_energy_and_metal_sources": {"total_shap": 0.42, "n_features": 2, "top_features": []},
                        "2_crustal_conduits": {"total_shap": 0.35, "n_features": 2, "top_features": []},
                        "3_physicochemical_traps": {"total_shap": 0.51, "n_features": 3, "top_features": []},
                        "4_alteration_and_anomaly": {"total_shap": 0.65, "n_features": 3, "top_features": []},
                    },
                    "skarn_fe_cu": {
                        "1_energy_and_metal_sources": {"total_shap": 0.30, "n_features": 2, "top_features": []},
                        "2_crustal_conduits": {"total_shap": 0.40, "n_features": 2, "top_features": []},
                        "3_physicochemical_traps": {"total_shap": 0.55, "n_features": 3, "top_features": []},
                        "4_alteration_and_anomaly": {"total_shap": 0.20, "n_features": 2, "top_features": []},
                    },
                },
                {
                    "laterite_fe_ni": {
                        "1_energy_and_metal_sources": {"total_shap": 0.10, "n_features": 1, "top_features": []},
                        "2_crustal_conduits": {"total_shap": 0.05, "n_features": 1, "top_features": []},
                        "3_physicochemical_traps": {"total_shap": 0.82, "n_features": 3, "top_features": []},
                        "4_alteration_and_anomaly": {"total_shap": 0.15, "n_features": 2, "top_features": []},
                    }
                },
                {},
            ],
            "al_oh_crystallinity": [1.25, 0.90, 0.85],
            "kaolinite_alunite_doublet": [0.18, 0.0, 0.0],
            "aem_conductance_0_300m_S": [14.0, 4.0, 6.0],
            "geometry": [poly1, poly2, poly3],
        },
        crs="EPSG:4326",
    )

    out_file = tmp_path / "borneo_site_cards.json"
    cards = emit_site_cards_json(gdf, str(out_file), elkan_noto_c_hat=0.74, tau_drill=0.50)

    assert len(cards) == 3
    loaded = json.loads(out_file.read_text(encoding="utf-8"))

    card0 = loaded[0]
    # Check 5 mandatory multi-mineral provenance fields
    assert "conformal_90pct_target_set" in card0
    assert "telescoped_system_flag" in card0
    assert "outmogc_boundary_gate" in card0
    assert "exploration_logistics_propensity" in card0
    assert "mineral_systems_shap_attribution" in card0

    # Target 0: Telescoped
    assert card0["conformal_90pct_target_set"] == ["porphyry_cu", "skarn_fe_cu"]
    assert card0["telescoped_system_flag"] is True
    assert card0["exploration_logistics_propensity"] == 0.22
    assert "porphyry_cu" in card0["mineral_systems_shap_attribution"]
    assert "skarn_fe_cu" in card0["mineral_systems_shap_attribution"]
    for b in SKIRROW_MINERAL_SYSTEMS_BUCKETS:
        assert b in card0["mineral_systems_shap_attribution"]["porphyry_cu"]

    # Target 1: Single
    card1 = loaded[1]
    assert card1["conformal_90pct_target_set"] == ["laterite_fe_ni"]
    assert card1["telescoped_system_flag"] is False
    assert "laterite_fe_ni" in card1["mineral_systems_shap_attribution"]

    # Target 2: Subthreshold
    card2 = loaded[2]
    assert card2["conformal_90pct_target_set"] == []
    assert card2["telescoped_system_flag"] is False
    assert card2["mineral_systems_shap_attribution"] == {}

    # Verify standard legacy schemas remain intact
    assert set(card0["geometry"].keys()) == {"centroid_lat_lon", "area_sq_km", "window_footprint_km"}
    assert set(card0["probabilities"].keys()) == {
        "fused_raw_score",
        "elkan_noto_c_hat",
        "calibrated_deposit_prob",
        "conformal_90_ci",
        "exploration_propensity_e_x",
        "propensity_weight_w",
    }
    assert set(card0["physics_and_geology_summary"].keys()) == {
        "alteration",
        "geomechanics_and_structure",
        "cover_and_conductivity",
    }
