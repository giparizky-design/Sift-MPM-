"""Unit tests for leakage prevention, provenance honesty, and read-only reranking."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from src.features.feature_stack import FeatureStack
from src.labels.pu_negatives import OUTMOGCGate
from src.outputs.prospectivity_map import emit_site_cards_json, resolve_physical_layers


def test_resolve_physical_layers_aliases():
    """Verify that canonical aliases for physics layers resolve correctly and return None when missing."""
    # Stack with exact canonical names
    H, W = 20, 20
    band_td = np.random.uniform(0.1, 0.9, size=(H, W))
    band_alt = np.random.uniform(0.2, 0.8, size=(H, W))
    band_cov = np.random.uniform(10.0, 80.0, size=(H, W))

    mock_stack = MagicMock()
    mock_stack.feature_names = ["dilation_tendency_Td", "swir_clay_iron_alteration", "regolith_cover_m"]
    mock_stack.array = np.stack([band_td, band_alt, band_cov], axis=0)

    phys = resolve_physical_layers(mock_stack)
    assert phys["dilation_tendency_Td"] is not None
    assert phys["alteration_index"] is not None
    assert phys["regolith_cover_depth_m"] is not None
    assert phys["crustal_thickness_km"] is None  # Missing layer returns None, not a fake array

    # Stack with distance-to-faults fallback (inverted proxy)
    mock_stack_faults = MagicMock()
    mock_stack_faults.feature_names = ["dist_active_faults_m"]
    mock_stack_faults.array = np.array([np.random.uniform(0.0, 5000.0, size=(H, W))])

    phys_faults = resolve_physical_layers(mock_stack_faults)
    assert phys_faults["dilation_tendency_Td"] is not None
    assert phys_faults["alteration_index"] is None


def test_site_cards_no_fabricated_sensor_data(tmp_path: Path):
    """Verify that emit_site_cards_json never fabricates sensor or geomechanical values when absent."""
    poly = Polygon([(115.0, 0.0), (115.1, 0.0), (115.1, 0.1), (115.0, 0.1)])
    # Minimal GeoDataFrame without optional sensor columns (no AEM, no SWIR doublet, no Moho step, etc.)
    gdf = gpd.GeoDataFrame(
        {
            "target_id": ["TEST_TGT_0001"],
            "rank": [1],
            "rank_score": [0.88],
            "calibrated_deposit_prob": [0.85],
            "lower_bound_L": [0.72],
            "upper_bound_U": [0.94],
            "geometry": [poly],
        },
        crs="EPSG:4326",
    )

    out_json = tmp_path / "honest_site_cards.json"
    cards = emit_site_cards_json(gdf, str(out_json), elkan_noto_c_hat=0.70)
    assert len(cards) == 1

    loaded = json.loads(out_json.read_text(encoding="utf-8"))[0]
    phys = loaded["physics_and_geology_summary"]

    # When sensor data is absent, sensor_source must be 'not_available' and values must be None
    assert phys["alteration"]["sensor_source"] == "not_available"
    assert phys["alteration"]["al_oh_crystallinity_2200nm"] is None
    assert phys["alteration"]["kaolinite_alunite_doublet"] is None
    assert phys["alteration"]["fe_mg_carbonate_shift_nm"] is None

    assert phys["geomechanics_and_structure"]["sar_cloude_pottier_alpha_deg"] is None
    assert phys["geomechanics_and_structure"]["moho_gravity_gradient_step_mGal_km"] is None
    assert phys["cover_and_conductivity"]["aem_conductance_0_300m_S"] is None
    assert phys["cover_and_conductivity"]["reductant_trap_inferred"] is None


def test_gate_no_val_leakage():
    """Verify that OUTMOGCGate fitted strictly on training positives scores unseen validation samples without error."""
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    X_val = np.random.randn(50, 10)

    # Positive samples in training
    X_pos_train = X_train[:15]
    gate = OUTMOGCGate(n_pca_components=3, n_gmm_components=2, random_state=42)
    gate.fit(X_pos_train)

    # Scoring on validation split should be strictly out-of-sample
    val_scores = gate.score(X_val)
    assert len(val_scores) == len(X_val)
    assert np.all((val_scores >= 0.0) & (val_scores <= 1.0))


def test_rerank_is_read_only(tmp_path: Path, monkeypatch):
    """Verify that _recompute_targets_live does not modify persistent target artifacts."""
    from scripts.dashboard_server import _recompute_targets_live, ROOT_DIR
    import os

    # Check that calling rerank on EAST_KALIMANTAN returns a bundle with targets
    # without touching the original site_cards.json or gpkg modification timestamps
    cards_path = ROOT_DIR / "outputs" / "seven_pass_east_kalimantan" / "targets" / "site_cards.json"
    if cards_path.exists():
        mtime_before = cards_path.stat().st_mtime
        content_before = cards_path.read_text(encoding="utf-8")

        bundle = _recompute_targets_live(
            region="EAST_KALIMANTAN",
            l_threshold=0.70,
            min_area_ha=2.0,
            weights=(0.4, 0.25, 0.2, 0.15),
        )

        # Artifacts on disk must remain pristine
        assert cards_path.stat().st_mtime == mtime_before
        assert cards_path.read_text(encoding="utf-8") == content_before
        # Dynamic in-memory bundle must reflect preview
        assert "targets_geojson" in bundle
        assert "features" in bundle["targets_geojson"]

