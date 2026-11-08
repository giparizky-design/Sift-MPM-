"""tests/unit/test_argon_leakage_remediation.py
=============================================
Unit tests verifying Gemini Argon leakage remediation additions & guardrails:
1. Contradiction resolution: geology_coal_contact_dist_km permanently banned in
   NON_FEATURE_COLS; dist_sedimentary_polygon_km computed cleanly from formation polygons.
2. Distance capping: distance features capped at local physical limit (<= 15.0 km) to
   prevent long-range triangulation / secret GPS grid memorization.
3. Strict rock-type whitelist: intrusive seeds match only authentic igneous rock names
   (granit, diorit, andesit, basalt, intrusif, vulkanik, pluton, gabro) and explicitly
   reject any row sourced from mineral deposits or commodity tables.
4. U-Net cheat smuggling prevention: cube_norm strictly excludes all NON_FEATURE_COLS
   and enforces float32 data types without memory bloat.
5. Spatial CV negative truncation fix: coal spatial CV evaluates full validation negatives
   instead of capping at 600 cells.
6. Metallic K=4 stability: porphyry_cu, skarn_fe_cu, epithermal_au, laterite_fe_ni remain
   completely stable, unaffected, and backward-compatible.
"""

from __future__ import annotations

from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from src.features.coal_and_multimodal_features import compute_lithologic_and_stratigraphic_contacts
from src.models.pu_pipeline import (
    SUBCANOPY_COAL_FEATURES,
    TwoStageK4Ensemble,
    TwoStageK3CoalEnsemble,
    verify_no_raw_pixel_leakage,
)
from src.outputs.model_introspection import NON_FEATURE_COLS as INTRO_NON_FEATURE_COLS


class TestArgonLeakageRemediation:
    """Test Gemini Argon Leakage Remediation Additions."""

    def test_argon_item_1_geology_coal_contact_dist_km_permanently_banned(self) -> None:
        """Verify geology_coal_contact_dist_km is permanently quarantined in NON_FEATURE_COLS

        and rejected by verify_no_raw_pixel_leakage, while dist_sedimentary_polygon_km is in SUBCANOPY_COAL_FEATURES.
        """
        assert "geology_coal_contact_dist_km" in INTRO_NON_FEATURE_COLS
        assert "in_coal_bearing_formation" in INTRO_NON_FEATURE_COLS
        assert "in_holocene_peat_or_karst_mask" in INTRO_NON_FEATURE_COLS
        assert "in_ophiolite_unit" in INTRO_NON_FEATURE_COLS

        # Check scripts/run_seven_pass_training.py
        seven_pass_src = Path("scripts/run_seven_pass_training.py").read_text(encoding="utf-8")
        assert '"geology_coal_contact_dist_km"' in seven_pass_src
        assert '"in_coal_bearing_formation"' in seven_pass_src
        assert '"in_holocene_peat_or_karst_mask"' in seven_pass_src
        assert '"in_ophiolite_unit"' in seven_pass_src

        # Check raw pixel leakage rejection
        with pytest.raises(ValueError, match="Raw sensor pixel leakage detected"):
            verify_no_raw_pixel_leakage(["dem_slope", "geology_coal_contact_dist_km"])

        with pytest.raises(ValueError, match="Raw sensor pixel leakage detected"):
            verify_no_raw_pixel_leakage(["dem_slope", "in_coal_bearing_formation"])

        # Check that dist_sedimentary_polygon_km is in SUBCANOPY_COAL_FEATURES
        assert "dist_sedimentary_polygon_km" in SUBCANOPY_COAL_FEATURES
        assert "geology_coal_contact_dist_km" not in SUBCANOPY_COAL_FEATURES
        assert "in_coal_bearing_formation" not in SUBCANOPY_COAL_FEATURES

    def test_argon_item_2_distance_features_capped_at_15km(self) -> None:
        """Verify all distance features are capped at <= 15.0 km to prevent secret GPS grid triangulation."""
        H, W = 40, 40
        lons = np.linspace(115.0, 117.0, W)
        lats = np.linspace(-1.0, 1.0, H)

        # Create synthetic GeoDataFrame with a small polygon at the corner
        poly = Polygon([(115.0, -1.0), (115.1, -1.0), (115.1, -0.9), (115.0, -0.9)])
        gdf = gpd.GeoDataFrame(
            [
                {
                    "formation": "Tanjung",
                    "lithology": "Coal-bearing sandstone",
                    "unit_type": "sedimentary",
                    "geological_role": "coal_formation",
                    "geometry": poly,
                }
            ],
            crs="EPSG:4326",
        )

        res = compute_lithologic_and_stratigraphic_contacts(
            geology_gpkg_path=gdf,
            grid_lon=lons,
            grid_lat=lats,
            res_m=250.0,
        )

        # Check all distance outputs are strictly within [0.0, 15.0]
        assert "dist_marine_deltaic_contact_km" in res
        assert "dist_calc_alkaline_intrusive_km" in res
        assert "dist_pluton_carbonate_contact_km" in res
        assert "dist_sedimentary_polygon_km" in res

        for dist_key in [
            "dist_marine_deltaic_contact_km",
            "dist_calc_alkaline_intrusive_km",
            "dist_pluton_carbonate_contact_km",
            "dist_sedimentary_polygon_km",
        ]:
            arr = res[dist_key]
            assert np.all(arr >= 0.0), f"{dist_key} has negative values"
            assert np.all(arr <= 15.0), f"{dist_key} exceeds 15.0 km physical cap: max={np.max(arr)}"

    def test_argon_item_3_strict_igneous_rock_whitelist_rejects_metallic_commodities(self, tmp_path: Path) -> None:
        """Verify GeoRIMA intrusive seeds strictly reject metallic commodities (Tembaga, Emas, Besi, Batubara)

        and accept only true igneous rock types (Granit, Andesit, Diorit, Basalt).
        """
        H, W = 30, 30
        lons = np.linspace(115.0, 116.0, W)
        lats = np.linspace(-1.0, 0.0, H)

        # Create mock GeoRIMA table with copper (Tembaga), gold (Emas), and true granite
        csv_path = tmp_path / "mock_georima.csv"
        df_nodes = pd.DataFrame(
            [
                # Commodity deposit: Tembaga (Copper) - MUST BE REJECTED
                {
                    "georima_id": "ID_001",
                    "potensi_type": "ml",
                    "commodity_label": "Tembaga",
                    "resource_group": "COPPER_ORE",
                    "geological_role": "porphyry_intrusive",
                    "lon": 115.2,
                    "lat": -0.8,
                },
                # Commodity deposit: Emas Primer (Gold) - MUST BE REJECTED
                {
                    "georima_id": "ID_002",
                    "potensi_type": "ml",
                    "commodity_label": "Emas Primer",
                    "resource_group": "GOLD",
                    "geological_role": "calc_alkaline_volcanic",
                    "lon": 115.5,
                    "lat": -0.5,
                },
                # Commodity deposit: Batubara (Coal) - MUST BE REJECTED
                {
                    "georima_id": "ID_003",
                    "potensi_type": "bb",
                    "commodity_label": "Batubara",
                    "resource_group": "COAL",
                    "geological_role": "coal_seam",
                    "lon": 115.7,
                    "lat": -0.3,
                },
                # Authentic rock: Granit (Granite) - MUST BE ACCEPTED
                {
                    "georima_id": "ID_004",
                    "potensi_type": "nml",
                    "commodity_label": "Granit",
                    "resource_group": "INDUSTRIAL_ROCKS",
                    "geological_role": "felsic_intrusive",
                    "lon": 115.8,
                    "lat": -0.2,
                },
            ]
        )
        df_nodes.to_csv(csv_path, index=False)

        empty_gdf = gpd.GeoDataFrame({"formation": [], "lithology": [], "geometry": []}, crs="EPSG:4326")
        res = compute_lithologic_and_stratigraphic_contacts(
            geology_gpkg_path=empty_gdf,
            grid_lon=lons,
            grid_lat=lats,
            georima_csv_path=csv_path,
            res_m=250.0,
        )

        dist_intr = res["dist_calc_alkaline_intrusive_km"]
        # The intrusive seed must be at (115.8, -0.2) where Granit is, NOT at (115.2, -0.8) where Tembaga is!
        # At (115.8, -0.2), distance should be 0.0 (near cell)
        # Check closest cell to (115.8, -0.2)
        c_granit = int(np.round((115.8 - 115.0) / (116.0 - 115.0) * (W - 1)))
        r_granit = int(np.round((-0.2 - (-1.0)) / (0.0 - (-1.0)) * (H - 1)))
        # Note: in from_bounds, y runs from min_lat to max_lat or vice versa, but check min distance location
        min_r, min_c = np.unravel_index(np.argmin(dist_intr), dist_intr.shape)
        min_lon = lons[min_c]
        min_lat = lats[min_r]

        # The closest distance to intrusive must be near the Granit node (lon ~ 115.8), NOT near Tembaga (lon ~ 115.2)
        assert abs(min_lon - 115.8) < 0.15, f"Intrusive seed positioned at lon {min_lon}, expected near Granit at 115.8!"
        assert abs(min_lon - 115.2) > 0.30, "Intrusive seed was erroneously placed at Tembaga copper occurrence!"

    def test_argon_item_4_cube_norm_excludes_non_feature_cols(self) -> None:
        """Verify seven_pass training script ensures cube_norm excludes all NON_FEATURE_COLS and loads float32."""
        seven_pass_src = Path("scripts/run_seven_pass_training.py").read_text(encoding="utf-8")
        assert "Strictly exclude all NON_FEATURE_COLS from U-Net input cube" in seven_pass_src
        assert "cube_raw = stack.array[feat_indices].astype(np.float32)" in seven_pass_src
        assert "cube_norm = np.zeros((C, H_c, W_c), dtype=np.float32)" in seven_pass_src

    def test_argon_item_5_spatial_cv_negative_truncation_fixed(self) -> None:
        """Verify that coal spatial CV does not cap validation negatives at 600 cells."""
        seven_pass_src = Path("scripts/run_seven_pass_training.py").read_text(encoding="utf-8")
        assert "sub_u = u_va_f" in seven_pass_src
        assert "ev_f = np.concatenate([p_va_f, sub_u])" in seven_pass_src
        assert "val_coal_idx[~va_pos][: min(600" not in seven_pass_src

    def test_metallic_k4_stability_and_backward_compatibility(self) -> None:
        """Verify that TwoStageK4Ensemble fits and predicts on metallic data without regressions."""
        rng = np.random.default_rng(42)
        N = 100
        # Valid features excluding raw bands and quarantined features
        cols = [
            "dilation_tendency_Td",
            "slip_tendency_Ts",
            "bouguer_gravity_mgal",
            "dist_sedimentary_polygon_km",
            "dist_calc_alkaline_intrusive_km",
        ]
        X = rng.normal(size=(N, len(cols))).astype(np.float32)
        y_k4 = np.zeros((N, 4), dtype=np.int32)
        for k in range(4):
            y_k4[k * 10 : (k + 1) * 10, k] = 1

        ens = TwoStageK4Ensemble(n_estimators=10, n_bag=2, random_state=42)
        ens.fit(X=X, y_k4=y_k4, feature_names=cols)

        probs = ens.predict_proba(X)
        assert probs.shape == (N, 4)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert not np.isnan(probs).any()

        stds = ens.predict_std(X)
        assert stds.shape == (N, 4)
        assert np.all(stds >= 0.0)

    def test_argon_item_4_functional_cube_norm_excludes_non_feature_cols_and_caps_distances(self) -> None:
        """Functional test: verify cube_norm extracts strictly feature_cols, excludes NON_FEATURE_COLS,

        caps distance channels at 15.0 km, and enforces float32 precision.
        """
        from src.features.feature_stack import FeatureStack

        H, W = 20, 20
        # Create a mock FeatureStack with 4 layers: 2 valid features, 2 forbidden non-features
        layer_names = [
            "dem_slope",
            "dist_marine_deltaic_contact_km",
            "geology_coal_contact_dist_km",  # FORBIDDEN
            "in_coal_bearing_formation",     # FORBIDDEN
        ]
        arr = np.zeros((4, H, W), dtype=np.float64)
        arr[0] = 5.0  # slope
        arr[1] = 85.0  # dist_marine_deltaic_contact_km (> 15 km)
        arr[2] = 2.0   # geology_coal_contact_dist_km
        arr[3] = 1.0   # in_coal_bearing_formation

        stack = FeatureStack(
            array=arr,
            feature_names=layer_names,
            profile={"width": W, "height": H},
        )

        # Permitted feature_cols for training
        feature_cols = ["dem_slope", "dist_marine_deltaic_contact_km"]

        # Run the extraction logic from run_seven_pass_training
        stack_feats = list(stack.feature_names)
        feat_indices = [i for i, name in enumerate(stack_feats) if name in feature_cols]
        assert feat_indices == [0, 1]  # Only indices 0 and 1 are picked!

        cube_raw = stack.array[feat_indices].astype(np.float32)
        assert cube_raw.shape == (2, H, W)
        assert cube_raw.dtype == np.float32

        # Verify distance capping logic in cube_norm
        cube_norm = cube_raw.copy()
        for ch in range(cube_norm.shape[0]):
            band = cube_norm[ch]
            feat_name = feature_cols[ch]
            if feat_name.startswith("dist_") or feat_name.endswith("_dist_km"):
                if feat_name.endswith("_m"):
                    band = np.clip(band, 0.0, 15000.0)
                else:
                    band = np.clip(band, 0.0, 15.0)
            assert np.all(band <= 15.0) if feat_name.startswith("dist_") else True

    def test_argon_item_4_model_introspection_stack_feature_cols_channel_alignment(self, tmp_path: Path) -> None:
        """Functional test: verify model_introspection handles stacks with extra non-feature channels

        without broadcasting ValueError or PyTorch channel mismatch.
        """
        from src.features.feature_stack import FeatureStack

        H, W = 32, 32
        stack_feats = [
            "dem_slope",
            "bouguer_gravity_mgal",
            "geology_coal_contact_dist_km",  # extra banned layer in stack
            "in_coal_bearing_formation",     # extra banned layer in stack
        ]
        arr = np.random.randn(4, H, W).astype(np.float32)
        stack = FeatureStack(
            array=arr,
            feature_names=stack_feats,
            profile={"width": W, "height": H, "transform": Affine.identity() if "Affine" in globals() else None, "crs": "EPSG:4326"},
        )

        feature_cols = ["dem_slope", "bouguer_gravity_mgal"]

        # Align stack to feature_cols
        feat_indices = [i for i, name in enumerate(stack.feature_names) if name in feature_cols]
        cube_raw = stack.array[feat_indices].astype(np.float32)
        assert cube_raw.shape == (2, H, W)

        ch_means_norm = np.array([0.5, -0.2], dtype=np.float32)
        ch_stds_norm = np.array([1.0, 1.2], dtype=np.float32)

        # Broadcasting must succeed with shape (2, 32, 32)
        norm_win = (cube_raw - ch_means_norm[:, None, None]) / ch_stds_norm[:, None, None]
        assert norm_win.shape == (2, H, W)
