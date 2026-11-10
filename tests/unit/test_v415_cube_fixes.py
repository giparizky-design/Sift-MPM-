"""v4.15 (audit v4.14 T18/T19): real-cube fixes - water gating, label-derived features, feature lists."""

import numpy as np
import pytest

from src.features.coal_and_multimodal_features import compute_metallic_and_coal_gated_optical
from src.features.real_sources.water import compute_mndwi, compute_water_mask
from src.schema.canonical import (
    EXTENDED_NON_FEATURE_COLS,
    LABEL_DERIVED_COLS,
    load_feature_list,
)


def test_water_mask_rules_and_nan_safety():
    ndvi = np.array([[-0.2, 0.1, np.nan, 0.5]], dtype=np.float32)
    mndwi = np.array([[np.nan, 0.3, np.nan, -0.4]], dtype=np.float32)
    dem = np.array([[5.0, 5.0, -3.0, 5.0]], dtype=np.float32)
    osm = np.array([[0, 0, 0, 1]])
    w = compute_water_mask(ndvi, mndwi=mndwi, dem=dem, osm_water=osm)
    assert w.tolist() == [[True, True, True, True]]
    dry = compute_water_mask(np.array([[0.1, np.nan]], dtype=np.float32), mndwi=np.array([[-0.1, np.nan]], dtype=np.float32),
                             dem=np.array([[10.0, 10.0]], dtype=np.float32))
    assert dry.tolist() == [[False, False]]  # NaN never counts as water


def test_mndwi_masks_unphysical_reflectance():
    g = np.array([0.10, 0.001, 0.2], dtype=np.float32)
    s = np.array([0.05, 0.05, 0.4], dtype=np.float32)
    out = compute_mndwi(g, s)
    assert out[0] == pytest.approx((0.10 - 0.05) / 0.15)
    assert np.isnan(out[1])
    assert out[2] < 0


def test_gated_optical_is_nan_on_water_cells():
    h, w = 6, 6
    ndvi = np.full((h, w), 0.05, dtype=np.float32)  # bare / sparse: passes the canopy gate
    water = np.zeros((h, w), dtype=bool)
    water[2:4, 2:4] = True
    kw = dict(
        s2_stack={"ndvi": ndvi, "albedo": np.full((h, w), 0.2, np.float32), "ficr": np.full((h, w), 1.1, np.float32),
                  "clay_hydroxyl": np.full((h, w), 1.2, np.float32), "ferrous_ultramafic": np.full((h, w), 2.0, np.float32)},
        aster_tir_silica=np.full((h, w), 1.1, np.float32), thermal_st=np.zeros((h, w), np.float32),
        w_canopy=np.ones((h, w), np.float32), river_mask=np.zeros((h, w), np.uint8), disturbance_mask=np.zeros((h, w), np.float32),
    )
    without = compute_metallic_and_coal_gated_optical(**kw)
    with_w = compute_metallic_and_coal_gated_optical(**kw, water_mask=water)
    assert np.isfinite(without["spec_clay_hydroxyl_gated"][water]).all()  # the old behaviour: water passed
    assert np.isnan(with_w["spec_clay_hydroxyl_gated"][water]).all()
    assert np.isfinite(with_w["spec_clay_hydroxyl_gated"][~water]).all()


def test_label_derived_columns_are_never_automatic_features():
    for col in LABEL_DERIVED_COLS:
        assert col in EXTENDED_NON_FEATURE_COLS
    for col in ("water_mask", "osm_water_mask", "osm_quarry_mask", "dist_roads_km", "dist_ports_km"):
        assert col in EXTENDED_NON_FEATURE_COLS


def test_load_feature_list_comments_blanks_and_errors(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("# header\nslope  # inline\n\naspect\n", encoding="utf-8")
    assert load_feature_list(f) == ["slope", "aspect"]
    (tmp_path / "dup.txt").write_text("a\nb\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_feature_list(tmp_path / "dup.txt")
    (tmp_path / "empty.txt").write_text("# nothing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_feature_list(tmp_path / "empty.txt")


# ---------------------------------------------------------------------------
# T20: ExplorationPropensity
# ---------------------------------------------------------------------------
import pandas as pd

from src.labels.pu_negatives import CANONICAL_PROPENSITY_COLS, ExplorationPropensity


def _access_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dist_roads_km": rng.uniform(0, 30, n),
        "dist_major_rivers_km": rng.uniform(0, 20, n),
        "dist_ports_km": rng.uniform(0, 80, n),
    })


def test_propensity_default_columns_are_roads_rivers_ports():
    assert CANONICAL_PROPENSITY_COLS == ["dist_roads_km", "dist_major_rivers_km", "dist_ports_km"]
    assert ExplorationPropensity().columns == CANONICAL_PROPENSITY_COLS


@pytest.mark.parametrize("bad", ["gfw_forest_disturbance", "dem_slope", "dem_elevation", "dist_haul_roads_km", "outcrop_exposure_index", "regolith_cover_m"])
def test_propensity_rejects_forbidden_covariates(bad):
    with pytest.raises(ValueError, match="FORBIDDEN"):
        ExplorationPropensity(columns=["dist_roads_km", bad])


def test_propensity_scores_new_rows_and_matches_in_sample():
    df = _access_frame()
    target = (df["dist_roads_km"] < 8).to_numpy()
    prop = ExplorationPropensity().fit(df, target)
    e_all = prop.score(df)
    e_head = prop.score(df.iloc[:50])
    np.testing.assert_allclose(e_all[:50], e_head)
    assert e_all.min() >= 0.05 and e_all.max() <= 1.0
    # closer to a road -> more likely explored
    assert e_all[df["dist_roads_km"].to_numpy() < 3].mean() > e_all[df["dist_roads_km"].to_numpy() > 20].mean()
    # a chunk with NaN is scored through the fitted imputer, not zero-filled
    chunk = df.iloc[:5].copy()
    chunk.loc[chunk.index[0], "dist_roads_km"] = np.nan
    assert np.isfinite(prop.score(chunk)).all()


def test_propensity_missing_column_and_degenerate_target():
    df = _access_frame()
    with pytest.raises(KeyError):
        ExplorationPropensity().fit(df.drop(columns=["dist_ports_km"]), np.zeros(len(df)))
    prop = ExplorationPropensity().fit(df, np.zeros(len(df)))
    assert prop.degenerate_
    assert np.allclose(prop.score(df), 0.05)  # base rate 0 clipped to the floor
    with pytest.raises(ValueError, match="no finite value"):
        bad = df.copy()
        bad["dist_ports_km"] = np.nan
        ExplorationPropensity().fit(bad, (df["dist_roads_km"] < 8).to_numpy())
