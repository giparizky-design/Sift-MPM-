"""Unit tests for the v4.8 "find new ground" changes (audit v4.8).

Covers:
- coal neutral unlabeled weighting (Hájek-normalised positives, weight 1 on unlabeled cells),
- coal negative exclusion mask (known-coal disk cells never act as negatives),
- the legacy coal inverse-propensity weighting is unchanged by default,
- the rank softmax gets logits, not probabilities,
- the standardised OUTMOGC gate is invariant to the units of a column,
- the porphyry head without skarn/epithermal stacking,
- the min_child_weight override,
- greenfield / brownfield target split and its site-card fields.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from scripts.run_seven_pass_training import (
    GENERALISATION_DEFAULTS,
    GENERALISATION_LEGACY_V47,
    _prob_to_logit,
)
from src.labels.pu_negatives import OUTMOGCGate
from src.models.coal import TwoStageK3CoalEnsemble, compute_hierarchical_coal_probabilities
from src.models.metals import TwoStageK4Ensemble
from src.outputs.prospectivity_map import emit_site_cards_json, split_targets_by_known_ground


def _coal_labels(n: int, rng: np.random.Generator) -> np.ndarray:
    y = -np.ones((n, 3), dtype=np.int32)
    for c in range(3):
        y[rng.choice(n, size=20, replace=False), c] = 1
    return y


def test_coal_neutral_weights_unlabeled_one_and_positives_hajek():
    rng = np.random.default_rng(0)
    n = 400
    y = _coal_labels(n, rng)
    e = rng.uniform(0.05, 1.0, size=n)
    ens = TwoStageK3CoalEnsemble(unlabeled_weighting="neutral")
    targets, weights = ens._extract_coal_labels_and_weights(y_k3=y, tonnage_weights=None, e_pooled=e, N=n)
    for c in range(3):
        pos = targets[c] == 1
        assert np.allclose(weights[c][~pos], 1.0)
        assert np.mean(weights[c][pos]) == pytest.approx(1.0)
        # remote positives (low e) still weigh more than accessible ones
        assert np.corrcoef(weights[c][pos], 1.0 / np.clip(e[pos], 0.05, 1.0))[0, 1] > 0.99


def test_coal_legacy_weighting_unchanged_by_default():
    rng = np.random.default_rng(1)
    n = 300
    y = _coal_labels(n, rng)
    e = rng.uniform(0.05, 1.0, size=n)
    ens = TwoStageK3CoalEnsemble()
    assert ens.unlabeled_weighting == "inverse_propensity"
    targets, weights = ens._extract_coal_labels_and_weights(y_k3=y, tonnage_weights=None, e_pooled=e, N=n)
    inv = 1.0 / np.clip(e, 0.05, 1.0)
    inv = inv / inv.mean()
    for c in range(3):
        assert np.allclose(weights[c], inv)


def test_coal_negative_exclusion_zeroes_unlabeled_only():
    rng = np.random.default_rng(2)
    n = 300
    y = _coal_labels(n, rng)
    excl = np.zeros(n, dtype=bool)
    excl[:150] = True
    ens = TwoStageK3CoalEnsemble(unlabeled_weighting="neutral")
    targets, weights = ens._extract_coal_labels_and_weights(
        y_k3=y, tonnage_weights=None, e_pooled=None, N=n, negative_exclusion_mask=excl)
    for c in range(3):
        pos = targets[c] == 1
        assert np.all(weights[c][excl & ~pos] == 0.0)
        assert np.all(weights[c][pos] > 0.0)
        assert np.all(weights[c][~excl & ~pos] == 1.0)


def test_coal_invalid_weighting_rejected():
    with pytest.raises(ValueError):
        TwoStageK3CoalEnsemble(unlabeled_weighting="bogus")


def test_coal_fit_accepts_exclusion_and_min_child_weight():
    rng = np.random.default_rng(3)
    n, d = 600, 6
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = _coal_labels(n, rng)
    ens = TwoStageK3CoalEnsemble(n_estimators=5, n_bag=1, unlabeled_weighting="neutral", min_child_weight=7.0)
    ens.fit(X, y, e_pooled=np.full(n, 0.3), negative_exclusion_mask=np.arange(n) < 100,
            feature_names=[f"f{i}" for i in range(d)])
    assert ens.subcanopy_heads_[0].min_child_weight == 7.0
    out = ens.predict_full_outputs(X)
    assert out["p_coal_any"].shape == (n,)


def test_rank_softmax_on_logits_is_not_flattened():
    p = np.array([[0.90, 0.05, 0.02], [0.10, 0.60, 0.05]])
    p_any = np.array([0.9, 0.6])
    gate = np.ones(2)
    split_prob, _ = compute_hierarchical_coal_probabilities(p_any, p, gate)
    split_logit, total = compute_hierarchical_coal_probabilities(p_any, _prob_to_logit(p), gate)
    share_prob = split_prob[0, 0] / split_prob[0].sum()
    share_logit = split_logit[0, 0] / split_logit[0].sum()
    assert share_prob < 0.55          # softmax of probabilities is nearly uniform
    assert share_logit > 0.95         # softmax of logits keeps the rank decision
    assert np.all(split_logit.sum(axis=1) <= total + 1e-9)
    assert np.array_equal(np.argmax(split_logit, axis=1), np.argmax(p, axis=1))


def test_prob_to_logit_is_finite_at_bounds():
    v = _prob_to_logit(np.array([0.0, 0.5, 1.0]))
    assert np.all(np.isfinite(v)) and v[1] == pytest.approx(0.0)


def test_standardized_gate_is_unit_invariant():
    rng = np.random.default_rng(4)
    pos = rng.normal(size=(60, 5))
    ref = rng.normal(scale=2.0, size=(2000, 5))
    query = rng.normal(scale=2.0, size=(500, 5))
    scale = np.array([1.0, 1.0, 1000.0, 1.0, 1.0])  # e.g. a distance in metres instead of km

    g1 = OUTMOGCGate(n_pca_components=4, n_gmm_components=1, standardize=True).fit(pos, X_reference=ref)
    g2 = OUTMOGCGate(n_pca_components=4, n_gmm_components=1, standardize=True).fit(pos * scale, X_reference=ref * scale)
    assert np.allclose(g1.score(query), g2.score(query * scale), atol=1e-6)

    r1 = OUTMOGCGate(n_pca_components=2, n_gmm_components=1).fit(pos)
    r2 = OUTMOGCGate(n_pca_components=2, n_gmm_components=1).fit(pos * scale)
    assert not np.allclose(r1.score(query), r2.score(query * scale), atol=1e-3)


def test_legacy_gate_pickle_without_scaler_attrs_still_scores():
    rng = np.random.default_rng(5)
    g = OUTMOGCGate(n_pca_components=2, n_gmm_components=1).fit(rng.normal(size=(30, 4)))
    for a in ("standardize", "scaler_mean_", "scaler_std_"):
        delattr(g, a)
    assert g.score(rng.normal(size=(10, 4))).shape == (10,)


def _k4_data(rng, n=800, d=5):
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = -np.ones((n, 4), dtype=np.int32)
    for k in range(4):
        y[rng.choice(n, size=25, replace=False), k] = 1
    return X, y


def test_head0_without_stacking_uses_base_features():
    rng = np.random.default_rng(6)
    X, y = _k4_data(rng)
    names = [f"f{i}" for i in range(X.shape[1])]
    ens = TwoStageK4Ensemble(n_estimators=5, n_bag=1, head0_stacking=False, min_child_weight=4.0)
    ens.fit(X, y, feature_names=names, spatial_block_ids=np.arange(len(X)) % 5)
    assert ens.head_0_.models_[0].n_features_in_ == X.shape[1]
    assert ens.head_1_.min_child_weight == 4.0 and ens.head_0_.min_child_weight == 4.0
    p = ens.predict_proba(X)
    assert p.shape == (len(X), 4) and np.all(np.isfinite(p))
    assert ens.predict_std(X).shape == (len(X), 4)
    _, fnames = ens.compute_head_shap(0, X[:3], names)
    assert fnames == names


def test_head0_stacking_default_keeps_meta_features():
    rng = np.random.default_rng(7)
    X, y = _k4_data(rng)
    ens = TwoStageK4Ensemble(n_estimators=5, n_bag=1)
    ens.fit(X, y, spatial_block_ids=np.arange(len(X)) % 5)
    assert ens.head_0_.models_[0].n_features_in_ == X.shape[1] + 2
    assert ens.head_1_.min_child_weight == 1.0 and ens.head_0_.min_child_weight == 3.0


def _target_fixture():
    H = W = 100
    profile = {"transform": Affine(250.0, 0, 0, 0, -250.0, 25_000.0), "crs": "EPSG:32750"}
    known = np.zeros((H, W), dtype=bool)
    known[10:12, 10:12] = True  # known deposit near the top-left corner
    # polygon A touches the known deposit; polygon B sits ~20 km away
    a = box(2_500, 25_000 - 3_500, 3_500, 25_000 - 2_500)
    b = box(20_000, 25_000 - 21_000, 21_000, 25_000 - 20_000)
    c = box(22_000, 25_000 - 23_000, 23_000, 25_000 - 22_000)
    gdf = gpd.GeoDataFrame(
        {"rank_score": [0.9, 0.5, 0.4], "rank": [1, 2, 3], "target_id": ["T1", "T2", "T3"],
         "area_sq_km": [1.0, 1.0, 1.0], "calibrated_deposit_prob": [0.9, 0.5, 0.4]},
        geometry=[a, b, c], crs="EPSG:32750")
    return gdf, known, profile


def test_split_targets_by_known_ground_classes_and_caps():
    gdf, known, profile = _target_fixture()
    out = split_targets_by_known_ground(gdf, known, profile, greenfield_km=5.0, top_n_each=1)
    assert list(out["exploration_class"]) == ["brownfield", "greenfield"]
    assert list(out["target_id"]) == ["T1", "T2"]  # T3 is the 2nd greenfield, cut by top_n_each=1
    assert list(out["_source_pos"]) == [0, 1]
    assert out.loc[0, "dist_to_known_km"] < 1.0 and out.loc[1, "dist_to_known_km"] > 15.0
    full = split_targets_by_known_ground(gdf, known, profile, greenfield_km=5.0, top_n_each=0)
    assert list(full["class_rank"]) == [1, 1, 2]


def test_split_targets_handles_empty_and_no_known():
    gdf, known, profile = _target_fixture()
    empty = split_targets_by_known_ground(gdf.iloc[0:0], known, profile)
    assert len(empty) == 0 and "exploration_class" in empty.columns
    none_known = split_targets_by_known_ground(gdf, np.zeros_like(known), profile)
    assert set(none_known["exploration_class"]) == {"greenfield"}


def test_site_cards_carry_exploration_class(tmp_path):
    gdf, known, profile = _target_fixture()
    out = split_targets_by_known_ground(gdf, known, profile).drop(columns="_source_pos")
    cards = emit_site_cards_json(out, out_path=str(tmp_path / "cards.json"))
    assert [c["exploration_class"] for c in cards] == ["brownfield", "greenfield", "greenfield"]
    assert cards[1]["class_rank"] == 1 and cards[2]["class_rank"] == 2
    assert cards[1]["dist_to_known_km"] > 15.0
    json.loads((tmp_path / "cards.json").read_text())


def test_generalisation_defaults_have_every_switch():
    assert set(GENERALISATION_DEFAULTS) == {
        "weight_scheme", "gate_mode", "head0_stacking", "min_child_weight_metals",
        "coal_disk_mode", "coal_buffer_m", "drop_features", "drop_features_metals", "k4_exclusion_scale",
        # v4.15 additions
        "feature_list_metals", "feature_list_coal", "cluster_integrity", "split_buffer_m",
    }
    assert set(GENERALISATION_LEGACY_V47) == set(GENERALISATION_DEFAULTS)


def test_metals_exclude_features_selects_columns_everywhere():
    rng = np.random.default_rng(8)
    X, y = _k4_data(rng, d=6)
    names = [f"f{i}" for i in range(6)]
    ens = TwoStageK4Ensemble(n_estimators=5, n_bag=1, exclude_features=("f1", "f4"))
    ens.fit(X, y, feature_names=names, spatial_block_ids=np.arange(len(X)) % 5)
    assert ens.feature_names_ == names                      # full input schema kept
    assert ens.model_feature_names_ == ["f0", "f2", "f3", "f5"]
    assert ens.head_1_.models_[0].n_features_in_ == 4
    p_full = ens.predict_proba(X)
    p_sel = ens.predict_proba(X[:, [0, 2, 3, 5]])
    assert np.allclose(p_full, p_sel)
    assert ens.predict_std(X).shape == (len(X), 4)
    _, fnames = ens.compute_head_shap(1, X[:2], names)
    assert fnames == ["f0", "f2", "f3", "f5"]


def test_gate_feature_index_matches_manual_subset():
    rng = np.random.default_rng(9)
    pos, ref, q = rng.normal(size=(40, 6)), rng.normal(size=(500, 6)), rng.normal(size=(100, 6))
    idx = [0, 2, 3, 5]
    g1 = OUTMOGCGate(n_pca_components=3, n_gmm_components=1, standardize=True, feature_index=idx).fit(pos, X_reference=ref)
    g2 = OUTMOGCGate(n_pca_components=3, n_gmm_components=1, standardize=True).fit(pos[:, idx], X_reference=ref[:, idx])
    assert np.allclose(g1.score(q), g2.score(q[:, idx]))
