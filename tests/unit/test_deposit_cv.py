"""Tests for deposit-level spatial CV (src/evaluation/deposit_cv.py)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.deposit_cv import (
    assign_cluster_folds,
    deposit_area_fractions,
    label_deposit_clusters,
    success_rate_curve,
    summarize_clusters,
    summarize_success_rate,
    training_exclusion_mask,
)


def _three_disks():
    pos = np.zeros((60, 60), bool)
    pos[5:8, 5:8] = True       # cluster A, 9 cells
    pos[30:32, 30:32] = True   # cluster B, 4 cells
    pos[50, 10] = True         # cluster C, 1 cell
    return pos


def test_clusters_count_connected_components():
    cl = label_deposit_clusters(_three_disks())
    assert cl.max() == 2
    assert (cl == -1).sum() == 60 * 60 - 14
    summ = summarize_clusters(cl)
    assert sorted(summ["n_cells"].tolist()) == [1, 4, 9]


def test_diagonal_neighbours_join_only_with_8_connectivity():
    pos = np.zeros((5, 5), bool)
    pos[1, 1] = pos[2, 2] = True
    assert label_deposit_clusters(pos, 8).max() == 0
    assert label_deposit_clusters(pos, 4).max() == 1


def test_summarize_clusters_head_membership():
    pos = _three_disks()
    cl = label_deposit_clusters(pos)
    heads = np.zeros((2, 60, 60), bool)
    heads[0, 5, 5] = True
    heads[1, 30:32, 30:32] = True
    s = summarize_clusters(cl, heads, ["a", "b"]).set_index("n_cells")
    assert bool(s.loc[9, "has_a"]) and not bool(s.loc[9, "has_b"])
    assert bool(s.loc[4, "has_b"]) and not bool(s.loc[4, "has_a"])
    assert not bool(s.loc[1, "has_a"]) and not bool(s.loc[1, "has_b"])


def test_folds_keep_same_block_clusters_together_and_cover_all():
    df = pd.DataFrame({
        "cluster_id": range(6),
        "centroid_row": [1, 2, 25, 26, 55, 56],
        "centroid_col": [1, 3, 25, 27, 55, 57],
    })
    folds = assign_cluster_folds(df, block_px=10, n_folds=2, random_state=0)
    assert folds[0] == folds[1] and folds[2] == folds[3] and folds[4] == folds[5]
    assert set(folds) == {0, 1}
    lobo = assign_cluster_folds(df, block_px=10, n_folds=0)
    assert len(set(lobo)) == 3


def test_exclusion_mask_buffer_radius():
    cl = label_deposit_clusters(_three_disks())
    a_id = int(cl[6, 6])
    ex = training_exclusion_mask(cl, [a_id], buffer_px=3)
    assert ex[6, 6] and ex[5, 10] and not ex[5, 12]
    assert not ex[31, 31]  # other clusters stay trainable
    ex0 = training_exclusion_mask(cl, [a_id], buffer_px=0)
    assert ex0.sum() == 9


def test_area_fraction_extremes_and_ties():
    bg = np.arange(100, dtype=float)
    af = deposit_area_fractions(bg, [1000.0, -1.0, 49.5])
    assert af[0] == 0.0 and af[1] == 1.0 and af[2] == pytest.approx(0.5)
    af_tie = deposit_area_fractions(np.zeros(10), [0.0])
    assert af_tie[0] == pytest.approx(0.5)
    assert deposit_area_fractions(bg, [np.nan])[0] == 1.0


def test_random_scores_give_auc_near_half():
    rng = np.random.default_rng(1)
    bg = rng.random(20_000)
    dep = rng.random(400)
    s = summarize_success_rate(deposit_area_fractions(bg, dep), n_null=20_000)
    assert abs(s.auc - 0.5) < 0.05
    assert s.p_value_vs_random > 0.01


def test_perfect_model_is_significant():
    s = summarize_success_rate(np.full(10, 0.001), n_null=20_000)
    assert s.auc > 0.99
    assert s.p_value_vs_random < 1e-3
    assert s.capture["0.01"]["captured"] == 10
    assert s.capture["0.10"]["p_value"] < 1e-6


def test_insufficient_data_flag():
    s = summarize_success_rate(np.array([0.1, 0.2, 0.3]))
    assert s.insufficient_data and s.auc is None
    d = s.to_dict()
    assert d["insufficient_data"] is True and d["success_rate_auc"] is None


def test_runner_bounds_parsing_and_masking():
    from scripts.run_deposit_cv import in_bounds, parse_bounds

    assert parse_bounds("") is None
    b = parse_bounds("119.5,3.2,114.5,-2.8")  # corners in any order
    assert b == (114.5, -2.8, 119.5, 3.2)
    x = np.array([114.0, 115.0, 119.5, 120.0])
    y = np.array([0.0, 0.0, 3.2, 0.0])
    assert in_bounds(x, y, b).tolist() == [False, True, True, False]
    assert in_bounds(x, y, None).all()
    with pytest.raises(SystemExit):
        parse_bounds("1,2,3")


def test_success_rate_curve_monotone():
    c = success_rate_curve(np.array([0.05, 0.2, 0.6]))
    v = c["deposits_captured"].to_numpy()
    assert v[0] == 0.0 and v[-1] == 1.0 and np.all(np.diff(v) >= 0)


def _coal_comp(rng, n=200):
    return {"domain": "coal", "raw": rng.random((n, 3)), "basin_gate": rng.random(n),
            "c_hat": {0: 0.3, 1: 0.5, 2: 0.8}}


def test_unet_blend_weight_zero_reproduces_xgb_and_one_gives_unet():
    from scripts.run_deposit_cv import blend_outputs

    rng = np.random.default_rng(0)
    comp = _coal_comp(rng)
    s_u = rng.random((200, 3))
    b0 = blend_outputs(comp, s_u, 0.0)
    np.testing.assert_allclose(b0["raw"][:, :3], comp["raw"])
    # gated = coal ensemble calibration (0.65 p + 0.35 min(1, p/c)) times the basin gate
    c = np.array([0.3, 0.5, 0.8])
    cal = 0.65 * comp["raw"] + 0.35 * np.minimum(1.0, comp["raw"] / c)
    np.testing.assert_allclose(b0["gated"][:, :3], cal * comp["basin_gate"][:, None])
    np.testing.assert_allclose(b0["gated"][:, 3], (1 - np.prod(1 - cal, axis=1)) * comp["basin_gate"])
    b1 = blend_outputs(comp, s_u, 1.0)
    np.testing.assert_allclose(b1["raw"][:, :3], s_u)
    # per-head weights
    bh = blend_outputs(comp, s_u, [0.0, 1.0, 0.5])
    np.testing.assert_allclose(bh["raw"][:, 2], 0.5 * comp["raw"][:, 2] + 0.5 * s_u[:, 2])


def test_unet_blend_metals_zero_positive_head_stays_zero():
    from scripts.run_deposit_cv import blend_outputs

    rng = np.random.default_rng(1)
    comp = {"domain": "metals", "raw": rng.random((50, 4)), "gates": [None] * 4,
            "c_hat": {k: 0.35 for k in range(4)}, "n_train_pos": np.array([5, 5, 5, 0])}
    out = blend_outputs(comp, rng.random((50, 4)), 0.4)
    assert np.all(out["calibrated"][:, 3] == 0.0)
    assert out["raw"].shape == (50, 5) and np.allclose(out["raw"][:, 4], out["raw"][:, :4].max(axis=1))


def test_compare_pairs_across_models(tmp_path):
    import json

    from scripts.compare_deposit_cv import main as compare_main

    rng = np.random.default_rng(2)
    a = rng.random(20)
    for name, model, variant, af in (("b", "xgb_prod", "calibrated", a), ("c", "unet", "srauc_calibrated", a * 0.5)):
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame({"model": model, "variant": variant, "target": "any", "cluster_id": np.arange(20),
                      "area_fraction": af}).to_csv(d / "per_deposit.csv", index=False)
        (d / "deposit_cv_summary.json").write_text(json.dumps({"novelty": {}}))
    res = compare_main([str(tmp_path / "b"), str(tmp_path / "c"), "--cand-model", "unet",
                        "--pairs", "calibrated:srauc_calibrated"])
    r = res["xgb_prod/calibrated->unet/srauc_calibrated/any"]
    assert r["n"] == 20 and r["delta_auc"] == pytest.approx(float(np.mean(a * 0.5)))
