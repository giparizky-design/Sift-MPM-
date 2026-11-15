"""v4.15 (audit v4.14 T17): metal labels come from deposit records only."""

import numpy as np
import pandas as pd

from src.labels.borneo_labels import (
    EXCLUDED_DEPOSIT_SITES,
    build_k4_deposit_targets,
    filter_valid_ore_candidates,
    load_borneo_vector_sources,
    mrds_has_code,
)


def test_mrds_code_matching_is_whole_token():
    codes = pd.Series(["CU AU PB ZN", "CUR", "SCU", "AU", None, "gem_d au"])
    assert mrds_has_code(codes, "CU").tolist() == [True, False, False, False, False, False]
    assert mrds_has_code(codes, "AU").tolist() == [True, False, False, True, False, True]


def test_filter_drops_flagged_placers_named_sites_and_artifacts():
    df = pd.DataFrame({
        "site_name": ["Mamut", "Busang", "Riam Kanan River", "Kelian", "Centroid"],
        "commodity_label": ["Mamut", "Busang", "Riam Kanan River", "Kelian", "Centroid"],
        "is_placer": [False, False, True, False, False],
        "is_centroid_artifact": [False, False, False, False, True],
    })
    kept = filter_valid_ore_candidates(df)
    assert kept["site_name"].tolist() == ["Mamut", "Kelian"]
    assert "busang" in EXCLUDED_DEPOSIT_SITES


def test_filter_still_drops_emas_alluvial_without_is_placer_column():
    df = pd.DataFrame({"commodity_label": ["Emas Primer", "Emas Alluvial"]})
    assert filter_valid_ore_candidates(df)["commodity_label"].tolist() == ["Emas Primer"]


def test_real_sources_flag_the_two_mrds_placers():
    mrds = load_borneo_vector_sources()["mrds"]
    placers = set(mrds.loc[mrds["is_placer"], "site_name"])
    assert {"Riam Kanan River", "Cempaka-Danau Seran Paleochannels"} <= placers
    assert "Kelian Gold Deposit" not in placers


def test_head_contents_are_deposit_records_only():
    src = load_borneo_vector_sources()
    t = build_k4_deposit_targets(src["georima"], src["iron"], src["mrds"], src["coal"])

    # Head 0 sources are GeoRIMA COPPER_ORE nodes and MRDS Cu records, nothing else
    geo = src["georima"]
    cu = geo[geo["resource_group"] == "COPPER_ORE"]
    k0 = t["positives_k0"]
    geo_pts = {(round(g.x, 6), round(g.y, 6)) for g in k0[k0["source"] == "georima"].geometry}
    cu_pts = {(round(x, 6), round(y, 6)) for x, y in zip(cu["lon"], cu["lat"])}
    assert geo_pts and geo_pts <= cu_pts
    ind = geo[(geo["resource_group"] == "INDUSTRIAL_LITHOLOGY") | (geo["resource_group"] == "BASE_METALS")]
    bad = {(round(x, 6), round(y, 6)) for x, y in zip(ind["lon"], ind["lat"])}
    assert not (geo_pts & bad - cu_pts)

    # Head 1 is ESDM primary iron only (no marble)
    assert set(t["positives_k1"]["source"]) == {"esdm_iron"}

    # Head 2 has no placers and no Busang
    k2 = t["positives_k2"]
    mrds_pts = {(round(g.x, 4), round(g.y, 4)) for g in k2[k2["source"] == "mrds_gold"].geometry}
    m = src["mrds"]
    excluded = m[m["is_placer"] | m["site_name"].str.lower().isin(set(EXCLUDED_DEPOSIT_SITES))]
    excl_pts = {(round(g.x, 4), round(g.y, 4)) for g in excluded.geometry}
    assert not (mrds_pts & excl_pts)
