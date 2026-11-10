"""v4.9 guards: input layers must be real or documented functions of real layers."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import Affine

from src.features.real_sources.core import (
    ALLOW_UNVERIFIED_ENV,
    GridSpec,
    LayerProvenance,
    MEASURED,
    ProvenanceManifest,
    RealDataUnavailableError,
    assert_feature_provenance,
    decimation_factor,
)
from src.features.real_sources.sentinel2 import compute_indices

ROOT = Path(__file__).resolve().parents[2]

# Modules that build model input layers. None of them may draw random numbers.
LAYER_BUILDERS = [
    "src/features/real_sources/core.py",
    "src/features/real_sources/sentinel2.py",
    "src/features/real_sources/landsat_thermal.py",
    "src/features/real_sources/aster_tir.py",
    "src/features/real_sources/global_grids.py",
    "src/features/real_sources/osm_transport.py",
    "src/features/coal_and_multimodal_features.py",
    "src/features/sar_palsar2.py",
    "src/features/geophysics.py",
    "src/features/structural.py",
    "scripts/build_real_feature_cubes.py",
    "scripts/fetch_real_layers.py",
    "scripts/harvest_kalimantan_multimodal.py",
    "scripts/run_indonesia_hd_upgrade.py",
]


@pytest.mark.parametrize("rel", LAYER_BUILDERS)
def test_layer_builders_draw_no_random_numbers(rel: str) -> None:
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("default_rng", "RandomState"):
            raise AssertionError(f"{rel}:{node.lineno} uses np.random.{node.attr}")
        if isinstance(node, ast.Attribute) and node.attr == "random" and isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy"):
            raise AssertionError(f"{rel}:{node.lineno} uses np.random")


def _grid() -> GridSpec:
    return GridSpec(crs="EPSG:4326", transform=Affine(0.0025, 0, 114.5, 0, -0.0025, 3.2), height=40, width=50)


def test_manifest_requires_source_and_known_inputs(tmp_path: Path) -> None:
    m = ProvenanceManifest(_grid())
    with pytest.raises(ValueError):
        LayerProvenance(name="x", kind=MEASURED, source="")
    with pytest.raises(ValueError):
        LayerProvenance(name="x", kind="invented", source="s")
    with pytest.raises(KeyError):
        m.derive("ndvi", ["s2_b08"], "nd")  # input not registered
    m.add(LayerProvenance(name="s2_b08", kind=MEASURED, source="S2"))
    m.add(LayerProvenance(name="s2_b04", kind=MEASURED, source="S2"))
    m.derive("ndvi", ["s2_b08", "s2_b04"], "nd")
    m.write(tmp_path)
    back = ProvenanceManifest.read(tmp_path)
    assert back is not None and set(back.layers) == {"s2_b08", "s2_b04", "ndvi"}
    assert json.loads((tmp_path / "layer_provenance.json").read_text())["layers"]["ndvi"]["kind"] == "derived"


def test_training_gate_refuses_unverified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOW_UNVERIFIED_ENV, raising=False)
    with pytest.raises(RealDataUnavailableError):
        assert_feature_provenance(tmp_path, ["a"])  # no manifest
    m = ProvenanceManifest(_grid())
    m.add(LayerProvenance(name="a", kind=MEASURED, source="S"))
    m.write(tmp_path)
    assert_feature_provenance(tmp_path, ["a"])
    with pytest.raises(RealDataUnavailableError):
        assert_feature_provenance(tmp_path, ["a", "b"])
    monkeypatch.setenv(ALLOW_UNVERIFIED_ENV, "1")
    assert_feature_provenance(tmp_path, ["a", "b"])  # downgraded to a warning


def test_s2_indices_on_known_reflectance() -> None:
    b = {k: np.array([[v]], np.float32) for k, v in dict(s2_b02=0.05, s2_b03=0.08, s2_b04=0.10, s2_b05=0.15, s2_b08=0.30, s2_b8a=0.32, s2_b11=0.25, s2_b12=0.20).items()}
    out = compute_indices(b)
    assert out["spec_ndvi"][0, 0] == pytest.approx(0.5)
    assert out["spec_ndre"][0, 0] == pytest.approx(0.15 / 0.45)
    assert out["spec_ficr"][0, 0] == pytest.approx(2.0)
    assert out["spec_clay_hydroxyl"][0, 0] == pytest.approx(1.25)
    assert out["spec_ferrous_ultramafic"][0, 0] == pytest.approx(0.20 / 0.32 + 0.08 / 0.10)
    nan_b = {k: np.full((1, 1), np.nan, np.float32) for k in b}
    assert all(np.isnan(v).all() for v in compute_indices(nan_b).values())


def test_s2_indices_mask_dark_reflectance() -> None:
    """Near-zero/negative reflectance must give NaN, not ratios like NDVI = -5e5."""
    vals = dict(s2_b02=0.05, s2_b03=0.08, s2_b04=-0.0101, s2_b05=0.15, s2_b08=0.0100, s2_b8a=0.32, s2_b11=0.25, s2_b12=0.20)
    out = compute_indices({k: np.array([[v]], np.float32) for k, v in vals.items()})
    for k in ("spec_ndvi", "spec_ficr", "spec_albedo", "spec_ferrous_ultramafic"):
        assert np.isnan(out[k][0, 0]), k  # these use B04
    assert np.isfinite(out["spec_clay_hydroxyl"][0, 0])  # B11/B12 unaffected


def test_landsat_cold_cloud_residue_masked(tmp_path: Path) -> None:
    from src.features.real_sources.core import save_layer
    from src.features.real_sources.landsat_thermal import build_landsat_thermal_layers

    g = _grid()
    lst = np.full(g.shape, 31.0, np.float32)
    lst[5, 5] = -15.0  # cold cloud top
    lst[6, 6] = 12.0   # high summit: 12 C + 6.5 K/km * 2.5 km = 28 C at sea level, real
    dem = np.zeros(g.shape, np.float32)
    dem[6, 6] = 2500.0
    save_layer(lst, g, tmp_path / "thermal_lst_c.tif")
    m = ProvenanceManifest(g)
    m.add(LayerProvenance(name="dem_elevation", kind=MEASURED, source="DEM"))
    out = build_landsat_thermal_layers(g, m, tmp_path, dem=dem)
    assert np.isnan(out["thermal_lst_c"][5, 5]) and np.isnan(out["thermal_anomaly_st"][5, 5])
    assert out["thermal_lst_c"][6, 6] == pytest.approx(12.0)
    assert np.isfinite(out["thermal_anomaly_st"]).sum() == g.height * g.width - 1


def test_decimation_factor() -> None:
    assert decimation_factor(10.0, 250.0) == 16
    assert decimation_factor(20.0, 160.0) == 8
    assert decimation_factor(30.0, 240.0) == 8
    assert decimation_factor(300.0, 250.0) == 1


def test_gated_optical_does_not_fill_missing_indices() -> None:
    from src.features.coal_and_multimodal_features import compute_metallic_and_coal_gated_optical

    ndvi = np.array([[0.05, 0.8], [np.nan, 0.1]], np.float32)
    w = np.clip((0.30 - np.nan_to_num(ndvi, nan=1.0)) / 0.2, 0, 1).astype(np.float32)
    out = compute_metallic_and_coal_gated_optical(
        s2_stack={"ndvi": ndvi, "ficr": np.full((2, 2), 1.3, np.float32)},
        aster_tir_silica=None, thermal_st=None, w_canopy=w,
        river_mask=np.zeros((2, 2), np.uint8), disturbance_mask=np.zeros((2, 2), np.float32))
    assert "spec_albedo_gated" not in out and "spec_wcdi_gated" not in out and "spec_oci_gated" not in out
    assert "aster_tir_silica_gated" not in out and "thermal_anomaly_st_filtered" not in out
    f = out["spec_ficr_gated"]
    assert f[0, 0] == pytest.approx(1.3) and np.isnan(f[0, 1]) and np.isnan(f[1, 0])
    with pytest.raises(ValueError):
        compute_metallic_and_coal_gated_optical(s2_stack={}, aster_tir_silica=None, thermal_st=None, w_canopy=w,
                                                river_mask=np.zeros((2, 2)), disturbance_mask=np.zeros((2, 2)))


def test_potential_fields_require_real_magnetics() -> None:
    from src.features.coal_and_multimodal_features import compute_potential_field_derivatives

    with pytest.raises(ValueError):
        compute_potential_field_derivatives(bouguer_mgal=np.zeros((8, 8)), sed_thickness_m=np.zeros((8, 8)))


def test_lithologic_contacts_unknown_when_unit_unmapped() -> None:
    import geopandas as gpd

    from src.features.coal_and_multimodal_features import compute_lithologic_and_stratigraphic_contacts

    empty = gpd.GeoDataFrame({"formation": [], "lithology": [], "geometry": []}, crs="EPSG:4326")
    out = compute_lithologic_and_stratigraphic_contacts(
        empty, grid_lon=np.linspace(114.5, 114.6, 20), grid_lat=np.linspace(3.2, 3.1, 20),
        georima_csv_path=ROOT / "does_not_exist.csv")
    for k in ("dist_calc_alkaline_intrusive_km", "dist_sedimentary_polygon_km", "dist_marine_deltaic_contact_km"):
        assert np.isnan(out[k]).all(), k


def test_legacy_harvester_fabricators_are_disabled() -> None:
    import importlib

    h = importlib.import_module("scripts.harvest_kalimantan_multimodal")
    z = np.zeros((4, 4))
    calls = {
        "fetch_and_process_optical": (z, 4, 4),
        "fetch_and_process_thermal": (z, z, 4, 4),
        "fetch_and_process_geophysics": (z, 4, 4, None),
        "fetch_and_process_deposits_and_covariates": (4, 4, None),
        "build_kalimantan_multimodal_pipeline": (),
    }
    for fn, args in calls.items():
        with pytest.raises(RuntimeError, match="fabricated"):
            getattr(h, fn)(*args)
