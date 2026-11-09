"""Archipelago-wide K=4 deposit targets from GeoRIMA, ESDM and MRDS.

Inside the Borneo bounding box the rules are identical to
``borneo_labels.build_k4_deposit_targets`` (same sources, same filters), so
Kalimantan positives match production. Outside it the same GeoRIMA rules are
applied to every island. Two Kalimantan-only sources (the ESDM iron
GeoPackage and the Borneo MRDS GeoPackage) are replaced there by their
archipelago equivalents:

* skarn (k1): GeoRIMA ``IRON_ORE`` / ``Besi Primer`` stands in for ESDM ``Besi Primer``.
* laterite (k3): GeoRIMA ``Besi Laterit`` stands in for ESDM ``Besi Laterit``.
* MRDS Cu / Au / Ni: ``mrds-indonesia.csv`` commodity text stands in for the Borneo ``code_list``.

Hygiene: alluvial placers (``Emas Alluvial``) and centroid artifacts are
removed by ``filter_valid_ore_candidates``. MRDS rows whose coordinates are
flagged as a province/island centroid or as shared/unverified are dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.labels.borneo_labels import (
    BORNEO_BBOX,
    DEFAULT_GEORIMA_PATH,
    DEFAULT_IRON_PATH,
    DEFAULT_MRDS_PATH,
    filter_valid_ore_candidates,
)

DEFAULT_MRDS_INDONESIA_PATH = Path("data/indonesia/mrds-indonesia.csv")
EK_BBOX = (114.50, -2.80, 119.50, 3.20)
K4_TARGETS = ("k0_porphyry_cu", "k1_skarn_fe_cu", "k2_epithermal_au", "k3_laterite_fe_ni")
_BAD_MRDS_FLAGS = ("province_or_island_centroid", "shared coordinate")


def _in_bbox(lon: pd.Series, lat: pd.Series, bbox: tuple) -> pd.Series:
    return lon.between(bbox[0], bbox[2]) & lat.between(bbox[1], bbox[3])


def _points(df: pd.DataFrame, lon: str, lat: str, source: str) -> pd.DataFrame:
    return pd.DataFrame({"lon": df[lon].astype(float).to_numpy(), "lat": df[lat].astype(float).to_numpy(),
                         "source": source})


def clean_mrds_indonesia(mrds: pd.DataFrame) -> pd.DataFrame:
    """Drop MRDS rows with centroid or unverified shared coordinates, and rows without coordinates."""
    flag = mrds.get("coordinate_quality_flag", pd.Series("", index=mrds.index)).fillna("").astype(str)
    bad = np.zeros(len(mrds), dtype=bool)
    for f in _BAD_MRDS_FLAGS:
        bad |= flag.str.contains(f, case=False, regex=False).to_numpy()
    out = mrds[~bad].copy()
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    return out.dropna(subset=["longitude", "latitude"])


def _mrds_commodity(mrds: pd.DataFrame, word: str) -> pd.Series:
    text = mrds[[c for c in ("commod1", "commod2", "commod3") if c in mrds.columns]].fillna("").astype(str).agg(" ".join, axis=1)
    return text.str.contains(word, case=False, regex=False)


def build_archipelago_k4_targets(
    georima: pd.DataFrame,
    mrds_indonesia: pd.DataFrame,
    iron_gdf=None,
    borneo_mrds_gdf=None,
    borneo_bbox: tuple = BORNEO_BBOX,
) -> dict[str, pd.DataFrame]:
    """Return {target: DataFrame(lon, lat, source)} for the four K=4 heads, de-duplicated by coordinate."""
    geo = filter_valid_ore_candidates(georima)
    in_b = _in_bbox(geo["lon"], geo["lat"], borneo_bbox)
    # Production Kalimantan rules select GeoRIMA rows with this mask (island_region OR bbox)
    in_borneo_src = (geo.get("island_region", pd.Series("", index=geo.index)) == "Kalimantan (Borneo)") | in_b
    mrds = clean_mrds_indonesia(mrds_indonesia)
    mrds_out = mrds[~_in_bbox(mrds["longitude"], mrds["latitude"], borneo_bbox)]

    b_mrds = None
    if borneo_mrds_gdf is not None:
        b_mrds = filter_valid_ore_candidates(borneo_mrds_gdf)
        code = b_mrds.get("code_list", pd.Series("", index=b_mrds.index)).fillna("")
        b_mrds = b_mrds.assign(_lon=b_mrds.geometry.x, _lat=b_mrds.geometry.y, _code=code)
    iron = filter_valid_ore_candidates(iron_gdf) if iron_gdf is not None else None
    if iron is not None:
        iron = iron.assign(_lon=iron.geometry.x, _lat=iron.geometry.y)

    out: dict[str, list[pd.DataFrame]] = {t: [] for t in K4_TARGETS}

    # k0 porphyry Cu-Au
    ek_intrusive = geo["geological_role"].isin(["felsic_intrusive", "potassic_host"]) & _in_bbox(geo["lon"], geo["lat"], EK_BBOX)
    k0 = ((geo["resource_group"] == "COPPER_ORE") | (geo["commodity_label"] == "Tembaga")
          | (geo["resource_group"] == "BASE_METALS") | (geo["geological_role"] == "porphyry_intrusive") | ek_intrusive)
    out["k0_porphyry_cu"].append(_points(geo[k0], "lon", "lat", "georima"))
    if b_mrds is not None:
        out["k0_porphyry_cu"].append(_points(b_mrds[b_mrds["_code"].str.contains("CU", case=False)], "_lon", "_lat", "mrds_borneo"))
    out["k0_porphyry_cu"].append(_points(mrds_out[_mrds_commodity(mrds_out, "Copper")], "longitude", "latitude", "mrds_indonesia"))

    # k1 contact Fe-Cu skarn. Production restricts GeoRIMA carbonate_skarn to Kalimantan; here every island.
    out["k1_skarn_fe_cu"].append(_points(geo[geo["geological_role"] == "carbonate_skarn"], "lon", "lat", "georima_skarn"))
    if iron is not None:
        out["k1_skarn_fe_cu"].append(_points(iron[iron["commodity_type"] == "Besi Primer"], "_lon", "_lat", "esdm_iron"))
    k1_out = (geo["resource_group"] == "IRON_ORE") & (geo["commodity_label"] == "Besi Primer") & ~in_borneo_src
    out["k1_skarn_fe_cu"].append(_points(geo[k1_out], "lon", "lat", "georima_besi_primer"))

    # k2 epithermal Au-Ag
    k2 = (geo["resource_group"] == "GOLD_ORE") & (geo["commodity_label"] == "Emas Primer")
    out["k2_epithermal_au"].append(_points(geo[k2], "lon", "lat", "georima_gold"))
    if b_mrds is not None:
        out["k2_epithermal_au"].append(_points(b_mrds[b_mrds["_code"].str.contains("AU", case=False)], "_lon", "_lat", "mrds_borneo"))
    out["k2_epithermal_au"].append(_points(mrds_out[_mrds_commodity(mrds_out, "Gold")], "longitude", "latitude", "mrds_indonesia"))

    # k3 lateritic Fe-Ni
    out["k3_laterite_fe_ni"].append(_points(geo[geo["resource_group"] == "NICKEL_COBALT"], "lon", "lat", "georima_ni"))
    if iron is not None:
        out["k3_laterite_fe_ni"].append(_points(iron[iron["commodity_type"] == "Besi Laterit"], "_lon", "_lat", "esdm_laterite"))
    k3_out = (geo["resource_group"] == "IRON_ORE") & (geo["commodity_label"] == "Besi Laterit") & ~in_borneo_src
    out["k3_laterite_fe_ni"].append(_points(geo[k3_out], "lon", "lat", "georima_besi_laterit"))
    if b_mrds is not None:
        out["k3_laterite_fe_ni"].append(_points(b_mrds[b_mrds["_code"].str.contains("NI", case=False)], "_lon", "_lat", "mrds_borneo"))
    out["k3_laterite_fe_ni"].append(_points(mrds_out[_mrds_commodity(mrds_out, "Nickel")], "longitude", "latitude", "mrds_indonesia"))

    result = {}
    for t, parts in out.items():
        df = pd.concat(parts, ignore_index=True)
        df = df[np.isfinite(df["lon"]) & np.isfinite(df["lat"])]
        result[t] = df.drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    return result


def load_archipelago_sources(georima_path: Optional[Path] = None, mrds_indonesia_path: Optional[Path] = None,
                             iron_path: Optional[Path] = None, borneo_mrds_path: Optional[Path] = None) -> dict:
    import geopandas as gpd

    def _gpkg(p):
        p = Path(p)
        if not p.exists():
            return None
        g = gpd.read_file(p)
        if g.crs is not None and g.crs.to_epsg() != 4326:
            g = g.to_crs("EPSG:4326")
        if "is_centroid_artifact" not in g.columns:
            g["is_centroid_artifact"] = False
        if "commodity_label" not in g.columns:
            g["commodity_label"] = g["commodity_type"] if "commodity_type" in g.columns else g.get("site_name", "")
        return g

    geo = pd.read_csv(georima_path or DEFAULT_GEORIMA_PATH)
    if "is_centroid_artifact" not in geo.columns:
        geo["is_centroid_artifact"] = False
    return {
        "georima": geo,
        "mrds_indonesia": pd.read_csv(mrds_indonesia_path or DEFAULT_MRDS_INDONESIA_PATH),
        "iron": _gpkg(iron_path or DEFAULT_IRON_PATH),
        "borneo_mrds": _gpkg(borneo_mrds_path or DEFAULT_MRDS_PATH),
    }


def points_to_cells(lon: np.ndarray, lat: np.ndarray, transform, H: int, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map points to (row, col) with the grid's affine transform. Returns rows, cols and an in-grid mask."""
    inv = ~transform
    x = np.asarray(lon, dtype=np.float64)
    y = np.asarray(lat, dtype=np.float64)
    c = inv.a * x + inv.b * y + inv.c
    r = inv.d * x + inv.e * y + inv.f
    r = np.floor(r).astype(np.int64)
    c = np.floor(c).astype(np.int64)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    return r, c, ok
