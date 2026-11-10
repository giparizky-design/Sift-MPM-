"""Borneo Multi-Mineral Vector Unmerger and K=4 Target Label Generator.

Provides 100% leak-free and placer-hygiene-enforced label extraction for the
Borneo K=4 deposit mechanics discovery engine:
  - Head 0: Cu-Mo deposits (GeoRIMA COPPER_ORE [Tembaga, Molibdenum] + MRDS records with a CU code).
            Key kept as ``porphyry_cu``; the records are copper deposits of any style, not a
            verified porphyry classification (v4.15; Pb-Zn, Hg-Sb and rock-type quarry nodes removed).
  - Head 1: Primary Fe (ESDM Besi Primer only; marble is a possible skarn host, not a deposit)
  - Head 2: Primary Au-Ag (GeoRIMA Emas Primer + MRDS Au records that are not placers or excluded sites)
  - Head 3: Lateritic Fe-Ni (ESDM Besi Laterit [21] + GeoRIMA NICKEL_COBALT along Meratus/Sebuku)
  - Alteration Benchmark: Exact strings argillic_alteration, advanced_argillic, propylitic_zeolite,
    Kaolin, Pirofilit, Zeolit
  - Pooled Exploration Footprint: Union of P0-P3, Emas Alluvial, clay quarries, and ESDM concessions (N ≈ 1,800)

Mandatory Guardrails Enforced:
  - Hard exclusion of river placers ("Emas Alluvial") and centroid artifacts (is_centroid_artifact == True)
    from all positive ore targets P_k.
  - "Emas Alluvial" and concession centroids retained in the human exploration footprint (E_pooled).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Default file paths
DEFAULT_GEORIMA_PATH = Path("data/indonesia/georima_all_indonesia_nodes.csv")
DEFAULT_IRON_PATH = Path("data/borneo/kalimantan_iron_deposits.gpkg")
DEFAULT_MRDS_PATH = Path("data/borneo/borneo_mrds_occurrences.gpkg")
DEFAULT_COAL_PATH = Path("data/borneo/kalimantan_coal_deposits.gpkg")

# Bounding box covering Kalimantan and surrounding Borneo: 108.5°E - 119.5°E, 4.5°S - 7.0°N
BORNEO_BBOX = (108.5, -4.5, 119.5, 7.0)

# Canonical K=4 metallic deposit label columns
K4_LABEL_COLS = [
    "label_k0_porphyry_cu",
    "label_k1_skarn_fe_cu",
    "label_k2_epithermal_au",
    "label_k3_laterite_fe_ni",
]


# ---------------------------------------------------------------------------
# Global Hygiene Guard
# ---------------------------------------------------------------------------


# Sites removed from every positive set by name (lower-case site name -> reason).
# A decision, not a default: add a name here only with a documented reason.
EXCLUDED_DEPOSIT_SITES: dict[str, str] = {
    "busang": "Bre-X fraud (1997): the reported gold resource was fabricated; minor real mineralisation only.",
}

# MRDS site names that denote alluvial / placer workings (whole-word match).
PLACER_SITE_NAME_PATTERN = r"\b(?:river|alluvial|paleochannels?|placers?)\b"
# MRDS commodity code for gem diamonds, which in Borneo marks alluvial diamond-gold workings.
PLACER_MRDS_CODES = frozenset({"GEM_D"})


def mrds_code_tokens(code_list: pd.Series) -> pd.Series:
    """Split MRDS ``code_list`` strings into upper-case token lists (whole-token matching)."""
    return code_list.fillna("").astype(str).str.upper().str.split()


def mrds_has_code(code_list: pd.Series, code: str) -> pd.Series:
    """True where ``code`` is one of the whitespace-separated tokens ('CU' does not match 'CUR')."""
    code = code.upper()
    return mrds_code_tokens(code_list).apply(lambda toks: code in toks)


def _site_name_series(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> pd.Series:
    for col in ("site_name", "deposit_name", "name"):
        if col in df.columns:
            return df[col].fillna("").astype(str).str.strip().str.lower()
    return pd.Series("", index=df.index)


def filter_valid_ore_candidates(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Apply the Global Hygiene Guard to exclude alluvial placers, named exclusions and centroid artifacts.

    Enforces Reconciled Engineering Rule 3:
      River placers and island/province centroid artifacts must never contaminate bedrock
      positive ore training sets (P_k).

    A record is dropped when any of the following holds:
      - ``is_placer`` is True (set by the loaders: GeoRIMA ``Emas Alluvial``; MRDS ``GEM_D`` code or a
        placer word in the site name);
      - ``commodity_label == "Emas Alluvial"`` (kept for frames that do not carry ``is_placer``);
      - the site name is in ``EXCLUDED_DEPOSIT_SITES``;
      - ``is_centroid_artifact`` is True.

    Args:
        df: DataFrame or GeoDataFrame with optional columns 'commodity_label', 'is_placer',
            'site_name' and 'is_centroid_artifact'.

    Returns:
        Filtered DataFrame / GeoDataFrame retaining only valid bedrock ore candidates.
    """
    comm_label = df["commodity_label"] if "commodity_label" in df.columns else pd.Series("", index=df.index)
    artifact_series = df["is_centroid_artifact"] if "is_centroid_artifact" in df.columns else pd.Series(False, index=df.index)
    placer_series = df["is_placer"] if "is_placer" in df.columns else pd.Series(False, index=df.index)
    excluded_site = _site_name_series(df).isin(set(EXCLUDED_DEPOSIT_SITES))

    is_valid_ore_candidate = (
        (comm_label != "Emas Alluvial")
        & (~placer_series.fillna(False).astype(bool))
        & (~excluded_site)
        & (~artifact_series.fillna(False).astype(bool))
    )
    return df[is_valid_ore_candidate].copy()


# ---------------------------------------------------------------------------
# Vector Source Ingestion
# ---------------------------------------------------------------------------


def load_borneo_vector_sources(
    georima_path: Optional[Union[str, Path]] = None,
    iron_path: Optional[Union[str, Path]] = None,
    mrds_path: Optional[Union[str, Path]] = None,
    coal_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Load and normalize source vector datasets across Borneo/Kalimantan.

    Args:
        georima_path: Path to GeoRIMA CSV file. Defaults to data/indonesia/georima_all_indonesia_nodes.csv.
        iron_path: Path to Kalimantan Iron GeoPackage. Defaults to data/borneo/kalimantan_iron_deposits.gpkg.
        mrds_path: Path to Borneo MRDS GeoPackage. Defaults to data/borneo/borneo_mrds_occurrences.gpkg.
        coal_path: Optional path to Kalimantan Coal GeoPackage for exploration footprint.

    Returns:
        Dictionary with normalized DataFrames / GeoDataFrames:
          - 'georima': GeoRIMA nodes for Kalimantan (Borneo)
          - 'iron': ESDM iron deposits
          - 'mrds': USGS MRDS occurrences in Borneo
          - 'coal': ESDM coal deposits (if available)
    """
    p_geo = Path(georima_path or DEFAULT_GEORIMA_PATH)
    p_iron = Path(iron_path or DEFAULT_IRON_PATH)
    p_mrds = Path(mrds_path or DEFAULT_MRDS_PATH)
    p_coal = Path(coal_path or DEFAULT_COAL_PATH)

    if not p_geo.exists():
        raise FileNotFoundError(f"GeoRIMA nodes file not found: {p_geo}")
    if not p_iron.exists():
        raise FileNotFoundError(f"Kalimantan iron deposits file not found: {p_iron}")
    if not p_mrds.exists():
        raise FileNotFoundError(f"Borneo MRDS file not found: {p_mrds}")

    df_geo_raw = pd.read_csv(p_geo)
    # Filter to Kalimantan / Borneo
    mask_borneo = (
        (df_geo_raw["island_region"] == "Kalimantan (Borneo)")
        | (
            df_geo_raw["lon"].between(BORNEO_BBOX[0], BORNEO_BBOX[2])
            & df_geo_raw["lat"].between(BORNEO_BBOX[1], BORNEO_BBOX[3])
        )
    )
    b_geo = df_geo_raw[mask_borneo].copy()
    if "is_centroid_artifact" not in b_geo.columns:
        b_geo["is_centroid_artifact"] = False
    b_geo["is_placer"] = b_geo["commodity_label"].eq("Emas Alluvial") if "commodity_label" in b_geo.columns else False

    gdf_iron = gpd.read_file(p_iron)
    if gdf_iron.crs is None or gdf_iron.crs.to_epsg() != 4326:
        gdf_iron = gdf_iron.to_crs("EPSG:4326")
    if "is_centroid_artifact" not in gdf_iron.columns:
        gdf_iron["is_centroid_artifact"] = False
    if "commodity_label" not in gdf_iron.columns and "commodity_type" in gdf_iron.columns:
        gdf_iron["commodity_label"] = gdf_iron["commodity_type"]

    gdf_mrds = gpd.read_file(p_mrds)
    if gdf_mrds.crs is None or gdf_mrds.crs.to_epsg() != 4326:
        gdf_mrds = gdf_mrds.to_crs("EPSG:4326")
    if "is_centroid_artifact" not in gdf_mrds.columns:
        gdf_mrds["is_centroid_artifact"] = False
    if "commodity_label" not in gdf_mrds.columns:
        gdf_mrds["commodity_label"] = gdf_mrds.get("site_name", "")
    # Placer flag: MRDS carries no commodity_label of its own, so the placer test must read the
    # commodity codes (GEM_D = gem diamond, alluvial in Borneo) and the site name (v4.14 B/C: Riam
    # Kanan River and Cempaka-Danau Seran Paleochannels passed the old string test).
    _codes = mrds_code_tokens(gdf_mrds["code_list"]) if "code_list" in gdf_mrds.columns else pd.Series([[]] * len(gdf_mrds), index=gdf_mrds.index)
    gdf_mrds["is_placer"] = (
        _codes.apply(lambda toks: bool(PLACER_MRDS_CODES.intersection(toks)))
        | _site_name_series(gdf_mrds).str.contains(PLACER_SITE_NAME_PATTERN, regex=True)
    )

    gdf_coal = None
    if p_coal.exists():
        try:
            gdf_coal = gpd.read_file(p_coal)
            if gdf_coal.crs is None or gdf_coal.crs.to_epsg() != 4326:
                gdf_coal = gdf_coal.to_crs("EPSG:4326")
        except Exception as e:
            logger.warning("Could not load coal GPKG (%s): %s", p_coal, e)

    return {
        "georima": b_geo,
        "iron": gdf_iron,
        "mrds": gdf_mrds,
        "coal": gdf_coal,
    }


# ---------------------------------------------------------------------------
# K=4 Deposit Mechanics Target Builder
# ---------------------------------------------------------------------------


def build_k4_deposit_targets(
    georima_df: Optional[pd.DataFrame] = None,
    iron_gdf: Optional[gpd.GeoDataFrame] = None,
    mrds_gdf: Optional[gpd.GeoDataFrame] = None,
    coal_gdf: Optional[gpd.GeoDataFrame] = None,
) -> dict[str, Any]:
    """Build the K=4 target sets, alteration benchmarks, and pooled exploration footprint.

    Exact Implementation:
      - Head k=0 (Porphyry Cu-Au):
        GeoRIMA resource_group == 'COPPER_ORE' ('Tembaga'), Cu-bearing 'BASE_METALS',
        and porphyry intrusive nodes, plus Borneo MRDS Cu-bearing nodes (N ≈ 35-45).
      - Head k=1 (Contact Fe-Cu Skarn):
        Kalimantan ESDM commodity_type == 'Besi Primer' (57) +
        GeoRIMA geological_role == 'carbonate_skarn' (14) (Total N = 71).
      - Head k=2 (Epithermal Au-Ag):
        GeoRIMA resource_group == 'GOLD_ORE' with commodity_label == 'Emas Primer' (87) +
        MRDS primary epithermal Au-Ag occurrences.
      - Head k=3 (Lateritic Fe-Ni):
        Kalimantan ESDM commodity_type == 'Besi Laterit' (21) +
        GeoRIMA resource_group == 'NICKEL_COBALT' along Meratus/Sebuku ophiolite belt.
      - Alteration Benchmark:
        Exact strings argillic_alteration, advanced_argillic, propylitic_zeolite,
        Kaolin, Pirofilit, Zeolit.
      - Pooled Exploration Footprint:
        Union of P0-P3, 'Emas Alluvial' sites, industrial clay quarries, and ESDM
        concession centroids (N ≈ 1,800).

    Returns:
        Dictionary containing GeoDataFrames for:
          'positives_k0', 'positives_k1', 'positives_k2', 'positives_k3',
          'alteration_benchmark', 'explored_footprint'.
    """
    if georima_df is None or iron_gdf is None or mrds_gdf is None:
        sources = load_borneo_vector_sources(coal_path=coal_gdf)
        georima_df = sources["georima"]
        iron_gdf = sources["iron"]
        mrds_gdf = sources["mrds"]
        coal_gdf = sources["coal"]

    # Apply global hygiene guard to separate raw candidate nodes
    clean_geo = filter_valid_ore_candidates(georima_df)
    clean_iron = filter_valid_ore_candidates(iron_gdf)
    clean_mrds = filter_valid_ore_candidates(mrds_gdf)

    # ------------------------------------------------------------------
    # Head k=0: Porphyry Cu-Au & Intrusive-Hosted Magmatic-Hydrothermal Systems
    # ------------------------------------------------------------------
    # v4.15: deposit records only. The Pb-Zn / Hg-Sb (BASE_METALS), rock-type
    # (porphyry_intrusive, felsic_intrusive, potassic_host) and East-Kalimantan-box rules were removed:
    # they put quarry / industrial-mineral occurrences into the head (v4.13 section 1.1).
    k0_geo_mask = clean_geo["resource_group"] == "COPPER_ORE"  # Tembaga (Cu) and Molibdenum (Mo)
    k0_geo = clean_geo[k0_geo_mask].copy()

    # MRDS Cu-bearing nodes (whole-token match on the commodity code list)
    mrds_code = clean_mrds.get("code_list", pd.Series("", index=clean_mrds.index)).fillna("")
    k0_mrds_mask = mrds_has_code(mrds_code, "CU")
    k0_mrds = clean_mrds[k0_mrds_mask].copy()

    pts_k0: list[Point] = [Point(float(r["lon"]), float(r["lat"])) for _, r in k0_geo.iterrows()]
    pts_k0.extend(list(k0_mrds.geometry))
    gdf_k0 = gpd.GeoDataFrame(
        {"target_class": "k0_porphyry_cu", "source": ["georima"] * len(k0_geo) + ["mrds"] * len(k0_mrds)},
        geometry=pts_k0,
        crs="EPSG:4326",
    ).drop_duplicates(subset=["geometry"])

    # ------------------------------------------------------------------
    # Head k=1: Contact Fe-Cu Skarn
    # ------------------------------------------------------------------
    k1_iron_mask = clean_iron["commodity_type"] == "Besi Primer"
    k1_iron = clean_iron[k1_iron_mask].copy()

    # v4.15: GeoRIMA marble (``carbonate_skarn`` role) is a possible skarn host, not a deposit, so it is
    # no longer a positive. The empty frame keeps the downstream concatenation unchanged.
    k1_geo = clean_geo.iloc[0:0].copy()

    pts_k1: list[Point] = list(k1_iron.geometry)
    pts_k1.extend([Point(float(r["lon"]), float(r["lat"])) for _, r in k1_geo.iterrows()])
    gdf_k1 = gpd.GeoDataFrame(
        {"target_class": "k1_skarn_fe_cu", "source": ["esdm_iron"] * len(k1_iron) + ["georima_skarn"] * len(k1_geo)},
        geometry=pts_k1,
        crs="EPSG:4326",
    ).drop_duplicates(subset=["geometry"])

    # ------------------------------------------------------------------
    # Head k=2: Epithermal Au-Ag
    # ------------------------------------------------------------------
    k2_geo_mask = (clean_geo["resource_group"] == "GOLD_ORE") & (clean_geo["commodity_label"] == "Emas Primer")
    k2_geo = clean_geo[k2_geo_mask].copy()

    k2_mrds_mask = mrds_has_code(mrds_code, "AU")
    k2_mrds = clean_mrds[k2_mrds_mask].copy()

    pts_k2: list[Point] = [Point(float(r["lon"]), float(r["lat"])) for _, r in k2_geo.iterrows()]
    pts_k2.extend(list(k2_mrds.geometry))
    gdf_k2 = gpd.GeoDataFrame(
        {"target_class": "k2_epithermal_au", "source": ["georima_gold"] * len(k2_geo) + ["mrds_gold"] * len(k2_mrds)},
        geometry=pts_k2,
        crs="EPSG:4326",
    ).drop_duplicates(subset=["geometry"])

    # ------------------------------------------------------------------
    # Head k=3: Lateritic Fe-Ni
    # ------------------------------------------------------------------
    k3_iron_mask = clean_iron["commodity_type"] == "Besi Laterit"
    k3_iron = clean_iron[k3_iron_mask].copy()

    k3_geo_mask = clean_geo["resource_group"] == "NICKEL_COBALT"
    k3_geo = clean_geo[k3_geo_mask].copy()

    k3_mrds_mask = mrds_has_code(mrds_code, "NI")
    k3_mrds = clean_mrds[k3_mrds_mask].copy()

    pts_k3: list[Point] = list(k3_iron.geometry)
    pts_k3.extend([Point(float(r["lon"]), float(r["lat"])) for _, r in k3_geo.iterrows()])
    pts_k3.extend(list(k3_mrds.geometry))
    gdf_k3 = gpd.GeoDataFrame(
        {
            "target_class": "k3_laterite_fe_ni",
            "source": ["esdm_laterite"] * len(k3_iron) + ["georima_ni"] * len(k3_geo) + ["mrds_ni"] * len(k3_mrds),
        },
        geometry=pts_k3,
        crs="EPSG:4326",
    ).drop_duplicates(subset=["geometry"])

    # ------------------------------------------------------------------
    # Alteration Benchmark
    # ------------------------------------------------------------------
    # Exact string matching rule
    alt_mask = georima_df["geological_role"].isin([
        "argillic_alteration",
        "advanced_argillic",
        "propylitic_zeolite",
    ]) | georima_df["commodity_label"].isin(["Kaolin", "Pirofilit", "Zeolit"])
    alt_df = georima_df[alt_mask].copy()
    pts_alt: list[Point] = [Point(float(r["lon"]), float(r["lat"])) for _, r in alt_df.iterrows()]
    gdf_alt = gpd.GeoDataFrame(
        {
            "target_class": ["alteration_benchmark"] * len(pts_alt),
            "commodity": list(alt_df["commodity_label"]),
            "role": list(alt_df["geological_role"]),
        },
        geometry=pts_alt,
        crs="EPSG:4326",
    ).drop_duplicates(subset=["geometry"])

    # ------------------------------------------------------------------
    # Pooled Human Exploration Footprint (E_pooled)
    # ------------------------------------------------------------------
    # Union of P0-P3, Emas Alluvial, industrial clay quarries, and ESDM concessions
    footprint_pts: list[Point] = []
    footprint_pts.extend(list(gdf_k0.geometry))
    footprint_pts.extend(list(gdf_k1.geometry))
    footprint_pts.extend(list(gdf_k2.geometry))
    footprint_pts.extend(list(gdf_k3.geometry))

    # Alluvial gold (river placers)
    alluvial_mask = georima_df["commodity_label"] == "Emas Alluvial"
    for _, r in georima_df[alluvial_mask].iterrows():
        footprint_pts.append(Point(float(r["lon"]), float(r["lat"])))

    # Industrial clay quarries
    clay_mask = georima_df["commodity_label"].isin(["Lempung", "Kaolin"]) | (
        georima_df["geological_role"] == "regolith_clay_cover"
    )
    for _, r in georima_df[clay_mask].iterrows():
        footprint_pts.append(Point(float(r["lon"]), float(r["lat"])))

    # Concession centroids (coal deposits and concessions)
    if coal_gdf is not None and len(coal_gdf) > 0:
        footprint_pts.extend(list(coal_gdf.geometry))

    # All coal in GeoRIMA
    coal_geo_mask = georima_df["resource_group"] == "COAL"
    for _, r in georima_df[coal_geo_mask].iterrows():
        footprint_pts.append(Point(float(r["lon"]), float(r["lat"])))

    gdf_footprint = gpd.GeoDataFrame(
        {"target_class": ["explored_footprint"] * len(footprint_pts)},
        geometry=footprint_pts,
        crs="EPSG:4326",
    ).drop_duplicates(subset=["geometry"])

    logger.info(
        "Built Borneo K=4 Deposit Targets: P0=%d, P1=%d, P2=%d, P3=%d | Alteration=%d | Explored Footprint=%d",
        len(gdf_k0),
        len(gdf_k1),
        len(gdf_k2),
        len(gdf_k3),
        len(gdf_alt),
        len(gdf_footprint),
    )

    return {
        "positives_k0": gdf_k0,
        "positives_k1": gdf_k1,
        "positives_k2": gdf_k2,
        "positives_k3": gdf_k3,
        "alteration_benchmark": gdf_alt,
        "explored_footprint": gdf_footprint,
    }


# ---------------------------------------------------------------------------
# Spatial-Join onto Grid / DataFrame
# ---------------------------------------------------------------------------


def spatial_join_borneo_k4(
    grid_df: pd.DataFrame,
    deposit_targets: Optional[dict[str, Any]] = None,
    match_radius_m: float = 1000.0,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> pd.DataFrame:
    """Spatially join K=4 targets, alteration benchmark, and explored footprint onto a grid DataFrame.

    Produces binary target columns in {-1, 1} for PU training:
      - 'label_k0_porphyry_cu': 1 for positive P0 cells, -1 for unlabeled
      - 'label_k1_skarn_fe_cu': 1 for positive P1 cells, -1 for unlabeled
      - 'label_k2_epithermal_au': 1 for positive P2 cells, -1 for unlabeled
      - 'label_k3_laterite_fe_ni': 1 for positive P3 cells, -1 for unlabeled
      - 'alteration_benchmark': 1 for cells within match_radius_m of alteration benchmark, else 0
      - 'explored_footprint': 1 for cells within match_radius_m of explored footprint, else 0

    Strict Placers & Centroid Hygiene:
      No cell is marked 1 in any label_k* column due to an 'Emas Alluvial' placer
      or centroid artifact.

    Args:
        grid_df: DataFrame containing geographic coordinate columns (default 'lon', 'lat' or 'longitude', 'latitude').
        deposit_targets: Target dictionary from build_k4_deposit_targets(). If None, built automatically.
        match_radius_m: Buffer radius in metres to assign a positive occurrence to a cell.
        lon_col: Name of longitude column.
        lat_col: Name of latitude column.

    Returns:
        DataFrame with added target and benchmark columns.
    """
    out_df = grid_df.copy()

    # Detect coordinate column variants
    col_map = {c.lower(): c for c in out_df.columns}
    real_lon = col_map.get(lon_col.lower(), col_map.get("longitude", col_map.get("x", None)))
    real_lat = col_map.get(lat_col.lower(), col_map.get("latitude", col_map.get("y", None)))

    if real_lon is None or real_lat is None:
        raise ValueError(f"Could not find coordinate columns in grid_df. Checked for {lon_col}, {lat_col}, longitude, latitude, x, y.")

    lons = out_df[real_lon].to_numpy(dtype=np.float64)
    lats = out_df[real_lat].to_numpy(dtype=np.float64)

    if deposit_targets is None:
        deposit_targets = build_k4_deposit_targets()

    # Convert coordinates to approximate local metric projection for distance queries
    mean_lat = float(np.nanmean(lats))
    m_per_deg_lon = 111320.0 * np.cos(np.radians(mean_lat))
    m_per_deg_lat = 110570.0

    grid_metric_xy = np.column_stack([lons * m_per_deg_lon, lats * m_per_deg_lat])
    grid_tree = cKDTree(grid_metric_xy)

    # Initialize all 4 heads to Unlabeled (-1)
    head_keys = [
        ("label_k0_porphyry_cu", "positives_k0"),
        ("label_k1_skarn_fe_cu", "positives_k1"),
        ("label_k2_epithermal_au", "positives_k2"),
        ("label_k3_laterite_fe_ni", "positives_k3"),
    ]

    for col_name, target_key in head_keys:
        out_df[col_name] = -1
        gdf_pos = deposit_targets.get(target_key)
        if gdf_pos is not None and len(gdf_pos) > 0:
            pos_lons = gdf_pos.geometry.x.to_numpy(dtype=np.float64)
            pos_lats = gdf_pos.geometry.y.to_numpy(dtype=np.float64)
            pos_metric_xy = np.column_stack([pos_lons * m_per_deg_lon, pos_lats * m_per_deg_lat])

            # Query grid cells within match_radius_m
            indices = grid_tree.query_ball_point(pos_metric_xy, r=match_radius_m)
            matched_cells = np.unique([idx for sublist in indices for idx in sublist])
            if len(matched_cells) > 0:
                out_df.iloc[matched_cells, out_df.columns.get_loc(col_name)] = 1

    # Alteration Benchmark (0 or 1)
    out_df["alteration_benchmark"] = 0
    gdf_alt = deposit_targets.get("alteration_benchmark")
    if gdf_alt is not None and len(gdf_alt) > 0:
        alt_lons = gdf_alt.geometry.x.to_numpy(dtype=np.float64)
        alt_lats = gdf_alt.geometry.y.to_numpy(dtype=np.float64)
        alt_metric_xy = np.column_stack([alt_lons * m_per_deg_lon, alt_lats * m_per_deg_lat])
        indices = grid_tree.query_ball_point(alt_metric_xy, r=match_radius_m)
        matched_cells = np.unique([idx for sublist in indices for idx in sublist])
        if len(matched_cells) > 0:
            out_df.iloc[matched_cells, out_df.columns.get_loc("alteration_benchmark")] = 1

    # Explored Footprint (0 or 1)
    out_df["explored_footprint"] = 0
    gdf_fp = deposit_targets.get("explored_footprint")
    if gdf_fp is not None and len(gdf_fp) > 0:
        fp_lons = gdf_fp.geometry.x.to_numpy(dtype=np.float64)
        fp_lats = gdf_fp.geometry.y.to_numpy(dtype=np.float64)
        fp_metric_xy = np.column_stack([fp_lons * m_per_deg_lon, fp_lats * m_per_deg_lat])
        indices = grid_tree.query_ball_point(fp_metric_xy, r=match_radius_m)
        matched_cells = np.unique([idx for sublist in indices for idx in sublist])
        if len(matched_cells) > 0:
            out_df.iloc[matched_cells, out_df.columns.get_loc("explored_footprint")] = 1

    return out_df


# ---------------------------------------------------------------------------
# Rasterization Helper
# ---------------------------------------------------------------------------


def rasterize_borneo_k4_labels(
    deposit_targets: dict[str, Any],
    target_profile: dict[str, Any],
    buffer_m: float = 500.0,
) -> dict[str, np.ndarray]:
    """Rasterize K=4 positive targets, alteration benchmark, and explored footprint to 2D arrays.

    Args:
        deposit_targets: Target dictionary from build_k4_deposit_targets().
        target_profile: Rasterio profile dictionary containing 'height', 'width', 'transform', and 'crs'.
        buffer_m: Radius in metres to buffer positive occurrences before rasterization.

    Returns:
        Dictionary of 2D NumPy arrays:
          - 'label_k0_porphyry_cu': values in {-1, 1}
          - 'label_k1_skarn_fe_cu': values in {-1, 1}
          - 'label_k2_epithermal_au': values in {-1, 1}
          - 'label_k3_laterite_fe_ni': values in {-1, 1}
          - 'alteration_benchmark': values in {0, 1}
          - 'explored_footprint': values in {0, 1}
    """
    import warnings
    import rasterio.features
    from rasterio.crs import CRS

    height = target_profile["height"]
    width = target_profile["width"]
    transform = target_profile["transform"]
    target_crs = CRS.from_user_input(target_profile["crs"])

    is_geo = target_crs.is_geographic
    # Convert buffer in metres to CRS units
    buffer_units = (buffer_m / 111320.0) if is_geo else buffer_m

    head_keys = [
        ("label_k0_porphyry_cu", "positives_k0"),
        ("label_k1_skarn_fe_cu", "positives_k1"),
        ("label_k2_epithermal_au", "positives_k2"),
        ("label_k3_laterite_fe_ni", "positives_k3"),
    ]

    out_rasters: dict[str, np.ndarray] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for col_name, target_key in head_keys:
            raster = np.full((height, width), -1, dtype=np.int8)
            gdf = deposit_targets.get(target_key)
            if gdf is not None and len(gdf) > 0:
                gdf_proj = gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf
                buffered = gdf_proj.geometry.buffer(buffer_units)
                shapes = [(geom, 1) for geom in buffered if geom is not None and not geom.is_empty]
                if shapes:
                    rasterio.features.rasterize(
                        shapes=shapes,
                        out=raster,
                        transform=transform,
                        all_touched=True,
                        dtype=np.int8,
                    )
            out_rasters[col_name] = raster

        # Alteration Benchmark (0 or 1)
        alt_raster = np.zeros((height, width), dtype=np.int8)
        gdf_alt = deposit_targets.get("alteration_benchmark")
        if gdf_alt is not None and len(gdf_alt) > 0:
            gdf_alt_proj = gdf_alt.to_crs(target_crs) if gdf_alt.crs != target_crs else gdf_alt
            buffered_alt = gdf_alt_proj.geometry.buffer(buffer_units)
            shapes_alt = [(geom, 1) for geom in buffered_alt if geom is not None and not geom.is_empty]
            if shapes_alt:
                rasterio.features.rasterize(
                    shapes=shapes_alt,
                    out=alt_raster,
                    transform=transform,
                    all_touched=True,
                    dtype=np.int8,
                )
        out_rasters["alteration_benchmark"] = alt_raster

        # Explored Footprint (0 or 1)
        fp_raster = np.zeros((height, width), dtype=np.int8)
        gdf_fp = deposit_targets.get("explored_footprint")
        if gdf_fp is not None and len(gdf_fp) > 0:
            gdf_fp_proj = gdf_fp.to_crs(target_crs) if gdf_fp.crs != target_crs else gdf_fp
            buffered_fp = gdf_fp_proj.geometry.buffer(buffer_units)
            shapes_fp = [(geom, 1) for geom in buffered_fp if geom is not None and not geom.is_empty]
            if shapes_fp:
                rasterio.features.rasterize(
                    shapes=shapes_fp,
                    out=fp_raster,
                    transform=transform,
                    all_touched=True,
                    dtype=np.int8,
                )
        out_rasters["explored_footprint"] = fp_raster

    return out_rasters


# ---------------------------------------------------------------------------
# Kalimantan K=3 Coal Rank Label Engineering & Rim-Anchored Halo Sampling
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
BORNEO_LON_MIN: float = 108.0
BORNEO_LON_MAX: float = 119.5
BORNEO_LAT_MIN: float = -5.0
BORNEO_LAT_MAX: float = 7.5
_ARTIFACT_LAT: float = -1.9982
_ARTIFACT_LON: float = 113.9995
_ARTIFACT_TOL_DEG: float = 0.01

COAL_HEAD_NAMES: dict[int | str, str] = {
    0: "coal_k0_high_cv",
    1: "coal_k1_med_cv",
    2: "coal_k2_low_cv",
    "coal_k0_high_cv": "coal_k0_high_cv",
    "coal_k1_med_cv": "coal_k1_med_cv",
    "coal_k2_low_cv": "coal_k2_low_cv",
}

# Physical strike-extension halo distances in meters along strike (D6):
#   - High CV (coal_k0_high_cv, > 6,100 kcal/kg): 1,000–2,000 m
#   - Med CV (coal_k1_med_cv, 5,100–6,100 kcal/kg): 1,500–3,000 m
#   - Low CV (coal_k2_low_cv, < 5,100 kcal/kg): 1,500–3,500 m
COAL_RIM_HALO_METERS: dict[int | str, tuple[float, float]] = {
    0: (1000.0, 2000.0),
    1: (1500.0, 3000.0),
    2: (1500.0, 3500.0),
    "coal_k0_high_cv": (1000.0, 2000.0),
    "coal_k1_med_cv": (1500.0, 3000.0),
    "coal_k2_low_cv": (1500.0, 3500.0),
}


def get_rank_adaptive_offsets_cells(px_size_m: float = 250.0) -> dict[int | str, tuple[int, int]]:
    """Compute rank-adaptive strike offsets in cells for a given physical pixel size (D6).

    Args:
        px_size_m: Physical pixel size in meters (default 250.0).

    Returns:
        Dictionary mapping coal rank head (int or name) to (d_min_cells, d_max_cells).
    """
    px = max(1.0, float(px_size_m))
    return {
        0: (int(round(1000.0 / px)), int(round(2000.0 / px))),
        1: (int(round(1500.0 / px)), int(round(3000.0 / px))),
        2: (int(round(1500.0 / px)), int(round(3500.0 / px))),
        "coal_k0_high_cv": (int(round(1000.0 / px)), int(round(2000.0 / px))),
        "coal_k1_med_cv": (int(round(1500.0 / px)), int(round(3000.0 / px))),
        "coal_k2_low_cv": (int(round(1500.0 / px)), int(round(3500.0 / px))),
    }


# Default rank-adaptive offsets in 250m grid cells (backward compatibility)
RANK_ADAPTIVE_OFFSETS_CELLS: dict[int | str, tuple[int, int]] = get_rank_adaptive_offsets_cells(250.0)


def _classify_coal_rank_head(row: pd.Series) -> int:
    """Map ESDM coal quality attributes to K=3 calorific rank head {0, 1, 2}.

    Returns:
        0: High-CV (> 6,100 kcal/kg, Kalori Tinggi / Bituminous-Anthracite)
        1: Med-CV (5,100 - 6,100 kcal/kg, Kalori Sedang / Sub-bituminous A/B)
        2: Low-CV (< 5,100 kcal/kg, Kalori Rendah / Lignite-Sub-bituminous C)
        -1: Unclassified / invalid
    """
    for num_col in ("kalori_kcal_kg", "cv_kcal_kg", "calorific_value"):
        if num_col in row and pd.notna(row[num_col]):
            try:
                cv = float(row[num_col])
                if cv > 6100.0:
                    return 0
                if cv >= 5100.0:
                    return 1
                if cv > 0.0:
                    return 2
            except (ValueError, TypeError):
                pass

    for str_col in ("klsbb", "kelas_kalori", "coal_rank", "rank_class"):
        if str_col in row and pd.notna(row[str_col]):
            val = str(row[str_col]).strip().lower()
            if "tinggi" in val or "high" in val or "bitumin" in val or "antrasit" in val or val == "coal_k0_high_cv":
                return 0
            if "sedang" in val or "med" in val or val == "coal_k1_med_cv":
                return 1
            if "rendah" in val or "low" in val or "lignit" in val or val == "coal_k2_low_cv":
                return 2

    if "coal_rank_head" in row and pd.notna(row["coal_rank_head"]):
        val_str = str(row["coal_rank_head"]).strip().lower()
        if val_str in ("0", "coal_k0_high_cv") or "high" in val_str:
            return 0
        if val_str in ("1", "coal_k1_med_cv") or "med" in val_str:
            return 1
        if val_str in ("2", "coal_k2_low_cv") or "low" in val_str:
            return 2
        try:
            h = int(row["coal_rank_head"])
            if h in (0, 1, 2):
                return h
        except (ValueError, TypeError):
            pass

    return -1


def load_kalimantan_coal_sources(
    coal_csv_path: Path | str | pd.DataFrame | None = None,
    purge_centroid_artifact: bool = True,
) -> gpd.GeoDataFrame:
    """Load and standardize the 850 ESDM Kalimantan coal deposit records.

    Computes resource tonnage sample weights:
        w_i = 1.0 + ln(1.0 + Resources_Mt)
    and assigns each deposit to one of the K=3 calorific rank heads:
        0: coal_k0_high_cv (> 6,100 kcal/kg, Kalori Tinggi)
        1: coal_k1_med_cv  (5,100 - 6,100 kcal/kg, Kalori Sedang)
        2: coal_k2_low_cv  (< 5,100 kcal/kg, Kalori Rendah)
    """
    if isinstance(coal_csv_path, (pd.DataFrame, gpd.GeoDataFrame)):
        df = coal_csv_path.copy()
    else:
        if coal_csv_path is None:
            default_paths = [
                _PROJECT_ROOT / "data" / "borneo" / "kalimantan_coal_deposits.csv",
                _PROJECT_ROOT / "data" / "kalimantan" / "raw" / "kalimantan_coal_deposits.csv",
            ]
            resolved = next((p for p in default_paths if p.exists()), default_paths[0])
        else:
            resolved = Path(coal_csv_path)

        if not resolved.exists():
            logger.warning("Coal source file not found at %s; returning empty GeoDataFrame.", resolved)
            return gpd.GeoDataFrame(
                columns=[
                    "concession_id",
                    "namobj",
                    "klsbb",
                    "coal_rank_head",
                    "coal_rank_name",
                    "resources_mt",
                    "tonnage_weight",
                    "latitude",
                    "longitude",
                    "geometry",
                ],
                geometry="geometry",
                crs="EPSG:4326",
            )
        df = pd.read_csv(resolved)

    if len(df) == 0:
        return gpd.GeoDataFrame(
            columns=[
                "concession_id",
                "namobj",
                "klsbb",
                "coal_rank_head",
                "coal_rank_name",
                "resources_mt",
                "tonnage_weight",
                "latitude",
                "longitude",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    # Standardize coordinate columns
    lon_col = next((c for c in ("longitude", "lon", "x") if c in df.columns), None)
    lat_col = next((c for c in ("latitude", "lat", "y") if c in df.columns), None)
    if lon_col is not None and lat_col is not None:
        df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
        df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
        valid_coords = (
            df["longitude"].notna()
            & df["latitude"].notna()
            & (df["longitude"] >= BORNEO_LON_MIN)
            & (df["longitude"] <= BORNEO_LON_MAX)
            & (df["latitude"] >= BORNEO_LAT_MIN)
            & (df["latitude"] <= BORNEO_LAT_MAX)
        )
        df = df.loc[valid_coords].copy()
        if purge_centroid_artifact and len(df) > 0:
            is_artifact = (
                np.isclose(df["latitude"].values, _ARTIFACT_LAT, atol=_ARTIFACT_TOL_DEG)
                & np.isclose(df["longitude"].values, _ARTIFACT_LON, atol=_ARTIFACT_TOL_DEG)
            )
            df = df.loc[~is_artifact].copy()
    else:
        df["longitude"] = np.nan
        df["latitude"] = np.nan

    df = df.reset_index(drop=True)

    # Assign unique integer concession_id >= 1
    if "concession_id" in df.columns:
        df["concession_id"] = pd.to_numeric(df["concession_id"], errors="coerce").fillna(0).astype(int)
        zero_mask = df["concession_id"] <= 0
        if zero_mask.any():
            df.loc[zero_mask, "concession_id"] = np.arange(1, int(zero_mask.sum()) + 1)
    elif "objectid" in df.columns:
        df["concession_id"] = pd.to_numeric(df["objectid"], errors="coerce").fillna(
            pd.Series(np.arange(1, len(df) + 1), index=df.index)
        ).astype(int)
    else:
        df["concession_id"] = np.arange(1, len(df) + 1, dtype=int)

    # Map calorific rank head
    df["coal_rank_head"] = df.apply(_classify_coal_rank_head, axis=1).astype(int)
    df = df.loc[df["coal_rank_head"].isin([0, 1, 2])].copy().reset_index(drop=True)
    df["coal_rank_name"] = df["coal_rank_head"].map(COAL_HEAD_NAMES)

    # Compute resource tonnage in Mt and logarithmic sample weight w_i = 1 + ln(1 + Resources_Mt)
    res_col = next(
        (c for c in ("totsdbb", "resources_mt", "tonnage_mt", "totcadbb") if c in df.columns),
        None,
    )
    if res_col is not None:
        resources_mt = pd.to_numeric(df[res_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    else:
        resources_mt = np.zeros(len(df), dtype=np.float64)

    df["resources_mt"] = resources_mt
    df["tonnage_weight"] = (1.0 + np.log1p(resources_mt)).astype(np.float32)

    if "geometry" not in df.columns or not isinstance(df, gpd.GeoDataFrame):
        if df["longitude"].notna().all() and df["latitude"].notna().all():
            geom = gpd.points_from_xy(df["longitude"], df["latitude"])
        else:
            geom = [Point(0.0, 0.0) for _ in range(len(df))]
        gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    return gdf


def build_k3_coal_rank_targets(
    concessions: gpd.GeoDataFrame | pd.DataFrame | Path | str | None,
    ndvi: np.ndarray,
    active_mine_mask: np.ndarray,
    formation_polygon_id: np.ndarray,
    sar_bedding_strike_deg: np.ndarray | float | None = None,
    gfw_forest_disturbance: np.ndarray | None = None,
    sar_hh_db: np.ndarray | None = None,
    transform: object | None = None,
    bounds: tuple[float, float, float, float] = (108.5, -4.5, 119.5, 4.5),
    canopy_ndvi_min: float = 0.25,
    disturbance_ndvi_max: float = 0.20,
    max_rim_search_cells: int = 40,
    px_size_m: float = 250.0,
) -> dict[str, object]:
    """Build K=3 Coal Rank positive halo targets with 3 Non-Negotiable Spatial Guardrails.

    Parameterized in physical meters divided by pixel size px_size_m (D6, Bug 10):
      - High CV (coal_k0_high_cv, > 6,100 kcal/kg): 1,000–2,000 m along strike
      - Med CV (coal_k1_med_cv, 5,100–6,100 kcal/kg): 1,500–3,000 m along strike
      - Low CV (coal_k2_low_cv, < 5,100 kcal/kg): 1,500–3,500 m along strike
      - Canopy NDVI footprint: 750 x 750 m kernel (k_size = max(3, round(750 / px_size_m)))

    Three Non-Negotiable Kalimantan Spatial Guardrails:
      1. Rim-Anchored Origin (d=0 at Disturbance Boundary):
         Traces along the bedding strike ray through any contiguous bare-earth/mine
         disturbance zone (NDVI < 0.20, active_mine_mask == 1, gfw_forest_disturbance >= 0.5,
         (sar_hh_db is ignored since v4.15)) to find the outer boundary pixel (r_rim, c_rim).
         Sets d = 0 at (r_rim, c_rim) and steps out by d in [d_min^(k), d_max^(k)].
      2. Same-Formation Polygon Lock:
         Candidate (r', c') must satisfy:
             formation_polygon_id[r', c'] == formation_polygon_id[r_i, c_i] > 0
         If a strike-offset vector crosses into a different formation polygon, it is
         immediately rejected.
      3. Canopy Integrity Check:
         Candidate (r', c') must pass a neighborhood check requiring every pixel in
         the 750 x 750 m footprint to have NDVI >= 0.25 and active_mine_mask == 0
         (plus undisturbed GFW/SAR where provided).
    """
    ndvi_arr = np.asarray(ndvi, dtype=np.float32)
    if ndvi_arr.ndim != 2:
        raise ValueError(f"Expected 2D ndvi array, got shape {ndvi_arr.shape}")
    H, W = ndvi_arr.shape

    mine_arr = np.asarray(active_mine_mask, dtype=np.int8)
    poly_arr = np.asarray(formation_polygon_id, dtype=np.int32)
    if mine_arr.shape != (H, W) or poly_arr.shape != (H, W):
        raise ValueError("ndvi, active_mine_mask, and formation_polygon_id must have identical 2D shape.")

    if sar_bedding_strike_deg is None:
        strike_arr = np.full((H, W), 45.0, dtype=np.float32)
    elif np.ndim(sar_bedding_strike_deg) == 0:
        strike_arr = np.full((H, W), float(sar_bedding_strike_deg), dtype=np.float32)
    else:
        strike_arr = np.asarray(sar_bedding_strike_deg, dtype=np.float32)

    gfw_arr = (
        np.asarray(gfw_forest_disturbance, dtype=np.float32)
        if gfw_forest_disturbance is not None
        else np.zeros((H, W), dtype=np.float32)
    )
    # v4.15: ``sar_hh_db`` is accepted for API compatibility but no longer used. The old rule treated
    # HH > -2 dB as disturbed ground, but L-band HH is about -7 dB over forest and about -13 dB over
    # sparsely vegetated land in this region, so the clause selected almost nothing and none of it bare
    # (audit v4.13 section 2.7).

    # Load or standardize concessions DataFrame
    if concessions is None or isinstance(concessions, (Path, str)):
        gdf_conc = load_kalimantan_coal_sources(concessions)
    else:
        gdf_conc = load_kalimantan_coal_sources(concessions, purge_centroid_artifact=False)

    # Map concessions to grid (row, col) if not already provided
    lon_min, lat_min, lon_max, lat_max = bounds
    res_lon = (lon_max - lon_min) / float(W)
    res_lat = (lat_max - lat_min) / float(H)

    if "row" in gdf_conc.columns and "col" in gdf_conc.columns and gdf_conc["row"].notna().all():
        rows_origin = gdf_conc["row"].to_numpy(dtype=np.int32)
        cols_origin = gdf_conc["col"].to_numpy(dtype=np.int32)
    elif transform is not None and hasattr(transform, "a"):
        # Affine transform from rasterio
        inv_t = ~transform
        cols_f, rows_f = inv_t * (
            gdf_conc["longitude"].to_numpy(dtype=np.float64),
            gdf_conc["latitude"].to_numpy(dtype=np.float64),
        )
        rows_origin = np.floor(rows_f).astype(np.int32)
        cols_origin = np.floor(cols_f).astype(np.int32)
    else:
        rows_origin = np.floor((lat_max - gdf_conc["latitude"].to_numpy(dtype=np.float64)) / res_lat).astype(np.int32)
        cols_origin = np.floor((gdf_conc["longitude"].to_numpy(dtype=np.float64) - lon_min) / res_lon).astype(np.int32)

    gdf_conc = gdf_conc.copy()
    gdf_conc["row"] = rows_origin
    gdf_conc["col"] = cols_origin

    # Precompute disturbance mask (for Guardrail 1 rim tracing)
    is_disturbed = (
        (ndvi_arr < float(disturbance_ndvi_max))
        | (mine_arr > 0)
        | (gfw_arr >= 0.5)
        | ~np.isfinite(ndvi_arr)
    )

    # Precompute canopy footprint integrity pass mask (Guardrail 3, D6)
    # Every pixel in N_{k_size}(r', c') (750 x 750 m footprint) must have NDVI >= canopy_ndvi_min and active_mine_mask == 0
    single_pixel_intact = (
        np.isfinite(ndvi_arr)
        & (ndvi_arr >= float(canopy_ndvi_min))
        & (mine_arr == 0)
        & (gfw_arr < 0.5)
    )
    k_size = max(3, int(round(750.0 / max(1.0, float(px_size_m)))))
    if k_size % 2 == 0:
        k_size += 1
    k_half = k_size // 2

    canopy_3x3_valid = np.zeros((H, W), dtype=bool)
    if H >= k_size and W >= k_size:
        intact_core = np.ones((H - 2 * k_half, W - 2 * k_half), dtype=bool)
        for dr in range(-k_half, k_half + 1):
            for dc in range(-k_half, k_half + 1):
                intact_core &= single_pixel_intact[
                    k_half + dr : H - k_half + dr,
                    k_half + dc : W - k_half + dc,
                ]
        canopy_3x3_valid[k_half : H - k_half, k_half : W - k_half] = intact_core

    # Dynamic rank-adaptive offsets in grid cells
    rank_offsets = get_rank_adaptive_offsets_cells(px_size_m)

    # Output rasters
    label_k0 = np.full((H, W), -1, dtype=np.int8)
    label_k1 = np.full((H, W), -1, dtype=np.int8)
    label_k2 = np.full((H, W), -1, dtype=np.int8)
    head_rasters = {0: label_k0, 1: label_k1, 2: label_k2}

    tonnage_weights_grid = np.ones((H, W), dtype=np.float32)
    parent_concession_grid = np.zeros((H, W), dtype=np.int32)

    halo_records: list[dict[str, object]] = []

    for idx, row in gdf_conc.iterrows():
        r0 = int(row["row"])
        c0 = int(row["col"])
        if not (0 <= r0 < H and 0 <= c0 < W):
            continue

        origin_poly_id = int(poly_arr[r0, c0])
        # Guardrail 2 requires a valid mapped formation polygon (> 0)
        if origin_poly_id <= 0:
            continue

        rank_head = int(row["coal_rank_head"])
        if rank_head not in rank_offsets:
            continue
        d_min, d_max = rank_offsets[rank_head]
        concession_id = int(row["concession_id"])
        w_tonnage = float(row["tonnage_weight"])

        theta_deg = float(strike_arr[r0, c0])
        if not np.isfinite(theta_deg):
            theta_deg = 45.0
        theta_rad = np.deg2rad(theta_deg)
        # Unit strike vector (ur, uc) in (row, col). v4.15: matches the convention of
        # coal_and_multimodal_features._structure_tensor_eigen_and_strike, (row, col) = (cos theta, sin theta),
        # where theta = 0 is a north-south (along-row) bedding strike. It was (sin, cos) before, which laid
        # the halo across strike whenever theta was 0 or 90 deg (audit v4.13 section 1.5).
        ur_base = float(np.cos(theta_rad))
        uc_base = float(np.sin(theta_rad))

        # Step along both strike directions (+1 and -1)
        for sign in (+1.0, -1.0):
            ur = sign * ur_base
            uc = sign * uc_base

            # Guardrail 1: Rim-Anchored Origin (d=0 at Disturbance Boundary)
            # If the concession origin (r0, c0) sits inside a contiguous disturbance / open-pit
            # zone, trace along the ray until the outer boundary of that disturbance zone.
            r_rim, c_rim = r0, c0
            rim_step = 0
            if is_disturbed[r0, c0]:
                for s in range(1, int(max_rim_search_cells) + 1):
                    rs = int(np.round(r0 + s * ur))
                    cs = int(np.round(c0 + s * uc))
                    if not (0 <= rs < H and 0 <= cs < W):
                        break
                    if is_disturbed[rs, cs]:
                        r_rim, c_rim = rs, cs
                        rim_step = s
                    else:
                        # Reached undisturbed canopy just outside the disturbance rim
                        break

            # Step out from (r_rim, c_rim) by d in [d_min, d_max] cells
            seen_cells: set[tuple[int, int]] = set()
            for d in range(int(d_min), int(d_max) + 1):
                rc = int(np.round(r_rim + d * ur))
                cc = int(np.round(c_rim + d * uc))
                if (rc, cc) in seen_cells:
                    continue
                seen_cells.add((rc, cc))

                # Must lie inside valid footprint interior bounds
                if not (k_half <= rc < H - k_half and k_half <= cc < W - k_half):
                    continue

                # Guardrail 2: Same-Formation Polygon Lock
                cand_poly_id = int(poly_arr[rc, cc])
                if cand_poly_id != origin_poly_id or cand_poly_id <= 0:
                    continue

                # Guardrail 3: Canopy Integrity Check
                if not canopy_3x3_valid[rc, cc]:
                    continue

                # Valid undisturbed sub-canopy strike-extension halo positive!
                head_rasters[rank_head][rc, cc] = 1
                if w_tonnage > tonnage_weights_grid[rc, cc]:
                    tonnage_weights_grid[rc, cc] = np.float32(w_tonnage)
                if parent_concession_grid[rc, cc] == 0:
                    parent_concession_grid[rc, cc] = concession_id

                halo_records.append(
                    {
                        "row": rc,
                        "col": cc,
                        "origin_row": r0,
                        "origin_col": c0,
                        "rim_row": r_rim,
                        "rim_col": c_rim,
                        "rim_step_cells": rim_step,
                        "offset_cells": d,
                        "direction_sign": int(sign),
                        "strike_deg": theta_deg,
                        "coal_rank_head": rank_head,
                        "coal_rank_name": COAL_HEAD_NAMES[rank_head],
                        "parent_concession_id": concession_id,
                        "formation_polygon_id": cand_poly_id,
                        "origin_formation_polygon_id": origin_poly_id,
                        "tonnage_weight": w_tonnage,
                        "ndvi_center": float(ndvi_arr[rc, cc]),
                    }
                )

    halo_samples_df = pd.DataFrame(halo_records)
    if len(halo_samples_df) == 0:
        halo_samples_df = pd.DataFrame(
            columns=[
                "row",
                "col",
                "origin_row",
                "origin_col",
                "rim_row",
                "rim_col",
                "rim_step_cells",
                "offset_cells",
                "direction_sign",
                "strike_deg",
                "coal_rank_head",
                "coal_rank_name",
                "parent_concession_id",
                "formation_polygon_id",
                "origin_formation_polygon_id",
                "tonnage_weight",
                "ndvi_center",
            ]
        )

    return {
        "label_coal_k0_high_cv": label_k0,
        "label_coal_k1_med_cv": label_k1,
        "label_coal_k2_low_cv": label_k2,
        "coal_k0_high_cv": label_k0,
        "coal_k1_med_cv": label_k1,
        "coal_k2_low_cv": label_k2,
        "coal_tonnage_weights": tonnage_weights_grid,
        "coal_parent_concession_id": parent_concession_grid,
        "halo_samples_df": halo_samples_df,
        "concessions_df": gdf_conc,
        "rank_offset_config": dict(rank_offsets),
    }


def generate_rim_anchored_coal_labels(
    concessions: gpd.GeoDataFrame | pd.DataFrame | Path | str | None = None,
    ndvi: Optional[np.ndarray] = None,
    active_mine_mask: Optional[np.ndarray] = None,
    formation_polygon_id: Optional[np.ndarray] = None,
    sar_bedding_strike_deg: np.ndarray | float | None = None,
    gfw_forest_disturbance: np.ndarray | None = None,
    sar_hh_db: np.ndarray | None = None,
    transform: object | None = None,
    bounds: tuple[float, float, float, float] = (108.5, -4.5, 119.5, 4.5),
    canopy_ndvi_min: float = 0.25,
    disturbance_ndvi_max: float = 0.20,
    max_rim_search_cells: int = 40,
    px_size_m: float = 250.0,
    **kwargs: Any,
) -> dict[str, object]:
    """Generate rim-anchored K=3 coal labels parameterized in physical meters (D6, Bug 10).

    Alias for build_k3_coal_rank_targets ensuring physical metric scaling.
    """
    return build_k3_coal_rank_targets(
        concessions=concessions,
        ndvi=ndvi,
        active_mine_mask=active_mine_mask,
        formation_polygon_id=formation_polygon_id,
        sar_bedding_strike_deg=sar_bedding_strike_deg,
        gfw_forest_disturbance=gfw_forest_disturbance,
        sar_hh_db=sar_hh_db,
        transform=transform,
        bounds=bounds,
        canopy_ndvi_min=canopy_ndvi_min,
        disturbance_ndvi_max=disturbance_ndvi_max,
        max_rim_search_cells=max_rim_search_cells,
        px_size_m=px_size_m,
        **kwargs,
    )


def build_pooled_exploration_footprint(
    metallic_explored_footprint: np.ndarray,
    coal_concessions: gpd.GeoDataFrame | pd.DataFrame | None = None,
    active_mine_mask: np.ndarray | None = None,
    gfw_forest_disturbance: np.ndarray | None = None,
    bounds: tuple[float, float, float, float] = (108.5, -4.5, 119.5, 4.5),
    buffer_cells: Optional[int] = None,
    buffer_m: float = 2000.0,
    px_size_m: float = 250.0,
) -> np.ndarray:
    """Build pooled exploration intensity target E_pooled in {0, 1} for Pass 2 Propensity Model.

    Pools all documented geological exploration and extraction footprints across Kalimantan:
      1. K=4 Metallic deposit and exploration footprints (`metallic_explored_footprint`)
      2. All 850 ESDM coal concessions (`coal_concessions`) buffered by `buffer_m` (default 2,000 m)
      3. Mapped active mining polygons (`active_mine_mask`) and severe GFW mining disturbances
    """
    from scipy.ndimage import binary_dilation

    if buffer_cells is None:
        buffer_cells = max(1, int(round(float(buffer_m) / max(1.0, float(px_size_m)))))

    base_fp = (np.asarray(metallic_explored_footprint) > 0).copy()
    H, W = base_fp.shape

    coal_pts = np.zeros((H, W), dtype=bool)
    if coal_concessions is not None and len(coal_concessions) > 0:
        lon_min, lat_min, lon_max, lat_max = bounds
        res_lon = (lon_max - lon_min) / float(W)
        res_lat = (lat_max - lat_min) / float(H)

        if "row" in coal_concessions.columns and "col" in coal_concessions.columns:
            rows = coal_concessions["row"].to_numpy(dtype=np.int32)
            cols = coal_concessions["col"].to_numpy(dtype=np.int32)
        else:
            rows = np.floor((lat_max - coal_concessions["latitude"].to_numpy(dtype=np.float64)) / res_lat).astype(np.int32)
            cols = np.floor((coal_concessions["longitude"].to_numpy(dtype=np.float64) - lon_min) / res_lon).astype(np.int32)

        valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        coal_pts[rows[valid], cols[valid]] = True

        if buffer_cells > 0 and coal_pts.any():
            y, x = np.ogrid[-buffer_cells : buffer_cells + 1, -buffer_cells : buffer_cells + 1]
            struct = (x * x + y * y) <= (buffer_cells * buffer_cells)
            coal_pts = binary_dilation(coal_pts, structure=struct)

    pooled = base_fp | coal_pts
    if active_mine_mask is not None:
        pooled |= np.asarray(active_mine_mask) > 0
    # v4.15: forest loss is no longer treated as exploration. 77% of the old footprint was cells with
    # loss >= 0.75, i.e. plantation, logging and fire, not prospecting (audit v4.13 section 2.8).
    # ``gfw_forest_disturbance`` is accepted for API compatibility and ignored.

    return pooled.astype(np.int8)


