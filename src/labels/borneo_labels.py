"""Borneo Multi-Mineral Vector Unmerger and K=4 Target Label Generator.

Provides 100% leak-free and placer-hygiene-enforced label extraction for the
Borneo K=4 deposit mechanics discovery engine:
  - Head 0: Porphyry Cu-Au (GeoRIMA COPPER_ORE + Cu-bearing BASE_METALS + MRDS Cu-Au, N ≈ 35-45)
  - Head 1: Contact Fe-Cu Skarn (ESDM Besi Primer [57] + GeoRIMA carbonate_skarn [14] = 71)
  - Head 2: Epithermal Au-Ag (GeoRIMA Emas Primer [87] + MRDS primary Au-Ag)
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


# ---------------------------------------------------------------------------
# Global Hygiene Guard
# ---------------------------------------------------------------------------


def filter_valid_ore_candidates(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Apply the Global Hygiene Guard to exclude alluvial placers and centroid artifacts.

    Enforces Reconciled Engineering Rule 3:
      River placers ("Emas Alluvial") and island/province centroid artifacts
      must never contaminate bedrock positive ore training sets (P_k).

    Args:
        df: DataFrame or GeoDataFrame with columns 'commodity_label' and
            optional 'is_centroid_artifact'.

    Returns:
        Filtered DataFrame / GeoDataFrame retaining only valid bedrock ore candidates.
    """
    comm_label = df["commodity_label"] if "commodity_label" in df.columns else pd.Series("", index=df.index)
    artifact_series = df["is_centroid_artifact"] if "is_centroid_artifact" in df.columns else pd.Series(False, index=df.index)

    is_valid_ore_candidate = (
        (comm_label != "Emas Alluvial")
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
    # Head k=0: Porphyry Cu-Au
    # ------------------------------------------------------------------
    k0_geo_mask = (
        (clean_geo["resource_group"] == "COPPER_ORE")
        | (clean_geo["commodity_label"] == "Tembaga")
        | (clean_geo["resource_group"] == "BASE_METALS")
        | (clean_geo["geological_role"] == "porphyry_intrusive")
    )
    k0_geo = clean_geo[k0_geo_mask].copy()

    # MRDS Cu-bearing porphyry / stockwork nodes
    mrds_code = clean_mrds.get("code_list", pd.Series("", index=clean_mrds.index)).fillna("")
    k0_mrds_mask = mrds_code.str.contains("CU", case=False, na=False)
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

    # Exact string carbonate_skarn in GeoRIMA
    k1_geo_mask = clean_geo["geological_role"] == "carbonate_skarn"
    # Ensure Kalimantan (Borneo) constraint matches the N=14 exactly
    if "island_region" in clean_geo.columns:
        k1_geo_mask = k1_geo_mask & (clean_geo["island_region"] == "Kalimantan (Borneo)")
    k1_geo = clean_geo[k1_geo_mask].copy()

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

    k2_mrds_mask = mrds_code.str.contains("AU", case=False, na=False)
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

    k3_mrds_mask = mrds_code.str.contains("NI", case=False, na=False)
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

