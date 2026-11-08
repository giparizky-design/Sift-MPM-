"""Canonical schema and column definitions for the MPM v4.0 Pipeline.

Centralizes all non-feature column names, forbidden raw sensor band patterns,
commodity head definitions, domain priors, and label conventions across the
MPM pipeline.
"""

from __future__ import annotations

import re
from typing import Dict, Pattern, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Canonical Non-Feature Columns (24 Items)
# ---------------------------------------------------------------------------
# Authoritative single source of truth for non-feature, spatial index, label,
# and quarantined circular-leakage columns.
NON_FEATURE_COLS: Tuple[str, ...] = (
    # Spatial coordinates & grid indices (10)
    "x",
    "y",
    "lon",
    "lat",
    "row",
    "col",
    "spatial_block",
    "block_id",
    "block_x",
    "block_y",
    # Targets & commodity labels (5)
    "label",
    "label_coal",
    "deposit_class",
    "formation_polygon_id",
    "coal_parent_concession_id",
    # Quarantined circular-distance leakage columns (6)
    "geology_coal_contact_dist_km",
    "geology_felsic_intrusive_dist_km",
    "dist_kelian_center_m",
    "dist_transport_km",
    "dist_major_rivers_km",
    "dist_haul_roads_km",
    # Quarantined binary polygon masks (3)
    "in_coal_bearing_formation",
    "in_ophiolite_unit",
    "in_holocene_peat_or_karst_mask",
)

NON_FEATURE_COLS_SET: Set[str] = set(NON_FEATURE_COLS)

# Extended non-feature columns including environmental & temporal disturbance masks (33 items)
EXTENDED_NON_FEATURE_COLS: Tuple[str, ...] = (
    *NON_FEATURE_COLS,
    "active_mine_mask",
    "gfw_forest_disturbance",
    "forest_disturbance_rate",
    "coal_tonnage_weight",
    "explored_footprint",
    "mine_dist",
    "deposit_dist",
    "regional_fe",
    "regional_coal",
)

# ---------------------------------------------------------------------------
# Forbidden Raw Band Patterns
# ---------------------------------------------------------------------------
# String substring patterns preserved for backward compatibility
_FORBIDDEN_RAW_BAND_PATTERNS: Tuple[str, ...] = (
    "raw_band",
    "aster_b",
    "sentinel_b",
    "s2_b",
    "b01",
    "b02",
    "b03",
    "b04",
    "b05",
    "b06",
    "b07",
    "b08",
    "b8a",
    "b09",
    "b10",
    "b11",
    "b12",
    "b13",
    "b14",
    "landsat_b",
    "emit_raw",
    "prisma_raw",
    "enmap_raw",
    "dn_",
    "radiance_",
    "reflectance_band",
    "kelian",
    "georima",
    "mrds",
    "minedex",
    "mine_dist",
    "deposit_dist",
    "regional_fe",
    "regional_coal",
    "dist_major_rivers_km",
    "dist_haul_roads_km",
    "dist_transport_km",
    "gfw_forest_disturbance",
    "forest_disturbance_rate",
    "explored_footprint",
    "active_mine_mask",
    "formation_polygon_id",
    "geology_coal_contact_dist_km",
    "in_coal_bearing_formation",
    "in_holocene_peat_or_karst_mask",
    "in_ophiolite_unit",
)

# Compiled regular expressions for strict pattern matching
FORBIDDEN_RAW_BAND_REGEXES: Tuple[Pattern[str], ...] = (
    re.compile(r"^b\d+$", re.IGNORECASE),
    re.compile(r"^s2_b\d+[a-z]?$", re.IGNORECASE),
    re.compile(r"^aster_b\d+$", re.IGNORECASE),
    re.compile(r"^sentinel_b\d+$", re.IGNORECASE),
    re.compile(r"^landsat_b\d+$", re.IGNORECASE),
    re.compile(r"^raw_band.*", re.IGNORECASE),
    re.compile(r"^dn_.*", re.IGNORECASE),
    re.compile(r"^radiance_.*", re.IGNORECASE),
    re.compile(r"^reflectance_band.*", re.IGNORECASE),
    re.compile(r"^emit_raw.*", re.IGNORECASE),
    re.compile(r"^prisma_raw.*", re.IGNORECASE),
    re.compile(r"^enmap_raw.*", re.IGNORECASE),
    re.compile(r"^dist_kelian_center_m$", re.IGNORECASE),
    re.compile(r"^geology_coal_contact_dist_km$", re.IGNORECASE),
    re.compile(r"^in_coal_bearing_formation$", re.IGNORECASE),
    re.compile(r"^in_ophiolite_unit$", re.IGNORECASE),
    re.compile(r"^in_holocene_peat_or_karst_mask$", re.IGNORECASE),
)

# ---------------------------------------------------------------------------
# Commodity Head Names
# ---------------------------------------------------------------------------
K4_HEAD_NAMES: Tuple[str, ...] = (
    "porphyry_cu",
    "skarn_fe_cu",
    "epithermal_au",
    "laterite_fe_ni",
)

K3_COAL_HEAD_NAMES: Tuple[str, ...] = (
    "coal_k0_high_cv",
    "coal_k1_med_cv",
    "coal_k2_low_cv",
)

# ---------------------------------------------------------------------------
# Domain Priors
# ---------------------------------------------------------------------------
# Discovery / observation priors (c_hat floors)
DISCOVERY_PRIORS_K4: Dict[str, float] = {
    "porphyry_cu": 0.05,
    "skarn_fe_cu": 0.03,
    "epithermal_au": 0.08,
    "laterite_fe_ni": 0.02,
}

COAL_PRIORS_K3: Dict[str, float] = {
    "coal_k0_high_cv": 0.02,
    "coal_k1_med_cv": 0.04,
    "coal_k2_low_cv": 0.05,
}

# Domain geological class priors (true regional prevalence pi_k)
METALS_PREVALENCE_PRIORS: Dict[str, float] = {
    "porphyry_cu": 0.005,
    "skarn_fe_cu": 0.010,
    "epithermal_au": 0.015,
    "laterite_fe_ni": 0.008,
}

COAL_PREVALENCE_PRIORS: Dict[str, float] = {
    "coal_k0_high_cv": 0.020,
    "coal_k1_med_cv": 0.050,
    "coal_k2_low_cv": 0.040,
}

# ---------------------------------------------------------------------------
# Label Constants
# ---------------------------------------------------------------------------
LABEL_P: int = 1
LABEL_RN: int = 0
LABEL_U: int = -1


def verify_no_raw_pixel_leakage(feature_names: Sequence[str]) -> bool:
    """Verify feature columns contain only Pass 1 physical/geological indices.

    Enforces Acceptance Criterion 1 (No Raw Pixel Leakage): raw sensor
    reflectance/radiance bands (e.g. aster_b4, raw_band_1, dn_3) or quarantined
    spatial target distance columns must never be passed to tabular or spatial
    learners.

    Args:
        feature_names: Sequence of feature channel/column names.

    Returns:
        True if all feature names pass verification.

    Raises:
        ValueError: If any feature name matches a forbidden raw band or leakage pattern.
    """
    violating: list[str] = []
    for name in feature_names:
        low = name.strip().lower()
        if any(pat in low for pat in _FORBIDDEN_RAW_BAND_PATTERNS):
            violating.append(name)
        elif any(regex.match(low) for regex in FORBIDDEN_RAW_BAND_REGEXES):
            violating.append(name)
    if violating:
        raise ValueError(
            f"Raw sensor pixel leakage detected in feature matrix: {violating}. "
            "Only Pass 1 physical, structural, and geomechanical indices are allowed."
        )
    return True


WARM_START_SCHEMA_POLICIES: Tuple[str, ...] = ("retrain", "prior")


def resolve_warm_start_feature_schema(
    candidate_cols: Sequence[str],
    prior_feature_names: Sequence[str] | None,
    available_cols: Sequence[str],
    scan_only: bool = False,
    policy: str = "retrain",
    excluded_cols: Sequence[str] = (),
) -> Dict[str, object]:
    """Decide which feature schema to use when a prior checkpoint is loaded.

    Boosted-tree warm-starts require the exact feature schema of the prior model.
    Previously the prior schema silently overrode the dataset's columns, so newly
    ingested channels (e.g. real PALSAR-2 SAR) were dropped without any log.

    Policies (training only; ``scan_only`` always uses the prior schema because
    inference must match the trained model):
      - ``"retrain"``: if the dataset offers channels the prior model lacks, or the
        prior requires channels the dataset lacks, discard the warm-start and train
        from zero on the dataset's full feature set.
      - ``"prior"``: keep the prior schema (legacy behaviour); dropped / missing
        columns are reported explicitly.

    Returns:
        Dict with ``feature_cols`` (list), ``use_prior`` (bool), ``dropped``
        (dataset columns ignored), ``missing`` (prior columns absent from the
        dataset and zero-filled when ``use_prior``), and ``reason`` (str).
    """
    if policy not in WARM_START_SCHEMA_POLICIES:
        raise ValueError(f"policy must be one of {WARM_START_SCHEMA_POLICIES}, got '{policy}'")
    cand = list(candidate_cols)
    if not prior_feature_names:
        return {"feature_cols": cand, "use_prior": False, "dropped": [], "missing": [], "reason": "no_prior"}

    excluded = set(excluded_cols)
    prior = [f for f in prior_feature_names if f not in excluded and not f.startswith("label_")]
    avail = set(available_cols)
    prior_set = set(prior)
    dropped = [c for c in cand if c not in prior_set]
    missing = [f for f in prior if f not in avail]

    if not dropped and not missing:
        return {"feature_cols": prior, "use_prior": True, "dropped": [], "missing": [], "reason": "schema_match"}
    if scan_only:
        return {"feature_cols": prior, "use_prior": True, "dropped": dropped, "missing": missing, "reason": "scan_only_prior_schema"}
    if policy == "prior":
        return {"feature_cols": prior, "use_prior": True, "dropped": dropped, "missing": missing, "reason": "policy_prior"}
    return {"feature_cols": cand, "use_prior": False, "dropped": dropped, "missing": missing, "reason": "schema_mismatch_retrain"}
