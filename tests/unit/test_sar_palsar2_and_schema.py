"""Unit tests for real ALOS-2 PALSAR-2 ingestion and warm-start schema resolution."""

import numpy as np
import pytest
from src.features.sar_palsar2 import (
    SAR_CHANNEL_NAMES,
    compute_palsar2_channels,
    dn_to_gamma0_db,
    dn_to_linear_power,
    linear_power_to_db,
    steger_ridge_strength,
)
from src.schema.canonical import resolve_warm_start_feature_schema


def test_palsar2_calibration_math():
    """Verify JAXA PALSAR-2 calibration: gamma0_dB = 10 * log10(DN^2) - 83.0."""
    # Test DN = 10000 -> 10 * log10(1e8) - 83.0 = 80.0 - 83.0 = -3.0 dB
    dn = np.array([10000.0, 0.0, -10.0])
    lin = dn_to_linear_power(dn)
    assert np.isclose(lin[0], 1e8 * (10.0 ** (-8.3)))
    assert np.isnan(lin[1])
    assert np.isnan(lin[2])

    db = linear_power_to_db(lin)
    assert np.isclose(db[0], -3.0)
    assert np.isnan(db[1])
    assert np.isnan(db[2])

    db_direct = dn_to_gamma0_db(dn)
    assert np.isclose(db_direct[0], -3.0)
    assert np.isnan(db_direct[1])


def test_steger_ridge_strength():
    """Verify Steger ridge filter outputs [0, 1] normalized response."""
    img = np.zeros((50, 50), dtype=np.float32)
    # Put a linear bright ridge
    img[:, 25] = 10.0
    ridge = steger_ridge_strength(img, sigma=1.5)
    assert ridge.shape == (50, 50)
    assert ridge.min() >= 0.0
    assert ridge.max() <= 1.0
    # Ridge should be strongest along the line
    assert ridge[:, 25].mean() > ridge[:, 5].mean()


def test_compute_palsar2_channels():
    """Verify calculation of all 6 canonical SAR channels with NaN preservation."""
    H, W = 40, 40
    hh = np.full((H, W), -8.0, dtype=np.float32)
    hv = np.full((H, W), -14.0, dtype=np.float32)
    dem = np.full((H, W), 100.0, dtype=np.float32)

    # Set some invalid NaN cells
    hh[0, 0] = np.nan
    hv[0, 0] = np.nan

    channels = compute_palsar2_channels(hh_db=hh, hv_db=hv, dem=dem, res_m=250.0)

    for name in SAR_CHANNEL_NAMES:
        assert name in channels
        arr = channels[name]
        assert arr.shape == (H, W)
        assert np.isnan(arr[0, 0])  # Masked cell remains NaN
        assert np.isfinite(arr[1:, 1:]).all()

    # Dual-pol ratio check: HV/HH in linear power = 10^(-1.4) / 10^(-0.8) = 10^(-0.6) ~ 0.251
    dpr_expected = 10.0 ** (-0.6)
    assert np.isclose(channels["sar_palsar2_dpr"][1, 1], dpr_expected, rtol=1e-3)


def test_resolve_warm_start_feature_schema_no_prior():
    """When no prior exists, returns candidate columns directly."""
    candidates = ["dem_elevation", "sar_palsar2_hh"]
    decision = resolve_warm_start_feature_schema(
        candidate_cols=candidates,
        prior_feature_names=None,
        available_cols=candidates,
    )
    assert decision["feature_cols"] == candidates
    assert decision["use_prior"] is False
    assert decision["reason"] == "no_prior"


def test_resolve_warm_start_feature_schema_matching():
    """When prior matches candidates, uses prior."""
    cols = ["dem_elevation", "sar_palsar2_hh"]
    decision = resolve_warm_start_feature_schema(
        candidate_cols=cols,
        prior_feature_names=cols,
        available_cols=cols,
    )
    assert decision["feature_cols"] == cols
    assert decision["use_prior"] is True
    assert decision["reason"] == "schema_match"
    assert len(decision["dropped"]) == 0
    assert len(decision["missing"]) == 0


def test_resolve_warm_start_feature_schema_retrain_on_mismatch():
    """When dataset has new channels (e.g. SAR) and prior does not, retrain policy trains from zero."""
    candidates = ["dem_elevation", "sar_palsar2_hh", "sar_palsar2_hv"]
    prior = ["dem_elevation"]  # 21-feature legacy model without SAR
    decision = resolve_warm_start_feature_schema(
        candidate_cols=candidates,
        prior_feature_names=prior,
        available_cols=candidates,
        policy="retrain",
    )
    # Under retrain, warm start is discarded so new SAR channels are learned!
    assert decision["feature_cols"] == candidates
    assert decision["use_prior"] is False
    assert decision["reason"] == "schema_mismatch_retrain"
    assert "sar_palsar2_hh" in decision["dropped"]
    assert "sar_palsar2_hv" in decision["dropped"]


def test_resolve_warm_start_feature_schema_prior_policy_and_scan_only():
    """When policy is 'prior' or scan_only=True, prior schema is enforced."""
    candidates = ["dem_elevation", "sar_palsar2_hh"]
    prior = ["dem_elevation"]
    decision = resolve_warm_start_feature_schema(
        candidate_cols=candidates,
        prior_feature_names=prior,
        available_cols=candidates,
        policy="prior",
    )
    assert decision["feature_cols"] == prior
    assert decision["use_prior"] is True
    assert decision["reason"] == "policy_prior"

    # scan_only always uses prior even if policy="retrain"
    decision_scan = resolve_warm_start_feature_schema(
        candidate_cols=candidates,
        prior_feature_names=prior,
        available_cols=candidates,
        scan_only=True,
        policy="retrain",
    )
    assert decision_scan["feature_cols"] == prior
    assert decision_scan["use_prior"] is True
    assert decision_scan["reason"] == "scan_only_prior_schema"
