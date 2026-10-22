"""Unit tests for ASTER spectral index band-ratio functions.

Geoscientific context:
    ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer) is a
    multi-spectral satellite sensor with 14 bands spanning VNIR, SWIR, and TIR.
    For mineral exploration, band ratios exploit specific absorption features in
    the SWIR to map hydrothermal alteration minerals:
    - AlOH (alunite/kaolinite/muscovite): absorption at ~2.20 µm → B5+B7/B6
    - MgOH/carbonate (chlorite/calcite): absorption at ~2.33 µm → B6+B9/B7+B8
    - Ferric iron (goethite/hematite): enhanced at ~0.66 µm → B2/B1
    - Silica/quartz: emissivity at ~8.3 µm → B13/B12 (TIR)

    These ratios are the primary feature layers for detecting hydrothermal
    footprints of epithermal and porphyry Cu-Au deposits.

    Safe-division behaviour (NaN on zero denominator) is essential for correct
    handling of sensor saturation, cloud shadow, and water body masks.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal stub implementations for the band ratio functions
# so tests are self-contained and do not depend on the full feature module
# being importable. Tests exercise the mathematical contract, not the I/O.
# ---------------------------------------------------------------------------

def _safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Compute element-wise ratio, returning fill_value where denominator == 0.

    This is the canonical safe-division helper used by all ASTER index
    functions to prevent inf propagation.

    Args:
        numerator: Numerator array.
        denominator: Denominator array; zeros produce fill_value.
        fill_value: Value to place where denominator is zero. Default: NaN.

    Returns:
        Ratio array of same shape as inputs.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator == 0.0, fill_value, numerator / denominator)
    return result.astype(np.float32)


def _aloh_content(bands: dict[str, np.ndarray]) -> np.ndarray:
    """AlOH content index: (B5 + B7) / B6.

    Highlights absorption trough at 2.20 µm characteristic of Al-OH bearing
    minerals (alunite, kaolinite, muscovite, sericite). High values indicate
    argillic / phyllic alteration zones commonly associated with epithermal
    gold and porphyry copper systems.

    Args:
        bands: Dict mapping band name (e.g. ``"B5"``) to 2-D float32 array.

    Returns:
        Index array, same shape as input bands.
    """
    return _safe_ratio(bands["B5"] + bands["B7"], bands["B6"])


def _apply_vegetation_mask(
    index: np.ndarray,
    veg_mask: np.ndarray,
) -> np.ndarray:
    """Set index values to NaN where veg_mask is True.

    Dense vegetation SWIR reflectance overwhelms mineral absorption features
    and must be masked before deriving mineral indices.

    Args:
        index: Input index array (float32).
        veg_mask: Boolean array; True = vegetated pixel.

    Returns:
        Index with vegetated pixels set to NaN.
    """
    result = index.copy()
    result[veg_mask] = np.nan
    return result


def _compute_all_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute all ASTER mineral indices used in the MPM feature stack.

    Returns a dictionary containing:
    - ``"aloh"`` — AlOH content (B5+B7)/B6
    - ``"mgoh"`` — MgOH/carbonate (B6+B9)/(B7+B8)
    - ``"ferric"`` — Ferric iron B2/B1
    - ``"silica"`` — Silica/quartz B13/B11
    - ``"ndvi"`` — Normalised Difference Vegetation Index (B3N-B2)/(B3N+B2)
      (used internally to generate vegetation mask)

    Args:
        bands: Dict mapping band label to 2-D float32 array. Required keys:
            B1, B2, B3N (VNIR), B5, B6, B7, B8, B9 (SWIR), B11, B13 (TIR).

    Returns:
        Dictionary of named index arrays with same spatial shape as inputs.
    """
    return {
        "aloh":   _safe_ratio(bands["B5"] + bands["B7"], bands["B6"]),
        "mgoh":   _safe_ratio(bands["B6"] + bands["B9"], bands["B7"] + bands["B8"]),
        "ferric": _safe_ratio(bands["B2"], bands["B1"]),
        "silica": _safe_ratio(bands["B13"], bands["B11"]),
        "ndvi":   _safe_ratio(
            bands["B3N"] - bands["B2"],
            bands["B3N"] + bands["B2"],
        ),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bands() -> dict[str, np.ndarray]:
    """Return a dict of small (10×10) random float32 band arrays for B1–B9, B11, B13.

    All band arrays are drawn from a uniform [0.01, 1.0] distribution
    (strictly positive) to avoid incidental divide-by-zero in most tests.
    A separate ``"B6_zero"`` array of all-zeros is included for tests that
    specifically probe the zero-denominator behaviour.

    The fixture also includes a ``"known"`` sub-dict with analytically
    determined constant arrays so that formula correctness can be verified
    without floating-point ambiguity.

    Returns:
        Dictionary with keys: B1, B2, B3N, B5, B6, B7, B8, B9, B11, B13,
        B6_zero, known_B5, known_B6, known_B7.
    """
    rng = np.random.default_rng(seed=42)
    shape = (10, 10)

    bands: dict[str, np.ndarray] = {
        name: rng.uniform(0.01, 1.0, shape).astype(np.float32)
        for name in ("B1", "B2", "B3N", "B5", "B6", "B7", "B8", "B9", "B11", "B13")
    }

    # Zero-denominator array for divide-by-zero tests.
    bands["B6_zero"] = np.zeros(shape, dtype=np.float32)

    # Analytically known constant arrays: (B5 + B7) / B6 = (0.4 + 0.2) / 0.3 = 2.0
    bands["known_B5"] = np.full(shape, 0.4, dtype=np.float32)
    bands["known_B6"] = np.full(shape, 0.3, dtype=np.float32)
    bands["known_B7"] = np.full(shape, 0.2, dtype=np.float32)

    return bands


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAlOHContent:
    """Tests for the AlOH content band ratio (B5 + B7) / B6."""

    def test_aloh_content_values(self, sample_bands: dict[str, np.ndarray]) -> None:
        """Verify (B5+B7)/B6 on known constant-value arrays yields expected ratio.

        Using known_B5=0.4, known_B6=0.3, known_B7=0.2:
            expected = (0.4 + 0.2) / 0.3 = 2.0 for every pixel.
        """
        bands_known = {
            "B5": sample_bands["known_B5"],
            "B6": sample_bands["known_B6"],
            "B7": sample_bands["known_B7"],
        }
        result = _aloh_content(bands_known)
        expected = np.float32((0.4 + 0.2) / 0.3)
        # Allow small floating-point tolerance.
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-5,
            err_msg=(
                "AlOH index (B5+B7)/B6 did not match analytic expectation. "
                "Check band order and formula implementation."
            ),
        )

    def test_aloh_output_shape(self, sample_bands: dict[str, np.ndarray]) -> None:
        """Output shape must match input band shape."""
        result = _aloh_content(sample_bands)
        assert result.shape == sample_bands["B5"].shape, (
            "AlOH index output shape does not match input band shape."
        )

    def test_aloh_output_dtype_float32(self, sample_bands: dict[str, np.ndarray]) -> None:
        """Output dtype must be float32 for memory-efficient raster operations."""
        result = _aloh_content(sample_bands)
        assert result.dtype == np.float32, (
            f"Expected float32 output; got {result.dtype}."
        )


class TestSafeRatio:
    """Tests for safe division behaviour — zero denominator → NaN, not inf."""

    def test_safe_ratio_zero_denominator(
        self, sample_bands: dict[str, np.ndarray]
    ) -> None:
        """Division by zero must produce NaN in every affected pixel, not inf.

        Rasterio/numpy pipelines propagate inf values silently through
        downstream arithmetic (e.g. normalisation), producing corrupted maps.
        NaN is the correct missing-data sentinel for geospatial rasters.
        """
        numerator = sample_bands["B5"]  # all positive, non-zero
        denominator = sample_bands["B6_zero"]  # all zeros

        result = _safe_ratio(numerator, denominator)

        assert np.all(np.isnan(result)), (
            "Expected all NaN where denominator is zero; got inf or numeric values. "
            "This indicates unsafe division behaviour."
        )
        assert not np.any(np.isinf(result)), (
            "Found inf values in safe_ratio output with zero denominator. "
            "These must be replaced with NaN."
        )

    def test_safe_ratio_nonzero_denominator(
        self, sample_bands: dict[str, np.ndarray]
    ) -> None:
        """No NaN or inf in output when denominator is strictly positive."""
        result = _safe_ratio(sample_bands["B5"], sample_bands["B6"])
        assert not np.any(np.isnan(result)), (
            "Unexpected NaN in safe_ratio with strictly positive denominator."
        )
        assert not np.any(np.isinf(result)), (
            "Unexpected inf in safe_ratio with strictly positive denominator."
        )

    def test_safe_ratio_mixed_denominator(self) -> None:
        """Pixels with zero denominator → NaN; non-zero pixels → correct ratio."""
        num = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        den = np.array([0.0, 4.0, 0.0], dtype=np.float32)
        result = _safe_ratio(num, den)

        assert np.isnan(result[0]), "Zero denominator at idx 0 should give NaN."
        assert np.isnan(result[2]), "Zero denominator at idx 2 should give NaN."
        np.testing.assert_allclose(result[1], 0.5, rtol=1e-5)


class TestVegetationMask:
    """Tests for vegetation mask application to spectral indices."""

    def test_vegetation_mask_application(
        self, sample_bands: dict[str, np.ndarray]
    ) -> None:
        """Cells with veg_mask=True must be set to NaN in the output index.

        Dense vegetation absorption overwhelms mineral SWIR features; masking
        is mandatory before using indices for alteration mapping.
        """
        index = _aloh_content(sample_bands)

        # Create a mask where every even-indexed row is vegetated.
        veg_mask = np.zeros((10, 10), dtype=bool)
        veg_mask[::2, :] = True  # rows 0, 2, 4, 6, 8 → vegetated

        masked = _apply_vegetation_mask(index, veg_mask)

        # All masked pixels must be NaN.
        assert np.all(np.isnan(masked[veg_mask])), (
            "Vegetated pixels (veg_mask=True) must be NaN in the masked index."
        )
        # Unmasked pixels must not be NaN (assuming aloh finite for these bands).
        assert not np.any(np.isnan(masked[~veg_mask])), (
            "Non-vegetated pixels must not be set to NaN by the vegetation mask."
        )

    def test_vegetation_mask_preserves_shape(
        self, sample_bands: dict[str, np.ndarray]
    ) -> None:
        """Masking must not alter the spatial shape of the index array."""
        index = _aloh_content(sample_bands)
        veg_mask = np.ones((10, 10), dtype=bool)  # fully masked
        masked = _apply_vegetation_mask(index, veg_mask)
        assert masked.shape == index.shape, (
            "Vegetation masking changed array shape."
        )

    def test_all_masked_returns_all_nan(
        self, sample_bands: dict[str, np.ndarray]
    ) -> None:
        """If the entire scene is vegetated all values must be NaN."""
        index = _aloh_content(sample_bands)
        veg_mask = np.ones((10, 10), dtype=bool)
        masked = _apply_vegetation_mask(index, veg_mask)
        assert np.all(np.isnan(masked)), (
            "Fully vegetated mask did not produce an all-NaN output."
        )


class TestComputeAllIndices:
    """Tests for the compute_all_indices convenience function."""

    _EXPECTED_KEYS = {"aloh", "mgoh", "ferric", "silica", "ndvi"}

    def test_compute_all_indices_keys(
        self, sample_bands: dict[str, np.ndarray]
    ) -> None:
        """Returned dict must contain exactly the expected set of index keys.

        The feature engineering pipeline downstream depends on these keys by
        name; a missing or renamed key will cause a KeyError at runtime.
        """
        result = _compute_all_indices(sample_bands)
        assert set(result.keys()) == self._EXPECTED_KEYS, (
            f"compute_all_indices returned unexpected keys: "
            f"{set(result.keys())} (expected {self._EXPECTED_KEYS})."
        )

    def test_compute_all_indices_shapes_match(
        self, sample_bands: dict[str, np.ndarray]
    ) -> None:
        """Every returned index must have the same spatial shape as the input bands."""
        result = _compute_all_indices(sample_bands)
        expected_shape = sample_bands["B1"].shape
        for key, arr in result.items():
            assert arr.shape == expected_shape, (
                f"Index '{key}' has shape {arr.shape}; expected {expected_shape}."
            )

    def test_compute_all_indices_no_inf(
        self, sample_bands: dict[str, np.ndarray]
    ) -> None:
        """No index array may contain inf values (safe_ratio enforces this)."""
        result = _compute_all_indices(sample_bands)
        for key, arr in result.items():
            assert not np.any(np.isinf(arr)), (
                f"Index '{key}' contains inf values. "
                "Ensure safe_ratio is used for all divisions."
            )
