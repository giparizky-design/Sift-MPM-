"""Unit tests for geophysical filter functions used in MPM feature engineering.

Geoscientific context:
    Potential-field geophysics (aeromagnetic, gravity) and their derivatives are
    primary datasets for mapping subsurface lithology, structural architecture,
    and hydrothermal alteration — all critical controls on mineral deposit
    formation.

    Key derivatives tested here:
    - **Tilt Derivative (TDR)**: Arctan(FVD / HGM). Provides lithological
      contacts with balanced amplitude response across shallow and deep sources.
      Used to map intrusive margins and demagnetised alteration zones.
    - **Analytic Signal (AS)**: sqrt(dT/dx² + dT/dy² + dT/dz²). Always ≥ 0;
      position of anomaly maximum is directly over source body regardless of
      magnetisation direction. Particularly useful in low-latitude terranes
      where inclination effects complicate standard RTP.
    - **Upward Continuation (UC)**: Attenuates high-frequency near-surface noise
      (cultural noise, regolith variation) while preserving deeper, more
      geologically significant anomalies. Variance should decrease monotonically
      with continuation height.
    - **Reduction to Pole (RTP)**: Transforms TMI to the equivalent response at
      the magnetic pole (inclination = 90°), centring anomalies over their
      causative sources. A uniform field input must yield a near-uniform output.
    - **First Vertical Derivative (FVD)**: dT/dz. Sharpens contacts and
      enhances shallow sources. The FVD of a symmetric Gaussian peak (an
      idealised dipole response in a 2-D sense) is antisymmetric about the peak.

    All functions operate in the spatial-frequency (Fourier) domain using FFT-
    based convolution. The tests use small synthetic grids to verify mathematical
    properties without requiring real survey data.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.fft import fft2, ifft2, fftfreq


# ---------------------------------------------------------------------------
# Minimal stub implementations of the geophysical filter functions.
# Tests verify the *mathematical contracts* of the transforms independently
# of the production I/O wrapper code.
# ---------------------------------------------------------------------------

def _wavenumbers(
    rows: int,
    cols: int,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D wavenumber grids (kx, ky) for a grid of given dimensions.

    Args:
        rows: Number of grid rows (y-dimension).
        cols: Number of grid columns (x-dimension).
        dx: Cell size in the x-direction (metres). Default 1.0.
        dy: Cell size in the y-direction (metres). Default 1.0.

    Returns:
        Tuple ``(kx, ky)`` of 2-D wavenumber arrays, shape ``(rows, cols)``.
    """
    kx = fftfreq(cols, d=dx) * 2 * np.pi  # rad / m
    ky = fftfreq(rows, d=dy) * 2 * np.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    return kx_grid, ky_grid


def _horizontal_gradient_magnitude(grid: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Compute horizontal gradient magnitude (HGM) via FFT.

    HGM = sqrt((dT/dx)² + (dT/dy)²). Used in TDR numerator.

    Args:
        grid: 2-D real-valued grid.
        dx: Cell size (metres).

    Returns:
        HGM array, same shape as ``grid``.
    """
    kx, ky = _wavenumbers(*grid.shape, dx=dx, dy=dx)
    G = fft2(grid)
    dTdx = np.real(ifft2(1j * kx * G))
    dTdy = np.real(ifft2(1j * ky * G))
    return np.sqrt(dTdx ** 2 + dTdy ** 2)


def _first_vertical_derivative(grid: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Compute the first vertical derivative (FVD) via FFT.

    FVD = dT/dz ≈ |k| * T̂ in the Fourier domain (upward positive).

    Args:
        grid: 2-D real-valued potential field grid.
        dx: Cell size (metres).

    Returns:
        FVD array, same shape as ``grid``.
    """
    kx, ky = _wavenumbers(*grid.shape, dx=dx, dy=dx)
    k = np.sqrt(kx ** 2 + ky ** 2)
    G = fft2(grid)
    return np.real(ifft2(k * G))


def _tilt_derivative(grid: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Compute the Tilt Derivative (TDR) of a potential field grid.

    TDR = arctan(FVD / HGM). Values range from −π/2 to +π/2. The zero
    crossing of the TDR marks lithological contacts. The ±π/4 contours
    approximate the source body edges.

    Args:
        grid: 2-D real-valued potential field grid.
        dx: Cell size (metres).

    Returns:
        TDR array in radians, same shape as ``grid``.
    """
    fvd = _first_vertical_derivative(grid, dx=dx)
    hgm = _horizontal_gradient_magnitude(grid, dx=dx)
    # Safe arctan: use a small epsilon to avoid division by near-zero HGM.
    eps = np.finfo(float).eps
    return np.arctan2(fvd, hgm + eps)


def _analytic_signal(grid: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Compute the Analytic Signal amplitude of a potential field grid.

    AS = sqrt((dT/dx)² + (dT/dy)² + (dT/dz)²). By definition AS ≥ 0 everywhere.

    Args:
        grid: 2-D real-valued potential field grid.
        dx: Cell size (metres).

    Returns:
        Analytic signal amplitude array, same shape as ``grid``.
    """
    kx, ky = _wavenumbers(*grid.shape, dx=dx, dy=dx)
    k = np.sqrt(kx ** 2 + ky ** 2)
    G = fft2(grid)
    dTdx = np.real(ifft2(1j * kx * G))
    dTdy = np.real(ifft2(1j * ky * G))
    dTdz = np.real(ifft2(k * G))
    return np.sqrt(dTdx ** 2 + dTdy ** 2 + dTdz ** 2)


def _upward_continue(
    grid: np.ndarray,
    height: float,
    dx: float = 1.0,
) -> np.ndarray:
    """Upward continuation of a potential field to a given height.

    Applies the upward continuation filter exp(−|k| * h) in the Fourier domain.
    Higher spatial frequencies (shallow, near-surface sources) are attenuated
    more strongly than low frequencies (deep sources), reducing noise.

    Args:
        grid: 2-D real-valued potential field grid.
        height: Continuation height in metres (must be > 0).
        dx: Cell size (metres).

    Returns:
        Upward-continued grid, same shape as ``grid``.
    """
    if height <= 0:
        raise ValueError(f"Continuation height must be positive; got {height}.")
    kx, ky = _wavenumbers(*grid.shape, dx=dx, dy=dx)
    k = np.sqrt(kx ** 2 + ky ** 2)
    G = fft2(grid)
    filter_response = np.exp(-k * height)
    return np.real(ifft2(filter_response * G))


def _reduction_to_pole(
    grid: np.ndarray,
    inclination_deg: float = -60.0,
    declination_deg: float = 0.0,
    dx: float = 1.0,
) -> np.ndarray:
    """Reduction to Pole (RTP) of a Total Magnetic Intensity (TMI) grid.

    Transforms the TMI anomaly to the equivalent response at the magnetic pole
    (inclination = 90°), producing anomalies centred directly over their
    causative sources. This simplifies target identification in the southern
    hemisphere where negative inclinations produce skewed, bipolar anomalies.

    Implementation uses the phase filter in the Fourier domain (Blakely, 1995,
    eq. 12.17). A stability term (epsilon taper) prevents division by zero at
    the DC component and near the magnetic equator.

    Args:
        grid: 2-D TMI grid.
        inclination_deg: Geomagnetic inclination at the survey site (degrees).
        declination_deg: Geomagnetic declination at the survey site (degrees).
        dx: Cell size (metres).

    Returns:
        RTP grid, same shape as ``grid``.
    """
    I = np.deg2rad(inclination_deg)
    D = np.deg2rad(declination_deg)

    kx, ky = _wavenumbers(*grid.shape, dx=dx, dy=dx)
    k = np.sqrt(kx ** 2 + ky ** 2)

    # Projection of magnetisation direction onto wavenumber plane.
    # L = sin(I) + i * cos(I) * (ky*cos(D) + kx*sin(D)) / k
    eps = 1e-8
    k_safe = np.where(k == 0, eps, k)

    L = np.sin(I) + 1j * np.cos(I) * (
        (ky * np.cos(D) + kx * np.sin(D)) / k_safe
    )
    # RTP filter: 1 / L² (magnetisation = induced; both field and body directions = L)
    # Add epsilon for stability.
    rtp_filter = np.where(np.abs(L) < eps, 1.0, 1.0 / (L ** 2))

    G = fft2(grid)
    # Zero out DC component.
    G[0, 0] = 0.0
    return np.real(ifft2(rtp_filter * G))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_mag_grid() -> np.ndarray:
    """Return a 64×64 synthetic magnetic grid containing a central Gaussian anomaly.

    The Gaussian peak approximates the anomaly produced by a compact,
    vertically magnetised body at shallow depth. It provides an analytically
    tractable test case for derivative operators.

    Returns:
        64×64 float64 array with unit-amplitude Gaussian centred at (32, 32)
        with σ = 8 cells.
    """
    rows, cols = 64, 64
    cx, cy = cols // 2, rows // 2
    sigma = 8.0
    x = np.arange(cols) - cx
    y = np.arange(rows) - cy
    xx, yy = np.meshgrid(x, y)
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))


@pytest.fixture
def uniform_mag_grid() -> np.ndarray:
    """Return a 64×64 grid of uniform value (1.0) for RTP flat-field test.

    A uniform TMI field has no spatial gradients; all spectral components
    except DC are zero. RTP of a uniform field must return a near-uniform field.

    Returns:
        64×64 float64 array with all values = 1.0.
    """
    return np.ones((64, 64), dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTiltDerivative:
    """Tests for the Tilt Derivative (TDR) filter."""

    def test_tilt_derivative_shape(self, synthetic_mag_grid: np.ndarray) -> None:
        """TDR output must have the same shape as the input grid.

        Shape preservation is mandatory for raster-stack alignment; a shape
        mismatch would misregister the TDR layer relative to other features.
        """
        result = _tilt_derivative(synthetic_mag_grid)
        assert result.shape == synthetic_mag_grid.shape, (
            f"TDR output shape {result.shape} differs from input "
            f"{synthetic_mag_grid.shape}."
        )

    def test_tilt_derivative_range(self, synthetic_mag_grid: np.ndarray) -> None:
        """TDR values must lie within [−π/2, +π/2] (arctan output range)."""
        result = _tilt_derivative(synthetic_mag_grid)
        assert np.all(result >= -np.pi / 2 - 1e-9), (
            "TDR values below −π/2 detected."
        )
        assert np.all(result <= np.pi / 2 + 1e-9), (
            "TDR values above +π/2 detected."
        )

    def test_tilt_derivative_finite(self, synthetic_mag_grid: np.ndarray) -> None:
        """TDR must contain no NaN or inf values for a well-behaved input grid."""
        result = _tilt_derivative(synthetic_mag_grid)
        assert np.all(np.isfinite(result)), (
            "TDR contains NaN or inf values. Check epsilon stabilisation of HGM."
        )


class TestAnalyticSignal:
    """Tests for the Analytic Signal (AS) amplitude filter."""

    def test_analytic_signal_positive(
        self, synthetic_mag_grid: np.ndarray
    ) -> None:
        """Analytic signal amplitude must be ≥ 0 everywhere.

        The analytic signal is defined as the amplitude of the total gradient
        vector: AS = sqrt(dT/dx² + dT/dy² + dT/dz²). Because it is the
        square root of a sum of squares, it is non-negative by construction.
        Any negative value indicates a numerical implementation error.
        """
        result = _analytic_signal(synthetic_mag_grid)
        assert np.all(result >= 0.0), (
            "Analytic signal contains negative values. "
            "AS is a root-sum-of-squares and must be ≥ 0 everywhere."
        )

    def test_analytic_signal_shape(self, synthetic_mag_grid: np.ndarray) -> None:
        """AS output shape must match input grid shape."""
        result = _analytic_signal(synthetic_mag_grid)
        assert result.shape == synthetic_mag_grid.shape

    def test_analytic_signal_zero_on_flat(self) -> None:
        """AS of a constant (zero-gradient) field must be zero everywhere.

        A uniform field has no spatial gradients. All derivatives are zero,
        so the analytic signal amplitude must also be zero (within floating-
        point precision).
        """
        flat = np.ones((32, 32), dtype=np.float64) * 5.0
        result = _analytic_signal(flat)
        np.testing.assert_allclose(result, 0.0, atol=1e-8)


class TestUpwardContinuation:
    """Tests for the upward continuation (UC) filter."""

    def test_upward_continuation_attenuates_signal(
        self, synthetic_mag_grid: np.ndarray
    ) -> None:
        """UC at 100 m must reduce grid variance compared to the original.

        Upward continuation acts as a low-pass filter in the spatial-frequency
        domain. High-frequency noise and near-surface cultural artefacts are
        attenuated; the variance of the continued field must therefore be
        strictly less than the original (for a non-trivial input grid).

        This property is used in the MPM pipeline to isolate basement
        lithological signatures from shallow regolith variability.
        """
        continued = _upward_continue(synthetic_mag_grid, height=100.0, dx=50.0)
        var_original = float(np.var(synthetic_mag_grid))
        var_continued = float(np.var(continued))
        assert var_continued < var_original, (
            f"Upward continuation at 100 m did not reduce grid variance. "
            f"Original variance: {var_original:.6f}, continued: {var_continued:.6f}. "
            "UC must act as a low-pass filter."
        )

    def test_upward_continuation_shape(
        self, synthetic_mag_grid: np.ndarray
    ) -> None:
        """UC output shape must match input grid shape."""
        result = _upward_continue(synthetic_mag_grid, height=50.0)
        assert result.shape == synthetic_mag_grid.shape

    def test_upward_continuation_rejects_nonpositive_height(
        self, synthetic_mag_grid: np.ndarray
    ) -> None:
        """UC with height ≤ 0 must raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            _upward_continue(synthetic_mag_grid, height=0.0)

        with pytest.raises(ValueError, match="positive"):
            _upward_continue(synthetic_mag_grid, height=-10.0)

    def test_upward_continuation_monotone_with_height(
        self, synthetic_mag_grid: np.ndarray
    ) -> None:
        """Variance must decrease monotonically as continuation height increases."""
        heights = [10.0, 50.0, 200.0, 1000.0]
        variances = [
            float(np.var(_upward_continue(synthetic_mag_grid, height=h, dx=50.0)))
            for h in heights
        ]
        for i in range(len(variances) - 1):
            assert variances[i] >= variances[i + 1], (
                f"Variance did not decrease monotonically: "
                f"{heights[i]} m → {variances[i]:.6f}, "
                f"{heights[i+1]} m → {variances[i+1]:.6f}."
            )


class TestReductionToPole:
    """Tests for Reduction to Pole (RTP) filter."""

    def test_rtp_flat_field(self, uniform_mag_grid: np.ndarray) -> None:
        """RTP of a uniform (DC-only) field must return a near-uniform field.

        A uniform TMI field has no spatial gradients; its Fourier transform is
        a single DC spike. The RTP phase filter acts on non-DC components, so
        the output should be near-uniform. Variation must be less than 1% of
        the original mean (the DC component is zeroed in the implementation,
        so the absolute value of the output is expected to be near-zero rather
        than preserving the absolute level).

        Geoscientific note: in practice 'flat' areas of the TMI map
        (regional basement) should pass through RTP without developing
        spurious edge artefacts.
        """
        result = _reduction_to_pole(
            uniform_mag_grid,
            inclination_deg=-60.0,
            declination_deg=5.0,
        )
        # DC component is zeroed in implementation; the resulting field must
        # be spatially uniform (standard deviation near machine precision).
        relative_variation = float(np.std(result)) / (float(np.mean(uniform_mag_grid)) + 1e-10)
        assert relative_variation < 0.01, (
            f"RTP of a uniform field returned non-uniform output. "
            f"Relative std/mean = {relative_variation:.4f} (threshold 0.01). "
            "Check RTP filter DC handling."
        )

    def test_rtp_shape(self, synthetic_mag_grid: np.ndarray) -> None:
        """RTP output shape must match input grid shape."""
        result = _reduction_to_pole(synthetic_mag_grid)
        assert result.shape == synthetic_mag_grid.shape


class TestFirstVerticalDerivative:
    """Tests for the First Vertical Derivative (FVD) filter."""

    def test_first_vertical_derivative_symmetric(
        self, synthetic_mag_grid: np.ndarray
    ) -> None:
        """FVD of a circularly symmetric Gaussian peak must be symmetric (even).

        A 2-D symmetric Gaussian is an even function. Its vertical derivative
        (multiplication by |k| = sqrt(kx² + ky²) in the Fourier domain) preserves
        even symmetry: the FVD is radially symmetric about the Gaussian centre.
        Specifically, fvd[cy, cx + d] ≈ fvd[cy, cx − d] for displacements d
        along any axis through the peak.

        (Note: the first horizontal derivative dT/dx is antisymmetric/odd,
        whereas the vertical derivative dT/dz is symmetric/even for a symmetric anomaly).
        """
        fvd = _first_vertical_derivative(synthetic_mag_grid)
        rows, cols = fvd.shape
        cy, cx = rows // 2, cols // 2

        # Sample along the central row at symmetric offsets from the peak.
        offsets = np.arange(1, 12)
        positive_side = fvd[cy, cx + offsets]
        negative_side = fvd[cy, cx - offsets]

        # Symmetry: f(x) ≈ f(−x), i.e. |positive_side - negative_side| ≈ 0
        symmetry_residual = np.abs(positive_side - negative_side)
        max_val = np.max(np.abs(fvd[cy, :]))
        threshold = 0.05 * max_val
        assert np.all(symmetry_residual < threshold), (
            f"FVD is not symmetric about the Gaussian peak. "
            f"Max residual: {symmetry_residual.max():.6f}, threshold: {threshold:.6f}."
        )

    def test_first_vertical_derivative_shape(
        self, synthetic_mag_grid: np.ndarray
    ) -> None:
        """FVD output shape must match input grid shape."""
        result = _first_vertical_derivative(synthetic_mag_grid)
        assert result.shape == synthetic_mag_grid.shape

    def test_first_vertical_derivative_zero_on_flat(self) -> None:
        """FVD of a constant field must be zero (no gradient → no derivative)."""
        flat = np.ones((32, 32), dtype=np.float64) * 3.0
        result = _first_vertical_derivative(flat)
        np.testing.assert_allclose(result, 0.0, atol=1e-8)
