"""Physics assertions for the LWIR band.

No GPU, no Blender. These are the check that the radiometry is right; a render only
shows that it is plausible.

Each assertion names what it is measured against. Every function in the module is
exercised by one, planck and fresnel_emissivity through emissivity_curve.
See "Where the numbers come from" in seascape/lwir.py.
"""

import numpy as np
import pytest

from seascape import lwir

# Derived from the three SI defining constants rather than hardcoded: sigma is
# 2 pi^5 k^4 / (15 h^3 c^2) exactly, so this also checks they are self-consistent.
STEFAN_BOLTZMANN = (
    2 * np.pi**5 * lwir.BOLTZMANN_K**4 / (15 * lwir.PLANCK_H**3 * lwir.LIGHT_C**2)
)


@pytest.fixture(scope="module")
def curve() -> tuple[np.ndarray, np.ndarray]:
    return lwir.emissivity_curve()


def eps_at(curve: tuple[np.ndarray, np.ndarray], deg: float) -> float:
    angles, eps = curve
    return float(np.interp(np.radians(deg), angles, eps))


def test_normal_incidence_emissivity_is_about_0_99(curve) -> None:
    """Water looks almost black in the LWIR when viewed straight down.

    0.98-0.99 is the standard handbook emissivity for water in this band, and the one
    every IR thermometer ships as its water preset.
    """
    assert 0.98 <= eps_at(curve, 0.0) <= 0.995


def test_emissivity_falls_with_angle(curve) -> None:
    assert eps_at(curve, 60.0) > eps_at(curve, 80.0) > eps_at(curve, 89.0)


def test_grazing_emissivity_collapses(curve) -> None:
    """Near the horizon the sea turns into a mirror for the sky."""
    assert eps_at(curve, 89.0) < 0.6


def test_optical_constants_match_downing_williams_at_10um() -> None:
    """Spot-check the transcribed table against Downing & Williams 1975, Table 1.

    The table was transcribed from a PDF with three OCR artifacts corrected by hand, so
    this guards the transcription, not the physics.
    """
    lam, n, k = lwir.optical_constants()
    j = int(np.argmin(np.abs(lam - 10e-6)))
    assert n[j] == pytest.approx(1.214, abs=1e-3)
    assert k[j] == pytest.approx(0.0534, abs=1e-4)


def test_stefan_boltzmann_matches_its_published_value() -> None:
    """Guards the transcription of h, c and k_B against CODATA's rounded sigma."""
    sigma = (
        STEFAN_BOLTZMANN  # local: ruff reads a bare constant here as a Yoda condition
    )
    assert sigma == pytest.approx(5.670374419e-8, rel=1e-9)


def test_the_wavenumber_grid_is_unbroken() -> None:
    """The wavenumber column is a per-row checksum, so check it.

    Downing & Williams sampled every 10 cm^-1 from 1250 to 710 with no gaps. Asserting
    that catches a mistyped wavenumber, a dropped row, a duplicate and a transposition
    -- none of which a spot-check or a length count would notice.
    """
    wavenumbers = np.array([row[0] for row in lwir.WATER_NK])
    assert np.array_equal(wavenumbers, np.arange(1250, 705, -10))


def test_optical_constants_vary_smoothly() -> None:
    """Catch a mistyped digit anywhere in the table, not just at the one spot-check.

    n and k change gradually with wavelength, so every value should sit near the
    midpoint of its two neighbours. The k column around 1010 cm^-1 reads 0.0497,
    0.0515, 0.0534: the middle one is 0.1% off that midpoint, and the worst row in
    the table is 1.1% off. The table was transcribed from a PDF with three OCR
    artifacts fixed by hand; restoring the worst, k(1010) as 0.515, puts that row
    90% off. Hence the 10% bar.
    """
    _, n, k = lwir.optical_constants()
    for values in (n, k):
        midpoint_of_neighbours = (values[:-2] + values[2:]) / 2
        off_by = np.abs(values[1:-1] - midpoint_of_neighbours) / values[1:-1]
        assert off_by.max() < 0.10


def test_band_holds_a_plausible_share_of_total_emission() -> None:
    """The band is a fraction of a 288 K body's total emission.

    Stefan-Boltzmann gives the total exactly; the 35-50% window is a chosen tolerance
    around the ~36% this integration produces, wide enough to survive a change of
    integration scheme and narrow enough to catch a unit error in Planck.
    """
    total = STEFAN_BOLTZMANN * 288.0**4 / np.pi
    assert 0.35 <= lwir.band_radiance(288.0) / total <= 0.50
