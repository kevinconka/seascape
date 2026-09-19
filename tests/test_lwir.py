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
    """The shape between the endpoints, which checking 0° and 89° alone would miss.

    Fresnel reflectance rises monotonically with incidence for an absorbing medium, so
    emissivity has to fall monotonically. A wobble means the complex sqrt took a wrong
    branch or the band integration is picking up sign errors.
    """
    assert eps_at(curve, 60.0) > eps_at(curve, 80.0) > eps_at(curve, 89.0)


def test_grazing_emissivity_collapses(curve) -> None:
    """Near the horizon the sea turns into a mirror for the sky."""
    assert eps_at(curve, 89.0) < 0.6


def test_table_reproduces_downing_williams_at_their_own_temperature() -> None:
    """End-to-end check of the shipped table against a value published in 1975.

    The table is Downing & Williams extended across temperature by Nalli et al. Asked
    for 300 K -- the 27 C those original measurements were made at -- it has to give
    back what they printed: n=1.214, k=0.0534 at 10 um. That exercises the file, the
    reader and the interpolation in one assertion, against a number none of them
    produced.
    """
    lam, n, k = lwir.optical_constants(300.0)
    j = int(np.argmin(np.abs(lam - 10e-6)))
    assert n[j] == pytest.approx(1.214, abs=1e-3)
    assert k[j] == pytest.approx(0.0534, abs=5e-4)


def test_stefan_boltzmann_matches_its_published_value() -> None:
    """Guards the transcription of h, c and k_B against CODATA's rounded sigma."""
    sigma = (
        STEFAN_BOLTZMANN  # local: ruff reads a bare constant here as a Yoda condition
    )
    assert sigma == pytest.approx(5.670374419e-8, rel=1e-9)


def test_the_shipped_grid_is_unbroken() -> None:
    """A hole in either axis would silently pair values with the wrong coordinate.

    `_table` already raises on a missing cell; this pins the grid it expects, so a
    regenerated file with a different span or step fails loudly rather than quietly
    changing the physics.
    """
    wavenumbers, temperatures, _ = lwir._table()
    assert np.array_equal(wavenumbers, np.arange(700.0, 1261.0, 20.0))
    assert np.array_equal(temperatures, np.arange(271.0, 312.0, 4.0))


def test_optical_constants_vary_smoothly() -> None:
    """Catch a corrupted value anywhere in the band, not just at the one spot-check.

    n and k change gradually with wavelength, so every value should sit near the
    midpoint of its two neighbours. A slipped decimal in the shipped file, or a
    transposed column in the reader, breaks that immediately: the real data stays
    within a few percent, so the 10% bar has a wide margin either side.
    """
    _, n, k = lwir.optical_constants()
    for values in (n, k):
        midpoint_of_neighbours = (values[:-2] + values[2:]) / 2
        off_by = np.abs(values[1:-1] - midpoint_of_neighbours) / values[1:-1]
        assert off_by.max() < 0.10


def test_emissivity_rises_with_sea_temperature_but_barely() -> None:
    """Both halves matter: the trend is real, and it is small.

    Warmer water is slightly less reflective in this band, so emissivity climbs with
    temperature. Across 271-311 K -- colder and warmer than any sea -- it moves 0.016
    at most: 0.003 looking straight down, peaking at 80 degrees where the curve is
    steepest. Small, but it peaks exactly where the horizon and the distant targets
    are, which is why it is worth carrying rather than freezing at one temperature.

    The epsilon absorbs 1e-17 wobble at 90 degrees, where emissivity is exactly zero.
    """
    cold = lwir.emissivity_curve(t_sea_k=271.0)[1]
    warm = lwir.emissivity_curve(t_sea_k=311.0)[1]
    assert np.all(warm >= cold - 1e-12)
    assert np.abs(warm - cold).max() < 0.02


def test_temperature_is_clamped_to_the_measured_range() -> None:
    """Outside 271-311 K the table has nothing, so hold the endpoint.

    Extrapolating optical constants past the measurements would invent data. Only the
    lookup clamps: Planck still uses the temperature it was given, which is why the
    curves are close rather than identical.
    """
    below = lwir.optical_constants(200.0)
    at_edge = lwir.optical_constants(271.0)
    assert all(np.array_equal(a, b) for a, b in zip(below, at_edge, strict=True))


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan")])
def test_planck_rejects_impossible_temperatures(bad: float) -> None:
    """Every public entry point taking a temperature has to reject a non-temperature.

    A negative kelvin returned a negative radiance and propagated in silence; nan
    survived the clamp in optical_constants and turned the whole lookup into NaN.
    """
    with pytest.raises(ValueError, match="positive"):
        lwir.planck(1e-5, bad)
    with pytest.raises(ValueError, match="positive"):
        lwir.band_radiance(bad)
    with pytest.raises(ValueError, match="positive"):
        lwir.optical_constants(bad)
    with pytest.raises(ValueError, match="positive"):
        lwir.emissivity_curve(t_sea_k=bad)


def test_band_holds_a_plausible_share_of_total_emission() -> None:
    """The band is a fraction of a 288 K body's total emission.

    Stefan-Boltzmann gives the total exactly; the 35-50% window is a chosen tolerance
    around the ~36% this integration produces, wide enough to survive a change of
    integration scheme and narrow enough to catch a unit error in Planck.
    """
    total = STEFAN_BOLTZMANN * 288.0**4 / np.pi
    assert 0.35 <= lwir.band_radiance(288.0) / total <= 0.50
