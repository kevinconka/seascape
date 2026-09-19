"""Physics assertions for the LWIR band.

Published values, no GPU, no Blender. These are the check that the radiometry is
right; a render only shows that it is plausible.
"""

import numpy as np
import pytest

from seascape import lwir

STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4


@pytest.fixture(scope="module")
def curve() -> tuple[np.ndarray, np.ndarray]:
    return lwir.emissivity_curve()


def eps_at(curve: tuple[np.ndarray, np.ndarray], deg: float) -> float:
    angles, eps = curve
    return float(np.interp(np.radians(deg), angles, eps))


def test_normal_incidence_emissivity_is_about_0_99(curve) -> None:
    """Water looks almost black in the LWIR when viewed straight down."""
    assert 0.98 <= eps_at(curve, 0.0) <= 0.995


def test_emissivity_falls_with_angle(curve) -> None:
    assert eps_at(curve, 60.0) > eps_at(curve, 80.0) > eps_at(curve, 89.0)


def test_grazing_emissivity_collapses(curve) -> None:
    """Near the horizon the sea turns into a mirror for the sky."""
    assert eps_at(curve, 89.0) < 0.6


def test_optical_constants_match_downing_williams_at_10um() -> None:
    """Spot-check the transcribed table against its source at 1000 cm^-1."""
    lam, n, k = lwir.optical_constants()
    j = int(np.argmin(np.abs(lam - 10e-6)))
    assert n[j] == pytest.approx(1.214, abs=1e-3)
    assert k[j] == pytest.approx(0.0534, abs=1e-4)


def test_band_holds_a_plausible_share_of_total_emission() -> None:
    """Stefan-Boltzmann sanity: 8-14 um is a fraction of a 288 K body's output."""
    total = STEFAN_BOLTZMANN * 288.0**4 / np.pi
    assert 0.35 <= lwir.band_radiance(288.0) / total <= 0.50


def test_sea_radiance_is_finite_over_the_full_angular_sweep() -> None:
    """Guards the clip at grazing, where tan() would otherwise diverge."""
    radiance = lwir.sea_radiance(np.linspace(0.0, np.pi / 2, 91))
    assert radiance.shape == (91,)
    assert np.all(np.isfinite(radiance)) and np.all(radiance > 0.0)
