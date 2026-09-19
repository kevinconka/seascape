"""Band-integrated LWIR radiometry for a water surface.

Blender is an RGB renderer with no concept of an 8-14 um band, and its Fresnel node
takes a scalar IOR where seawater needs a complex one (n + i*k). So the angular
emissivity curve is evaluated here, in numpy, and the shader consumes it as a 1D
lookup. Band-integrating costs nothing, because the sensor does it anyway.

Angles are radians throughout; this module sits well inside the degrees boundary.

Where the numbers come from
---------------------------
Two kinds of number live here and they are not interchangeable. Measured values carry a
source; the rest are chosen to look right and are marked as such below and at their
definition. Only the first kind can back a claim.

*Measured.* The optical constants are Downing & Williams, "Optical Constants of Water in
the Infrared", J. Geophys. Res. 80(12), 1975, Table 1 -- water at 27 C, 1250-710 cm^-1.
Pure water, not seawater: salinity shifts k slightly in this band, and Friedman (1969)
is the seawater source if that correction is ever needed. The physical constants are the
SI defining constants, exact by definition since the 2019 redefinition.

*Derived.* Planck's law and Fresnel for an absorbing medium are textbook. Two
assumptions ride on them and are easier to forget than to spot:

- Kirchhoff's law, eps = 1 - R, which holds because water is opaque across this band
  well inside any depth the sensor resolves. No transmitted term.

- Band emissivity is the Planck-weighted mean of the spectral emissivity. This is exact
  only for a sensor whose spectral response is flat across 8-14 um. A real
  microbolometer is not, so a specific sensor wants its own response curve here.

*Chosen.* The sky model: SKY_EPS_ZENITH and the 1/sin(elevation) airmass, which further
assumes a plane-parallel atmosphere and so overstates the path near the horizon. It
reproduces the shape of a thermal horizon rather than its radiometry, and a defensible
figure needs a real model (lowtran, HITRAN) in its place.

Path extinction is deliberately absent. Beer-Lambert over a homogeneous medium is what
Blender's Volume Absorption and Volume Emission nodes already do, per ray and with the
real geometry, so computing it here would only work for a flat sea and would be thrown
away the moment there are waves and targets at different ranges.

The band itself is the atmospheric window an uncooled microbolometer sees.
"""

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]

# Wavenumber (cm^-1), n, k -- Downing & Williams 1975 Table 1, 1250..710 cm^-1
# (8.00-14.08 um). Three OCR artifacts in the source PDF were corrected against
# the smoothness of their neighbours: k(1010) 0.515->0.0515, k(930) 0.O828->
# 0.0828, k(900) _0.107->0.107.
WATER_NK = (
    (1250, 1.291, 0.0351), (1240, 1.288, 0.0352), (1230, 1.286, 0.0356),
    (1220, 1.285, 0.0359), (1210, 1.283, 0.0361), (1200, 1.281, 0.0362),
    (1190, 1.279, 0.0366), (1180, 1.276, 0.0370), (1170, 1.274, 0.0374),
    (1160, 1.271, 0.0378), (1150, 1.269, 0.0383), (1140, 1.267, 0.0387),
    (1130, 1.264, 0.0392), (1120, 1.261, 0.0398), (1110, 1.259, 0.0405),
    (1100, 1.256, 0.0411), (1090, 1.253, 0.0417), (1080, 1.249, 0.0424),
    (1070, 1.246, 0.0434), (1060, 1.242, 0.0443), (1050, 1.238, 0.0453),
    (1040, 1.234, 0.0467), (1030, 1.230, 0.0481), (1020, 1.224, 0.0497),
    (1010, 1.220, 0.0515), (1000, 1.214, 0.0534), (990, 1.208, 0.0557),
    (980, 1.202, 0.0589), (970, 1.194, 0.0622), (960, 1.189, 0.0661),
    (950, 1.181, 0.0707), (940, 1.174, 0.0764), (930, 1.168, 0.0828),
    (920, 1.162, 0.0898), (910, 1.156, 0.0973), (900, 1.149, 0.1070),
    (890, 1.143, 0.1180), (880, 1.139, 0.1300), (870, 1.135, 0.1440),
    (860, 1.132, 0.1590), (850, 1.132, 0.1760), (840, 1.131, 0.1920),
    (830, 1.132, 0.2080), (820, 1.130, 0.2260), (810, 1.130, 0.2430),
    (800, 1.134, 0.2600), (790, 1.138, 0.2770), (780, 1.142, 0.2920),
    (770, 1.157, 0.3050), (760, 1.171, 0.3170), (750, 1.182, 0.3280),
    (740, 1.189, 0.3380), (730, 1.201, 0.3470), (720, 1.213, 0.3560),
    (710, 1.223, 0.3650),
)  # fmt: skip

BAND_M = (8.0e-6, 14.0e-6)

# SI defining constants, exact.
PLANCK_H = 6.62607015e-34  # J s
LIGHT_C = 2.99792458e8  # m s^-1
BOLTZMANN_K = 1.380649e-23  # J K^-1

T_SEA_K = 288.0

# Chosen, not measured. See "Where the numbers come from" above before quoting anything
# downstream of these three.
T_AIR_K = 288.0
SKY_EPS_ZENITH = 0.30  # in-band zenith emissivity, clear dry sky


def optical_constants() -> tuple[FloatArray, FloatArray, FloatArray]:
    """Wavelength (m), n, k across the band, ascending in wavelength."""
    wn, n, k = np.array(WATER_NK, dtype=np.float64).T
    lam = 1e-2 / wn  # cm^-1 -> m
    order = np.argsort(lam)
    lam, n, k = lam[order], n[order], k[order]
    inside = (lam >= BAND_M[0]) & (lam <= BAND_M[1])
    return lam[inside], n[inside], k[inside]


def planck(lam_m: npt.ArrayLike, t_k: float) -> FloatArray:
    """Spectral radiance of a blackbody, W m^-2 sr^-1 m^-1."""
    lam = np.asarray(lam_m, dtype=np.float64)
    numerator = 2 * PLANCK_H * LIGHT_C**2 / lam**5
    return numerator / (np.exp(PLANCK_H * LIGHT_C / (lam * BOLTZMANN_K * t_k)) - 1.0)


def fresnel_emissivity(
    theta_rad: npt.ArrayLike, n: npt.ArrayLike, k: npt.ArrayLike
) -> FloatArray:
    """Unpolarised emissivity 1-R at incidence `theta_rad` from vacuum.

    Textbook Fresnel for an absorbing medium; numpy's complex sqrt takes the
    correct branch, so no special-casing of grazing angles is needed. Inputs
    broadcast, so angle and wavelength can be evaluated as a grid.
    """
    n_c = np.asarray(n, dtype=np.float64) + 1j * np.asarray(k, dtype=np.float64)
    theta = np.asarray(theta_rad, dtype=np.float64)
    cos_i = np.cos(theta)
    q = np.sqrt(n_c**2 - np.sin(theta) ** 2)
    r_s = (cos_i - q) / (cos_i + q)
    r_p = (n_c**2 * cos_i - q) / (n_c**2 * cos_i + q)
    return 1.0 - 0.5 * (np.abs(r_s) ** 2 + np.abs(r_p) ** 2)


def emissivity_curve(
    *, t_sea_k: float = T_SEA_K, n_angles: int = 91
) -> tuple[FloatArray, FloatArray]:
    """Planck-weighted, band-integrated emissivity against incidence angle (rad)."""
    lam, n, k = optical_constants()
    weight = planck(lam, t_sea_k)
    theta = np.linspace(0.0, np.pi / 2, n_angles)
    eps = fresnel_emissivity(theta[:, None], n, k)
    return theta, np.trapezoid(eps * weight, lam, axis=-1) / np.trapezoid(weight, lam)


def band_radiance(t_k: float) -> float:
    """Blackbody radiance integrated over the band, W m^-2 sr^-1."""
    lam, _, _ = optical_constants()
    return float(np.trapezoid(planck(lam, t_k), lam))


def sky_radiance(
    elevation_rad: npt.ArrayLike,
    *,
    t_air_k: float = T_AIR_K,
    eps_zenith: float = SKY_EPS_ZENITH,
) -> FloatArray:
    """Downwelling in-band sky radiance at an elevation above the horizon.

    Sky emissivity grows toward the horizon as the slant path lengthens, so the sky
    warms from a cold zenith to ambient at the horizon. That convergence is what
    makes a thermal horizon read correctly.

    Shape, not radiometry — the zenith emissivity and the 1/sin airmass are chosen
    values. See "Where the numbers come from".
    """
    s = np.clip(np.sin(np.asarray(elevation_rad, dtype=np.float64)), 1e-3, 1.0)
    return (1.0 - (1.0 - eps_zenith) ** (1.0 / s)) * band_radiance(t_air_k)
