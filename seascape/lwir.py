"""Band-integrated LWIR emissivity of a water surface.

Blender is an RGB renderer with no concept of the 8-14 um band, and its Fresnel node
takes a scalar IOR where water needs a complex one (n + i*k). So the emissivity curve is
evaluated here and the shader consumes it as a 1D lookup. Nothing else belongs in this
module: path extinction is Blender's volume nodes, and the sky term stays a guess until
someone brings a real model. Angles are radians.

Sources
-------
Optical constants: Downing & Williams, "Optical Constants of Water in the Infrared",
J. Geophys. Res. 80(12), 1975, Table 1 -- water at 27 C, 1250-710 cm^-1. Pure water, not
seawater; Friedman (1969) is the seawater source if that correction is ever needed.
h, c and k_B are the SI defining constants, exact since the 2019 redefinition.

Planck's law and Fresnel for an absorbing medium are textbook, but carry two assumptions
that fail silently:

- Kirchhoff's law, eps = 1 - R. Holds because water is opaque across this band well
  inside any depth the sensor resolves, so there is no transmitted term.
- Band emissivity is the Planck-weighted mean of the spectral emissivity. Exact only for
  a flat sensor response; a specific microbolometer wants its own curve here.
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
