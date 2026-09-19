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

# Downing & Williams 1975, Table 1: water at 27 C on a regular 10 cm^-1 grid, so the
# grid is generated rather than typed out a third time.
WAVENUMBERS_CM1 = np.arange(1250, 705, -10)

# n and k are laboratory measurements. There is no formula for them: a Lorentz
# oscillator fit would trade 110 measured numbers for ~16 fitted ones plus fit
# error, and would still need these to validate against.
WATER_N = (
    1.291, 1.288, 1.286, 1.285, 1.283, 1.281, 1.279, 1.276, 1.274, 1.271, 1.269,
    1.267, 1.264, 1.261, 1.259, 1.256, 1.253, 1.249, 1.246, 1.242, 1.238, 1.234,
    1.230, 1.224, 1.220, 1.214, 1.208, 1.202, 1.194, 1.189, 1.181, 1.174, 1.168,
    1.162, 1.156, 1.149, 1.143, 1.139, 1.135, 1.132, 1.132, 1.131, 1.132, 1.130,
    1.130, 1.134, 1.138, 1.142, 1.157, 1.171, 1.182, 1.189, 1.201, 1.213, 1.223,
)  # fmt: skip

# Three OCR artifacts in the source PDF were corrected against neighbouring rows:
# k(1010) 0.515->0.0515, k(930) 0.O828->0.0828, k(900) _0.107->0.107.
WATER_K = (
    0.0351, 0.0352, 0.0356, 0.0359, 0.0361, 0.0362, 0.0366, 0.0370, 0.0374, 0.0378,
    0.0383, 0.0387, 0.0392, 0.0398, 0.0405, 0.0411, 0.0417, 0.0424, 0.0434, 0.0443,
    0.0453, 0.0467, 0.0481, 0.0497, 0.0515, 0.0534, 0.0557, 0.0589, 0.0622, 0.0661,
    0.0707, 0.0764, 0.0828, 0.0898, 0.0973, 0.1070, 0.1180, 0.1300, 0.1440, 0.1590,
    0.1760, 0.1920, 0.2080, 0.2260, 0.2430, 0.2600, 0.2770, 0.2920, 0.3050, 0.3170,
    0.3280, 0.3380, 0.3470, 0.3560, 0.3650,
)  # fmt: skip

BAND_M = (8.0e-6, 14.0e-6)

# SI defining constants, exact.
PLANCK_H = 6.62607015e-34  # J s
LIGHT_C = 2.99792458e8  # m s^-1
BOLTZMANN_K = 1.380649e-23  # J K^-1

T_SEA_K = 288.0


def optical_constants() -> tuple[FloatArray, FloatArray, FloatArray]:
    """Wavelength (m), n, k across the band, ascending in wavelength."""
    n = np.array(WATER_N, dtype=np.float64)
    k = np.array(WATER_K, dtype=np.float64)
    lam = 1e-2 / WAVENUMBERS_CM1  # cm^-1 -> m
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
