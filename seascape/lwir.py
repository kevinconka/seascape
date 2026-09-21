"""Band-integrated LWIR emissivity of a water surface, and the sky it reflects.

Blender is an RGB renderer with no concept of the 8-14 um band, and its Fresnel node
takes a scalar IOR where water needs a complex one (n + i*k). So the emissivity curve is
evaluated here and the shader consumes it as a 1D lookup. The sky curve is here for the
same reason: the Sky Texture is a visible-band scattering model with nothing to say
about 8-14 um. Path extinction is not here, because that is Blender's volume nodes.
Angles are radians.

A sea surface emits and reflects, and the two are complements: `1 - eps` of what it does
not emit comes back as reflected sky. Leave the reflection out and the sea goes black at
grazing incidence, which is most of a maritime image.

Sources
-------
Optical constants: Nalli et al., "Temperature-dependent optical constants of water in
the thermal infrared derived from data archaeology", Optics Continuum 1(4) 738, 2022
(doi:10.1364/OPTCON.450833); data doi:10.6084/m9.figshare.19341533, CC BY 4.0. That is
Downing & Williams 1975 extended across 271-311 K using Pinkley et al. 1977. The table
ships as data/water_nk.csv, which carries the same citation.

Sky emissivity: one LOWTRAN7 run, midlatitude summer profile with the navy maritime
aerosol, observer at 12 m, integrated over the band. LOWTRAN7 is public-domain
(AFGL-TR-88-0177); the run is reproducible with `lowtran` on PyPI, which needs gfortran.
It is a band model, not line-by-line, and the profile is fixed: good to a few percent,
not a radiometric reference.

h, c and k_B are the SI defining constants, exact since the 2019 redefinition.

Planck's law and Fresnel for an absorbing medium are textbook, but carry two assumptions
that fail silently:

- Kirchhoff's law, eps = 1 - R. Holds because water is opaque across this band well
  inside any depth the sensor resolves, so there is no transmitted term.
- Band emissivity is the Planck-weighted mean of the spectral emissivity. Exact only for
  a flat sensor response; a specific microbolometer wants its own curve here.
"""

import functools
from pathlib import Path

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]

_TABLE_CSV = Path(__file__).parent / "data" / "water_nk.csv"

BAND_M = (8.0e-6, 14.0e-6)

# SI defining constants, exact.
PLANCK_H = 6.62607015e-34  # J s
LIGHT_C = 2.99792458e8  # m s^-1
BOLTZMANN_K = 1.380649e-23  # J K^-1

T_SEA_K = 288.0
T_AIR_K = 288.0

# Downwelling sky emissivity against elevation above the horizon, in degrees for
# legibility and converted once below. Normalised by the horizon value, which is ambient
# by construction: a horizontal path is optically thick, so the sky at the horizon is a
# blackbody at air temperature. That normalisation is what lets one curve serve any air
# temperature -- the shape belongs to the atmosphere, the scale to Planck.
_SKY_EPS = (
    (0.0, 1.0000),
    (1.0, 0.9898),
    (2.0, 0.9863),
    (3.0, 0.9789),
    (5.0, 0.9507),
    (7.0, 0.9129),
    (10.0, 0.8535),
    (15.0, 0.7663),
    (20.0, 0.6973),
    (30.0, 0.6001),
    (45.0, 0.5143),
    (60.0, 0.4671),
    (90.0, 0.4352),
)


def _checked_kelvin(t_k: float) -> float:
    """Reject anything that is not a temperature.

    Written as `not > 0` so nan raises as well as zero and negatives; nan would
    otherwise survive np.clip and turn a whole lookup into silent NaN.
    """
    if not t_k > 0.0:
        raise ValueError(
            f"temperature must be a positive number of kelvin, got {t_k!r}"
        )
    return float(t_k)


@functools.lru_cache(maxsize=1)
def _table() -> tuple[FloatArray, FloatArray, FloatArray]:
    """The shipped table as (wavenumber, temperature, n and k on that grid).

    n and k come back shaped (temperature, wavenumber). Rows are placed by index
    rather than reshaped, so the file's row order does not matter. Arrays are frozen
    because the cache hands the same objects to every caller.
    """
    raw = np.loadtxt(_TABLE_CSV, delimiter=",", comments="#")
    grid, column = np.unique(raw[:, 0], return_inverse=True)
    temperatures, row = np.unique(raw[:, 1], return_inverse=True)

    nk = np.full((2, len(temperatures), len(grid)), np.nan)
    nk[0, row, column] = raw[:, 2]
    nk[1, row, column] = raw[:, 3]
    if np.isnan(nk).any():
        raise ValueError(f"{_TABLE_CSV.name} is missing rows: the grid has holes")

    for a in (grid, temperatures, nk):
        a.setflags(write=False)
    return grid, temperatures, nk


def optical_constants(
    t_k: float = T_SEA_K,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Wavelength (m), n, k across the band at `t_k`, ascending in wavelength.

    Linearly interpolated between the table's 4 K steps and clamped outside 271-311 K,
    which already spans any sea surface. Water's optical constants move little across
    that range: emissivity shifts by 0.016 at most, peaking at 80 degrees where the
    curve is steepest, and by 0.003 looking straight down.
    """
    grid, temperatures, nk = _table()
    t = float(np.clip(_checked_kelvin(t_k), temperatures[0], temperatures[-1]))
    n, k = (
        np.array([np.interp(t, temperatures, col) for col in plane.T]) for plane in nk
    )

    lam = 1e-2 / grid  # cm^-1 -> m
    order = np.argsort(lam)
    lam, n, k = lam[order], n[order], k[order]
    inside = (lam >= BAND_M[0]) & (lam <= BAND_M[1])
    return lam[inside], n[inside], k[inside]


def planck(lam_m: npt.ArrayLike, t_k: float) -> FloatArray:
    """Spectral radiance of a blackbody, W m^-2 sr^-1 m^-1.

    A negative temperature otherwise returns a negative radiance rather than failing.
    """
    _checked_kelvin(t_k)
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
    lam, n, k = optical_constants(t_sea_k)
    weight = planck(lam, t_sea_k)
    theta = np.linspace(0.0, np.pi / 2, n_angles)
    eps = fresnel_emissivity(theta[:, None], n, k)
    return theta, np.trapezoid(eps * weight, lam, axis=-1) / np.trapezoid(weight, lam)


def band_radiance(t_k: float) -> float:
    """Blackbody radiance integrated over the band, W m^-2 sr^-1."""
    lam, _, _ = optical_constants(t_k)
    return float(np.trapezoid(planck(lam, t_k), lam))


def sky_radiance(elev_rad: npt.ArrayLike, t_air_k: float = T_AIR_K) -> FloatArray:
    """Downwelling in-band sky radiance at an elevation above the horizon.

    The sky cools from ambient at the horizon, where the slant path is optically thick,
    to roughly 0.44 of it at the zenith. That convergence is what makes a thermal
    horizon read correctly: sea and sky meet at the same radiance, so contrast collapses
    exactly where a target is hardest to see.

    Below the horizon the curve holds at ambient, which is what a ray that misses the
    sea should see.
    """
    elev, eps = np.array(_SKY_EPS, dtype=np.float64).T
    fraction = np.interp(np.asarray(elev_rad, dtype=np.float64), np.radians(elev), eps)
    return fraction * band_radiance(t_air_k)
