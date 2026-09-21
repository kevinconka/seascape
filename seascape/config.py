"""Scenario config: pydantic models, and a TOML loader that never guesses.

A scenario is a TOML file. Two forms of reuse, both resolved here:

- `extends = "baseline.toml"` at the top level, for a variant that is a diff.
- `preset = "twin_pod"` inside a block, next to the overrides it applies to.

Rules:

1. `preset` is a key inside its block, so there are no precedence rules between
   parents.
2. Syntax decides what a name is. A bare name is a preset shipped under `cfg/`,
   anything with `/` or ending in `.toml` is a path relative to the including file.
   Never try one form and fall back to the other.
3. Tables merge, everything else replaces. A list is replaced whole.
"""

import tomllib
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from seascape import lwir

CFG_DIR = Path(__file__).parent / "cfg"

# A camera's kind is the band it sees in, and a scene is built for one band at a time.
type Band = Literal["eo", "ir"]

type ImageFormat = Literal["exr", "png"]


class Model(BaseModel):
    """Strictness shared by everything this package parses from TOML."""

    # extra: a typo in a scenario is a silent wrong render otherwise. `preset` is
    # consumed by the loader before validation, so this also catches it leaking.
    # inf_nan: tomllib parses `nan` and `inf`, and pydantic accepts both by default.
    # A nan bearing renders a camera pointing nowhere and reports no error.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Camera(Model):
    kind: Band
    fan_deg: float = 0.0  # relative to the pod axis, positive to starboard
    hfov_deg: float = Field(gt=0.0, lt=180.0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


class Pod(Model):
    """One enclosure: several cameras behind one yaw and one mount point.

    Offsets matter: pods a beam apart overlap in angle before they overlap in space,
    which is the blind wedge over the bow. Coincident cameras would hide it.
    """

    # Becomes a filename; path text would write outside the output directory.
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    yaw_deg: float  # pod axis relative to the bow, positive to starboard
    offset_x_m: float = 0.0  # from the centreline, positive to starboard
    offset_y_m: float = 0.0  # from midships, positive forward
    cameras: list[Camera] = Field(min_length=1)


class Mount(NamedTuple):
    """A camera and the pod that aims it."""

    pod: Pod
    camera: Camera

    @property
    def name(self) -> str:
        return f"{self.pod.name}_{self.camera.kind}_{self.bearing_deg:+g}"

    @property
    def bearing_deg(self) -> float:
        """Relative to the bow: pod yaw + fan, so re-aiming a pod moves its cameras."""
        return self.pod.yaw_deg + self.camera.fan_deg


class Rig(Model):
    """The pods on the ownship."""

    height_m: float = Field(gt=0.0)
    tilt_deg: float = 0.0
    pods: list[Pod] = Field(min_length=1)

    @property
    def mounts(self) -> list[Mount]:
        return [Mount(pod, camera) for pod in self.pods for camera in pod.cameras]

    @model_validator(mode="after")
    def _names_are_unique(self) -> "Rig":
        """Two cameras of one name share a datablock and overwrite each other's file.

        Checked over mounts: two pods may share a fan angle, not a bearing.
        """
        names = [mount.name for mount in self.mounts]
        if len(set(names)) != len(names):
            raise ValueError(f"two cameras share a name: {sorted(names)}")
        return self


class Sea(Model):
    """Sea state. Wind reaches the waves through wavelength and slope; see `scene`.

    271-311 K is the span of the shipped optical-constant table. `lwir` clamps to it;
    here it is an error.
    """

    t_sea_k: float = Field(default=lwir.T_SEA_K, ge=271.0, le=311.0)
    wind_speed_mps: float = Field(default=7.0, ge=0.0)


class Sky(Model):
    """Blender's Sky Texture in EO, and the downwelling radiance the sea reflects in IR.

    Haze is `aerosol_density`, the node's own parameter. Its `turbidity` belongs to the
    Preetham and Hosek-Wilkie models and is ignored by this one, so naming it that would
    be a knob that changes nothing.

    `t_air_k` scales the IR sky and nothing in EO. Its bound is where the fixed sky
    profile stays credible; a colder or hotter sea needs a new profile, not a wider
    bound.
    """

    sun_elevation_deg: float = Field(default=30.0, ge=-90.0, le=90.0)
    sun_bearing_deg: float = 0.0
    aerosol_density: float = Field(default=1.0, ge=0.0, le=10.0)
    t_air_k: float = Field(default=lwir.T_AIR_K, ge=250.0, le=320.0)


class Object(Model):
    """Something to detect."""

    asset: str
    range_m: float = Field(gt=0.0)
    bearing_deg: float
    heading_deg: float = 0.0
    t_k: float = Field(default=293.0, ge=250.0, le=400.0)


class Ownship(Model):
    """The vessel the rig is bolted to. Its own cameras see it, which is the point:
    the forecastle occludes the lower frame and the bow has a blind wedge."""

    asset: str
    t_k: float = Field(default=296.0, ge=250.0, le=400.0)


class Targets(Model):
    """A ring of vessels in the world, at one range, spread over a span of bearings.

    World objects, not a per-camera construct: nothing here refers to the rig. That
    every camera ends up with one in frame is a property of the two geometries, so it
    is asserted in a test rather than guaranteed by the placement.
    """

    asset: str
    count: int = Field(gt=0)
    range_m: float = Field(gt=0.0)
    # Inclusive span, spread evenly. Both ends are used, so `count` targets sit at
    # `count - 1` intervals across it.
    bearing_deg: tuple[float, float]
    # Spread evenly too, so aspect varies between targets and projected size with it.
    heading_deg: tuple[float, float] = (0.0, 315.0)
    t_k: float = Field(default=293.0, ge=250.0, le=400.0)

    def _spread(self, span: tuple[float, float], i: int) -> float:
        low, high = span
        return low + (high - low) * i / max(self.count - 1, 1)

    def poses(self) -> list[tuple[float, float]]:
        """Bearing and heading per target, in degrees."""
        return [
            (self._spread(self.bearing_deg, i), self._spread(self.heading_deg, i))
            for i in range(self.count)
        ]


class Outputs(Model):
    """What a render writes.

    Every camera of a listed band is rendered; a scene is built per band, in Cycles.

    EXR by default: it is float, so an LWIR pixel stays the radiance in W m^-2 sr^-1
    that the render produced. PNG is 8-bit and needs a mapping onto it -- for EO the
    exposure below and Blender's AgX film curve, for LWIR an auto-contrast over the
    frame, which is a picture rather than a measurement.
    """

    # uniqueItems so an editor validating against the schema catches a repeat too,
    # not just `load`. `_bands_are_distinct` is what actually enforces it.
    bands: tuple[Band, ...] = Field(
        default=("eo", "ir"), min_length=1, json_schema_extra={"uniqueItems": True}
    )
    # eo is denoised: 16 is enough. ir is not, and needs 64 for the grain to go.
    samples: dict[Band, int] = {"eo": 16, "ir": 64}
    format: ImageFormat = "exr"
    # Stops, and EO clips to white without them: a sunlit sea renders at 3 to 13 where
    # a display wants 1. -5 puts the frame's median luminance on the 18% grey card
    # every light meter is calibrated to -- the baseline reads 7.0, and
    # log2(0.18 / 7.0) = -5.3, rounded to the nearest stop. Display transform only, so
    # the exr keeps its radiance and ir ignores this. Blender clamps to +/-32 in
    # silence, so -50 would read back as -32.
    exposure_ev: float = Field(default=-5.0, ge=-32.0, le=32.0)

    @model_validator(mode="after")
    def _bands_are_distinct(self) -> "Outputs":
        """A repeat renders the same cameras twice, onto the same files."""
        if len(set(self.bands)) != len(self.bands):
            raise ValueError(f"a band is listed twice: {self.bands}")
        return self


class Scenario(Model):
    seed: int = 0
    rig: Rig
    ownship: Ownship | None = None
    targets: Targets | None = None
    sea: Sea = Field(default_factory=Sea)
    sky: Sky = Field(default_factory=Sky)
    objects: list[Object] = Field(default_factory=list)
    outputs: Outputs = Field(default_factory=Outputs)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`over` wins. Two tables merge; anything else replaces outright."""
    out = dict(base)
    for key, value in over.items():
        current = out.get(key)
        out[key] = (
            _merge(current, value)
            if isinstance(current, dict) and isinstance(value, dict)
            else value
        )
    return out


def _include_path(name: Any, base: Path, block: str | None = None) -> Path:
    """Resolve rule 2. `block` is the preset directory; None allows only a path."""
    if not isinstance(name, str):
        raise TypeError(f"include must be a string, got {name!r}")
    if "/" in name or name.endswith(".toml"):
        return base / name
    if block is None:
        raise ValueError(f"{name!r} must be a path: it needs a '/' or a '.toml' suffix")
    return CFG_DIR / block / f"{name}.toml"


def _expand(node: Any, block: str | None, base: Path, chain: tuple[Path, ...]) -> Any:
    """Resolve every `preset` key in the tree, innermost first."""
    if isinstance(node, list):
        return [_expand(item, block, base, chain) for item in node]
    if not isinstance(node, dict):
        return node
    out = {key: _expand(value, key, base, chain) for key, value in node.items()}
    if (name := out.pop("preset", None)) is None:
        return out
    return _merge(_read(_include_path(name, base, block), chain), out)


def _read(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ValueError(f"circular include: {cycle}")
    chain = (*chain, path)

    with path.open("rb") as handle:  # TOML is UTF-8 by spec, so never read_text
        data = tomllib.load(handle)

    # This file's presets resolve before the parent merges in, or an inherited value
    # would outrank a preset this file names explicitly.
    data = _expand(data, None, path.parent, chain)
    if (parent := data.pop("extends", None)) is not None:
        data = _merge(_read(_include_path(parent, path.parent), chain), data)
    return data


def load(path: str | Path) -> Scenario:
    """Read a scenario TOML, resolving `extends` and `preset`, and validate it."""
    return Scenario.model_validate(_read(Path(path)))
