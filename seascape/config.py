"""Scenario config: pydantic models, and a TOML loader that never guesses.

A scenario is a TOML file. Two forms of reuse, both resolved here:

- `extends = "baseline.toml"` at the top level, for a variant that is a diff.
- `preset = "twin_pod"` inside a block, next to the overrides it applies to.

Three rules keep the loader dumb:

1. `preset` is a key inside its block, not a list of parents. No precedence rules.
2. Syntax decides what a name is. A bare identifier is a shipped preset, anything
   with `/` or ending in `.toml` is a path relative to the including file, and
   anything else is an error. Never try one form and fall back to the other.
3. Tables merge, everything else replaces. A list is replaced whole.

Degrees at the boundary: every angle here is `*_deg`, converted to radians exactly
once by whoever builds the scene.
"""

import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from seascape import lwir

CFG_DIR = Path(__file__).parent / "cfg"

# Which preset directory a block draws from, keyed by the block's own name. The one
# place a new component registers itself; `sea` and `sky` are absent because a single
# table of five defaults has nothing to reuse.
PRESET_DIRS = {"rig": "rig", "cameras": "camera", "objects": "object"}


class _Model(BaseModel):
    # A typo in a scenario is a silent wrong render otherwise. `preset` is consumed by
    # the loader before validation, so forbidding extras also catches it leaking.
    model_config = ConfigDict(extra="forbid")


class _Camera(_Model):
    """Mount and optics shared by every modality."""

    pod: str
    bearing_deg: float  # relative to the bow, positive to starboard
    tilt_deg: float = 0.0
    hfov_deg: float = Field(gt=0.0, lt=180.0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


class EOCamera(_Camera):
    kind: Literal["eo"] = "eo"


class IRCamera(_Camera):
    kind: Literal["ir"] = "ir"
    band_m: tuple[float, float] = lwir.BAND_M


type Camera = Annotated[EOCamera | IRCamera, Field(discriminator="kind")]


class Rig(_Model):
    """The sensor mast: where it sits on the ownship, and what is bolted to it."""

    height_m: float = Field(gt=0.0)
    tilt_deg: float = 0.0
    cameras: list[Camera] = Field(min_length=1)


class Sea(_Model):
    """Blender's Ocean modifier is driven by wind, so the config is too.

    The temperature bound is the span of the shipped optical-constant table, which
    already covers any sea surface. `lwir` clamps to it; here it is an error, because
    a scenario asking for 400 K water is a mistake, not something to silently correct.
    """

    t_sea_k: float = Field(default=lwir.T_SEA_K, ge=271.0, le=311.0)
    wind_speed_mps: float = Field(default=7.0, ge=0.0)
    choppiness: float = Field(default=1.0, ge=0.0, le=4.0)


class Sky(_Model):
    """Blender's Sky Texture (Nishita). Turbidity's 1-10 is the node's own range."""

    sun_elevation_deg: float = Field(default=30.0, ge=-90.0, le=90.0)
    sun_bearing_deg: float = 0.0
    turbidity: float = Field(default=2.0, ge=1.0, le=10.0)


class Object(_Model):
    """Something to detect. A person in the water and a floating container count."""

    asset: str
    range_m: float = Field(gt=0.0)
    bearing_deg: float
    heading_deg: float = 0.0
    t_k: float = Field(default=293.0, ge=250.0, le=400.0)


class Scenario(_Model):
    seed: int = 0
    rig: Rig
    sea: Sea = Field(default_factory=Sea)
    sky: Sky = Field(default_factory=Sky)
    objects: list[Object] = Field(default_factory=list)


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


def _preset_path(block: str | None, name: Any, base: Path) -> Path:
    if not isinstance(name, str):
        raise TypeError(f"preset must be a string, got {name!r}")
    if "/" in name or name.endswith(".toml"):
        return base / name
    if not name.isidentifier():
        raise ValueError(
            f"{name!r} is neither a preset name nor a path: "
            "write a bare name, or a path containing '/' or ending in '.toml'"
        )
    if block not in PRESET_DIRS:
        raise ValueError(f"no presets exist for the {block!r} block")
    return CFG_DIR / PRESET_DIRS[block] / f"{name}.toml"


def _expand(node: Any, block: str | None, base: Path, chain: tuple[Path, ...]) -> Any:
    """Resolve every `preset` key in the tree, innermost first."""
    if isinstance(node, list):
        return [_expand(item, block, base, chain) for item in node]
    if not isinstance(node, dict):
        return node
    out = {key: _expand(value, key, base, chain) for key, value in node.items()}
    if (name := out.pop("preset", None)) is None:
        return out
    return _merge(_read(_preset_path(block, name, base), chain), out)


def _read(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in chain:
        cycle = " -> ".join(p.name for p in (*chain, path))
        raise ValueError(f"circular include: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(path)
    chain = (*chain, path)

    data = tomllib.loads(path.read_text())
    if (parent := data.pop("extends", None)) is not None:
        data = _merge(_read(path.parent / parent, chain), data)
    return _expand(data, None, path.parent, chain)


def load(path: str | Path) -> Scenario:
    """Read a scenario TOML, resolving `extends` and `preset`, and validate it."""
    return Scenario.model_validate(_read(Path(path)))
