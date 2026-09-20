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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from seascape import lwir

CFG_DIR = Path(__file__).parent / "cfg"


class Model(BaseModel):
    """Strictness shared by everything this package parses from TOML."""

    # extra: a typo in a scenario is a silent wrong render otherwise. `preset` is
    # consumed by the loader before validation, so this also catches it leaking.
    # inf_nan: tomllib parses `nan` and `inf`, and pydantic accepts both by default.
    # A nan bearing renders a camera pointing nowhere and reports no error.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Camera(Model):
    kind: Literal["eo", "ir"]
    pod: str
    bearing_deg: float  # relative to the bow, positive to starboard
    hfov_deg: float = Field(gt=0.0, lt=180.0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


class Rig(Model):
    """The sensor mast on the ownship."""

    height_m: float = Field(gt=0.0)
    tilt_deg: float = 0.0
    cameras: list[Camera] = Field(min_length=1)


class Sea(Model):
    """Blender's Ocean modifier is driven by wind, so the config is too.

    271-311 K is the span of the shipped optical-constant table. `lwir` clamps to it;
    here it is an error.
    """

    t_sea_k: float = Field(default=lwir.T_SEA_K, ge=271.0, le=311.0)
    wind_speed_mps: float = Field(default=7.0, ge=0.0)
    choppiness: float = Field(default=1.0, ge=0.0, le=4.0)


class Sky(Model):
    """Blender's Sky Texture (Nishita). Turbidity's 1-10 is the node's own range."""

    sun_elevation_deg: float = Field(default=30.0, ge=-90.0, le=90.0)
    sun_bearing_deg: float = 0.0
    turbidity: float = Field(default=2.0, ge=1.0, le=10.0)


class Object(Model):
    """Something to detect."""

    asset: str
    range_m: float = Field(gt=0.0)
    bearing_deg: float
    heading_deg: float = 0.0
    t_k: float = Field(default=293.0, ge=250.0, le=400.0)


class Scenario(Model):
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
