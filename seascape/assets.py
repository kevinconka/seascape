"""Asset manifest: where a mesh comes from, and the cache it lands in."""

import hashlib
import os
import shutil
import tomllib
import urllib.request
import uuid
from pathlib import Path, PurePosixPath

from pydantic import Field

from seascape.config import Model

# XDG ignores a relative XDG_CACHE_HOME; honouring one puts meshes in the source tree.
_XDG = os.environ.get("XDG_CACHE_HOME", "")
CACHE = (Path(_XDG) if _XDG.startswith("/") else Path.home() / ".cache") / "seascape"
MANIFEST = Path(__file__).parent / "assets.toml"


class Asset(Model):
    """One fetchable mesh."""

    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    length_m: float = Field(gt=0.0)  # bow to stern; the mesh arrives in arbitrary units
    # A real figure for the vessel, not a proportion of the mesh: assets are stylised.
    # Required, because defaulting it to zero floats the hull and looks almost right.
    draught_m: float = Field(ge=0.0)
    licence: str = Field(min_length=1)
    attribution: str = Field(min_length=1)


def manifest() -> dict[str, Asset]:
    with MANIFEST.open("rb") as handle:  # TOML is UTF-8 by spec, so never read_text
        return {name: Asset(**body) for name, body in tomllib.load(handle).items()}


def _verify(path: Path, sha256: str) -> None:
    with path.open("rb") as handle:
        got = hashlib.file_digest(handle, "sha256").hexdigest()
    if got != sha256:
        raise ValueError(f"{path}: sha256 {got}, manifest says {sha256}")


def fetch(name: str) -> Path:
    """The cached file for `name`, downloaded once. Verified on every call."""
    asset = manifest()[name]
    path = CACHE / f"{name}{PurePosixPath(asset.url).suffix}"
    if path.exists():
        _verify(path, asset.sha256)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    # A killed or corrupt transfer must never take the cache name, and concurrent
    # callers must not share a scratch file. A pid is not enough: threads share one.
    part = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    # The default socket timeout is None, so a server that stops sending hangs the
    # build forever. (urlretrieve, the obvious alternative, takes no timeout at all.)
    with (
        urllib.request.urlopen(asset.url, timeout=30) as response,
        part.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    _verify(part, asset.sha256)
    part.replace(path)
    return path
