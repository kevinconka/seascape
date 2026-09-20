"""Asset manifest: where a mesh comes from, and the cache it lands in.

Meshes are never committed. `assets.toml` records the source, the digest that makes a
cached copy trustworthy, and the credit a released dataset has to carry.
"""

import hashlib
import os
import tomllib
import urllib.request
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from pydantic import Field

from seascape.config import Model

CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "seascape"
MANIFEST = Path(__file__).parent / "assets.toml"


class Asset(Model):
    """One fetchable mesh.

    `licence` and `attribution` are here rather than in a README because they have to
    reach whatever ships the renders, and only this file knows what was used.
    """

    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    licence: str
    attribution: str
    page: str  # where a human reads the terms; `url` is the bytes


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
    path = CACHE / f"{name}{PurePosixPath(urlparse(asset.url).path).suffix}"
    if path.exists():
        _verify(path, asset.sha256)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    # Download to a sibling: a killed transfer must not read as a cache hit, and bytes
    # that fail the digest must never reach `path`.
    part = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(asset.url, part)
    _verify(part, asset.sha256)
    part.replace(path)
    return path
