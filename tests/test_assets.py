"""Manifest parsing, and the digest gate in front of the cache."""

import hashlib
import tomllib
from pathlib import Path

import pytest

from seascape import assets
from seascape.config import CFG_DIR

BODY = b"not really a mesh"
DIGEST = hashlib.sha256(BODY).hexdigest()


@pytest.fixture
def one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-entry manifest served over `file://`, cached under `tmp_path`."""
    source = tmp_path / "source.fbx"
    source.write_bytes(BODY)
    manifest = tmp_path / "assets.toml"
    manifest.write_text(
        f'[ship]\nurl = "{source.as_uri()}"\nsha256 = "{DIGEST}"\n'
        'licence = "CC0-1.0"\nattribution = "nobody"\n'
    )
    monkeypatch.setattr(assets, "MANIFEST", manifest)
    monkeypatch.setattr(assets, "CACHE", tmp_path / "cache")
    return source


def test_every_object_preset_names_an_asset() -> None:
    """A preset naming a mesh nothing can fetch fails at render time, not load time."""
    for preset in sorted((CFG_DIR / "objects").glob("*.toml")):
        with preset.open("rb") as handle:
            assert tomllib.load(handle)["asset"] in assets.manifest(), preset


def test_every_url_carries_an_extension() -> None:
    """The cached name takes its suffix from the URL, and Blender picks by extension."""
    for name, asset in assets.manifest().items():
        assert Path(asset.url).suffix, name


def test_fetch_downloads_once(one: Path) -> None:
    first = assets.fetch("ship")
    assert first.read_bytes() == BODY
    assert first.suffix == ".fbx"

    one.unlink()
    assert assets.fetch("ship") == first


def test_fetch_rejects_a_corrupt_cache(one: Path) -> None:
    assets.fetch("ship").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="manifest says"):
        assets.fetch("ship")


def test_fetch_rejects_bytes_that_miss_the_digest(one: Path) -> None:
    one.write_bytes(b"a different mesh")
    with pytest.raises(ValueError, match="manifest says"):
        assets.fetch("ship")
    assert not (assets.CACHE / "ship.fbx").exists()
    assert [p.read_bytes() for p in assets.CACHE.glob("*.part")] == [
        b"a different mesh"
    ]
