"""The manifest is the only record of what a render owes credit to."""

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
        'licence = "CC0-1.0"\nattribution = "nobody"\npage = "https://example.invalid"\n'
    )
    monkeypatch.setattr(assets, "MANIFEST", manifest)
    monkeypatch.setattr(assets, "CACHE", tmp_path / "cache")
    return source


def test_the_shipped_manifest_parses() -> None:
    assert assets.manifest()


def test_every_object_preset_names_an_asset() -> None:
    """A preset naming a mesh nothing can fetch fails at render time, not load time."""
    for preset in sorted((CFG_DIR / "objects").glob("*.toml")):
        with preset.open("rb") as handle:
            assert tomllib.load(handle)["asset"] in assets.manifest(), preset


def test_fetch_downloads_once(one: Path) -> None:
    first = assets.fetch("ship")
    assert first.read_bytes() == BODY
    assert first.suffix == ".fbx"  # the importer picks off the extension

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
    # The bytes stay as a `.part` for inspection; what must not exist is a cache hit.
    assert not (assets.CACHE / "ship.fbx").exists()


def test_a_failed_download_is_not_cached(
    one: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed transfer leaves a `.part`; only a verified file takes the cache name."""
    monkeypatch.setattr(
        assets.urllib.request,
        "urlretrieve",
        lambda url, path: Path(path).write_bytes(BODY[:4]) and None,
    )
    with pytest.raises(ValueError):
        assets.fetch("ship")
    assert not (assets.CACHE / "ship.fbx").exists()
