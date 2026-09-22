"""Laying rendered frames out for review."""

from pathlib import Path

import pytest
from PIL import Image

from seascape import montage
from seascape.config import Scenario, load

SCENARIOS = Path(__file__).parents[1] / "scenarios"


@pytest.fixture
def twin_pod() -> Scenario:
    return load(SCENARIOS / "twin-pod.toml", ['outputs.format = "png"'])


def frames(scenario: Scenario, into: Path, size: tuple[int, int] = (64, 36)) -> Path:
    """Stand-ins for a render, one per camera, named as `render` names them."""
    into.mkdir(parents=True, exist_ok=True)
    for mount in scenario.rig.mounts:
        Image.new("RGB", size, (90, 110, 130)).save(into / f"{mount.name}.png")
    return into


def test_a_montage_carries_every_camera(twin_pod, tmp_path) -> None:
    into = frames(twin_pod, tmp_path)

    out = montage.compose(twin_pod, into)

    assert out == into / "montage.png"
    assert Image.open(out).size[0] > 0


def test_a_band_gets_its_own_row(twin_pod, tmp_path) -> None:
    """An ir png is stretched per frame, so its greys never read as comparable to eo."""
    into = frames(twin_pod, tmp_path)

    height = Image.open(montage.compose(twin_pod, into)).height

    rows = len({mount.camera.kind for mount in twin_pod.rig.mounts})
    assert height == rows * (montage.TILE_H + montage.CAPTION_H) + montage.GUTTER


def test_the_caption_sits_under_the_frame_and_never_on_it(twin_pod, tmp_path) -> None:
    into = frames(twin_pod, tmp_path)
    flat = (90, 110, 130)

    sheet = Image.open(montage.compose(twin_pod, into))

    assert sheet.getpixel((0, montage.TILE_H // 2)) == flat  # frame, untouched
    assert sheet.getpixel((0, montage.TILE_H + 2)) == montage.MATTE  # caption band


def test_the_caption_band_holds_the_font_it_is_drawn_in() -> None:
    """A band fixed in pixels clips the descenders as soon as the font size moves."""
    line_box = sum(montage._font().getmetrics())

    assert line_box < montage.CAPTION_H <= line_box + 8


def test_a_missing_frame_names_itself(twin_pod, tmp_path) -> None:
    """Silently dropping it gives a montage that looks complete and is not."""
    into = frames(twin_pod, tmp_path)
    absent = next(into.glob("*.png"))
    absent.unlink()

    with pytest.raises(FileNotFoundError, match=absent.stem):
        montage.compose(twin_pod, into)


def test_an_unrendered_scenario_is_an_error(twin_pod, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        montage.compose(twin_pod, tmp_path)


def test_a_sliver_of_a_frame_still_gets_a_tile(twin_pod, tmp_path) -> None:
    """A 1:1000 camera rounds to no width at all, which Pillow refuses to resize."""
    into = frames(twin_pod, tmp_path, size=(1, 1000))

    assert Image.open(montage.compose(twin_pod, into)).width > 0


def test_a_camera_named_montage_is_an_error(twin_pod, tmp_path) -> None:
    """Its frame is the output file: composed in, then written over."""
    twin_pod.rig.pods[0].cameras[0].name = "montage"
    into = frames(twin_pod, tmp_path)

    with pytest.raises(ValueError, match="montage"):
        montage.compose(twin_pod, into)
