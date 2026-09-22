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

    sheet = Image.open(montage.compose(twin_pod, into))

    rows = len({mount.camera.kind for mount in twin_pod.rig.mounts})
    tile_h = (sheet.height - montage.GUTTER * (rows - 1)) / rows - montage.CAPTION_H
    assert sheet.height == rows * (tile_h + montage.CAPTION_H) + montage.GUTTER * (
        rows - 1
    )


def test_the_caption_sits_under_the_frame_and_never_on_it(twin_pod, tmp_path) -> None:
    into = frames(twin_pod, tmp_path)
    flat = (90, 110, 130)

    sheet = Image.open(montage.compose(twin_pod, into))
    tile_h = sheet.height // 2 - montage.CAPTION_H - montage.GUTTER // 2

    assert sheet.getpixel((0, tile_h // 2)) == flat  # frame, untouched
    assert sheet.getpixel((0, tile_h + 2)) == montage.MATTE  # caption band


def test_the_busiest_row_fills_the_sheet(twin_pod, tmp_path) -> None:
    """Tile height follows the content: a six-camera row and a two-camera one cannot
    both be sized by one constant without one of them wasting the sheet.

    Frames bigger than a tile, so the sheet's width governs rather than the cap that
    stops a tile being scaled past the resolution it was rendered at."""
    into = frames(twin_pod, tmp_path, size=(640, 360))

    width = Image.open(montage.compose(twin_pod, into)).width

    assert width == pytest.approx(montage.SHEET_W, abs=montage.GUTTER * 8)


def test_a_tile_is_never_scaled_past_its_own_resolution(twin_pod, tmp_path) -> None:
    """Upscaling invents pixels, and a row of narrow frames would demand a great many
    of them to reach the sheet's width."""
    into = frames(twin_pod, tmp_path, size=(64, 36))

    sheet = Image.open(montage.compose(twin_pod, into))

    assert sheet.width < montage.SHEET_W


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
    """A frame far taller than wide rounds to no width at all, which Pillow refuses to
    resize, and sizing the sheet off its aspect alone runs to millions of pixels."""
    into = frames(twin_pod, tmp_path, size=(1, 400))

    sheet = Image.open(montage.compose(twin_pod, into))

    assert sheet.width > 0
    assert sheet.height < 4 * montage.SHEET_W


def test_a_camera_named_montage_is_an_error(twin_pod, tmp_path) -> None:
    """Its frame is the output file: composed in, then written over."""
    twin_pod.rig.pods[0].cameras[0].name = "montage"
    into = frames(twin_pod, tmp_path)

    with pytest.raises(ValueError, match="montage"):
        montage.compose(twin_pod, into)
