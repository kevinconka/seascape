"""Compose a render's frames into one labelled image, for review.

No Blender: this reads the images `render` already wrote, so it runs without the bpy
wheel and a layout can be redone without re-rendering eight 4K frames.

Captions sit in a band under each frame: text burnt into a frame is an artefact that
travels with the dataset, and in an exr it would corrupt radiance.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

from seascape.config import Scenario

# The one size worth choosing is the finished sheet's; a tile's height follows from
# how many frames its row has to carry.
SHEET_W = 2400
GUTTER = 4

# Dark enough that a frame's edge shows against it, and pale ink stays legible over
# both a bright EO frame and a dark thermal one.
MATTE = (24, 24, 24)
INK = (232, 232, 232)


def _font() -> FreeTypeFont:
    """Pillow's built-in face: no font file to ship, or to find missing."""
    font = ImageFont.load_default(size=16)
    # load_default falls back to a bitmap face only when called without a size.
    assert isinstance(font, FreeTypeFont)
    return font


# A line box, so a taller font cannot clip its own descenders.
CAPTION_H = sum(_font().getmetrics()) + 4


def _tile(
    frame: Image.Image, height: int, font: FreeTypeFont, caption: str
) -> Image.Image:
    # Pillow rejects a zero-width resize, naming neither the camera nor the file.
    width = max(1, round(frame.width * height / frame.height))
    frame = frame.resize((width, height), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", (width, height + CAPTION_H), MATTE)
    tile.paste(frame, (0, 0))
    draw = ImageDraw.Draw(tile)
    left, top, right, bottom = draw.textbbox((0, 0), caption, font=font)
    draw.text(
        ((width - (right - left)) / 2, height + (CAPTION_H - (bottom - top)) / 2 - top),
        caption,
        font=font,
        fill=INK,
    )
    return tile


def compose(scenario: Scenario, into: Path) -> Path:
    """Write `montage.png` beside the frames in `into`, one row per band.

    Tiles follow rig order, so a row reads port to starboard. eo and ir never share
    a row: their pixels mean different things.
    """
    out = into / "montage.png"
    # `montage` passes the camera name pattern, so that camera's frame is this file:
    # composed in, then written over, and the next run lays out the sheet itself.
    if any(mount.name == out.stem for mount in scenario.rig.mounts):
        raise ValueError(f"a camera named {out.stem} writes over {out.name}")

    font = _font()
    rows: list[list[tuple[str, Image.Image]]] = []
    for band in scenario.outputs.bands:
        frames = []
        for mount in scenario.rig.mounts:
            if mount.camera.kind != band:
                continue
            # png whatever `outputs.format` says: an exr is float radiance, and
            # turning one into a picture is the render's display transform, not this.
            path = into / f"{mount.name}.png"
            if not path.exists():
                raise FileNotFoundError(f"{path}: render it, as png rather than exr")
            frames.append((mount.name, Image.open(path).convert("RGB")))
        if frames:
            rows.append(frames)
    if not rows:
        raise ValueError(f"no frames for any of {scenario.outputs.bands} in {into}")

    # The busiest row fills the sheet; every other row is narrower and centres. Never
    # past native, or one portrait camera scales the sheet to millions of pixels to
    # make its row reach the width.
    aspect, count = max(
        (sum(f.width / f.height for _, f in row), len(row)) for row in rows
    )
    native = max(f.height for row in rows for _, f in row)
    tile_h = min(round((SHEET_W - GUTTER * (count - 1)) / aspect), native)
    rows = [[(name, _tile(f, tile_h, font, name)) for name, f in row] for row in rows]

    widths = [sum(t.width for _, t in row) + GUTTER * (len(row) - 1) for row in rows]
    heights = [max(t.height for _, t in row) for row in rows]
    sheet = Image.new(
        "RGB", (max(widths), sum(heights) + GUTTER * (len(rows) - 1)), MATTE
    )
    y = 0
    for row, row_w, row_h in zip(rows, widths, heights, strict=True):
        x = (sheet.width - row_w) // 2  # a short ir row sits under the eo
        for _, tile in row:
            sheet.paste(tile, (x, y))
            x += tile.width + GUTTER
        y += row_h + GUTTER

    sheet.save(out)
    return out
