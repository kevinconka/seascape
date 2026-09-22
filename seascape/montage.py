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

# Tall enough to read at a glance, small enough that eight frames fit across a screen.
# Absolute rather than a fraction of the frame, as in ImageMagick's montage (a 120x120
# tile) and feh's --index (60 px): percentage geometry is resolved against each input's
# own resolution, so a rig of mixed-resolution cameras would tile unevenly.
TILE_H = 260
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


# The band comes off the font, not the frame: ImageMagick's montage sizes a label
# ascent - descent + 4 and feh's --index the rendered font height + 5. Pillow reports
# descent as a magnitude, so the line box is the sum.
CAPTION_H = sum(_font().getmetrics()) + 4


def _tile(path: Path, font: FreeTypeFont, caption: str) -> Image.Image:
    frame = Image.open(path).convert("RGB")
    # A frame taller than 260:1 rounds to nothing, and Pillow rejects a zero-width
    # resize with an error that names neither the camera nor the file.
    width = max(1, round(frame.width * TILE_H / frame.height))
    frame = frame.resize((width, TILE_H), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", (width, TILE_H + CAPTION_H), MATTE)
    tile.paste(frame, (0, 0))
    draw = ImageDraw.Draw(tile)
    left, top, right, bottom = draw.textbbox((0, 0), caption, font=font)
    draw.text(
        ((width - (right - left)) / 2, TILE_H + (CAPTION_H - (bottom - top)) / 2 - top),
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
    rows: list[list[Image.Image]] = []
    for band in scenario.outputs.bands:
        tiles = []
        for mount in scenario.rig.mounts:
            if mount.camera.kind != band:
                continue
            # png whatever `outputs.format` says: an exr is float radiance, and
            # turning one into a picture is the render's display transform, not this.
            frame = into / f"{mount.name}.png"
            if not frame.exists():
                raise FileNotFoundError(f"{frame}: render it, as png rather than exr")
            tiles.append(_tile(frame, font, mount.name))
        if tiles:
            rows.append(tiles)
    if not rows:
        raise ValueError(f"no frames for any of {scenario.outputs.bands} in {into}")

    widths = [sum(t.width for t in row) + GUTTER * (len(row) - 1) for row in rows]
    heights = [max(t.height for t in row) for row in rows]
    sheet = Image.new(
        "RGB", (max(widths), sum(heights) + GUTTER * (len(rows) - 1)), MATTE
    )
    y = 0
    for row, row_w, row_h in zip(rows, widths, heights, strict=True):
        x = (sheet.width - row_w) // 2  # a short ir row sits under the eo
        for tile in row:
            sheet.paste(tile, (x, y))
            x += tile.width + GUTTER
        y += row_h + GUTTER

    sheet.save(out)
    return out
