"""Compose a render's frames into one labelled image, for review.

No Blender: this reads the images `render` already wrote, so it runs without the bpy
wheel and a layout can be redone without re-rendering eight 4K frames.

Captions sit in a band under each frame, never over it. These images are detection and
radiometry data; text burnt into one is an artefact that travels with the dataset, and
in an exr it would corrupt radiance.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

from seascape.config import Scenario

# Tall enough to read at a glance, small enough that eight frames fit across a screen.
TILE_H = 260
CAPTION_H = 26
GUTTER = 4

# Matte, so a frame's own edge is visible against it and the text is legible on both a
# bright EO frame and a dark thermal one.
MATTE = (24, 24, 24)
INK = (232, 232, 232)


def _font() -> FreeTypeFont | ImageFont.ImageFont:
    """Pillow's built-in face, scaled. No font file to ship, or to find missing."""
    return ImageFont.load_default(size=16)


def _tile(
    path: Path, font: FreeTypeFont | ImageFont.ImageFont, caption: str
) -> Image.Image:
    """One frame scaled to `TILE_H`, with its caption in a band underneath."""
    frame = Image.open(path).convert("RGB")
    width = round(frame.width * TILE_H / frame.height)
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

    Rows follow the rig, so a row reads port to starboard the way the pods are bolted
    on, and the bands stay apart because their pixels mean different things.
    """
    font = _font()
    suffix = scenario.outputs.format
    rows: list[list[Image.Image]] = []
    for band in scenario.outputs.bands:
        tiles = []
        for mount in scenario.rig.mounts:
            if mount.camera.kind != band:
                continue
            frame = into / f"{mount.name}.{suffix}"
            if not frame.exists():
                raise FileNotFoundError(f"{frame}: render the scenario first")
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
        x = (sheet.width - row_w) // 2  # centred, so a short ir row sits under the eo
        for tile in row:
            sheet.paste(tile, (x, y))
            x += tile.width + GUTTER
        y += row_h + GUTTER

    out = into / "montage.png"
    sheet.save(out)
    return out
