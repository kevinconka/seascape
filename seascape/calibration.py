"""The geometry a render was made with, written beside its images.

Imports no Blender, so a consumer reads it without the bpy wheel.
"""

from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from seascape.config import Model

FILENAME = "calibration.json"

# Each is +X right, +Y ahead, +Z up; see the `T_<frame>_cam` fields.
type Frame = Literal["world", "vessel", "pod"]

type Row3 = tuple[float, float, float]
type Row4 = tuple[float, float, float, float]
type Matrix3 = tuple[Row3, Row3, Row3]
type Matrix4 = tuple[Row4, Row4, Row4, Row4]


class _Record(Model):
    # Another writer's file, or a newer one, carries fields this reader has no use for.
    model_config = ConfigDict(extra="ignore")


class CameraCalibration(_Record):
    """A pinhole with no distortion, in OpenCV's conventions.

    Camera frame: +X right, +Y down, +Z along the optical axis. Pixel centres sit at
    integer coordinates.
    """

    name: str
    band: str
    image: str = Field(description="relative to the calibration file's folder")
    pod: str | None = None
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    K: Matrix3
    extrinsics: dict[str, Matrix4] = Field(
        min_length=1,
        description=(
            "camera to each named frame: world (+X east, +Y north, +Z up, at sea "
            "level), vessel (+X starboard, +Y bow, +Z up, moving with "
            "the hull) and pod (the enclosure, +Y along its axis)"
        ),
    )


class Calibration(_Record):
    cameras: list[CameraCalibration] = Field(min_length=1)

    def write(self, folder: Path) -> Path:
        path = folder / FILENAME
        path.write_text(self.model_dump_json(indent=2) + "\n")
        return path

    @classmethod
    def read(cls, folder: Path) -> "Calibration":
        return cls.model_validate_json((folder / FILENAME).read_text())
