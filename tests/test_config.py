"""Loader rules and the shipped presets.

No Blender. Geometry is asserted from the bearings and FOVs as configured; measuring
it off a built scene is a separate check.
"""

import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from seascape.config import CFG_DIR, Camera, Outputs, Rig, Scenario, load

BASELINE = Path(__file__).parents[1] / "scenarios" / "baseline.toml"
SCHEMA = Path(__file__).parents[1] / "schema" / "scenario.json"


def variant(tmp_path: Path, body: str) -> Path:
    """A scenario extending the baseline; `body` is the override TOML."""
    path = tmp_path / "variant.toml"
    path.write_text(f'extends = "{BASELINE}"\n\n{body}')
    return path


@pytest.fixture(scope="module")
def baseline() -> Scenario:
    return load(BASELINE)


def test_baseline_has_eight_cameras(baseline) -> None:
    """Six EO plus an LWIR pair, from twin_pod.toml."""
    kinds = [camera.kind for camera in baseline.rig.cameras]
    assert kinds.count("eo") == 6
    assert kinds.count("ir") == 2


def test_preset_supplies_optics_and_block_supplies_the_mount(baseline) -> None:
    """Optics come from the camera preset, mount from the rig block that names it."""
    eo, ir = baseline.rig.cameras[0], baseline.rig.cameras[-1]
    assert (eo.hfov_deg, eo.width_px, eo.height_px) == (45.0, 1920, 1080)
    assert (ir.hfov_deg, ir.width_px, ir.height_px) == (24.0, 640, 512)
    assert (eo.pod, eo.bearing_deg) == ("port", -100.0)


def test_objects_merge_their_preset(baseline) -> None:
    """The only list-of-tables preset: asset and temperature from cfg, pose here."""
    obj = baseline.objects[0]
    assert (obj.asset, obj.t_k) == ("container_ship", 295.0)
    assert (obj.range_m, obj.bearing_deg) == (2000.0, 15.0)


def test_a_block_overrides_its_own_preset(tmp_path) -> None:
    """Rule 1's whole point. Disjoint keys would pass whichever way the merge ran."""
    scenario = load(
        variant(
            tmp_path,
            '[[rig.cameras]]\npreset = "eo"\npod = "bow"\n'
            "bearing_deg = 0.0\nhfov_deg = 10.0\n",
        )
    )
    assert scenario.rig.cameras[0].hfov_deg == 10.0  # preset says 45.0
    assert scenario.rig.cameras[0].width_px == 1920  # untouched by the block


def test_a_preset_outranks_an_inherited_value(tmp_path) -> None:
    """A preset the variant names explicitly beats what `extends` brought in.

    Expanding after the parent merge inverts this, and nothing else notices.
    """
    (tmp_path / "single.toml").write_text(
        'height_m = 2.0\n\n[[cameras]]\npreset = "ir"\npod = "bow"\nbearing_deg = 0.0\n'
    )
    scenario = load(variant(tmp_path, '[rig]\npreset = "./single.toml"\n'))
    assert scenario.rig.height_m == 2.0  # baseline says 12.0
    assert [camera.kind for camera in scenario.rig.cameras] == ["ir"]


def test_tables_merge_and_lists_replace(tmp_path, baseline) -> None:
    """A variant changes one key; siblings survive, a list does not."""
    scenario = load(
        variant(
            tmp_path,
            "[sea]\nwind_speed_mps = 3.0\n\n"
            '[[rig.cameras]]\npreset = "ir"\npod = "bow"\nbearing_deg = 0.0\n',
        )
    )
    assert scenario.sea.wind_speed_mps == 3.0
    # Read off the parent, not written out: this is about the merge, not the value.
    assert scenario.sea.t_sea_k == baseline.sea.t_sea_k
    assert scenario.rig.height_m == 12.0  # sibling table survived it too
    assert len(scenario.rig.cameras) == 1  # the list did not


@pytest.mark.parametrize("name", ["./mine.toml", "mine.toml"])
def test_preset_can_be_a_path(tmp_path, name) -> None:
    """A `/` or a `.toml` means a path, so a scenario can carry its own presets."""
    (tmp_path / "mine.toml").write_text('kind = "eo"\nhfov_deg = 12.0\n')
    scenario = load(
        variant(
            tmp_path,
            f'[[rig.cameras]]\npreset = "{name}"\npod = "bow"\nbearing_deg = 0.0\n'
            "width_px = 1920\nheight_px = 1080\n",
        )
    )
    assert scenario.rig.cameras[0].hfov_deg == 12.0


@pytest.mark.parametrize(
    ("preset", "error", "match"),
    [
        ("not_a_real_preset", FileNotFoundError, "not_a_real_preset.toml"),
        (12, TypeError, "must be a string"),
    ],
)
def test_bad_presets_fail_loudly(tmp_path, preset, error, match) -> None:
    """The loader never guesses: it does not try one form and fall back to another."""
    with pytest.raises(error, match=match):
        load(variant(tmp_path, f"[rig]\npreset = {json.dumps(preset)}\n"))


def test_extends_demands_a_path(tmp_path) -> None:
    """`extends` has no preset directory to draw from, so a bare name is an error."""
    path = tmp_path / "bare.toml"
    path.write_text('extends = "baseline"\n')
    with pytest.raises(ValueError, match="must be a path"):
        load(path)


def test_circular_extends_raises_rather_than_recursing(tmp_path) -> None:
    (tmp_path / "a.toml").write_text('extends = "b.toml"\n')
    (tmp_path / "b.toml").write_text('extends = "a.toml"\n')
    with pytest.raises(ValueError, match="circular include"):
        load(tmp_path / "a.toml")


def test_a_mistyped_key_is_an_error_not_a_silent_default(tmp_path) -> None:
    with pytest.raises(ValidationError, match="height_metres"):
        load(variant(tmp_path, "[rig]\nheight_metres = 22.0\n"))


@pytest.mark.parametrize("t_sea_k", [260.0, 400.0])
def test_sea_temperature_is_bounded_at_the_config_boundary(tmp_path, t_sea_k) -> None:
    """The bound is 271-311 K, the span of the shipped optical-constant table."""
    with pytest.raises(ValidationError, match="t_sea_k"):
        load(variant(tmp_path, f"[sea]\nt_sea_k = {t_sea_k}\n"))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_numbers_are_rejected(tmp_path, value) -> None:
    """tomllib parses these and pydantic accepts them by default.

    `bearing_deg` carries no bound, so nothing else would catch one.
    """
    with pytest.raises(ValidationError, match="bearing_deg"):
        load(
            variant(
                tmp_path,
                f'[[rig.cameras]]\npreset = "ir"\npod = "bow"\nbearing_deg = {value}\n',
            )
        )


def test_committed_schema_matches_the_models() -> None:
    """Editors validate against the committed file; a stale one is worse than none."""
    assert json.loads(SCHEMA.read_text()) == Scenario.model_json_schema(), (
        "schema/scenario.json is stale. Regenerate it:\n"
        "    uv run seascape schema > schema/scenario.json"
    )


def test_baseline_points_at_the_committed_schema() -> None:
    """The `#:schema` line is a comment, so nothing else would ever notice it rot."""
    line = BASELINE.read_text().splitlines()[0]
    assert line.startswith("#:schema ")
    assert (BASELINE.parent / line.removeprefix("#:schema ")).resolve() == SCHEMA


def test_every_shipped_preset_parses() -> None:
    """A preset directory is named after the block it serves, and holds valid TOML."""
    presets = sorted(CFG_DIR.rglob("*.toml"))
    assert {path.parent.name for path in presets} == {"rig", "cameras", "objects"}
    for preset in presets:
        tomllib.load(preset.open("rb"))


@pytest.mark.parametrize("exposure_ev", [-50.0, 100.0])
def test_an_exposure_blender_would_clamp_is_rejected(exposure_ev: float) -> None:
    """Blender pins it to +/-32 and says nothing, so the render is not as configured."""
    with pytest.raises(ValidationError, match="exposure_ev"):
        Outputs(exposure_ev=exposure_ev)


@pytest.mark.parametrize("pod", ["../escaped", "/tmp/absolute", "sub/dir"])
def test_a_pod_cannot_be_path_text(pod: str) -> None:
    """A camera's name is a filename, so path text writes outside the output dir."""
    with pytest.raises(ValidationError, match="pod"):
        Camera(
            kind="eo", pod=pod, bearing_deg=0.0, hfov_deg=60.0, width_px=8, height_px=8
        )


def test_two_cameras_cannot_share_a_name() -> None:
    """They would share a datablock and overwrite each other's render."""
    twice = {
        "kind": "eo",
        "pod": "bow",
        "bearing_deg": 0.0,
        "width_px": 8,
        "height_px": 8,
    }
    with pytest.raises(ValidationError, match="share a name"):
        Rig(
            height_m=12.0,
            cameras=[
                Camera(**twice, hfov_deg=60.0),
                Camera(**twice, hfov_deg=30.0),
            ],
        )
