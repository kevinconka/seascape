"""Loader rules and the shipped presets.

No Blender. The geometry assertions check the numbers issue #6 accepts against, at
the config level; measuring them from a built scene is a separate check.
"""

import itertools
import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from seascape.config import Scenario, load

BASELINE = Path(__file__).parents[1] / "scenarios" / "baseline.toml"
SCHEMA = Path(__file__).parents[1] / "schema" / "scenario.json"


@pytest.fixture(scope="module")
def baseline() -> Scenario:
    return load(BASELINE)


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_baseline_has_eight_cameras(baseline) -> None:
    """Six EO plus an LWIR pair, which is the eight images acceptance asks for."""
    kinds = [camera.kind for camera in baseline.rig.cameras]
    assert kinds.count("eo") == 6
    assert kinds.count("ir") == 2


@pytest.mark.parametrize(
    ("pod", "span_deg", "overlap_deg"), [("port", 125.0, 5.0), ("bow", 44.0, 4.0)]
)
def test_pod_geometry(baseline, pod, span_deg, overlap_deg) -> None:
    """Combined span and neighbour overlap, from the bearings and FOVs as configured.

    A typo in twin_pod.toml or in a camera preset moves these; nothing else does.
    """
    edges = sorted(
        (c.bearing_deg - c.hfov_deg / 2, c.bearing_deg + c.hfov_deg / 2)
        for c in baseline.rig.cameras
        if c.pod == pod
    )
    assert edges[-1][1] - edges[0][0] == pytest.approx(span_deg)
    overlaps = [left[1] - right[0] for left, right in itertools.pairwise(edges)]
    assert overlaps == pytest.approx([overlap_deg] * len(overlaps))


def test_preset_supplies_optics_and_block_supplies_the_mount(baseline) -> None:
    """The half of a camera that is reusable comes from the preset, the rest doesn't."""
    camera = baseline.rig.cameras[0]
    assert (camera.hfov_deg, camera.width_px) == (45.0, 1920)
    assert (camera.pod, camera.bearing_deg) == ("port", -100.0)


def test_extends_overrides_one_key_and_keeps_the_rest(tmp_path) -> None:
    """Acceptance check 5: change the mounting height without touching the cameras."""
    variant = write(
        tmp_path / "taller.toml",
        f'extends = "{BASELINE}"\n\n[rig]\nheight_m = 22.0\n',
    )
    scenario = load(variant)
    assert scenario.rig.height_m == 22.0
    assert len(scenario.rig.cameras) == 8
    assert scenario.sky.sun_bearing_deg == 135.0


def test_tables_merge_and_lists_replace(tmp_path) -> None:
    """The one merge rule. A replaced list is what makes an override predictable."""
    variant = write(
        tmp_path / "one_camera.toml",
        f'extends = "{BASELINE}"\n\n[sea]\nchoppiness = 0.2\n\n'
        '[[rig.cameras]]\npreset = "ir"\npod = "bow"\nbearing_deg = 0.0\n',
    )
    scenario = load(variant)
    assert scenario.sea.choppiness == 0.2
    assert scenario.sea.t_sea_k == 288.0  # sibling key survived the merge
    assert len(scenario.rig.cameras) == 1  # the list did not


def test_preset_can_be_a_path(tmp_path) -> None:
    """A `/` or a `.toml` means a path, so a scenario can carry its own presets."""
    write(tmp_path / "mine.toml", 'kind = "eo"\nhfov_deg = 12.0\n')
    variant = write(
        tmp_path / "narrow.toml",
        f'extends = "{BASELINE}"\n\n[[rig.cameras]]\n'
        'preset = "./mine.toml"\npod = "bow"\nbearing_deg = 0.0\n'
        "width_px = 1920\nheight_px = 1080\n",
    )
    assert load(variant).rig.cameras[0].hfov_deg == 12.0


@pytest.mark.parametrize(
    ("preset", "match"),
    [
        ("no such preset", "neither a preset name nor a path"),
        ("not_a_real_preset", "not_a_real_preset.toml"),
        (12, "must be a string"),
    ],
)
def test_bad_preset_names_fail_loudly(tmp_path, preset, match) -> None:
    """The loader never guesses: it does not try one form and fall back to another."""
    variant = write(
        tmp_path / "bad.toml",
        f'extends = "{BASELINE}"\n\n[rig]\npreset = {json.dumps(preset)}\n',
    )
    with pytest.raises((ValueError, TypeError, OSError), match=match):
        load(variant)


def test_presets_are_rejected_where_none_exist(tmp_path) -> None:
    variant = write(
        tmp_path / "sky.toml",
        f'extends = "{BASELINE}"\n\n[sky]\npreset = "clear"\n',
    )
    with pytest.raises(ValueError, match="no presets exist for the 'sky' block"):
        load(variant)


def test_circular_extends_raises_rather_than_recursing(tmp_path) -> None:
    write(tmp_path / "a.toml", 'extends = "b.toml"\n')
    write(tmp_path / "b.toml", 'extends = "a.toml"\n')
    with pytest.raises(ValueError, match="circular include"):
        load(tmp_path / "a.toml")


def test_a_mistyped_key_is_an_error_not_a_silent_default(tmp_path) -> None:
    variant = write(
        tmp_path / "typo.toml",
        f'extends = "{BASELINE}"\n\n[rig]\nheight_metres = 22.0\n',
    )
    with pytest.raises(ValidationError, match="height_metres"):
        load(variant)


@pytest.mark.parametrize("t_sea_k", [260.0, 400.0])
def test_sea_temperature_is_bounded_at_the_config_boundary(tmp_path, t_sea_k) -> None:
    """`lwir` clamps out-of-range temperatures; a scenario asking for one is a bug."""
    variant = write(
        tmp_path / "hot.toml",
        f'extends = "{BASELINE}"\n\n[sea]\nt_sea_k = {t_sea_k}\n',
    )
    with pytest.raises(ValidationError, match="t_sea_k"):
        load(variant)


def test_committed_schema_matches_the_models() -> None:
    """Editors validate against the committed file, so a stale one is worse than none.

    Regenerate with `uv run seascape schema > schema/scenario.json`.
    """
    assert json.loads(SCHEMA.read_text()) == Scenario.model_json_schema()


def test_baseline_points_at_the_committed_schema() -> None:
    """The `#:schema` line is a comment, so nothing else would ever notice it rot."""
    line = BASELINE.read_text().splitlines()[0]
    assert line.startswith("#:schema ")
    assert (BASELINE.parent / line.removeprefix("#:schema ")).resolve() == SCHEMA


def test_every_shipped_preset_parses() -> None:
    presets = sorted((Path(__file__).parents[1] / "seascape" / "cfg").rglob("*.toml"))
    assert len(presets) >= 4
    for preset in presets:
        tomllib.loads(preset.read_text())
