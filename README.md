<h1 align="center">seascape</h1>

<p align="center">
  <em>Synthetic maritime scenes with exact ground truth — EO and LWIR, from a TOML file.</em>
</p>

<p align="center">
  <a href="https://github.com/kevinconka/seascape/actions/workflows/ci.yml"><img src="https://github.com/kevinconka/seascape/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/kevinconka/seascape"><img src="https://codecov.io/gh/kevinconka/seascape/graph/badge.svg" alt="codecov"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/kevinconka/seascape" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.13-blue" alt="Python 3.13"></a>
  <a href="https://www.blender.org/"><img src="https://img.shields.io/badge/blender-5.2%20LTS-orange" alt="Blender 5.2 LTS"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

> [!NOTE]
> Early days. The scaffolding is landing first — see [Status](#status).

Real footage can't put a vessel at exactly 7 NM, hold the visibility constant, or show you the
same ship from eight aspects. `seascape` renders maritime scenes where you choose all of that,
and tells you exactly where everything was.

## Highlights

- **Ground truth you can trust.** Rigs are built from code, so camera extrinsics and intrinsics
  are known by construction rather than estimated. Every target comes back with its true range,
  bearing and pixel extent.
- **Thermal, not just visible.** LWIR renders use real seawater emissivity, angle-dependent sky
  reflection and atmospheric attenuation over the 8–14 µm band.
- **Multi-sensor rigs.** Several cameras, each with its own resolution, optics and modality,
  in one scene with known relative geometry.

## Install

Blender ships as a Python package, so there is nothing else to install.

```bash
git clone https://github.com/kevinconka/seascape.git
cd seascape
uv sync
```

Working on the physics or the config only? `uv sync` skips Blender entirely. Add it when you
want to render:

```bash
uv sync --group blender
```

## Quickstart

```bash
uv run seascape render scenarios/baseline.toml            # images + ground truth
uv run seascape render scenarios/baseline.toml --set rig.tilt_deg=-5
uv run seascape build  scenarios/baseline.toml            # a .blend you can open
```

A scenario is a TOML file describing the world, the platform, the sensors and the targets.
Variants are diffs:

```toml
# scenarios/tilt-down.toml
extends = "baseline.toml"

[rig]
tilt_deg = -5.0
```

Scenarios carry a `#:schema` line, so editors with a TOML language server give you key
completion, inline validation and hover docs.

## Blender MCP (optional)

Only needed if you want to drive Blender interactively from an AI agent. Rendering from the
CLI never touches it.

```bash
blender --command extension repo-add lab_blender_org --url https://lab.blender.org/
blender --command extension install mcp --repo lab_blender_org --enable --sync
claude mcp add blender -- blender-mcp
```

Then in Blender: **Edit → Preferences → System → Network → Allow Online Access**, and
**Save Preferences**.

<details>
<summary>Troubleshooting</summary>

| Symptom | Cause |
|---|---|
| Add-on gone after restarting Blender | Preferences were never saved. Run **Save Preferences**, or enable auto-save. |
| "Online access must be enabled" | Turn on **Allow Online Access**, or pass `--online-mode` for background runs. |
| Connection refused on port 9876 | Blender isn't running, or another instance already holds the port. MCP needs the GUI. |
| Searching "mcp" finds nothing | The repository index syncs at startup. Restart Blender after adding it. |

</details>

## Status

| | |
|---|---|
| Scaffolding and CI | in progress |
| LWIR radiometry | porting |
| Scenario config and schema | porting |
| Scene build and rendering | porting |

Not yet: multi-frame sequences, vessel motion, COCO labels.

## Contributing

`AGENTS.md` covers the conventions and the Blender-specific traps worth knowing before you
change anything.

## Licence

MIT — see [LICENSE](LICENSE). Bundled 3D assets carry their own licences, recorded in
`assets.toml`; some require attribution, which travels with any released dataset.
