<h1 align="center">seascape</h1>

<p align="center">
  <em>Maritime scene generator for sensor validation. Multi-camera rigs, EO/LWIR, exact ground truth.</em>
</p>

<p align="center">
  <a href="https://github.com/kevinconka/seascape/actions/workflows/ci.yml"><img src="https://github.com/kevinconka/seascape/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
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
- **LWIR as well as EO.** Thermal renders use real seawater emissivity, angle-dependent sky
  reflection and atmospheric attenuation across the 8–14 µm band.
- **Multi-sensor rigs.** Several cameras, each with its own resolution, optics and modality,
  in one scene with known relative geometry.

## Install

Blender ships as a Python package, so there is nothing else to install.

```bash
git clone https://github.com/kevinconka/seascape.git
cd seascape
uv sync
```

Only touching the radiometry or the scenario config? `uv sync --no-group blender` skips the
Blender wheel — 245 MB on Apple silicon, 402 MB on Linux. Those parts are plain NumPy and run
without it.

## Quickstart

```bash
uv run seascape render scenarios/baseline.toml            # images + ground truth
uv run seascape render scenarios/baseline.toml --set rig.tilt_deg=-5
uv run seascape build  scenarios/baseline.toml --open     # a .blend, opened in Blender
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

Lets an AI agent inspect and edit whatever scene you have open in Blender. Rendering from the
CLI never touches it — skip this unless you want the interactive workflow.

1. **Add the connector.** In Claude Desktop: **Customize → Connectors**, search *Blender*,
   click **Add**. It's first-party, so there's no config file and no `.mcpb`.
2. **Install the Blender add-on.** Open the [add-on install page](https://www.blender.org/lab/mcp-server/#add-on)
   next to Blender and drag the install link onto the Blender window — **twice**. The first
   drop allows the Blender Lab extension repository, the second installs the add-on.
3. **Start it.** In Blender: **Edit → Preferences → Add-ons**, find *BlenderMCP*, enable
   **start MCP server**. Then **Save Preferences**, or it's gone on restart.

Check it's listening:

```bash
lsof -nP -iTCP:9876 -sTCP:LISTEN
```

> [!WARNING]
> The add-on runs generated code in your Blender session with no sandbox, and the port is
> unauthenticated. Changes only persist when you save in Blender.

<details>
<summary>Troubleshooting</summary>

| Symptom | Cause |
|---|---|
| Add-on gone after restarting Blender | Preferences were never saved. Run **Save Preferences**, or enable auto-save. |
| "Online access must be enabled" | **Edit → Preferences → System → Network → Allow Online Access**. |
| Nothing listening on 9876 | Blender isn't running, or the add-on is disabled. MCP needs the GUI. |
| Listening, but the wrong scene answers | Another Blender instance bound the port first. Only one can hold it. |
| Dragging the link does nothing | Drop it twice — the first drop only registers the repository. |
| A guide tells you to run `uvx blender-mcp` | That's [`ahujasid/blender-mcp`](https://github.com/ahujasid/blender-mcp), a different community server. Both work; don't mix their instructions. |

</details>

## Status

| Area | State |
|---|---|
| Scaffolding and CI | in progress |
| LWIR radiometry | porting |
| Scenario config and schema | porting |
| Scene build and rendering | porting |
| Multi-frame sequences | planned |
| Vessel motion | planned |
| COCO labels | planned |

## Contributing

`AGENTS.md` covers the conventions and the Blender traps to know before changing anything.

## Licence

MIT — see [LICENSE](LICENSE). Bundled 3D assets carry their own licences, recorded in
`assets.toml`; some require attribution, which travels with any released dataset.
