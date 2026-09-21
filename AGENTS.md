# AGENTS.md

What an agent needs that a human picks up by working here. Everything else — what the project is, how to install it, how to run it — lives in [README.md](README.md).

## The rule

**Blender does anything it can do. Code is a liability.**

Before writing a module, check whether Blender already has the feature. The sky is the Sky Texture. Panoramas are a panoramic camera. Depth and segmentation are render passes.

Two things are **not** Blender's, both documented so nobody helpfully puts them back.

**Waves are bump normals on a flat plane, not the Ocean modifier.** Displaced geometry goes sub-pixel before the horizon, and sub-pixel geometry aliases instead of averaging; a bump normal is evaluated per pixel, so the far field averages out on its own. It needs no distance fade, and a measured one changed the far-field texture by 3% and the aliasing not at all -- do not add one back without a render to show it earns its place. The accepted cost is that a bump normal cannot occlude, so a wave can never hide a target. `tests/test_render_drift.py` holds this in place; run it with `--render` before touching the sea shader.

**LWIR radiometry lives in numpy.** Blender has no concept of an 8–14 µm band, and its Fresnel node takes a scalar IOR where seawater emissivity needs complex IOR (n + i·k).

If a new dependency looks necessary, say why Blender or the standard library can't do it.

## The dev loop

**Code never runs inside Blender.** `seascape` builds a `.blend` in its own process; Blender opens it. The dependency between them is a file, not an import — so there is nothing to hot-reload and no module cache to defeat.

Do not add `sys.path` entries, `.pth` files, `importlib.reload`, symlinks into Blender's script directories, or an add-on wrapper to get around this. Every one of them exists to let Blender host your code, which is the thing to avoid.

The loop is `seascape build`, then reload in Blender. Building a scene is cheap; process startup dominates a cycle.

To reload without losing where the user had the viewport, via MCP:

```python
import bpy
from mathutils import Matrix


def _view3d():
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == "VIEW_3D":
                return next(s.region_3d for s in a.spaces if s.type == "VIEW_3D")


path = bpy.data.filepath  # or the .blend seascape just wrote
rv = _view3d()
view = Matrix(rv.view_matrix), rv.view_distance, rv.view_location.copy()
bpy.ops.wm.open_mainfile(filepath=path)
rv = _view3d()  # regions are rebuilt by open_mainfile
rv.view_matrix, rv.view_distance, rv.view_location = view
rv.update()
```

Without MCP, **File → Revert** (`bpy.ops.wm.revert_mainfile`) does the same minus the view restore. Either way, reloading discards unsaved in-memory changes, so anything hand-tweaked in Blender is lost — put it in the build code instead.

Building fresh each time is also what keeps a long-lived session from accumulating state that the next build inherits.

## Traps that fail silently

These produce wrong output with no error. They are the reason this file exists.

- **Blender's +Z rotation turns a forward-facing object to port.** Every nautical bearing goes through the single conversion helper and is negated there and nowhere else. Two negations cancel and look plausible.
- **`rotation_mode` is often `QUATERNION`.** Assigning `rotation_euler` is then ignored entirely — no exception, no warning, object doesn't move. Set the mode first.
- **Address shader sockets by name, never by index.** `inputs["Distance"]` raises if Blender renames it; `inputs[1]` happily writes to whatever now sits in that slot.
- **Objects can share a mesh datablock.** Material slots link to mesh data by default, so assigning a material to one object silently changes the other. Use `slot.link = "OBJECT"` when they must differ.
- **Shader node trees leak.** If you build a chain, cleanup must remove the whole chain, not just the node you tagged. Re-running a build should leave the node count unchanged.
- **A sea at air temperature has no LWIR waves.** Emission and reflected sky are then the same radiance, so tilting a facet changes nothing and the surface renders as a flat plate. `t_sea_k - t_air_k` is the wave signal, not a refinement of it.
- **The engine identifier is version-dependent.** `BLENDER_EEVEE` means EEVEE Legacy on ≤4.1 and EEVEE Next on ≥5.0, with `BLENDER_EEVEE_NEXT` in between. The Sky Texture moved the same way: `NISHITA` is `SINGLE_SCATTERING` and `MULTIPLE_SCATTERING` on ≥5.0.
- **Do not validate an engine against the enum.** Under the `bpy` module `render.engine` reports only `['BLENDER_EEVEE']`, on the class and the instance alike, because Cycles registers as an add-on. Assigning `CYCLES` works anyway and reads back. Assign it and let Blender raise: an identifier it does not know is a `TypeError`.
- **A camera's `clip_end` defaults to 1000 m.** A target at 2 km renders as sky and the clip boundary reads as a convincing horizon. Nothing warns. Set it from the scene's reach.
- **`matrix_world` is stale until the depsgraph runs.** Parent an object, move the parent, read a child's `matrix_world`, and you get where it used to be. `view_layer.update()` first, or every measurement quietly describes the wrong scene.
- **An empty's `bound_box` is a unit cube at its origin.** An imported FBX is largely empties, so measuring the extent of "everything I just imported" inflates it and the fit comes out wrong.
- **Cycles denoising is on by default and is not radiometric.** OIDN is an edge-aware image filter. On a world flat at 290.00 K it returns 282.43-293.00 K, worst at the frame border, and it breaks the R=G=B that an LWIR scene guarantees. Turn it off for the `ir` band; EO is a picture and keeps it.
- **The Sky Texture's `turbidity` does nothing under the scattering models.** It belongs to Preetham and Hosek-Wilkie. Haze there is `aerosol_density`. Setting the wrong one is accepted in silence and changes no pixel, which was verified by rendering both.

## Conventions

- **Degrees at the boundary, radians inside.** Config and ground truth are `*_deg`; internals are radians, converted exactly once. Degrees-versus-radians is the live bug class here.
- **Every physical number cites a source**, in a comment beside it: a published relation, or a derivation from one, or a measurement of Blender itself that a test pins. Never a constant fitted to a render. Renders of an eyeballed scene are not a physics target, and a fitted number cannot be checked by anyone reading the diff.
- **Units in field names.** `height_m`, `t_sea_k`, `range_m`. No units library.
- **Randomness comes from named substreams** off the scenario seed — `substream(seed, "sea/surface")`. Never `np.random` module functions. Named streams mean adding a component doesn't perturb an existing one.
- **Poses are functions of time**, called with `t = 0` today. Ground truth records a timestamp, not a frame index: sensors run at different rates, so frame *n* is not one instant.
- **Never commit a `.blend` or an asset.** `.blend` files are build artifacts; assets are fetched from `assets.toml` into a gitignored cache.
- **New components are pydantic models in a discriminated union**, registered in one place. Not a registry dict, not entry points.
- Annotate every function signature.

## Checks — run before committing

These mirror CI; all must pass.

```bash
uvx ruff check .
uvx ruff format --check .
uvx ty check
uv run pytest
```

ruff and ty go through `uvx` deliberately: never add them to a dependency group and never pin them, `.pre-commit-config.yaml` included. `uvx pre-commit install` automates the ruff lines.

The render-drift check needs a GPU and skips without one, which is also why CI never runs it. Run it locally before touching anything in the shader chain.

## Working here

- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`, `ci:`).
- **Branches:** matching prefixes (`feat/...`, `fix/...`).
- **Be concise.** Commit messages, PR descriptions, comments and docs state the fact, not the journey. No debugging narration, no restating the diff, no closing paragraph that repeats what was just said.
- **Comments are for what the code cannot say.** A better name beats a comment explaining a worse one. Write one for a non-obvious constraint, a unit, a workaround, a spec reference — never to restate the line. `# negate the bearing` is noise; `# Blender's +Z turns to port` is the reason.
- **Avoid the machine cadence.** `X, not just Y` for emphasis, lists of exactly three, uniformly long sentences, hedges like "it's worth noting". Prefer the specific: a number, a file name or a flag beats an adjective.
- **Make PRs scannable.** A table, a before/after, or a rendered frame beats a paragraph.
- Renders are cheap and settle arguments. If a change affects what the camera sees, show it.
