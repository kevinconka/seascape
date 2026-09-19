# AGENTS.md

What an agent needs that a human picks up by working here. Everything else — what the project
is, how to install it, how to run it — lives in [README.md](README.md).

## The rule

**Blender does anything it can do. Code is a liability.**

Before writing a module, check whether Blender already has the feature. The sea is the Ocean
modifier, not a noise shader. The sky is the Sky Texture. Panoramas are a panoramic camera.
Depth and segmentation are render passes.

There is exactly **one** exception, and it is documented so nobody helpfully removes it: LWIR
radiometry lives in numpy because Blender is an RGB renderer with no concept of an 8–14 µm
band. Its Fresnel node takes a scalar IOR; seawater emissivity needs complex IOR (n + i·k).
Do not migrate that into shader nodes.

If a new dependency looks necessary, say why Blender or the standard library can't do it.

## The dev loop

**Code never runs inside Blender.** `seascape` builds a `.blend` in its own process; Blender
opens it. The dependency between them is a file, not an import — so there is nothing to
hot-reload and no module cache to defeat.

Do not add `sys.path` entries, `.pth` files, `importlib.reload`, symlinks into Blender's
script directories, or an add-on wrapper to get around this. Every one of them exists to let
Blender host your code, which is the thing to avoid.

The loop is `seascape build`, then reload in Blender. Building a scene is cheap; process
startup dominates a cycle.

To reload without losing where the user had the viewport, via MCP:

```python
import bpy
from mathutils import Matrix

def _view3d():
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == "VIEW_3D":
                return next(s.region_3d for s in a.spaces if s.type == "VIEW_3D")

path = bpy.data.filepath            # or the .blend seascape just wrote
rv = _view3d()
view = Matrix(rv.view_matrix), rv.view_distance, rv.view_location.copy()
bpy.ops.wm.open_mainfile(filepath=path)
rv = _view3d()                      # regions are rebuilt by open_mainfile
rv.view_matrix, rv.view_distance, rv.view_location = view
rv.update()
```

Without MCP, **File → Revert** (`bpy.ops.wm.revert_mainfile`) does the same minus the view
restore. Either way, reloading discards unsaved in-memory changes, so anything hand-tweaked in
Blender is lost — put it in the build code instead.

Building fresh each time is also what keeps a long-lived session from accumulating state that
the next build inherits.

## Traps that fail silently

These produce wrong output with no error. They are the reason this file exists.

- **Blender's +Z rotation turns a forward-facing object to port.** Every nautical bearing goes
  through the single conversion helper and is negated there and nowhere else. Two negations
  cancel and look plausible.
- **`rotation_mode` is often `QUATERNION`.** Assigning `rotation_euler` is then ignored
  entirely — no exception, no warning, object doesn't move. Set the mode first.
- **Address shader sockets by name, never by index.** `inputs["Distance"]` raises if Blender
  renames it; `inputs[1]` happily writes to whatever now sits in that slot.
- **Objects can share a mesh datablock.** Material slots link to mesh data by default, so
  assigning a material to one object silently changes the other. Use `slot.link = "OBJECT"`
  when they must differ.
- **Shader node trees leak.** If you build a chain, cleanup must remove the whole chain, not
  just the node you tagged. Re-running a build should leave the node count unchanged.
- **The engine identifier is version-dependent.** `BLENDER_EEVEE` means EEVEE Legacy on ≤4.1
  and EEVEE Next on ≥5.0, with `BLENDER_EEVEE_NEXT` in between. Assert against the enum.

## Conventions

- **Degrees at the boundary, radians inside.** Config and ground truth are `*_deg`; internals
  are radians, converted exactly once. Degrees-versus-radians is the live bug class here.
- **Units in field names.** `height_m`, `t_sea_k`, `range_m`. No units library.
- **Randomness comes from named substreams** off the scenario seed — `substream(seed,
  "sea/surface")`. Never `np.random` module functions. Named streams mean adding a component
  doesn't perturb an existing one.
- **Poses are functions of time**, called with `t = 0` today. Ground truth records a timestamp,
  not a frame index: sensors run at different rates, so frame *n* is not one instant.
- **Never commit a `.blend` or an asset.** `.blend` files are build artifacts; assets are
  fetched from `assets.toml` into a gitignored cache.
- **New components are pydantic models in a discriminated union**, registered in one place.
  Not a registry dict, not entry points.
- Annotate every function signature.

## Checks — run before committing

These mirror CI; all must pass.

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

The render-drift check needs a GPU and skips without one, which is also why CI never runs it.
Run it locally before touching anything in the shader chain.

## Writing

Applies to comments, docstrings, commit messages, PR descriptions and docs.

**Comments earn their place or go.** Good code needs few: a better name beats a comment
explaining a worse one. Write one when the *why* cannot be recovered from the code — a
non-obvious constraint, a unit, a workaround for a specific bug, a reference to a spec.

```python
angle = -bearing        # bad: negate the bearing
angle = -bearing        # good: Blender's +Z rotation turns to port
```

Docstrings say what a function is for and what a caller must know. Not how it works — the
code says that, and the docstring will drift from it.

**Avoid the machine cadence.** These read as generated, and most are padding:

- `X, not just Y` and `it's not X, it's Y` used for emphasis. Naming a real alternative
  someone would pick is useful ("the Ocean modifier, not a hand-rolled noise shader").
  Inventing one to sound balanced is not.
- Lists of exactly three where two or five would be truer.
- Every sentence the same length. Vary it, or the prose flattens.
- A long word doing a short word's job.
- Hedges: "it's worth noting", "essentially", "in order to".
- A closing paragraph that restates what was just said.

Prefer the specific. A number, a file name or a flag beats an adjective.

## Working here

- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`, `ci:`).
- **Branches:** matching prefixes (`feat/...`, `fix/...`).
- **Be concise.** PR descriptions and commit messages state the fact, not the journey. No
  debugging narration, no restating the diff.
- **Make PRs scannable.** A table, a before/after, or a rendered frame beats a paragraph.
- Renders are cheap and settle arguments. If a change affects what the camera sees, show it.
