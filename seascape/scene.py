"""Build a Blender scene from a scenario. Nothing here renders.

The scene is written to a `.blend` and opened separately, so this module runs in its
own process and never inside Blender.
"""

import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from seascape import lwir
from seascape.assets import fetch, manifest
from seascape.config import Band, Object, Rig, Scenario, Sea, Sky

# An ocean tile is `spatial_size` across with `resolution**2` samples, so spacing is
# spatial_size / resolution**2, and the whole grid spans spatial_size * repeat.
# 4 m resolves the ~34 m dominant wave of a 7 m/s sea. Reaching the 12.4 km horizon of
# a 12 m mast at that spacing would take 157 M vertices, so the grid stops short of it
# and the sea ends in a visible edge.
OCEAN_RESOLUTION = 7
OCEAN_SPACING_M = 4.0

# No target sits closer to that edge than this, and an empty scenario still gets a sea.
REACH_MARGIN = 1.5
MIN_REACH_M = 2000.0

EMISSIVITY_SAMPLES = 256


def _yaw(bearing_deg: float) -> float:
    """Bearing to Blender yaw, in radians.

    Blender's +Z rotation turns a forward-facing object to port, so every bearing is
    negated. This is the only place it happens: two negations cancel and look plausible.
    """
    return -math.radians(bearing_deg)


def _substream(seed: int, name: str) -> np.random.Generator:
    """A named substream, so adding a component cannot perturb an existing one."""
    return np.random.default_rng([seed, *name.encode()])


def _place(obj: bpy.types.Object, east_m: float, north_m: float, up_m: float) -> None:
    """Position and orient with the rotation mode set first.

    A new object's `rotation_mode` is XYZ, but one loaded from an asset may be
    QUATERNION, where assigning `rotation_euler` is ignored with no error.
    """
    obj.rotation_mode = "XYZ"
    obj.location = (east_m, north_m, up_m)


def _sky(sky: Sky, band: Band) -> bpy.types.World:
    world = bpy.data.worlds.new("sky")
    if band == "ir":
        # The Sky Texture is a visible-band scattering model, so it has nothing to say
        # about 8-14 um. Downwelling atmospheric radiance needs LOWTRAN or HITRAN, which
        # this project does not have, so the IR sky is empty rather than invented.
        world.color = (0.0, 0.0, 0.0)
        return world
    tree = world.node_tree
    node = tree.nodes.new("ShaderNodeTexSky")
    # Multiple scattering is Blender 5's name for Nishita. `turbidity` belongs to the
    # Preetham and Hosek-Wilkie models and is silently ignored here.
    node.sky_type = "MULTIPLE_SCATTERING"
    node.sun_elevation = math.radians(sky.sun_elevation_deg)
    node.sun_rotation = _yaw(sky.sun_bearing_deg)
    node.aerosol_density = sky.aerosol_density
    tree.links.new(node.outputs["Color"], tree.nodes["Background"].inputs["Color"])
    return world


def _emissivity_image(t_sea_k: float) -> bpy.types.Image:
    """`lwir.emissivity_curve` baked against cos(theta), which is what the shader has.

    The curve is sampled uniformly in angle; the shader's dot product is uniform in its
    cosine, so it is resampled here rather than corrected in nodes.
    """
    theta, eps = lwir.emissivity_curve(t_sea_k=t_sea_k)
    mu = np.cos(theta)[::-1]
    values = np.interp(np.linspace(0.0, 1.0, EMISSIVITY_SAMPLES), mu, eps[::-1])

    image = bpy.data.images.new(
        "sea_emissivity", EMISSIVITY_SAMPLES, 1, float_buffer=True, is_data=True
    )
    pixels = np.ones((EMISSIVITY_SAMPLES, 4), dtype=np.float32)
    pixels[:, :3] = values[:, None]
    image.pixels.foreach_set(pixels.ravel())
    return image


def _emissive_material(
    name: str, radiance: float, eps: bpy.types.Image | None
) -> bpy.types.Material:
    """Emission of `radiance` W m^-2 sr^-1, scaled by `eps` against cos(theta) if given.

    Blender has no 8-14 um band, so an LWIR surface is an emission whose strength is the
    band radiance. A target with no measured emissivity radiates as a blackbody.
    """
    material = bpy.data.materials.new(name)
    tree = material.node_tree
    tree.nodes.clear()
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    link = tree.links.new
    link(emission.outputs["Emission"], output.inputs["Surface"])

    if eps is None:
        emission.inputs["Strength"].default_value = radiance
        return material

    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    dot = tree.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    facing = tree.nodes.new("ShaderNodeMath")
    facing.operation = "ABSOLUTE"
    lookup = tree.nodes.new("ShaderNodeCombineXYZ")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = eps
    texture.extension = "EXTEND"
    scale = tree.nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = radiance

    link(geometry.outputs["Incoming"], dot.inputs[0])
    # Vector Math names both inputs "Vector", so the second one can only be indexed.
    link(geometry.outputs["Normal"], dot.inputs[1])
    link(dot.outputs["Value"], facing.inputs[0])
    link(facing.outputs["Value"], lookup.inputs["X"])
    link(lookup.outputs["Vector"], texture.inputs["Vector"])
    link(texture.outputs["Color"], scale.inputs[0])
    link(scale.outputs["Value"], emission.inputs["Strength"])
    return material


def _water_material() -> bpy.types.Material:
    """Daylight water: rough enough to catch the sun, refracting at seawater's IOR."""
    material = bpy.data.materials.new("sea")
    principled = material.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (0.004, 0.02, 0.035, 1.0)
    principled.inputs["Roughness"].default_value = 0.05
    principled.inputs["IOR"].default_value = 1.33
    return material


def _sea_material(sea: Sea, band: Band) -> bpy.types.Material:
    if band == "eo":
        return _water_material()
    return _emissive_material(
        "sea", lwir.band_radiance(sea.t_sea_k), _emissivity_image(sea.t_sea_k)
    )


def _sea(sea: Sea, seed: int, reach_m: float, band: Band) -> bpy.types.Object:
    tile_m = OCEAN_SPACING_M * OCEAN_RESOLUTION**2
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    water = bpy.context.object
    water.name = "sea"
    _place(water, 0.0, 0.0, 0.0)

    ocean = water.modifiers.new("ocean", "OCEAN")
    ocean.geometry_mode = "GENERATE"
    ocean.resolution = ocean.viewport_resolution = OCEAN_RESOLUTION
    ocean.spatial_size = round(tile_m)
    ocean.repeat_x = ocean.repeat_y = math.ceil(2 * reach_m / tile_m)
    # Pierson-Moskowitz is the fully developed wind sea, which is what a single wind
    # speed describes. Phillips is the graphics default; JONSWAP needs a fetch length.
    ocean.spectrum = "PIERSON_MOSKOWITZ"
    ocean.wind_velocity = sea.wind_speed_mps
    ocean.choppiness = sea.choppiness
    ocean.random_seed = int(_substream(seed, "sea/surface").integers(2**31))

    water.data.materials.append(_sea_material(sea, band))
    return water


def _cameras(rig: Rig, far_m: float) -> list[bpy.types.Object]:
    cameras = []
    for spec in rig.cameras:
        data = bpy.data.cameras.new(f"{spec.pod}_{spec.kind}_{spec.bearing_deg:+g}")
        # AUTO fits the field of view to whichever image dimension is larger, so a
        # portrait sensor would silently reinterpret hfov as a vertical angle.
        data.sensor_fit = "HORIZONTAL"
        data.angle_x = math.radians(spec.hfov_deg)
        # The default 1000 m puts a 2 km target behind the far plane, where it renders
        # as sky and the clip boundary reads as the horizon. Neither reports anything.
        data.clip_end = far_m
        camera = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(camera)
        _place(camera, 0.0, 0.0, rig.height_m)
        # A camera looks down its local -Z, so +90 deg about X aims it at the horizon.
        camera.rotation_euler = (
            math.radians(90.0 + rig.tilt_deg),
            0.0,
            _yaw(spec.bearing_deg),
        )
        cameras.append(camera)
    return cameras


def _bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    """World-space extent of the meshes in `objects`.

    An empty's `bound_box` is a unit cube at its origin, and an FBX rig is mostly
    empties, so including them silently inflates the extent.
    """
    corners = [
        o.matrix_world @ Vector(corner)
        for o in objects
        if o.type == "MESH"
        for corner in o.bound_box
    ]
    axes = list(zip(*corners, strict=True))
    return Vector([min(a) for a in axes]), Vector([max(a) for a in axes])


def _object(spec: Object, band: Band) -> bpy.types.Object:
    """Import the mesh, fit it to its manifest length, and pose it.

    An asset arrives in whatever units its author used, off-origin, in many parts. It
    is scaled by its bow-to-stern extent, centred, and set down with its lowest point on
    the waterline. Draft is not modelled: nothing knows the hull's displacement.
    """
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(fetch(spec.asset)))
    parts = [o for o in set(bpy.data.objects) - before if o.parent is None]

    low, high = _bounds(parts)
    fit = Matrix.Scale(manifest()[spec.asset].length_m / (high.y - low.y), 4)
    for part in parts:
        part.matrix_world = fit @ part.matrix_world

    low, high = _bounds(parts)
    centre = Matrix.Translation((-(low.x + high.x) / 2, -(low.y + high.y) / 2, -low.z))
    for part in parts:
        part.matrix_world = centre @ part.matrix_world

    if band == "ir":
        # The asset's own materials are albedo, which says nothing about 8-14 um.
        skin = _emissive_material(
            f"{spec.asset}_ir", lwir.band_radiance(spec.t_k), None
        )
        for part in parts:
            for mesh in [part, *part.children_recursive]:
                if mesh.type == "MESH":
                    mesh.data.materials.clear()
                    mesh.data.materials.append(skin)

    anchor = bpy.data.objects.new(spec.asset, None)
    bpy.context.collection.objects.link(anchor)
    for part in parts:
        part.parent = anchor
    _place(
        anchor,
        spec.range_m * math.sin(math.radians(spec.bearing_deg)),
        spec.range_m * math.cos(math.radians(spec.bearing_deg)),
        0.0,
    )
    anchor.rotation_euler = (0.0, 0.0, _yaw(spec.heading_deg))
    return anchor


def build(scenario: Scenario, band: Band = "eo") -> None:
    """Replace the current Blender session's contents with `scenario` in one band.

    A scene is EO or LWIR, never both: the two describe different physics and share no
    units. Rendering both bands means building twice, which costs seconds.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.world = _sky(scenario.sky, band)
    reach_m = REACH_MARGIN * max(
        MIN_REACH_M, *(o.range_m for o in scenario.objects), 0.0
    )
    _sea(scenario.sea, scenario.seed, reach_m, band)
    # The sea's far corner is reach * sqrt(2) away, so the clip plane has to clear that.
    cameras = _cameras(scenario.rig, 1.5 * reach_m)
    for spec in scenario.objects:
        _object(spec, band)
    bpy.context.scene.camera = cameras[0]
    # Until the depsgraph runs, every child still reports its pre-parenting
    # matrix_world, so anything measuring the scene reads the wrong place.
    bpy.context.view_layer.update()
