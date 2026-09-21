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

CURVE_SAMPLES = 256

GRAVITY_MS2 = 9.81

# Cox & Munk 1954, mean square surface slope of a clean sea against wind speed,
# measured off sun glitter photographs. Slope is what decides how rough water looks,
# and it comes mostly from waves far shorter than the dominant one, so it is measured
# rather than derived from wave height.
SLOPE_VARIANCE_INTERCEPT = 0.003
SLOPE_VARIANCE_PER_MPS = 0.00512

# Pierson-Moskowitz peak frequency is 0.877 g / U, which fixes the dominant wavelength.
PM_PEAK = 0.877
MIN_WAVELENGTH_M = 1.0

# How much of Cox & Munk's slope a four-octave noise field actually carries.
RESOLVED_SLOPE_FRACTION = 0.68

# Relief fades out with camera distance. Perspective already smooths distant water;
# this takes the last of the stipple off the approach to the horizon.
WAVE_FADE_M = 30000.0


def wave_length_m(wind_speed_mps: float) -> float:
    """Dominant wavelength of a fully developed sea, Pierson-Moskowitz."""
    length = 2 * math.pi * wind_speed_mps**2 / (PM_PEAK**2 * GRAVITY_MS2)
    return max(length, MIN_WAVELENGTH_M)


def wave_slope(wind_speed_mps: float) -> float:
    """Total RMS surface slope, Cox & Munk. Dimensionless, a tangent."""
    return math.sqrt(SLOPE_VARIANCE_INTERCEPT + SLOPE_VARIANCE_PER_MPS * wind_speed_mps)


def bump_slope(wind_speed_mps: float) -> float:
    """The part of that slope the noise field can carry.

    Cox & Munk measured the whole spectrum down to capillaries. The bump runs four
    octaves below the dominant wave and stops, so it cannot hold the short-wave slope,
    and asking it to reproduce the full figure makes the sea about twice as textured as
    the reference renders. The fraction is calibrated against those renders, not
    derived: it is the one number here that a spectrum integral should eventually
    replace.
    """
    return RESOLVED_SLOPE_FRACTION * wave_slope(wind_speed_mps)


def unresolved_slope(wind_speed_mps: float) -> float:
    """RMS slope the bump cannot carry, left for the shading to account for.

    Variances add, so this is a difference of squares rather than of slopes.
    """
    u = wind_speed_mps
    return math.sqrt(max(wave_slope(u) ** 2 - bump_slope(u) ** 2, 0.0))


def specular_roughness(wind_speed_mps: float) -> float:
    """Blender roughness for a reflection lobe matching the unresolved slope.

    Cycles' GGX takes alpha = roughness^2, and a Gaussian slope of sigma maps to
    alpha = sqrt(2) sigma. This is the consistent partner to an emissivity curve
    averaged over the same slopes: the surface cannot be rough enough to change how
    much it reflects and still be smooth enough to reflect sharply.

    It is expensive. Measured at 12 m against a hull at 2 km, the reflection in the
    water goes from +1.2 W m^-2 sr^-1 over the surrounding sea to nothing, and the sea
    just below the horizon from 1.03 of ambient to 0.90. Both are the physics working,
    not a defect, but a target's wake in the water is not available as a cue.
    """
    return math.sqrt(min(math.sqrt(2.0) * unresolved_slope(wind_speed_mps), 1.0))


def sea_reach_m(rig: Rig, band: Band) -> float:
    """Half-width of the sea plane: far enough that its edge lands inside a pixel.

    The sea is flat, so it has no horizon of its own and runs to the vanishing point.
    Pushing the edge under the angular resolution of the sharpest camera in the band is
    what makes that vanishing point read as a horizon. Earth curvature is not modelled,
    so a target past the true horizon shows when it should be hull-down.
    """
    ifov_rad = min(
        math.radians(camera.hfov_deg) / camera.width_px
        for camera in rig.cameras
        if camera.kind == band
    )
    return rig.height_m / math.tan(ifov_rad / 2)


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
    tree = world.node_tree
    if band == "ir":
        return _thermal_sky(world, sky.t_air_k)
    node = tree.nodes.new("ShaderNodeTexSky")
    # Multiple scattering is Blender 5's name for Nishita. `turbidity` belongs to the
    # Preetham and Hosek-Wilkie models and is silently ignored here.
    node.sky_type = "MULTIPLE_SCATTERING"
    node.sun_elevation = math.radians(sky.sun_elevation_deg)
    node.sun_rotation = _yaw(sky.sun_bearing_deg)
    node.aerosol_density = sky.aerosol_density
    tree.links.new(node.outputs["Color"], tree.nodes["Background"].inputs["Color"])
    return world


def _curve_image(name: str, values: np.ndarray) -> bpy.types.Image:
    """A 1-D lookup the shader samples with a Combine XYZ into an Image Texture."""
    image = bpy.data.images.new(name, len(values), 1, float_buffer=True, is_data=True)
    pixels = np.ones((len(values), 4), dtype=np.float32)
    pixels[:, :3] = np.asarray(values, dtype=np.float32)[:, None]
    image.pixels.foreach_set(pixels.ravel())
    return image


def _emissivity_image(t_sea_k: float, slope_sigma: float) -> bpy.types.Image:
    """`lwir.emissivity_curve` baked against cos(theta), which is what the shader has.

    The curve is sampled uniformly in angle; the shader's dot product is uniform in its
    cosine, so it is resampled here rather than corrected in nodes.
    """
    theta, eps = lwir.emissivity_curve(t_sea_k=t_sea_k, slope_sigma=slope_sigma)
    mu = np.cos(theta)[::-1]
    return _curve_image(
        "sea_emissivity",
        np.interp(np.linspace(0.0, 1.0, CURVE_SAMPLES), mu, eps[::-1]),
    )


def _sky_image(t_air_k: float) -> bpy.types.Image:
    """`lwir.sky_radiance` baked against sin(elevation), which is what the shader has.

    A world shader's ray direction is a unit vector, so its Z is already sin(elevation)
    and no arcsine node is needed. Below the horizon Z is clamped to 0, where the curve
    holds at ambient.
    """
    return _curve_image(
        "sky_radiance",
        lwir.sky_radiance(np.arcsin(np.linspace(0.0, 1.0, CURVE_SAMPLES)), t_air_k),
    )


def _thermal_sky(world: bpy.types.World, t_air_k: float) -> bpy.types.World:
    """Downwelling radiance against elevation, as raw W m^-2 sr^-1.

    This is what the sea reflects, so it is not decoration: leave it black and the sea
    turns black with it wherever emissivity falls, which is most of a maritime image.
    """
    tree = world.node_tree
    tree.nodes.clear()
    link = tree.links.new
    coord = tree.nodes.new("ShaderNodeTexCoord")
    height = tree.nodes.new("ShaderNodeSeparateXYZ")
    above = tree.nodes.new("ShaderNodeMath")
    above.operation = "MAXIMUM"
    above.use_clamp = True
    above.inputs[1].default_value = 0.0
    lookup = tree.nodes.new("ShaderNodeCombineXYZ")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = _sky_image(t_air_k)
    texture.extension = "EXTEND"
    background = tree.nodes.new("ShaderNodeBackground")
    output = tree.nodes.new("ShaderNodeOutputWorld")

    link(coord.outputs["Generated"], height.inputs["Vector"])
    link(height.outputs["Z"], above.inputs[0])
    link(above.outputs["Value"], lookup.inputs["X"])
    link(lookup.outputs["Vector"], texture.inputs["Vector"])
    link(texture.outputs["Color"], background.inputs["Color"])
    link(background.outputs["Background"], output.inputs["Surface"])
    return world


def _blackbody_material(name: str, radiance: float) -> bpy.types.Material:
    """Emission of `radiance` W m^-2 sr^-1.

    Blender has no 8-14 um band, so an LWIR surface is an emission whose strength is the
    band radiance. A target with no measured emissivity radiates as a blackbody.
    """
    material = bpy.data.materials.new(name)
    tree = material.node_tree
    tree.nodes.clear()
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = radiance
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _wave_normals(
    tree: bpy.types.NodeTree, sea: Sea, seed: int
) -> bpy.types.NodeSocket:
    """Wave normals from world position, fading out with camera distance.

    Shading, not geometry, and that is the whole point. A bump normal is evaluated per
    pixel and varies continuously, so distant water averages smooth. Displaced geometry
    at any affordable spacing goes sub-pixel before the horizon and aliases instead --
    measured against the reference renders at nine times the texture it should have,
    and unchanged between 48 and 512 samples, so not the renderer.

    Being shader-only also means coverage is unbounded and circular, with no patch edge
    to hide, and it costs no vertices.
    """
    length_m = wave_length_m(sea.wind_speed_mps)
    scale = tree.nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs["Scale"].default_value = 1.0 / length_m
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    # Scale stays 1 so the vector above carries the wavelength in metres. W is the
    # fourth axis, which moves the field without moving the sea.
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.55
    noise.inputs["W"].default_value = float(
        _substream(seed, "sea/surface").random() * 1e3
    )

    camera = tree.nodes.new("ShaderNodeCameraData")
    rate = tree.nodes.new("ShaderNodeMath")
    rate.operation = "MULTIPLY"
    rate.inputs[1].default_value = -1.0 / WAVE_FADE_M
    fade = tree.nodes.new("ShaderNodeMath")
    fade.operation = "POWER"
    fade.inputs[0].default_value = math.e

    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = bump_slope(sea.wind_speed_mps) * length_m

    link = tree.links.new
    link(geometry.outputs["Position"], scale.inputs[0])
    link(scale.outputs["Vector"], noise.inputs["Vector"])
    link(noise.outputs["Fac"], bump.inputs["Height"])
    link(camera.outputs["View Distance"], rate.inputs[0])
    link(rate.outputs["Value"], fade.inputs[1])
    link(fade.outputs["Value"], bump.inputs["Strength"])
    return bump.outputs["Normal"]


def _incidence_lookup(
    tree: bpy.types.NodeTree,
    curve: bpy.types.Image,
    normal: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Sample `curve` at |cos(theta)| between the wave normal and the viewing ray.

    Against the wave normal, not the plane's: emissivity has to follow the surface a
    ray actually meets, or a flat sea's worth of emissivity gets applied to water that
    is visibly not flat.
    """
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    dot = tree.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    facing = tree.nodes.new("ShaderNodeMath")
    facing.operation = "ABSOLUTE"
    lookup = tree.nodes.new("ShaderNodeCombineXYZ")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = curve
    texture.extension = "EXTEND"

    link = tree.links.new
    link(geometry.outputs["Incoming"], dot.inputs[0])
    # Vector Math names both inputs "Vector", so the second one can only be indexed.
    link(normal, dot.inputs[1])
    link(dot.outputs["Value"], facing.inputs[0])
    link(facing.outputs["Value"], lookup.inputs["X"])
    link(lookup.outputs["Vector"], texture.inputs["Vector"])
    return texture.outputs["Color"]


def _thermal_sea(sea: Sea, seed: int) -> bpy.types.Material:
    """eps(theta) of the sea emitted, the remaining 1 - eps reflected from the sky.

    Emission and reflection are complements, so the two very nearly cancel: the sea
    holds close to ambient at every angle. Emission alone would fall to a fiftieth of
    that by 2 km, because emissivity collapses at grazing incidence and nothing fills
    the gap.

    Blender does the reflection. A Glossy BSDF against the displaced ocean surface
    reflects the real sky in the real mirror direction, and reflects a warm hull in the
    swell too, which a baked sky curve cannot.
    """
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    tree.nodes.clear()
    mirror = tree.nodes.new("ShaderNodeBsdfGlossy")
    # The same unresolved slope the emissivity curve is averaged over. Leaving this
    # specular while eps is not would be the cheaper picture and the inconsistent one.
    mirror.inputs["Roughness"].default_value = specular_roughness(sea.wind_speed_mps)
    # Glossy BSDF ships at 0.8 grey. The Mix Shader already applies the 1 - eps
    # weighting, so anything but white here absorbs a fifth of the reflected sky and
    # cuts a dark notch along the horizon.
    mirror.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = lwir.band_radiance(sea.t_sea_k)
    mix = tree.nodes.new("ShaderNodeMixShader")
    output = tree.nodes.new("ShaderNodeOutputMaterial")

    link = tree.links.new
    normal = _wave_normals(tree, sea, seed)
    link(normal, mirror.inputs["Normal"])
    # Mix Shader names both shader inputs "Shader", so they can only be indexed. Factor
    # is emissivity: 0 at grazing incidence takes the mirror, 1 head-on takes emission.
    link(mirror.outputs["BSDF"], mix.inputs[1])
    link(emission.outputs["Emission"], mix.inputs[2])
    link(
        _incidence_lookup(
            tree,
            _emissivity_image(sea.t_sea_k, unresolved_slope(sea.wind_speed_mps)),
            normal,
        ),
        mix.inputs["Factor"],
    )
    link(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def _water_material(sea: Sea, seed: int) -> bpy.types.Material:
    """Daylight water: rough enough to catch the sun, refracting at seawater's IOR."""
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    principled = tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (0.004, 0.02, 0.035, 1.0)
    principled.inputs["Roughness"].default_value = 0.05
    principled.inputs["IOR"].default_value = 1.33
    tree.links.new(_wave_normals(tree, sea, seed), principled.inputs["Normal"])
    return material


def _sea_material(sea: Sea, seed: int, band: Band) -> bpy.types.Material:
    return _water_material(sea, seed) if band == "eo" else _thermal_sea(sea, seed)


def _sea(sea: Sea, seed: int, reach_m: float, band: Band) -> bpy.types.Object:
    """One flat plane. The waves are in its material.

    Nothing is displaced, so the sea costs four vertices and covers every range the
    camera can see without a patch edge, a tiling seam, or a grid to alias.
    """
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    water = bpy.context.object
    water.name = "sea"
    _place(water, 0.0, 0.0, 0.0)
    water.scale = (2 * reach_m, 2 * reach_m, 1.0)
    water.data.materials.append(_sea_material(sea, seed, band))
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
        skin = _blackbody_material(f"{spec.asset}_ir", lwir.band_radiance(spec.t_k))
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
    if band == "ir":
        # Pixels are radiance in W m^-2 sr^-1, not a picture. Blender defaults to the
        # AgX film curve, which is a lookup built to make photographs pleasant and
        # destroys the one property these pixels have. Any AGC belongs in post.
        view = bpy.context.scene.view_settings
        view.view_transform, view.look = "Standard", "None"
        view.exposure, view.gamma = 0.0, 1.0
    bpy.context.scene.world = _sky(scenario.sky, band)
    reach_m = sea_reach_m(scenario.rig, band)
    _sea(scenario.sea, scenario.seed, reach_m, band)
    # The sea's far corner is reach * sqrt(2) away, so the clip plane has to clear it.
    cameras = _cameras(scenario.rig, 1.5 * reach_m)
    for spec in scenario.objects:
        _object(spec, band)
    bpy.context.scene.camera = cameras[0]
    # Until the depsgraph runs, every child still reports its pre-parenting
    # matrix_world, so anything measuring the scene reads the wrong place.
    bpy.context.view_layer.update()
