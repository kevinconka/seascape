"""Build a Blender scene from a scenario. Nothing here renders.

The scene is written to a `.blend` and opened separately, so this module runs in its
own process and never inside Blender.

Sources
-------
Dominant wavelength: Pierson & Moskowitz, "A proposed spectral form for fully developed
wind seas based on the similarity theory of S. A. Kitaigorodskii", Journal of
Geophysical Research 69(24) 5181, 1964 (doi:10.1029/JZ069i024p05181).

Slope variance: Cox & Munk, "Measurement of the roughness of the sea surface from
photographs of the sun's glitter", JOSA 44(11) 838, 1954 (doi:10.1364/JOSA.44.000838),
clean-sea fit, equation 13.

Slope spectrum: Phillips, "The equilibrium range in the spectrum of wind-generated
waves", Journal of Fluid Mechanics 4(4) 426, 1958 (doi:10.1017/S0022112058000550).

Microfacet lobe: Walter, Marschner, Li & Torrance, "Microfacet models for refraction
through rough surfaces", EGSR 2007 (doi:10.2312/EGWR/EGSR07/195-206) for GGX; Burley,
"Physically-based shading at Disney", SIGGRAPH 2012 course notes, for the alpha =
roughness^2 convention Cycles follows.
"""

import math
from collections.abc import Iterable

import bpy
import numpy as np
from mathutils import Matrix, Vector

from seascape import lwir
from seascape.assets import Asset, fetch, manifest
from seascape.config import (
    Band,
    ImageFormat,
    Object,
    Outputs,
    Rig,
    Scenario,
    Sea,
    Sky,
    Targets,
)

CURVE_SAMPLES = 256

GRAVITY_MS2 = 9.81

# Waves, end to end. Each step is a published relation or follows from one:
#
#   wavelength      2 pi U^2 / (0.877^2 g)          Pierson-Moskowitz 1964
#   total slope     sqrt(0.003 + 0.00512 U)         Cox & Munk 1954, eq. 13
#   resolved share  sqrt(octaves / log2(lam/1.7cm)) Phillips 1958 equilibrium range
#   bump relief     resolved share x slope x lam    over the noise transfer below
#   unresolved      sqrt(total^2 - resolved^2)      variances subtract
#   emissivity      Fresnel over unresolved slopes  Masuda 1988, in lwir.py
#   lobe roughness  sqrt(sqrt(2) x unresolved)      GGX alpha = roughness^2
SLOPE_VARIANCE_INTERCEPT = 0.003
SLOPE_VARIANCE_PER_MPS = 0.00512
PM_PEAK = 0.877
MIN_WAVELENGTH_M = 1.0

# Blender's Detail input, which is octaves *beyond* the first: Detail = 0 already
# carries one, measured at 0.51 RMS slope against 0.61 at 4.
NOISE_DETAIL = 4.0

# Amplitude ratio between octaves. Slope goes as amplitude x wavenumber and wavenumber
# doubles each octave, so 0.5 is the ratio that puts equal slope variance in each --
# which is what the Phillips equilibrium range says a wind sea does.
NOISE_ROUGHNESS = 0.5

# 2 pi sqrt(gamma / rho g) = 1.73 cm at gamma = 0.074 N/m: the wavelength of minimum
# phase speed, where surface tension takes over from gravity. The bottom of the slope
# spectrum, not a chosen resolution.
CAPILLARY_WAVELENGTH_M = 0.0173

# RMS gradient of the noise's Fac per noise unit, so a Distance of slope x wavelength
# delivers 0.55 of the slope asked for. Quoted at 2 cm sampling: finer sampling finds
# more. `test_the_noise_delivers_the_slope_it_is_asked_for` pins it.
NOISE_SLOPE_PER_UNIT = 0.55


def wave_length_m(wind_speed_mps: float) -> float:
    """Dominant wavelength of a fully developed sea, Pierson-Moskowitz."""
    length = 2 * math.pi * wind_speed_mps**2 / (PM_PEAK**2 * GRAVITY_MS2)
    return max(length, MIN_WAVELENGTH_M)


def wave_slope(wind_speed_mps: float) -> float:
    """Total RMS surface slope, Cox & Munk. Dimensionless, a tangent."""
    return math.sqrt(SLOPE_VARIANCE_INTERCEPT + SLOPE_VARIANCE_PER_MPS * wind_speed_mps)


def resolved_slope_fraction(wind_speed_mps: float) -> float:
    """Fraction of the RMS slope the noise field can carry, over its octaves.

    Cox & Munk measured the whole spectrum down to capillaries; the noise stops a few
    octaves below the dominant wave. In the Phillips equilibrium range the slope
    spectrum goes as 1/k, so mean-square slope accumulates equally per octave and the
    captured share is a ratio of logs rather than an integral, square-rooted because
    this is slope and that was variance.
    """
    octaves = math.log(wave_length_m(wind_speed_mps) / CAPILLARY_WAVELENGTH_M, 2.0)
    return math.sqrt(min((NOISE_DETAIL + 1.0) / octaves, 1.0))


def bump_slope(wind_speed_mps: float) -> float:
    """The part of that slope the noise field can carry."""
    return resolved_slope_fraction(wind_speed_mps) * wave_slope(wind_speed_mps)


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
    much it reflects and still be smooth enough to reflect sharply. A hull at
    7 NM leaves no measurable reflection (0.000 change); a 400 K slab at 300 m moves
    the sea under it by 136 W m^-2 sr^-1, a hundred times the wave variation.
    """
    return math.sqrt(min(math.sqrt(2.0) * unresolved_slope(wind_speed_mps), 1.0))


# Flat paint over steel, 8-14 um. Paints sit at 0.94-0.96 across this band and the
# colour does not matter, only how flat the finish is; metallic paints are far lower
# and are not what a hull is coated with.
PAINT_EMISSIVITY = 0.94

# Mean radius, IUGG.
EARTH_RADIUS_M = 6_371_000.0

# Cells per side. A cell's sagitta is under a millimetre, 1e-7 of a pixel at the
# horizon: grid enough for the tangent point to land on a face, not an accuracy knob.
SEA_CELLS = 128

# Margin on the horizon, or the grid's own edge becomes the horizon.
SEA_MARGIN = 1.5


def earth_radius_m(refraction_k: float) -> float:
    """Effective radius, R / (1 - k).

    Surveying's standard refraction treatment: a bent ray over R is straight over R'.
    """
    return EARTH_RADIUS_M / (1.0 - refraction_k)


def sea_z_m(east_m: float, north_m: float, radius_m: float) -> float:
    """Height of the sea at a point, relative to the tangent plane at the origin.

    The parabola that osculates the sphere; one definition, so hull and mesh share it.
    """
    return -(east_m * east_m + north_m * north_m) / (2.0 * radius_m)


def horizon_m(height_m: float, refraction_k: float) -> float:
    """Distance to the horizon from `height_m`, tangent to the effective sphere.

    51.8 m gives 27.5 km at k = 0.13, 25.7 km geometric; the 3.86 sqrt(h_m) km rule
    of thumb agrees to 1%.
    """
    return math.sqrt(2.0 * earth_radius_m(refraction_k) * height_m)


def sea_reach_m(rig: Rig, sea: Sea) -> float:
    """Half-width of the sea, a margin past the horizon."""
    return SEA_MARGIN * horizon_m(rig.height_m, sea.refraction_k)


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
    """Position, with the rotation mode set first.

    `rotation_mode` is often QUATERNION, where assigning `rotation_euler` afterwards is
    ignored with no error.
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
    """Downwelling radiance against elevation, as raw W m^-2 sr^-1."""
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


def _sun_vector(sky: Sky) -> tuple[float, float, float]:
    """Unit vector towards the sun. A direction, so the bearing is not negated: that
    belongs to rotations, and `_pose` places by the same sin/cos."""
    elevation = math.radians(sky.sun_elevation_deg)
    bearing = math.radians(sky.sun_bearing_deg)
    return (
        math.cos(elevation) * math.sin(bearing),
        math.cos(elevation) * math.cos(bearing),
        math.sin(elevation),
    )


def _thermal_skin(name: str, t_k: float, sky: Sky) -> bpy.types.Material:
    """eps of a painted hull emitted, the remaining 1 - eps reflected from the sky.

    The sea's shape, for the sea's reason. A pure emitter leaves the same radiance in
    every direction, so a vessel renders as one flat value however it is lit or turned.
    Reflecting the other 6% gives it back the angular structure a real hull has: a deck
    faces the cold zenith, a vertical side sees half sky and half sea.

    Diffuse rather than glossy, which is where this parts from the sea: flat marine
    paint is near-Lambertian in this band, so a hull scatters the sky rather than
    mirroring it.
    """
    material = bpy.data.materials.new(name)
    tree = material.node_tree
    tree.nodes.clear()
    # White: the Mix Shader already applies the 1 - eps weighting, so a grey here would
    # absorb part of the reflected sky a second time.
    scatter = tree.nodes.new("ShaderNodeBsdfDiffuse")
    scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    mix = tree.nodes.new("ShaderNodeMixShader")
    # Mix Shader names both shader inputs "Shader", so they can only be indexed.
    mix.inputs["Factor"].default_value = PAINT_EMISSIVITY
    output = tree.nodes.new("ShaderNodeOutputMaterial")

    link = tree.links.new
    link(scatter.outputs["BSDF"], mix.inputs[1])
    link(_sunlit_emission(tree, t_k, sky), mix.inputs[2])
    link(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def _sunlit_emission(
    tree: bpy.types.NodeTree, t_k: float, sky: Sky
) -> bpy.types.NodeSocket:
    """Emission graded from shaded to sunlit by Lambert's cosine on the real sun.

    Two emissions mixed by `max(0, n . sun)`, so both ends are the exact band radiance
    and only the middle interpolates -- 0.2% out against evaluating the Planck integral
    at the blended temperature, which no shader node can do.

    A single temperature leaves the pattern a hull shows in this band on the floor: a
    lit side against a shaded one, and decks hotter than either.
    """
    shaded = tree.nodes.new("ShaderNodeEmission")
    shaded.inputs["Strength"].default_value = lwir.band_radiance(t_k)
    sunlit = tree.nodes.new("ShaderNodeEmission")
    sunlit.inputs["Strength"].default_value = lwir.band_radiance(t_k + sky.solar_gain_k)

    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    facing = tree.nodes.new("ShaderNodeVectorMath")
    facing.operation = "DOT_PRODUCT"
    facing.inputs[1].default_value = _sun_vector(sky)
    # Clamped, or a surface turned away from the sun would cool below shaded.
    lit = tree.nodes.new("ShaderNodeMath")
    lit.operation = "MAXIMUM"
    lit.inputs[1].default_value = 0.0

    grade = tree.nodes.new("ShaderNodeMixShader")
    link = tree.links.new
    link(geometry.outputs["Normal"], facing.inputs[0])
    link(facing.outputs["Value"], lit.inputs[0])
    link(lit.outputs["Value"], grade.inputs["Factor"])
    link(shaded.outputs["Emission"], grade.inputs[1])
    link(sunlit.outputs["Emission"], grade.inputs[2])
    return grade.outputs["Shader"]


def _wave_normals(
    tree: bpy.types.NodeTree, sea: Sea, seed: int
) -> bpy.types.NodeSocket:
    """Wave normals from world position.

    Shading, not geometry. A bump normal is evaluated per pixel and varies
    continuously, so distant water averages smooth; displaced geometry at any
    affordable spacing goes sub-pixel before the horizon and aliases instead.
    `tests/test_render_drift.py` holds this in place.

    Relief does not fade with range. A fade reads as an obvious fix for the stipple
    past the point waves go sub-pixel, and measurably is not one: at 30 km it changed
    the far field by 3% and the aliasing not at all, because it scales amplitude
    uniformly rather than filtering anything. Shortening it to where waves actually go
    sub-pixel made the far field more aliased relative to its own texture, not less.
    """
    length_m = wave_length_m(sea.wind_speed_mps)
    # z multiplier 0: the seed owns that axis, so the sea curving under it cannot slide
    # the wave field. Scaling z drifts the sample three noise periods and ties it to k.
    # 3-D rather than 4-D with the seed in W: same field, 20% cheaper at 4K.
    scale = tree.nodes.new("ShaderNodeVectorMath")
    scale.operation = "MULTIPLY_ADD"
    scale.inputs[1].default_value = (1.0 / length_m, 1.0 / length_m, 0.0)
    scale.inputs[2].default_value = (
        0.0,
        0.0,
        _substream(seed, "sea/surface").random() * 1e3,
    )
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    # Scale stays 1 so the vector above carries the wavelength in metres.
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = NOISE_DETAIL
    noise.inputs["Roughness"].default_value = NOISE_ROUGHNESS

    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = (
        bump_slope(sea.wind_speed_mps) * length_m / NOISE_SLOPE_PER_UNIT
    )

    link = tree.links.new
    link(geometry.outputs["Position"], scale.inputs[0])
    link(scale.outputs["Vector"], noise.inputs["Vector"])
    link(noise.outputs["Fac"], bump.inputs["Height"])
    return bump.outputs["Normal"]


def _incidence_lookup(
    tree: bpy.types.NodeTree,
    curve: bpy.types.Image,
    normal: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """Sample `curve` at |cos(theta)| between the wave normal and the viewing ray.

    Against the wave normal, not the plane's, or a flat sea's emissivity gets applied
    to water that is visibly not flat.
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

    Complements, so the two very nearly cancel and the sea holds close to ambient at
    every angle. Blender does the reflection: a Glossy BSDF against the wave normals
    reflects the real sky in the real mirror direction, and a warm hull with it, which
    a baked sky curve cannot.
    """
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    tree.nodes.clear()
    mirror = tree.nodes.new("ShaderNodeBsdfGlossy")
    # The same unresolved slope the emissivity curve is averaged over.
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


def _sea(sea: Sea, seed: int, reach_m: float, band: Band) -> bpy.types.Object:
    """A grid curved to the earth. The waves are in its material.

    z = -(x^2 + y^2) / 2R osculates the sphere, within a millimetre over the grid.
    Geometry here and not for waves: the bulge is kilometres across, never sub-pixel.
    """
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=SEA_CELLS, y_subdivisions=SEA_CELLS, size=2 * reach_m
    )
    water = bpy.context.object
    water.name = "sea"
    _place(water, 0.0, 0.0, 0.0)
    radius_m = earth_radius_m(sea.refraction_k)
    for vertex in water.data.vertices:
        vertex.co.z = sea_z_m(vertex.co.x, vertex.co.y, radius_m)
    # Flat faces would show their edges in the specular.
    for face in water.data.polygons:
        face.use_smooth = True
    water.data.materials.append(
        _water_material(sea, seed) if band == "eo" else _thermal_sea(sea, seed)
    )
    return water


def _rig(rig: Rig, far_m: float) -> dict[str, bpy.types.Object]:
    root = bpy.data.objects.new("rig", None)
    bpy.context.collection.objects.link(root)
    _place(root, 0.0, 0.0, rig.height_m)

    pods: dict[str, bpy.types.Object] = {}
    for pod in rig.pods:
        empty = bpy.data.objects.new(f"pod_{pod.name}", None)
        bpy.context.collection.objects.link(empty)
        empty.parent = root
        _place(empty, pod.offset_x_m, pod.offset_y_m, 0.0)
        # XYZ euler is Rz @ Ry @ Rx: yaw, then pitch about the pod's own transverse
        # axis. The enclosure pitches as one rigid unit, so its off-axis cameras see a
        # rolled horizon.
        empty.rotation_euler = (math.radians(rig.pitch_deg), 0.0, _yaw(pod.yaw_deg))
        pods[pod.name] = empty

    cameras: dict[str, bpy.types.Object] = {}
    for mount in rig.mounts:
        data = bpy.data.cameras.new(mount.name)
        # AUTO fits the field of view to whichever image dimension is larger, so a
        # portrait sensor would silently reinterpret hfov as a vertical angle.
        data.sensor_fit = "HORIZONTAL"
        data.angle_x = math.radians(mount.camera.hfov_deg)
        # The default 1000 m puts a 2 km target behind the far plane, where it
        # renders as sky and the clip boundary reads as the horizon. Nothing warns.
        data.clip_end = far_m
        camera = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(camera)
        camera.parent = pods[mount.pod.name]
        camera.rotation_mode = "XYZ"
        # A camera looks down its local -Z; +90 deg about X aims it at the horizon.
        # Rx inside the camera's yaw keeps its transverse axis in the pod's horizontal.
        camera.rotation_euler = (
            math.radians(90.0 + mount.camera.pitch_deg),
            0.0,
            _yaw(mount.camera.yaw_deg),
        )
        cameras[mount.name] = camera
    return cameras


def boresight_deg(camera: bpy.types.Object) -> tuple[float, float]:
    """Bearing and elevation a built camera actually points at, in degrees.

    Measured, not summed: the rig's pitch sits between the two yaws, so an off-axis
    camera's azimuth is not their sum -- 0.108 deg at -5 deg of pitch, 9 px at 4K.
    `matrix_world` is stale until the depsgraph runs, so build first.
    """
    forward = camera.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
    forward.normalize()
    return (
        math.degrees(math.atan2(forward.x, forward.y)),
        math.degrees(math.asin(min(1.0, max(-1.0, forward.z)))),
    )


def _corners(objects: Iterable[bpy.types.Object]) -> list[Vector]:
    """World-space bounding corners of the meshes in `objects`.

    An empty's `bound_box` is a unit cube at its origin, and an FBX rig is mostly
    empties, so including them silently inflates the extent.
    """
    return [
        o.matrix_world @ Vector(corner)
        for o in objects
        if o.type == "MESH"
        for corner in o.bound_box
    ]


def _fit(corners: Iterable[Vector], asset: Asset) -> Matrix:
    """Bow to +Y, scaled to the manifest length, centred, keel at the draught.

    Turned first so the length is measured bow to stern whatever the authored axis.
    """
    turn = Matrix.Rotation(_yaw(-asset.bow_deg), 4, "Z")
    axes = list(zip(*(turn @ c for c in corners), strict=True))
    low, high = Vector([min(a) for a in axes]), Vector([max(a) for a in axes])
    scale = asset.length_m / (high.y - low.y)
    low, high = low * scale, high * scale
    return (
        Matrix.Translation(
            (-(low.x + high.x) / 2, -(low.y + high.y) / 2, -low.z - asset.draught_m)
        )
        @ Matrix.Scale(scale, 4)
        @ turn
    )


def _vessel(name: str, t_k: float, band: Band, sky: Sky) -> bpy.types.Object:
    """Import a hull, fit it, and anchor it at the origin under an empty.

    An asset arrives in its author's units, off-origin, in many parts.
    """
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(fetch(name)))
    imported = set(bpy.data.objects) - before
    # Measure everything, move the roots. The shipped ship keeps 40 of its 88 meshes
    # under empties, and measuring only the roots would leave them out of the fit.
    parts = [o for o in imported if o.parent is None]

    fit = _fit(_corners(imported), manifest()[name])
    for part in parts:
        part.matrix_world = fit @ part.matrix_world

    if band == "ir":
        # The asset's own materials are albedo, which says nothing about 8-14 um.
        skin = _thermal_skin(f"{name}_ir", t_k, sky)
        for part in parts:
            for mesh in [part, *part.children_recursive]:
                if mesh.type == "MESH":
                    mesh.data.materials.clear()
                    mesh.data.materials.append(skin)

    anchor = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(anchor)
    for part in parts:
        part.parent = anchor
    _place(anchor, 0.0, 0.0, 0.0)
    return anchor


def _pose(
    anchor: bpy.types.Object,
    range_m: float,
    bearing_deg: float,
    heading_deg: float,
    radius_m: float,
) -> None:
    """Put a hull on the sea at a bearing and range, steering the given course.

    A hull left at z = 0 flies: 11.5 m at 7 NM, 109 m at 40 km.

    Not tilted to the local vertical: range / R is 0.18 m across a 200 m hull at
    7 NM, under a 5 m draught.
    """
    east = range_m * math.sin(math.radians(bearing_deg))
    north = range_m * math.cos(math.radians(bearing_deg))
    _place(anchor, east, north, sea_z_m(east, north, radius_m))
    anchor.rotation_euler = (0.0, 0.0, _yaw(heading_deg))


def _copy_tree(
    obj: bpy.types.Object, parent: bpy.types.Object | None
) -> bpy.types.Object:
    """Duplicate an object tree. `copy()` shares `data`, so N targets cost one mesh.

    The parent inverse copies too, so the tree keeps its shape rather than flattening.
    """
    clone = obj.copy()
    bpy.context.collection.objects.link(clone)
    clone.parent = parent
    for child in obj.children:
        _copy_tree(child, clone)
    return clone


def _targets(
    spec: Targets, band: Band, radius_m: float, sky: Sky
) -> list[bpy.types.Object]:
    first = _vessel(spec.asset, spec.t_k, band, sky)
    poses = spec.poses()
    anchors = [first, *(_copy_tree(first, None) for _ in poses[1:])]
    for i, (anchor, (bearing_deg, heading_deg)) in enumerate(
        zip(anchors, poses, strict=True)
    ):
        anchor.name = f"target_{i}"
        _pose(anchor, spec.range_m, bearing_deg, heading_deg, radius_m)
    return anchors


def _object(spec: Object, band: Band, radius_m: float, sky: Sky) -> bpy.types.Object:
    anchor = _vessel(spec.asset, spec.t_k, band, sky)
    _pose(anchor, spec.range_m, spec.bearing_deg, spec.heading_deg, radius_m)
    return anchor


# Blender's identifier and bit depth. EXR is 32-bit float; ir pixels are radiance.
_FORMATS: dict[ImageFormat, tuple[str, str]] = {
    "exr": ("OPEN_EXR", "32"),
    "png": ("PNG", "8"),
}


def _enable_gpu() -> bool:
    """Point Cycles at a GPU. Without `refresh_devices()` it stays on the CPU."""
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            preferences.compute_device_type = backend
        except TypeError:
            continue  # not compiled into this build
        preferences.refresh_devices()
        if any(device.type != "CPU" for device in preferences.devices):
            for device in preferences.devices:
                # CPU alongside the GPU wins nothing here.
                device.use = device.type != "CPU"
            return True
    return False


def _output(outputs: Outputs, band: Band) -> None:
    """Render and display settings, in the .blend, so F12 renders what `render` does."""
    sc = bpy.context.scene
    # Not EEVEE: no second bounce for world light, so the sea renders at half radiance.
    sc.render.engine = "CYCLES"
    sc.cycles.device = "GPU" if _enable_gpu() else "CPU"
    sc.cycles.samples = getattr(outputs.samples, band)
    # OIDN is on by default and is a picture filter: a world flat at 290.00 K comes
    # back 282.43-293.00 K, and R=G=B, which `render._thermal_png` reads, breaks.
    sc.cycles.use_denoising = band == "eo"
    # HIGH: 16 s vs 5 s per 4K frame, 0.8% pixel change.
    sc.cycles.denoising_quality = "FAST"
    view = sc.view_settings
    if band == "eo":
        view.exposure = outputs.exposure_ev
    else:
        # Radiance in W m^-2 sr^-1, not a picture; the default AgX film curve bends it.
        view.view_transform, view.look = "Standard", "None"
        view.exposure, view.gamma = 0.0, 1.0
    # 8-bit radiance is not radiance; `render` maps the ir png from the exr.
    file_format, depth = _FORMATS[outputs.format if band == "eo" else "exr"]
    sc.render.image_settings.file_format = file_format
    sc.render.image_settings.color_depth = depth


def build(scenario: Scenario, band: Band = "eo") -> None:
    """Replace the current Blender session's contents with `scenario` in one band.

    A scene is EO or LWIR, never both: the two describe different physics and share no
    units. Rendering both bands means building twice, which costs seconds.
    """
    if not any(mount.camera.kind == band for mount in scenario.rig.mounts):
        raise ValueError(f"the rig has no {band} camera to build a {band} scene for")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _output(scenario.outputs, band)
    bpy.context.scene.world = _sky(scenario.sky, band)
    reach_m = sea_reach_m(scenario.rig, scenario.sea)
    _sea(scenario.sea, scenario.seed, reach_m, band)
    # The sea's far corner is reach * sqrt(2) away, so the clip plane has to clear it.
    cameras = _rig(scenario.rig, 1.5 * reach_m)
    if scenario.ownship is not None:
        # At the origin, bow to +Y: the rig's offsets are in that frame. Named for
        # its role, or a target on the same asset takes the name by build order.
        _vessel(
            scenario.ownship.asset, scenario.ownship.t_k, band, scenario.sky
        ).name = "ownship"
    radius_m = earth_radius_m(scenario.sea.refraction_k)
    for spec in scenario.objects:
        _object(spec, band, radius_m, scenario.sky)
    if scenario.targets is not None:
        _targets(scenario.targets, band, radius_m, scenario.sky)
    # Scenario order, so the first camera is EO in the baseline: an IR build would
    # otherwise open on a camera whose optics belong to the other band.
    first = next(mount for mount in scenario.rig.mounts if mount.camera.kind == band)
    sc = bpy.context.scene
    sc.camera = cameras[first.name]
    sc.render.resolution_x, sc.render.resolution_y = (
        first.camera.width_px,
        first.camera.height_px,
    )
    # Until the depsgraph runs, every child still reports its pre-parenting
    # matrix_world, so anything measuring the scene reads the wrong place.
    bpy.context.view_layer.update()
