"""Scene exports and independent estimators for the browser photon camera.

Beam and ceiling photons use importance weights for their sampled source
mixture. Energy is expressed relative to one 450nm photon, so a surviving
packet contributes source_scale*450/wavelength, normalized by the total N.
The finite photon maps estimate radiance in relative energy
units per square millimetre and steradian; exposure maps that estimate to a
display. The camera does not draw trajectories.
"""

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .optical_showcase import SCENES, PlaygroundScene, build_playground_scene
from .webgpu_physics import physics_manifest

XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)
BEAM_PROBABILITY = 0.8
FILL_POWER = 0.00003 * 180 * 80 * np.pi
CAMERA_SCENES = (*SCENES, "pmt")


def camera_source_is_fill(count, seed=901):
    from chroma.triton.optical_response import uniform

    return uniform(np.arange(count, dtype=np.uint64), seed, 0x10000008) >= np.float32(
        BEAM_PROBABILITY
    )


def camera_source_scales(count, seed, source_wavelengths):
    """Photon importance weights for stable-ID beam/ceiling source sampling."""
    return np.where(
        camera_source_is_fill(count, seed),
        FILL_POWER / (1 - BEAM_PROBABILITY) * np.asarray(source_wavelengths) / 450,
        1 / BEAM_PROBABILITY,
    )


def camera_source_energy_sum(name, count, seed=901):
    """Expected event energy before interactions, evaluated directly from IDs."""
    from chroma.triton.optical_response import uniform

    total = 0.0
    for start in range(0, count, 1_000_000):
        ids = np.arange(start, min(count, start + 1_000_000), dtype=np.uint64)
        fill = uniform(ids, seed, 0x10000008) >= np.float32(BEAM_PROBABILITY)
        low, high = {"fluorescence": (300.0, 310.0), "pmt": (126.0, 130.0)}.get(
            name, (390.0, 710.0)
        )
        wavelength = low + (high - low) * uniform(ids, seed, 0x10000003)
        energy = np.where(
            fill,
            FILL_POWER / (1 - BEAM_PROBABILITY),
            450 / wavelength.astype(float) / BEAM_PROBABILITY,
        )
        total += energy.sum(dtype=np.float64)
    return float(total)


@dataclass(frozen=True)
class CameraScene(PlaygroundScene):
    beam_width: float = 2.0

    def photons(self, count, seed=901, polarization="random"):
        from chroma.triton.optical_response import uniform
        from chroma.triton.spectral import _hemisphere, _polarization

        batch = super().photons(count, seed, polarization)
        position = batch.pos.copy()
        for axis in (1, 2):
            position[:, axis] = (
                self.source_center[axis]
                + (uniform(batch.global_photon_ids, seed, 0x10000000 + axis - 1) - 0.5)
                * self.beam_width
            )
        ids = batch.global_photon_ids
        fill = camera_source_is_fill(count, seed)
        selected = ids[fill]
        position[fill, 0] = -30 + (uniform(selected, seed, 0x10000004) - 0.5) * 180
        position[fill, 1] = 60 + (uniform(selected, seed, 0x10000005) - 0.5) * 80
        position[fill, 2] = np.float32(119.999)
        direction, pol, wavelength = (
            batch.direction.copy(),
            batch.polarization.copy(),
            batch.wavelengths.copy(),
        )
        normal = np.tile(np.array([0, 0, -1], np.float32), (len(selected), 1))
        direction[fill] = _hemisphere(
            normal,
            uniform(selected, seed, 0x10000006),
            uniform(selected, seed, 0x10000007),
            lambert=True,
        )
        pol[fill] = _polarization(direction[fill], uniform(selected, seed, 0x10000002))
        wavelength[fill] = 390 + 320 * uniform(selected, seed, 0x10000003)
        # Native spectral transport validates unit statistical weights. Source
        # importance weights affect map deposition, not sampled trajectories;
        # camera_source_scales supplies them to the independent observer.
        return replace(
            batch,
            pos=position,
            direction=direction,
            polarization=pol,
            wavelengths=wavelength,
        )


def camera_color_table(count=64):
    """Bin-averaged native CIE1964 matching functions; UV has no fake violet."""
    import chroma.color

    source = Path(chroma.color.__file__).parent / "ciexyz64_1.csv"
    table = np.loadtxt(source, delimiter=",")
    edges = np.linspace(280, 740, count + 1)
    xyz = []
    for low, high in zip(edges[:-1], edges[1:]):
        wavelength = np.linspace(low, high, 65)
        matching = np.stack(
            [
                np.interp(wavelength, table[:, 0], table[:, axis], left=0, right=0)
                for axis in (1, 2, 3)
            ],
            axis=1,
        )
        xyz.append(np.trapezoid(matching, wavelength, axis=0) / (high - low))
    xyz = np.asarray(xyz)
    return dict(
        edges=edges.tolist(),
        centers=((edges[:-1] + edges[1:]) / 2).tolist(),
        xyz=xyz.tolist(),
        linear_srgb=(xyz @ XYZ_TO_LINEAR_SRGB.T).tolist(),
        observer="CIE1964 10-degree matching functions from Chroma's bundled table",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        negative_components="Retain until spectral integration, then gamut clip for display",
        units="relative radiant packet energy; no per-spectrum or per-frame normalization",
    )


def build_camera_scene(name):
    """Independent camera variants; diagnostic fixtures stay unchanged.

    A weak Rayleigh haze makes the prism and fluorescent beams visible from
    the side through actual scattering. Original diagnostic scenes retain
    their infinite inactive lengths.
    """
    if name == "pmt":
        from .pmt_camera import build_pmt_camera_scene

        return build_pmt_camera_scene()
    fixture = build_playground_scene(name)
    if name != "rayleigh":
        for material in fixture.geometry.unique_materials:
            if material.name == "synthetic_air":
                material.set(
                    "scattering_length",
                    10_000 * (fixture.wavelengths / 450) ** 4,
                    fixture.wavelengths,
                )
    if name in ("prism", "fluorescence"):
        from chroma.detector import Detector
        from chroma.geometry import Material, Solid

        air = next(
            material
            for material in fixture.geometry.unique_materials
            if material.name == "synthetic_air"
        )
        geometry = Detector(air)
        geometry.add_pmt(fixture.geometry.solids[0], displacement=(0, 0, 0))
        solid = fixture.geometry.solids[1]
        if name == "fluorescence":
            glass = Material("synthetic_fluorescent_camera_glass")
            glass.set("refractive_index", 1.5, fixture.wavelengths)
            glass.set("absorption_length", 60.0, fixture.wavelengths)
            glass.set("scattering_length", np.inf, fixture.wavelengths)
            solid = Solid(solid.mesh, glass, air, surface=solid.surface, color=solid.color)
        geometry.add_solid(solid, displacement=(0, 0, -58))
        geometry.flatten()
        fixture = replace(
            fixture, geometry=geometry, source_center=(*fixture.source_center[:2], -58.0)
        )
    updated = replace(
        fixture,
        explanation=fixture.explanation
        + (
            " · weak wavelength-dependent Rayleigh haze for the camera"
            if name != "rayleigh"
            else ""
        ),
    )
    return CameraScene(**updated.__dict__, beam_width=20.0 if name == "fluorescence" else 2.0)


def camera_manifest(name):
    fixture = build_camera_scene(name)
    manifest, _, scene = physics_manifest(name, fixture=fixture)
    manifest["max_steps"] = 8192
    manifest["source"]["beam_width"] = fixture.beam_width
    colors = camera_color_table()
    half = np.array([300.0, 200.0, 120.0])
    shape = np.array([64, 40, 24])
    manifest["camera"] = dict(
        version=1,
        estimator="forward photon density maps with a spectral camera gather",
        wavelength_bins=len(colors["centers"]),
        colors=colors,
        default_eye=[-240, -170, 95],
        default_target=[30, 0, -40 if name != "rayleigh" else 0],
        field_of_view_degrees=55,
        room_half_extent=half.tolist(),
        face_order=["+x", "-x", "+y", "-y", "+z", "-z"],
        face_uv_axes=[[1, 2], [1, 2], [0, 2], [0, 2], [0, 1], [0, 1]],
        wall_shape=[128, 128],
        fill_wall_shape=[32, 32],
        wall_reflectance=0.65,
        volume_shape=shape.tolist(),
        voxel_volume_mm3=float(np.prod(2 * half / shape)),
        fluorescence_shape=[64, 64],
        fluorescence_half_extent=[2.5, 90, 60],
        fluorescence_center=[0, 0, -58],
        object_translation=[0, 0, -58] if name != "rayleigh" else [0, 0, 0],
        relative_photon_energy_reference_nm=450,
        source_mix=dict(
            beam_probability=BEAM_PROBABILITY,
            choice_stream=0x10000008,
            fill_x_stream=0x10000004,
            fill_y_stream=0x10000005,
            fill_cosine_stream=0x10000006,
            fill_azimuth_stream=0x10000007,
            fill_origin_z=119.999,
            weighting="beam1/p; fill L*A*pi/(1-p) radiant energy per packet",
        ),
        fill_light=dict(
            center=[-30, 60, 120],
            full_size=[180, 80],
            normal=[0, 0, -1],
            tangent_u=[1, 0, 0],
            tangent_v=[0, 1, 0],
            integrated_radiance=0.00003,
            radiance_per_bin=(
                0.00003
                * np.maximum(
                    0,
                    np.minimum(np.asarray(colors["edges"])[1:], 710)
                    - np.maximum(np.asarray(colors["edges"])[:-1], 390),
                )
                / 320
            ).tolist(),
            scope="Ceiling packets undergo the same full scattering, refraction and absorption transport as beam packets",
        ),
        normalize_by="all emitted photons N, never detected photon count or histogram maximum",
        wall_radiance="sum(source_scale*450/lambda)/(N*cell_area_mm2) * reflectance/pi",
        volume_source="3/(8*pi*N*voxel_volume) * (trace(M)-omega^T*M*omega)",
        polarization_moments=["xx", "yy", "zz", "xy", "xz", "yz"],
        fluorescence_radiance="per-outgoing-hemisphere sum(source_scale*450/lambda)/(2*pi*N*area*abs(cos(theta)))",
        scattering_length_at_bin_centers_mm=(
            (100 if name == "rayleigh" else 10_000) * (np.asarray(colors["centers"]) / 450) ** 4
        ).tolist(),
        scope=[
            "steady-state illumination",
            "all emitted photons deposit into maps",
            "first diffuse wall reflection; no later diffuse wall interreflection",
            "finite spatial and wavelength bins introduce density-estimation bias",
            "camera visibility and dielectric refraction remain wavelength dependent",
            "RGB display uses a finite spectral approximation and exposure, not absolute photometric calibration",
        ],
    )
    manifest["display"].update(
        radiance=True, ultraviolet="invisible below the bundled visible matching functions"
    )
    if name == "pmt":
        from .pmt_camera import pmt_camera_metadata
        from chroma.triton.webgpu.export import bvh_escape_links

        manifest["camera"]["pmt"] = pmt_camera_metadata(scene)
        manifest["camera"]["geometry_bvh"] = dict(
            nodes=scene.bvh.nodes.tolist(),
            escape_links=bvh_escape_links(scene.bvh.nodes).tolist(),
            world_origin=scene.bvh.world_origin.tolist(),
            world_scale=float(scene.bvh.world_scale),
            node_count=scene.bvh.node_count,
            layout="Chroma uint32[N,4]; xyz pack lower16/upper16; w packs child_count:4 and first_child_or_triangle:28",
            bounds_padding_quantization_units=1,
            triangle_ids="original exported triangle indices, unchanged by Morton ordering",
        )
        manifest["camera"]["default_eye"] = [-30, -185, 45]
        manifest["camera"]["default_target"] = [100, 0, -15]
        manifest["camera"]["scattering_length_at_bin_centers_mm"] = [950.0] * len(colors["centers"])
        manifest["camera"]["object_translation"] = [20, 0, -15]
    return manifest, fixture, scene


def export_camera_scene(name, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest, fixture, scene = camera_manifest(name)
    payload = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    filename = f"camera-{name}.json"
    (destination / filename).write_bytes(payload)
    return (
        dict(
            name=name,
            title=manifest["title"],
            manifest=filename,
            sha256=hashlib.sha256(payload).hexdigest(),
            byte_length=len(payload),
            scene_fingerprint=scene.fingerprint,
        ),
        fixture,
        scene,
    )


def export_camera_catalog(destination):
    destination = Path(destination)
    entries = [export_camera_scene(name, destination)[0] for name in CAMERA_SCENES]
    catalog = dict(format="trichroma-webgpu-photon-camera-v1", scenes=entries)
    (destination / "camera-catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog


def export_camera_bundle(destination):
    """Write a self-contained camera bundle, including the diagnostic optics page."""
    from chroma.triton.webgpu import export as webgpu_export
    from .webgpu_physics import export_physics_catalog
    import re

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    assets = Path(webgpu_export.__file__).with_name("assets")
    for name in (
        "camera.html",
        "camera.js",
        "camera.wgsl",
        "deposition.wgsl",
        "physics.html",
        "physics.js",
        "physics.wgsl",
        "theme.css",
        "fonts.css",
    ):
        shutil.copyfile(assets / name, destination / name)
    # This focused bundle contains optical scenes, without the detector meshes
    # supplied by the larger viewer exporter. Avoid navigation to absent pages.
    for name in ("camera.html", "physics.html"):
        path = destination / name
        page = path.read_text().replace(
            'class="brand" href="index.html"', 'class="brand" href="camera.html"'
        )
        path.write_text(
            re.sub(r'<a\b[^>]*href="index.html"[^>]*>.*?</a>', "", page, flags=re.DOTALL)
        )
    export_physics_catalog(destination)
    return export_camera_catalog(destination)


def notebook_camera_html(destination, *, height=1000, manual=False, software=False):
    """Embed exact bundle bytes without Jupyter's sandboxed /files HTML route.

    The notebook must be trusted, as with other executable notebook output.
    Scene tables and shaders stay byte-identical to the portable bundle;
    module imports use data URLs and fetch resolves only bundled resources.
    No notebook-server security setting or external rendering server is used.
    """
    import base64
    import gzip
    from html import escape
    import re

    destination = Path(destination)
    query = "?" + "&".join(
        value for enabled, value in ((manual, "manual=1"), (software, "fallback=1")) if enabled
    )
    encode = lambda value: base64.b64encode(value).decode("ascii")
    files = {
        path.name: encode(path.read_bytes())
        for path in destination.iterdir()
        if path.suffix in (".json", ".wgsl")
    }
    source = (destination / "physics.js").read_text().replace("location.search", json.dumps(query))
    physics_url = "data:text/javascript;base64," + encode(source.encode())
    camera = (destination / "camera.js").read_text().replace("location.search", json.dumps(query))
    camera, imports = re.subn(
        r"from\s+['\"]\./physics\.js['\"]", lambda _: "from " + json.dumps(physics_url), camera
    )
    if imports != 1:
        raise ValueError("Could not resolve the camera module import")
    bootstrap = """<script>
    const cameraBundle=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('%s'),c=>c.charCodeAt(0))));
    window.fetch=async resource=>{
        const name=new URL(typeof resource==='string'?resource:resource.url,'https://trichroma.invalid/').pathname.split('/').pop();
        if(!Object.hasOwn(cameraBundle,name))throw Error('Resource is not in the camera bundle: '+name);
        return new Response(Uint8Array.from(atob(cameraBundle[name]),c=>c.charCodeAt(0)),{headers:{'Content-Type':name.endsWith('.json')?'application/json':'text/plain'}});
    };
    </script>""" % encode(json.dumps(files).encode())
    page = (destination / "camera.html").read_text()
    page = page.replace('href="camera.html"', 'href="#"')
    for name in ("theme.css", "fonts.css"):
        page = re.sub(
            r'<link\b(?=[^>]*\bhref=["\']' + re.escape(name) + r'["\'])[^>]*>',
            lambda _, name=name: "<style>" + (destination / name).read_text() + "</style>",
            page,
        )
    page = re.sub(
        r'<a\b[^>]*href="physics.html"[^>]*>.*?</a>',
        '<a href="#camera-notes">Model details</a>',
        page,
        flags=re.DOTALL,
    )
    page = page.replace("<details>", '<details id="camera-notes">', 1)
    replacement = (
        bootstrap
        + '<script type="module" src="data:text/javascript;base64,'
        + encode(camera.encode())
        + '"></script>'
    )
    page, entries = re.subn(
        r'<script\b(?=[^>]*\bsrc=["\']camera\.js["\'])[^>]*>\s*</script\s*>',
        lambda _: replacement,
        page,
    )
    if entries != 1:
        raise ValueError("Camera HTML does not contain one module entry")
    # Avoid a multi-megabyte IOPub message once the PMT tables and licensed font
    # are included. Decompression restores the exact document and bundle bytes.
    compressed = encode(gzip.compress(page.encode(), mtime=0))
    loader = """<!doctype html><meta charset="utf-8"><script>
    (async()=>{
        const bytes=Uint8Array.from(atob('%s'),c=>c.charCodeAt(0));
        const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
        const html=await new Response(stream).text();
        document.open();document.write(html);document.close();
    })().catch(error=>{document.documentElement.textContent='Camera failed to load: '+error.message;});
    </script>""" % compressed
    return f'<iframe title="Spectral photon camera" style="width:100%;height:{height}px;border:0" srcdoc="{escape(loader, quote=True)}"></iframe>'


def packet_energy(wavelength_nm):
    wavelength = np.asarray(wavelength_nm, float)
    if np.any(~np.isfinite(wavelength) | (wavelength <= 0)):
        raise ValueError("wavelengths must be finite and positive")
    return 450 / wavelength


def wall_radiance(energy_sum, photons, area_mm2, reflectance=0.65):
    if photons <= 0 or area_mm2 <= 0 or not 0 <= reflectance <= 1:
        raise ValueError("positive photon count/area and reflectance in[0,1] required")
    return np.asarray(energy_sum) * reflectance / (photons * area_mm2 * np.pi)


def polarization_moments(polarization, energy):
    p = np.asarray(polarization, float)
    if p.shape[-1] != 3 or not np.allclose(np.linalg.norm(p, axis=-1), 1, atol=1e-5):
        raise ValueError("polarization must consist of unit3-vectors")
    return np.asarray(energy)[..., None, None] * p[..., :, None] * p[..., None, :]


def rayleigh_source(moment_sum, direction, photons, volume_mm3):
    """Scattering-collision estimator; sigma_s is already in collision density."""
    m, d = np.asarray(moment_sum, float), np.asarray(direction, float)
    if photons <= 0 or volume_mm3 <= 0 or not np.allclose(np.linalg.norm(d, axis=-1), 1):
        raise ValueError("positive count/volume and unit camera directions required")
    directional = np.einsum("...i,...ij,...j->...", d, m, d)
    density = np.maximum(0, np.trace(m, axis1=-2, axis2=-1) - directional)
    return 3 * density / (8 * np.pi * photons * volume_mm3)


def integrate_constant_segment(source_radiance_per_mm, extinction_per_mm, length_mm):
    """Exact attenuated integral for a piecewise-constant camera segment."""
    extinction, length = np.broadcast_arrays(
        np.asarray(extinction_per_mm, float), np.asarray(length_mm, float)
    )
    if np.any(extinction < 0) or np.any(length < 0):
        raise ValueError("extinction and length must be nonnegative")
    integral = np.divide(
        -np.expm1(-extinction * length),
        extinction,
        out=np.array(length, copy=True),
        where=extinction != 0,
    )
    return np.asarray(source_radiance_per_mm) * integral


def fluorescence_radiance(hemisphere_energy, photons, area_mm2, outgoing_cosine):
    """Uniform-hemisphere emission has radiance proportional to1/|cos(theta)|."""
    cosine = np.abs(np.asarray(outgoing_cosine, float))
    if photons <= 0 or area_mm2 <= 0 or np.any((cosine <= 0) | (cosine > 1)):
        raise ValueError("positive count/area and a nongrazing direction required")
    return np.asarray(hemisphere_energy) / (2 * np.pi * photons * area_mm2 * cosine)


def collect_camera_reference(
    name, count=256, *, seed=901, max_steps=8192, fixture=None, scene=None, compact_prefixes=True
):
    """Independent CPU collision records for small photon-map audit batches.

    Records use incoming polarization for scattering and the new wavelength
    for fluorescence. Prefix simulations always restart original photons/IDs.
    Completed photons are omitted from later prefix replays; every replay
    still starts from the original source state and retains its original ID.
    This observer is only for verification.
    """
    from chroma.event import SURFACE_DETECT, SURFACE_REEMIT, RAYLEIGH_SCATTER
    from chroma.triton.spectral import SpectralSimulation, STEP_LIMIT, _finish
    from chroma.triton.runtime import PhotonBatch
    from .webgpu_physics import TERMINAL

    if (fixture is None) != (scene is None):
        raise ValueError("supply both camera fixture and compiled scene, or neither")
    if fixture is None:
        _, fixture, scene = camera_manifest(name)
    batch = fixture.photons(count, seed=seed)
    source_scale = camera_source_scales(count, seed, batch.wavelengths)
    source_is_fill = camera_source_is_fill(count, seed)
    simulation = SpectralSimulation(scene, backend="reference")
    previous = dict(
        pos=batch.pos,
        polarization=batch.polarization,
        wavelengths=batch.wavelengths,
        flags=batch.flags,
    )
    records = dict(wall=[], volume=[], fluorescence=[], photocathode=[])
    cathode = None
    if name == "pmt":
        cathode = list(scene.host.optics.surfaces.names).index("perfect_pmt_photocathode")
    for limit in range(1, max_steps + 1):
        active_ids = np.flatnonzero((previous["flags"] & TERMINAL) == 0)
        replay = batch
        if compact_prefixes and limit > 1:
            replay = PhotonBatch(
                **{
                    field: getattr(batch, field)[active_ids]
                    for field in (
                        "pos",
                        "direction",
                        "polarization",
                        "wavelengths",
                        "times",
                        "last_hit_triangles",
                        "flags",
                        "weights",
                        "event_indices",
                        "global_photon_ids",
                        "channels",
                    )
                }
            )
        result = simulation.simulate(replay, seed=seed, max_steps=limit)
        if compact_prefixes and limit > 1:
            current = {field: value.copy() for field, value in previous.items()}
            for field, value in result.final_state.items():
                current[field][active_ids] = value
        else:
            current = result.final_state
        flags = current["flags"] & ~STEP_LIMIT
        active = (previous["flags"] & TERMINAL) == 0
        selections = dict(
            wall=active & ((flags & SURFACE_DETECT) != 0),
            volume=active
            & ((flags & RAYLEIGH_SCATTER) != 0)
            & ((flags & TERMINAL) == 0)
            & (current["last_hit"] < 0),
            fluorescence=active
            & ((flags & SURFACE_REEMIT) != 0)
            & ((previous["flags"] & SURFACE_REEMIT) == 0),
        )
        if cathode is not None:
            at_cathode = scene.host.surface_index[np.maximum(current["last_hit"], 0)] == cathode
            selections["photocathode"] = selections["wall"] & at_cathode
            selections["wall"] &= ~at_cathode
        for kind, selection in selections.items():
            for photon in np.flatnonzero(selection):
                triangle = int(current["last_hit"][photon])
                records[kind].append(
                    dict(
                        photon_id=int(photon),
                        step=limit,
                        position=current["pos"][photon].tolist(),
                        wavelength=float(current["wavelengths"][photon]),
                        energy=float(
                            packet_energy(current["wavelengths"][photon]) * source_scale[photon]
                        ),
                        source_is_fill=bool(source_is_fill[photon]),
                        incoming_polarization=previous["polarization"][photon].tolist(),
                        outgoing_direction=current["direction"][photon].tolist(),
                        triangle=triangle,
                        normal=None if triangle < 0 else scene.normals[triangle].tolist(),
                    )
                )
        previous = dict(current, flags=flags)
        if not result.step_limit_count:
            break
    else:
        raise RuntimeError("CPU photon-camera reference reached the step limit")
    result = _finish(current, limit, scene.fingerprint)
    return dict(
        count=count,
        seed=seed,
        scene_fingerprint=scene.fingerprint,
        records=records,
        result=result,
        fixture=fixture,
        scene=scene,
    )


def reference_camera_maps(oracle, manifest):
    """Sparse photon maps from independent CPU interactions, in WGSL layout."""
    from collections import defaultdict

    settings = manifest["camera"]
    edges = np.asarray(settings["colors"]["edges"])
    bins = settings["wavelength_bins"]
    half = np.asarray(settings["room_half_extent"])
    volume_shape = np.asarray(settings["volume_shape"])
    maps = {name: defaultdict(float) for name in ("wall", "fill_wall", "volume", "wls")}

    def axis_bin(position, extent, count):
        return int(
            np.clip(
                np.searchsorted(np.linspace(-extent, extent, count + 1), position, side="right")
                - 1,
                0,
                count - 1,
            )
        )

    for kind, records in oracle["records"].items():
        if kind == "photocathode":
            continue
        for record in records:
            position = np.asarray(record["position"])
            wavelength_bin = int(
                np.clip(np.searchsorted(edges, record["wavelength"], side="right") - 1, 0, bins - 1)
            )
            energy = record["energy"]
            if kind == "volume":
                x, y, z = [
                    axis_bin(position[axis], half[axis], volume_shape[axis]) for axis in range(3)
                ]
                base = (((z * 40 + y) * 64 + x) * bins + wavelength_bin) * 6
                moment = polarization_moments(record["incoming_polarization"], energy)
                for component, (a, b) in enumerate(
                    ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
                ):
                    maps["volume"][base + component] += float(moment[a, b])
                continue
            normal = np.asarray(record["normal"])
            axis = int(np.argmax(np.abs(normal)))
            face = 2 * axis + int(normal[axis] < 0)
            u_axis, v_axis = settings["face_uv_axes"][face]
            if kind == "wall":
                resolution = 32 if record["source_is_fill"] else 128
                u = axis_bin(position[u_axis], half[u_axis], resolution)
                v = axis_bin(position[v_axis], half[v_axis], resolution)
                index = ((face * resolution + v) * resolution + u) * bins + wavelength_bin
                maps["fill_wall" if record["source_is_fill"] else "wall"][index] += energy
            else:
                if "pmt" in settings:
                    chart = settings["pmt"]["triangle_chart"][record["triangle"]]
                    if chart < 0:
                        raise ValueError("Fluorescent collision has no PMT facet chart")
                    hemisphere = int(np.dot(normal, record["outgoing_direction"]) < 0)
                    index = (2 * chart + hemisphere) * bins + wavelength_bin
                    maps["wls"][index] += energy
                    continue
                extent = settings["fluorescence_half_extent"]
                position = position - settings["fluorescence_center"]
                u = axis_bin(position[u_axis], extent[u_axis], 64)
                v = axis_bin(position[v_axis], extent[v_axis], 64)
                hemisphere = int(np.dot(normal, record["outgoing_direction"]) < 0)
                index = (((face * 2 + hemisphere) * 64 + v) * 64 + u) * bins + wavelength_bin
                maps["wls"][index] += energy
    return {
        name: [[index, value] for index, value in sorted(values.items()) if value != 0]
        for name, values in maps.items()
    }


def wall_bin_boundary_oracle(browser_states, oracle, manifest, *, position_tolerance_mm=0.02):
    """Isolate discontinuous wall-bin choices from already checked flight errors.

    The independent transport oracle remains unchanged. Only a terminal wall
    collision that straddles an immediately adjacent spatial-bin edge may use
    the browser's recorded position for an independent deposition check.
    Wavelength, packet energy, source identity and all interior collision
    records still come from the CPU oracle. This is explicitly not exact
    CPU/GPU map parity: every changed bin and physical displacement is reported.
    """
    records = dict(oracle["records"], wall=list(oracle["records"]["wall"]))
    settings = manifest["camera"]
    bins = settings["wavelength_bins"]
    half = np.asarray(settings["room_half_extent"])
    ambiguities = []
    for slot, record in enumerate(records["wall"]):
        photon = record["photon_id"]
        old = np.asarray(record["position"])
        new = np.asarray(browser_states["pos"][photon])
        moved = dict(record, position=new.tolist())
        kind = "fill_wall" if record["source_is_fill"] else "wall"
        old_index = reference_camera_maps(dict(records={"wall": [record]}), manifest)[kind][0][0]
        new_index = reference_camera_maps(dict(records={"wall": [moved]}), manifest)[kind][0][0]
        if old_index == new_index:
            continue
        resolution = 32 if record["source_is_fill"] else 128
        normal = np.asarray(record["normal"])
        axis = int(np.argmax(np.abs(normal)))
        face = 2 * axis + int(normal[axis] < 0)
        uv_axes = settings["face_uv_axes"][face]
        old_cell, new_cell = old_index // bins, new_index // bins
        old_uv = (old_cell % resolution, old_cell // resolution % resolution)
        new_uv = (new_cell % resolution, new_cell // resolution % resolution)
        edges = []
        for coordinate, before, after in zip(uv_axes, old_uv, new_uv):
            if before != after:
                edge = -half[coordinate] + max(before, after) * 2 * half[coordinate] / resolution
                edges.append(dict(axis=coordinate, edge_mm=float(edge)))
        displacement = np.abs(new - old)
        valid = bool(
            np.isfinite(new).all()
            and displacement.max() <= position_tolerance_mm
            and browser_states["last_hit_triangles"][photon] == record["triangle"]
            and old_cell // resolution**2 == new_cell // resolution**2
            and all(abs(a - b) <= 1 for a, b in zip(old_uv, new_uv))
            and all(
                (old[e["axis"]] - e["edge_mm"]) * (new[e["axis"]] - e["edge_mm"]) <= 0
                for e in edges
            )
        )
        ambiguities.append(
            dict(
                photon_id=photon,
                map=kind,
                cpu_index=old_index,
                browser_position_index=new_index,
                cpu_position=old.tolist(),
                browser_position=new.tolist(),
                displacement_mm=displacement.tolist(),
                crossed_edges=edges,
                packet_energy=record["energy"],
                valid=valid,
            )
        )
        if valid:
            records["wall"][slot] = moved
    return dict(
        valid=all(row["valid"] for row in ambiguities),
        position_tolerance_mm=position_tolerance_mm,
        ambiguities=ambiguities,
        maps=reference_camera_maps(dict(oracle, records=records), manifest),
        scope="Independent deposition at recorded GPU wall positions only for reported adjacent-bin crossings; interior collision records and packet energies remain CPU-derived.",
    )


def compare_camera_maps(actual, expected, *, atol=0.0001, rtol=0.0001):
    """Report sparse-cell differences without dropping sign or empty cells."""
    comparisons = {}
    for kind in ("wall", "fill_wall", "volume", "wls"):
        a = {int(index): float(value) for index, value in actual[kind]}
        b = {int(index): float(value) for index, value in expected[kind]}
        ids = sorted(a.keys() | b.keys())
        av = np.array([a.get(index, 0) for index in ids])
        bv = np.array([b.get(index, 0) for index in ids])
        delta = np.abs(av - bv)
        bad = ~np.isfinite(delta) | (delta > atol + rtol * np.abs(bv))
        comparisons[kind] = dict(
            cells=len(ids),
            mismatches=int(np.count_nonzero(bad)),
            finite=bool(np.isfinite(av).all() and np.isfinite(bv).all()),
            max_abs=(
                float(delta.max())
                if len(delta) and np.isfinite(delta).all()
                else (None if len(delta) else 0)
            ),
            browser_sum=float(av.sum()) if np.isfinite(av).all() else None,
            cpu_sum=float(bv.sum()) if np.isfinite(bv).all() else None,
            disagreement_indices=np.asarray(ids)[bad][:32].tolist(),
        )
    return dict(
        passed=not any(row["mismatches"] for row in comparisons.values()),
        atol=atol,
        rtol=rtol,
        comparisons=comparisons,
    )
