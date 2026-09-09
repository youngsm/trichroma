"""Viewer-ready versions of the repository's detector configurations.

These helpers do not alter the original configuration files. The main wire viewer
uses the same analytic cylinders as optical transport; an explicit meshed
comparison is also available. The main pixel view preserves the config's explicit
area-averaged pixel-surface approximation.
"""

from dataclasses import dataclass

import numpy as np

from chroma.triton.viewer import Camera


@dataclass
class ViewerExample:
    geometry: object
    metadata: dict
    camera: Camera
    solid_colors: dict
    boundary_layers: tuple = ()

    def viewer(self, **kwargs):
        from chroma.triton.viewer import DetectorViewer

        result = DetectorViewer(
            self.geometry,
            solid_colors=self.solid_colors,
            boundary_layers=self.boundary_layers,
            **kwargs
        )
        result.camera = self.camera
        return result


def build_viewer_example(name="theia", *, radius=25500.0, coverage=0.81, diameter=508.0):
    """Return geometry, provenance, display colors, and a useful interior camera.

    Names: ``theia``, ``reflect3wires``, ``reflect3wires-mesh``, ``pixelTPC``, or ``pixelPads`` (a 32x32
    resolved-pad patch; this closeup is not a complete detector).
    """
    if name == "theia":
        from chroma.triton.examples.theia import build_theia

        fixture = build_theia(size=radius, coverage=coverage, diameter=diameter)
        geometry = fixture.detector()
        metadata = dict(
            name=name,
            radius_mm=radius,
            diameter_mm=diameter,
            coverage=coverage,
            sensor_instances=fixture.channel_count,
            canonical_sensor_triangles=len(fixture.sensor.mesh.triangles),
            enclosure_triangles=len(fixture.enclosure.triangles),
            representation="instanced PMT meshes and faceted enclosure",
        )
        colors = {i + 1: 0xFFCC923C for i in range(fixture.channel_count)}
        colors[0] = 0xFFAAC3CE
        camera = Camera.orbit((0.0, 0.0, 0.0), (radius + 2000.0) * 0.65)
        return ViewerExample(geometry, metadata, camera, colors)
    if name == "pixelPads":
        return _pixel_pad_closeup()
    from chroma_lar.geometry.config_loader import load_config_from_file, build_detector_from_dict

    names = {
        "reflect3wires": "detector_config_reflect_reflect3wires",
        "reflect3wires-mesh": "detector_config_reflect_reflect3wires",
        "pixelTPC": "detector_config_pixel",
    }
    if name not in names:
        raise ValueError("unknown viewer example: " + str(name))
    config = load_config_from_file(names[name])
    config["flatten"] = False
    if name.startswith("reflect3wires"):
        config["analytic_wires"] = name == "reflect3wires"
    geometry = build_detector_from_dict(config)
    channels = geometry.channel_index_to_solid_id
    colors = {int(solid): 0xFFCC923C for solid in channels}
    metadata = dict(
        name=name,
        config=names[name],
        sensor_instances=len(channels),
        solid_instances=len(geometry.solids),
        triangles=sum(len(s.mesh.triangles) for s in geometry.solids),
        active_dimensions_mm=config["active_dimensions"],
        diameter_mm=config["pmt_diameter_in"] * 25.4,
    )
    layers = ()
    if name == "reflect3wires":
        from .triton_scene.compiler import compile_reflect3wires_scene
        from .viewer_analytic import AnalyticWireLayer

        scene = compile_reflect3wires_scene(retain_all_geometry=True)
        layer = AnalyticWireLayer(scene)
        layer.validate_geometry(geometry)
        layers = (layer,)
        metadata.update(
            representation="original six analytic periodic-cylinder wire planes",
            wire_planes=scene.wires.count,
            wire_cylinders=int(np.sum(scene.wires.kmax - scene.wires.kmin + 1)),
            wire_diameter_mm=config["wire_diameter"],
            wire_pitch_mm=config["wire_pitch"],
            analytic_wires=True,
        )
        camera = Camera.orbit((-2160.0, 0.0, 0.0), 1350.0, azimuth=0.0, elevation=10.0)
    elif name == "reflect3wires-mesh":
        # Builder order: cavity, PMTs, active box, cathode, then all wire solids.
        first_wire = len(channels) + 3
        wires = geometry.solids[first_wire:]
        if not wires or len(getattr(geometry, "wireplanes", ())):
            raise RuntimeError("wire viewer did not receive the expected fully meshed wires")
        metadata.update(
            representation="original fully meshed wires; analytic_wires=False",
            wire_planes=2 * len(config["wire_angles"]),
            wire_solids=len(wires),
            wire_triangles=sum(len(s.mesh.triangles) for s in wires),
            wire_diameter_mm=config["wire_diameter"],
            wire_pitch_mm=config["wire_pitch"],
            wire_nsteps=config["wire_nsteps"],
        )
        camera = Camera.orbit((-2160.0, 0.0, 0.0), 1350.0, azimuth=0.0, elevation=10.0)
    else:
        metadata.update(
            representation="original pixel_simplified=True; area-averaged pixel faces",
            pixel_simplified=bool(config["pixel_simplified"]),
            pads_per_face=config["n_pixels_y"] * config["n_pixels_z"],
            resolved_pad_triangles=False,
            pixel_pitch_mm=config["pixel_pitch"],
        )
        camera = Camera.orbit((-1100.0, 2160.0, 0.0), 3300.0, azimuth=-90.0, elevation=10.0)
    return ViewerExample(geometry, metadata, camera, colors, layers)


def _pixel_pad_closeup():
    from chroma.geometry import Geometry, Mesh, Solid
    from chroma_lar.geometry.pixelplane import make_pixel_face
    from chroma_lar.config.detector_config_pixel import get_config

    config = get_config()
    pitch, cells = config["pixel_pitch"], 32
    half = cells * pitch / 2
    vertices, triangles, pad, border = make_pixel_face(
        0.0,
        (-half, half),
        (-half, half),
        cells,
        cells,
        pitch,
        config["pixel_pad_size"],
        config["pixel_chamfer_radius"],
    )
    colors = np.where(pad, 0xFFFFD700, 0xFF2E8B57).astype(np.uint32)
    colors[border] = 0xFFA0A0A0
    geometry = Geometry()
    geometry.add_solid(Solid(Mesh(vertices, triangles), None, None, color=colors))
    metadata = dict(
        name="pixelPads",
        representation="resolved 32x32-pad closeup only",
        complete_detector=False,
        sensor_instances=0,
        pads=cells * cells,
        triangles=len(triangles),
        pixel_pitch_mm=pitch,
        pixel_pad_size_mm=config["pixel_pad_size"],
        pixel_chamfer_radius_mm=config["pixel_chamfer_radius"],
    )
    camera = Camera.orbit((0.0, 0.0, 0.0), 140.0, azimuth=0.0, elevation=20.0)
    return ViewerExample(geometry, metadata, camera, {})
