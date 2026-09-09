"""Viewer-ready versions of the repository's detector configurations.

These helpers do not alter the original configuration files. The main wire viewer
uses the same analytic cylinders as optical transport; an explicit meshed
comparison is also available. The pixelTPC helper preserves the configuration's
area-averaged pixel surface; pixelTPC-resolved replaces both faces with explicit
pads for the complete browser detector.
"""

from dataclasses import dataclass, field

import numpy as np

from chroma.triton.viewer import Camera


@dataclass
class ViewerExample:
    geometry: object
    metadata: dict
    camera: Camera
    solid_colors: dict
    boundary_layers: tuple = ()
    views: dict = field(default_factory=dict)
    export_groups: object = None

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

    Names: ``theia``, ``reflect3wires``, ``reflect3wires-mesh``, ``pixelTPC``,
    ``pixelTPC-resolved`` (complete detector with two million explicit pads),
    or ``pixelPads`` (legacy 32x32 pad patch).
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
    if name == "pixelTPC-resolved":
        return _resolved_pixel_detector()
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
    example = ViewerExample(geometry, metadata, camera, colors, layers)
    if name == "reflect3wires-mesh":
        example.export_groups = lambda: _wire_export_groups(geometry, colors, config)
        example.views["Wire planes"] = Camera.orbit(
            (-2160.0, 0.0, 0.0), 28.0, azimuth=0.0, elevation=10.0
        )
    return example


def _wire_export_groups(geometry, colors, config):
    """Group parallel wire meshes with oriented bounds and untouched leaf vertices."""
    from chroma.triton.viewer import _geometry_groups, _MeshGroup

    first_wire = len(geometry.channel_index_to_solid_id) + 3
    groups = _geometry_groups(geometry, solid_colors=colors,
                              hidden_solids=range(first_wire, len(geometry.solids)))
    for angle, color in zip(config["wire_angles"], (0xFF, 0xFF00, 0xFF0000)):
        meshes, locations = {}, set()
        for i in range(first_wire, len(geometry.solids)):
            solid = geometry.solids[i]
            if np.all(solid.color == color):
                if not np.array_equal(geometry.solid_rotations[i], np.eye(3)):
                    raise ValueError("wire builder must supply its original baked mesh positions")
                meshes[id(solid.mesh)] = solid.mesh
                locations.add(tuple(geometry.solid_displacements[i]))
        if len(locations) != 2 or not meshes:
            raise ValueError("expected matching wire meshes on both drift faces")
        vertices, triangles, offset = [], [], 0
        for mesh in meshes.values():
            vertices.append(mesh.vertices)
            triangles.append(mesh.triangles + offset)
            offset += len(mesh.vertices)
        vertices, triangles = np.concatenate(vertices), np.concatenate(triangles)
        c, s = np.cos(angle), np.sin(angle)
        bounds_rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)
        groups.append(_MeshGroup(vertices, triangles, np.full(len(triangles), color, np.uint32),
                                  np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
                                  np.array(sorted(locations), np.float32), bounds_rotation))
    return groups


def _resolved_pixel_detector(tile_cells=25):
    """Keep the complete detector and instance every configured pad on both faces.

    Tiles contain the same chamfered pad and FR-4 triangles as make_pixel_face.
    Only their storage is shared; there is no distance-dependent substitution or
    pad averaging. The production detector's optical configuration is unchanged.
    """
    from chroma.geometry import Mesh, Solid
    from chroma_lar.geometry.pixelplane import make_pixel_face
    from chroma_lar.config.detector_config_pixel import get_config

    example = build_viewer_example("pixelTPC")
    config = get_config()
    geometry = example.geometry
    pitch = config["pixel_pitch"]
    ny, nz = config["n_pixels_y"], config["n_pixels_z"]
    if tile_cells <= 0 or ny % tile_cells or nz % tile_cells:
        raise ValueError("tile size must divide both configured pixel counts")
    # Remove only the two averaged pixel rectangles, retaining all steel borders,
    # the other active-volume faces, cavity, cathode, and every PMT.
    found = []
    for index, solid in enumerate(geometry.solids):
        mask = np.array([getattr(s, "name", None) == "averaged_pixel" for s in solid.surface])
        if mask.any():
            found.append(index)
            keep = ~mask
            geometry.solids[index] = Solid(
                Mesh(solid.mesh.vertices, solid.mesh.triangles[keep], round=False),
                solid.material1[keep], solid.material2[keep], solid.surface[keep], solid.color[keep],
            )
    if len(found) != 1:
        raise RuntimeError("expected one active box with averaged pixel faces")
    half_x = np.ptp(config["active_dimensions"]["x"]) / 2
    half_tile = tile_cells * pitch / 2
    target = config["target_material"]
    for sign in (-1, 1):
        vertices, triangles, pad, border = make_pixel_face(
            0.0, (-half_tile, half_tile), (-half_tile, half_tile),
            tile_cells, tile_cells, pitch, config["pixel_pad_size"],
            config["pixel_chamfer_radius"], face_normal=sign,
        )
        assert not border.any()
        surfaces = np.where(pad, config["pixel_surface"], config["pcb_surface"])
        colors = np.where(pad, 0xFFFFD700, 0xFF2E8B57).astype(np.uint32)
        tile = Solid(Mesh(vertices, triangles, round=False), target,
                     config["default_optics"].vacuum, surfaces, colors)
        for iy in range(0, ny, tile_cells):
            for iz in range(0, nz, tile_cells):
                geometry.add_solid(tile, displacement=(
                    sign * half_x,
                    (iy + tile_cells / 2 - ny / 2) * pitch,
                    (iz + tile_cells / 2 - nz / 2) * pitch,
                ))
    example.metadata.update(
        name="pixelTPC-resolved", complete_detector=True, pixel_simplified=False,
        representation="complete detector with all chamfered pixel pads and FR-4 in shared mesh tiles",
        resolved_pad_triangles=True, pads_per_face=ny*nz, pads=2*ny*nz,
        pixel_pad_size_mm=config["pixel_pad_size"],
        pixel_chamfer_radius_mm=config["pixel_chamfer_radius"],
        tile_cells=tile_cells, tile_instances=2*(ny//tile_cells)*(nz//tile_cells),
        solid_instances=len(geometry.solids),
        triangles=sum(len(s.mesh.triangles) for s in geometry.solids),
    )
    example.views["Pixel pads"] = Camera.orbit(
        (-half_x, 0.0, 0.0), 140.0, azimuth=0.0, elevation=20.0
    )
    return example


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
