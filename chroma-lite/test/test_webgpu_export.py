"""Host-side WebGPU export invariants; these tests never initialize CUDA."""

from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from chroma.geometry import Geometry, Solid
from chroma.make import box
from chroma.triton.viewer import Camera
from chroma.triton.webgpu.export import (
    bvh_escape_links,
    export_example,
    pack_groups,
    prepare_groups,
)


def test_threaded_bvh_visits_original_leaves_once():
    groups = prepare_groups(box(2, 3, 4))
    group = groups[0]
    visited, leaves, index = [], [], 0
    while index != 0xFFFFFFFF:
        visited.append(index)
        count, child = (
            int(group.bvh.nodes[index, 3]) >> 28,
            int(group.bvh.nodes[index, 3]) & 0x0FFFFFFF,
        )
        if count:
            index = child
        else:
            leaves.append(child)
            index = int(group.escape[index])
    assert len(visited) == len(set(visited)) == group.bvh.node_count
    assert sorted(leaves) == list(range(group.bvh.triangle_count))
    broken = group.bvh.nodes.copy()
    broken[0, 3] = 1 << 28
    with pytest.raises(ValueError, match="cycle"):
        bvh_escape_links(broken)


def test_shared_mesh_export_keeps_transforms_and_leaf_instances(tmp_path):
    geometry = Geometry()
    mesh = box(2, 3, 4)
    for index in range(7):
        geometry.add_solid(
            Solid(mesh, None, None, color=0xFFC08040), displacement=(index * 10, 2, 3)
        )
    example = SimpleNamespace(
        geometry=geometry,
        solid_colors=None,
        boundary_layers=(),
        camera=Camera((20, 20, 20)),
        metadata={"name": "boxes"},
    )
    manifest, groups = export_example(example, tmp_path)
    assert len(groups) == 1 and manifest["mesh_triangles"] == len(mesh.triangles)
    assert manifest["instances"] == 7
    group = groups[0]
    leaf_ids = group.tlas[group.tlas[:, 3] & 0x80000000 != 0, 3] & 0x7FFFFFFF
    np.testing.assert_array_equal(leaf_ids, np.arange(7))
    np.testing.assert_array_equal(group.transforms[:, 9], np.arange(7) * 10)
    data = (tmp_path / "boxes.bin").read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["sha256"]
    assert json.loads((tmp_path / "boxes.json").read_text()) == manifest
    words = pack_groups(groups)
    np.testing.assert_array_equal(words[:4], [0x54524957, 1, 1, len(words)])
    import gzip
    compressed, _ = export_example(example, tmp_path, name="boxes-compressed", compress=True)
    downloaded = (tmp_path / compressed["binary"]).read_bytes()
    assert len(downloaded) == compressed["download_byte_length"]
    assert gzip.decompress(downloaded) == data
    assert compressed["sha256"] == manifest["sha256"]


def test_nonrigid_transforms_are_baked_and_unhandled_wires_rejected(tmp_path):
    geometry = Geometry()
    solid = Solid(box(1, 1, 1), None, None)
    for index in range(4):
        geometry.add_solid(
            solid, rotation=np.diag([2.0, 1.0, 1.0]), displacement=(index * 10, 0, 0)
        )
    groups = prepare_groups(geometry)
    assert len(groups) == 1 and len(groups[0].transforms) == 1
    assert groups[0].bvh.triangle_count == 4 * len(solid.mesh.triangles)
    example = SimpleNamespace(geometry=geometry, boundary_layers=(object(),))
    with pytest.raises(ValueError, match="no wires are omitted"):
        export_example(example, tmp_path / "unwritten")
    assert not (tmp_path / "unwritten").exists()


def test_oriented_bounds_preserve_original_triangle_coordinates():
    from chroma.triton.viewer import _MeshGroup

    mesh = box(1, 1000, 0.1)
    angle = .63
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]], np.float32)
    vertices = np.asarray(mesh.vertices @ rotation.T, np.float32)
    group = _MeshGroup(vertices, mesh.triangles, np.full(len(mesh.triangles), 0xFFFFFFFF, np.uint32),
                       np.eye(3, dtype=np.float32)[None], np.zeros((1, 3), np.float32), rotation)
    prepared = prepare_groups(None, mesh_groups=[group])
    np.testing.assert_array_equal(prepared[0].bvh.triangle_vertices, vertices[mesh.triangles])
    packed = pack_groups(prepared)
    assert packed[1] == 2
    matrix_offset = packed[4 + 14]
    np.testing.assert_array_equal(packed[matrix_offset:matrix_offset+9].view(np.float32).reshape(3, 3), rotation)


def test_portable_browser_pages_include_their_local_styles_and_modules(tmp_path):
    """A styled notebook does not prove its separately exported HTML is complete."""
    from html.parser import HTMLParser
    from chroma.triton.webgpu.export import copy_browser_assets

    class Dependencies(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths = []

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if tag == "script" and "src" in values:
                self.paths.append(values["src"])
            if tag == "link" and values.get("rel") == "stylesheet":
                self.paths.append(values["href"])

    copy_browser_assets(tmp_path, [])
    for page in ("index.html", "physics.html", "camera.html"):
        parser = Dependencies()
        parser.feed((tmp_path / page).read_text())
        assert {"theme.css", "fonts.css"}.issubset(parser.paths)
        assert all((tmp_path / path).is_file() for path in parser.paths)
    assert "data:font/otf;base64," in (tmp_path / "fonts.css").read_text()
