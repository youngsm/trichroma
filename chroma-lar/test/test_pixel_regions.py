"""Pixel-TPC region discovery using the existing area-averaged configuration."""

from pathlib import Path

import numpy as np
import pytest

from chroma_lar.optical_calibration import OpticalCalibration
from chroma_lar.triton_scene.pixel_adapter import pixel_primitives
from chroma.triton.regions import compile_regions

CALIBRATION = (
    Path(__file__).parents[1]
    / "benchmarks/optical_validation/primitive_regions/pixel_synthetic_calibration.json"
)


@pytest.fixture(scope="module")
def pixel_geometry():
    calibration = OpticalCalibration.load(CALIBRATION)
    return calibration, calibration.build_detector("detector_config_pixel")


def test_pixel_tpc_discovers_both_components_and_excludes_sensor_geometry(pixel_geometry):
    calibration, geometry = pixel_geometry
    scene, materials = pixel_primitives(geometry)
    result = compile_regions(scene)
    assert len(scene.instances) == 164 and not scene.wires
    assert calibration.manifest["pixel_model"]["simplified"] is True
    assert len(scene.volumes) == 3
    assert result.rejected_domains == (("solid:0", "inconsistent or unknown exterior material"),)
    lar = materials.index("liquid_argon")
    negative = result.select([[-1000, 0, 0]], material=lar)
    positive = result.select([[1000, 0, 0]], material=lar)
    assert negative.bounds.contains([-1000, 0, 0]) and negative.bounds.upper[0] == -3
    assert positive.bounds.contains([1000, 0, 0]) and positive.bounds.lower[0] == 3
    points = np.concatenate(
        [i.mesh.vertices @ i.rotation.T + i.translation for i in scene.instances]
    )
    for region in result.regions:
        assert not region.bounds.contains(points).any()


def test_pixel_adapter_rejects_new_interior_boundary_in_box(pixel_geometry):
    from copy import deepcopy

    _, geometry = pixel_geometry
    changed = deepcopy(geometry)
    active = changed.solids[-2]
    active.mesh.vertices[0, 0] = 0.0
    with pytest.raises(ValueError, match="box"):
        pixel_primitives(changed)
