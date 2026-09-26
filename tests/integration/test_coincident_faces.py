"""A solid whose face lies in the plane of an analytic box's face.

chroma-lar mounts its PMTs with their backs in the plane of the TPC walls. A
photon that leaves such a PMT through its back is left on the wall plane,
heading out of the TPC: the wall is at distance ~0, and a triangle test that
needs t > 1e-6 (Chroma's, and the analytic boxes' before the fix) lets the
photon through whenever the rounding of its position puts it on or beyond the
plane. It then leaves the detector (1.0-1.3% of the photons of the LUT
fixture). Here a glass cylinder (an instanced mesh, like a PMT) stands against
the inside of a mirror box at metre-scale coordinates; photons start in the
glass heading for the shared face. None may end outside the box.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

X0 = -2295.4148  # the reflect3wires TPC wall
N = 20_000


def _geometry():
    from chroma.geometry import Geometry, Material, Mesh, Solid, Surface
    from chroma.loader import create_geometry_from_obj
    from chroma.make import box, cylinder

    def material(name, index):
        m = Material(name)
        m.set("refractive_index", index)
        m.set("absorption_length", 1e30)
        m.set("scattering_length", 1e30)
        return m

    def surface(name, **properties):
        s = Surface(name)
        for key, value in properties.items():
            s.set(key, value)
        return s

    def against_wall(mesh, axis):
        """The mesh with ``axis`` turned to x and its low-x end exactly on x = X0."""
        v = np.asarray(mesh.vertices, np.float64)[:, [axis, (axis + 1) % 3, (axis + 2) % 3]]
        v[:, 0] += X0 - v[:, 0].min()
        return Mesh(v, np.asarray(mesh.triangles), remove_duplicate_vertices=False)

    argon, glass, vacuum = material("argon", 1.38), material("glass", 1.5), material("vacuum", 1.0)
    mirror = surface("mirror", reflect_specular=1.0)
    absorber = surface("absorber", absorb=1.0)
    geometry = Geometry(vacuum)
    geometry.add_solid(Solid(against_wall(box(600.0, 800.0, 800.0), 0), argon, vacuum, surface=mirror))
    tube = cylinder(50.0, 10.0, nsteps=32)
    length_axis = int(np.argmin(np.ptp(np.asarray(tube.vertices), axis=0)))
    geometry.add_solid(Solid(against_wall(tube, length_axis), glass, argon))
    geometry.add_solid(Solid(box(8000.0, 3000.0, 3000.0), vacuum, vacuum, surface=absorber))
    return create_geometry_from_obj(geometry)


def _escaped(seed=5):
    from chroma.event import Photons
    from trichroma.simulation import Simulation

    rng = np.random.default_rng(seed)
    r, phi = 40.0 * np.sqrt(rng.random(N)), rng.uniform(0, 2 * np.pi, N)
    pos = np.stack([np.full(N, X0 + 5.0), r * np.cos(phi), r * np.sin(phi)], 1)
    d = rng.normal(size=(N, 3))
    d[:, 0] = -np.abs(d[:, 0])
    d /= np.linalg.norm(d, axis=1)[:, None]
    keep = d[:, 0] < -0.2
    pos, d = pos[keep], d[keep]
    pol = np.cross(d, rng.normal(size=d.shape))
    pol /= np.linalg.norm(pol, axis=1)[:, None]
    ev = next(Simulation(_geometry(), seed=seed).simulate(
        [Photons(pos, d, pol, np.full(len(d), 400.0))], keep_photons_end=True, keep_flat_hits=False,
        keep_hits=False, max_steps=20))
    return int((ev.photons_end.pos[:, 0] < X0 - 1e-3).sum()), len(d)


def test_photons_leaving_a_solid_on_a_box_face_meet_the_box():
    escaped, n = _escaped()
    assert n > 10_000 and escaped == 0


def test_bug_compatible_mode_keeps_the_leak(monkeypatch):
    monkeypatch.setenv("CHROMA_TRITON_FIXES", "0")
    escaped, n = _escaped()
    assert escaped > 0.01 * n  # Chroma's t > 1e-6 lets photons on the plane through
