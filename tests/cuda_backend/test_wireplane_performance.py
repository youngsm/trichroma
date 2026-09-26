"""
Test and benchmark wire plane intersection performance.

This tests the FP32 optimization for analytic wire plane intersections.
"""

from .unittest_find import unittest
import numpy as np
import time

from chroma.geometry import Solid, Geometry, Material, Surface, vacuum
from chroma.loader import create_geometry_from_obj
from chroma.make import box
from chroma.sim import Simulation
from chroma.event import Photons


def make_test_material():
    """Create a simple test material (liquid argon-like)."""
    mat = Material('test_lar')
    mat.set('refractive_index', 1.38)
    mat.set('absorption_length', 1000.0)
    mat.set('scattering_length', 1000.0)
    return mat


def make_reflective_surface():
    """Create a highly reflective surface."""
    surf = Surface('reflect99')
    surf.set('reflect_specular', 0.99)
    surf.set('absorb', 0.01)
    return surf


def make_wireplane_geometry(n_planes=3, pitch=3.0, radius=0.075, 
                            box_size=100.0, wire_angles=None):
    """
    Create a test geometry with analytic wire planes.
    
    Parameters
    ----------
    n_planes : int
        Number of wire planes
    pitch : float
        Wire pitch in mm
    radius : float
        Wire radius in mm
    box_size : float
        Size of containing box in mm
    wire_angles : list
        Angles of wire planes in radians
    """
    if wire_angles is None:
        wire_angles = [np.pi/2, np.pi/3, -np.pi/3][:n_planes]
    
    lar = make_test_material()
    steel = Material('steel')
    steel.set('refractive_index', 2.5)
    steel.set('absorption_length', 0.001)
    steel.set('scattering_length', 0.001)
    
    wire_surface = make_reflective_surface()
    
    # create geometry
    geo = Geometry(lar)
    
    # add containing box - use steel as material2 to register it
    container = box(box_size, box_size, box_size, center=(0, 0, 0))
    container_solid = Solid(container, lar, steel, surface=wire_surface)
    geo.add_solid(container_solid)
    
    # add analytic wire planes
    half_size = box_size / 2.0
    geo.wireplanes = []
    
    for i, angle in enumerate(wire_angles):
        cA = np.cos(angle)
        sA = np.sin(angle)
        
        # u direction (along wires)
        u_dir = np.array([0.0, cA, sA], dtype=np.float32)
        
        # plane at +x boundary
        desc = {
            'origin': np.array([half_size - 1.0, 0.0, 0.0], dtype=np.float32),
            'u': u_dir,
            'v': np.array([0.0, -sA, cA], dtype=np.float32),  # perpendicular to u in YZ plane
            'pitch': float(pitch),
            'radius': float(radius),
            'umin': -half_size,
            'umax': half_size,
            'vmin': -half_size,
            'vmax': half_size,
            'v0': 0.0,
            'surface': wire_surface,
            'material_inner': steel,
            'material_outer': lar,
            'color': 0xFF0000,
        }
        geo.wireplanes.append(desc)
    
    return create_geometry_from_obj(geo, update_bvh_cache=False)


def generate_test_photons(nphotons, box_size=100.0):
    """Generate random photons inside a box."""
    half = box_size / 2.0 * 0.8  # stay inside boundary
    
    pos = np.random.uniform(-half, half, (nphotons, 3)).astype(np.float32)
    
    # random directions
    theta = np.arccos(2 * np.random.random(nphotons) - 1)
    phi = np.random.uniform(0, 2*np.pi, nphotons)
    dir = np.zeros((nphotons, 3), dtype=np.float32)
    dir[:, 0] = np.sin(theta) * np.cos(phi)
    dir[:, 1] = np.sin(theta) * np.sin(phi)
    dir[:, 2] = np.cos(theta)
    
    # random polarization perpendicular to direction
    pol = np.zeros_like(dir)
    pol[:, 0] = np.cos(phi + np.pi/2)
    pol[:, 1] = np.sin(phi + np.pi/2)
    
    t = np.zeros(nphotons, dtype=np.float32)
    wavelengths = np.full(nphotons, 128.0, dtype=np.float32)  # VUV
    
    return Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)


class TestWirePlaneCorrectness(unittest.TestCase):
    """Test that wire plane intersection produces valid results."""
    
    def test_no_nan_abort(self):
        """Photons should not produce NaN or abort when hitting wires."""
        geo = make_wireplane_geometry(n_planes=1, pitch=5.0, radius=0.1)
        sim = Simulation(geo)
        
        nphotons = 1000
        photons = generate_test_photons(nphotons, box_size=100.0)
        
        # simulate
        for ev in sim.simulate([photons], keep_photons_end=True, max_steps=50):
            photons_end = ev.photons_end
        
        # check for NaN
        self.assertFalse(np.isnan(photons_end.pos).any(), "NaN in positions")
        self.assertFalse(np.isnan(photons_end.dir).any(), "NaN in directions")
        self.assertFalse(np.isnan(photons_end.t).any(), "NaN in times")
        
        # check for aborts
        aborted = (photons_end.flags & (1 << 31)) > 0
        abort_frac = float(np.sum(aborted)) / nphotons
        self.assertLess(abort_frac, 0.01, f"Too many aborted photons: {abort_frac*100:.1f}%")
    
    def test_wire_hits_detected(self):
        """Photons directed at wires should hit wire surface."""
        geo = make_wireplane_geometry(n_planes=1, pitch=5.0, radius=0.5)
        sim = Simulation(geo)
        
        nphotons = 100
        # photons directed at +x where wire plane is
        pos = np.zeros((nphotons, 3), dtype=np.float32)
        dir = np.zeros((nphotons, 3), dtype=np.float32)
        dir[:, 0] = 1.0  # all going toward +x
        pol = np.zeros_like(dir)
        pol[:, 1] = 1.0
        t = np.zeros(nphotons, dtype=np.float32)
        wavelengths = np.full(nphotons, 128.0, dtype=np.float32)
        
        photons = Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)
        
        for ev in sim.simulate([photons], keep_photons_end=True, max_steps=10):
            photons_end = ev.photons_end
        
        # some photons should have been reflected or absorbed
        SURFACE_ABSORB = 0x1 << 3
        REFLECT_SPECULAR = 0x1 << 6
        REFLECT_DIFFUSE = 0x1 << 5
        
        interacted = (photons_end.flags & (SURFACE_ABSORB | REFLECT_SPECULAR | REFLECT_DIFFUSE)) > 0
        interact_frac = float(np.sum(interacted)) / nphotons
        print(f"Interaction fraction: {interact_frac*100:.1f}%")
        # with such large wires, most should interact
        self.assertGreater(interact_frac, 0.1, "Too few photons interacted with wires")


class TestWirePlanePerformance(unittest.TestCase):
    """Benchmark wire plane intersection performance."""
    
    def test_benchmark_3_planes(self):
        """Benchmark with 3 wire planes (typical LAr TPC config)."""
        geo = make_wireplane_geometry(n_planes=3, pitch=3.0, radius=0.075, box_size=200.0)
        sim = Simulation(geo)
        
        nphotons = 100000
        photons = generate_test_photons(nphotons, box_size=200.0)
        
        # warmup
        for ev in sim.simulate([photons], keep_photons_end=True, max_steps=10):
            pass
        
        # timed run
        n_trials = 3
        times = []
        for _ in range(n_trials):
            photons = generate_test_photons(nphotons, box_size=200.0)
            t0 = time.time()
            for ev in sim.simulate([photons], keep_photons_end=True, max_steps=100):
                pass
            times.append(time.time() - t0)
        
        avg_time = np.mean(times)
        photons_per_sec = nphotons / avg_time
        
        print(f"\n=== Wire Plane Benchmark (3 planes) ===")
        print(f"Photons: {nphotons}")
        print(f"Time: {avg_time:.3f}s (avg of {n_trials})")
        print(f"Rate: {photons_per_sec/1e6:.2f}M photons/sec")
        print(f"=======================================\n")
        
        # just ensure it completes in reasonable time
        self.assertLess(avg_time, 60.0, "Benchmark took too long")


if __name__ == '__main__':
    unittest.main()

