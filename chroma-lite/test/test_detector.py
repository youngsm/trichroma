from .unittest_find import unittest
import numpy as np

from chroma.geometry import Solid, Geometry, vacuum
from chroma.loader import create_geometry_from_obj
from chroma.detector import Detector
from chroma.make import box
from chroma.sim import Simulation
from chroma.event import Photons

from chroma.demo.optics import r7081hqe_photocathode

class TestDetector(unittest.TestCase):
    def setUp(self):
        # Setup geometry
        cube = Detector(vacuum)
        cube.add_pmt(Solid(box(10.0,10,10), vacuum, vacuum, surface=r7081hqe_photocathode))
        cube.set_time_dist_gaussian(1.2, -6.0, 6.0)
        cube.set_charge_dist_gaussian(1.0, 0.1, 0.5, 1.5)

        geo = create_geometry_from_obj(cube, update_bvh_cache=False)

        self.geo = geo
        self.sim = Simulation(self.geo, geant4_processes=0)
        
    def testTime(self):
        '''Test PMT time distribution'''

        # Run only one photon at a time
        nphotons = 1
        pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
        dir = np.tile([0,0,1], (nphotons,1)).astype(np.float32)
        pol = np.zeros_like(pos)
        phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
        pol[:,0] = np.cos(phi)
        pol[:,1] = np.sin(phi)
        t = np.zeros(nphotons, dtype=np.float32) + 100.0 # Avoid negative photon times
        wavelengths = np.empty(nphotons, np.float32)
        wavelengths.fill(400.0)

        photons = Photons(pos=pos, dir=dir, pol=pol, t=t,
                          wavelengths=wavelengths)

        hit_times = []
        for ev in self.sim.simulate(photons for i in range(1000)):
            if ev.channels.hit[0]:
                hit_times.append(ev.channels.t[0])
        hit_times = np.array(hit_times)

        self.assertAlmostEqual(hit_times.std(),  1.2, delta=1e-1)


    def testCharge(self):
        '''Test PMT charge distribution'''

        # Run only one photon at a time
        nphotons = 1
        pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
        dir = np.tile([0,0,1], (nphotons,1)).astype(np.float32)
        pol = np.zeros_like(pos)
        phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
        pol[:,0] = np.cos(phi)
        pol[:,1] = np.sin(phi)
        t = np.zeros(nphotons, dtype=np.float32)
        wavelengths = np.empty(nphotons, np.float32)
        wavelengths.fill(400.0)

        photons = Photons(pos=pos, dir=dir, pol=pol, t=t,
                          wavelengths=wavelengths)

        hit_charges = []
        for ev in self.sim.simulate(photons for i in range(1000)):
            if ev.channels.hit[0]:
                hit_charges.append(ev.channels.q[0])
        hit_charges = np.array(hit_charges)
        self.assertAlmostEqual(hit_charges.mean(),  1.0, delta=1e-1)
        self.assertAlmostEqual(hit_charges.std(), 0.1, delta=1e-1)
