from .unittest_find import unittest
import numpy as np

from chroma.geometry import Solid, Geometry, vacuum
from chroma.loader import create_geometry_from_obj
from chroma.make import box
from chroma.sim import Simulation
from chroma.event import Photons
from chroma.tools import count_nonzero

class TestPropagation(unittest.TestCase):
    def testAbort(self):
        '''Photons that hit a triangle at normal incidence should not abort.

        Photons that hit a triangle at exactly normal incidence can sometimes
        produce a dot product that is outside the range allowed by acos().
        Trigger these with axis aligned photons in a box.
        '''

        # Setup geometry
        cube = Geometry(vacuum)
        cube.add_solid(Solid(box(100,100,100), vacuum, vacuum))
        geo = create_geometry_from_obj(cube, update_bvh_cache=False)
        sim = Simulation(geo, geant4_processes=0)

        # Create initial photons
        nphotons = 10000
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

        # First make one step to check for strangeness
        photons_end = next(sim.simulate([photons], keep_photons_end=True,
                                   max_steps=1)).photons_end

        self.assertFalse(np.isnan(photons_end.pos).any())
        self.assertFalse(np.isnan(photons_end.dir).any())
        self.assertFalse(np.isnan(photons_end.pol).any())
        self.assertFalse(np.isnan(photons_end.t).any())
        self.assertFalse(np.isnan(photons_end.wavelengths).any())

        # Now let it run the usual ten steps
        photons_end = next(sim.simulate([photons], keep_photons_end=True,
                                   max_steps=10)).photons_end
        aborted = (photons_end.flags & (1 << 31)) > 0
        print('aborted photons: %1.1f' % \
            (float(count_nonzero(aborted)) / nphotons))
        self.assertFalse(aborted.any())
        
