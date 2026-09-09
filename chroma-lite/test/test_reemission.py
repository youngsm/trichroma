from .unittest_find import unittest
import numpy as np
#import matplotlib.pyplot as plt

from chroma.geometry import Solid, Geometry, Surface, Material
from chroma.loader import create_geometry_from_obj
from chroma.make import sphere
from chroma.sim import Simulation
from chroma.demo.optics import vacuum
from chroma.event import Photons, SURFACE_DETECT
from chroma.tools import enable_debug_on_crash

class TestReemission(unittest.TestCase):
    @unittest.skip('need to implement scipy stats functions here')
    def testBulkReemission(self):
        '''Test bulk reemission 

        Start a bunch of monoenergetic photons at the center of a wavelength-
        shifting sphere, forcing reemission, and check that the final
        wavelength distribution matches the wls spectrum.
        '''
        import scipy.stats
        nphotons = 1e5

        # set up detector -- a sphere of 'scintillator' surrounded by a
        # detecting sphere
        scint = Material('scint')
        scint.set('refractive_index', 1)
        scint.set('absorption_length', 1.0)
        scint.set('scattering_length', 1e7)
        scint.set('reemission_prob', 1)

        x = np.arange(0,1000,10)
        norm = scipy.stats.norm(scale=50, loc=600)
        pdf = 10 * norm.pdf(x)
        cdf = norm.cdf(x)
        scint.reemission_cdf = np.array(list(zip(x, cdf)))

        detector = Surface('detector')
        detector.set('detect', 1)

        world = Geometry(vacuum)
        world.add_solid(Solid(sphere(1000), vacuum, vacuum, surface=detector))
        world.add_solid(Solid(sphere(500), scint, vacuum))
        w = create_geometry_from_obj(world, update_bvh_cache=False)

        sim = Simulation(w, geant4_processes=0)

        # initial photons -- isotropic 250 nm at the origin
        pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
        dir = np.random.rand(nphotons, 3).astype(np.float32) * 2 - 1
        dir /= np.sqrt(dir[:,0]**2 + dir[:,1]**2 + dir[:,2]**2)[:,np.newaxis]
        pol = np.zeros_like(pos)
        t = np.zeros(nphotons, dtype=np.float32)
        wavelengths = np.ones(nphotons).astype(np.float32) * 250

        photons = Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)

        # run simulation and extract final wavelengths
        event = next(sim.simulate([photons], keep_photons_end=True))
        mask = (event.photons_end.flags & SURFACE_DETECT) > 0
        final_wavelengths = event.photons_end.wavelengths[mask]

        # compare wavelength distribution to scintillator's reemission pdf
        hist, edges = np.histogram(final_wavelengths, bins=x)
        print('detected', hist.sum(), 'of', nphotons, 'photons')
        hist_norm = 1.0 * hist / (1.0 * hist.sum() / 1000)
        pdf /= (1.0 * pdf.sum() / 1000)

        chi2 = scipy.stats.chisquare(hist_norm, pdf[:-1])[1]
        print('chi2 =', chi2)

        # show histogram comparison
        #plt.figure(1)
        #width = edges[1] - edges[0]
        #plt.bar(left=edges, height=pdf, width=width, color='red')
        #plt.bar(left=edges[:-1], height=hist_norm, width=width)
        #plt.show()

        self.assertTrue(chi2 > 0.75)

