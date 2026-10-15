from .unittest_find import unittest
import numpy as np
import itertools

import chroma.demo
from chroma.loader import create_geometry_from_obj
from chroma.generator.photon import G4ParallelGenerator
from chroma.generator.vertex import constant_particle_gun
from chroma.demo.optics import water
from chroma import gpu
from chroma.sim import Simulation

class TestPDF(unittest.TestCase):
    def setUp(self):
        self.detector = create_geometry_from_obj(chroma.demo.tiny(), update_bvh_cache=False)
        self.vertex_gen = constant_particle_gun('e-', (0,0,0), (1,0,0), 10)

    def testGPUPDF(self):
        '''Create a hit count and (q,t) PDF for 10 MeV events in MicroLBNE'''

        g4generator = G4ParallelGenerator(1, water)

        context = gpu.create_cuda_context()

        gpu_geometry = gpu.GPUDetector(self.detector)

        nthreads_per_block, max_blocks = 64, 1024

        rng_states = gpu.get_rng_states(nthreads_per_block*max_blocks)

        gpu_daq = gpu.GPUDaq(gpu_geometry)
        gpu_pdf = gpu.GPUPDF()
        gpu_pdf.setup_pdf(self.detector.num_channels(), 100, (-0.5, 999.5), 10, (-0.5, 9.5))
        
        gpu_pdf.clear_pdf()

        for ev in g4generator.generate_events(itertools.islice(self.vertex_gen, 10)):
            gpu_photons = gpu.GPUPhotons(ev.photons_beg)
            gpu_photons.propagate(gpu_geometry, rng_states, nthreads_per_block,
                                  max_blocks)
            gpu_daq.begin_acquire()
            gpu_daq.acquire(gpu_photons, rng_states,
                            nthreads_per_block, max_blocks)
            gpu_channels = gpu_daq.end_acquire()
            gpu_pdf.add_hits_to_pdf(gpu_channels)

        hitcount, pdf = gpu_pdf.get_pdfs()
        self.assertTrue( (hitcount > 0).any() )
        self.assertTrue( (pdf > 0).any() )

        # Consistency checks
        for i, nhits in enumerate(hitcount):
            self.assertEqual(nhits, pdf[i].sum())

        context.pop()

    def testSimPDF(self):
        sim = Simulation(self.detector)

        vertex_iter = itertools.islice(self.vertex_gen, 10)

        hitcount, pdf = sim.create_pdf(vertex_iter, 100, (-0.5, 999.5), 10, (-0.5, 9.5))

        self.assertTrue( (hitcount > 0).any() )
        self.assertTrue( (pdf > 0).any() )

        # Consistency checks
        for i, nhits in enumerate(hitcount):
            self.assertEqual(nhits, pdf[i].sum())
