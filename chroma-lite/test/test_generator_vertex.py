from .unittest_find import unittest
import itertools
import numpy as np
import chroma.generator.vertex

class TestParticleGun(unittest.TestCase):
    def test_constant_particle_gun_center(self):
        '''Generate electron vertices at the center of the world volume.'''
        vertex = chroma.generator.vertex.constant_particle_gun('e-', (0,0,0), (1,0,0), 100)
        for ev in itertools.islice(vertex, 100):
            self.assertEqual(ev.primary_vertex.particle_name, 'e-')
            self.assertTrue(np.allclose(ev.primary_vertex.pos, [0,0,0]))
            self.assertTrue(np.allclose(ev.primary_vertex.dir, [1,0,0]))
            self.assertTrue(np.allclose(ev.primary_vertex.ke, 100))

    def test_off_center(self):
        '''Generate electron vertices at (1,0,0) in the world volume.'''
        vertex = chroma.generator.vertex.constant_particle_gun('e-', (1,0,0), (1,0,0), 100)
        for ev in itertools.islice(vertex, 100):
            self.assertEqual(ev.primary_vertex.particle_name, 'e-')
            self.assertTrue(np.allclose(ev.primary_vertex.pos, [1,0,0]))
            self.assertTrue(np.allclose(ev.primary_vertex.dir, [1,0,0]))
            self.assertTrue(np.allclose(ev.primary_vertex.ke, 100))
