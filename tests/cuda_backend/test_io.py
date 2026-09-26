from .unittest_find import unittest
from chroma.io import root
from chroma import event
import numpy as np

class TestRootIO(unittest.TestCase):
    def test_file_write_and_read(self):
        ev = event.Event(1, event.Vertex('e-', pos=(0,0,1), dir=(1,0,0),
                                         ke=15.0, pol=(0,1,0), t0=40.0))

        photons_beg = root.make_photon_with_arrays(1)
        photons_beg.pos[0] = (1,2,3)
        photons_beg.dir[0] = (4,5,6)
        photons_beg.pol[0] = (7,8,9)
        photons_beg.wavelengths[0] = 400.0
        photons_beg.t[0] = 100.0
        photons_beg.last_hit_triangles[0] = 5
        photons_beg.flags[0] = 20
        ev.photons_beg = photons_beg

        photons_end = root.make_photon_with_arrays(1)
        photons_end.pos[0] = (1,2,3)
        photons_end.dir[0] = (4,5,6)
        photons_end.pol[0] = (7,8,9)
        photons_end.wavelengths[0] = 400.0
        photons_end.t[0] = 100.0
        photons_end.last_hit_triangles[0] = 5
        photons_end.flags[0] = 20
        ev.photons_end = photons_end

        ev.vertices = [ev.primary_vertex]

        channels = event.Channels(hit=np.array([True, False]), 
                                  t=np.array([20.0, 1e9], dtype=np.float32),
                                  q=np.array([2.0, 0.0], dtype=np.float32),
                                  flags=np.array([8, 32], dtype=np.uint32))
        ev.channels = channels

        filename = '/tmp/chroma-filewritertest.root'
        writer = root.RootWriter(filename)
        writer.write_event(ev)
        writer.close()

        # Exercise the RootReader methods
        reader = root.RootReader(filename)
        self.assertEqual(len(reader), 1)

        self.assertRaises(StopIteration, reader.prev)

        next(reader)

        self.assertEqual(reader.index(), 0)
        self.assertRaises(StopIteration, reader.__next__)
        
        reader.jump_to(0)

        # Enough screwing around, let's get the one event in the file
        newev = reader.current()

        # Now check if everything is correct in the event
        for attribute in ['id']:
            self.assertEqual(getattr(ev, attribute), getattr(newev, attribute), 'compare %s' % attribute)

        for attribute in ['pos', 'dir', 'pol', 'ke', 't0']:
            self.assertTrue(np.allclose(getattr(ev.primary_vertex, attribute), getattr(newev.primary_vertex, attribute)), 'compare %s' % attribute)

            for i in range(len(ev.vertices)):
                self.assertTrue(np.allclose(getattr(ev.vertices[i], attribute), getattr(newev.vertices[i], attribute)), 'compare %s' % attribute)

        for attribute in ['pos', 'dir', 'pol', 'wavelengths', 't', 'last_hit_triangles', 'flags']:    
            self.assertTrue(np.allclose(getattr(ev.photons_beg, attribute), 
                                        getattr(newev.photons_beg, attribute)), 'compare %s' % attribute)
            self.assertTrue(np.allclose(getattr(ev.photons_end, attribute), 
                                        getattr(newev.photons_end, attribute)), 'compare %s' % attribute)
