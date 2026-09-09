
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from chroma import event, gpu
import chroma.demo
from chroma.loader import create_geometry_from_obj
from chroma.sim import Simulation

class TestGPUPhotonsGPUInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.context = gpu.create_cuda_context()
        except Exception as err:
            raise unittest.SkipTest(f"CUDA context unavailable: {err}")

    @classmethod
    def tearDownClass(cls):
        cls.context.pop()

    def _make_photons(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        polarizations = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        wavelengths = np.array([400.0, 420.0], dtype=np.float32)
        times = np.array([0.1, 0.2], dtype=np.float32)
        flags = np.array([0, 1], dtype=np.uint32)
        weights = np.array([1.0, 0.5], dtype=np.float32)
        evidx = np.array([0, 0], dtype=np.uint32)
        return event.Photons(pos, directions, polarizations, wavelengths, times,
                             flags=flags, weights=weights, evidx=evidx)

    def _gpu_view(self, gpu_photons):
        return SimpleNamespace(
            pos=gpu_photons.pos,
            dir=gpu_photons.dir,
            pol=gpu_photons.pol,
            wavelengths=gpu_photons.wavelengths,
            t=gpu_photons.t,
            last_hit_triangles=gpu_photons.last_hit_triangles,
            flags=gpu_photons.flags,
            weights=gpu_photons.weights,
            evidx=gpu_photons.evidx,
        )

    def test_alias_when_single_copy(self):
        cpu_photons = self._make_photons()
        gpu_source = gpu.GPUPhotons(cpu_photons)
        gpu_alias = gpu.GPUPhotons(self._gpu_view(gpu_source))

        self.assertEqual(gpu_alias.true_nphotons, len(cpu_photons))
        self.assertEqual(int(gpu_alias.pos.gpudata), int(gpu_source.pos.gpudata))
        self.assertEqual(int(gpu_alias.dir.gpudata), int(gpu_source.dir.gpudata))
        self.assertEqual(int(gpu_alias.flags.gpudata), int(gpu_source.flags.gpudata))
        self.assertEqual(int(gpu_alias.evidx.gpudata), int(gpu_source.evidx.gpudata))

    def test_duplicate_from_gpu_arrays(self):
        cpu_photons = self._make_photons()
        gpu_source = gpu.GPUPhotons(cpu_photons)
        gpu_dupe = gpu.GPUPhotons(self._gpu_view(gpu_source), ncopies=2)

        n = len(cpu_photons)
        self.assertEqual(len(gpu_dupe.pos), n * 2)

        pos_host = gpu_dupe.pos.get().view(np.float32).reshape((-1, 3))
        np.testing.assert_allclose(pos_host[:n], pos_host[n:])

        flags_host = gpu_dupe.flags.get()
        np.testing.assert_array_equal(flags_host[:n], flags_host[n:])

    def test_reset_optional_fields(self):
        cpu_photons = self._make_photons()
        gpu_source = gpu.GPUPhotons(cpu_photons)
        gpu_reset = gpu.GPUPhotons(self._gpu_view(gpu_source), copy_flags=False,
                                   copy_triangles=False, copy_weights=False)

        self.assertNotEqual(int(gpu_reset.flags.gpudata), int(gpu_source.flags.gpudata))
        self.assertTrue(np.all(gpu_reset.flags.get() == 0))
        self.assertTrue(np.all(gpu_reset.last_hit_triangles.get() == -1))
        self.assertTrue(np.allclose(gpu_reset.weights.get(), 1.0))

class TestSimulationGPUPhotonInput(unittest.TestCase):
    def test_simulate_accepts_gpu_photons(self):
        detector = create_geometry_from_obj(chroma.demo.tiny(), update_bvh_cache=False)
        sim = Simulation(detector)
        gpu_photons = None
        try:
            pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            directions = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
            polarizations = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            wavelengths = np.array([400.0], dtype=np.float32)
            times = np.array([0.0], dtype=np.float32)

            cpu_photons = event.Photons(pos, directions, polarizations, wavelengths, times)
            gpu_photons = gpu.GPUPhotons(cpu_photons)
            ev = event.Event(photons_beg=gpu_photons)

            with mock.patch('chroma.event.Photons.join', side_effect=AssertionError('CPU join should not be used for GPU sources')):
                results = list(sim.simulate([ev], keep_hits=False, keep_flat_hits=False, run_daq=False, max_steps=1))

            self.assertEqual(len(results), 1)
        finally:
            if gpu_photons is not None:
                del gpu_photons
            del sim

