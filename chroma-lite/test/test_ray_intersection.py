from .unittest_find import unittest
import chroma
import numpy as np
import os
from pycuda import gpuarray as ga

class TestRayIntersection(unittest.TestCase):
    def setUp(self):
        self.context = chroma.gpu.create_cuda_context()
        self.module = chroma.gpu.get_cu_module('mesh.h')
        self.gpu_funcs = chroma.gpu.GPUFuncs(self.module)
        cube = chroma.loader.create_geometry_from_obj(chroma.make.cube(size=1000.0), update_bvh_cache=False)
        self.box = chroma.gpu.GPUGeometry(cube)

        pos, dir = chroma.project.from_film()
        self.pos_gpu = ga.to_gpu(chroma.gpu.to_float3(pos))
        self.dir_gpu = ga.to_gpu(chroma.gpu.to_float3(dir))

        testdir = os.path.dirname(os.path.abspath(__file__))
        self.dx_standard = np.load(os.path.join(testdir,
                                                'data/ray_intersection.npy'))

    @unittest.skip('Ray data file needs to be updated')
    def test_intersection_distance(self):
        dx = ga.zeros(self.pos_gpu.size, dtype=np.float32)
        self.gpu_funcs.distance_to_mesh(np.int32(self.pos_gpu.size), self.pos_gpu, self.dir_gpu, self.box.gpudata, dx, block=(64,1,1), grid=(self.pos_gpu.size//64+1,1))

        self.assertTrue((dx.get() == self.dx_standard).all())

    def tearDown(self):
        self.context.pop()
