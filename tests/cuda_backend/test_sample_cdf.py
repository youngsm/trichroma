import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
import ROOT
import os
from .unittest_find import unittest
import chroma

class TestSampling(unittest.TestCase):
    def setUp(self):
        self.context = chroma.gpu.create_cuda_context()
        current_directory = os.path.split(os.path.realpath(__file__))[0]
        from chroma.cuda import srcdir as source_directory
        source = open(current_directory + '/test_sample_cdf.cu').read()
        self.mod = SourceModule(source, options=['-I' + source_directory], no_extern_c=True, cache_dir=False)
        self.test_sample_cdf = self.mod.get_function('test_sample_cdf')

    def compare_sampling(self, hist, reps=10):
        nbins = hist.GetNbinsX();
        xaxis = hist.GetXaxis()
        intg = hist.GetIntegral()
        cdf_y = np.empty(nbins+1, dtype=float)
        cdf_x = np.empty_like(cdf_y)

        cdf_x[0] = xaxis.GetBinLowEdge(1)
        cdf_y[0] = 0.0
        for i in range(1,len(cdf_x)):
            cdf_y[i] = intg[i]
            cdf_x[i] = xaxis.GetBinUpEdge(i)

        cdf_x_gpu = gpuarray.to_gpu(cdf_x.astype(np.float32))
        cdf_y_gpu = gpuarray.to_gpu(cdf_y.astype(np.float32))
        block =(128,1,1)
        grid = (128, 1)
        out_gpu = gpuarray.empty(shape=int(block[0]*grid[0]), dtype=np.float32)

        out_h = ROOT.TH1D('out_h', '', hist.GetNbinsX(), 
                          xaxis.GetXmin(),
                          xaxis.GetXmax())
        out_h.SetLineColor(ROOT.kGreen)

        for i in range(reps):
            self.test_sample_cdf(np.int32(i),
                                 np.int32(len(cdf_x_gpu)), 
                                 cdf_x_gpu, cdf_y_gpu, out_gpu, block=block, grid=grid)
            out = out_gpu.get()
            for v in out:
                out_h.Fill(v)

        prob = out_h.KolmogorovTest(hist)
        return prob, out_h 

    def test_sampling(self):
        '''Verify that the CDF-based sampler on the GPU reproduces a binned
        Gaussian distribution'''
        f = ROOT.TF1('f_gaussian', 'gaus(0)', -5, 5)
        f.SetParameters(1.0/np.sqrt(np.pi * 2), 0.0, 1.0)
        gaussian = ROOT.TH1D('gaussian', '', 100, -5, 5)
        gaussian.Add(f)

        prob, out_h = self.compare_sampling(gaussian, reps=50)

        assert prob > 0.01

    def tearDown(self):
        self.context.pop()
