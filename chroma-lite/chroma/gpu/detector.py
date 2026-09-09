import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray as ga
from pycuda import characterize

from chroma.geometry import standard_wavelengths
from chroma.gpu.tools import get_cu_module, get_cu_source, cuda_options, \
    chunk_iterator, format_array, format_size, to_uint3, to_float3, \
    make_gpu_struct
from chroma.log import logger

from chroma.gpu.geometry import GPUGeometry

class GPUDetector(GPUGeometry):
    def __init__(self, detector, wavelengths=None, print_usage=False):
        GPUGeometry.__init__(self, detector, wavelengths=wavelengths, print_usage=False)
        self.solid_id_to_channel_index_gpu = \
            ga.to_gpu(detector.solid_id_to_channel_index.astype(np.int32))
        self.nchannels = detector.num_channels()


        self.time_cdf_x_gpu = ga.to_gpu(detector.time_cdf[0].astype(np.float32))
        self.time_cdf_y_gpu = ga.to_gpu(detector.time_cdf[1].astype(np.float32))

        self.charge_cdf_x_gpu = ga.to_gpu(detector.charge_cdf[0].astype(np.float32))
        self.charge_cdf_y_gpu = ga.to_gpu(detector.charge_cdf[1].astype(np.float32))

        detector_source = get_cu_source('detector.h')
        detector_struct_size = characterize.sizeof('Detector', detector_source)
        self.detector_gpu = make_gpu_struct(detector_struct_size,
                                            [self.solid_id_to_channel_index_gpu,
                                             self.time_cdf_x_gpu,
                                             self.time_cdf_y_gpu,
                                             self.charge_cdf_x_gpu,
                                             self.charge_cdf_y_gpu,
                                             np.int32(self.nchannels),
                                             np.int32(len(detector.time_cdf[0])),
                                             np.int32(len(detector.charge_cdf[0])),
                                             np.float32(detector.charge_cdf[0][-1] / 2**16)])
                                             
