import numpy as np
from pycuda import gpuarray as ga
from pycuda import driver as cuda

from chroma.gpu.tools import get_cu_module, cuda_options, GPUFuncs, \
    chunk_iterator
from chroma import event

class GPUChannels(object):
    def __init__(self, t, q, flags, ndaq=1, stride=None):
        self.t = t
        self.q = q
        self.flags = flags
        self.ndaq = ndaq
        if stride is None:
            self.stride = len(t)
        else:
            self.stride = stride

    def iterate_copies(self):
        for i in range(self.ndaq):
            yield GPUChannels(self.t[i*self.stride:(i+1)*self.stride],
                              self.q[i*self.stride:(i+1)*self.stride],
                              self.flags[i*self.stride:(i+1)*self.stride])

    def get(self):
        t = self.t.get()
        q = self.q.get()

        # For now, assume all channels with small
        # enough hit time were hit.
        return event.Channels(t<1e8, t, q, self.flags.get())

    def __len__(self):
        return self.t.size

class GPUDaq(object):
    def __init__(self, gpu_detector, ndaq=1):
        assert gpu_detector.nchannels>0, "Geometry has no detectors, DAQ can't be initialized."
        self.earliest_time_gpu = ga.empty(gpu_detector.nchannels*ndaq, dtype=np.float32)
        self.earliest_time_int_gpu = ga.empty(gpu_detector.nchannels*ndaq, dtype=np.uint32)
        self.channel_history_gpu = ga.zeros_like(self.earliest_time_int_gpu)
        self.channel_q_int_gpu = ga.zeros_like(self.earliest_time_int_gpu)
        self.channel_q_gpu = ga.zeros(len(self.earliest_time_int_gpu), dtype=np.float32)
        self.detector_gpu = gpu_detector.detector_gpu
        self.solid_id_map_gpu = gpu_detector.solid_id_map
        self.solid_id_to_channel_index_gpu = gpu_detector.solid_id_to_channel_index_gpu

        self.module = get_cu_module('daq.cu', options=cuda_options, 
                                    include_source_directory=True)
        self.gpu_funcs = GPUFuncs(self.module)
        self.ndaq = ndaq
        self.stride = gpu_detector.nchannels

    def begin_acquire(self, nthreads_per_block=64):
        self.gpu_funcs.reset_earliest_time_int(np.float32(1e9), np.int32(len(self.earliest_time_int_gpu)), self.earliest_time_int_gpu, block=(nthreads_per_block,1,1), grid=(len(self.earliest_time_int_gpu)//nthreads_per_block+1,1))
        self.channel_q_int_gpu.fill(0)
        self.channel_q_gpu.fill(0)
        self.channel_history_gpu.fill(0)

    def acquire(self, gpuphotons, rng_states, nthreads_per_block=64, max_blocks=1024, start_photon=None, nphotons=None, weight=1.0):
        if start_photon is None:
            start_photon = 0
        if nphotons is None:
            nphotons = len(gpuphotons.pos) - start_photon

        if self.ndaq == 1:
            for first_photon, photons_this_round, blocks in \
                    chunk_iterator(nphotons, nthreads_per_block, max_blocks):
                self.gpu_funcs.run_daq(rng_states, np.uint32(0x1 << 2), 
                                       np.int32(start_photon+first_photon), np.int32(photons_this_round), gpuphotons.t, 
                                       gpuphotons.flags, gpuphotons.last_hit_triangles, gpuphotons.weights,
                                       self.solid_id_map_gpu,
                                       self.detector_gpu,
                                       self.earliest_time_int_gpu, 
                                       self.channel_q_int_gpu, self.channel_history_gpu,
                                       np.float32(weight),
                                       block=(nthreads_per_block,1,1), grid=(blocks,1))
        else:
            for first_photon, photons_this_round, blocks in \
                    chunk_iterator(nphotons, 1, max_blocks):
                self.gpu_funcs.run_daq_many(rng_states, np.uint32(0x1 << 2), 
                                            np.int32(start_photon+first_photon), np.int32(photons_this_round), gpuphotons.t, 
                                            gpuphotons.flags, gpuphotons.last_hit_triangles, gpuphotons.weights,
                                            self.solid_id_map_gpu,
                                            self.detector_gpu,
                                            self.earliest_time_int_gpu, 
                                            self.channel_q_int_gpu, self.channel_history_gpu, 
                                            np.int32(self.ndaq), np.int32(self.stride),
                                            np.float32(weight),
                                            block=(nthreads_per_block,1,1), grid=(blocks,1))
        cuda.Context.get_current().synchronize()
    
    def end_acquire(self, nthreads_per_block=64):
        self.gpu_funcs.convert_sortable_int_to_float(np.int32(len(self.earliest_time_int_gpu)), self.earliest_time_int_gpu, self.earliest_time_gpu, block=(nthreads_per_block,1,1), grid=(len(self.earliest_time_int_gpu)//nthreads_per_block+1,1))

        self.gpu_funcs.convert_charge_int_to_float(self.detector_gpu, self.channel_q_int_gpu, self.channel_q_gpu, block=(nthreads_per_block,1,1), grid=(len(self.channel_q_int_gpu)//nthreads_per_block+1,1))

        cuda.Context.get_current().synchronize()

        return GPUChannels(self.earliest_time_gpu, self.channel_q_gpu, self.channel_history_gpu, self.ndaq, self.stride)

