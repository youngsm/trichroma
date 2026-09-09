#!/usr/bin/env python
import time
import os
import numpy as np
from types import SimpleNamespace

from chroma import gpu
from chroma import event
from chroma import itertoolset
from chroma.tools import profile_if_possible

from pycuda import gpuarray as ga
import pycuda.driver as cuda

from timeit import default_timer as timer

def pick_seed():
    """Returns a seed for a random number generator selected using
    a mixture of the current time and the current process ID."""
    return int(time.time()) ^ (os.getpid() << 16) & 2**32-1

class Simulation(object):
    def __init__(self, detector, seed=None, cuda_device=None, photon_tracking=False,
                 nthreads_per_block=512, max_blocks=1024):
        self.detector = detector

        self.nthreads_per_block = nthreads_per_block
        self.max_blocks = max_blocks
        self.photon_tracking = photon_tracking

        if seed is None:
            self.seed = pick_seed()
        else:
            self.seed = seed

        # We have three generators to seed: numpy.random, GEANT4, and CURAND.
        # The latter two are done below.
        np.random.seed(self.seed)

        self.context = gpu.create_cuda_context(cuda_device)

        if hasattr(detector, 'num_channels'):
            self.gpu_geometry = gpu.GPUDetector(detector)
            self.gpu_daq = gpu.GPUDaq(self.gpu_geometry)
            self.gpu_pdf = gpu.GPUPDF()
            self.gpu_pdf_kernel = gpu.GPUKernelPDF()
        else:
            self.gpu_geometry = gpu.GPUGeometry(detector)

        self.rng_states = gpu.get_rng_states(self.nthreads_per_block*self.max_blocks, seed=self.seed)

        self.pdf_config = None
     
    def _simulate_batch(self,batch_events,keep_photons_beg=False,keep_photons_end=False,keep_hits=True,keep_flat_hits=True,run_daq=False, max_steps=100, verbose=False):
        '''Assumes batch_events is a list of Event objects with photons_beg having evidx set to the index in the array.
           
           Yields the fully formed events. Do not call directly.'''
        
        t_start = timer()

        photon_sources = [ev.photons_beg for ev in batch_events]
        batch_bounds = np.cumsum(np.concatenate([[0], [len(src) for src in photon_sources]]))

        copy_flags = True
        copy_triangles = False
        copy_weights = False

        # Join photons either on the GPU (preferred) or fall back to CPU concatenation.
        stacked_gpu_view = self._stack_gpu_photon_sources(
            photon_sources,
            copy_flags=copy_flags,
            copy_triangles=copy_triangles,
            copy_weights=copy_weights,
        )
        batch_photons_source = stacked_gpu_view
        if stacked_gpu_view is None:
            batch_photons_source = event.Photons.join(photon_sources)

        #This copy to gpu has a _lot_ of overhead, want 100k photons at least, hence batches
        #Assume triangles, and weights are unimportant to copy to GPU
        t_copy_start = timer()
        gpu_photons = gpu.GPUPhotons(
            batch_photons_source,
            copy_flags=copy_flags,
            copy_triangles=copy_triangles,
            copy_weights=copy_weights,
        )
        t_copy_end = timer()
        if verbose:
            print('GPU copy took %0.2f s' % (t_copy_end-t_copy_start))

        t_prop_start = timer()
        tracking = gpu_photons.propagate(self.gpu_geometry, self.rng_states,
                              nthreads_per_block=self.nthreads_per_block,
                              max_blocks=self.max_blocks,
                              max_steps=max_steps,track=self.photon_tracking)
            
        t_prop_end = timer()
        if verbose:
            print('GPU propagate took %0.2f s' % (t_prop_end-t_prop_start))
                              
        t_end = timer()
        if verbose:
            print('Batch took %0.2f s' % (t_end-t_start))

        if keep_photons_end:
            batch_photons_end = gpu_photons.get()
            
        if hasattr(self.detector, 'num_channels') and (keep_hits or keep_flat_hits):
            batch_hits = gpu_photons.get_flat_hits(self.gpu_geometry)

        for i,(batch_ev,(start_photon,end_photon)) in enumerate(zip(batch_events,zip(batch_bounds[:-1],batch_bounds[1:]))):
                    
            if not keep_photons_beg:
                batch_ev.photons_beg = None
                
            if self.photon_tracking:
                step_photon_ids,step_photons = tracking
                nphotons = end_photon-start_photon
                photon_tracks = [[] for i in range(nphotons)]
                for step_ids,step_photons in zip(step_photon_ids,step_photons):
                    mask = np.logical_and(step_ids >= start_photon,step_ids<end_photon)
                    if np.count_nonzero(mask) == 0:
                        break
                    photon_ids = step_ids[mask]-start_photon
                    photons = step_photons[mask]
                    #Indexing Photons with a scalar changes the internal array shapes...
                    any(photon_tracks[id].append(photons[i]) for i,id in enumerate(photon_ids))
                batch_ev.photon_tracks = [event.Photons.join(photons,concatenate=False) if len(photons) > 0 else event.Photons() for photons in photon_tracks]

            if keep_photons_end:
                batch_ev.photons_end = batch_photons_end[start_photon:end_photon]
                        
            if hasattr(self.detector, 'num_channels') and (keep_hits or keep_flat_hits):
                ev_hits = batch_hits[batch_hits.evidx == i]
                if keep_hits:
                    #This is kind of expensive computationally, but keep_hits is for diagnostics
                    batch_ev.hits = { int(chan):ev_hits[ev_hits.channel == chan] for chan in np.unique(ev_hits.channel) }
                if keep_flat_hits:
                    batch_ev.flat_hits = ev_hits
                    
                        
            if hasattr(self, 'gpu_daq') and run_daq:
                #Must run DAQ per event, or design a much more complicated daq algorithm
                self.gpu_daq.begin_acquire()
                self.gpu_daq.acquire(gpu_photons, self.rng_states, 
                                     start_photon=start_photon, 
                                     nphotons=(end_photon-start_photon),
                                     nthreads_per_block=self.nthreads_per_block, 
                                     max_blocks=self.max_blocks)
                gpu_channels = self.gpu_daq.end_acquire()
                batch_ev.channels = gpu_channels.get()
                    
            yield batch_ev

    @staticmethod
    def _is_gpu_photon_source(photons, copy_flags=True, copy_triangles=False, copy_weights=False):
        required_fields = ['pos', 'dir', 'pol', 'wavelengths', 't', 'evidx']
        if copy_flags:
            required_fields.append('flags')
        if copy_triangles:
            required_fields.append('last_hit_triangles')
        if copy_weights:
            required_fields.append('weights')
        for name in required_fields:
            arr = getattr(photons, name, None)
            if not isinstance(arr, ga.GPUArray):
                return False
        return True

    @classmethod
    def _stack_gpu_photon_sources(cls, photon_sources, copy_flags=True, copy_triangles=False, copy_weights=False):
        if not photon_sources or not all(cls._is_gpu_photon_source(ph, copy_flags, copy_triangles, copy_weights) for ph in photon_sources):
            return None

        total = sum(len(ph) for ph in photon_sources)
        reference = photon_sources[0]

        def _allocate(field):
            template = getattr(reference, field, None)
            if template is None:
                raise AttributeError(f"GPU photon input missing '{field}' field")
            return ga.empty(shape=total, dtype=template.dtype)

        arrays = {
            'pos': _allocate('pos'),
            'dir': _allocate('dir'),
            'pol': _allocate('pol'),
            'wavelengths': _allocate('wavelengths'),
            't': _allocate('t'),
            'evidx': _allocate('evidx'),
        }

        if copy_flags:
            arrays['flags'] = _allocate('flags')
        if copy_triangles:
            arrays['last_hit_triangles'] = _allocate('last_hit_triangles')
        if copy_weights:
            arrays['weights'] = _allocate('weights')

        def _copy_field(field, dest):
            if dest is None:
                return
            offset = 0
            element_size = dest.dtype.itemsize
            for photons in photon_sources:
                count = len(photons)
                if count == 0:
                    continue
                src = getattr(photons, field)
                dst_slice = dest[offset:offset+count]
                src_slice = src[:count]
                cuda.memcpy_dtod(dst_slice.gpudata, src_slice.gpudata, int(count * element_size))
                offset += count

        for field_name, dest_array in arrays.items():
            if dest_array is not None:
                _copy_field(field_name, dest_array)

        namespace_kwargs = {key: value for key, value in arrays.items() if value is not None}
        namespace_kwargs['true_nphotons'] = total

        return SimpleNamespace(**namespace_kwargs)

    def simulate(self, iterable, keep_photons_beg=False, keep_photons_end=False,
                 keep_hits=True, keep_flat_hits=True, run_daq=False, max_steps=1000,
                 photons_per_batch=1000000):
        if isinstance(iterable, event.Photons):
            first_element, iterable = iterable, [iterable]
        else:
            first_element, iterable = itertoolset.peek(iterable)

        if isinstance(first_element, event.Event):
            pass
        elif isinstance(first_element, event.Photons):
            iterable = (event.Event(photons_beg=x) for x in iterable)
        elif isinstance(first_element, event.Vertex):
            raise NotImplementedError("Vertex input not supported in Chroma")

        nphotons = 0
        batch_events = []
        
        for ev in iterable:
            
            ev.nphotons = len(ev.photons_beg)
            event_index = len(batch_events)
            evidx_field = getattr(ev.photons_beg, 'evidx', None)
            if evidx_field is not None:
                if isinstance(evidx_field, ga.GPUArray):
                    if ev.nphotons > 0:
                        evidx_slice = evidx_field[:ev.nphotons]
                        evidx_slice.fill(np.uint32(event_index))
                else:
                    evidx_field[:ev.nphotons] = np.uint32(event_index)
            
            nphotons += ev.nphotons
            batch_events.append(ev)
            
            #FIXME need an alternate implementation to split an event that is too large
            if nphotons >= photons_per_batch:
                yield from self._simulate_batch(batch_events,
                                                keep_photons_beg=keep_photons_beg,
                                                keep_photons_end=keep_photons_end,
                                                keep_hits=keep_hits,
                                                keep_flat_hits=keep_flat_hits,
                                                run_daq=run_daq, max_steps=max_steps,
                                                )
                nphotons = 0
                batch_events = []
                
        if len(batch_events) != 0:
            yield from self._simulate_batch(batch_events,
                                            keep_photons_beg=keep_photons_beg,
                                            keep_photons_end=keep_photons_end,
                                            keep_hits=keep_hits,
                                            keep_flat_hits=keep_flat_hits,
                                            run_daq=run_daq, max_steps=max_steps,
                                            )


    def __del__(self):
        self.context.pop()
