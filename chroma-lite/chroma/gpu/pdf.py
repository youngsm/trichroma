import numpy as np
from pycuda import gpuarray as ga
import pycuda.driver as cuda
from chroma.gpu.tools import get_cu_module, cuda_options, GPUFuncs, chunk_iterator
from chroma.tools import profile_if_possible, count_nonzero

class GPUKernelPDF(object):
    def __init__(self):
        self.module = get_cu_module('pdf.cu', options=cuda_options,
                                    include_source_directory=True)
        self.gpu_funcs = GPUFuncs(self.module)

    def setup_moments(self, nchannels, trange, qrange, time_only=True):
        """Setup GPU arrays to accumulate moments and eventually
        compute a kernel estimate of PDF values for each hit channel.

            trange: (float, float)
              Range of time dimension in PDF
            qrange: (float, float)
              Range of charge dimension in PDF
            time_only: bool
              If True, only the time observable will be used in the PDF.
        """
        self.hitcount_gpu = ga.zeros(nchannels, dtype=np.uint32)
        self.tmom1_gpu = ga.zeros(nchannels, dtype=np.float32)
        self.tmom2_gpu = ga.zeros(nchannels, dtype=np.float32)
        self.qmom1_gpu = ga.zeros(nchannels, dtype=np.float32)
        self.qmom2_gpu = ga.zeros(nchannels, dtype=np.float32)

        self.trange = trange
        self.qrange = qrange
        self.time_only = time_only

    def clear_moments(self):
        "Reset PDF evaluation counters to start accumulating new Monte Carlo."
        self.hitcount_gpu.fill(0)
        self.tmom1_gpu.fill(0.0)
        self.tmom2_gpu.fill(0.0)
        self.qmom1_gpu.fill(0.0)
        self.qmom2_gpu.fill(0.0)

    def accumulate_moments(self, gpuchannels, nthreads_per_block=64):
        """Add the most recent results of run_daq() to the accumulate of 
        moments for future bandwidth calculation."""
        self.gpu_funcs.accumulate_moments(np.int32(self.time_only),
                                          np.int32(len(gpuchannels.t)),
                                          gpuchannels.t,
                                          gpuchannels.q,
                                          np.float32(self.trange[0]),
                                          np.float32(self.trange[1]),
                                          np.float32(self.qrange[0]),
                                          np.float32(self.qrange[1]),
                                          self.hitcount_gpu,
                                          self.tmom1_gpu,
                                          self.tmom2_gpu,
                                          self.qmom1_gpu,
                                          self.qmom2_gpu,
                                          block=(nthreads_per_block,1,1), 
                                          grid=(len(gpuchannels.t)//nthreads_per_block+1,1))
        
    def compute_bandwidth(self, event_hit, event_time, event_charge, 
                          scale_factor=1.0):
        """Use the MC information accumulated by accumulate_moments() to
        estimate the best bandwidth to use when kernel estimating."""

        rho = 1.0

        hitcount = self.hitcount_gpu.get()
        mom0 = np.maximum(hitcount, 1)
        tmom1 = self.tmom1_gpu.get()
        tmom2 = self.tmom2_gpu.get()

        tmean = tmom1 / mom0
        tvar = np.maximum(tmom2 / mom0 - tmean**2, 0.0) # roundoff can go neg
        trms = tvar**0.5

        if self.time_only:
            d = 1
        else:
            d = 2
        dimensionality_factor = ((4.0/(d+2)) / (mom0/scale_factor))**(-1.0/(d+4))
        gaussian_density = np.minimum(1.0/trms, (1.0/np.sqrt(2.0*np.pi)) * np.exp(-0.5*((event_time - tmean)/trms))  / trms)
        time_bandwidths = dimensionality_factor / gaussian_density * rho
        inv_time_bandwidths = np.zeros_like(time_bandwidths)
        inv_time_bandwidths[time_bandwidths  > 0] = time_bandwidths[time_bandwidths > 0] ** -1

        # precompute inverse to speed up GPU evaluation
        self.inv_time_bandwidths_gpu = ga.to_gpu(
            inv_time_bandwidths.astype(np.float32)
            )

        # Compute charge bandwidths if needed
        if self.time_only:
            self.inv_charge_bandwidths_gpu = ga.empty_like(
                self.inv_time_bandwidths_gpu
                )
            self.inv_charge_bandwidths_gpu.fill(0.0)
        else:
            qmom1 = self.qmom1_gpu.get()
            qmom2 = self.qmom2_gpu.get()

            qmean = qmom1 / mom0
            qrms = (qmom2 / mom0 - qmean**2)**0.5

            gaussian_density = np.minimum(1.0/qrms, (1.0/np.sqrt(2.0*np.pi)) * np.exp(-0.5*((event_charge - qmean)/qrms))  / qrms)

            charge_bandwidths = dimensionality_factor / gaussian_density * rho

            # precompute inverse to speed up GPU evaluation
            self.inv_charge_bandwidths_gpu = ga.to_gpu( 
                (charge_bandwidths**-1).astype(np.float32)
                )

    def setup_kernel(self, event_hit, event_time, event_charge):
        """Setup GPU arrays to accumulate moments and eventually
        compute a kernel estimate of PDF values for each hit channel.

            event_hit: ndarray
              Hit or not-hit status for each channel in the detector.
            event_time: ndarray
              Hit time for each channel in the detector.  If channel 
              not hit, the time will be ignored.
            event_charge: ndarray
              Integrated charge for each channel in the detector.
              If channel not hit, the charge will be ignored.
        """
        self.event_hit_gpu = ga.to_gpu(event_hit.astype(np.uint32))
        self.event_time_gpu = ga.to_gpu(event_time.astype(np.float32))
        self.event_charge_gpu = ga.to_gpu(event_charge.astype(np.float32))
        self.hitcount_gpu.fill(0)
        self.time_pdf_values_gpu = ga.zeros(len(event_hit), dtype=np.float32)
        self.charge_pdf_values_gpu = ga.zeros(len(event_hit), dtype=np.float32)

    def clear_kernel(self):
        self.hitcount_gpu.fill(0)
        self.time_pdf_values_gpu.fill(0.0)
        self.charge_pdf_values_gpu.fill(0.0)
            
    def accumulate_kernel(self, gpuchannels, nthreads_per_block=64):
        "Add the most recent results of run_daq() to the kernel PDF evaluation."
        self.gpu_funcs.accumulate_kernel_eval(np.int32(self.time_only),
                                              np.int32(len(self.event_hit_gpu)),
                                              self.event_hit_gpu,
                                              self.event_time_gpu,
                                              self.event_charge_gpu,
                                              gpuchannels.t,
                                              gpuchannels.q,
                                              np.float32(self.trange[0]),
                                              np.float32(self.trange[1]),
                                              np.float32(self.qrange[0]),
                                              np.float32(self.qrange[1]),
                                              self.inv_time_bandwidths_gpu,
                                              self.inv_charge_bandwidths_gpu,
                                              self.hitcount_gpu,
                                              self.time_pdf_values_gpu,
                                              self.charge_pdf_values_gpu,
                                              block=(nthreads_per_block,1,1), 
                                              grid=(len(gpuchannels.t)//nthreads_per_block+1,1))


    def get_kernel_eval(self):
        hitcount = self.hitcount_gpu.get()
        hit = self.event_hit_gpu.get().astype(bool)
        time_pdf_values = self.time_pdf_values_gpu.get()
        time_pdf_values /= np.maximum(1, hitcount) # avoid divide by zero

        charge_pdf_values = self.charge_pdf_values_gpu.get()
        charge_pdf_values /= np.maximum(1, hitcount) # avoid divide by zero

        if self.time_only:
            pdf_values = time_pdf_values
        else:
            pdf_values = time_pdf_values * charge_pdf_values
        
        return hitcount, pdf_values, np.zeros_like(pdf_values)

class GPUPDF(object):
    def __init__(self):
        self.module = get_cu_module('pdf.cu', options=cuda_options,
                                    include_source_directory=True)
        self.gpu_funcs = GPUFuncs(self.module)

    def setup_pdf(self, nchannels, tbins, trange, qbins, qrange):
        """Setup GPU arrays to hold PDF information.

           nchannels: int, number of channels
           tbins: number of time bins
           trange: tuple of (min, max) time in PDF
           qbins: number of charge bins
           qrange: tuple of (min, max) charge in PDF
        """
        self.events_in_histogram = 0
        self.hitcount_gpu = ga.zeros(nchannels, dtype=np.uint32)
        self.pdf_gpu = ga.zeros(shape=(nchannels, tbins, qbins), 
                                      dtype=np.uint32)
        self.tbins = tbins
        self.trange = trange
        self.qbins = qbins
        self.qrange = qrange

    def clear_pdf(self):
        """Rezero the PDF counters."""
        self.hitcount_gpu.fill(0)
        self.pdf_gpu.fill(0)

    def add_hits_to_pdf(self, gpuchannels, nthreads_per_block=64):
        self.gpu_funcs.bin_hits(np.int32(len(self.hitcount_gpu)),
                                gpuchannels.q,
                                gpuchannels.t,
                                self.hitcount_gpu,
                                np.int32(self.tbins),
                                np.float32(self.trange[0]),
                                np.float32(self.trange[1]),
                                np.int32(self.qbins),
                                np.float32(self.qrange[0]),
                                np.float32(self.qrange[1]),
                                self.pdf_gpu,
                                block=(nthreads_per_block,1,1), 
                                grid=(len(gpuchannels.t)//nthreads_per_block+1,1))


        self.events_in_histogram += 1

    def get_pdfs(self):
        """Returns the 1D hitcount array and the 3D [channel, time, charge]
        histogram."""
        return self.hitcount_gpu.get(), self.pdf_gpu.get()

    def setup_pdf_eval(self, event_hit, event_time, event_charge, min_twidth,
                       trange, min_qwidth, qrange, min_bin_content=10,
                       time_only=True):
        """Setup GPU arrays to compute PDF values for the given event.
        The pdf_eval calculation allows the PDF to be evaluated at a
        single point for each channel as the Monte Carlo is run.  The
        effective bin size will be as small as (`min_twidth`,
        `min_qwidth`) around the point of interest, but will be large
        enough to ensure that `min_bin_content` Monte Carlo events
        fall into the bin.

            event_hit: ndarray
              Hit or not-hit status for each channel in the detector.
            event_time: ndarray
              Hit time for each channel in the detector.  If channel 
              not hit, the time will be ignored.
            event_charge: ndarray
              Integrated charge for each channel in the detector.
              If channel not hit, the charge will be ignored.

            min_twidth: float
              Minimum bin size in the time dimension
            trange: (float, float)
              Range of time dimension in PDF
            min_qwidth: float
              Minimum bin size in charge dimension
            qrange: (float, float)
              Range of charge dimension in PDF
            min_bin_content: int
              The bin will be expanded to include at least this many events
            time_only: bool
              If True, only the time observable will be used in the PDF.
        """
        self.event_nhit = count_nonzero(event_hit)
        
        # Define a mapping from an array of len(event_hit) to an array of length event_nhit
        self.map_hit_offset_to_channel_id = np.where(event_hit)[0].astype(np.uint32)
        self.map_hit_offset_to_channel_id_gpu = ga.to_gpu(self.map_hit_offset_to_channel_id)
        self.map_channel_id_to_hit_offset = np.maximum(0, event_hit.cumsum() - 1).astype(np.uint32)
        self.map_channel_id_to_hit_offset_gpu = ga.to_gpu(self.map_channel_id_to_hit_offset)

        self.event_hit_gpu = ga.to_gpu(event_hit.astype(np.uint32))
        self.event_time_gpu = ga.to_gpu(event_time.astype(np.float32))
        self.event_charge_gpu = ga.to_gpu(event_charge.astype(np.float32))

        self.eval_hitcount_gpu = ga.zeros(len(event_hit), dtype=np.uint32)
        self.eval_bincount_gpu = ga.zeros(len(event_hit), dtype=np.uint32)
        self.nearest_mc_gpu = ga.empty(shape=self.event_nhit * min_bin_content, 
                                             dtype=np.float32)
        self.nearest_mc_gpu.fill(1e9)
        
        self.min_twidth = min_twidth
        self.trange = trange
        self.min_qwidth = min_qwidth
        self.qrange = qrange
        self.min_bin_content = min_bin_content

        assert time_only # Only support time right now
        self.time_only = time_only

    def clear_pdf_eval(self):
        "Reset PDF evaluation counters to start accumulating new Monte Carlo."
        self.eval_hitcount_gpu.fill(0)
        self.eval_bincount_gpu.fill(0)
        self.nearest_mc_gpu.fill(1e9)

    @profile_if_possible
    def accumulate_pdf_eval(self, gpuchannels, nthreads_per_block=64, max_blocks=10000):
        "Add the most recent results of run_daq() to the PDF evaluation."
        self.work_queues = ga.empty(shape=self.event_nhit * (gpuchannels.ndaq+1), dtype=np.uint32)
        self.work_queues.fill(1)

        self.gpu_funcs.accumulate_bincount(np.int32(self.event_hit_gpu.size),
                                           np.int32(gpuchannels.ndaq),
                                           self.event_hit_gpu,
                                           self.event_time_gpu,
                                           gpuchannels.t,
                                           self.eval_hitcount_gpu,
                                           self.eval_bincount_gpu,
                                           np.float32(self.min_twidth),
                                           np.float32(self.trange[0]),
                                           np.float32(self.trange[1]),
                                           np.int32(self.min_bin_content),
                                           self.map_channel_id_to_hit_offset_gpu,
                                           self.work_queues,
                                           block=(nthreads_per_block,1,1), 
                                           grid=(self.event_hit_gpu.size//nthreads_per_block+1,1))
        cuda.Context.get_current().synchronize()

        self.gpu_funcs.accumulate_nearest_neighbor_block(np.int32(self.event_nhit),
                                                         np.int32(gpuchannels.ndaq),
                                                         self.map_hit_offset_to_channel_id_gpu,
                                                         self.work_queues,
                                                         self.event_time_gpu,
                                                         gpuchannels.t,
                                                         self.nearest_mc_gpu,
                                                         np.int32(self.min_bin_content),
                                                         block=(nthreads_per_block,1,1), 
                                                         grid=(self.event_nhit,1))
        cuda.Context.get_current().synchronize()

    def get_pdf_eval(self):
        evhit = self.event_hit_gpu.get().astype(bool)
        hitcount = self.eval_hitcount_gpu.get()
        bincount = self.eval_bincount_gpu.get()

        pdf_value = np.zeros(len(hitcount), dtype=float)
        pdf_frac_uncert = np.zeros_like(pdf_value)

        # PDF value for high stats bins
        high_stats = (bincount >= self.min_bin_content)
        if high_stats.any():
            if self.time_only:
                pdf_value[high_stats] = bincount[high_stats].astype(float) / hitcount[high_stats] / self.min_twidth
            else:
                assert Exception('Unimplemented 2D (time,charge) mode!')

            pdf_frac_uncert[high_stats] = 1.0/np.sqrt(bincount[high_stats])

        # PDF value for low stats bins
        low_stats = ~high_stats & (hitcount > 0) & evhit

        nearest_mc_by_hit = self.nearest_mc_gpu.get().reshape((self.event_nhit, self.min_bin_content))
        nearest_mc = np.empty(shape=(len(hitcount), self.min_bin_content), dtype=np.float32)
        nearest_mc.fill(1e9)
        nearest_mc[self.map_hit_offset_to_channel_id,:] = nearest_mc_by_hit

        # Deal with the case where we did not even get min_bin_content events
        # in the PDF but also clamp the lower range to ensure we don't index
        # by a negative number in 2 lines
        last_valid_entry = np.maximum(0, (nearest_mc < 1e9).astype(int).sum(axis=1) - 1)
        distance = nearest_mc[np.arange(len(last_valid_entry)),last_valid_entry]
        if low_stats.any():
            if self.time_only:
                pdf_value[low_stats] = (last_valid_entry[low_stats] + 1).astype(float) / hitcount[low_stats] / distance[low_stats] / 2.0
            else:
                assert Exception('Unimplemented 2D (time,charge) mode!')

            pdf_frac_uncert[low_stats] = 1.0/np.sqrt(last_valid_entry[low_stats] + 1)

        # PDFs with no stats got zero by default during array creation
        
        print('high_stats:', high_stats.sum(), 'low_stats', low_stats.sum())
        return hitcount, pdf_value, pdf_value * pdf_frac_uncert
