import numpy as np
from math import sqrt
from uncertainties import ufloat, unumpy
from itertools import islice, repeat
from chroma.tools import profile_if_possible, count_nonzero

class Likelihood(object):
    "Class to evaluate likelihoods for detector events."
    def __init__(self, sim, event=None, tbins=100, trange=(-0.5, 999.5), 
                 qbins=10, qrange=(-0.5, 49.5), time_only=True):
        """
        Args:
            - sim: chroma.sim.Simulation
                The simulation object used to simulate events and build pdfs.
            - event: chroma.event.Event, *optional*
                The detector event being reconstructed. If None, you must call
                set_event() before eval().
            - tbins: int, *optional*
                Number of time bins in PDF
            - trange: tuple of 2 floats, *optional*
                Min and max time to include in PDF
            - qbins: int, *optional*
                Number of charge bins in PDF
            - qrange: tuple of 2 floats, *optional*
                Min and max charge to include in PDF
            - time_only: bool, *optional*
                Only use time observable (instead of time and charge) in computing
                likelihood.  Defaults to True.
        """
        self.sim = sim
        self.tbins = tbins
        self.trange = trange
        self.qbins = qbins
        self.qrange = qrange
        self.time_only = time_only

        if event is not None:
            self.set_event(event)

    def set_event(self, event):
        "Set the detector event being reconstructed."
        self.event = event

    def eval_channel_vbin(self, vertex_generator, nevals, nreps=16, ndaq=50):
        '''Evaluate the hit probability and observable (time or
        time+charge) probability density for each channel using the
        variable bin window method.

        Returns: (array of hit probabilities, array of PDF values, 
                  array of PDF uncertainties)
        '''

        ntotal = nevals * nreps * ndaq

        vertex_generator = islice(vertex_generator, nevals)

        hitcount, pdf_prob, pdf_prob_uncert = \
            self.sim.eval_pdf(self.event.channels,
                              vertex_generator,
                              0.2, self.trange, 
                              1, self.qrange,
                              nreps=nreps,
                              ndaq=ndaq,
                              time_only=self.time_only,
                              min_bin_content=320)
        
        # Normalize probabilities
        hit_prob = hitcount.astype(np.float32) / ntotal

        # Set all zero or nan probabilities to limiting PDF value
        bad_value = (pdf_prob <= 0.0) | np.isnan(pdf_prob)
        if self.time_only:
            pdf_floor = 1.0 / (self.trange[1] - self.trange[0])
        else:
            pdf_floor = 1.0 / (self.trange[1] - self.trange[0]) / (self.qrange[1] - self.qrange[0])
        pdf_prob[bad_value] = pdf_floor
        pdf_prob_uncert[bad_value] = pdf_floor

        print('channels with no data:', (bad_value & self.event.channels.hit).astype(int).sum())

        return hit_prob, pdf_prob, pdf_prob_uncert
        
    def eval(self, vertex_generator, nevals, nreps=16, ndaq=50):
        """
        Return the negative log likelihood that the detector event set in the
        constructor or by set_event() was the result of a particle generated
        by `vertex_generator`.  If `nreps` set to > 1, each set of photon
        vertices will be propagated `nreps` times.
        """
        ntotal = nevals * nreps * ndaq

        hit_prob, pdf_prob, pdf_prob_uncert = \
            self.eval_channel_vbin(vertex_generator, nevals, nreps, ndaq)

        # NLL calculation: note that negation is at the end
        # Start with the probabilties of hitting (or not) the channels
        
        # Flip probability for channels in event that were not hit
        hit_prob[~self.event.channels.hit] = 1.0 - hit_prob[~self.event.channels.hit]
        # Apply a floor so that we don't take the log of zero
        hit_prob = np.maximum(hit_prob, 0.5 / ntotal)

        hit_channel_prob = np.log(hit_prob).sum()
        log_likelihood = ufloat((hit_channel_prob, 0.0))

        # Then include the probability densities of the observed
        # charges and times.
        log_likelihood += ufloat((np.log(pdf_prob[self.event.channels.hit]).sum(),
                                  0.0))
        
        return -log_likelihood

    def setup_kernel(self, vertex_generator, nevals, nreps, ndaq, oversample_factor):
        bandwidth_generator = islice(vertex_generator, nevals*oversample_factor)

        self.sim.setup_kernel(self.event.channels,
                              bandwidth_generator,
                              self.trange,
                              self.qrange,
                              nreps=nreps,
                              ndaq=ndaq,
                              time_only=self.time_only,
                              scale_factor=oversample_factor)

    @profile_if_possible
    def eval_kernel(self, vertex_generator, nevals, nreps=16, ndaq=50, navg=10):
        """
        Return the negative log likelihood that the detector event set in the
        constructor or by set_event() was the result of a particle generated
        by `vertex_generator`.  If `nreps` set to > 1, each set of photon
        vertices will be propagated `nreps` times.
        """
        ntotal = nevals * nreps * ndaq

        mom0 = 0
        mom1 = 0.0
        mom2 = 0.0
        for i in range(navg):
            kernel_generator = islice(vertex_generator, nevals)
            hitcount, pdf_prob, pdf_prob_uncert = \
                self.sim.eval_kernel(self.event.channels,
                                     kernel_generator,
                                     self.trange, 
                                     self.qrange,
                                     nreps=nreps,
                                     ndaq=ndaq,
                                     time_only=self.time_only)
        
            # Normalize probabilities and put a floor to keep the log finite
            hit_prob = hitcount.astype(np.float32) / ntotal
            hit_prob[self.event.channels.hit] = np.maximum(hit_prob[self.event.channels.hit], 0.5 / ntotal)

            # Set all zero or nan probabilities to limiting PDF value
            bad_value = (pdf_prob <= 0.0) | np.isnan(pdf_prob)
            if self.time_only:
                pdf_floor = 1.0 / (self.trange[1] - self.trange[0])
            else:
                pdf_floor = 1.0 / (self.trange[1] - self.trange[0]) / (self.qrange[1] - self.qrange[0])
            pdf_prob[bad_value] = pdf_floor
            pdf_prob_uncert[bad_value] = pdf_floor

            print('channels with no data:', (bad_value & self.event.channels.hit).astype(int).sum())

            # NLL calculation: note that negation is at the end
            # Start with the probabilties of hitting (or not) the channels
            log_likelihood = np.log(hit_prob[self.event.channels.hit]).sum() + np.log(1.0-hit_prob[~self.event.channels.hit]).sum()
            log_likelihood = 0.0 # FIXME: Skipping hit/not-hit probabilities for now

            # Then include the probability densities of the observed
            # charges and times.
            log_likelihood += np.log(pdf_prob[self.event.channels.hit]).sum()
            print('ll', log_likelihood)
            if np.isfinite(log_likelihood):
                mom0 += 1
                mom1 += log_likelihood
                mom2 += log_likelihood**2
        
        avg_like = mom1 / mom0
        rms_like = (mom2 / mom0 - avg_like**2)**0.5
        # Don't forget to return a negative log likelihood
        return ufloat((-avg_like, rms_like/sqrt(mom0)))

if __name__ == '__main__':
    from chroma.demo import detector as build_detector
    from chroma.sim import Simulation
    from chroma.generator import constant_particle_gun
    from chroma import tools
    import time

    tools.enable_debug_on_crash()

    detector = build_detector()
    sim = Simulation(detector, seed=0)

    event = next(sim.simulate(islice(constant_particle_gun('e-',(0,0,0),(1,0,0),100.0), 1)))

    print('nhit = %i' % count_nonzero(event.channels.hit))

    likelihood = Likelihood(sim, event)

    x = np.linspace(-10.0, 10.0, 100)
    l = []

    for pos in zip(x, repeat(0), repeat(0)):
        t0 = time.time()
        ev_vertex_iter = constant_particle_gun('e-',pos,(1,0,0),100.0)
        l.append(likelihood.eval(ev_vertex_iter, 1000))
        elapsed = time.time() - t0

        print('(%.1f, %.1f, %.1f), %s (%1.1f sec)' % \
            (pos[0], pos[1], pos[2], tools.ufloat_to_str(l[-1]), elapsed))

    import matplotlib.pyplot as plt

    plt.errorbar(x, [v.nominal_value for v in l], [v.std_dev() for v in l])
    plt.xlabel('X Position (m)')
    plt.ylabel('Negative Log Likelihood')
    plt.show()
