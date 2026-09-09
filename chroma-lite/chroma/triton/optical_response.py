"""Photon distributions and electronics response, independent of transport.

Lengths are mm, wavelengths nm, times ns, and photoelectron charges PE.
Random draws are keyed by photon ID so hit reordering and tiling do not change
the PMT response. No GPU or PyCUDA imports are required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _philox4x32(counter, key):
    """Random123 Philox4x32-10; integer arithmetic also checked by known vectors."""
    c0, c1, c2, c3 = np.broadcast_arrays(*[np.asarray(v, np.uint32) for v in counter])
    k0, k1 = (np.asarray(v, np.uint32) for v in key)
    with np.errstate(over="ignore"):
        for _ in range(10):
            a = c0.astype(np.uint64) * np.uint64(0xd2511f53)
            b = c2.astype(np.uint64) * np.uint64(0xcd9e8d57)
            c0, c1, c2, c3 = ((b >> np.uint64(32)).astype(np.uint32) ^ c1 ^ k0,
                              b.astype(np.uint32),
                              (a >> np.uint64(32)).astype(np.uint32) ^ c3 ^ k1,
                              a.astype(np.uint32))
            k0 = k0 + np.uint32(0x9e3779b9)
            k1 = k1 + np.uint32(0xbb67ae85)
    return c0, c1, c2, c3


def validate_seed(seed):
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    return int(seed)


def uniform(ids, seed=1, stream=0):
    """Open-interval Philox draws keyed by 64-bit IDs, stream, and 64-bit seed.

    This is the CPU counterpart of ``spectral_kernels.random_uniform``.
    Stream namespaces: transport < 2**27, source 2**28+, response 2**29+.
    """
    seed = validate_seed(seed)
    if not isinstance(stream, (int, np.integer)) or not 0 <= stream < 2**32:
        raise ValueError("stream must be an integer in [0, 2**32)")
    ids = np.asarray(ids, dtype=np.uint64)
    x, _, _, _ = _philox4x32((ids.astype(np.uint32),
                              (ids >> np.uint64(32)).astype(np.uint32), stream, 0),
                             (seed & 0xffffffff, seed >> 32))
    # Use 23 bits: the largest value remains strictly below 1 in float32.
    return ((x >> np.uint32(9)).astype(np.float32) + np.float32(0.5)) * np.float32(2.0**-23)


@dataclass(frozen=True)
class TabulatedCDF:
    """Tabulated CDF, or the exact integral of a piecewise-linear PDF.

    Direct CDF inputs interpolate probability linearly and support atoms.
    ``from_pdf`` retains the PDF slopes for exact integration and inversion.
    """

    x: np.ndarray
    cdf: np.ndarray
    provenance: str = "user supplied"
    density: Optional[np.ndarray] = None

    def __post_init__(self):
        x, cdf = (np.array(v, dtype=np.float64, copy=True) for v in (self.x, self.cdf))
        if x.ndim != 1 or len(x) < 2 or cdf.shape != x.shape:
            raise ValueError("CDF x and probability must be matching vectors of length >= 2")
        if not np.isfinite(x).all() or not np.isfinite(cdf).all():
            raise ValueError("CDF must be finite")
        if np.any(np.diff(x) < 0) or np.any(np.diff(cdf) < 0):
            raise ValueError("CDF coordinates and probability must be nondecreasing")
        if cdf[0] != 0 or cdf[-1] != 1:
            raise ValueError("CDF endpoints must be exactly 0 and 1")
        x.flags.writeable = cdf.flags.writeable = False
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "cdf", cdf)
        if self.density is not None:
            pdf = np.array(self.density, dtype=float, copy=True)
            if pdf.shape != x.shape or np.any(np.diff(x) <= 0) or np.any(pdf < 0) or not np.isfinite(pdf).all():
                raise ValueError("PDF density must be finite, nonnegative, and have increasing coordinates")
            integrals = np.r_[0, np.cumsum(np.diff(x)*(pdf[:-1]+pdf[1:])/2)]
            if not np.allclose(integrals, cdf, rtol=1e-6, atol=1e-7):
                raise ValueError("PDF density must integrate to the supplied CDF")
            pdf.flags.writeable = False
            object.__setattr__(self, "density", pdf)

    @classmethod
    def from_pdf(cls, x, density, provenance="user supplied"):
        x, density = np.asarray(x, float), np.asarray(density, float)
        if x.ndim != 1 or len(x) < 2 or density.shape != x.shape:
            raise ValueError("PDF must have matching coordinate and density vectors")
        if not np.isfinite(x).all() or not np.isfinite(density).all():
            raise ValueError("PDF must be finite")
        if np.any(np.diff(x) <= 0) or np.any(density < 0):
            raise ValueError("PDF coordinates must increase and density must be nonnegative")
        mass = np.diff(x) * (density[:-1] + density[1:]) / 2
        cdf = np.r_[0, np.cumsum(mass)]
        if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
            raise ValueError("PDF must have a finite positive integral")
        return cls(x, cdf / cdf[-1], provenance, density / cdf[-1])

    @classmethod
    def from_csv(cls, path, *, kind="pdf", provenance=None):
        values = np.loadtxt(Path(path), delimiter=",", comments="#")
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("distribution CSV must have two numeric columns")
        source = str(path) if provenance is None else provenance
        if kind == "pdf":
            return cls.from_pdf(values[:, 0], values[:, 1], source)
        if kind == "cdf":
            return cls(values[:, 0], values[:, 1], source)
        raise ValueError("kind must be 'pdf' or 'cdf'")

    def sample(self, u):
        u = np.asarray(u, dtype=float)
        if np.any(~np.isfinite(u)) or np.any((u < 0) | (u >= 1)):
            raise ValueError("CDF draws must be in [0, 1)")
        hi = np.clip(np.searchsorted(self.cdf, u, side="right"), 1, len(self.x) - 1)
        lo = hi - 1
        fraction = (u - self.cdf[lo]) / (self.cdf[hi] - self.cdf[lo])
        if self.density is not None:
            # Use the interval's normalized fraction, avoiding accumulated
            # CDF roundoff and cancellation for flat or decreasing densities.
            p0, p1 = self.density[lo], self.density[hi]
            root = np.sqrt(np.maximum(0, p0*p0 + fraction*(p1*p1-p0*p0)))
            fraction = np.divide(fraction*(p0+p1), p0+root,
                                 out=np.zeros_like(fraction), where=(p0+root) > 0)
        return self.x[lo] + fraction * (self.x[hi] - self.x[lo])

    def evaluate(self, x):
        """Evaluate the CDF, including the quadratic integral of PDF segments."""
        x = np.asarray(x, dtype=float)
        if self.density is None:
            return np.interp(x, self.x, self.cdf)
        lo = np.clip(np.searchsorted(self.x, x, side="right")-1, 0, len(self.x)-2)
        dx = np.clip(x-self.x[lo], 0, self.x[lo+1]-self.x[lo])
        slope = (self.density[lo+1]-self.density[lo])/(self.x[lo+1]-self.x[lo])
        value = self.cdf[lo] + dx*(self.density[lo]+.5*dx*slope)
        return np.where(x <= self.x[0], 0, np.where(x >= self.x[-1], 1, value))


def _channel_values(value, channels, name):
    values = np.asarray(value, dtype=float)
    if values.ndim > 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite scalar or channel vector")
    if values.ndim == 0:
        return np.full(len(channels), values.item())
    if len(channels) and channels.max() >= len(values):
        raise ValueError(f"{name} does not cover every hit channel")
    return values[channels]


@dataclass(frozen=True)
class OpticalHits:
    """Detected photons; IDs remain attached through response and digitization."""

    times: np.ndarray
    channels: np.ndarray
    photon_ids: np.ndarray
    event_indices: np.ndarray
    wavelengths: Optional[np.ndarray] = None

    def __post_init__(self):
        n = len(np.asarray(self.times))
        for name, dtype in (("times", float), ("channels", np.int32),
                            ("photon_ids", np.int64), ("event_indices", np.int64)):
            raw = np.asarray(getattr(self, name))
            if raw.shape != (n,) or not np.isfinite(raw).all():
                raise ValueError(f"hits.{name} must be a finite vector of length {n}")
            if name != "times" and (np.any(raw < 0) or np.any(raw != raw.astype(dtype))):
                raise ValueError(f"hits.{name} must contain nonnegative integers")
            array = np.array(raw, dtype=dtype, copy=True)
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        if len(np.unique(self.photon_ids)) != n:
            raise ValueError("hit photon IDs must be unique")
        if self.wavelengths is not None:
            wl = np.array(self.wavelengths, dtype=float, copy=True)
            if wl.shape != (n,) or np.any(~np.isfinite(wl) | (wl <= 0)):
                raise ValueError("hit wavelengths must be finite and positive")
            wl.flags.writeable = False
            object.__setattr__(self, "wavelengths", wl)

    def __len__(self):
        return len(self.times)


@dataclass(frozen=True)
class Photoelectrons:
    times: np.ndarray
    charges: np.ndarray
    channels: np.ndarray
    photon_ids: np.ndarray
    event_indices: np.ndarray
    optical_times: np.ndarray

    def __post_init__(self):
        hits = OpticalHits(self.times, self.channels, self.photon_ids, self.event_indices)
        for name in ("times", "channels", "photon_ids", "event_indices"):
            object.__setattr__(self, name, getattr(hits, name))
        for name in ("charges", "optical_times"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if values.shape != (len(hits),) or not np.isfinite(values).all():
                raise ValueError(f"photoelectron {name} must be a matching finite vector")
            if name == "charges" and np.any(values < 0):
                raise ValueError("photoelectron charges must be nonnegative")
            values.flags.writeable = False
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class PMTResponse:
    """Response conditional on optical detection; QE belongs in transport.

    Scalar or per-channel parameters are accepted. ``time_cdf`` is a residual
    transit-time distribution and replaces the Gaussian if provided. ``tts_sigma``
    is RMS in ns, not FWHM. ``charge_cdf`` is in PE and is scaled by ``gain``.
    """

    tts_sigma: object = 0.0
    transit_time: object = 0.0
    gain: object = 1.0
    collection_efficiency: object = 1.0
    time_cdf: Optional[TabulatedCDF] = None
    charge_cdf: Optional[TabulatedCDF] = None

    def __post_init__(self):
        for name in ("tts_sigma", "transit_time", "gain", "collection_efficiency"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim > 1 or value.size == 0 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite scalar or nonempty channel vector")
            if name != "transit_time" and np.any(value < 0):
                raise ValueError(f"{name} must be nonnegative")
            if name == "collection_efficiency" and np.any(value > 1):
                raise ValueError("collection_efficiency must be in [0,1]")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        for name in ("time_cdf", "charge_cdf"):
            distribution = getattr(self, name)
            if distribution is not None and not isinstance(distribution, TabulatedCDF):
                raise ValueError(f"{name} must be a TabulatedCDF")
        if self.charge_cdf is not None and self.charge_cdf.x[0] < 0:
            raise ValueError("single-photoelectron charge distribution must be nonnegative")

    def apply(self, hits: OpticalHits, *, seed=1):
        sigma = _channel_values(self.tts_sigma, hits.channels, "tts_sigma")
        mean = _channel_values(self.transit_time, hits.channels, "transit_time")
        gain = _channel_values(self.gain, hits.channels, "gain")
        ce = _channel_values(self.collection_efficiency, hits.channels, "collection_efficiency")
        if np.any(sigma < 0) or np.any(gain < 0) or np.any((ce < 0) | (ce > 1)):
            raise ValueError("sigma/gain must be nonnegative and collection efficiency in [0, 1]")
        ids = hits.photon_ids
        if self.time_cdf is None:
            z = np.sqrt(-2 * np.log(uniform(ids, seed, 0x20000000))) * np.cos(2*np.pi*uniform(ids, seed, 0x20000001))
            delay = mean + sigma * z
        else:
            delay = mean + self.time_cdf.sample(uniform(ids, seed, 0x20000000))
        charge = gain if self.charge_cdf is None else gain * self.charge_cdf.sample(uniform(ids, seed, 0x20000002))
        if np.any(charge < 0):
            raise ValueError("single-photoelectron charge distribution must be nonnegative")
        keep = uniform(ids, seed, 0x20000003) < ce
        return Photoelectrons((hits.times + delay)[keep], charge[keep], hits.channels[keep],
                              ids[keep], hits.event_indices[keep], hits.times[keep])


@dataclass(frozen=True)
class Waveforms:
    event_indices: np.ndarray
    sample_times: np.ndarray
    samples: np.ndarray  # [event, channel, sample], ADC units


def digitizer_parameters(*, event_indices, channel_count, start_ns, sample_period_ns,
                         sample_count, pulse_times_ns, pulse_adc_per_pe, baseline, noise_rms,
                         adc_bits):
    """Validate the common CPU/GPU digitizer contract and return host axes."""
    raw_events = np.asarray(event_indices)
    events = np.asarray(event_indices, dtype=np.int64)
    px, py = np.asarray(pulse_times_ns, float), np.asarray(pulse_adc_per_pe, float)
    if events.ndim != 1 or np.any(raw_events != events) or len(np.unique(events)) != len(events) or np.any(events < 0):
        raise ValueError("event_indices must be unique nonnegative integers")
    if (not isinstance(channel_count, (int, np.integer)) or not isinstance(sample_count, (int, np.integer))
            or channel_count <= 0 or sample_count <= 0 or sample_period_ns <= 0):
        raise ValueError("channel/sample counts and sample period must be positive")
    if not np.isfinite([start_ns, sample_period_ns, baseline, noise_rms]).all() or noise_rms < 0:
        raise ValueError("digitizer parameters must be finite, noise RMS nonnegative")
    if px.ndim != 1 or len(px) < 2 or py.shape != px.shape or np.any(np.diff(px) <= 0):
        raise ValueError("pulse template must have matching vectors and increasing times")
    if not np.isfinite(px).all() or not np.isfinite(py).all() or px[0] < 0:
        raise ValueError("pulse template must be finite and causal")
    if adc_bits is not None and (not isinstance(adc_bits, (int, np.integer)) or not 1 <= adc_bits <= 24):
        raise ValueError("adc_bits must be an integer in [1, 24]")
    times = start_ns + np.arange(sample_count) * sample_period_ns
    return events, px, py, times


def digitize(photoelectrons: Photoelectrons, *, event_indices, channel_count,
             start_ns, sample_period_ns, sample_count, pulse_times_ns,
             pulse_adc_per_pe, baseline=0.0, noise_rms=0.0, adc_bits=None, seed=1):
    """Evaluate an SPE pulse template at the actual PE arrival time.

    Pulse amplitudes are ADC counts per PE; no implicit pulse normalization is
    applied. Explicit event IDs preserve events with zero detected photons.
    """
    events, px, py, times = digitizer_parameters(event_indices=event_indices, channel_count=channel_count,
        start_ns=start_ns, sample_period_ns=sample_period_ns, sample_count=sample_count,
        pulse_times_ns=pulse_times_ns, pulse_adc_per_pe=pulse_adc_per_pe, baseline=baseline,
        noise_rms=noise_rms, adc_bits=adc_bits)
    samples = np.full((len(events), channel_count, sample_count), baseline, dtype=float)
    event_rows = {int(event): row for row, event in enumerate(events)}
    for t, q, ch, ev in zip(photoelectrons.times, photoelectrons.charges,
                           photoelectrons.channels, photoelectrons.event_indices):
        if int(ev) not in event_rows or ch < 0 or ch >= channel_count:
            raise ValueError("photoelectron refers to an unspecified event or channel")
        if not np.isfinite(t) or not np.isfinite(q) or q < 0:
            raise ValueError("photoelectron time/charge must be finite, charge nonnegative")
        # Only touch samples on which this pulse has support.
        lo = max(0, int(np.ceil((t + px[0] - start_ns) / sample_period_ns)))
        hi = min(sample_count, int(np.floor((t + px[-1] - start_ns) / sample_period_ns)) + 1)
        if hi > lo:
            samples[event_rows[int(ev)], ch, lo:hi] += q * np.interp(times[lo:hi] - t, px, py)
    if noise_rms:
        # Per-event RNG makes event batching/order irrelevant.
        for row, event in enumerate(events):
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(event), 2052]))
            samples[row] += rng.normal(0, noise_rms, samples[row].shape)
    if adc_bits is not None:
        samples = np.rint(np.clip(samples, 0, 2**adc_bits - 1)).astype(np.uint32)
    return Waveforms(events.copy(), times, samples)
