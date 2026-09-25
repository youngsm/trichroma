"""Triton implementation of :class:`chroma.sim.Simulation` (no PyCUDA).

The constructor and :meth:`Simulation.simulate` follow the installed Chroma
working tree (``chroma/sim_cuda.py``): the same arguments and defaults, the
same batching, ``evidx`` rewriting and event wrapping, and the same ``Event``
fields and dtypes. Propagation, random numbers and the DAQ response belong
to a :class:`chroma.triton.engine.api.TransportEngine`:

:class:`chroma.triton.engine.core.ProductionEngine`, in production mode by
default and in its exact (bitwise legacy) mode when
``CHROMA_TRITON_TAPE=replay:<dir>`` names a tape recorded by the CUDA backend.

Documented differences from the CUDA backend in production mode:

* hits are returned in photon order (the CUDA backend's order depends on
  warp-level atomics and varies from run to run);
* ``use_packed=True`` returns the true final photon states;
* random numbers are counter-based, so results do not depend on the
  batch size, ``nthreads_per_block`` or ``max_blocks``.

In exact mode ``use_packed=True`` reproduces the CUDA backend instead:
``photons_end`` carries the *initial* position, direction, polarization,
wavelength, time and weight words (``GPUPhotons.get()`` reads the unpacked
arrays, which ``propagate_packed`` never updates), and the DAQ reads the
initial times and weights unless hits were extracted first
(``get_flat_hits()`` syncs the packed state back).
"""

import os
import time

import numpy as np

from chroma import event
from chroma import itertoolset
from chroma.backend import tape_mode
from chroma.triton.engine.api import SURFACE_DETECT, DevicePhotons, to_device


def pick_seed():
    """Returns a seed for a random number generator selected using
    a mixture of the current time and the current process ID."""
    return int(time.time()) ^ (os.getpid() << 16) & 2**32 - 1


def _torch():
    import torch

    return torch


class _RawDeviceBytes(object):
    """Expose a device buffer as bytes via ``__cuda_array_interface__``."""

    def __init__(self, pointer, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1",
                                         "data": (pointer, False), "version": 2}


def _device_array(value):
    """A torch view of a CUDA array, or ``None`` for host data.

    Accepts torch tensors and any object with ``__cuda_array_interface__``,
    including PyCUDA ``GPUArray`` of ``float3`` (exposed as raw bytes).
    """
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value if value.is_cuda else None
    iface = getattr(value, "__cuda_array_interface__", None)
    if iface is None:
        return None
    if iface.get("strides") not in (None, ()):
        itemsize = int(iface["typestr"][2:])
        expected = []
        stride = itemsize
        for extent in reversed(iface["shape"]):
            expected.insert(0, stride)
            stride *= extent
        if tuple(iface["strides"]) != tuple(expected):
            raise ValueError("non-contiguous device photon arrays are not supported")
    kind = iface["typestr"][1]
    if kind == "V":  # structured element (e.g. PyCUDA float3): raw bytes
        nbytes = int(np.prod(iface["shape"])) * int(iface["typestr"][2:])
        return torch.as_tensor(_RawDeviceBytes(iface["data"][0], nbytes), device="cuda")
    return torch.as_tensor(value, device="cuda")


def _photon_count(photons):
    try:
        return len(photons)
    except TypeError:
        pass
    count = getattr(photons, "true_nphotons", None)
    if count is not None:
        return int(count)
    return len(photons.pos)


def _as_field(value, count, device, dtype, width=None):
    """Copy one photon field (host or device) to a torch tensor of ``dtype``.

    Integer fields hold uint32 bit patterns in int32 tensors.
    """
    torch = _torch()
    shape = (count,) if width is None else (count, width)
    elements = count * (width or 1)
    gpu = _device_array(value)
    if gpu is not None:
        gpu = gpu.reshape(-1)
        if gpu.dtype == torch.uint8:
            gpu = gpu.view(dtype)
        elif gpu.dtype != dtype:
            gpu = gpu.to(torch.int64).to(dtype) if not dtype.is_floating_point else gpu.to(dtype)
        return gpu[:elements].reshape(shape).to(device)
    array = np.asarray(value)
    if dtype.is_floating_point:
        array = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
    elif array.dtype in (np.uint32, np.int32):
        array = np.ascontiguousarray(array).view(np.int32).reshape(-1)
    else:
        array = np.ascontiguousarray(array.astype(np.int64) & 0xFFFFFFFF, dtype=np.uint32).view(np.int32).reshape(-1)
    return to_device(array[:elements].reshape(shape), device)


class _LazyHits(dict):
    """``Event.hits`` built on first use: the per-channel split of the flat
    hits costs about as much as the flat hits themselves for large events,
    and many callers (e.g. LUT generation) only read ``flat_hits``."""

    def __init__(self, flat):
        super().__init__()
        self._flat = flat

    def _fill(self):
        flat = self.__dict__.pop("_flat", None)
        if flat is not None:
            dict.update(self, _hits_by_channel(flat))


def _lazy(name):
    method = getattr(dict, name)

    def wrapper(self, *args, **kwargs):
        self._fill()
        return method(self, *args, **kwargs)

    wrapper.__name__ = name
    return wrapper


for _name in ("__getitem__", "__iter__", "__len__", "__contains__", "__repr__", "__eq__", "__ne__", "__reversed__",
              "keys", "values", "items", "get", "copy", "pop", "popitem", "setdefault", "update", "clear",
              "__setitem__", "__delitem__", "__or__", "__ior__"):
    if hasattr(dict, _name):
        setattr(_LazyHits, _name, _lazy(_name))
_LazyHits.__bool__ = lambda self: self.__len__() > 0


def _hits_by_channel(hits):
    """``{channel: hits[hits.channel == channel]}`` for every channel in
    ``np.unique`` order, photons in their original order (one stable sort
    instead of one boolean mask per channel)."""
    if len(hits) == 0:
        return {}
    order = np.argsort(hits.channel, kind="stable")
    ordered = hits[order]
    channels, starts = np.unique(ordered.channel, return_index=True)
    stops = np.append(starts[1:], len(ordered))
    return {int(c): ordered[a:b] for c, a, b in zip(channels, starts, stops)}


class Simulation(object):
    def __init__(self, detector, seed=None, cuda_device=None, photon_tracking=False,
                 nthreads_per_block=512, max_blocks=1024, use_packed=False):
        torch = _torch()
        self.detector = detector
        self.nthreads_per_block = nthreads_per_block
        self.max_blocks = max_blocks
        self.photon_tracking = photon_tracking
        self.use_packed = use_packed

        if seed is None:
            self.seed = pick_seed()
        else:
            self.seed = seed

        # Same global side effect as the CUDA backend.
        np.random.seed(self.seed)

        override = os.environ.get("CHROMA_TRITON_DEVICE")
        if override is not None:
            cuda_device = int(override)
        if cuda_device is None:
            cuda_device = torch.cuda.current_device()
        self.device = torch.device("cuda", cuda_device)

        self.has_channels = hasattr(detector, "num_channels")
        if self.has_channels:
            # GPUDaq refuses a detector without channels.
            assert detector.num_channels() > 0, "Geometry has no detectors, DAQ can't be initialized."

        from chroma.triton.engine.core import ProductionEngine

        self.tape = tape_mode()
        if self.tape.enabled:
            if self.tape.mode == "record":
                raise ValueError("CHROMA_TRITON_TAPE=record applies to the CUDA backend; use replay:<dir>")
            # Exact mode: the tape must come from a Simulation with these parameters.
            self.engine = ProductionEngine(
                detector, seed=self.seed, device=self.device, tape=self.tape,
                simulation=dict(nthreads_per_block=nthreads_per_block, max_blocks=max_blocks,
                                photon_tracking=photon_tracking, use_packed=use_packed))
        else:
            self.engine = ProductionEngine(detector, seed=self.seed, device=self.device)

        # Batch-independent photon ids for counter-based random numbers.
        self._next_photon_id = 0
        # Attributes that scripts read from the CUDA backend.
        self.gpu_geometry = self.engine
        self.rng_states = None

    # ------------------------------------------------------------------ batch

    def _gather(self, photon_sources):
        """Concatenate the batch's photon sources into one DevicePhotons."""
        torch = _torch()
        counts = [_photon_count(p) for p in photon_sources]
        total = int(sum(counts))
        photons = DevicePhotons.empty(total, self.device)
        offset = 0
        for source, count in zip(photon_sources, counts):
            if count == 0:
                continue
            rows = slice(offset, offset + count)
            photons.pos[rows] = _as_field(source.pos, count, self.device, torch.float32, 3)
            photons.dir[rows] = _as_field(source.dir, count, self.device, torch.float32, 3)
            photons.pol[rows] = _as_field(source.pol, count, self.device, torch.float32, 3)
            photons.wavelengths[rows] = _as_field(source.wavelengths, count, self.device, torch.float32)
            photons.t[rows] = _as_field(source.t, count, self.device, torch.float32)
            photons.evidx[rows] = _as_field(source.evidx, count, self.device, torch.int32)
            # copy_flags=True, copy_triangles=False, copy_weights=False:
            # last_hit_triangles=-1 and weights=1 as in the CUDA backend.
            photons.flags[rows] = _as_field(source.flags, count, self.device, torch.int32)
            offset += count
        photons.ids.copy_(torch.arange(self._next_photon_id, self._next_photon_id + total,
                                       device=self.device, dtype=torch.int64))
        self._next_photon_id += total
        return photons

    def _flat_hits(self, photons):
        """Detected photons with channel index, in photon order (numpy Photons)."""
        torch = _torch()
        tri = photons.last_hit_triangles
        detected = ((photons.flags & SURFACE_DETECT) != 0) & (tri >= 0)
        solid = torch.where(detected, self.engine.triangle_solid(tri.clamp(min=0)), 0)
        channel = torch.where(detected, self.engine.solid_id_to_channel_index[solid.long()], -1)
        rows = torch.nonzero(detected & (channel >= 0)).flatten()
        hits = photons.select(rows).to_numpy(skip=("ids",))
        return event.Photons(hits["pos"], hits["dir"], hits["pol"], hits["wavelengths"], hits["t"],
                             hits["last_hit_triangles"], hits["flags"], hits["weights"],
                             hits["evidx"], channel[rows].cpu().numpy().astype(np.int32))

    def _simulate_batch(self, batch_events, keep_photons_beg=False, keep_photons_end=False,
                        keep_hits=True, keep_flat_hits=True, run_daq=False, max_steps=100,
                        use_weights=False, verbose=False):
        '''Assumes batch_events is a list of Event objects with photons_beg having evidx set to the index in the array.

           Yields the fully formed events. Do not call directly.'''
        photon_sources = [ev.photons_beg for ev in batch_events]
        batch_bounds = np.cumsum(np.concatenate([[0], [_photon_count(src) for src in photon_sources]]))
        photons = self._gather(photon_sources)
        # Exact mode with use_packed: keep the initial words the CUDA backend returns.
        packed_view = self.use_packed and getattr(self.engine, "exact", None) is not None
        initial = photons.clone() if packed_view else None
        tracking = self.engine.propagate(photons, max_steps=max_steps, use_weights=use_weights,
                                         track=self.photon_tracking)

        if keep_photons_end:
            end = photons.to_numpy(skip=("ids",))
            if packed_view:
                start = initial.to_numpy(skip=("ids",))
                for name in ("pos", "dir", "pol", "wavelengths", "t", "weights"):
                    end[name] = start[name]
            batch_photons_end = event.Photons(end["pos"], end["dir"], end["pol"], end["wavelengths"],
                                              end["t"], end["last_hit_triangles"], end["flags"],
                                              end["weights"], end["evidx"])
        if self.has_channels and (keep_hits or keep_flat_hits):
            batch_hits = self._flat_hits(photons)
            # Hits are in photon order and events occupy consecutive rows, so
            # evidx is non-decreasing: each event's hits are one slice.
            hit_bounds = None
            if np.all(batch_hits.evidx[1:] >= batch_hits.evidx[:-1]):
                hit_bounds = np.searchsorted(batch_hits.evidx, np.arange(len(batch_events) + 1), side="left")
        daq_photons = photons
        if packed_view and not (self.has_channels and (keep_hits or keep_flat_hits)):
            daq_photons = photons.select(slice(None))
            daq_photons.t = initial.t
            daq_photons.weights = initial.weights

        for i, (batch_ev, (start_photon, end_photon)) in enumerate(zip(batch_events, zip(batch_bounds[:-1], batch_bounds[1:]))):
            if not keep_photons_beg:
                batch_ev.photons_beg = None

            if self.photon_tracking:
                nphotons = end_photon - start_photon
                photon_tracks = [[] for _ in range(nphotons)]
                for step_ids, step_photons in tracking:
                    step_ids = step_ids.cpu().numpy()
                    mask = np.logical_and(step_ids >= start_photon, step_ids < end_photon)
                    if np.count_nonzero(mask) == 0:
                        break
                    photon_ids = step_ids[mask] - start_photon
                    state = step_photons.select(np.flatnonzero(mask)).to_numpy()
                    snapshot = event.Photons(state["pos"], state["dir"], state["pol"], state["wavelengths"],
                                             state["t"], state["last_hit_triangles"], state["flags"],
                                             state["weights"], state["evidx"])
                    for j, pid in enumerate(photon_ids):
                        photon_tracks[pid].append(snapshot[j])
                batch_ev.photon_tracks = [event.Photons.join(p, concatenate=False) if len(p) > 0 else event.Photons()
                                          for p in photon_tracks]

            if keep_photons_end:
                batch_ev.photons_end = batch_photons_end[start_photon:end_photon]

            if self.has_channels and (keep_hits or keep_flat_hits):
                if hit_bounds is not None:
                    ev_hits = batch_hits[hit_bounds[i]:hit_bounds[i + 1]]
                else:
                    ev_hits = batch_hits[batch_hits.evidx == i]
                if keep_hits:
                    batch_ev.hits = _LazyHits(ev_hits)
                if keep_flat_hits:
                    batch_ev.flat_hits = ev_hits

            if self.has_channels and run_daq:
                channels = self.engine.acquire(daq_photons, int(start_photon), int(end_photon - start_photon))
                batch_ev.channels = event.Channels(channels.t < 1e8, channels.t, channels.q, channels.flags)

            yield batch_ev

    # --------------------------------------------------------------- simulate

    def simulate(self, iterable, keep_photons_beg=False, keep_photons_end=False,
                 keep_hits=True, keep_flat_hits=True, run_daq=False, max_steps=1000,
                 use_weights=False, photons_per_batch=1000000):
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

            ev.nphotons = _photon_count(ev.photons_beg)
            event_index = len(batch_events)
            evidx_field = getattr(ev.photons_beg, 'evidx', None)
            if evidx_field is not None:
                gpu = _device_array(evidx_field)
                if gpu is not None:
                    if ev.nphotons > 0:
                        gpu.reshape(-1)[:ev.nphotons].fill_(event_index)
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
                                                use_weights=use_weights,
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
                                            use_weights=use_weights,
                                            )
