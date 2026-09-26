"""Triton implementation of :class:`chroma.sim.Simulation` (no PyCUDA).

The constructor and :meth:`Simulation.simulate` follow the installed Chroma
working tree (``chroma/sim_cuda.py``): the same arguments and defaults, the
same batching, ``evidx`` rewriting and event wrapping, and the same ``Event``
fields and dtypes. Propagation, random numbers and the DAQ response belong
to a :class:`trichroma.engine.api.TransportEngine`:

:class:`trichroma.engine.core.ProductionEngine`, in production mode by
default and in its exact (bitwise legacy) mode when
``CHROMA_TRITON_TAPE=replay:<dir>`` names a tape recorded by the CUDA backend.

Documented differences from the CUDA backend in production mode:

* hits are returned in photon order (the CUDA backend's order depends on
  warp-level atomics and varies from run to run);
* ``use_packed=True`` returns the true final photon states;
* random numbers are counter-based, so results do not depend on the
  batch size, ``nthreads_per_block`` or ``max_blocks``;
* :meth:`Simulation.simulate` reads one batch ahead: batch k+1 is taken from
  the input and propagated on the GPU while the caller consumes the events of
  batch k (the results are unchanged; ``CHROMA_TRITON_PIPELINE=0`` restores
  the CUDA backend's order of reading input and yielding events).

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
from trichroma.engine.api import SURFACE_DETECT, DevicePhotons, to_device


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


# Photon fields copied to the host, in order, with their widths and host dtypes.
_PHOTON_FIELDS = (("pos", 3, np.float32), ("dir", 3, np.float32), ("pol", 3, np.float32),
                  ("wavelengths", 1, np.float32), ("t", 1, np.float32), ("last_hit_triangles", 1, np.int32),
                  ("flags", 1, np.uint32), ("weights", 1, np.float32), ("evidx", 1, np.uint32))


def _float_words(tensor):
    """``tensor`` flattened, 32-bit words reinterpreted as float32 (no copy)."""
    torch = _torch()
    flat = tensor.reshape(-1)
    return flat if flat.dtype == torch.float32 else flat.view(torch.float32)


def _photon_words(photons, channel=None):
    """Field-major float32 words of DevicePhotons (and a channel column)."""
    words = [_float_words(getattr(photons, name)) for name, _, _ in _PHOTON_FIELDS]
    if channel is not None:
        words.append(_float_words(channel))
    return words


def _unpack_photons(host, offset, count, channel=False):
    """event.Photons viewing the field-major words at ``host[offset:]``;
    returns (photons, offset after them)."""
    fields = []
    for _, width, dtype in _PHOTON_FIELDS + ((("channel", 1, np.uint32),) if channel else ()):
        words = host[offset:offset + count * width]
        offset += count * width
        words = words if dtype == np.float32 else words.view(dtype)
        fields.append(words.reshape(count, width) if width > 1 else words)
    return event.Photons(*fields), offset


class _Batch(object):
    """A batch in flight: its events, photon rows and, once extracted, the
    page-locked host copy of what the events need."""

    def __init__(self, events, bounds, photons):
        self.events = events
        self.bounds = bounds
        self.photons = photons
        self.nhits = 0
        self.daq_shape = None
        self.flat = None  # packed device words, until copied
        self.packed = None
        self.host = None
        self.ready = None


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

        from trichroma.engine.core import ProductionEngine

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
        self._copy_stream = None  # device->host copies of finished batches
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
        sources = [(source, count) for source, count in zip(photon_sources, counts) if count > 0]
        # copy_flags=True, copy_triangles=False, copy_weights=False:
        # last_hit_triangles=-1 and weights=1 as in the CUDA backend.
        for name, dtype, width in (("pos", torch.float32, 3), ("dir", torch.float32, 3), ("pol", torch.float32, 3),
                                   ("wavelengths", torch.float32, None), ("t", torch.float32, None),
                                   ("evidx", torch.int32, None), ("flags", torch.int32, None)):
            parts = [_as_field(getattr(source, name), count, self.device, dtype, width) for source, count in sources]
            if len(parts) == 1:
                getattr(photons, name).copy_(parts[0])
            elif parts:
                torch.cat(parts, out=getattr(photons, name))
        torch.arange(self._next_photon_id, self._next_photon_id + total, out=photons.ids)
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

    # ------------------------------------------------------ pipelined batches
    #
    # Production mode overlaps the GPU with the caller: batch k+1 is gathered
    # and propagated while the caller consumes the events of batch k. What
    # the events of batch k need (hits, final photons, DAQ words) is packed
    # on the device before batch k+1 starts and copied to page-locked host
    # memory on a separate stream while it runs.

    def _pipelined(self):
        return (getattr(self.engine, "exact", None) is None and not self.photon_tracking
                and os.environ.get("CHROMA_TRITON_PIPELINE", "1") not in ("", "0"))

    def _pack(self, batch, keep_photons_end, want_hits, run_daq):
        """Pack everything the batch's events need into one device vector
        (waits for the batch's propagation: the hit count sizes it)."""
        torch = _torch()
        photons = batch.photons
        words = []
        if want_hits:
            # Detected photons (their channel is looked up here; the few
            # without one are dropped on the host, see _emit).
            tri = photons.last_hit_triangles
            rows = torch.nonzero(((photons.flags & SURFACE_DETECT) != 0) & (tri >= 0)).flatten()
            batch.nhits = int(rows.numel())
            channel = self.engine.solid_id_to_channel_index[self.engine.triangle_solid(tri[rows])]
            words += _photon_words(photons.select(rows), channel)
        if keep_photons_end:
            words += _photon_words(photons)
        if run_daq:
            daq = self.engine.acquire_batch(photons, batch.bounds)
            batch.daq_shape = daq[0].shape
            words += [_float_words(x) for x in daq]
        if not words:
            return
        batch.flat = torch.cat(words) if len(words) > 1 else words[0]
        batch.packed = torch.cuda.Event()
        batch.packed.record(torch.cuda.current_stream(self.device))

    def _copy(self, batch):
        """Queue the device->host copy of the packed words on the copy stream.

        Called after the next batch was launched: the next propagation does
        not wait for the copy, and a slow page-locked allocation (a new block
        costs ~1 ms per MB) holds up only the host.
        """
        torch = _torch()
        if batch.flat is None:
            return
        stream = self._copy_stream
        if stream is None:
            stream = self._copy_stream = torch.cuda.Stream(device=self.device)
        batch.host = torch.empty(batch.flat.shape, dtype=torch.float32, pin_memory=True)
        with torch.cuda.stream(stream):
            stream.wait_event(batch.packed)
            batch.host.copy_(batch.flat, non_blocking=True)
            batch.ready = torch.cuda.Event()
            batch.ready.record(stream)
        batch.flat.record_stream(stream)  # the allocator must not reuse it before the copy is done
        batch.flat = None

    def _emit(self, batch, keep_photons_beg, keep_photons_end, keep_hits, keep_flat_hits, want_hits, run_daq):
        """Yield the batch's events once its host copy has landed."""
        host = None
        if batch.host is not None:
            batch.ready.synchronize()
            host = batch.host.numpy()
        offset = 0
        if want_hits:
            batch_hits, offset = _unpack_photons(host, offset, batch.nhits, channel=True)
            no_channel = batch_hits.channel.view(np.int32) < 0
            if no_channel.any():
                batch_hits = batch_hits[~no_channel]
            if np.all(batch_hits.evidx[1:] >= batch_hits.evidx[:-1]):
                hit_bounds = np.searchsorted(batch_hits.evidx, np.arange(len(batch.events) + 1), side="left")
            else:
                hit_bounds = None
        if keep_photons_end:
            batch_photons_end, offset = _unpack_photons(host, offset, len(batch.photons))
        if run_daq:
            size = int(np.prod(batch.daq_shape))
            time_bits, q, hist = (host[offset + k * size:offset + (k + 1) * size].view(np.int32).reshape(batch.daq_shape)
                                  for k in range(3))
        bounds = batch.bounds
        for i, batch_ev in enumerate(batch.events):
            if not keep_photons_beg:
                batch_ev.photons_beg = None
            if keep_photons_end:
                batch_ev.photons_end = batch_photons_end[bounds[i]:bounds[i + 1]]
            if want_hits:
                if hit_bounds is not None:
                    ev_hits = batch_hits[hit_bounds[i]:hit_bounds[i + 1]]
                else:
                    ev_hits = batch_hits[batch_hits.evidx == i]
                if keep_hits:
                    batch_ev.hits = _LazyHits(ev_hits)
                if keep_flat_hits:
                    batch_ev.flat_hits = ev_hits
            if run_daq:
                channels = self.engine.daq_channels(time_bits[i], q[i], hist[i])
                batch_ev.channels = event.Channels(channels.t < 1e8, channels.t, channels.q, channels.flags)
            yield batch_ev

    def _simulate_pipelined(self, batches, keep_photons_beg, keep_photons_end, keep_hits, keep_flat_hits,
                            run_daq, max_steps, use_weights):
        want_hits = self.has_channels and (keep_hits or keep_flat_hits)
        run_daq = self.has_channels and run_daq
        previous = None
        for batch_events in batches:
            sources = [ev.photons_beg for ev in batch_events]
            bounds = np.cumsum(np.concatenate([[0], [_photon_count(src) for src in sources]])).astype(np.int64)
            batch = _Batch(batch_events, bounds, self._gather(sources))
            del sources
            if not keep_photons_beg:
                # The input photons are copied: let them go now rather than
                # when the events are yielded (GPU-drawn sources hold memory).
                for batch_ev in batch_events:
                    batch_ev.photons_beg = None
            if previous is not None:
                self._pack(previous, keep_photons_end, want_hits, run_daq)
            self.engine.propagate(batch.photons, max_steps=max_steps, use_weights=use_weights)
            if previous is not None:
                self._copy(previous)
                yield from self._emit(previous, keep_photons_beg, keep_photons_end, keep_hits, keep_flat_hits,
                                      want_hits, run_daq)
            previous = batch
        if previous is not None:
            self._pack(previous, keep_photons_end, want_hits, run_daq)
            self._copy(previous)
            yield from self._emit(previous, keep_photons_beg, keep_photons_end, keep_hits, keep_flat_hits,
                                  want_hits, run_daq)

    # --------------------------------------------------------------- simulate

    def _batches(self, iterable, photons_per_batch):
        """The CUDA backend's batching: lists of events of at least
        ``photons_per_batch`` photons (the last one may be smaller), with
        ``evidx`` set to the index in the batch."""
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
                yield batch_events
                nphotons = 0
                batch_events = []

        if len(batch_events) != 0:
            yield batch_events

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

        batches = self._batches(iterable, photons_per_batch)
        if self._pipelined():
            yield from self._simulate_pipelined(batches, keep_photons_beg, keep_photons_end, keep_hits,
                                                keep_flat_hits, run_daq, max_steps, use_weights)
            return
        for batch_events in batches:
            yield from self._simulate_batch(batch_events,
                                            keep_photons_beg=keep_photons_beg,
                                            keep_photons_end=keep_photons_end,
                                            keep_hits=keep_hits,
                                            keep_flat_hits=keep_flat_hits,
                                            run_daq=run_daq, max_steps=max_steps,
                                            use_weights=use_weights,
                                            )
