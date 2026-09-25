"""Contract between :mod:`chroma.triton.compat` and the transport engines.

Every engine (the production engine in :mod:`chroma.triton.engine` and the
bitwise legacy engine in :mod:`chroma.triton.legacy`) implements
:class:`TransportEngine`. The compatibility layer owns batching, event
bookkeeping, hit extraction and host conversion; engines own geometry
compilation, random numbers, propagation and the DAQ response.

Photon state lives on one CUDA device as a :class:`DevicePhotons` of
structure-of-arrays torch tensors. Engines update it in place.
"""

from dataclasses import dataclass, fields
from typing import List, Optional, Protocol, Tuple

import numpy as np

# Chroma photon history bits (chroma/event.py). The production engine keeps
# all 32 bits; the original device code stores 16 and uses 1<<15 for NaN.
NO_HIT = 1 << 0
BULK_ABSORB = 1 << 1
SURFACE_DETECT = 1 << 2
SURFACE_ABSORB = 1 << 3
RAYLEIGH_SCATTER = 1 << 4
REFLECT_DIFFUSE = 1 << 5
REFLECT_SPECULAR = 1 << 6
SURFACE_REEMIT = 1 << 7
SURFACE_TRANSMIT = 1 << 8
BULK_REEMIT = 1 << 9
CHERENKOV = 1 << 10
SCINTILLATION = 1 << 11
NAN_ABORT = 1 << 31
# Photons with any of these bits are not propagated (W propagate.cu:295 also
# skips bit 15, the device NaN flag).
TERMINAL = NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT
DEVICE_NAN_ABORT_16 = 1 << 15

# last_hit_triangles value for an analytic wire (W photon.h).
WIRE_HIT = -2


@dataclass
class DevicePhotons:
    """Photon state on one CUDA device.

    Shapes/dtypes: ``pos``, ``dir``, ``pol`` [N,3] float32; ``wavelengths``,
    ``t``, ``weights`` [N] float32; ``last_hit_triangles`` [N] int32;
    ``flags``, ``evidx`` [N] int32 holding the uint32 bit patterns;
    ``ids`` [N] int64 global photon ids (keys for counter-based RNG; unique
    over the lifetime of a Simulation).
    """

    pos: "object"
    dir: "object"
    pol: "object"
    wavelengths: "object"
    t: "object"
    last_hit_triangles: "object"
    flags: "object"
    weights: "object"
    evidx: "object"
    ids: "object"

    def __len__(self):
        return int(self.wavelengths.shape[0])

    @classmethod
    def empty(cls, count, device):
        import torch

        f = dict(device=device, dtype=torch.float32)
        return cls(
            pos=torch.empty((count, 3), **f),
            dir=torch.empty((count, 3), **f),
            pol=torch.empty((count, 3), **f),
            wavelengths=torch.empty(count, **f),
            t=torch.empty(count, **f),
            last_hit_triangles=torch.full((count,), -1, device=device, dtype=torch.int32),
            flags=torch.zeros(count, device=device, dtype=torch.int32),
            weights=torch.ones(count, **f),
            evidx=torch.zeros(count, device=device, dtype=torch.int32),
            ids=torch.zeros(count, device=device, dtype=torch.int64),
        )

    def select(self, index):
        """New DevicePhotons with rows ``index`` (a torch index tensor or slice)."""
        return DevicePhotons(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def clone(self):
        return DevicePhotons(**{f.name: getattr(self, f.name).clone() for f in fields(self)})

    def to_numpy(self, skip=()):
        """Host arrays with Chroma's dtypes (flags/evidx as uint32)."""
        out = {}
        for f in fields(self):
            if f.name in skip:
                continue
            value = to_host(getattr(self, f.name))
            if f.name in ("flags", "evidx"):
                value = value.view(np.uint32)
            out[f.name] = value
        return out


# ------------------------------------------------------------ host transfers
#
# Host-to-device copies of large arrays are pipelined through two reusable
# pinned staging buffers (the DMA of one chunk overlaps the host memcpy of the
# other).

_CHUNK = 32 << 20
_STAGING = {}


def _staging(device):
    import torch

    key = (device.type, device.index)
    bufs = _STAGING.get(key)
    if bufs is None:
        bufs = [torch.empty(_CHUNK, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        _STAGING[key] = bufs
    return bufs


def _numpy_dtype(dtype):
    import torch

    return {torch.float32: np.float32, torch.int32: np.int32, torch.int64: np.int64, torch.uint8: np.uint8,
            torch.float64: np.float64, torch.int16: np.int16, torch.bool: np.bool_}[dtype]


def to_host(tensor):
    """A numpy array with the contents of a CUDA tensor.

    The copy lands in page-locked memory from PyTorch's pinned-memory cache
    (the array keeps its block alive): a pageable destination is faulted in
    page by page and runs at ~1 GB/s, the pinned copy at bus speed.
    """
    import torch

    t = tensor.detach()
    if not t.is_cuda:
        return t.numpy()
    host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    host.copy_(t, non_blocking=True)
    torch.cuda.current_stream(t.device).synchronize()
    return host.numpy()


def to_device(array, device):
    """A new CUDA tensor with the contents of a C-contiguous numpy array."""
    import torch

    array = np.ascontiguousarray(array)
    device = torch.device(device)
    if array.nbytes <= 2 * _CHUNK:
        return torch.from_numpy(array).to(device)
    out = torch.empty(array.shape, dtype=torch.from_numpy(array[:0]).dtype, device=device)
    src = array.reshape(-1).view(np.uint8)
    dst = out.reshape(-1).view(torch.uint8)
    bufs = _staging(device)
    stream = torch.cuda.current_stream(device)
    events = [None, None]
    for i, start in enumerate(range(0, src.size, _CHUNK)):
        k = i & 1
        if events[k] is not None:
            events[k].synchronize()  # the staging buffer's previous DMA is done
        stop = min(start + _CHUNK, src.size)
        bufs[k][:stop - start].numpy()[:] = src[start:stop]
        dst[start:stop].copy_(bufs[k][:stop - start], non_blocking=True)
        events[k] = torch.cuda.Event()
        events[k].record(stream)
    for event in events:
        if event is not None:
            event.synchronize()
    return out


@dataclass
class DaqChannels:
    """Per-channel DAQ output for one event (numpy): Chroma ``Channels`` fields."""

    t: np.ndarray  # float32 [nchannels], 1e9 when not hit
    q: np.ndarray  # float32 [nchannels]
    flags: np.ndarray  # uint32 [nchannels]


# One tracking record: the ids (batch-local row numbers) of the photons that
# were alive at the start of a step, and their states after that step.
TrackStep = Tuple["object", DevicePhotons]


class TransportEngine(Protocol):
    """Interface implemented by every Triton transport engine."""

    #: torch.device used for all state
    device: "object"
    def triangle_solid(self, triangles):
        """Solid id (int64 tensor) of flattened triangle ids ``triangles``."""
    #: int32 tensor [nsolids]: channel index or -1 (Detector only)
    solid_id_to_channel_index: Optional["object"]

    def propagate(
        self,
        photons: DevicePhotons,
        *,
        max_steps: int,
        use_weights: bool = False,
        track: bool = False,
    ) -> Optional[List[TrackStep]]:
        """Propagate ``photons`` in place until terminal or ``max_steps``.

        Photons already carrying a TERMINAL bit are left untouched. With
        ``track`` the engine returns W-compatible per-step records.
        """

    def acquire(self, photons: DevicePhotons, start: int, count: int) -> DaqChannels:
        """Run the DAQ for rows ``[start, start+count)`` (one event)."""
