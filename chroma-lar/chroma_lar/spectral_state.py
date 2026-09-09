"""Named structure-of-arrays contracts for the spectral GPU kernels.

Tuple order is the kernel ABI: append/change fields only together with the
kernel signatures. Host code uses names; unpacking at launch adds no GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from torch import Tensor


class PhotonState(NamedTuple):
    pos: Tensor
    direction: Tensor
    polarization: Tensor
    times: Tensor
    flags: Tensor
    photon_ids: Tensor
    last_instance: Tensor
    last_hit: Tensor
    channels: Tensor
    steps: Tensor
    wavelengths: Tensor
    event_indices: Tensor

    @classmethod
    def allocate(cls, count: int, device="cuda") -> PhotonState:
        """Allocate uninitialized storage; a source must initialize every field."""
        import torch

        vector_fields = {"pos", "direction", "polarization"}
        float_fields = vector_fields | {"times", "wavelengths"}
        id_fields = {"photon_ids", "event_indices"}
        return cls(
            *(
                torch.empty(
                    (count, 3) if name in vector_fields else (count,),
                    dtype=(
                        torch.float32
                        if name in float_fields
                        else torch.int64 if name in id_fields else torch.int32
                    ),
                    device=device,
                )
                for name in cls._fields
            )
        )

    def slice(self, start: int, stop: int) -> PhotonState:
        """Return views into a contiguous range, retaining the named contract."""
        return type(self)(*(value[start:stop] for value in self))

    def to_host(self) -> dict:
        import numpy as np

        values = {name: value.cpu().numpy() for name, value in self._asdict().items()}
        values["flags"] = values["flags"].view(np.uint32)
        return values


class SpectralProperties(NamedTuple):
    refractive_index: Tensor
    absorption_length: Tensor
    scattering_length: Tensor
    group_velocity: Tensor
    surface_model: Tensor
    detect: Tensor
    absorb: Tensor
    reflect_diffuse: Tensor
    reflect_specular: Tensor
    reemit: Tensor
    reemission_cdf: Tensor
    time_offsets: Tensor
    time_x: Tensor
    time_cdf: Tensor
    time_density: Tensor
    reemission_to_material1: Tensor


class SourceProperties(NamedTuple):
    offsets: Tensor
    wavelengths: Tensor
    cdf: Tensor
    density: Tensor
    lifetimes: Tensor
    fractions: Tensor


@dataclass(frozen=True)
class TransportSettings:
    """Scheduling controls; none changes optical probabilities or RNG streams."""

    history_length: int = 8
    epochs_per_poll: int = 2
    block_size: int = 128
    fused_pmt: bool = False

    def __post_init__(self):
        for name in ("history_length", "epochs_per_poll", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.block_size not in (64, 128, 256, 512):
            raise ValueError("block_size must be 64, 128, 256, or 512")
        if not isinstance(self.fused_pmt, bool):
            raise ValueError("fused_pmt must be a boolean")
