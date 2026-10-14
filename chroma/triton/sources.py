"""Photon sources drawn on the GPU.

With the Triton backend, ``Simulation.simulate`` takes events whose
``photons_beg`` holds CUDA tensors, so generated photons never pass through
host memory::

    from chroma.event import Event
    from trichroma import sources

    photons = sources.photon_bombs(200_000, centres, voxel_size=30, wavelength=450)
    events = (Event(photons_beg=photons[i * 200_000:(i + 1) * 200_000])
              for i in range(len(centres)))
    for ev in sim.simulate(events, photons_per_batch=5_000_000):
        ...

Nothing here waits for the GPU (no synchronizing copies), so while one batch
propagates the next can already be drawn. The CUDA (PyCUDA) backend needs
host ``chroma.event.Photons`` instead; :func:`available` tells them apart.
"""

from types import SimpleNamespace

import numpy as np

FIELDS = ("pos", "dir", "pol", "wavelengths", "t", "last_hit_triangles", "flags", "weights", "evidx")


def available():
    """True when ``chroma.sim.Simulation`` is the Triton backend and a CUDA
    device exists (then sources from this module can be simulated)."""
    from chroma.backend import backend_name

    if backend_name() != "triton":
        return False
    import torch

    return torch.cuda.is_available()


class SourcePhotons(SimpleNamespace):
    """The fields of ``chroma.event.Photons`` as CUDA tensors (float32;
    ``last_hit_triangles``, ``flags`` and ``evidx`` as int32)."""

    def __len__(self):
        return int(self.wavelengths.shape[0])

    def __getitem__(self, rows):
        """The photons ``rows`` (a slice gives views)."""
        return SourcePhotons(**{name: getattr(self, name)[rows] for name in FIELDS})


def photon_bomb(nphotons, pos, voxel_size=30, wavelength=128, generator=None):
    """``nphotons`` isotropic photons with random polarization, uniform in a
    cube of side ``voxel_size`` (mm) centred on ``pos``, at t = 0.

    ``wavelength`` is a value or a (min, max) range (uniform). ``generator``
    is an optional ``torch.Generator`` on the CUDA device.
    """
    return photon_bombs(nphotons, np.asarray(pos, dtype=np.float64).reshape(1, 3), voxel_size, wavelength,
                        generator)


def photon_bombs(nphotons, centres, voxel_size=30, wavelength=128, generator=None):
    """:func:`photon_bomb` for many voxels at once: ``nphotons`` photons for
    each row of ``centres`` [k, 3], voxel after voxel (slice the result for
    one voxel's photons). A few large GPU operations instead of many small
    ones per voxel."""
    import torch

    dev = torch.device("cuda")
    centres = np.asarray(centres, dtype=np.float32).reshape(-1, 3)
    corner = torch.from_numpy(centres - np.float32(voxel_size / 2)).pin_memory().to(dev, non_blocking=True)
    total = nphotons * len(centres)
    u = torch.rand((9, total), device=dev, generator=generator)
    # Rows 0/1: direction (cos theta, phi); rows 2/3: the random unit vector
    # crossed with it for the polarization; row 4: wavelength; rows 6-8: position.
    costheta = u[0:4:2] * 2 - 1
    sintheta = (1 - costheta * costheta).sqrt()
    phi = u[1:4:2] * (2 * np.pi)
    unit = torch.stack([sintheta * phi.cos(), sintheta * phi.sin(), costheta], 2)
    pdir = unit[0]
    ppol = torch.linalg.cross(pdir, unit[1])
    ppol /= ppol.norm(dim=1, keepdim=True)
    if isinstance(wavelength, tuple):
        pwavelength = u[4] * (wavelength[1] - wavelength[0]) + wavelength[0]
    else:
        pwavelength = torch.full((total,), float(wavelength), device=dev)
    ppos = corner.repeat_interleave(nphotons, 0).add_(u[6:9].T, alpha=float(voxel_size))
    return SourcePhotons(
        pos=ppos, dir=pdir, pol=ppol, wavelengths=pwavelength,
        t=torch.zeros(total, device=dev),
        last_hit_triangles=torch.full((total,), -1, device=dev, dtype=torch.int32),
        flags=torch.zeros(total, device=dev, dtype=torch.int32),
        weights=torch.ones(total, device=dev),
        evidx=torch.zeros(total, device=dev, dtype=torch.int32),
    )
