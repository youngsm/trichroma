# User guide

TriChroma is a second implementation of `chroma.sim.Simulation`. Chroma
itself (geometry, materials, detectors, events) still comes from chroma-lite;
TriChroma is only the part that runs on the GPU. This guide covers what you
need to use it day to day.

## Choosing the backend

`chroma.sim.Simulation` is one of two implementations:

- **CUDA Chroma**, Chroma's original PyCUDA code;
- **TriChroma**, this package.

Set `CHROMA_BACKEND=triton` or `CHROMA_BACKEND=cuda` to choose one. Without
it, the CUDA backend is used when PyCUDA is installed, and TriChroma
otherwise. The choice is fixed when `chroma.sim` is first imported.

Select the GPU as usual, with `CUDA_VISIBLE_DEVICES` or
`Simulation(detector, cuda_device=1)`.

## What stays the same

- `Simulation(...)` and `simulate(...)` have the same arguments, including
  `use_weights`, `photons_per_batch`, `run_daq`, `photon_tracking` and
  `use_packed`.
- Events come back with the same fields and types: `photons_end`, `hits`,
  `flat_hits`, `channels` (from the DAQ) and `photon_tracks`. Triangle, solid
  and channel numbers are Chroma's.
- Any Chroma detector works, including analytic wire planes.

Input photons can be ordinary `chroma.event.Photons`, or arrays already on
the GPU (PyTorch, CuPy or PyCUDA). Photons on the GPU are used in place,
without a copy where possible.

## What changes in your results

TriChroma simulates the same physics, but the numbers differ from CUDA
Chroma's, for three reasons.

1. **Different random numbers.** Each photon's random numbers depend only on
   the seed, the photon's position in your input, and its step. The result is
   a different random realisation than CUDA Chroma's, with the same
   distributions. As a bonus, the same seed yields identical results for any
   batch size, and every run repeats exactly on the same kind of GPU. (CUDA
   Chroma's results depend on how its threads happen to be scheduled.)
2. **Fixes of CUDA Chroma bugs**, on by default:
   - **Wire planes.** CUDA Chroma's wire intersection is dominated by float32
     rounding. Its results include wire hits that do not exist, omit real
     ones, and leave photons inside wires. On chroma-lar's liquid-argon TPC,
     this fix alone raises the detected light by about 18%.
   - **Faces that touch.** A photon could slip through a face that lies
     exactly in another face's plane, such as a PMT backplate flush with a
     wall (0.08% of the light on the same detector).
   - **Polarization after a mirror reflection** was left unchanged (-0.4% of
     detected light).
   - **Fresnel reflection at normal incidence** computed 0/0, so such
     photons were always transmitted.
   - **DAQ time and charge tables** built with `set_time_dist*` or
     `set_charge_dist*` were one entry short, so memory past their end was
     read.
3. **Rounding.** TriChroma's float32 arithmetic differs from CUDA Chroma's in
   the last bits. A few photons in a million end up on a different branch;
   the distributions do not change.

`CHROMA_TRITON=legacy` switches the bug fixes off, for like-for-like
comparisons with CUDA Chroma. [Correctness](correctness.md) has the evidence,
and [the reference](reference/differences.md) lists every difference.

Two output details also differ:
- photon `flags` have all 32 bits (CUDA Chroma's are truncated to 16), so an
  aborted photon is flagged with `chroma.event.NAN_ABORT` (bit 31);
- with `use_packed=True`, `photons_end` contains the photons' final state;
  in CUDA Chroma it contains their initial state.

## Not supported yet

- The `complex`, `dichroic` and `angular` surface models raise
  `NotImplementedError`.
- Chroma's interactive viewers depend on PyCUDA; run them with the CUDA
  backend.

## Options

`CHROMA_TRITON` is a comma-separated list of options, all off by default:

```bash
CHROMA_TRITON=legacy python my_script.py
CHROMA_TRITON=strict,roulette=0.05 python my_script.py
```

| Option | Effect |
|---|---|
| `legacy` | Keeps CUDA Chroma's bugs (listed above), for like-for-like comparisons. |
| `strict` | Turns off the few optimizations that can change a result in the last bit. Slightly slower. |
| `roulette=<w>` | Weighted mode only: Russian roulette below weight `w` (for example 0.05). Unbiased, and about twice as fast. |

An unknown option is an error. The
[engine reference](reference/engine.md#debugging-options) has four more
options for debugging.

## Getting full speed

A GPU runs at full speed only with millions of photons in flight. Three
habits make the difference:

- **Make photons on the GPU.** Generating photons with NumPy can take longer
  than simulating them. `trichroma.sources` generates them on the GPU, where
  `simulate()` reads them directly:

  ```python
  from chroma.event import Event
  from trichroma import sources

  photons = sources.photon_bombs(200_000, voxel_centres, voxel_size=30, wavelength=450)
  events = (Event(photons_beg=photons[i * 200_000:(i + 1) * 200_000])
            for i in range(len(voxel_centres)))
  for event in sim.simulate(events, keep_flat_hits=True, photons_per_batch=10_000_000):
      ...
  ```

- **Launch big.** `simulate()` groups events into launches of
  `photons_per_batch` photons (default one million). On a large GPU, about
  10^7 is close to full speed. Memory sets the limit: 11 GB holds about
  3·10^7 photons.
- **Keep the kernel cache.** Compiled kernels are cached in
  `TRITON_CACHE_DIR` (default `~/.triton`). If your home directory is small or
  slow, point it at a local disk.

While your code reads the events of one batch, the next batch is already
running, so the GPU is rarely idle.

## Comparing runs

`trichroma.utils` has two comparisons:

- `compare_events(a, b)` compares bit for bit. Use it between two TriChroma
  runs, for example before and after a change to your code.
- `compare_statistics(a, b)` compares within statistical errors: history
  fractions, hits per channel, and the time and wavelength distributions. Use
  it against CUDA Chroma, with `CHROMA_TRITON=legacy` so that both run the
  same physics.

Run each backend in its own process and save what you need:

```python
from trichroma.utils import compare_statistics, event_statistics_arrays

arrays = event_statistics_arrays(events)          # in each run; save with np.savez
result = compare_statistics(cuda_arrays, triton_arrays)
print(result)                                      # a short report
```

## Replaying CUDA Chroma bit for bit

A CUDA Chroma run can also be recorded and then replayed by TriChroma
exactly, random numbers and rounding included. This mode exists to show that
the two compute the same physics, not for production; see
[exact mode](reference/exact_mode.md).
