# Triton drop-in backend for Chroma: design

Status: working design for branch `feature/triton-dropin`. This file is the
shared specification for everyone working on the drop-in.

## Goals

1. **Drop-in.** Unchanged scripts written against the *installed* Chroma
   (`/sdf/group/neutrino/youngsam/sim/chroma-lite`, revision b67bc6d plus its
   uncommitted local changes, called **W** below) run on the Triton backend by
   setting one environment variable. The Triton backend must not import PyCUDA.
2. **Arbitrary meshes.** Any Chroma `Geometry`/`Detector` works. Analytic wire
   planes use W's `geometry.wireplanes` declarations and FP32 semantics. No
   detector-specific compiler is allowed in the general path.
3. **Throughput.** The general mechanism sustains at least 25M input photons/s
   on the local A100 for the reflect3wires detector built by W's chroma-lar,
   with the same event contents the fast LAr path produces. Theia-like and
   pixel-TPC detectors are reported as generality checks.
4. **Bitwise mode.** A second environment variable switches the Triton backend
   to Chroma's exact arithmetic, XORWOW streams, RNG-slot assignment and launch
   schedule, including every known original defect, so that its outputs are
   bitwise identical to the original CUDA Chroma under the same schedule. The
   schedule (the "tape") is recorded so the equality can be demonstrated.

## Environment variables

| Variable | Values | Meaning |
|---|---|---|
| `CHROMA_BACKEND` | `cuda` (default), `triton` | Implementation behind `chroma.sim.Simulation` |
| `CHROMA_TRITON_TAPE` | unset/`off`, `canonical`, `record:<dir>`, `replay:<dir>` | Bitwise legacy mode (see below) |
| `CHROMA_TRITON_TAPE_SORT` | `1` | Record with sorted survivor queues (the `canonical` schedule) |
| `CHROMA_TRITON_DEVICE` | CUDA ordinal | Overrides `cuda_device` for the Triton backend |
| `TRITON_CACHE_DIR` | path | Put Triton's JIT cache on `/lscratch`; `$HOME` has little quota |

`chroma.sim` reads `CHROMA_BACKEND` at import time. `chroma.sim.Simulation`
and `chroma.Simulation` both resolve to the selected class. The original code
moves verbatim to `chroma/sim_cuda.py`.

## Contract (from W)

Constructor: `Simulation(detector, seed=None, cuda_device=None,
photon_tracking=False, nthreads_per_block=512, max_blocks=1024,
use_packed=False)`. `simulate(iterable, keep_photons_beg=False,
keep_photons_end=False, keep_hits=True, keep_flat_hits=True, run_daq=False,
max_steps=1000, use_weights=False, photons_per_batch=1000000)` is a generator
with W's batching, `evidx` rewriting and event wrapping. Outputs are the same
`Event` fields with the same dtypes: `photons_end`, `hits`
(`{channel_index: Photons}`), `flat_hits`, `channels` (`Channels` from the DAQ),
`photon_tracks`. Hit rows: `flags & SURFACE_DETECT`, `last_hit_triangle >= 0`,
`solid_id_to_channel_index[solid_id[tri]] >= 0`. Wire hits use
`last_hit_triangle = -2` and are never hits.

Inputs may be numpy `Photons`, or any object whose arrays expose
`__cuda_array_interface__` (PyCUDA `GPUArray`, torch tensors, CuPy); they are
wrapped with `torch.as_tensor` without copying when possible.

Production-mode deviations from W, all deliberate and documented:
* NaN-free Fresnel at normal incidence and at the critical angle; transverse
  specular polarization; no `NAN_ABORT` from those paths.
* `use_packed=True` returns the true final photon states (W returns the
  initial positions/directions).
* Photon histories are kept in 32 bits (W truncates to 16 on the device).
* Counter-based Philox RNG keyed by photon id: results do not depend on the
  batch size, thread count or queue order.
Flight time uses the phase velocity like W unless a material provides
`group_velocity`; surface re-emission is instantaneous unless the surface
provides `reemission_time_cdf`.

## Engine (production mode)

Compiled once per `Simulation` from the unflattened detector when available.

* **Instances.** Solids that share one `Mesh` object become instances of one
  bottom-level structure (BLAS) with their rotation and displacement. Each
  unique mesh gets one BLAS in its local frame. Global triangle ids stay
  identical to `geometry.flatten()` so `last_hit_triangles`, solid ids and
  channel ids are unchanged.
* **Top level.** A threaded (escape-link) BVH over instance world bounds.
* **Bottom level.** Threaded BVHs with small leaves in local FP32 coordinates.
  Traversal is stackless: no global-memory stack.
* **Wires.** W's wire-plane records and FP32 intersection, merged with the
  mesh hit exactly as W does (`t_wire + 1e-6 < t_mesh`).
* **Empty-space grid.** A uniform grid over the world bounds marks cells that
  no triangle, wire slab or instance bound touches. Connected empty cells of
  one certified material are expanded to per-cell safe boxes. Inside a safe box,
  absorption/scattering/re-emission run without geometry queries. A photon
  whose free path would leave the box is handed to the boundary query with the
  same random streams, so the shortcut does not change the random process.
* **Scheduler.** Device queues with block-level atomics, several bulk
  collisions per launch, one fused boundary query + interaction kernel, and at
  most one host synchronization per round. Photon state stays on the device
  from input to output.
* **Physics.** All of W's surface models (default, complex, WLS, dichroic,
  angular), multi-component bulk re-emission, weights, `max_steps`.

## Bitwise legacy mode

Bitwise equality is demonstrated on the *recorded schedule* of a real CUDA run
(reference: `docs/triton_legacy_tape.md`).

* **Record** (`CHROMA_BACKEND=cuda CHROMA_TRITON_TAPE=record:<dir>`): the
  unmodified CUDA backend runs through `chroma.sim.Simulation`; a hook in
  `chroma/sim_cuda.py` snapshots the XORWOW slot states around every kernel
  chunk and regenerates each photon's `curand_uniform` draws from the Weyl
  counter (checked against the full exit state). The tape holds the inputs as
  copied to the GPU, every photon's ordered propagation draws (CSR:
  `draws` float bits + `offsets[N+1]`), the launch schedule
  (`launch_starts`, `launch_nsteps`, chunks, actual queues), DAQ draws, the
  uploaded scene words (including the word after every table), Simulation
  parameters, source hashes and all outputs. Recording does not change the
  outputs (checked on every fixture).
* **Schedule.** W appends survivors to the next launch's queue with
  warp-level atomics, so multi-launch runs are not repeatable; the tape keeps
  the order that happened. `CHROMA_TRITON_TAPE=canonical` sorts every queue
  (a legal execution of the unmodified kernels) for repeatable CUDA runs.
* **Exact arithmetic** (`chroma/triton/engine/exact.py`): Triton functions
  that reproduce W's machine arithmetic bit for bit (explicit `.rn` and
  `fma.rn` wherever the original SASS contracted, fast-math intrinsics,
  libdevice transcriptions, FP32 wires, the thin-film block as extracted PTX)
  and take pre-drawn uniforms.
* **Replay** (`CHROMA_BACKEND=triton CHROMA_TRITON_TAPE=replay:<dir>`):
  consumes the per-photon draws. Until the production engine has an exact
  mode, `chroma.triton.legacy.engine.LegacyEngine` plugs the minimal
  reference loop (`chroma.triton.legacy.reference`, built from `exact.py`)
  into the compatibility layer; `python -m chroma.triton.legacy.verify`
  records, replays and compares `photons_end`, hits, channels and tracks
  word by word.
* **Fail closed** where W's behaviour is undefined or unrecorded (DAQ CDFs
  whose `cdf_y` is one entry short, indices >= 128, BVH stack > 1000,
  out-of-table dichroic/angular lookups, `scatter_first`, device profiling,
  mismatching parameters/inputs/detector).

## Validation

* Unit tests against the CPU reference for every physics branch.
* Statistical comparisons with the original CUDA backend (fractions within
  6 standard errors, DKW bounds on distributions) for reflect3wires, the pixel
  TPC and Theia.
* Bitwise-mode equality on the recorded native fixtures and on new canonical
  runs through the `Simulation` interface.
* Throughput: `benchmark_mesh_occupancy.py` and a reflect3wires full-event
  benchmark through `Simulation`.
