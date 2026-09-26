# Triton drop-in backend for Chroma: design

This is the design reference of the Triton backend: its goals, the API
contract it keeps, the deliberate differences from CUDA Chroma, the engine
and its measured throughput. Code layout: `src/chroma` (Chroma's API),
`src/trichroma` (this implementation). [Exact vs production](exact_vs_production.md)
lists every difference between the bitwise mode and the production engine,
with the evidence that the production engine is unbiased and what each
difference contributes to speed.

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
| `CHROMA_BACKEND` | `cuda`, `triton` | Implementation behind `chroma.sim.Simulation`; unset: `cuda` when PyCUDA is installed, `triton` otherwise |
| `CHROMA_TRITON_TAPE` | unset/`off`, `canonical`, `record:<dir>`, `replay:<dir>` | Bitwise mode (see below) |
| `CHROMA_TRITON_TAPE_SORT` | `1` | Record with sorted survivor queues (the `canonical` schedule) |
| `CHROMA_TRITON_DEVICE` | CUDA ordinal | Overrides `cuda_device` for the Triton backend |
| `CHROMA_TRITON_FUSED` | `1` (default), `0` | Fused register-resident transport kernel; `0` selects the wavefront scheduler |
| `CHROMA_TRITON_GRID` | `1` (default), `0` | Certified empty-space grid (bulk shortcut of the wavefront scheduler; built only when that scheduler is selected) |
| `CHROMA_TRITON_PIPELINE` | `1` (default), `0` | `simulate` propagates batch k+1 while the caller consumes batch k; `0` restores W's order of reading input and yielding events (results are identical either way) |
| `CHROMA_TRITON_FIXES` | `1` (default), `0` | `0` keeps W's behaviour where production corrects it (specular polarization, literal Fresnel formulas, NaN-abort bit 1<<15, W's wire algorithm, W's t > 1e-6 at box faces), with the production RNG, geometry and arithmetic: a statistical like-for-like comparison with CUDA Chroma. It does not restore W's specular-direction formula or 16-bit history truncation ([exact vs production](exact_vs_production.md)) |
| `CHROMA_TRITON_LEGACY_WIRES` | unset, `0`, `1` | Override the wire algorithm alone (default: legacy iff `CHROMA_TRITON_FIXES=0`) |
| `CHROMA_TRITON_ROULETTE` | weight, e.g. `0.05` | Opt-in, weighted mode only: Russian roulette below that weight (unbiased for every tally; not W's weighted-mode semantics) |
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
wrapped with `torch.as_tensor` without copying when possible. Photons drawn on
the GPU (events whose `photons_beg` holds CUDA tensors) never pass through
host memory.

In production mode `simulate` is pipelined: batch k+1 is taken from the input
and propagated while the caller consumes the events of batch k. The outputs
of batch k are packed on the device before batch k+1 starts and copied to
page-locked memory on a second stream while it runs; the propagation itself
needs no host synchronization. Photon ids (the RNG keys) are assigned in input
order, so the results do not change; only the input is read one batch ahead
(`CHROMA_TRITON_PIPELINE=0` turns this off). Exact mode and photon tracking
are not pipelined.

Production-mode deviations from W, all deliberate and documented:
* NaN-free Fresnel at normal incidence and at the critical angle; transverse
  specular polarization; no `NAN_ABORT` from those paths.
* `use_packed=True` returns the true final photon states (W returns the
  initial positions/directions).
* Photon histories are kept in 32 bits (W truncates to 16 on the device).
* Counter-based Philox RNG: each event (one iteration of W's loop) takes its
  uniforms from Philox blocks keyed by (photon id, steps done, block), four
  per call. Results do not depend on batch size, thread count, queue order or
  scheduler (fused, wavefront, bulk shortcut).
* Analytic wires: three defects of W, all of which lose light.
  * Far hits. W forms the discriminant as `B*B - A*C`, whose two terms are
    ~`t*t` while their difference is ~`r*r`; beyond a few hundred mm FP32
    rounding decides far hits. On 16.5M captured LAr boundary rays W reports
    42% more wire hits than a float64 reference (and a 4 degree median normal
    error); production uses the identical `A*r*r - (wv*dn - wn0*dv)**2`.
  * Leaving a wire. At metre-scale coordinates the FP32 reflection point on
    a 75 um wire often lands just inside the cylinder. W then takes the exit
    for a hit from inside, places the photon in the wire's material and
    (steel: absorption length 0) absorbs it: 7.2% of all photons of a 450 nm
    voxel bomb in reflect3wires, although the wire surface transmits
    nothing. Production skips the wire a photon has just left outward (exact:
    a ray leaving a convex cylinder cannot meet it again; the skip ends at the
    photon's next event) and matches a float64 Monte Carlo of a wire plane at
    x = 2160 mm within statistics (`tests/integration/test_wires.py`).
  * Culled planes. W skips a plane when the distance along the ray to the
    plane of the wire axes exceeds the mesh hit plus one radius; an oblique
    ray enters the wires up to r/|dn| earlier, so a ray ending on a wall the
    wires pass through (chroma-lar's rotated planes extend through the TPC
    walls) can reach a wall point inside a wire, and the photon is lost in
    the steel. Production culls a plane only when the slab entry is beyond
    the mesh hit (1568 photons in 10M of the weighted LUT fixture were lost
    this way; +0.0016% detected light).
  * Together they raise the detected light on the LAr LUT workload
    (reflect3wires, 128 nm, weighted) by 18.0% (with every production
    correction: 3.145% of the photons' weight in W, 3.697% in production),
    and at 450 nm from 3.87% to 4.54%. On 2M rays of real
    histories, W's wire answer is wrong for 98.8% of the rays where it differs
    from production (spurious hits up to 25 radii from any wire, missed hits,
    wrong wires, hit points up to 2.9 mm off); production agrees with float64
    except within 0.2% of tangency. What is left in production are float32
    limits (about 1 photon in 10^6 ends in steel): a Rayleigh scatter within
    ~1e-4 mm of a steel surface, or a wall reflection at the edge of an end
    wire that chroma-lar places half inside the wall.
    `CHROMA_TRITON_LEGACY_WIRES=1` (or `CHROMA_TRITON_FIXES=0`) reproduces W's
    algorithm (in Triton arithmetic, so its noise-decided far hits differ ray
    by ray).
* Analytic boxes do not re-hit the coplanar neighbour of the face a photon
  just left (W's mesh traversal occasionally does, at t ~ 1e-5 mm, and can
  then put the photon inside the solid: 8 photons in 10^6 end in the
  cathode's steel on reflect3wires).
* Coincident faces. A photon that leaves a solid through a face lying in the
  plane of a box face is left on that plane; W's triangle test (t > 1e-6)
  lets it through whenever rounding puts it on or beyond the plane.
  chroma-lar's PMT backs lie in the TPC walls, so 0.5% of the photons of the
  LUT fixture leave the TPC through them in W. Production meets the box face
  at distance 0 when the photon is on its plane within rounding (a few ulp)
  and has not just met it: no photon leaves the TPC (+0.08% detected light).
* Specular reflection uses the mirror formula `d - 2(d.n)n` (also with
  `CHROMA_TRITON_FIXES=0`); W rotates the normal by the incidence angle,
  which gives a NaN direction (NaN abort) for a photon exactly anti-parallel
  to the normal and rounds directions within 2.4e-4 rad of normal incidence.
* The DAQ clamps the time and charge CDFs at their ends and adds nothing for
  a negative charge sample, as W does.

Flight time uses the phase velocity `c/n` of the incident material and
surface re-emission is instantaneous, as in W.

## Engine (production mode)

Compiled once per `Simulation` (`engine/scene.py`).

* **Instances.** Solids with identical meshes (content hash) become instances
  of one bottom-level structure (BLAS) in the mesh's local frame. Global
  triangle ids stay identical to `geometry.flatten()`, so `last_hit_triangles`,
  solid ids and channel ids are unchanged.
* **Trees.** Top-level and bottom-level trees are binned-SAH BVHs stored as
  eight threaded (escape-link) copies, one per ray-direction octant, near
  child first. Traversal is stackless and picks the copy from the sign bits of
  the (local) ray direction.
* **Analytic boxes.** Axis-aligned box solids (up to 64 triangles) are tested
  analytically before the top-level tree: slab method for the crossing face
  (entry face, or exit face when starting inside or on the box; the face of
  the last-hit triangle decides a ray leaving it), then Moller-Trumbore on that
  face's triangles for exact triangle ids.
* **Wires.** W's wire-plane records; the accurate intersection above (or W's
  exactly with `CHROMA_TRITON_LEGACY_WIRES=1`), merged with the mesh hit as W
  does (`t_wire + 1e-6 < t_mesh`).
* **Fused transport** (`engine/fused.py`, default). Persistent one-warp
  programs keep 32 photons in registers and run them to completion, taking new
  photons from the work list as lanes free up. Each iteration takes one W step
  for every live lane: boxes and the top-level tree, wires, boundary physics.
  A lane whose ray reaches an instance waits with its top-level result until
  8 lanes of the warp (or all its live lanes) need an instance descent; the
  warp then descends for them together, resuming the top-level walk where it
  stopped. Rare branches (re-emission, WLS, diffuse, Fresnel) run only when
  some lane takes them.
* **Wavefront scheduler** (`CHROMA_TRITON_FUSED=0`, and photon tracking).
  Device queues with one host read per round: a certified empty-space grid
  lets bulk collisions run without geometry queries (same draws, so the
  outcome equals a full step), a two-pass boundary query (top level for every
  ray, instance descent for the compacted rays that reach one), and the same
  boundary physics. The last rounds replay from a CUDA graph.
* **Physics.** W's default and WLS surface models, multi-component bulk
  re-emission, weights, `max_steps`. The complex, dichroic and angular surface
  models are implemented in exact mode only; the production engine raises
  `NotImplementedError` for them (to do).

Throughput on the A100 (reflect3wires, LUT voxel at (-450, 60, -120), 128 nm,
30M photons, one `Simulation.simulate` call with flat hits,
`benchmarks/simulate_throughput.py`): 63M photons/s unweighted, 11.7M
photons/s weighted (~90 steps and ~5.7 PMT-mesh descents per photon),
23M photons/s weighted with `CHROMA_TRITON_ROULETTE=0.05` (same detected weight). CUDA
Chroma: 1.6M and 0.55M photons/s (with its wire losses, which end many
histories early).

The chroma-lar waveform map (`pyrat macros/waveform_map_pyrat.py`: 200K
photons per 30 mm voxel at 450 nm, one event per voxel, flat hits, a 2D
histogram and HDF5 rows per voxel): the engine alone does 66M photons/s with
5M photons per launch and 87M/s with 20M (1.5G steps/s; the last photons of
each launch dominate small launches). Through `Simulation.simulate`, with
photons drawn on the GPU and 5M photons per batch, 720 voxels take 2.2 s (64M
photons/s), against ~60 ms per voxel with the
macro's original numpy photons and one voxel per batch. Kernels are never
specialized on the seed (a new seed used to recompile the fused kernel, ~20
s), and the hashing Triton does before its first cache lookup (~0.4 s) runs on
a worker thread while the scene compiles.

## Bitwise mode (tape replay)

Bitwise equality is demonstrated on the *recorded schedule* of a real CUDA run
(reference: `docs/bitwise_mode.md`).

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
* **Exact arithmetic** (`src/trichroma/engine/exact.py`): Triton functions
  that reproduce W's machine arithmetic bit for bit (explicit `.rn` and
  `fma.rn` wherever the original SASS contracted, fast-math intrinsics,
  libdevice transcriptions, FP32 wires, the thin-film block as extracted PTX)
  and take pre-drawn uniforms.
* **Replay** (`CHROMA_BACKEND=triton CHROMA_TRITON_TAPE=replay:<dir>`)
  runs through the same `Simulation`, the same `ProductionEngine` class, its
  device-queue round scheduler and the same DAQ plumbing as production: the
  simulation layer (`trichroma.simulation`) builds `ProductionEngine(detector, seed=, device=,
  tape=TapeMode, simulation=...)`, whose exact mode
  (`src/trichroma/engine/exact_mode.py`) swaps in exact kernels. A round is
  W's launch-entry normalization at the recorded launch starts (exact
  `normalize3`), `fill_state`'s boundary part (W's BVH and FP32 wires) and one
  loop-body step whose uniforms come from the photon's tape segment;
  survivors are appended to the next device queue, one host read per round.
  `acquire()` replays `run_daq` with the recorded draws. No grid, bulk
  shortcut or graph tail in this mode. The scene is the recorded one after a
  check against the detector (material/surface labels matched by content).
  With `use_packed` the simulation layer returns W's initial
  pos/dir/pol/wavelength/time/weight words in `photons_end` and feeds the DAQ
  the initial times/weights unless hits were extracted, as W does.
  `trichroma.tape.reference` drives the same kernels launch by launch
  from the host (reference loop, timing baseline);
  `python -m trichroma.tape.verify` records, replays (public API and
  reference loop) and compares `photons_end`, hits, channels and tracks word
  by word.
* **Fail closed** where W's behaviour is undefined or unrecorded (DAQ CDFs
  whose `cdf_y` is one entry short, indices >= 128, BVH stack > 1000,
  out-of-table dichroic/angular lookups, `scatter_first`, device profiling,
  mismatching parameters/inputs/detector, a replay whose launch queue lengths
  or per-photon draw counts differ from the tape). `canonical` on the Triton
  backend is not implemented (there is no tape to replay).

## Validation

* Unit tests against the CPU reference for every physics branch.
* Statistical comparisons with the original CUDA backend (fractions within
  6 standard errors, DKW bounds on distributions) for reflect3wires, the pixel
  TPC and Theia.
* Bitwise-mode equality on the recorded native fixtures and on new canonical
  runs through the `Simulation` interface.
* Throughput: `benchmark_mesh_occupancy.py` and a reflect3wires full-event
  benchmark through `Simulation`.
