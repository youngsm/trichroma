# Bitwise mode: native RNG tapes and exact arithmetic

This is the reference for the bitwise mode of the Triton drop-in
backend. It describes how a CUDA Chroma run is recorded, the tape format, the
order in which the original kernels consume random numbers, the exact-math
Triton library, and how to demonstrate equality. The reference implementation
of the CUDA side is **W** (`/sdf/group/neutrino/youngsam/sim/chroma-lite`,
b67bc6d plus local changes), compiled by PyCUDA with its `cuda_options`
(`--use_fast_math -Xptxas=-dlcm=ca`) for sm_80 with CUDA 12.4.

## 1. What is claimed and how it is shown

Given the same inputs, the same detector words, the same Simulation parameters
and the same *schedule* (the queue order of every launch, see below), the
Triton side must produce the same bits as W for `photons_end` (every field),
`flat_hits`/`hits` (in photon order), `channels` (t, q, flags, hit) and
`photon_tracks`, known defects included.

1. **Record** a run of the *unmodified* CUDA backend through the public API:
   `CHROMA_BACKEND=cuda CHROMA_TRITON_TAPE=record:<dir>`. The tape holds the
   inputs as copied to the GPU, every uniform draw each photon consumed,
   the launch schedule, the DAQ draws, and the outputs.
2. **Replay** the tape on the Triton side through the same public API and the
   same engine as production: `CHROMA_BACKEND=triton
   CHROMA_TRITON_TAPE=replay:<dir>` makes `chroma.sim.Simulation` build
   `ProductionEngine(detector, ..., tape=...)`, whose exact mode runs the
   production round scheduler with W's arithmetic and draws every uniform
   from the tape instead of XORWOW. Every output word is compared.
3. **Transparency**: the recorded run's outputs are bitwise identical to the
   same run without recording (shown on every fixture, section 7).

Original Chroma is not repeatable once a batch needs more than one propagate
launch: survivors are appended to the next queue with warp-aggregated atomics
in arbitrary warp order, and the queue position selects the RNG slot. The tape
records the order that actually happened. `CHROMA_TRITON_TAPE=canonical`
(CUDA backend, no recording) sorts every survivor queue before the next launch,
which is one legal execution of the unmodified kernels; with
`CHROMA_TRITON_TAPE_SORT=1` the recorder applies the same sort. Canonical runs
are repeatable and are what the transparency check compares.

## 2. Recording without touching the kernels

`chroma/sim_cuda.py` installs `trichroma.tape.record_cuda` when
`CHROMA_TRITON_TAPE` is set (`record:<dir>` or `canonical`); with the variable
unset the CUDA backend is unchanged. The recorder subclasses `Simulation`
(`_simulate_batch`), `GPUPhotons` (to observe each batch) and wraps the
PyCUDA kernel handles of `propagate`, `propagate_packed` and `run_daq`:

* Every draw is one `curand_uniform` on the slot's `curandStateXORWOW`; each
  call adds 362437 to the Weyl word `d`. Around every kernel chunk the
  recorder copies the slot states (48 bytes each) before and after the
  unmodified kernel. Draw count of slot `i` = `(d_after - d_before) *
  362437^-1 mod 2^32`. The draws are regenerated from the entry state by a
  separate kernel with the same inline `curand_uniform`
  (`cvt.rn.f32.u32` + `fma.rn.ftz(x, 2^-32, 2^-33)`), and regeneration must
  reproduce the complete exit state (`d`, `v[0..4]`) of every slot, or
  recording fails.
* Queues, chunking, step counts, final photon state and the returned events
  are read, never written (except the canonical sort).
* The scene is read back from device memory (`export_device_scene`), including
  the word that follows every per-wavelength table (see section 5).

## 3. Tape format (version 1)

Directory `<dir>`:

| File | Content |
|---|---|
| `manifest.json` | `format="chroma-triton-tape"`, `version=1`, environment (Python, NumPy, PyCUDA, nvcc, device, driver, `cuda_options`), SHA-256 of every `chroma/*.py|.cu|.h`, `argv`, `queue_sorted`; one entry per `Simulation` (`seed`, `nthreads_per_block`, `max_blocks`, `photon_tracking`, `use_packed`, `rng_slots`, `rng_initial_sha256`, `detector_class`, `has_channels`, `scene_file`) and one per batch (file, simulation, `params` = the `_simulate_batch` keywords, counts, per-array SHA-256, `rng_final_sha256`, `complete`). |
| `sim<S>_scene.npz` | The words the CUDA backend uploaded for simulation `S`: `wavelength_grid`, `material_*` tables and `material_header` [M,7], `comp_*` re-emission tables and `material_comp_offsets`, `surface_*` tables, `surface_header` [S,6], `surface_present`, `dichroic_*`, `angular_*`, `wireplanes` [P,31], `vertices`, `triangles`, `material_codes`, `colors`, `solid_id_map`, `nodes` (BVH), `world_origin`, `world_scale`, DAQ `solid_id_to_channel_index`, `time_cdf_x/y`, `charge_cdf_x/y`, `detector_header`; `pad_<table>` = the 32-bit word that follows each row in device memory. |
| `sim<S>_batch<B>.npz` | One `_simulate_batch` call (arrays below). |

Batch arrays (`N` photons, `E` events, `L` launches, `C` kernel chunks; `u32`
draws are IEEE-754 float bits):

| Key | Type | Meaning |
|---|---|---|
| `in_pos`, `in_dir`, `in_pol` | f32 [N,3] | Inputs exactly as `GPUPhotons` holds them before `propagate` |
| `in_wavelengths`, `in_t`, `in_weights` | f32 [N] | |
| `in_last_hit_triangles` | i32 [N] | |
| `in_flags`, `in_evidx` | u32 [N] | |
| `event_bounds` | i64 [E+1] | Photon range of every event |
| `draws` | u32 [D] | Propagation draws, CSR by photon: photon `i` consumed `draws[offsets[i]:offsets[i+1]]` over all its launches, in call order |
| `offsets` | i64 [N+1] | |
| `launch_starts` | i32 [L] | Global step index at which launch `k` began |
| `launch_nsteps`, `launch_nphotons` | i32 [L] | Steps and queue length of launch `k` |
| `chunk_launch`, `chunk_first`, `chunk_count`, `chunk_blocks` | i32 [C] | Kernel chunks (`nthreads_per_block*max_blocks` photons each) |
| `queue`, `queue_offsets` | u32 [Q], i64 [L+1] | Actual input queue of every launch |
| `queue_draws` | u32 [Q] | Draws consumed by each queue entry in that launch |
| `daq_draws`, `daq_offsets` | u32, i64 [N+1] | DAQ draws, CSR by photon |
| `daq_event_ran` | u8 [E] | Events with `run_daq` |
| `final_*` | photon fields | Device state after propagation (packed arrays with `use_packed`) |
| `end_*` | photon fields | `photons_end` as returned (with `use_packed` W returns the *initial* pos/dir/pol/wavelength/time/weight words) |
| `hits_*`, `hits_event` | | `flat_hits` as returned (warp-atomic order) |
| `hits_expected_*`, `hits_expected_photon`, `hits_expected_event` | | The same hits derived from `final_*` in photon order |
| `channels_t`, `channels_q` / `channels_flags` / `channels_hit` | f32 / u32 / u8 [E,nch] | DAQ output |
| `track_step_offsets`, `track_ids`, `track_*` | | `photon_tracks` input (`GPUPhotons.propagate(track=True)` lists) |

Reader (`trichroma.tape.format`, NumPy only; device views need torch):

```python
tape = Tape(dir)                      # checks format/version, array hashes
batch = tape.batch(i)                 # TapeBatch; tape.scene(sim) -> dict
batch.draws(device)                   # float32 [D]  (bit-identical curand_uniform values)
batch.offsets(device)                 # int64 [N+1]
batch.launch_starts(device), batch.launch_nsteps(device)   # int32 [L]
batch.daq_draws(device), batch.daq_offsets(device)
batch.device_inputs(device)           # trichroma.engine.api.DevicePhotons
batch.inputs(), batch.final(), batch.fields("end"|"hits"|...)   # numpy dicts
batch.params                          # max_steps, use_weights, run_daq, keep_*
replay = TapeReplay(dir)              # sequential consumer for one Simulation
replay.check_simulation(seed=, nthreads_per_block=, max_blocks=, photon_tracking=, use_packed=)
batch = replay.next_batch(photons, max_steps=, use_weights=)   # checks inputs bitwise
```

## 4. Replay by the production engine (exact mode)

`ProductionEngine(detector, seed=, device=, tape=TapeMode('replay', dir),
simulation=dict(nthreads_per_block=, max_blocks=, photon_tracking=,
use_packed=))` (`trichroma.simulation` passes the Simulation's values)
selects the exact mode, implemented in `src/trichroma/engine/exact_mode.py`
(`ExactMode`); `propagate()` and `acquire()` dispatch to it. Construction
checks the tape's Simulation parameters, compares the detector's uploaded
words with the recorded scene (materials and surfaces are matched by content,
because W numbers them in Python set order), applies the fail-closed limits
and uploads the *recorded* words (W's BVH and the word after every table).
There is no empty-space grid, bulk shortcut or CUDA-graph tail.

`propagate()` reads the next batch, checks the inputs bitwise on the device,
uploads the draws, and runs the production scheduler: the live photons (16-bit
history not terminal) form the first device queue; every round is one global
step; survivors are appended to the next queue on the device
(`physics.append_rows`) and the host reads one queue length per round. A round
is three kernels, as in production:

1. `exact_renorm_kernel` -- W's launch-entry `dir/pol /= norm()` (exact
   `normalize3`) for photons whose step count is a recorded launch start and
   that were not yet normalized at that step (launched only at launch starts);
2. `exact_geometry_kernel` -- `fill_state`'s boundary part: W's flattened BVH
   (global-memory stack, bound from the tree) and the FP32 wires;
3. `exact_step_kernel` -- one iteration of W's loop body with every uniform
   read from the photon's tape segment; the history is stored as W's 16-bit
   word; survivors with fewer than `max_steps` steps are appended.

After the rounds the replay checks the schedule (at every recorded launch
start the queue held exactly `launch_nphotons` photons; no photon outlived the
last launch) and that every photon consumed exactly its recorded draws, with no
harness error (stack overflow, table lookups past the end, unknown model).
`track=True` returns W's records: all photons before propagation and after the
first step (W's first queue holds every photon), then the queue of every later
step. `acquire(photons, start, count)` replays `run_daq` for the next recorded
event with the recorded DAQ draws and W's integer atomics.
`ExactMode.timing` holds the seconds of the last `propagate()` (tape read,
preparation, rounds, checks).

With `use_packed=True` the simulation layer reproduces W's
`GPUPhotons(use_packed=True)` in exact mode: `photons_end` holds the *initial*
position, direction, polarization, wavelength, time and weight words
(`get()` reads the unpacked arrays that `propagate_packed` never updates), and
the DAQ reads the initial times and weights unless hits were extracted first
(`get_flat_hits()` syncs the packed state back). Production mode keeps
returning the true final states.

`src/trichroma/tape/reference.py` drives the same exact kernels launch by
launch from the host (active rows compacted after every step), like W's host
loop. It needs no detector, backs `python -m trichroma.tape.verify tape`
and is the timing baseline.

**Launch schedule.** Launch `k` covers global steps
`[launch_starts[k], launch_starts[k] + launch_nsteps[k])`. Launch 0's queue is
every photon; launch `k+1`'s queue is every photon still alive after launch
`k` (W's rule: `nsteps = max_steps - step` when `nphotons <
nthreads_per_block*128` or `use_weights`, else 1; always 1 with tracking).
For every photon in a launch's queue W:

1. loads the photon and normalizes `dir` and `pol` (`normalize3`), every
   launch;
2. keeps only the low 16 bits of `flags` (`unsigned short history`;
   `NAN_ABORT` is `0x8000` inside the kernel);
3. returns without storing anything if `history & 0x800F` (terminal at entry,
   possible only for input flags; such photons keep their input words);
4. runs up to `nsteps` steps, then stores all fields (history zero-extended).

The replay does not need RNG slots or queue order: the per-photon draw
sequence already encodes them. One engine round per global step reproduces
every launch shape, because a photon alive at the end of a launch has run all
of that launch's steps (every early exit of W's loop sets a terminal bit).

**Draw order of one step** (`propagate`, photon alive):

| Stage | Draws |
|---|---|
| NaN check of `dir.x*dir.y*dir.z*pos.x*pos.y*pos.z` (sets `NO_HIT|NAN_ABORT`), `fill_state` (no hit: `NO_HIT`) | 0 |
| `propagate_to_boundary`: absorption distance, scattering distance | 2 |
| bulk absorption with re-emission components: component, re-emit test | +2 |
| re-emitted: wavelength CDF, time CDF, direction (theta, z), polarization (theta, z) | +6 |
| Rayleigh: cos(theta) variate, phi | +2 |
| surface, default model: one draw, taken *before* the use_weights re-weighting | 1 |
| surface, WLS: one draw; absorbed: re-emit test (+1), re-emitted: wavelength CDF, direction, polarization (+5); reflected: reflect draw (+1) | 1-7 |
| surface, dichroic / angular: one draw | 1 |
| surface, complex: none if `use_weights` and `detect > 0` (forced detection); else one; absorbed: detect test (+1); reflected: diffuse test (+1) | 0-2 |
| diffuse reflector: per rejection iteration (theta, z, accept), then polarization (theta, z) | 3k+2 |
| specular reflector | 0 |
| Fresnel boundary (`propagate_at_boundary`): polarization choice, reflect test | 2 |

`uniform_sphere` draws theta (`2*pi*u`) then z (`2u-1`); `sample_cdf` draws
one uniform. DAQ (`run_daq`, once per event, slot = index within the event):
for a photon whose last triangle is in a channel and whose history matches
`SURFACE_DETECT`: weight test (1); if it passes: time CDF (1), charge CDF (1).

**Known original behaviour that must be (and is) reproduced**: fast-math
FP32 everywhere (`div.approx`, `sqrt.approx`, `sin/cos/lg2/ex2.approx`,
libdevice `acosf/asinf/atan2f`, FTZ); FP32 analytic wires merged with
`t_wire + 1e-6f < t_mesh`; `interp_property` at the top grid point reads
`fp[n]` (the recorded `pad_*` word); Fresnel NaN at normal incidence and at the
critical angle (reflects); 16-bit history; the complex model's thin-film
arithmetic including `cuCsqrtf`/`cuCargf`; `(unsigned)roundf` charge
quantization; `float_to_sortable_int` times with integer `atomicMin/Add/Or`.

## 5. Fail-closed cases

The recorder or the replay raises instead of guessing when W's behaviour is
undefined or unrecorded:

* DAQ tables whose `cdf_y` is one entry shorter than `cdf_x`
  (`Detector._pdf_to_cdf`, i.e. every `set_time_dist*`/`set_charge_dist*`):
  `run_daq` reads one float past the allocation. The default 2-point
  distributions are fine.
* Material/surface indices >= 128 (sign-extended by `convert`), a BVH whose
  root is a leaf, a traversal that needs more than 1000 stack entries,
  dichroic/angular lookups at the last tabulated angle (`iidx+1` out of the
  table), unknown surface models.
* `scatter_first != 0`, `CHROMA_DEVICE_PROFILE`, replaying with other
  Simulation parameters, inputs, detector words or batch boundaries than
  recorded, a replay whose queue lengths at the recorded launch starts differ
  or whose photons use more or fewer draws than recorded, DAQ calls that are
  not the next recorded event. `CHROMA_TRITON_TAPE=canonical` on the Triton
  backend (no tape to replay) raises `NotImplementedError`.
* Bitwise mode needs W's BVH (`geometry.bvh`, from Chroma's cache or the
  PyCUDA builder). A NumPy/Triton port of the builder is not bit-identical
  (the leaf quantization runs in a fast-math kernel and NumPy's argsort tie
  order differs between versions), so the replay uses the recorded `nodes`
  after checking that every other scene word matches the detector.

## 6. Exact-math Triton library (`src/trichroma/engine/exact.py`)

**Method.** ptxas contracts `mul`+`add` into FMA depending on the data flow of
the whole kernel, so a straightforward transcription is not reproducible. The
integrated original `propagate` kernel's SASS was mapped back to its PTX
(`nvdisasm -gp`), every contraction decision was written back as an explicit
`fma.rn`, and every other `mul/add/sub` got `.rn` (which ptxas never
contracts); recompiling that explicit PTX reproduces the original SASS
instruction stream. `exact.py` uses the same explicit operations through
`tl.inline_asm_elementwise`, so its functions yield the original bits from any
Triton kernel. The thin-film block of `propagate_complex` is included verbatim
(`_complex_ptx.py`, generated by `src/trichroma/tape/tools/ptx_extract.py`).

**Conventions.** fp32/int32 tensors of one block shape; `u*` arguments are
pre-drawn uniforms in the order of section 4; history words are uint32 in
int32; masks are int1. Kernels that call `intersect_mesh`/`fill_state_geometry`
must launch with `BLOCK == 32 * num_warps` (the traversal stack is in global
memory; replicated lanes would race).

**Pitfalls found while building it** (relevant to any integration):
Triton's on-disk kernel cache keys on jit *source*, not on the values of
global constexprs, so the complex block's digest is part of
`complex_probabilities`' docstring and checked at import; inline-asm blocks
must not declare registers named like LLVM's (`%f<N>`, `%r<N>`, `%p<N>`),
because a scoped declaration can shadow the register an `$n` operand
refers to (the generated block uses `%cx...`); a probe compiled in isolation
can be lowered differently from the integrated kernel (NVVM keeps `x*x`
instead of `y*y` in two polarization normalizations), so unit probes are
compared against the integrated lowering where they differ.

**API** (all `@triton.jit`):

* Explicit operations: `fmul fadd fsub ffma fnms fms fnmsn fneg fabs fmin fmax
  frcp fdiv fdiv_add fdiv_full fsqrt frsqrt fsin fcos flg2 fex2`, comparisons
  `flt fle fgt fge feq fltu fleu fgtu fgeu fisnan fisfinite`, conversions
  `cvt_f32_u32 cvt_f32_s32 cvt_rzi_s32 floor_to_s32 ceil_to_s32`, the double
  tests of `intersect_triangle` (`d_lt_neg_eps d_gt_one_eps d_gt_eps`),
  `interp_idx_value`, `fbits ffrom_bits`.
* Library functions as lowered: `logf_ expf_ acosf_ asinf_ atan2f_`.
* Vectors: `normalize3(x,y,z)`, `normalize3_xfirst` (isolated-probe variant),
  `dot3`, `dot3_neg`, `cross3`, `rotate(a, cos, sin, axis)`,
  `get_theta_neg(n, d)`, `uniform_sphere(u_theta, u_z)`,
  `random_polarization(u_theta, u_z, dir)`.
* Tables: `interp_property(table, x, start, step, n, mask)`,
  `sample_cdf_uniform(cdf, n, x0, delta, u, mask)`,
  `interp_nonuniform(x, xp, fp, n, mask)`, `interp_idx(x, xp, n, mask) ->
  (idx, iidx, at_top)`.
* Geometry: `node_box(packed xyz, o, d, world, scale) -> (tmin, tmax)`,
  `intersect_triangle(v0, v1, v2, o, d) -> (hit, t)`,
  `intersect_mesh(nodes, vertices, triangles, o, d, last_triangle, active,
  stack, stack_base, world, scale, STACK) -> (triangle, distance, overflow)`,
  `wire_hit(planes, nplanes, o, d, mesh_triangle, mesh_distance, active)`,
  `fill_state_geometry(nodes, vertices, triangles, material_codes, planes,
  nplanes, pos, dir, last_triangle, active, stack, stack_base, world, scale,
  STACK) -> (triangle, distance, surface, material1, material2, normal,
  overflow)`.
* Bulk: `bulk_distances(abs_len, scat_len, u_abs, u_scat, weight, use_weights)
  -> (da, ds, weights_active)`, `bulk_outcome(da, ds, distance)`,
  `advance(pos, dir, t, distance, n1)`, `attenuate(weight, distance,
  abs_len)`, `select_component(comp_absorption, comp_first, num_comp,
  wavelength, abs_len, start, step, n, u, stride, mask)`,
  `rayleigh_scatter(dir, pol, u_theta, u_phi)`, `rayleigh_scatter_raw`.
* Reflectors and boundary: `specular_reflect(dir, normal)`,
  `diffuse_candidate(normal, u_theta, u_z) -> (dir, ndotv)`,
  `diffuse_accept(u, ndotv)`, `fresnel(dir, pol, normal, n1, n2, u_pol,
  u_reflect) -> (dir, pol, reflected)`.
* Surfaces (return an `ACT_*` action; the caller applies it):
  `surface_default(detect, absorb, diffuse, specular, u, weight, use_weights)
  -> (action, weight)`, `surface_wls(absorb, specular, diffuse, reemit, u,
  weight, use_weights) -> (stage, weight, specular, diffuse)`,
  `wls_reemits`, `wls_reflect`, `surface_dichroic(normal, dir, angles,
  nangles, reflect, transmit, stride, wavelength, start, step, n, u, mask) ->
  (action, out_of_table)`, `surface_angular(normal, dir, angles, transmit,
  reflect_specular, reflect_diffuse, nangles, u, weight, use_weights, mask) ->
  (action, weight, out_of_table)`, `complex_probabilities(dir, normal, pol,
  n1, n2, eta, k, wavelength, thickness, transmissive) -> (reflect, absorb,
  axis)`, `surface_complex(detect, reflect, absorb, weight, use_weights) ->
  (forced, detect, reflect, absorb, weight)`, `complex_stage(u, absorb,
  reflect, transmissive)`, `complex_refract(dir, normal, n1, n2, axis) ->
  (dir, pol)`.
* DAQ: `daq_passes(u, weight, global_weight)`, `daq_charge_int(charge,
  charge_unit)`, `daq_charge_float(q_int, charge_unit)`.

A complete step built from these functions is
`src/trichroma/engine/exact_mode.py` (`exact_geometry_kernel` +
`exact_step_kernel`), used by the production engine's exact mode and by the
reference loop.

## 7. Demonstrating equality

```bash
# CUDA environment (PyCUDA): record through the public Simulation API
CHROMA_BACKEND=cuda CHROMA_TRITON_TAPE=record:/lscratch/$USER/t \
    python -m trichroma.tape.verify run --fixture reflect3wires --run visible --output cuda.npz
# Triton environment, public API: chroma.sim.Simulation -> ProductionEngine exact mode;
# every Event output compared with the outputs the tape recorded
CHROMA_BACKEND=triton python -m trichroma.tape.verify replay --fixture reflect3wires --run visible \
    --tape /lscratch/$USER/t --report replay.json
# Reference loop (no detector needed): final device words, photons_end, hits, channels, tracks
python -m trichroma.tape.verify tape --tape /lscratch/$USER/t --report report.json
# Or everything, including the transparency check and the Simulation-level replay:
python -m trichroma.tape.verify all --fixture synthetic --work /lscratch/$USER/w \
    --cuda-python <cuda env python> --triton-python <triton env python> --transparency
```

`verify replay` and `verify compare` report the first differing photon and
word of `photons_end`, hits (in photon order), channels and `photon_tracks`;
`verify tape` reports the same per batch plus the final device state and the
draw-count check (every photon must consume exactly its recorded draws).
`python -m trichroma.tape.probes generate` regenerates the unit ground
truth used by `tests/bitwise/test_exact.py`.

### Results (A100, CUDA 12.4 / PyCUDA for W, Triton 3.1.0, 2026-09-25)

"Public API" is `CHROMA_BACKEND=triton CHROMA_TRITON_TAPE=replay:<dir>`
through `chroma.sim.Simulation`, i.e. `ProductionEngine` in exact mode
(`verify replay`: every Event output against the outputs the tape recorded;
`verify all`: against the CUDA run's outputs). "Reference loop" is `verify
tape` (final device words, `photons_end`, hits, channels, tracks, draw
counts). Both were run from a fresh Triton cache on every tape below, in the
recorded (warp-atomic) and in the sorted queue order.

| Fixture / run | Photons | Launches | Draws | Hits | Public API | Reference loop | Transparency |
|---|---|---|---|---|---|---|---|
| reflect3wires / visible | 100,000 | 9 | 6,113,660 | 3,838 | equal | equal | canonical = recorded |
| reflect3wires_vuv (TPB WLS) / vuv | 100,000 | 9 | 6,198,479 | 1,614 | equal | equal | canonical = recorded |
| pixel_vuv / vuv | 100,000 | 6 | 4,162,873 | 952 | equal | equal | canonical = recorded |
| synthetic / multi_launch (2 batches) | 200,000 | 4+4 | 8,123,179 | 6,716 | equal | equal | canonical = recorded |
| synthetic / weights (use_weights) | 50,000 | 1 | 9,209,627 | 13,458 | equal | equal | plain = recorded |
| synthetic / tracking | 4,000 | 60 | 154,361 | 120 | equal (+tracks) | equal (+tracks) | canonical = recorded |
| synthetic / packed (use_packed) | 80,000 | 3 | 3,243,394 | 2,593 | equal | equal | canonical = recorded |
| synthetic / small_threads (64x16) | 30,000 | 9 | 1,207,309 | 1,052 | equal | equal | canonical = recorded |
| synthetic / adversarial | 32,000 | 1 | 1,341,297 | 2,186 | equal | equal | plain = recorded |
| synthetic / tiny (test data) | 72 | 30 | 7,392 | 20 | equal (+tracks) | equal | canonical = recorded |
| synthetic / tiny_packed (test data; DAQ on W's initial times) | 72 | 1 | 2,348 | -- | equal | equal | canonical = recorded |

`python -m trichroma.tape.verify all --transparency` (fresh CUDA
recordings with sorted queues, the unrecorded canonical run, the reference
loop and the public-API replay) prints `ALL BITWISE EQUAL` for
`reflect3wires`, `reflect3wires_vuv`, `pixel_vuv` and all eight synthetic runs.
The synthetic detector exercises every surface model (default, WLS with
re-emission, dichroic, angular, complex incl. refraction and forced
detection), bulk re-emission, Rayleigh, diffuse/specular reflection, Fresnel,
analytic wires and NaN aborts (adversarial inputs). Two *unrecorded* CUDA
runs of reflect3wires differ in 96,391 of 100,000 photons (queue order after
the first one-step launch), which is why equality is stated against the
recorded schedule or the canonical one.

**Replay time per 100k photons** (warm; the A100 was shared with other jobs):

| Fixture | Engine rounds | Tape read + hash check | Engine `propagate()` | `simulate()` (whole batch) | Reference loop |
|---|---|---|---|---|---|
| reflect3wires (152 steps) | 0.16 s | 0.11 s | 0.28 s | 0.30 s | 0.19 s |
| reflect3wires_vuv (163 steps) | 0.17 s | 0.10 s | 0.28 s | 0.31 s | 0.19 s |
| pixel_vuv (118 steps) | 0.05 s | 0.07 s | 0.12 s | 0.14 s | 0.07 s |

"Engine rounds" is the round loop of the exact mode (renorm, geometry, step,
one host read per round); the reference loop's time excludes reading the
tape. The first `simulate()` in a process adds about 0.5 s (Triton loads the
cached kernels); compiling the exact kernels into an empty cache takes about
40 s. For scale: the CUDA backend's `simulate()` of the same batch took
2.2-2.5 s.

Recording costs about 0.3 s per 100k-photon LAr batch (2.2-2.5 s unrecorded,
2.8-3.0 s recorded) plus about 2 s at Simulation construction (scene
read-back). A 100k-photon reflect3wires batch takes 51 MB (24.5 MB of draws;
36 MB after `tape.compact`) and its scene 14 MB compressed.
