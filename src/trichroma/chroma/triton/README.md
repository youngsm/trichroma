# Triton upgrade for Chroma

> This document records the earlier Triton work that preceded the drop-in
> `chroma.sim.Simulation` backend (see `docs/triton_dropin_design.md` for the
> current engine). Its benchmark scripts, reports and the
> `chroma_lar.triton_scene` module lived in the `chroma-lar` copy that was
> removed from this repository; they remain at commit
> [`78e4b1a`](https://github.com/youngsm/trichroma/tree/78e4b1ab4b64bf5be9f78d848c8490170dae8d68/chroma-lar), where
> the links below point.

This directory contains the foundations of a Triton propagation backend for
Chroma.  The goal is not a separate photon-bomb program or a backend tied to
one detector.  The goal is to preserve Chroma's existing concepts and public
event semantics while replacing the CUDA/PyCUDA propagation implementation
with a data-driven backend that can be faster, asynchronous, and distributed
over several GPUs.

The short version is:

- arbitrary Chroma geometry, optical properties, and event inputs are the
  intended interface;
- work is internally divided according to available memory, not a fixed
  user-visible photon count;
- the reusable geometry, optics, event, memory-planning, BVH, physics, and
  queue layers now exist;
- a fast end-to-end detector specialization demonstrates the performance
  headroom, but it is not yet the general `chroma.sim.Simulation` backend;
- the remaining work is the generic device scene, complete GPU boundary
  dispatch, public `Simulation` adapter, DAQ/tracking integration, and a
  concrete GPU-process IPC adapter.  The local three-stream executor and the
  backend-neutral multi-GPU coordinator are implemented already.

This status distinction matters.  Importing `chroma.triton` does not currently
change the behavior of `chroma.sim.Simulation`, whose default remains the
legacy CUDA backend.

## How this maps onto ordinary Chroma

A normal Chroma simulation has three broad layers:

1. A `Geometry` or `Detector` is flattened into a global triangle namespace.
   Each triangle identifies its two materials, optional surface, solid, and,
   for detector solids, channel.
2. `Simulation.simulate` batches `Photons` or `Event` objects, propagates them,
   and produces optional terminal photons, tracks, hits, and DAQ channels.
3. The CUDA hot loop repeatedly finds the nearest boundary, samples bulk
   physics before that boundary, then applies the relevant surface or Fresnel
   interaction.

The Triton organization keeps those same responsibilities but gives them
explicit, backend-neutral interfaces:

```text
Geometry / Detector
        |
        v
compile_scene() --> immutable HostTritonScene --> device scene (next stage)
                         |                              |
Photons / Events        |                              v
        |                +--------------------> wavefront transport
        v                                               |
EventBatchPlan --> asynchronous H2D / compute / D2H ----+
        |                                               |
        +<-------- PropagationResult + compact hits -----+
        |
        v
ordered Chroma Events / Photons / hits / DAQ
```

The generic compiler treats Chroma's flattened arrays as authoritative.  It
does not assume PMT ordering, a particular detector material, one wavelength,
one source location, or a particular number of photons.

## What has changed

### Generic scene compilation

[`scene.py`](scene.py) compiles Chroma's conventional flattened triangle
geometry into an immutable `HostTritonScene`.  It preserves:

- vertices and indexed triangles;
- the original global triangle IDs;
- `material1_index`, `material2_index`, and `surface_index` per triangle;
- solid IDs and the triangle-to-channel mapping;
- detector channel maps and time/charge response CDFs;
- an existing compatible Chroma BVH, or a newly built packed BVH;
- a feature inventory and deterministic SHA-256 fingerprint of the serialized
  scene contents.

All arrays are owning, contiguous, and read-only.  Mutating the original
Python geometry after compilation therefore cannot silently mutate a cached
scene.  The fingerprint is intended to key versioned host/device caches for
those serialized contents.

There is an important current limitation for non-mesh geometry extensions.
The compiler inventories analytic wire planes, but stores only their count;
it does not yet serialize or hash their parameters.  Its fingerprint is
therefore not a complete cache key for a scene containing those descriptors.

For example, CPU scene compilation is already usable without importing a GPU
runtime:

```python
from chroma.triton.scene import compile_scene

host_scene = compile_scene(detector)
print(host_scene.triangle_count)
print(host_scene.channel_count)
print(host_scene.fingerprint)
```

The flattened mesh portion of the reference `reflect3wires` detector compiles
through this path as 829,488 triangles, 165 solids, 162 channels, four
materials, five surface slots, and 1,105,988 packed BVH nodes. That detector
also contains six analytic wire planes which replace wire meshes and are real
physical primitives. The corrected production specialization retains the
three planes proved reachable from its fixed source region. Strict
compatibility instead retains all six in ascending Chroma scan order and
requires an explicit artifact containing Chroma's cached full global BVH and
global triangle IDs. That artifact pins the mesh MD5, traversal SHA-256, and
optical-semantics SHA-256. The generic host scene is not yet sufficient to
simulate the analytic extensions by itself because it records only their
count; the specialization supplies their complete descriptors.

### Full optical-property representation

[`optics.py`](optics.py) converts Chroma's pointer-rich `Material` and
`Surface` objects into immutable, pointer-free structure-of-arrays tables.
It follows `GPUGeometry`'s established interpolation rule:

```python
numpy.interp(target_wavelengths, property[:, 0], property[:, 1]).astype(float32)
```

The representation includes:

- refractive index, absorption length, and scattering length;
- an arbitrary uniform wavelength grid, chosen when tables are compiled,
  rather than one hard-coded wavelength;
- ragged bulk re-emission components, wavelength CDFs, and time CDFs;
- default, complex thin-film, WLS, dichroic, and angular surface models;
- WLS re-emission, complex `eta`/`k`, thickness, and transmissive state;
- a feature mask intended to let the future device dispatcher select the
  smallest valid GPU kernel variant.

This is important for performance.  Generality does not require every photon
to execute every branch.  Once the generic device dispatcher exists, a scene
with no WLS or complex surface can select a smaller compiled kernel, while the
same host format can represent a scene that does use those features.

### Chroma-compatible event and result contracts

[`runtime.py`](runtime.py) defines CPU-only contracts between
`Simulation.simulate`-style orchestration and a propagation backend:

- `PhotonBatch` contains the complete Chroma photon state;
- `EventBatchPlan` preserves event boundaries and `evidx` while allowing a
  large event to be divided internally;
- every photon receives a stable 64-bit global ID, independent of tile layout;
- `OutputRequest` describes terminal-photon, hit, tracking, and DAQ needs;
- `PropagationResult` and `EventResultAssembler` accept out-of-order tile
  completion and restore original event/photon order;
- compact tracking records are keyed by `(global_photon_id, step)`;
- `PropagationBackend` is the structural interface a CUDA or Triton backend
  implements.

These objects do not yet construct or mutate public `Event` objects.  That
small adapter belongs in `chroma.sim`, where the existing generator behavior
and `keep_*` options can remain stable.

### Memory planning instead of a magic photon count

The public workload may contain any number of photons.  `MemoryTilePlanner`
chooses an internal capacity from:

- free device bytes;
- immutable scene allocation;
- input and live photon-state bytes;
- traversal/queue scratch;
- requested terminal, hit, and tracking output;
- a reserve and safety fraction.

The fast detector benchmark currently happens to use a large A100 work unit,
but that number is benchmark metadata, not part of this architecture.  A
different GPU, scene, or output request should result in a different capacity.

### Overlapping transfer and transport

`AsyncPipelinePlan` describes a ring-buffered three-stream dependency graph:

```text
next tile:       pinned host -> device
current tile:                   propagation
previous tile:                                compact device -> host
```

The default remains the full three-stage graph.  Device-resident inputs omit
H2D, while supplying an `OutputRequest` omits D2H when no host result is
needed.  Slot reuse depends on the earlier tile's actual terminal stage—D2H
when present, otherwise compute—so these reduced graphs cannot overwrite an
in-flight slot.

[`executor.py`](executor.py) executes the graph through user-provided enqueue
callbacks.  Its lazy `TorchCudaPipelineDriver` creates independent CUDA
streams and records an event for every stage; cross-stream and slot-reuse
dependencies become stream-local event waits.  The executor submits every
stage of every tile before synchronizing any terminal event, allowing real
H2D/compute/D2H overlap.  Callbacks own the ring buffers and must use
asynchronous kernels or pinned, nonblocking copies.  A CPU fake-driver suite
checks submission order and every dependency without requiring CUDA.  The
executor exists, but generic device-scene callbacks still need to be connected
to it.

### BVH, physics, and wavefront queues

The lower-level modules provide reusable hot-loop components:

- [`bvh.py`](bvh.py) and [`bvh_kernels.py`](bvh_kernels.py): packed BVH
  construction and nearest-hit traversal while preserving global triangle
  IDs and previous-hit suppression;
- [`physics.py`](physics.py) and
  [`physics_kernels.py`](physics_kernels.py): Rayleigh scattering, competing
  bulk hazards, reflection, refraction, Fresnel sampling, and deterministic
  photon generation;
- [`transport.py`](transport.py): device queues, compaction, and a
  collision-first history kernel for certified homogeneous regions.

The optimized queue path lets collision and boundary kernels append directly
to reusable continuing, boundary, absorbed, invalid, and survivor buffers.
This removes the former full-size status array and its follow-up compaction
launches.  Queue counts can remain on the device across ping-pong history
epochs.  The detector specialization also compacts the rare histories left
after a dense prefix into one cross-tile reservoir, so a large event drains
one sparse tail instead of one tail per input slab.  Boundary rays, analytic
query outputs, PMT traversal results, and merge outputs reuse grow-on-demand
storage rather than allocating on every round.

Certified empty/homogeneous regions are optional accelerators.  The general
correctness path must always be able to fall back to an ordinary global BVH
query.

## Correctness policy

The intended bar is Chroma semantic and distributional equivalence:

- the same geometry/material/surface interpretation;
- the same event boundaries and requested output objects;
- matching distributions of terminal flags, hits, channels, wavelengths,
  times, and weights;
- explicit failure for an unsupported feature or traversal overflow;
- optimized paths differentially checked against the generic flattened path.

### Random-stream alignment and current exact certificates

Equal seeds alone cannot give bitwise agreement with CUDA/XORWOW trajectories.
Legacy Chroma assigns RNG state by worker slot and atomically reorders work, so
its photon-to-random-stream mapping depends on scheduling.  The fast Triton
path instead uses counter-based RNG keyed by stable global photon ID and is
invariant under tiling or GPU count.

For exact differential debugging, [`rng_alignment.py`](rng_alignment.py) and
the matching CUDA [`rng_alignment.h`](../cuda/rng_alignment.h) define an
opt-in shared random tape keyed by `(global photon ID, physical interaction,
draw)`.  A CPU oracle materializes canonical float32 words; CUDA and Triton
load the same words and maintain audited interaction/draw cursors.  Sticky
flags identify draw exhaustion, interaction exhaustion, an ID mismatch, or an
invalid row.  Probe tests verify output words, NaN sentinels, cursors, and
flags bit-for-bit on both runtimes, including permuted worker order.  This
solves RNG-stream alignment itself; ordinary production runs retain the faster
XORWOW/Philox generators.

The `reflect3wires` specialization wires that tape through its supported
450-nm, nonweighted, non-reemitting bulk, default-surface, and dielectric
paths. Alongside the audit cursors, a dense interaction certificate records
one raw word for every committed interaction: four process bits and 28 bits
for the exact semantic draw count. Unused cells remain a sentinel. The
validator compares every CUDA/Triton word and requires each ledger to be a
hole-free prefix ending at its interaction cursor.

A second debug-only certificate now records the complete `Photon` state after
every committed physical interaction. Each record is 15 raw 32-bit words:
position `(x, y, z)`, direction `(x, y, z)`, polarization `(x, y, z)`,
wavelength, time, history, last triangle, weight, and event index. Floating
values are compared by their bit patterns, so signed zero and NaN payloads are
not normalized away; integer fields are stored raw. The process ledger is the
occupancy authority, and all unused state cells must retain the `0xffffffff`
sentinel. Both certificate stores are compiled out of production kernels.

The current proof set replays one fixed 8,192-photon population generated with
source seed `8123` and tape seed `99173`, using a schema-v4 global-BVH artifact:

| Report | Scope | Result |
|---|---|---|
| [`detector_lockstep_global_64_steps_trajectory_certificate.json`](https://github.com/youngsm/trichroma/blob/78e4b1ab4b64bf5be9f78d848c8490170dae8d68/chroma-lar/benchmarks/detector_lockstep_global_64_steps_trajectory_certificate.json) | all 8,192 photons through at most 64 interactions | exact; 134,958 records, 89 survivors |
| [`detector_lockstep_global_tail89_256_steps_trajectory_certificate.json`](https://github.com/youngsm/trichroma/blob/78e4b1ab4b64bf5be9f78d848c8490170dae8d68/chroma-lar/benchmarks/detector_lockstep_global_tail89_256_steps_trajectory_certificate.json) | exactly those 89 original global IDs, replayed from their original source words through at most 256 interactions | exact; all terminal, maximum cursor 126 |
| [`detector_lockstep_global_full_trajectory_certificate.json`](https://github.com/youngsm/trichroma/blob/78e4b1ab4b64bf5be9f78d848c8490170dae8d68/chroma-lar/benchmarks/detector_lockstep_global_full_trajectory_certificate.json) | fail-closed composition of the retained prefix and tail arrays | exact full trajectories for all 8,192 photons |

The first two reports use validator schema v6. Here `exact` means raw-word
equality for every post-interaction state record and process/draw record, as
well as the final position, direction, polarization, time, history, event
index, detected channel, boundary kind, last triangle, cursors, and audit
flags. It also requires equal active global-ID sets, the strict global-mode
`last_instance == -1` invariant, hole-free sentinel suffixes, and no tape or
traversal overflow.

The composite independently reopens the retained CUDA and Triton arrays. It
proves that the selected tail IDs are exactly the prefix survivor queue, that
their initial normalized source words are unchanged, and that their process
and state prefixes agree with the population run. This rechecks 5,696
overlapping interactions rather than trusting the two runs merely because
their seeds match. After removing that deliberate overlap, the complete
population contains 136,280 committed interactions, 509,853 random draws, and
136,280 state records--2,044,200 exact raw state words. The ordered state
transcript has SHA-256
`2aa7f1c3272c40a91b50a4f3697024f7bc1aa022753b7a7b6a8e8a6061a1c446`.
All photons terminate, and the transcript exercises all eight supported
physical outcomes: bulk absorption/scattering; surface absorption, detection,
diffuse reflection, and specular reflection; and dielectric
reflection/transmission.

The certificate also pins the proof sources and compiler policy. The Chroma
tape kernel was compiled by NVCC with the legacy fast-math options and
`CHROMA_FORCE_SCATTER_AT_PASS=0`, i.e. ordinary Chroma behavior rather than the
historical forced-scatter debug hook.

The
[`full-detector ray certificate`](https://github.com/youngsm/trichroma/blob/78e4b1ab4b64bf5be9f78d848c8490170dae8d68/chroma-lar/benchmarks/full_detector_geometry_certificate.json),
backed by a schema-v4 global-BVH artifact, separately checks 16,396 rays
spanning both detector halves and targeted hits on all six wires. Global
triangle, distance, raw oriented normal, solid,
channel, material side, surface name, refractive indices, absorption length,
and scattering length are exact for every ray.

From the `chroma-lar` directory of commit `78e4b1a`, reproduce the 64-step population run
on a CUDA host with the reference Chroma container installed:

```bash
PYTHONPATH=../chroma-lite:. python benchmarks/validate_detector_lockstep.py \
  --count 8192 --max-steps 64 \
  --source-seed 8123 --tape-seed 99173 \
  --tape-interactions 65 --draws-per-interaction 64 \
  --json benchmarks/detector_lockstep_global_64_steps_trajectory_certificate.json \
  --npz-prefix benchmarks/detector_lockstep_global_64_steps_trajectory_certificate
```

Replay exactly the survivor rows to termination, preserving their original
global IDs and source-population addressing:

```bash
PYTHONPATH=../chroma-lite:. python benchmarks/validate_detector_lockstep.py \
  --photon-id-from-npz \
    benchmarks/detector_lockstep_global_64_steps_trajectory_certificate.cuda.npz \
  --source-population 8192 --max-steps 256 \
  --source-seed 8123 --tape-seed 99173 \
  --tape-interactions 257 --draws-per-interaction 64 \
  --json \
    benchmarks/detector_lockstep_global_tail89_256_steps_trajectory_certificate.json \
  --npz-prefix \
    benchmarks/detector_lockstep_global_tail89_256_steps_trajectory_certificate
```

Then make the small fail-closed composite from the retained arrays:

```bash
python benchmarks/compose_detector_full_history_certificate.py \
  --prefix-report \
    benchmarks/detector_lockstep_global_64_steps_trajectory_certificate.json \
  --prefix-npz \
    benchmarks/detector_lockstep_global_64_steps_trajectory_certificate \
  --tail-report \
    benchmarks/detector_lockstep_global_tail89_256_steps_trajectory_certificate.json \
  --tail-npz \
    benchmarks/detector_lockstep_global_tail89_256_steps_trajectory_certificate \
  --output \
    benchmarks/detector_lockstep_global_full_trajectory_certificate.json
```

The older 1-, 16-, and 64-step endpoint/process reports are useful historical
artifacts but are superseded as primary evidence by these state-bearing
trajectory reports; their embedded source hashes predate the current proof
code.

The validator exports the artifact from the same detector object passed to
Chroma. A reusable checked artifact can also be generated from `chroma-lar`
(commit `78e4b1a`) with:

```bash
PYTHONPATH=../chroma-lite:. python -m \
  chroma_lar.triton_scene.chroma_global_bvh reference-global-bvh.npz
```

The artifact is versioned and pickle-free. Schema-v3 artifacts are
intentionally incompatible with schema v4 and must be regenerated after
geometry, optical-table, cached-BVH, or artifact-schema changes.

The independent 4,096-state
[`fresnel_compatibility.json`](https://github.com/youngsm/trichroma/blob/78e4b1ab4b64bf5be9f78d848c8490170dae8d68/chroma-lar/benchmarks/fresnel_compatibility.json)
probe also has exact incident angle, refracted argument/angle, normalized
incidence axis, output direction/polarization, and decisions.  Its only
nonexact diagnostic is the sign bit of a zero reflection coefficient in 164
synthetic equal-index cases; squaring removes that sign, so reflectance,
branching, and output words are unchanged.

This is an exhaustive bitwise certificate for every recorded interaction in
this fixed population and detector specialization, not a universal
mathematical proof over all IEEE-754 inputs, random words, sources, geometries,
optical tables, or unsupported Chroma branches. Its supported scope is the
450-nm, weight-one, no-re-emission specialization described above. Source
generation is outside the propagation tape: exact replay begins from the same
CUDA-normalized input words, and the composite verifies that the tail uses the
same source rows. The corrected production Philox/instancing/tiling/reservoir
path is a deliberately different numerical policy and is not certified by
this tape result. The generic backend still needs tape integration and new
feature-by-feature certificates for paths absent from this specialization.

### Corrected production behavior versus compatibility behavior

The detector specialization keeps these policies explicit.  Its default is
the corrected, high-throughput path: algebraic specular reflection, optimized
instanced PMTs, the three reachability-proved wire planes, Philox RNG, and the
representable-progress PMT rule. A compatibility diagnostic instead sets
`legacy_specular_reflection=True` and
`chroma_mesh_box_compatibility=True`, supplies an explicit
`chroma_global_bvh_artifact=`, and uses the shared tape. That selects Chroma's
legacy reflection/refraction arithmetic, complete cached global BVH, global
triangle IDs, and all six wire planes in ascending scan order. It also turns
the production progress guard off. The artifact pins mesh, traversal, and
optical-semantic hashes. The flags and artifact do not by themselves align
random streams: exact replay must also start from identical normalized photon
words. The performance table below measures the corrected default, not this
diagnostic path.

The full-trajectory certificate is concrete evidence for that strict setting:
with the algebraic-reflection, progress-guard, instancing, and wire-pruning
changes disabled, the forced-scatter change disabled, the full global scene
selected, and the shared tape enabled, every recorded CUDA/Triton interaction
in the certified population is bitwise equal. This is the controlled baseline
against which each corrected production change should be introduced and
measured.

One corrected-mode issue found in the reference study is that Chroma's
fast-math `acos`/Rodrigues specular reflection can
round a ray back inside a 0.075 mm wire, immediately absorbing it in steel with
zero distance and zero time advance.  Algebraic reflection is the same
physical law, avoids that nonphysical self-entry, and is faster. The corrected
production policy also changes geometry organization and applies a progress
guard, so aggregate differences should be attributed to that policy as a
whole unless a controlled A/B test isolates one change.

At 300M photons per replicate:

- the strict aggregate acceptance result is currently **failed**: hit
  fraction, maximum time-quantile shift, and time-quantile RMSE exceed their
  calibrated Chroma-noise gates;
- Chroma hit fraction: `0.0425899`;
- corrected Triton hit fraction: `0.0427338`;
- relative increase: about `0.338%`;
- 99th-percentile arrival time: `205.871 ns` versus `206.316 ns`, a
  `0.445 ns` (`0.216%`) shift;
- channel distribution, binned time KS, joint channel-time shape, and invalid
  hit checks pass.

Changes of this kind must be documented as corrected-physics deviations, not
silently presented as bitwise Chroma compatibility.

The corrected production scheduler also exposed two PMT limit cycles. Adjacent
triangles alternated at 1--7 nm reported distances while every float32
position component remained bitwise unchanged (the local coordinate ULP was
as large as 244 nm).  Last-triangle suppression merely selected the neighbor,
so the rays consumed all 1,000 steps without moving.  CPU and both GPU PMT
traversals now accept a hit only when the exact float32 update
`origin + distance * direction` changes at least one component; traversal
continues to the next triangle otherwise.  This parameter-free progress rule
reduced the reproduced bad seed from 1,000 to 121 boundary rounds and removed
four spurious detections.  It does not truncate legitimate long histories,
which still enter the shared reservoir.

This progress rule remains enabled in corrected production and is disabled in
strict compatibility. The latter can therefore reproduce Chroma's historical
non-advancing intersections. `legacy_specular_reflection=True` selects only
the legacy reflection arithmetic; the global artifact, global-BVH mode,
shared tape, and identical input words are also required for exact replay.

## Current performance evidence

The complete GPU path presently measured is a specialization of
`detector_config_reflect_reflect3wires.py` at 450 nm. It provides measured
performance evidence and a detailed parity study for the reusable components,
not yet the generic runtime.

The timing interval includes source construction through host-readable compact
`(time, channel)` hits.  Geometry initialization and post-run diagnostics are
excluded.  Rates are photons divided by p95 elapsed time across three runs on
an NVIDIA A100-SXM4-40GB.

| Photons per run | Chroma CUDA | Triton | Speedup |
|---:|---:|---:|---:|
| 15M | 1.010M/s | 20.081M/s | 19.88x |
| 30M | 1.226M/s | 25.216M/s | 20.56x |
| 60M | 1.413M/s | 26.112M/s | 18.48x |
| 120M | 1.163M/s | 26.078M/s | 22.43x |
| 180M | 1.319M/s | 26.410M/s | 20.02x |
| 240M | 1.421M/s | 26.477M/s | 18.63x |
| 300M | 1.484M/s | 26.555M/s | 17.89x |

The optimized sweep keeps one scene/workspace set resident and lets the live
memory planner select each internal tile; the 300M call used nine roughly
34.4M-photon source slabs and pooled only 34--45 legitimate tail photons per
replicate.  Its p95 result is 5.31 times the 5M photons/s target and 2.60 times
the earlier Triton implementation.  The requested log--log Chroma/Triton plot,
Opticks reference markers, raw per-run data, and CSV live in the companion
`chroma-lar/benchmarks` directory (commit `78e4b1a`) as
`throughput_scaling_optimized_loglog.*` and
`throughput_scaling_optimized_triton.json`.

The Opticks markers are context, not a head-to-head benchmark. The
[2017 paper](https://doi.org/10.1088/1742-6596/898/4/042001) reports 0.15 s for
0.5M PMT-in-oil photons and 0.25--0.28 s for 1M water-drop photons on a GT
750M. The exact 10.42 s per 100M-photon full-JUNO point on an RTX 5000 Ada is
from the author's [presentation table](https://simoncblyth.github.io/env/presentation/opticks_20241021_krakow_chep2024.html);
the related [2025 proceedings](https://doi.org/10.1051/epjconf/202533701093)
describe it approximately as 10 s. Those geometries, physics, GPUs, and timing
scopes differ from `reflect3wires`, so connecting the markers or dividing
their rates by this A100 result would imply a comparison the data cannot
support.

It would be misleading to promise that every arbitrary geometry and tiny event
will exceed 5M photons/s.  The production rule should instead be: a feature
class is routed to Triton only after semantic parity and warmed sustained
performance both beat the legacy backend; unsupported classes fail closed or
remain on CUDA until completed.

## Multi-GPU organization

Photon transport is nearly embarrassingly parallel.  The intended design is
one process per GPU, each with a resident copy of the immutable device scene.
Workers dynamically claim event fragments identified by stable global photon
ID, overlap input/compute/output locally, and return compact results to an
order-preserving coordinator.

No communication is needed inside the transport hot loop.  Large-event DAQ is
also mergeable: Chroma reduces each channel using minimum time, integer charge
sum, and history bitwise OR.  Per-GPU partials can therefore be merged with the
same `min / sum / OR` operations.  On multiple nodes, hit output should
normally remain sharded with a manifest rather than being gathered through a
single rank.

[`distributed.py`](distributed.py) implements the CPU-only process-level
control plane.  An `EventBatchPlan` first becomes immutable tile assignments;
this happens before worker topology is known, so changing `N` cannot change a
photon's global ID or aligned RNG row.  The coordinator dynamically gives the
next tile to the eligible worker with the lowest capacity-normalized
outstanding load.  Faster workers refill sooner and heterogeneous workers can
advertise weights and maximum tile sizes; there is no static equal partition.

Each worker has a bounded `max_inflight` queue.  A value of two or three lets
its local asynchronous executor copy the next tile while transporting the
current tile and returning compact output from the previous tile.  An
optional `max_inflight_photons` bound also caps staged photons, which matters
when event-derived tiles have different sizes; a tile-count bound alone is
not a memory bound.  The scheduler can skip a temporarily blocked large tile
and feed smaller work to another eligible GPU without changing tile identity
or final order.

Completed results now enter a bounded host-output thread.  It sorts each
compact-hit shard and incrementally reduces sparse DAQ rows while GPU queues
are refilled.  A `result_consumer=` callback can write tile shards to storage
or another process on that same thread, overlapping this I/O with subsequent
H2D and transport.  Consumer calls are in completion order and carry stable
tile/global-ID ranges; returned `tile_results` and merged hits are always
restored to plan/global-photon order.  Equal DAQ times are tied by plan order,
so even the `+0.0`/`-0.0` bit selected by the historical ordered reducer is
independent of worker timing.  The output-future queue is bounded and applies
backpressure only after free GPU slots have been refilled.

The only built-in distributed payloads are compact hit rows and sparse DAQ
partials; DAQ merge is exactly minimum time, checked signed-integer charge
sum, and history bitwise OR.  Full live photon state never passes through a
central reducer.  A typical coordinator configuration is:

```python
coordinator = DistributedCoordinator(
    worker_specs,
    overlap_output=True,
    max_pending_output_tiles=2 * len(worker_specs),
)
result = coordinator.run(assignments, result_consumer=write_tile_shard)
```

`WorkerClient` is the explicit boundary for a production IPC adapter: its
nonblocking `submit` returns a standard future and its `cancel` control message
propagates run failure or generator cancellation.  One server process should
select one GPU once, create one CUDA context, retain its scene, and feed its
three-stream executor.  A threaded fake-worker suite proves weighted dynamic
claiming, bounded overlap, out-of-order completion repair, and fail-fast
cancellation without needing CUDA.  The remaining distributed work is to
connect that protocol to actual persistent GPU processes and the public
`Simulation` adapter, plus optional multi-node transports/manifests.

### Aggregate scaling model (not a multi-GPU measurement)

`estimate_distributed_scaling` makes the throughput accounting explicit.  It
takes per-worker transport, H2D, and D2H rates; because those three stages are
overlapped, the slowest rate limits each worker.  Worker rates add after a
load-balance factor, then optional shared source and output rates cap the
aggregate.  It is a steady-state model: startup/drain latency and a workload
with fewer tiles than GPUs are not represented.

Using the measured **single-A100** large-run rate of 26.555M photons/s and no
additional bottleneck gives these ideal aggregate bounds:

| A100 workers | Measured input rate per GPU | Ideal aggregate model |
|---:|---:|---:|
| 1 | 26.555M/s | 26.555M/s |
| 2 | 26.555M/s | 53.110M/s |
| 4 | 26.555M/s | 106.220M/s |
| 8 | 26.555M/s | 212.440M/s |

Thus two GPUs reach **50M/s aggregate** only if the complete distributed path
retains at least `50 / 53.11 = 94.14%` parallel efficiency.  This narrow 5.86%
overhead budget is why pinned nonblocking copies, three in-flight slots,
dynamic load balancing, compact output, and overlapped reduction matter.  It
does **not** mean one GPU runs at 50M/s: reaching **50M/s on one GPU** still
requires a 1.88x improvement over the measured 26.555M/s kernel/runtime rate.
No multi-GPU hardware result has been measured yet, so the table must not be
reported as benchmark data.

The same calculation is available programmatically:

```python
from chroma.triton.distributed import estimate_distributed_scaling

model = estimate_distributed_scaling(
    26.555e6,
    worker_count=2,
    load_balance_efficiency=0.95,
    # Add measured common-service caps when known:
    # shared_input_rate=...,
    # shared_output_rate=...,
)
print(model.estimated_aggregate_photons_per_second)
print(model.required_parallel_efficiency(50e6))  # 0.9414...
```

## What remains before this is a general backend

The remaining production work is substantial but contains no known Triton
capability blocker:

1. Complete serialization and hashing for non-mesh extension primitives, then
   upload `HostTritonScene` into a generic immutable device scene.
2. Promote the strict specialization's implemented global-hit normal
   orientation, material-side selection, solid identity, and channel lookup
   into the generic device runtime, then add spectral interpolation.
3. Implement and validate device dispatch for bulk re-emission, weighting,
   default/complex/WLS/dichroic/angular surfaces, and dielectric boundaries.
4. Use the direct device queues and device-resident counts throughout the
   generic scheduler, then connect its scene/transport callbacks to the
   implemented asynchronous executor.
5. Adapt `chroma.sim.Simulation` without changing its event-generator API,
   `keep_*` behavior, tracking, flat/grouped hits, or DAQ results.
6. Unify CUDA context ownership so PyCUDA fallback and Torch/Triton cannot
   corrupt one another's context stack.
7. Add versioned scene/kernel/autotuning caches, overflow recovery, local
   multi-GPU workers, and multi-node output/checkpoint manifests.
8. Extend the shared random tape beyond the detector specialization's current
   feature set and broaden bounded/terminal certificates to new populations,
   optical branches, and generic scheduler paths alongside ensemble tests.
9. Run feature-by-feature statistical parity and performance gates before
   changing the default backend.

The existing detector specialization should remain an optimization plugin.
Its analytic wires, instanced PMTs, source reachability, and monochromatic
kernel variants should be selected because generic scene analysis proves them
safe—not because the runtime recognizes a detector configuration by name.

## Tests

CPU-only tests can be run without a CUDA context:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  test/test_triton_bvh.py \
  test/test_triton_physics.py \
  test/test_triton_transport.py \
  test/test_triton_rng_alignment.py \
  test/test_triton_optics.py \
  test/test_triton_scene.py \
  test/test_triton_runtime.py
```

On an available CUDA GPU, remove the `CUDA_VISIBLE_DEVICES=''` prefix while
keeping the same test list; that also exercises the Triton BVH, physics,
source, and queue kernels plus a real four-tile/three-slot Torch executor
smoke.  The CPU fake-driver executor tests remain active in both modes.
