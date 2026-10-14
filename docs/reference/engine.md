# Engine internals

This page is for people working on TriChroma's code. [How TriChroma
scales](../how_it_scales.md) explains the same ideas without code.

## Repository layout

| Location | Contents |
|---|---|
| `src/trichroma/simulation.py` | the `Simulation` class: batching, events, hits, DAQ, pipelining |
| `src/trichroma/engine/core.py` | `ProductionEngine`: owns the compiled scene and runs the kernels |
| `src/trichroma/engine/scene.py`, `bvh_build.py` | scene compiler and tree builder |
| `src/trichroma/engine/fused.py`, `traverse.py`, `physics.py` | transport kernel, geometry search, physics |
| `src/trichroma/engine/exact.py`, `exact_mode.py` | [exact mode](exact_mode.md) |
| `src/trichroma/options.py` | the `CHROMA_TRITON` options |
| `src/trichroma/sources.py` | photon sources generated on the GPU |
| `src/trichroma/tape/` | exact mode's recording format, CUDA-side recorder and checks |
| `src/trichroma/utils.py` | bitwise and statistical comparison of two runs |
| `tests/` | `unit/`, `integration/`, `bitwise/` (recordings) |
| `benchmarks/` | throughput, scaling, profiling, backend comparison, `validation/` |

The `chroma` package itself, including the backend switch
(`chroma.backend`) and the CUDA backend (`chroma.sim_cuda`), lives in
chroma-lite.

## From `simulate()` to the GPU

`Simulation.simulate()` batches events exactly as Chroma does
(`photons_per_batch`), and numbers the photons of all calls in input order;
those numbers key the random numbers. Batches are pipelined: batch k+1 is
propagated while the caller reads the events of batch k. Each batch's outputs
(hits, final photons, DAQ results) are packed on the GPU and copied to pinned
host memory on a second CUDA stream; the propagation needs no host
synchronisation. Exact mode and photon tracking run without the pipeline.

Input photons on the GPU (anything with `__cuda_array_interface__`) are
wrapped with `torch.as_tensor`, without a copy where possible.

## Scene compilation

The scene is compiled once per `Simulation` (`engine/scene.py`). Triangle,
solid and channel numbers stay those of `geometry.flatten()`.

- **Repeated meshes.** Solids with identical meshes (by content hash) become
  instances of one shared mesh tree in the mesh's own frame. A top-level tree
  holds the instances.
- **Trees.** All trees are binned surface-area-heuristic BVHs, built level by
  level with NumPy. Each is stored eight times, once per ray-direction
  octant, in the order a ray of that octant visits it (near child first),
  with escape links, so the search is stackless; the copy is selected from
  the signs of the ray direction. Each node stores the box corner its rays
  reach first and the far one, which saves a min/max per axis in the box
  test. A top-level leaf holds the box of its instance's placed vertices.
- **Box and triangle tests.** Boxes use conservative slabs (Ize 2013) with
  finite inverse directions. Triangles use the watertight test of Woop,
  Benthin and Wald (JCGT 2013) on exact shared float32 vertices, so a ray
  through a shared edge or vertex meets at least one of its triangles.
- **Analytic boxes.** Axis-aligned box solids (up to 64 triangles each) are
  tested as boxes before the top-level tree: a slab test gives the crossing
  face, then 2D edge functions with exact float64 coefficients locate the
  triangle within that face. Boxes nested inside the box a ray starts in are
  skipped.
- **Wires.** Chroma's wire-plane records (`geometry.wireplanes`), intersected
  with a formula free of float32 cancellation and merged with the mesh hit as
  in Chroma (`t_wire + 1e-6 < t_mesh`). The wire a photon has just left is
  skipped. [Differences](differences.md#wire-planes) explains why.
- **Enclosing instances.** An instance whose box covers at least a quarter of
  the scene (a tank or cavity around the rest) is searched after all other
  instances, with exact ties resolved in the walk's order. It is skipped
  entirely while the ray segment up to the nearest hit lies inside its empty
  core: the largest box, sphere or axis-aligned cylinder certified in float64
  to contain no triangle, shrunk by a margin far above float32 rounding.
  `CHROMA_TRITON=strict` turns the reordering off ([differences](differences.md#geometry), G12).

## Transport

**Fused kernel** (`engine/fused.py`, the default). Persistent programs of one
warp each hold up to 128 photons in flight in their own queues and run them
to completion. Each iteration runs one kind of work for the whole warp:

- a step for 32 photons held in registers (analytic boxes, the top-level
  tree, wires, boundary physics);
- a descent batch: 32 queued instance searches, 8 tree nodes per round per
  thread, with idle threads refilled from the queue;
- the completion of steps whose descent has finished;
- new photons from the work list.

Leaf triangle tests are postponed and run across the warp, one (ray,
triangle) pair per thread, then merged in each ray's own test order.
Warp-wide decisions are warp votes. Rare branches (re-emission, wavelength
shifting, diffuse reflection, Fresnel) run only when some thread's photon
requires them. No
scheduling choice changes a photon's result.

**Wavefront scheduler** (`CHROMA_TRITON=wavefront`, and photon tracking).
Device queues with one host read per round, a two-pass boundary search (top
level for every ray, then instance descent for the rays that reach one) and
the same physics. With a certified empty-space grid, the geometry search is
skipped for bulk interactions; the outcome is the same as a full step except for rare
float32 back-face hits at silhouette edges. The last rounds replay from a CUDA
graph.

**Physics.** Chroma's default and wavelength-shifting surface models,
multi-component bulk re-emission, weights and `max_steps`. The complex,
dichroic and angular surface models exist in exact mode only. Flight time
uses the phase velocity `c/n`, and surface re-emission is instantaneous, as
in Chroma.

**Random numbers.** Philox-4x32-10 keyed by the seed; each kind of event
(transport, surface, re-emission, each diffuse-reflection trial, roulette)
is assigned a block of four numbers at counter (photon, steps done, block). The DAQ
uses counters that transport never reaches.

**Compilation.** Kernels are compiled by Triton on first use and cached in
`TRITON_CACHE_DIR`. They are never specialised on the seed. Triton's
first-lookup hashing (~0.4 s) runs on a worker thread while the scene
compiles.

## Debugging options

These `CHROMA_TRITON` options exist for debugging. Except for `legacy-wires`,
results are the same either way.

| Option | Effect |
|---|---|
| `legacy-wires` | CUDA Chroma's wire algorithm alone (`legacy` includes it) |
| `wavefront` | the wavefront scheduler instead of the fused kernel |
| `no-grid` | no empty-space grid for the wavefront scheduler |
| `no-pipeline` | `simulate()` reads, propagates and returns one batch at a time |

## Tests and benchmarks

- `python -m pytest` runs the unit, integration and bitwise tests; tests that
  need a GPU are skipped without one.
- `python -m trichroma.tape.verify` checks exact mode against CUDA Chroma
  ([exact mode](exact_mode.md#checking-equality)).
- `benchmarks/simulate_throughput.py`: `simulate()` throughput on either
  backend.
- `benchmarks/throughput_scaling.py` and `plot_scaling.py`: throughput
  against photons per launch, for any detector.
- `benchmarks/profile_engine.py`: the engine alone, with profiling.
- `benchmarks/compare_backends.py`: statistical comparison of the two
  backends.
- `benchmarks/validation/`: the statistical study behind
  [differences](differences.md#measurements).
