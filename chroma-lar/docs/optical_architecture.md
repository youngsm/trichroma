# Optical engine structure and extension points

The fast engine is a detector specialization with reusable optical physics.
Calibration changes are straightforward. A new physical process or arbitrary
detector geometry still requires implementation and validation.

## Responsibilities

| Layer | Implementation | Responsibility |
|---|---|---|
| Application API | `optical_simulation.py` | Photon/deposition/voxel entry points and complete event membership |
| Readout | `optical_readout.py` | Shared PMT response, CPU/GPU digitizer selection, result format and provenance |
| Calibration | `optical_calibration.py` | Load and validate spectra, timing laws, materials, surfaces and electronics |
| Compiled model | `spectral_model.py` | Validate supported physics, upload spectral/source tables, fingerprint effective inputs |
| State contract | `spectral_state.py` | Named structure-of-arrays fields and validated scheduling settings |
| Scheduler | `spectral_backend.py` | Source launches, live queues, bulk/boundary handoff, completion audit and hit extraction |
| Boundary service | `triton_scene/detector_query.py` | Own analytic/PMT accelerators and scratch storage; return nearest boundaries |
| Physics kernels | `spectral_kernels.py`, `chroma.triton.*_kernels` | Numerical source, bulk and surface interactions |
| Geometry compiler | `triton_scene/compiler.py` | Translate detector configuration into full analytic/instanced scene and prove specialization assumptions |

The spectral path no longer constructs `Reflect3WiresTritonSimulation` or calls
its private methods. The older monochromatic/compatibility backend remains
available independently. Both optical pipelines share readout and metadata
construction. Public simulation entry points and saved NPZ fields are retained.

`PhotonState`, `SpectralProperties`, `SourceProperties` and `BoundaryHit` provide
named fields on the host. Their tuple order defines the flat kernel argument
contract. Changing that order requires updating every consuming kernel. Naming
the fields does not introduce GPU object allocation or per-photon Python work.

## Adding capabilities

| Change | Where to work | Validation needed |
|---|---|---|
| LAr spectrum, yield, emission lifetimes, material optical tables | Calibration manifest/tables | Coverage, normalization and source distribution checks |
| TPB spectrum, delay, efficiency or escape probability within the effective surface model | Calibration | WLS spectrum/time/material-side tests |
| PMT collection efficiency, TTS, charge, pulse shape, noise or ADC settings | Calibration and existing response models | PE/readout statistics and CPU/GPU digitizer agreement |
| Different source distribution | Source generator plus GPU source adapter if needed | Photon identity, event membership, spectrum/timing and CPU/GPU agreement |
| A new bulk process, such as bulk reemission | Table compiler, shared samplers and both transport implementations | Independent process-level reference and detector distributions; reserve/document RNG streams |
| New geometric primitives | Scene representation/compiler and boundary service | Intersection/material-side tests, boundary-position invariants and empty-region certification |
| Arbitrary detector geometry at comparable throughput | General compiler/acceleration selection | General mesh robustness, source-domain support and benchmarks across geometries |

Construct a new simulation after changing calibration. Calibration objects and
device buffers are not immutable snapshots; mutating them in an existing
simulation can mix compiled and live parameters and invalidate provenance.
Do not share a simulation across concurrent events or CUDA streams: queue and
boundary workspaces are reused. Geometry query outputs are views valid until
the next query. The scheduler consumes them before mutating their queues.

Remaining limitations include the negative-x source adapter, opaque enclosure
requirements, no automatic tiling for oversized source populations, no bulk
reemission, and no general weighted-photon support. The generic and specialized
transport implementations must continue to be checked against each other and
independent references when extending physics. The old mesh boundary issues and
native-CUDA comparison discrepancy remain documented in the validation reports.

## Can this become one standalone Chroma replacement package?

Yes. There is no requirement that the Triton implementation live in two
distributions. Extraction and full API compatibility are separate work items;
the present checkout is not an installable drop-in replacement.

The reusable kernels, state/optical tables, general scene compilation, readout
and acceleration services can live in one core package. LAr configurations,
PMT assets and scintillation defaults should be detector inputs or an optional
adapter, rather than imports required by the core engine. In particular,
`compile_reflect3wires_scene()` currently loads `chroma_lar` configuration and
geometry builders, so the measured fast path is not yet detector-independent.

For unchanged application imports, a distribution named `triton-chroma` could
provide the Python package `chroma`, retaining `chroma.geometry`,
`chroma.detector`, `chroma.event` and `chroma.sim.Simulation`. It would replace
the old distribution in an environment. Installing two distributions that own
the same `chroma` files is not a reliable coexistence strategy. A separate
`triton_chroma` import namespace is easier to install alongside Chroma but
requires application import changes.

A replacement would need:

1. **A standalone build and dependency graph.** Package the required CPU data
   types, geometry utilities, assets and kernels; replace the current mandatory
   PyCUDA setup dependency with Torch/Triton and make legacy CUDA optional.
   Verify wheel installation and imports outside the source checkout.
2. **A `Simulation` compatibility adapter.** Accept existing detectors,
   `Photons`/`Event` inputs and iterables, constructor options, batching and
   retention flags; yield existing `Event` objects with the expected photon,
   hit and DAQ fields. Current spectral entry points return different result
   types. Accepting `Photons` inputs alone does not provide this contract.
   It should also obtain optics and readout settings from existing material,
   surface and `Detector` definitions, so unchanged scripts do not suddenly
   require a separate calibration manifest.
3. **General scene support.** Compile a Chroma `Geometry`/`Detector` directly,
   choosing analytic/instanced acceleration where certified and a robust mesh
   path elsewhere. Existing general mesh transport provides a starting point,
   but its unresolved boundary behavior needs fixing. The 25M/s detector result
   does not establish that rate for arbitrary meshes.
4. **A declared compatibility scope.** Current spectral physics supports
   default/WLS surfaces and unweighted photons. Full replacement also requires
   the requested legacy features, such as bulk reemission, other surface
   models, tracking, weighted transport and relevant DAQ/PDF APIs. Low-level
   clients of `chroma.gpu` expect PyCUDA objects and need separate compatibility
   work. Unsupported features should fail explicitly.
5. **Compatibility tests from unchanged applications.** Check imports,
   signatures, array layouts/dtypes, event ordering, flags, channel mapping,
   retention behavior, serialization and physics distributions. Statistical
   equivalence and deterministic Triton runs are reasonable contracts; matching
   the old CUDA random histories bit for bit is a separate requirement, and
   known numerical defects should not silently become the specification.

The practical first release would promise the high-level optical
`Simulation`/`Photons`/`Event` interface for an explicit set of supported
features, with this detector's fast compiler as one backend. Claiming every
legacy Chroma feature and low-level GPU API would require substantially more
work than consolidating source directories.
