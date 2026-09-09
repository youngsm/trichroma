# Primitive region compiler prototype

This work lives on `feature/primitive-regions` in the `trichroma-primitives`
worktree. Published `main` remains the consolidated simulation baseline.

The compiler now discovers homogeneous bulk regions from geometry and material
declarations. The fast LAr transport consumes those certificates, including
regions in either detector half. Geometry descriptions and certificate discovery
live in `chroma.triton`; the adapter for the existing LAr artifact remains in
`chroma_lar.triton_scene`.

## What is implemented

`chroma.triton.primitives` provides immutable box volumes, closed convex mesh
volumes, shared triangle meshes, rotated mesh instances, finite periodic wire
arrays, and explicitly bounded obstacles. A convex volume retains the actual
input mesh planes: a faceted Chroma cylinder is not replaced by a smooth
cylinder. An arbitrary nonconvex mesh can be an obstacle, but cannot yet certify
its own interior as a volume.

`compile_regions(scene)` subtracts conservative obstacle bounds from eligible
interiors. Convex mesh seed boxes are calculated using face-plane support
functions. This first version keeps one inscribed seed box per convex volume;
it does not attempt to cover all remaining wedges near a curved enclosure.
Nested interfaces must expose the declared surrounding material. Open,
partially overlapping or ambiguous material regions are left uncertified.
The existing LAr cavity declaration has inconsistent surrounding-material
labels, so it is rejected for bulk acceleration while its physical boundaries
remain in exact traversal.

Each emitted region is checked against every original blocker and rounded inward
to representable float32 faces. Exactly representable faces are retained, with
open membership excluding points on the face. Bounded cell budgets can discard acceleration
opportunities; they never discard physical geometry. Renaming detector labels,
changing world axes, or changing PMT counts does not alter these rules.
For more than 4,096 exposed obstacles, the default compiler contracts each seed
about its center against **every** obstacle in one vectorized pass. This bounds
construction cost for dense sensor arrays and reports the affected domain in
`coarse_domains`. If the center is obstructed, that seed produces no certificate.

At event setup, the LAr scheduler chooses one region using source positions and
photon counts. The existing bulk kernel receives its six bounds. There is no
per-photon search over all certificates. A photon outside that region, or whose
next flight would leave it, uses the exact boundary query. Crossing an
acceleration-cell face is not a physical interaction, does not move the photon,
and consumes no random numbers.

`FastOpticalSimulation(..., region_mode="automatic")` is the default.
`region_mode="legacy"` retains the old detector-specific region for matched
benchmarks; `"disabled"` forces all flights through geometry queries.

## Water Cherenkov example

The second fixture follows the supplied Theia-like cap/barrel and rectangular
PMT placement rules, using the user's requested water Cherenkov interpretation.
It has no dependency on the separate scintillator/dichroicon database.

```python
from chroma.triton.examples.theia import build_theia, cherenkov_photons
from chroma.triton.regions import compile_regions
from chroma.triton.spectral import SpectralSimulation
import numpy as np

fixture = build_theia()  # 25.5 m radius, 81% nominal coverage, 20-inch PMTs
# Small regression fixture: build_theia(5000., coverage=.05, diameter=304.8)
# Rectangular example: build_theia((10000., 8000., 12000.), coverage=.05, diameter=304.8)
certificates = compile_regions(fixture.primitives())
simulation = SpectralSimulation(fixture.detector(), wavelengths=np.arange(300., 651., 5.))
result = simulation.simulate(cherenkov_photons(100000, seed=17), seed=17)
```

The primary fixture has **49,684 nominal 20-inch sensors**, a 25.5 m placement
radius and 51 m placement height. The size/coverage are assumptions chosen to
match the requested roughly 50,000 PMTs. It retains one shared 660-triangle
sensor mesh (32,791,840 total triangles if flattened, including the enclosure).
The small regression cylinder has 326 sensors and 215,560 flattened triangles;
the rectangular regression fixture has 340 sensors.
A proper rigid rotation points each sensor's local +Y face inward;
parallel/antiparallel cases do not reflect the mesh.

The placement removes 582 barrel-edge sensors to clear the end caps. A
conservative finite-cylinder enclosure of every sensor is checked against all
147,804 potentially overlapping pairs, including a float32 transform guard.
The minimum proven separating gap is 7.63 mm. Layouts that cannot be certified
disjoint raise an error by default. The original 24 m / 90% layout had actual
cap/barrel intersections; it is retained only as a numerical regression with
explicitly disabled clearance validation.

Explicit assumptions:

* Water, glass and photocathode tables come from the bundled
  `chroma.demo.optics`; these are example inputs, not a Theia calibration.
* The bundled reduced SNO PMT profile is scaled from 8 to 20 inches for the main
  fixture (12 inches for the small regression fixture). Glass and
  photocathode interfaces are explicit. The enclosure and PMT backs absorb.
* The source is a fixed photon count along a 1 m straight beta=1 track, using
  the Frank–Tamm spectral shape over 300–650 nm, a dispersive Cherenkov cone,
  and polarization in the track/photon plane. It does not simulate charged
  particle energy loss or determine photon yield from deposited energy.
* The benchmark adds illustrative 2 ns RMS TTS, 30 ns transit time, 90%
  collection efficiency, a tabulated SPE charge distribution and noisy ADC
  waveforms. All parameters are recorded in its JSON output.
* There is no LAB/PPO, TPB, bulk reemission, dichroicon or LAPPD in this fixture.

The water fixture exercises automatic region discovery and the existing exact
shared-mesh GPU traversal independently. Its **full optical benchmark currently
uses the general flattened-mesh spectral engine**, which does not consume
region certificates. Its rate must not be described as the generalized fast
transport rate, or compared directly with LAr as an isolated compiler speedup.

## Pixel TPC example

The existing `detector_config_pixel` supplies 164 PMTs on the ±Y walls,
pixel surfaces on ±X and the central cathode. `pixel_primitives(geometry)`
declares its three box volumes and shared PMT mesh instances; the same generic
compiler discovers clear LAr cells in both detector halves. The adapter checks
the known builder's axis-aligned box-face and material contracts before issuing
volume declarations. It is not an arbitrary-mesh topology recognizer.

The original configuration sets **`pixel_simplified=True`**. Its optical model
uses the area-weighted gold/FR-4/copper response of one million pads per face,
with the configured steel border. The full-detector benchmark preserves that
approximation. The viewer's separate resolved 32 × 32 pad patch is a geometry
close-up and is not the full optical benchmark.

`build_pixel_calibration.py` reproduces the saved synthetic bundle from the
original pixel configuration and the existing synthetic LAr/TPB/PMT/noisy ADC
bundle. The pixel benchmark samples equal populations in both detector halves,
compares spectral transport with the CPU reference, checks GPU digitization
against CPU readout, and times complete host-readable events. Like water, its
transport currently uses the general flattened-mesh engine.

## Scalable independent CPU reference

`SpectralSimulation(..., backend="reference_bvh")` retains the NumPy optical
physics, adding CPU traversal of conservative BVH bounds. It is checked against the original
`backend="reference"`, which remains available. Candidate triangles are sorted
by original ID to preserve ties; traversal has no candidate cap or nearest-hit
pruning. This permits CPU comparisons on the actual 32.8-million-triangle water
detector instead of validating only a reduced fixture.

Packed BVH parent construction now uses bounded vectorized NumPy chunks. Tests
compare serialized parent words with the former scalar construction, including
partial groups and chunk boundaries. This reduces host construction overhead;
the GPU traversal format and kernel are unchanged.

The general spectral engine selects `high_precision=True` for CPU and GPU
triangle queries. Vertices and rays remain stored in float32; slab and triangle
calculations promote them to float64 **before** subtracting world coordinates,
and return float32 distances. This matters for small, grazing PMT facets at
tens of metres: a captured ray falsely intersected different faces in both
float32 implementations, while float64 rejects both. The case and 4,096 nearby
rays are checked against the independent CPU oracle. The high-precision mode
also uses conservative padded slabs and a deterministic lowest-ID tie rule.
The original query mode remains the default for existing local-coordinate
traversal and the fast LAr scheduler. Fixed intersection tolerances and stored
float32 geometry are still numerical limits, not a formal watertight predicate
at arbitrary scales.

General mesh transport now constructs outgoing origins from the actual
triangle plane and its float32 error scale. This prevents false second
interactions with numerically coplanar adjacent triangles. The CPU and GPU
constructions agree exactly for 2,048 tested origins at the water detector's
scale. A captured failing PMT ray clears its former false interface; a separate
0.15 mm physical interface remains visible. This numerical offset adds no
flight time and consumes no random numbers. Geometry much smaller than the
floating-point uncertainty at its world coordinates still needs local
coordinates or greater precision; arbitrary scale invariance is not claimed.

## Remaining work for general fast transport

The fast boundary scheduler still consumes the existing LAr artifact and its
box/wire boundary projection data. General primitive boundary lowering,
multiple shared mesh asset groups in optical transport, general material
initialization/source validation, and binding generic spectral tables to that
scheduler remain to be implemented. The new shared-instance traversal adapter
handles geometry only. A drop-in `chroma.sim.Simulation` replacement is not
claimed by this prototype.

The next useful measurement is the same water event on a generic instanced
boundary scheduler with and without its discovered bulk region. That will
separate gains from instancing, region discovery and physics scheduling.

## Reproduction

Use the local GPU environment described in the root README, with
`PYTHONPATH=chroma-lite:chroma-lar`. No cloud service is used.

```bash
python scripts/check_optics.py --output-dir /tmp/trichroma-primitive-checks
python chroma-lar/benchmarks/check_optical_refactor.py compare \
  --output-dir chroma-lar/benchmarks/optical_validation/primitive_regions
python chroma-lar/benchmarks/benchmark_fast_full.py --counts 30000000 \
  --repeats 5 --depositions --region-mode automatic \
  --calibration chroma-lar/benchmarks/optical_validation/full_detector_synthetic_calibration_noise.json \
  --output chroma-lar/benchmarks/optical_validation/primitive_regions/automatic_30m.json
python chroma-lar/benchmarks/benchmark_fast_full.py --counts 30000000 \
  --repeats 5 --depositions --region-mode legacy \
  --calibration chroma-lar/benchmarks/optical_validation/full_detector_synthetic_calibration_noise.json \
  --output chroma-lar/benchmarks/optical_validation/primitive_regions/legacy_30m.json
python chroma-lar/benchmarks/benchmark_theia.py --counts 100000 1000000 \
  --repeats 3 --output chroma-lar/benchmarks/optical_validation/primitive_regions/theia_water.json
python chroma-lar/benchmarks/build_pixel_calibration.py
python chroma-lar/benchmarks/benchmark_pixel.py --counts 100000 1000000 \
  --repeats 3 --output chroma-lar/benchmarks/optical_validation/primitive_regions/pixel_tpc.json
```

Archived pre-prototype reports and their source hashes remain untouched. New
validation evidence is stored under `optical_validation/primitive_regions`.
