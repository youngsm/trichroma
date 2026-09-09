# Original Chroma-Lite bitwise parity

Full bitwise parity with the original installation is an acceptance requirement,
**not a property currently established for the spectral production engine**.
Passing NumPy/Triton comparisons, matching distributions, and reproducing an
older Triton snapshot do not establish that requirement.

The reference installation must be pinned by its source-file hashes, local
patches, CUDA compiler/options, runtime, geometry/BVH, optical tables, normalized
input photon words, and random-number history. A candidate reference manifest is
saved under `optical_validation/primitive_regions/legacy_parity/`; the installed
working tree and its clean Git revision differ, including wire arithmetic.
Neither reference is to be modified in place.

## Acceptance contract

Compare raw IEEE-754 words, including signed zero and NaN payloads, for every
photon's position, direction, polarization, wavelength, time and weight. Compare
history, last triangle, event identity, channel identity, active populations,
interaction counts and random draw counts exactly. Record the complete state
after each interaction so a final-state match cannot hide an intermediate
divergence. Output ordering must be stated; reordering by persistent photon ID
does not make an unordered hit list byte-identical as an array.

Ordinary Chroma assigns XORWOW state to worker slots. Queue compaction can change
which photon uses which slot. Equal seeds alone therefore do not align its
trajectories with the production Philox engine. A controlled transport proof
must supply identical random words and verify draw consumption. A proof tied
to the original native generator additionally needs to connect those recorded
words to an original CUDA execution, rather than substitute an unrelated tape
and call that native-run parity.

Compatibility execution must preserve the original geometry traversal,
normalization, arithmetic order, interpolation, random call order and historical
boundary behavior. Numerical corrections need explicit, separately exercised
switches. New physical laws without an original counterpart require separate
validation and must not be included in an original-equivalence claim.

## Native reference and current evidence

The native audits use the installed working tree at
`/sdf/group/neutrino/youngsam/sim/chroma-lite`, Git revision
`b67bc6db09dcf4eb7f52052f2451736012829883` **plus its existing local changes**.
This selection was inferred from the available installation. It was not replaced
with the clean revision: the installed analytic-wire implementation uses FP32
arithmetic and differs from the older FP64 implementation. Each capture pins the
actual source files, CUDA options, fixture, uploaded tables and geometry.

`capture_original_chroma.py` includes the unchanged original headers and records
native XORWOW draws and all 15 photon words after every interaction. Before
using a recording, it compares the recorder with the original public
`GPUPhotons.propagate` call, including the final native RNG bytes. New captures
also rerun the ordinary API at every interaction-prefix length and require its
output to equal that recorded prefix.

`replay_original_chroma.py --native-rng` independently normalizes the supplied
raw photon words and initializes/evolves XORWOW in Triton. Recorded random draws
are not used as simulation inputs in this mode. The separate RNG test covers
225 combinations of 64-bit seeds, subsequences and offsets, and 7,200 native
integer/uniform draws, with zero different raw words.

Passed evidence snapshots under
`chroma-lar/benchmarks/optical_validation/primitive_regions/legacy_parity/` include:

| Fixture | Native photons | Comparison |
| --- | ---: | --- |
| Ten elementary transport fixtures | 81,920 | Every recorded photon word and draw count; initial normalized words supplied by CUDA |
| Spectral Fresnel, mixed spectral transport, two-component bulk reemission | 24,576 | Every recorded word and draw count, independent native RNG; bounded 128-step runs |
| Mixed spectral transport through completion | 8,192 | 312,357 interactions, 1,297,899 draws; all states and final RNG words equal, no survivors |
| Two-component bulk reemission through completion | 8,192 | 74,245 interactions, 360,457 draws; all states and final RNG words equal, no survivors |
| Full pixel TPC, visible and VUV/TPB fixtures | 16,384 | All intermediate/final photon words and final native RNG words equal, no survivors |
| Full reflect3wires, visible and VUV/TPB fixtures | 16,384 | All intermediate/final photon words and final native RNG words equal, no survivors |
| Full reflect3wires, visible and VUV/TPB tenfold follow-ups | 163,840 | Every native launch output and final RNG word equal; 2,706,864 interactions, 10,129,627 draws, no survivors |
| Analytic wire fixture, tenfold follow-up | 81,920 | Every native launch output and final RNG word equal; 126,306 interactions, 407,876 draws, no survivors |
| Mixed spectral transport across native queue compaction | 65,536 | All original launch outputs and native RNG words equal under the observed queue schedule; 128-step budget |
| Mixed spectral transport across native queue compaction, through completion | 81,920 | All 16 native launch outputs and native RNG words equal; 3,230,709 interactions, 13,411,702 draws, no survivors |

These are finite, source-pinned tests, not a proof of all possible inputs.
The corrected production engine is not asserted to produce native Chroma's raw
words. Failed earlier wire reports are retained and superseded by the passing
`triton_detectors_final` and `triton_schedule_final_81920_analytic_wires` audits.

The installed wire primitive required matching an otherwise hidden compiler
choice: the integrated original CUDA kernel evaluates its quadratic coefficient
as `fma(dn, dn, round(dv*dv))`. Compiling the identical isolated C++ expression
reverses the rounded product. Cancellation in the discriminant changes both
near-tangent hit positions and some hit/miss outcomes. The compatibility adapter
pins the integrated operation order explicitly, while retaining the installed
FP32 algorithm. It does not dispatch the original CUDA transport kernel.
A small immutable native fixture, including the first divergent photon, tests
both complete histories and final-only recording in the regular GPU suite.
Both versions also pass Compute Sanitizer with zero memory errors.
The four completed 81,920-photon follow-ups total 327,680 photons, 6,063,879
interactions and 23,949,205 independently generated draws. Their aggregate report
is [FINAL_REPORT.json](../chroma-lar/benchmarks/optical_validation/primitive_regions/legacy_parity/FINAL_REPORT.json).

## Original queue ordering is not seed-repeatable

`check_original_repeatability.py` compares the untouched original with itself,
holding the geometry, input photon words and complete initial RNG bytes fixed.
Three 8,192-photon runs were identical. Three 65,536-photon one-step runs were
also identical. At 128 steps, the first pair of 65,536-photon runs differed for
61,550 photons, including 53,895 history flags. The original atomic queue changes
which worker RNG state a surviving photon receives.

Consequently, an arbitrary large original run is not determined by its seed
alone. A seed-only promise of identical output to every such execution would be
false even for the original installation compared with itself.

`capture_original_schedule.py` observes the actual original public API's kernel
launches and downloads their queue ordering and outputs. It does not replace
the CUDA kernels or their random generator. `replay_original_schedule.py` uses
only that ordering and launch metadata, then independently evolves photon and
RNG states in Triton. Expected native states are read as comparison targets.
The 65,536-photon check passed all 12 launches, covering 1,738,429 propagated
interactions and 7,335,490 independently generated draws. Its final launch
contains 117 interactions per surviving worker at most; the original state is
verified at the launch boundary, not after each internal interaction of that
tail. 4,568 photons remain alive at this explicitly bounded 128-step budget.

## Compatibility executor and limits

`chroma_lar.triton_scene.legacy_spectral.propagate_legacy` accepts scene arrays,
raw `[N,15]` photon words, an interaction budget, and exactly one random source:
a seed, explicit six-word XORWOW worker states, or an audited random tape.
It does not require expected output states. `record_history=False` keeps final
states, interaction counts and total draw counts without allocating a complete
history tensor.

Historical mode retains the original normalization, interpolation, arithmetic,
boundary behavior and specular-polarization behavior. It currently supports
dielectrics, Rayleigh scattering, default/WLS surfaces and multi-component bulk
reemission. It rejects unsupported surface models and non-unit input weights.
Complex films, dichroic/angular models, importance weighting and scatter-first
controls are not yet covered by this compatibility executor.

The original has no counterpart for several added source/readout laws, including
the chosen effective TPB delay and PMT response calibration. Those require
separate physical and implementation tests. A new-law output cannot be part of
an original-equivalence claim. Compatibility audit speed is also separate from
the corrected full-pipeline throughput measurement.

## Reproduce a native comparison

Run capture commands in an environment with the selected original installation,
PyCUDA and its CUDA compiler. The original Chroma root is explicit; the detector
fixture builder comes from this checkout. No source under the reference root is
modified.

```sh
python chroma-lar/benchmarks/capture_original_chroma.py \
  --reference-root /path/to/original/chroma-lite \
  --cases analytic_wires --count 8192 --output /tmp/native-wire-capture
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/replay_original_chroma.py \
  --capture /tmp/native-wire-capture --native-rng --output /tmp/triton-wire-replay
```

The large-queue version observes an actual original execution, then independently
replays its worker assignment. Its expected output words never become the
Triton simulation's input states:

```sh
python chroma-lar/benchmarks/capture_original_schedule.py \
  --reference-root /path/to/original/chroma-lite \
  --case reflect3wires_vuv --count 81920 --max-steps 8192 \
  --output /tmp/native-schedule
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/replay_original_schedule.py \
  --capture /tmp/native-schedule --output /tmp/triton-schedule-replay
```

JSON reports and the small immutable native regression fixture are versioned.
Large generated NPZ recordings are retained in the shared workspace and excluded
from Git; their SHA-256 hashes are recorded in the reports. A new capture may
observe a different valid atomic queue ordering, so compare its own paired
replay instead of expecting the archive hash to repeat across captures.

## Earlier certificates

The older 450 nm specialization has a complete shared-tape certificate for
8,192 photons, 136,280 interactions and 2,044,200 raw state words. It covers bulk
absorption/scattering, default surfaces, diffuse/specular reflection, and
dielectric reflection/transmission. The report pins the relevant sources and
compiler policy. See the [detailed certificate documentation](../chroma-lite/chroma/triton/README.md#random-stream-alignment-and-current-exact-certificates).

That certificate does not cover the newer general spectral engine, WLS, arbitrary
detectors, native XORWOW stream assignment, or the added source/readout stages.
The separate 749,196-photon regression compares the earlier and current Triton
implementations. Neither is a full original-installation parity certificate.

The native spectral evidence above extends that earlier scope. Unsupported
features and untested inputs remain outside this compatibility certificate.
