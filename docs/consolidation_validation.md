# Consolidated checkout verification

The source snapshot was assembled in a clean checkout of `youngsm/trichroma`.
The original Chroma-Lite and Chroma-LAr working trees were preserved.

- **685 copied files** match their source snapshots byte for byte. The
  [manifest](consolidation_manifest.json) records every SHA-256 and the upstream
  base revisions, including the fact that both source working trees had local
  changes.
- **167 regression tests passed**, with zero failures, errors or skips, when
  run from the consolidated checkout on the local A100. See the
  [test log](consolidation_validation/tests.log),
  [JUnit results](consolidation_validation/tests.xml), and
  [command/environment record](consolidation_validation/run.json).
- **749,196 photons across three seeds matched exactly** against the saved
  pre-refactor baseline, including terminal photon states, optical hits,
  photoelectrons, noisy ADC waveforms and metadata. See the
  [comparison log](consolidation_validation/equivalence.log).
- Runtime source hashes match the recorded
  [25.50M photons/s sustained benchmark](../chroma-lar/benchmarks/optical_validation/maintainability/performance_30m.json).
  Consolidation changed file organization at the repository level and added
  root documentation and a test runner; it did not change the optical runtime.
- The saved [GPU memcheck](../chroma-lar/benchmarks/optical_validation/maintainability/memcheck.log)
  reports zero errors across 17 tests for this runtime.

The focused suite covers source sampling, spectral physics, TPB coating,
PMT response, waveform digitization, analytic and mesh geometry, queue/runtime
behavior and the full detector pipeline. Full legacy Chroma feature parity and
physical validation against measured detector data are outside these claims.
The recorded native-CUDA comparison discrepancy and mesh-boundary limitations
remain in the [full validation report](../chroma-lar/benchmarks/optical_validation/FAST_FULL_REPORT.md).

Large generated photon populations and repeated BVH dumps are inventoried by
size and hash in the consolidation manifest rather than embedded in Git.
The optical regression baseline and small comparison fixtures are included.
No primitive/region compiler prototype is included in this baseline snapshot.
