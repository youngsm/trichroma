# Matched transport comparison

The earlier 1.96× end-to-end advantage is not a transport speedup. With identical prepared inputs, the measured median transport advantage at one million photons is 1.30×. At 100,000 photons CUDA is 2.73× faster.

Both backends ran alone on the local A100-SXM4-40GB, with one discarded full-size warm-up and five measured repeats per size. The source is the same 450 nm photon population in a 30 mm cube centered at (-1000, 0, 0) mm. Input arrays are persisted once and their SHA-256 hashes are checked on the host and after GPU upload for both backends. Both paths download every photon's position, direction, polarization, time and history: exactly 44 bytes per input photon.

Transport means synchronized wall time from resident state to completed propagation, including queue allocation and host scheduling. Prepared-event time adds allocation/upload and the matching output download. Source generation, input-file reads, setup and input-hash auditing are excluded. No 30-million-photon run was performed.

| Photons | CUDA transport | Triton transport | CUDA throughput | Triton throughput | Triton speedup |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.0511 s | 0.1398 s | 1.956 M photons/s | 0.715 M photons/s | 0.37× |
| 1,000,000 | 0.2497 s | 0.1914 s | 4.004 M photons/s | 5.224 M photons/s | 1.30× |

| Photons | CUDA prepared event | Triton prepared event | Triton speedup |
|---:|---:|---:|---:|
| 100,000 | 0.0536 s | 0.1419 s | 0.38× |
| 1,000,000 | 0.2781 s | 0.2018 s | 1.38× |

At one million photons, the fastest observed resident-transport rates were 4.059 M photons/s for CUDA and 5.510 M photons/s for Triton. These are maxima over five repeats at this size, not established hardware peak rates.

All measured repeats are retained, including the 1.533-second CUDA outlier at one million photons. CUDA uses its persistent XORWOW states and nondeterministic work queue; Triton uses its existing Philox streams. Identical inputs therefore do not imply identical random trajectories.

## Validity and remaining differences

An initial CUDA series stopped at one million photons, seed 1910, after finding one aborted/escaped photon and three nonfinite vector components. The harness was changed to preserve invalid states and continue recording rather than stop before writing the record. The complete CUDA series reported above was then run from fresh setup. All completed-series runs were finite and terminated, but the initial failure remains recorded in `cuda_initial_failure.json` and the initial partial timings remain in `cuda_initial_partial.json`. This is not a correctness sign-off.

CUDA uses the repository's existing full detector with analytic wires. Triton uses its existing instance and analytic-wire acceleration, including the reachability specialization for a source on the negative-x side. The Triton harness calls the existing resident propagation loop and retains all final states; it does not compact surviving photons into the optional reservoir. No transport physics code was changed. Geometry acceleration, numerical edge behavior, and RNG algorithms still differ. Both benchmarks are 450 nm transport, without the new VUV/WLS/TTS/waveform pipeline.

Reproduce with `benchmark_prepared_transport.py --backend prepare`, followed by separate `--backend triton` and `--backend cuda` processes in their respective local environments. Raw complete timings are in `triton.json` and `cuda.json`; input hashes are in `inputs.json`.
