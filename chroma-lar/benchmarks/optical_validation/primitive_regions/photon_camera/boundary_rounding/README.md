# Rounded bulk collisions at room boundaries

The original native WebGPU run lost eight of 30,000,000 Rayleigh photons
(seed 901) after a bulk collision rounded exactly onto a monitor wall. The next
outward triangle query rejected the zero-distance intersection. The retained
`checkpoints-before.json` and `cpu-reference.json` record those stable IDs.

A boundary-first policy was tested and rejected: it matched terminal flags but
removed a CPU-resolved scatter for two paths. Its failed paired comparison is
preserved as `comparison-boundary-first.json`.

The final policy preserves every already-selected bulk collision and its
random stream. A rounded boundary origin is moved conservatively inside its
incident medium. This avoids numerical escape without replacing the sampled
scatter with a wall hit. Near-tie f32 and CPU-f64 event ordering can still
differ; this is not a proof of bitwise transport equality.

`300m/capture.json` records ten complete 30M events, seeds 901–910, on the native
NVIDIA/ampere adapter: 300M detected, no escapes, step limits or nonfinite
states. Forty rounded origins were corrected. The seed 901 wall-energy sum
agrees with the independently reconstructed weighted source to 1.95e−7 relative.
Queue times were collected during correctness work and are not sustained
throughput measurements.
