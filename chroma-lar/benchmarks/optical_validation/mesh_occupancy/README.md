# High-occupancy arbitrary-mesh transport: CUDA Chroma vs general Triton engine

Theia-like water detector (49,684 PMTs, 32,791,840 triangles), one Chroma-built
BVH (47,175,746 nodes) shared by both backends, identical persisted Cherenkov
photons (1 m track, 300-650 nm), A100-SXM4-40GB, `max_steps=256`, 2026-09-25.
Transport time runs from resident device photons to device synchronization.
Every seed is warmed once before its timed run. Harness:
`chroma-lar/benchmarks/benchmark_mesh_occupancy.py`.

| Backend | 15M photons: runs (s) | median rate | 25M photons: runs (s) | median rate |
|---|---|---:|---|---:|
| Triton general flattened-mesh engine (FP64 traversal) | 1.153, 1.154, 1.159 | 13.0M/s | 1.764, 1.768, 1.775 | 14.1M/s |
| CUDA Chroma, installed defaults (512x1024) | 1.146, 1.188, **7.465** | 12.6M/s | 1.900, 1.911, **8.506** | 13.1M/s |
| CUDA Chroma, `use_packed=True` | 0.912, 0.940, 0.950 | 16.0M/s | 1.524, 1.542, **17.61** | 16.2M/s |

Outcome fractions agree between backends (detected 13.6%, bulk absorbed 62.6%,
surface absorbed 23.8%, Rayleigh scattered 30.1%). CUDA leaves 2-4e-7 of the
photons unfinished at 256 steps; Triton finishes every photon within 48 steps.
CUDA runs show recurring multi-second outliers (also in warmups: 2.3, 5.2 and
5.5 s). The packed path's download time is not comparable: W's packed `get()`
does not unpack the final states.

At high occupancy the general Triton mesh engine matches the installed CUDA
defaults and trails the packed kernel by about 15%. Earlier comparisons at
0.1-1M photons were dominated by the Triton engine's per-step host overhead.
