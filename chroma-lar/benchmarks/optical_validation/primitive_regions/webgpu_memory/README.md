# Browser GPU memory and recovery

The reported failure was `vkAllocateMemory failed with
VK_ERROR_OUT_OF_DEVICE_MEMORY`. No browser/OS/GPU model was available; the user
specified that an 8 GiB device should be sufficient. The local test device is an
NVIDIA A100-SXM4-40GB, not the colleague's device.

The published `01fe3b5` renderer peaked at 258 MiB of total device memory during
two cycles through prism and PMT scenes, 2,500,000 photons per event, and two
camera samples each at 320, 640 and 960 pixels. The revised renderer using
reduced maps peaked at 148 MiB under the same workload. `nvidia-smi` sampled
every 100 ms. Application-owned buffers/textures peaked at 141.2 MiB before
and 33.9 MiB in reduced mode; tracked resources returned to zero after release.
These sampled peaks include local driver overhead but cannot bound another
driver's internal allocations. No accumulation across repeated runs was found.

Full detail remains the hardware default. PMT fluorescence storage now uses
exactly its 624 facet charts, reducing map storage from 127.5 to 115.8 MiB
without changing map entries. Reduced detail uses 17.9 MiB for PMT maps and
20.6 MiB for the other scenes. All 64 wavelength bins and all requested photons
remain. Spatial grids become coarser, so rendered images can differ.

`recovery.json` covers synchronous allocation failure under a 64 MiB buffer
budget, an asynchronous scoped `GPUOutOfMemoryError`, native device destruction
with the reported Vulkan OOM reason, camera-pose preservation, and cleanup when
the reduced-memory retry also fails. The retry recreates the device once. All
tests ended with zero leaked tracked buffers. These are deliberate fault
injections, not a claim to reproduce the colleague's actual driver failure.

Reproduce with:

```sh
python chroma-lar/benchmarks/check_webgpu_memory.py BUNDLE --output recovery.json
```

`parity.json` compares 327,720 complete raw GPU terminal records against the
published implementation, using 81,930 photons per scene across ten seeds.
Both full and reduced maps give identical SHA-256 hashes and outcome counts.
The PMT uses the original ceiling-light input for this comparison. This is a
tenfold sample expansion after restoring the original source expression's FMA
rounding during the PMT-lighting change. These finite comparisons do not establish
universal equivalence with original Chroma transport.

`compact-oracle.json` independently recounts every reduced PMT map from 1,024
CPU photon histories. All cells agree within the existing deposition tolerances.
The focused Python suites passed 31 tests.

`cpu-fallback.json` records both browser cases: ordinary Chromium with
`--disable-gpu` exposed neither hardware nor software WebGPU; an explicitly
enabled SwiftShader test environment executed the actual shaders through the
automatic software-adapter retry. The software UI starts at 10,000 photons,
160-pixel camera width, reduced maps and Low compute use. Availability of a CPU
adapter remains browser-controlled.
