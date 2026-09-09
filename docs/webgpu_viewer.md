# Browser WebGPU detector viewer

The prototype traces camera rays on the browser's own GPU and presents them
directly to a canvas. After a scene is exported, neither a Python render server
nor CUDA is needed. Theia and pixelTPC preserve shared PMT meshes, rigid instance
transforms, and the existing host-built BVHs. The browser uses threaded escape
links for traversal and retains every exported mesh triangle.

The detector page is **opaque geometry rendering**. The same bundle includes a
[photon camera](photon_camera.md) at `camera.html`, which forms a 3D image from
full optical photon maps, and [optical diagnostics](webgpu_physics.md) at
`physics.html` for detailed spectra, timing and sampled trajectories. PixelTPC
keeps the original configuration's area-averaged pixel faces. The separate
`pixelPads` scene shows a resolved 32 × 32 pad patch. Analytic `reflect3wires`
export is explicitly rejected: its validated renderer uses FP64 cylinder
queries, while [WGSL provides f32/f16 floating-point types](https://www.w3.org/TR/WGSL/#floating-point-types).
Use the [Triton notebook](../notebooks/detector_viewer.ipynb) for all six wire
planes. No wire geometry is silently dropped.

## Open it

From the checkout, using the same Python environment as the detector examples:

```bash
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/export_webgpu_viewer.py \
  --output /tmp/trichroma-browser --detector theia --detector pixelTPC --detector pixelPads
python -m http.server 8765 --bind 127.0.0.1 --directory /tmp/trichroma-browser
```

Open **http://localhost:8765** in a browser with WebGPU support. If Python runs on
a remote host, either forward its port with `ssh -L 8765:localhost:8765 HOST`, or
copy the exported directory to your computer and serve it there. The browser's
GPU does the rendering in either case. `file://` cannot load this bundle; WebGPU
also requires a secure context such as HTTPS or localhost.

[`notebooks/webgpu_detector_viewer.ipynb`](../notebooks/webgpu_detector_viewer.ipynb)
exports the same bundle and provides a ZIP download. The existing Triton notebook
continues to render on the server GPU and includes the analytic-wire detector.

The scene dropdown selects the detector. Drag to orbit, wheel to zoom, and
Shift-drag to move the target. The default full frame traces **2,500,000 camera
rays**, averaging four samples into each pixel of a 1000 × 625 image. The ray
budget control also offers 625,000 and 10 million rays. Motion initially uses
100,000 rays; subsequent preview budgets adapt toward roughly 30 ms of queue
work on the selected adapter. A full frame follows after motion stops.

The details panel reports the actual detector dimensions/count, exported scene
SHA-256, and browser adapter. The corrected main fixture has 49,684 twenty-inch
PMTs, a 25.5 m radius and 0.81 requested coverage; its shared geometry is about
5.64 MB. The pixelTPC export is about 0.36 MB. Runtime labels always come from
the export metadata.

## Precision and validation

WGSL computes triangle queries in float32. This is an inspection renderer and
does not promise bitwise equality with CUDA optical transport. Exported bounds
are conservative; traversal guards fail the frame if a tree is invalid. Scene
bytes are checked against their SHA-256 before upload. Glass and enclosures are
opaque; refraction, photon histories, QE, timing and readout are absent.

The browser checker compares actual WGSL rays with an independent float64
Möller–Trumbore oracle on the exported triangles and transforms. It tests camera
direction rounding separately, checks exact group/instance/triangle identities,
distances and normals, and captures the actual browser canvas. It uses small
ray sets for the numerical oracle; the frame benchmark uses a separate chosen
budget. These finite checks do not establish correctness for every possible
camera or boundary case, particularly grazing rays and subpixel geometry.

```bash
# Optional browser-test dependency; normal viewing does not require Playwright.
python -m pip install playwright==1.60.0
python -m playwright install chromium
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/check_webgpu_viewer.py \
  /tmp/trichroma-browser --output /tmp/browser-check
```

By default the checker forces Chromium's **SwiftShader software adapter** and
records it explicitly. Add `--hardware --full-frame` to require a hardware
adapter and measure 2.5 million rays per frame; coordinate this with other GPU
benchmarks on the same host. `--browser /path/to/chromium` selects an executable.
The supplied Linux headless flags keep Vulkan rendering and canvas compositing
on the same backend. A failed hardware adapter request is an unsupported test
environment, not a measurement of hardware performance.

Reported queue time includes command encoding, traversal, shading and the
canvas render pass through GPU queue completion. It excludes initial shader
compilation, scene upload, test-only buffer readback, and browser painting. It
must not be reported as measured interactive browser FPS or optical photons/s.

The native NVIDIA/ampere adapter on the local A100-SXM4-40GB produced these
three-frame warm averages from each default camera, varying jitter seeds, in
Chromium 148.0.7778.96:

| Scene | Camera rays/frame | Mean queue time | Camera rays/s |
| --- | ---: | ---: | ---: |
| Theia, 49,684 twenty-inch PMTs | 2,500,000 | 112.53 ms | 22.22 million |
| PixelTPC, original averaged pixel faces | 2,500,000 | 5.30 ms | 471.70 million |
| Resolved 32 × 32 pad patch | 2,500,000 | 4.23 ms | 590.55 million |

All 2,560 tested center-ray hit identities matched the float64 oracle in each
scene. Maximum distance differences were 0.03862 mm, 0.001074 mm and 0.00000762 mm,
respectively. The query preserves low bits when subtracting world translations
and advances the ray near an instance before applying its rotation; this reduces
cancellation when centimetre-scale PMT triangles are tens of metres from the
camera. These values cover the recorded cameras, not arbitrary grazing rays.

The [native hardware report](../chroma-lar/benchmarks/optical_validation/primitive_regions/webgpu_hardware/verification.json)
includes individual timings and source/asset hashes, with actual browser
screenshots and oracle arrays in the same directory. The separate
[software report](../chroma-lar/benchmarks/optical_validation/primitive_regions/webgpu_software/verification.json)
uses SwiftShader; its timings are not hardware performance results. The
[control smoke report](../chroma-lar/benchmarks/optical_validation/primitive_regions/webgpu_software/controls.json)
checks real browser selection, drag orbit, wheel zoom, ray-budget changes, reset
and adaptive previews on that software adapter.

## Implementation boundaries

The exporter is in `chroma.triton.webgpu.export`; `export_example()` accepts the
same `ViewerExample` geometry used by the Triton notebook. Shared rigid meshes
use one BLAS plus an instance TLAS. Unique meshes and non-rigid transforms are
baked into a static mesh. WGSL, JavaScript, and HTML live in its `assets/`
directory. `window.trichroma.render({width, height, rays, camera, debug})` exposes
a headless-friendly render API; `window.trichromaReady` resolves after adapter,
pipeline, and initial scene loading. Diagnostic readback is only enabled when
`debug: true`, apart from a four-byte traversal-guard counter read after each
frame. Ordinary frames present directly from a GPU texture without image readback.
