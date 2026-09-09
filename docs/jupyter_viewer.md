# Jupyter detector viewer

`chroma.triton.viewer.DetectorViewer` traces fresh camera rays against detector
triangles on the local CUDA GPU. The default is **2,500,000 rays per frame**,
resolved to a 1000 × 625 RGB image. These are geometry inspection rays: the
viewer does not generate or propagate an optical event, apply QE, or simulate
readout. The optical simulation benchmarks measure a different workload. For a
3D camera illuminated by simulated optical photons, open
[`notebooks/photon_camera.ipynb`](../notebooks/photon_camera.ipynb); the
[optical diagnostics](../notebooks/optical_showcase.ipynb) show spectra and timing.

Open [`notebooks/detector_viewer.ipynb`](../notebooks/detector_viewer.ipynb) in a
Jupyter kernel with PyTorch, Triton, NumPy, Pillow and ipywidgets. On this host,
select **TriChroma (GPU, pimm-bench)**.
The tested notebook dependencies are pinned in
[`requirements-viewer.txt`](../requirements-viewer.txt). Keep both
`chroma-lite` and `chroma-lar` on `PYTHONPATH` for shared mesh instancing:

```bash
PYTHONPATH=chroma-lite:chroma-lar jupyter lab notebooks/detector_viewer.ipynb
```

```python
from chroma.triton.examples.theia import build_theia
from chroma.triton.viewer import DetectorViewer

fixture = build_theia(size=25500., coverage=.81, diameter=508.)
viewer = DetectorViewer(fixture.detector(), rays=2_500_000)
controls = viewer.show()
```

The notebook has a detector dropdown and **Load detector** button for Theia,
reflect3wires, pixelTPC, and a resolved pixel-pad closeup.

The main Theia fixture has 49,684 20-inch PMTs; use `size=5000., coverage=.05,
diameter=304.8` for the smaller 326-sensor smoke fixture.

Orbit, elevation and distance update a 100,000-ray preview; after motion
settles, the running notebook event loop requests a full frame. Change the
**Camera rays** integer field to set its exact budget (default 2.5 million).
A budget below 625,000 also reduces the image size; every pixel gets a sample.
The full-render button also works in environments without that event loop.
The initial frame includes JIT compilation and may take longer. Changing
camera parameters traces new rays; it does not rotate a saved image.

The default camera sits inside the detector. Triangle colors are rendered as
opaque surfaces with a headlight, including glass and enclosure triangles;
alpha blending, refraction, optical emission and shadows are not simulated.
For an exterior cutaway, construct with `hidden_solids=(0,)` to omit the first
solid (the enclosure in the supplied Theia fixture). Ordinary Chroma Geometry
objects keep frequently repeated mesh instances (at least four by default); a flattened Geometry or Mesh also
works. The caller's geometry is not mutated or flattened by the viewer.
`solid_colors={solid_index: 0xAARRGGBB}` overrides display colors without changing
the detector; the notebook uses blue walls and gold sensors for contrast.

For a headless image or a custom camera:

```python
from chroma.triton.viewer import Camera

frame = viewer.render_frame(Camera(eye=(0., -12000., 3000.), target=(0., 0., 0.)))
display(frame)  # RGB image is also available as frame.image
print(frame.rays, frame.seconds, frame.rays_per_second)
with open('detector.png', 'wb') as output:
    output.write(frame.png())
```

`frame.seconds` includes ray generation, exact triangle traversal, shading,
anti-alias sample reduction, overflow checking, and RGB download. It excludes
PNG encoding, notebook communication, and browser painting. The widget
reports PNG time separately; network speed can limit notebook responsiveness.
The first frame also includes allocation and JIT, so warm the viewer before
measuring sustained speed. `viewer.close()` cancels pending refreshes and
closes its widgets; drop the viewer object to release its GPU buffers.

Run the reproducible geometry-only benchmark with:

```bash
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/benchmark_viewer.py \
  --radius 25500 --coverage .81 --diameter 508 \
  --rays 2500000 --frames 5 --output /tmp/viewer-benchmark
```

The JSON records each frame, GPU/software versions, sensor count, and hashes
of the rendering/traversal source. PNG encoding is measured separately.

The kernel/widget smoke check also verifies the preview and delayed full-frame
callbacks (it does not open a browser):

```bash
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/check_viewer_notebook.py \
  --output /tmp/viewer-notebook-smoke.json
```

## Measured local A100 performance

Five warm frames with different camera angles/seeds, on an A100-SXM4-40GB,
during an exclusive GPU window with the current source:

| Geometry | Rays/frame | Render + RGB download | PNG encoding |
| --- | ---: | ---: | ---: |
| 49,684 twenty-inch PMTs, 25.5 m radius | 2,500,000 | 169.91 ms | 67.53 ms |
| Same detector, preview | 100,000 | 10.70 ms | 10.18 ms |
| reflect3wires, six analytic wire planes | 2,500,000 | 8.54 ms | 77.44 ms |
| Same wire detector, preview | 100,000 | 1.17 ms | 7.84 ms |
| pixelTPC, original averaged pixel faces | 2,500,000 | 4.80 ms | 26.88 ms |
| Same pixel detector, preview | 100,000 | 0.99 ms | 4.95 ms |

The full-frame camera-ray rates are 14.71M/s, 292.83M/s, and 521.06M/s,
respectively. These are geometry-camera rays, not optical-event photons.
The main detector produces about 4.2 encoded frames/s before notebook transfer
and browser painting; actual browser frame rate was not measured. Initial
construction took 9.92 s, followed by 1.20 s of warmup with cached Triton kernels.
Cold compilation may take longer.

The main detector retains 49,684 transforms of one 660-triangle sensor mesh
(32.79 million equivalent sensor triangles). Current reports and images are in
[`viewer_final/theia`](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer_final/theia/timings.json),
[`viewer_final/reflect3wires`](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer_final/reflect3wires/timings.json), and
[`viewer_final/pixelTPC`](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer_final/pixelTPC/timings.json).
The current [notebook check](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer_final/notebook.json)
exercised the registered GPU kernel, exact ray-count control, scheduled camera
refreshes, and every detector selection. It did not measure browser painting.
Earlier independent small-fixture measurements are retained in
[`viewer/timings.json`](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer/timings.json).


## reflect3wires and pixelTPC

The supplied examples load the existing repository configurations directly:

```python
from chroma_lar.viewer_examples import build_viewer_example

example = build_viewer_example("reflect3wires")  # or "pixelTPC", "pixelPads"
print(example.metadata)
viewer = example.viewer(rays=2_500_000)
controls = viewer.show()
```

The main reflect3wires view contains all six analytic periodic-cylinder wire
planes (10,750 wires), 162 PMTs, active enclosure, cavity, and cathode.
Its `AnalyticWireLayer` reuses the same FP64 intersection kernel as optical
transport. All wire frames, radii, pitches and extents are checked against the
supplied geometry before rendering. Original box/PMT triangles remain in the
mesh query, and the nearest wire is merged into that result.

Direct `DetectorViewer` calls reject unhandled analytic-wire metadata rather
than silently omitting wires. The example attaches the validated adapter.
For a mesh comparison, select `build_viewer_example("reflect3wires-mesh")`:
this uses the original 32-facet cylinder builder and 1,376,000 wire triangles.
Long thin triangles make that generic mesh traversal much slower; the main
interactive view uses analytic intersections. Reduce Distance to resolve
individual wires; their 0.15 mm diameters are smaller than a pixel in an overview,
so a four-sample full frame shows sampling noise at the wire planes.

pixelTPC uses the original configuration's **`pixel_simplified=True`** setting.
It has 164 PMTs, plus two surfaces representing the area-averaged response of
one million pads per face. These full-detector surfaces do not resolve the
individual gold pads. The separate `pixelPads` selection renders a 32 × 32
patch from the original detailed pixel-face builder, with 20,480 triangles
and the configured gold/FR-4 colors. That closeup is not a full detector.

Benchmark either detector with:

```bash
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/benchmark_viewer.py \
  --detector reflect3wires --frames 5 --output /tmp/viewer-reflect3wires
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/benchmark_viewer.py \
  --detector pixelTPC --frames 5 --output /tmp/viewer-pixelTPC
```

Add `--all-detectors` to `check_viewer_notebook.py` to execute the notebook's
Load detector callback for the wire detector, pixel detector, and pad closeup.


The final reflect3wires measurements and image are in
[`viewer_final/reflect3wires/timings.json`](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer_final/reflect3wires/timings.json)
and [`reflect3wires.png`](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer_final/reflect3wires/reflect3wires.png).
The one-frame, fully meshed-wire comparison took **9.283 s** for 2.5M rays
(excluding its additional 78.63 ms PNG encoding); its separate
[report](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer_reflect3wires_mesh/timings.json)
and [image](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer_reflect3wires_mesh/reflect3wires-mesh.png)
retain the 32-facet representation explicitly.

The pixel detector's [report](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer_final/pixelTPC/timings.json)
and [image](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer_final/pixelTPC/pixelTPC.png)
are separate from the resolved pad patch's
[report](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer_pixelPads/timings.json)
and [image](../chroma-lar/benchmarks/optical_validation/primitive_regions/viewer_pixelPads/pixelPads.png).
