# Browser optical spectral lab

The browser lab runs actual optical Monte Carlo for three small synthetic scenes:

- **Dispersive prism:** polarized Fresnel reflection/transmission, wavelength-dependent refractive index and group velocity.
- **Fluorescent surface:** ultraviolet absorption, 85% reemission probability, a tabulated visible emission spectrum, a tabulated emission delay, and random hemispherical emission.
- **Polarized Rayleigh:** a wavelength-dependent scattering length proportional to wavelength to the fourth power, with direction and outgoing polarization sampled together.

The geometry is a 600 × 400 × 240 mm ideal detecting monitor around each demonstration. These are the same fixtures and compiled tables as the native [optical showcase](../notebooks/optical_showcase.ipynb). WGSL implements their transport directly; Python is used only to export scene tables. There is no server simulation or prerecorded event behind the browser page.

## Open the lab

Run [webgpu_detector_viewer.ipynb](../notebooks/webgpu_detector_viewer.ipynb) with the **TriChroma (GPU, pimm-bench)** kernel, then download the generated ZIP. Unzip it on your computer and run:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory webgpu_detectors
```

Open **http://localhost:8765/physics.html** in a current WebGPU-capable browser. The adjacent `index.html` page remains the detector geometry explorer. If serving from a remote host, first forward its port using `ssh -L 8765:localhost:8765 HOST`; `localhost` in your browser refers to your computer. HTTPS serving is also supported. `file://` cannot fetch the assets.

The CPU exporter can also build the bundle directly:

```bash
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/export_webgpu_viewer.py \
  --output /tmp/trichroma-browser
```

For a lab-only bundle, call `export_physics_catalog(destination)` from `chroma_lar.webgpu_physics` and copy `physics.html`, `physics.js`, and `physics.wgsl` from `chroma.triton.webgpu`'s assets directory.

## Controls and observables

Select the experiment, an optical photon count from **1 to 30 million**, a seed, and random/y/z linear polarization. The event computes all selected photons. Source and detected spectra, monitor arrival times, and each scene-specific observable use the **full population**. The default event is 2.5 million photons; browser hardware determines how long it takes.

Only the first requested **0–2,048 photon trajectories** are recorded for display; the UI offers 128–2,048. These paths come from the same WGSL invocations as the full event. Display colors track wavelength, with UV mapped to violet as a false color. The x–y projection and additive path brightness aid inspection; they do not represent radiance or a physical camera response.

The time gate reveals completed flights whose endpoint time is within the gate. It redraws cached trajectories without rerunning the event. A fluorescent waiting time is retained at the reemitting surface; the plot does not interpret that waiting time as a slower flight through space. The spectrum/timing charts remain full-event distributions while the path gate changes.

The page reports GPU queue completion time separately from buffer readback and JavaScript decoding. The first dispatch may include driver warm-up. Optical photons/s divides the full event count by the queue time; it is neither detector-camera rays/s nor browser painting FPS. Software fallback adapters are explicitly labeled.

## API and boundaries

```javascript
await window.opticalLabReady;
const result = await opticalLab.run({
  scene: 'fluorescence', photons: 2500000, seed: 901,
  polarization: 'random', paths: 512, maxSteps: 256
});
opticalLab.setTimeGate(20); // ns; no simulation rerun
```

The return value includes event counts, full-count histograms, timing, and sampled vertices. `debug: true` additionally downloads every terminal state for batches of at most 65,536 photons. Large events allocate only bounded path/statistics buffers. Dispatches use a second workgroup dimension above the one-dimensional WebGPU dispatch limit.

This is a bounded prototype for these three exported scenes, **not a general Chroma transport backend**. It supports no detector event loading, bulk reemission, PMT electronics, TTS or digitization. Analytic detector wires remain unsupported in the browser geometry page; use the native Triton viewer. Invalid inputs, unsupported formats/surface models, step-limit survivors, and non-finite results fail the event instead of displaying a partial success.

WGSL arithmetic is float32. Philox uses exact 32-bit integer multiplication/carry logic and matches native random-stream goldens, including 64-bit photon IDs and seeds in the probe. Boundary intersection and Fresnel/Rayleigh math can differ from the CPU float64 intersection reference. Finite validation does not establish correctness for arbitrary geometries, grazing cases or all seeds.

## Reproduce validation

The checker exports current scene tables, retains the matching CPU oracle objects, launches actual Chromium WebGPU, and tests random numbers, terminal states, full-count histogram conservation, and recorded paths:

```bash
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/check_webgpu_physics.py \
  chroma-lite/chroma/triton/webgpu/assets --output /tmp/webgpu-physics-check \
  --photons 4096 --seed 901 2027
```

It requires Playwright and Chromium and forces SwiftShader software by default. Use `--hardware` only when the local GPU is available. Native hardware and software results must be kept separate. See the generated report for exact source hashes, sample sizes, observed numerical discrepancies, and tolerances; no universal bitwise equivalence is asserted.
