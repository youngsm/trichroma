# Optical playground validation

The new `notebooks/optical_showcase.ipynb` runs actual spectral transport in three
small synthetic experiments. It uses the registered **TriChroma (GPU,
pimm-bench)** kernel. No Modal or browser compute is used for this notebook.

All three experiments were run with **2,500,000 photons** on the local A100.
Each image shows **512 actual trajectories**, with spectra and arrival-time
histograms derived from the complete event. The two counts are printed in each
image and exposed separately in the notebook.

| Experiment | Monitor arrivals | Additional observable | Source / transport / path recording |
|---|---:|---|---|
| Dispersive prism | 2,500,000 | 259,918 photons reflected at least once | 1.194 / 1.054 / 0.018 s |
| Fluorescent coating | 2,124,171 | 84.97% reemitted; delay median 6.612 ns, 90th percentile 49.987 ns | 1.142 / 0.599 / 0.008 s |
| Rayleigh chamber | 2,500,000 | 2,204,851 photons scattered at least once | 0.999 / 1.058 / 0.602 s |

This gallery was refreshed after the final general-transport boundary and
FP64-intersection fixes. The recorded source hashes include both BVH query
implementations. These are individual illustrative runs, not sustained performance
benchmarks; an independent CPU-only browser validation ran concurrently.
The general optical path includes CPU source generation and full terminal-state
download. Its rate is separate from both the camera-ray renderer and the
specialized full-detector pipeline. Setup, plotting and notebook transfer are
excluded from the three timings. No photons reached the step limit in the
7,500,000-photon gallery.

Validation passed:

- **16 tests**, covering the native showcase and browser scene export/oracle support,
  including CPU/GPU comparisons of 1,024 photons per experiment.
  Final flags, channels and photon IDs agree exactly; time and wavelength
  comparisons use the stated numerical tolerances in the tests.
- The 512 trajectory endpoints per large experiment match the corresponding
  full event's position, wavelength, time and history flags exactly.
- Stable source generation retains the same photons when the count changes.
  Display colors change at the wavelength-shifting vertex, not along the
  incoming ultraviolet segment.
- `nbclient` executed the actual notebook's default 2.5M-photon event and all
  three experiment callbacks. Photon-count and path-count changes updated the
  simulation and image. Time gates redrew the existing event without rerunning
  physics. Browser painting itself was not tested.
- All three exported PNGs were visually inspected. The committed notebook is
  clean, without stored execution output.

See [report.json](report.json) for exact timings, scene fingerprints and source
hashes; [tests.txt](tests.txt) records the test result. Gallery:
[prism](prism.png), [fluorescence](fluorescence.png), [Rayleigh](rayleigh.png).

Inactive bulk processes use infinite interaction lengths. A dedicated test
checks that the fluorescent delay subtraction has no competing bulk process.
The original finite-length evidence is retained under `initial_finite_lengths`;
all recorded event counts and fluorescence delay quantiles remained unchanged.
The time slider now adjusts to each experiment and labels completion of
interactions, including any fluorescent delay.

The geometry and calibration are synthetic. Fluorescence is an effective
surface process, not bulk reemission. The image is an x/y projection of Monte
Carlo trajectories, with a display palette and glow effect; it is not a
camera-radiance, brightness or colorimetric prediction. UV is explicitly shown
using violet false color.

Reproduce from the repository root with the CUDA Python environment:

```sh
PYTHONPATH=chroma-lite:chroma-lar python -m pytest -q chroma-lar/test/test_optical_showcase.py
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/check_optical_showcase.py --photons 2500000 --paths 512 --notebook --output-dir chroma-lar/benchmarks/optical_validation/primitive_regions/showcase
```
