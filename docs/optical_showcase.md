# Play with wavelength-dependent light

Open [`notebooks/optical_showcase.ipynb`](../notebooks/optical_showcase.ipynb) and
select **TriChroma (GPU, pimm-bench)** (`trichroma-gpu`). Run its cells, choose an
experiment and press **Trace the light**. The notebook uses the local GPU.

- **Split white light:** a triangular dispersive dielectric separates a
  broadband beam. Reflected branches come from polarized Fresnel sampling.
- **Turn ultraviolet into visible light:** a fluorescent coating absorbs UV,
  reemits with probability 0.85, draws a new visible wavelength and a tabulated
  delay, and emits into either hemisphere. Its afterglow comes from those actual
  delays. This is the supported effective surface model, not bulk fluorescence.
- **Scatter blue, transmit red:** a chamber uses polarized Rayleigh scattering
  with a scattering length proportional to wavelength to the fourth power.

The **Optical photons** control sets the full simulation count, initially
2,500,000 and adjustable through 30,000,000. **Drawn paths** selects up to 2,048
of those same photons, retaining their original IDs and random streams. The
histograms use every simulated photon. Choose random linear polarization or a
fixed y/z polarization to explore its effect on scattering and reflection.

**Time** and **Show all times** only redraw the saved trajectories. The time
control reveals segments once the interaction at their endpoint has completed;
it does not interpolate fluorescent waiting time into an invented speed.
Its range adjusts to the recorded paths in each experiment.
Changing the experiment, count, source seed, polarization or path count takes
effect on the next **Trace the light** click.

The image is a projection of physical photon trajectories into x/y. Color
encodes wavelength; UV is assigned violet so it can be inspected. The line glow
is a display effect. It is not a camera-radiance calculation, a brightness
prediction or human colorimetry. Material tables are explicitly synthetic and
live in `build_playground_scene`. A virtual enclosing photon monitor records
arrival position, time and wavelength, without PMT response or electronics.
Inactive bulk processes use infinite interaction lengths, so the fluorescent
delay can be obtained by subtracting its actual straight-flight time without
an unmodeled rare scattering contribution.

The code runs the existing general `SpectralSimulation`. To recover trajectory
vertices without modifying kernels, it repeats increasingly long prefixes on
the small subset. Every prefix restarts the original input, IDs and seed. The
final subset state must exactly match the corresponding full event state, or
the display raises an error. Prefix recording is separately timed, alongside
source generation and full transport. Setup, plotting and notebook transfer are
excluded from the optical throughput shown. This generic spectral workload is
separate from the detector geometry renderer's camera-ray rate.

Use it without Jupyter:

```python
from chroma_lar.optical_showcase import OpticalShowcase

lab = OpticalShowcase()
event = lab.run("fluorescence", photons=2_500_000, paths=512, seed=901)
lab.save_figure("fluorescence.png")
```

The notebook also includes a three-experiment gallery and example exports.
Optional dependencies are the existing viewer environment plus Matplotlib
(locally tested with 3.10.9). CPU validation can use
`OpticalShowcase(backend="reference_bvh", device=None)` with small counts.
