# Spectral optical simulation

The optical pipeline connects arbitrary optical photons or energy depositions to
spectral transport, effective TPB coatings, PMT photoelectrons, and digitized
waveforms. `FastOpticalSimulation` combines these physics models with the
detector's analytic wires, boxes, and shared PMT mesh. `SpectralSimulation`
retains the general triangle-mesh transport implementation.
The geometry builders accept `pmt_coating_surface` on the outer PMT window.

See [architecture and extension points](optical_architecture.md) for module
responsibilities, supported extensions, and the work needed for a standalone
Chroma-compatible package.

The earlier full comparison found an unresolved numerical issue in meshed-wire boundary
handling: a lossless reflective-wire test spuriously absorbs some photons in
both the original CUDA and spectral Triton mesh paths. The accelerated detector
uses analytic wire cylinders. See the
[accelerated full-pipeline report](../benchmarks/optical_validation/FAST_FULL_REPORT.md)
for its separate checks and sustained throughput, and the earlier
[correctness and local GPU performance report](../benchmarks/optical_validation/REPORT.md)
for mesh-path failures and model differences. The supplied calibration is
synthetic; software validation does not establish measured detector accuracy.

## Accelerated detector pipeline

From the workspace root, using a local CUDA GPU:

```bash
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/run_optical_simulation.py \
  --backend triton-fast \
  --calibration chroma-lar/benchmarks/optical_validation/full_detector_synthetic_calibration.json \
  --detector-config detector_config_reflect_reflect3wires \
  --input depositions.npz --max-steps 2048 --output optical_events.npz
```

The input NPZ contains `pos` (mm), `energy_mev`, and optionally `t` (ns) and
`evidx` (event IDs). Zero-light deposition events retain their waveforms.
The same backend also accepts prepared photon NPZ inputs described below.

```python
from chroma_lar.optical_calibration import OpticalCalibration
from chroma_lar.optical_simulation import FastOpticalSimulation

calibration = OpticalCalibration.load("my_calibration.json")
simulation = FastOpticalSimulation(calibration)
result = simulation.simulate_depositions(
    [[-1000, 0, 0], [-1000, 0, 0]], [1500., 0.],
    event_indices=[11, 99], seed=901, max_steps=2048,
)
result.save("optical_events.npz")
```

Yield comes from the calibration: 1500 MeV corresponds to a mean 30M photons
only for the bundled 20,000 photons/MeV fixture. `simulate_voxel(count, ...)`
instead generates a specified count in a cube. `simulate_photons(photons, ...)`
propagates prepared live photons. Optical hits, PE hits, and waveforms return
as CPU arrays. `keep_final_states=True` additionally copies every terminal
photon's state to the CPU for debugging; this is excluded from throughput
measurements. NPZ compression and writes are also excluded.

The accelerated source adapter currently accepts positions in the negative-x
component of the reflect3wires detector. Source points outside it are rejected.
The scene retains **all 162 PMTs, all six wire planes, and every box face**,
including cavity paths reached through coincident PMT/wall boundaries. The
current specialization requires opaque default-model cathode and enclosure
surfaces over the entire wavelength grid. Coating and glass boundaries retain
the original PMT triangles.

The GPU executes multiple bulk interactions inside a certified empty LAr
region, querying exact detector geometry when a sampled free path could reach
a boundary. A handoff neither moves a photon nor consumes a new random draw.
Wavelength-dependent absorption/scattering, group velocity, Rayleigh scattering,
Fresnel interactions, and TPB spectrum/delay/escape all remain active. PMT
collection, TTS and charge use the validated CPU response; GPU waveform
superposition preserves fractional PE arrival times. Electronics noise and
ADC processing use the same model as the general pipeline.

Analytic wire and box boundary points are reconstructed in FP64 and rounded
toward the outgoing material. Ordinary float32 flight arithmetic can otherwise
place a reflected photon inside a wire at metre-scale detector coordinates.
This correction changes the representable boundary position; interaction
probabilities and the distance/time of the completed flight are unchanged.

The source population is resident on the GPU; this API currently does not
automatically tile oversized requests. Thirty million photons fit the tested
A100 40 GB configuration. `max_photons` limits Poisson source allocation, and
the source ID/population bounds are checked. Truncation, nonfinite state, and
traversal overflow raise errors.

Reproduce sustained throughput with actual Poisson photon counts:

```bash
PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/benchmark_fast_full.py \
  --calibration chroma-lar/benchmarks/optical_validation/full_detector_synthetic_calibration_noise.json \
  --depositions --counts 30000000 --repeats 5 --output full_throughput.json
```

The benchmark warms the full pipeline, then times source generation through
CPU-readable hits, PE and waveform output. It reports every repeat, aggregate
photons divided by aggregate seconds, individual rates, memory use, physics
diagnostics, and source/calibration hashes.
The original recorded run processed 149,998,442 photons in 5.84839 s: **25.65M/s
sustained**, 24.11M/s for the slowest repeat and 26.94M/s for the fastest.
After separating the engine into maintainable components, the same five events
ran at **25.50M/s sustained** (23.91M/s minimum, 26.43M/s maximum), with identical
physics summary counts. See the [refactor validation report](../benchmarks/optical_validation/maintainability/REPORT.md)
for exact per-photon regression comparisons and the current runtime hashes.

## Run the controlled example

From `chroma-lar`, with NumPy, SciPy, Torch, Triton, and a CUDA GPU available:

```bash
PYTHONPATH=../chroma-lite:. python benchmarks/run_optical_simulation.py \
  --demo --backend triton --output /tmp/optical-demo.npz
```

Use `--backend reference` for the slow CPU reference on small geometries.
The default example has a WLS shell inside a detecting box and two events,
including one with zero deposited energy. It is **synthetic software test
data**, not a prediction for the production detector. Its calibration status
and provenance are written into every output file.

## Interfaces and units

Positions/path lengths are mm; wavelength is nm; time is ns; energy is MeV;
photoelectron charge is PE; digitized waveform samples are ADC counts.

```python
from chroma.triton.spectral import SpectralSimulation

simulation = SpectralSimulation(geometry, wavelengths=wavelength_grid_nm,
                                backend="triton", tile_size=65536)
result = simulation.simulate(photons, seed=123, max_steps=1000)
hits = result.hits  # times, channels, photon_ids, event_indices, wavelengths
```

`photons` may be `chroma.event.Photons` or `chroma.triton.runtime.PhotonBatch`.
Input position, direction, polarization, wavelength, time, event index, and
photon identity are preserved. `PhotonBatch` IDs provide reproducibility under
tiling; when using separate `Photons` calls, give disjoint `photon_id_base`
ranges. Directions/polarizations must be normalized and transverse. Sources
must contain live, unweighted photons inside the compiled spectral range.

`ScintillationSource.photons()` samples a supplied wavelength CDF and a mixture
of exponential emission times, optionally convolved with an exponential rise
time. It generates isotropic directions and random transverse polarizations.
`from_depositions()` samples Poisson photon counts from energy, yield, and an
explicit quenching factor. Deposit positions/times and particle transport are
external inputs; no recombination model is selected implicitly. The count
sampler is reproducible for a fixed deposition batch; repartitioning the
deposition list can change sampled counts. The per-photon samplers are keyed
by stable IDs.

The per-photon RNG is Philox4x32-10, with 64-bit photon IDs and seeds and a
separate 32-bit stream coordinate. Transport, source, and response use disjoint
stream ranges. Its CPU integer implementation is checked against Random123
known-answer vectors and its GPU implementation. Uniform draws use 23 bits
strictly inside (0,1); this imposes finite resolution on extremely rare tails.

## Transport model

The spectral path supports bulk absorption, polarized Rayleigh scattering,
polarization-resolved dielectric Fresnel reflection/refraction, default
detect/absorb/diffuse/specular surfaces, and effective WLS surfaces. It uses the
existing packed triangle BVH in the general implementation. That implementation
requires wireplanes tessellated using `analytic_wires=False`; the accelerated
detector uses analytic cylinders. Both reject weighted transport, bulk re-emission components,
complex-film surfaces, angular surfaces, and dichroics are rejected explicitly.
They are not required for the implemented pure-LAr/effective-TPB model.

Optical properties are interpolated at each photon's current wavelength.
Material `group_velocity` tables (mm/ns) control travel time. When omitted,
velocity is derived as `c / (n - wavelength * dn/dwavelength)` on the compiled
grid. Refractive index still controls Fresnel interactions. No causal flight
time is simulated for dead photons or photons that miss all geometry.

Default-surface residual probability invokes the dielectric boundary. Diffuse
reflection is Lambertian; specular reflection updates both direction and
polarization to preserve transversality. This is a physical polarization
choice and is not bitwise parity with the old CUDA specular branch, which
leaves polarization unchanged.

The general mesh engine launches traversal and interaction kernels per
optical step and compacts active photons with Torch. It prioritizes validated
physics over matching the throughput of the detector specialization. Source
tiling bounds device state; the returned full final state still occupies host
memory proportional to the total input count. The accelerated API above uses
the detector's collision-first scheduler and returns compact outputs by default.

`STEP_LIMIT` (bit 30) identifies photons still alive at the requested step
limit. They are distinct from absorption or escape. The full pipeline raises
on truncation unless explicitly allowed.

## TPB coatings

`build_r5912_pmt(..., coating_surface=surface)` applies the coating only to the
outer window above the photocathode. The surrounding material is material 2;
glass is material 1. Both wire and pixel detector builders expose this as
`pmt_coating_surface`.

For a WLS surface (`Surface(model=2)`), `absorb` is the probability of coating
absorption and `reemit` is the conditional probability of producing one
escaping visible photon. `reemission_cdf` describes outgoing wavelength.
`reemission_time_cdf` is a `TabulatedCDF` or two-column time/CDF array; it adds
a nonnegative delay. Plain WLS surfaces without timing metadata are explicitly
instantaneous. Calibration bundles require a timing distribution for WLS.

`reemission_to_material1` is the probability of escape into glass, independent
of incidence side. The direction is isotropic over the selected hemisphere,
with random transverse polarization. The emitted ray starts at the coating
boundary and continues through the appropriate surrounding medium. Visible
light returning to the coating can interact again according to its spectrum.

This is an **effective coating model**: thickness, internal reflection,
self-absorption, and escape losses must be represented in the calibrated
probabilities and spectra. The interface is not applied again to a photon
that has already escaped the effective coating. There is no explicit film
volume, incidence-angle dependence, incident-wavelength-conditioned emission
spectrum, or multi-photon conversion. A conversion probability above one is
invalid; a yield model that produces multiple visible photons would require
photon branching. Probability, spectrum, delay, and escape direction are
factorized in this version.

## PMT and waveform response

`PMTResponse.apply(hits)` applies Gaussian TTS (RMS, not FWHM) or a tabulated
residual time distribution, transit offsets, collection efficiency, and a
single-PE charge distribution. Gaussian width, offset, gain, and collection
efficiency may be scalars or per-channel arrays. Tabulated timing and charge
distributions are shared across channels in this version. The output retains
the original optical arrival times alongside PE times.

Photocathode detection probability is evaluated in optical transport. Do not
apply the same QE or collection efficiency again in the response. Likewise,
if a quoted tube QE already includes window transmission, its interpretation
must be reconciled with explicit glass transport.

`digitize()` evaluates a caller-supplied SPE pulse template at each actual PE
time without rounding arrivals to sample bins. It supports per-channel pulse
superposition, a baseline, Gaussian electronics noise, and optional integer
ADC clipping/quantization. The template amplitude is ADC/PE and is not silently
normalized. Explicit event IDs preserve empty events. Pulse contributions
outside the acquisition window are omitted; metadata records PE times before
and after the sample window. There is no dark-count, afterpulse, discriminator,
or PMT space-charge saturation model; ADC clipping is the implemented
saturation mechanism.

## Calibration bundles

See `examples/synthetic_optical_calibration.json` for the schema. Real bundles
must use `status: "measured"` or `"provisional"` with accurate provenance.
Changing a label does not validate the values. Referenced CSVs have two numeric
columns, optional `#` comments, and no header row. Property units follow the
conventions above. A property can be a numeric constant, inline `[[x,y], ...]`,
or `{"file": "relative.csv"}`. Optical property tables must cover the entire
compiled wavelength range, preventing implicit extrapolation in bundle loading.

PDF/CDF entries additionally accept `kind` and `provenance`. PDFs are linear
between their density knots; integration and inverse sampling preserve those
slopes exactly for sources, PMT response, and WLS delays. Direct CDF inputs use
linear CDF interpolation. WLS wavelength CDFs are evaluated on the compiled
wavelength grid and interpolated there; select adequate wavelength resolution
and check grid convergence. Delays retain their original irregular table.
Repeated CDF abscissae can represent prompt point masses. CDF endpoints must
be 0 and 1. Source and WLS spectra must fit inside the compiled wavelength grid.

For the repository detector, supply `roles` identifying material keys `lar`
and `glass`, and surface keys `photocathode` and `tpb`. Include calibrated
entries for every other named material/surface used by that detector, including
vacuum, backs, wires, and cavity walls. `build_detector()` refuses to inherit
unlisted optics from the monochromatic setup:

```python
from chroma_lar.optical_calibration import OpticalCalibration
from chroma_lar.optical_simulation import OpticalSimulation

calibration = OpticalCalibration.load("my_calibration.json")
geometry = calibration.build_detector("detector_config_reflect_reflect3wires")
simulation = OpticalSimulation(geometry, calibration, backend="triton")
result = simulation.simulate_depositions(positions_mm, energies_mev,
                                         times=deposit_times_ns,
                                         event_indices=event_ids, seed=123)
result.save("optical_events.npz")
```

The full CLI accepts NPZ depositions (`pos`, `energy_mev`, optional `t`, `evidx`)
or photons (`pos`, `dir`, `pol`, `wavelengths`, optional `t`, `evidx`). Supply both
`--detector-config` and `--input` for a detector run. The NPZ output includes
optical hits, PE hits, waveforms, units, scene/calibration hashes, seed, and
truncation counts. A calibration hash includes all referenced CSV contents.

## Validation

From the workspace root:

```bash
PYTHONPATH=chroma-lite:chroma-lar python -m pytest -q \
  chroma-lite/test/test_optical_response.py \
  chroma-lite/test/test_spectral_transport.py \
  chroma-lar/test/test_scintillation_source.py \
  chroma-lar/test/test_pmt_coating.py \
  chroma-lar/test/test_optical_pipeline.py
```

Tests cover TTS statistics, CDFs/point masses, charge collection, waveform
superposition and clipping, empty events, scintillation yield/time/spectrum,
Beer-Lambert survival, dispersive group velocity, dielectric reflection,
WLS spectra/delays/escape sides, ID/tiling preservation, and CPU/Triton parity.
An actual PMT mesh test follows VUV photons through the coated outer window to
visible detection. GPU tests skip only when no GPU is accessible; run them on
a GPU before claiming device validation.

For matched single-PE measurements, the comparison CLI reports per-channel
light yield, time quantiles, and empirical CDF distances:

```bash
python benchmarks/compare_optical_measurements.py \
  --simulation optical_events.npz --observed measured_pe.npz \
  --output measurement_comparison.json
```

The observed file requires `times_ns`, integer `channels`, and positive scalar
`emitted_photons`. Match the source distribution, timing origin, channel map,
window, and signal definition. Backgrounds and normalization uncertainty must
be handled in the experimental analysis. These descriptive comparisons do not
automatically certify a detector calibration. No measured calibration or
detector measurement dataset is bundled with this implementation.
