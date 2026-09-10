# Browser photon histories

Recorded HK particle steps drive a WebGPU Cherenkov event display inside the
existing 49,684-PMT Theia geometry. The browser propagates individual photons,
stores actual flight segments and renders them directly from GPU buffers.
Each polyline identifies one simulated photon. RGB is the absolute value of
its polarization components; PMT RGB is `(normal + 1) / 2` in world coordinates.
The display includes time scrubbing, short trails or complete histories,
stable path subsets, camera orbit/pan/zoom, view presets and PNG export.
“Explore first 10 ns” selects a close-up of the emission region, a short trail,
and playback at 0.1 simulated ns per real second. The speed can drop to
0.01 ns/s (100 real seconds per simulated ns). Separate 0–1 ns and 0–10 ns
slider windows expose early emission precisely. Playback uses elapsed wall
time and a floating-point clock independent of slider quantization.

Photon-age fading has an adjustable half-life and can be disabled. Moving
trails fade according to the current time minus that photon's emission time;
complete histories fade along each flight according to its age. This changes
visual opacity only. A secondary photon emitted at 2.9 µs starts at age zero.

All PMT arrivals are recorded, including photons without a displayed path.
The 3D PMTs default to 15% normal-color brightness (10% in the early close-up),
sum an exponential pulse from every arrival, and retain a brightness
contribution proportional to accumulated hit count. Only brightness changes;
the original surface-normal RGB color is preserved.

For PMT arrival times `t_i <= t`, the pulse amplitude is
`A(t) = sum(exp(-(t-t_i)/tau))` and accumulated charge proxy is the count `N(t)`.
Every photon has unit weight. Pulses add before the nonlinear display mapping:
`brightness = base + (1-base) * (1-exp(-gain*(A + weight*N)))`.
Default decay is 20 ns, accumulated-hit weight 0.35, and display gain 1.
Setting the accumulated-hit weight to zero gives a pure decaying response.
Complete histories show `N` alone. These are relative signals, not calibrated
charge or voltage: the model does not include QE, gain fluctuations or PMT
electronics. The display gain adjusts brightness, not simulated PMT gain.
The GPU recomputes the amplitude from recorded arrival times, so backward
scrubbing and changes to decay time are independent of frame rate or history.
The arrival counter lives directly over the 3D canvas; there is no separate
hit-map panel. Scrubbing backward removes later arrivals. The First PMT
arrivals window jumps to the first recorded arrivals in the current event.
Per-PMT sorted arrival lists are queried on the GPU, so playback needs neither repeated
transport nor GPU-to-CPU readback. These are ideal-surface photon intersections,
not quantum-efficiency-weighted or electronics-level detections.

```sh
python examples/photon_histories/build.py /tmp/photon-histories \
  --geometry PATH_TO_EXISTING_THEIA_BROWSER_BUNDLE
python -m http.server 8765 --directory /tmp/photon-histories
```

The website destination is `/chroma/photons/`. No homepage link is needed.
The build copies the unchanged shared traversal shader and scheduler. The
GPU helper has an optional storage-buffer limit request; other viewers keep
their default limits. The full Theia payload is SHA-256 verified on load. PMT and enclosure geometry participate in both
transport and camera depth clipping.

## Recorded events

| Particle | HK configuration / event | Recorded energy | Tracks | Input steps |
|---|---|---:|---:|---:|
| Muon | `config_01 / event_327` | 1999.748413 MeV | 306 | 16,875 |
| Electron | `config_03 / event_072` | 1998.289185 MeV | 525 | 21,364 |

`export_events.py` reads `DORAEMON/WAND/HK/GeV/test` HDF5 step and label files
read-only. `events.json` retains source event index, run, source-code commit,
step-data checksum and exact coordinate transform. Source positions undergo
only rigid rotation/translation and conversion from meters to millimeters;
recorded event t0 is subtracted from times. No track length is rescaled.
The muon sample includes late secondary light at about 2.9 microseconds.

New emission uses the Frank–Tamm spectrum over 360–700 nm. Each step is sampled
in proportion to its integrated yield, with its recorded starting beta and
direction held constant over the chord. Positions are uniform along that
chord; emission time advances by chord distance divided by beta*c. Zero-length
and subthreshold steps emit no new photons. The photon count is the rounded
mean yield, without Poisson fluctuations: 272,157 and 269,860 respectively.
The original G4 optical-photon wavelength band is unconfirmed, so its photon
counts are retained as metadata, not used to normalize this band.

This reconstructs optical emission from G4 particle steps; it does not replay
the original optical photons or HK hits. No bitwise parity with Geant4/Chroma
is claimed for this separate browser event display.

## Optical model and resource bounds

Water has `n = 1.322 + 3000/lambda_nm²`,
`n_group = 1.322 + 9000/lambda_nm²`, Rayleigh length 80 m at 450 nm proportional
to wavelength⁴, and illustrative absorption lengths interpolated in log space
from the explicit table in `event.wgsl`. Absorption is analog; Rayleigh direction
is sampled from `1 - (E dot direction)²`, and polarization is projected onto the
new transverse plane. These are the same illustrative water functions used in
the separate Cherenkov camera, not a calibrated HK optical database.

Actual PMT and enclosure triangle intersections terminate transport. PMTs are
ideal absorbing surfaces here; glass refraction and photocathode QE are not
modeled. The visual color is a vector diagnostic, not radiometric brightness.
The visual composition is inspired by Simon Blyth's photon-propagation images;
the implementation and transport here use WebGPU and Chroma geometry.

Each photon can make 32 flights; unfinished and sampling-failure counters are
reported explicitly. By default, up to 32,768 evenly spaced photon IDs retain
histories (64 MiB path buffer). Higher limits include 65,536, 131,072, 262,144,
and all photons; the full muon event uses 531.6 MiB of path storage. The
available limits follow the device's supported storage-buffer size, capped
at 300,000 retained histories. Larger buffers are allocated only when selected.
Allocation errors release partially created resources and retry with fewer
retained paths. This is bounded storage, not a guarantee of available VRAM.
Software adapters retain at most 2,048 histories. Increasing the retained
limit reruns the same seed; reducing the displayed limit reuses the event.
Clicking Simulate after reducing the limit also releases the larger allocation.
Neither changes the physical photon count. All photons contribute to counters
and the 3D hit display. Display controls never reseed the event. Transport and geometry rendering use bounded GPU batches. Time animation
is capped at 30 updates/s and pauses in hidden tabs. Camera geometry is cached
while scrubbing; motion uses a smaller preview before a 1440-pixel final image.

## Validation

```sh
PYTHONPATH=PATH_TO_PLAYWRIGHT python examples/photon_histories/check.py \
  /tmp/photon-histories --seeds 10 --baseline PATH_TO_PREVIOUS_BUNDLE \
  --output /tmp/history-check
```

`check.py` verifies Cherenkov angles, unit/transverse polarization, source
positions and emission times, the marginal wavelength spectrum, flight group
velocity, continuous histories and polarization updates at scattering. It
compares every history/counter/debug byte across different batch sizes, checks
terminal accounting and exercises time/path controls, playback, Stop and pan.
Arrival records must match terminal flight IDs, times and wavelengths. GPU
hit counts and latest-arrival times are checked against an independent CPU
scan during forward and backward scrubbing. Full-event hit records must be
bitwise unchanged when all paths are retained. Tests also exercise precise
slow playback, early close-ups, fading, late-light navigation, a constrained
128 MiB buffer limit, and injected allocation failure recovery. Thirty synthetic
PMT impulse-train cases check coincident hits, pulse overlap, decay, late hits
and backward scrubbing against independent exponential sums; recorded event
arrivals are checked the same way. The late-time render is compared with
and without the accumulated-hit contribution. With
`--baseline`, original history/counter/debug bytes must match the old page.
It also renders the full muon and electron events and saves screenshots.
Browser-native WebGPU is required for the default local hardware test.

`validation/` records the local NVIDIA A100 browser run. Complete histories for
163,860 photons (ten seeds per particle) match bitwise across batch sizes and
against the previously published transport implementation; see
`expanded-transport.json`. `report.json` records the final presentation and
resource checks, with another 16,386-history transport comparison.
Both full events have zero unfinished photons and zero traversal failures;
their measured simulation wall times were 0.211 s and 0.220 s, including
allocation, readback and arrival indexing but excluding initialization and rendering.
Cached early-event frames took 3.5 ms with 8,192 displayed paths and 11.0 ms
with all 269,860 electron paths (three frames each, at 1440 pixels wide).
The display still caps playback at 30 updates/s.
`validation/ring-integrated.png` and `ring-pulses-only.png` compare the same
muon event at 500 ns from the Along the cone view.
These are local measurements, not a performance guarantee for other GPUs.

References: [Geant4 Cherenkov model](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/xray_production/cerenkov.html),
[Optical polarization in Geant4](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/optical_photons/optical.html).
