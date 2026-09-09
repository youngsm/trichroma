# Browser photon histories

Recorded HK particle steps drive a WebGPU Cherenkov event display inside the
existing 49,684-PMT Theia geometry. The browser propagates individual photons,
stores actual flight segments and renders them directly from GPU buffers.
Each polyline identifies one simulated photon. RGB is the absolute value of
its polarization components; PMT RGB is `(normal + 1) / 2` in world coordinates.
The display includes time scrubbing, short trails or complete histories,
stable path subsets, camera orbit/pan/zoom, view presets and PNG export.

```sh
python examples/photon_histories/build.py /tmp/photon-histories \
  --geometry PATH_TO_EXISTING_THEIA_BROWSER_BUNDLE
python -m http.server 8765 --directory /tmp/photon-histories
```

The website destination is `/chroma/photons/`. No homepage link is needed.
The build copies the shared traversal shader, GPU allocator and scheduler;
it does not change those shared implementations. The full Theia payload is
SHA-256 verified on load. PMT and enclosure geometry participate in both
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
reported explicitly. At most 32,768 evenly spaced photon IDs retain histories
(64 MiB maximum path buffer); all simulated photons contribute to counters.
Software adapters retain 2,048 histories. Display controls never reseed the
event. Transport and geometry rendering use bounded GPU batches. Time animation
is capped at 30 updates/s and pauses in hidden tabs. Camera geometry is cached
while scrubbing; motion uses a smaller preview before a 1440-pixel final image.

## Validation

```sh
PYTHONPATH=PATH_TO_PLAYWRIGHT python examples/photon_histories/check.py \
  /tmp/photon-histories --seeds 10 --output /tmp/history-check
```

`check.py` verifies Cherenkov angles, unit/transverse polarization, source
positions and emission times, the marginal wavelength spectrum, flight group
velocity, continuous histories and polarization updates at scattering. It
compares every history/counter/debug byte across different batch sizes, checks
terminal accounting and exercises time/path controls, playback, Stop and pan.
It also renders the full muon and electron events and saves screenshots.
Browser-native WebGPU is required for the default local hardware test.

`validation/` records the local NVIDIA A100 browser run. Complete histories for
163,860 photons (ten seeds per particle) match bitwise across batch sizes.
Both full events have zero unfinished photons and zero traversal failures;
their measured simulation wall times were 0.186 s and 0.192 s, including
allocation and counter readback but excluding initialization and rendering.
These are local measurements, not a performance guarantee for other GPUs.

References: [Geant4 Cherenkov model](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/xray_production/cerenkov.html),
[Optical polarization in Geant4](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/optical_photons/optical.html).
