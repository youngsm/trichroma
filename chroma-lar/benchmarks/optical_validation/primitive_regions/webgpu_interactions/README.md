# Browser interaction validation

Local NVIDIA A100 / Chromium WebGPU, compared against the published f830795 implementation.

- Five geometry views, including both TPC close-ups: 12,800 unchanged hit records and original surface-color pixels. Normal RGB checked against the returned world-space normals, within one output quantization level.
- The normal switch preserves the camera. Actual middle-button drags translate eye and target equally; photon-camera panning stops at room boundaries without changing direction. Rerunning each camera scene preserves the translation.
- All three diagnostics and four photon-camera scenes retain identical terminal states for 8,193 photons each (57,351 total). Existing diagnostic histograms and sampled paths also remain identical.
- Independently recounted positive-time bins and under/overflow from downloaded GPU terminal states. A 2.5-million-photon fluorescence event preserves the complete arrival/delay populations. Log/linear switching redraws the stored event without another simulation.

The screenshots are browser output. These finite checks are not universal bitwise-equivalence claims for Chroma optical transport.
