# Primitive compiler and optical rendering validation

The automatic region compiler retains the full reflect3wires optical pipeline's
20 million input photons/s target on the local NVIDIA A100-SXM4-40GB. The current
water and pixel benchmarks use the general mesh transport engine; their rates
are not measurements of the specialized LAr scheduler.

## Full event performance

Rates below are total simulated input photons divided by total measured event
time. Construction, compilation warmup and file writes are excluded. Outputs
are downloaded and usable by the host before the timer stops. Each measured
seed is warmed before the general-mesh timings to exclude specialization of
previously unseen active-population sizes.

| Configuration | Photons per event | Repeats | Sustained photons/s | Best event photons/s |
| --- | ---: | ---: | ---: | ---: |
| reflect3wires, automatic regions | approximately 30,000,000 | 5 | 25,462,272 | 26,704,309 |
| reflect3wires, former region selection | approximately 30,000,000 | 5 | 25,468,203 | 26,657,535 |
| Pixel TPC, general mesh | 1,000,000 | 3 | 803,075 | 809,880 |
| Theia-like water, general mesh | 1,000,000 | 3 | 330,671 | 332,784 |

The two reflect3wires runs differ only in acceleration-region selection. Here
“former” does **not** mean the original Chroma CUDA physics implementation.
The rate difference is below the observed event-to-event timing variation.

Reflect3wires includes Poisson source yield, the supplied LAr spectrum and timing,
spectral propagation through all six wire planes and 162 PMTs, TPB response,
PMT collection efficiency/TTS/charge, pulse superposition, electronics noise,
ADC conversion and host-readable hits, photoelectrons and waveforms. Its
calibration is explicitly synthetic. Unsupported or absent physical laws must
not be inferred from the phrase “full event.” The source and calibration hashes,
model settings, stage times and memory measurements are in
[full_automatic_final.json](full_automatic_final.json) and
[full_legacy_final.json](full_legacy_final.json).

The water fixture contains 49,684 nominal 20-inch PMTs and 32,791,840 triangles.
It uses illustrative water/glass/QE tables and a fixed-count Cherenkov source.
At one million photons, transport takes about 0.7 s and the dense 49,684-channel
waveforms take about 2 s. The pixel fixture has 164 PMTs and 839,732 triangles;
it preserves the original configuration's area-averaged pixel-pad optics. Its
CPU source sampling is about 0.66 s, transport about 0.49 s, and readout about
0.10 s. Both include source, transfer, transport, terminal-state download, PMT
response and noisy digitized waveforms. Their fixed photon counts do not time
Poisson yield sampling. See [theia_water_final.json](theia_water_final.json) and
[pixel_tpc_final.json](pixel_tpc_final.json).

Every measured event has zero aborted, nonfinite, escaped, step-limited or
unfinished photons according to its detector's completion checks.

## Correctness evidence and its scope

The final local suite passed **275 tests, zero failures/errors/skips**. See
[final_tests/run.json](final_tests/run.json) and its full test log. The focused
native arithmetic regressions also pass Compute Sanitizer with zero errors.

* The pre-refactor Triton regression compares 749,196 photons and all 81 saved
  arrays, including terminal states, hits, photoelectrons, noisy ADC waveforms
  and metadata. All arrays are exactly equal: [equivalence.json](equivalence.json).
  This compares two Triton versions, not native Chroma.
* The actual water detector passed 131,000 CPU/general-Triton comparisons with
  exact terminal flags, detected photon identities, channels and wavelengths.
  Times use the recorded tolerance. Two boundary precision defects were
  reproduced, fixed, and followed with the requested tenfold larger sample.
  The CPU reference uses independent NumPy physics and CPU BVH traversal.
* The pixel detector passed 31,000 CPU/general-Triton comparisons and the
  CPU/GPU readout comparison. Numerical state tolerances are explicitly recorded
  in the report; this is not a raw-word native Chroma certificate.
* Historical original-Chroma checks independently reproduce normalization and
  native XORWOW draws and compare raw state words. Both full wire-detector
  fixtures passed every intermediate/final state in the 8,192-photon audit,
  then every native launch output in the 81,920-photon follow-up. Together the
  two larger wire runs cover 2,706,864 interactions and 10,129,627 draws with
  zero different words and no survivors. Detailed acceptance limits,
  the original's nondeterministic atomic queue behavior, and the remaining
  compatibility work are documented in
  [legacy_parity.md](../../../../docs/legacy_parity.md).

Failed intermediate reports are retained for diagnosis. In particular, old
Theia performance reports containing previously unseen JIT compilation and old
overlapping-layout experiments are not the final benchmark evidence.

## Browser photon camera

The browser implements forward optical transport, spectral illumination maps,
and a movable camera in WGSL. Jupyter embeds the same exported assets. The image
is a radiance estimate with documented spatial/spectral discretization and
finite camera samples. It is not a drawing of selected photon trajectories.

The prism, fluorescent coating and Rayleigh scenes have 30-million-photon native
WebGPU captures under [photon_camera/hardware_final](photon_camera/hardware_final).
Doubling 2.5 million to 5 million photons changes measured image luminance by
about 0.1%, rather than doubling exposure. The maps normalize by all launched
photons, including absorbed packets. A 300-million-photon follow-up of a near-wall
float32 rounding defect completed with no unexpected escapes or unfinished
photons; see [boundary_rounding/300m](photon_camera/boundary_rounding/300m).

The PMT scene uses the curved R5912 mesh, a synthetic TPB surface model, glass
and photocathode. Its optional camera cutaway leaves forward transport geometry
intact. UV false color is explicitly selectable and labeled. This is an
effective coating model, not a resolved microscopic TPB-film calculation.
The finer 3,856-triangle demonstration completed 2.5 million photons with
1,383,153 reemitted photons and 171,754 photocathode detections, with no escapes
or unfinished photons. Its images are under
[pmt_fine_hardware](photon_camera/pmt_fine_hardware).
The polished [PMT cutaway](photon_camera/pmt_presentation/pmt-cutaway-camera.png)
uses those same 2.5 million photons with 256 camera samples per pixel.
See [photon_camera.md](../../../../docs/photon_camera.md) for the estimator,
controls, notebook and physical limits.

The expanded 40,960-photon prism comparison has exact terminal outcomes and
wavelengths. Two photons straddle adjacent display-map cells in CPU/GPU
calculations; strict map equality remains false and is retained in the report.
The independent deposition check accounts for every affected cell and preserves
energy within the stated transport/accumulation tolerances. This is distinct
from the native-Chroma raw-word comparisons above. See the
[camera validation report](photon_camera/REPORT.md).

## Interactive detector geometry viewer

The native Triton viewer was measured separately with five complete
2.5-million-camera-ray frames per detector and an otherwise idle GPU:

| Detector | Mean frame time including RGB download | Camera rays/s |
| --- | ---: | ---: |
| Theia-like, 49,684 twenty-inch PMTs | 169.91 ms | 14.71 million |
| reflect3wires, all six analytic wire planes | 8.54 ms | 292.83 million |
| Pixel TPC, configured area-averaged model | 4.80 ms | 521.06 million |

These rays perform geometry visualization, not complete optical propagation.
PNG encoding is recorded separately; widget/network/browser painting is outside
the timer. Source hashes, preview measurements and images are under
[viewer_final](viewer_final).
