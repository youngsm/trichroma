# Browser scheduling and camera responsiveness

Measured with Chromium WebGPU on a local NVIDIA A100-SXM4-40GB. The old
implementation is `f830795`; the new implementation removes deliberate Balanced
idle time and increases the starting batch to use available GPU parallelism.
Each submission must finish before another is queued. Low mode still adds idle
time, hidden tabs pause new submissions, and Stop cancels at a batch boundary.
Batch duration targets are adaptive scheduling heuristics, not driver deadlines.

Both camera scenes use maps from 2,500,000 simulated photons. Times below are
wall time for the first camera sample, excluding simulation and map building.
These are single measurements on this GPU, not estimates for other hardware.
The unfinished old runs were stopped after six seconds; they are lower bounds.

| Scene | Image | Old | New |
| --- | --- | ---: | ---: |
| Prism | 160 × 100 | 1,311 ms | 79 ms |
| Prism | 640 × 400 | >6,000 ms | 685 ms |
| PMT | 160 × 100 | >6,000 ms | 209 ms |
| PMT | 640 × 400 | >6,000 ms | 3,281 ms |

The new UI coalesces pointer positions while allowing a small preview to finish.
Motion cancels old full-resolution refinement. The canvas retains its last
presented image while the next image computes, including resolution changes.
In a PMT scene with 640-pixel quality selected, actual middle-button dragging
presented four previews during 18 pointer moves spaced 35 ms apart. The final
pose appeared 205 ms after release; Stop completed in 181 ms. These measurements
include browser automation overhead. No further refinement occurred after Stop.
Controlled document visibility also verified that hidden tabs submit no new
work and resume when visible. See `controls.json`.

The scheduling change also passed the full interaction comparison again:
12,800 unchanged detector hit records and original surface pixels; 57,351
unchanged photon terminal states across three diagnostics and four camera
scenes; exact independent log-bin recounts; 2.5-million-photon fluorescence
arrival/delay population conservation; normal colors, middle-button panning,
camera preservation on resimulation, and cached log/linear axis switching.
See `parity-and-controls.json`. These finite checks do not establish universal
bitwise equivalence with original Chroma.
