# PMT room lighting

Six inward-facing Lambertian panels share the existing visible fill power.
The VUV beam, TPB conversion and material calibration are unchanged. At
2,500,000 photons, every fill-map spatial cell on all six walls receives visible
light (1,024/1,024 cells per face in full detail). Previously several faces had
unoccupied cells and the +X wall received much less fill energy.

The attached image is a four-sample browser render using those photon maps.
The independent 1,024-photon CPU audit passes transport and map comparisons;
6,617 VUV beam histories remain exactly unchanged in the 8,193-photon lighting
comparison. Restoring the old lighting layout retains all 8,193 original states.
The larger unchanged-input comparison is recorded in `../../webgpu_memory`.
