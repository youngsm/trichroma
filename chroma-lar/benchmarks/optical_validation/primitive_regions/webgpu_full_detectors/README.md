# Complete browser detector validation

Local NVIDIA A100, Chromium WebGPU. Geometry rendering only.

* Wire planes: 960 exact nearest group/instance/triangle IDs against the independent float64 oracle, expanded tenfold after fixing a neighboring-facet rounding discrepancy. Compensated arithmetic retains original leaf vertices. The camera reference follows the shader float32 operation order; its original precomputed aspect ratio introduced one 2.38e-7 component error at the expanded resolution.
* Full pixel TPC pad close-up: 2,560 exact hit identities, both full million-pad faces retained in shared mesh tiles.
* Theia overview: 2,560 exact hit identities.

Each report includes distance and normal errors, scene/source hashes, adapter identity, and a separate 2.5-million-ray frame. Timings separate summed active batch queue completion from elapsed time including cooperative pauses. Screenshots are actual browser output. These finite checks do not establish universal equivalence with Chroma optical transport.

Reproduce with `check_webgpu_viewer.py BUNDLE --hardware --full-frame --repeat 1 --detector NAME --output OUTPUT`; add `--view 'Wire planes' --debug-width 40 --debug-height 24` for wires or `--view 'Pixel pads'` for pixels. The exporter defaults to the three complete detector scenes.
