The PMT cutaway uses one unchanged event of 2,500,000 optical photons, seed 901,
with the 3,856-triangle demonstration mesh. Camera angle, exposure, resolution,
and camera sample count change only the view of the resident photon maps.
`presentation.json` records the event counts and each view. Its copied
`validated_assets` are the exact shader, scene, stylesheet, and module bytes served.

`pmt-cutaway-camera.png` is the final 1280 × 800 view with 256 camera samples.
UV false color is disabled. The cutaway is a diagnostic camera view; forward
transport retains the complete PMT. The visible facet structure includes the
finite triangle emission-map reconstruction. No image enhancement or generated
image is used. Queue times are diagnostic and are not throughput claims.

`final_ui.json` checks the final exported page on software WebGPU: styles and
embedded font, PMT-only layer legend, exposure and cutaway controls, finite image,
debug triangle decoding, and navigation. The final debug decoder adds triangle
IDs after the presentation browser loaded; this changes no shader or image.
The export regression suite also passed all four tests.

The scripts beside this file reproduce the presentation controller and focused
UI check from the worktree root. The controller accepts one JSON object per line
with `name`, `samples`, `width`, `height`, `eye`, `target`, `exposure`, `cutaway`,
and `uvFalseColor`; `{"quit": true}` closes its browser cleanly.
