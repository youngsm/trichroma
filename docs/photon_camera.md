# Spectral photon camera

Open [photon_camera.ipynb](../notebooks/photon_camera.ipynb) with the
**TriChroma (GPU, pimm-bench)** kernel. The notebook exports a portable scene
bundle and embeds its interactive WebGPU camera as self-contained, trusted
notebook HTML. Shader and table bytes are identical to the portable bundle;
the embed requires no HTML file route or notebook-server security changes.
The browser GPU performs the
simulation and rendering. The exported ZIP also works independently of Python
and Jupyter after it is served over localhost or HTTPS.

The main image is a three-dimensional camera view. Forward optical photons
populate illumination maps; camera rays gather that illumination through the
scene. Orbit changes reuse the maps. Changing the optical photon count changes
their statistical precision, while exposure controls display brightness.

The scenes contain a dispersive glass prism, an effective fluorescent coating,
a polarized Rayleigh-scattering medium, and a TPB-coated PMT. The prism and
coating camera variants include a weak Rayleigh haze with scattering length
`10,000 mm × (wavelength / 450 nm)^4`. The original diagnostic fixtures retain
their infinite inactive scattering lengths. The camera starts inside the room
and faces the prism's forward caustic, preserving ordinary geometric visibility.

The fluorescent surface absorbs ultraviolet photons, reemits visible photons
with probability 0.85, and samples the exported emission spectrum and delay.
Its camera variant coats a glass substrate with refractive index 1.5 and
60 mm absorption length, illuminated by a 20 mm beam. The glass objects sit
2 mm above the floor. A ceiling area light emits visible photons that undergo
the same refraction, scattering and absorption as the main beam.
The camera integrates steady-state illumination. The
[diagnostic notebook](../notebooks/optical_showcase.ipynb) exposes the time
distribution separately. Ultraviolet illumination is invisible in the camera;
visible light appears when wavelength shifting produces it.

The PMT uses the repository's R5912 contour, 203.2 mm nominal diameter and
3 mm glass, with the existing synthetic full-detector calibration. A
126–130 nm beam reaches the effective TPB surface. It reemits with probability
0.9 into the tabulated 400–460 nm spectrum and samples the tabulated delay,
including its tail to 10 microseconds. The glass absorbs unconverted VUV;
visible photocathode detection probability is 0.25, with the remainder
absorbed. These are software-validation inputs, not measured PMT calibration.
Optical-table endpoint values above 500 nm extend the visible fill spectrum.
For this PMT scene, the visible fill is shared by six inward-facing area lights
centered on the room faces. Each emits one sixth of the original ceiling
light's power, so the total stays fixed while illumination surrounds the PMT.
The focused VUV beam and optical calibration are unchanged.
The same compiled BVH accelerates forward photons and camera queries while
preserving the original triangle indices. The curved TPB emission maps use
each triangle's actual area and separate its two outgoing hemispheres.
The demonstration samples the profile with 20 axial rings and 48 azimuth
segments, producing 3,840 PMT facets and 624 coating charts. This is a finer
polygonal approximation of the same contour, not an exact subdivision of the
earlier coarse mesh. Both forward photons and camera queries use it; the
original detector geometry is unchanged.

The PMT cutaway changes camera visibility only: the complete PMT remains in
every forward photon simulation. The exposed view has no invented luminous
cut cap. The coating is an effective interface, without a resolved microscopic
TPB film. Its emitted hemisphere and later dielectric Fresnel crossings are
modeled explicitly. A labeled UV false-color view, when selected, is a
diagnostic display rather than visible radiance.
For the PMT cross-section presentation, enable the cutaway and set exposure
to `400000`. The default view uses eye `[-30, -185, 45]` and target
`[100, 0, -15]` mm; the initial scene remains uncut with UV false color off.

Drag to orbit, middle-drag (or Shift-drag) to pan, and use the wheel to zoom.
Panning translates both the camera and its target in world coordinates, keeping
the viewing direction and distance fixed. Translation stops at the room walls.
These controls reuse the current photon maps; rerunning the same scene preserves
the composed camera view. Motion coalesces to the newest view while a small
preview finishes; changing the view cancels older full-resolution refinement.
The first image is a quick preview before the selected quality refines.
The hardware view accumulates 8 camera passes per pixel (two on a software
adapter). Each pass jitters the pixel ray and samples optical interactions
across the 64 wavelength bins, skipping bins with negligible color response.
These passes reuse the same forward photon maps: more passes reduce camera
sampling noise, while more simulated photons reduce photon-map sampling noise.

The **Light-map detail** control selects full or reduced spatial maps for the
next simulation. Reduced mode retains all 64 wavelength bins and the complete
requested photon population, with the same transport and random streams.
It uses a 32 × 20 × 12 volume grid, 64 × 64 beam-wall maps and 16 × 16 fill-wall
maps. The curved PMT keeps its original per-facet fluorescence charts. Its maps
need 17.9 MiB in reduced mode, compared with 115.8 MiB in full mode. Other scenes
use 20.6 MiB in reduced mode. Driver allocations and image buffers are additional.

Allocation failures release any buffers already created by that operation.
A GPU memory failure triggers one retry on a fresh device with reduced maps,
preserving the requested photons and camera pose. This also handles a lost
device reporting `VK_ERROR_OUT_OF_DEVICE_MEMORY`. A failed retry stops with an
error instead of repeatedly allocating. Lowering the photon count alone does
not reduce these fixed-size lighting maps.

Adapter selection tries hardware first and then requests a software adapter if
needed. When the browser supplies a CPU adapter, the page starts with 10,000
photons, reduced maps, a 160-pixel camera and smaller compute batches. Browsers
may refuse software WebGPU, particularly when hardware acceleration is disabled;
a website cannot enable that browser capability. The fallback is conditional
on the browser exposing a usable adapter.

## What the camera estimates

Packet energy is `source_scale × 450 / wavelength_nm`. A stable Philox draw
selects the main beam with probability 0.8 and visible fill with
probability 0.2. Beam packets have `source_scale = 1/0.8`. Ceiling packets
sample a Lambertian rectangle with constant radiant energy
`L × area × pi / 0.2` per packet; their scale includes the initial wavelength
so wavelength shifting still reduces their energy correctly. Here
`L = 0.00003` in relative energy per square millimetre and steradian, and the
emitter measures 180 by 80 mm. For the PMT, a separate stable draw selects
one of six equal-area panels; each has radiance `L/6`. The packet weight and
total fill power remain identical to the single ceiling source. Source choices remain identical for existing
photon IDs when N changes.

Every estimate divides by the **total emitted photon count N**, including
photons that are subsequently absorbed. Doubling N therefore reduces sampling
noise without doubling the expected brightness. A wavelength-shifted photon
carries its new, lower energy.

Diffuse wall cells estimate outgoing radiance as
`reflectance × deposited_energy / (pi × N × cell_area)`. This represents the
first diffuse wall reflection. Subsequent diffuse wall interreflection is
outside this bounded studio renderer.

Scattering collisions accumulate the incoming polarization second-moment
matrix `M = sum(energy × p pᵀ)` in spatial and wavelength bins. The
unpolarized camera source term is
`3/(8 pi N voxel_volume) × (trace(M) − omegaᵀ M omega)`. A camera carrying a
random transverse polarization `e` uses the equivalent polarized estimator
`2 × 3/(8 pi N voxel_volume) × eᵀ M e`, then carries that polarization through
Fresnel interactions. This avoids treating a real scattering collision as a
second scattering of its already sampled outgoing polarization.

Camera segments integrate their source term with Beer attenuation using the
segment's material and wavelength. Clear glass does not acquire the outside
medium's extinction. Refractive camera transmission includes the index-squared
factor required by radiance transport.

Fluorescent emission is uniform in its sampled outgoing hemisphere. Its
surface radiance is proportional to `1 / abs(cos(theta))`; the projected flux
integrates to the deposited hemisphere energy. It must not be silently treated
as Lambertian emission. Contributions from all six coating faces are retained.

The maps use finite spatial cells and 64 wavelength bands, introducing density
and spectral approximation errors. Beam walls use 128 by 128 cells per face;
the broad ceiling illumination uses a smoother 32 by 32 wall map. The volume
uses 64 by 40 by 24 cells, with trilinear reconstruction; surface maps use
bilinear reconstruction. The camera follows at most 64 dielectric interfaces
and drops negligible transmission below its numerical threshold. These bounds
and finite map kernels limit this renderer's accuracy.
The curved PMT coating uses constant density over each exact-area facet instead
of the rectangular coating's bilinear chart.

Color uses the repository's bundled CIE1964
matching functions and an XYZ-to-linear-sRGB transform. Signed spectral RGB
components are retained through spectral integration, then display exposure,
tone mapping and gamut clipping produce the image. These relative energy
estimates are not an absolute photometric calibration.

## Reproduce and inspect

```python
from chroma_lar.photon_camera import export_camera_bundle
export_camera_bundle("/tmp/spectral-studio")
```

Serve that directory locally and open `camera.html`. The portable bundle also
includes `physics.html`, the detailed optical-event diagnostic page.

The Python support module provides independent flux-conservation,
polarization, attenuation and normalization checks, plus an intentionally slow
CPU collision recorder for small GPU map audits. Run them with:

```sh
PYTHONPATH=chroma-lite:chroma-lar python -m pytest -q chroma-lar/test/test_photon_camera.py
```

The shared optical transport was checked separately against the CPU oracle on
245,760 photons across the three original diagnostic scenes. That comparison
checks flags, channels, stable IDs, numerical terminal states, recorded path
vertices, full-count histograms, RNG vectors and interactive controls. It does
not establish correctness for every possible geometry or camera. See the
[recorded browser comparison](../chroma-lar/benchmarks/optical_validation/primitive_regions/webgpu_physics/final/verification.json)
for the exact tolerances and provenance.

The camera audits separately check its emitted-count normalization, polarized
scattering source, attenuated integration along all six axis-aligned directions,
PMT photocathode counts, and BVH versus brute-force transport and camera queries.
They retain a strict CPU/GPU map comparison. A small flight-coordinate difference
can move a packet across a spatial cell edge even when its wavelength, fate and
energy agree. The additional deposition check reports every such adjacent-cell
crossing and bins the recorded GPU wall position independently; it keeps CPU
packet energies and interior collision records. This is a quantified spatial
quantization difference, not exact map parity.

After the initial cell-edge difference, the prism comparison expanded from
4,096 to **40,960 photons**. Flags, channels, stable IDs and wavelengths agree
exactly; numeric terminal states meet the unchanged tolerances, with maximum
position difference 0.01220 mm and time difference 0.0000621 ns. Two packets,
IDs 738 and 39164, land across adjacent wall-cell edges. Their strict map
comparison remains explicitly false. Independently binning their recorded
GPU wall positions accounts for every cell difference while retaining CPU
packet energies and interior collision records; summed energy is unchanged
by the cell reassignment. The separate native 40,960-photon run completes
every photon and conserves weighted source energy to 5.6e-8 relative.
See the [expanded comparison](../chroma-lar/benchmarks/optical_validation/primitive_regions/photon_camera/prism_expanded/verification.json)
and [camera validation report](../chroma-lar/benchmarks/optical_validation/primitive_regions/photon_camera/REPORT.md)
for exact bounds, retained failures, hardware PMT checks and rendered images.
