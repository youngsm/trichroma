# Every difference from CUDA Chroma

This page lists every difference between TriChroma's default engine and CUDA
Chroma that can change an output, with the measurements behind each claim.
The list comes from a line-by-line comparison of the two code paths.
[Correctness](../correctness.md) has the short version.

Terms used below:
- **CUDA Chroma** is chroma-lite's PyCUDA backend. TriChroma's
  [exact mode](exact_mode.md) reproduces it bit for bit, so the CUDA Chroma
  column also describes exact mode.
- **TriChroma** is the default engine.
- **The LAr TPC** is chroma-lar's liquid-argon TPC with three wire planes
  (configuration `detector_config_reflect_reflect3wires`), used for the
  measurements.

Each difference has a kind:
- **(a)** another random realisation, or last-bit rounding: same
  distributions;
- **(b)** a fix of a CUDA Chroma bug;
- **(c)** output format or API behaviour only;
- **(d)** an edge case where one or both codes are undefined.

"Switch" names the `CHROMA_TRITON` option that restores CUDA Chroma's
behaviour; "Effect" is the measured effect on the LAr TPC
([measurements](#measurements)).

## Random numbers

| ID | CUDA Chroma | TriChroma | Switch | Effect | Kind |
|---|---|---|---|---|---|
| R1 | XORWOW stream of the thread a photon lands on | Philox-4x32-10, keyed by seed, with counter (photon, steps done, block) | none | another realisation, same distributions | (a) |
| R2 | draws taken one after another | a block of four numbers per kind of event (transport, surface, re-emission, each diffuse-reflection trial, roulette) | none | same | (a) |
| R3 | `curand_uniform`: values in (0, 1], steps of 2^-32 near 0 | 2^23 values from 2^-24 to 1 - 2^-24 | none | sampled distances end at 16.6 attenuation lengths (probability 6e-8); probabilities quantised to 2^-23 | (a) |
| R4 | 64-bit seed | seed modulo 2^32 | none | seeds equal modulo 2^32 give the same run | (a) |
| R5 | a photon's numbers depend on queue order, chunking, `nthreads_per_block`, `max_blocks` and tracking | only on (seed, photon, step) | none | runs repeat exactly, for any batch size | (a) |
| R6 | no photon numbering | photons are numbered across all `simulate()` calls of a `Simulation`, in input order; input is read one batch ahead | `no-pipeline` for the read-ahead | another realisation | (a) |
| R7 | the DAQ continues the transport streams | separate Philox counters, never reached by transport | none | DAQ only | (a) |

No random number is used twice or shared between photons.

## Arithmetic

| ID | CUDA Chroma | TriChroma | Switch | Effect | Kind |
|---|---|---|---|---|---|
| A1 | nvcc fast-math instructions | Triton/LLVM float32, with its own fused multiply-adds and `libdevice` functions | none | last bits; a photon's path diverges once a branch flips | (a) |
| A2 | normalisation by three divisions, trigonometric Fresnel formulas, rotations | multiplication by 1/sqrt, algebraic Fresnel formulas, mirror formula | Fresnel: `legacy` (formulas, not bits) | last bits | (a) |
| A3 | reflected and refracted directions from `acosf` and a rotation; within 2.4e-4 rad of normal incidence the angle rounds to 0 | vector formulas | Fresnel: `legacy` | CUDA Chroma's directions are off by up to 2.4e-4 rad near normal incidence | (a)/(b) |

## Geometry

| ID | CUDA Chroma | TriChroma | Switch | Effect | Kind |
|---|---|---|---|---|---|
| G1 | Chroma's BVH (16-bit quantised boxes) with a stack; on an exact tie the first triangle in search order is kept | surface-area-heuristic trees in eight direction-ordered copies, stackless, with a top-level tree over repeated meshes | none | edge ties only | (a) |
| G2 | triangles with abs(det) < FLT_EPSILON are skipped (an absolute cut) | only det = 0 is skipped | none | the cut only loses grazing hits on sub-millimetre triangles | (b)/(d) |
| G3 | repeated meshes stored in world coordinates | the ray is transformed into the mesh's frame in float32 | none | hit points agree to 1.2e-3 mm (99.99%), normals to 2.4e-7 | (a) |
| G4 | a box face is two triangles; a photon crossing it, or reflected at a point that rounds past it, can hit the other triangle at t ~ 1e-5 mm and end up inside the solid | up to 16 axis-aligned box solids are tested as boxes; the triangle within the face is located with exact edge functions | none | 8 photons in a million end inside the cathode's steel in CUDA Chroma | (b) |
| G5 | 1000-entry traversal stack; an overflow breaks the search | stackless | none | none (exact mode stops with an error on such trees) | (d) |
| G6 | material and surface numbers stored in 8 bits; 128 and above read the wrong table rows | 32-bit numbers | none | none (few materials) | (b) |
| G7 | zero-area triangle: NaN normal, photon aborted | zero normal, photon passes | none | none | (d) |
| G8 | a bulk event's material is that of the next boundary | with the optional empty-space grid, it comes from a grid cell certified by sampling rays | `no-grid`; the default engine never uses the grid | none | (d) |
| G9 | a photon leaving a solid through a face in the plane of a box face is on that plane, and the box face needs t > 1e-6, so rounding can let it through | the box face is met at distance 0 when the photon is on its plane within a few ulp | `legacy` | the LAr TPC's PMT backplates lie in its walls: 0.5% of photons leave the TPC in CUDA Chroma; +0.08% detected light | (b) |
| G10 | a ray through the edge or vertex shared by two triangles can miss both (a crack) | watertight triangle test (Woop, Benthin and Wald 2013) on shared float32 vertices: such a ray meets at least one | none | not measurable (+0.0002% ± 0.01% detected light) | (b) |
| G11 | - | a box whose entry face was at most 1e-6 mm ahead was missed, and the ray passed through the solid; it is now met | `legacy` | 4.66M adversarial rays against float64: 808 wrong answers before, none after; no effect on the LAr TPC | (b) |
| G12 | - | an enclosing mesh (a tank or cavity) is searched after everything else, so where one of its faces lies within float32 rounding of another mesh's face (a part mounted flush on a tank), the other face can be the hit | `strict` | none on the test detectors; 140 in a million photons aimed at such faces | (a) |

## Wire planes

CUDA Chroma's three wire bugs all lose light. Each one leaves a photon inside
a wire's steel (absorption and scattering length 0), where the photon is
absorbed or its weight becomes 0 or NaN (`exp(-0/0)`).

| ID | CUDA Chroma | TriChroma | Switch | Effect | Kind |
|---|---|---|---|---|---|
| W1 | the discriminant is formed as `B*B - A*C`, two terms of order t² whose difference is of order r², so float32 rounding determines hits beyond a few hundred mm | `A*r*r - (wv*dn - wn0*dv)^2`, the same quantity without cancellation | `legacy-wires` or `legacy` | W1-W3 together: +18.0% ± 0.1% detected light | (b) |
| W2 | the float32 reflection point often lands inside the wire; the next search returns the exit point as a hit from inside, and the photon moves into the steel | the wire a photon has just left is skipped | same | (in W1) | (b) |
| W3 | hit point and normal come from the noisy root | from the same exact decomposition | same | (in W1) | (b) |
| W6 | a plane is skipped when the distance to the plane of the wire axes exceeds the mesh hit plus one radius, but an oblique ray enters the wires earlier, so a ray ending on a wall that the wires pass through can reach wall points inside a wire | skipped only when the plane's entry lies beyond the mesh hit | same | +0.0016% (1,568 photons in 10M no longer lost) | (b) |
| W4 | CUDA Chroma's float32 bits | `legacy-wires` is CUDA Chroma's algorithm in Triton arithmetic, so its noise differs | none | the two answers differ on 0.73% of rays in each direction | (a)/(d) |
| W5 | a NaN wire distance (ray exactly parallel to the wires) is accepted and corrupts the nearest-wire choice | rejected | none | none | (d) |

## Surfaces and bulk

| ID | CUDA Chroma | TriChroma | Switch | Effect | Kind |
|---|---|---|---|---|---|
| S1 | Fresnel within 2.4e-4 rad of normal incidence: 0/0 = NaN coefficients, so the photon is always transmitted | angle-free coefficients (4% reflected for 1.0/1.5) | `legacy` | not measurable (~1e-7 of crossings) | (b) |
| S2 | Fresnel elsewhere: trigonometric formulas | algebraic, with the same branches and draws | `legacy` | last bits | (a) |
| S3 | specular direction by rotating the normal: NaN, and the photon aborted, when it is exactly anti-parallel to the normal | mirror formula `d - 2(d.n)n` | none (`legacy` keeps the mirror formula) | none | (b) |
| S4 | polarization unchanged by a mirror reflection, so it is no longer transverse | mirrored, `p - 2(p.n)n` | `legacy` | -0.37% ± 0.14% detected light, and a shift of the time spectrum (Rayleigh scattering depends on the polarization) | (b) |
| S5 | complex, dichroic and angular surface models | `NotImplementedError` | none | not used | (c) |
| B1 | a property at the top wavelength of the grid is read one entry past the table | clamped | none | none | (b) |
| B2 | absorption, scattering, re-emission, Rayleigh | the same algorithms | - | only through R1-R3 and A1 | (a) |

## Weights

The weighted-mode rules are the same in both: the 1e-4 weight threshold, bulk
absorption as `exp(-d/L)`, surface absorption folded into the weight, and
forced detection (weight × detect) on every detecting surface.

| ID | CUDA Chroma | TriChroma | Switch | Effect | Kind |
|---|---|---|---|---|---|
| WT1 | none | optional Russian roulette (`roulette=<w>`): below weight w a photon continues with probability weight/w at weight w, or ends (flag bit 29) | off by default | +0.003% ± 0.006% (unbiased); fewer, heavier hits | (b)/(c) |

## Flags, outputs, DAQ, scheduling

| ID | CUDA Chroma | TriChroma | Switch | Kind |
|---|---|---|---|---|
| H1 | NaN abort flagged with bit 15 | bit 31 (`chroma.event.NAN_ABORT`) | `legacy` | (c) |
| H2 | photon history kept in 16 bits (input flag bits 16-31 are lost) | 32 bits | none | (c) |
| H3 | input photons with bits 0-3 or 15 set are not propagated | bits 0-3, 29 or 31 | none | (c)/(d) |
| N1 | direction and polarization renormalised at every launch | once, at the start | none | (a) |
| O1 | `use_packed=True`: `photons_end` holds the initial positions and directions, and the DAQ is given the initial weights when no hits were extracted | final states | none | (b) |
| O2 | `photon_tracks` include photons already finished on input | omitted | none | (c) |
| O3 | optical tables outside their valid range (probabilities outside [0, 1], unnormalised CDFs, ...) are accepted | such tables raise an error | none | (c) |
| D2 | DAQ tables from `Detector._pdf_to_cdf` (`set_time_dist*`, `set_charge_dist*`) are one entry short, so memory past their end is read: a 0 in fresh memory, which puts every time and charge at the last bin edge, and later whatever the memory held | the missing leading 0 restored (chroma-lite now builds complete tables; detectors built or pickled earlier still carry short ones) | `legacy` appends the 0 of fresh memory | (b) |
| SC1 | a batch is read, propagated and returned before the next is read | the next batch is read and propagated before the current one's events are returned | `no-pipeline` | (a)/(c) |
| SC2 | - | three schedulers (fused kernel, wavefront rounds, grid shortcut) share the physics code and random-number keys | `wavefront`, `no-grid` | (a) |

## Measurements

### Setup

- **Detector:** the LAr TPC, 128 nm light, weighted mode
  (`use_weights=True`), `max_steps=1000`, 10M photons per run, on an A100.
- **Point source:** isotropic photons at (-450, 60, -120) mm with polarization
  +x, the voxel of chroma-lar's `generate_lut.py`.
- **Spread source:** photons uniform over the simulated half of the detector,
  isotropic, with random transverse polarization.
- **Observables:** each photon's detected weight (0 if not detected), summed
  in total, per PMT, and per arrival-time bin (5 ns up to 400 ns, then wider).
  Bins are merged until both samples have 100 hits.
- **Scripts:** `benchmarks/validation/`.

### Exact mode against CUDA Chroma

A weighted CUDA Chroma run of the point source (200K photons) was recorded and
replayed: `photons_end` is identical in every bit, and so are the hits as a
set (CUDA Chroma returns hits in a different order on every run).

### Same physics, different random numbers

With `legacy`, TriChroma runs CUDA Chroma's physics, and only the (a)
differences remain. Detected weight per photon, averaged over seeds
(difference = CUDA Chroma minus TriChroma):

| Source | Seeds × 10M | CUDA Chroma | TriChroma `legacy` | Difference | Per-PMT χ²/dof | Time χ²/dof | Histories |
|---|---|---|---|---|---|---|---|
| point, no wires | 4 | 0.043891 | 0.043876 | +0.035% ± 0.073% | 88.6/81 | 99.4/101 | within 1.0σ |
| spread, no wires | 4 | 0.052468 | 0.052497 | -0.055% ± 0.071% | 71.5/81 | 118.0/101 | within 1.2σ |
| point | 6 | 0.031450 | 0.031417 | +0.106% ± 0.072% | 71.5/81 | 103.2/97 | within 1.4σ |
| spread | 6 | 0.038308 | 0.038260 | +0.124% ± 0.070% | 99.2/81 | 91.7/97 | within 1.8σ |

The 0.1% offset with wires is W4: CUDA Chroma's wire bug is float32 noise,
which other arithmetic reproduces only statistically. In both codes, about 1%
of photons leak out through the PMT backplates in the walls (G9); the exact
rate depends on how positions round.

With every fix on (the default), the detected light is higher:

| Source | CUDA Chroma | TriChroma | Difference |
|---|---|---|---|
| point | 0.031450 | 0.036965 | +17.53% ± 0.07% |
| spread | 0.038308 | 0.044471 | +16.09% ± 0.07% |

### Each change on its own

With the same seed, every photon gets the same random numbers, so the
differences are paired:

| Change | Point | Spread | Photons changed (point) | Per-PMT χ²/dof | Time χ²/dof |
|---|---|---|---|---|---|
| polarization, Fresnel, NaN bit (S1, S2, S4, H1) | -0.366% ± 0.141% | -0.268% ± 0.120% | 3.57M | 95.6/81 | 193.6/93 |
| wires (W1-W3, W6) | +17.96% ± 0.10% | +16.27% ± 0.09% | 3.87M | 30643/81 | 112415/93 |
| box faces (G9) | +0.082% ± 0.002% | +0.082% ± 0.002% | 21,475 | 1868/81 | 7349/93 |
| roulette 0.05 (should be 0) | +0.003% ± 0.006% | +0.003% ± 0.005% | 2.34M | 88.0/81 | 86.9/81 |
| wavefront scheduler (should be 0) | +0.0001% ± 0.0001% | -0.0001% ± 0.0002% | 56 | 47.2/43 | 40.7/43 |
| grid shortcut (should be 0) | -0.0004% ± 0.0003% | -0.0006% ± 0.0003% | 97 | 63.1/63 | 75.0/64 |

All fixes together: +17.62% ± 0.15% (point) and +16.06% ± 0.13% (spread).
The schedulers change a few dozen photons in 10M, where a different fused
multiply-add flips a branch. Roulette changes 2.3M photons but no tally; its
cost is extra variance in the late time tail only:

| Arrival time | Share of the light | Variance, roulette over default |
|---|---|---|
| all, per PMT | 100% | 1.00 |
| < 100 ns | 85-90% | 1.00 |
| 100-200 ns | 9-14% | 1.06-1.09 |
| 200-300 ns | 1-1.5% | 2.5-2.8 |
| 300-400 ns | 0.1-0.2% | 12-13 |
| > 400 ns | < 0.05% | ~80 |

### Geometry, ray by ray

Every step of 16,000 photons (half from each source) gives 1.96M rays from
real paths. For each ray, the nearest boundary was computed with both
TriChroma's geometry and CUDA Chroma's (reproduced by exact mode from a
recorded scene).

- **Meshes.** 1,718,878 rays hit the same triangle in both. Distances agree
  to 1.2e-3 mm for 99.99% of them (float32 at metre scale; at most 0.037 mm),
  and normals to 2.4e-7. 166 rays (8.5e-5) hit different triangles: 164 are
  ties within 1e-3 mm at shared edges or vertices, and 2 are coincident
  PMT/wall faces (G9). No mesh is missed by one and hit by the other.
- **Wires.** 226,742 rays involve a wire in either code, and the results
  differ on 221,032. Each of those was compared with a float64 calculation:

  | Rays where the two differ | TriChroma | CUDA Chroma |
  |---|---|---|
  | matches float64 | 221,003 (99.987%) | 2,589 (1.2%) |
  | wire hit where float64 has none | 17 | 117,745 (up to 25.5 radii from any wire) |
  | misses a wire hit | 9 | 32,698 |
  | wrong wire | 1 | 7,464 |
  | right wire, distance off | 2 | 60,536 (median 0.056 mm, up to 2.9 mm) |

  TriChroma's 29 errors all lie within 0.2% of tangency: float32 noise at
  grazing incidence.

### Tests against exact expectations

- `tests/integration/test_wires.py`: a wire plane in front of a mirror. The
  outcomes (absorbed on a wire, detected on either wall) match a float64 Monte
  Carlo of the same photons within 5 standard errors on 300K photons, and no
  photon enters a wire. Rays aimed obliquely at wall points inside a wire all
  hit the wire first.
- `tests/integration/test_coincident_faces.py`: a glass cylinder stands
  against the inside of a mirror box, its end in the plane of the box face.
  None of 16K photons sent at that face leaves the box (with `legacy`, more
  than 1% do).

### What remains

About 1.5 photons in a million still end in steel with a non-finite weight,
none of them detected. The causes are float32 limits shared with CUDA Chroma:
a Rayleigh scatter within ~1e-4 mm of a steel surface whose rounded position
lies inside it, or a reflection exactly at the edge of an end wire that
chroma-lar places half inside a wall.

## Speed of each difference

These rows were measured with an earlier TriChroma version (65M photons/s on
this workload, against 96M today): the point source unweighted, 30M photons
prepared in NumPy, one `simulate()` call, on an A100.

| Configuration | Photons/s | Speed-up | Steps per photon |
|---|---|---|---|
| CUDA Chroma (fastest of six calls) | 1.77M | 1 | 16.4 |
| TriChroma `legacy`, wavefront scheduler | 12.1M | 6.8 | 16.4 |
| + fixed wires | 28.9M | 16 | 17.1 |
| + every fix (default physics) | 32.1M | 18 | 17.1 |
| fused kernel, `legacy` | 6.7M | 3.8 | 16.4 |
| fused kernel, fixed wires, `legacy` otherwise | 62.3M | 35 | 17.1 |
| **fused kernel, default** | **65.4M** | **37** | 17.1 |

- **Random numbers, geometry, scheduling:** 6.8 times at equal physics.
- **Fixed wires:** another 2.4 times. CUDA Chroma's algorithm tests every
  wire between the ray's crossing points of the plane, up to all 1,967 wires
  of a plane for a grazing ray; the fixed one tests only the wires within
  reach.
- **Fused kernel:** another 2.0 times with the fixed wires, but slower with
  CUDA Chroma's: their long, uneven loop stalls every thread of a warp.
- **Other fixes:** no measurable cost.
- **CUDA Chroma's rate depends on the CPU.** While more than 65,536 photons
  are queued, each launch advances one step and is followed by a host
  synchronisation. Six calls on the same photons took 17 to 84 s on a busy node.
- **Exact mode** is 14 times slower than CUDA Chroma. It exists to show
  equality, not to run productions.
- **Weighted mode** (30M photons): 0.55M photons/s for CUDA Chroma, 11.7M
  for TriChroma (now 21M), 23M with `roulette=0.05` (now 41M).

## Other findings

- **NaN weights in CUDA Chroma.** In weighted mode, CUDA Chroma leaves 52.6%
  of the point source's photons with a NaN weight, and 0.14% of its hits carry
  one: photons put in steel by W2, W6 and G4, where `exp(-0/0)` is NaN. Their
  true weight is 0, so dropping such hits (as chroma-lar's `generate_lut.py`
  does) is correct.
- **The point source's polarization.** `generate_lut.py` gives every photon
  polarization +x whatever its direction, which is not transverse. Both codes
  use it as given.
- **Forced detection.** In weighted mode both codes end a photon at a
  detecting surface with weight × detect and drop the rest of its weight.
  That is exact when detect = 1, as on the LAr TPC's photocathodes, but
  biased for a surface that also reflects.
