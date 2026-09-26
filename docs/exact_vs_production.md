# Exact mode and production mode: every difference

TriChroma runs `chroma.sim.Simulation` in one of two modes.

- **Exact mode** (`CHROMA_TRITON_TAPE=replay:<dir>`) replays a CUDA Chroma
  run recorded on tape: its random numbers, arithmetic, BVH, wire code and
  launch schedule. Its outputs equal the recorded run's word for word. Flat
  hits come in photon order, because CUDA Chroma's order changes from run to
  run. It replays recorded runs only ([bitwise mode](bitwise_mode.md)).
- **Production mode** (the default) is the fast engine.

This page lists every difference between the two that can change an output,
found by a line-by-line audit of both code paths. For each one it says what
it does to the results, whether it can bias a measurement and on what
evidence, and what it contributes to speed. The measurements use the weighted
mode (`use_weights=True`) of the LUT workloads on chroma-lar's reflect3wires
detector. "CUDA Chroma" is the installed CUDA code that exact mode
reproduces.

## Summary

Production mode computes CUDA Chroma's physics model. It differs in three
ways.

1. **Random numbers, geometry structures, arithmetic, scheduling.** These are
   Philox instead of XORWOW, SAH trees with instancing and analytic boxes
   instead of Chroma's BVH, Triton instead of nvcc arithmetic, and a fused
   persistent kernel. They change which random realisation a run produces,
   not the distributions.
   - With CUDA Chroma's physics switched back on (`CHROMA_TRITON_FIXES=0`)
     and the wire planes removed, the two engines agree on 40M weighted
     photons each per workload. Detected weight: +0.035% ± 0.073% and
     -0.055% ± 0.071%. Channel spectra: χ² 89/81 and 72/81. Time spectra:
     99/101 and 118/101. History bits: within 1.2σ.
   - The wires are removed because their CUDA defect is float32 rounding
     noise, which other arithmetic cannot reproduce.
   - Ray by ray, the nearest mesh boundary is the same triangle as CUDA
     Chroma's for all but 8.5e-5 of 2M rays from real photon histories. The
     exceptions are edge ties, plus two rays at a coincident PMT/wall face.
2. **Corrections of CUDA Chroma defects**, on by default. They change results
   on purpose: +17.5% ± 0.07% detected light on the LUT fixture, and +16.1%
   ± 0.07% for photons spread over the detector. Each correction is
   checked against a float64 or analytic reference. `CHROMA_TRITON_FIXES=0`
   restores all but the specular-direction formula.
   - **Wires** (three defects): +18.0%.
   - **Photons leaking through coincident PMT and wall faces:** +0.08%.
   - **Polarization after a specular reflection:** -0.37% ± 0.14%.
   - **Fresnel reflection and specular direction at normal incidence:** not
     measurable.
3. **Opt-in Russian roulette** (`CHROMA_TRITON_ROULETTE`): unbiased (checked
   to ±0.006%), twice as fast, and noisier only in the late tail of the time
   spectrum.

**Speed.** On the weighted LUT fixture (10M photons, A100), CUDA Chroma
takes 15-17 s and production 0.91 s (0.46 s with roulette), although
production follows each photon for 47% more steps. Per difference:
- The engine alone (Philox, the geometry structures, the scheduler) is
  3.2 times faster at equal physics.
- The corrected wires cut the cost of a step by a factor of 3.2.
- The fused kernel is another 2.5 times faster.
- The other corrections cost at most 4%.
- Exact mode is 10 times slower than CUDA Chroma itself.

Section 4 has the details.

## 1. The differences

The class says what kind of difference an entry is:

- **(a)** Only another random realisation, or last-bit rounding: same distributions.
- **(b)** A deliberate correction of a CUDA Chroma defect.
- **(c)** Output format or API semantics only.
- **(d)** An edge case where one or both modes are undefined or look unintended.

The "Restored by" column names the setting that brings back CUDA Chroma's
behaviour in production mode. The "Weighted LUT" column gives the measured
effect on the LUT fixture (section 3).

### Random numbers

| ID | Exact mode (CUDA Chroma) | Production | Restored by | Weighted LUT | Class |
|---|---|---|---|---|---|
| R1 | XORWOW stream of the thread slot a photon lands in | Philox-4x32-10: key = seed, counter = (photon id, steps done, block) | none | another realisation; same distributions (3.2) | (a) |
| R2 | draws taken one after another in call order | a block of four uniforms per kind of event (transport, surface, WLS, re-emission, each diffuse trial, roulette) | none | same | (a) |
| R3 | `curand_uniform`: (0, 1], steps of 2^-32 near 0 | 2^23 values from 2^-24 to 1 - 2^-24 | none | distances cut at 16.6 attenuation lengths (probability 6e-8); probabilities quantised to 2^-23 | (a) |
| R4 | 64-bit seed | seed modulo 2^32 | none | seeds equal modulo 2^32 give the same run | (a) |
| R5 | a photon's numbers depend on queue order, chunking, `nthreads_per_block`, `max_blocks`, tracking | only on (seed, photon id, step) | none | production runs are repeatable and do not depend on batch size | (a) |
| R6 | no photon id | ids number the photons of all `simulate` calls of a `Simulation` in input order; the pipeline reads one batch ahead | `CHROMA_TRITON_PIPELINE=0` for the read-ahead | another realisation | (a) |
| R7 | DAQ continues the transport streams | Philox counters 0x7FFFFF00 + k, never reached by transport | none | DAQ only | (a) |

Every event takes its uniforms from its own Philox counter (photon id, step,
block). No uniform is used twice, and none is shared between photons.

### Arithmetic

| ID | Exact mode | Production | Restored by | Weighted LUT | Class |
|---|---|---|---|---|---|
| A1 | nvcc fast-math instructions, reproduced exactly | Triton/LLVM float32 (its own FMA contraction, `libdevice` functions) | none | last bits; trajectories diverge once a branch decision flips | (a) |
| A2 | normalisation by three divisions, trig Fresnel, `rotate` | multiplication by 1/sqrt, algebraic Fresnel, mirror formula | Fresnel: `FIXES=0` (formula, not bits) | last bits | (a) |
| A3 | reflected and refracted directions from `acosf` and a rotation; within 2.4e-4 rad of normal incidence the angle rounds to 0 | vector formulas | Fresnel: `FIXES=0` | CUDA Chroma's directions off by up to 2.4e-4 rad, near normal incidence only | (a)/(b) |

### Geometry

| ID | Exact mode | Production | Restored by | Weighted LUT | Class |
|---|---|---|---|---|---|
| G1 | Chroma's BVH (16-bit quantised boxes), stack traversal; the first triangle found wins exact ties | binned-SAH trees in eight octant copies, stackless, a top-level tree over instances | none | edge ties only (3.4) | (a) |
| G2 | triangle test rejects abs(det) < FLT_EPSILON (an absolute cut) and compares in double | rejects det = 0 only | none | none: the cut only misses near-grazing hits on sub-mm triangles | (b)/(d) |
| G3 | instance triangles in float32 world coordinates | intersected in the mesh's frame with a float32 ray transform | none | hit points agree to 1.2e-3 mm (99.99%), normals to 2.4e-7 (3.4) | (a) |
| G4 | a box face is two BVH triangles; a photon crossing it, or reflected at a point that rounds to the far side, can hit the other triangle at t ~ 1e-5 mm and end up inside the solid | up to 16 axis-aligned box solids are tested analytically, crossing face only | none | 8 photons in 10^6 end in the cathode's steel in CUDA Chroma (3.2) | (b) |
| G5 | 1000-entry traversal stack (an overflow breaks the traversal) | stackless | none | none (exact mode refuses such trees) | (d) |
| G6 | material and surface indices packed in 8 bits (128 and above read wrong table rows) | int32 indices | none | none (few materials) | (b) |
| G7 | zero-area triangle: NaN normal, photon aborted | zero normal, photon passes | none | none | (d) |
| G8 | the material of a bulk event is always that of the next boundary | the wavefront grid shortcut takes it from a grid cell whose material is certified by sampling rays | `CHROMA_TRITON_GRID=0`; the default fused kernel never uses the grid | none (3.3) | (d) |
| G9 | a photon that leaves a solid through a face lying in a box face's plane is on that plane; the box face then needs t > 1e-6, so rounding lets the photon through | the box face is met at distance 0 when the photon is on its plane within a few ulp and has not just met it | `FIXES=0` | chroma-lar's PMT backs lie in the TPC walls: 0.5% of the photons leave the TPC in CUDA Chroma; +0.08% detected light (3.3) | (b) |

### Wires

The three CUDA Chroma defects lose light. Each ends with a photon inside a
wire's steel (absorption and scattering length 0), where it is absorbed or
its weight becomes 0 or NaN (`exp(-0/0)`).

| ID | Exact mode | Production | Restored by | Weighted LUT | Class |
|---|---|---|---|---|---|
| W1 | discriminant `B*B - A*C`: the two terms are ~t^2 and differ by ~r^2, so float32 rounding decides hits beyond a few hundred mm | `A*r*r - (wv*dn - wn0*dv)^2`, the same quantity without cancellation | `CHROMA_TRITON_LEGACY_WIRES=1` or `FIXES=0` | W1-W3 together: +18.0% ± 0.1% detected light (3.3) | (b) |
| W2 | the float32 reflection point often lands inside the wire; the next query takes the exit for a hit from inside and puts the photon in the steel | the next query skips the wire the photon has just left outward | same | (in W1) | (b) |
| W3 | hit point and normal from the noisy root | from the same exact decomposition | same | (in W1) | (b) |
| W6 | a plane is culled when the distance to the plane of the wire axes exceeds the mesh hit plus one radius; the ray enters the wires r/abs(dn) earlier, so an oblique ray ending on a wall the wires pass through reaches wall points inside a wire | culled only when the slab entry is beyond the mesh hit | same | +0.0016% ± 0.0002% (1568 photons in 10M no longer lost) | (b) |
| W4 | CUDA Chroma's float32 bits | `LEGACY_WIRES=1` is CUDA Chroma's algorithm in Triton arithmetic: its far hits are decided by other rounding noise | none | legacy and CUDA wire answers differ on 0.73% of rays in each direction (3.4) | (a)/(d) |
| W5 | a NaN wire distance (ray exactly parallel to the wire axes) is accepted and corrupts the nearest-wire choice | rejected | none | none | (d) |

### Surface and bulk physics

| ID | Exact mode | Production | Restored by | Weighted LUT | Class |
|---|---|---|---|---|---|
| S1 | Fresnel within 2.4e-4 rad of normal incidence: 0/0 = NaN coefficients, so the photon is always transmitted | angle-free coefficients (4% reflected for 1.0/1.5) | `FIXES=0` | not measurable (~1e-7 of crossings) | (b) |
| S2 | Fresnel elsewhere: trig formulas | algebraic; same branches and draws | `FIXES=0` | last bits | (a) |
| S3 | specular direction by rotating the normal: NaN, and the photon is aborted, when it is exactly anti-parallel to the normal | mirror formula `d - 2(d.n)n` | none (`FIXES=0` keeps the mirror formula) | none (isotropic photons) | (b) |
| S4 | polarization unchanged by a specular reflection, so it stops being transverse | mirrored, `p - 2(p.n)n` | `FIXES=0` | -0.37% ± 0.14% detected light, and a shift of the time spectrum (Rayleigh scattering follows the polarization) | (b) |
| S5 | complex, dichroic and angular surface models | `NotImplementedError` | none | not used | (c) |
| B1 | a property lookup at the top wavelength of the grid reads one word past the table | clamped | none | none (128 nm) | (b) |
| B2 | absorption, scattering, re-emission, Rayleigh | the same algorithms | - | only through R1-R3 and A1 | (a) |

### Weights

The weighted-mode rules are identical in both modes:
- the 1e-4 weight threshold;
- bulk absorption replaced by `exp(-d/L)`;
- surface absorption folded into the weight;
- forced detection with weight × detect on every detecting surface.

| ID | Exact mode | Production | Restored by | Weighted LUT | Class |
|---|---|---|---|---|---|
| WT1 | none | opt-in Russian roulette (`CHROMA_TRITON_ROULETTE=w`): below weight w a photon survives with probability weight/w at weight w, else ends (flag bit 29) | off by default | +0.003% ± 0.006% (unbiased); fewer, heavier hits | (b)/(c) |

### Flags, outputs, DAQ, scheduling

| ID | Exact mode | Production | Restored by | Class |
|---|---|---|---|---|
| H1 | NaN abort flagged with bit 15 | bit 31 (`chroma.event.NAN_ABORT`) | `FIXES=0` | (c) |
| H2 | photon history kept in 16 bits (input flag bits 16-31 are dropped) | 32 bits | none | (c) |
| H3 | input photons with bits 0-3 or 15 set are not propagated | bits 0-3, 29 or 31 | none | (c)/(d) |
| N1 | direction and polarization renormalised at every launch (unweighted runs launch every step while the queue is long) | once, on entry | none; weighted runs use one launch, so no difference | (a) |
| O1 | `use_packed=True`: `photons_end` holds the initial positions and directions, and the DAQ sees initial weights when no hits were extracted | final states | none | (b) |
| O2 | `photon_tracks` include photons already finished on input | omitted | none | (c) |
| O3 | accepts optical tables production rejects (probabilities outside [0, 1], unnormalised CDFs, ...) | validates and refuses them | none | (c) |
| D1 | DAQ time and charge CDFs clamped at their ends | the same (fixed with this audit) | - | - |
| D2 | `Detector._pdf_to_cdf` builds a CDF one entry short, and both modes read past it | the same bug (in Chroma's API) | - | (d) |
| D3 | a negative charge sample adds nothing | the same (fixed with this audit) | - | - |
| M1 | `max_steps <= 0`: nothing is propagated | the same (fixed with this audit; the fused kernel took one step) | - | - |
| SC1 | reads a batch, propagates it, yields its events | pipelined: batch k+1 is read and propagated before batch k's events are yielded | `CHROMA_TRITON_PIPELINE=0` | (a)/(c) |
| SC2 | - | fused kernel, wavefront rounds and grid shortcut share the physics code and the Philox keys | `CHROMA_TRITON_FUSED`, `CHROMA_TRITON_GRID` | (a) (3.3) |

## 2. How production could be biased, and how that was tested

A difference can bias a measurement only if it changes a distribution.
- The (a) items change the realisation only.
- The (b) items change the physics on purpose.
- The (c) and (d) items do not arise in these workloads.

Section 3 tests each of these claims on the weighted workloads:

- **(a) items.** Production with CUDA Chroma's physics restored must agree
  with CUDA Chroma statistically (3.2), and ray by ray on the geometry (3.4).
  Scheduler variants must produce the same photons (3.3).
- **(b) items.** Each correction is measured on its own, with the same random
  numbers (3.3), and checked against a float64 or analytic reference (3.4,
  3.5).
- **Roulette.** Must leave every tally unchanged (3.3).

**Observables.** Each photon's detected weight (0 if not detected), summed in
total, per channel, and per detection-time bin (5 ns up to 400 ns, then
wider). Errors are per photon, since photons are independent.

**χ² tests.** Sparse bins are merged until both samples have 100 hits. A
roulette sample has few, heavy hits in the late tail, and an almost empty bin
has no usable variance estimate. A detected photon with a non-finite weight
counts as 0, as chroma-lar's `generate_lut.py` drops such hits.

## 3. Measurements

Setup: reflect3wires detector, 128 nm, weighted, `max_steps=1000`, 10M
photons per run, A100. The scripts are in `benchmarks/validation/`. Two
sources:

- **W1:** the LUT fixture of `generate_lut.py`: isotropic photons at
  (-450, 60, -120) mm, polarization +x.
- **W2:** photons uniform over the simulated half of the detector (x -2310..0,
  y and z ±2160 mm), isotropic, random transverse polarization.

### 3.1 Exact mode equals CUDA Chroma

A weighted CUDA Chroma run of the LUT fixture (200K photons, seed 5) was
recorded and replayed. `photons_end` is identical in every word. The flat
hits are identical as a set: CUDA Chroma's hit order changes from run to
run, so only the set is compared.

### 3.2 Production with CUDA Chroma's physics vs CUDA Chroma

`CHROMA_TRITON_FIXES=0` restores CUDA Chroma's polarization, Fresnel
formulas, wire algorithm and box-face threshold. What remains are the (a)
items. Detected weight per photon, pooled over seeds; the difference is CUDA
minus production:

| Workload | Seeds x 10M | CUDA Chroma | Production `FIXES=0` | Difference | Channels χ²/dof | Time χ²/dof | History bits |
|---|---|---|---|---|---|---|---|
| W1 no wires | 4 | 0.043891 | 0.043876 | +0.035% ± 0.073% | 88.6/81 | 99.4/101 | within 1.0σ |
| W2 no wires | 4 | 0.052468 | 0.052497 | -0.055% ± 0.071% | 71.5/81 | 118.0/101 | within 1.2σ |
| W1 | 6 | 0.031450 | 0.031417 | +0.106% ± 0.072% | 71.5/81 | 103.2/97 | within 1.4σ |
| W2 | 6 | 0.038308 | 0.038260 | +0.124% ± 0.070% | 99.2/81 | 91.7/97 | within 1.8σ |

Seed-to-seed scatter is statistical in both engines: χ² per degree of
freedom of 3.1/3 to 8.9/5, with no significant seed dependence of CUDA
Chroma.

Two things differ, both known:

- **How photons end.** In bug-compatible mode, 0.2% fewer photons are
  detected (at any weight) and more end on the world box. Both leak photons
  through the PMT backs that lie in the TPC walls (G9). Without the wires
  that is 0.96% of the photons in CUDA Chroma and 1.33% in bug-compatible
  production; with them, 0.54% and 0.74%. Either rate is decided by whether
  rounding puts a photon on, beyond or inside the wall plane.
- **The wire defect.** With the wires, the small offset in detected weight
  (+0.1%, 2.3σ combined) is W4: CUDA Chroma's wire defect is float32 noise,
  and Triton arithmetic reproduces it only statistically (3.4).

Production mode corrects both (G9, and W1-W6), so neither affects its
results.

With all corrections (the default), production detects more light than CUDA
Chroma. Six seeds of 10M photons per engine:

| Workload | CUDA Chroma | Production | Difference |
|---|---|---|---|
| W1 (LUT fixture) | 0.031450 | 0.036965 | +17.53% ± 0.07% |
| W2 | 0.038308 | 0.044471 | +16.09% ± 0.07% |

The corrections account for all of it (3.3).

### 3.3 Each change on its own, with the same random numbers

Same seed and photon ids: every photon gets the same Philox numbers, so
differences are paired. Detected weight difference, relative to the second
run:

| Change | W1 | W2 | Photons changed (W1) | Channels χ²/dof (W1) | Time χ²/dof (W1) |
|---|---|---|---|---|---|
| polarization, Fresnel, NaN bit (S1, S2, S4, H1) | -0.366% ± 0.141% | -0.268% ± 0.120% | 3.57M | 95.6/81 | 193.6/93 |
| wires (W1-W3, W6) | +17.96% ± 0.10% | +16.27% ± 0.09% | 3.87M | 30643/81 | 112415/93 |
| box faces (G9) | +0.082% ± 0.002% | +0.082% ± 0.002% | 21,475 | 1868/81 | 7349/93 |
| roulette 0.05 (must be 0) | +0.003% ± 0.006% | +0.003% ± 0.005% | 2.34M | 88.0/81 | 86.9/81 |
| wavefront instead of fused (must be 0) | +0.0001% ± 0.0001% | -0.0001% ± 0.0002% | 56 | 47.2/43 | 40.7/43 |
| grid shortcut (must be 0) | -0.0004% ± 0.0003% | -0.0006% ± 0.0003% | 97 | 63.1/63 | 75.0/64 |
| wavefront instead of fused, `FIXES=0` (must be 0) | 0.0000% ± 0.0002% | +0.0002% ± 0.0002% | 27 | 25.0/25 | 23.5/24 |

The rows above compare runs made before and after the box-face correction
(G9), and they are all repeated with the final code:

- All corrections together: +17.62% ± 0.15% (W1) and +16.06% ± 0.13% (W2).
- Roulette: +0.003% ± 0.006% and +0.004% ± 0.005%. Channel χ² 89.6/81 and
  85.5/81; time χ² 83.2/83 and 76.3/84.
- Wavefront and grid: they change 57/68 and 99/128 photons, with every
  χ² consistent.
- `FIXES=0`: bitwise identical before and after.

The schedulers change a few dozen photons in 10M: an FMA contraction that
differs between kernels flips a branch decision. Roulette changes 2.3M
photons but no tally.

Roulette trades variance for speed only in the late tail of the time
spectrum. Per-photon variance, roulette over default:

| Observable | Share of the light | Variance ratio |
|---|---|---|
| total, per channel | 100% | 1.00 |
| times < 100 ns | 85-90% | 1.00 |
| 100-200 ns | 9-14% | 1.06-1.09 |
| 200-300 ns | 1-1.5% | 2.5-2.8 |
| 300-400 ns | 0.1-0.2% | 12-13 |
| > 400 ns | < 0.05% | ~80 |

### 3.4 Geometry, ray by ray

The rays are those of real histories: every step of 8000 W1 and 8000 W2
photons, 1.96M rays in all. Both geometries answer each ray: production's
query, and CUDA Chroma's BVH and wire code as reproduced by the exact mode
from the recorded scene.

- **Meshes.** 1,718,878 rays hit the same triangle in both. Distances agree
  to 1.2e-3 mm for 99.99% of them (float32 at metre scale; max 0.037 mm);
  normals agree to 2.4e-7; materials and surfaces map one to one. 166 rays
  (8.5e-5) hit different triangles:
  - 164 are ties within 1e-3 mm at shared edges or vertices;
  - 2 are coincident PMT/wall faces 1-10 µm apart (G9).
  In the 2M rays, no mesh is missed by one engine and hit by the other.
- **Wires.** 226,742 rays involve a wire in either engine (rays leaving a
  wire are left out; see 3.5). The engines disagree on 221,032 of them, and
  float64 arithmetic decides each case:

  | Rays where the engines disagree | Production | CUDA Chroma |
  |---|---|---|
  | matches float64 | 221,003 (99.987%) | 2,589 (1.2%) |
  | wire hit where float64 has none | 17 | 117,745 (closest approach up to 25.5 radii) |
  | misses a float64 wire hit | 9 | 32,698 |
  | hits the wrong wire | 1 | 7,464 |
  | right wire, distance off | 2 | 60,536 (median 0.056 mm, max 2.9 mm) |

  Production's 29 errors all lie within 0.2% of tangency (closest approach
  0.998-1.0015 r), in both directions: float32 noise at grazing incidence.
  On 5,710 sampled rays where the engines agree, both match float64.
- **Legacy wires.** With `CHROMA_TRITON_LEGACY_WIRES=1` (1.33M rays), 98.5%
  of the answers equal CUDA Chroma's. The legacy wires see a wire where CUDA
  Chroma does not on 0.73% of rays, and the reverse on 0.73%: the same
  defect with other rounding noise (W4).

### 3.5 Wires and box faces against float64 or exact expectations

`tests/integration/test_wires.py`:
- A wire plane at x = 2160 mm (75 µm wires, 20% absorbing, 80% specular)
  facing a mirror 5 mm behind it. The production engine's outcomes (absorbed
  on a wire, detected on either wall) match a float64 Monte Carlo of the same
  photons within 5 standard errors on 300K photons. No photon enters a wire.
  With legacy wires, photons do enter.
- Rays aimed at wall points inside a wire, obliquely, from outside the
  wires: all hit the wire first. With CUDA Chroma's cull, more than 10% reach
  the wall.

`tests/integration/test_coincident_faces.py`: a glass cylinder stands
against the inside of a mirror box at x = -2295.4148 mm, with its end in the
plane of the box face. Of 16K photons sent from the glass to that face, none
leaves the box. With `FIXES=0`, more than 1% do.

### 3.6 What remains: float32 limits

On the LUT fixture, 11-18 photons in 10M (~1.5e-6) still end in steel with
a non-finite weight, and none of them is detected. The traces show two
causes:
- a Rayleigh scatter within ~1e-4 mm of a steel surface (cathode or wire)
  whose rounded position lies inside it;
- a wall reflection exactly at the edge of an end wire that chroma-lar places
  half inside the wall.

CUDA Chroma has the same limits. They bound any remaining bias at ~1e-6.

## 4. Speed

The weighted LUT fixture (W1) with 10M photons in one `simulate()` call. The
times are warm (the last of four calls), measured on an A100 that no other
job used during the measurement; the node's CPUs were shared (load ~17).
Production's step count is read from the engine. CUDA Chroma's is not
reported, but its physics is that of the `FIXES=0` rows (84.7 steps).

| Configuration | Time | Photons/s | Steps/photon | ns/step | Speed-up |
|---|---|---|---|---|---|
| CUDA Chroma | 15.2 s (17.4 s in an earlier run) | 0.66M | ~85 | ~18 | 1 |
| production engine, CUDA Chroma's physics (`FIXES=0`), wavefront scheduler | 4.75 s | 2.1M | 84.7 | 5.6 | 3.2 |
| + production wires (`FIXES=0 LEGACY_WIRES=0`) | 2.16 s | 4.6M | 123.7 | 1.75 | 7.0 |
| + every correction (default physics) | 2.25 s | 4.4M | 124.6 | 1.8 | 6.8 |
| + grid bulk shortcut (`CHROMA_TRITON_FUSED=0`) | 2.62 s | 3.8M | 124.6 | 2.1 | 5.8 |
| fused kernel, CUDA Chroma's physics (`FIXES=0`) | 7.58 s | 1.3M | 84.7 | 8.9 | 2.0 |
| fused kernel, corrections but legacy wires | 7.65 s | 1.3M | 85.2 | 9.0 | 2.0 |
| fused kernel, production wires, `FIXES=0` otherwise | 0.87 s | 11.5M | 123.7 | 0.70 | 17 |
| **fused kernel, default** | **0.91 s** | **11.0M** | 124.6 | 0.73 | **17** |
| fused kernel, default, `CHROMA_TRITON_ROULETTE=0.05` | 0.46 s | 21.7M | 56.3 | 0.82 | 33 |

At 200K photons: CUDA Chroma 0.15 s, production 0.04 s. Exact mode takes
1.59 s: 0.55 s reading the 299 MB tape, 0.92 s for the 348 rounds.

What each difference buys:

- **Random numbers, geometry structures, scheduler (R, G, SC):** 3.2x at
  equal physics (CUDA Chroma to the first production row). These were not
  timed separately.
- **Wires (W1-W3, W6):** the step cost falls from 5.6 to 1.75 ns, while
  photons live 46% longer because the wires no longer kill them. CUDA
  Chroma's algorithm tests every wire between the ray origin's position
  across the plane and the slab exit, up to all 1,967 wires of a plane for a
  ray at a grazing angle. Production tests only the wires the slab crossing
  can reach.
- **Polarization, Fresnel, box faces (S1-S4, G9):** 0% to 4% (2.16 s to
  2.25 s), mostly from the 0.7% more steps the extra light takes.
- **Fused kernel (SC2):** 2.5x with the production wires (2.25 s to 0.91 s),
  but 0.6x with CUDA Chroma's wires. Presumably their long, uneven loop holds
  up all 32 lanes of a warp, since the fused kernel steps a warp's photons
  together; this was not profiled.
- **Grid shortcut:** slower on this workload (2.62 s); the default fused
  kernel does not use it.
- **Roulette (WT1):** 2x, from half as many steps.
- **Exact mode:** reading the tape and CUDA Chroma's arithmetic make it
  10 times slower than CUDA Chroma. It exists to prove equality, not to run
  productions.

## 5. Other findings

- **CUDA Chroma's NaN weights.** In weighted mode CUDA Chroma leaves 52.6% of
  the LUT fixture's photons with a NaN weight, and 0.14% of its hits carry
  one. These are photons put in steel, in the wires by W2 and W6 and in the
  cathode by G4, where `exp(-0/0)` makes the weight NaN. `generate_lut.py`
  drops such hits, which is right: their true weight is 0. Production leaves
  ~1.5e-6 of the photons with a NaN weight (3.6) and no NaN hits on the
  fixture.
- **The LUT fixture's polarization.** `generate_lut.py` gives every photon
  polarization +x whatever its direction, so it is not transverse. Both
  engines use it as given; the macros draw transverse polarizations.
- **Forced detection.** Weighted mode (both engines) ends a photon at a
  detecting surface with weight × detect and drops the rest of its weight.
  On reflect3wires this is exact, since the photocathodes have detect = 1.
  It would bias results for a surface that also reflects.
- **Seed dependence.** CUDA Chroma's seed dependence, seen earlier in
  unweighted runs, does not appear in these weighted runs (one launch per
  run).
- **A bug shared by both modes.** `Detector._pdf_to_cdf` (`set_time_dist*`,
  `set_charge_dist*`) builds its CDF one entry short (D2). It is in Chroma's
  API and left as is.
