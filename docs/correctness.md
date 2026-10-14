# Correctness

TriChroma's results differ from CUDA Chroma's: other random numbers, and some
CUDA Chroma bugs fixed. So how do we know it simulates the same physics?
Three checks address three questions.

## Is the physics code the same?

TriChroma has an *exact mode* that replays a recorded CUDA Chroma run with
the same random numbers, the same float32 rounding, the same geometry tree
and the same order of work. Its outputs (final photons, hits, DAQ channels,
photon tracks) match the CUDA run in every bit, on eleven test runs that
together cover every surface model, re-emission, scattering, wire planes,
weighted mode, packed mode and photon tracking.

So TriChroma implements CUDA Chroma's physics exactly, bugs included.
[Exact mode](reference/exact_mode.md) describes the recording and the
replay.

## Do the distributions agree?

The fast engine uses other random numbers, so it is compared statistically.
With CUDA Chroma's bugs switched back on (`CHROMA_TRITON=legacy`), both
engines simulated 40M photons each on chroma-lar's liquid-argon TPC, from a
point source and from photons spread through the detector, in weighted mode:

- **Detected light** differs by +0.035% ± 0.073% and -0.055% ± 0.071%.
- **Light per PMT and arrival times** agree: χ² per degree of freedom close
  to 1.
- **How photons end** (absorbed, scattered, reflected, detected) agrees
  within 1.2 standard errors.

The wire planes were removed for this comparison. CUDA Chroma's wire bug is
rounding noise, which no other arithmetic reproduces exactly; with the wires,
a 0.1% offset remains.

The geometry was also compared ray by ray, on 2 million rays taken from real
photon paths. The nearest triangle is the same in both engines for all but
0.0085% of the rays, and almost all of those are ties at edges shared by two
triangles.

## Is each fix right?

The fixes change results on purpose. Each was measured on its own, with the
same random numbers before and after, and checked against an independent
calculation:

| Fix | Detected light on the LAr TPC | Checked against |
|---|---|---|
| Wire intersection (three bugs) | +18.0% | float64 geometry: on the rays where the two results differ, TriChroma's is correct for 99.987%, CUDA Chroma's for 1.2% |
| Faces lying in the same plane | +0.08% | a test where no photon may leave a box through such a face |
| Polarization after a mirror reflection | -0.37% | the reflection law |
| Fresnel reflection at normal incidence | too small to measure | the Fresnel formulas |
| Box faces closer than 1e-6 mm | too small to measure | 4.66M adversarial rays against float64: 808 wrong answers before, none after |
| DAQ tables one entry short | DAQ output only | the tables Chroma meant to build |

Together they account for the whole difference from CUDA Chroma: on this
detector the detected light is 17.5% higher with TriChroma for the point
source, and 16% higher for photons spread through the detector.

## What is left

About one photon in a million still ends inside a steel surface, for
example after scattering within 0.0001 mm of it, where float32 positions run
out of precision. CUDA Chroma has the same limit.

Russian roulette (`CHROMA_TRITON=roulette=<w>`, off by default) leaves every
result unchanged within 0.006%. Its extra noise is confined to the few
photons that arrive after 200 ns.

## Checking for yourself

- `python -m pytest` runs the unit, integration and exact-mode tests.
- `python -m trichroma.tape.verify all` records CUDA Chroma runs and
  compares the replay bit for bit ([exact mode](reference/exact_mode.md#checking-equality)).
- `benchmarks/validation/` contains the statistical study above.

[Every difference](reference/differences.md) lists the full catalogue and
the measurements.
