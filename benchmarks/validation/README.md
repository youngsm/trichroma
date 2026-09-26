# Weighted-mode validation of the production engine

The scripts behind [docs/exact_vs_production.md](../../docs/exact_vs_production.md).
They need chroma-lar (for the reflect3wires detector) on `PYTHONPATH`, and
PyCUDA for the CUDA Chroma runs.

## Statistics: CUDA Chroma and production variants

One run is one weighted `simulate()` call with 10M photons; every photon's
end state is saved. `W1` is the LUT fixture of chroma-lar's `generate_lut.py`,
`W2` photons spread over the simulated half detector; `W1nw` and `W2nw` are
the same without the wire planes.

```bash
run() {  # TAG WORKLOAD SEED [VAR=VALUE ...]
  tag=$1 w=$2 seed=$3; shift 3
  env "$@" python benchmarks/validation/weighted_run.py $w $seed 10000000 data/${tag}_${w}_s${seed}.npz
}
for w in W1 W2 W1nw W2nw; do for seed in 1 2 3 4 5 6; do
  run C_cuda $w $seed CHROMA_BACKEND=cuda
  run P0_fixes0 $w $seed CHROMA_BACKEND=triton CHROMA_TRITON_FIXES=0
  run P2_default $w $seed CHROMA_BACKEND=triton
done; done
for w in W1 W2; do  # same seed, same random numbers: paired differences
  run P1_fixes_legacywires $w 1 CHROMA_BACKEND=triton CHROMA_TRITON_LEGACY_WIRES=1
  run P3_roulette $w 1 CHROMA_BACKEND=triton CHROMA_TRITON_ROULETTE=0.05
  run P4_wavefront $w 1 CHROMA_BACKEND=triton CHROMA_TRITON_FUSED=0 CHROMA_TRITON_GRID=0
  run P5_wavefront_grid $w 1 CHROMA_BACKEND=triton CHROMA_TRITON_FUSED=0
  run P6_wavefront_fixes0 $w 1 CHROMA_BACKEND=triton CHROMA_TRITON_FUSED=0 CHROMA_TRITON_GRID=0 CHROMA_TRITON_FIXES=0
done
python benchmarks/validation/weighted_stats.py data
```

## Geometry, ray by ray

`geometry_rays.py` collects the rays of production photon histories and asks
both geometries for the nearest boundary: production's, and CUDA Chroma's BVH
and wire code as reproduced by the exact mode from a recorded scene (record
any reflect3wires run with `CHROMA_BACKEND=cuda CHROMA_TRITON_TAPE=record:<dir>`).
`wires_float64.py` then decides every ray on which they disagree about a wire
with float64 arithmetic.

```bash
CHROMA_BACKEND=triton CHROMA_TRITON_FUSED=0 python benchmarks/validation/geometry_rays.py TAPE_DIR 8000 geometry.npz
python benchmarks/validation/geometry_stats.py geometry.npz
python benchmarks/validation/wires_float64.py geometry.npz
```
