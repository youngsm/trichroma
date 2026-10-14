# Exact mode

Exact mode replays a recorded CUDA Chroma run inside TriChroma and produces
the same output, bit for bit: the same random numbers, the same float32
rounding, the same geometry tree and the same order of work, bugs included.
Its purpose is to show that TriChroma's physics code is CUDA Chroma's. It is
slower than CUDA Chroma and needs a recording, so it is not meant for
production runs.

## Why a recording is needed

CUDA Chroma runs do not repeat exactly. When a batch needs more than one
kernel launch, the surviving photons are re-queued in whatever order the GPU's
atomic operations produce, and a photon's queue position selects its
random-number stream. Two runs of the same script differ (in one test, in
96,391 of 100,000 photons).

So equality is shown against a recording, which contains every random
number each photon used and the order of work that actually happened.

## Use

The same script runs twice, once per backend, with the same inputs and the
same `Simulation` arguments:

```bash
# 1. record, in an environment with PyCUDA
CHROMA_BACKEND=cuda   CHROMA_TRITON_TAPE=record:/path/to/tape python my_script.py
# 2. replay in TriChroma
CHROMA_BACKEND=triton CHROMA_TRITON_TAPE=replay:/path/to/tape python my_script.py
```

Then `trichroma.utils.compare_events` compares the two runs' events bit for
bit. Recording leaves CUDA Chroma's output unchanged.

`CHROMA_TRITON_TAPE` takes these values:

| Value | Backend | Effect |
|---|---|---|
| `record:<dir>` | CUDA | record the run to `<dir>` |
| `record-sorted:<dir>` | CUDA | the same, with surviving photons re-queued in ascending order |
| `canonical` | CUDA | no recording; surviving photons re-queued in ascending order, so CUDA runs repeat |
| `replay:<dir>` | TriChroma | replay the recording in `<dir>` |

Sorting the queue is one legal order of CUDA Chroma's unmodified kernels, so
a sorted run is still a genuine CUDA Chroma run.

## Checking equality

`python -m trichroma.tape.verify` automates the whole check:

```bash
# everything: record with CUDA Chroma, replay in TriChroma, compare every output
python -m trichroma.tape.verify all --fixture synthetic --work /scratch/tapes \
    --cuda-python <python with PyCUDA> --triton-python <python with TriChroma> --transparency

# or step by step
CHROMA_BACKEND=cuda CHROMA_TRITON_TAPE=record:/scratch/t \
    python -m trichroma.tape.verify run --fixture reflect3wires --run visible --output cuda.npz
CHROMA_BACKEND=triton python -m trichroma.tape.verify replay --fixture reflect3wires --run visible \
    --tape /scratch/t --report replay.json
python -m trichroma.tape.verify tape --tape /scratch/t --report report.json   # no detector needed
```

- `verify replay` compares every event output with the outputs in the
  recording and reports the first photon and field that differ.
- `verify tape` replays the recording with a simple host-driven loop instead
  of the full `Simulation`, and also checks that every photon used exactly
  its recorded random numbers.
- `--transparency` also runs CUDA Chroma without recording, to show that
  recording changes nothing.

### Results

On an A100 (CUDA Chroma compiled with CUDA 12.4, TriChroma on Triton 3.1),
each run below was recorded with CUDA Chroma and replayed in TriChroma, in
both the recorded and the sorted queue order. Every output was equal.

| Detector and run | Photons | Launches | Random numbers | Hits |
|---|---|---|---|---|
| LAr TPC, visible light | 100,000 | 9 | 6,113,660 | 3,838 |
| LAr TPC with TPB wavelength shifter, VUV light | 100,000 | 9 | 6,198,479 | 1,614 |
| pixel TPC, VUV light | 100,000 | 6 | 4,162,873 | 952 |
| synthetic, two batches | 200,000 | 4+4 | 8,123,179 | 6,716 |
| synthetic, weighted mode | 50,000 | 1 | 9,209,627 | 13,458 |
| synthetic, photon tracking | 4,000 | 60 | 154,361 | 120 |
| synthetic, packed mode | 80,000 | 3 | 3,243,394 | 2,593 |
| synthetic, small thread blocks (64 × 16) | 30,000 | 9 | 1,207,309 | 1,052 |
| synthetic, adversarial inputs | 32,000 | 1 | 1,341,297 | 2,186 |
| synthetic, tiny (test data) | 72 | 30 | 7,392 | 20 |
| synthetic, tiny packed (test data) | 72 | 1 | 2,348 | - |

The synthetic detector covers every surface model (default, wavelength
shifting with re-emission, dichroic, angular, complex with refraction and
forced detection), bulk re-emission, Rayleigh scattering, diffuse and
specular reflection, Fresnel reflection, analytic wires and NaN aborts.

Replaying 100,000 photons takes 0.3 s on the LAr TPC (CUDA Chroma's own run
takes 2.2-2.5 s). The first replay in a process adds about 0.5 s to load
kernels, and compiling them the first time takes about 40 s. Recording costs
about 0.3 s per 100,000-photon batch plus 2 s per `Simulation`; such a batch
takes 51 MB on disk (36 MB after `trichroma.tape.compact`).

## How the replay works

- **Random numbers** come from the recording instead of a generator. Each
  photon's numbers are stored in the order they were used, across all its
  launches.
- **Arithmetic** is CUDA Chroma's, instruction for instruction. CUDA
  Chroma's kernels are compiled with fast math, where the compiler fuses
  multiplications and additions depending on the whole kernel. Each of those
  fusions was read from the compiled machine code and written explicitly into
  `src/trichroma/engine/exact.py`, which therefore produces the same bits
  inside any Triton kernel (appendix C).
- **Geometry** is CUDA Chroma's: its BVH, its triangle test and its float32
  wire code, taken from the recorded scene.
- **Work order.** TriChroma's production scheduler runs one step per round.
  CUDA Chroma's launches are reproduced exactly: a photon alive at the end of
  a launch has run all of that launch's steps, and directions are
  renormalised at each recorded launch start, as in CUDA Chroma. After the
  run, the queue lengths at every launch start and every photon's count of
  random numbers are checked against the recording.
- **DAQ** results are replayed with the recorded random numbers and CUDA
  Chroma's integer arithmetic.
- **Packed mode** (`use_packed=True`) is reproduced too: like CUDA Chroma,
  `photons_end` then holds the initial photon words, and the DAQ is given the
  initial times and weights unless hits were extracted first.

## Limits

Where CUDA Chroma's behaviour is undefined or was not recorded, recording or
replay stops with an error:

- DAQ tables one entry short (built by `set_time_dist*` or
  `set_charge_dist*` before chroma-lite fixed them), which CUDA Chroma reads
  past their end;
- material or surface numbers of 128 or more, a BVH whose root is a leaf, a
  search that needs more than 1000 stack entries, dichroic or angular lookups
  at the last tabulated angle, unknown surface models;
- `scatter_first != 0`, device profiling (`CHROMA_DEVICE_PROFILE`);
- a replay with other `Simulation` arguments, inputs, detector or batches
  than recorded, or whose queue lengths or random-number counts differ from
  the recording;
- `canonical` with the TriChroma backend (there is nothing to replay).

Exact mode needs CUDA Chroma's BVH (`geometry.bvh`). A NumPy port of Chroma's
BVH builder is not bit-identical, so the recorded tree is used, after a check
that every other part of the scene matches the detector.

## Appendix A: recording format (version 1)

A recording is a directory:

| File | Contents |
|---|---|
| `manifest.json` | format and version; environment (Python, NumPy, PyCUDA, nvcc, GPU, driver, compiler options); SHA-256 of every Chroma source file; the command line; one entry per `Simulation` (seed, thread and block counts, flags, scene file, RNG state hashes) and one per batch (file, `simulate()` arguments, counts, array hashes) |
| `sim<S>_scene.npz` | everything CUDA Chroma uploaded for simulation `S`: wavelength grid, material and surface tables, re-emission tables, wire planes, vertices, triangles, material codes, BVH nodes, DAQ tables; `pad_<table>` holds the word after each table row in device memory |
| `sim<S>_batch<B>.npz` | one batch (below) |

Batch arrays (`N` photons, `E` events, `L` launches; random numbers are stored
as float32 bit patterns):

| Key | Type | Meaning |
|---|---|---|
| `in_pos`, `in_dir`, `in_pol` | f32 [N,3] | input photons, as copied to the GPU |
| `in_wavelengths`, `in_t`, `in_weights` | f32 [N] | |
| `in_last_hit_triangles` | i32 [N] | |
| `in_flags`, `in_evidx` | u32 [N] | |
| `event_bounds` | i64 [E+1] | photon range of every event |
| `draws`, `offsets` | u32, i64 [N+1] | random numbers: photon `i` used `draws[offsets[i]:offsets[i+1]]`, in order |
| `launch_starts`, `launch_nsteps`, `launch_nphotons` | i32 [L] | first step, step count and queue length of every launch |
| `chunk_*` | i32 | kernel chunks within launches |
| `queue`, `queue_offsets`, `queue_draws` | | the queue of every launch, and the random numbers each entry used |
| `daq_draws`, `daq_offsets` | u32, i64 [N+1] | DAQ random numbers per photon |
| `daq_event_ran` | u8 [E] | events with a DAQ readout |
| `final_*` | | GPU state after propagation |
| `end_*` | | `photons_end` as returned |
| `hits_*`, `hits_expected_*` | | `flat_hits` as returned (arbitrary order), and the same hits in photon order |
| `channels_t`, `channels_q`, `channels_flags`, `channels_hit` | [E, channels] | DAQ output |
| `track_*` | | `photon_tracks` |

`trichroma.tape.format` reads recordings (NumPy only):

```python
tape = Tape(dir)                 # checks format, version and array hashes
batch = tape.batch(i)
batch.draws(device), batch.offsets(device)
batch.inputs(), batch.final(), batch.fields("end")
```

## Appendix B: order of the random numbers in one step

How a recording is read depends on the order in which CUDA Chroma's kernel
uses random numbers. For one step of a live photon:

| Stage | Numbers used |
|---|---|
| NaN check, boundary search | 0 |
| absorption and scattering distances | 2 |
| bulk absorption with re-emission components: component, re-emission test | +2 |
| re-emitted: wavelength, time, direction (2), polarization (2) | +6 |
| Rayleigh scatter: angle, azimuth | +2 |
| surface, default model (drawn before any re-weighting) | 1 |
| surface, wavelength shifter: one; absorbed: re-emission test (+1), re-emitted: wavelength, direction, polarization (+5); reflected: reflection test (+1) | 1-7 |
| surface, dichroic or angular | 1 |
| surface, complex: none for forced detection in weighted mode, otherwise one; absorbed: detection test (+1); reflected: diffuse test (+1) | 0-2 |
| diffuse reflection: three per rejection trial, then polarization (2) | 3k+2 |
| specular reflection | 0 |
| Fresnel boundary: polarization choice, reflection test | 2 |

DAQ, once per event, for a photon detected on a channel: weight test (1);
if it passes, time (1) and charge (1).

CUDA Chroma behaviour that is reproduced deliberately: fast-math float32
throughout, the float32 wires, the property lookup one entry past a table at
the top of the wavelength grid, Fresnel NaN at normal incidence and at the
critical angle, the 16-bit photon history, the complex surface model's
thin-film arithmetic, and the DAQ's integer charge and time arithmetic.

## Appendix C: the exact-arithmetic library

`src/trichroma/engine/exact.py` provides Triton functions whose results equal
CUDA Chroma's machine arithmetic bit for bit, and which take their random
numbers as arguments.

**Method.** The CUDA compiler fuses a multiplication and an addition into
one instruction depending on the data flow of the whole kernel, so a direct
transcription of the CUDA source does not reproduce its bits. The compiled
`propagate` kernel was mapped back to its PTX, every fusion was written as an
explicit `fma.rn`, and every other operation got `.rn`, which is never fused.
Recompiling that explicit PTX reproduces the original machine code. The
functions use the same explicit operations through
`tl.inline_asm_elementwise`. The complex surface model's thin-film block is
included as extracted PTX (`_complex_ptx.py`, made by
`src/trichroma/tape/tools/ptx_extract.py`).

**Pitfalls** for anyone changing it:
- Triton's kernel cache is keyed on source text, not on the values of
  global constants, so the thin-film block's digest is part of a docstring
  and checked at import.
- Inline assembly must not declare registers named like LLVM's (`%f<N>`,
  `%r<N>`, `%p<N>`): a scoped declaration can shadow an operand's register.
- A function compiled alone can be lowered differently from the same code
  inside the full kernel, so unit probes are compared against the full
  kernel's lowering where the two differ.

`python -m trichroma.tape.probes generate` regenerates the reference values
used by `tests/bitwise/test_exact.py`.
