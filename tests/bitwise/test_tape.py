"""Native RNG tape: format round trip, scene checks and bitwise replays.

``tests/bitwise/data/legacy_tape_tiny`` and ``legacy_tape_tiny_packed`` were recorded
from the unmodified CUDA backend through ``chroma.sim.Simulation``
(``CHROMA_BACKEND=cuda CHROMA_TRITON_TAPE=record:<dir> python -m
trichroma.tape.verify run --fixture synthetic --run tiny`` /
``tiny_packed``) and compacted with ``trichroma.tape.format.compact``:
72 photons each; ``tiny`` with photon tracking, use_weights, 30 launches and
the DAQ; ``tiny_packed`` with use_packed and the DAQ reading W's initial
times (no hit extraction).

The GPU tests replay them through the public API (the Triton ``Simulation``,
i.e. ``ProductionEngine`` in exact mode) and through the reference loop. Set
``CHROMA_LEGACY_TAPE`` to replay another tape in the reference-loop test.
"""

import os

import numpy as np
import pytest

from trichroma.tape import format as tapefmt
from trichroma.tape.scene import (check_daq_limits, check_legacy_limits, compare_scene_relabeled,
                                        relabel_scene)

TINY = os.path.join(os.path.dirname(__file__), "data", "legacy_tape_tiny")
TINY_PACKED = os.path.join(os.path.dirname(__file__), "data", "legacy_tape_tiny_packed")


def _fake_batch(n=5, draws_per=(3, 0, 7, 1, 2)):
    rng = np.random.default_rng(0)
    offsets = np.concatenate([[0], np.cumsum(draws_per)]).astype(np.int64)
    arrays = dict(
        in_pos=rng.normal(size=(n, 3)).astype(np.float32), in_dir=rng.normal(size=(n, 3)).astype(np.float32),
        in_pol=rng.normal(size=(n, 3)).astype(np.float32), in_wavelengths=np.full(n, 400, np.float32),
        in_t=np.zeros(n, np.float32), in_last_hit_triangles=np.full(n, -1, np.int32),
        in_flags=np.zeros(n, np.uint32), in_weights=np.ones(n, np.float32), in_evidx=np.zeros(n, np.uint32),
        event_bounds=np.asarray([0, n], np.int64),
        draws=rng.random(int(offsets[-1])).astype(np.float32).view(np.uint32), offsets=offsets,
        launch_starts=np.zeros(1, np.int32), launch_nsteps=np.full(1, 10, np.int32),
        launch_nphotons=np.full(1, n, np.int32))
    return arrays


def test_tape_round_trip(tmp_path):
    writer = tapefmt.TapeWriter(str(tmp_path / "t"), dict(mode="record"))
    sim = writer.add_simulation(dict(seed=1, nthreads_per_block=64, max_blocks=4, photon_tracking=False,
                                     use_packed=False), scene=dict(vertices=np.zeros((3, 3), np.float32)))
    arrays = _fake_batch()
    writer.add_batch(sim, arrays, dict(params=dict(max_steps=10, use_weights=False), nphotons=5, nevents=1))
    tapefmt.compact(str(tmp_path / "t"))

    tape = tapefmt.Tape(str(tmp_path / "t"))
    assert len(tape) == 1 and tape.simulations[0]["seed"] == 1
    batch = tape.batch(0)
    assert batch.nphotons == 5 and batch.nevents == 1
    np.testing.assert_array_equal(batch.photon_draws(2).view(np.uint32),
                                  arrays["draws"][arrays["offsets"][2]:arrays["offsets"][3]])
    assert len(batch.photon_draws(1)) == 0

    replay = tapefmt.TapeReplay(str(tmp_path / "t"), simulation=0)
    replay.check_simulation(seed=1, nthreads_per_block=64, max_blocks=4)
    with pytest.raises(tapefmt.TapeError):
        replay.check_simulation(seed=2)
    photons = {k[3:]: v for k, v in arrays.items() if k.startswith("in_")}
    bad = dict(photons, wavelengths=np.full(5, 401, np.float32))
    with pytest.raises(tapefmt.TapeError):
        tapefmt.TapeReplay(str(tmp_path / "t"), simulation=0).next_batch(bad)
    assert replay.next_batch(photons, max_steps=10).index == 0
    with pytest.raises(tapefmt.TapeError):
        replay.next_batch(photons)


def test_tape_detects_corruption(tmp_path):
    writer = tapefmt.TapeWriter(str(tmp_path / "t"), dict(mode="record"))
    sim = writer.add_simulation(dict(seed=1), scene=None)
    writer.add_batch(sim, _fake_batch(), dict(params={}, nphotons=5, nevents=1))
    path = tmp_path / "t" / "sim0000_batch00000.npz"
    with np.load(str(path)) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["draws"] = arrays["draws"].copy()
    arrays["draws"][0] ^= 1
    np.savez(str(path), **arrays)
    with pytest.raises(tapefmt.TapeError):
        tapefmt.Tape(str(tmp_path / "t")).batch(0).arrays


@pytest.fixture(scope="module")
def tiny():
    if not os.path.exists(os.path.join(TINY, "manifest.json")):
        pytest.skip("tiny tape not found")
    return tapefmt.Tape(TINY)


def _permuted(words, mperm, sperm):
    """The same scene with materials/surfaces renumbered (new label = perm[old])."""
    from trichroma.tape import scene as sc

    out = dict(words)
    minv = np.argsort(mperm)
    sinv = np.argsort(sperm)
    for key in sc._MATERIAL_ROWS:
        out[key] = np.asarray(words[key])[minv]
    out.update(sc._reorder_ranges(words, "material_comp_offsets", sc._COMP_ROWS, minv))
    for key in sc._SURFACE_ROWS:
        out[key] = np.asarray(words[key])[sinv]
    out.update(sc._reorder_ranges(words, "dichroic_offsets", sc._DICHROIC_ROWS, sinv))
    out.update(sc._reorder_ranges(words, "angular_offsets", sc._ANGULAR_ROWS, sinv))
    mmap = np.arange(256, dtype=np.uint32)
    mmap[:len(mperm)] = mperm
    smap = np.arange(256, dtype=np.uint32)
    smap[:len(sperm)] = sperm
    codes = np.asarray(words["material_codes"], np.uint32)
    out["material_codes"] = ((mmap[(codes >> 24) & 0xFF] << 24) | (mmap[(codes >> 16) & 0xFF] << 16)
                             | (smap[(codes >> 8) & 0xFF] << 8) | (codes & 0xFF)).astype(np.uint32)
    planes = np.array(words["wireplanes"], np.uint32, copy=True).reshape(-1, 31)
    if len(planes):
        surf = planes[:, 16].view(np.int32)
        planes[:, 16] = np.where(surf >= 0, smap[np.clip(surf, 0, 255)], planes[:, 16])
        planes[:, 17] = mmap[planes[:, 17] & 0xFF]
        planes[:, 18] = mmap[planes[:, 18] & 0xFF]
    out["wireplanes"] = planes
    return out


def test_scene_relabel(tiny):
    words = tiny.scene(0)
    words = {k: v for k, v in words.items() if not k.startswith("pad_")}
    nm, ns = len(words["material_header"]), len(words["surface_header"])
    rng = np.random.default_rng(3)
    other = _permuted(words, rng.permutation(nm).astype(np.uint32), rng.permutation(ns).astype(np.uint32))
    assert compare_scene_relabeled(other, words) == []
    back = relabel_scene(other, words)
    for key in ("material_codes", "material_refractive_index", "surface_header", "dichroic_reflect"):
        np.testing.assert_array_equal(back[key], words[key])
    changed = dict(other, material_refractive_index=other["material_refractive_index"] * np.float32(1.0001))
    assert compare_scene_relabeled(changed, words) != []


def test_fail_closed_limits(tiny):
    words = tiny.scene(0)
    assert check_legacy_limits(words) and check_daq_limits(words)
    codes = np.asarray(words["material_codes"], np.uint32).copy()
    codes[0] = (codes[0] & 0x00FFFFFF) | (130 << 24)
    with pytest.raises(NotImplementedError):
        check_legacy_limits(dict(words, material_codes=codes))
    with pytest.raises(NotImplementedError):
        check_daq_limits(dict(words, time_cdf_y=np.asarray(words["time_cdf_y"])[:-1]))


@pytest.mark.parametrize("directory", [TINY, TINY_PACKED])
def test_replay_bitwise(directory):
    """The reference loop (same exact kernels, host-scheduled) reproduces the recorded outputs."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    directory = os.environ.get("CHROMA_LEGACY_TAPE", directory)
    if not os.path.exists(os.path.join(directory, "manifest.json")):
        pytest.skip("tape %s not found" % directory)
    from trichroma.tape.reference import replay_tape

    report = replay_tape(directory)
    for batch in report["batches"]:
        assert batch["draw_count_mismatch"] == 0 and batch["harness_errors"] == 0, batch
        for key in ("final_equal", "photons_end_equal", "hits_equal", "channels_equal", "tracks_equal"):
            assert batch.get(key, True) in (True, None), (key, batch)
    assert report["equal"]


@pytest.mark.parametrize("run, directory", [("tiny", TINY), ("tiny_packed", TINY_PACKED)])
def test_public_api_replay(run, directory):
    """trichroma.simulation Simulation -> ProductionEngine(tape=...) -> every output equals the tape's."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if not os.path.exists(os.path.join(directory, "manifest.json")):
        pytest.skip("tape %s not found" % directory)
    from trichroma.simulation import Simulation
    from trichroma.tape.verify import replay_fixture

    ok, report, info = replay_fixture("synthetic", run, directory, simulation_class=Simulation)
    assert ok, report
    assert set(report) >= {"photons_end", "channels_t"}, report
    assert info["simulation"] == "trichroma.simulation.Simulation"


def test_exact_mode_fails_closed(tiny):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from chroma.backend import TapeMode
    from trichroma.engine.core import ProductionEngine

    with pytest.raises(tapefmt.TapeError):  # a different seed than recorded
        ProductionEngine(None, seed=1, device="cuda", tape=TapeMode("replay", TINY),
                         simulation=dict(nthreads_per_block=512, max_blocks=1024, photon_tracking=True))
    with pytest.raises(NotImplementedError):  # canonical schedules are not replayed without a tape
        ProductionEngine(None, seed=23, device="cuda", tape=TapeMode("canonical"))
