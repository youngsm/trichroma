"""Bitwise CUDA/Triton comparison through the public ``chroma.sim.Simulation`` API.

Subcommands
-----------
``run``      Run one fixture in *this* process (backend and tape mode from the
             environment) and save every Event output to an ``.npz``.
``compare``  Compare two ``run`` outputs byte for byte (hits in photon order).
``tape``     Replay a recorded tape through the reference harness and compare
             with the recorded outputs (no second process needed).
``all``      Orchestrate: record with the CUDA backend, replay with the Triton
             backend (``CHROMA_TRITON_TAPE=replay:<dir>``), compare.

Example::

    python -m chroma.triton.legacy.verify all --fixture synthetic \\
        --cuda-python /path/to/cuda/env/bin/python \\
        --triton-python /path/to/triton/env/bin/python --work /lscratch/$USER/tape-check
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from chroma.triton.legacy import tape as tapefmt


# ----------------------------------------------------------------- run fixture


def _event_arrays(events, has_channels):
    """Flatten a list of Events into arrays with per-event offsets."""
    out = {}
    end = [ev.photons_end for ev in events]
    if all(p is not None for p in end) and end:
        out["end_offsets"] = np.concatenate([[0], np.cumsum([len(p) for p in end])]).astype(np.int64)
        for name in tapefmt.PHOTON_FIELD_NAMES:
            out["end_" + name] = np.concatenate([np.asarray(getattr(p, name)) for p in end])
    hits = [getattr(ev, "flat_hits", None) for ev in events]
    if has_channels and all(h is not None for h in hits) and hits:
        out["hits_offsets"] = np.concatenate([[0], np.cumsum([len(h) for h in hits])]).astype(np.int64)
        for name in tapefmt.PHOTON_FIELD_NAMES + ("channel",):
            out["hits_" + name] = np.concatenate([np.asarray(getattr(h, name)) for h in hits])
    chans = [getattr(ev, "channels", None) for ev in events]
    if chans and all(c is not None for c in chans):
        out["channels_t"] = np.stack([np.asarray(c.t, np.float32) for c in chans])
        out["channels_q"] = np.stack([np.asarray(c.q, np.float32) for c in chans])
        out["channels_flags"] = np.stack([np.asarray(c.flags, np.uint32) for c in chans])
        out["channels_hit"] = np.stack([np.asarray(c.hit, np.uint8) for c in chans])
    tracks = [getattr(ev, "photon_tracks", None) for ev in events]
    if tracks and all(t is not None for t in tracks):
        lengths, fields = [], {name: [] for name in tapefmt.PHOTON_FIELD_NAMES}
        for ev_tracks in tracks:
            for track in ev_tracks:
                lengths.append(len(track))
                for name in fields:
                    if len(track):
                        fields[name].append(np.asarray(getattr(track, name)).reshape(len(track), -1))
        out["track_lengths"] = np.asarray(lengths, np.int64)
        for name, parts in fields.items():
            out["track_" + name] = np.concatenate(parts) if parts else np.zeros(0)
    return out


def run_fixture(fixture, run_name, output, scale=1.0, detector=None):
    """Run one fixture run through chroma.sim.Simulation; save outputs to ``output``."""
    from chroma.backend import backend_name, tape_mode
    from chroma.sim import Simulation
    from chroma.triton.legacy import fixtures

    run = [r for r in fixtures.RUNS[fixture] if r["name"] == run_name][0]
    t0 = time.time()
    if detector is None:
        detector = fixtures.build_detector(fixture)
    t_build = time.time() - t0
    events = fixtures.make_events(fixture, run, scale)
    t0 = time.time()
    sim = Simulation(detector, **run["sim"])
    t_init = time.time() - t0
    t0 = time.time()
    produced = list(sim.simulate(events, **run["simulate"]))
    t_sim = time.time() - t0
    arrays = _event_arrays(produced, hasattr(detector, "num_channels"))
    info = dict(fixture=fixture, run=run_name, scale=scale, backend=backend_name(),
                tape=str(tape_mode()), nphotons=int(sum(len(e) for e in events)), nevents=len(events),
                seconds=dict(build=t_build, init=t_init, simulate=t_sim))
    arrays["info"] = np.frombuffer(json.dumps(info).encode(), np.uint8)
    np.savez(output, **arrays)
    return info


# ------------------------------------------------------------------ comparison


def _hits_photon_order(arrays, prefix="hits"):
    """Sort each event's hits into photon order via their full words.

    Hits carry no photon index; CUDA returns them in warp-atomic order. Two
    hit lists are equal as multisets iff their lexicographically sorted
    word rows are equal, which is what the comparison uses.
    """
    offsets = arrays[prefix + "_offsets"]
    words = tapefmt.photon_words({name: arrays[prefix + "_" + name] for name in tapefmt.PHOTON_FIELD_NAMES})
    words = np.concatenate([words, arrays[prefix + "_channel"].astype(np.uint32).reshape(-1, 1)], axis=1)
    rows = []
    for i in range(len(offsets) - 1):
        block = words[offsets[i]:offsets[i + 1]]
        order = np.lexsort(block.T[::-1])
        rows.append(block[order])
    return np.concatenate(rows) if rows else words


def first_difference(a, b, names=None):
    """(row, column) of the first differing uint32 word of two [N,K] arrays, or None."""
    if a.shape != b.shape:
        return ("shape", a.shape, b.shape)
    diff = np.flatnonzero(np.any(a != b, axis=1)) if a.ndim == 2 else np.flatnonzero(a != b)
    if not len(diff):
        return None
    row = int(diff[0])
    if a.ndim == 2:
        col = int(np.flatnonzero(a[row] != b[row])[0])
        return (row, names[col] if names else col, int(a[row, col]), int(b[row, col]), int(len(diff)))
    return (row, None, int(a[row]), int(b[row]), int(len(diff)))


def compare_outputs(expected, actual):
    """Byte comparison of two ``run`` outputs. Returns (ok, report dict)."""
    report = {}
    ok = True
    if "end_pos" in expected or "end_pos" in actual:
        wa = tapefmt.photon_words({n: expected["end_" + n] for n in tapefmt.PHOTON_FIELD_NAMES})
        wb = tapefmt.photon_words({n: actual["end_" + n] for n in tapefmt.PHOTON_FIELD_NAMES})
        d = first_difference(wa, wb, tapefmt.WORD_NAMES)
        report["photons_end"] = dict(photons=int(len(wa)), equal=d is None, first_difference=d)
        ok &= d is None
    if "hits_offsets" in expected or "hits_offsets" in actual:
        same_counts = np.array_equal(expected["hits_offsets"], actual["hits_offsets"])
        wa = _hits_photon_order(expected)
        wb = _hits_photon_order(actual) if same_counts else None
        d = first_difference(wa, wb, tapefmt.WORD_NAMES + ("channel",)) if same_counts else "hit counts differ"
        report["hits"] = dict(hits=int(len(wa)), equal=d is None, first_difference=d)
        ok &= d is None
    for key in ("channels_t", "channels_q", "channels_flags", "channels_hit"):
        if key in expected or key in actual:
            a = np.ascontiguousarray(expected[key]).view(np.uint8)
            b = np.ascontiguousarray(actual[key]).view(np.uint8)
            equal = a.shape == b.shape and a.tobytes() == b.tobytes()
            report[key] = dict(equal=equal)
            if not equal and a.shape == b.shape:
                ea = np.ascontiguousarray(expected[key]).reshape(len(expected[key]), -1)
                eb = np.ascontiguousarray(actual[key]).reshape(len(actual[key]), -1)
                report[key]["first_difference"] = first_difference(ea.view(np.uint32) if ea.itemsize == 4 else ea,
                                                                   eb.view(np.uint32) if eb.itemsize == 4 else eb)
            ok &= equal
    if "track_lengths" in expected or "track_lengths" in actual:
        same = np.array_equal(expected["track_lengths"], actual["track_lengths"])
        d = None
        if same:
            wa = tapefmt.photon_words({n: expected["track_" + n].reshape(len(expected["track_" + n]), *(
                (3,) if n in ("pos", "dir", "pol") else ())) for n in tapefmt.PHOTON_FIELD_NAMES})
            wb = tapefmt.photon_words({n: actual["track_" + n].reshape(len(actual["track_" + n]), *(
                (3,) if n in ("pos", "dir", "pol") else ())) for n in tapefmt.PHOTON_FIELD_NAMES})
            d = first_difference(wa, wb, tapefmt.WORD_NAMES)
        else:
            d = "track lengths differ"
        report["photon_tracks"] = dict(equal=d is None, first_difference=d)
        ok &= d is None
    return bool(ok), report


def _load(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# ------------------------------------------------------------------------ main


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m chroma.triton.legacy.verify", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="run a fixture in this process")
    p.add_argument("--fixture", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p = sub.add_parser("compare", help="compare two run outputs")
    p.add_argument("expected")
    p.add_argument("actual")
    p = sub.add_parser("tape", help="replay a tape through the reference harness")
    p.add_argument("--tape", required=True)
    p.add_argument("--report")
    p = sub.add_parser("all", help="record (CUDA) + replay (Triton) + compare")
    p.add_argument("--fixture", default="synthetic")
    p.add_argument("--runs", nargs="*")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--work", required=True)
    p.add_argument("--cuda-python", default=os.environ.get("CHROMA_VERIFY_CUDA_PYTHON", sys.executable))
    p.add_argument("--triton-python", default=os.environ.get("CHROMA_VERIFY_TRITON_PYTHON", sys.executable))
    p.add_argument("--sort-queues", action="store_true",
                   help="record with ascending survivor queues (CHROMA_TRITON_TAPE_SORT=1)")
    p.add_argument("--transparency", action="store_true",
                   help="also run CUDA without recording (CHROMA_TRITON_TAPE=canonical) and require identical "
                        "outputs; implies --sort-queues (unsorted multi-launch runs are not repeatable)")
    p.add_argument("--reference-only", action="store_true",
                   help="replay with the reference harness only (no Triton Simulation run)")
    args = parser.parse_args(argv)

    if args.command == "run":
        info = run_fixture(args.fixture, args.run, args.output, scale=args.scale)
        print(json.dumps(info))
        return 0
    if args.command == "compare":
        ok, report = compare_outputs(_load(args.expected), _load(args.actual))
        print(json.dumps(report, indent=1, default=str))
        print("BITWISE EQUAL" if ok else "DIFFERENT")
        return 0 if ok else 1
    if args.command == "tape":
        from chroma.triton.legacy.reference import replay_tape

        report = replay_tape(args.tape)
        text = json.dumps(report, indent=1, default=str)
        if args.report:
            with open(args.report, "w") as f:
                f.write(text)
        print(text)
        return 0 if report["equal"] else 1
    if args.command == "all":
        return _all(args)
    return 2


def _all(args):
    from chroma.triton.legacy import fixtures

    os.makedirs(args.work, exist_ok=True)
    if args.transparency and not args.sort_queues:
        # Original Chroma appends survivors to the next launch's queue with
        # warp-level atomics, so two unrecorded multi-launch runs already
        # differ from each other; compare under the canonical (sorted) order.
        print("--transparency: recording with sorted queues (CHROMA_TRITON_TAPE_SORT=1)")
        args.sort_queues = True
    runs = args.runs or [r["name"] for r in fixtures.RUNS[args.fixture]]
    summary = {}
    base_env = dict(os.environ)
    for run in runs:
        tag = "%s_%s" % (args.fixture, run)
        tape_dir = os.path.join(args.work, tag + "_tape")
        if os.path.exists(os.path.join(tape_dir, "manifest.json")):
            raise SystemExit("%s already holds a tape; choose a fresh --work directory" % tape_dir)
        cuda_out = os.path.join(args.work, tag + "_cuda.npz")
        env = dict(base_env, CHROMA_BACKEND="cuda", CHROMA_TRITON_TAPE="record:" + tape_dir)
        if args.sort_queues:
            env["CHROMA_TRITON_TAPE_SORT"] = "1"
        cmd = [args.cuda_python, "-m", "chroma.triton.legacy.verify", "run", "--fixture", args.fixture,
               "--run", run, "--output", cuda_out, "--scale", str(args.scale)]
        subprocess.run(cmd, env=env, check=True)
        entry = {"tape": tape_dir}
        if args.transparency:
            plain_out = os.path.join(args.work, tag + "_cuda_plain.npz")
            env = dict(base_env, CHROMA_BACKEND="cuda")
            env.pop("CHROMA_TRITON_TAPE", None)
            if args.sort_queues:
                env["CHROMA_TRITON_TAPE"] = "canonical"
            subprocess.run([args.cuda_python, "-m", "chroma.triton.legacy.verify", "run", "--fixture",
                            args.fixture, "--run", run, "--output", plain_out, "--scale", str(args.scale)],
                           env=env, check=True)
            ok, report = compare_outputs(_load(plain_out), _load(cuda_out))
            entry["recorder_transparent"] = dict(equal=ok, report=report)
        env = dict(base_env, CHROMA_BACKEND="triton")
        env.pop("CHROMA_TRITON_TAPE", None)
        ref_report = os.path.join(args.work, tag + "_reference.json")
        subprocess.run([args.triton_python, "-m", "chroma.triton.legacy.verify", "tape", "--tape", tape_dir,
                        "--report", ref_report], env=env, check=False)
        with open(ref_report) as f:
            entry["reference_replay"] = json.load(f)
        if not args.reference_only:
            triton_out = os.path.join(args.work, tag + "_triton.npz")
            env = dict(base_env, CHROMA_BACKEND="triton", CHROMA_TRITON_TAPE="replay:" + tape_dir)
            result = subprocess.run([args.triton_python, "-m", "chroma.triton.legacy.verify", "run", "--fixture",
                                     args.fixture, "--run", run, "--output", triton_out, "--scale",
                                     str(args.scale)], env=env)
            if result.returncode == 0:
                ok, report = compare_outputs(_load(cuda_out), _load(triton_out))
                entry["simulation_replay"] = dict(equal=ok, report=report)
            else:
                entry["simulation_replay"] = dict(equal=False, error="Triton run failed (exit %d)" % result.returncode)
        summary[tag] = entry
    text = json.dumps(summary, indent=1, default=str)
    with open(os.path.join(args.work, "summary_%s.json" % args.fixture), "w") as f:
        f.write(text)
    print(text)
    good = all(e.get("reference_replay", {}).get("equal", False)
               and e.get("simulation_replay", {"equal": True}).get("equal", False)
               and e.get("recorder_transparent", {"equal": True}).get("equal", False) for e in summary.values())
    print("ALL BITWISE EQUAL" if good else "DIFFERENCES FOUND")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
