"""Replay an observed original queue using independently evolved Triton states.

Only photon ordering and launch sizes come from the recorded schedule. Native
RNG words and photon outputs are comparison targets, never simulation inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_original_chroma import digest
from chroma.triton.xorwow import initialize_xorwow
from chroma_lar.triton_scene.legacy_spectral import propagate_legacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture_path = args.capture / "schedule.json"
    capture = json.loads(capture_path.read_text())
    with np.load(args.capture / "scene.npz") as archive:
        scene = dict(archive)
    for key, sha in capture["scene_array_sha256"].items():
        if digest(scene[key]) != sha:
            raise ValueError(f"scene array hash mismatch: {key}")
    with np.load(args.capture / "source.npz") as archive:
        words = archive["source_words"].copy()
    if digest(words) != capture["initial_photons_sha256"]:
        raise ValueError("source hash mismatch")
    rng = initialize_xorwow(capture["seed"], np.arange(capture["rng_slots"], dtype=np.uint64))
    if digest(rng) != capture["initial_rng_words_sha256"]:
        raise ValueError("independent RNG initialization differs from original")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    files = [
        Path(__file__),
        root / "chroma-lar/chroma_lar/triton_scene/legacy_spectral.py",
        root / "chroma-lar/chroma_lar/triton_scene/legacy_wires.py",
        root / "chroma-lar/chroma_lar/triton_scene/_legacy_wire_ptx.py",
        root / "chroma-lar/chroma_lar/triton_scene/chroma_global_traversal.py",
        root / "chroma-lar/chroma_lar/triton_backend.py",
        root / "chroma-lite/chroma/triton/physics_kernels.py",
        root / "chroma-lite/chroma/triton/xorwow.py",
    ]
    report = {
        "scope": "original queue ordering with independently normalized photon states and independently seeded/evolved XORWOW in Triton",
        "capture_report_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        "photons": len(words),
        "launches": [],
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "limits": "A captured atomic schedule is required; equal seeds alone do not determine the original queue ordering. Original states are verified after every actual launch, which may contain multiple interactions in the final tail.",
    }
    for launch in capture["launches"]:
        index, count = launch["index"], launch["threads"]
        path = args.capture / f"launch_{index:04d}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != launch["archive_sha256"]:
            raise ValueError("launch archive hash mismatch")
        if launch["use_weights"] or launch["scatter_first"]:
            raise NotImplementedError("weighted/scatter-first compatibility is not implemented")
        with np.load(path) as archive:
            ids = archive["photon_ids"]
        if len(ids) != count or len(np.unique(ids)) != count or np.any(ids >= len(words)):
            raise ValueError("invalid original launch queue")
        actual = propagate_legacy(
            scene,
            words[ids],
            max_steps=launch["max_steps"],
            rng_words=rng[:count],
            record_history=False,
        )
        words[ids] = actual["final_words"]
        rng[:count] = actual["native_rng_words"]
        comparisons = {}
        with np.load(path) as archive:
            for key, expected in (
                ("final_words", archive["state_words"]),
                ("native_rng_words", archive["native_rng_words"]),
            ):
                different = np.argwhere(actual[key] != expected)
                first = tuple(different[0]) if len(different) else None
                comparisons[key] = {
                    "mismatches": len(different),
                    "first": (
                        None
                        if first is None
                        else {
                            "index": list(map(int, first)),
                            "original": int(expected[first]),
                            "triton": int(actual[key][first]),
                            "photon_id": int(ids[first[0]]),
                        }
                    ),
                }
        row = {
            "index": index,
            "threads": count,
            "max_steps": launch["max_steps"],
            "committed_interactions": int(actual["interaction_counts"].sum()),
            "draws": int(np.maximum(actual["draw_counts"], 0).sum()),
            "comparisons": comparisons,
            "passed": all(v["mismatches"] == 0 for v in comparisons.values()),
        }
        print(json.dumps(row), flush=True)
        report["launches"].append(row)
        report["passed"] = all(r["passed"] for r in report["launches"])
        (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
        if not row["passed"]:
            np.savez_compressed(args.output / f"mismatch_{index:04d}.npz", **actual)
            raise AssertionError("Triton diverged from the observed native execution")
    report["final_words_sha256"] = digest(words)
    report["passed"] &= report["final_words_sha256"] == capture["final_words_sha256"]
    report["unfinished_photons"] = int(np.count_nonzero((words[:, 11] & 32783) == 0))
    np.savez_compressed(args.output / "final.npz", final_words=words, rng_words=rng)
    (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError("complete original output differs")


if __name__ == "__main__":
    main()
