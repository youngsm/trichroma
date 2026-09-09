"""Raw-word audit of Triton replay against verified native Chroma captures."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from capture_original_chroma import FIELDS
from chroma_lar.triton_scene.legacy_spectral import replay_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--native-rng",
        action="store_true",
        help="initialize and generate XORWOW independently from the original seed; do not use recorded draws as random inputs",
    )
    args = parser.parse_args()
    capture_path = args.capture / "capture.json"
    capture = json.loads(capture_path.read_text())
    if not capture["passed"]:
        raise ValueError("native recorder has not passed its original-kernel audit")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__),
        root / "chroma-lar/chroma_lar/triton_scene/legacy_spectral.py",
        root / "chroma-lar/chroma_lar/triton_scene/legacy_wires.py",
        root / "chroma-lar/chroma_lar/triton_scene/_legacy_wire_ptx.py",
        root / "chroma-lar/chroma_lar/triton_scene/_legacy_wire_ptx.json",
        root / "chroma-lar/chroma_lar/triton_scene/chroma_global_traversal.py",
        root / "chroma-lar/chroma_lar/triton_backend.py",
        root / "chroma-lite/chroma/triton/physics_kernels.py",
        root / "chroma-lite/chroma/triton/xorwow.py",
    ]
    report = {
        "scope": "Raw-word comparison of independently evolved Triton photon histories against verified original Chroma captures",
        "random_source": (
            "independently initialized and generated XORWOW"
            if args.native_rng
            else "captured native random tape"
        ),
        "capture_report_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "cases": [],
    }
    for row in capture["cases"]:
        name = row["name"]
        if args.cases and name not in args.cases:
            continue
        path = args.capture / f"{name}.npz"
        with np.load(path, allow_pickle=False) as source:
            data = dict(source)
        if (
            hashlib.sha256(data["original_words"].tobytes()).hexdigest()
            != row["original_words_sha256"]
        ):
            raise ValueError("capture output hash mismatch")
        for key, sha in row.get("scene_array_sha256", {}).items():
            if hashlib.sha256(np.ascontiguousarray(data[key]).tobytes()).hexdigest() != sha:
                raise ValueError(f"capture scene hash mismatch: {key}")
        for key, field in (
            ("source_words", "initial_source_sha256"),
            ("initial_words", "normalized_source_sha256"),
            ("random_tape", "random_tape_sha256"),
        ):
            if field in row and hashlib.sha256(data[key].tobytes()).hexdigest() != row[field]:
                raise ValueError(f"capture input hash mismatch: {key}")
        actual = replay_capture(data, native_seed=capture["seed"] if args.native_rng else None)
        comparisons = {}
        pairs = [
            ("final_words", data["original_words"]),
            ("state_words", data["state_words"]),
            ("draw_counts", data["draw_counts"]),
            ("interaction_counts", data["interaction_counts"]),
            ("overflow", np.zeros_like(data["capture_overflow"])),
        ]
        if args.native_rng and "original_rng_words" in data:
            pairs.append(("native_rng_words", data["original_rng_words"]))
        for key, reference in pairs:
            different = np.argwhere(actual[key] != reference)
            first = different[0] if len(different) else None
            detail = (
                None
                if first is None
                else {
                    "index": first.tolist(),
                    "original_word": int(reference[tuple(first)]),
                    "triton_word": int(actual[key][tuple(first)]),
                }
            )
            if detail and key in ("final_words", "state_words"):
                detail["field"] = FIELDS[first[-1]]
            elif detail and key == "native_rng_words":
                detail["field"] = ("d", "v0", "v1", "v2", "v3", "v4")[first[-1]]
            comparisons[key] = {"mismatches": len(different), "first": detail}
        result = {
            "case": name,
            "photons": len(data["initial_words"]),
            "initialization": (
                "independent normalization of raw input photon words"
                if "source_words" in data
                else "captured CUDA-normalized photon words"
            ),
            "unfinished_reference_photons": row.get("unfinished_photons"),
            "capture_archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "comparisons": comparisons,
            "passed": all(v["mismatches"] == 0 for v in comparisons.values()),
        }
        np.savez_compressed(args.output / f"{name}.npz", **actual)
        print(json.dumps(result), flush=True)
        report["cases"].append(result)
        report["passed"] = all(r["passed"] for r in report["cases"])
        (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["cases"] or not report["passed"]:
        raise AssertionError("original Chroma / Triton raw-word replay differs")


if __name__ == "__main__":
    main()
