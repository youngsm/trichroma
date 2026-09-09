#!/usr/bin/env python3
"""Compose a fail-closed full-trajectory certificate from population + tail runs.

The first run proves a complete population through a bounded prefix.  The
second reruns exactly the prefix survivors, addressed by their original source
population/global photon IDs, until every survivor is terminal.  This script
independently checks the retained NPZ endpoint, process/draw, and post-
interaction state-ledger arrays before emitting a small composite certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXACT_FIELDS = (
    "position",
    "direction",
    "polarization",
    "time",
    "history",
    "evidx",
    "last_triangle",
    "boundary_kind",
    "detected_channel",
    "interaction_cursor",
    "draw_cursor",
    "overflow",
    "interaction_certificate",
    "state_certificate",
    "global_ids",
)
STABLE_GEOMETRY_FIELDS = (
    "mesh_md5",
    "traversal_sha256",
    "optical_semantics_sha256",
)
CERTIFICATE_SENTINEL = np.uint32(0xFFFFFFFF)
STATE_CERTIFICATE_FIELDS = (
    "position_x",
    "position_y",
    "position_z",
    "direction_x",
    "direction_y",
    "direction_z",
    "polarization_x",
    "polarization_y",
    "polarization_z",
    "wavelength",
    "time",
    "history",
    "last_triangle",
    "weight",
    "evidx",
)
PROCESS_NAMES = {
    1: "bulk_absorb",
    2: "bulk_scatter",
    3: "surface_absorb",
    4: "surface_detect",
    5: "surface_diffuse",
    6: "surface_specular",
    7: "dielectric_reflect",
    8: "dielectric_transmit",
    11: "no_hit",
    12: "invalid",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"certificate is not a JSON object: {path}")
    return value


def _require_report(report: dict[str, Any], *, label: str) -> None:
    requirements = {
        "matched_bitwise": True,
        "draw_consumption_certified": True,
        "post_interaction_state_certified": True,
        "tape_overflow_free": True,
        "legacy_specular_reflection": True,
        "chroma_mesh_box_compatibility": True,
        "chroma_global_bvh_compatibility": True,
        "representable_progress_guard": False,
    }
    # One-step reports omit the multistep-only convenience field.  The
    # composite inputs are always multistep, so absence here is an error.
    for name, expected in requirements.items():
        if report.get(name) is not expected:
            raise RuntimeError(
                f"{label} report did not certify {name}={expected!r}"
            )
    if report.get("wire_scan_source_indices") != list(range(6)):
        raise RuntimeError(f"{label} report did not scan all six wires")
    evidx = report.get("single_event_evidx_invariant", {})
    if evidx.get("satisfied") is not True:
        raise RuntimeError(f"{label} report did not certify the evidx invariant")
    interaction = report.get("interaction_certificate", {})
    for name in ("ledger_equal", "cursor_equal", "overflow_free"):
        if interaction.get(name) is not True:
            raise RuntimeError(f"{label} interaction certificate failed {name}")
    prefix = interaction.get("prefix_valid", {})
    if prefix.get("chroma") is not True or prefix.get("triton") is not True:
        raise RuntimeError(f"{label} ledger is not a hole-free cursor prefix")


def _compare_npz(cuda_path: Path, triton_path: Path, *, label: str) -> dict[str, Any]:
    with np.load(cuda_path) as cuda, np.load(triton_path) as triton:
        for field in EXACT_FIELDS:
            if field not in cuda or field not in triton:
                raise RuntimeError(f"{label} NPZ is missing {field}")
            left = np.ascontiguousarray(cuda[field])
            right = np.ascontiguousarray(triton[field])
            if left.dtype != right.dtype or left.shape != right.shape:
                raise RuntimeError(
                    f"{label} {field} shape/dtype differs: "
                    f"{left.shape}/{left.dtype} vs {right.shape}/{right.dtype}"
                )
            if not np.array_equal(left.view(np.uint8), right.view(np.uint8)):
                raise RuntimeError(f"{label} {field} is not byte-exact")
        last_instance = np.ascontiguousarray(triton["last_instance"], dtype=np.int32)
        if not np.all(last_instance == -1):
            raise RuntimeError(f"{label} escaped the global triangle namespace")
        if np.any(np.asarray(cuda["overflow"], dtype=np.uint32)):
            raise RuntimeError(f"{label} CUDA tape overflowed")
        if np.any(np.asarray(triton["overflow"], dtype=np.uint32)):
            raise RuntimeError(f"{label} Triton tape overflowed")
        cuda_active = np.ascontiguousarray(cuda["active_ids"], dtype=np.int64)
        triton_active = np.ascontiguousarray(
            triton["active_ids"], dtype=np.int64
        )
        if (
            np.unique(cuda_active).size != cuda_active.size
            or np.unique(triton_active).size != triton_active.size
            or not np.array_equal(np.sort(cuda_active), np.sort(triton_active))
        ):
            raise RuntimeError(f"{label} active global-ID sets differ")
        return {
            "global_ids": np.ascontiguousarray(cuda["global_ids"], dtype=np.int64),
            "initial_position": np.ascontiguousarray(
                cuda["initial_position"], dtype=np.float32
            ),
            "normalized_direction": np.ascontiguousarray(
                cuda["normalized_direction"], dtype=np.float32
            ),
            "normalized_polarization": np.ascontiguousarray(
                cuda["normalized_polarization"], dtype=np.float32
            ),
            # Preserve the CUDA queue order because the selected-tail command
            # records that exact ordered list; semantic equality above is a
            # global-ID set because scheduler queue order is unobservable.
            "active_ids": cuda_active,
            "interaction_cursor": np.ascontiguousarray(
                cuda["interaction_cursor"], dtype=np.int32
            ),
            "interaction_certificate": np.ascontiguousarray(
                cuda["interaction_certificate"], dtype=np.uint32
            ),
            "state_certificate": np.ascontiguousarray(
                cuda["state_certificate"], dtype=np.uint32
            ),
            "history": np.ascontiguousarray(cuda["history"], dtype=np.uint32),
            "ledger_shape": list(cuda["interaction_certificate"].shape),
        }


def _ledger_summary(*chunks: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    process_words: list[np.ndarray] = []
    draw_words: list[np.ndarray] = []
    total_rows = 0
    for words, cursor in chunks:
        total_rows += int(words.shape[0])
        expected = (
            np.arange(words.shape[1], dtype=np.int32)[None, :]
            < cursor[:, None]
        )
        committed = words != CERTIFICATE_SENTINEL
        if not np.array_equal(committed, expected):
            raise RuntimeError("combined ledger is not a hole-free cursor prefix")
        selected = words[committed]
        process_words.append((selected >> np.uint32(28)) & np.uint32(0xF))
        draw_words.append(selected & np.uint32(0x0FFFFFFF))
    processes = np.concatenate(process_words)
    draws = np.concatenate(draw_words)
    by_process: dict[str, Any] = {}
    for code in np.unique(processes):
        selected = processes == code
        selected_draws = draws[selected]
        name = PROCESS_NAMES.get(int(code), f"process_{int(code)}")
        by_process[name] = {
            "count": int(np.count_nonzero(selected)),
            "draws_total": int(np.sum(selected_draws, dtype=np.uint64)),
            "minimum_draws": int(np.min(selected_draws)),
            "maximum_draws": int(np.max(selected_draws)),
        }
    return {
        "photon_rows": total_rows,
        "committed_interactions": int(processes.size),
        "random_draws": int(np.sum(draws, dtype=np.uint64)),
        "by_process": by_process,
    }


def _state_ledger_summary(
    *chunks: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> dict[str, Any]:
    """Validate dense state records against process-ledger commit masks."""

    digest = hashlib.sha256()
    committed_records = 0
    total_rows = 0
    for states, processes, cursor in chunks:
        total_rows += int(states.shape[0])
        expected_shape = processes.shape + (len(STATE_CERTIFICATE_FIELDS),)
        if states.dtype != np.dtype(np.uint32) or states.shape != expected_shape:
            raise RuntimeError(
                "state certificate must be uint32 [row, interaction, 15]"
            )
        expected = (
            np.arange(processes.shape[1], dtype=np.int32)[None, :]
            < cursor[:, None]
        )
        committed = processes != CERTIFICATE_SENTINEL
        if not np.array_equal(committed, expected):
            raise RuntimeError("state-ledger process mask is not a cursor prefix")
        suffix = states[~committed]
        if suffix.size and not np.all(suffix == CERTIFICATE_SENTINEL):
            row, interaction, field = np.argwhere(
                (~committed)[..., None]
                & (states != CERTIFICATE_SENTINEL)
            )[0]
            raise RuntimeError(
                "state certificate has a non-sentinel suffix at "
                f"row {int(row)}, interaction {int(interaction)}, "
                f"field {STATE_CERTIFICATE_FIELDS[int(field)]}"
            )
        selected = np.ascontiguousarray(states[committed], dtype=np.uint32)
        committed_records += int(selected.shape[0])
        digest.update(selected.view(np.uint8))
    return {
        "photon_rows": total_rows,
        "committed_state_records": committed_records,
        "raw_words_per_record": len(STATE_CERTIFICATE_FIELDS),
        "fields": list(STATE_CERTIFICATE_FIELDS),
        "ordered_raw_words_sha256": digest.hexdigest(),
    }


def _validate_prefix_tail_continuity(
    prefix: dict[str, np.ndarray], tail: dict[str, np.ndarray]
) -> int:
    """Prove that an independently replayed tail has the same bounded prefix."""

    prefix_rows = {
        int(global_id): row
        for row, global_id in enumerate(prefix["global_ids"])
    }
    continuity_interactions = 0
    for tail_row, raw_global_id in enumerate(tail["global_ids"]):
        global_id = int(raw_global_id)
        prefix_row = prefix_rows.get(global_id)
        if prefix_row is None:
            raise RuntimeError(f"tail global ID {global_id} is absent from prefix")
        for field in (
            "initial_position",
            "normalized_direction",
            "normalized_polarization",
        ):
            left = np.ascontiguousarray(prefix[field][prefix_row]).view(np.uint8)
            right = np.ascontiguousarray(tail[field][tail_row]).view(np.uint8)
            if not np.array_equal(left, right):
                raise RuntimeError(
                    f"tail source discontinuity for global ID {global_id}, {field}"
                )
        depth = int(prefix["interaction_cursor"][prefix_row])
        if int(tail["interaction_cursor"][tail_row]) < depth:
            raise RuntimeError(
                f"tail global ID {global_id} ended before prefix cursor {depth}"
            )
        if not np.array_equal(
            prefix["interaction_certificate"][prefix_row, :depth],
            tail["interaction_certificate"][tail_row, :depth],
        ):
            raise RuntimeError(
                f"tail process/draw history diverges before splice for ID {global_id}"
            )
        if not np.array_equal(
            prefix["state_certificate"][prefix_row, :depth].view(np.uint8),
            tail["state_certificate"][tail_row, :depth].view(np.uint8),
        ):
            raise RuntimeError(
                f"tail state history diverges before splice for ID {global_id}"
            )
        continuity_interactions += depth
    return continuity_interactions


def _terminal_summary(histories: np.ndarray) -> dict[str, Any]:
    terminal_rules = (
        (1 << 15, "invalid"),
        (1 << 0, "no_hit"),
        (1 << 1, "bulk_absorb"),
        (1 << 2, "surface_detect"),
        (1 << 3, "surface_absorb"),
    )
    unresolved = np.ones(histories.size, dtype=np.bool_)
    outcomes: dict[str, int] = {}
    for bit, name in terminal_rules:
        selected = unresolved & ((histories & np.uint32(bit)) != 0)
        outcomes[name] = int(np.count_nonzero(selected))
        unresolved[selected] = False
    if np.any(unresolved):
        raise RuntimeError("composed population contains a nonterminal history")
    return {"total": int(histories.size), "outcomes": outcomes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-report", type=Path, required=True)
    parser.add_argument("--prefix-npz", type=Path, required=True)
    parser.add_argument("--tail-report", type=Path, required=True)
    parser.add_argument("--tail-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prefix_report = _load_json(args.prefix_report)
    tail_report = _load_json(args.tail_report)
    _require_report(prefix_report, label="population prefix")
    _require_report(tail_report, label="selected tail")
    prefix = _compare_npz(
        Path(str(args.prefix_npz) + ".cuda.npz"),
        Path(str(args.prefix_npz) + ".triton.npz"),
        label="population prefix",
    )
    tail = _compare_npz(
        Path(str(args.tail_npz) + ".cuda.npz"),
        Path(str(args.tail_npz) + ".triton.npz"),
        label="selected tail",
    )

    population = int(prefix_report["photons"])
    if not np.array_equal(prefix["global_ids"], np.arange(population, dtype=np.int64)):
        raise RuntimeError("prefix report does not cover the full contiguous population")
    if not np.array_equal(prefix["active_ids"], tail["global_ids"]):
        raise RuntimeError("tail selection is not exactly the prefix survivor queue")
    if tail["active_ids"].size:
        raise RuntimeError("selected tail did not reach terminal state")
    if int(tail_report["photons"]) != int(prefix["active_ids"].size):
        raise RuntimeError("tail report photon count does not match prefix survivors")
    if prefix_report["source_seed"] != tail_report["source_seed"]:
        raise RuntimeError("source seeds differ")
    if prefix_report["tape_seed"] != tail_report["tape_seed"]:
        raise RuntimeError("tape seeds differ")
    for field in ("center", "voxel_size", "draws_per_interaction"):
        if prefix_report.get(field) != tail_report.get(field):
            raise RuntimeError(f"source/tape configuration differs for {field}")
    if int(prefix_report.get("source_population", -1)) != population:
        raise RuntimeError("prefix source_population does not cover its population")
    if int(tail_report.get("source_population", -1)) != population:
        raise RuntimeError("tail source_population differs from the prefix population")
    if int(prefix_report.get("tape_interactions", -1)) != int(
        prefix["interaction_certificate"].shape[1]
    ):
        raise RuntimeError("prefix tape extent differs between report and arrays")
    if int(tail_report.get("tape_interactions", -1)) != int(
        tail["interaction_certificate"].shape[1]
    ):
        raise RuntimeError("tail tape extent differs between report and arrays")
    if prefix_report["provenance"] != tail_report["provenance"]:
        raise RuntimeError("proof source/runtime provenance differs between phases")
    for field in STABLE_GEOMETRY_FIELDS:
        left = prefix_report["global_bvh_certificate"][field]
        right = tail_report["global_bvh_certificate"][field]
        if left != right:
            raise RuntimeError(f"global BVH {field} differs between phases")

    survivor_mask = np.isin(prefix["global_ids"], tail["global_ids"])
    continuity_interactions = _validate_prefix_tail_continuity(prefix, tail)
    ledger = _ledger_summary(
        (
            prefix["interaction_certificate"][~survivor_mask],
            prefix["interaction_cursor"][~survivor_mask],
        ),
        (tail["interaction_certificate"], tail["interaction_cursor"]),
    )
    state_ledger = _state_ledger_summary(
        (
            prefix["state_certificate"][~survivor_mask],
            prefix["interaction_certificate"][~survivor_mask],
            prefix["interaction_cursor"][~survivor_mask],
        ),
        (
            tail["state_certificate"],
            tail["interaction_certificate"],
            tail["interaction_cursor"],
        ),
    )
    if (
        state_ledger["committed_state_records"]
        != ledger["committed_interactions"]
    ):
        raise RuntimeError("state and process ledgers have different lengths")
    terminal_histories = prefix["history"].copy()
    tail_rows = {
        int(global_id): row for row, global_id in enumerate(tail["global_ids"])
    }
    for prefix_row in np.flatnonzero(survivor_mask):
        global_id = int(prefix["global_ids"][prefix_row])
        terminal_histories[prefix_row] = tail["history"][tail_rows[global_id]]
    terminal = _terminal_summary(terminal_histories)

    paths = {
        "prefix_report": args.prefix_report,
        "prefix_cuda_npz": Path(str(args.prefix_npz) + ".cuda.npz"),
        "prefix_triton_npz": Path(str(args.prefix_npz) + ".triton.npz"),
        "prefix_global_bvh": Path(str(args.prefix_npz) + ".chroma-global-bvh.npz"),
        "tail_report": args.tail_report,
        "tail_cuda_npz": Path(str(args.tail_npz) + ".cuda.npz"),
        "tail_triton_npz": Path(str(args.tail_npz) + ".triton.npz"),
        "tail_global_bvh": Path(str(args.tail_npz) + ".chroma-global-bvh.npz"),
        "composer": Path(__file__).resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing {label}: {path}")

    result = {
        "schema_version": 3,
        "detector": prefix_report["detector"],
        "scope": "complete post-interaction trajectories for one fixed source/tape population",
        "matched_bitwise": True,
        "terminal_histories_certified": True,
        "intermediate_state_words_certified": True,
        "draw_consumption_certified": True,
        "population_photons": population,
        "source_seed": int(prefix_report["source_seed"]),
        "tape_seed": int(prefix_report["tape_seed"]),
        "source_configuration": {
            "center": prefix_report["center"],
            "voxel_size": prefix_report["voxel_size"],
            "source_population": population,
            "draws_per_interaction": prefix_report["draws_per_interaction"],
        },
        "prefix": {
            "max_steps": int(prefix_report["max_steps"]),
            "active_after_prefix": int(prefix["active_ids"].size),
            "ledger_shape": prefix["ledger_shape"],
        },
        "tail": {
            "selected_original_global_ids": int(tail["global_ids"].size),
            "max_steps": int(tail_report["max_steps"]),
            "maximum_committed_interactions": int(np.max(tail["interaction_cursor"])),
            "active_after_tail": 0,
            "ledger_shape": tail["ledger_shape"],
        },
        "prefix_tail_continuity": {
            "source_words_equal": True,
            "process_draw_prefix_equal": True,
            "post_interaction_state_prefix_equal": True,
            "survivor_photons": int(tail["global_ids"].size),
            "rechecked_prefix_interactions": continuity_interactions,
        },
        "complete_population_ledger": ledger,
        "complete_population_state_ledger": state_ledger,
        "terminal_histories": terminal,
        "strict_compatibility_policy": {
            "legacy_specular_reflection": True,
            "chroma_global_bvh": True,
            "representable_progress_guard": False,
            "wire_scan_source_indices": list(range(6)),
        },
        "global_bvh_certificate": {
            field: prefix_report["global_bvh_certificate"][field]
            for field in STABLE_GEOMETRY_FIELDS
        },
        "provenance": prefix_report["provenance"],
        "input_sha256": {
            label: _sha256(path) for label, path in paths.items()
        },
        "qualification": (
            "This is an exhaustive post-interaction raw-state and process/draw "
            "certificate for the specified 8,192-photon population. It is not "
            "a mathematical proof over every possible IEEE-754 input, source, "
            "or unsupported Chroma feature."
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
