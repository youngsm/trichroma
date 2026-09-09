"""Export recorded HK particle steps; reconstruct new visible Cherenkov emission.

The source is opened read-only. Only a rigid transform, m -> mm conversion and
subtraction of event t0 are applied to geometry/times. No track is stretched.
"""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def export(base, output):
    events = []
    for config, name, kind in [
        ("01", "event_327", "muon"),
        ("03", "event_072", "electron"),
    ]:
        root = base / ("config_" + config)
        with h5py.File(root / "step/wc_step_0000.h5", "r") as sf, h5py.File(
            root / "labl/wc_labl_0000.h5", "r"
        ) as lf:
            s, truth = sf[name], lf[name]
            assert s.attrs["source_event_idx"] == truth.attrs["source_event_idx"]
            start = np.column_stack([s["start_" + a][:] for a in "xyz"]).astype(float)
            end = np.column_stack([s["end_" + a][:] for a in "xyz"]).astype(float)
            direction = np.column_stack([s["dir_" + a][:] for a in "xyz"]).astype(float)
            norms = np.linalg.norm(direction, axis=1)
            direction /= np.maximum(norms[:, None], 1e-30)
            axis = direction[0]
            helper = np.eye(3)[np.argmin(np.abs(axis))]
            right = np.cross(helper, axis)
            right /= np.linalg.norm(right)
            rotation = np.stack((right, np.cross(axis, right), axis))
            origin = start[0].copy()
            a, b, d = (
                (start - origin) @ rotation.T * 1000,
                (end - origin) @ rotation.T * 1000,
                direction @ rotation.T,
            )
            # Center the event's longitudinal extent, preserving every relative position.
            shift = (float(a[:, 2].min()) + float(b[:, 2].max())) / 2
            a[:, 2] -= shift
            b[:, 2] -= shift
            beta = s["beta_start"][:].astype(float)
            time = s["time"][:].astype(float) - float(truth["per_event/t0"][()])
            track = s["track_idx"][:]
            pdg = truth["per_track/pdg"][:][track]
            # All emitting species in these two samples have unit charge.
            charged = np.isin(np.abs(pdg), [11, 13, 211, 321, 2212])
            wavelengths = np.arange(360.5, 700, 1)
            n = 1.322 + 3000 / wavelengths**2
            q = np.maximum(0, 1 - 1 / np.maximum((beta[:, None] * n) ** 2, 1e-30))
            yield_per_mm = (
                (q / wavelengths**2).sum(axis=1) * 2 * np.pi / 137.035999084 * 1e6
            )
            lengths = np.linalg.norm(b - a, axis=1)
            weights = yield_per_mm * lengths * charged * (norms > 0)
            mask = weights > 0
            cdf = np.cumsum(weights[mask])
            total = float(cdf[-1])
            cdf /= total
            cdf[-1] = 1
            rows = np.column_stack(
                (a[mask], time[mask], b[mask], beta[mask], d[mask], cdf)
            )
            raw = np.column_stack((a, b, d, beta, time, track, s["n_cherenkov"][:]))
            assert np.allclose(
                lengths, np.linalg.norm(end - start, axis=1) * 1000, atol=1e-8
            )
            assert np.isfinite(rows).all() and (time >= -1e-4).all()
            events.append(
                dict(
                    id=kind,
                    energy_MeV=float(truth["per_track/initial_energy"][0]),
                    pdg=int(truth["per_track/pdg"][0]),
                    name=name,
                    config="config_" + config,
                    source_event_idx=int(s.attrs["source_event_idx"]),
                    source_file=str(root / "step/wc_step_0000.h5"),
                    source_run=str(sf["config"].attrs["run_id"]),
                    source_git=str(sf["config"].attrs["git_commit"]),
                    original_t0_ns=float(truth["per_event/t0"][()]),
                    origin_m=origin.tolist(),
                    rotation=rotation.tolist(),
                    longitudinal_shift_mm=shift,
                    tracks=int(truth.attrs["n_tracks"]),
                    input_steps=len(a),
                    emitting_steps=int(mask.sum()),
                    source_steps_sha256=hashlib.sha256(
                        raw.astype("<f8").tobytes()
                    ).hexdigest(),
                    g4_photons=int(s["n_cherenkov"][:].sum()),
                    total_yield=total,
                    rows=rows.tolist(),
                    source_step_indices=np.flatnonzero(mask).tolist(),
                    prompt_duration_ns=float(np.percentile(time[mask], 95)),
                    latest_emission_ns=float(time[mask].max()),
                    extent_mm=[a.min(axis=0).tolist(), b.max(axis=0).tolist()],
                )
            )
    result = dict(
        format="chroma-hk-photon-histories-v1",
        units={"position": "mm", "time": "ns", "wavelength": "nm"},
        dataset="DORAEMON/WAND/HK/GeV/test",
        transform="Rigid rotation and translation; m to mm; subtract recorded event t0. No track rescaling.",
        emission="New Frank-Tamm sampling over 360-700 nm; source G4 photon wavelength band unconfirmed.",
        events=events,
    )
    output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                {k: v for k, v in e.items() if k not in ("rows", "source_step_indices")}
                for e in events
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    export(
        Path("/sdf/data/neutrino/cjesus/DORAEMON/WAND/HK/GeV/test"),
        Path(__file__).with_name("events.json"),
    )
