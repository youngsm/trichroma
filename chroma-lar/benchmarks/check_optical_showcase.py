"""Render all optical-playground scenes and optionally exercise notebook controls."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--photons", type=int, default=2_500_000)
    parser.add_argument("--paths", type=int, default=512)
    parser.add_argument("--notebook", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from chroma.event import RAYLEIGH_SCATTER, SURFACE_REEMIT, SURFACE_DETECT, REFLECT_SPECULAR
    from chroma_lar.optical_showcase import OpticalShowcase, SCENES
    import torch

    root = Path(__file__).resolve().parents[2]
    report = dict(
        host=platform.node(),
        device=torch.cuda.get_device_name(),
        scenes=[],
        calibration="synthetic illustrative optical tables",
        trajectory_prefix_endpoint_exact=True,
        camera_radiance=False,
        source_sha256={},
    )
    for relative in (
        "chroma-lar/chroma_lar/optical_showcase.py",
        "notebooks/optical_showcase.ipynb",
        "chroma-lite/chroma/triton/spectral.py",
        "chroma-lite/chroma/triton/spectral_kernels.py",
        "chroma-lite/chroma/triton/bvh.py",
        "chroma-lite/chroma/triton/bvh_kernels.py",
        "chroma-lite/chroma/triton/boundary.py",
        "chroma-lite/chroma/triton/boundary_kernels.py",
        "chroma-lite/chroma/triton/physics.py",
        "chroma-lite/chroma/triton/physics_kernels.py",
    ):
        report["source_sha256"][relative] = hashlib.sha256(
            (root / relative).read_bytes()
        ).hexdigest()
    lab = OpticalShowcase()
    for name in SCENES:
        event = lab.run(name, args.photons, paths=args.paths, seed=901)
        flags = event.result.final_state["flags"]
        row = dict(
            name=name,
            photons=event.photon_count,
            drawn_photons=len(event.paths.photon_ids),
            scene_fingerprint=event.result.scene_fingerprint,
            source_seconds=event.source_seconds,
            transport_seconds=event.transport_seconds,
            trace_seconds=event.trace_seconds,
            photons_per_second=event.photons_per_second,
            trajectory_snapshots=len(event.paths.positions),
            seed=event.seed,
            step_limit_count=event.result.step_limit_count,
        )
        for label, mask in (
            ("detected", SURFACE_DETECT),
            ("scattered", RAYLEIGH_SCATTER),
            ("reemitted", SURFACE_REEMIT),
            ("reflected", REFLECT_SPECULAR),
        ):
            row[label] = int(np.count_nonzero(flags & mask))
        if name == "fluorescence":
            row["delay_quantiles_ns"] = np.quantile(
                event.fluorescence_delays, [0, 0.25, 0.5, 0.9, 1]
            ).tolist()
            row["input_delay_cdf"] = [
                event.scene.fluorescence_time.x.tolist(),
                event.scene.fluorescence_time.cdf.tolist(),
            ]
        lab.save_figure(args.output_dir / f"{name}.png")
        report["scenes"].append(row)
        print(json.dumps(row), flush=True)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lab.close()
    if args.notebook:
        import nbformat
        from nbclient import NotebookClient

        notebook = nbformat.read(root / "notebooks/optical_showcase.ipynb", as_version=4)
        notebook.cells.append(
            nbformat.v4.new_code_cell("""assert lab.event.photon_count == 2_500_000
assert bytes(lab.controls["image"].value).startswith(b"\\x89PNG")
for name in ("fluorescence", "rayleigh", "prism"):
    lab.controls["experiment"].value = name
    lab.controls["photons"].value = 10_000
    lab.controls["paths"].value = 32
    lab.controls["run"].click()
    assert lab.event.scene.name == name
    assert lab.event.photon_count == 10_000
    assert len(lab.event.paths.photon_ids) == 32
    assert "failed" not in lab.controls["status"].value.lower()
    assert bytes(lab.controls["image"].value).startswith(b"\\x89PNG")
before = lab.event
lab.controls["all_times"].value = False
lab.controls["time"].value = 1.0
assert lab.event is before
lab.controls["all_times"].value = True
lab.close()
print("All scene, photon-count, path-count and time-gate callbacks passed")""")
        )
        NotebookClient(
            notebook,
            timeout=300,
            kernel_name="trichroma-gpu",
            resources={"metadata": {"path": str(root)}},
        ).execute()
        report["notebook"] = dict(
            kernel="trichroma-gpu",
            default_2_5m_run=True,
            all_scene_callbacks=True,
            count_control=True,
            path_count_control=True,
            time_gate_without_resimulation=True,
            browser_painting_tested=False,
        )
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Notebook smoke passed", flush=True)


if __name__ == "__main__":
    main()
