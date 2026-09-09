"""Measure complete camera-ray frames, including GPU-to-host RGB transfer."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import triton

from chroma_lar.viewer_examples import build_viewer_example
from chroma.triton.viewer import Camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector",
        choices=("theia", "reflect3wires", "reflect3wires-mesh", "pixelTPC", "pixelPads"),
        default="theia",
    )
    parser.add_argument("--radius", type=float, default=25500.0)
    parser.add_argument("--coverage", type=float, default=0.81)
    parser.add_argument("--diameter", type=float, default=508.0)
    parser.add_argument("--rays", type=int, default=2_500_000)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    example = build_viewer_example(
        args.detector, radius=args.radius, coverage=args.coverage, diameter=args.diameter
    )
    viewer = example.viewer(rays=args.rays)
    setup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    viewer.render_frame().png()
    viewer.render_frame(rays=100_000).png()
    warmup_seconds = time.perf_counter() - start
    target = np.asarray(example.camera.target, float)
    offset = np.asarray(example.camera.eye, float) - target
    orbit_distance = float(np.linalg.norm(offset))
    base_azimuth = float(np.rad2deg(np.arctan2(offset[1], offset[0])))
    elevation = float(np.rad2deg(np.arctan2(offset[2], np.linalg.norm(offset[:2]))))
    records = []
    for rays in (100_000, args.rays):
        frames = []
        for index in range(args.frames):
            camera = Camera.orbit(
                target,
                orbit_distance,
                azimuth=base_azimuth - 10.0 + index * 4.0,
                elevation=elevation,
            )
            frame = viewer.render_frame(camera, rays=rays, seed=index)
            start = time.perf_counter()
            png = frame.png()
            encode_seconds = time.perf_counter() - start
            frames.append(
                dict(
                    rays=frame.rays,
                    render_seconds=frame.seconds,
                    png_encode_seconds=encode_seconds,
                    png_bytes=len(png),
                    image_sha256=hashlib.sha256(frame.image.tobytes()).hexdigest(),
                )
            )
        total = sum(f["render_seconds"] for f in frames)
        records.append(
            dict(
                rays_per_frame=rays,
                frames=frames,
                sustained_rays_per_second=rays * len(frames) / total,
                mean_frame_seconds=total / len(frames),
                median_frame_seconds=float(np.median([f["render_seconds"] for f in frames])),
            )
        )
    (args.output / (args.detector + ".png")).write_bytes(png)
    root = Path(__file__).resolve().parents[2]
    hashes = {}
    for relative in (
        "chroma-lite/chroma/triton/viewer.py",
        "chroma-lar/chroma_lar/viewer_examples.py",
        "chroma-lar/chroma_lar/viewer_analytic.py",
        "chroma-lar/chroma_lar/triton_scene/intersect.py",
        "chroma-lar/chroma_lar/triton_scene/compiler.py",
        "chroma-lar/chroma_lar/geometry/build_larcube.py",
        "chroma-lar/chroma_lar/geometry/build_larcube_pixel.py",
        "chroma-lar/chroma_lar/geometry/pixelplane.py",
        "chroma-lar/chroma_lar/geometry/wireplane.py",
        "chroma-lar/chroma_lar/config/detector_config_reflect_reflect3wires.py",
        "chroma-lar/chroma_lar/config/detector_config_pixel.py",
        "chroma-lar/chroma_lar/config/detector_config.yaml",
        "chroma-lar/chroma_lar/config/detector_config_pixel.yaml",
        "chroma-lite/chroma/triton/examples/theia.py",
        "chroma-lite/chroma/triton/primitives.py",
        "chroma-lar/benchmarks/benchmark_viewer.py",
        "chroma-lite/chroma/triton/viewer_kernels.py",
        "chroma-lite/chroma/triton/bvh_kernels.py",
        "chroma-lite/chroma/triton/bvh.py",
        "chroma-lar/chroma_lar/triton_scene/instances.py",
        "chroma-lar/chroma_lar/triton_scene/primitive_adapter.py",
    ):
        hashes[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    report = dict(
        kind="opaque_geometry_camera_rays",
        optical_transport=False,
        includes=[
            "ray generation",
            "nearest triangle/analytic boundary traversal",
            "headlight shading",
            "antialias sample reduction",
            "overflow check",
            "RGB image download",
        ],
        excludes=[
            "setup/JIT/warmup",
            "PNG encoding (reported separately)",
            "widget transfer",
            "browser painting",
        ],
        detector=example.metadata,
        device=torch.cuda.get_device_name(),
        python=platform.python_version(),
        torch=torch.__version__,
        triton=triton.__version__,
        setup_seconds=setup_seconds,
        warmup_seconds=warmup_seconds,
        records=records,
        source_hashes=hashes,
    )
    (args.output / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("detector", "device", "setup_seconds", "warmup_seconds")},
            indent=2,
        )
    )
    for record in records:
        print(
            f"{record['rays_per_frame']:,} rays: {record['mean_frame_seconds']*1000:.2f} ms/frame, "
            f"{record['sustained_rays_per_second']/1e6:.2f}M camera rays/s"
        )


if __name__ == "__main__":
    main()
