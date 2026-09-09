"""Export browser detector views and three optical demonstrations; export uses only CPU."""

import argparse
from pathlib import Path

from chroma_lar.viewer_examples import build_viewer_example
from chroma_lar.webgpu_physics import export_physics_catalog
from chroma_lar.photon_camera import export_camera_catalog
from chroma.triton.webgpu.export import export_example, copy_browser_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--detector", action="append", choices=("theia", "pixelTPC", "pixelPads", "reflect3wires")
    )
    parser.add_argument("--small-theia", action="store_true")
    args = parser.parse_args()
    scenes = []
    for name in args.detector or ["theia", "reflect3wires", "pixelTPC"]:
        options = (
            dict(radius=5000.0, coverage=0.05, diameter=304.8)
            if name == "theia" and args.small_theia
            else {}
        )
        source_name = {"reflect3wires": "reflect3wires-mesh", "pixelTPC": "pixelTPC-resolved"}.get(name, name)
        example = build_viewer_example(source_name, **options)
        scene, groups = export_example(example, args.output, name=name, compress=True)
        print(
            f"{name}: {scene['instances']:,} instances, {scene['mesh_triangles']:,} shared triangles, {scene['byte_length']/1e6:.2f} MB"
        )
        scenes.append(scene)
    copy_browser_assets(args.output, scenes)
    export_physics_catalog(args.output)
    export_camera_catalog(args.output)
    print(f"Serve with: python -m http.server 8765 --bind 127.0.0.1 --directory {args.output}")
    print(
        "Open /camera.html for the photon studio, /physics.html for diagnostics, or /index.html for detector geometry."
    )
    print(
        "If serving remotely, forward port 8765 over SSH before opening localhost in your browser."
    )


if __name__ == "__main__":
    main()
