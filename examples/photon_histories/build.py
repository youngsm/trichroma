"""Package the browser photon-history scene with the shared Theia geometry."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def build(destination, geometry):
    here = Path(__file__).resolve().parent
    assets = here.parents[1] / "chroma-lite/chroma/triton/webgpu/assets"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "event.js", "event_pose.js", "event.wgsl", "events.json"):
        shutil.copyfile(here / name, destination / name)
    for name in ("trace.wgsl", "gpu.js", "scheduler.js", "theme.css", "fonts.css"):
        shutil.copyfile(assets / name, destination / name)
    manifest = json.loads((geometry / "theia.json").read_text())
    binary = manifest["binary"].split("?")[0]
    shutil.copyfile(geometry / binary, destination / binary)
    manifest["binary"] = binary
    (destination / "theia.json").write_text(json.dumps(manifest, indent=2) + "\n")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
    version = commit[:12]
    for path in destination.glob("*.html"):
        path.write_text(re.sub(r'((?:src|href)="[^"?]+\.(?:js|css))"',
                               lambda m: m[1] + '?v=' + version + '"', path.read_text()))
    for path in destination.glob("*.js"):
        path.write_text(re.sub(r'''(from\s+['"])(\./[\w-]+\.js)(['"])''',
                               lambda m: m[1] + m[2] + '?v=' + version + m[3], path.read_text()))
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.iterdir()
        if p.is_file() and p.name != "source.json"
    }
    (destination / "source.json").write_text(
        json.dumps({"source_repository": "https://github.com/youngsm/trichroma",
                    "source_commit": commit, "source_directory": "examples/photon_histories",
                    "sha256": files}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--geometry",
        type=Path,
        required=True,
        help="Existing exported Theia browser bundle",
    )
    args = parser.parse_args()
    build(args.destination, args.geometry)
