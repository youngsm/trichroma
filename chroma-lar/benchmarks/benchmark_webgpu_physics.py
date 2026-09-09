"""Capture rendered browser optical events and measure warm compute/readback times.

This does not replace check_webgpu_physics.py's independent CPU validation.
Coordinate hardware use with other local GPU benchmarks. Software fallback is
explicit and is never labeled hardware performance.
"""

import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

import numpy as np


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--photons", type=int, default=100_000)
    parser.add_argument("--paths", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--browser", type=Path)
    args = parser.parse_args()
    if not 1 <= args.photons <= 30_000_000 or args.repeat < 1:
        parser.error("photons must be in 1..30,000,000 and repeat positive")
    from playwright.sync_api import sync_playwright

    if args.browser is None:
        candidates = sorted(
            Path.home().glob(
                ".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
            )
        )
        if not candidates:
            parser.error("pass --browser or install Playwright Chromium")
        args.browser = candidates[-1]
    flags = [
        "--no-sandbox",
        "--enable-gpu",
        "--enable-unsafe-webgpu",
        "--use-angle=vulkan",
        "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
    ]
    env = dict(os.environ)
    if args.hardware:
        flags += ["--use-vulkan=native", "--ignore-gpu-blocklist"]
    else:
        flags += ["--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader"]
        env["VK_ICD_FILENAMES"] = str(args.browser.parent / "vk_swiftshader_icd.json")
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((args.bundle / "physics-catalog.json").read_text())
    report = dict(
        hardware=args.hardware,
        photons=args.photons,
        paths=args.paths,
        repeat=args.repeat,
        seed=args.seed,
        scenes=[],
        asset_hashes={
            name: hashlib.sha256((args.bundle / name).read_bytes()).hexdigest()
            for name in ("physics.html", "physics.js", "physics.wgsl")
        },
        scene_manifests=catalog,
        timing_scope="Warm optical compute submission through queue completion; readback and JavaScript decoding are separately measured. Browser plotting/painting and initial pipeline creation excluded.",
    )
    root = Path(__file__).resolve().parents[2]
    report["source_hashes"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "chroma-lar/benchmarks/benchmark_webgpu_physics.py",
            "chroma-lar/chroma_lar/webgpu_physics.py",
            "chroma-lar/chroma_lar/optical_showcase.py",
        )
    }
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(QuietHandler, directory=str(args.bundle.resolve()))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True, executable_path=str(args.browser), args=flags, env=env
            )
            page = browser.new_page(viewport=dict(width=1500, height=1100))
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(
                f"http://localhost:{server.server_port}/physics.html?manual=1"
                + ("" if args.hardware else "&fallback=1")
            )
            report["adapter"] = page.evaluate("async()=>await opticalLabReady")
            report["browser_version"] = browser.version
            if bool(report["adapter"]["isFallbackAdapter"]) == args.hardware:
                raise RuntimeError(
                    "Requested hardware/software adapter mode did not match actual adapter"
                )
            for entry in catalog["scenes"]:
                name = entry["name"]
                # Retain a small actual native WGSL terminal batch for a separate CPU audit.
                debug = page.evaluate(
                    "async o=>await opticalLab.run(o)",
                    dict(scene=name, photons=4096, seed=args.seed, paths=64, debug=True),
                )
                np.savez_compressed(
                    args.output / f"{name}-debug.npz",
                    **{k: np.asarray(v) for k, v in debug["states"].items()},
                )
                options = dict(scene=name, photons=args.photons, seed=args.seed, paths=args.paths)
                page.evaluate("async o=>{await opticalLab.run(o);return true;}", options)
                frames = []
                for repeat in range(args.repeat):
                    options["seed"] = args.seed + repeat
                    row = page.evaluate(
                        "async o=>{const r=await opticalLab.run(o);return {metrics:r.metrics,counts:r.counts,histograms:r.histograms,photons:r.photons,seed:r.seed,pathCount:r.pathCount};}",
                        options,
                    )
                    frames.append(row)
                page.screenshot(path=str(args.output / f"{name}.png"), full_page=True)
                queue = [row["metrics"]["queue_ms"] for row in frames]
                result = dict(
                    name=name,
                    frames=frames,
                    mean_queue_ms=float(np.mean(queue)),
                    mean_readback_ms=float(np.mean([r["metrics"]["readback_ms"] for r in frames])),
                    mean_wall_ms=float(np.mean([r["metrics"]["wall_ms"] for r in frames])),
                    optical_photons_per_second=args.photons / (float(np.mean(queue)) / 1000),
                )
                report["scenes"].append(result)
                (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: v for k, v in result.items() if k != "frames"}), flush=True)
            report["page_errors"] = errors
            report["passed"] = not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("Browser errors occurred; inspect benchmark.json")


if __name__ == "__main__":
    main()
