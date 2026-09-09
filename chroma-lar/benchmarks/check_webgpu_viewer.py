"""Validate the browser renderer against independent float64 mesh queries.

Starts a loopback-only HTTP server and Chromium. Software mode explicitly
restricts Vulkan to Chromium's SwiftShader ICD. Hardware mode must be run
separately from CUDA benchmarks sharing the same device.
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

from chroma.triton.webgpu.export import pack_groups, prepare_groups
from chroma.triton.webgpu.reference import camera_rays, nearest_reference
from chroma_lar.viewer_examples import build_viewer_example


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument(
        "--full-frame", action="store_true", help="Measure 2.5M rays rather than a 100k-ray preview"
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--debug-width", type=int, default=64)
    parser.add_argument("--debug-height", type=int, default=40)
    parser.add_argument("--detector", action="append")
    parser.add_argument("--view")
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    browser_path = args.browser
    if browser_path is None:
        candidates = sorted(
            Path.home().glob(
                ".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
            )
        )
        if not candidates:
            parser.error("Pass --browser /path/to/chromium or install Playwright Chromium")
        browser_path = candidates[-1]
    flags = [
        "--no-sandbox",
        "--enable-gpu",
        "--enable-unsafe-webgpu",
        "--use-angle=vulkan",
        "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
    ]
    browser_env = dict(os.environ)
    if args.hardware:
        flags += ["--use-vulkan=native", "--ignore-gpu-blocklist"]
    else:
        flags += ["--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader"]
        browser_env["VK_ICD_FILENAMES"] = str(browser_path.parent / "vk_swiftshader_icd.json")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(QuietHandler, directory=str(args.bundle.resolve()))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True, executable_path=str(browser_path), args=flags, env=browser_env
            )
            page = browser.new_page(viewport={"width": 1100, "height": 1000})
            page.goto(
                f"http://localhost:{server.server_port}/?manual=1"
                + ("" if args.hardware else "&fallback=1")
            )
            adapter = page.evaluate("async()=>await window.trichromaReady")
            if args.hardware and (
                adapter["isFallbackAdapter"] or adapter["architecture"] == "swiftshader"
            ):
                raise RuntimeError("Hardware requested but browser selected a software adapter")
            for entry in json.loads((args.bundle / "catalog.json").read_text()):
                if args.detector and entry["name"] not in args.detector:
                    continue
                manifest = page.evaluate(
                    "async path=>await window.trichroma.loadScene(path)", entry["manifest"]
                )
                if args.view:
                    manifest["camera"] = manifest["views"][args.view]
                    page.evaluate("camera=>trichroma.camera=structuredClone(camera)", manifest["camera"])
                options = (
                    {
                        key: manifest["geometry"][field]
                        for key, field in (
                            ("radius", "radius_mm"),
                            ("coverage", "coverage"),
                            ("diameter", "diameter_mm"),
                        )
                    }
                    if manifest["name"] == "theia"
                    else {}
                )
                example = build_viewer_example(manifest["geometry"]["name"], **options)
                groups = prepare_groups(example.geometry, example.solid_colors,
                                        None if example.export_groups is None else example.export_groups())
                rebuilt_hash = hashlib.sha256(pack_groups(groups).tobytes()).hexdigest()
                if rebuilt_hash != manifest["sha256"]:
                    raise RuntimeError(
                        "Current geometry does not match exported scene; re-export before verification"
                    )
                frame = page.evaluate(
                    "async options=>await window.trichroma.render(options)",
                    dict(width=args.debug_width, height=args.debug_height,
                         rays=args.debug_width*args.debug_height, debug=True, jitter=False),
                )
                actual = np.asarray(frame.pop("diagnostic")).reshape(-1, 12)
                origins, directions = camera_rays(manifest["camera"], args.debug_width, args.debug_height)
                np.testing.assert_allclose(actual[:, 8:11], directions, atol=2e-7, rtol=0)
                oracle_source = (
                    Path(__file__).resolve().parents[2]
                    / "chroma-lite/chroma/triton/webgpu/reference.py"
                )
                oracle_key = hashlib.sha256(
                    manifest["sha256"].encode()
                    + origins.tobytes()
                    + actual[:, 8:11].tobytes()
                    + oracle_source.read_bytes()
                ).hexdigest()
                oracle_cache = args.output / (manifest["name"] + "_reference.npz")
                cached = False
                if oracle_cache.exists():
                    with np.load(oracle_cache, allow_pickle=False) as cache:
                        if str(cache["key"]) == oracle_key:
                            expected = cache["expected"]
                            cached = True
                if not cached:
                    expected = nearest_reference(groups, origins, actual[:, 8:11])
                    np.savez_compressed(oracle_cache, key=oracle_key, expected=expected)
                np.savez_compressed(args.output / (manifest["name"] + "_hits.npz"),
                                    actual=actual, expected=expected, origins=origins)
                np.testing.assert_array_equal(actual[:, 1:4], expected[:, 1:4])
                np.testing.assert_array_equal(actual[:, 7], expected[:, 7])
                mask = expected[:, 7] != 0
                np.testing.assert_allclose(actual[mask, 0], expected[mask, 0], atol=0.01, rtol=4e-6)
                np.testing.assert_allclose(
                    actual[mask, 4:7], expected[mask, 4:7], atol=2e-5, rtol=0
                )
                dimensions = (
                    dict(width=1000, height=625, rays=2_500_000)
                    if args.full_frame
                    else dict(width=400, height=250, rays=100_000)
                )
                await_render = "async options=>await window.trichroma.render(options)"
                page.evaluate(await_render, dimensions)
                frames = [
                    page.evaluate(await_render, dict(**dimensions, seed=seed + 20))
                    for seed in range(args.repeat)
                ]
                page.locator("#canvas").screenshot(
                    path=str(args.output / (manifest["name"] + ".png"))
                )
                from PIL import Image

                pixels = np.asarray(Image.open(args.output / (manifest["name"] + ".png")))[:, :, :3]
                assert (
                    np.std(pixels.astype(float), axis=(0, 1)).max() > 5
                ), "Browser canvas is blank or uniform"
                report = dict(
                    scene=manifest,
                    oracle="Independent float64 Moller-Trumbore leaves on float32 exported triangles/transforms; no exported BVH traversal",
                    compared_rays=len(actual),
                    max_camera_direction_error=float(np.max(np.abs(actual[:, 8:11] - directions))),
                    hit_identity_mismatches=0,
                    max_distance_error_mm=float(
                        np.max(np.abs(actual[mask, 0] - expected[mask, 0]), initial=0)
                    ),
                    max_normal_error=float(
                        np.max(np.abs(actual[mask, 4:7] - expected[mask, 4:7]), initial=0)
                    ),
                    frames=frames,
                    canvas_nonuniform=True,
                )
                reports.append(report)
                print(
                    f"{manifest['name']}: {len(actual)} exact hit IDs; max distance error {report['max_distance_error_mm']:.6g} mm; {dimensions['rays']:,} rays mean {np.mean([f['milliseconds'] for f in frames]):.3f} ms",
                    flush=True,
                )
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    root = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__).resolve(),
        root / "chroma-lite/chroma/triton/webgpu/export.py",
        root / "chroma-lite/chroma/triton/webgpu/reference.py",
        root / "chroma-lite/chroma/triton/viewer.py",
        root / "chroma-lite/chroma/triton/examples/theia.py",
        root / "chroma-lar/chroma_lar/viewer_examples.py",
        root / "chroma-lite/chroma/triton/bvh.py",
        root / "chroma-lite/chroma/triton/primitives.py",
        root / "chroma-lar/chroma_lar/triton_scene/instances.py",
        root / "chroma-lar/chroma_lar/triton_scene/primitive_adapter.py",
        root / "chroma-lar/chroma_lar/geometry/build_larcube_pixel.py",
        root / "chroma-lar/chroma_lar/geometry/pixelplane.py",
        root / "chroma-lar/chroma_lar/config/detector_config_pixel.py",
        root / "chroma-lar/chroma_lar/config/detector_config_pixel.yaml",
    ]
    report = dict(
        adapter=adapter,
        hardware=args.hardware,
        browser=str(browser_path),
        browser_version=browser.version,
        geometry_rendering_only=True,
        browser_canvas_verified=True,
        source_hashes={
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        asset_hashes={
            name: hashlib.sha256((args.bundle / name).read_bytes()).hexdigest()
            for name in ("index.html", "viewer.js", "trace.wgsl")
        },
        scenes=reports,
    )
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
