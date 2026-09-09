"""Render real photon-camera scenes and audit large-event map accumulation.

The hardware GPU must be coordinated with other jobs. Queue timings are image
review diagnostics, not sustained throughput claims. Full-count source energy
is checked independently after the browser closes.
"""

import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import shutil
from pathlib import Path
import threading

from chroma_lar.photon_camera import camera_source_energy_sum


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--photons", type=int, default=2_500_000)
    parser.add_argument("--energy-counts", type=int, nargs="*", default=[])
    parser.add_argument("--brightness-counts", type=int, nargs="*", default=[])
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument(
        "--completion-batches",
        type=int,
        default=1,
        help="Validate additional full-count events with consecutive seeds",
    )
    parser.add_argument("--scene", nargs="+", default=["prism", "fluorescence", "rayleigh"])
    parser.add_argument("--browser", type=Path)
    args = parser.parse_args()
    for count in [args.photons, *args.energy_counts, *args.brightness_counts]:
        if not 1 <= count <= 30_000_000:
            parser.error("photon counts must be in1..30,000,000")
    if args.completion_batches < 1:
        parser.error("completion batches must be positive")
    if args.samples < 1 or not 8 <= args.width <= 1920:
        parser.error("positive samples and width8..1920 required")
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
    root = Path(__file__).resolve().parents[2]
    asset_names = [
        "physics.wgsl",
        "physics.js",
        "deposition.wgsl",
        "camera.wgsl",
        "camera.js",
        "camera.html",
        "theme.css",
        "fonts.css",
    ]
    preserved = args.output / "validated_assets"
    preserved.mkdir(exist_ok=True)
    catalog = json.loads((args.bundle / "camera-catalog.json").read_text())
    for name in [
        *asset_names,
        "camera-catalog.json",
        *(entry["manifest"] for entry in catalog["scenes"]),
    ]:
        shutil.copyfile(args.bundle / name, preserved / name)
    source_names = [
        "chroma-lar/benchmarks/benchmark_photon_camera.py",
        "chroma-lar/chroma_lar/photon_camera.py",
        "chroma-lar/chroma_lar/pmt_camera.py",
        "chroma-lar/chroma_lar/geometry/pmt.py",
        "chroma-lar/chroma_lar/optical_calibration.py",
        "chroma-lar/benchmarks/optical_validation/full_detector_synthetic_calibration_noise.json",
        "chroma-lite/chroma/triton/bvh.py",
        "chroma-lite/chroma/triton/webgpu/export.py",
        "chroma-lar/chroma_lar/webgpu_physics.py",
        "chroma-lar/chroma_lar/optical_showcase.py",
    ]
    report = dict(
        hardware=args.hardware,
        photons=args.photons,
        seed=args.seed,
        width=args.width,
        samples=args.samples,
        scenes=[],
        energy_checks=[],
        completion_checks=[],
        brightness_checks=[],
        source_hashes={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_names
        },
        asset_hashes={
            name: hashlib.sha256((args.bundle / name).read_bytes()).hexdigest()
            for name in asset_names
        },
        catalog=json.loads((args.bundle / "camera-catalog.json").read_text()),
        timing_scope="Diagnostic queue times during image review; other correctness jobs may share this GPU. No sustained performance or browser FPS claim.",
    )
    save = lambda: (args.output / "capture.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(QuietHandler, directory=str(args.bundle.resolve()))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True, executable_path=str(args.browser), args=flags, env=env
            )
            page = browser.new_page(viewport=dict(width=1500, height=1400))
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(
                f"http://localhost:{server.server_port}/camera.html?manual=1"
                + ("" if args.hardware else "&fallback=1")
            )
            report["adapter"] = page.evaluate("async()=>await photonCameraReady")
            report["browser_version"] = browser.version
            if bool(report["adapter"]["isFallbackAdapter"]) == args.hardware:
                raise RuntimeError("Actual adapter does not match requested hardware/software mode")
            print("ADAPTER", report["adapter"], flush=True)
            for name in args.scene:
                counts = sorted(
                    set(
                        [
                            args.photons,
                            *args.brightness_counts,
                            *(args.energy_counts if name in ("prism", "rayleigh") else []),
                        ]
                    )
                )
                for count in counts:
                    event = page.evaluate(
                        "async o=>JSON.stringify(await photonCamera.simulate(o))",
                        dict(scene=name, photons=count, seed=args.seed),
                    )
                    event = json.loads(event)
                    summary = page.evaluate("async()=>await photonCamera.mapSummary()")
                    record = dict(name=name, photons=count, event=event, map_summary=summary)
                    print("EVENT", name, count, event["counts"], event["metrics"], flush=True)
                    if count in args.brightness_counts:
                        for sample in range(16):
                            page.evaluate(
                                "async()=>await photonCamera.render({width:128,height:80})"
                            )
                        record["raw_brightness"] = page.evaluate("""async()=>{
                            const values=await photonCamera.linearImage(); let rgb=[0,0,0];
                            for(let i=0;i<values.length;i+=4) for(let c=0;c<3;c++) rgb[c]+=values[i+c];
                            rgb=rgb.map(v=>v/(values.length/4));
                            return {mean_linear_rgb:rgb,mean_luminance:.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2],
                                width:128,height:80,camera_samples:16};
                        }""")
                    if count == args.photons:
                        frames = []
                        for sample in range(args.samples):
                            frames.append(
                                page.evaluate(
                                    "async o=>await photonCamera.render(o)",
                                    dict(
                                        width=args.width, height=max(8, round(args.width * 0.625))
                                    ),
                                )
                            )
                        record["frames"] = frames
                        page.evaluate(
                            "text=>document.getElementById('samples').textContent=text",
                            f"{count:,} simulated photons · {args.samples} camera samples · {args.width} × {round(args.width*.625)}",
                        )
                        page.locator("#camera").screenshot(
                            path=str(args.output / f"{name}-camera.png")
                        )
                        page.screenshot(path=str(args.output / f"{name}-page.png"), full_page=True)
                        if name == "pmt":
                            record["inspection_views"] = []
                            for label, false_color in (
                                ("pmt-cutaway", False),
                                ("pmt-cutaway-uv", True),
                            ):
                                view_frames = []
                                for sample in range(args.samples):
                                    view_frames.append(
                                        page.evaluate(
                                            "async o=>await photonCamera.render(o)",
                                            dict(
                                                width=args.width,
                                                height=max(8, round(args.width * 0.625)),
                                                reset=sample == 0,
                                                cutaway=True,
                                                uvFalseColor=false_color,
                                            ),
                                        )
                                    )
                                page.locator("#camera").screenshot(
                                    path=str(args.output / f"{label}-camera.png")
                                )
                                page.screenshot(
                                    path=str(args.output / f"{label}-page.png"), full_page=True
                                )
                                record["inspection_views"].append(
                                    dict(
                                        name=label,
                                        frames=view_frames,
                                        description="Diagnostic clipping of unchanged physical photon maps; violet UV display is false color when enabled.",
                                    )
                                )
                    report["scenes"].append(record)
                    save()
            for batch in range(1, args.completion_batches):
                for name in args.scene:
                    event = json.loads(
                        page.evaluate(
                            "async o=>JSON.stringify(await photonCamera.simulate(o))",
                            dict(scene=name, photons=args.photons, seed=args.seed + batch),
                        )
                    )
                    report["completion_checks"].append(event)
                    print(
                        "COMPLETION",
                        name,
                        args.photons,
                        args.seed + batch,
                        event["counts"],
                        flush=True,
                    )
                    save()
            report["page_errors"] = errors
            browser.close()
            print("GPU browser closed; independent source-energy audit now runs on CPU", flush=True)
    finally:
        server.shutdown()
        server.server_close()
    for case in report["scenes"]:
        if case["name"] not in ("prism", "rayleigh"):
            continue
        expected = camera_source_energy_sum(case["name"], case["photons"], seed=args.seed)
        actual = case["map_summary"]["wall_energy"]
        relative = abs(actual - expected) / expected
        row = dict(
            scene=case["name"],
            photons=case["photons"],
            expected_source_energy=expected,
            deposited_wall_energy=actual,
            relative_error=relative,
            relative_tolerance=1e-4,
            passed=relative <= 1e-4,
        )
        report["energy_checks"].append(row)
        print("ENERGY", row, flush=True)
        save()
    for name in args.scene:
        comparisons = sorted(
            (row for row in report["scenes"] if row["name"] == name and "raw_brightness" in row),
            key=lambda row: row["photons"],
        )
        if len(comparisons) >= 2:
            low, high = comparisons[0], comparisons[-1]
            ratio = (
                high["raw_brightness"]["mean_luminance"] / low["raw_brightness"]["mean_luminance"]
            )
            row = dict(
                scene=name,
                low_count=low["photons"],
                high_count=high["photons"],
                mean_luminance_ratio=ratio,
                relative_tolerance=0.03,
                passed=abs(ratio - 1) <= 0.03,
            )
            report["brightness_checks"].append(row)
            print("BRIGHTNESS", row, flush=True)
    report["passed"] = (
        all(row["passed"] for row in report["brightness_checks"])
        and not report["page_errors"]
        and all(row["passed"] for row in report["energy_checks"])
    )
    save()
    if not report["passed"]:
        raise SystemExit("Camera capture or energy audit failed; inspect capture.json")


if __name__ == "__main__":
    main()
