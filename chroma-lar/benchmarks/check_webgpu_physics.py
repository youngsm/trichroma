"""Compare actual WGSL photon transport with the independent CPU optics oracle.

Uses an isolated loopback bundle and a forced SwiftShader adapter by default.
Hardware mode must be coordinated with other jobs on the same local GPU.
"""

import argparse
from functools import partial
import hashlib
from itertools import product
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading

import numpy as np

from chroma_lar.webgpu_physics import (
    check_browser_histograms,
    compare_browser_source,
    compare_ensemble,
    compare_recorded_paths,
    compare_terminal_states,
    cpu_reference,
    export_physics_scene,
    terminal_summary,
)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Directory containing physics.html/js/wgsl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--photons", type=int, default=4096)
    parser.add_argument("--paths", type=int, default=32)
    parser.add_argument("--seed", type=int, nargs="+", default=[901, 2027])
    parser.add_argument(
        "--polarization", choices=["random", "y", "z"], nargs="+", default=["random"]
    )
    args = parser.parse_args()
    if not 1 <= args.photons <= 65536 or not 1 <= args.paths <= min(args.photons, 2048):
        parser.error("require photons in 1..65536 and paths in 1..min(photons,2048)")
    from playwright.sync_api import sync_playwright

    browser_path = args.browser
    if browser_path is None:
        candidates = sorted(
            Path.home().glob(
                ".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
            )
        )
        if not candidates:
            parser.error("pass --browser /path/to/chromium or install Playwright Chromium")
        browser_path = candidates[-1]
    browser_env = dict(os.environ)
    flags = [
        "--no-sandbox",
        "--enable-gpu",
        "--enable-unsafe-webgpu",
        "--use-angle=vulkan",
        "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
    ]
    if args.hardware:
        flags += ["--use-vulkan=native", "--ignore-gpu-blocklist"]
    else:
        flags += ["--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader"]
        browser_env["VK_ICD_FILENAMES"] = str(browser_path.parent / "vk_swiftshader_icd.json")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sources = [
        "chroma-lar/chroma_lar/webgpu_physics.py",
        "chroma-lar/chroma_lar/optical_showcase.py",
        "chroma-lar/benchmarks/check_webgpu_physics.py",
        "chroma-lite/chroma/triton/spectral.py",
        "chroma-lite/chroma/triton/bvh.py",
        "chroma-lite/chroma/triton/boundary.py",
        "chroma-lite/chroma/triton/physics.py",
        "chroma-lite/chroma/triton/optical_response.py",
    ]
    report = dict(
        hardware=args.hardware,
        browser=str(browser_path),
        scenes=[],
        source_hashes={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
        },
        asset_hashes={
            name: hashlib.sha256((args.bundle / name).read_bytes()).hexdigest()
            for name in ("physics.html", "physics.js", "physics.wgsl")
        },
        comparison="Paired CPU/WGSL state discrepancies and independent marginal statistical checks; no universal bitwise claim",
    )
    all_passed = True
    with tempfile.TemporaryDirectory(prefix="trichroma-webgpu-physics-") as temporary:
        bundle = Path(temporary)
        for name in ("physics.html", "physics.js", "physics.wgsl"):
            shutil.copyfile(args.bundle / name, bundle / name)
        exports = {
            name: export_physics_scene(name, bundle)
            for name in ("prism", "fluorescence", "rayleigh")
        }
        catalog = dict(
            format="trichroma-webgpu-spectral-v1", scenes=[entry[0] for entry in exports.values()]
        )
        (bundle / "physics-catalog.json").write_text(json.dumps(catalog))
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(bundle)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True, executable_path=str(browser_path), args=flags, env=browser_env
                )
                page = browser.new_page(viewport=dict(width=1500, height=1100))
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(
                    f"http://localhost:{server.server_port}/physics.html?manual=1"
                    + ("" if args.hardware else "&fallback=1")
                )
                adapter = page.evaluate("async()=>await window.opticalLabReady")
                report["adapter"] = adapter
                report["browser_version"] = browser.version
                if bool(adapter["isFallbackAdapter"]) == args.hardware:
                    raise RuntimeError(
                        "browser adapter does not match requested hardware/fallback mode"
                    )
                goldens = json.loads((bundle / "physics-prism.json").read_text())["rng"]["goldens"]
                rng_words = page.evaluate("async g=>await opticalLab.checkRng(g)", goldens)
                expected_words = [row["float32_bits"] for row in goldens]
                report["rng"] = dict(
                    count=len(goldens),
                    passed=rng_words == expected_words,
                    actual=rng_words,
                    expected=expected_words,
                )
                all_passed &= report["rng"]["passed"]
                for name, (entry, fixture, scene) in exports.items():
                    rows = []
                    for seed, polarization in product(args.seed, args.polarization):
                        expected = cpu_reference(
                            name,
                            args.photons,
                            seed=seed,
                            polarization=polarization,
                            paths=args.paths,
                            fixture=fixture,
                            scene=scene,
                        )
                        actual_json = page.evaluate(
                            "async options=>JSON.stringify(await opticalLab.run(options))",
                            dict(
                                scene=name,
                                photons=args.photons,
                                seed=seed,
                                polarization=polarization,
                                paths=args.paths,
                                maxSteps=256,
                                debug=True,
                            ),
                        )
                        actual = json.loads(actual_json)
                        state = {
                            key: np.asarray(value) for key, value in actual.pop("states").items()
                        }
                        paired = compare_terminal_states(state, expected["result"])
                        marginal = compare_ensemble(
                            terminal_summary(state, name), expected["summary"]
                        )
                        recorded_paths = compare_recorded_paths(
                            actual["paths"], expected["paths"], state
                        )
                        histograms = check_browser_histograms(actual, state)
                        source = compare_browser_source(state, expected["batch"])
                        rows.append(
                            dict(
                                seed=seed,
                                polarization=polarization,
                                paired=paired,
                                ensemble=marginal,
                                cpu=expected["summary"],
                                browser=terminal_summary(state, name),
                                recorded_paths=recorded_paths,
                                histograms=histograms,
                                source=source,
                                metrics=actual["metrics"],
                            )
                        )
                        all_passed &= (
                            paired["all_within_tolerances"]
                            and marginal["passed"]
                            and recorded_paths["passed"]
                            and histograms["passed"]
                            and source["passed"]
                        )
                        if not paired["all_within_tolerances"]:
                            np.savez_compressed(
                                args.output / f"{name}-{seed}-{polarization}-disagreements.npz",
                                **{
                                    "cpu_" + key: value
                                    for key, value in expected["result"].final_state.items()
                                },
                                **{"browser_" + key: value for key, value in state.items()},
                            )
                        print(
                            json.dumps(
                                dict(
                                    name=name,
                                    seed=seed,
                                    polarization=polarization,
                                    paired=paired,
                                    ensemble_passed=marginal["passed"],
                                    paths=recorded_paths,
                                    histogram_passed=histograms["passed"],
                                )
                            ),
                            flush=True,
                        )
                    controls = page.evaluate("""()=>{
                        const lab=opticalLab, result=lab.result, canvas=document.getElementById('paths');
                        const before=canvas.toDataURL();
                        const sampledMax=Math.max(...result.paths.flatMap(p=>p.vertices.map(v=>v.time)));
                        const gate=document.getElementById('gate');gate.value=String(sampledMax/2);
                        gate.dispatchEvent(new Event('input'));
                        const changed=before!==canvas.toDataURL(), sameEvent=lab.result===result;
                        document.getElementById('allTimes').click();
                        return {changed,sameEvent,restored:before===canvas.toDataURL(),
                                photons:Number(document.getElementById('photons').value)===result.photons,
                                seed:Number(document.getElementById('seed').value)===result.seed,
                                polarization:document.getElementById('polarization').value===result.polarization};
                    }""")
                    all_passed &= all(controls.values())
                    page.screenshot(path=str(args.output / f"{name}.png"), full_page=True)
                    report["scenes"].append(dict(scene=entry, comparisons=rows, controls=controls))
                    (args.output / "verification.json").write_text(
                        json.dumps(report, indent=2, allow_nan=False) + "\n"
                    )
                report["page_errors"] = errors
                all_passed &= not errors
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
    report["passed"] = bool(all_passed)
    (args.output / "verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if not all_passed:
        raise SystemExit(
            "Browser optical validation found discrepancies; inspect verification.json and saved states"
        )


if __name__ == "__main__":
    main()
