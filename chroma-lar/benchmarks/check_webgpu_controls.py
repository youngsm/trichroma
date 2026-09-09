"""Exercise browser controls on a forced software adapter, without using CUDA."""

import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(args.bundle.resolve()))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=str(args.browser),
                headless=True,
                env={
                    **os.environ,
                    "VK_ICD_FILENAMES": str(args.browser.parent / "vk_swiftshader_icd.json"),
                },
                args=[
                    "--no-sandbox",
                    "--enable-gpu",
                    "--enable-unsafe-webgpu",
                    "--use-angle=vulkan",
                    "--use-vulkan=swiftshader",
                    "--use-webgpu-adapter=swiftshader",
                    "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
                ],
            )
            page = browser.new_page(viewport={"width": 1100, "height": 1000})
            page.on("pageerror", lambda error: print("Browser error:", error, flush=True))
            page.goto(f"http://localhost:{server.server_port}/?fallback=1&manual=1")
            adapter = page.evaluate("async()=>await window.trichromaReady")
            assert adapter["isFallbackAdapter"]
            page.evaluate('''() => {
                const originalRender = trichroma.render.bind(trichroma);
                trichroma.render = options => { window.lastRenderOptions = options; return originalRender(options); };
            }''')
            page.evaluate('document.getElementById("rays").value="625000"')
            page.select_option("#scene", "pixelPads.json")
            settled = (
                "!trichroma.busy && !trichroma.pending && trichroma.lastFrame?.rays === 625000"
            )
            try:
                page.wait_for_function(settled)
            except Exception:
                print(page.evaluate('({options:window.lastRenderOptions,rayValue:document.getElementById("rays").value,limit:trichroma.device.limits.maxTextureDimension2D,error:document.getElementById("error").textContent,status:document.getElementById("status").textContent,busy:trichroma.busy,pending:trichroma.pending,last:trichroma.lastFrame})'), flush=True)
                raise
            assert page.evaluate("trichroma.manifest.name") == "pixelPads"
            original_eye = page.evaluate("trichroma.camera.eye")
            page.mouse.move(500, 400)
            page.mouse.down()
            page.mouse.move(540, 410, steps=2)
            page.mouse.up()
            page.wait_for_function(settled)
            assert page.evaluate("trichroma.camera.eye") != original_eye
            distance = page.evaluate(
                "Math.hypot(...trichroma.camera.eye.map((x,i)=>x-trichroma.camera.target[i]))"
            )
            page.mouse.wheel(0, 150)
            page.wait_for_function(settled)
            assert (
                page.evaluate(
                    "Math.hypot(...trichroma.camera.eye.map((x,i)=>x-trichroma.camera.target[i]))"
                )
                > distance
            )
            page.select_option("#rays", "2500000")
            page.wait_for_function("!trichroma.busy && trichroma.lastFrame?.rays === 2500000")
            assert not page.evaluate("trichroma.errors")
            page.click("#reset")
            page.wait_for_function("!trichroma.busy && trichroma.lastFrame?.rays === 2500000")
            assert page.evaluate("trichroma.camera.eye") == original_eye
            preview_budget = page.evaluate("trichroma.previewRays")
            assert 1000 <= preview_budget <= 100000
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    report = dict(
        adapter=adapter,
        detector_selection=True,
        orbit=True,
        zoom=True,
        ray_count_change=[625_000, 2_500_000],
        reset=True,
        adaptive_preview_rays=preview_budget,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        asset_hashes={
            name: hashlib.sha256((args.bundle / name).read_bytes()).hexdigest()
            for name in ("index.html", "viewer.js", "trace.wgsl")
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
