"""Final focused presentation/export smoke; software WebGPU, no timings."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import threading

sys.path.append(
    "/sdf/home/y/youngsam/sw/dune/.conda/envs/pointcept-torch2.5.0-cu12.4/lib/python3.10/site-packages"
)
from playwright.sync_api import sync_playwright
from chroma_lar.photon_camera import export_camera_bundle

output = Path(
    "chroma-lar/benchmarks/optical_validation/primitive_regions/photon_camera/pmt_presentation"
)
output.mkdir(parents=True, exist_ok=True)
binary = Path(
    "/sdf/home/y/youngsam/.cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell"
)


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


with tempfile.TemporaryDirectory(prefix="camera-presentation-ui-") as temporary:
    bundle = Path(temporary)
    export_camera_bundle(bundle)
    report = {
        "scope": "Final exported HTML, stylesheet, legend, exposure, and debug-decoder checks",
        "assets": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in bundle.iterdir()
            if p.is_file()
        },
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(bundle)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=str(binary),
                args=[
                    "--no-sandbox",
                    "--enable-gpu",
                    "--enable-unsafe-webgpu",
                    "--use-angle=vulkan",
                    "--use-vulkan=swiftshader",
                    "--use-webgpu-adapter=swiftshader",
                    "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
                ],
                env=dict(
                    os.environ, VK_ICD_FILENAMES=str(binary.parent / "vk_swiftshader_icd.json")
                ),
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://localhost:{server.server_port}/camera.html?manual=1&fallback=1")
            report["adapter"] = page.evaluate("async()=>await photonCameraReady")
            report["pmt"] = page.evaluate("""async()=>{
              const event=await photonCamera.simulate({scene:'pmt',photons:128,seed:901,debug:true});
              await photonCamera.render({width:64,height:40,reset:true,exposure:400000,cutaway:true});
              await document.fonts.ready;
              return {count:event.counts, triangle_ids:event.states.last_hit_triangles.length,
                legend_visible:!document.getElementById('pmt_legend').hidden,
                exposure:Number(document.getElementById('exposure').value),
                cutaway:document.getElementById('cutaway').checked,
                uv_false_color:document.getElementById('uv_false_color').checked,
                font:getComputedStyle(document.body).fontFamily,
                background:getComputedStyle(document.body).backgroundColor,
                finite:(await photonCamera.linearImage()).every(Number.isFinite),
                missing_detector_links:document.querySelectorAll('a[href="index.html"]').length};
            }""")
            pmt = report["pmt"]
            assert pmt["triangle_ids"] == 128 and pmt["legend_visible"] and pmt["cutaway"]
            assert abs(pmt["exposure"] - 5.6) < 0.001 and not pmt["uv_false_color"]
            assert pmt["finite"] and pmt["missing_detector_links"] == 0
            assert "Libertine" in pmt["font"]
            page.screenshot(path=str(output / "final-ui.png"), full_page=True)
            report["prism_legend_hidden"] = page.evaluate("""async()=>{
              await photonCamera.simulate({scene:'prism',photons:16,seed:901});
              return document.getElementById('pmt_legend').hidden;
            }""")
            assert report["prism_legend_hidden"] and not errors
            report["page_errors"] = errors
            browser.close()
    finally:
        server.shutdown()
    report["passed"] = True
    (output / "final_ui.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
