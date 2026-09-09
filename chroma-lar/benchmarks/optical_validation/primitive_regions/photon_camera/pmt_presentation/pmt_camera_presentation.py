"""Present multiple cameras from one unchanged optical event; no timing claims."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import hashlib, json, threading, sys, shutil
from playwright.sync_api import sync_playwright

bundle = Path("notebooks/photon_camera_bundle").resolve()
out = Path(
    "chroma-lar/benchmarks/optical_validation/primitive_regions/photon_camera/pmt_presentation"
)
out.mkdir(parents=True, exist_ok=True)


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(bundle)))
threading.Thread(target=server.serve_forever, daemon=True).start()
assets = [
    "camera.html",
    "camera.js",
    "camera.wgsl",
    "deposition.wgsl",
    "physics.js",
    "physics.wgsl",
    "theme.css",
    "fonts.css",
    "camera-catalog.json",
]
catalog = json.loads((bundle / "camera-catalog.json").read_text())
assets += [x["manifest"] for x in catalog["scenes"]]
archive = out / "validated_assets"
archive.mkdir(exist_ok=True)
for name in assets:
    shutil.copyfile(bundle / name, archive / name)
report = {
    "scope": "Presentation-only camera changes reuse one fixed2.5M-photon event; queue times are diagnostic, not exclusive throughput.",
    "assets": {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in assets},
    "views": [],
}
with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        executable_path="/sdf/home/y/youngsam/.cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell",
        args=[
            "--no-sandbox",
            "--enable-gpu",
            "--enable-unsafe-webgpu",
            "--use-angle=vulkan",
            "--use-vulkan=native",
            "--ignore-gpu-blocklist",
            "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
        ],
    )
    page = browser.new_page(viewport={"width": 1500, "height": 1400})
    page.goto(f"http://localhost:{server.server_port}/camera.html?manual=1")
    report["adapter"] = page.evaluate("async()=>await photonCameraReady")
    report["event"] = page.evaluate(
        'async()=>await photonCamera.simulate({scene:"pmt",photons:2500000,seed:901})'
    )
    print("READY", report["event"]["counts"], flush=True)
    for line in sys.stdin:
        command = json.loads(line)
        if command.get("quit"):
            break
        name = command.pop("name")
        samples = command.pop("samples")
        frames = []
        for sample in range(samples):
            opts = (
                command if sample == 0 else {"width": command["width"], "height": command["height"]}
            )
            if sample == 0:
                opts = {**opts, "reset": True}
            frames.append(page.evaluate("async o=>await photonCamera.render(o)", opts))
        page.evaluate(
            's=>document.getElementById("samples").textContent=s',
            f'2,500,000 simulated photons · {samples} camera samples · {command["width"]} × {command["height"]}',
        )
        page.locator("#camera").screenshot(path=str(out / (name + "-camera.png")))
        page.screenshot(path=str(out / (name + "-page.png")), full_page=True)
        report["views"].append({"name": name, "options": command, "frames": frames})
        (out / "presentation.json").write_text(json.dumps(report, indent=2) + "\n")
        print("DONE", name, flush=True)
    browser.close()
server.shutdown()
print("CLOSED", flush=True)
