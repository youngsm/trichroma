"""Execute the camera notebook and exercise its trusted, self-contained iframe."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    browser_path = sorted(
        Path.home().glob(
            ".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
        )
    )[-1]
    with tempfile.TemporaryDirectory(prefix="trichroma-camera-notebook-") as directory:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        environment = dict(os.environ, JUPYTER_RUNTIME_DIR=directory)
        prefix = f"http://localhost:{port}/studio-test/"
        with open(Path(directory) / "server.log", "w") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "jupyter_server",
                    "--ServerApp.ip=127.0.0.1",
                    f"--ServerApp.port={port}",
                    "--ServerApp.port_retries=0",
                    "--ServerApp.base_url=/studio-test/",
                    f"--ServerApp.root_dir={root}",
                    "--ServerApp.open_browser=False",
                    "--IdentityProvider.token=",
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                for _ in range(200):
                    if server.poll() is not None:
                        raise RuntimeError((Path(directory) / "server.log").read_text())
                    try:
                        with urlopen(prefix + "api/status", timeout=0.5) as response:
                            assert response.status == 200
                        break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise RuntimeError("Jupyter server did not become ready")
                notebook = nbformat.read(root / "notebooks/photon_camera.ipynb", as_version=4)
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        cell.source = cell.source.replace(
                            "notebook_camera_html(output)",
                            "notebook_camera_html(output, manual=True, software=True)",
                        )
                old_runtime = os.environ.get("JUPYTER_RUNTIME_DIR")
                os.environ["JUPYTER_RUNTIME_DIR"] = directory
                try:
                    NotebookClient(
                        notebook,
                        timeout=120,
                        kernel_name="trichroma-gpu",
                        resources={"metadata": {"path": str(root / "notebooks")}},
                    ).execute()
                finally:
                    if old_runtime is None:
                        os.environ.pop("JUPYTER_RUNTIME_DIR", None)
                    else:
                        os.environ["JUPYTER_RUNTIME_DIR"] = old_runtime
                html = "\n".join(
                    output.get("data", {}).get("text/html", "")
                    for cell in notebook.cells
                    if cell.cell_type == "code"
                    for output in cell.outputs
                )
                assert "<iframe" in html and "srcdoc=" in html
                report = dict(
                    kernel="trichroma-gpu",
                    executed_cells=sum(cell.cell_type == "code" for cell in notebook.cells),
                    iframe_mode="trusted srcdoc, bundled module and shader bytes",
                    jupyter_base_url="/studio-test/",
                    archive_exists=(root / "notebooks/photon_camera_bundle.zip").is_file(),
                )
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(
                        headless=True,
                        executable_path=str(browser_path),
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
                            os.environ,
                            VK_ICD_FILENAMES=str(browser_path.parent / "vk_swiftshader_icd.json"),
                        ),
                    )
                    page = browser.new_page(viewport=dict(width=1400, height=1000))
                    errors = []
                    messages = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.on(
                        "console",
                        lambda message: messages.append(dict(type=message.type, text=message.text)),
                    )
                    response = page.goto(prefix)
                    page.set_content(html)
                    iframe = next(frame for frame in page.frames if frame != page.main_frame)
                    try:
                        iframe.wait_for_function(
                            "typeof photonCameraReady !== 'undefined'", timeout=10000
                        )
                    except Exception:
                        (args.output / "route_failure.json").write_text(
                            json.dumps(
                                dict(
                                    headers=response.headers,
                                    errors=errors,
                                    messages=messages,
                                    html=page.content(),
                                ),
                                indent=2,
                            )
                        )
                        raise
                    report["adapter"] = iframe.evaluate("async()=>await photonCameraReady")
                    result = iframe.evaluate("""async()=>{
                        const event=await photonCamera.simulate({scene:'fluorescence',photons:128,seed:73});
                        const frame=await photonCamera.render({width:64,height:40,reset:true});
                        return {counts:event.counts,frame,finite:(await photonCamera.linearImage()).every(Number.isFinite)};
                    }""")
                    report.update(result)
                    report["page_errors"] = errors
                    assert result["finite"] and not errors
                    page.screenshot(path=str(args.output / "jupyter-camera.png"), full_page=True)
                    browser.close()
                report["passed"] = True
                (args.output / "notebook_smoke.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                nbformat.write(notebook, Path(directory) / "executed.ipynb")
                print(json.dumps(report), flush=True)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
