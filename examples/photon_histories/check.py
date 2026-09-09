"""Audit browser emission, stored histories, controls and deterministic batching."""

import argparse
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--url")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Quiet, directory=str(args.bundle))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = args.url or f"http://localhost:{server.server_port}"
    report = {}
    errors = []
    try:
        with sync_playwright() as p:
            executable = sorted(
                Path.home().glob(
                    ".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
                )
            )[-1]
            browser = p.chromium.launch(
                executable_path=str(executable),
                headless=True,
                args=[
                    "--no-sandbox",
                    "--enable-gpu",
                    "--enable-unsafe-webgpu",
                    "--use-angle=vulkan",
                    "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
                    "--use-vulkan=native",
                    "--ignore-gpu-blocklist",
                ],
            )
            page = browser.new_page(viewport={"width": 1500, "height": 1200})
            page.set_default_timeout(120000)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(base.rstrip("/") + "/?manual=1")
            report["adapter"] = page.evaluate("async()=>await photonHistoriesReady")
            page.evaluate("""()=>{
                window.app=photonHistories;
                // Audit calls explicitly render; keep the normal UI scheduler for controls below.
                window.schedule=app.requestRender.bind(app);app.requestRender=()=>{};
                window.digest=async buffer=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),x=>x.toString(16).padStart(2,'0')).join('');
                window.snapshot=async()=>{
                    const result={};for(const key of ['flights','counts','stats','debug'])result[key]=await digest(await app.download(app.eventBuffers[key]));return result;
                };
            }""")
            for event in ["muon", "electron"]:
                r = page.evaluate(
                    """async event=>{
                    const r=await app.simulate({event,count:8193,debug:4096});
                    const hashes=await snapshot();
                    const records=new Float32Array(await app.download(app.eventBuffers.flights));
                    const counts=new Uint32Array(await app.download(app.eventBuffers.counts));
                    let timeError=0,orthogonalError=0,unitError=0,continuityError=0,scatterProjectionError=0,stored=0;
                    for(let slot=0;slot<app.retained;slot++)for(let step=0;step<counts[slot];step++){
                        const b=(slot*32+step)*16,x=records.slice(b,b+3),end=records.slice(b+4,b+7),e=records.slice(b+8,b+11),w=records[b+11];
                        const delta=end.map((v,i)=>v-x[i]),length=Math.hypot(...delta),ng=1.322+9000/(w*w);
                        timeError=Math.max(timeError,Math.abs((records[b+7]-records[b+3])-length*ng/299.792458));
                        if(length>.1)orthogonalError=Math.max(orthogonalError,Math.abs(delta.reduce((s,v,i)=>s+v*e[i],0)/length));
                        unitError=Math.max(unitError,Math.abs(Math.hypot(...e)-1));
                        if(step){
                            const prev=b-16;continuityError=Math.max(continuityError,...x.map((v,i)=>Math.abs(v-records[prev+4+i])),Math.abs(records[b+3]-records[prev+7]));
                            if(length>.1){const d=delta.map(v=>v/length),old=records.slice(prev+8,prev+11),dot=d.reduce((s,v,i)=>s+v*old[i],0),projected=old.map((v,i)=>v-dot*d[i]),scale=Math.hypot(...projected);
                                scatterProjectionError=Math.max(scatterProjectionError,Math.hypot(...e.map((v,i)=>v-projected[i]/scale)));}
                        }
                        if(records[b+12]!==slot*app.stride)throw Error('Stored photon identity changed');
                        if(!Number.isFinite(records[b+7]))throw Error('Nonfinite history');stored++;
                    }
                    return {debug:r.debug,counters:r.counters,hashes,stored,timeError,orthogonalError,unitError,continuityError,scatterProjectionError};
                }""",
                    event,
                )
                debug = np.array(r.pop("debug")).reshape(-1, 16)
                export = next(
                    e
                    for e in json.loads((args.bundle / "events.json").read_text())[
                        "events"
                    ]
                    if e["id"] == event
                )
                rows = np.array(export["rows"], dtype=np.float32)
                source = rows[debug[:, 12].astype(int)]
                d, e, w, beta = debug[:, 4:7], debug[:, 8:11], debug[:, 7], debug[:, 11]
                axis = source[:, 8:11].astype(float)
                axis /= np.linalg.norm(axis, axis=1)[:, None]
                cosine = 1 / (beta * (1.322 + 3000 / w**2))
                angle_error = float(np.max(np.abs(np.sum(d * axis, axis=1) - cosine)))
                assert angle_error < 5e-6, angle_error
                assert np.max(np.abs(np.sum(e * d, axis=1))) < 3e-6
                assert np.max(np.abs(np.linalg.norm(d, axis=1) - 1)) < 3e-6
                assert np.all((w >= 360) & (w <= 700))
                start, end = source[:, :3].astype(float), source[:, 4:7].astype(float)
                v = end - start
                length = np.linalg.norm(v, axis=1)
                fraction = np.sum((debug[:, :3] - start) * v, axis=1) / length**2
                distance = np.linalg.norm(
                    debug[:, :3] - (start + fraction[:, None] * v), axis=1
                )
                assert distance.max() < 0.002, distance.max()
                assert np.all((fraction > -0.01) & (fraction < 1.01))
                emission_time_error = np.abs(
                    debug[:, 3] - source[:, 3] - fraction * length / (beta * 299.792458)
                ).max()
                assert emission_time_error < 0.001, emission_time_error
                assert r["timeError"] < 0.001, r["timeError"]
                assert r["orthogonalError"] < 0.005, r["orthogonalError"]
                assert r["unitError"] < 3e-6 and r["continuityError"] == 0
                assert r["scatterProjectionError"] < 0.01, r["scatterProjectionError"]
                assert r["counters"][5] == 0 and r["counters"][7] == 0
                # Five broad spectral bins compare to independent quadrature, marginalizing all source steps.
                edges = np.linspace(360, 700, 6)
                expected = []
                weights = np.diff(np.r_[0, rows[:, 11].astype(float)])
                wavelengths = np.arange(360.25, 700, 0.5)
                spectrum = np.zeros_like(wavelengths)
                for begin in range(0, len(rows), 512):
                    q = (
                        np.maximum(
                            0,
                            1
                            - 1
                            / (
                                rows[begin : begin + 512, 7, None].astype(float)
                                * (1.322 + 3000 / wavelengths**2)
                            )
                            ** 2,
                        )
                        / wavelengths**2
                    )
                    q /= np.maximum(q.sum(axis=1)[:, None], 1e-30)
                    spectrum += (q * weights[begin : begin + 512, None]).sum(axis=0)
                expected = np.array(
                    [
                        spectrum[(wavelengths >= a) & (wavelengths < b)].sum()
                        for a, b in zip(edges[:-1], edges[1:])
                    ]
                )
                observed = np.histogram(w, edges)[0]
                chi2 = float(
                    ((observed - len(w) * expected) ** 2 / (len(w) * expected)).sum()
                )
                assert chi2 < 30, chi2
                # Change batch size only; every stored byte and integer counter must agree.
                second = page.evaluate(
                    """async event=>{
                    const moduleURL=new URL(document.querySelector('script[type=module]').src);
                    const budget=(await import(new URL('scheduler.js'+moduleURL.search,moduleURL))).gpuBudget;
                    const run=budget.run;budget.run=function(d,n,encode,opts){return run.call(this,d,n,encode,{...opts,initial:3,maximum:3});};
                    try{await app.simulate({event,count:8193,debug:4096});return await snapshot();}finally{budget.run=run;}
                }""",
                    event,
                )
                assert r.pop("hashes") == second
                r["counters"] = r["counters"][:16]
                r.update(
                    cone_cosine_error=angle_error,
                    source_position_error_mm=float(distance.max()),
                    emission_time_error_ns=float(emission_time_error),
                    spectrum_chi2=chi2,
                    batch_bitwise_equal=True,
                )
                report[event] = r
                # Retain every photon in each additional sample; only batch size varies.
                for seed in range(902, 901 + args.seeds):
                    equal = page.evaluate("""async ({event,seed})=>{
                        const oldCapacity=app.capacity;app.capacity=8193;
                        const moduleURL=new URL(document.querySelector('script[type=module]').src);
                        const budget=(await import(new URL('scheduler.js'+moduleURL.search,moduleURL))).gpuBudget;
                        const run=budget.run;
                        try{
                            await app.simulate({event,seed,count:8193,debug:4096});const first=await snapshot();
                            budget.run=function(d,n,encode,opts){return run.call(this,d,n,encode,{...opts,initial:3,maximum:3});};
                            await app.simulate({event,seed,count:8193,debug:4096});
                            return JSON.stringify(first)===JSON.stringify(await snapshot());
                        }finally{budget.run=run;app.capacity=oldCapacity;}
                    }""", {"event": event, "seed": seed})
                    assert equal, (event, seed)
                report[event]["bitwise_photon_histories"] = 8193 * args.seeds
                full = page.evaluate(
                    """async event=>{const r=await app.simulate({event,seed:901});await app.render();return {photons:r.photons,wall_ms:r.wall_ms,counters:r.counters.slice(0,16),retained:r.retained};}""",
                    event,
                )
                report[event]["full_event"] = full
                page.locator("#view").screenshot(
                    path=str(args.output / (event + ".png"))
                )
            page.evaluate(
                "app.requestRender=window.schedule;window.savedView=JSON.stringify(app.view);window.generation=app.eventGeneration;"
            )
            page.select_option("#paths", "2048")
            page.locator("#time").fill("65")
            page.locator("#time").dispatch_event("input")
            page.wait_for_function("app.lastFrame.time===65")
            assert page.evaluate(
                "app.eventGeneration===generation && JSON.stringify(app.view)===savedView"
            )
            before = page.evaluate("app.frame")
            page.click("#play")
            page.wait_for_function(f"app.frame>{before+2}")
            page.click("#stop")
            page.wait_for_function("!app.rendering")
            assert not page.evaluate("app.playing")
            box = page.locator("#view").bounding_box()
            x = box["x"] + box["width"] / 2
            y = box["y"] + box["height"] / 2
            old = page.evaluate("structuredClone(app.view)")
            page.mouse.move(x, y)
            page.mouse.down(button="middle")
            page.mouse.move(x + 40, y + 15, steps=4)
            page.mouse.up(button="middle")
            page.wait_for_function("!app.rendering && !app.pending")
            new = page.evaluate("structuredClone(app.view)")
            np.testing.assert_allclose(
                np.array(new["eye"]) - old["eye"],
                np.array(new["target"]) - old["target"],
                atol=1e-8,
            )
            assert not np.array_equal(new["eye"], old["eye"])
            report["controls"] = {
                "scrubbing_reuses_event": True,
                "path_count_reuses_event": True,
                "play_stop": True,
                "middle_pan": True,
            }
            assert not errors, errors
            assert not page.evaluate("app.errors"), page.evaluate("app.errors")
            browser.close()
    finally:
        report["errors"] = errors
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        server.shutdown()
        server.server_close()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
