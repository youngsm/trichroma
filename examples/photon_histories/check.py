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
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Previous browser bundle for bitwise transport regression",
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Quiet, directory=str(args.bundle))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = args.url or f"http://localhost:{server.server_port}"
    baseline_server = None
    if args.baseline:
        baseline_server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(Quiet, directory=str(args.baseline))
        )
        threading.Thread(target=baseline_server.serve_forever, daemon=True).start()
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
                    const result={};for(const key of ['flights','counts','stats','debug','hits'])result[key]=await digest(await app.download(app.eventBuffers[key]));return result;
                };
            }""")
            baseline_page = None
            if baseline_server:
                baseline_page = browser.new_page()
                baseline_page.goto(
                    f"http://localhost:{baseline_server.server_port}/?manual=1"
                )
                baseline_page.evaluate(
                    "async()=>{await photonHistoriesReady;photonHistories.requestRender=()=>{};}"
                )

            def baseline_hashes(event, seed, capacity):
                return baseline_page.evaluate(
                    """async ({event,seed,capacity})=>{
                    const app=photonHistories;app.capacity=capacity;
                    await app.simulate({event,seed,count:8193,debug:4096});
                    const hashes={};for(const key of ['flights','counts','stats','debug']){
                        const bytes=await crypto.subtle.digest('SHA-256',await app.download(app.eventBuffers[key]));
                        hashes[key]=Array.from(new Uint8Array(bytes),x=>x.toString(16).padStart(2,'0')).join('');
                    }return hashes;
                }""",
                    {"event": event, "seed": seed, "capacity": capacity},
                )

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
                if baseline_page:
                    baseline = baseline_hashes(event, 901, 32768)
                    assert all(
                        r["hashes"][key] == value for key, value in baseline.items()
                    ), (event, "baseline mismatch")
                hit_check = page.evaluate("""async()=>{
                    const hits=new Float32Array(await app.download(app.eventBuffers.hits));
                    const flights=new Float32Array(await app.download(app.eventBuffers.flights));
                    const counts=new Uint32Array(await app.download(app.eventBuffers.counts));
                    let matched=0;
                    for(let slot=0;slot<app.retained;slot++){
                        const id=slot*app.stride,n=counts[slot];if(!n)continue;
                        const end=(slot*32+n-1)*16;
                        if(flights[end+13]===1){
                            if(hits[id*4]!==flights[end+7]||hits[id*4+1]!==flights[end+14]+1||hits[id*4+2]!==flights[end+11])throw Error('Arrival does not match terminal flight');matched++;
                        }else if(hits[id*4+1]!==0)throw Error('Spurious PMT arrival');
                    }
                    let amplitudeError=0;
                    for(const time of [0,10,50,100,180,3000,100,0]){
                        app.timeWindow('full');app.setEventTime(time);await app.render({width:340});
                        const states=new Float32Array(await app.download(app.eventBuffers.hitStates));
                        let sum=0;
                        for(let sensor=0;sensor<app.sensorCount;sensor++){
                            let count=0,last=-1e30,amplitude=0;
                            for(let i=app.hitOffsets[sensor];i<app.hitOffsets[sensor+1];i++)if(app.hitTimes[i]<=Math.fround(app.eventTime)){
                                count++;last=app.hitTimes[i];amplitude+=Math.exp(-(Math.fround(app.eventTime)-app.hitTimes[i])/Number(document.getElementById('pmtDecay').value));
                            }
                            amplitudeError=Math.max(amplitudeError,Math.abs(states[sensor*4+2]-amplitude));
                            if(states[sensor*4]!==count || (count && states[sensor*4+1]!==last))throw Error('PMT timeline mismatch');sum+=count;
                        }
                        if(sum!==app.arrivedAt(Math.fround(app.eventTime)))throw Error('Total arrival mismatch');
                    }
                    if(amplitudeError>5e-5)throw Error('Additive PMT signal mismatch');
                    app.timeWindow('prompt');return {matched,forward_backward_scrub:true,amplitudeError};
                }""")
                r["arrivals"] = hit_check
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
                    equal = page.evaluate(
                        """async ({event,seed})=>{
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
                    }""",
                        {"event": event, "seed": seed},
                    )
                    assert equal, (event, seed)
                    if baseline_page:
                        baseline = baseline_hashes(event, seed, 8193)
                        current = page.evaluate("async()=>await snapshot()")
                        assert all(
                            current[key] == value for key, value in baseline.items()
                        ), (event, seed, "baseline mismatch")
                report[event]["bitwise_photon_histories"] = 8193 * args.seeds
                report[event]["previous_transport_bitwise_equal"] = bool(baseline_page)
                full = page.evaluate(
                    """async event=>{const r=await app.simulate({event,seed:901});await app.render();return {photons:r.photons,wall_ms:r.wall_ms,counters:r.counters.slice(0,16),retained:r.retained,hit_sha256:await digest(await app.download(app.eventBuffers.hits))};}""",
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
            # High precision playback must advance even below the range-input step per frame.
            page.click("#early")
            page.wait_for_function("!app.rendering && !app.pending")
            assert page.evaluate(
                "app.eventTime===3 && document.getElementById('window').value==='early'"
            )
            assert page.evaluate("app.eventGeneration===generation")
            page.select_option("#window", "first")
            page.select_option("#speed", "0.01")
            start = page.evaluate("app.eventTime")
            page.click("#play")
            page.wait_for_timeout(1100)
            page.click("#stop")
            elapsed = page.evaluate("app.eventTime") - start
            assert 0.006 < elapsed < 0.035, elapsed
            report["controls"]["slow_playback_ns_approx_1s"] = elapsed
            page.select_option("#fade", "1")
            page.wait_for_function("!app.rendering && !app.pending")
            assert page.evaluate("app.eventGeneration===generation")
            page.locator("#view").screenshot(path=str(args.output / "early.png"))
            page.select_option("#window", "full")
            page.locator("#time").fill("180")
            page.locator("#time").dispatch_event("input")
            page.wait_for_function("!app.rendering && !app.pending")
            page.locator("#view").screenshot(path=str(args.output / "hits.png"))
            visible_hits = page.locator("#view").screenshot()
            page.uncheck("#highlight")
            page.wait_for_function("!app.rendering && !app.pending")
            assert (
                page.locator("#view").screenshot() != visible_hits
            ), "3D PMT highlights did not change the render"
            page.check("#highlight")
            page.select_option("#window", "arrivals")
            page.wait_for_function("!app.rendering && !app.pending")
            assert page.evaluate(
                "app.arrivedAt(app.eventTime)>0 && app.eventTime>app.allHitTimes[0]"
            )
            report["controls"]["hits_in_3d"] = True
            report["controls"]["first_arrivals_window"] = True
            # Keep every physical photon, then compare arrivals and all non-retention counters.
            old = page.evaluate(
                "async()=>({hits:await digest(await app.download(app.eventBuffers.hits)),counters:app.result.counters,eye:app.view,capacity:app.maxCapacity})"
            )
            if old["capacity"] >= 272157:
                page.select_option("#paths", "all")
                page.wait_for_function(
                    "!app.busy && app.retained===app.count && !app.rendering && !app.pending"
                )
                new = page.evaluate(
                    "async()=>({hits:await digest(await app.download(app.eventBuffers.hits)),counters:app.result.counters,eye:app.view,paths:app.retained,bytes:app.eventBuffers.flights.size})"
                )
                assert old["hits"] == new["hits"] and old["eye"] == new["eye"]
                assert all(
                    a == b
                    for i, (a, b) in enumerate(zip(old["counters"], new["counters"]))
                    if i != 8
                )
                report["controls"]["all_paths"] = {
                    key: new[key] for key in ["paths", "bytes"]
                }
                page.click("#early")
                page.wait_for_function("!app.rendering && !app.pending")
                page.locator("#view").screenshot(
                    path=str(args.output / "early-all.png")
                )
                report["controls"]["cached_frame_ms"] = page.evaluate("""async()=>{
                    const times={};for(const limit of ['8192','all']){
                        document.getElementById('paths').value=limit;const start=performance.now();
                        for(let i=0;i<3;i++)await app.render();times[limit]=(performance.now()-start)/3;
                    }return times;
                }""")
                page.select_option("#event", "muon")
                page.wait_for_function(
                    "!app.busy && app.source.id==='muon' && app.retained===app.count && !app.rendering && !app.pending"
                )
                muon_all = page.evaluate(
                    "async()=>({hits:await digest(await app.download(app.eventBuffers.hits)),count:app.count,counters:app.result.counters.slice(0,16)})"
                )
                assert muon_all["hits"] == report["muon"]["full_event"]["hit_sha256"]
                assert all(
                    a == b
                    for i, (a, b) in enumerate(
                        zip(
                            muon_all["counters"],
                            report["muon"]["full_event"]["counters"],
                        )
                    )
                    if i != 8
                )
                report["controls"]["all_muon_paths"] = muon_all["count"]
                page.locator("#view").screenshot(
                    path=str(args.output / "muon-early-all.png")
                )
                page.select_option("#window", "late")
                page.wait_for_function("!app.rendering && !app.pending")
                assert page.evaluate("app.eventTime>2900")
                report["controls"]["late_window"] = True
            # Independent synthetic impulse trains test coincidence, pileup, decay and replay.
            report["pmt_signal"] = page.evaluate("""async()=>{
                const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
                const owned=[];const buffer=(data,usage)=>{const b=app.device.createBuffer({size:data.byteLength,usage});owned.push(b);app.device.queue.writeBuffer(b,0,data);return b;};
                const lists=[[1,1,3],[2,10,2900],[]];
                try{
                    const offsets=buffer(new Uint32Array([0,3,6,6]),usage),times=buffer(new Float32Array(lists.flat()),usage);
                    const states=buffer(new Float32Array(12),usage),params=buffer(new Float32Array(20),GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
                    const group=app.group(app.hitPipeline,{6:params,13:offsets,14:times,15:states});let maximumError=0,cases=0;
                    for(const tau of [4,20,100])for(const time of [0,.9999,1,2,3,10,15,200,2910,3]){
                        const settings=app.settings();settings[7]=0;settings[8]=time;settings[16]=3;settings[17]=tau;app.device.queue.writeBuffer(params,0,settings);
                        const encoder=app.device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(app.hitPipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();app.device.queue.submit([encoder.finish()]);
                        const result=new Float32Array(await app.download(states));
                        for(let sensor=0;sensor<3;sensor++){
                            const arrived=lists[sensor].filter(t=>t<=settings[8]);const expected=arrived.reduce((a,t)=>a+Math.exp(-(settings[8]-t)/tau),0);
                            if(result[sensor*4]!==arrived.length)throw Error('Integrated PMT count mismatch');
                            maximumError=Math.max(maximumError,Math.abs(result[sensor*4+2]-expected));
                        }
                        if(time===1 && result[2]!==2)throw Error('Coincident hits did not add');cases++;
                    }
                    if(maximumError>5e-5)throw Error('Exponential PMT pulse mismatch');
                    return {cases,maximumError,coincident_hits_add:true,delayed_hit:true,backward_scrub:true};
                }finally{owned.forEach(b=>b.destroy());}
            }""")
            for control, value in [("#pmtDecay", "50"), ("#pmtGain", ".5")]:
                page.select_option(control, value)
            page.locator("#pmtMemory").fill("0.5")
            page.locator("#pmtMemory").dispatch_event("input")
            page.wait_for_function("!app.rendering && !app.pending")
            signal_event_generation = page.evaluate("app.eventGeneration")
            page.select_option("#window", "full")
            page.locator("#time").fill("500")
            page.locator("#time").dispatch_event("input")
            page.wait_for_function("!app.rendering && !app.pending")
            retained_ring = page.locator("#view").screenshot(
                path=str(args.output / "integrated-ring.png")
            )
            page.locator("#pmtMemory").fill("0")
            page.locator("#pmtMemory").dispatch_event("input")
            page.wait_for_function("!app.rendering && !app.pending")
            assert (
                page.locator("#view").screenshot() != retained_ring
            ), "Accumulated hits did not retain the ring"
            assert page.evaluate("app.eventGeneration") == signal_event_generation
            report["pmt_signal"]["controls_reuse_event"] = True
            report["pmt_signal"]["integrated_ring_persists"] = True
            # Inject an allocation failure without exhausting the host GPU.
            report["resource_limits"] = page.evaluate("""async()=>{
                const create=app.device.createBuffer.bind(app.device);let injected=0;
                app.device.createBuffer=descriptor=>{
                    if(!injected && descriptor.size===64*1048576){injected++;const error=Error('Injected allocation failure');error.name='GpuMemoryError';throw error;}
                    return create(descriptor);
                };
                try{
                    app.capacity=32768;document.getElementById('paths').value='8192';
                    const result=await app.simulate({event:'muon',seed:901});
                    if(injected!==1||app.capacity!==16384)throw Error('Allocation retry did not reduce history storage');
                    return {allocation_failure_retry:true,retained:result.retained,hit_sha256:await digest(await app.download(app.eventBuffers.hits))};
                }finally{app.device.createBuffer=create;}
            }""")
            assert (
                report["resource_limits"]["hit_sha256"]
                == report["muon"]["full_event"]["hit_sha256"]
            )
            limited = browser.new_page()
            limited.add_init_script("""(()=>{
                const request=GPUAdapter.prototype.requestDevice;
                GPUAdapter.prototype.requestDevice=function(descriptor={}){
                    return request.call(this,{...descriptor,requiredLimits:{...descriptor.requiredLimits,maxStorageBufferBindingSize:128*1048576,maxBufferSize:256*1048576}});
                };
            })()""")
            limited.goto(base.rstrip("/") + "/?manual=1")
            limits = limited.evaluate("""async()=>{
                await photonHistoriesReady;const app=photonHistories;app.requestRender=()=>{};
                await app.simulate({count:8193});
                return {capacity:app.maxCapacity,all_disabled:document.querySelector('#paths [value="all"]').disabled,errors:app.errors};
            }""")
            assert limits == {
                "capacity": 65536,
                "all_disabled": True,
                "errors": [],
            }, limits
            report["resource_limits"]["128_MiB_adapter"] = limits
            limited.close()
            report["controls"]["fade_reuses_event"] = True
            report["controls"]["early_closeup"] = True
            assert not errors, errors
            assert not page.evaluate("app.errors"), page.evaluate("app.errors")
            browser.close()
    finally:
        report["errors"] = errors
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        server.shutdown()
        server.server_close()
        if baseline_server:
            baseline_server.shutdown()
            baseline_server.server_close()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
