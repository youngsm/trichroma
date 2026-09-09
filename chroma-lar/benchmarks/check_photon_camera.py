"""Audit actual browser photon maps and the normalized camera against CPU data."""

import argparse
from functools import partial
import hashlib
import gzip
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading

import numpy as np

from chroma_lar.photon_camera import (
    collect_camera_reference,
    compare_camera_maps,
    export_camera_bundle,
    export_camera_scene,
    integrate_constant_segment,
    reference_camera_maps,
    wall_bin_boundary_oracle,
)
from chroma_lar.webgpu_physics import compare_terminal_states


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def check_axis_aligned_volume(page, manifest, scene, count):
    """Run the real WGSL DDA on an independent constant physical source field."""
    material = manifest["materials"]["names"].index("synthetic_rayleigh_medium")
    wavelength = manifest["camera"]["colors"]["centers"][23]
    values = page.evaluate(
        """async options=>{
        const c=photonCamera,d=c.device;
        const code=(await Promise.all(['physics.wgsl','deposition.wgsl','camera.wgsl'].map(p=>fetch(p).then(r=>r.text())))).join('\\n');
        const probe=`@compute @workgroup_size(1) fn probe_axis(@builtin(global_invocation_id) id:vec3<u32>){
            var directions=array<vec3<f32>,6>(vec3<f32>(1.,0.,0.),vec3<f32>(-1.,0.,0.),vec3<f32>(0.,1.,0.),vec3<f32>(0.,-1.,0.),vec3<f32>(0.,0.,1.),vec3<f32>(0.,0.,-1.));
            let direction=directions[id.x];let polarization=select(vec3<f32>(0.,1.,0.),vec3<f32>(1.,0.,0.),abs(direction.y)>.5);
            let answer=integrate_volume(vec3<f32>(0.),direction,polarization,100.,23u,${options.material}u,${options.wavelength});
            final_state[id.x*2u]=bitcast<u32>(answer.radiance);final_state[id.x*2u+1u]=bitcast<u32>(answer.transmittance);
        }`;
        const module=d.createShaderModule({code:code+'\\n'+probe});
        const pipeline=await d.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'probe_axis'}});
        const data=new Float32Array(61440*64*6);
        for(let cell=0;cell<61440;cell++){const base=(cell*64+23)*6;data[base]=data[base+1]=data[base+2]=1;}
        const volume=c.buffer(data,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
        const output=c.buffer(48,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
        try{
            const group=d.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
                {binding:0,resource:{buffer:c.resources.tables}},{binding:1,resource:{buffer:c.resources.config}},
                {binding:4,resource:{buffer:output}},{binding:6,resource:{buffer:volume}}]});
            const encoder=d.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(6);pass.end();d.queue.submit([encoder.finish()]);
            return Array.from(new Float32Array(await c.download(output)));
        }finally{volume.destroy();output.destroy();}
    }""",
        dict(material=material, wavelength=wavelength),
    )
    grid = scene.host.optics.wavelength_grid
    wavelengths = grid.start + np.arange(grid.count) * grid.step
    length = np.interp(
        wavelength, wavelengths, scene.host.optics.materials.scattering_length[material]
    )
    source = 3 / (4 * np.pi * count * manifest["camera"]["voxel_volume_mm3"])
    expected = [
        float(integrate_constant_segment(source, 1 / length, 100)),
        float(np.exp(-100 / length)),
    ]
    actual = np.asarray(values).reshape(6, 2)
    return dict(
        actual=actual.tolist(),
        expected=expected,
        passed=bool(np.allclose(actual, expected, rtol=2e-5, atol=1e-10)),
        directions=["+x", "-x", "+y", "-y", "+z", "-z"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photons", type=int, default=256)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--browser", type=Path)
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=("prism", "fluorescence", "rayleigh", "pmt"),
        default=("prism", "fluorescence", "rayleigh", "pmt"),
    )
    args = parser.parse_args()
    if not 1 <= args.photons <= 65536:
        parser.error("debug photon count must be in1..65536")
    from playwright.sync_api import sync_playwright

    if args.browser is None:
        args.browser = sorted(
            Path.home().glob(
                ".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
            )
        )[-1]
    flags = [
        "--no-sandbox",
        "--enable-gpu",
        "--enable-unsafe-webgpu",
        "--use-angle=vulkan",
        "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
    ]
    environment = dict(os.environ)
    if args.hardware:
        flags += ["--use-vulkan=native", "--ignore-gpu-blocklist"]
    else:
        flags += ["--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader"]
        environment["VK_ICD_FILENAMES"] = str(args.browser.parent / "vk_swiftshader_icd.json")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    report = dict(
        hardware=args.hardware,
        scenes=[],
        source_hashes={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "chroma-lar/chroma_lar/photon_camera.py",
                "chroma-lar/chroma_lar/pmt_camera.py",
                "chroma-lar/chroma_lar/optical_showcase.py",
                "chroma-lar/chroma_lar/webgpu_physics.py",
                "chroma-lar/benchmarks/check_photon_camera.py",
                "chroma-lite/chroma/triton/spectral.py",
                "chroma-lite/chroma/triton/bvh.py",
                "chroma-lite/chroma/triton/boundary.py",
            )
        },
    )
    passed = True
    with tempfile.TemporaryDirectory(prefix="trichroma-photon-camera-") as temporary:
        bundle = Path(temporary)
        export_camera_bundle(bundle)
        exports = {name: export_camera_scene(name, bundle) for name in args.scenes}
        (bundle / "camera-catalog.json").write_text(
            json.dumps(
                dict(
                    format="trichroma-webgpu-photon-camera-v1",
                    scenes=[row[0] for row in exports.values()],
                )
            )
        )
        report["asset_hashes"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in bundle.iterdir()
            if p.suffix in (".js", ".wgsl", ".html")
        }
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(bundle)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True, executable_path=str(args.browser), args=flags, env=environment
                )
                page = browser.new_page(viewport=dict(width=1450, height=1050))
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(
                    f"http://localhost:{server.server_port}/camera.html?manual=1"
                    + ("" if args.hardware else "&fallback=1")
                )
                report["adapter"] = page.evaluate("async()=>await photonCameraReady")
                if bool(report["adapter"]["isFallbackAdapter"]) == args.hardware:
                    raise RuntimeError(
                        "actual adapter does not match requested software/hardware mode"
                    )
                for name, (entry, fixture, scene) in exports.items():
                    manifest = json.loads((bundle / entry["manifest"]).read_text())
                    expected = collect_camera_reference(
                        name, args.photons, fixture=fixture, scene=scene
                    )
                    # Persist the expensive independent oracle before entering
                    # browser code or report checks. Debug/UI failures must not
                    # discard completed CPU transport and collision records.
                    cache = args.output / "cpu_oracles"
                    cache.mkdir(exist_ok=True)
                    np.savez_compressed(cache / f"{name}.npz", **expected["result"].final_state)
                    with gzip.open(cache / f"{name}.json.gz", "wt") as stream:
                        json.dump(
                            dict(
                                count=args.photons,
                                seed=901,
                                steps=expected["result"].steps,
                                scene_fingerprint=scene.fingerprint,
                                manifest=manifest,
                                source_hashes=report["source_hashes"],
                                records=expected["records"],
                            ),
                            stream,
                            allow_nan=False,
                        )
                    actual = json.loads(
                        page.evaluate(
                            "async options=>JSON.stringify(await photonCamera.simulate(options))",
                            dict(
                                scene=name,
                                photons=args.photons,
                                seed=901,
                                debug=True,
                                debugMaps=True,
                            ),
                        )
                    )
                    state = compare_terminal_states(actual["states"], expected["result"])
                    maps = compare_camera_maps(
                        actual["maps"], reference_camera_maps(expected, manifest)
                    )
                    boundary = wall_bin_boundary_oracle(actual["states"], expected, manifest)
                    deposition = compare_camera_maps(actual["maps"], boundary.pop("maps"))
                    boundary["deposition"] = deposition
                    boundary["passed"] = boundary["valid"] and deposition["passed"]
                    camera = page.evaluate(
                        """async width=>{
                        const c=photonCamera, height=Math.round(width*.625), n=c.photons;
                        const packed=c.scenes[c.scene].packed, light=packed[41];
                        const saved=packed.slice(light,light+64);
                        c.device.queue.writeBuffer(c.resources.tables,light*4,new Uint32Array(64));
                        await c.render({width,height,reset:true});const first=await c.linearImage();
                        c.device.queue.writeBuffer(c.resources.config,0,new Uint32Array([2*n]));
                        await c.render({width,height,reset:true});const doubled=await c.linearImage();
                        let scalingError=0,maxSignal=0;
                        for(let i=0;i<first.length;i++)if(i%4!==3){scalingError=Math.max(scalingError,Math.abs(doubled[i]-.5*first[i]));maxSignal=Math.max(maxSignal,Math.abs(first[i]));}
                        c.device.queue.writeBuffer(c.resources.config,0,new Uint32Array([n]));
                        c.device.queue.writeBuffer(c.resources.tables,light*4,saved);
                        const frame=await c.render({width,height,reset:true});const baseline=await c.linearImage();
                        let bvhCameraMatchesBrute=true;
                        if(c.scene==='pmt'){
                            const header=c.scenes.pmt.packed[45];
                            c.device.queue.writeBuffer(c.resources.tables,45*4,new Uint32Array([0]));
                            await c.render({width,height,reset:true});const brute=await c.linearImage();
                            bvhCameraMatchesBrute=baseline.every((v,i)=>v===brute[i]);
                            c.device.queue.writeBuffer(c.resources.tables,45*4,new Uint32Array([header]));
                        }
                        const map=c.resources, eye=c.eye.slice(),target=c.target.slice();
                        await c.render({width,height,reset:true,exposure:c.exposure*2});
                        const exposed=await c.linearImage();
                        await c.render({width,height,eye:[-200,-150,90]});const orbit=await c.linearImage();
                        await c.render({width,height,eye,target,exposure:frame.exposure});
                        return {frame,scalingError,maxSignal,bvhCameraMatchesBrute,finite:baseline.every(Number.isFinite),
                            exposurePreservesRaw:baseline.every((v,i)=>v===exposed[i]),
                            orbitChangesImage:baseline.some((v,i)=>v!==orbit[i]),sameMaps:c.resources===map,
                            displayedPhotonCount:Number(document.getElementById('photons').value)===n};
                    }""",
                        args.width,
                    )
                    camera["passed"] = bool(
                        camera["finite"]
                        and camera["maxSignal"] > 0
                        and camera["scalingError"] <= max(1e-12, camera["maxSignal"] * 1e-6)
                        and all(
                            camera[key]
                            for key in (
                                "exposurePreservesRaw",
                                "orbitChangesImage",
                                "sameMaps",
                                "displayedPhotonCount",
                                "bvhCameraMatchesBrute",
                            )
                        )
                    )
                    row = dict(
                        scene=entry,
                        photons=args.photons,
                        state=state,
                        maps=maps,
                        wall_bin_boundaries=boundary,
                        camera=camera,
                    )
                    if name == "pmt":
                        row["photocathode"] = dict(
                            browser=actual["counts"]["photocathode"],
                            cpu=len(expected["records"]["photocathode"]),
                        )
                        passed &= row["photocathode"]["browser"] == row["photocathode"]["cpu"]
                        row["cutaway_controls"] = page.evaluate(
                            """async width=>{
                            const c=photonCamera,options={width,height:Math.round(width*.625),reset:true};
                            const maps=c.resources,event=c.result;
                            await c.render({...options,cutaway:false,uvFalseColor:false});const physical=await c.linearImage();
                            await c.render({...options,cutaway:true,uvFalseColor:false});const cut=await c.linearImage();
                            await c.render({...options,cutaway:true,uvFalseColor:true});const uv=await c.linearImage();
                            await c.render({...options,cutaway:false,uvFalseColor:false});const restored=await c.linearImage();
                            return {sameMaps:maps===c.resources,sameEvent:event===c.result,
                                cutChangesImage:physical.some((v,i)=>v!==cut[i]),uvChangesImage:cut.some((v,i)=>v!==uv[i]),
                                restoringModeIsExact:physical.every((v,i)=>v===restored[i]),
                                finite:[cut,uv].every(values=>values.every(Number.isFinite))};
                        }""",
                            args.width,
                        )
                        passed &= all(row["cutaway_controls"].values())
                        brute = json.loads(
                            page.evaluate(
                                """async options=>{
                            const c=photonCamera,packed=c.scenes.pmt.packed,header=packed[45];
                            packed[45]=0;
                            try{return JSON.stringify(await c.simulate(options));}
                            finally{packed[45]=header;}
                        }""",
                                dict(
                                    scene="pmt",
                                    photons=args.photons,
                                    seed=901,
                                    debug=True,
                                    debugMaps=True,
                                ),
                            )
                        )
                        row["bvh_transport"] = dict(
                            terminal_states_exact=all(
                                np.array_equal(np.asarray(value), np.asarray(brute["states"][key]))
                                for key, value in actual["states"].items()
                            ),
                            maps=compare_camera_maps(brute["maps"], actual["maps"]),
                        )
                        passed &= (
                            row["bvh_transport"]["terminal_states_exact"]
                            and row["bvh_transport"]["maps"]["passed"]
                        )
                    if name == "rayleigh":
                        row["axis_aligned_volume"] = check_axis_aligned_volume(
                            page, manifest, scene, args.photons
                        )
                        passed &= row["axis_aligned_volume"]["passed"]
                    report["scenes"].append(row)
                    passed &= (
                        state["all_within_tolerances"] and boundary["passed"] and camera["passed"]
                    )
                    if not maps["passed"]:
                        (args.output / f"{name}-maps.json").write_text(
                            json.dumps(
                                dict(
                                    browser=actual["maps"],
                                    cpu=reference_camera_maps(expected, manifest),
                                )
                            )
                        )
                    page.screenshot(path=str(args.output / f"{name}.png"), full_page=True)
                    print(json.dumps(row), flush=True)
                    (args.output / "verification.json").write_text(
                        json.dumps(report, indent=2, allow_nan=False)
                    )
                report["page_errors"] = errors
                passed &= not errors
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
    report["passed"] = bool(passed)
    (args.output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if not passed:
        raise SystemExit("Photon-camera validation failed; inspect saved evidence")


if __name__ == "__main__":
    main()
