"""Check photon parity, allocation cleanup, and GPU memory-failure recovery.

Uses native browser GPU devices with injected allocation failures and an actual
GPUDevice.destroy() whose loss reason is substituted with the reported Vulkan
OOM message. This tests recovery without exhausting system VRAM. It is not a
claim that the colleague's device-specific failure was reproduced.
"""

import argparse, json, threading
from pathlib import Path
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from playwright.sync_api import sync_playwright


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path)
    parser.add_argument(
        "--url", help="Published bundle URL, instead of a local directory"
    )
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.bundle and not args.url:
        parser.error("Provide a bundle directory or --url")
    s = None
    if args.bundle:
        s = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(Quiet, directory=str(args.bundle.resolve()))
        )
        threading.Thread(target=s.serve_forever, daemon=True).start()
    base = args.url.rstrip("/") if args.url else f"http://localhost:{s.server_port}"
    report = {}
    errors = []
    try:
        with sync_playwright() as p:
            exe = (
                args.browser
                or sorted(
                    Path.home().glob(
                        ".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
                    )
                )[-1]
            )
            b = p.chromium.launch(
                executable_path=str(exe),
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
            page = b.new_page()
            page.set_default_timeout(120000)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.add_init_script("""
    window.memoryCheck={live:0,peak:0,devices:0,failures:0};const owned=new WeakMap();
    const make=GPUDevice.prototype.createBuffer,release=GPUBuffer.prototype.destroy;
    GPUDevice.prototype.createBuffer=function(options){
     if((window.memoryLimit&&memoryCheck.live+options.size>window.memoryLimit)||window.failEveryBuffer){memoryCheck.failures++;const e=new Error('GPU memory allocation failed (test injection).');e.name='GpuMemoryError';throw e;}
     const value=make.call(this,options);owned.set(value,options.size);memoryCheck.live+=options.size;memoryCheck.peak=Math.max(memoryCheck.peak,memoryCheck.live);return value;
    };
    GPUBuffer.prototype.destroy=function(){if(owned.has(this)){memoryCheck.live-=owned.get(this);owned.delete(this);}return release.call(this);};
    const request=GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice=async function(options){const device=await request.call(this,options);memoryCheck.devices++;
     const lost=device.lost;Object.defineProperty(device,'lost',{value:lost.then(info=>{
      if(window.injectLoss){window.injectLoss=false;return {reason:'unknown',message:'vkAllocateMemory failed with VK_ERROR_OUT_OF_DEVICE_MEMORY (test injection)'};}return info;
     })});return device;
    };
   """)
            page.goto(base + "/camera.html?manual=1")
            report["adapter"] = page.evaluate("async()=>await photonCameraReady")
            for scene in ["prism", "pmt"]:
                r = page.evaluate(
                    """async scene=>{
     const a=await photonCamera.simulate({scene,photons:8193,debug:true,memory:'full'});
     const b=await photonCamera.simulate({scene,photons:8193,debug:true,memory:'compact'});
     if(JSON.stringify(a.states)!==JSON.stringify(b.states)||JSON.stringify(a.counts)!==JSON.stringify(b.counts))throw Error('Profile changed photon states');
     await photonCamera.render({width:160,height:100,reset:true});
     return {states:8193,full_map_bytes:a.map_bytes,compact_map_bytes:b.map_bytes};
    }""",
                    scene,
                )
                report[scene] = r
            report["allocation_recovery"] = page.evaluate("""async()=>{
    photonCamera.releaseMaps();window.memoryLimit=64*1024*1024;memoryCheck.peak=memoryCheck.live;const before=memoryCheck.devices;
    const r=await photonCamera.simulate({scene:'pmt',photons:8193,memory:'full'});
    window.memoryLimit=0;
    if(r.map_profile!=='compact'||memoryCheck.devices!==before+1)throw Error('No compact allocation recovery');
    await photonCamera.render({width:320,height:200,reset:true});
    return {...memoryCheck,map_profile:r.map_profile,note:document.getElementById('memory_note').textContent};
   }""")
            report["scoped_allocation_recovery"] = page.evaluate("""async()=>{
    const stacks=new WeakMap(),pending=new WeakSet();
    const push=GPUDevice.prototype.pushErrorScope,pop=GPUDevice.prototype.popErrorScope,create=GPUDevice.prototype.createBuffer;
    GPUDevice.prototype.pushErrorScope=function(kind){if(!stacks.has(this))stacks.set(this,[]);stacks.get(this).push(kind);return push.call(this,kind);};
    GPUDevice.prototype.popErrorScope=async function(){const kind=stacks.get(this)?.pop();const error=await pop.call(this);if(kind==='out-of-memory'&&pending.has(this)){pending.delete(this);return new GPUOutOfMemoryError('Scoped allocation failure (test injection)');}return error;};
    GPUDevice.prototype.createBuffer=function(options){const buffer=create.call(this,options);if(options.size>64000000)pending.add(this);return buffer;};
    try{const result=await photonCamera.simulate({scene:'pmt',photons:8193,memory:'full'});
     if(result.map_profile!=='compact')throw Error('Asynchronous allocation failure was not recovered');
     photonCamera.releaseMaps();if(memoryCheck.live)throw Error('Scoped failure leaked buffers');
     return {recovered:true,live_bytes_after_release:memoryCheck.live};
    }finally{GPUDevice.prototype.pushErrorScope=push;GPUDevice.prototype.popErrorScope=pop;GPUDevice.prototype.createBuffer=create;}
   }""")
            # Trigger native GPUDevice destruction, substituting the reported OOM reason.
            page.evaluate("""async()=>{
    await photonCamera.simulate({scene:'pmt',photons:8193,memory:'full'});
    photonCamera.eye=[-80,-180,60];photonCamera.target=[70,0,-10];photonCamera.exposure=400000;
    window.old=photonCamera.device;window.injectLoss=true;old.destroy();
   }""")
            page.wait_for_function(
                'photonCamera.lost?.includes("VK_ERROR_OUT_OF_DEVICE_MEMORY")'
            )
            report["lost_device_recovery"] = page.evaluate("""async()=>{
    const r=await photonCamera.render({width:160,height:100,reset:true});
    if(photonCamera.device===old||photonCamera.mapProfile!=='compact'||r.eye.join(',')!=='-80,-180,60'||r.exposure!==400000)throw Error('Lost-device recovery failed');
    return {new_device:true,profile:photonCamera.mapProfile,eye:r.eye,photons:r.photons};
   }""")
            report["failed_retry_cleanup"] = page.evaluate("""async()=>{
    window.failEveryBuffer=true;let failed=false;
    try{await photonCamera.simulate({scene:'pmt',photons:8193,memory:'full'});}catch(e){failed=true;}
    window.failEveryBuffer=false;
    if(!failed||photonCamera.busy||photonCamera.resources||memoryCheck.live)throw Error('Leaked partial allocations: '+JSON.stringify(memoryCheck));
    await photonCamera.simulate({scene:'pmt',photons:4096,memory:'compact'});
    photonCamera.releaseMaps();return {failed_cleanly:true,live_bytes_after_release:memoryCheck.live};
   }""")
            assert not errors, errors
            b.close()
    finally:
        report["errors"] = errors
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if s:
            s.shutdown()
            s.server_close()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
