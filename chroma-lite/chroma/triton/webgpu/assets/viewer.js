// Browser-only geometry rendering. The Python exporter is not a render server.
import {gpuBudget} from './scheduler.js';
const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, k) => a.map(v => v * k);
const dot = (a, b) => a.reduce((v, x, i) => v + x * b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = a => mul(a, 1 / Math.hypot(...a));
const copyCamera = camera => structuredClone(camera);
const presentShader = `
@group(0) @binding(0) var picture: texture_2d<f32>;
@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));
  return vec4<f32>(p[i],0.,1.);
}
@fragment fn fragment(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
  return textureLoad(picture, vec2<i32>(p.xy), 0);
}`;

class DetectorRenderer {
  constructor() {
    this.canvas = $("canvas"); this.pending = null; this.busy = false;
    this.scene = null; this.seed = 0; this.timer = null; this.errors = [];
    this.previewRays = 100000;
  }
  async initialize() {
    if (!navigator.gpu) throw Error("WebGPU is unavailable. Open this page in a WebGPU-capable browser over HTTPS or localhost.");
    this.adapter = await navigator.gpu.requestAdapter({powerPreference: "high-performance", forceFallbackAdapter: params.has("fallback")});
    if (!this.adapter) throw Error("No WebGPU adapter was available. The Triton Jupyter viewer remains available on the server.");
    this.device = await this.adapter.requestDevice();
    this.device.addEventListener("uncapturederror", event => {
      this.errors.push(event.error.message); $("error").textContent = event.error.message;
    });
    this.device.lost.then(info => { $("error").textContent = `GPU device lost: ${info.message}. Reload to reconnect.`; });
    const info = this.adapter.info;
    this.adapterInfo = {vendor: info.vendor, architecture: info.architecture, device: info.device,
      description: info.description, isFallbackAdapter: info.isFallbackAdapter ?? params.has("fallback")};
    this.context = this.canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({device:this.device, format:this.format, alphaMode:"opaque"});
    const shader = this.device.createShaderModule({code:await (await fetch("trace.wgsl")).text()});
    const compilation = await shader.getCompilationInfo();
    const errors = compilation.messages.filter(m => m.type === "error");
    if (errors.length) throw Error(errors.map(e => `WGSL ${e.lineNum}:${e.linePos}: ${e.message}`).join("\n"));
    this.compute = await this.device.createComputePipelineAsync({layout:"auto", compute:{module:shader, entryPoint:"main"}});
    const presentation = this.device.createShaderModule({code:presentShader});
    this.presentation = await this.device.createRenderPipelineAsync({layout:"auto",
      vertex:{module:presentation,entryPoint:"vertex"},fragment:{module:presentation,entryPoint:"fragment",targets:[{format:this.format}]},primitive:{topology:"triangle-list"}});
    this.uniform = this.device.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.failures = this.device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    this.catalog = await (await fetch("catalog.json")).json();
    if (!this.catalog.length) throw Error("The scene catalog is empty.");
    $("scene").replaceChildren(...this.catalog.map(item => new Option(item.label || item.name, item.manifest)));
    await this.loadScene(this.catalog[0].manifest);
    this.installControls();
    if (!params.has("manual")) this.schedule(false);
    return this.adapterInfo;
  }
  async loadScene(manifestPath) {
    if (this.busy) throw Error("Wait for the active frame before changing scenes.");
    clearTimeout(this.timer); this.pending = null;
    $("status").textContent = "Loading shared detector geometry…";
    const manifest = await (await fetch(manifestPath)).json();
    if (!["trichroma-webgpu-v1", "trichroma-webgpu-v2"].includes(manifest.format)) throw Error("Unsupported scene format.");
    if (manifest.byte_length > this.device.limits.maxStorageBufferBindingSize) throw Error('This detector exceeds the browser GPU storage-buffer limit.');
    const response = await fetch(manifest.binary);
    if (!response.ok) throw Error('Could not download detector geometry.');
    const payload = await (manifest.compression === 'gzip'
      ? new Response(response.body.pipeThrough(new DecompressionStream('gzip')))
      : response).arrayBuffer();
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", payload)), b=>b.toString(16).padStart(2,"0")).join("");
    if (digest !== manifest.sha256 || payload.byteLength !== manifest.byte_length) throw Error("Scene data failed its length/SHA-256 integrity check.");
    const header = new Uint32Array(payload,0,4);
    if (header[0] !== 0x54524957 || manifest.format !== `trichroma-webgpu-v${header[1]}` || header[2] !== manifest.groups || header[3]*4 !== payload.byteLength) throw Error("Invalid scene header.");
    if (payload.byteLength > this.device.limits.maxStorageBufferBindingSize) throw Error(`Scene needs ${payload.byteLength} bytes, exceeding this adapter's storage-buffer limit.`);
    await this.device.queue.onSubmittedWorkDone();
    this.scene?.destroy();
    this.scene = this.device.createBuffer({size:payload.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.device.queue.writeBuffer(this.scene,0,payload);
    this.manifest = manifest; this.camera = copyCamera(manifest.camera);
    this.views = {Overview: manifest.camera, ...manifest.views};
    if ($('view')) $('view').replaceChildren(...Object.keys(this.views).map(label => new Option(label, label)));
    $("details").textContent = JSON.stringify({adapter:this.adapterInfo, ...manifest},null,2);
    $("status").textContent = `${manifest.name}: ${manifest.geometry.sensor_instances?.toLocaleString() ?? manifest.instances.toLocaleString()} sensors/instances · ${(payload.byteLength/1e6).toFixed(2)} MB shared geometry`;
    return manifest;
  }
  configureTarget(width,height,debug) {
    if (this.width === width && this.height === height && this.debug === debug) return;
    this.texture?.destroy(); this.diagnostic?.destroy();
    this.width = width; this.height = height; this.debug = debug;
    this.canvas.width = width; this.canvas.height = height;
    this.texture = this.device.createTexture({size:[width,height],format:"rgba8unorm",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});
    this.diagnostic = this.device.createBuffer({size:debug ? width*height*48 : 48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  }
  async readBuffer(buffer,size) {
    const staging = this.device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer,0,staging,0,size);
    this.device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
    const result = staging.getMappedRange().slice(0); staging.unmap(); staging.destroy(); return result;
  }
  async render({width=1000,height=625,rays=2500000,seed=0,debug=false,jitter=true,camera=this.camera}={}) {
    if (![width,height,rays].every(Number.isSafeInteger) || width<=0 || height<=0 || rays<width*height || rays>0xffffffff || width>this.device.limits.maxTextureDimension2D || height>this.device.limits.maxTextureDimension2D) throw Error("Invalid image dimensions or ray budget; use at least one ray per pixel.");
    if (![...camera.eye,...camera.target,...camera.up,camera.fov].every(Number.isFinite) || camera.fov<=0 || camera.fov>=179 || Math.hypot(...sub(camera.target,camera.eye))===0 || Math.hypot(...cross(sub(camera.target,camera.eye),camera.up))===0) throw Error("Invalid camera basis or field of view.");
    $("status").textContent = `${this.manifest.name} · rendering ${rays.toLocaleString()} camera rays${this.adapterInfo.isFallbackAdapter ? " on a software adapter" : ""}…`;
    this.configureTarget(width,height,debug);
    const forward = norm(sub(camera.target,camera.eye)), right = norm(cross(forward,camera.up)), up = cross(right,forward);
    const config = new ArrayBuffer(96); const floats = new Float32Array(config), words = new Uint32Array(config);
    floats.set(camera.eye,0); floats.set(forward,4); floats[7]=Math.tan(camera.fov*Math.PI/360);
    floats.set(right,8); floats.set(up,12); words.set([width,height,rays,seed>>>0],16); words.set([+debug,+jitter,0,0xffffffff],20);
    this.device.queue.writeBuffer(this.uniform,0,config); this.device.queue.writeBuffer(this.failures,0,new Uint32Array(1));
    const bindings = this.device.createBindGroup({layout:this.compute.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.scene}},{binding:1,resource:{buffer:this.uniform}},
      {binding:2,resource:this.texture.createView()},{binding:3,resource:{buffer:this.diagnostic}},
      {binding:4,resource:{buffer:this.failures}}]});
    const presentation = this.device.createBindGroup({layout:this.presentation.getBindGroupLayout(0),entries:[{binding:0,resource:this.texture.createView()}]});
    const start = performance.now();
    const timing = await gpuBudget.run(this.device, Math.ceil(width/8)*Math.ceil(height/8), (offset,count)=>{
      this.device.queue.writeBuffer(this.uniform,88,new Uint32Array([offset]));
      const encoder=this.device.createCommandEncoder(),compute=encoder.beginComputePass();
      compute.setPipeline(this.compute);compute.setBindGroup(0,bindings);compute.dispatchWorkgroups(count);compute.end();
      return encoder;
    }, {initial:1, progress:fraction=>{$('status').textContent=`${this.manifest.name} · ${rays.toLocaleString()} camera rays · ${Math.floor(fraction*100)}%`;}});
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});
    pass.setPipeline(this.presentation); pass.setBindGroup(0,presentation); pass.draw(3); pass.end();
    this.device.queue.submit([encoder.finish()]); await this.device.queue.onSubmittedWorkDone();
    const milliseconds = timing.compute_ms;
    const failures = new Uint32Array(await this.readBuffer(this.failures,4))[0];
    if (failures) throw Error(`${failures} traversal guards failed; this frame is invalid.`);
    if (this.errors.length) throw Error(this.errors.join("\n"));
    const result = {width,height,rays,milliseconds,wall_ms:performance.now()-start,batches:timing.batches,failures,seed,adapter:this.adapterInfo,
      timing:"sum of GPU batch queue completion times; wall_ms includes cooperative pauses; excludes browser paint and asset loading"};
    if (debug) result.diagnostic = Array.from(new Float32Array(await this.readBuffer(this.diagnostic,width*height*48)));
    this.lastFrame = result;
    if (!debug && rays<=100000) this.previewRays=Math.round(Math.max(1000,Math.min(100000,rays*30/Math.max(1,milliseconds))));
    $("status").textContent = `${this.manifest.name} · ${rays.toLocaleString()} camera rays · ${result.wall_ms.toFixed(1)} ms including pauses · ${(rays/result.wall_ms/1000).toFixed(2)} M rays/s${this.adapterInfo.isFallbackAdapter ? " · software adapter" : ""}`;
    return result;
  }
  schedule(preview) {
    if (this.busy) gpuBudget.cancel();
    if (preview) {
      clearTimeout(this.timer); this.timer=setTimeout(()=>this.schedule(false),220);
    }
    const budget=preview?this.previewRays:Number($("rays").value);
    const scale=preview?Math.sqrt(budget/100000):1;
    this.pending = {width:preview?Math.max(1,Math.floor(400*scale)):1000,height:preview?Math.max(1,Math.floor(250*scale)):625,rays:budget,seed:this.seed++,camera:copyCamera(this.camera)};
    this.drain();
  }
  async drain() {
    if (this.busy) return;
    this.busy=true;
    try {
      while(this.pending) {
        const next=this.pending; this.pending=null;
        try { await this.render(next); }
        catch(error) { if(error.name !== 'AbortError') throw error; }
      }
    } catch(error) { $("error").textContent=error.stack ?? String(error); }
    finally { this.busy=false; }
  }
  installControls() {
    $("scene").addEventListener("change",async event=>{
      try { while(this.busy) await new Promise(resolve=>setTimeout(resolve,20)); await this.loadScene(event.target.value); this.schedule(false); }
      catch(error) { $("error").textContent=String(error); }
    });
    $("rays").addEventListener("change",()=>this.schedule(false));
    $("render").addEventListener("click",()=>this.schedule(false));
    const resetView=()=>{this.camera=copyCamera(this.views[$('view')?.value] || this.manifest.camera);this.schedule(false);};
    $("reset").addEventListener("click",resetView);
    $('view')?.addEventListener('change',resetView);
    document.addEventListener('gpu-stop',()=>{this.pending=null;clearTimeout(this.timer);$('status').textContent='Stopped.';});
    let pointer=null;
    this.canvas.addEventListener("pointerdown",event=>{pointer=[event.clientX,event.clientY];this.canvas.setPointerCapture(event.pointerId);});
    this.canvas.addEventListener("pointerup",()=>{pointer=null;this.schedule(false);});
    this.canvas.addEventListener("pointercancel",()=>{pointer=null;});
    this.canvas.addEventListener("pointermove",event=>{
      if(!pointer)return;
      const dx=event.clientX-pointer[0],dy=event.clientY-pointer[1];pointer=[event.clientX,event.clientY];
      const offset=sub(this.camera.eye,this.camera.target), distance=Math.hypot(...offset);
      if(event.shiftKey){
        const forward=norm(mul(offset,-1)),right=norm(cross(forward,this.camera.up)),up=cross(right,forward);
        const movement=add(mul(right,-dx*distance*.002),mul(up,dy*distance*.002));
        this.camera.eye=add(this.camera.eye,movement);this.camera.target=add(this.camera.target,movement);
      }else{
        const azimuth=Math.atan2(offset[1],offset[0])-dx*.006;
        const elevation=Math.max(-1.56,Math.min(1.56,Math.asin(offset[2]/distance)+dy*.006));
        this.camera.eye=add(this.camera.target,mul([Math.cos(elevation)*Math.cos(azimuth),Math.cos(elevation)*Math.sin(azimuth),Math.sin(elevation)],distance));
      }
      this.schedule(true);
    });
    this.canvas.addEventListener("wheel",event=>{event.preventDefault();
      this.camera.eye=add(this.camera.target,mul(sub(this.camera.eye,this.camera.target),Math.exp(Math.max(-2,Math.min(2,event.deltaY*.001)))));this.schedule(true);
    },{passive:false});
  }
}

const renderer = new DetectorRenderer();
window.trichroma = renderer;
window.trichromaReady = renderer.initialize().catch(error=>{
  $("error").textContent=error.stack ?? String(error); $("status").textContent="Renderer unavailable"; throw error;
});
