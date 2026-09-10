import {requestGpuDevice, allocateGpu, checkedGpuWork, isGpuMemoryError} from './gpu.js';
import {poseEvent, rotatePoint, rotateVector} from './event_pose.js';
import {gpuBudget} from './scheduler.js';
const $ = id => document.getElementById(id);
const assetURL = name => {const url=new URL(name,import.meta.url);url.search=new URL(import.meta.url).search;return url;};
const norm = v => {const n=Math.hypot(...v);return v.map(x=>x/n);};
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const PRESETS = {
  oblique:{eye:[11500,-14500,6500],target:[0,0,800],fov:62},
  side:{eye:[17000,0,0],target:[0,0,0],fov:60},
  cone:{eye:[9000,-9000,-12500],target:[0,0,1200],fov:68},
  wide:{eye:[19000,-15000,13000],target:[0,0,0],fov:80}
};
class PhotonHistories {
  constructor(){
    this.view=structuredClone(PRESETS.oblique);this.seed=901;this.errors=[];
    this.pending=false;this.rendering=false;this.busy=false;this.playing=false;
    this.frame=0;this.eventGeneration=0;this.eventTime=42;this.aim={azimuth:0,elevation:90};
  }
  async initialize(){
    const selected=await requestGpuDevice({storageBufferBytes:640*1048576});Object.assign(this,selected);
    gpuBudget.software=this.software;
    this.device.addEventListener('uncapturederror',event=>{this.errors.push(event.error.message);this.fail(event.error);});
    this.device.lost.then(info=>{this.playing=false;gpuBudget.cancel();this.fail(Error('GPU device lost: '+info.message));});
    const [trace,code,data,manifest]=await Promise.all(['trace.wgsl','event.wgsl','events.json','theia.json'].map(async name=>{
      const r=await fetch(assetURL(name));if(!r.ok)throw Error(`${name}: ${r.status}`);return name.endsWith('.wgsl')?r.text():r.json();
    }));
    if(data.format!=='chroma-hk-photon-histories-v1'||manifest.groups!==2)throw Error('Unsupported event or geometry export');
    this.events=data.events;this.manifest=manifest;
    $('event').replaceChildren(...this.events.map(e=>new Option(`${e.id==='muon'?'Muon':'Electron'} · ${(e.energy_MeV/1000).toFixed(4)} GeV`,e.id)));
    const response=await fetch(assetURL(manifest.binary));if(!response.ok)throw Error('Geometry download failed');
    const payload=await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',payload)),b=>b.toString(16).padStart(2,'0')).join('');
    if(digest!==manifest.sha256||payload.byteLength!==manifest.byte_length)throw Error('Geometry checksum mismatch');
    const [compute,raster]=code.split('// RASTER MODULE');
    this.computeModule=this.device.createShaderModule({code:trace+'\n'+compute});
    this.rasterModule=this.device.createShaderModule({code:raster});
    for(const module of [this.computeModule,this.rasterModule]){
      const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');
      if(errors.length)throw Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join('\n'));
    }
    this.transport=await this.device.createComputePipelineAsync({layout:'auto',compute:{module:this.computeModule,entryPoint:'propagate'}});
    this.hitPipeline=await this.device.createComputePipelineAsync({layout:'auto',compute:{module:this.computeModule,entryPoint:'update_hits'}});
    this.cameraPipeline=await this.device.createComputePipelineAsync({layout:'auto',compute:{module:this.computeModule,entryPoint:'detector_camera'}});
    this.canvas=$('view');this.context=this.canvas.getContext('webgpu');this.format=navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({device:this.device,format:this.format,alphaMode:'opaque'});
    this.backgroundPipeline=await this.device.createRenderPipelineAsync({layout:'auto',vertex:{module:this.rasterModule,entryPoint:'fullscreen'},fragment:{module:this.rasterModule,entryPoint:'background',targets:[{format:this.format}]},primitive:{topology:'triangle-list'}});
    this.linePipeline=await this.device.createRenderPipelineAsync({layout:'auto',vertex:{module:this.rasterModule,entryPoint:'photon_vertex'},fragment:{module:this.rasterModule,entryPoint:'photon_fragment',targets:[{format:this.format,blend:{color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha'}}}]},primitive:{topology:'triangle-list'}});
    const words=new Uint32Array(payload),header=20;
    this.sensorCount=words[header+13];
    if(this.sensorCount!==manifest.geometry.sensor_instances)throw Error('PMT instance count mismatch');
    const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
    this.base=await allocateGpu(this.device,a=>({
      scene:a.buffer(payload,storage),camera:a.buffer(96,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),
      settings:a.buffer(80,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),failures:a.buffer(4,storage)
    }));
    this.maxCapacity=this.software?2048:Math.min(300000,Math.floor(Math.min(this.device.limits.maxStorageBufferBindingSize,this.device.limits.maxBufferSize)/2048));
    this.capacity=Math.min(this.maxCapacity,32768);
    for(const option of $('paths').options)if(option.value!=='all')option.disabled=Number(option.value)>this.maxCapacity;
    if(this.software){$('paths').value='2048';$('gpu_mode').value='eco';}
    const info=this.adapter.info;
    $('adapter').textContent=`WebGPU · ${info.description||info.vendor||'browser GPU'} ${info.architecture||''}${this.software?' · software adapter':''} · high-performance preference`;
    this.controls();$('simulate').disabled=false;
    $('status').textContent='Ready. Simulate the recorded HK event.';
    if(!new URLSearchParams(location.search).has('manual'))await this.simulate();
    return {vendor:info.vendor,architecture:info.architecture,software:this.software};
  }
  fail(error){if(error.name==='AbortError'){$('status').textContent='Stopped.';return;}$('status').textContent=error.message||String(error);$('status').classList.add('error');}
  group(pipeline,entries){return this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:Object.entries(entries).map(([binding,value])=>({binding:Number(binding),resource:value instanceof GPUBuffer?{buffer:value}:value}))});}
  basis(){
    const forward=norm(this.view.target.map((x,i)=>x-this.view.eye[i]));
    const right=norm(cross(forward,Math.abs(forward[2])>.99?[0,1,0]:[0,0,1]));
    return {forward,right,up:cross(right,forward)};
  }
  cameraConfig(width,height,offset=0){
    const u=new Uint32Array(24),f=new Float32Array(u.buffer),b=this.basis();
    f.set(this.view.eye);f.set(b.forward,4);f[7]=Math.tan(this.view.fov*Math.PI/360);f.set(b.right,8);f.set(b.up,12);
    u.set([width,height,width*height,901],16);u.set([2,0,offset,0xffffffff],20);return u;
  }
  settings(offset=0,disableOptics=false){
    const p=new Float32Array(20);
    p.set([this.count||1,offset,this.source?.rows.length||1,this.seed]);
    p.set([this.capacity,this.stride||1,this.displayCount(),$('mode').value==='history'?1:0],4);
    p.set([this.eventTime,Number($('trail').value),Number($('opacity').value),Number($('brightness').value)],8);
    p.set([$('highlight').checked?1:0,Number($('fade').value),this.debugCount||0,disableOptics?1:0],12);
    p.set([this.sensorCount,Number($('pmtDecay').value),Number($('pmtMemory').value),Number($('pmtGain').value)],16);return p;
  }
  displayCount(){return Math.min($('paths').value==='all'?this.count:Number($('paths').value),this.retained||0);}
  setEventTime(value){
    this.eventTime=Math.max(Number($('time').min),Math.min(Number($('time').max),Number(value)));
    $('time').value=this.eventTime;$('timeLabel').textContent=this.eventTime.toFixed(this.eventTime<10?3:1)+' ns';
  }
  timeWindow(name,reset=true){
    const latest=this.source?.latest_emission_ns||0;
    if(name==='late'&&latest<=180)name='prompt';
    const firstHit=this.allHitTimes?.[0]??110;
    const windows={arrivals:[Math.max(0,Math.floor(firstHit-5)),Math.ceil(firstHit+30),firstHit+5],first:[0,1,.3],early:[0,10,3],prompt:[0,180,42],late:[Math.max(0,Math.floor(latest-100)),Math.ceil(latest+200),latest+25],full:[0,Math.max(180,Math.ceil(latest+200)),42]};
    const [min,max,time]=windows[name];$('window').value=name;$('time').min=min;$('time').max=max;$('time').step=name==='first'?'.0001':'.001';
    this.setEventTime(reset?time:this.eventTime);
    $('windowNote').textContent=name==='late'?`Late secondary emission near ${latest.toFixed(1)} ns. This is a different time in the same event.`:`${name==='arrivals'?'First PMT arrivals':name==='first'?'First nanosecond':name==='early'?'Early light':name==='prompt'?'Prompt light':'Whole event'}: ${min}–${max} ns. Time windows navigate the same simulated event.`;
    this.requestRender();
  }
  syncTimeControls(){
    const history=$('mode').value==='history';
    for(const id of ['time','trail','play','window','speed','early','pmtDecay','pmtMemory'])$(id).disabled=history;
    $('window').querySelector('[value="late"]').disabled=(this.source?.latest_emission_ns||0)<=180;
    for(const option of $('paths').options)if(option.value!=='all')option.disabled=Number(option.value)>this.maxCapacity;
    $('paths').querySelector('[value="all"]').disabled=Math.round(this.source?.total_yield||272157)>this.maxCapacity;
  }
  readAim(){return {azimuth:Number($('azimuth').value),elevation:Number($('elevation').value)};}
  aimLabels(){
    const aim=this.readAim();$('azimuthLabel').textContent=aim.azimuth.toFixed(1)+'°';$('elevationLabel').textContent=aim.elevation.toFixed(1)+'°';
    $('aimLabel').textContent=`az ${aim.azimuth.toFixed(1)}° · el ${aim.elevation.toFixed(1)}°`;
  }
  async applyAim(aim=this.readAim()){
    if(this.busy)return;
    $('azimuth').value=aim.azimuth;$('elevation').value=aim.elevation;this.aimLabels();
    await this.simulate({aim:this.readAim()});
  }
  setViewPreset(name){
    if(name==='early'||name==='origin'){this.focusEmission(name==='origin');return;}
    this.view=structuredClone(PRESETS[name]);
    if(name==='cone' && this.pose){
      for(const key of ['eye','target'])this.view[key]=rotatePoint(this.pose.rotation,this.view[key],this.pose.pivot);
    }
    this.requestRender();
  }
  focusEmission(tight=false){
    if(!this.source)return;
    const first=this.sourceRows.reduce((a,b)=>b[3]<a[3]?b:a);
    const target=first.slice(0,3).map((x,i)=>x+first[8+i]*(tight?120:700));
    const offset=rotateVector(this.pose.rotation,tight?[450,-600,220]:[1500,-2000,700]);
    this.view={eye:target.map((x,i)=>x+offset[i]),target,fov:62};this.requestRender();
  }
  async prepareHits(records,expected){
    const offsets=new Uint32Array(this.sensorCount+1),times=new Float32Array(expected);let total=0;
    for(let i=0;i<records.length;i+=4){const sensor=records[i+1];if(!sensor)continue;
      if(!Number.isInteger(sensor)||sensor<1||sensor>this.sensorCount||!Number.isFinite(records[i])||records[i]<0||records[i+3]!==i/4)throw Error('Invalid PMT arrival');
      offsets[sensor]++;total++;
    }
    if(total!==expected)throw Error('PMT arrival accounting mismatch');
    for(let i=1;i<offsets.length;i++)offsets[i]+=offsets[i-1];
    const cursor=offsets.slice();
    for(let i=0;i<records.length;i+=4)if(records[i+1])times[cursor[records[i+1]-1]++]=records[i];
    for(let i=0;i<this.sensorCount;i++)times.subarray(offsets[i],offsets[i+1]).sort();
    const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
    Object.assign(this.eventBuffers,await allocateGpu(this.device,a=>({hitOffsets:a.buffer(offsets,usage),hitTimes:a.buffer(times,usage),hitStates:a.buffer(this.sensorCount*16,usage)})));
    this.allHitTimes=times.slice().sort();this.hitOffsets=offsets;this.hitTimes=times;
  }
  arrivedAt(time){let lo=0,hi=this.allHitTimes?.length||0;while(lo<hi){const mid=(lo+hi)>>>1;if(this.allHitTimes[mid]<=time)lo=mid+1;else hi=mid;}return lo;}
  releaseEvent(){if(this.eventBuffers)Object.values(this.eventBuffers).forEach(b=>b.destroy());this.eventBuffers=null;this.allHitTimes=null;}
  async download(buffer){
    const read=await allocateGpu(this.device,a=>a.buffer(buffer.size,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ));
    try{const encoder=this.device.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,read,0,buffer.size);this.device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);return read.getMappedRange().slice(0);}finally{read.destroy();}
  }
  async simulate({event=$('event').value,seed=this.seed,count=null,debug=0,disableOptics=false,aim=this.readAim()}={}){
    if(this.busy)throw Error('An event is already being simulated');
    this.busy=true;for(const id of ['azimuth','elevation','aimCamera','aimReset'])$(id).disabled=true;$('simulate').disabled=true;$('event').disabled=true;$('paths').disabled=true;$('status').classList.remove('error');
    this.playing=false;$('play').textContent='Play';gpuBudget.cancel();
    while(this.rendering)await new Promise(r=>setTimeout(r,10));
    const started=performance.now();
    try{
      this.source=this.events.find(e=>e.id===event);if(!this.source)throw Error('Unknown event');$('event').value=event;
      this.pose=poseEvent(this.source.rows,aim);this.sourceRows=this.pose.rows;this.aim=this.pose.aim;
      $('azimuth').value=this.aim.azimuth;$('elevation').value=this.aim.elevation;this.aimLabels();
      this.syncTimeControls();this.timeWindow($('window').value,false);
      this.seed=seed;this.count=count??Math.round(this.source.total_yield);
      if(!Number.isInteger(this.count)||this.count<1||this.count>3000000)throw Error('Invalid photon count');
      this.stride=Math.ceil(this.count/this.capacity);this.retained=Math.ceil(this.count/this.stride);this.debugCount=Math.min(debug,this.count,4096);
      this.releaseEvent();await this.device.queue.onSubmittedWorkDone();
      const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
      // Keep the default small. A larger requested path limit allocates only on simulation.
      const requested=$('paths').value==='all'?this.count:Number($('paths').value);
      this.capacity=Math.min(this.maxCapacity,Math.max(this.capacity,Math.min(requested,this.count)));
      this.stride=Math.ceil(this.count/this.capacity);this.retained=Math.ceil(this.count/this.stride);
      for(;;){try{this.eventBuffers=await allocateGpu(this.device,a=>({
        sources:a.buffer(new Float32Array(this.sourceRows.flat()),storage),
        flights:a.buffer(this.capacity*32*64,storage),counts:a.buffer(this.capacity*4,storage),
        hits:a.buffer(this.count*16,storage),stats:a.buffer((128+this.source.rows.length)*4,storage),debug:a.buffer(Math.max(16,this.debugCount*64),storage)
      }));break;}catch(error){
        if(!isGpuMemoryError(error)||this.capacity<=2048)throw error;
        this.capacity=Math.max(2048,Math.floor(this.capacity/2));this.maxCapacity=Math.min(this.maxCapacity,this.capacity);
        this.stride=Math.ceil(this.count/this.capacity);this.retained=Math.ceil(this.count/this.stride);
      }}
      this.device.queue.writeBuffer(this.base.camera,0,this.cameraConfig(1,1));
      this.device.queue.writeBuffer(this.base.settings,0,this.settings(0,disableOptics));
      this.device.queue.writeBuffer(this.base.failures,0,new Uint32Array(1));
      const e=this.eventBuffers,b=this.base;
      const group=this.group(this.transport,{0:b.scene,1:b.camera,4:b.failures,6:b.settings,7:e.sources,8:e.flights,9:e.counts,10:e.stats,11:e.debug,12:e.hits});
      await checkedGpuWork(this.device,()=>gpuBudget.run(this.device,Math.ceil(this.count/128),(offset,n)=>{
        this.device.queue.writeBuffer(b.settings,4,new Float32Array([offset*128]));
        const encoder=this.device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(this.transport);pass.setBindGroup(0,group);pass.dispatchWorkgroups(n);pass.end();return encoder;
      },{initial:32,maximum:128,progress:(done,total)=>{$('status').textContent=`Transporting ${this.count.toLocaleString()} photons · ${Math.round(100*done/total)}%`;}}));
      const counters=Array.from(new Uint32Array(await this.download(e.stats)));
      const failed=new Uint32Array(await this.download(b.failures))[0];
      if(failed)throw Error(`${failed} geometry traversal failures`);
      const terminal=counters.slice(1,6).reduce((a,b)=>a+b,0)+counters[7];
      if(terminal!==this.count)throw Error('Incomplete photon accounting');
      await this.prepareHits(new Float32Array(await this.download(e.hits)),counters[1]);
      this.result={event,aim:{...this.aim},primary_direction:this.pose.axis.slice(),photons:this.count,retained:this.retained,counters,wall_ms:performance.now()-started,geometryFailures:failed};
      if(this.debugCount)this.result.debug=Array.from(new Float32Array(await this.download(e.debug)));
      if($('window').value==='arrivals')this.timeWindow('arrivals',false);
      this.eventGeneration++;
      $('status').textContent=`${this.count.toLocaleString()} photons · ${counters[1].toLocaleString()} PMT intersections · ${counters[6].toLocaleString()} Rayleigh scatters · ${counters[5]+counters[7]} unfinished · ${(this.result.wall_ms/1000).toFixed(2)} s`;
      $('meta').textContent=`HK ${this.source.config} / ${this.source.name} · ${this.source.energy_MeV.toFixed(3)} MeV recorded primary energy · ${this.source.tracks} tracks · ${this.source.input_steps.toLocaleString()} recorded steps · ${(this.eventBuffers.flights.size/1048576).toFixed(0)} MiB path storage`;
      this.syncTimeControls();return this.result;
    }catch(error){this.releaseEvent();throw error;}
    finally{this.busy=false;for(const id of ['azimuth','elevation','aimCamera','aimReset'])$(id).disabled=false;$('simulate').disabled=false;$('event').disabled=false;$('paths').disabled=false;if(this.eventBuffers)this.requestRender();}
  }
  async render({width=this.software?340:(innerWidth<=700?720:1440),height=Math.round(width/(innerWidth<=700?1.2:1.7))}={}){
    if(this.busy||this.rendering||!this.eventBuffers)return;
    this.rendering=true;
    try{
      const key=JSON.stringify([this.view,width,height]);
      if(this.width!==width||this.height!==height){
        this.targets&&Object.values(this.targets).forEach(t=>t.destroy());this.targets=null;
        this.targets=await allocateGpu(this.device,a=>({
          color:a.texture({size:[width,height],format:'rgba8unorm',usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING}),
          sensor:a.texture({size:[width,height],format:'r32uint',usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING}),
          depth:a.texture({size:[width,height],format:'r32float',usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING})
        }));this.width=width;this.height=height;this.cameraKey=null;
      }
      const b=this.base,e=this.eventBuffers,t=this.targets;
      this.device.queue.writeBuffer(b.camera,0,this.cameraConfig(width,height));
      const frameSettings=this.settings(),renderedTime=this.eventTime;
      this.device.queue.writeBuffer(b.settings,0,frameSettings);
      if(key!==this.cameraKey){
        const group=this.group(this.cameraPipeline,{0:b.scene,1:b.camera,2:t.color.createView(),4:b.failures,5:t.depth.createView(),16:t.sensor.createView()});
        await checkedGpuWork(this.device,()=>gpuBudget.run(this.device,Math.ceil(width/8)*Math.ceil(height/8),(offset,n)=>{
          this.device.queue.writeBuffer(b.camera,88,new Uint32Array([offset]));
          const encoder=this.device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(this.cameraPipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(n);pass.end();return encoder;
        },{initial:128,maximum:512,interactive:this.dragging}));this.cameraKey=key;
      }
      if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
      const encoder=this.device.createCommandEncoder();
      const hits=encoder.beginComputePass();hits.setPipeline(this.hitPipeline);hits.setBindGroup(0,this.group(this.hitPipeline,{6:b.settings,13:e.hitOffsets,14:e.hitTimes,15:e.hitStates}));hits.dispatchWorkgroups(Math.ceil(this.sensorCount/128));hits.end();
      const pass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:1}}]});
      pass.setPipeline(this.backgroundPipeline);pass.setBindGroup(0,this.group(this.backgroundPipeline,{2:b.settings,4:t.color.createView(),6:t.sensor.createView(),7:e.hitStates}));pass.draw(3);
      pass.setPipeline(this.linePipeline);pass.setBindGroup(0,this.group(this.linePipeline,{0:e.flights,1:b.camera,2:b.settings,3:t.depth.createView(),5:e.counts}));
      pass.draw(6,this.displayCount()*32);pass.end();
      this.device.queue.submit([encoder.finish()]);await this.device.queue.onSubmittedWorkDone();
      this.frame++;this.lastFrame={frame:this.frame,view:structuredClone(this.view),time:renderedTime,width,height,eventGeneration:this.eventGeneration};
      const history=frameSettings[7]>.5,arrivals=this.arrivedAt(history?Infinity:frameSettings[8]);
      $('hitCount').textContent=`${arrivals.toLocaleString()} / ${this.allHitTimes.length.toLocaleString()} PMT arrivals${history?' · complete event':` by ${renderedTime.toFixed(3)} ns`}${!arrivals&&this.allHitTimes.length?` · first at ${this.allHitTimes[0].toFixed(1)} ns`:''}`;
      $('pathInfo').textContent=`${this.count.toLocaleString()} simulated photons · ${this.retained.toLocaleString()} retained histories · display limit ${this.displayCount().toLocaleString()} · ${(e.flights.size/1048576).toFixed(1)} MiB path storage. Paths outside the time window are hidden.`;
      return this.lastFrame;
    }finally{this.rendering=false;}
  }
  requestRender(){this.pending=true;if(this.scheduled)return;this.scheduled=true;requestAnimationFrame(()=>this.drain());}
  async drain(){
    try{
      if(!document.hidden&&!this.busy&&!this.rendering&&this.pending){
        this.pending=false;await this.render({width:this.dragging?(this.software?240:510):(this.software?340:(innerWidth<=700?720:1440))});
      }
    }catch(error){this.fail(error);}finally{this.scheduled=false;if(this.pending&&!document.hidden&&!this.busy)this.requestRender();}
  }
  controls(){
    for(const id of ['azimuth','elevation']){
      $(id).oninput=()=>this.aimLabels();
      $(id).onchange=()=>this.applyAim().catch(e=>this.fail(e));
    }
    $('aimReset').onclick=()=>this.applyAim({azimuth:0,elevation:90}).catch(e=>this.fail(e));
    $('aimCamera').onclick=()=>{
      const d=this.basis().forward;
      this.applyAim({azimuth:Math.atan2(d[1],d[0])*180/Math.PI,elevation:Math.asin(Math.max(-1,Math.min(1,d[2])))*180/Math.PI}).catch(e=>this.fail(e));
    };
    $('simulate').onclick=()=>{const requested=$('paths').value==='all'?Math.round(this.source?.total_yield||272157):Number($('paths').value);this.capacity=Math.min(this.maxCapacity,Math.max(this.software?2048:32768,requested));this.simulate().catch(e=>this.fail(e));};
    $('event').onchange=()=>{this.seed=901;this.simulate().then(()=>{if(['early','origin'].includes($('preset').value))this.focusEmission($('preset').value==='origin');}).catch(e=>this.fail(e));};
    $('stop').onclick=()=>{this.playing=false;$('play').textContent='Play';this.pending=false;gpuBudget.cancel();};
    for(const id of ['paths','mode','brightness','opacity','time','trail','highlight','fade','pmtDecay','pmtMemory','pmtGain'])$(id).addEventListener('input',()=>{
      if(id==='pmtMemory')$('pmtMemoryLabel').textContent=Math.round(Number($('pmtMemory').value)*100)+'%';
      if(id==='time')this.setEventTime($('time').value);
      if(id==='paths'){const requested=$('paths').value==='all'?this.count:Number($('paths').value);if(requested>this.capacity&&!this.busy){this.simulate().catch(e=>this.fail(e));return;}}
      if(id==='trail'){$('trailLabel').textContent=$('trail').value+' ns';}
      if(id==='mode'){
        this.syncTimeControls();
        this.playing=false;$('play').textContent='Play';
      }
      this.requestRender();
    });
    $('preset').onchange=()=>this.setViewPreset($('preset').value);
    $('window').onchange=()=>this.timeWindow($('window').value);
    $('early').onclick=()=>{
      this.playing=false;$('play').textContent='Play';this.timeWindow('early');this.setEventTime(3);
      $('speed').value='0.1';$('brightness').value='.1';$('trail').value='.5';$('trailLabel').textContent='.5 ns';$('fade').value='5';
      $('preset').value='early';this.focusEmission();
    };
    $('play').onclick=()=>{this.playing=!this.playing;$('play').textContent=this.playing?'Pause':'Play';if(this.playing)this.animate();};
    $('save').onclick=async()=>{
      if(this.busy)return;this.playing=false;$('play').textContent='Play';
      while(this.rendering)await new Promise(r=>setTimeout(r,10));
      await this.render();this.canvas.toBlob(blob=>{if(!blob)return;const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${this.source.id}-photon-histories.png`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
    };
    window.addEventListener('resize',()=>this.requestRender());
    document.addEventListener('visibilitychange',()=>{if(!document.hidden){this.requestRender();}});
    let pointer=null;
    this.canvas.onpointerdown=e=>{
      if(this.busy||![0,1].includes(e.button))return;e.preventDefault();this.dragging=true;
      pointer={id:e.pointerId,x:e.clientX,y:e.clientY,pan:e.button===1||e.shiftKey};this.canvas.setPointerCapture(e.pointerId);
    };
    this.canvas.onpointermove=e=>{
      if(!pointer||pointer.id!==e.pointerId)return;
      const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;pointer.x=e.clientX;pointer.y=e.clientY;
      const delta=this.view.eye.map((x,i)=>x-this.view.target[i]),radius=Math.hypot(...delta);
      if(pointer.pan){
        const b=this.basis(),scale=2*radius*Math.tan(this.view.fov*Math.PI/360)/this.canvas.clientHeight;
        const move=b.right.map((x,i)=>scale*(-dx*x+dy*b.up[i]));
        this.view.eye=this.view.eye.map((x,i)=>x+move[i]);this.view.target=this.view.target.map((x,i)=>x+move[i]);
      }else{
        const az=Math.atan2(delta[1],delta[0])-dx*.005,el=Math.max(-1.5,Math.min(1.5,Math.asin(delta[2]/radius)+dy*.004));
        this.view.eye=[Math.cos(az)*Math.cos(el),Math.sin(az)*Math.cos(el),Math.sin(el)].map((x,i)=>this.view.target[i]+x*radius);
      }
      this.requestRender();
    };
    const release=()=>{pointer=null;this.dragging=false;this.requestRender();};
    this.canvas.onpointerup=release;this.canvas.onpointercancel=release;
    this.canvas.onauxclick=e=>{if(e.button===1)e.preventDefault();};
    this.canvas.onwheel=e=>{e.preventDefault();if(this.busy)return;const scale=Math.exp(Math.max(-1,Math.min(1,e.deltaY*.001)));this.view.eye=this.view.eye.map((x,i)=>this.view.target[i]+(x-this.view.target[i])*scale);this.requestRender();};
  }
  async animate(){
    if(this.animating)return;this.animating=true;
    let previous=performance.now();
    try{while(this.playing){
      const start=performance.now();
      const elapsed=Math.min(.1,(start-previous)/1000);previous=start;
      if(!document.hidden&&!this.busy){
        const min=Number($('time').min),span=Number($('time').max)-min;
        this.setEventTime(min+(this.eventTime-min+elapsed*Number($('speed').value))%span);this.requestRender();
      }
      await new Promise(r=>setTimeout(r,Math.max(0,33-(performance.now()-start))));
    }}finally{this.animating=false;}
  }
}
window.photonHistories=new PhotonHistories();
window.photonHistoriesReady=photonHistories.initialize().catch(error=>{photonHistories.fail(error);throw error;});
