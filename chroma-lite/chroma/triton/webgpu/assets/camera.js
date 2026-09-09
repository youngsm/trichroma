import {
  OpticalLab, STAT_WORDS
} from './physics.js';
import {gpuBudget} from './scheduler.js';
import {allocateGpu, checkedGpuWork, isGpuMemoryError} from './gpu.js';
const $ = id => document.getElementById(id);
const MAP_PROFILES = {
  full: {wall:128, fill:32, emission:64, volume:[64,40,24]},
  compact: {wall:64, fill:16, emission:32, volume:[32,20,12]}
};
function mapSizes(profile, pmt) {
  const m=MAP_PROFILES[profile], bytes=64*4;
  return {
    wall:6*m.wall*m.wall*bytes,
    volume:m.volume.reduce((a,b)=>a*b)*6*bytes,
    wls:(pmt ? 2*pmt.wls_charts : 12*m.emission*m.emission)*bytes,
    fill_wall:6*m.fill*m.fill*bytes
  };
}
const HALF = [300, 200, 120];

function checkedInteger(value, low, high, name) {
  if (!Number.isSafeInteger(value) || value < low || value > high) throw Error(
    `${name} must be an integer in ${low}…${high}`);
  return value;
}

function status(message, error = false) {
  if ($('status')) {
    $('status').textContent = message;
    $('status').classList.toggle('error', error);
  }
}
export class PhotonCamera extends OpticalLab {
  async initialize() {
    try { return await this.initializeOnce(); }
    catch (error) {
      if ((!isGpuMemoryError(error) && !isGpuMemoryError(this.lost)) || this.preferredMapProfile==='compact') throw error;
      const oldDevice=this.device;
      this.device=null;
      if (oldDevice) {oldDevice.destroy();await oldDevice.lost;}
      this.lost=null;
      this.preferredMapProfile='compact';
      this.memoryRecovered=true;
      if ($('memory_mode')) $('memory_mode').value='compact';
      return this.initializeOnce();
    }
  }
  async initializeOnce() {
    await super.initialize({
      catalogName: 'camera-catalog.json'
    });
    if (this.software) this.preferredMapProfile='compact';
    for (const {
        scene
      }
      of Object.values(this.scenes)) {
      if (scene.camera?.version !== 1 || scene.camera.wavelength_bins !== 64) throw Error(
        'Unsupported photon camera manifest');
    }
    for (const entry of Object.values(this.scenes)) {
      const old = entry.packed,
        pmt = entry.scene.camera.pmt,
        bvh = entry.scene.camera.geometry_bvh,
        count = entry.scene.triangles.length,
        packed = new Uint32Array(old.length + 288 + (pmt ? 3 * count : 0) + (bvh ? 5 * bvh
          .node_count : 0));
      packed.set(old);
      packed[40] = old.length;
      packed[41] = old.length + 192;
      const pf = new Float32Array(packed.buffer),
        settings = entry.scene.camera,
        light = settings.fill_light;
      pf.set(settings.colors.linear_srgb.flat(), old.length);
      const data = new Float32Array(96);
      data.set(light.radiance_per_bin);
      data.set(light.center, 64);
      data.set(light.full_size.map(x => x / 2), 67);
      data[69] = light.full_size[0] * light.full_size[1];
      data.set(settings.fluorescence_center || [0, 0, 0], 70);
      data.set(settings.fluorescence_half_extent, 73);
      data[76] = settings.wall_reflectance;
      data[77] = settings.source_mix.beam_probability;
      data[78] = light.integrated_radiance * data[69] * Math.PI;
      data[79] = 120 - settings.source_mix.fill_origin_z;
      const fillLayout = light.layout || 'ceiling';
      if (!['ceiling', 'room_faces'].includes(fillLayout)) throw Error('Unsupported fill-light layout');
      data[85] = fillLayout === 'room_faces' ? 1 : 0;
      if (pmt) {
        if ([pmt.triangle_role, pmt.triangle_chart, pmt.triangle_area_mm2].some(a =>
            !Array.isArray(a) || a.length !== count)) throw Error('Invalid PMT triangle atlas');
        checkedInteger(pmt.wls_charts, 1, count, 'WLS charts');
        if (pmt.triangle_role.some(v => !Number.isInteger(v) || v < 0 || v > 3) ||
          pmt.triangle_chart.some(v => !Number.isInteger(v) || v < -1 || v >= pmt.wls_charts) ||
          pmt.triangle_area_mm2.some(v => !Number.isFinite(v) || v <= 0) ||
          pmt.triangle_role.some((role, i) => (role === 1) !== (pmt.triangle_chart[i] >= 0)))
          throw Error('Invalid PMT role, chart or facet area');
        if (pmt.clip_plane_normal.length !== 3 || pmt.clip_plane_normal.some(v => !Number
            .isFinite(v)) ||
          Math.abs(Math.hypot(...pmt.clip_plane_normal) - 1) > 1e-6 || !Number.isFinite(pmt
            .clip_plane_offset))
          throw Error('Invalid PMT clip plane');
        packed[42] = old.length + 288;
        packed[43] = packed[42] + count;
        packed[44] = packed[43] + count;
        packed.set(pmt.triangle_role, packed[42]);
        new Int32Array(packed.buffer).set(pmt.triangle_chart, packed[43]);
        pf.set(pmt.triangle_area_mm2, packed[44]);
        data.set(pmt.clip_plane_normal, 80);
        data[83] = pmt.clip_plane_offset;
        data[84] = pmt.keep_positive ? 1 : -1;
      }
      if (bvh) {
        checkedInteger(bvh.node_count, 1, 65536, 'BVH nodes');
        if (bvh.nodes.length !== bvh.node_count || bvh.escape_links.length !== bvh.node_count ||
          bvh.nodes.some(node => node.length !== 4 || node.some(v => !Number.isInteger(v) || v <
            0 || v > 0xffffffff)) ||
          bvh.escape_links.some(v => !Number.isInteger(v) || v < 0 || (v !== 0xffffffff && v >=
            bvh.node_count)) ||
          bvh.world_origin.length !== 3 || bvh.world_origin.some(v => !Number.isFinite(v)) ||
          !Number.isFinite(bvh.world_scale) || bvh.world_scale <= 0)
          throw Error('Invalid camera BVH');
        for (const node of bvh.nodes) {
          const child = node[3] & 0x0fffffff,
            children = node[3] >>> 28;
          if (children ? child + children > bvh.node_count : child >= count)
            throw Error('Invalid camera BVH child');
        }
        packed[45] = old.length + 288 + (pmt ? 3 * count : 0);
        packed[46] = packed[45] + bvh.node_count * 4;
        packed[47] = bvh.node_count;
        packed.set(bvh.nodes.flat(), packed[45]);
        packed.set(bvh.escape_links, packed[46]);
        data.set(bvh.world_origin, 88);
        data[91] = bvh.world_scale;
      }
      pf.set(data, old.length + 192);
      entry.packed = packed;
    }
    let [physics, deposition, camera] = await Promise.all(['physics.wgsl', 'deposition.wgsl',
      'camera.wgsl'
    ].map(name => fetch(name).then(r => r.text())));
    const hooks = {
      '// CAMERA_BVH_HOOK': 'if(tables[45u]!=0u){return nearest_camera_bvh(origin,direction,previous,false);}',
      '// CAMERA_SOURCE_HOOK': `
        let light=tables[41u];
        let beam_probability=f(light+77u);
        let is_fill=random_uniform(id,0x10000008u)>=beam_probability;
        var packet_scale=1./beam_probability;
        if(is_fill){
          if(f(light+85u)>.5){
            let x=(2.*random_uniform(id,0x10000004u)-1.)*f(light+67u);
            let y=(2.*random_uniform(id,0x10000005u)-1.)*f(light+68u);
            let face=min(5u,u32(6.*random_uniform(id,0x10000009u)));
            let axis=face/2u;
            let sign=select(-1.,1.,face%2u==0u);
            let u_axis=select(0u,1u,axis==0u);
            let v_axis=select(2u,1u,axis==2u);
            position=vec3<f32>(0.);
            position[axis]=sign*(ROOM[axis]-.001);
            position[u_axis]=x;position[v_axis]=y;
            var fill_normal=vec3<f32>(0.);fill_normal[axis]=-sign;
            direction=hemisphere(fill_normal,random_uniform(id,0x10000006u),random_uniform(id,0x10000007u),true);
          }else{
            // Retain the original expression so driver FMA rounding is unchanged.
            position=v3(light+64u)+vec3<f32>((2.*random_uniform(id,0x10000004u)-1.)*f(light+67u),(2.*random_uniform(id,0x10000005u)-1.)*f(light+68u),-f(light+79u));
            direction=hemisphere(vec3<f32>(0.,0.,-1.),random_uniform(id,0x10000006u),random_uniform(id,0x10000007u),true);
          }
          initial_position=position;
          polarization=random_polarization(direction,random_uniform(id,0x10000002u));
          wavelength=390.+320.*random_uniform(id,0x10000003u);
          packet_scale=f(light+78u)/(1.-beam_probability)*wavelength/450.;
          atomicAdd(&statistics[21u],1u);
        }else{atomicAdd(&statistics[20u],1u);}`,
      '// CAMERA_SCATTER_HOOK': 'deposit_scatter(position,polarization,wavelength,packet_scale);',
      '// CAMERA_EMISSION_HOOK': 'deposit_emission(position,normal,direction,wavelength,packet_scale,tri);',
      '// CAMERA_BOUNDARY_HOOK': 'if((flags&4u)!=0u){if(tables[42u]!=0u&&tables[tables[42u]+tri]==2u){atomicAdd(&statistics[25u],1u);}else{deposit_wall(position,normal,wavelength,packet_scale,is_fill);}}',
      '// CAMERA_ESCAPE_HOOK': 'if(config.options.z==0u){let escaped=atomicAdd(&statistics[23u],1u);if(escaped<256u){final_state[256u+escaped]=id;}}',
      '// CAMERA_TAIL_HOOK': 'if(config.options.z==0u){let tail=atomicAdd(&statistics[22u],1u);if(tail<256u){final_state[tail]=id;}}'
    };
    for (const [marker, code] of Object.entries(hooks)) {
      if (physics.split(marker).length !== 2) throw Error(
        'Transport instrumentation hook changed: ' + marker);
      physics = physics.replace(marker, code);
    }
    const module = this.device.createShaderModule({
      code: physics + '\n' + deposition + '\n' + camera
    });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (errors.length) throw Error(errors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join(
      '\n'));
    this.shaderModule = module;
    this.mapProfile = null;
    await this.setMapProfile(this.preferredMapProfile || 'full');
    this.canvas = $('camera');
    this.context = this.canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque'
    });
    const present = this.device.createShaderModule({
      code: `@group(0) @binding(0) var rendered:texture_2d<f32>;
@vertex fn vertex(@builtin(vertex_index)i:u32)->@builtin(position)vec4<f32>{var p=array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));return vec4<f32>(p[i],0.,1.);}
@fragment fn fragment(@builtin(position)p:vec4<f32>)->@location(0)vec4<f32>{return textureLoad(rendered,vec2<i32>(p.xy),0);}`
    });
    this.presentPipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: present,
        entryPoint: 'vertex'
      },
      fragment: {
        module: present,
        entryPoint: 'fragment',
        targets: [{
          format: this.format
        }]
      },
      primitive: {
        topology: 'triangle-list'
      }
    });
    this.frame = 0;
    this.exposure = 100000;
    return this.info;
  }
  async setMapProfile(profile) {
    if (!Object.hasOwn(MAP_PROFILES, profile)) throw Error('Unsupported light-map detail.');
    if (this.mapProfile === profile) return;
    const m=MAP_PROFILES[profile];
    const constants={WALL_RES:m.wall,FILL_RES:m.fill,EMISSION_RES:m.emission,
      VOLUME_X:m.volume[0],VOLUME_Y:m.volume[1],VOLUME_Z:m.volume[2],
      VOXEL_VOLUME_MM3:57600000/m.volume.reduce((a,b)=>a*b)};
    // Compile sequentially to avoid overlapping driver compilation peaks.
    const pipelines = await checkedGpuWork(this.device, async () => {
      const deposition = await this.device.createComputePipelineAsync({layout:'auto',
        compute:{module:this.shaderModule,entryPoint:'simulate',constants}});
      const camera = await this.device.createComputePipelineAsync({layout:'auto',
        compute:{module:this.shaderModule,entryPoint:'render_camera',constants}});
      return {deposition,camera};
    });
    this.depositionPipeline=pipelines.deposition;
    this.cameraPipeline=pipelines.camera;
    this.mapProfile=profile;
  }
  async recoverMemory(error) {
    if (this.mapProfile === 'compact') {
      throw Error('GPU memory allocation failed even with reduced light-map detail. Reload after freeing GPU memory. '+(error.message || error));
    }
    const generation=gpuBudget.generation;
    const view={eye:this.eye?.slice(),target:this.target?.slice(),fov:this.fov,
      exposure:this.exposure,cutaway:this.cutaway,uvFalseColor:this.uvFalseColor};
    this.busy=true;
    try {
      this.releaseMaps();
      const oldDevice=this.device;
      this.device=null;
      oldDevice.destroy();
      await oldDevice.lost;
      this.lost=null;
      this.preferredMapProfile='compact';
      if ($('memory_mode')) $('memory_mode').value='compact';
      if ($('memory_note')) $('memory_note').textContent='Recovering from a GPU memory allocation failure with reduced light-map detail…';
      await this.initialize();
      Object.assign(this,view);
      if (generation !== gpuBudget.generation) throw new DOMException('Stopped.','AbortError');
      this.memoryRecovered=true;
    } finally { this.busy=false; }
  }
  async simulate(options = {}) {
    if (this.busy || this.rendering) throw Error('The browser GPU is already working');
    const profile=options.memory || this.preferredMapProfile || 'full';
    try {
      // Keep camera gestures from rendering old maps while their pipelines change.
      this.busy=true;
      try { await this.setMapProfile(profile); }
      finally { this.busy=false; }
      const result=await this.simulateOnce(options);
      this.lastSimulationOptions={...options,scene:result.scene,photons:result.photons,seed:result.seed,polarization:result.polarization};
      return result;
    } catch (error) {
      if (!isGpuMemoryError(error) && !isGpuMemoryError(this.lost)) throw error;
      await this.recoverMemory(error);
      const result=await this.simulateOnce(options);
      this.lastSimulationOptions={...options,scene:result.scene,photons:result.photons,seed:result.seed,polarization:result.polarization};
      return result;
    }
  }
  async render(options = {}) {
    try { return await this.renderOnce(options); }
    catch (error) {
      if ((!isGpuMemoryError(error) && !isGpuMemoryError(this.lost)) || !this.lastSimulationOptions) throw error;
      await this.recoverMemory(error);
      await this.simulateOnce(this.lastSimulationOptions);
      return this.renderOnce({...options,reset:true});
    }
  }
  async download(buffer) {
    const read = this.buffer(buffer.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, read, 0, buffer.size);
      this.device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const data = read.getMappedRange().slice(0);
      read.unmap();
      return data;
    } finally {
      read.destroy();
    }
  }
  releaseMaps() {
    if (this.resources) {
      Object.values(this.resources).forEach(b => b.destroy());
      this.resources = null;
    }
    this.image?.destroy();
    this.image = null;
    this.accumulation?.destroy();
    this.accumulation = null;
  }
  async simulateOnce({
    scene = 'prism',
    photons = 2500000,
    seed = 901,
    polarization = 'random',
    debug = false,
    debugMaps = false
  } = {}) {
    if (this.busy || this.rendering) throw Error('The browser GPU is already working');
    if (this.lost) throw Error('GPU device lost: '+this.lost);
    if (!this.scenes[scene]) throw Error('Unsupported camera scene');
    checkedInteger(photons, 1, 30000000, 'Photons');
    checkedInteger(seed, 0, Number.MAX_SAFE_INTEGER, 'Seed');
    if (!['random', 'y', 'z'].includes(polarization)) throw Error('Unsupported polarization');
    if (debug && photons > 65536) throw Error('Debug event limit is 65,536');
    this.busy = true;
    this.releaseMaps();
    const start = performance.now();
    try {
      const d = this.device,
        store = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        // The camera stores maps and optional terminal states; trajectories are not used.
        paths = 0;
      const cfg = new Uint32Array(12);
      cfg.set([photons, seed >>> 0, Math.floor(seed / 4294967296), paths, 8192, ['prism',
        'fluorescence', 'rayleigh', 'pmt'
      ].indexOf(scene), debug ? photons : 0, ['random', 'y', 'z'].indexOf(polarization)]);
      new Float32Array(cfg.buffer)[8] = scene === 'pmt' ? 10000 : scene === 'fluorescence' ? 205 :
        25;
      await d.queue.onSubmittedWorkDone();
      const mapBytes=mapSizes(this.mapProfile,this.scenes[scene].scene.camera.pmt);
      const resources = await allocateGpu(d, arena => ({
        tables: arena.buffer(this.scenes[scene].packed, store),
        config: arena.buffer(cfg, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
        statistics: arena.buffer((STAT_WORDS + paths) * 4, store),
        paths: arena.buffer(Math.max(4, paths * 8193 * 32), store),
        final: arena.buffer(debug ? photons * 80 : 2048, store),
        wall: arena.buffer(mapBytes.wall, store),
        volume: arena.buffer(mapBytes.volume, store),
        wls: arena.buffer(mapBytes.wls, store),
        fill_wall: arena.buffer(mapBytes.fill_wall, store)
      }));
      this.resources = resources;
      const group = d.createBindGroup({
        layout: this.depositionPipeline.getBindGroupLayout(0),
        entries: Object.values(resources).map((buffer, index) => ({
          binding: index === 8 ? 11 : index,
          resource: {
            buffer
          }
        }))
      });
      d.pushErrorScope('out-of-memory');
      d.pushErrorScope('validation');
      let timing;
      try {
        timing = await gpuBudget.run(d, Math.ceil(photons / 128), (offset, count) => {
          d.queue.writeBuffer(resources.config, 36, new Uint32Array([offset * 128, Math.min(photons, (offset + count) * 128)]));
          const encoder = d.createCommandEncoder(), pass = encoder.beginComputePass();
          pass.setPipeline(this.depositionPipeline); pass.setBindGroup(0, group);
          pass.dispatchWorkgroups(count); pass.end();
          return encoder;
        }, {progress: fraction => status(`Simulating ${photons.toLocaleString()} photons · ${Math.floor(fraction * 100)}%`)});
      } finally {
        const error = await d.popErrorScope();
        const memory = await d.popErrorScope();
        if (memory) {const e=new Error(memory.message || 'GPU memory allocation failed.');e.name='GpuMemoryError';throw e;}
        if (error) throw Error(error.message);
      }
      const compute_ms = timing.compute_ms;
      const stats = new Uint32Array(await this.download(resources.statistics));
      if (stats[0] !== photons || stats[4] || stats[5] || stats[6]) {
        const tails = !debug && stats[5] ? Array.from(new Uint32Array(
          await this.download(resources.final)).slice(0, Math.min(stats[22], 256))) : [];
        const escaped = !debug && stats[4] ? Array.from(new Uint32Array(
          await this.download(resources.final)).slice(256, 256 + Math.min(stats[23], 256))) : [];
        throw Error(`Incomplete camera event: processed ${stats[0]}, step limits ${stats[5]}, ` +
          `nonfinite ${stats[6]}; surviving photon IDs ${JSON.stringify(tails)}; escaped photon IDs ${JSON.stringify(escaped)}`
        );
      }
      const manifest = this.scenes[scene].scene;
      // Rebuilding a scene's photon maps should preserve the composed camera view.
      if (this.scene !== scene) {
        this.eye = manifest.camera.default_eye.slice();
        this.target = manifest.camera.default_target.slice();
        this.fov = manifest.camera.field_of_view_degrees;
        this.cutaway = false;
        this.uvFalseColor = false;
      }
      this.scene = scene;
      this.photons = photons;
      this.seed = seed;
      this.polarization = polarization;
      this.frame = 0;
      for (const [id, key] of [['cutaway', 'cutaway'], ['uv_false_color', 'uvFalseColor']])
        if ($(id)) {
          $(id).checked = this[key];
          $(id).disabled = !manifest.camera.pmt;
        }
      const result = {
        scene,
        photons,
        seed,
        polarization,
        counts: {
          detected: stats[1],
          absorbed: stats[2] + stats[3],
          escaped: stats[4],
          reemitted: stats[7],
          scattered: stats[8],
          max_steps: stats[12],
          beam: stats[20],
          fill: stats[21],
          boundary_roundings: stats[24],
          photocathode: stats[25]
        },
        metrics: {
          compute_ms,
          batches: timing.batches,
          wall_ms: performance.now() - start
        },
        map_bytes: mapBytes,
        map_profile: this.mapProfile,
        estimator: manifest.camera,
        adapter: this.info
      };
      if (debug) {
        const raw = await this.download(resources.final);
        const f = new Float32Array(raw),
          u = new Uint32Array(raw),
          i = new Int32Array(raw);
        const states = {
          pos: [],
          direction: [],
          polarization: [],
          wavelengths: [],
          times: [],
          flags: [],
          channels: [],
          last_hit_triangles: [],
          photon_ids: []
        };
        for (let n = 0; n < photons; n++) {
          const b = n * 20;
          states.pos.push(Array.from(f.slice(b, b + 3)));
          states.direction.push(Array.from(f.slice(b + 4, b + 7)));
          states.polarization.push(Array.from(f.slice(b + 8, b + 11)));
          states.wavelengths.push(f[b + 7]);
          states.times.push(f[b + 3]);
          states.flags.push(u[b + 12]);
          states.channels.push(i[b + 13]);
          states.last_hit_triangles.push(i[b + 14]);
          states.photon_ids.push(n);
        }
        result.states = states;
      }
      if (debugMaps) {
        result.maps = await this.debugMaps();
      }
      const m=MAP_PROFILES[this.mapProfile];
      result.estimator={...manifest.camera,wall_shape:[m.wall,m.wall],fill_wall_shape:[m.fill,m.fill],
        volume_shape:m.volume.slice(),voxel_volume_mm3:57600000/m.volume.reduce((a,b)=>a*b),
        fluorescence_shape:[m.emission,m.emission]};
      if ($('memory_note')) $('memory_note').textContent=
        `${this.memoryRecovered?'Recovered after a GPU memory allocation failed. ':''}${this.mapProfile==='compact'?'Reduced':'Full'} light-map detail · ${(Object.values(mapBytes).reduce((a,b)=>a+b)/1048576).toFixed(1)} MiB of maps · 64 wavelength bins.`;
      this.result = result;
      const cathode = result.counts.photocathode ?
        ` · ${result.counts.photocathode.toLocaleString()} photocathode detections` : '';
      status(
        `${photons.toLocaleString()} optical photons · ${result.counts.scattered.toLocaleString()} scattered · ${result.counts.reemitted.toLocaleString()} reemitted${cathode}`
      );
      if ($('pmt_legend')) $('pmt_legend').hidden = scene !== 'pmt';
      if ($('scene_note')) $('scene_note').textContent = scene === 'pmt' ?
        'R5912 geometry · 126–130 nm illumination · synthetic TPB, glass and photocathode calibration' :
        manifest.explanation;
      if ($('photons')) {
        $('photons').value = photons;
        $('scene').value = scene;
        $('seed').value = seed;
      }
      return result;
    } catch (error) {
      // A rejected event must never leave a partial map available to the camera.
      this.releaseMaps();
      this.result = null;
      throw error;
    } finally {
      this.busy = false;
    }
  }
  async mapSummary() {
    const summary = {};
    for (const key of ['wall', 'fill_wall']) {
      const values = new Float32Array(await this.download(this.resources[key]));
      let sum = 0,
        maximum = 0,
        nonzero = 0;
      for (const value of values) {
        sum += value;
        maximum = Math.max(maximum, value);
        if (value !== 0) nonzero++;
      }
      summary[key] = {
        energy_sum: sum,
        maximum_cell: maximum,
        nonzero_cells: nonzero
      };
    }
    summary.wall_energy = summary.wall.energy_sum + summary.fill_wall.energy_sum;
    return summary;
  }
  async debugMaps() {
    const output = {};
    for (const key of ['wall','volume','wls','fill_wall']) {
      const values = new Float32Array(await this.download(this.resources[key]));
      const pairs = [];
      for (let index = 0; index < values.length; index++)
        if (values[index] !== 0) pairs.push([index, values[index]]);
      output[key] = pairs;
    }
    return output;
  }
  async renderOnce({
    width = 640,
    height = 400,
    reset = false,
    eye = null,
    target = null,
    exposure = null,
    cutaway = null,
    uvFalseColor = null,
    interactive = false
  } = {}) {
    if (this.busy || this.rendering) throw Error('The browser GPU is already working');
    if (this.lost) throw Error('GPU device lost: '+this.lost);
    if (!this.resources) throw Error('Simulate an optical event first');
    checkedInteger(width, 8, 1920, 'Width');
    checkedInteger(height, 8, 1200, 'Height');
    if (eye) {
      if (eye.length !== 3 || eye.some((v, a) => !Number.isFinite(v) || Math.abs(v) >= HALF[a]))
        throw Error('Camera eye must remain inside the room');
      this.eye = eye.slice();
      reset = true;
    }
    if (target) {
      if (target.length !== 3 || target.some(v => !Number.isFinite(v))) throw Error(
        'Camera target must have 3 finite coordinates');
      this.target = target.slice();
      reset = true;
    }
    if (exposure !== null) {
      if (!Number.isFinite(exposure) || exposure <= 0) throw Error(
        'Exposure must be finite and positive');
      this.exposure = exposure;
      if ($('exposure')) $('exposure').value = Math.log10(exposure);
    }
    if (Math.hypot(...this.eye.map((x, i) => x - this.target[i])) < 1e-5) throw Error(
      'Camera eye and target must differ');
    for (const [key, value] of Object.entries({
        cutaway,
        uvFalseColor
      })) {
      if (value !== null) {
        if (typeof value !== 'boolean') throw Error(key + ' must be boolean');
        if (value && !this.scenes[this.scene].scene.camera.pmt)
          throw Error(key + ' is available only for the PMT scene');
        if (value !== this[key]) reset = true;
        this[key] = value;
        const control = $(key === "uvFalseColor" ? "uv_false_color" : key);
        if (control) control.checked = value;
      }
    }
    this.rendering = true;
    let uniform=null;
    try {
      const d = this.device;
      if (!this.image || width !== this.width || height !== this.height) {
        this.image?.destroy();
        this.accumulation?.destroy();
        this.image=null;this.accumulation=null;
        await d.queue.onSubmittedWorkDone();
        const target=await allocateGpu(d, arena => ({
          image:arena.texture({size:[width,height],format:'rgba16float',
            usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING}),
          accumulation:arena.buffer(width*height*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST)
        }));
        this.width=width;this.height=height;
        this.image=target.image;this.accumulation=target.accumulation;
        reset = true;
      }
      if (reset) this.frame = 0;
      const cfg = new Uint32Array(16),
        f = new Float32Array(cfg.buffer);
      f.set([...this.eye, this.exposure, ...this.target, Math.tan(this.fov * Math.PI / 360)]);
      cfg.set([width, height, this.frame, 0], 8);
      f.set([this.cutaway ? 1 : 0, this.uvFalseColor ? 1 : 0, 0, 0], 12);
      uniform = await allocateGpu(d, arena => arena.buffer(cfg, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));
      const r = this.resources;
      const bind = d.createBindGroup({
        layout: this.cameraPipeline.getBindGroupLayout(0),
        entries: [{
          binding: 0,
          resource: {
            buffer: r.tables
          }
        }, {
          binding: 1,
          resource: {
            buffer: r.config
          }
        }, {
          binding: 5,
          resource: {
            buffer: r.wall
          }
        }, {
          binding: 6,
          resource: {
            buffer: r.volume
          }
        }, {
          binding: 7,
          resource: {
            buffer: r.wls
          }
        }, {
          binding: 8,
          resource: {
            buffer: uniform
          }
        }, {
          binding: 9,
          resource: this.image.createView()
        }, {
          binding: 10,
          resource: {
            buffer: this.accumulation
          }
        }, {
          binding: 11,
          resource: {
            buffer: r.fill_wall
          }
        }]
      });
      const present = d.createBindGroup({
        layout: this.presentPipeline.getBindGroupLayout(0),
        entries: [{
          binding: 0,
          resource: this.image.createView()
        }]
      });
      d.pushErrorScope('out-of-memory');
      d.pushErrorScope('validation');
      let queue_ms;
      try {
        const timing = await gpuBudget.run(d, Math.ceil(width / 8) * Math.ceil(height / 8), (offset, count) => {
          d.queue.writeBuffer(uniform, 44, new Uint32Array([offset]));
          const encoder = d.createCommandEncoder(), compute = encoder.beginComputePass();
          compute.setPipeline(this.cameraPipeline); compute.setBindGroup(0, bind);
          compute.dispatchWorkgroups(count); compute.end();
          return encoder;
        }, {interactive});
        queue_ms = timing.compute_ms;
      } catch (error) {
        // An interrupted sample must not be mixed into a later camera sample.
        this.frame = 0;
        throw error;
      } finally {
        const error = await d.popErrorScope();
        const memory = await d.popErrorScope();
        if (memory) {const e=new Error(memory.message || 'GPU memory allocation failed.');e.name='GpuMemoryError';throw e;}
        if (error) throw Error(error.message);
      }
      if (this.lost) throw Error('GPU device lost: '+this.lost);
      // Keep the last presented preview visible while a larger image computes.
      // Resizing the canvas earlier would blank it for the whole refinement.
      if (this.canvas.width !== width) this.canvas.width = width;
      if (this.canvas.height !== height) this.canvas.height = height;
      const encoder = d.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: {
            r: 0,
            g: 0,
            b: 0,
            a: 1
          }
        }]
      });
      pass.setPipeline(this.presentPipeline);
      pass.setBindGroup(0, present);
      pass.draw(3);
      pass.end();
      const start = performance.now();
      d.queue.submit([encoder.finish()]);
      await d.queue.onSubmittedWorkDone();
      queue_ms += performance.now() - start;
      if (this.lost) throw Error('GPU device lost: '+this.lost);
      this.frame++;
      return this.lastFrame = {
        width,
        height,
        frame: this.frame,
        queue_ms,
        photons: this.photons,
        eye: Array.from(f.slice(0,3)),
        target: Array.from(f.slice(4,7)),
        exposure: f[3],
        cutaway: !!f[12],
        uv_false_color: !!f[13]
      };
    } finally {
      uniform?.destroy();
      this.rendering = false;
    }
  }
  async linearImage() {
    return Array.from(new Float32Array(await this.download(this.accumulation)));
  }
}
const camera = new PhotonCamera();
window.photonCamera = camera;
let request = 0, settle = null, pendingView = null, drainingViews = false;
let activePreview = false, previewWidth = 128;

function updateInfo(frame) {
  $('samples').textContent =
    `${camera.photons.toLocaleString()} photons · ${frame.frame} camera ${frame.frame === 1 ? 'pass' : 'passes'} · ${frame.queue_ms.toFixed(0)} ms camera`;
}
async function drainViews() {
  if (drainingViews) return;
  drainingViews = true;
  try {
    while (pendingView) {
      const next = pendingView;
      pendingView = null;
      while (camera.rendering) await new Promise(resolve => setTimeout(resolve,8));
      if (next.version !== request || camera.busy || !camera.resources) continue;
      try {
        // Finish a small preview while newer pointer positions coalesce. Repeated
        // cancellation during dragging would prevent any frame being presented.
        activePreview = true;
        const start = performance.now(), width = previewWidth;
        updateInfo(await camera.render({width,height:Math.round(width*.625),reset:true,interactive:true}));
        previewWidth = Math.max(camera.software?24:64,Math.min(camera.software?64:160,8*Math.round(width*Math.sqrt(80/Math.max(1,performance.now()-start))/8)));
        activePreview = false;
        if (next.preview || next.version !== request || pendingView) continue;
        const quality = Number($('quality').value), height = Math.round(quality*.625);
        for (let sample=0;sample<(camera.software?2:8) && next.version===request && !pendingView;sample++) {
          const begin = performance.now();
          updateInfo(await camera.render({width:quality,height,reset:sample===0,interactive:sample===0}));
          // Limit refinement of cheap scenes to 30 frames/s (10 in Low mode).
          // Expensive frames need no additional idle time.
          const interval = gpuBudget.mode==='fast' ? 0 : gpuBudget.mode==='eco' ? 100 : 1000/30;
          const wait = interval-(performance.now()-begin);
          if (sample<15 && wait>0) await new Promise(resolve=>setTimeout(resolve,wait));
        }
      } catch (error) {
        if (error.name !== 'AbortError') status(error.message,true);
      } finally { activePreview = false; }
    }
  } finally { drainingViews = false; }
}
function schedule(preview = false) {
  const version = ++request;
  clearTimeout(settle);
  pendingView = {version,preview};
  if (camera.rendering && !activePreview) gpuBudget.cancel();
  drainViews();
  if (preview) settle = setTimeout(()=>{
    if (version !== request) return;
    pendingView = {version,preview:false};
    drainViews();
  },250);
}
document.addEventListener('gpu-stop', () => {
  request++;
  pendingView = null;
  clearTimeout(settle);
  status('Stopped.');
});
window.photonCameraReady = camera.initialize().then(info => {
  if (info.isFallbackAdapter) {
    $('photons').value=10000;$('gpu_mode').value='eco';$('memory_mode').value='compact';
    $('quality').add(new Option('160 px','160'));$('quality').value='160';previewWidth=32;
  }
  $('adapter').textContent =
    `${info.vendor} ${info.architecture}${info.isFallbackAdapter?' · software adapter':''}`;
  $('simulate').disabled = false;
  $('simulate').onclick = async () => {
    request++;
    pendingView = null;
    clearTimeout(settle);
    if (camera.rendering) gpuBudget.cancel();
    $('simulate').disabled = true;
    while (camera.rendering) await new Promise(resolve => requestAnimationFrame(resolve));
    status('Simulating photons and building spectral light maps…');
    try {
      const result = await camera.simulate({
        scene: $('scene').value,
        photons: Number($('photons').value),
        seed: Number($('seed').value),
        polarization: $('polarization').value,
        memory: $('memory_mode')?.value || 'full'
      });
      status(
        `${result.photons.toLocaleString()} optical photons (${result.counts.beam.toLocaleString()} beam + ${result.counts.fill.toLocaleString()} fill) · ${result.counts.scattered.toLocaleString()} scattered · ${result.counts.reemitted.toLocaleString()} reemitted${result.scene === "pmt" ? " · " + result.counts.photocathode.toLocaleString() + " photocathode detections" : ""} · ${result.metrics.compute_ms.toFixed(1)} ms transport and deposition`
      );
      schedule();
    } catch (error) {
      status(error.message, true);
    } finally {
      $('simulate').disabled = false;
    }
  };
  $('exposure').oninput = () => {
    camera.exposure = 10 ** Number($('exposure').value);
    schedule();
  };
  $('quality').onchange = () => schedule();
  for (const [id, key] of [
      ['cutaway', 'cutaway'],
      ['uv_false_color', 'uvFalseColor']
    ])
    if ($(id)) $(id).onchange = () => {
      camera[key] = $(id).checked;
      schedule();
    };
  $('reset').onclick = () => {
    if (!camera.resources || camera.busy) return;
    const m = camera.scenes[camera.scene].scene.camera;
    camera.eye = m.default_eye.slice();
    camera.target = m.default_target.slice();
    schedule();
  };
  let drag = null;
  camera.canvas.onpointerdown = e => {
    if (camera.busy || !camera.resources || (e.button !== 0 && e.button !== 1)) return;
    e.preventDefault();
    drag = {
      id: e.pointerId,
      pan: e.button === 1 || e.shiftKey,
      x: e.clientX,
      y: e.clientY,
      eye: camera.eye.slice(),
      target: camera.target.slice()
    };
    camera.canvas.setPointerCapture(e.pointerId);
  };
  camera.canvas.onpointermove = e => {
    if (!drag || drag.id !== e.pointerId) return;
    const v = drag.eye.map((x, i) => x - drag.target[i]),
      radius = Math.hypot(...v);
    if (drag.pan) {
      // Use the same world-up fallback as the camera shader near the poles.
      const forward = v.map(x => -x / radius);
      const reference = Math.abs(forward[2]) > .99 ? [0,1,0] : [0,0,1];
      const cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
      const horizontal = cross(forward,reference), length = Math.hypot(...horizontal);
      const right = horizontal.map(x => x/length), up = cross(right,forward);
      const scale = 2*radius*Math.tan(camera.fov*Math.PI/360)/Math.max(1,camera.canvas.clientHeight);
      const movement = right.map((x,i) => scale*(-(e.clientX-drag.x)*x+(e.clientY-drag.y)*up[i]));
      // Clip the translation to the room, moving eye and target by exactly the
      // same amount so reaching a wall cannot rotate or stretch the view.
      let fraction = 1;
      for (let i=0;i<3;i++) if(movement[i]!==0)
        fraction = Math.min(fraction, Math.max(0, ((movement[i]>0 ? HALF[i]-2 : -HALF[i]+2)-drag.eye[i])/movement[i]));
      camera.eye = drag.eye.map((x,i) => x+fraction*movement[i]);
      camera.target = drag.target.map((x,i) => x+fraction*movement[i]);
      schedule(true);
      return;
    }
    const az = Math.atan2(v[1], v[0]) - (e.clientX - drag.x) * .005,
      el = Math.max(-1.3, Math.min(1.3, Math.asin(v[2] / radius) + (e.clientY - drag.y) *
        .004));
    camera.eye = [radius * Math.cos(el) * Math.cos(az), radius * Math.cos(el) * Math.sin(az),
      radius * Math.sin(el)
    ].map((x, i) => Math.max(-HALF[i] + 2, Math.min(HALF[i] - 2, x + drag.target[i])));
    schedule(true);
  };
  camera.canvas.onpointerup = e => {
    if (!drag || drag.id !== e.pointerId) return;
    drag = null;
    schedule();
  };
  camera.canvas.onpointercancel = camera.canvas.onlostpointercapture = () => { drag = null; };
  camera.canvas.onauxclick = e => { if(e.button===1)e.preventDefault(); };
  camera.canvas.onwheel = e => {
    e.preventDefault();
    if (!camera.resources || camera.busy) return;
    const scale = Math.exp(e.deltaY * .001);
    camera.eye = camera.eye.map((x, i) => Math.max(-HALF[i] + 2, Math.min(HALF[i] - 2, camera
      .target[i] + (x - camera.target[i]) * scale)));
    schedule(true);
  };
  status('Ready. Simulate photons to render the scene.');
  if (!new URLSearchParams(location.search).has('manual')) $('simulate').click();
  return info;
}).catch(error => {
  status(error.message, true);
  throw error;
});
