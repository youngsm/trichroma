import {
  OpticalLab
} from './physics.js';
const $ = id => document.getElementById(id);
const MAP_SIZES = {
  wall: 6 * 128 * 128 * 64 * 4,
  volume: 64 * 40 * 24 * 64 * 6 * 4,
  wls: 12 * 64 * 64 * 64 * 4,
  fill_wall: 6 * 32 * 32 * 64 * 4
};
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
    await super.initialize({
      catalogName: 'camera-catalog.json'
    });
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
      if (pmt) {
        if ([pmt.triangle_role, pmt.triangle_chart, pmt.triangle_area_mm2].some(a =>
            !Array.isArray(a) || a.length !== count)) throw Error('Invalid PMT triangle atlas');
        checkedInteger(pmt.wls_charts, 1, MAP_SIZES.wls / (2 * 64 * 4), 'WLS charts');
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
      '// CAMERA_SOURCE_HOOK': `let is_fill=random_uniform(id,0x10000008u)>=f(tables[41u]+77u);var packet_scale=1./f(tables[41u]+77u);if(is_fill){position=v3(tables[41u]+64u)+vec3<f32>((2.*random_uniform(id,0x10000004u)-1.)*f(tables[41u]+67u),(2.*random_uniform(id,0x10000005u)-1.)*f(tables[41u]+68u),-f(tables[41u]+79u));initial_position=position;direction=hemisphere(vec3<f32>(0.,0.,-1.),random_uniform(id,0x10000006u),random_uniform(id,0x10000007u),true);polarization=random_polarization(direction,random_uniform(id,0x10000002u));wavelength=390.+320.*random_uniform(id,0x10000003u);packet_scale=f(tables[41u]+78u)/(1.-f(tables[41u]+77u))*wavelength/450.;atomicAdd(&statistics[21u],1u);}else{atomicAdd(&statistics[20u],1u);}`,
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
    this.depositionPipeline = await this.device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'simulate'
      }
    });
    this.cameraPipeline = await this.device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'render_camera'
      }
    });
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
  async simulate({
    scene = 'prism',
    photons = 2500000,
    seed = 901,
    polarization = 'random',
    debug = false,
    debugMaps = false
  } = {}) {
    if (this.busy || this.rendering) throw Error('The browser GPU is already working');
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
      const resources = {
        tables: this.buffer(this.scenes[scene].packed, store),
        config: this.buffer(cfg, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
        statistics: this.buffer((15788 + paths) * 4, store),
        paths: this.buffer(Math.max(4, paths * 8193 * 32), store),
        final: this.buffer(debug ? photons * 80 : 2048, store),
        wall: this.buffer(MAP_SIZES.wall, store),
        volume: this.buffer(MAP_SIZES.volume, store),
        wls: this.buffer(MAP_SIZES.wls, store),
        fill_wall: this.buffer(MAP_SIZES.fill_wall, store)
      };
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
      d.pushErrorScope('validation');
      const encoder = d.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.depositionPipeline);
      pass.setBindGroup(0, group);
      const groups = Math.ceil(photons / 128);
      pass.dispatchWorkgroups(Math.min(65535, groups), Math.ceil(groups / 65535));
      pass.end();
      const begin = performance.now();
      d.queue.submit([encoder.finish()]);
      await d.queue.onSubmittedWorkDone();
      const compute_ms = performance.now() - begin;
      const error = await d.popErrorScope();
      if (error) throw Error(error.message);
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
      this.scene = scene;
      this.photons = photons;
      this.seed = seed;
      this.polarization = polarization;
      this.frame = 0;
      const manifest = this.scenes[scene].scene;
      this.eye = manifest.camera.default_eye.slice();
      this.target = manifest.camera.default_target.slice();
      this.fov = manifest.camera.field_of_view_degrees;
      this.cutaway = false;
      this.uvFalseColor = false;
      for (const id of ['cutaway', 'uv_false_color'])
        if ($(id)) {
          $(id).checked = false;
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
          wall_ms: performance.now() - start
        },
        map_bytes: MAP_SIZES,
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
    for (const key of Object.keys(MAP_SIZES)) {
      const values = new Float32Array(await this.download(this.resources[key]));
      const pairs = [];
      for (let index = 0; index < values.length; index++)
        if (values[index] !== 0) pairs.push([index, values[index]]);
      output[key] = pairs;
    }
    return output;
  }
  async render({
    width = 640,
    height = 400,
    reset = false,
    eye = null,
    target = null,
    exposure = null,
    cutaway = null,
    uvFalseColor = null
  } = {}) {
    if (this.busy || this.rendering) throw Error('The browser GPU is already working');
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
    try {
      const d = this.device;
      if (!this.image || width !== this.width || height !== this.height) {
        this.image?.destroy();
        this.accumulation?.destroy();
        this.width = width;
        this.height = height;
        this.canvas.width = width;
        this.canvas.height = height;
        this.image = d.createTexture({
          size: [width, height],
          format: 'rgba16float',
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });
        this.accumulation = this.buffer(width * height * 16, GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        reset = true;
      }
      if (reset) this.frame = 0;
      const cfg = new Uint32Array(16),
        f = new Float32Array(cfg.buffer);
      f.set([...this.eye, this.exposure, ...this.target, Math.tan(this.fov * Math.PI / 360)]);
      cfg.set([width, height, this.frame, 0], 8);
      f.set([this.cutaway ? 1 : 0, this.uvFalseColor ? 1 : 0, 0, 0], 12);
      const uniform = this.buffer(cfg, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
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
      d.pushErrorScope('validation');
      const encoder = d.createCommandEncoder();
      const compute = encoder.beginComputePass();
      compute.setPipeline(this.cameraPipeline);
      compute.setBindGroup(0, bind);
      compute.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      compute.end();
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
      const queue_ms = performance.now() - start;
      const error = await d.popErrorScope();
      uniform.destroy();
      if (error) throw Error(error.message);
      this.frame++;
      return {
        width,
        height,
        frame: this.frame,
        queue_ms,
        photons: this.photons,
        eye: this.eye.slice(),
        target: this.target.slice(),
        exposure: this.exposure,
        cutaway: this.cutaway,
        uv_false_color: this.uvFalseColor
      };
    } finally {
      this.rendering = false;
    }
  }
  async linearImage() {
    return Array.from(new Float32Array(await this.download(this.accumulation)));
  }
}
const camera = new PhotonCamera();
window.photonCamera = camera;
let request = 0,
  settle = null;

function updateInfo(frame) {
  $('samples').textContent =
    `${camera.photons.toLocaleString()} photons · ${frame.frame} spectral camera samples · ${frame.queue_ms.toFixed(0)} ms camera`;
}
async function refine(version, preview = false) {
  if (version !== request || camera.busy || camera.rendering) return;
  try {
    const width = preview ? 160 : Number($('quality').value),
      height = Math.round(width * .625);
    const frame = await camera.render({
      width,
      height,
      reset: true
    });
    updateInfo(frame);
    if (!preview) {
      for (let i = 1; i < 16 && version === request; i++) {
        updateInfo(await camera.render({
          width,
          height
        }));
      }
    }
  } catch (error) {
    status(error.message, true);
  }
}

function schedule(preview = false) {
  const version = ++request;
  clearTimeout(settle);
  refine(version, preview);
  if (preview) settle = setTimeout(() => refine(version, false), 250);
}
window.photonCameraReady = camera.initialize().then(info => {
  $('adapter').textContent =
    `${info.vendor} ${info.architecture}${info.isFallbackAdapter?' · software adapter':''}`;
  $('simulate').disabled = false;
  $('simulate').onclick = async () => {
    request++;
    $('simulate').disabled = true;
    while (camera.rendering) await new Promise(resolve => requestAnimationFrame(resolve));
    status('Simulating photons and building spectral light maps…');
    try {
      const result = await camera.simulate({
        scene: $('scene').value,
        photons: Number($('photons').value),
        seed: Number($('seed').value),
        polarization: $('polarization').value
      });
      status(
        `${result.photons.toLocaleString()} optical photons (${result.counts.beam.toLocaleString()} beam + ${result.counts.fill.toLocaleString()} ceiling) · ${result.counts.scattered.toLocaleString()} scattered · ${result.counts.reemitted.toLocaleString()} reemitted${result.scene === "pmt" ? " · " + result.counts.photocathode.toLocaleString() + " photocathode detections" : ""} · ${result.metrics.compute_ms.toFixed(1)} ms transport and deposition`
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
    if (camera.busy || !camera.resources) return;
    drag = {
      x: e.clientX,
      y: e.clientY,
      eye: camera.eye.slice()
    };
    camera.canvas.setPointerCapture(e.pointerId);
  };
  camera.canvas.onpointermove = e => {
    if (!drag) return;
    const v = drag.eye.map((x, i) => x - camera.target[i]),
      radius = Math.hypot(...v),
      az = Math.atan2(v[1], v[0]) - (e.clientX - drag.x) * .005,
      el = Math.max(-1.3, Math.min(1.3, Math.asin(v[2] / radius) + (e.clientY - drag.y) *
        .004));
    camera.eye = [radius * Math.cos(el) * Math.cos(az), radius * Math.cos(el) * Math.sin(az),
      radius * Math.sin(el)
    ].map((x, i) => Math.max(-HALF[i] + 2, Math.min(HALF[i] - 2, x + camera.target[i])));
    schedule(true);
  };
  camera.canvas.onpointerup = () => {
    drag = null;
    schedule();
  };
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
