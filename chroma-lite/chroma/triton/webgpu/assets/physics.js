// Small-scene optical Monte Carlo. Full-count observables stay on the GPU;
// only histograms and a bounded set of real photon paths are downloaded.
const STAT_WORDS = 15788;
const SCENES = ['prism', 'fluorescence', 'rayleigh', 'pmt'];
const $ = id => document.getElementById(id);
const colorKnots = [
  [280, 164, 64, 255],
  [380, 164, 64, 255],
  [440, 61, 89, 255],
  [490, 31, 230, 255],
  [510, 51, 255, 115],
  [580, 255, 230, 38],
  [620, 255, 89, 20],
  [700, 255, 31, 33],
  [740, 255, 31, 33]
];
export function spectralColor(wavelength, alpha = 1) {
  const w = Math.max(280, Math.min(740, wavelength));
  const hi = colorKnots.findIndex(row => row[0] >= w),
    a = colorKnots[Math.max(0, hi - 1)],
    b = colorKnots[hi];
  const t = b[0] === a[0] ? 0 : (w - a[0]) / (b[0] - a[0]);
  return `rgba(${a.slice(1).map((v,i)=>Math.round(v+t*(b[i+1]-v))).join(',')},${alpha})`;
}

function integer(value, low, high, label) {
  if (!Number.isSafeInteger(value) || value < low || value > high) throw Error(
    `${label} must be an integer in ${low}…${high}`);
  return value;
}

function flatten(value) {
  return Array.isArray(value) ? value.flat(Infinity) : [value];
}
export function packPhysicsScene(scene) {
  if (scene.format !== 'trichroma-webgpu-spectral-v1' || scene.version !== 1 || !SCENES.includes(
      scene.name)) throw Error('Unsupported optical scene format');
  const triangles = integer(scene.triangles.length, 1, 4096, 'Triangle count'),
    materials = integer(scene.materials.names.length, 1, 256, 'Material count'),
    surfaces = integer(scene.surfaces.names.length, 1, 256, 'Surface count');
  integer(scene.wavelengths.count, 2, 4096, 'Wavelength grid count');
  if (!(scene.wavelengths.step > 0) || !Number.isFinite(scene.wavelengths.start)) throw Error(
    'Invalid wavelength grid');
  for (const [name, minimum, maximum] of [
      ['material1', 0, materials - 1],
      ['material2', 0, materials - 1],
      ['surface', -1, surfaces - 1]
    ]) {
    if (scene[name].length !== triangles || scene[name].some(v => !Number.isInteger(v) || v <
        minimum || v > maximum)) throw Error('Invalid ' + name + ' triangle table');
  }
  if (flatten(scene.triangles).length !== triangles * 9 || flatten(scene.normals).length !==
    triangles * 3 || scene.channels.length !== triangles) throw Error(
    'Inconsistent geometry table dimensions');
  for (const key of ['refractive_index', 'absorption_length', 'scattering_length',
      'group_velocity'
    ])
    if (flatten(scene.materials[key]).length !== materials * scene.wavelengths.count) throw Error(
      'Inconsistent material table: ' + key);
  for (const key of ['detect', 'absorb', 'reflect_diffuse', 'reflect_specular', 'reemit',
      'reemission_cdf'
    ])
    if (flatten(scene.surfaces[key]).length !== surfaces * scene.wavelengths.count) throw Error(
      'Inconsistent surface table: ' + key);
  if (scene.source.direction.join(',') !== '1,0,0' || scene.source.time0 !== 0) throw Error(
    'This prototype requires its exported +X source');
  if (scene.surfaces.model.some((m, i) => scene.surfaces.present[i] && ![0, 2].includes(m)))
    throw Error('Unsupported surface model');
  const header = new Uint32Array(64),
    hf = new Float32Array(header.buffer),
    chunks = [];
  let length = 64;
  const append = (slot, values, kind = 'float', allowInfinity = false) => {
    const flat = flatten(values).map(v => v === null && allowInfinity ? Infinity : v);
    if (flat.some(v => typeof v !== 'number' || Number.isNaN(v) || (!allowInfinity && !Number
        .isFinite(v)))) throw Error(`Invalid table at slot ${slot}`);
    const array = kind === 'int' ? new Int32Array(flat) : new Float32Array(flat);
    header[slot] = length;
    chunks.push(new Uint32Array(array.buffer));
    length += array.length;
  };
  header[0] = SCENES.indexOf(scene.name);
  header[1] = scene.triangles.length;
  header[2] = scene.materials.names.length;
  header[3] = scene.surfaces.names.length;
  header[4] = scene.wavelengths.count;
  hf[5] = scene.wavelengths.start;
  hf[6] = scene.wavelengths.step;
  append(8, scene.triangles);
  append(9, scene.normals);
  ['material1', 'material2', 'surface', 'channels'].forEach((name, i) => append(10 + i, scene[name],
    'int'));
  ['refractive_index', 'absorption_length', 'scattering_length', 'group_velocity'].forEach((name,
    i) => append(14 + i, scene.materials[name], 'float', i === 1 || i === 2));
  append(18, scene.surfaces.present.map(Number), 'int');
  append(19, scene.surfaces.model, 'int');
  ['detect', 'absorb', 'reflect_diffuse', 'reflect_specular', 'reemit', 'reemission_cdf',
    'time_offsets', 'time_x', 'time_cdf', 'time_pdf', 'reemit_to_material1'
  ].forEach((name, i) => append(20 + i, scene.surfaces[name], name === 'time_offsets' ? 'int' :
    'float'));
  hf.set(scene.source.center, 31);
  hf.set(scene.source.band, 34);
  hf[36] = scene.source.beam_width;
  const packed = new Uint32Array(length);
  packed.set(header);
  let offset = 64;
  for (const chunk of chunks) {
    packed.set(chunk, offset);
    offset += chunk.length;
  }
  return packed;
}
export class OpticalLab {
  async initialize({
    catalogName = 'physics-catalog.json'
  } = {}) {
    if (!navigator.gpu) throw Error(
      'WebGPU needs a compatible browser and a secure context (HTTPS or localhost).');
    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
      forceFallbackAdapter: new URLSearchParams(location.search).has('fallback')
    });
    if (!this.adapter) throw Error(
      'No WebGPU adapter. Try current Chrome with hardware acceleration enabled.');
    this.device = await this.adapter.requestDevice();
    this.device.lost.then(info => {
      this.lost = info.message;
      showError(`GPU device lost: ${info.message}`);
    });
    this.errors = [];
    this.device.addEventListener('uncapturederror', event => {
      this.errors.push(event.error.message);
      showError(event.error.message);
    });
    const code = await fetch('physics.wgsl').then(r => {
      if (!r.ok) throw Error('Missing physics.wgsl');
      return r.text();
    });
    const shader = this.device.createShaderModule({
      code
    });
    const messages = (await shader.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (messages.length) throw Error(messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`)
      .join('\n'));
    // Explicit layout allows the RNG probe to share all binding slots.
    this.layout = this.device.createBindGroupLayout({
      entries: [{
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'read-only-storage'
          }
        }, {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'uniform'
          }
        },
        ...[2, 3, 4].map(binding => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage'
          }
        }))
      ]
    });
    const layout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.layout]
    });
    [this.pipeline, this.rngPipeline] = await Promise.all(['simulate', 'rng_probe'].map(
      entryPoint => this.device.createComputePipelineAsync({
        layout,
        compute: {
          module: shader,
          entryPoint
        }
      })));
    const catalog = await fetch(catalogName).then(r => r.json());
    this.scenes = {};
    for (const entry of catalog.scenes) {
      const response = await fetch(entry.manifest);
      if (!response.ok) throw Error('Missing optical table manifest ' + entry.manifest);
      const bytes = await response.arrayBuffer();
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), x =>
        x.toString(16).padStart(2, '0')).join('');
      if (digest !== entry.sha256) throw Error('Optical manifest checksum mismatch: ' + entry
        .manifest);
      const scene = JSON.parse(new TextDecoder().decode(bytes));
      this.scenes[scene.name] = {
        scene,
        packed: packPhysicsScene(scene)
      };
    }
    const info = this.adapter.info;
    return this.info = {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
      isFallbackAdapter: info.isFallbackAdapter ?? false,
      limits: {
        maxStorageBufferBindingSize: this.device.limits.maxStorageBufferBindingSize
      }
    };
  }
  buffer(data, usage) {
    const size = typeof data === 'number' ? data : data.byteLength;
    const b = this.device.createBuffer({
      size: Math.max(4, size),
      usage
    });
    if (typeof data !== 'number') this.device.queue.writeBuffer(b, 0, data);
    return b;
  }
  async checkRng(goldens) {
    if (this.busy) throw Error('An optical run is already active');
    const vectors = new Uint32Array(goldens.flatMap(g => [g.id_low, g.id_high, g.seed_low, g
      .seed_high, g.stream
    ]));
    const cfg = new Uint32Array(12);
    cfg[0] = goldens.length;
    const result = await this.dispatch(vectors, cfg, 4, 4, goldens.length * 4, this.rngPipeline,
      Math.ceil(goldens.length / 64), 1);
    return Array.from(new Uint32Array(result.debug));
  }
  async dispatch(tables, cfg, statBytes, pathBytes, debugBytes, pipeline, x, y) {
    const d = this.device,
      storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const buffers = [this.buffer(tables, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST), this
      .buffer(cfg, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST), this.buffer(statBytes,
        storage), this.buffer(pathBytes, storage), this.buffer(debugBytes, storage)
    ];
    const sizes = [statBytes, pathBytes, debugBytes],
      offsets = [0, statBytes, statBytes + pathBytes];
    const readback = this.buffer(Math.max(4, statBytes + pathBytes + debugBytes), GPUBufferUsage
      .MAP_READ | GPUBufferUsage.COPY_DST);
    try {
      d.pushErrorScope('validation');
      const group = d.createBindGroup({
        layout: this.layout,
        entries: buffers.map((buffer, binding) => ({
          binding,
          resource: {
            buffer
          }
        }))
      });
      let encoder = d.createCommandEncoder();
      let pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(x, y);
      pass.end();
      const begin = performance.now();
      d.queue.submit([encoder.finish()]);
      await d.queue.onSubmittedWorkDone();
      const queueMs = performance.now() - begin;
      const validation = await d.popErrorScope();
      if (validation) throw Error(validation.message);
      encoder = d.createCommandEncoder();
      sizes.forEach((size, i) => {
        if (size) encoder.copyBufferToBuffer(buffers[i + 2], 0, readback, offsets[i], size);
      });
      const readStart = performance.now();
      d.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = readback.getMappedRange();
      const output = {
        statistics: mapped.slice(0, statBytes),
        paths: mapped.slice(statBytes, statBytes + pathBytes),
        debug: mapped.slice(statBytes + pathBytes, statBytes + pathBytes + debugBytes),
        queueMs,
        readbackMs: performance.now() - readStart
      };
      readback.unmap();
      return output;
    } finally {
      buffers.forEach(b => b.destroy());
      readback.destroy();
    }
  }
  async run({
    scene = 'prism',
    photons = 2500000,
    seed = 901,
    polarization = 'random',
    paths = 512,
    maxSteps = 256,
    debug = false
  } = {}) {
    if (this.busy) throw Error('An optical run is already active');
    if (this.lost) throw Error(this.lost);
    if (!this.scenes[scene]) throw Error('Choose one of the three exported optical scenes');
    integer(photons, 1, 30000000, 'Photon count');
    integer(seed, 0, Number.MAX_SAFE_INTEGER, 'Seed');
    integer(paths, 0, Math.min(2048, photons), 'Path count');
    integer(maxSteps, 1, 256, 'Maximum steps');
    if (!['random', 'y', 'z'].includes(polarization)) throw Error(
      'Polarization must be random, y or z');
    if (debug && photons > 65536) throw Error(
      'Terminal debug readback is limited to 65,536 photons; use histogram mode for larger runs.'
    );
    const begin = performance.now();
    this.busy = true;
    setBusy(true);
    try {
      const cfg = new Uint32Array(12),
        f = new Float32Array(cfg.buffer),
        timeMax = scene === 'fluorescence' ? 205 : scene === 'rayleigh' ? 25 : 5;
      cfg.set([photons, seed >>> 0, Math.floor(seed / 4294967296), paths, maxSteps, SCENES
        .indexOf(scene), debug ? photons : 0, ['random', 'y', 'z'].indexOf(polarization)
      ]);
      f[8] = timeMax;
      const pathBytes = Math.max(4, paths * (maxSteps + 1) * 32),
        debugBytes = debug ? photons * 80 : 4;
      if (Math.max(pathBytes, debugBytes) > this.device.limits.maxStorageBufferBindingSize)
        throw Error('Requested path/debug buffer exceeds this browser GPU limit');
      const groups = Math.ceil(photons / 128);
      const data = await this.dispatch(this.scenes[scene].packed, cfg, (STAT_WORDS + paths) * 4,
        pathBytes, debugBytes, this.pipeline, Math.min(65535, groups), Math.ceil(groups / 65535)
      );
      const stats = new Uint32Array(data.statistics),
        pf = new Float32Array(data.paths),
        pu = new Uint32Array(data.paths);
      if (stats[0] !== photons || stats[5] || stats[6]) throw Error(
        `Incomplete optical event: ${stats[0]}/${photons} processed, ${stats[5]} step limits, ${stats[6]} non-finite states. Increase the supported step limit or reduce this scene; no partial result is displayed.`
      );
      const sample = [];
      for (let id = 0; id < paths; id++) {
        const vertices = [];
        for (let j = 0; j < stats[STAT_WORDS + id]; j++) {
          const b = (id * (maxSteps + 1) + j) * 8;
          vertices.push({
            position: Array.from(pf.slice(b, b + 3)),
            time: pf[b + 3],
            wavelength: pf[b + 4],
            flags: pu[b + 5],
            flight: pf[b + 6],
            arrival_time: pf[b + 7]
          });
        }
        sample.push({
          photon_id: id,
          vertices
        });
      }
      const result = {
        scene,
        photons,
        seed,
        polarization,
        pathCount: paths,
        maxSteps,
        adapter: this.info,
        metrics: {
          queue_ms: data.queueMs,
          readback_ms: data.readbackMs,
          wall_ms: 0,
          photons_per_second: photons / (data.queueMs / 1000),
          timing_scope: 'compute submission to queue completion; first dispatch may include driver warm-up; readback and JavaScript decoding reported separately'
        },
        counts: {
          detected: stats[1],
          surface_absorbed: stats[2],
          bulk_absorbed: stats[3],
          escaped: stats[4],
          step_limit: stats[5],
          nonfinite: stats[6],
          reemitted: stats[7],
          scattered: stats[8],
          reflected: stats[9],
          transmitted: stats[10],
          max_steps: stats[12],
          forward_monitor: stats[13],
          time_overflow: stats[14]
        },
        histograms: {
          source: Array.from(stats.slice(32, 124)),
          detected: Array.from(stats.slice(124, 216)),
          arrival: Array.from(stats.slice(216, 344)),
          delay: Array.from(stats.slice(344, 424)),
          scattered: Array.from(stats.slice(424, 456)),
          scatter_source: Array.from(stats.slice(456, 488)),
          dispersion: Array.from(stats.slice(488, STAT_WORDS)),
          time_max: timeMax
        },
        maxTime: new Float32Array(data.statistics)[16],
        paths: sample
      };
      if (debug) {
        const u = new Uint32Array(data.debug),
          f = new Float32Array(data.debug),
          i = new Int32Array(data.debug);
        const state = {
          pos: [],
          direction: [],
          polarization: [],
          wavelengths: [],
          times: [],
          flags: [],
          channels: [],
          last_hit_triangles: [],
          steps: [],
          photon_ids: [],
          source_wavelengths: [],
          fluorescence_delay: [],
          flight_distance: [],
          source_y: [],
          source_z: []
        };
        for (let id = 0; id < photons; id++) {
          const b = id * 20;
          state.pos.push(Array.from(f.slice(b, b + 3)));
          state.direction.push(Array.from(f.slice(b + 4, b + 7)));
          state.polarization.push(Array.from(f.slice(b + 8, b + 11)));
          state.wavelengths.push(f[b + 7]);
          state.times.push(f[b + 3]);
          state.flags.push(u[b + 12]);
          state.channels.push(i[b + 13]);
          state.last_hit_triangles.push(i[b + 14]);
          state.steps.push(u[b + 15]);
          state.photon_ids.push(id);
          state.source_wavelengths.push(f[b + 11]);
          state.fluorescence_delay.push(f[b + 16]);
          state.flight_distance.push(f[b + 17]);
          state.source_y.push(f[b + 18]);
          state.source_z.push(f[b + 19]);
        }
        result.states = state;
      }
      result.metrics.wall_ms = performance.now() - begin;
      this.result = result;
      this.timeGate = Infinity;
      renderResult(this, result);
      return result;
    } finally {
      this.busy = false;
      setBusy(false);
    }
  }
  setTimeGate(time) {
    this.timeGate = Number(time);
    if (this.result) drawPaths(this, this.result);
  }
}

function showError(error) {
  const s = $('status');
  if (s) {
    s.textContent = String(error);
    s.classList.add('error');
  }
}

function setBusy(busy) {
  if ($('run')) {
    $('run').disabled = busy;
    $('run').textContent = busy ? 'Tracing optical photons…' : 'Run optical event';
  }
  if (busy) {
    $('status').classList.remove('error');
    $('status').textContent = 'Transporting photons on the browser GPU…';
  }
}

function canvas(id) {
  const c = $(id),
    rect = c.getBoundingClientRect(),
    scale = Math.min(devicePixelRatio || 1, 2),
    w = Math.max(1, rect.width),
    h = Math.max(1, rect.height);
  c.width = Math.round(w * scale);
  c.height = Math.round(h * scale);
  const ctx = c.getContext('2d');
  ctx.scale(scale, scale);
  ctx.clearRect(0, 0, w, h);
  return {
    ctx,
    w,
    h
  };
}

function chartAxes(ctx, w, h, xlabel, ylabel) {
  ctx.strokeStyle = '#293346';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(48, 16);
  ctx.lineTo(48, h - 34);
  ctx.lineTo(w - 12, h - 34);
  ctx.stroke();
  ctx.fillStyle = '#8593aa';
  ctx.font = '11px system-ui';
  ctx.fillText(xlabel, 50, h - 8);
  ctx.save();
  ctx.translate(13, h - 38);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylabel, 0, 0);
  ctx.restore();
}

function histogramChart(id, series, low, high, xlabel, {
  secondary = null,
  ylabel = 'photons',
  log = false
} = {}) {
  const {
    ctx,
    w,
    h
  } = canvas(id), left = 48, bottom = h - 34, plotw = w - 62, ploth = h - 55;
  chartAxes(ctx, w, h, xlabel, ylabel);
  const transform = v => log ? Math.log1p(v) : v,
    max = Math.max(1, ...series.map(transform), ...(secondary || []).map(transform));
  for (let b = 0; b < series.length; b++) {
    const wavelength = low + (b + .5) * (high - low) / series.length;
    ctx.fillStyle = id === 'spectrum' ? spectralColor(wavelength, .82) : '#7bd6e7';
    ctx.fillRect(left + b * plotw / series.length, bottom - transform(series[b]) / max * ploth, Math
      .max(1, plotw / series.length - .4), transform(series[b]) / max * ploth);
  }
  if (secondary) {
    ctx.strokeStyle = '#e1e9ff88';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    secondary.forEach((v, b) => {
      const x = left + (b + .5) * plotw / secondary.length,
        y = bottom - transform(v) / max * ploth;
      b ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  ctx.fillStyle = '#aebacf';
  ctx.font = '10px system-ui';
  ctx.fillText(String(low), left, bottom + 13);
  ctx.fillText(String(high), w - 35, bottom + 13);
  ctx.fillText(log ? 'log counts' : Math.max(...series).toLocaleString(), left + 4, 14);
}

function drawPaths(lab, result) {
  const {
    ctx,
    w,
    h
  } = canvas('paths'), scene = lab.scenes[result.scene].scene, left = 26, right = w - 26, top = 25,
    bottom = h - 28;
  const project = p => [left + (p[0] + 305) / 610 * (right - left), bottom - (p[1] + 205) / 410 * (
    bottom - top)];
  ctx.fillStyle = '#080d18';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#263144';
  ctx.lineWidth = .7;
  for (let x = -300; x <= 300; x += 100) {
    let a = project([x, -200]),
      b = project([x, 200]);
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
  }
  for (let y = -200; y <= 200; y += 100) {
    let a = project([-300, y]),
      b = project([300, y]);
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
  }
  const a = project([-300, -200]),
    b = project([300, 200]);
  ctx.strokeStyle = '#5bb8d966';
  ctx.lineWidth = 2;
  ctx.strokeRect(a[0], b[1], b[0] - a[0], a[1] - b[1]);
  if (scene.display.outline) {
    ctx.beginPath();
    scene.display.outline.forEach((p, i) => {
      const q = project(p);
      i ? ctx.lineTo(...q) : ctx.moveTo(...q);
    });
    ctx.closePath();
    ctx.fillStyle = result.scene === 'prism' ? '#aad1ff0b' : '#b66bff1c';
    ctx.fill();
    ctx.strokeStyle = result.scene === 'prism' ? '#abcfff77' : '#d59bffc9';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  const gate = lab.timeGate,
    alpha = Math.min(.7, Math.max(.075, 10 / Math.sqrt(Math.max(1, result.pathCount))));
  ctx.globalCompositeOperation = 'lighter';
  for (const path of result.paths) {
    for (let j = 1; j < path.vertices.length; j++) {
      const from = path.vertices[j - 1],
        to = path.vertices[j];
      if (to.arrival_time > gate) continue;
      const p = project(from.position),
        q = project(to.position);
      ctx.strokeStyle = spectralColor(from.wavelength, alpha);
      ctx.lineWidth = .8;
      ctx.beginPath();
      ctx.moveTo(...p);
      ctx.lineTo(...q);
      ctx.stroke();
      if (to.time <= gate && (to.flags & 128) && !(from.flags & 128)) {
        ctx.fillStyle = spectralColor(to.wavelength, .5);
        ctx.beginPath();
        ctx.arc(q[0], q[1], 2.3, 0, Math.PI * 2);
        ctx.fill();
      }
      if (to.flags & 4) {
        ctx.fillStyle = spectralColor(to.wavelength, .7);
        ctx.beginPath();
        ctx.arc(q[0], q[1], 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = '#7e90aa';
  ctx.font = '10px system-ui';
  ctx.fillText('−300 mm', left, h - 9);
  ctx.fillText('x → +300 mm', w - 100, h - 9);
  ctx.fillText('y / mm · x–y projection', left, 14);
  $('pathCaption').textContent =
    `${result.pathCount.toLocaleString()} actual trajectories from the ${result.photons.toLocaleString()}-photon event · ${Number.isFinite(gate)?`completed flights by ${gate.toFixed(2)} ns`:'all recorded flights'} · UV shown in violet`;
}

function physicsChart(result) {
  if (result.scene === 'fluorescence') {
    $('physicsTitle').textContent = 'Fluorescence delay';
    histogramChart('physicsChart', result.histograms.delay, 0, 200, 'added delay / ns');
    return;
  }
  const {
    ctx,
    w,
    h
  } = canvas('physicsChart');
  if (result.scene === 'rayleigh') {
    $('physicsTitle').textContent = 'Scattering vs. wavelength';
    chartAxes(ctx, w, h, 'source wavelength / nm', 'fraction scattered');
    const left = 48,
      bottom = h - 34,
      ph = h - 55,
      pw = w - 62;
    for (let i = 0; i < 32; i++) {
      const p = result.histograms.scattered[i] / Math.max(1, result.histograms.scatter_source[i]);
      ctx.fillStyle = spectralColor(395 + i * 10, .85);
      ctx.fillRect(left + i * pw / 32, bottom - p * ph, pw / 32 - 1, p * ph);
    }
    ctx.fillStyle = '#aebacf';
    ctx.font = '10px system-ui';
    ctx.fillText('390', left, bottom + 13);
    ctx.fillText('710', w - 35, bottom + 13);
    ctx.fillText('1.0', left + 3, 14);
    return;
  }
  $('physicsTitle').textContent = 'Prism dispersion · forward monitor';
  chartAxes(ctx, w, h, 'wavelength / nm', 'outgoing angle / deg');
  const values = result.histograms.dispersion,
    max = Math.max(1, ...values),
    left = 48,
    bottom = h - 34,
    pw = w - 62,
    ph = h - 55;
  // Restrict the angle view to occupied bins while retaining every histogram entry.
  const occupied = [];
  for (let a = 0; a < 180; a++)
    if (values.slice(a * 85, (a + 1) * 85).some(Boolean)) occupied.push(a);
  const low = Math.max(0, (occupied[0] ?? 65) - 2),
    high = Math.min(180, (occupied.at(-1) ?? 115) + 3);
  for (let a = low; a < high; a++)
    for (let b = 0; b < 85; b++) {
      const value = values[a * 85 + b];
      if (!value) continue;
      ctx.fillStyle = spectralColor(382 + b * 4, .25 + .75 * Math.log1p(value) / Math.log1p(max));
      ctx.fillRect(left + b * pw / 85, bottom - (a - low + 1) * ph / (high - low), pw / 85 + 1, ph /
        (high - low) + 1);
    }
  ctx.fillStyle = '#aebacf';
  ctx.font = '10px system-ui';
  ctx.fillText('380', left, bottom + 13);
  ctx.fillText('720', w - 35, bottom + 13);
  ctx.fillText(`${high-90}°`, left + 3, 14);
  ctx.fillText(`${low-90}°`, left + 3, bottom - 4);
}

function renderResult(lab, result) {
  if (!$('paths')) return;
  $('scene').value = result.scene;
  $('photons').value = result.photons;
  $('seed').value = result.seed;
  $('polarization').value = result.polarization;
  const manifest = lab.scenes[result.scene].scene;
  $('sceneTitle').textContent = manifest.title;
  $('explanation').textContent = manifest.explanation;
  $('eventCount').textContent = result.photons.toLocaleString();
  $('detected').textContent = (100 * result.counts.detected / result.photons).toFixed(2) + '%';
  $('compute').textContent = result.metrics.queue_ms.toFixed(1) + ' ms';
  $('throughput').textContent = (result.metrics.photons_per_second / 1e6).toFixed(2) + ' M/s';
  $('gate').max = Math.max(.01, result.maxTime);
  $('gate').value = Number.isFinite(lab.timeGate) ? lab.timeGate : $('gate').max;
  $('gateValue').textContent = Number.isFinite(lab.timeGate) ? lab.timeGate.toFixed(2) + ' ns' :
    'all arrival times';
  $('status').textContent =
    `Complete · ${result.counts.max_steps} maximum interactions · ${result.metrics.readback_ms.toFixed(1)} ms readback · ${result.metrics.wall_ms.toFixed(1)} ms compute + readback + decode · ${lab.info.isFallbackAdapter?'software adapter':'hardware adapter'}`;
  $('status').classList.remove('error');
  histogramChart('spectrum', result.histograms.detected, 280, 740,
    'wavelength / nm · pale line = emitted source', {
      secondary: result.histograms.source
    });
  histogramChart('arrival', result.histograms.arrival, 0, result.histograms.time_max,
    `arrival time / ns${result.counts.time_overflow?' · overflow in final bin':''}`);
  physicsChart(result);
  drawPaths(lab, result);
}
if ($('paths')) {
  const lab = new OpticalLab();
  window.opticalLab = lab;
  window.opticalLabReady = lab.initialize().then(info => {
    $('adapter').textContent =
      `${info.vendor||'GPU'} ${info.architecture||info.description||''}${info.isFallbackAdapter?' · software fallback':''}`;
    $('run').disabled = false;
    $('status').textContent = 'Ready. Run an event to compute real photon transport.';
    $('run').addEventListener('click', () => lab.run({
      scene: $('scene').value,
      photons: Number($('photons').value),
      seed: Number($('seed').value),
      polarization: $('polarization').value,
      paths: Math.min(Number($('pathsCount').value), Number($('photons').value))
    }).catch(showError));
    $('scene').addEventListener('change', () => {
      $('status').textContent = 'Selected ' + lab.scenes[$('scene').value].scene.title +
        ' · press Run optical event to update the plots.';
    });
    $('gate').addEventListener('input', () => {
      const value = Number($('gate').value);
      $('gateValue').textContent = value.toFixed(2) + ' ns';
      lab.setTimeGate(value);
    });
    $('allTimes').addEventListener('click', () => {
      $('gate').value = $('gate').max;
      $('gateValue').textContent = 'all arrival times';
      lab.setTimeGate(Infinity);
    });
    window.addEventListener('resize', () => {
      if (lab.result) renderResult(lab, lab.result);
    });
    if (!new URLSearchParams(location.search).has('manual')) lab.run({
      photons: Number($('photons').value)
    }).catch(showError);
    return info;
  }).catch(error => {
    showError(error);
    throw error;
  });

}
