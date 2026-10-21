// stamp.js — the v12 "stamped machine" running in the browser on WebGPU.
//
// One compute dispatch per NCA step: fixed identity/Sobel perception, the
// learned 48->256->16 per-pixel MLP, one stochastic fire decision per cell,
// then a clamp. Between instructions the harness ASSIGNS channels 1..3 — the
// three rectangles — and the rule is expected to erase them again by the end
// of the cycle. Nothing else is injected; channel 0 is the memory and it
// persists untouched across the whole program.
//
// The memory can be any width: v12 is translation invariant, so a program may
// use as many registers as it likes provided each instruction stays local. One
// pipeline is compiled per memory width and cached.
//
// Programs run LIVE: the network is stepped on the GPU as you watch (see
// Machine). Long programs -- hundreds of instructions -- would need hundreds of
// MB to keep every frame, so instead the full state is checkpointed every few
// instructions and anything earlier is re-simulated on demand.

const PARAM_STRIDE = 256;   // minUniformBufferOffsetAlignment
export const HIST_CH = 4;          // channels kept per frame: memory + the 3 rectangles

// ── model ───────────────────────────────────────────────────────────────────

function b64ToF32(b64) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function checked(meta, f) {
  if (f.length !== meta.weights.floats)
    throw new Error(`weights: expected ${meta.weights.floats} floats, got ${f.length}`);
  return { meta, weights: f };
}

/** Model from an already-parsed metadata object — no network at all. This is
 *  what the single-file build uses; relative URLs are not always resolvable
 *  when the page is embedded. */
export function modelFromMeta(meta) {
  if (!meta.weights.b64) throw new Error('inline model needs weights.b64');
  return checked(meta, b64ToF32(meta.weights.b64));
}

export async function loadModel(base = 'data/') {
  const meta = await (await fetch(base + 'stamp_v12.json')).json();
  if (meta.weights.b64) return modelFromMeta(meta);   // inline base64
  return checked(meta, new Float32Array(
    await (await fetch(base + meta.weights.file)).arrayBuffer()));   // sidecar .bin
}

// ── geometry (mirrors TileLayoutV8; the layout is a few inequalities, so the
//    demo rebuilds it rather than shipping a second copy of the masks) ───────

export class Layout {
  constructor(meta, prog) {
    Object.assign(this, {
      H: prog.H, W: prog.W, r: meta.radius, colX: prog.col_x, rowY: prog.row_y,
      nRegs: prog.n_regs, wordBits: prog.word_bits ?? meta.word_bits,
    });
  }

  // the squares' extent plus a 1px margin — neighbouring columns never touch
  columnRect(c) {
    const m = this.r + 1, cx = this.colX[c];
    const y0 = this.rowY[0] - m, y1 = this.rowY[this.rowY.length - 1] + m;
    const px = [];
    for (let y = Math.max(0, y0); y <= Math.min(this.H - 1, y1); y++)
      for (let x = Math.max(0, cx - m); x <= Math.min(this.W - 1, cx + m); x++)
        px.push(y * this.W + x);
    return px;
  }

  stamp(ins) {
    const n = this.H * this.W, s = new Float32Array(3 * n);
    if (ins.op === 'NOP') return s;
    for (const i of this.columnRect(ins.a)) s[i] = 1;
    for (const i of this.columnRect(ins.b)) s[n + i] = 1;
    const v = ins.op === 'SUBLEQ' ? -1 : 1;
    for (const i of this.columnRect(ins.dst)) s[2 * n + i] = v;
    return s;
  }

  dataScreen(regs) {
    const s = new Float32Array(this.H * this.W);
    for (let c = 0; c < this.nRegs; c++)
      for (let b = 0; b < this.wordBits; b++) {
        const on = (regs[c] >> (this.wordBits - 1 - b)) & 1;
        this._disc(s, this.colX[c], this.rowY[b], on ? 1 : -1);
      }
    return s;
  }

  _disc(s, cx, cy, v) {
    for (let y = cy - this.r; y <= cy + this.r; y++)
      for (let x = cx - this.r; x <= cx + this.r; x++) s[y * this.W + x] = v;
  }

  // mean over the square EXCLUDING its centre pixel — same read the trainer
  // scored itself with, so "what the demo says" and "what the eval said" agree
  _ring(ch0, cx, cy) {
    let s = 0, n = 0;
    for (let y = cy - this.r; y <= cy + this.r; y++)
      for (let x = cx - this.r; x <= cx + this.r; x++) {
        if (x === cx && y === cy) continue;
        s += ch0[y * this.W + x]; n++;
      }
    return s / n;
  }

  // null for a register whose bits are too faint to call
  decode(ch0, tol = 0.25) {
    const out = [];
    for (let c = 0; c < this.nRegs; c++) {
      let v = 0, ok = true;
      for (let b = 0; b < this.wordBits; b++) {
        const m = this._ring(ch0, this.colX[c], this.rowY[b]);
        if (Math.abs(m) < tol) ok = false;
        v = (v << 1) | (m > 0 ? 1 : 0);
      }
      out.push(ok ? v : null);
    }
    return out;
  }
}

// ── WebGPU ──────────────────────────────────────────────────────────────────

// One thread per cell, fully unrolled: weights in uniform buffers (broadcast
// reads), Sobel taps baked in as literals, every per-cell array a named vec4
export function packWeights(meta, f) {
  const kk = meta.kernel ** 2, C = meta.channels, HID = meta.hidden, PERC = 3 * C;
  const oW1 = 3 * kk, oB1 = oW1 + HID * PERC, oW2 = oB1 + HID;
  const w1 = f.slice(oW1, oB1);                       // (HID, PERC) row-major already
  const w2t = new Float32Array(HID * C + HID);        // (HID, C) transposed, then b1
  for (let j = 0; j < HID; j++) for (let o = 0; o < C; o++) w2t[j * C + o] = f[oW2 + o * HID + j];
  w2t.set(f.slice(oB1, oW2), HID * C);
  return { w1, w2t, sx: f.slice(kk, 2 * kk), sy: f.slice(2 * kk, 3 * kk) };
}

export function wgsl(meta, W, H, pk) {
  const n = H * W, k = meta.kernel, pad = (k - 1) / 2;
  const C = meta.channels, HID = meta.hidden, PV = (3 * C) / 4, CV = C / 4;
  const lit = (x) => (Number.isInteger(x) ? x.toFixed(1) : String(x));
  const perc = [];
  for (let c = 0; c < C; c++) {
    let sx = [], sy = [];
    for (let dy = 0; dy < k; dy++) for (let dx = 0; dx < k; dx++) {
      const a = pk.sx[dy * k + dx], b = pk.sy[dy * k + dx];
      if (a === 0 && b === 0) continue;
      const v = `ld(${c}u, ${dx - pad}i, ${dy - pad}i)`;
      if (a !== 0 || b !== 0) {
        perc.push(`  { let v = ${v};${a ? ` sx = sx + ${lit(a)} * v;` : ''}${b ? ` sy = sy + ${lit(b)} * v;` : ''} }`);
      }
    }
    perc.push(`  pc[${3 * c}] = src[${c}u * N + idx]; pc[${3 * c + 1}] = sx; pc[${3 * c + 2}] = sy; sx = 0.0; sy = 0.0;`);
  }
  // pc[] is unrolled into named vec4s by constant indexing only
  const body = perc.join('\n').replace(/pc\[(\d+)\]/g, (_, i) => `p${(i / 4) | 0}.${'xyzw'[i % 4]}`);
  const pdecl = Array.from({ length: PV }, (_, i) => `var p${i} = vec4<f32>();`).join(' ');
  const hdot = Array.from({ length: PV }, (_, i) => `dot(w1[b + ${i}u], p${i})`).join(' + ');
  const ddecl = Array.from({ length: CV }, (_, i) => `var d${i} = vec4<f32>();`).join(' ');
  const dacc = Array.from({ length: CV }, (_, i) => `d${i} = d${i} + h * w2[j * ${CV}u + ${i}u];`).join(' ');
  const outs = [];
  for (let c = 0; c < C; c++)
    outs.push(`  dst[${c}u * N + idx] = clamp(src[${c}u * N + idx] + fire * d${(c / 4) | 0}.${'xyzw'[c % 4]}, ${-meta.state_bound}.0, ${meta.state_bound}.0);`);
  return /* wgsl */ `
struct P { seed: u32, fire: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             w1  : array<vec4<f32>, ${HID * PV}>;
@group(0) @binding(3) var<uniform>             p   : P;
@group(0) @binding(4) var<uniform>             w2  : array<vec4<f32>, ${HID * CV + HID / 4}>;

const W = ${W}i; const H = ${H}i; const N = ${n}u;
var<private> X : i32; var<private> Y : i32;

fn ld(c: u32, dx: i32, dy: i32) -> f32 {           // ZERO padding
  let gx = X + dx; let gy = Y + dy;
  if (gx < 0i || gx >= W || gy < 0i || gy >= H) { return 0.0; }
  return src[c * N + u32(gy * W + gx)];
}

fn hash(x0: u32) -> u32 {
  var x = x0;
  x = x ^ (x >> 16u); x = x * 0x7feb352du;
  x = x ^ (x >> 15u); x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= N) { return; }
  X = i32(idx % u32(W)); Y = i32(idx / u32(W));
  ${pdecl}
  var sx = 0.0; var sy = 0.0;
${body}
  ${ddecl}
  for (var j = 0u; j < ${HID}u; j = j + 1u) {
    let b = j * ${PV}u;
    let h = max(w2[${HID * CV}u + j / 4u][j % 4u] + ${hdot}, 0.0);
    ${dacc}
  }
  // one fire decision per CELL, shared by all its channels
  let rnd = f32(hash((idx * 0x9e3779b9u) ^ hash(p.seed))) * 2.3283064365386963e-10;
  let fire = select(0.0, 1.0, rnd < p.fire);
${outs.join('\n')}
}`;
}

export class StampGPU {
  static async create(model) {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no WebGPU adapter');
    const device = await adapter.requestDevice();
    return new StampGPU(device, model, adapter);
  }

  constructor(device, model, adapter) {
    this.device = device;
    this.adapter = adapter;
    this.m = model.meta;
    // W1 and (W2 transposed, b1) as two uniform buffers: every cell reads the same
    // weight at the same moment, which is exactly what the constant cache is for
    this.pk = packWeights(this.m, model.weights);
    const ub = (a) => { const b = device.createBuffer({ size: a.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, a); return b; };
    this.w1 = ub(this.pk.w1); this.w2 = ub(this.pk.w2t);
    // explicit layout: an 'auto' layout never has dynamic offsets, and the
    // per-step params are one uniform buffer indexed by offset
    const CS = GPUShaderStage.COMPUTE;
    this.bgl = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: CS, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: CS, buffer: { type: 'storage' } },
      { binding: 2, visibility: CS, buffer: { type: 'uniform' } },
      { binding: 3, visibility: CS, buffer: { type: 'uniform', hasDynamicOffset: true } },
      { binding: 4, visibility: CS, buffer: { type: 'uniform' } },
    ] });
    this.plLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    this.geoms = new Map();
  }

  /** Pipeline + ping-pong state for one memory size, compiled once and cached.
   *  WebGPU validation errors do not throw -- a broken shader just gives an
   *  invalid pipeline that silently does nothing -- so they are surfaced here. */
  async _geometry(W, H) {
    const key = `${W}x${H}`;
    if (this.geoms.has(key)) return this.geoms.get(key);
    const d = this.device, n = W * H;
    d.pushErrorScope('validation');
    const module = d.createShaderModule({ code: wgsl(this.m, W, H, this.pk) });
    const pipeline = d.createComputePipeline({
      layout: this.plLayout, compute: { module, entryPoint: 'main' } });
    const S = GPUBufferUsage.STORAGE, C = GPUBufferUsage.COPY_SRC, D = GPUBufferUsage.COPY_DST;
    const stateBytes = this.m.channels * n * 4;
    const g = {
      W, H, n, pipeline, stateBytes, frameBytes: HIST_CH * n * 4,
      A: d.createBuffer({ size: stateBytes, usage: S | C | D }),
      B: d.createBuffer({ size: stateBytes, usage: S | C | D }),
      wg: [Math.ceil(n / 64), 1],
    };
    const info = await module.getCompilationInfo();
    const bad = info.messages.filter((x) => x.type === 'error');
    const err = await d.popErrorScope();
    if (bad.length)
      throw new Error('WGSL: ' + bad.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join('; '));
    if (err) throw new Error('WebGPU: ' + err.message);
    this.geoms.set(key, g);
    return g;
  }

  _bind(src, dst, params) {
    return this.device.createBindGroup({
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: this.w1 } },
        { binding: 3, resource: { buffer: params, offset: 0, size: 16 } },
        { binding: 4, resource: { buffer: this.w2 } },
      ],
    });
  }

  /** The register decoder for one program shape, compiled once and cached. */
  async _decoder(prog, meta) {
    const key = `${prog.W}x${prog.H}:${prog.n_regs}:${prog.word_bits}`;
    this.decs = this.decs || new Map();
    if (this.decs.has(key)) return this.decs.get(key);
    const d = this.device;
    d.pushErrorScope('validation');
    const module = d.createShaderModule({ code: wgslDecode(meta, prog) });
    const pipeline = d.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const err = await d.popErrorScope();
    if (err) throw new Error('WebGPU (decoder): ' + err.message);
    this.decs.set(key, pipeline);
    return pipeline;
  }
}

/** Reads every register straight off channel 0, exactly as the trainer scored
 *  itself: the mean of each bit's square excluding its centre pixel, sign = bit,
 *  |mean| < 0.25 = can't call it (written as 0xffffffff). */
export function wgslDecode(meta, prog) {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read>       st  : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<u32>;
const W = ${prog.W}i; const NR = ${prog.n_regs}u; const WB = ${prog.word_bits}u;
const R = ${meta.radius}i; const PITCH = ${meta.pitch}i; const X0 = ${prog.col_x[0]}i; const Y0 = ${prog.row_y[0]}i;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= NR) { return; }
  let cx = X0 + i32(r) * PITCH;
  var v = 0u; var ok = true;
  for (var b = 0i; b < i32(WB); b = b + 1i) {
    let cy = Y0 + b * PITCH;
    var s = 0.0;
    for (var dy = -R; dy <= R; dy = dy + 1i) {
      for (var dx = -R; dx <= R; dx = dx + 1i) {
        if (dx != 0i || dy != 0i) { s = s + st[u32((cy + dy) * W + cx + dx)]; }
      }
    }
    let m = s / f32((2i * R + 1i) * (2i * R + 1i) - 1i);
    if (abs(m) < 0.25) { ok = false; }
    v = (v << 1u) | select(0u, 1u, m > 0.0);
  }
  out[r] = select(0xffffffffu, v, ok);
}`;
}

const CHUNK = 2048;                       // NCA steps per command buffer = param slots
export const UNKNOWN = 0xffffffff;

/** One program running live on the GPU.
 *
 *  Frame f is (instruction c, step w) with f = c*(K+1) + w: w = 0 is the moment
 *  the rectangles have been stamped in, w = K is the end of that instruction.
 *  The machine only ever moves forward; seeking backwards restores the nearest
 *  checkpoint and re-simulates. That is exact, not approximate: the fire mask of
 *  global step s is seeded by s alone, so a re-run draws the same randomness. */
export class Machine {
  static async create(gpu, layout, prog, seed = (Math.random() * 1e9) | 0) {
    const g = await gpu._geometry(layout.W, layout.H);
    const dec = await gpu._decoder(prog, gpu.m);
    return new Machine(gpu, g, dec, layout, prog, seed);
  }

  constructor(gpu, g, dec, layout, prog, seed) {
    const d = gpu.device, U = GPUBufferUsage;
    Object.assign(this, { gpu, g, d, layout, P: prog, K: gpu.m.k_cycle, seed });
    this.NI = prog.instrs.length;
    this.NR = prog.n_regs;
    this.total = this.NI * (this.K + 1);
    if (this.K % 2) throw new Error('k_cycle must be even: every instruction must end on buffer A');

    this.params = d.createBuffer({ size: CHUNK * PARAM_STRIDE, usage: U.UNIFORM | U.COPY_DST });
    this.seeds = new ArrayBuffer(CHUNK * PARAM_STRIDE);
    this.bgAB = gpu._bind(g.A, g.B, this.params);
    this.bgBA = gpu._bind(g.B, g.A, this.params);

    // checkpoints: at most 256 of them and ~96 MB, whatever the program length
    const maxCk = Math.max(2, Math.min(256, Math.floor(96e6 / g.stateBytes)));
    this.I = Math.max(1, Math.ceil(this.NI / maxCk));
    this.ckpt = d.createBuffer({ size: (Math.floor((this.NI - 1) / this.I) + 1) * g.stateBytes,
                                 usage: U.COPY_SRC | U.COPY_DST });

    this.decPipe = dec;
    this.decOut = d.createBuffer({ size: Math.max(16, this.NR * 4), usage: U.STORAGE | U.COPY_SRC });
    this.decHist = d.createBuffer({ size: this.NI * this.NR * 4, usage: U.COPY_SRC | U.COPY_DST });
    this.decBG = d.createBindGroup({ layout: dec.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: g.A } },
      { binding: 1, resource: { buffer: this.decOut } }] });

    this.dispBytes = HIST_CH * g.n * 4;
    this.stage = d.createBuffer({ size: this.dispBytes, usage: U.MAP_READ | U.COPY_DST });
    this.stageDec = d.createBuffer({ size: this.NI * this.NR * 4, usage: U.MAP_READ | U.COPY_DST });

    this.decoded = new Array(this.NI);    // decoded registers at the END of each instruction
    this.decRead = 0;                     // decoded[0 .. decRead) are on the CPU
    this.completed = 0;                   // instructions whose end has been simulated
    this.reset(seed);
  }

  reset(seed = this.seed) {
    const { d, g, layout, P } = this;
    this.seed = seed;
    const init = new Float32Array(this.gpu.m.channels * g.n);
    init.set(layout.dataScreen(P.regs0), 0);
    init.set(layout.stamp(P.instrs[0]), g.n);
    d.queue.writeBuffer(g.A, 0, init);
    d.queue.writeBuffer(g.B, 0, init);
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(g.A, 0, this.ckpt, 0, g.stateBytes);
    d.queue.submit([enc.finish()]);
    this.cur = 0;
    this.completed = 0; this.decRead = 0; this.decoded = new Array(this.NI);
  }

  /** Move to frame f (clamped). Pure command encoding; nothing here waits. */
  seek(f) {
    const { d, g, K } = this;
    f = Math.max(0, Math.min(this.total - 1, Math.round(f)));
    if (f < this.cur) {                         // backwards: restore, then replay
      const ck = Math.floor(Math.floor(f / (K + 1)) / this.I);
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(this.ckpt, ck * g.stateBytes, g.A, 0, g.stateBytes);
      d.queue.submit([enc.finish()]);
      this.cur = ck * this.I * (K + 1);
    }
    this._advance(f);
  }

  _advance(target) {
    const { d, g, K, P } = this;
    let enc = d.createCommandEncoder(), slot = 0;
    const flush = () => {
      if (slot) d.queue.writeBuffer(this.params, 0, this.seeds, 0, slot * PARAM_STRIDE);
      d.queue.submit([enc.finish()]);
      enc = d.createCommandEncoder(); slot = 0;
    };
    while (this.cur < target) {
      const c = Math.floor(this.cur / (K + 1)), w = this.cur % (K + 1);
      if (w === K) {                             // next instruction: stamp it in
        flush();
        d.queue.writeBuffer(g.A, g.n * 4, this.layout.stamp(P.instrs[c + 1]));
        this.cur++;
        if ((c + 1) % this.I === 0)
          enc.copyBufferToBuffer(g.A, 0, this.ckpt, ((c + 1) / this.I) * g.stateBytes, g.stateBytes);
        continue;
      }
      const dv = new DataView(this.seeds, slot * PARAM_STRIDE);
      dv.setUint32(0, (this.seed + (c * K + w) * 2654435761) >>> 0, true);
      dv.setFloat32(4, this.gpu.m.fire_rate, true);
      const pass = enc.beginComputePass();
      pass.setPipeline(g.pipeline);
      pass.setBindGroup(0, w % 2 === 0 ? this.bgAB : this.bgBA, [slot * PARAM_STRIDE]);
      pass.dispatchWorkgroups(g.wg[0], g.wg[1]);
      pass.end();
      slot++; this.cur++;
      if (w + 1 === K) {                         // instruction c just ended (on buffer A)
        const dp = enc.beginComputePass();
        dp.setPipeline(this.decPipe); dp.setBindGroup(0, this.decBG);
        dp.dispatchWorkgroups(Math.ceil(this.NR / 64));
        dp.end();
        enc.copyBufferToBuffer(this.decOut, 0, this.decHist, c * this.NR * 4, this.NR * 4);
        this.completed = Math.max(this.completed, c + 1);
      }
      if (slot === CHUNK) flush();
    }
    flush();
  }

  /** Channels 0..3 of the current frame, plus any newly decoded boundaries. */
  async read() {
    const { d, g, K } = this;
    const w = this.cur % (K + 1);
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(w % 2 === 0 ? g.A : g.B, 0, this.stage, 0, this.dispBytes);
    const lo = this.decRead, hi = this.completed;
    if (hi > lo) enc.copyBufferToBuffer(this.decHist, lo * this.NR * 4, this.stageDec, 0, (hi - lo) * this.NR * 4);
    d.queue.submit([enc.finish()]);
    const waits = [this.stage.mapAsync(GPUMapMode.READ)];
    if (hi > lo) waits.push(this.stageDec.mapAsync(GPUMapMode.READ, 0, (hi - lo) * this.NR * 4));
    await Promise.all(waits);
    const disp = new Float32Array(this.stage.getMappedRange().slice(0));
    this.stage.unmap();
    if (hi > lo) {
      const u = new Uint32Array(this.stageDec.getMappedRange(0, (hi - lo) * this.NR * 4).slice(0));
      this.stageDec.unmap();
      for (let c = lo; c < hi; c++) this.decoded[c] = Array.from(u.subarray((c - lo) * this.NR, (c - lo + 1) * this.NR));
      this.decRead = hi;
    }
    return disp;
  }

  destroy() {
    for (const b of [this.params, this.ckpt, this.decOut, this.decHist, this.stage, this.stageDec]) b.destroy();
  }
}
