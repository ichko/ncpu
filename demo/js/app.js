// app.js — the interactive plate. Set the input bits, run the NCA simulation
// up to the number of steps it was trained on (its "horizon"), then pause and
// check whether the bits read off the grid match the correct answer.

import { NCASim } from './nca.js';
import { geometry, buildImage, outputMasks } from './screen.js';

// ground-truth functions + the trained rollout horizon (max steps seen in training)
const TASKS = {
  and:        { label: 'AND',       group: 'gate',  horizon: 64,  fn: (b) => [b[0] & b[1]] },
  or:         { label: 'OR',        group: 'gate',  horizon: 64,  fn: (b) => [b[0] | b[1]] },
  xor:        { label: 'XOR',       group: 'gate',  horizon: 64,  fn: (b) => [b[0] ^ b[1]] },
  nand:       { label: 'NAND',      group: 'gate',  horizon: 64,  fn: (b) => [1 - (b[0] & b[1])] },
  nor:        { label: 'NOR',       group: 'gate',  horizon: 64,  fn: (b) => [1 - (b[0] | b[1])] },
  xnor:       { label: 'XNOR',      group: 'gate',  horizon: 64,  fn: (b) => [1 - (b[0] ^ b[1])] },
  majority3:  { label: 'MAJ-3',     group: 'gate',  horizon: 64,  fn: (b) => [b[0] + b[1] + b[2] >= 2 ? 1 : 0] },
  half_adder: { label: 'HALF-ADD',  group: 'gate',  horizon: 64,  fn: (b) => [b[0] ^ b[1], b[0] & b[1]] },
  adder4:     { label: 'ADD 4-BIT', group: 'adder', horizon: 128, bits: 4 },
  adder8:     { label: 'ADD 8-BIT', group: 'adder', horizon: 128, bits: 8 },
};

const $ = (id) => document.getElementById(id);
const els = {
  canvas: $('field'), overlay: $('ioOverlay'), screen: $('screen'), plate: $('plate'),
  demo: $('demo'), conn: $('connSvg'),
  tabs: $('taskTabs'), inputs: $('inputBits'), outputs: $('outputReadout'),
  expr: $('exprLine'), liveNum: $('liveNum'),
  bar: $('pbar'), pstep: $('pstep'), phorizon: $('phorizon'),
  play: $('playBtn'), again: $('againBtn'), rand: $('randBtn'),
  fps: $('fpsSlider'), fpsVal: $('fpsVal'),
  debug: $('debugToggle'), status: $('simStatus'),
};

let sim = null, cfg = null, geo = null, masks = null, taskKey = null;
let inputBits = [], step = 0, horizon = 64;
let phase = 'hold';                 // 'hold' (show input) | 'run' | 'done'
let holdUntil = 0;
let ema = null, expArr = [];        // per-output-bit moving average + expected bits
let fps = 30, debug = true;         // fire rate is fixed to the trained value
let raf = null, lastT = 0;
const cache = {};
const HOLD_MS = 1500;               // pause on the encoded input before running

const ERR = '#c23b2a';                 // only to flag a wrong output bit
let inputRings = [], outputRings = [];
let inputBtns = [], inPaths = [], outPaths = [];

// viridis colormap (same fit as the display shader): value in [-1,1] -> css rgb,
// so a wire is exactly the colour its circle shows on the grid.
function viridis(v) {
  const t = Math.max(0, Math.min(1, v * 0.5 + 0.5));
  const C = [
    [0.2777, 0.0054, 0.3341], [0.1051, 1.4046, 1.3846], [-0.3309, 0.2148, 0.0951],
    [-4.6342, -5.7991, -19.3324], [6.2283, 14.1799, 56.6906], [4.7764, -13.7451, -65.3530],
    [-5.4355, 4.6459, 26.3124],
  ];
  const ch = (k) => {
    let s = C[6][k];
    for (let j = 5; j >= 0; j--) s = C[j][k] + t * s;
    return Math.round(Math.max(0, Math.min(1, s)) * 255);
  };
  return `rgb(${ch(0)},${ch(1)},${ch(2)})`;
}

// orthogonal polyline with rounded corners (from web/wires.js)
function roundedPath(pts, r) {
  let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [px, py] = pts[i - 1], [x, y] = pts[i], [nx, ny] = pts[i + 1];
    const rIn = Math.min(r, Math.hypot(x - px, y - py) / 2, Math.hypot(nx - x, ny - y) / 2);
    const [ix, iy] = towards(x, y, px, py, rIn);
    const [ox, oy] = towards(x, y, nx, ny, rIn);
    d += ` L ${ix.toFixed(1)} ${iy.toFixed(1)} Q ${x.toFixed(1)} ${y.toFixed(1)} ${ox.toFixed(1)} ${oy.toFixed(1)}`;
  }
  const last = pts[pts.length - 1];
  return d + ` L ${last[0].toFixed(1)} ${last[1].toFixed(1)}`;
}
function towards(x, y, tx, ty, dist) {
  const len = Math.hypot(tx - x, ty - y) || 1;
  return [x + ((tx - x) / len) * dist, y + ((ty - y) / len) * dist];
}

const loadModel = async (k) => (cache[k] ||= await fetch(`data/${k}.json`).then((r) => r.json()));
const intBits = (v, w) => { const o = []; for (let i = w - 1; i >= 0; i--) o.push((v >> i) & 1); return o; };
const bitsInt = (a) => a.reduce((x, b) => x * 2 + b, 0);
const nInputs = (k) => TASKS[k].group === 'gate' ? (k === 'majority3' ? 3 : 2) : TASKS[k].bits * 2;

// the plate is a FIXED box; every grid is scaled to fit inside it so switching
// tasks (which have different aspect ratios) never shifts the surrounding layout
const BOX_W = 244, BOX_H = 330;

async function selectTask(key) {
  taskKey = key;
  cfg = await loadModel(key);
  geo = geometry(cfg);
  masks = outputMasks(cfg, geo);
  horizon = TASKS[key].horizon;

  const s = Math.min(BOX_W / cfg.W, BOX_H / cfg.H);
  const cw = Math.round(cfg.W * s), ch = Math.round(cfg.H * s);
  els.plate.style.width = cw + 'px'; els.plate.style.height = ch + 'px';
  els.canvas.style.width = cw + 'px'; els.canvas.style.height = ch + 'px';
  // reserve gutters for the wire ribbon so traces route around the plate rather
  // than over it: a top band (one lane per input) and a left gutter (the input
  // circles sit on the grid's far edge, opposite the control panel).

  try {
    if (sim) sim.dispose();
    sim = new NCASim(els.canvas, cfg);
    els.status.textContent = '';
  } catch (e) { els.status.textContent = 'needs WebGL2 + float buffers · ' + e.message; throw e; }

  if (TASKS[key].group === 'adder') {
    const b = TASKS[key].bits;
    inputBits = intBits((1 << b) - 1, b).concat(intBits(1, b)); // e.g. 15 + 1
  } else {
    inputBits = new Array(nInputs(key)).fill(1);
  }
  // build the DOM and wires ONCE per task; reseed() reuses them thereafter
  buildTabs(); buildInputUI(); buildOutputUI(); drawOverlay(); drawConnectors();
  reseed();
}

function buildTabs() {
  if (!els.tabs.childElementCount) {
    Object.keys(TASKS).forEach((k) => {
      const b = document.createElement('button');
      b.className = 'tab'; b.textContent = TASKS[k].label; b.dataset.key = k;
      b.onclick = () => selectTask(k);
      els.tabs.appendChild(b);
    });
  }
  [...els.tabs.children].forEach((c) => c.classList.toggle('active', c.dataset.key === taskKey));
}

// input order matching the grid: rows top→bottom, columns left→right, so each
// toggle sits at a unique height and lines up with its circle's row. This is
// what lets every wire keep its own lane.
function inputOrder() {
  return geo.inputs.map((c, i) => i).sort((a, b) =>
    (geo.inputs[a].y - geo.inputs[b].y) || (geo.inputs[a].x - geo.inputs[b].x));
}
function inColumns() { return [...new Set(geo.inputs.map((c) => c.x))].sort((a, b) => a - b); }

function buildInputUI() {
  els.inputs.innerHTML = '';
  inputBtns = [];
  const t = TASKS[taskKey];
  if (t.group === 'adder') {
    els.inputs.appendChild(operandRow('A', 0, t.bits));
    els.inputs.appendChild(operandRow('B', t.bits, t.bits));
  } else {
    // one input per row (A, B, [C]) — same shape as the adders, so the wires
    // never have to cross a neighbouring control
    const names = taskKey === 'majority3' ? ['A', 'B', 'C'] : ['A', 'B'];
    names.forEach((nm, i) => {
      const row = document.createElement('div'); row.className = 'bit-row';
      const tag = document.createElement('span'); tag.className = 'operand-tag'; tag.textContent = nm;
      row.append(tag, bitToggle(i));
      els.inputs.appendChild(row);
    });
  }
  updateExpr();
}
function operandRow(name, off, w) {
  const row = document.createElement('div'); row.className = 'bit-row';
  const tag = document.createElement('span'); tag.className = 'operand-tag'; tag.textContent = name; row.append(tag);
  // a real number input: type a value, it becomes bits that wire to the grid
  const num = document.createElement('input');
  num.type = 'number'; num.className = 'num-in'; num.min = 0; num.max = (1 << w) - 1;
  num.dataset.off = off; num.dataset.w = w;
  num.value = bitsInt(inputBits.slice(off, off + w));
  num.oninput = () => {
    let v = parseInt(num.value, 10); if (isNaN(v)) return;
    v = Math.max(0, Math.min((1 << w) - 1, v));
    setOperand(off, w, v);
  };
  row.append(num);
  const bits = document.createElement('span'); bits.className = 'bit-cluster';
  for (let i = 0; i < w; i++) bits.append(bitToggle(off + i));
  row.append(bits);
  return row;
}
function setOperand(off, w, v) {
  for (let i = 0; i < w; i++) inputBits[off + i] = (v >> (w - 1 - i)) & 1;
  refreshInputVisuals(); updateExpr(); reseed();
}
// sync the bit buttons and number fields to inputBits
function refreshInputVisuals() {
  inputBtns.forEach((b, idx) => {
    if (!b) return;
    b.classList.toggle('on', !!inputBits[idx]); b.textContent = inputBits[idx] ? '1' : '0';
  });
  els.inputs.querySelectorAll('.num-in').forEach((n) => {
    if (document.activeElement === n) return;
    const off = +n.dataset.off, w = +n.dataset.w;
    n.value = bitsInt(inputBits.slice(off, off + w));
  });
}
// row layout of an input: which operand row it's in and its position there
function rowInfo(i) {
  const t = TASKS[taskKey];
  if (t.group === 'adder') return { pos: i % t.bits, len: t.bits };
  return { pos: i, len: nInputs(taskKey) };
}
function bitToggle(idx) {
  const b = document.createElement('button');
  b.className = 'bit' + (inputBits[idx] ? ' on' : '');
  b.textContent = inputBits[idx] ? '1' : '0';
  b.onclick = () => {
    inputBits[idx] ^= 1;
    refreshInputVisuals(); updateExpr(); reseed();
  };
  inputBtns[idx] = b;
  return b;
}
function updateExpr() {
  const t = TASKS[taskKey];
  if (t.group === 'adder') {
    const b = t.bits, a = bitsInt(inputBits.slice(0, b)), bb = bitsInt(inputBits.slice(b, 2 * b));
    els.expr.textContent = `${a} + ${bb} = ${a + bb}`;
  } else {
    els.expr.textContent = `${t.label}(${inputBits.join(', ')}) → ${t.fn(inputBits).join(', ')}`;
  }
}

function expected() {
  const t = TASKS[taskKey];
  if (t.group === 'adder') {
    const b = t.bits, s = bitsInt(inputBits.slice(0, b)) + bitsInt(inputBits.slice(b, 2 * b));
    return intBits(s, cfg.n_right);
  }
  return t.fn(inputBits);
}

// ---- overlay: rings on the circles, coloured to match the field so wires blend
function drawOverlay() {
  const { W, H } = cfg, ns = 'http://www.w3.org/2000/svg';
  els.overlay.setAttribute('viewBox', `0 0 ${W} ${H}`);
  els.overlay.innerHTML = '';
  els.overlay.style.display = debug ? '' : 'none';
  const ring = (c) => {
    const el = document.createElementNS(ns, 'circle');
    el.setAttribute('cx', c.x + 0.5); el.setAttribute('cy', c.y + 0.5);
    el.setAttribute('r', geo.r + 1.2); el.setAttribute('fill', 'none');
    el.setAttribute('stroke-width', 1.2);
    els.overlay.appendChild(el);
    return el;
  };
  inputRings = geo.inputs.map(ring);
  outputRings = geo.outputs.map(ring);
  updateOverlay();
}

function updateOverlay() {
  inputRings.forEach((el, i) => {
    el.setAttribute('stroke', '#fff');
    el.setAttribute('stroke-opacity', inputBits[i] ? 0.95 : 0.35);
  });
  outputRings.forEach((el, i) => {
    if (phase === 'hold' || !ema) { el.setAttribute('stroke', '#fff'); el.setAttribute('stroke-opacity', 0.4); return; }
    const got = ema[i] > 0, ok = (got ? 1 : 0) === expArr[i];
    el.setAttribute('stroke', (!ok && phase === 'done') ? ERR : '#fff');
    el.setAttribute('stroke-opacity', got ? 0.95 : 0.4);
  });
}

// ---- chip-style traces (after web/wires.js of the classification demo):
// each wire leaves its circle radially, hops onto its own nested ring around the
// plate, and runs out to its control. Colour matches the circle; the only part
// over the simulation is a faint stub from the rim to the plate edge.
const LANE0 = 22, CORNER = 5;   // how far the nearest wire bends out from the grid
function drawConnectors() {
  const ns = 'http://www.w3.org/2000/svg';
  els.conn.innerHTML = '';
  const mk = () => {
    const stub = document.createElementNS(ns, 'path');
    const route = document.createElementNS(ns, 'path');
    stub.setAttribute('fill', 'none'); route.setAttribute('fill', 'none');
    route.setAttribute('stroke-linecap', 'round');
    els.conn.append(stub, route);
    return { stub, route };
  };
  inPaths = geo.inputs.map(mk);
  outPaths = geo.outputs.map(mk);
  requestAnimationFrame(layoutConnectors);
}

function layoutConnectors() {
  if (!cfg || !inPaths.length) return;
  const dRect = els.demo.getBoundingClientRect();
  const cRect = els.canvas.getBoundingClientRect();
  const btn0 = inputBtns[0] && inputBtns[0].getBoundingClientRect();
  const stacked = !cRect.width || (btn0 && btn0.top > cRect.bottom - 4);
  if (stacked || !debug) { els.conn.style.display = 'none'; return; }
  els.conn.style.display = '';
  els.conn.setAttribute('viewBox', `0 0 ${dRect.width} ${dRect.height}`);
  els.conn.setAttribute('width', dRect.width); els.conn.setAttribute('height', dRect.height);
  const sc = cRect.width / cfg.W;
  const gL = cRect.left - dRect.left, gT = cRect.top - dRect.top;
  const gR = cRect.right - dRect.left, gB = cRect.bottom - dRect.top;
  const LX = (x) => gL + (x + 0.5) * sc, LY = (y) => gT + (y + 0.5) * sc;
  const rr = (geo.r + 1.2) * sc;                       // ring rim in px
  const cols = inColumns(), nCols = cols.length;

  // lane spacing — wide enough to fan out clearly, capped so the nest fits the gutter
  const gapFor = (n) => Math.max(2.2, Math.min(4, 34 / Math.max(1, n - 1)));
  // rank each side top→bottom (then left→right) so the lanes nest in order
  const byRow = (arr) => arr.map((c, i) => i).sort((a, b) => (arr[a].y - arr[b].y) || (arr[a].x - arr[b].x));
  const rankIn = new Array(geo.inputs.length); byRow(geo.inputs).forEach((idx, k) => rankIn[idx] = k);
  const rankOut = new Array(geo.outputs.length); byRow(geo.outputs).forEach((idx, k) => rankOut[idx] = k);
  const gIn = gapFor(geo.inputs.length), gOut = gapFor(geo.outputs.length);

  // A wire leaves its circle sideways (L for inputs, R for outputs), rides its
  // own lane just outside the grid, and rises/falls to its control. It never
  // dips below the grid, so it never reaches the progress bar. The stub from the
  // rim to the grid edge is the only part over the simulation.
  function route(c, target, off, side, yoff) {
    const px = LX(c.x), py = LY(c.y) + yoff;
    const dir = side === 'L' ? -1 : 1;
    const edgeX = side === 'L' ? gL : gR;
    const rimX = px + dir * rr;
    const railX = edgeX + dir * off;
    const ty = Math.min(target.y, gB - 3);             // clamp: stay above the grid bottom
    const pts = [[edgeX, py], [railX, py], [railX, ty], [target.x, ty]];
    return { stubD: `M ${rimX.toFixed(1)} ${py.toFixed(1)} L ${edgeX.toFixed(1)} ${py.toFixed(1)}`,
             routeD: roundedPath(pts, CORNER) };
  }

  geo.inputs.forEach((c, i) => {
    const btn = inputBtns[c.bit]; if (!btn) return;
    const r = btn.getBoundingClientRect();
    const target = { x: r.right - dRect.left, y: r.top + r.height / 2 - dRect.top };
    const yoff = (cols.indexOf(c.x) - (nCols - 1) / 2) * 3.2;   // split same-row A/B
    const d = route(c, target, LANE0 + rankIn[i] * gIn, 'L', yoff);
    inPaths[i].stub.setAttribute('d', d.stubD);
    inPaths[i].route.setAttribute('d', d.routeD);
  });
  geo.outputs.forEach((c, i) => {
    const dot = outCells[i] && outCells[i].dot; if (!dot) return;
    const r = dot.getBoundingClientRect();
    const target = { x: r.left - dRect.left, y: r.top + r.height / 2 - dRect.top };
    const d = route(c, target, LANE0 + rankOut[i] * gOut, 'R', 0);
    outPaths[i].stub.setAttribute('d', d.stubD);
    outPaths[i].route.setAttribute('d', d.routeD);
  });
  updateConnectors();
}

// wires: fixed hair-thin width; only opacity responds. Black over the paper,
// white where the stub crosses the (dark) simulation.
const WIRE_W = 0.75;
function paintWire(w, a, err) {
  w.route.setAttribute('stroke', err ? ERR : '#000');
  w.route.setAttribute('stroke-width', WIRE_W);
  w.route.setAttribute('opacity', (0.13 + 0.42 * a).toFixed(3));
  w.stub.setAttribute('stroke', err ? ERR : '#fff');
  w.stub.setAttribute('stroke-width', WIRE_W);
  w.stub.setAttribute('opacity', (0.16 + 0.5 * a).toFixed(3));
}
function updateConnectors() {
  inPaths.forEach((w, i) => paintWire(w, inputBits[geo.inputs[i].bit] ? 1 : 0.28, false));
  outPaths.forEach((w, i) => {
    const v = ema ? ema[i] : -1;
    const bad = phase === 'done' && ema && ((v > 0 ? 1 : 0) !== expArr[i]);
    paintWire(w, Math.min(1, Math.max(0, v * 0.5 + 0.5)), bad);
  });
}

let outCells = [];
function buildOutputUI() {
  els.outputs.innerHTML = '';
  const exp = expected();
  // one line per output circle, stacked top→bottom to match the grid's output
  // column, so each output wire keeps its own lane
  outCells = exp.map((e, i) => {
    const line = document.createElement('div'); line.className = 'out-line';
    const lbl = document.createElement('span'); lbl.className = 'out-lbl'; lbl.textContent = bitLabel(i);
    const dot = document.createElement('div'); dot.className = 'out-dot';
    line.append(dot, lbl); els.outputs.appendChild(line);
    return { dot, exp: e };
  });
}
function updateExpectations() {
  expArr = expected();
  outCells.forEach((c, i) => { if (c) c.exp = expArr[i]; });
}
function bitLabel(i) {
  const t = TASKS[taskKey];
  if (t.group === 'adder') return '2^' + (cfg.n_right - 1 - i);
  if (taskKey === 'half_adder') return i === 0 ? 'sum' : 'carry';
  return 'out';
}

function readMeans() {
  const ch0 = sim.readChannel0();
  return masks.map((idx) => { let s = 0; for (const p of idx) s += ch0[p]; return s / idx.length; });
}
function updateLiveDots() {
  if (!ema) return;
  ema.forEach((v, i) => { const c = outCells[i]; if (c) c.dot.classList.toggle('hi', v > 0); });
}
// the number the grid is currently spelling out, and whether it's right yet
function updateLive() {
  if (!els.liveNum) return;
  const t = TASKS[taskKey];
  if (!ema || phase === 'hold') { els.liveNum.textContent = ''; els.liveNum.className = 'live-num'; return; }
  const got = ema.map((v) => v > 0 ? 1 : 0);
  let text, ok;
  if (t.group === 'adder') {
    const n = bitsInt(got);
    const a = bitsInt(inputBits.slice(0, t.bits)), b = bitsInt(inputBits.slice(t.bits, 2 * t.bits));
    ok = n === a + b; text = '= ' + n;
  } else {
    text = got.join(''); ok = got.join('') === expArr.join('');
  }
  // once the run finishes, mark the number with a simple ✓ / ✗
  if (phase === 'done') text += ok ? '  ✓' : '  ✗';
  els.liveNum.textContent = text;
  els.liveNum.className = 'live-num ' + (ok ? 'ok' : (phase === 'done' ? 'bad' : ''));
}

// re-seed the grid and reset the run WITHOUT rebuilding any DOM or wires — the
// controls and connector geometry are stable, so toggling an input never makes
// the output wires jump.
function reseed() {
  if (!sim) return;
  sim.seed(buildImage(cfg, geo, inputBits));
  sim.draw();
  step = 0; ema = null; lastT = 0;
  phase = 'hold'; holdUntil = performance.now() + HOLD_MS;
  updateExpectations();
  outCells.forEach((c) => { if (c) c.dot.classList.remove('hi', 'ok', 'bad'); });
  els.play.textContent = 'Pause';
  updateLive(); updateOverlay(); updateConnectors(); updateProgress();
}

function finish() {
  const exp = expected();
  const got = ema.map((v) => v > 0 ? 1 : 0);
  let ok = true;
  got.forEach((g, i) => {
    const c = outCells[i]; if (!c) return;
    c.dot.classList.toggle('hi', g === 1);
    c.dot.classList.toggle('ok', g === c.exp);
    c.dot.classList.toggle('bad', g !== c.exp);
    if (g !== c.exp) ok = false;
  });
  els.play.textContent = 'Replay';
  updateLive(); updateOverlay(); updateConnectors();
}

function updateProgress() {
  els.bar.style.width = Math.min(100, (step / horizon) * 100) + '%';
  els.pstep.textContent = String(step).padStart(3, '0');
  els.phorizon.textContent = String(horizon).padStart(3, '0');
}

function frame(ts) {
  raf = requestAnimationFrame(frame);
  if (!sim) return;
  if (phase === 'hold') { if (ts >= holdUntil) { phase = 'run'; lastT = 0; } return; }
  if (phase !== 'run') return;
  if (ts - lastT < 1000 / fps) return;
  lastT = ts;
  sim.step(cfg.fire_rate); step++;   // always the trained fire rate
  const m = readMeans();
  ema = ema ? ema.map((v, i) => v * 0.8 + m[i] * 0.2) : m;
  sim.draw();
  updateLiveDots(); updateLive(); updateOverlay(); updateConnectors(); updateProgress();
  if (step >= horizon) { phase = 'done'; finish(); }
}

// ---- controls ------------------------------------------------------------
els.play.onclick = () => {
  if (phase === 'done') { reseed(); return; }          // "Replay"
  if (phase === 'hold') { phase = 'run'; lastT = 0; els.play.textContent = 'Pause'; return; }
  // toggle run/pause
  if (els.play.textContent === 'Pause') { phase = '__paused'; els.play.textContent = 'Play'; }
  else { phase = 'run'; lastT = 0; els.play.textContent = 'Pause'; }
};
els.again.onclick = () => {
  const wasPaused = els.play.textContent === 'Play';   // don't resume a paused run
  reseed();
  if (wasPaused) { phase = '__paused'; els.play.textContent = 'Play'; }
};
els.rand.onclick = () => {
  inputBits = inputBits.map(() => Math.random() < 0.5 ? 0 : 1);
  buildInputUI(); updateExpr(); layoutConnectors(); reseed();
};
els.fps.oninput = () => { fps = +els.fps.value; els.fpsVal.textContent = fps; };
els.debug.onchange = () => {
  debug = els.debug.checked;
  els.overlay.style.display = debug ? '' : 'none';
  els.conn.style.display = debug ? '' : 'none';   // "mark cells" also shows/hides the wires
};

(async function () {
  els.fpsVal.textContent = fps;
  els.debug.checked = debug;
  if (window.ResizeObserver) new ResizeObserver(() => layoutConnectors()).observe(els.demo);
  window.addEventListener('resize', layoutConnectors);
  const initTask = TASKS[location.hash.slice(1)] ? location.hash.slice(1) : 'adder4';
  try { await selectTask(initTask); raf = requestAnimationFrame(frame); }
  catch (e) { console.error(e); }
})();
