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
  canvas: $('field'), overlay: $('ioOverlay'), screen: $('screen'),
  demo: $('demo'), conn: $('connSvg'),
  tabs: $('taskTabs'), inputs: $('inputBits'), outputs: $('outputReadout'),
  expr: $('exprLine'), verdict: $('verdict'),
  bar: $('pbar'), pstep: $('pstep'), phorizon: $('phorizon'),
  play: $('playBtn'), again: $('againBtn'), rand: $('randBtn'),
  fps: $('fpsSlider'), fpsVal: $('fpsVal'),
  fire: $('fireSlider'), fireVal: $('fireVal'),
  debug: $('debugToggle'), runName: $('runName'), status: $('simStatus'),
};

let sim = null, cfg = null, geo = null, masks = null, taskKey = null;
let inputBits = [], step = 0, horizon = 64;
let phase = 'hold';                 // 'hold' (show input) | 'run' | 'done'
let holdUntil = 0;
let ema = null, expArr = [];        // per-output-bit moving average + expected bits
let fps = 20, fireRate = 0.5, debug = true;
let raf = null, lastT = 0;
const cache = {};
const HOLD_MS = 1000;               // pause on the encoded input before running

// overlay markers: monochrome white with opacity, red only to flag a wrong bit
const ERR = '#ff5a3c';
let inputRings = [], outputRings = [];
let inputBtns = [], inPaths = [], outPaths = [];
const TRACE = '#4a443b', TRACE_ERR = '#ab3520';
function setRing(el, stroke, op, w, fillOp) {
  el.setAttribute('stroke', stroke);
  el.setAttribute('stroke-opacity', op);
  el.setAttribute('stroke-width', w);
  el.setAttribute('fill', stroke === '#fff' ? `rgba(255,255,255,${fillOp})` : `rgba(255,90,60,${fillOp})`);
}

const loadModel = async (k) => (cache[k] ||= await fetch(`data/${k}.json`).then((r) => r.json()));
const intBits = (v, w) => { const o = []; for (let i = w - 1; i >= 0; i--) o.push((v >> i) & 1); return o; };
const bitsInt = (a) => a.reduce((x, b) => x * 2 + b, 0);
const nInputs = (k) => TASKS[k].group === 'gate' ? (k === 'majority3' ? 3 : 2) : TASKS[k].bits * 2;

// display scale: keep the grid modest on screen (~300px on its long side)
function scaleFor(W, H) { return Math.max(2, Math.round(300 / Math.max(W, H))); }

async function selectTask(key) {
  taskKey = key;
  cfg = await loadModel(key);
  geo = geometry(cfg);
  masks = outputMasks(cfg, geo);
  horizon = TASKS[key].horizon;

  const s = scaleFor(cfg.W, cfg.H);
  els.canvas.style.width = cfg.W * s + 'px';
  els.canvas.style.height = cfg.H * s + 'px';
  // reserve gutters for the wire ribbon so traces route around the plate rather
  // than over it: a top band (one lane per input) and a left gutter (the input
  // circles sit on the grid's far edge, opposite the control panel).
  const band = 7 + geo.inputs.length * 3;   // one thin 3px lane per input wire
  els.demo.style.paddingTop = band + 'px';
  const stageEl = els.demo.querySelector('.stage');
  if (stageEl) stageEl.style.marginLeft = band + 'px';

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
  els.runName.textContent = cfg.run;
  buildTabs(); buildInputUI(); drawOverlay();
  restart();
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

function buildInputUI() {
  els.inputs.innerHTML = '';
  inputBtns = [];
  const t = TASKS[taskKey];
  if (t.group === 'adder') {
    els.inputs.appendChild(operandRow('A', 0, t.bits));
    els.inputs.appendChild(operandRow('B', t.bits, t.bits));
  } else {
    const names = taskKey === 'majority3' ? ['A', 'B', 'C'] : ['A', 'B'];
    const row = document.createElement('div'); row.className = 'bit-row';
    names.forEach((nm, i) => row.appendChild(bitToggle(i)));
    els.inputs.appendChild(row);
  }
  updateExpr();
}
function operandRow(name, off, w) {
  const row = document.createElement('div'); row.className = 'bit-row';
  const tag = document.createElement('span'); tag.className = 'operand-tag'; tag.textContent = name; row.appendChild(tag);
  for (let i = 0; i < w; i++) row.appendChild(bitToggle(off + i));
  const val = document.createElement('span'); val.className = 'operand-val'; val.dataset.for = name; row.appendChild(val);
  return row;
}
function bitToggle(idx) {
  const b = document.createElement('button');
  b.className = 'bit' + (inputBits[idx] ? ' on' : '');
  b.textContent = inputBits[idx] ? '1' : '0';
  b.onclick = () => {
    inputBits[idx] ^= 1;
    b.classList.toggle('on', !!inputBits[idx]); b.textContent = inputBits[idx] ? '1' : '0';
    updateExpr(); restart();
  };
  inputBtns[idx] = b;
  return b;
}
function updateExpr() {
  const t = TASKS[taskKey];
  if (t.group === 'adder') {
    const b = t.bits, a = bitsInt(inputBits.slice(0, b)), bb = bitsInt(inputBits.slice(b, 2 * b));
    els.expr.textContent = `${a} + ${bb} = ${a + bb}`;
    els.inputs.querySelectorAll('.operand-val').forEach((s) => { s.textContent = s.dataset.for === 'A' ? a : bb; });
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

// ---- overlay: rings that react to state (input activation, output validity)
function drawOverlay() {
  const { W, H } = cfg, ns = 'http://www.w3.org/2000/svg';
  els.overlay.setAttribute('viewBox', `0 0 ${W} ${H}`);
  els.overlay.innerHTML = '';
  els.overlay.style.display = debug ? '' : 'none';
  const ring = (c) => {
    const el = document.createElementNS(ns, 'circle');
    el.setAttribute('cx', c.x + 0.5); el.setAttribute('cy', c.y + 0.5);
    el.setAttribute('r', geo.r + 1.5); el.setAttribute('fill', 'none');
    els.overlay.appendChild(el);
    return el;
  };
  inputRings = geo.inputs.map(ring);
  outputRings = geo.outputs.map(ring);
  updateOverlay();
}

function updateOverlay() {
  inputRings.forEach((el, i) => {
    const on = !!inputBits[i];
    setRing(el, '#fff', on ? 0.95 : 0.32, on ? 1.1 : 0.7, on ? 0.14 : 0);
  });
  outputRings.forEach((el, i) => {
    if (phase === 'hold' || !ema) { setRing(el, '#fff', 0.4, 0.8, 0); return; }
    const got = ema[i] > 0, ok = (got ? 1 : 0) === expArr[i];
    if (!ok) setRing(el, ERR, 0.95, 1.2, got ? 0.18 : 0.05);        // wrong bit -> red
    else setRing(el, '#fff', got ? 0.95 : 0.4, got ? 1.2 : 0.8, got ? 0.16 : 0);
  });
}

// ---- circuit-style connector traces between controls and grid circles ----
function drawConnectors() {
  const ns = 'http://www.w3.org/2000/svg';
  els.conn.innerHTML = '';
  const mk = () => { const p = document.createElementNS(ns, 'path'); els.conn.appendChild(p); return p; };
  inPaths = geo.inputs.map(mk);
  outPaths = geo.outputs.map(mk);
  requestAnimationFrame(layoutConnectors);
}

function layoutConnectors() {
  if (!cfg || !inPaths.length) return;
  const dRect = els.demo.getBoundingClientRect();
  const cRect = els.canvas.getBoundingClientRect();
  // hide when the panel has wrapped below the grid (narrow screens)
  const btn0 = inputBtns[0] && inputBtns[0].getBoundingClientRect();
  const stacked = !cRect.width || (btn0 && btn0.top > cRect.bottom - 4);
  if (stacked) { els.conn.style.display = 'none'; return; }
  els.conn.style.display = '';
  els.conn.setAttribute('viewBox', `0 0 ${dRect.width} ${dRect.height}`);
  els.conn.setAttribute('width', dRect.width); els.conn.setAttribute('height', dRect.height);
  const sc = cRect.width / cfg.W;
  const gL = cRect.left - dRect.left, gT = cRect.top - dRect.top;
  const gR = cRect.right - dRect.left, gB = cRect.bottom - dRect.top;
  const LX = (x) => gL + (x + 0.5) * sc, LY = (y) => gT + (y + 0.5) * sc;

  const scr = els.screen.getBoundingClientRect();     // outer plate boundary
  const sL = scr.left - dRect.left, sR = scr.right - dRect.left, sT = scr.top - dRect.top;
  // Grid is on the LEFT, controls on the RIGHT. Traces live only in the empty
  // gutters around the plate — never over the simulation and never over another
  // control. They meet the plate at its edge (aligned with each circle's row)
  // and meet a control at its edge. Each wire keeps its own thin 3px lane.
  //
  // inputs sit on the plate's FAR (left) edge, so each trace leaves its bit
  // button's top edge, rises into its own lane in the top band, crosses left,
  // drops down its own lane in the left gutter, and meets the left plate edge.
  const N = geo.inputs.length;
  geo.inputs.forEach((c, i) => {
    const btn = inputBtns[c.bit]; if (!btn) return;
    const r = btn.getBoundingClientRect();
    const bx = r.left + r.width / 2 - dRect.left;     // meet the button at its top edge
    const bTop = r.top - dRect.top;
    const cy = LY(c.y);
    const topBus = sT - 5 - (N - 1 - i) * 3;
    const leftBus = sL - 4 - i * 3;
    inPaths[i].setAttribute('d', `M ${bx} ${bTop} V ${topBus} H ${leftBus} V ${cy} H ${sL}`);
  });
  // outputs sit on the plate's NEAR (right) edge: each trace leaves the right
  // plate edge, runs out into its own lane in the gutter, and meets its readout
  // dot at the dot's left edge.
  geo.outputs.forEach((c, i) => {
    const dot = outCells[i] && outCells[i].dot; if (!dot) return;
    const r = dot.getBoundingClientRect();
    const dLeft = r.left - dRect.left, dy = r.top + r.height / 2 - dRect.top;
    const cy = LY(c.y);
    const rightBus = sR + 4 + i * 3;
    outPaths[i].setAttribute('d', `M ${sR} ${cy} H ${rightBus} V ${dy} H ${dLeft}`);
  });
  updateConnectors();
}

function updateConnectors() {
  inPaths.forEach((p, i) => {
    const on = !!inputBits[geo.inputs[i].bit];
    p.setAttribute('stroke', TRACE);
    p.setAttribute('stroke-opacity', on ? 0.8 : 0.26);
    p.setAttribute('stroke-width', on ? 2.2 : 1);
  });
  outPaths.forEach((p, i) => {
    const got = ema && ema[i] > 0;
    const settled = phase === 'run' || phase === 'done' || phase === '__paused';
    const bad = settled && ema && (got ? 1 : 0) !== expArr[i];
    p.setAttribute('stroke', bad ? TRACE_ERR : TRACE);
    p.setAttribute('stroke-opacity', got ? 0.8 : 0.26);
    p.setAttribute('stroke-width', got ? 2.2 : 1);
  });
}

let outCells = [];
function buildOutputUI() {
  els.outputs.innerHTML = '';
  const exp = expected();
  outCells = exp.map((e, i) => {
    const cell = document.createElement('div'); cell.className = 'out-bit';
    const dot = document.createElement('div'); dot.className = 'out-dot';
    const lbl = document.createElement('span'); lbl.className = 'out-lbl'; lbl.textContent = bitLabel(i);
    cell.append(dot, lbl); els.outputs.appendChild(cell);
    return { dot, exp: e };
  });
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

function restart() {
  if (!sim) return;
  sim.seed(buildImage(cfg, geo, inputBits));
  sim.draw();
  step = 0; ema = null; lastT = 0;
  phase = 'hold'; holdUntil = performance.now() + HOLD_MS;
  expArr = expected();
  buildOutputUI();
  drawConnectors();
  updateOverlay();
  els.play.textContent = 'Pause';
  els.verdict.className = 'verdict run';
  els.verdict.innerHTML = '<span class="mk">▸</span> encoded input…';
  updateProgress();
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
  const t = TASKS[taskKey], gs = got.join('');
  let expr;
  if (t.group === 'adder') {
    const b = t.bits, a = bitsInt(inputBits.slice(0, b)), bb = bitsInt(inputBits.slice(b, 2 * b));
    expr = `${a} + ${bb} = ${bitsInt(got)}` + (ok ? '' : `  (want ${a + bb})`);
  } else {
    expr = `read ${gs}` + (ok ? '' : `  · want ${exp.join('')}`);
  }
  els.verdict.className = 'verdict ' + (ok ? 'pass' : 'fail');
  els.verdict.innerHTML = `<span class="mk">${ok ? '✓' : '✗'}</span> ${expr}`;
  els.play.textContent = 'Replay';
  updateOverlay(); updateConnectors();
}

function updateProgress() {
  els.bar.style.width = Math.min(100, (step / horizon) * 100) + '%';
  els.pstep.textContent = String(step).padStart(3, '0');
  els.phorizon.textContent = String(horizon).padStart(3, '0');
}

function frame(ts) {
  raf = requestAnimationFrame(frame);
  if (!sim) return;
  if (phase === 'hold') { if (ts >= holdUntil) { phase = 'run'; lastT = 0; els.verdict.innerHTML = '<span class="mk">▸</span> running…'; } return; }
  if (phase !== 'run') return;
  if (ts - lastT < 1000 / fps) return;
  lastT = ts;
  sim.step(fireRate); step++;
  const m = readMeans();
  ema = ema ? ema.map((v, i) => v * 0.8 + m[i] * 0.2) : m;
  sim.draw();
  updateLiveDots(); updateOverlay(); updateConnectors(); updateProgress();
  if (step >= horizon) { phase = 'done'; finish(); }
}

// ---- controls ------------------------------------------------------------
els.play.onclick = () => {
  if (phase === 'done') { restart(); return; }        // "Replay"
  if (phase === 'hold') { phase = 'run'; lastT = 0; els.play.textContent = 'Pause'; return; }
  // toggle run/pause
  if (els.play.textContent === 'Pause') { phase = '__paused'; els.play.textContent = 'Play'; }
  else { phase = 'run'; lastT = 0; els.play.textContent = 'Pause'; }
};
els.again.onclick = () => {
  const wasPaused = els.play.textContent === 'Play';   // don't resume a paused run
  restart();
  if (wasPaused) {
    phase = '__paused'; els.play.textContent = 'Play';
    els.verdict.className = 'verdict run';
    els.verdict.innerHTML = '<span class="mk">▮</span> paused — encoded input';
  }
};
els.rand.onclick = () => { inputBits = inputBits.map(() => Math.random() < 0.5 ? 0 : 1); buildInputUI(); restart(); };
els.fps.oninput = () => { fps = +els.fps.value; els.fpsVal.textContent = fps; };
els.fire.oninput = () => { fireRate = +els.fire.value / 100; els.fireVal.textContent = fireRate.toFixed(2).slice(1); };
els.debug.onchange = () => { debug = els.debug.checked; els.overlay.style.display = debug ? '' : 'none'; };

(async function () {
  els.fpsVal.textContent = fps;
  els.fireVal.textContent = fireRate.toFixed(2).slice(1);
  els.debug.checked = debug;
  if (window.ResizeObserver) new ResizeObserver(() => layoutConnectors()).observe(els.demo);
  window.addEventListener('resize', layoutConnectors);
  try { await selectTask('adder4'); raf = requestAnimationFrame(frame); }
  catch (e) { console.error(e); }
})();
