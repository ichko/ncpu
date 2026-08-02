// carry_tsne.js — latent-dimensions viewer. Baked 8-bit adder rollouts
// (scripts/bake_carry_tsne.py); switch examples from the menu. Everything moves with
// the movie scrubber:
//   * a row of KEY hidden-channel spatial maps under the movie (RdBu filmstrips)
//   * the tracked cells in t-SNE space, as trajectories (default) or as a dot cloud
//   * FILTERS that dim the field down to the I/O bits or the compute band
//   * a COLOUR BASIS: what the cell is (role), how it moves (cluster), or where it
//     sits on the grid (position)
//
// Recency is carried by the trail — brightest where the cell just was, fading behind
// it — which is what makes an individual path followable while colour stays free to
// encode something else.
//
// The cell cloud is CANVAS, not SVG: ~135k states is orders of magnitude past what SVG
// elements can animate. Coordinates arrive as a binary blob (uint16 x, uint16 y, int8
// value, frame-major) — see the bake script's header.
// Pure viewer; nothing here touches the live demo (index.html / app.js).

const $ = (id) => document.getElementById(id);
const MOVIE_SCALE = 4, KEY_SCALE = 2, CROP = 6;
const SCAT = 620, PAD = 18;
const PAPER = [236, 225, 199], ACCENT = [171, 53, 32], PURPLE = [59, 15, 79];
// high-contrast, clearly separated hues; role 0 (plain field) stays neutral and quiet
const ROLE_COLOR = { 0: '#9a8f7d', 1: '#1864ab', 2: '#e8590c', 3: '#2f9e44' };
const MOVIE_ROLE_COLOR = { 1: '#4dabf7' };   // brighter blue for the dark movie only
const ROLES = [0, 3, 1, 2];              // draw order: field, band, then the bits on top
const NBUCKET = 24;                      // colour quantisation for the dot cloud
const LONGSEG = 0.08;                    // steps past this share of the plot are "chords"
const TRAIL = 26;                        // frames of decaying trail behind each cell
// movement clusters (KMeans over each cell's whole path — scripts/cluster_carry_cells.py)
const CLUSTER_COLOR = ['#c1121f', '#0353a4', '#2a9d3f', '#e07a00', '#7b2cbf', '#0b8f8f', '#b5179e', '#5c5c5c'];
const POS_NX = 8, POS_NY = 5;            // position colouring: 8 x 5 = 40 grid patches
// The compute band spans the full height of the grid, so one flat green throws away
// where in the column a cell sits. Shade it top to bottom instead.
const BAND_TOP = [150, 214, 90], BAND_BOT = [8, 84, 52], BAND_BINS = 8;
// Time ramp, used ONLY for a selected path: amber early, red middle, purple late.
const SEL_A = [224, 163, 60], SEL_B = [171, 53, 32], SEL_C = [59, 15, 79];
const RECENT_TAIL = 12;                  // frames of fading dots in 'recent only'
const PICK_RADIUS = 14;                  // px within which a click selects a cell
const CLOUD_CACHE_MAX = 6;               // rendered static layers kept as bitmaps
// Backing store is capped at 1.5x device pixels: at 2x a rebuild costs 4x the fill for
// hairlines that look the same, which is what made switching feel slow on retina.
// The static layer is ~287k segments and canvas charges for every one. It is drawn at
// 0.10 alpha as context, so stepping over every other frame is visually identical and
// halves the cost of a rebuild. The trail stays full-fidelity — that one you read.
const STATIC_STEP = 2;

const els = {
  movie: $('cwMovie'), cloud: $('cwCloud'), live: $('cwLive'),
  keymaps: $('cwKeymaps'), scatterLegend: $('cwScatterLegend'), example: $('cwExample'),
  viewTabs: $('cwViewTabs'), focusTabs: $('cwFocusTabs'), colorTabs: $('cwColorTabs'), modeTabs: $('cwModeTabs'),
  play: $('cwPlay'), time: $('cwTime'), speed: $('cwSpeed'),
  status: $('cwStatus'), caption: $('cwCaption'), head: $('cwHead'), perp: $('cwPerp'), busy: $('cwBusy'),
  selInfo: $('cwSelInfo'),
};

// Every control is the same .tab button the article uses, and selection is the same
// inverted-ink .active state — one compact bar instead of stacked rows of radios.
const VIEWS = [
  { v: 'lines', label: 'trajectories', title: "Every tracked cell's whole path through the embedding, coloured by what the cell is. A bright trail marks where each cell has just been and fades behind it, so you can follow one path at a time." },
  { v: 'dots', label: 'dots', title: 'The state cloud: every tracked cell at every frame, from its 14 hidden channels (ch2–15). Colour is the channel-0 value, so you see which cells read 1 and which read 0.' },
];
const FILTERS = [
  { roles: [1, 2], label: 'I/O cells', on: false, title: 'Dim the field and keep only the cells sitting on an input or output bit. The movie marks where they are.' },
  { roles: [3], label: 'compute band', on: false, title: 'Keep only the cells between the input and output columns. Nothing is read in or written out there, so whatever crosses it is the computation itself. The movie shades the band.' },
];
const MODES = [
  { key: 'recent', label: 'recent only', on: false, title: 'Drop the full run and show only what is happening now: the last few frames, fading fast. Nothing competes with the current state, so the active cells read clearly.' },
];
const COLORS = [
  { v: 'role', label: 'by role', title: 'Colour each path by what the cell is: an input bit, an output bit, the band between the columns, or plain field.' },
  { v: 'cluster', label: 'by movement', title: 'Colour by movement: cells are clustered on their whole path through the embedding, so cells that travel together share a colour. Recomputed per example, since the same cell moves differently in 255+1 than in 170+85.' },
  { v: 'pos', label: 'by position', title: 'Colour by where the cell sits on the grid: hue runs left to right (blue at the input column, red at the output column), lightness runs top to bottom. Shows whether cells that are neighbours on the grid stay neighbours in latent space.' },
];
let exTabs = [], viewTabs = [], filterTabs = [], colorTabs = [], modeTabs = [];
function tabBtn(label, title, dots) {
  const b = document.createElement('button');
  b.className = 'tab'; b.title = title;
  for (const d of dots || []) {
    const i = document.createElement('i'); i.className = 'cw-dot'; i.style.background = d; b.appendChild(i);
  }
  b.appendChild(document.createTextNode(label));
  return b;
}
// Click the figure to follow one cell. In path space we pick by the path dot, elsewhere
// by where the cell is at the current frame.
function pickAt(mx, my) {
  const n = cells.n;
  let best = -1, bd = PICK_RADIUS * PICK_RADIUS;
  if (cur) {
    const off = frame * n, { PX, PY } = cur;
    for (let i = 0; i < n; i++) {
      const d = (PX[off + i] - mx) ** 2 + (PY[off + i] - my) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
  }
  return best;
}
function describeSel() {
  if (!els.selInfo) return;
  if (sel < 0) {
    els.selInfo.innerHTML = 'click any dot to follow one cell — its whole path is drawn from '
      + '<span class="cw-swatch" style="background:#e0a33c"></span>start to '
      + '<span class="cw-swatch" style="background:#3b0f4f"></span>end';
    return;
  }
  const NAMES = { 0: 'field', 1: 'input bit', 2: 'output bit', 3: 'compute band' };
  const slot = cells.slot[sel];
  els.selInfo.innerHTML = `following cell (x&nbsp;${cells.x[sel]}, y&nbsp;${cells.y[sel]}) · `
    + `${NAMES[cells.role[sel]]}${slot >= 0 ? ` #${slot}` : ''} · click empty space to release`;
}
function syncTabs() {
  exTabs.forEach((b, i) => b.classList.toggle('active', i === curIdx));
  viewTabs.forEach((b, i) => b.classList.toggle('active', VIEWS[i].v === view));
  filterTabs.forEach((b, i) => b.classList.toggle('active', FILTERS[i].on));
  colorTabs.forEach((b, i) => b.classList.toggle('active', COLORS[i].v === colorBy));
  modeTabs.forEach((b, i) => b.classList.toggle('active', MODES[i].on));
}

let data, cells, T, H, W, cw, ch, kw, kh, dpr = 1;
let frame = 0, playing = true, fps = 18, lastT = 0, raf = 0, holdUntil = 0;
let curIdx = -1, selToken = 0, view = 'lines', colorBy = 'role', busy = false;
let film = null, cur = null;             // current movie Image / current {PX,PY,V}
let keyviews = [];
let roleIdx = {};                        // role -> Int32Array of cell indices
let groups = [];                         // draw units: {idx, color, movieColor, dim}
let curCluster = null;                    // per-example movement clusters
let sel = -1;                            // selected cell index, or -1
let recentOnly = false;                  // show only the recent window, not the whole run
let focus = new Set();                   // roles kept by the filters; empty = keep all
const imgCache = new Map(), binCache = new Map(), cloudCache = new Map();

// ── colour ───────────────────────────────────────────────────────────────────
const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
const lerp = (a, b, k) => a.map((x, i) => Math.round(x + (b[i] - x) * k));
// In the dot view activation is the signal, so the two signs get separate ramps AND
// separate sizes — one shared ramp washes out around zero.
const ACT = Array.from({ length: NBUCKET }, (_, i) => rgb(lerp(PAPER, ACCENT, i / (NBUCKET - 1))));
const INACT = Array.from({ length: NBUCKET }, (_, i) => rgb(lerp(PAPER, PURPLE, i / (NBUCKET - 1))));
const bucket = (v) => Math.min(NBUCKET - 1, Math.round((Math.abs(v) / 127) * (NBUCKET - 1)));
const shown = (role) => focus.size === 0 || focus.has(role);
// grid position -> colour, on a fine 8x5 patchwork. Hue sweeps 265 degrees left to
// right and lightness drops top to bottom, so both grid axes are readable and adjacent
// patches stay far apart. Interpolation happens in HSL, NOT in RGB: blending opposite
// hues in RGB was what turned the middle of the grid into one muddy brown.
const selColor = (u) => (u < 0.5 ? rgb(lerp(SEL_A, SEL_B, u / 0.5)) : rgb(lerp(SEL_B, SEL_C, (u - 0.5) / 0.5)));
const posColor = (bx, by) => {
  const u = (bx + 0.5) / POS_NX, v = (by + 0.5) / POS_NY;
  return `hsl(${Math.round(200 + 265 * u)}, 80%, ${Math.round(60 - 30 * v)}%)`;
};
const paint = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
const loadImg = (src) => new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.src = src; });
function drawFilm(ctx, img, w, h) { ctx.imageSmoothingEnabled = false; ctx.drawImage(img, frame * W + CROP, CROP, W - 2 * CROP, H - 2 * CROP, 0, 0, w, h); }
function ctx2d(cv) { const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, SCAT, SCAT); return c; }
function circle(c, x, y, r) { c.moveTo(x + r, y); c.arc(x, y, r, 0, Math.PI * 2); }

// ── draw units ───────────────────────────────────────────────────────────────
// One list of {indices, colour} per batch, so every draw routine is the same loop no
// matter which colour basis is active. Filtered-out cells become a single dim group.
function buildGroups() {
  const keep = (i) => shown(cells.role[i]);
  const mk = (list, color, movieColor, dim, quiet) => ({ idx: Int32Array.from(list), color, movieColor: movieColor || color, dim: !!dim, quiet: !!quiet });
  groups = [];
  const dimmed = [];
  for (let i = 0; i < cells.n; i++) if (!keep(i)) dimmed.push(i);
  if (dimmed.length) groups.push(mk(dimmed, ROLE_COLOR[0], ROLE_COLOR[0], true));

  if (colorBy === 'cluster' && curCluster) {
    const nk = data.n_clusters || CLUSTER_COLOR.length;
    for (let k = 0; k < nk; k++) {
      const list = [];
      for (let i = 0; i < cells.n; i++) if (curCluster[i] === k && keep(i)) list.push(i);
      if (list.length) groups.push(mk(list, CLUSTER_COLOR[k % CLUSTER_COLOR.length]));
    }
  } else if (colorBy === 'pos') {
    const bins = new Map();
    for (let i = 0; i < cells.n; i++) {
      if (!keep(i)) continue;
      const bx = Math.min(POS_NX - 1, Math.floor((cells.x[i] / W) * POS_NX));
      const by = Math.min(POS_NY - 1, Math.floor((cells.y[i] / H) * POS_NY));
      const k = by * POS_NX + bx;
      if (!bins.has(k)) bins.set(k, []);
      bins.get(k).push(i);
    }
    for (const [k, list] of bins) {
      groups.push(mk(list, posColor(k % POS_NX, (k / POS_NX) | 0)));
    }
  } else {
    for (const r of ROLES) {
      const list = [];
      for (const i of roleIdx[r]) if (keep(i)) list.push(i);
      if (!list.length) continue;
      if (r === 3) {                             // band: one batch per horizontal slice
        const bins = new Map();
        for (const i of list) {
          const k = Math.min(BAND_BINS - 1, Math.floor((cells.y[i] / H) * BAND_BINS));
          if (!bins.has(k)) bins.set(k, []);
          bins.get(k).push(i);
        }
        for (const [k, sub] of bins) {
          const col = rgb(lerp(BAND_TOP, BAND_BOT, (k + 0.5) / BAND_BINS));
          groups.push(mk(sub, col, col));
        }
        continue;
      }
      // field is most of the lattice: draw it, but let the meaningful roles lead
      groups.push(mk(list, ROLE_COLOR[r], MOVIE_ROLE_COLOR[r], false, r === 0));
    }
  }
  // paint order is back-to-front: backdrop, then the quiet majority, then the rest.
  // Canvas has no z-index, so this ordering is the only thing keeping 1380 field cells
  // from covering the bits that matter.
  groups.sort((a, b) => (b.dim - a.dim) || (b.quiet - a.quiet));
}

// ── static layer ─────────────────────────────────────────────────────────────
// Whole paths, one stroke batch per (role, chord-or-not) — kept deliberately faint.
// It is context for the trail, not the thing you read.
function drawTrajectories(c) {
  const { PX, PY } = cur, n = cells.n, lim2 = Math.pow(LONGSEG * (SCAT - 2 * PAD), 2);
  c.lineCap = 'round';
  for (const g of groups) {
    for (const long of [true, false]) {          // chords under real steps
      c.globalAlpha = g.dim ? (long ? 0.006 : 0.022) : g.quiet ? (long ? 0.01 : 0.05) : (long ? 0.02 : 0.10);
      c.lineWidth = g.dim ? 0.5 : 0.6;
      c.strokeStyle = g.color;
      c.beginPath(); let any = false;
      for (let f = STATIC_STEP; f < T; f += STATIC_STEP) {
        for (let k = 0; k < g.idx.length; k++) {
          const a = (f - STATIC_STEP) * n + g.idx[k], b = f * n + g.idx[k];
          const dx = PX[b] - PX[a], dy = PY[b] - PY[a];
          if ((dx * dx + dy * dy > lim2) !== long) continue;
          c.moveTo(PX[a], PY[a]); c.lineTo(PX[b], PY[b]); any = true;
        }
      }
      if (any) c.stroke();
    }
  }
  c.globalAlpha = 1;
}
function drawCloudDots(c) {
  const { PX, PY, V } = cur, N = PX.length, n = cells.n;
  const k = focus.size ? 1.7 : 1;                // a focused subset needs more weight
  for (const [pos, ramp, r, alpha] of [[false, INACT, 0.8 * k, 0.10 * k], [true, ACT, 1.1 * k, 0.22 * k]]) {
    c.globalAlpha = alpha;
    for (let b = 0; b < NBUCKET; b++) {
      c.beginPath(); let any = false;
      for (let i = 0; i < N; i++) {
        if ((V[i] > 0) !== pos || bucket(V[i]) !== b || !shown(cells.role[i % n])) continue;
        circle(c, PX[i], PY[i], r); any = true;
      }
      if (any) { c.fillStyle = ramp[b]; c.fill(); }
    }
  }
  if (focus.size) {                              // the rest: present but quiet
    c.globalAlpha = 0.03; c.beginPath();
    for (let i = 0; i < N; i++) if (!shown(cells.role[i % n])) circle(c, PX[i], PY[i], 0.7);
    c.fillStyle = ROLE_COLOR[0]; c.fill();
  }
  c.globalAlpha = 1;
}
// The static layer costs a few hundred ms at this cell count, so keep rendered copies:
// flipping a filter or returning to an example is then instant.
const cloudKey = () => `${data.examples[curIdx].id}|${view}|${colorBy}|${recentOnly ? 'r' : 'a'}|${[...focus].sort().join(',')}`;
function drawCloud() {
  const key = cloudKey(), hit = cloudCache.get(key);
  const c = ctx2d(els.cloud);
  if (hit) { c.drawImage(hit, 0, 0, SCAT, SCAT); return Promise.resolve(); }
  // 'recent only' means there IS no background layer: the whole point is that nothing
  // sits behind the current state competing with it.
  if (recentOnly) return Promise.resolve();
  if (view === 'lines') drawTrajectories(c);
  else drawCloudDots(c);
  // Snapshotting is a GPU readback and costs as much as the render itself, so it is
  // deliberately NOT awaited: the layer is already on screen: the bitmap only makes the
  // NEXT visit to this key instant.
  createImageBitmap(els.cloud).then((bmp) => {
    cloudCache.set(key, bmp);
    while (cloudCache.size > CLOUD_CACHE_MAX) {
      const k0 = cloudCache.keys().next().value;
      cloudCache.get(k0).close?.(); cloudCache.delete(k0);
    }
  }).catch(() => {});                            // the cache is an optimisation only
  return Promise.resolve();
}

// ── live layer (redrawn every frame) ─────────────────────────────────────────
// The trail IS the visualisation: brightest at the step just taken, decaying back over
// TRAIL frames, so each path reads as a moving comet rather than a static tangle.
function drawTrail(c) {
  const { PX, PY } = cur, n = cells.n, lim2 = Math.pow(LONGSEG * (SCAT - 2 * PAD), 2);
  c.lineCap = 'round';
  for (let w = 0; w < TRAIL && frame - w > 0; w++) {
    const decay = Math.pow(1 - w / TRAIL, recentOnly ? 3.4 : 2.2);
    for (const g of groups) {
      if (g.dim) continue;                       // dim groups stay in the static layer
      c.strokeStyle = g.color;
      const b = (frame - w) * n, a = b - n;
      // A chord is a seam in the embedding, and full-strength chords are all you see:
      // they are long, straight and cross everything. Keep them, but barely.
      for (const long of [true, false]) {
        c.globalAlpha = (g.quiet ? 0.22 : 0.95) * decay * (long ? 0.13 : 1);
        c.lineWidth = 2.1 * (g.quiet ? 0.6 : 1) * (0.35 + 0.65 * decay) * (long ? 0.6 : 1);
        c.beginPath(); let any = false;
        for (let k = 0; k < g.idx.length; k++) {
          const i = g.idx[k], dx = PX[b + i] - PX[a + i], dy = PY[b + i] - PY[a + i];
          if ((dx * dx + dy * dy > lim2) !== long) continue;
          c.moveTo(PX[a + i], PY[a + i]); c.lineTo(PX[b + i], PY[b + i]); any = true;
        }
        if (any) c.stroke();
      }
    }
  }
  c.globalAlpha = 1;
}
// A selected cell gets its WHOLE path drawn, coloured early-to-late, and every other
// cell's trail is suppressed. One path against faint context is the only way to
// actually follow a route through a map this dense.
function drawSelectedPath(c) {
  const { PX, PY } = cur, n = cells.n;
  const px = (f) => PX[f * n + sel], py = (f) => PY[f * n + sel];
  c.lineCap = 'round';
  c.lineWidth = 2.2; c.globalAlpha = 0.92;
  for (let f = 1; f < T; f++) {
    c.strokeStyle = selColor((f - 1) / (T - 2));
    c.beginPath(); c.moveTo(px(f - 1), py(f - 1)); c.lineTo(px(f), py(f)); c.stroke();
  }
  c.globalAlpha = 1;
  c.beginPath(); circle(c, px(0), py(0), 4.2);                // where the run begins
  c.lineWidth = 1.6; c.strokeStyle = rgb(SEL_A); c.stroke();
  c.beginPath(); circle(c, px(T - 1), py(T - 1), 3.4);
  c.fillStyle = rgb(SEL_C); c.fill();                         // where it ends
  c.beginPath(); circle(c, px(frame), py(frame), 5.2);        // where it is right now
  c.fillStyle = '#f5d90a'; c.fill();
  c.lineWidth = 1.4; c.strokeStyle = '#221f19'; c.stroke();
}

function drawLive() {
  const c = ctx2d(els.live), { PX, PY, V } = cur, n = cells.n, off = frame * n;
  if (view === 'lines' && sel >= 0) { drawSelectedPath(c); return; }
  if (view === 'lines') {
    drawTrail(c);
    // head dot: role colour, size carries whether the cell reads 1 or 0. No outline.
    for (const g of groups) {
      if (g.dim) continue;
      c.fillStyle = g.color;
      for (const pos of [false, true]) {
        c.globalAlpha = (g.quiet ? 0.34 : 1) * (pos ? 1 : 0.55);
        const rr = (pos ? 2.7 : 1.4) * (g.quiet ? 0.72 : 1);
        c.beginPath(); let any = false;
        for (let k = 0; k < g.idx.length; k++) {
          const i = g.idx[k];
          if ((V[off + i] > 0) !== pos) continue;
          circle(c, PX[off + i], PY[off + i], rr); any = true;
        }
        if (any) c.fill();
      }
    }
  } else {
    if (recentOnly) {                            // a short fading tail stands in for the cloud
      for (let w = RECENT_TAIL; w >= 1; w--) {
        if (frame - w < 0) continue;
        const o = (frame - w) * n, k = Math.pow(1 - w / (RECENT_TAIL + 1), 2.4);
        for (const pos of [false, true]) {
          c.globalAlpha = (pos ? 0.85 : 0.45) * k;
          c.beginPath(); let any = false;
          for (let i = 0; i < n; i++) {
            if ((V[o + i] > 0) !== pos || !shown(cells.role[i])) continue;
            circle(c, PX[o + i], PY[o + i], (pos ? 2.4 : 1.3) * (0.4 + 0.6 * k)); any = true;
          }
          if (any) { c.fillStyle = (pos ? ACT : INACT)[NBUCKET - 1]; c.fill(); }
        }
      }
      c.globalAlpha = 1;
    }
    for (const pos of [false, true]) {           // activation ramp, no outline
      const r = pos ? 3.0 : 1.7;
      c.globalAlpha = pos ? 1 : 0.6;
      for (let b = 0; b < NBUCKET; b++) {
        c.beginPath(); let any = false;
        for (let i = 0; i < n; i++) {
          const v = V[off + i];
          if ((v > 0) !== pos || bucket(v) !== b || !shown(cells.role[i])) continue;
          circle(c, PX[off + i], PY[off + i], r); any = true;
        }
        if (any) { c.fillStyle = (pos ? ACT : INACT)[b]; c.fill(); }
      }
    }
    if (focus.size) {                            // rings name the highlighted group
      c.globalAlpha = 0.95; c.lineWidth = 1.4;
      for (const role of focus) {
        const idx = roleIdx[role]; if (!idx.length) continue;
        c.beginPath();
        for (let k = 0; k < idx.length; k++) circle(c, PX[off + idx[k]], PY[off + idx[k]], 4.4);
        c.strokeStyle = ROLE_COLOR[role]; c.stroke();
      }
    }
  }
  c.globalAlpha = 1;
}

// With a highlight on, mark the highlighted cells ON THE MOVIE too, so it is obvious
// where in the grid the emphasised paths are being read from. The movie is drawn
// cropped by CROP px, so grid (x,y) lands at ((x-CROP)*S, (y-CROP)*S).
function drawMovieMarks(ctx) {
  if (!focus.size && colorBy === 'role') return;
  const S = MOVIE_SCALE;
  if (focus.has(3)) {                             // the band is a region, so shade it
    let lo = Infinity, hi = -Infinity;
    for (const i of roleIdx[3]) { lo = Math.min(lo, cells.x[i]); hi = Math.max(hi, cells.x[i]); }
    const x0 = (lo - 0.5 - CROP) * S, x1 = (hi + 0.5 - CROP) * S;
    ctx.globalAlpha = 0.16; ctx.fillStyle = ROLE_COLOR[3];
    ctx.fillRect(x0, 0, x1 - x0, ch);
    ctx.globalAlpha = 0.9; ctx.strokeStyle = ROLE_COLOR[3]; ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]); ctx.beginPath();
    ctx.moveTo(x0, 0); ctx.lineTo(x0, ch); ctx.moveTo(x1, 0); ctx.lineTo(x1, ch);
    ctx.stroke(); ctx.setLineDash([]);
  }
  // small centred pips, not full cells: at stride 3 a cell-sized mark tiles solid and
  // hides the simulation underneath, which is the thing you came to look at
  // movieColor differs from color where needed: the movie background is dark viridis,
  // where the deep input blue disappears, while the plot background is cream, where a
  // light blue would.
  ctx.globalAlpha = 0.85;
  for (const g of groups) {
    if (g.dim) continue;
    ctx.fillStyle = g.movieColor;
    for (let k = 0; k < g.idx.length; k++) {
      const i = g.idx[k], x = (cells.x[i] - CROP) * S, y = (cells.y[i] - CROP) * S;
      if (x < 0 || y < 0 || x >= cw || y >= ch) continue;
      ctx.fillRect(x + S / 2 - 1, y + S / 2 - 1, 2, 2);
    }
  }
  ctx.globalAlpha = 1;
}

// Keep this SHORT: .cw-status is nowrap, so a long string widens the movie column
// and wraps the figure onto its own row.
function updateStatus() {
  els.status.innerHTML = `frame <b>${String(frame).padStart(3, '0')}</b> / ${T - 1} &nbsp;·&nbsp; ${cells.n} cells`;
}

// Anything that rebuilds the static layer shows the spinner and yields a frame first,
// otherwise the browser never gets to paint it before the blocking redraw begins.
async function restyle() {
  if (!cur) return;
  syncTabs();
  const cached = cloudCache.has(cloudKey());
  if (!cached) { busy = true; els.busy.classList.add('on'); await paint(); }
  buildGroups();
  await drawCloud();
  render();
  busy = false; els.busy.classList.remove('on');
}

function render() {
  if (film) {
    const mc = els.movie.getContext('2d');
    drawFilm(mc, film, cw, ch);
    drawMovieMarks(mc);
  }
  for (const k of keyviews) if (k.img) drawFilm(k.ctx, k.img, kw, kh);
  if (cur) drawLive();
  updateStatus();
  els.time.value = String(frame);
}

function loop(ts) {
  raf = requestAnimationFrame(loop);
  if (holdUntil === 0) holdUntil = ts + 500;   // half-second pause on first load
  if (busy || !playing) return;                // never animate over a load or a rebuild
  if (ts < holdUntil) return;                  // hold (start of load, and end of each cycle)
  if (ts - lastT < 1000 / fps) return;
  lastT = ts;
  frame = (frame + 1) % T;
  render();
  if (frame === T - 1) holdUntil = ts + 500;   // pause on the final frame before looping
}

async function loadExample(ex) {
  if (!imgCache.has(ex.id)) {
    const [movie, ...keys] = await Promise.all([loadImg(ex.film), ...ex.keymaps.map((k) => loadImg(k.film))]);
    imgCache.set(ex.id, { movie, keys });
  }
  if (!binCache.has(ex.id)) {
    const buf = await (await fetch(ex.cellbin)).arrayBuffer();
    const N = cells.n * T, span = SCAT - 2 * PAD;
    const X = new Uint16Array(buf, 0, N), Y = new Uint16Array(buf, 2 * N, N);
    const PX = new Float32Array(N), PY = new Float32Array(N);
    for (let i = 0; i < N; i++) { PX[i] = PAD + (X[i] / 65535) * span; PY[i] = PAD + (1 - Y[i] / 65535) * span; }
    binCache.set(ex.id, { PX, PY, V: new Int8Array(buf, 4 * N, N) });
  }
  return { imgs: imgCache.get(ex.id), bin: binCache.get(ex.id) };
}

async function selectExample(i) {
  if (i === curIdx) return;
  const token = ++selToken;
  const ex = data.examples[i];
  busy = true; els.busy.classList.add('on');
  await paint();
  const { imgs, bin } = await loadExample(ex);
  if (token !== selToken) return;                 // a newer click won
  curIdx = i;
  film = imgs.movie; cur = bin; curCluster = ex.cluster || null;
  describeSel();
  keyviews.forEach((kv, k) => { kv.img = imgs.keys[k]; kv.capEl.textContent = `ch${ex.keymaps[k].ch} · tvar ${ex.keymaps[k].tvar}`; });
  els.head.textContent = `${ex.a} + ${ex.b} = ${ex.decoded}` + (ex.decoded === ex.sum ? '' : ` (want ${ex.sum})`);
  els.caption.textContent =
    `${ex.a} + ${ex.b} on the 8-bit adder NCA. It decoded ${ex.decoded}. Under the movie are the ${ex.keymaps.length} most ` +
    `dynamic hidden channels as spatial maps (red +, blue −). The vertical "spine" is the carry chain up the operand column. ` +
    `On the right we follow ${cells.n} cells — a regular grid every ${cells.stride} px plus the centre of every input and ` +
    `output bit — through all ${T} frames, reading their 14 hidden channels (ch2–15). All ${ex.n_points.toLocaleString()} ` +
    `states go into one t-SNE at perplexity ${data.perplexity}, so a cell's position is comparable across frames and it ` +
    `traces a path. Paths are coloured by what the cell is: input bit, output bit, the band between the columns where the ` +
    `computation has to happen, or plain field. The bright trail marks where each cell has just been and decays behind it.`;
  frame = 0; lastT = 0; holdUntil = 0;            // restart from the beginning (with the hold)
  playing = true; els.play.textContent = 'Pause';
  await restyle();
}

async function main() {
  data = await (await fetch('data/carry_tsne.json')).json();
  T = data.T; H = data.H; W = data.W; cells = data.cells;
  dpr = Math.min(window.devicePixelRatio || 1, 1.5);   // see MAX_DPR note below
  // group cell indices by role once: the draw loops run per role every frame
  for (const role of ROLES) roleIdx[role] = Int32Array.from(cells.role.flatMap((v, i) => (v === role ? [i] : [])));
  cw = (W - 2 * CROP) * MOVIE_SCALE; ch = (H - 2 * CROP) * MOVIE_SCALE;
  kw = (W - 2 * CROP) * KEY_SCALE; kh = (H - 2 * CROP) * KEY_SCALE;
  els.movie.width = cw; els.movie.height = ch; els.movie.style.width = cw + 'px'; els.movie.style.height = ch + 'px';
  // pin the movie column to the movie itself: the nowrap status line under it would
  // otherwise set the column width and wrap the figure to the next row
  els.movie.closest('.cw-col').style.width = cw + 2 + 'px';
  for (const cv of [els.cloud, els.live]) {
    cv.width = Math.round(SCAT * dpr); cv.height = Math.round(SCAT * dpr);
    cv.style.width = SCAT + 'px'; cv.style.height = SCAT + 'px';
  }
  els.scatterLegend.style.width = SCAT + 'px';
  els.time.max = String(T - 1);
  if (els.perp) els.perp.textContent = String(data.perplexity ?? '');

  exTabs = data.examples.map((ex, i) => {
    const b = tabBtn(ex.label, `${ex.a} + ${ex.b} = ${ex.sum}`);
    b.onclick = () => selectExample(i);
    els.example.appendChild(b);
    return b;
  });
  viewTabs = VIEWS.map((v) => {
    const b = tabBtn(v.label, v.title);
    b.onclick = () => { view = v.v; restyle(); };
    els.viewTabs.appendChild(b);
    return b;
  });
  filterTabs = FILTERS.map((f) => {
    const b = tabBtn(f.label, f.title, f.roles.map((r) => ROLE_COLOR[r]));
    b.onclick = () => {
      f.on = !f.on;
      focus = new Set();
      for (const g of FILTERS) if (g.on) for (const r of g.roles) focus.add(r);
      restyle();
    };
    els.focusTabs.appendChild(b);
    return b;
  });
  modeTabs = MODES.map((m) => {
    const b = tabBtn(m.label, m.title);
    b.onclick = () => { m.on = !m.on; recentOnly = MODES[0].on; restyle(); };
    els.modeTabs.appendChild(b);
    return b;
  });
  colorTabs = COLORS.map((cb) => {
    const b = tabBtn(cb.label, cb.title);
    b.onclick = () => { colorBy = cb.v; restyle(); };
    els.colorTabs.appendChild(b);
    return b;
  });
  data.keymap_channels.forEach((cch) => {
    const fig = document.createElement('figure'); fig.className = 'cw-km';
    const cv = document.createElement('canvas'); cv.width = kw; cv.height = kh;   // CSS scales display
    fig.appendChild(cv);
    const cap = document.createElement('figcaption'); cap.textContent = `ch${cch}`; fig.appendChild(cap);
    els.keymaps.appendChild(fig);
    keyviews.push({ ch: cch, ctx: cv.getContext('2d'), capEl: cap, img: null });
  });

  els.live.style.cursor = 'pointer';
  els.live.onclick = (e) => {
    const r = els.live.getBoundingClientRect();
    sel = pickAt(e.clientX - r.left, e.clientY - r.top);
    describeSel(); render();
  };

  await selectExample(0);
  raf = requestAnimationFrame(loop);
}

// ---- controls ----
els.play.onclick = () => { playing = !playing; els.play.textContent = playing ? 'Pause' : 'Play'; lastT = 0; };
els.time.oninput = () => { playing = false; els.play.textContent = 'Play'; frame = +els.time.value; render(); };
els.speed.oninput = () => { fps = +els.speed.value; };

main();
