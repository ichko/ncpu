// carry_tsne.js — carry-wave analysis viewer. Baked 8-bit adder rollouts
// (scripts/bake_carry_tsne.py); switch examples from the tabs. Each drives three
// views in sync with the movie scrubber:
//   1. a row of KEY hidden-channel spatial maps under the movie (RdBu filmstrips)
//   2. a t-SNE map of every active cell's hidden-channel state (frame-highlighted)
//   3. every channel drawn as a path in ONE shared t-SNE space
// Pure viewer; nothing here touches the live demo (index.html / app.js).

const $ = (id) => document.getElementById(id);
const SVGNS = 'http://www.w3.org/2000/svg';
const MOVIE_SCALE = 4, KEY_SCALE = 2, CROP = 6;
const SCAT = 400, OVERLAY = 400, PAD = 16;
const R_BASE = 2.0, R_HI = 4.4;

const els = {
  movie: $('cwMovie'), scatter: $('cwScatter'), overlay: $('cwOverlay'),
  keymaps: $('cwKeymaps'), legend: $('cwColorLegend'), scatterLegend: $('cwScatterLegend'), example: $('cwExample'),
  play: $('cwPlay'), time: $('cwTime'), speed: $('cwSpeed'), fps: $('cwFps'),
  status: $('cwStatus'), caption: $('cwCaption'), head: $('cwHead'), perp: $('cwPerp'),
};

let data, T, H, W, cw, ch, kw, kh;
let frame = 0, playing = true, fps = 18, lastT = 0, raf = 0, holdUntil = 0;
let curIdx = -1, selToken = 0;
let film = null;                        // current movie Image
let circles = [], byFrame = [], prevHi = [];
let keyviews = [];                      // {ch, ctx, capEl, img}
let ovMarks = [];
const cache = new Map();                // eid -> {movie:Image, keys:[Image]}

function valColor(v) {                   // scatter: purple(neg) -> paper(0) -> accent(pos)
  v = Math.max(-1, Math.min(1, v));
  const t = (v + 1) / 2, lo = [59, 15, 79], mid = [220, 210, 186], hi = [171, 53, 32];
  let c;
  if (t < 0.5) { const k = t / 0.5; c = lo.map((x, i) => Math.round(x + (mid[i] - x) * k)); }
  else { const k = (t - 0.5) / 0.5; c = mid.map((x, i) => Math.round(x + (hi[i] - x) * k)); }
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function chColor(cinfo) {                 // overlay: distinct per-channel colour
  if (cinfo.role.startsWith('input')) return '#9a9a9a';
  return `hsl(${Math.round((cinfo.ch * 360) / 16) % 360}, 60%, 45%)`;
}
function mk(tag, attrs) { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
function loadImg(src) { return new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.src = src; }); }
function drawFilm(ctx, img, w, h) { ctx.imageSmoothingEnabled = false; ctx.drawImage(img, frame * W + CROP, CROP, W - 2 * CROP, H - 2 * CROP, 0, 0, w, h); }

function buildScatter(points) {
  clear(els.scatter);
  els.scatter.setAttribute('viewBox', `0 0 ${SCAT} ${SCAT}`);
  els.scatter.setAttribute('width', SCAT); els.scatter.setAttribute('height', SCAT);
  const span = SCAT - 2 * PAD, frag = document.createDocumentFragment();
  circles = new Array(points.ex.length); byFrame = Array.from({ length: T }, () => []); prevHi = [];
  for (let i = 0; i < points.ex.length; i++) {
    const c = mk('circle', { cx: (PAD + points.ex[i] * span).toFixed(1), cy: (PAD + (1 - points.ey[i]) * span).toFixed(1),
      r: R_BASE, fill: valColor(points.v[i]), 'fill-opacity': 0.5 });
    frag.appendChild(c); circles[i] = c;
    const f = points.t[i]; if (f >= 0 && f < T) byFrame[f].push(i);
  }
  els.scatter.appendChild(frag);
}
function highlightScatter() {
  for (const i of prevHi) { const c = circles[i]; c.setAttribute('r', R_BASE); c.setAttribute('fill-opacity', '0.5'); c.removeAttribute('stroke'); }
  const idx = byFrame[frame] || [];
  for (const i of idx) { const c = circles[i]; c.setAttribute('r', R_HI); c.setAttribute('fill-opacity', '1'); c.setAttribute('stroke', '#221f19'); c.setAttribute('stroke-width', '1.1'); }
  prevHi = idx;
}

function buildOverlay(channels) {
  clear(els.overlay);
  els.overlay.setAttribute('viewBox', `0 0 ${OVERLAY} ${OVERLAY}`);
  els.overlay.setAttribute('width', OVERLAY); els.overlay.setAttribute('height', OVERLAY);
  const span = OVERLAY - 2 * PAD;
  ovMarks = channels.map((cinfo) => {
    const col = chColor(cinfo);
    const px = cinfo.ex.map((e) => PAD + e * span);
    const py = cinfo.ey.map((e) => PAD + (1 - e) * span);
    if (!cinfo.role.startsWith('input')) {
      els.overlay.appendChild(mk('polyline', { points: px.map((x, i) => `${x.toFixed(1)},${py[i].toFixed(1)}`).join(' '),
        fill: 'none', stroke: col, 'stroke-width': 1, 'stroke-opacity': 0.22, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    }
    const marker = mk('circle', { r: cinfo.role.startsWith('input') ? 3.5 : 4.6, fill: col, stroke: '#221f19', 'stroke-width': 1 });
    els.overlay.appendChild(marker);
    return { px, py, marker };
  });
}
function updateOverlay() { for (const m of ovMarks) { m.marker.setAttribute('cx', m.px[frame]); m.marker.setAttribute('cy', m.py[frame]); } }

function buildColorLegend(channels) {
  els.legend.innerHTML = '';
  for (const c of channels) {
    const row = document.createElement('div');
    row.className = 'cw-cl' + (c.role.startsWith('input') ? ' frozen' : '');
    row.innerHTML = `<i style="background:${chColor(c)}"></i> ch${c.ch} · ${c.role}`;
    els.legend.appendChild(row);
  }
}

function updateStatus() { els.status.innerHTML = `frame <b>${String(frame).padStart(3, '0')}</b> / ${T - 1} &nbsp;·&nbsp; ${(byFrame[frame] || []).length} cells lit`; }

function setView(mode) {                          // 'cells' (scatter) or 'paths' (overlay)
  const paths = mode === 'paths';
  els.scatter.style.display = paths ? 'none' : 'block';
  els.overlay.style.display = paths ? 'block' : 'none';
  els.scatterLegend.style.display = paths ? 'none' : '';
  els.legend.style.display = paths ? 'grid' : 'none';
}

function render() {
  if (film) drawFilm(els.movie.getContext('2d'), film, cw, ch);
  for (const k of keyviews) if (k.img) drawFilm(k.ctx, k.img, kw, kh);
  highlightScatter(); updateOverlay(); updateStatus();
  els.time.value = String(frame);
}

function loop(ts) {
  raf = requestAnimationFrame(loop);
  if (holdUntil === 0) holdUntil = ts + 500;   // half-second pause on first load
  if (!playing) return;
  if (ts < holdUntil) return;                  // hold (start of load, and end of each cycle)
  if (ts - lastT < 1000 / fps) return;
  lastT = ts;
  frame = (frame + 1) % T;
  render();
  if (frame === T - 1) holdUntil = ts + 500;   // pause on the final frame before looping
}

async function loadExampleImages(ex) {
  if (cache.has(ex.id)) return cache.get(ex.id);
  const [movie, ...keys] = await Promise.all([loadImg(ex.film), ...ex.keymaps.map((k) => loadImg(k.film))]);
  const entry = { movie, keys };
  cache.set(ex.id, entry);
  return entry;
}

async function selectExample(i) {
  if (i === curIdx) return;
  const token = ++selToken;
  const ex = data.examples[i];
  els.example.value = String(i);
  const imgs = await loadExampleImages(ex);
  if (token !== selToken) return;                 // a newer click won
  curIdx = i;
  film = imgs.movie;
  keyviews.forEach((kv, k) => { kv.img = imgs.keys[k]; kv.capEl.textContent = `ch${ex.keymaps[k].ch} · tvar ${ex.keymaps[k].tvar}`; });
  buildScatter(ex.points);
  buildOverlay(ex.overlay.channels);
  els.head.textContent = `${ex.a} + ${ex.b} = ${ex.decoded}` + (ex.decoded === ex.sum ? '' : ` (want ${ex.sum})`);
  els.caption.textContent =
    `${ex.a} + ${ex.b} on the 8-bit adder NCA. It decoded ${ex.decoded}. Under the movie are the ${ex.keymaps.length} most ` +
    `dynamic hidden channels as spatial maps (red +, blue −). The vertical "spine" is the carry chain up the operand column. ` +
    `On the right are ${ex.n_points.toLocaleString()} points, one per active cell per frame, from their 14 hidden channels (ch2–15). ` +
    `We embed them with t-SNE at perplexity ${data.perplexity}. Colour shows the channel-0 value, and the outlined dots are the ` +
    `current frame. Below, all 16 channels appear as paths in one shared ${data.overlay_method} space. Each view moves with the movie.`;
  frame = 0; lastT = 0; holdUntil = 0;            // restart from the beginning (with the half-second hold)
  playing = true; els.play.textContent = 'Pause';
  render();
}

async function main() {
  data = await (await fetch('data/carry_tsne.json')).json();
  T = data.T; H = data.H; W = data.W;
  cw = (W - 2 * CROP) * MOVIE_SCALE; ch = (H - 2 * CROP) * MOVIE_SCALE;
  kw = (W - 2 * CROP) * KEY_SCALE; kh = (H - 2 * CROP) * KEY_SCALE;
  els.movie.width = cw; els.movie.height = ch; els.movie.style.width = cw + 'px'; els.movie.style.height = ch + 'px';
  els.time.max = String(T - 1);
  if (els.perp) els.perp.textContent = String(data.perplexity ?? '');

  // example dropdown
  data.examples.forEach((ex, i) => {
    const o = document.createElement('option'); o.value = String(i); o.textContent = ex.label;
    els.example.appendChild(o);
  });
  els.example.onchange = () => selectExample(+els.example.value);
  // colour legend + key-map canvases are the same across examples (fixed channels)
  buildColorLegend(data.examples[0].overlay.channels);
  data.keymap_channels.forEach((cch) => {
    const fig = document.createElement('figure'); fig.className = 'cw-km';
    const cv = document.createElement('canvas'); cv.width = kw; cv.height = kh;   // CSS scales display
    fig.appendChild(cv);
    const cap = document.createElement('figcaption'); cap.textContent = `ch${cch}`; fig.appendChild(cap);
    els.keymaps.appendChild(fig);
    keyviews.push({ ch: cch, ctx: cv.getContext('2d'), capEl: cap, img: null });
  });

  // view toggle: scatter (cells) vs channel paths
  document.querySelectorAll('input[name="cwView"]').forEach((r) => { r.onchange = () => setView(r.value); });
  setView('cells');

  await selectExample(0);
  raf = requestAnimationFrame(loop);
}

// ---- controls ----
els.play.onclick = () => { playing = !playing; els.play.textContent = playing ? 'Pause' : 'Play'; lastT = 0; };
els.time.oninput = () => { playing = false; els.play.textContent = 'Play'; frame = +els.time.value; render(); };
els.speed.oninput = () => { fps = +els.speed.value; if (els.fps) els.fps.textContent = String(fps); };

main();
