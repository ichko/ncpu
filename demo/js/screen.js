// screen.js — reproduces the circle/screen geometry used to train the NCAs.
// Ported 1:1 from src/ncpu/utils.py (make_io_screen, make_io_screen_cols1).
// cv2.circle filled == Euclidean disk (dx*dx+dy*dy <= r*r), verified exactly.

const WHITE = 255 / 128 - 1;   // bit 1  ->  +0.9921875
const BLACK = 0 / 128 - 1;     // bit 0  ->  -1.0
const NEUTRAL = 128 / 128 - 1; // background -> 0.0

// ---- circle centres ------------------------------------------------------

// make_io_screen_cols1: inputs one vertical column (x=side), outputs one column (x=W-side)
function colCenters(n, x, H, r, among) {
  const v = n * 2 * r + among * (n - 1);
  const top = Math.floor((H - v) / 2);
  const out = [];
  for (let i = 0; i < n; i++) out.push({ x, y: top + r + i * (2 * r + among), bit: i });
  return out;
}

// make_io_screen: left inputs in up-to-2 columns, right outputs one column
function twoColLeftCenters(n, side, H, r, among) {
  const nRows = Math.ceil(n / 2);
  const v = nRows * 2 * r + among * (nRows - 1);
  const top = Math.floor((H - v) / 2);
  const out = [];
  for (let i = 0; i < n; i++) {
    const col = Math.floor(i / nRows);
    const row = i % nRows;
    out.push({ x: side + col * (2 * r + among), y: top + r + row * (2 * r + among), bit: i });
  }
  return out;
}

// Given a model config, return {inputs:[{x,y,bit}], outputs:[{x,y,bit}], r}
export function geometry(cfg) {
  const [among, side] = cfg.spacing;
  const r = cfg.r, H = cfg.H, W = cfg.W;
  let inputs, outputs;
  if (cfg.screen === 'make_io_screen_cols1') {
    inputs = colCenters(cfg.n_left, side, H, r, among);
    outputs = colCenters(cfg.n_right, W - side, H, r, among);
  } else { // make_io_screen (two-column left)
    inputs = twoColLeftCenters(cfg.n_left, side, H, r, among);
    outputs = colCenters(cfg.n_right, W - side, H, r, among);
  }
  return { inputs, outputs, r, H, W };
}

// Build the normalised channel-0 image (Float32Array H*W) for a set of input bits.
export function buildImage(cfg, geo, inputBits) {
  const { H, W } = cfg, r = geo.r;
  const img = new Float32Array(H * W).fill(NEUTRAL);
  const r2 = r * r;
  geo.inputs.forEach((c) => {
    const val = inputBits[c.bit] ? WHITE : BLACK;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx * dx + dy * dy > r2) continue;
        const x = c.x + dx, y = c.y + dy;
        if (x < 0 || y < 0 || x >= W || y >= H) continue;
        img[y * W + x] = val;
      }
    }
  });
  return img;
}

// Precompute, per output bit, the list of pixel indices inside its disk (for readout).
export function outputMasks(cfg, geo) {
  const { H, W } = cfg, r = geo.r, r2 = r * r;
  return geo.outputs.map((c) => {
    const idx = [];
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx * dx + dy * dy > r2) continue;
        const x = c.x + dx, y = c.y + dy;
        if (x < 0 || y < 0 || x >= W || y >= H) continue;
        idx.push(y * W + x);
      }
    }
    return idx;
  });
}
