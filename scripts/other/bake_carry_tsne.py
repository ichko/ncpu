#!/usr/bin/env python3
"""Bake several 8-bit adder rollouts for the carry-wave analysis page.

For each example (255+1, 255+255, 170+85, ...) we reproduce the exact rollout the
browser runs live (same weights / geometry / fire rate) and dump, per example:

  * a channel-0 FILMSTRIP PNG (the movie);
  * RdBu FILMSTRIP PNGs for a FIXED set of key hidden channels (live spatial maps);
  * a t-SNE embedding of a FIXED set of tracked cells at EVERY frame, so each cell
    has a continuous trajectory (one curve per cell), as a compact BINARY blob;
  * a t-SNE "paths" embedding of the per-(channel,frame) states (one curve/channel).

The tracked-cell set is drawn once and reused for every frame and every example. A
uniform draw over (cell, frame) pairs instead — what this script used to do — gives
~1.3 frames per cell, so consecutive frames light up disjoint dots and the animation
flashes instead of moving.

The cell embedding goes out as carry_<eid>_cells.bin, not JSON: at 5 bytes per state
a thousand cells cost ~650 KB, where the same numbers as JSON text cost ~14 MB. Layout
is frame-major (all cells at frame 0, then frame 1, ...), three arrays back to back:

    uint16 x[T*n]   uint16 y[T*n]   int8 v[T*n]

x/y are the embedding scaled to 0..65535; v is the channel-0 value scaled by 127.

Everything is baked offline (the embeddings are too slow to run live) into one
JSON (demo/data/carry_tsne.json) that the page switches between with tabs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.cm as cm
from PIL import Image
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from openTSNE import TSNE as OpenTSNE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from ncpu.nca import NeuralCA

RUN = "runs/E3_adder8_cols2_s0_20260316_033201"
CKPT = f"{RUN}/checkpoints/nca_099501.pt"
BITS, HORIZON, SEED = 8, 128, 0
dev = "cuda" if torch.cuda.is_available() else "cpu"

# contrasting carry behaviours to switch between (label, a, b)
EXAMPLES = [
    ("255 + 1",   255, 1),     # longest carry chain — one bit ripples all the way up
    ("255 + 255", 255, 255),   # a carry at every single bit (max input)
    ("170 + 85",  170, 85),    # 10101010 + 01010101 = 11111111 — NO carry anywhere
    ("15 + 1",    15, 1),      # short 4-bit ripple
    ("127 + 1",   127, 1),     # 7-bit ripple that flips to a single high bit (128)
    ("239 + 33",  239, 33),    # TWO chains, one per nibble: bits 0-3 ripple into bit 4, which
                               # absorbs the carry; then bits 5-7 ripple out into bit 8
    ("1 + 255",   1, 255),     # the long chain again, operands swapped between the input columns
]
PERPLEXITY = 50            # scatter t-SNE
OVR_PERPLEXITY = 30        # overlay ("paths") t-SNE
GRID_STRIDE = 2            # tracked cells: a regular grid over the field, every STRIDE px
                           # (a plain lattice: no I/O-disk extras, see tracked_cells)
N_KEYMAPS = 4              # number of key hidden channels rendered as live maps
IMG_FMT = "png"           # filmstrip format: "webp" (small) or "png" (revert with a re-bake)
WEBP_QUALITY = 90         # near-lossless for these flat colormap strips; raise toward 100 if unhappy

# ── geometry: ported 1:1 from demo/js/screen.js (make_io_screen, two-col left) ──
WHITE, BLACK, NEUTRAL = 255 / 128 - 1, 0 / 128 - 1, 128 / 128 - 1
H, W, r, among, side = 112, 80, 4, 2, 21
N_LEFT, N_RIGHT = 16, 9


def two_col_left_centers(n):
    n_rows = -(-n // 2)
    v = n_rows * 2 * r + among * (n_rows - 1)
    top = (H - v) // 2
    return [(side + (i // n_rows) * (2 * r + among), top + r + (i % n_rows) * (2 * r + among)) for i in range(n)]


def col_centers(n, x):
    v = n * 2 * r + among * (n - 1)
    top = (H - v) // 2
    return [(x, top + r + i * (2 * r + among)) for i in range(n)]


IN_CENTERS = two_col_left_centers(N_LEFT)
OUT_CENTERS = col_centers(N_RIGHT, W - side)


def lattice(n, stride):
    """Centred 1-D lattice: as many points as fit at this stride, with the leftover
    margin split evenly so the pattern is symmetric about the middle of the axis."""
    count = (n - 1) // stride + 1
    off = (n - 1 - (count - 1) * stride) // 2
    return np.arange(count) * stride + off


def tracked_cells():
    """One even, symmetric lattice over the whole field. Nothing else.

    Earlier versions bolted extras onto the grid — a whole patch per I/O disk, then just
    the disk centres — and both broke the lattice: the off-grid points read as clumps and
    the picture stopped looking regular. At a fine enough stride every disk already holds
    a dozen grid cells, so the bits are covered without special cases.
    """
    ys, xs = np.meshgrid(lattice(H, GRID_STRIDE), lattice(W, GRID_STRIDE), indexing="ij")
    return np.unique((ys * W + xs).ravel())


def cell_roles(cells):
    """Per tracked cell: 0 = field, 1 = inside an input disk, 2 = inside an output disk,
    3 = the band strictly between the two columns, plus which bit slot it belongs to
    (-1 for anything that is not a bit).

    Band 3 is where the computation has to happen: it holds no input and no output, so
    whatever crosses it is the NCA moving information across the grid.
    """
    role = np.zeros(len(cells), np.int8)
    slot = np.full(len(cells), -1, np.int8)
    x_all = cells % W
    lo = max(cx for cx, _ in IN_CENTERS) + r + 1                 # right edge of the input column
    hi = min(cx for cx, _ in OUT_CENTERS) - r - 1                # left edge of the output column
    role[(x_all >= lo) & (x_all <= hi)] = 3
    pos = {c: i for i, c in enumerate(cells.tolist())}
    for kind, cs in ((1, IN_CENTERS), (2, OUT_CENTERS)):
        for k, (cx, cy) in enumerate(cs):
            for y, x in disk_idx(cx, cy):
                i = pos.get(y * W + x)
                if i is not None:
                    role[i], slot[i] = kind, k
    return role, slot


def disk_idx(cx, cy):
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r and 0 <= cx + dx < W and 0 <= cy + dy < H:
                out.append((cy + dy, cx + dx))
    return out


def int_bits_msb(v, n):
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def build_image(a, b):
    img = np.full((H, W), NEUTRAL, dtype=np.float32)
    bits = int_bits_msb(a, BITS) + int_bits_msb(b, BITS)
    for (cx, cy), bit in zip(IN_CENTERS, bits):
        for y, x in disk_idx(cx, cy):
            img[y, x] = WHITE if bit else BLACK
    return img


# ── load the exact demo NCA ───────────────────────────────────────────────────
cfg = json.loads(Path(f"{RUN}/config.json").read_text())["nca"]
nca = NeuralCA(**cfg).to(dev)
nca.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False), strict=False)
nca.eval()
C = cfg["channels"]


def rollout(a, b):
    torch.manual_seed(SEED)
    st = torch.zeros(1, C, H, W, device=dev)
    seed_img = torch.from_numpy(build_image(a, b)).to(dev)
    st[:, 0] = seed_img; st[:, 1] = seed_img
    with torch.no_grad():
        roll = nca.forward(st, steps=HORIZON)[0].cpu().numpy()   # (T, C, H, W)
    out = [np.mean([roll[-1, 0, y, x] for y, x in disk_idx(cx, cy)]) for (cx, cy) in OUT_CENTERS]
    got = sum((1 if m > 0 else 0) << (N_RIGHT - 1 - i) for i, m in enumerate(out))
    return roll, got


def save_film(m_thw, path, cmap, lim):
    """m_thw: (T,H,W). Map to [0,1] by ((m/lim)+1)/2 and save an (H, T*W, 3) strip."""
    t = np.clip((m_thw / lim + 1) / 2, 0, 1)
    rgb = (cmap(t)[..., :3] * 255).astype(np.uint8)          # (T,H,W,3)
    img = Image.fromarray(rgb.transpose(1, 0, 2, 3).reshape(H, -1, 3))
    if str(path).endswith(".webp"):
        img.save(path, "WEBP", quality=WEBP_QUALITY, method=6)
    else:
        # A colormapped strip holds a couple of hundred distinct colours, so an 8-bit
        # palette stores it exactly at a third of the size. scripts/other/shrink_carry_films.py
        # does the same job to strips baked before this.
        img.convert("P", palette=Image.ADAPTIVE, colors=256).save(path, "PNG", optimize=True)


DATA = Path("demo/data")
MANIFEST = DATA / "carry_tsne.json"
DATA.mkdir(parents=True, exist_ok=True)

# Importable: everything above is definitions, so other scripts (bake_path_space.py)
# can reuse the exact geometry, cell set and rollout without triggering a bake.
if __name__ == "__main__":
    # Pass example indices to bake a subset ("bake ... 0 1 2"); the rest are carried over
    # from the existing manifest. One t-SNE per example is minutes of work, so re-baking
    # all five to change one is wasteful.
    ONLY = sorted({int(v) for v in sys.argv[1:] if v.isdigit()}) or list(range(len(EXAMPLES)))
    prev = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    prev_ex = {e["id"]: e for e in prev.get("examples", [])}
    KEY = list(prev["keymap_channels"]) if ONLY != list(range(len(EXAMPLES))) and "keymap_channels" in prev else None

    CELLS = tracked_cells()                                      # same cells for every example
    ROLE, SLOT = cell_roles(CELLS)
    print(f"tracking {len(CELLS)} cells ({100 * len(CELLS) / (H * W):.1f}% of the {H}x{W} field): "
          f"symmetric lattice, stride {GRID_STRIDE} — {(ROLE == 1).sum()} input, "
          f"{(ROLE == 2).sum()} output, {(ROLE == 3).sum()} in the compute band")

    # An old example can only be carried over if its blob was baked against THIS cell set —
    # otherwise the page would read the wrong stride's coordinates.
    if prev.get("cells", {}).get("n") != len(CELLS) or prev.get("cells", {}).get("stride") != GRID_STRIDE:
        if prev_ex:
            print(f"cell set changed (was {prev.get('cells', {}).get('n')} @ stride "
                  f"{prev.get('cells', {}).get('stride')}) — old examples cannot be reused")
        prev_ex, KEY = {}, None
    print(f"baking examples {ONLY}: {', '.join(EXAMPLES[i][0] for i in ONLY)}")

    def write_manifest():
        """Rewrite the manifest from whatever is baked now plus whatever survived from the
        previous run. Called after EVERY example so a chunked bake that gets cut short
        still leaves a loadable page and does not have to redo finished work."""
        examples = [baked.get(f"e{a}_{b}", prev_ex.get(f"e{a}_{b}")) for _, a, b in EXAMPLES]
        missing = [EXAMPLES[i][0] for i, e in enumerate(examples) if e is None]
        examples = [e for e in examples if e is not None]
        out = {
            "H": H, "W": W, "T": T,
            "perplexity": PERPLEXITY, "overlay_method": f"t-SNE (perplexity {OVR_PERPLEXITY})",
            "keymap_channels": list(KEY),
            # the tracked-cell set is shared by every example, so it lives here once
            "cells": {"n": len(CELLS), "stride": GRID_STRIDE,
                      "y": [int(c // W) for c in CELLS], "x": [int(c % W) for c in CELLS],
                      "role": [int(v) for v in ROLE], "slot": [int(v) for v in SLOT]},
            "examples": examples,
        }
        MANIFEST.write_text(json.dumps(out))
        return examples, missing



    baked = {}
    for label, a, b in [EXAMPLES[i] for i in ONLY]:
        roll, got = rollout(a, b)
        T = roll.shape[0]
        eid = f"e{a}_{b}"
        for p in list(DATA.glob(f"carry_{eid}_*.png")) + list(DATA.glob(f"carry_{eid}_*.webp")):
            p.unlink()                                           # drop this example's stale assets
        tvar = roll.reshape(T, C, -1).std(0).mean(1)
        if KEY is None:                                          # fix key channels from the first example
            KEY = sorted(range(2, C), key=lambda c: -tvar[c])[:N_KEYMAPS]
        print(f"{label}: decoded {got} ({'OK' if got == a + b else f'!= {a + b}'})")

        # (movie) channel-0 filmstrip
        save_film(roll[:, 0], DATA / f"carry_{eid}_film.{IMG_FMT}", cm.viridis, 1.0)

        # (key maps) RdBu filmstrips for the fixed key hidden channels
        keymaps = []
        for c in KEY:
            lim = float(max(abs(roll[:, c].min()), abs(roll[:, c].max()), 1e-3))
            save_film(roll[:, c], DATA / f"carry_{eid}_ch{c}_film.{IMG_FMT}", cm.RdBu_r, lim)
            keymaps.append({"ch": c, "tvar": round(float(tvar[c]), 3), "film": f"data/carry_{eid}_ch{c}_film.{IMG_FMT}"})

        # (cells) t-SNE of the tracked cells' hidden-channel vectors at EVERY frame, so the
        # page can move the SAME dots along continuous trajectories as it scrubs. openTSNE,
        # not sklearn: this is ~10x the point count sklearn's Barnes-Hut is comfortable with.
        hidden = list(range(2, C))
        vecs = roll[:, hidden].transpose(0, 2, 3, 1).reshape(T, H * W, len(hidden))
        flat = vecs[:, CELLS].reshape(-1, len(hidden))           # rows are frame-major: (t, cell)
        Xn = ((flat - flat.mean(0)) / (flat.std(0) + 1e-6)).astype(np.float32)
        emb = np.asarray(OpenTSNE(n_components=2, perplexity=PERPLEXITY, initialization="pca",
                                  n_jobs=-1, random_state=SEED).fit(Xn), dtype=np.float64)
        emb = (emb - emb.min(0)) / (emb.max(0) - emb.min(0) + 1e-9)
        ch0v = roll[:, 0].reshape(T, H * W)[:, CELLS].reshape(-1)
        xq = np.clip(np.rint(emb[:, 0] * 65535), 0, 65535).astype("<u2")
        yq = np.clip(np.rint(emb[:, 1] * 65535), 0, 65535).astype("<u2")
        vq = np.clip(np.rint(ch0v * 127), -127, 127).astype("<i1")
        (DATA / f"carry_{eid}_cells.bin").write_bytes(xq.tobytes() + yq.tobytes() + vq.tobytes())

        # (paths) t-SNE of per-(channel,frame) states -> one curve per channel. A channel that
        # is constant in time (the frozen input) contributes ONE row: its T duplicates would
        # otherwise be degenerate for t-SNE and its marker would random-walk across the map.
        static = [bool(roll[:, c].std(0).max() < 1e-6) for c in range(C)]
        rows, keys = [], []
        for c in range(C):
            for f in ([0] if static[c] else range(T)):
                rows.append(roll[f, c].ravel() / (roll[:, c].std() + 1e-6))
                keys.append(c)
        keys = np.array(keys)
        Xp = PCA(n_components=50, random_state=SEED).fit_transform(np.stack(rows))
        Eall = TSNE(n_components=2, perplexity=OVR_PERPLEXITY, init="pca", learning_rate="auto",
                    random_state=SEED).fit_transform(Xp)
        Eall = (Eall - Eall.min(0)) / (Eall.max(0) - Eall.min(0) + 1e-9)
        role_of = lambda c: "output" if c == 0 else ("input (frozen)" if c == 1 else "hidden")
        overlay = []
        for c in range(C):
            E = Eall[keys == c]
            if static[c]:
                E = np.repeat(E, T, axis=0)                      # frozen: parked at one spot
            overlay.append({"ch": c, "role": role_of(c), "static": static[c], "tvar": round(float(tvar[c]), 3),
                            "ex": [round(float(v), 4) for v in E[:, 0]],
                            "ey": [round(float(v), 4) for v in E[:, 1]]})

        baked[eid] = {"id": eid, "label": label, "a": a, "b": b, "sum": a + b, "decoded": got,
                      "film": f"data/carry_{eid}_film.{IMG_FMT}", "n_points": T * len(CELLS),
                      "cellbin": f"data/carry_{eid}_cells.bin",
                      "keymaps": keymaps, "overlay": {"channels": overlay}}
        write_manifest()
        print(f"  baked {eid}: {len(CELLS)} tracked cells x {T} frames, {len(keymaps)} key maps", flush=True)

    examples, missing = write_manifest()
    if missing:
        print(f"note: not in this run and not in the old manifest, so dropped: {', '.join(missing)}")
    sz = sum(p.stat().st_size for p in DATA.glob("carry_*_cells.bin")) / 1e6
    print(f"saved {MANIFEST} ({MANIFEST.stat().st_size / 1e6:.2f} MB) + {len(examples)} cell blobs "
          f"({sz:.2f} MB total), {len(CELLS)} tracked cells, key channels {list(KEY)}")
