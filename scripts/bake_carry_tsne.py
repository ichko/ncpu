#!/usr/bin/env python3
"""Bake several 8-bit adder rollouts for the carry-wave analysis page.

For each example (255+1, 255+255, 170+85, ...) we reproduce the exact rollout the
browser runs live (same weights / geometry / fire rate) and dump, per example:

  * a channel-0 FILMSTRIP PNG (the movie);
  * RdBu FILMSTRIP PNGs for a FIXED set of key hidden channels (live spatial maps);
  * a t-SNE embedding of every active cell's hidden-channel state (frame-tagged);
  * a t-SNE "paths" embedding of the per-(channel,frame) states (one curve/channel).

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
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
]
PERPLEXITY = 50            # scatter t-SNE
OVR_PERPLEXITY = 30        # overlay ("paths") t-SNE
POINT_CAP, ACT_THRESH = 5000, 0.4
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
        img.save(path)


# fresh assets
DATA = Path("demo/data")
DATA.mkdir(parents=True, exist_ok=True)
for pat in ("carry_*.png", "carry_*.webp"):
    for p in DATA.glob(pat):
        p.unlink()

rng = np.random.default_rng(SEED)
KEY = None
examples = []
for label, a, b in EXAMPLES:
    roll, got = rollout(a, b)
    T = roll.shape[0]
    eid = f"e{a}_{b}"
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

    # (scatter) t-SNE of per-(cell,frame) hidden-channel vectors
    hidden = list(range(2, C))
    vecs = roll[:, hidden].transpose(0, 2, 3, 1).reshape(T * H * W, len(hidden))
    ts = np.repeat(np.arange(T), H * W)
    ch0v = roll[:, 0].reshape(T * H * W)
    idx = np.where(np.linalg.norm(vecs, axis=1) > ACT_THRESH)[0]
    if idx.size > POINT_CAP:
        idx = rng.choice(idx, POINT_CAP, replace=False)
    Xn = (vecs[idx] - vecs[idx].mean(0)) / (vecs[idx].std(0) + 1e-6)
    emb = TSNE(n_components=2, perplexity=PERPLEXITY, init="pca", learning_rate="auto",
               random_state=SEED).fit_transform(Xn)
    emb = (emb - emb.min(0)) / (emb.max(0) - emb.min(0) + 1e-9)
    points = {"ex": [round(float(v), 4) for v in emb[:, 0]], "ey": [round(float(v), 4) for v in emb[:, 1]],
              "t": [int(v) for v in ts[idx]], "v": [round(float(v), 3) for v in ch0v[idx]]}

    # (paths) t-SNE of per-(channel,frame) states -> one curve per channel
    feats = [roll[:, c].reshape(T, -1) / (roll[:, c].std() + 1e-6) for c in range(C)]
    Xp = PCA(n_components=50, random_state=SEED).fit_transform(np.concatenate(feats, 0))
    Eall = TSNE(n_components=2, perplexity=OVR_PERPLEXITY, init="pca", learning_rate="auto",
                random_state=SEED).fit_transform(Xp)
    Eall = (Eall - Eall.min(0)) / (Eall.max(0) - Eall.min(0) + 1e-9)
    role_of = lambda c: "output" if c == 0 else ("input (frozen)" if c == 1 else "hidden")
    overlay = [{"ch": c, "role": role_of(c), "tvar": round(float(tvar[c]), 3),
                "ex": [round(float(v), 4) for v in Eall[c * T:(c + 1) * T, 0]],
                "ey": [round(float(v), 4) for v in Eall[c * T:(c + 1) * T, 1]]} for c in range(C)]

    examples.append({"id": eid, "label": label, "a": a, "b": b, "sum": a + b, "decoded": got,
                     "film": f"data/carry_{eid}_film.{IMG_FMT}", "n_points": len(points["ex"]),
                     "points": points, "keymaps": keymaps, "overlay": {"channels": overlay}})
    print(f"  baked {eid}: {len(points['ex'])} scatter pts, {len(keymaps)} key maps")

out = {
    "H": H, "W": W, "T": T,
    "perplexity": PERPLEXITY, "overlay_method": f"t-SNE (perplexity {OVR_PERPLEXITY})",
    "keymap_channels": list(KEY),
    "examples": examples,
}
(DATA / "carry_tsne.json").write_text(json.dumps(out))
print(f"saved demo/data/carry_tsne.json  ({len(examples)} examples, key channels {list(KEY)})")
