#!/usr/bin/env python3
"""Build the rollout animations for the lightning-talk deck (demo/talk.html).

The films in demo/data already hold every frame of a run tiled horizontally at
one pixel per cell. That *is* a sprite sheet, so the deck animates it with a CSS
`steps()` animation instead of shipping a video: 129 frames of the 8-bit adder
cost about 550 KB as the source PNG against 6.6 MB as a GIF, and the browser gets
exact frames it can pause on.

The visible-channel films ship as they are. The hidden-channel films need work
first: each channel sits on a large constant offset that hides everything else,
so every frame's own mean comes off and the rest is scaled by a percentile. Those
films are also the only surviving record of the channels, so the values are read
back out of the render — matplotlib's RdBu_r is the eleven ColorBrewer RdBu
anchors interpolated, and they are palette PNGs, so a nearest-colour lookup
inverts them exactly.

    uv run python scripts/build_talk.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "demo/data"
OUT = ROOT / "demo/assets/talk"

W, H = 80, 112

RDBU = ["67001f", "b2182b", "d6604d", "f4a582", "fddbc7", "f7f7f7",
        "d1e5f0", "92c5de", "4393c3", "2166ac", "053061"]


def lut(n=256):
    a = np.array([[int(c[i:i + 2], 16) for i in (0, 2, 4)] for c in RDBU], float)[::-1]
    x = np.linspace(0, 1, len(a))
    return np.stack([np.interp(np.linspace(0, 1, n), x, a[:, k]) for k in range(3)], 1)


def film_values(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(float)
    L = lut()
    d = ((a.reshape(-1, 1, 3) - L[None]) ** 2).sum(2)
    v = (d.argmin(1).reshape(a.shape[:2]) / (len(L) - 1)) * 2 - 1
    return np.stack([v[:, t * W:(t + 1) * W] for t in range(a.shape[1] // W)])


def save(name, img):
    """Write as an 8-bit palette PNG — the films are flat colour-mapped output,
    so a palette costs nothing and roughly halves the file."""
    img.convert("P", palette=Image.ADAPTIVE, colors=256).save(OUT / name, optimize=True)
    n = img.width // W
    print(f"  {name:<28} {n:>3} frames  {img.width}x{img.height}  "
          f"{(OUT / name).stat().st_size / 1e3:6.0f} KB")
    return n


def copy_visible(eid, name):
    return save(name, Image.open(DATA / f"carry_{eid}_film.png").convert("RGB"))


def build_hidden(eid, ch, name):
    v = film_values(DATA / f"carry_{eid}_ch{ch}_film.png")
    d = v - v.mean((1, 2), keepdims=True)
    lim = np.percentile(np.abs(d), 99)
    x = np.clip(d / (lim + 1e-9), -1, 1)
    rgb = lut()[np.clip(((x + 1) / 2 * 255).round(), 0, 255).astype(int)].astype(np.uint8)
    return save(name, Image.fromarray(np.concatenate(list(rgb), 1)))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("sprite sheets")
    counts = {
        # 239 + 33 sets off two separate carry runs; 255 + 1 is the single longest
        "two_waves": copy_visible("e239_33", "film_two_waves.png"),
        "longest": copy_visible("e255_1", "film_longest.png"),
        "hidden": build_hidden("e255_1", 4, "film_hidden_ch4.png"),
    }
    total = sum(p.stat().st_size for p in OUT.glob("*.png")) / 1e6
    print(f"\n{len(counts)} sheets, {total:.2f} MB total -> {OUT}")
    print("frame counts (the deck's steps() must match):", counts)
