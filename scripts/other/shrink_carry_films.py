#!/usr/bin/env python3
"""Re-encode the carry filmstrips as 8-bit palette PNGs, in place and losslessly.

The strips come out of matplotlib colormaps, so a frame holds at most a couple of
hundred distinct colours. Stored as truecolour PNG that wastes three bytes per pixel on
a palette that would fit in one. Converting to mode "P" is about 3x smaller and, for
these images, bit-exact.

Every file is checked before it is replaced: the palette version is decoded back to RGB
and compared against the original. Any file that does not round-trip exactly is left
alone and reported. Nothing here touches the embeddings, so it needs no re-bake.

    uv run python scripts/other/shrink_carry_films.py            # report only
    uv run python scripts/other/shrink_carry_films.py --apply     # rewrite the files
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path("demo/data")


def repack(path, apply):
    im = Image.open(path).convert("RGB")
    rgb = np.asarray(im)
    colors = len(np.unique(rgb.reshape(-1, 3), axis=0))
    before = path.stat().st_size

    pal = im.convert("P", palette=Image.ADAPTIVE, colors=256)
    tmp = path.with_suffix(".png.tmp")
    pal.save(tmp, "PNG", optimize=True)

    # decode what we would ship and insist it matches the original exactly
    back = np.asarray(Image.open(tmp).convert("RGB"))
    exact = np.array_equal(back, rgb)
    after = tmp.stat().st_size

    if not exact or after >= before:
        tmp.unlink()
        return before, before, colors, ("NOT EXACT" if not exact else "no gain")
    if apply:
        tmp.replace(path)
    else:
        tmp.unlink()
    return before, after, colors, "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite the files (default: report only)")
    a = ap.parse_args()

    films = sorted(DATA.glob("carry_*_film.png"))
    if not films:
        raise SystemExit("no carry filmstrips in demo/data — run scripts/other/bake_carry_tsne.py first")

    tot_b = tot_a = 0
    skipped = []
    for p in films:
        b, n, colors, note = repack(p, a.apply)
        tot_b += b
        tot_a += n
        if note != "ok":
            skipped.append((p.name, note))
        print(f"  {p.name:<34} {b/1e6:5.2f} -> {n/1e6:5.2f} MB  {b/max(n,1):4.1f}x  "
              f"{colors:>4} colours  {note}")

    print(f"\n{len(films)} filmstrips: {tot_b/1e6:.1f} MB -> {tot_a/1e6:.1f} MB "
          f"({tot_b/max(tot_a,1):.1f}x smaller, {(tot_b-tot_a)/1e6:.1f} MB saved)")
    for name, note in skipped:
        print(f"  left alone: {name} ({note})")
    if not a.apply:
        print("\nreport only — re-run with --apply to rewrite the files")
