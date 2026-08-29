#!/usr/bin/env python3
"""Cluster the tracked cells BY HOW THEY MOVE and write the labels into the manifest.

This is a post-process over what bake_carry_tsne.py already produced: it reads each
example's cells.bin, clusters the per-cell trajectories, and injects a "cluster" array
per example back into demo/data/carry_tsne.json. It never re-runs t-SNE, so it costs
seconds rather than the ~5 minutes per example the embedding costs.

A cell's feature vector is its whole path through the embedding — x and y at every
frame, concatenated. Cells that travel the same route through latent space therefore
land in the same cluster, which is what "clusters of movement" should mean. Labels are
per example: the same grid cell moves differently in 255+1 than in 170+85.

    uv run python scripts/other/cluster_carry_cells.py [--k 6]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

DATA = Path("demo/data")
MANIFEST = DATA / "carry_tsne.json"
SEED = 0


def trajectories(path, n, T):
    """(n, 2T) — each row is one cell's x,y over every frame, from the baked blob."""
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    N = n * T
    x = buf[0:2 * N].view("<u2").astype(np.float32) / 65535.0
    y = buf[2 * N:4 * N].view("<u2").astype(np.float32) / 65535.0
    # the blob is frame-major, so reshape to (T, n) then transpose to per-cell rows
    return np.concatenate([x.reshape(T, n).T, y.reshape(T, n).T], axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="number of movement clusters")
    a = ap.parse_args()

    man = json.loads(MANIFEST.read_text())
    n, T = man["cells"]["n"], man["T"]
    for ex in man["examples"]:
        X = trajectories(Path("demo") / ex["cellbin"], n, T)
        km = KMeans(n_clusters=a.k, n_init=10, random_state=SEED).fit(X)
        lab = km.labels_.astype(int)
        # relabel by size so cluster 0 is always the biggest — keeps the palette stable
        order = np.argsort(-np.bincount(lab, minlength=a.k))
        remap = np.zeros(a.k, int)
        remap[order] = np.arange(a.k)
        ex["cluster"] = [int(v) for v in remap[lab]]
        sizes = np.bincount(remap[lab], minlength=a.k)
        print(f"{ex['label']:<10} clusters {sizes.tolist()}  (of {n} cells)")

    man["n_clusters"] = a.k
    MANIFEST.write_text(json.dumps(man))
    print(f"wrote {a.k} movement clusters per example into {MANIFEST} "
          f"({MANIFEST.stat().st_size / 1e6:.2f} MB)")
