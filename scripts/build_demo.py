#!/usr/bin/env python3
"""Rebuild the demo's heavy generated assets, then optionally deploy the site.

The site itself (HTML/CSS/JS, demo/assets, and the small hand-exported weight JSONs
in demo/data) is under source control. The heavy assets are NOT — they are ~120 MB of
filmstrip PNGs and embeddings that this script regenerates from the training runs:

    demo/data/rollouts/*.png + subleq_fib.json   scripts/capture_subleq_demo.py
    demo/data/carry_*                            scripts/bake_carry_tsne.py
    movement clusters inside carry_tsne.json     scripts/cluster_carry_cells.py

Because the inputs live in runs/ (also untracked), a fresh clone cannot build these —
you need the machine that holds the checkpoints. Each target is checked for its
checkpoint first and reported as SKIPPED rather than failing the whole build.

    uv run python scripts/build_demo.py                # build what is missing
    uv run python scripts/build_demo.py --force        # rebuild everything
    uv run python scripts/build_demo.py --only carry   # one target
    uv run python scripts/build_demo.py --deploy       # build, then Cloudflare Pages

Deploy shells out to `npx wrangler pages deploy`. Log in once, interactively, with
`npx wrangler login` — this script will not try to authenticate for you.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "demo" / "data"
PROJECT = "ncpu-demo"                     # Cloudflare Pages project name

# name -> (checkpoint it needs, command, a path that exists once it is built)
TARGETS = {
    "carry": (
        "runs/E3_adder8_cols2_s0_20260316_033201/checkpoints/nca_099501.pt",
        [sys.executable, "scripts/bake_carry_tsne.py"],
        DATA / "carry_tsne.json",
    ),
    # clustering is a post-process over the carry blobs, so it runs after "carry"
    "clusters": (
        "runs/E3_adder8_cols2_s0_20260316_033201/checkpoints/nca_099501.pt",
        [sys.executable, "scripts/cluster_carry_cells.py"],
        None,
    ),
    "subleq": (
        "runs/20260726_214420_SUBLEQ_dartboard_w8_s0/checkpoints/nca_last.pt",
        [sys.executable, "scripts/capture_subleq_demo.py", "5"],
        DATA / "subleq_fib.json",
    ),
}


def run(name, force):
    ckpt, cmd, marker = TARGETS[name]
    if not (ROOT / ckpt).exists():
        print(f"[{name}] SKIPPED — no checkpoint at {ckpt}")
        return False
    if marker is not None and marker.exists() and not force:
        print(f"[{name}] up to date ({marker.relative_to(ROOT)}) — use --force to rebuild")
        return True
    print(f"[{name}] building: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"[{name}] FAILED (exit {r.returncode})")
        return False
    print(f"[{name}] done")
    return True


def size_report():
    total = 0
    for pat in ("rollouts/*.png", "carry_*"):
        n = sum(p.stat().st_size for p in DATA.glob(pat) if p.is_file())
        total += n
        print(f"  {pat:<18} {n / 1e6:8.1f} MB")
    print(f"  {'total generated':<18} {total / 1e6:8.1f} MB")


def deploy():
    cmd = ["npx", "wrangler", "pages", "deploy", "demo", f"--project-name={PROJECT}"]
    print(f"deploying: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(TARGETS), action="append")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--deploy", action="store_true")
    a = ap.parse_args()

    ok = all(run(name, a.force) for name in (a.only or sorted(TARGETS)))
    print("\ngenerated assets in demo/data:")
    size_report()
    if a.deploy:
        if not ok:
            sys.exit("refusing to deploy: a build target skipped or failed")
        sys.exit(deploy())
