#!/usr/bin/env python3
"""Look at the Wang tiling task, and check it, with no model involved.

Three jobs, three modes.

**Render** (default).  The instance as the NCA sees it (a ring of committed
border cells, a blank — i.e. uniform-belief — interior), the solved state, and
the clamp mask, next to a human-readable picture of the tiles themselves.  The
state is a stack of ``T + 1`` planes over a one-pixel-per-cell grid, so it is
shown as a labelled plane montage: plane ``t`` is belief in tile ``t``, the
last plane is the reserved confidence channel ``K``.  This montage is the
answer to the one real argument for the old spatial-block layout — a plane
stack is perfectly legible, it just needs to be laid out.  Writes
``wang_preview.png`` to the PROJECT ROOT.

**``--check``**.  Validation that costs no GPU:

  * encode -> decode round-trips on the solution screen, on the seed screen,
    and on a noisy solution screen;
  * ``clamp_mask`` covers exactly the given discs and nothing else, and the
    seed and target agree wherever the clamp is 1;
  * ``random_candidate`` respects the clamp — the baseline must not be solving
    an easier puzzle than the model (``relax.matched_baseline`` samples it);
  * ``relax.givens_preserved`` reports 1.0 for a clamped rollout and less for
    an unclamped one, so the cheat detector is known to detect;
  * ``decode`` is repr-stable, so ``relax.count_distinct`` counts what it says;
  * energy sanity: 0 at a solution, monotone in the exact violation count, and
    the aggregate comparison — how much of the energy range ONE broken edge
    costs — measured rather than asserted.

**``--survey``**.  The difficulty distribution of the generated instances,
measured with the reference solver.  Reports acceptance rate, the fraction
arc-consistency decides outright (those are warm-ups, not instances), and the
node-expansion quantiles, against the first-moment order parameter
``alpha = T / C^2``.

Usage:
    uv run python scripts/preview_wang.py
    uv run python scripts/preview_wang.py --check
    uv run python scripts/preview_wang.py --survey --instances 64
"""

import random
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.relax import count_distinct, givens_preserved, matched_baseline
from ncpu.tasks.wang import (
    AGGREGATES,
    WangDifficulty,
    WangTiling,
    violation_list,
)

ROOT = Path(__file__).resolve().parent.parent

parser = ArgumentParser()
parser.add_argument("--rows", type=int, default=8)
parser.add_argument("--cols", type=int, default=8)
parser.add_argument("--tiles", type=int, default=12)
parser.add_argument("--colors", type=int, default=3)
parser.add_argument("--pitch", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tileset-seed", type=int, default=0)
parser.add_argument("--difficulty", default="easy")
parser.add_argument("--check", action="store_true")
parser.add_argument("--survey", action="store_true")
parser.add_argument("--instances", type=int, default=48)
parser.add_argument("--out", default="wang_preview.png")
args = parser.parse_args()


def build(rows=None, cols=None, tiles=None, colors=None, tseed=None):
    return WangTiling(
        rows=rows or args.rows, cols=cols or args.cols,
        n_tiles=tiles or args.tiles, n_colors=colors or args.colors,
        pitch=args.pitch, seed=args.tileset_seed if tseed is None else tseed,
        difficulty=args.difficulty,
    )


# ── rendering ────────────────────────────────────────────────────────────────

def draw_tileset(ax, task):
    """The tile set itself, so the encoding can be read against the physics."""
    ts = task.tileset
    cmap = plt.get_cmap("tab10")
    cols = min(6, ts.n_tiles)
    for t, tile in enumerate(ts.tiles):
        i, j = divmod(t, cols)
        cx, cy = j * 1.2, -i * 1.2
        tri = {
            0: [(cx - .5, cy + .5), (cx + .5, cy + .5), (cx, cy)],       # north
            1: [(cx + .5, cy + .5), (cx + .5, cy - .5), (cx, cy)],       # east
            2: [(cx - .5, cy - .5), (cx + .5, cy - .5), (cx, cy)],       # south
            3: [(cx - .5, cy + .5), (cx - .5, cy - .5), (cx, cy)],       # west
        }
        for d in range(4):
            ax.add_patch(plt.Polygon(tri[d], facecolor=cmap(tile[d] % 10),
                                     edgecolor="white", linewidth=0.6))
        ax.text(cx, cy - 0.72, str(t), ha="center", va="center", fontsize=6)
    ax.set_xlim(-1, cols * 1.2)
    ax.set_ylim(-((ts.n_tiles - 1) // cols) * 1.2 - 1.1, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"tile set: {ts.describe()}", fontsize=8)


def draw_tiling(ax, task, tiling, givens, title):
    """The tiling in human terms — four coloured triangles per cell."""
    cmap = plt.get_cmap("tab10")
    ts = task.tileset
    for k, t in enumerate(tiling):
        i, j = divmod(k, task.cols)
        cx, cy = j, -i
        if t < 0:                                   # free cell: no commitment yet
            ax.add_patch(plt.Rectangle((cx - .5, cy - .5), 1, 1,
                                       facecolor="0.9", edgecolor="0.75", lw=0.4))
            continue
        tri = {
            0: [(cx - .5, cy + .5), (cx + .5, cy + .5), (cx, cy)],
            1: [(cx + .5, cy + .5), (cx + .5, cy - .5), (cx, cy)],
            2: [(cx - .5, cy - .5), (cx + .5, cy - .5), (cx, cy)],
            3: [(cx - .5, cy + .5), (cx - .5, cy - .5), (cx, cy)],
        }
        for d in range(4):
            ax.add_patch(plt.Polygon(tri[d], facecolor=cmap(ts.side(t, d) % 10),
                                     edgecolor="none"))
        given = givens[k] >= 0
        ax.add_patch(plt.Rectangle((cx - .5, cy - .5), 1, 1, fill=False,
                                   edgecolor="black" if given else "0.75",
                                   linewidth=1.8 if given else 0.4))
    ax.set_xlim(-0.6, task.cols - 0.4)
    ax.set_ylim(-task.rows + 0.4, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8)


def montage(state, per_row=5, pad=1):
    """(P, H, W) -> one image with the planes tiled and NaN gutters."""
    p, h, w = state.shape
    rows = int(np.ceil(p / per_row))
    out = np.full((rows * (h + pad) + pad, per_row * (w + pad) + pad), np.nan)
    origins = []
    for k in range(p):
        r, c = divmod(k, per_row)
        y, x = r * (h + pad) + pad, c * (w + pad) + pad
        out[y:y + h, x:x + w] = state[k]
        origins.append((y, x))
    return out, origins


def draw_state(ax, task, state, title, vmin, vmax, cmap):
    img, origins = montage(state)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.55")
    im = ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    for k, (y, x) in enumerate(origins):
        label = "K" if k == task.conf_plane else str(k)
        ax.text(x - 0.4, y - 0.6, label, fontsize=5.5, color="black",
                ha="left", va="bottom")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def render(task, inst, path):
    seed = task.encode(inst)
    target = task.target(inst)
    clamp = task.clamp_mask(inst)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    draw_tileset(axes[0, 0], task)
    draw_tiling(axes[0, 1], task, inst.givens, inst.givens,
                f"instance: {inst.n_free} free cells inside a given ring\n"
                f"(black outline = given, grey = free)")
    draw_tiling(axes[0, 2], task, inst.solution, inst.givens,
                f"a solution (reference solver, {inst.work} node expansions)")

    panels = [
        (axes[1, 0], seed, -1, 1, "coolwarm",
         f"encode(): seed state, {task.n_planes} planes of {task.rows}x{task.cols}\n"
         f"plane t = belief in tile t; free cells at BACKGROUND=0"),
        (axes[1, 1], target, -1, 1, "coolwarm",
         "target(): every cell committed\n(supervised targets only)"),
        (axes[1, 2], clamp, 0, 1, "gray_r",
         "clamp_mask() = hold_mask(): the audited givens\n"
         "no structural planes, so the audit is undiluted"),
    ]
    for ax, img, vmin, vmax, cmap, title in panels:
        im = draw_state(ax, task, img, title, vmin, vmax, cmap)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(task.describe(), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


# ── checks ───────────────────────────────────────────────────────────────────

class Checks:
    def __init__(self):
        self.fails = 0

    def __call__(self, name, ok, detail=""):
        self.fails += 0 if ok else 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def run_checks(task, rng):
    ck = Checks()
    inst = task.sample(1, rng, args.difficulty)[0]
    seed = torch.from_numpy(task.encode(inst))
    target = torch.from_numpy(task.target(inst))
    clamp = torch.from_numpy(task.clamp_mask(inst))

    print(f"\n{task.describe()}")
    print(f"instance: {inst.n_free} free cells, reference work {inst.work}, "
          f"{task.deduction_rounds(inst)} synchronous AC rounds\n")

    print("shape contract")
    ck("state_shape == (n_planes, H, W)",
       task.state_shape == (task.n_planes, *task.shape), f"{task.state_shape}")
    ck("encode() matches state_shape", seed.shape == task.state_shape)
    ck("clamp_mask() matches state_shape", clamp.shape == task.state_shape)
    ck("hold_mask() is clamp_mask() (no structural planes)",
       np.array_equal(task.hold_mask(inst), task.clamp_mask(inst)))

    print("\nencode / decode")
    ck("decode(target) == solution", task.decode(target) == inst.solution)
    dec_seed = task.decode(seed)
    ck("decode(seed) agrees with the givens",
       all(dec_seed[k] == g for k, g in enumerate(inst.givens) if g >= 0))
    noisy = target + 0.20 * torch.randn(target.shape, generator=torch.Generator().manual_seed(0))
    ck("decode survives 0.20 noise on the target", task.decode(noisy) == inst.solution)
    ck("decode is a plain tuple of ints (repr-stable for count_distinct)",
       isinstance(task.decode(target), tuple)
       and "..." not in repr(task.decode(target))
       and len(repr(task.decode(target))) > 2 * len(inst.solution))
    frames = torch.stack([seed, target])
    ck("count_distinct separates seed from target",
       count_distinct(frames, task) == 2, f"got {count_distinct(frames, task)}")

    print("\nclamp")
    n_given = sum(1 for g in inst.givens if g >= 0)
    expect = n_given * task.n_planes * task.pitch ** 2
    ck("clamp covers exactly the given cells, all planes",
       int(clamp.sum()) == expect, f"{int(clamp.sum())} vs {expect}")
    ck("clamp is empty on free cells",
       all(float(clamp[(slice(None),) + task.cell_block(k)].sum()) == 0
           for k, g in enumerate(inst.givens) if g < 0))
    ck("seed == target wherever clamped",
       bool((((seed - target).abs() * clamp) < 1e-6).all()))
    cand = [task.random_candidate(inst, rng) for _ in range(200)]
    ck("random_candidate respects the clamp (200 draws)",
       all(c[k] == g for c in cand for k, g in enumerate(inst.givens) if g >= 0))
    free = task.interior_cells()[0]
    ck("random_candidate actually varies the free cells",
       len({c[free] for c in cand}) > 1, f"{len({c[free] for c in cand})} values seen")
    ck("givens_preserved = 1.0 on a clamped rollout",
       givens_preserved(torch.stack([seed, target]), seed, clamp) == 1.0)
    wrecked = target.clone()
    for k, g in enumerate(inst.givens):
        if g >= 0:
            task._stamp_cell(wrecked.numpy(), k, (g + 1) % task.T)
            break
    ck("the confidence plane K is inside the audit",
       float(clamp[task.conf_plane].sum()) > 0)
    ck("givens_preserved < 1.0 when a given is overwritten",
       givens_preserved(torch.stack([seed, wrecked]), seed, clamp) < 1.0,
       f"{givens_preserved(torch.stack([seed, wrecked]), seed, clamp):.3f}")

    print("\nexact semantics")
    ck("is_solved(solution)", task.is_solved(inst, inst.solution))
    ck("violations(solution) == 0", task.violations(inst, inst.solution) == 0)
    broken = list(inst.solution)
    broken[free] = (broken[free] + 1) % task.T
    ck("one changed interior cell breaks >= 1 edge",
       task.violations(inst, tuple(broken)) >= 1,
       f"{task.violations(inst, tuple(broken))} violations")
    overwrite = list(inst.solution)
    overwrite[task.border_cells()[0]] = (overwrite[task.border_cells()[0]] + 1) % task.T
    ck("overwriting a given counts as a violation",
       any(kind == "given" for kind, _, _ in violation_list(inst, tuple(overwrite))))
    sol, work = task.reference_solve(inst)
    ck("reference_solve returns a genuine solution", task.is_solved(inst, sol),
       f"work {work}")
    ok, decided, dom = task.propagate_only(inst)
    ck("instance is NOT decided by propagation alone", not decided,
       f"total domain {dom} over {inst.n_cells} cells")

    print("\nbaseline")
    base = matched_baseline(task, inst, n_visits=64, rng=rng, trials=64)
    ck("matched baseline over 64 visits is far below 1.0", base < 0.5, f"{base:.3f}")

    print("\nenergy")
    e_sol = float(task.energy(target, inst))
    e_seed = float(task.energy(seed, inst))
    e_flat = float(task.energy(torch.zeros(task.state_shape), inst))
    batch = torch.stack([seed, target, torch.zeros(task.state_shape)])
    e_batch = task.energy(batch, [inst, inst, inst])
    ck("batched energy: (B, P, H, W) + B instances -> (B,)",
       tuple(e_batch.shape) == (3,), f"{tuple(e_batch.shape)}")
    ck("batched agrees with unbatched",
       abs(float(e_batch[0]) - e_seed) < 1e-5 and abs(float(e_batch[1]) - e_sol) < 1e-5)
    ck("energy(solution) ~ 0", e_sol < 0.02, f"{e_sol:.4f}")
    ck("energy(seed) > energy(solution)", e_seed > e_sol, f"{e_seed:.4f} vs {e_sol:.4f}")
    ck("the FLAT state is not a minimum", e_flat > e_sol,
       f"flat {e_flat:.4f} ~ 1-1/C = {1 - 1 / task.C:.4f}, solution {e_sol:.4f}")
    grads = torch.zeros(task.state_shape, requires_grad=True)
    lo, _ = task.loss_terms(grads + seed, inst)
    lo.backward()
    per_plane = grads.grad.abs().sum(dim=(-1, -2))
    ck("confidence plane K gets NO gradient from the loss",
       float(per_plane[task.conf_plane]) == 0.0
       and float(per_plane[:task.T].min()) > 0.0,
       f"K {float(per_plane[task.conf_plane]):.2e}, "
       f"min belief plane {float(per_plane[:task.T].min()):.3f}")
    es, vs = [], []
    for _ in range(48):
        c = task.random_candidate(inst, rng)
        es.append(float(task.energy(torch.from_numpy(task.target(inst, c)), inst)))
        vs.append(task.violations(inst, c))
    rho = np.corrcoef(np.argsort(np.argsort(es)), np.argsort(np.argsort(vs)))[0, 1]
    ck("energy rank-correlates with the exact violation count", rho > 0.8, f"rho {rho:+.3f}")

    print("\naggregates: share of the energy range spent on ONE broken edge")
    print("  (committed beliefs; the range is solution -> uniformly random)")
    one = list(inst.solution)
    one[free] = _flip_one_edge(task, inst, free)
    print(f"  the minimal single-cell change available breaks "
          f"{task.edge_violations(inst, tuple(one))} edge(s) of "
          f"{len(task.edge_agreement(torch.from_numpy(task.target(inst)), inst))}")
    for agg in AGGREGATES:
        f_sol = torch.from_numpy(task.target(inst))
        f_one = torch.from_numpy(task.target(inst, tuple(one)))
        rand = [torch.from_numpy(task.target(inst, task.random_candidate(inst, rng)))
                for _ in range(16)]
        a = task.aggregate_energy(task.edge_agreement(f_sol, inst), aggregate=agg)
        b = task.aggregate_energy(task.edge_agreement(f_one, inst), aggregate=agg)
        c = np.mean([float(task.aggregate_energy(task.edge_agreement(x, inst), aggregate=agg))
                     for x in rand])
        share = (float(b) - float(a)) / max(1e-9, c - float(a))
        print(f"  {agg:8s} solution {float(a):7.4f}  one-broken {float(b):7.4f}  "
              f"random {c:7.4f}  share {share:5.1%}")

    print("\nundecided regime: rank agreement with the exact violation count")
    print("  (beliefs at a quarter magnitude, straight out of a noise seed)")
    for agg in AGGREGATES:
        es, vs = [], []
        for _ in range(48):
            c = task.random_candidate(inst, rng)
            f = torch.from_numpy(task.target(inst, c)) * 0.25
            es.append(float(task.aggregate_energy(task.edge_agreement(f, inst), aggregate=agg)))
            vs.append(task.edge_violations(inst, c))
        rho = np.corrcoef(np.argsort(np.argsort(es)), np.argsort(np.argsort(vs)))[0, 1]
        print(f"  {agg:8s} rho {rho:+.3f}")

    print(f"\n{'ALL CHECKS PASSED' if ck.fails == 0 else str(ck.fails) + ' CHECK(S) FAILED'}")
    return ck.fails


def _flip_one_edge(task, inst, cell):
    """A tile for ``cell`` that breaks as few edges as possible (ideally one)."""
    best, best_v = None, None
    for t in range(task.T):
        if t == inst.solution[cell]:
            continue
        c = list(inst.solution)
        c[cell] = t
        v = task.edge_violations(inst, tuple(c))
        if best_v is None or v < best_v:
            best, best_v = t, v
    return best


# ── survey ───────────────────────────────────────────────────────────────────

SURVEY_HEADER = (f"{'region':>7} {'C':>2} {'T':>3} {'alpha':>5} {'border':>8} "
                 f"{'accept':>7} {'prop':>6} {'q10':>5} {'med':>6} {'q90':>6} "
                 f"{'max':>8} {'uniq':>6} {'rollout':>8}")


def survey_row(rows, cols, colors, tiles, border, n, budget_s, count_unique=True):
    """One ensemble, measured.  Returns nothing; prints the row."""
    works, prop, uniq, seen, drawn = [], 0, 0, 0, 0
    t0 = time.time()
    task = None
    for tseed in range(6):
        task = build(rows=rows, cols=cols, tiles=tiles, colors=colors,
                     tseed=1000 + tseed)
        cfg = WangDifficulty(border=border, min_work=0)
        r = random.Random(1009 * tseed + 7)
        got = 0
        while got < max(1, n // 6) and drawn < 400 * (tseed + 1):
            if time.time() - t0 > budget_s:
                break
            drawn += 1
            inst = task.draw_instance(r, cfg)
            if inst is None:
                continue
            seen += 1
            got += 1
            works.append(inst.work)
            prop += int(task.propagate_only(inst)[1])
            if count_unique:
                uniq += int(task.solver.count(inst, cap=2) == 1)
        if time.time() - t0 > budget_s:
            break
    head = (f"{f'{rows}x{cols}':>7} {colors:>2} {tiles:>3} {tiles / colors ** 2:>5.2f} "
            f"{border:>8}")
    if not works:
        print(f"{head} {seen / max(1, drawn):>7.3f}   (no satisfiable instance drawn "
              f"in {drawn} tries)")
        return
    w = np.array(works)
    print(f"{head} {seen / drawn:>7.3f} {prop / seen:>6.2f} "
          f"{np.percentile(w, 10):>5.0f} {np.median(w):>6.0f} {np.percentile(w, 90):>6.0f} "
          f"{w.max():>8.0f} {uniq / seen:>6.2f} {task.min_rollout():>8d}")


def survey(rng):
    print("Difficulty of the generated instances, measured with the reference solver.")
    print("work   = node expansions (tile assignments tried) — the difficulty scale,")
    print("         the role WalkSAT flips played for 3SAT")
    print("prop   = fraction decided by arc consistency ALONE.  Those are warm-ups,")
    print("         not instances; a tile set with prop ~ 1 is disqualified")
    print("accept = acceptance rate of the ensemble (part of its description)")
    print("uniq   = fraction with exactly one solution (supervised targets need this)")
    print()
    print("alpha sweep, at the configured region")
    print(SURVEY_HEADER)
    print("-" * len(SURVEY_HEADER))
    for colors, tiles in [(3, 6), (3, 8), (3, 10), (3, 12), (4, 16), (4, 20), (5, 25)]:
        survey_row(args.rows, args.cols, colors, tiles, "planted", args.instances, 12.0)
    print()
    print("region ladder, at the default tile set (the curriculum axis)")
    print(SURVEY_HEADER)
    print("-" * len(SURVEY_HEADER))
    for n in (6, 8, 10, 12, 16):
        survey_row(n, n, args.colors, args.tiles, "planted", args.instances, 15.0,
                   count_unique=n <= 10)
    print()
    print("reconciliation depth: SYNCHRONOUS arc-consistency rounds to fixpoint")
    print("(the schedule an NCA step actually uses; grounds the propagation part")
    print(" of min_rollout, which is rounds x 2 x diameter)")
    print(f"{'region':>7} {'rounds q10':>11} {'med':>5} {'q90':>5} {'max':>5} "
          f"{'-> steps (med/max)':>20} {'critical path':>14} {'floor':>6}")
    for n in (6, 8, 10, 12):
        task = build(rows=n, cols=n)
        rounds = []
        for tseed in range(4):
            t2 = build(rows=n, cols=n, tseed=1000 + tseed)
            r = random.Random(97 * tseed + 3)
            rounds += [t2.deduction_rounds(i)
                       for i in t2.sample(max(1, args.instances // 4), r, "easy")]
        a = np.array(rounds)
        d2 = 2 * task.diameter()
        print(f"{f'{n}x{n}':>7} {np.percentile(a, 10):>11.0f} {np.median(a):>5.0f} "
              f"{np.percentile(a, 90):>5.0f} {a.max():>5.0f} "
              f"{f'{np.median(a) * d2:.0f} / {a.max() * d2:.0f}':>20} "
              f"{task.min_rollout():>14d} {task.min_rollout_floor():>6d}")
    print()
    print("planting bias: the same region drawn from an unbiased border ensemble")
    print(SURVEY_HEADER)
    print("-" * len(SURVEY_HEADER))
    for border in ("planted", "random"):
        survey_row(args.rows, args.cols, args.colors, args.tiles, border,
                   args.instances, 20.0)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(args.seed)
    task = build()
    if args.survey:
        survey(rng)
        return 0
    if args.check:
        return 1 if run_checks(task, rng) else 0
    inst = task.sample(1, rng, args.difficulty)[0]
    print(task.describe())
    print(f"instance: {inst.n_free} free cells, reference work {inst.work}")
    render(task, inst, ROOT / args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
