#!/usr/bin/env python3
"""Eyeball (and interrogate) the Sudoku relaxation task — no model needed.

Renders exactly what ``ncpu.tasks.sudoku.SudokuTask`` hands the NCA: the puzzle,
its unique solution, the clamp mask that must survive the rollout, and all
``N + 3`` encoded planes — the nine digit-belief planes are the pencil marks, so
the picture *is* the state.

``--check`` is the part worth running before any GPU time.  It is
``docs/karp_eval_protocol.md`` Tier 0 for this track, and it asserts:

* encode -> decode round-trips the clues, and round-trips a full grid exactly;
* every generated puzzle is **proper** — ``count_solutions(cap=2) == 1``;
* ``random_candidate`` respects the clues in both modes, so the baseline is
  measuring the same problem the model is;
* the clamp audit actually fires — a rollout that overwrites one given must
  score below 1.0 on ``relax.givens_preserved``, or the audit is decorative;
* the energy orders solution < near-miss < random < uniform, and correlates
  with the exact violation count;
* the **flat-state trap**: with a plain mean and no purity term the uniform
  "believes everything" state has energy exactly 0.0, i.e. is a global minimum.
  This is why ``purity_weight`` must stay positive;
* difficulty: the fraction of instances a naked-single sweep alone finishes,
  which is the fraction on which a good score would mean nothing.

``--survey`` re-measures the difficulty distribution quoted in the module
docstring (~80 ms per puzzle, so 400 puzzles is about half a minute).

Outputs (project root by default):
    sudoku_preview.png    puzzle, solution, clamp mask, and every encoded plane

Usage:
    uv run python scripts/preview_sudoku.py
    uv run python scripts/preview_sudoku.py --difficulty search --check
    uv run python scripts/preview_sudoku.py --check --survey --survey-n 400
"""

import random
import sys
import time
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ncpu.relax import givens_preserved, matched_baseline
from ncpu.tasks.sudoku import (
    CLASSES,
    SudokuInstance,
    SudokuTask,
    augment,
    carve,
    classify,
    count_solutions,
    generate,
    random_solution,
)

parser = ArgumentParser()
parser.add_argument("--n", type=int, default=3, help="box size; the grid is n*n a side")
parser.add_argument("--difficulty", default="search", choices=list(CLASSES) + ["any"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--check", action="store_true", help="Tier-0 assertions")
parser.add_argument("--check-n", type=int, default=24, help="instances per check")
parser.add_argument("--survey", action="store_true", help="re-measure the difficulty mix")
parser.add_argument("--survey-n", type=int, default=200)
parser.add_argument("--out-dir", default=str(ROOT))
args = parser.parse_args()

RNG = random.Random(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

task = SudokuTask(n=args.n, difficulty=args.difficulty, pool_size=0)
N = task.N
print(task.describe())
wrapped = SudokuTask(n=args.n, difficulty=args.difficulty, wrap=True).min_rollout()
print(f"min_rollout {task.min_rollout()} = 2 * (N-1) * depth = 2 * {N - 1} * {task.depth}"
      f"   [circular padding would give {wrapped}]")


# ── the picture ──────────────────────────────────────────────────────────────

def draw_grid(ax, grid, n, title, highlight=None, cmap_vals=None):
    """A Sudoku grid as text on a lightly shaded array, with box rules."""
    N = n * n
    g = np.asarray(grid)
    base = np.zeros((N, N)) if cmap_vals is None else np.asarray(cmap_vals, dtype=float)
    ax.imshow(base, cmap="Blues", vmin=0.0, vmax=1.6, interpolation="nearest")
    for r in range(N):
        for c in range(N):
            v = int(g[r, c])
            if v == 0:
                continue
            bold = highlight is not None and highlight[r, c]
            ax.text(c, r, str(v), ha="center", va="center",
                    fontsize=9 if N > 9 else 11,
                    fontweight="bold" if bold else "normal",
                    color="black" if bold else "0.45")
    for k in range(N + 1):
        lw = 1.6 if k % n == 0 else 0.4
        ax.axhline(k - 0.5, color="0.2", lw=lw)
        ax.axvline(k - 0.5, color="0.2", lw=lw)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


inst = generate(args.n, RNG, args.difficulty)
seed = task.encode(inst)
clamp = task.clamp_mask(inst)
given = np.asarray(inst.puzzle) > 0
_, work = task.reference_solve(inst)

print(f"\ninstance: {inst}   clues {inst.n_clues}/{N * N}   reference work {work} nodes")
print(inst.ascii())

fig = plt.figure(figsize=(11, 2.6 + 1.5 * ((N + 3 + 5) // 6)))
gs = fig.add_gridspec(1 + (N + 3 + 5) // 6, 6, height_ratios=[2.4] + [1.0] * ((N + 3 + 5) // 6))

ax = fig.add_subplot(gs[0, 0:2])
draw_grid(ax, inst.puzzle, args.n, f"puzzle ({inst.n_clues} clues, {inst.difficulty})",
          highlight=given)
ax = fig.add_subplot(gs[0, 2:4])
draw_grid(ax, inst.solution, args.n, "unique solution (clues bold)", highlight=given)
ax = fig.add_subplot(gs[0, 4:6])
draw_grid(ax, np.zeros_like(inst.puzzle), args.n,
          "clamp mask: 1 = held on all digit planes",
          cmap_vals=clamp[0:N].max(axis=0))

labels = ([f"belief d={d + 1}" for d in range(N)]
          + ["given/confidence", "box phase x", "box phase y"])
for k in range(N + 3):
    ax = fig.add_subplot(gs[1 + k // 6, k % 6])
    ax.imshow(seed[k], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(labels[k], fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(f"{task.describe()}   —   +1 circled, -1 crossed out, 0 still open",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = Path(args.out_dir) / "sudoku_preview.png"
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"\nwrote {out}")


# ── Tier 0 ───────────────────────────────────────────────────────────────────

FAILURES = []


def check(label, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)
    return ok


def as_tuple(grid):
    return tuple(tuple(int(v) for v in row) for row in np.asarray(grid))


if args.check:
    K = args.check_n
    print(f"\n── Tier 0: well-posedness ({K} instances) " + "─" * 24)
    t0 = time.time()
    insts = [generate(args.n, RNG, args.difficulty) for _ in range(K)]
    print(f"  generated in {time.time() - t0:.1f}s "
          f"({1000 * (time.time() - t0) / K:.0f} ms each)")

    # 1. encode / decode round trip
    clue_ok = full_ok = plane_ok = True
    for ins in insts:
        s = torch.from_numpy(task.encode(ins))
        dec = task.decode(s)
        p = np.asarray(ins.puzzle)
        clue_ok &= all(dec[r][c] == p[r, c]
                       for r in range(N) for c in range(N) if p[r, c] > 0)
        whole = SudokuInstance(ins.n, np.asarray(ins.solution), np.asarray(ins.solution))
        full_ok &= task.decode(torch.from_numpy(task.encode(whole))) == as_tuple(ins.solution)
        plane_ok &= task.encode(ins).shape == (task.n_planes, N, N)
    check("encode->decode round-trips every clue", clue_ok)
    check("encode->decode round-trips a full grid exactly", full_ok)
    check(f"encode shape is ({task.n_planes}, {N}, {N})", plane_ok)

    # free cells must read as "everything still possible", not as a digit
    free_seed = task.encode(insts[0])[0:N][:, ~(np.asarray(insts[0].puzzle) > 0)]
    check("free cells sit at BACKGROUND on every digit plane",
          bool(np.all(free_seed == 0.0)),
          "(a decoded free cell is an argmax tie, i.e. an honest coin flip)")

    # 2. uniqueness — the puzzles are proper
    counts = Counter(count_solutions(ins.puzzle, ins.n, cap=2) for ins in insts)
    check("every generated puzzle has exactly one solution", set(counts) == {1},
          f"counts={dict(counts)}")
    check("the stored solution is the solution the solver finds",
          all(task.reference_solve(ins)[0] == as_tuple(ins.solution) for ins in insts))

    # 3. random_candidate respects the clues
    for mode in ("uniform", "propagate"):
        ok, viols = True, []
        for ins in insts:
            cand = task.random_candidate(ins, RNG, mode=mode)
            p = np.asarray(ins.puzzle)
            ok &= all(cand[r][c] == p[r, c]
                      for r in range(N) for c in range(N) if p[r, c] > 0)
            viols.append(task.violations(ins, cand))
        check(f"random_candidate(mode={mode!r}) respects the clues", ok,
              f"mean violations {np.mean(viols):.1f}")

    # the matched baseline, and why it is uninformative here
    ins = insts[0]
    base = matched_baseline(task, ins, n_visits=1000, rng=RNG, trials=8)
    n_free = N * N - ins.n_clues
    check("matched_baseline over 1000 visits is ~0", base == 0.0,
          f"= {base:.3f}; a uniform candidate is 1 in {N}^{n_free} ~ 1e{n_free * np.log10(N):.0f}, "
          "so LIFT OVER THIS IS NOT EVIDENCE — use mode='propagate' and the "
          "reference node count as the real comparisons")

    # 4. the clamp audit must actually fire
    clean = torch.from_numpy(np.stack([task.encode(ins)] * 4))
    seed_t = torch.from_numpy(task.encode(ins))
    clamp_t = torch.from_numpy(task.clamp_mask(ins))
    ok_clean = givens_preserved(clean, seed_t, clamp_t)
    tampered = clean.clone()
    r, c = np.argwhere(np.asarray(ins.puzzle) > 0)[0]
    tampered[-1, :N, r, c] = -tampered[-1, :N, r, c]      # flip that clue's marks
    ok_tampered = givens_preserved(tampered, seed_t, clamp_t)
    check("clamp audit passes an untouched rollout", ok_clean == 1.0, f"= {ok_clean:.3f}")
    check("clamp audit catches ONE overwritten clue", ok_tampered < 1.0,
          f"= {ok_tampered:.4f} (1 clue of {int(clamp_t.sum())} clamped values)")
    check("clamp_mask marks only clue cells on digit planes",
          int(clamp_t.sum()) == ins.n_clues * N and float(clamp_t[N:].sum()) == 0.0)
    check("structure_mask covers the 3 structural planes and nothing else",
          float(task.structure_mask().sum()) == 3 * N * N
          and float(task.structure_mask()[:N].sum()) == 0.0)

    # 5. the energy is well-posed
    sol_state = torch.from_numpy(task.encode(
        SudokuInstance(ins.n, np.asarray(ins.solution), np.asarray(ins.solution))))
    seed_state = torch.from_numpy(task.encode(ins))
    flat_state = seed_state.clone()
    flat_state[:N] = 0.0
    near = np.asarray(ins.solution).copy()
    free = np.argwhere(np.asarray(ins.puzzle) == 0)
    (r1, c1), (r2, c2) = free[0], free[1]
    near[r1, c1], near[r2, c2] = near[r2, c2], near[r1, c1]
    near_state = torch.from_numpy(task.encode(
        SudokuInstance(ins.n, near, np.asarray(ins.solution))))
    rand_state = torch.from_numpy(task.encode(SudokuInstance(
        ins.n, np.array(task.random_candidate(ins, RNG, "uniform"), dtype=np.uint8),
        np.asarray(ins.solution))))
    es = {k: float(task.energy(v, ins)) for k, v in
          [("solution", sol_state), ("near-miss (2 cells swapped)", near_state),
           ("random uniform fill", rand_state), ("seed (clues only)", seed_state),
           ("uniform beliefs", flat_state)]}
    for k, v in es.items():
        print(f"        E[{k:28s}] = {v:.4f}")
    check("energy(solution) is the smallest",
          es["solution"] == min(es.values()))
    check("energy(solution) < energy(near-miss) < energy(random)",
          es["solution"] < es["near-miss (2 cells swapped)"] < es["random uniform fill"])

    # the flat-state trap, measured
    trap = SudokuTask(n=args.n, aggregate="mean", purity_weight=0.0,
                      given_weight=0.0, pool_size=0)
    e_trap_flat = float(trap.energy(flat_state, ins))
    e_trap_sol = float(trap.energy(sol_state, ins))
    check("THE TRAP: unit residuals alone are ~0 at uniform beliefs",
          e_trap_flat <= e_trap_sol + 1e-9,
          f"E_mean,no-purity(uniform)={e_trap_flat:.2e} <= E(solution)={e_trap_sol:.2e} "
          "— keep purity_weight > 0")
    check("with purity, uniform beliefs are NOT a minimum",
          es["uniform beliefs"] > es["solution"],
          f"{es['uniform beliefs']:.4f} > {es['solution']:.4f}")

    # energy tracks the exact violation count
    xs, ys = [], []
    for _ in range(64):
        cand = np.asarray(task.random_candidate(ins, RNG, "propagate"), dtype=np.uint8)
        st = torch.from_numpy(task.encode(SudokuInstance(ins.n, cand, cand)))
        xs.append(task.violations(ins, cand))
        ys.append(float(task.energy(st, ins)))
    xr = np.argsort(np.argsort(xs)).astype(float)
    yr = np.argsort(np.argsort(ys)).astype(float)
    rho = float(np.corrcoef(xr, yr)[0, 1])
    check("energy rank-correlates with the exact violation count", rho > 0.5,
          f"spearman rho = {rho:+.3f} over 64 constraint-respecting fills")

    # the worst-constraint aggregate vs a plain mean
    worst = SudokuTask(n=args.n, aggregate="softmax", pool_size=0)
    plain = SudokuTask(n=args.n, aggregate="mean", pool_size=0)
    one_bad = torch.from_numpy(task.encode(
        SudokuInstance(ins.n, near, np.asarray(ins.solution))))
    span_w = (float(worst.energy(one_bad, ins)) - float(worst.energy(sol_state, ins))) \
        / max(float(worst.energy(rand_state, ins)) - float(worst.energy(sol_state, ins)), 1e-9)
    span_m = (float(plain.energy(one_bad, ins)) - float(plain.energy(sol_state, ins))) \
        / max(float(plain.energy(rand_state, ins)) - float(plain.energy(sol_state, ins)), 1e-9)
    print(f"        one swapped pair costs {span_w:.1%} of the energy range under "
          f"softmax(tau={task.tau:g}) vs {span_m:.1%} under mean")
    check("the smooth max puts more of the range on a single broken constraint",
          span_w > span_m)

    # 6. exactness of is_solved
    ok = True
    for ins2 in insts[:8]:
        ok &= task.is_solved(ins2, ins2.solution)
        ok &= task.violations(ins2, ins2.solution) == 0
        other = np.asarray(ins2.solution).copy()
        free2 = np.argwhere(np.asarray(ins2.puzzle) == 0)[0]
        other[free2[0], free2[1]] = 1 + (other[free2[0], free2[1]] % N)
        ok &= not task.is_solved(ins2, other)
    check("is_solved accepts the solution and rejects a one-cell change", ok)
    # a valid grid that ignores the clues must NOT count as solved
    foreign = random_solution(args.n, RNG)
    conflict = any(foreign[r, c] != insts[0].puzzle[r, c]
                   for r in range(N) for c in range(N) if insts[0].puzzle[r, c] > 0)
    check("is_solved rejects a valid grid that contradicts the clues",
          (not conflict) or (not task.is_solved(insts[0], foreign)),
          "(a solver that answers some other legal grid solved another puzzle)")

    # 7. difficulty: what fraction needs no search at all
    cls = Counter(classify(ins2.puzzle, ins2.n)[0] for ins2 in insts)
    trivial = cls.get("naked_single", 0) / K
    print(f"        classes: {dict(cls)}")
    check(f"< 5% of difficulty={args.difficulty!r} instances are naked-single-only",
          trivial < 0.05,
          f"trivial fraction {trivial:.1%} — these need no search at all, so a "
          "score on them is not a solving result.  (n=2 fails this by nature: "
          "every proper 4x4 puzzle falls to naked singles.)")
    if args.difficulty == "search":
        works = [task.reference_solve(i2)[1] for i2 in insts]
        check("every 'search' instance costs the reference solver > 0 branches",
              min(works) > 0, f"nodes med {np.median(works):.0f} max {max(works)}")

    # 8. augmentation preserves class, depth and properness
    ok_cls = ok_uniq = ok_exact = True
    depth_shift = 0
    for ins2 in insts[:8]:
        a = augment(ins2, RNG)
        c2, d2, _ = classify(a.puzzle, a.n)
        ok_cls &= (c2 == ins2.difficulty)
        depth_shift += (d2 != ins2.depth)
        ok_uniq &= count_solutions(a.puzzle, a.n, cap=2) == 1
        b = augment(ins2, RNG, recompute=True)
        ok_exact &= classify(b.puzzle, b.n) == (b.difficulty, b.depth, b.nodes)
    check("augment preserves the difficulty CLASS", ok_cls)
    check("augment preserves properness", ok_uniq)
    check("augment(recompute=True) reports the augmented instance's own depth", ok_exact,
          f"({depth_shift}/8 copied depths were stale — the search path is not "
          "symmetry-invariant, the class is)")

    # 9. shortcut check: no cell/digit is privileged by the generator
    tally = np.zeros((N, N))
    for ins2 in insts:
        for r in range(N):
            for c in range(N):
                tally[r, c] += int(np.asarray(ins2.puzzle)[r, c] > 0)
    rate = tally / K
    pbar = float(rate.mean())
    # the floor is sampling noise, not a magic constant: with K puzzles the
    # per-cell clue rate has binomial sd sqrt(p(1-p)/K), so anything under 3 sd
    # is indistinguishable from a generator with no positional preference.
    noise = float(np.sqrt(max(pbar * (1 - pbar), 1e-9) / K))
    check("clue placement has no positional bias beyond sampling noise",
          float(rate.std()) < 3 * noise,
          f"clue rate {pbar:.2f} +- {rate.std():.2f} vs 3sd = {3 * noise:.2f} "
          f"(min {rate.min():.2f} max {rate.max():.2f})")

    # 10. light cone
    check("min_rollout exceeds the grid diameter by the deduction depth",
          task.min_rollout() >= 2 * (N - 1),
          f"{task.min_rollout()} steps vs a {N - 1}px diameter — "
          f"{task.min_rollout() / (N - 1):.0f}x")

    print(f"\n  {len(FAILURES)} failure(s)" + (": " + ", ".join(FAILURES) if FAILURES else ""))


# ── difficulty survey ────────────────────────────────────────────────────────

if args.survey:
    M = args.survey_n
    print(f"\n── difficulty survey: {M} unconstrained minimal carves " + "─" * 12)
    t0 = time.time()
    recs = []
    for _ in range(M):
        sol = random_solution(args.n, RNG)
        puz = carve(sol, args.n, RNG, keep_class="any")
        recs.append(classify(puz, args.n) + (int((puz > 0).sum()),))
    el = time.time() - t0
    print(f"  {M} puzzles in {el:.1f}s ({1000 * el / M:.0f} ms each)")
    print(f"  {'class':14s} {'share':>7s}  {'depth med':>9s} {'p90':>4s} {'max':>4s}"
          f"  {'nodes med':>9s} {'p90':>5s} {'max':>5s}  {'clues':>5s}")
    for cls_name in CLASSES:
        sub = [r for r in recs if r[0] == cls_name]
        if not sub:
            print(f"  {cls_name:14s} {0:7.1%}")
            continue
        d = np.array([r[1] for r in sub], float)
        nd = np.array([r[2] for r in sub], float)
        cl = np.array([r[3] for r in sub], float)
        print(f"  {cls_name:14s} {len(sub) / M:7.1%}  {np.median(d):9.0f} "
              f"{np.percentile(d, 90):4.0f} {d.max():4.0f}  {np.median(nd):9.0f} "
              f"{np.percentile(nd, 90):5.0f} {nd.max():5.0f}  {np.median(cl):5.0f}")
    triv = sum(1 for r in recs if r[0] == "naked_single") / M
    print(f"\n  naked-singles-only fraction: {triv:.1%}  <- instances on which a good "
          "score demonstrates nothing")
    print("  min_rollout implied by each class's depth p90:")
    for cls_name in CLASSES:
        sub = [r[1] for r in recs if r[0] == cls_name]
        if sub:
            p90 = int(np.percentile(sub, 90))
            print(f"    {cls_name:14s} depth {p90:3d} -> {2 * (N - 1) * p90:5d} steps "
                  f"(wrap: {2 * (N // 2) * p90:5d})")

sys.exit(1 if FAILURES else 0)
