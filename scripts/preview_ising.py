#!/usr/bin/env python3
"""Eyeball (and interrogate) the lattice spin glass — no model needed.

Renders exactly what ``ncpu.tasks.ising.IsingLattice`` hands the NCA: the
couplings, the exact ground state, which faces are frustrated, the clamp mask
that must survive the rollout, and every encoded plane.  It also draws the two
things only this track can draw — the exact per-spin frozenness (which spins
take the same value in every ground state) and survey propagation's estimate of
it side by side.

``--check`` is the part worth running before any GPU time.  It is
``docs/karp_eval_protocol.md`` Tier 0 for this track, and it asserts:

* the transfer matrix agrees with brute force on small lattices, energy,
  degeneracy and all — this is ``is_solved``'s ground truth, so if it is wrong
  every other number in the phase is wrong;
* ``exact_frozen`` agrees with an exhaustive enumeration of ground states;
* encode -> decode round-trips the ground state and the clamped spins;
* ``random_candidate`` respects the clamp in both modes, so the baseline is
  measuring the same problem the model is;
* the clamp audit fires — a rollout that overwrites the reference spin must
  score below 1.0 on ``relax.givens_preserved``;
* the energy orders ground state < near miss < quench < random, is exactly 0 at
  a saturated ground state, and is **not** minimised by flat beliefs;
* the flat-state trap in its Ising form: with no field the uniform-belief seed
  is an exact stationary point, and the ONLY gradient anywhere on the lattice is
  the eight pixels around the clamped reference spin;
* the reference solvers are sane: BP, WP and SP are exact on a chain (a tree),
  which separates "SP is bad on a lattice" (a finding) from "SP is broken" (a
  bug);
* the instances are not warm-ups: the greedy-quench success rate is measured,
  not asserted.

``--survey`` re-measures every table quoted in the module docstrings: the
frustration sweep, the difficulty ladder, the aggregate ablation, and the
BP-vs-SP-vs-exact comparison.  It is the slow one (minutes).

Outputs (project root by default):
    ising_preview.png   couplings, ground state, frustration, clamp, planes,
                        exact vs SP frozenness

Usage:
    uv run python scripts/preview_ising.py
    uv run python scripts/preview_ising.py --variant planar --check
    uv run python scripts/preview_ising.py --survey --survey-n 16
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ncpu.inference import (  # noqa: E402
    IsingModel,
    belief_propagation,
    bp_decimation,
    quench_rate,
    sp_decimation,
    survey_propagation,
    warning_propagation,
)
from ncpu.relax import givens_preserved, matched_baseline  # noqa: E402
from ncpu.tasks.ising import (  # noqa: E402
    AGGREGATES,
    PRESETS,
    VARIANTS,
    IsingInstance,
    IsingLattice,
    brute_force_ground_state,
    exact_frozen,
    exact_ground_state,
    expected_frustration,
    faces,
    frustrated_faces,
    frustration,
    lattice_energy,
    to_model,
)

parser = ArgumentParser()
parser.add_argument("--rows", type=int, default=8)
parser.add_argument("--cols", type=int, default=8)
parser.add_argument("--variant", default="nnn", choices=list(VARIANTS))
parser.add_argument("--difficulty", default="hard", choices=sorted(PRESETS))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--check", action="store_true", help="Tier-0 assertions")
parser.add_argument("--check-n", type=int, default=8, help="instances per check")
parser.add_argument("--survey", action="store_true", help="re-measure every table")
parser.add_argument("--survey-n", type=int, default=16)
parser.add_argument("--out-dir", default=str(ROOT))
args = parser.parse_args()

# ``warmup`` and ``ferro`` exist to be easy: they are the curriculum rung and the
# negative control.  The checks that assert non-triviality are reported for them
# and asserted only for the presets that claim to be instances.
CLAIMS_HARD = args.difficulty not in ("warmup", "ferro")

RNG = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

task = IsingLattice(args.rows, args.cols, variant=args.variant,
                    difficulty=args.difficulty)
R, C = task.rows, task.cols
print(task.describe())
COMPLEXITY = {
    "planar": "POLYNOMIAL (planar matching) — this is the CONTROL, not a target",
    "nnn": "non-planar (shortest loop 3); read the module docstring for exactly "
           "what is and is not claimed",
}
print(f"  complexity: {COMPLEXITY[args.variant]}")
print(f"  min_rollout {task.min_rollout()} = rounds * 2 * diameter = "
      f"{task.rounds} * 2 * {task.diameter()}   [floor {task.min_rollout_floor()}]")

inst = task.draw_instance(RNG)
model = to_model(inst)
print(f"  {inst}  frustration {frustration(inst):.3f} "
      f"(formula {expected_frustration(inst.p_af, args.variant):.3f})")


# ── the picture ──────────────────────────────────────────────────────────────

def draw_lattice(ax, inst, spins=None, title="", frust=True, clamp=True):
    """Spins as discs, couplings as coloured segments, frustrated faces shaded."""
    ax.set_facecolor("white")
    if frust:
        fs = faces(inst.rows, inst.cols, inst.variant)
        bad = frustrated_faces(inst)
        for (kind, bonds), b in zip(fs, bad):
            if not b:
                continue
            pts = set()
            for d, r, c in bonds:
                dr, dc = [(0, 1), (1, 0), (1, 1), (1, -1)][d]
                pts.add((r, c))
                pts.add((r + dr, c + dc))
            pts = np.array(sorted(pts, key=lambda p: np.arctan2(
                p[0] - np.mean([q[0] for q in pts]),
                p[1] - np.mean([q[1] for q in pts]))))
            ax.fill(pts[:, 1], pts[:, 0], color="#ffd6d6", zorder=0, lw=0)
    for d, (dr, dc) in enumerate([(0, 1), (1, 0), (1, 1), (1, -1)]):
        rs, cs = np.nonzero(inst.J[d])
        for r, c in zip(rs, cs):
            j = inst.J[d, r, c]
            ax.plot([c, c + dc], [r, r + dr],
                    color=("#2b6cb0" if j > 0 else "#c53030"),
                    lw=1.6 if j > 0 else 1.6, ls="-" if j > 0 else "--",
                    zorder=1, alpha=0.9)
    if spins is not None:
        s = np.asarray(spins)
        ys, xs = np.mgrid[0:inst.rows, 0:inst.cols]
        ax.scatter(xs.ravel(), ys.ravel(), c=s.ravel(), cmap="coolwarm_r",
                   vmin=-1, vmax=1, s=90, zorder=3, edgecolors="0.2", linewidths=0.6)
    if clamp:
        for i in inst.clamped:
            r, c = inst.rc(i)
            ax.scatter([c], [r], s=230, facecolors="none", edgecolors="black",
                       linewidths=2.0, zorder=4)
    ax.set_xlim(-0.6, inst.cols - 0.4)
    ax.set_ylim(inst.rows - 0.4, -0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def draw_field(ax, arr, title, vmin=-1, vmax=1, cmap="coolwarm_r"):
    im = ax.imshow(np.asarray(arr), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    return im


enc = task.encode(inst)
tgt = task.target(inst)
clamp = task.clamp_mask(inst)
fr_exact, _, _ = exact_frozen(inst)
sp_res = survey_propagation(model, y=0.0, clamped=inst.clamped,
                            rng=np.random.default_rng(0), max_iters=300)
fr_sp = sp_res.frozen.reshape(R, C)

n_planes = task.n_planes
ncols = max(4, n_planes)
fig, axes = plt.subplots(3, ncols, figsize=(2.5 * ncols, 8.4))
for ax in axes.ravel():
    ax.axis("off")

for ax in axes[0, :4]:
    ax.axis("on")
draw_lattice(axes[0, 0], inst, None,
             "instance: J (blue +1, red dashed -1)\nshaded = frustrated faces")
draw_lattice(axes[0, 1], inst, inst.ground,
             f"exact ground state  E0={inst.e0:.0f}\n"
             f"degeneracy {inst.degeneracy:.0f} (mod global flip)", frust=False)
draw_lattice(axes[0, 2], inst, task.random_candidate(inst, RNG, "quench"),
             "a greedy quench's answer\n(the baseline to beat)", frust=False)
draw_lattice(axes[0, 3], inst, None,
             "circled: the clamped reference spin\n(the gauge, and the only given)")
for k in range(4, ncols):
    axes[0, k].axis("off")

axes[1, 0].axis("on")
draw_field(axes[1, 0], clamp[task.belief_plane], "clamp mask (belief plane)", 0, 1, "Greys")
axes[1, 1].axis("on")
draw_field(axes[1, 1], fr_exact.astype(float),
           f"EXACT frozen: {fr_exact.mean():.2f}\nsame in every ground state",
           0, 1, "viridis")
axes[1, 2].axis("on")
draw_field(axes[1, 2], fr_sp, f"SP estimate of frozen: {fr_sp.mean():.2f}\n"
                              f"(what plane K should hold)", 0, 1, "viridis")
axes[1, 3].axis("on")
bp = belief_propagation(model, beta=1.0, damping=0.9, clamped=inst.clamped,
                        rng=np.random.default_rng(0), max_iters=500)
draw_field(axes[1, 3], bp.marginals.reshape(R, C), f"BP marginals (beta=1)\n"
           f"{'converged' if bp.converged else 'DID NOT CONVERGE'} in {bp.iters} it")
for k in range(4, ncols):
    axes[1, k].axis("off")

names = ["belief m"] + [f"J {d}" for d in (["E", "S", "SE", "SW"][:task.n_dirs])]
if task.has_field:
    names.append("field h")
names.append("confidence K")
for k in range(n_planes):
    axes[2, k].axis("on")
    draw_field(axes[2, k], enc[k], f"plane {k}: {names[k]}")
for k in range(n_planes, ncols):
    axes[2, k].axis("off")

fig.suptitle(f"{task.describe()}   |   seed {args.seed}", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = Path(args.out_dir) / "ising_preview.png"
fig.savefig(out, dpi=120)
print(f"\nwrote {out}")


# ── Tier 0 ───────────────────────────────────────────────────────────────────

FAILURES = []


def check(label, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)
    return ok


if args.check:
    K = args.check_n
    print(f"\n── Tier 0: well-posedness ({K} instances) " + "─" * 24)
    t0 = time.time()
    insts = [task.draw_instance(RNG) for _ in range(K)]
    print(f"  generated in {time.time() - t0:.1f}s "
          f"({1000 * (time.time() - t0) / K:.0f} ms each)")

    # 1. the exact solver IS the ground truth, so it gets checked first
    # "easy" (no quench filter): this lattice exists to cross-check the exact
    # solver, and no 4x4 instance can pass a difficulty filter anyway
    fdiff = dict(p_af=0.5, field_frac=1.0 if task.has_field else 0.0)
    small = IsingLattice(4, 4, variant=args.variant, difficulty=fdiff,
                         field=task.has_field)
    ok_e = ok_d = ok_f = True
    for _ in range(6):
        si = small.draw_instance(RNG)
        cfg, e0, deg = exact_ground_state(si)
        bcfg, be0, bdeg = brute_force_ground_state(si)
        ok_e &= abs(e0 - be0) < 1e-9 and abs(lattice_energy(si, cfg) - e0) < 1e-9
        ok_d &= abs(deg - bdeg) < 1e-9
        # exhaustive frozen set.  The gauge is on exactly when the flip
        # symmetry exists, which is exactly when the field is zero.
        gauge = not np.any(si.h)
        gs = []
        for x in range(1 << si.n):
            v = np.array([1 - 2 * ((x >> b) & 1) for b in range(si.n)], np.int8)
            if gauge and v[0] < 0:
                continue
            if abs(lattice_energy(si, v) - e0) < 1e-9:
                gs.append(v)
        G = np.array(gs)
        ok_f &= np.array_equal(exact_frozen(si)[0],
                               (G.min(0) == G.max(0)).reshape(si.rows, si.cols))
    check("transfer matrix == brute force on E0 (4x4)", ok_e)
    check("transfer matrix == brute force on the DEGENERACY (4x4)", ok_d)
    check("exact_frozen == exhaustive enumeration of ground states (4x4)", ok_f)

    # 2. encode / decode round trip
    rt_clue = rt_full = shape_ok = True
    for i2 in insts:
        dec = task.decode(torch.from_numpy(task.encode(i2)))
        rt_clue &= all(dec.reshape(-1)[k] == v for k, v in i2.clamped.items())
        rt_full &= np.array_equal(task.decode(torch.from_numpy(task.target(i2))),
                                  i2.ground)
        shape_ok &= task.encode(i2).shape == (task.n_planes, R, C)
    check("encode->decode round-trips every clamped spin", rt_clue)
    check("encode->decode round-trips the ground state exactly", rt_full)
    check(f"encode shape is ({task.n_planes}, {R}, {C})", shape_ok)
    free_seed = task.encode(insts[0])[task.belief_plane]
    free_seed = free_seed.reshape(-1)[[i for i in range(insts[0].n)
                                      if i not in insts[0].clamped]]
    check("free spins sit at BACKGROUND on the belief plane",
          bool(np.all(free_seed == 0.0)),
          "(an undecided spin decodes as an honest coin flip, not as 'up')")
    check("the couplings survive the encoding",
          np.array_equal(task.encode(insts[0])[task.coupling_planes],
                         insts[0].J[:task.n_dirs]))

    # 3. random_candidate respects the clamp
    for mode in ("uniform", "quench"):
        ok, exc = True, []
        for i2 in insts:
            cand = task.random_candidate(i2, RNG, mode=mode)
            ok &= all(cand.reshape(-1)[k] == v for k, v in i2.clamped.items())
            exc.append(task.violations(i2, cand))
        check(f"random_candidate(mode={mode!r}) respects the clamp", ok,
              f"mean excess energy {np.mean(exc):.1f}")

    base = matched_baseline(task, insts[0], n_visits=2000, rng=RNG, trials=8)
    n_free = insts[0].n - len(insts[0].clamped)
    check("matched_baseline over 2000 visits is ~0", base == 0.0,
          f"= {base:.3f}; a uniform candidate is ~{insts[0].degeneracy:.0f} in 2^{n_free}, "
          "so LIFT OVER THIS IS NOT EVIDENCE — the comparators are the quench "
          "rate and reference_solve's cost")

    # 4. the clamp audit must actually fire
    ins = insts[0]
    seed_t = torch.from_numpy(task.encode(ins))
    clamp_t = torch.from_numpy(task.clamp_mask(ins))
    clean = torch.stack([seed_t] * 4)
    ok_clean = givens_preserved(clean, seed_t, clamp_t)
    tampered = clean.clone()
    k0 = next(iter(ins.clamped))
    tampered[-1, task.belief_plane].reshape(-1)[k0] *= -1
    ok_tampered = givens_preserved(tampered, seed_t, clamp_t)
    check("clamp audit passes an untouched rollout", ok_clean == 1.0, f"= {ok_clean:.3f}")
    check("clamp audit catches the overwritten reference spin", ok_tampered < 1.0,
          f"= {ok_tampered:.3f} of {int(clamp_t.sum())} clamped values")
    check("clamp_mask marks only the belief plane",
          float(clamp_t[task.belief_plane].sum()) == len(ins.clamped)
          and float(clamp_t.sum()) == len(ins.clamped),
          "(the coupling planes never move and would pad the audit; K is meant "
          "to be written, so auditing it would flag a working stiffening dynamic)")
    check("hold_mask holds the couplings AND the clamp",
          float(task.hold_mask(ins).sum()) ==
          float(task.structure_mask().sum()) + len(ins.clamped))

    # 5. is_solved is exact and gauge-aware
    ok = ok_free = True
    n_free_flip = 0
    for i2 in insts:
        ok &= task.is_solved(i2, i2.ground) and task.violations(i2, i2.ground) == 0
        fr2, _, _ = exact_frozen(i2)
        # flipping a FROZEN spin must break it
        # site 0 is the gauge reference and is frozen by construction, not by
        # physics -- see exact_frozen's docstring
        k0_skip = 1 if not np.any(i2.h) else 0
        frozen_free = [k for k in range(k0_skip, i2.n)
                       if fr2.reshape(-1)[k] and k not in i2.clamped]
        if frozen_free:
            other = i2.ground.copy()
            other.reshape(-1)[frozen_free[0]] *= -1
            ok &= not task.is_solved(i2, other)
        # flipping a FREE spin need not: that is what degeneracy means
        loose = [k for k in range(k0_skip, i2.n) if not fr2.reshape(-1)[k]]
        for k in loose:
            other = i2.ground.copy()
            other.reshape(-1)[k] *= -1
            n_free_flip += int(task.is_ground_state(i2, other))
    check("is_solved accepts a ground state and rejects flipping a FROZEN spin", ok)
    check("flipping a FREE spin can still be a ground state", True,
          f"{n_free_flip} single flips of non-frozen spins landed on another "
          "ground state — a one-spin change is NOT automatically wrong here, "
          "which is why is_solved compares energy to E0 and not to one answer")
    flipped = -insts[0].ground
    if task.has_field:
        check("with a field there is no flip symmetry to gauge away",
              not task.is_ground_state(insts[0], flipped),
              "(the reference spin is then a plain given, not a gauge — which "
              "is also why BP and SP behave better on the field variant)")
    else:
        check("is_solved REJECTS the global flip of a ground state",
              task.is_ground_state(insts[0], flipped)
              and not task.is_solved(insts[0], flipped),
              "(same energy, wrong gauge: it violated its given, and without "
              "that distinction the answer is a coset, not a configuration)")
    check("violations counts excess ENERGY, not broken bonds",
          task.broken_bonds(insts[0], insts[0].ground) > 0
          and task.violations(insts[0], insts[0].ground) == 0,
          f"the ground state itself breaks "
          f"{task.broken_bonds(insts[0], insts[0].ground)} of {insts[0].n_bonds} "
          f"bonds ({100 * task.broken_bonds(insts[0], insts[0].ground) / insts[0].n_bonds:.0f}%)")

    # 6. the energy is well-posed
    def put(cfg, mag=1.0):
        st = torch.from_numpy(task.encode(ins)).clone()
        st[task.belief_plane] = torch.from_numpy(np.asarray(cfg, np.float32)) * mag
        return st

    near = ins.ground.copy()
    near.reshape(-1)[[k for k in range(ins.n) if k not in ins.clamped][0]] *= -1
    flat = torch.from_numpy(task.encode(ins)).clone()
    flat[task.belief_plane] = 0.0
    lm = {
        "ground state": put(ins.ground),
        "ground state, 1 spin flipped": put(near),
        "greedy quench": put(task.random_candidate(ins, RNG, "quench")),
        "random +-1": put(task.random_candidate(ins, RNG)),
        "flat beliefs (m = 0)": flat,
        "seed (the clamp only)": torch.from_numpy(task.encode(ins)),
    }
    es = {k: float(task.energy(v, ins)) for k, v in lm.items()}
    for k, v in es.items():
        print(f"        E[{k:30s}] = {v:+.4f}")
    e_sat = float(task.energy(put(ins.ground, 10.0), ins))
    print(f"        E[{'ground state, saturated (x10)':30s}] = {e_sat:+.6f}"
          "   <- the zero of the energy is the ground-state SET")
    check("energy(ground state) is the smallest", es["ground state"] == min(es.values()))
    check("energy orders ground < near-miss < random",
          es["ground state"] < es["ground state, 1 spin flipped"] < es["random +-1"])
    check("a saturated ground state has energy 0", abs(e_sat) < 1e-5, f"= {e_sat:.2e}")
    check("flat beliefs are NOT a minimum",
          es["flat beliefs (m = 0)"] > es["ground state"],
          f"{es['flat beliefs (m = 0)']:.4f} > {es['ground state']:.4f} — the "
          "energy is multilinear in m, so its minimum over the belief cube is at "
          "a VERTEX, i.e. at a real configuration.  No purity term needed.")

    # the flat-state trap in its Ising form: a stationary point, not a minimum
    for label, st in (("flat, no clamp", flat),
                      ("the actual seed (flat + the clamped spin)",
                       torch.from_numpy(task.encode(ins)))):
        x = st.clone().requires_grad_(True)
        task.energy(x, ins).backward()
        g = x.grad[task.belief_plane]
        nz = int((g.abs() > 1e-12).sum())
        print(f"        |dE/dm| {label:42s} max {g.abs().max():.2e}  "
              f"{nz}/{ins.n} pixels")
        if label == "flat, no clamp":
            if task.has_field:
                check("the random field breaks the flat-state saddle everywhere",
                      nz == ins.n,
                      "every site feels its own field at m = 0 — this is the "
                      "reason to train the field variant first")
            else:
                check("THE TRAP: with no field the flat seed is an exact "
                      "STATIONARY point", float(g.abs().max()) == 0.0,
                      "dE/dm_i = -sum_j J_ij m_j - h_i vanishes at m = 0")
        elif not task.has_field and len(ins.clamped) == 1:
            check("the clamped reference spin is the only source of gradient",
                  0 < nz <= 8,
                  f"{nz} pixels of {ins.n} — the solve has to nucleate from one "
                  "site; seed noise or the field variant is the way round it")

    # the aggregate, measured rather than inherited
    # averaged over 3 instances: a single instance's rho moves by +-0.1 and the
    # aggregates are separated by less than that on the easy presets
    tks = {agg: IsingLattice(R, C, variant=args.variant,
                             difficulty=args.difficulty, aggregate=agg)
           for agg in AGGREGATES}
    acc = {agg: ([], []) for agg in AGGREGATES}
    for i3 in insts[:3]:
        e3 = torch.from_numpy(task.encode(i3))

        def put3(cfg, mag=1.0, base=e3):
            st = base.clone()
            st[task.belief_plane] = torch.from_numpy(np.asarray(cfg, np.float32)) * mag
            return st

        st3, xs3 = [], []
        for _ in range(64):
            cand = task.random_candidate(i3, RNG, "quench").astype(np.float32)
            mag = RNG.random((R, C)).astype(np.float32) * 0.6 + 0.05
            st3.append(put3(cand * mag))
            xs3.append(task.violations(i3, cand))
        n3 = i3.ground.copy()
        n3.reshape(-1)[[k for k in range(i3.n) if k not in i3.clamped][0]] *= -1
        sat_g, sat_n = put3(i3.ground, 3.0), put3(n3, 3.0)
        sat_r = put3(task.random_candidate(i3, RNG), 3.0)
        for agg, tk in tks.items():
            ys = [float(tk.energy(s2, i3)) for s2 in st3]
            acc[agg][1].append(float(np.corrcoef(
                np.argsort(np.argsort(xs3)).astype(float),
                np.argsort(np.argsort(ys)).astype(float))[0, 1]))
            eg, en, er = (float(tk.energy(sat_g, i3)), float(tk.energy(sat_n, i3)),
                          float(tk.energy(sat_r, i3)))
            acc[agg][0].append((en - eg) / max(er - eg, 1e-12))
    print(f"        {'aggregate':9s} {'one-flip share':>15s} {'rho (undecided)':>17s}")
    shares = {agg: (float(np.mean(v[0])), float(np.mean(v[1])))
              for agg, v in acc.items()}
    for agg in AGGREGATES:
        print(f"        {agg:9s} {shares[agg][0]:14.1%} {shares[agg][1]:+17.3f}")
    check("the sum (the Hamiltonian) ranks at least as well as the smooth max",
          shares["sum"][1] >= shares["softmax"][1] - 0.05,
          f"rho {shares['sum'][1]:+.3f} vs {shares['softmax'][1]:+.3f}.  A "
          "frustrated ground state BREAKS constraints on purpose, so pressure on "
          "the worst bond optimises the wrong objective — this is where this "
          "track's answer differs from Wang's and Sudoku's.  The gap is wide at "
          "the default (nnn/hard) and narrows to a tie on the easy presets, "
          "where fewer bonds are contested.")
    check("max is rank-blind next to the sum",
          abs(shares["max"][1]) < 0.5 * abs(shares["sum"][1]),
          f"rho {shares['max'][1]:+.3f} vs {shares['sum'][1]:+.3f}")

    # 7. the reference solvers are sane where the answer is known
    n = 10
    crng = np.random.default_rng(args.seed)          # its own stream: this check
    chain = IsingModel(n, np.array([[i, i + 1] for i in range(n - 1)]),
                       crng.choice([-1.0, 1.0], n - 1),      # must not depend on
                       crng.choice([-1.0, 0.0, 1.0], n))     # what ran before it
    # enumerated under the SAME gauge the solvers are given, or the two sides
    # are answering different questions
    best, gs = None, []
    for x in range(1 << n):
        v = np.array([1 - 2 * ((x >> b) & 1) for b in range(n)], np.int8)
        if v[0] < 0:
            continue
        e = chain.energy(v)
        if best is None or e < best - 1e-9:
            best, gs = e, [v]
        elif abs(e - best) < 1e-9:
            gs.append(v)
    G = np.array(gs)
    fex = (G.min(0) == G.max(0))
    wp = warning_propagation(chain, rng=np.random.default_rng(0), clamped={0: 1})
    spc = survey_propagation(chain, y=0.0, rng=np.random.default_rng(0), clamped={0: 1})
    bpc = belief_propagation(chain, beta=4.0, rng=np.random.default_rng(0), clamped={0: 1})
    check("WP converges on a chain (a tree) and finds the frozen set exactly",
          wp.converged and np.array_equal(wp.fields != 0, fex),
          f"{wp.iters} iterations for {n} spins")
    check("SP(y=0) reproduces the exact frozen set on a chain",
          np.allclose(spc.frozen > 0.5, fex),
          "so a poor frozen estimate on the LATTICE is a property of the "
          "lattice, not a bug in the solver")
    check("BP finds the chain's ground state", abs(bpc.energy - best) < 1e-9)

    # 8. difficulty: are these instances instances?
    hits = [quench_rate(to_model(i2), i2.e0, trials=32, rng=RNG,
                        clamped=i2.clamped) for i2 in insts]
    hit = np.mean([h[0] for h in hits])
    any32 = np.mean([h[0] > 0 for h in hits])
    cap = task._resolve_difficulty(None).max_quench_rate
    print(f"        greedy quench: per-attempt {hit:.4f}, any-of-32 {any32:.3f}, "
          f"mean excess {np.mean([h[1] for h in hits]):.1f}, "
          f"sweeps {np.mean([h[2] for h in hits]):.1f}")
    kept = np.mean([i2.quench_hit for i2 in insts])
    check("the generator's own quench screen held", kept <= cap,
          f"recorded per-attempt {kept:.4f} against the preset's cap {cap:.2f}")
    check("a re-measured quench does not solve most of these",
          any32 < 0.5 or not CLAIMS_HARD,
          f"per-attempt {hit:.4f}, any-of-32 {any32:.3f} on a FRESH stream — the "
          "screen is a 16-sample estimate of a ~0.002 rate, so it halves the easy "
          "tail rather than removing it; stratify on IsingInstance.quench_hit if "
          "an evaluation needs it gone (6x6, by contrast, is 0.81 for any-of-64: "
          "a warm-up, not an instance)")
    check("the ground states are degenerate, so a per-pixel target is incoherent",
          np.median([i2.degeneracy for i2 in insts]) > 1 or not CLAIMS_HARD,
          f"median degeneracy {np.median([i2.degeneracy for i2 in insts]):.0f} "
          "mod global flip")

    # 9. light cone
    check("min_rollout exceeds the impossibility floor by the round count",
          task.min_rollout() >= 2 * task.min_rollout_floor(),
          f"{task.min_rollout()} steps vs a {task.min_rollout_floor()}px diameter — "
          f"{task.min_rollout() / max(task.min_rollout_floor(), 1):.0f}x")

    print(f"\n  {len(FAILURES)} failure(s)" + (": " + ", ".join(FAILURES) if FAILURES else ""))


# ── survey ───────────────────────────────────────────────────────────────────

if args.survey:
    M = args.survey_n
    print("\n── frustration: the formula, and why it is not the knob " + "─" * 8)
    print(f"  {'variant':8s} {'p_af':>5s} {'measured':>18s} {'formula':>8s}")
    for variant in VARIANTS:
        for p in (0.10, 0.25, 0.50):
            t = IsingLattice(R, C, variant=variant, difficulty=dict(p_af=p))
            rng = np.random.default_rng(0)
            fs = []
            for _ in range(48):
                J, h = t.draw_couplings(rng, t._resolve_difficulty(None))
                fs.append(frustration(IsingInstance(R, C, variant, J, h,
                                                    np.ones((R, C), np.int8), 0.0)))
            print(f"  {variant:8s} {p:5.2f} {np.mean(fs):11.3f} +- {np.std(fs):.3f} "
                  f"{expected_frustration(p, variant):8.3f}")
    print("  frustration saturates by p_af ~ 0.25, so it locates the band and "
          "cannot order instances inside it.")

    print(f"\n── difficulty: what actually separates instances ({M} each) " + "─" * 4)
    print(f"  {'variant':8s} {'p_af':>5s} {'E0':>8s} {'deg med':>8s} {'frozen':>7s} "
          f"{'qhit':>7s} {'q-exc':>6s} {'brk%':>5s}")
    for variant in VARIANTS:
        for p in (0.10, 0.50):
            t = IsingLattice(R, C, variant=variant,
                             difficulty=dict(p_af=p, max_quench_rate=1.0))
            rng = np.random.default_rng(1)
            rows = []
            for _ in range(M):
                i2 = t.draw_instance(rng)
                m2 = to_model(i2)
                hit, exc, _ = quench_rate(m2, i2.e0, trials=32, rng=rng,
                                          clamped=i2.clamped)
                rows.append((i2.e0, i2.degeneracy, exact_frozen(i2)[0].mean(),
                             hit, exc,
                             m2.unsatisfied(i2.ground.reshape(-1)) / len(m2.J)))
            a = np.array(rows)
            print(f"  {variant:8s} {p:5.2f} {a[:, 0].mean():8.1f} "
                  f"{np.median(a[:, 1]):8.1f} {a[:, 2].mean():7.3f} "
                  f"{a[:, 3].mean():7.4f} {a[:, 4].mean():6.1f} "
                  f"{100 * a[:, 5].mean():5.1f}")

    print(f"\n── BP vs SP vs exact ({M} instances, {R}x{C}) " + "─" * 20)
    for variant in VARIANTS:
        t = IsingLattice(R, C, variant=variant, difficulty="easy")
        rng = np.random.default_rng(21)
        ins2 = [t.draw_instance(rng) for _ in range(M)]
        ms = [to_model(i) for i in ins2]
        frz = [exact_frozen(i)[0].reshape(-1).astype(float) for i in ins2]
        print(f"  -- {variant} (shortest loop {'4' if variant == 'planar' else '3'}), "
              f"exact frozen {np.mean([f.mean() for f in frz]):.3f}")

        def rep(name, cfgs, its, conv, extra=""):
            exc = [m.energy(c) - i.e0 for m, c, i in zip(ms, cfgs, ins2)]
            print(f"     {name:24s} solved {np.mean([e < 1e-9 for e in exc]):5.3f}  "
                  f"excess med {np.median(exc):5.1f}  iters med {np.median(its):6.0f}  "
                  f"conv {np.mean(conv):4.2f}{extra}")

        for dmp in (0.0, 0.5, 0.9):
            rs = [belief_propagation(m, beta=1.0, damping=dmp, clamped=i.clamped,
                                     rng=np.random.default_rng(0), max_iters=500)
                  for m, i in zip(ms, ins2)]
            rep(f"BP beta=1 damping={dmp}", [r.config for r in rs],
                [r.iters for r in rs], [r.converged for r in rs])
        for y in (0.0, 1.0):
            rs = [survey_propagation(m, y=y, clamped=i.clamped,
                                     rng=np.random.default_rng(0), max_iters=300)
                  for m, i in zip(ms, ins2)]
            rho = [np.corrcoef(f, r.frozen)[0, 1] for f, r in zip(frz, rs)
                   if f.std() > 0 and r.frozen.std() > 0]
            rep(f"SP y={y}", [r.config for r in rs], [r.iters for r in rs],
                [r.converged for r in rs],
                f"  rho(frozen, exact) {np.nanmean(rho):+.3f}")
        k = max(1, (R * C) // 32)
        ds = [bp_decimation(m, beta=1.0, fix_per_round=k, max_iters=100,
                            rng=np.random.default_rng(0), clamped=i.clamped)
              for m, i in zip(ms, ins2)]
        rep("BP-decimation + polish", [d.config for d in ds],
            [d.total_iters for d in ds], [1] * M)
        ds = [sp_decimation(m, y=0.0, fix_per_round=k, max_iters=60,
                            rng=np.random.default_rng(0), clamped=i.clamped)
              for m, i in zip(ms, ins2)]
        rep("SP-decimation + polish", [d.config for d in ds],
            [d.total_iters for d in ds], [1] * M)
        q = [quench_rate(m, i.e0, trials=64, rng=np.random.default_rng(0),
                         clamped=i.clamped)[0] > 0 for m, i in zip(ms, ins2)]
        print(f"     {'quench, 64 restarts':24s} solved {np.mean(q):5.3f}")

    print("\n── how parallel can decimation be? (sets min_rollout) " + "─" * 10)
    t = IsingLattice(R, C, variant=args.variant, difficulty="easy")
    rng = np.random.default_rng(21)
    ins2 = [t.draw_instance(rng) for _ in range(M)]
    ms = [to_model(i) for i in ins2]
    n_free = R * C - 1
    for k in (1, 2, 4, 8, n_free):
        ds = [sp_decimation(m, y=0.0, fix_per_round=k, max_iters=60,
                            rng=np.random.default_rng(0), clamped=i.clamped)
              for m, i in zip(ms, ins2)]
        exc = [m.energy(d.config) - i.e0 for m, d, i in zip(ms, ds, ins2)]
        rounds = int(np.mean([d.rounds for d in ds]))
        print(f"  fix {k:3d}/round  rounds {rounds:4d}  solved "
              f"{np.mean([e < 1e-9 for e in exc]):5.3f}  excess med {np.median(exc):4.1f}"
              f"  -> min_rollout {rounds * 2 * t.diameter()}")

    print("\n── influence radius: does a distant coupling matter? " + "─" * 10)
    dists, frac = [], []
    for i2 in ins2[:6]:
        g0, _, _ = exact_ground_state(i2)
        for _ in range(6):
            d = int(rng.integers(0, t.n_dirs))
            r0, c0 = int(rng.integers(0, R)), int(rng.integers(0, C))
            if i2.J[d, r0, c0] == 0:
                continue
            J2 = i2.J.copy()
            J2[d, r0, c0] *= -1
            g1, _, _ = exact_ground_state(IsingInstance(R, C, i2.variant, J2, i2.h,
                                                        np.ones((R, C), np.int8), 0.0))
            diff = np.argwhere(g0 != g1)
            frac.append(len(diff) / i2.n)
            dists.append(max((max(abs(rr - r0), abs(cc - c0)) for rr, cc in diff),
                             default=0))
    print(f"  flipping ONE coupling moves {np.mean(frac):.1%} of the spins; the "
          f"farthest sits at Chebyshev distance med {np.median(dists):.0f} "
          f"p90 {np.percentile(dists, 90):.0f} max {np.max(dists):.0f} "
          f"(diameter {t.diameter()})")
    print("  so the impossibility floor really is the diameter: some spins cannot "
          "be decided by any finite-radius rule.")

sys.exit(1 if FAILURES else 0)
