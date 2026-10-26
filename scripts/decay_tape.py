#!/usr/bin/env python3
"""How the v15 tape deteriorates: three measurements on one checkpoint.

--programs random programs, each run --seeds times (same program and start state;
only the stochastic fire masks differ), closed loop for up to --cycles cycles.

1. error type   tape error (ch1-3, ring rows) split into
                  level    a band block's mean is off its value (fading or overshoot)
                  texture  pixels inside a block disagree with the block mean (speckle)
                  near     ink in the off pixels touching a band (blur, leaking into gaps)
                  far      ink elsewhere off the bands (ghosts)
                plus the stamp error in the memory rows and the signed fade of the band
                levels (+ = towards 0).
2. load/exec    error added while the bands move (steps 0-load) vs while they hold.
3. seeds        does a program fail at the same cycle under every fire mask?

Also records, per run, the first cycle a band decodes wrong, the first cycle the
delivered stamp decodes wrong and the first cycle memory is wrong. Writes
<out>.npz, <out>.json and a figure <out>.png in the project root.

    uv run python scripts/decay_tape.py --run runs/TAPE_... --ckpt nca_keep_20k.pt
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="nca_best.pt")
ap.add_argument("--n", type=int, default=32, help="program length")
ap.add_argument("--programs", type=int, default=8)
ap.add_argument("--seeds", type=int, default=16, help="runs per program (fire masks)")
ap.add_argument("--cycles", type=int, default=1000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="v15_decay")
ap.add_argument("--plot-only", action="store_true", help="redraw from <out>.npz")
opt = ap.parse_args()

KEYS = ("level", "texture", "near", "far", "stamp", "fade", "e_start", "e_load", "e_end",
        "bad_slots", "stamp_bad", "mem_ok")


def measure():
    run = Path(opt.run)
    ck = torch.load(run / opt.ckpt, map_location=opt.device, weights_only=False)
    cfg = ck["config"]

    # the trainer's own geometry, painting and targets: its source up to the model
    src = (ROOT / "scripts/train_tape.py").read_text()
    src = src[:src.index("# ── model")]
    sys.argv = ["train_tape.py"] + [a for k, v in cfg.items() if v is not None and v != ""
                                    for a in (f"--{k.replace('_', '-')}", str(v))]
    sys.argv += ["--device", opt.device]
    T = {"__file__": str(ROOT / "scripts/train_tape.py"), "__name__": "decay"}
    exec(compile(src, "train_tape.py", "exec"), T)
    args, K, dev = T["args"], T["K"], opt.device

    nca = T["NeuralCA"](channels=args.channels, hidden_channels=args.hidden, fire_rate=args.fire_rate,
                        alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
                        perception="static_sobel", padding_type="circular", read_only_dims=[],
                        state_bound=10.0).to(dev)
    nca.load_state_dict(ck["nca"])
    nca.eval()
    torch.manual_seed(opt.seed)

    rng = random.Random(opt.seed)
    progs = [T["fresh_run"](opt.n, args.eval_regs, rng) for _ in range(opt.programs)]
    runs = [p for p in progs for _ in range(opt.seeds)]
    B, g, L, nr = len(runs), runs[0]["g"], T["lay"](args.eval_regs), args.eval_regs
    H, W, N = g.H, L.W, g.N

    # rows -> slot (gap + band rows), band rows only, memory rows; columns -> register
    slot_of = torch.zeros(H, N, device=dev)
    band_of = torch.zeros(H, N, device=dev)
    for i in range(N):
        y = g.slot_row(i)
        slot_of[y - args.gap:y + args.band, i] = 1
        band_of[y:y + args.band, i] = 1
    ring_rows = slot_of.sum(1) > 0                                    # (H,)
    mem_rows = torch.zeros(H, device=dev)
    mem_rows[g.M0:g.M1] = 1
    reg_of = torch.zeros(W, nr, device=dev)
    for r in range(nr):
        reg_of[L.col_x[r] - L.r:L.col_x[r] + L.r + 1, r] = 1
    blk_n = args.band * (2 * L.r + 1)
    mem_n = (g.M1 - g.M0) * (2 * L.r + 1)
    n_ring = float(3 * ring_rows.sum() * W)

    def decode_blocks(m):
        """(B, 3, S, nr) block means -> (B, S, 4) = A, B, D, sign of R."""
        d = m[:, 2].abs().argmax(-1)
        sgn = torch.gather(m[:, 2], -1, d[..., None])[..., 0] > 0
        return torch.stack([m[:, 0].argmax(-1), m[:, 1].argmax(-1), d, sgn.long()], -1)

    def stats(x, t):
        """Error decomposition of the instruction channels x vs the target t."""
        x, t = x[:, 1:4], t
        e2 = (x - t) ** 2
        rr = ring_rows[None, None, :, None]
        # band blocks: sums over (band rows of slot n) x (columns of register r)
        sx = torch.einsum("bchw,hn,wr->bcnr", x, band_of, reg_of)
        sx2 = torch.einsum("bchw,hn,wr->bcnr", x * x, band_of, reg_of)
        tb = torch.einsum("bchw,hn,wr->bcnr", t, band_of, reg_of) / blk_n
        mx = sx / blk_n
        on = tb != 0
        level = (blk_n * (mx - tb) ** 2 * on).flatten(1).sum(1)
        texture = ((sx2 - blk_n * mx ** 2) * on).flatten(1).sum(1)
        on_px = (t != 0) & rr
        near_px = (F.max_pool2d((t != 0).float(), 3, 1, 1) > 0) & ~on_px & rr
        far_px = ~(F.max_pool2d((t != 0).float(), 3, 1, 1) > 0) & rr
        near = (e2 * near_px).flatten(1).sum(1)
        far = (e2 * far_px).flatten(1).sum(1)
        fade = ((tb - mx) / tb.where(on, torch.ones_like(tb)) * on).flatten(1).sum(1) \
            / on.flatten(1).sum(1)
        bad = (decode_blocks(mx) != decode_blocks(tb)).any(-1)          # (B, N)
        # stamp: the memory rows' register blocks
        ms = torch.einsum("bchw,h,wq->bcq", x, mem_rows, reg_of)[:, :, None] / mem_n
        mt = torch.einsum("bchw,h,wq->bcq", t, mem_rows, reg_of)[:, :, None] / mem_n
        stamp_bad = (decode_blocks(ms) != decode_blocks(mt)).any(-1)[:, 0]
        stamp = (e2 * (mem_rows > 0)[None, None, :, None]).flatten(1).sum(1)
        per_slot = torch.einsum("bchw,hn->bn", e2, slot_of)
        ring = (e2 * rr).flatten(1).sum(1) / n_ring
        return dict(level=level / n_ring, texture=texture / n_ring, near=near / n_ring,
                    far=far / n_ring, stamp=stamp / (3 * (g.M1 - g.M0) * W), fade=fade,
                    ring=ring, bad=bad, stamp_sum=stamp, stamp_bad=stamp_bad, per_slot=per_slot)

    rec = {k: np.full((opt.cycles, B), np.nan, np.float32) for k in KEYS}
    slot_err = np.zeros((opt.cycles, B, g.n), np.float16)  # per instruction (not position)
    first = {k: np.full(B, opt.cycles + 1) for k in ("tape", "stamp", "mem")}
    x = T["start_state"](runs)
    t_prev = x[:, 1:4].clone()
    t0 = time.time()
    with torch.no_grad():
        for c in range(opt.cycles):
            e_start = ((x[:, 1:4] - t_prev) ** 2 * ring_rows[None, None, :, None]).flatten(1).sum(1) / n_ring
            runs, t_load, _, t_tape = T["cycle_targets"](runs)
            for s in range(K):
                x = nca(x, 1)[:, -1]
                if s + 1 == args.load_steps:
                    sl = stats(x, t_load)
            se = stats(x, t_tape)
            t_prev = t_tape
            mem_ok = (T["decode"](runs, x[:, 0]) == T["want_bits"](runs)).flatten(1).all(1)
            row = dict(level=se["level"], texture=se["texture"], near=se["near"], far=se["far"],
                       stamp=se["stamp"], fade=se["fade"], e_start=e_start, e_load=sl["ring"],
                       e_end=se["ring"], bad_slots=se["bad"].sum(1), stamp_bad=sl["stamp_bad"],
                       mem_ok=mem_ok)
            for k, v in row.items():
                rec[k][c] = v.float().cpu().numpy()
            # per instruction: slot error re-indexed by which instruction sits there
            ps = se["per_slot"].cpu().numpy()
            st = se["stamp_sum"].cpu().numpy()
            for b, r in enumerate(runs):
                slot_err[c, b, r["inmem"]] = st[b]             # the loaded one: its stamp
                for k, i in r["ring"].items():
                    slot_err[c, b, k] = ps[b, i]
            for k, hit in (("tape", se["bad"].any(1)), ("stamp", sl["stamp_bad"]), ("mem", ~mem_ok)):
                h = hit.cpu().numpy()
                first[k][h & (first[k] > opt.cycles)] = c + 1
            if (c + 1) % 50 == 0:
                print(f"cycle {c + 1}: memory ok {mem_ok.float().mean():.2f}  "
                      f"ring err {se['ring'].mean():.4f}  {time.time() - t0:.0f}s", flush=True)
            if (first["mem"] <= c + 1 - 20).all():         # all failed 20 cycles ago
                break
    C = c + 1
    np.savez_compressed(ROOT / f"{opt.out}.npz", **{k: v[:C] for k, v in rec.items()},
                        slot_err=slot_err[:C], **{f"first_{k}": v for k, v in first.items()},
                        inmem_head=np.array([g.head, g.A, N]), seeds=opt.seeds,
                        programs=opt.programs, cycles=opt.cycles)
    return f"{run.name}/{opt.ckpt} step {ck['step']:,}"


def summarize(name):
    d = np.load(ROOT / f"{opt.out}.npz")
    C, S, P = d["level"].shape[0], int(d["seeds"]), int(d["programs"])
    fm, ft, fs = d["first_mem"], d["first_tape"], d["first_stamp"]
    cyc = np.arange(1, C + 1)
    alive = cyc[:, None] < fm[None, :]                 # memory still right after this cycle

    def am(k):                                          # mean over runs still alive
        v = np.where(alive, d[k], np.nan)
        m = np.nanmean(v, 1)
        m[alive.sum(1) < 8] = np.nan
        return m

    comp = {k: am(k) for k in ("level", "texture", "near", "far", "stamp", "fade")}
    d_load = am("e_load") - am("e_start")
    d_exec = am("e_end") - am("e_load")

    # 3. seeds: share of the variance of log(first fail) explained by the program
    lf = np.log(np.minimum(fm, int(d["cycles"]) + 1)).reshape(P, S)
    icc = float(lf.mean(1).var() / lf.var()) if lf.var() > 0 else float("nan")

    out = {"what": name, "cycles_run": C, "programs": P, "seeds": S}
    print(f"\n{name}: {P} programs x {S} fire-mask seeds, n={opt.n}\n")
    print("1. error type (mean per ring pixel, runs still right):")
    print(f"{'cycle':>7}{'level':>10}{'texture':>10}{'near':>10}{'far':>10}{'stamp':>10}{'fade':>8}")
    out["types"] = {}
    for c in (1, 10, 30, 100, 300, 1000):
        if c <= C and not np.isnan(comp["level"][c - 1]):
            r = {k: float(comp[k][c - 1]) for k in comp}
            out["types"][c] = r
            print(f"{c:>7}" + "".join(f"{r[k]:>10.5f}" for k in ("level", "texture", "near", "far", "stamp"))
                  + f"{r['fade']:>+8.3f}")
    ok = ~np.isnan(d_load)
    out["load_total"], out["exec_total"] = float(np.nansum(d_load)), float(np.nansum(d_exec))
    print(f"\n2. error added per cycle (runs still right), summed over {ok.sum()} cycles:"
          f"\n   load  {out['load_total']:+.4f}   execute {out['exec_total']:+.4f}"
          f"\n   mean per cycle: load {np.nanmean(d_load):+.5f}  execute {np.nanmean(d_exec):+.5f}")
    print("\n3. first memory failure per program, across seeds (min / median / max):")
    out["seeds_fail"] = fm.reshape(P, S).tolist()
    for p in range(P):
        f = fm.reshape(P, S)[p]
        print(f"   program {p}: {f.min():>5} {int(np.median(f)):>5} {f.max():>5}")
    out["icc_log_fail"] = icc
    print(f"   share of log-fail variance explained by the program: {icc:.2f}"
          f"  (1 = fixed per program, 0 = pure chance)")
    order = np.stack([ft, fs, fm], 1)
    out["first_median"] = dict(tape=float(np.median(ft)), stamp=float(np.median(fs)), mem=float(np.median(fm)))
    out["tape_before_mem"] = float((ft < fm).mean())
    out["stamp_before_mem"] = float((fs <= fm).mean())
    print(f"\norder of events (median cycle): band decodes wrong {np.median(ft):.0f}, "
          f"stamp wrong {np.median(fs):.0f}, memory wrong {np.median(fm):.0f}"
          f"\n   band wrong before memory {out['tape_before_mem']:.2f}, "
          f"stamp wrong at or before memory {out['stamp_before_mem']:.2f}")
    (ROOT / f"{opt.out}.json").write_text(json.dumps(out, indent=1))
    plot(d, comp, d_load, d_exec, fm, ft, fs, name, order)


def plot(d, comp, d_load, d_exec, fm, ft, fs, name, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
    INK, MUTED = "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 13, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": "#ecebe7", "axes.spines.top": False, "axes.spines.right": False})
    C = d["level"].shape[0]
    cyc = np.arange(1, C + 1)
    P, S = int(d["programs"]), int(d["seeds"])
    fig, ax = plt.subplots(2, 3, figsize=(24, 13), constrained_layout=True)
    fig.suptitle(f"How the tape deteriorates — {name}, {P} programs × {S} seeds, n={opt.n}",
                 fontsize=17, color=INK)

    a = ax[0, 0]
    for k, col in (("level", S1), ("texture", S2), ("near", S3), ("far", S4), ("stamp", S5)):
        a.plot(cyc, comp[k], color=col, lw=2, label=k)
    a.set_yscale("log"), a.set_xscale("log")
    a.set_title("1. error by type (runs whose memory is still right)", loc="left")
    a.set_xlabel("cycle"), a.set_ylabel("squared error per ring pixel")
    a.legend(frameon=False)

    a = ax[0, 1]
    a.plot(cyc, 100 * comp["fade"], color=S1, lw=2)
    a.axhline(0, color=MUTED, lw=1)
    a.set_xscale("log")
    a.set_title("1. band level vs its value (+ = faded towards 0)", loc="left")
    a.set_xlabel("cycle"), a.set_ylabel("% of the value lost")

    a = ax[0, 2]
    k = 9                                               # 9-cycle running mean
    sm = lambda v: np.convolve(np.nan_to_num(v), np.ones(k) / k, "same") * np.where(np.isnan(v), np.nan, 1)
    a.plot(cyc, sm(d_load), color=S2, lw=2, label="load (bands move)")
    a.plot(cyc, sm(d_exec), color=S1, lw=2, label="execute (bands hold)")
    a.axhline(0, color=MUTED, lw=1)
    a.set_xscale("log")
    a.set_title("2. error added per cycle, by phase", loc="left")
    a.set_xlabel("cycle"), a.set_ylabel("Δ squared error per ring pixel")
    a.legend(frameon=False)

    # 4. one run, per instruction: the run with the median first failure
    a = ax[1, 0]
    b = int(np.argsort(fm)[len(fm) // 2])
    h = np.log10(d["slot_err"][:, b].astype(np.float32).T + 1e-4)
    im = a.imshow(h, aspect="auto", cmap="Blues", origin="lower", extent=(0.5, C + 0.5, -0.5, h.shape[0] - 0.5),
                  vmin=-3, vmax=1)
    for v, lab, col in ((ft[b], "band wrong", S2), (fs[b], "stamp wrong", S5), (fm[b], "memory wrong", "#e34948")):
        if v <= C:
            a.axvline(v, color=col, lw=2, ls="--", label=f"{lab} @{v}")
    a.set_xlim(0.5, min(C, 3 * fm[b]) + 0.5)
    a.set_title(f"one run (program {b // S}, seed {b % S}): error of each instruction", loc="left")
    a.set_xlabel("cycle"), a.set_ylabel("instruction")
    a.legend(frameon=False, loc="upper right")
    fig.colorbar(im, ax=a, label="log10 squared error (sum over its slot)")

    a = ax[1, 1]
    for p in range(P):
        f = fm.reshape(P, S)[p]
        t = ft.reshape(P, S)[p]
        y = p + np.linspace(-0.25, 0.25, S)
        a.scatter(t, y, s=60, facecolors="none", edgecolors=S2, lw=1.5, label="band wrong" if p == 0 else None)
        a.scatter(f, y, s=60, color=S1, label="memory wrong" if p == 0 else None)
    a.set_xscale("log")
    a.set_yticks(range(P))
    a.set_title(f"3. first failure of each program under {S} fire-mask seeds", loc="left")
    a.set_xlabel("cycle"), a.set_ylabel("program")
    a.legend(frameon=False)

    a = ax[1, 2]
    a.hist(np.log10(np.maximum(ft, 1)) - np.log10(np.maximum(fm, 1)), bins=30, color=S1, alpha=.9,
           edgecolor="white", label="band wrong")
    a.hist(np.log10(np.maximum(fs, 1)) - np.log10(np.maximum(fm, 1)), bins=30, color=S5, alpha=.6,
           edgecolor="white", label="stamp wrong")
    a.axvline(0, color=MUTED, lw=1)
    a.set_title("when the instruction goes wrong, relative to memory", loc="left")
    a.set_xlabel("log10(cycle / memory-failure cycle)   (< 0 = before)"), a.set_ylabel("runs")
    a.legend(frameon=False)
    fig.savefig(ROOT / f"{opt.out}.png", dpi=80)
    print(f"wrote {opt.out}.png")


name = f"{Path(opt.run).name}/{opt.ckpt}" if opt.plot_only else measure()
summarize(name)
