#!/usr/bin/env python3
"""Progress report for a v15 tape-machine run (scripts/train_tape.py).

Loads a checkpoint, runs fresh random programs CLOSED LOOP (the model's own state
carried from cycle to cycle) and prints, split by the op that executed:

    loss     instr (load at step 20 + tape/stamp at step 40), mem (step 40), total
    acc      destination register: bits right, whole register right;
             whole memory right; and runs still exact at every check so far

then writes a GIF of one batch executing.

    uv run python scripts/progress_tape.py                  # newest TAPE run
    uv run python scripts/progress_tape.py --run runs/TAPE_... --cycles 8
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="", help="run dir (default: newest runs/TAPE_*)")
ap.add_argument("--ckpt", default="nca_latest.pt")
ap.add_argument("--device", default="cuda")
ap.add_argument("--batches", type=int, default=8, help="eval batches, each a random program length")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--cycles", type=int, default=8)
ap.add_argument("--gif", default="v15_progress.gif")
ap.add_argument("--gif-batch", type=int, default=6)
ap.add_argument("--gif-cycles", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
opt = ap.parse_args()

run = Path(opt.run) if opt.run else max(ROOT.glob("runs/TAPE_*"), key=lambda p: p.stat().st_mtime)
ck = torch.load(run / opt.ckpt, map_location=opt.device, weights_only=False)
cfg = ck["config"]

# the trainer's own geometry, painting and targets: its source up to the model
src = (ROOT / "scripts/train_tape.py").read_text()
src = src[:src.index("# ── model")]
sys.argv = ["train_tape.py"] + [a for k, v in cfg.items() if v is not None and v != ""
                                for a in (f"--{k.replace('_', '-')}", str(v))]
sys.argv += ["--device", opt.device]
T = {"__file__": str(ROOT / "scripts/train_tape.py"), "__name__": "progress"}
exec(compile(src, "train_tape.py", "exec"), T)
args, K, LS = T["args"], T["K"], T["args"].load_steps

nca = T["NeuralCA"](channels=args.channels, hidden_channels=args.hidden, fire_rate=args.fire_rate,
                    alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
                    perception="static_sobel", padding_type="circular", read_only_dims=[],
                    state_bound=10.0).to(opt.device)
nca.load_state_dict(ck["nca"])
T["apply_fire"](nca, cfg)
nca.eval()


def mse(a, b):                                        # per sample
    return ((a - b) ** 2).flatten(1).mean(1)


@torch.no_grad()
def report():
    rng = random.Random(opt.seed)
    rows = []                                         # one per (sample, cycle)
    for _ in range(opt.batches):
        n, nr = rng.randint(*T["N_RANGE"]), rng.randint(*T["R_RANGE"])
        runs = [T["fresh_run"](n, nr, rng) for _ in range(opt.batch)]
        x = T["start_state"](runs)
        alive = torch.ones(opt.batch, dtype=torch.bool, device=opt.device)
        prog = alive.clone()                          # memory right at every cycle so far
        for c in range(opt.cycles):
            runs, t_load, t_mem, t_tape = T["cycle_targets"](runs)
            seq = nca(x, K)
            xl, xe = seq[:, LS], seq[:, -1]
            li, lt, lm = mse(xl[:, 1:4], t_load), mse(xe[:, 1:4], t_tape), mse(xe[:, 0], t_mem)
            got, want = T["decode"](runs, xe[:, 0]), T["want_bits"](runs)
            alive &= torch.tensor(T["landed"](xl, xe, t_load, t_tape, runs), device=opt.device)
            prog &= (got == want).flatten(1).all(1)
            for i, r in enumerate(runs):
                op, a, b, d = r["tape"][r["inmem"]]
                rows.append(dict(op=op, instr=float(li[i] + lt[i]), mem=float(lm[i]),
                                 bit=float((got[i, d] == want[i, d]).float().mean()),
                                 reg=float((got[i, d] == want[i, d]).all()),
                                 memory=float((got[i] == want[i]).all()),
                                 alive=float(alive[i]), prog=float(prog[i]), cycle=c + 1))
            x = xe
    return rows


rows = report()
print(f"{run.name}  step {ck['step']:,}  ({opt.batches} batches x {opt.batch}, {opt.cycles} cycles, closed loop)\n")
print(f"{'':8}{'n':>6}{'loss instr':>12}{'loss mem':>10}{'loss total':>12}"
      f"{'dst bits':>10}{'dst reg':>9}{'memory':>8}{'prog ok':>9}{'run exact':>11}")
TABLE = {}
for name in ("SUB", "SUBLEQ", "all"):
    rs = [r for r in rows if name == "all" or r["op"] == name]
    m = {k: np.mean([r[k] for r in rs]) for k in ("instr", "mem", "bit", "reg", "memory", "prog", "alive")}
    TABLE[name] = dict(m, n=len(rs))
    print(f"{name:8}{len(rs):>6}{m['instr']:>12.5f}{m['mem']:>10.5f}{m['instr'] + m['mem']:>12.5f}"
          f"{m['bit']:>10.4f}{m['reg']:>9.4f}{m['memory']:>8.4f}{m['prog']:>9.4f}{m['alive']:>11.4f}")
BY_CYCLE = [float(np.mean([r["alive"] for r in rows if r["cycle"] == c])) for c in range(1, opt.cycles + 1)]
PROG_BY_CYCLE = [float(np.mean([r["prog"] for r in rows if r["cycle"] == c])) for c in range(1, opt.cycles + 1)]
print("\nprog ok by cycle:   " + "  ".join(f"{c}:{v:.3f}" for c, v in enumerate(PROG_BY_CYCLE, 1)))
print("run exact by cycle: " + "  ".join(f"{c}:{v:.3f}" for c, v in enumerate(BY_CYCLE, 1)))


# ── GIF: one batch executing, every NCA step ────────────────────────────────

COL = {k: np.array(v) for k, v in T["TileLayoutV8"].COLORS.items()}
FONT = ImageFont.truetype("DejaVuSans.ttf", 20)


def rgb(ch0, ins):
    img = np.full(ch0.shape + (3,), 0.5)
    mem = np.abs(ch0) > 0.05
    img[mem] = np.clip((ch0[mem] + 1) / 2, 0, 1)[:, None]
    mag = np.clip(np.stack([ins[0], ins[1], ins[2], -ins[2]]), 0, 1)       # A, B, SUB, SUBLEQ
    cols = np.stack([COL["A"], COL["B"], COL["R+"], COL["R-"]])
    tint = np.einsum("khw,kc->hwc", mag, cols) / np.maximum(mag.sum(0), 1e-6)[..., None]
    alpha = (mag.max(0) * np.where(mem, 0.5, 1.0))[..., None]   # a stamp over memory is see-through
    return img * (1 - alpha) + tint * alpha


@torch.no_grad()
def gif():
    rng = random.Random(opt.seed + 1)
    n, nr = rng.randint(*T["N_RANGE"]), rng.randint(*T["R_RANGE"])
    runs = [T["fresh_run"](n, nr, rng) for _ in range(opt.gif_batch)]
    x = T["start_state"](runs)
    frames, S, gap = [], 6, 12
    for c in range(opt.gif_cycles):
        seq = nca(x, K).cpu().numpy()
        for t in range(0 if c == 0 else 1, K + 1):
            tiles = [np.kron(rgb(seq[i, t, 0], seq[i, t, 1:4]), np.ones((S, S, 1)))
                     for i in range(opt.gif_batch)]
            h, w = tiles[0].shape[:2]
            canvas = np.ones((h + 34, len(tiles) * (w + gap) - gap, 3))
            for i, tl in enumerate(tiles):
                canvas[34:, i * (w + gap):i * (w + gap) + w] = tl
            im = Image.fromarray((canvas * 255).astype(np.uint8))
            phase = "load" if t <= LS and t > 0 else "execute" if t > LS else "start"
            ImageDraw.Draw(im).text((4, 4), f"cycle {c + 1}   step {t:2d}/{K}   {phase}", fill=(0, 0, 0),
                                    font=FONT)
            frames.append(im)
        x = torch.from_numpy(seq[:, -1]).to(opt.device)
    frames[0].save(ROOT / opt.gif, save_all=True, append_images=frames[1:], duration=60, loop=0)
    print(f"\nwrote {opt.gif}: {opt.gif_batch} samples, {n} instructions, {len(frames)} frames")


gif()


# ── curves: the training log, split by op ───────────────────────────────────

def curves():
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = [json.loads(l) for l in (run / "train.jsonl").read_text().splitlines() if l.strip()]
    if len(recs) < 2:
        print("no training log yet"); return

    def ema(v, a=0.05):
        out, m = [], None
        for x in v:
            m = x if m is None or x != x else (1 - a) * m + a * x
            out.append(m)
        return np.array(out)

    step = np.array([r["step"] for r in recs])
    get = lambda k: ema([r.get(k, float("nan")) for r in recs])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8), facecolor="white")
    a1.plot(step, get("loss"), color="#1d1c1a", lw=1.8, label="total")
    a1.plot(step, ema([r["load"] + r["tape"] for r in recs]), color="#2f6fe4", lw=1.8, alpha=0.35, label="instr (all)")
    a1.plot(step, get("mem"), color="#e8820c", lw=1.8, alpha=0.35, label="mem (all)")
    for op, ls in (("sub", "-"), ("subleq", "--")):
        a1.plot(step, get(f"instr_{op}"), color="#2f6fe4", ls=ls, label=f"instr {op.upper()}")
        a1.plot(step, get(f"mem_{op}"), color="#e8820c", ls=ls, label=f"mem {op.upper()}")
        a2.plot(step, get(f"bit_{op}"), color="#16a3a0", ls=ls, label=f"dst bits {op.upper()}")
        a2.plot(step, get(f"reg_{op}"), color="#d6247f", ls=ls, label=f"dst register {op.upper()}")
    a1.set_yscale("log"); a1.set_title("loss (training batches, smoothed)"); a1.set_xlabel("step")
    a2.set_ylim(0, 1.02); a2.set_title("accuracy when the instruction executes"); a2.set_xlabel("step")
    for a in (a1, a2):
        a.grid(alpha=0.25); a.legend(fontsize=9, frameon=False)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "v15_progress_curves.png", dpi=130, facecolor="white")
    print(f"wrote v15_progress_curves.png ({len(recs)} log records, up to step {step[-1]:,})")


curves()


# ── the report page: open it once, it reloads itself ────────────────────────

def page():
    """Self-contained (images inlined): published as an artifact, or opened locally."""
    import html
    import json
    import time
    ev = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()] \
        if (run / "metrics.jsonl").exists() else []
    cyc = [k for k in (ev[-1] if ev else {}) if k.startswith(("cyc", "prog"))]
    f = lambda v, d=4: f"{v:.{d}f}"
    trs = "".join(
        f"<tr class='{k.lower()}'><td>{k}</td><td>{m['n']}</td><td>{f(m['instr'], 5)}</td><td>{f(m['mem'], 5)}</td>"
        f"<td><b>{f(m['instr'] + m['mem'], 5)}</b></td><td>{f(m['bit'])}</td><td>{f(m['reg'])}</td>"
        f"<td>{f(m['memory'])}</td><td>{f(m['prog'])}</td><td>{f(m['alive'])}</td></tr>" for k, m in TABLE.items())
    evr = "".join(f"<tr><td>{e['step']:,}</td><td>{f(e['loss'], 5)}</td>"
                  + "".join(f"<td>{e[k]:.2f}</td>" if k in e else "<td></td>" for k in cyc)
                  + "</tr>" for e in ev[::-1][:12])
    import base64
    inline = lambda name, mime: f"data:{mime};base64," + base64.b64encode((ROOT / name).read_bytes()).decode()
    curves_src = inline("v15_progress_curves.png", "image/png") if (ROOT / "v15_progress_curves.png").exists() else ""
    gif_src = inline(opt.gif, "image/gif")
    out = f"""<title>v15 Tape Machine</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* one narrow column of cards: summary table first, then curves, film, history */
:root{{--bg:#f6f5f1;--card:#ffffff;--line:#e4e2db;--fg:#1d1c1a;--dim:#7d7a72;--accent:#2f6fe4;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:-apple-system,system-ui,"Segoe UI",sans-serif}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#17171a;--card:#202024;--line:#33333a;
  --fg:#e9e7e1;--dim:#9a978f;--accent:#7aa2ff;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#17171a;--card:#202024;--line:#33333a;--fg:#e9e7e1;--dim:#9a978f;--accent:#7aa2ff;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:13px/1.45 var(--sans)}}
main{{max-width:760px;margin:0 auto;padding-block:16px 28px;display:grid;gap:12px;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
h1{{font-size:16px;margin:0;text-wrap:balance}} .dim{{color:var(--dim)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:12px;min-width:0}}
h2{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin:0 0 8px;font-weight:600}}
.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font:12px var(--mono);font-variant-numeric:tabular-nums}}
td,th{{padding:3px 6px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}
td:first-child,th:first-child{{text-align:left}} th{{color:var(--dim);font-weight:normal}}
tr.all td{{font-weight:600}}
img{{width:100%;display:block;border-radius:3px;background:#fff}}
</style>
<main>
<div><h1>v15 tape machine <span class="dim">· {html.escape(run.name)}</span></h1>
<div class="dim">checkpoint step {ck['step']:,} · built {time.strftime('%Y-%m-%d %H:%M')}</div></div>
<div class="card"><h2>loss report · closed loop, {opt.batches}×{opt.batch} runs × {opt.cycles} cycles</h2>
<div class="scroll"><table><tr><th></th><th>n</th><th>loss instr</th><th>loss mem</th><th>loss total</th>
<th>dst bits</th><th>dst reg</th><th>memory</th><th>prog ok</th><th>run exact</th></tr>{trs}</table></div>
<div class="dim" style="margin-top:6px">prog ok (memory right every cycle so far) by cycle: {" · ".join(f"{c}: {v:.3f}" for c, v in enumerate(PROG_BY_CYCLE, 1))}</div>
<div class="dim">run exact (also every tape pixel within 0.5) by cycle: {" · ".join(f"{c}: {v:.3f}" for c, v in enumerate(BY_CYCLE, 1))}</div></div>
<div class="card"><h2>training curves</h2><img alt="loss and accuracy curves" src="{curves_src}"></div>
<div class="card"><h2>a batch executing</h2><img alt="NCA executing a batch" src="{gif_src}"></div>
<div class="card"><h2>eval history · trainer, {len(ev)} evals</h2>
<div class="scroll"><table><tr><th>step</th><th>loss</th>{"".join(f"<th>{k}</th>" for k in cyc)}</tr>{evr}</table></div></div>
</main>"""
    (ROOT / "v15_report.html").write_text(out)
    print("wrote v15_report.html")


page()
