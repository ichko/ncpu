#!/usr/bin/env python3
"""Status for a v11 run: per-phase loss curves + a free-running rollout GIF."""
import argparse, glob, json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from ncpu.nca import NeuralCA
from ncpu.program import Instr, TileLayoutV11

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="")
ap.add_argument("--every", type=int, default=6, help="keep every Nth CA frame in the gif")
ap.add_argument("--cur", type=int, default=0, help="program length for the gif (0 = run's current)")
a = ap.parse_args()

run = Path(a.run) if a.run else Path(sorted(glob.glob("runs/V11_*"))[-1])
cfg = json.loads((run / "config.json").read_text())
rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
print(f"{run.name}: {len(rows)} log rows, last step {rows[-1]['step']}")

# ── loss plot ───────────────────────────────────────────────────────────────
st = [r["step"] for r in rows]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(st, [r["loss"] for r in rows], lw=1, color="k")
ax[0].set_yscale("log"); ax[0].set_title("total loss"); ax[0].set_xlabel("step")
for ph, c in zip(("ADVANCE", "FETCH", "EXEC", "STORE"), ("#59a14f", "#4472d0", "#17becf", "#ff9d3a")):
    ys = [(r["step"], r[f"L_{ph}"]) for r in rows if f"L_{ph}" in r]
    if ys: ax[1].plot(*zip(*ys), lw=1, color=c, label=ph)
ax[1].set_yscale("log"); ax[1].legend(fontsize=8); ax[1].set_title("per-phase loss"); ax[1].set_xlabel("step")
ev = [r for r in rows if "acc_all" in r]
if ev:
    ax[2].plot([r["step"] for r in ev], [r["acc_regs"] for r in ev], lw=1.4, label="registers exact")
    ax[2].plot([r["step"] for r in ev], [r["acc_all"] for r in ev], lw=1.4, label="registers + tape")
    ax[2].plot([r["step"] for r in ev], [r["cur"] for r in ev], lw=1, ls="--", color="#999", label="program length")
ax[2].set_ylim(-0.05, max(2.2, cfg["n_instr"] + 0.2)); ax[2].legend(fontsize=8)
ax[2].set_title("free-running accuracy"); ax[2].set_xlabel("step")
fig.suptitle(run.name, fontsize=11); fig.tight_layout()
fig.savefig("v11_loss.png", dpi=140, facecolor="white"); print("wrote v11_loss.png")

# ── rollout gif ─────────────────────────────────────────────────────────────
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L = TileLayoutV11(n_instr=cfg["n_instr"], n_regs=cfg["n_regs"], word_bits=cfg["word_bits"],
                  r=cfg.get("r", 2), pitch=cfg.get("pitch", 8), pad=cfg.get("pad", 6))
nca = NeuralCA(channels=cfg["channels"], hidden_channels=cfg["hidden"], fire_rate=cfg["fire_rate"],
               alive_threshold=0, zero_initialization=False, kernel_size=3,
               padding_type="zeros", state_bound=cfg.get("state_bound", 10.0), soft_bound=not cfg.get("hard_bound", True)).to(DEV)
sd = torch.load(run / "nca_latest.pt", map_location=DEV)
nca.load_state_dict(sd["model"]); nca.eval()
cur = a.cur or sd.get("cur", 1)

rng = np.random.default_rng(0)
prog = [Instr(("SUB", "SUBLEQ", "NOP")[rng.integers(3)], int(rng.integers(cfg["n_regs"])),
              int(rng.integers(cfg["n_regs"])), int(rng.integers(cfg["n_regs"]))) for _ in range(cur)]
regs = [int(rng.integers(1 << cfg["word_bits"])) for _ in range(cfg["n_regs"])]
traj = L.cycle(prog, regs)
d, m = L.render(prog, traj[0][0])
x = torch.zeros(1, cfg["channels"], L.H, L.W, device=DEV)
x[0, 0] = torch.from_numpy(d).to(DEV); x[0, 1:5] = torch.from_numpy(m).to(DEV)

frames, tgt_rgb, labels = [], [], []
torch.set_grad_enabled(False)
for st_, ph, nx in traj:
    tg = L.to_rgb(*L.render(prog, nx))
    for _ in range(cfg["K"][ph]):
        x = nca(x, 1)[:, -1]
        frames.append(L.to_rgb(x[0, 0].cpu().numpy(), x[0, 1:5].cpu().numpy()))
        tgt_rgb.append(tg); labels.append(ph)

got_r, got_s = L.decode(x[0, 0].cpu().numpy())
want = traj[-1][2]
print(f"  program {[f'{i.op}({i.a},{i.b})->{i.dst}' for i in prog]}")
print(f"  regs want {tuple(want.regs)}  got {got_r}   slots {got_s}")

gif = []
for i in range(0, len(frames), a.every):
    pair = np.concatenate([frames[i], np.full((L.H, 3, 3), 255, np.uint8), tgt_rgb[i]], 1)
    gif.append(Image.fromarray(pair).resize((pair.shape[1] * 3, pair.shape[0] * 3), Image.NEAREST))
gif[0].save(run / "rollout.gif", save_all=True, append_images=gif[1:], duration=40, loop=0)
print(f"wrote {run / 'rollout.gif'}  (left: rule, right: phase target)")
