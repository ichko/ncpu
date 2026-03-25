#!/usr/bin/env python3
"""
ALU v2 — 8-bit full ALU with carry and branch output.

Grid: 160×112, r=4
  x=16  A[7:0]
  x=32  B[7:0]          (blank for NOT/RCL/RCR)
  x=60  CTRL: op[2:0] + carry_in + cond[2:0]   — 7 bits
  x=144 OUT:  result[7:0] + carry_out + branch_taken  — 10 bits

Operations: ADD SUB AND OR XOR NOT RCL RCR
Conditions:  EQ NE CS CC MI PL AL NV

Fixes over old ALU:
  - zi=False  (ablation shows zi=True fails)
  - ks=7
  - hidden=[256]  (more capacity for 8-op routing)
  - Curriculum: bitwise → +arith → +rotate
  - Step noise: σ=0.1 on 10% of steps, 50% of each batch
  - grad_clip=1.0

Usage:
    uv run python scripts/train_alu2.py
    uv run python scripts/train_alu2.py --seed 1 --steps 300000
    uv run python scripts/train_alu2.py --resume last
"""

import json
import random
import shutil
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.dataset import ALU2Dataset
from ncpu.nca import NeuralCA
from ncpu.normalizers import normalize_neg1_to_1
from ncpu.utils import freeze_frame, git_info, make_grid, save_grid_image

# ── Args ───────────────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--seed",   type=int, default=0)
parser.add_argument("--steps",  type=int, default=300_000)
parser.add_argument("--device", default="cuda")
parser.add_argument("--resume", default=None, help="run name or 'last'")
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
DEVICE = args.device if torch.cuda.is_available() else "cpu"

# ── Config ─────────────────────────────────────────────────────────────────────

TOTAL_STEPS = args.steps
PLOT_EVERY  = 500
SAVE_EVERY  = 5000

# Grid / dataset
DS = dict(W=96, H=112, r=4, among_sp=2, x_a=12, x_b=24, x_ctrl=48, x_out=82)

# NCA
NCA_CFG = dict(
    channels=16,
    hidden_channels=[128],
    fire_rate=0.5,
    alive_threshold=0,
    zero_initialization=False,   # zi=True provably fails (E_ZI ablation)
    kernel_size=7,
    num_perception_kernels=3,
    read_only_dims=[1],
    padding_type="zeros",
)

# Training
# Memory budget: batch × steps × hidden × H × W × 4 bytes (fwd + bwd)
#   4 × 128 × 128 × 112 × 160 × 4 ≈ 3.7 GB  — safe on 24 GB
OPTIM = dict(lr=1e-4, batch_size=8, steps_min=48, steps_max=96, grad_clip=1.0)

# Noise: σ=0.1 injected on 10% of steps, applied to 50% of each batch
NOISE_SIGMA   = 0.1
NOISE_P_STEP  = 0.1
NOISE_P_BATCH = 0.5

N_LAST = 8   # average loss over last N rollout frames

# No curriculum — all 8 ops sampled uniformly at random every step

# ── Run directory ──────────────────────────────────────────────────────────────

from argparse import Namespace
ds_ns = Namespace(**DS)

if args.resume == "last":
    candidates = sorted(
        d.name for d in Path("runs").iterdir()
        if d.name.startswith("E_alu2_") and (d / "checkpoints" / "nca_latest.pt").exists()
    )
    if not candidates:
        raise RuntimeError("No resumable E_alu2 runs found")
    args.resume = candidates[-1]

if args.resume:
    run_dir = Path("runs") / args.resume
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    saved_cfg = json.load(open(run_dir / "config.json"))
    start_step = saved_cfg["run"].get("last_step", 0)
    print(f"Resuming {run_dir.name} from step {start_step}")
else:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"E_alu2_s{args.seed}_{ts}"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "rollouts").mkdir()
    (run_dir / "snapshots").mkdir()
    start_step = 0
    json.dump({
        "run":   {"name": run_dir.name, "experiment": "E_alu2", "seed": args.seed,
                  "total_steps": TOTAL_STEPS, "last_step": 0, "device": DEVICE},
        "ds":    DS,
        "nca":   NCA_CFG,
        "optim": OPTIM,
        "env":   {"python": sys.version, "torch": torch.__version__, "git": git_info()},
    }, open(run_dir / "config.json", "w"), indent=2)

log_path = run_dir / "log.jsonl"
ckpt_dir = run_dir / "checkpoints"
print(f"ALU v2  seed={args.seed}  run: {run_dir.name}\n")

# ── Model ──────────────────────────────────────────────────────────────────────

nca = NeuralCA(**NCA_CFG).to(DEVICE)

if args.resume:
    ckpt = ckpt_dir / "nca_latest.pt"
    nca.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False)
    print(f"Loaded checkpoint: {ckpt}")

optim = torch.optim.Adam(nca.parameters(), lr=OPTIM["lr"])

# ── Loss & metric helpers ──────────────────────────────────────────────────────

def build_masks(ds):
    """Returns (out_mask, bit_masks) on DEVICE."""
    _, out_screen = ds.get_io_mask()
    out_mask = torch.from_numpy(out_screen > 200).float().to(DEVICE)   # (H, W)
    bit_masks = ds.get_output_bit_masks().to(DEVICE)                   # (10, H, W)
    return out_mask, bit_masks


def count_valid_bits(ch0_last, out_norm, bit_masks):
    """Fraction of the 10 output bits with correct sign, averaged over batch."""
    B = ch0_last.shape[0]
    n_bits = bit_masks.shape[0]
    correct = 0
    for i in range(n_bits):
        mask   = bit_masks[i]                                          # (H, W)
        pred   = (ch0_last * mask).sum(dim=(-2, -1)) / mask.sum()     # (B,)
        target = (out_norm    * mask).sum(dim=(-2, -1)) / mask.sum()  # (B,)
        correct += ((pred > 0) == (target > 0)).float().sum()
    return (correct / (B * n_bits)).item()


def rollout_loss(rollout, out_norm, out_mask):
    """MSE over last N_LAST frames, restricted to output-circle pixels."""
    pred   = rollout[:, -N_LAST:, 0]                    # (B, N, H, W)
    target = out_norm.unsqueeze(1).expand_as(pred)      # (B, N, H, W)
    mask   = out_mask.unsqueeze(0).unsqueeze(0)         # (1, 1, H, W)
    return ((pred - target) ** 2 * mask).sum() / (mask.sum() * pred.shape[0] * N_LAST)

# ── Dataset ────────────────────────────────────────────────────────────────────

ds = ALU2Dataset(ds_ns)   # all 8 ops, uniform random
out_mask, bit_masks = build_masks(ds)
loader_iter = iter(ds.get_dataloader(OPTIM["batch_size"]))

# ── Training loop ──────────────────────────────────────────────────────────────

metrics = []

pbar = tqdm(range(start_step, start_step + TOTAL_STEPS), ncols=110,
            desc=f"alu2 s{args.seed}")

for step in pbar:
    # ── Sample batch ───────────────────────────────────────────────────────────
    try:
        inp, out = next(loader_iter)
    except StopIteration:
        loader_iter = iter(ds.get_dataloader(OPTIM["batch_size"]))
        inp, out = next(loader_iter)

    inp = inp.to(DEVICE)
    out = out.to(DEVICE)
    B   = inp.shape[0]

    # Normalise: pixel → [-1, 1]
    inp_norm = inp / 128.0 - 1.0
    out_norm = out / 128.0 - 1.0

    # ── Build initial state ────────────────────────────────────────────────────
    H, W = DS["H"], DS["W"]
    C    = NCA_CFG["channels"]
    state = torch.zeros(B, C, H, W, device=DEVICE)
    state[:, 0] = inp_norm
    state[:, 1] = inp_norm   # read-only channel carries the input

    # ── Noise injection ────────────────────────────────────────────────────────
    if random.random() < NOISE_P_STEP:
        noise_sel = (torch.rand(B, device=DEVICE) < NOISE_P_BATCH)[:, None, None, None]
        state = state + noise_sel * torch.randn_like(state) * NOISE_SIGMA

    # ── Forward + loss ─────────────────────────────────────────────────────────
    T = random.randint(OPTIM["steps_min"], OPTIM["steps_max"])
    rollout = nca(state, steps=T)    # (B, T+1, C, H, W)

    loss = rollout_loss(rollout, out_norm, out_mask)

    optim.zero_grad()
    loss.backward()
    if OPTIM["grad_clip"]:
        torch.nn.utils.clip_grad_norm_(nca.parameters(), OPTIM["grad_clip"])
    optim.step()

    # ── Metrics ────────────────────────────────────────────────────────────────
    with torch.no_grad():
        valid = count_valid_bits(rollout[:, -1, 0], out_norm, bit_masks)

    loss_val = loss.item()
    metrics.append({"step": step, "loss": loss_val, "valid_bits": valid})

    pbar.set_description(
        f"alu2 s{args.seed}  loss={loss_val:.4f}  bits={valid*10:.1f}/10"
    )

    with open(log_path, "a") as f:
        f.write(json.dumps({"step": step, "loss": loss_val,
                            "num_valid_bits": valid * 10}) + "\n")

    if step % PLOT_EVERY != 0:
        continue

    # ── Checkpoint ─────────────────────────────────────────────────────────────
    ckpt_step = ckpt_dir / f"nca_{step:07d}.pt"
    torch.save(nca.state_dict(), ckpt_step)
    torch.save(nca.state_dict(), ckpt_dir / "nca_latest.pt")
    # update last_step in config
    cfg = json.load(open(run_dir / "config.json"))
    cfg["run"]["last_step"] = step
    json.dump(cfg, open(run_dir / "config.json", "w"), indent=2)

    # ── Plots ──────────────────────────────────────────────────────────────────
    losses = [m["loss"]        for m in metrics]
    bits   = [m["valid_bits"]  for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].scatter(range(len(losses)), losses, s=0.4, alpha=0.3, color="steelblue")
    axes[0].set_yscale("log"); axes[0].set_title("Loss"); axes[0].set_xlabel("step")
    axes[1].scatter(range(len(bits)), [b * 10 for b in bits],
                    s=0.4, alpha=0.3, color="darkorange")
    axes[1].set_ylim(0, 10.5); axes[1].set_title("Valid bits / 10")
    axes[1].set_xlabel("step")
    fig.suptitle(f"ALU v2 — seed={args.seed}  step={step}", fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=110)
    plt.close(fig)

    # ── Snapshot ───────────────────────────────────────────────────────────────
    snap_b  = min(B, 4)
    nca_out = rollout[:snap_b, -1, 0]            # (snap_b, H, W)
    inp_vis = inp_norm[:snap_b]                  # (snap_b, H, W)
    out_vis = out_norm[:snap_b]
    diff    = (nca_out - out_vis).abs()
    save_grid_image(
        run_dir / "snapshot_latest.png",
        [inp_vis.cpu(), out_vis.cpu(), nca_out.detach().cpu(), diff.detach().cpu()],
        row_vmin=[None, None, None, 0],
        row_vmax=[None, None, None, 2],
    )

    # ── Rollout GIF ────────────────────────────────────────────────────────────
    gif_b   = min(B, 4)
    gif_t   = rollout[:gif_b, ::2]                           # subsample time
    C_dim   = gif_t.shape[2]
    frames  = gif_t.permute(2, 0, 1, 3, 4).reshape(
        C_dim * gif_b, gif_t.shape[1], DS["H"], DS["W"]
    ).detach().cpu().numpy()
    frames_rgb = torch.from_numpy(
        media.to_rgb(frames, vmin=-1, vmax=1, cmap="viridis")
    )
    grid = make_grid(frames_rgb, nrow=gif_b, padding=1)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=8)
    media.write_video(str(run_dir / "rollout_latest.gif"),
                      grid.numpy(), fps=10, codec="gif")

    if step % SAVE_EVERY == 0 and step > 0:
        shutil.copy(run_dir / "snapshot_latest.png",
                    run_dir / "snapshots" / f"snapshot_{step:07d}.png")
        shutil.copy(run_dir / "rollout_latest.gif",
                    run_dir / "rollouts"  / f"rollout_{step:07d}.gif")

print(f"\nDone. Run: {run_dir}")
