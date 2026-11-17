#!/usr/bin/env python3
"""Growing-NCA image training (the "gecko" example), wired to the rolling window.

A single small rule learns to *grow* a target image (RGBA) from a sparse seed of
alive cells. The rule only ever sees a local ``window x window`` crop of the
board, so it learns a local growth texture — training on a small crop transfers
to a much larger canvas for free.

Two execution paths are demonstrated:

* **Train**  — classic Growing-NCA recipe (Mordvintsev 2020) on a fixed small
  board: alpha channel gates updates, a stochastic fire-rate masks the delta,
  and an experience-replay pool carries partially-grown states between steps.
  The loss is *damage-free* (RGB supervised only where the target is opaque)
  plus an overflow penalty on out-of-body alpha.
* **Eval**   — the trained rule is rolled out on a larger ``--canvas`` board
  with plain global updates, and (with ``--sweep-eval``) through a
  :class:`~ncpu.sweep.WindowSweep` to show the size-agnostic stitch.

Channel layout (``channels``, default 16):
    0       — alpha / alive
    1, 2, 3 — RGB (normalised to [-1, 1])
    4..15   — hidden

Usage:
    uv run python scripts/train/train_gecko.py --seed 0 --steps 4000
    uv run python scripts/train/train_gecko.py --image gecko.png --size 48
    uv run python scripts/train/train_gecko.py --sweep-eval --canvas 256
    uv run python scripts/train/train_gecko.py --resume last
"""

import json
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.nca_utils import GaussianNoise, NCARule, SobelPerception, StochasticUpdate
from ncpu.sweep import SweepPlan, WindowSweep
from ncpu.utils import git_info

# ── Channel layout ──────────────────────────────────────────────────────────────

ALPHA_CH = 0
RGB_CH = [1, 2, 3]


# ── The growing rule ────────────────────────────────────────────────────────────


class GrowingNCA(nn.Module):
    """A growing NCA whose ``step`` satisfies the ``WindowSweep`` contract.

    The update follows the Growing-NCA paper: the delta is fire-rate masked,
    then cells are kept alive iff they were alive *before or after* the update
    (an OR mask applied to the state), so dead cells adjacent to alive ones can
    turn on and growth proceeds outward from the seed.
    """

    def __init__(
        self,
        channels=16,
        hidden_channels=(128,),
        fire_rate=0.5,
        alive_threshold=0.1,
        kernel_size=3,
        zero_initialization=False,
    ):
        super().__init__()
        self.is_stateful = False
        self.channels = channels
        self.alive_threshold = alive_threshold
        self.perception = SobelPerception(kernel_size, channels, "zeros")
        self.rule = NCARule(3 * channels, hidden_channels, channels, zero_initialization)
        self.stochastic_update = StochasticUpdate(fire_rate)
        self.gaussian_noise = GaussianNoise(channels, [])

    def alive(self, x):
        alpha = x[:, :1]
        pad = self.perception.kernel_size // 2
        pooled = F.max_pool2d(
            F.pad(alpha, (pad,) * 4, self.perception.padding_type),
            self.perception.kernel_size,
            stride=1,
        )
        return pooled > self.alive_threshold

    def step(self, x, update_mask=None):
        dx = self.rule(self.perception(x))
        dx = self.stochastic_update(dx)
        if update_mask is not None:
            dx = dx * update_mask
        pre = self.alive(x)
        post = self.alive(x + dx)
        mask = (pre | post).to(x.dtype)
        x = (x + dx) * mask
        return self._clip(self.gaussian_noise(x))

    @staticmethod
    def _clip(x):
        return torch.cat(
            [
                x[:, :1].clamp(0, 1),      # alpha
                x[:, 1:4].clamp(-1, 1),    # RGB
                x[:, 4:].clamp(-5, 5),     # hidden
            ],
            dim=1,
        )

    def forward(self, x, steps):
        seq = [x]
        for _ in range(steps):
            x = self.step(x)
            seq.append(x)
        return torch.stack(seq).permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)


# ── Target image ────────────────────────────────────────────────────────────────


def make_demo_target(size=48):
    """A small lizard-ish sprite so the script runs with no asset on disk."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    s = size / 48.0
    body = ((round(18 * s), round(14 * s)), (round(20 * s), round(8 * s)), 0)
    cv2.ellipse(img, body, (60, 160, 60, 255), -1)  # body
    cv2.circle(img, (round(40 * s), round(16 * s)), round(9 * s), (40, 140, 40, 255), -1)  # head
    tail = np.array(
        [[round(6 * s), round(30 * s)], [round(16 * s), round(22 * s)],
         [round(6 * s), round(16 * s)]],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [tail], (100, 200, 100, 255))  # tail
    for dy in (-1, 1):  # legs
        cv2.ellipse(
            img,
            ((round(16 * s), round(16 * s + dy * 12 * s)), (round(9 * s), round(4 * s)), 30 * dy),
            (90, 170, 90, 255),
            -1,
        )
    cv2.circle(img, (round(43 * s), round(13 * s)), round(2 * s), (0, 0, 0, 255), -1)  # eye
    return img


def load_target(path, size):
    if path is None:
        rgba = make_demo_target(64)
    else:
        bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if bgra is None:
            raise FileNotFoundError(path)
        if bgra.shape[2] == 3:
            bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2BGRA)
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
    rgba = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA)
    return rgba


def normalize_target(rgba):
    rgb = (rgba[..., :3].astype(np.float32) / 255.0) * 2 - 1  # (H, W, 3) -> [-1,1]
    alpha = (rgba[..., 3:4].astype(np.float32) / 255.0)        # (H, W, 1) -> [0,1]
    rgb = torch.from_numpy(rgb).permute(2, 0, 1)               # (3, H, W)
    alpha = torch.from_numpy(alpha).permute(2, 0, 1)           # (1, H, W)
    rgb = rgb * (alpha[0] > 0.5).unsqueeze(0).float()          # background neutral
    return rgb, alpha


def make_seed(rgb_t, alpha_t, rng, channels=16, seed_radius=1):
    """One ``(C, H, W)`` state seeded with a small blob of alive cells."""
    _, H, W = alpha_t.shape
    alive = alpha_t[0] > 0.5
    ys, xs = alive.nonzero(as_tuple=True)
    if len(ys) == 0:
        ys, xs = torch.tensor([H // 2]), torch.tensor([W // 2])
    i = rng.randrange(len(ys))
    cy, cx = int(ys[i]), int(xs[i])
    state = torch.zeros(channels, H, W)
    for dy in range(-seed_radius, seed_radius + 1):
        for dx in range(-seed_radius, seed_radius + 1):
            y, x = cy + dy, cx + dx
            if 0 <= y < H and 0 <= x < W and alive[y, x]:
                state[ALPHA_CH, y, x] = 1.0
                state[RGB_CH, y, x] = rgb_t[:, y, x]
    return state


# ── Experience-replay pool ──────────────────────────────────────────────────────


class GrowingPool:
    """Stores partially-grown states so rollouts resume from a non-trivial stage."""

    def __init__(self, size, make_seed, reset_prob=0.05):
        self.size = size
        self.make_seed = make_seed
        self.reset_prob = reset_prob
        self.pool = [None] * size

    def sample(self, batch, device):
        idx = random.sample(range(self.size), batch)
        states = []
        for i in idx:
            if self.pool[i] is None:
                self.pool[i] = self.make_seed()
            states.append(self.pool[i])
        return torch.stack(states).to(device), idx

    def update(self, idx, finals):
        for i, s in zip(idx, finals.detach().cpu()):
            self.pool[i] = None if random.random() < self.reset_prob else s


def img_loss(x, rgb_t, alpha_t, overflow_weight):
    rgb = x[:, RGB_CH]
    alpha = x[:, :1]
    alive = (alpha_t > 0.1).to(x.dtype)
    rgb_loss = ((rgb - rgb_t) ** 2 * alive).sum() / alive.sum().clamp_min(1.0)
    overflow = torch.clamp(alpha - alpha_t, min=0).square().mean()
    return rgb_loss + overflow_weight * overflow


# ── Main ────────────────────────────────────────────────────────────────────────


def build_parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", type=str, default=None, help="run name, or 'last'")
    p.add_argument("--eval-every", type=int, default=250, dest="eval_every")
    p.add_argument("--save-rollout-npz", action="store_true")
    p.add_argument("--image", default=None, help="RGBA target; synthesized if omitted")
    p.add_argument("--size", type=int, default=40, help="train board size")
    p.add_argument("--canvas", type=int, default=160, help="eval board size")
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--hidden-channels", type=int, nargs="+", default=[128], dest="hidden_channels")
    p.add_argument("--fire-rate", type=float, default=0.5, dest="fire_rate")
    p.add_argument("--alive-threshold", type=float, default=0.1, dest="alive_threshold")
    p.add_argument("--kernel-size", type=int, default=3, dest="kernel_size")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    p.add_argument("--pool-size", type=int, default=1024, dest="pool_size")
    p.add_argument("--min-steps", type=int, default=32, dest="min_steps")
    p.add_argument("--max-steps", type=int, default=64, dest="max_steps")
    p.add_argument("--eval-steps", type=int, default=96, dest="eval_steps")
    p.add_argument("--overflow-weight", type=float, default=0.1, dest="overflow_weight")
    p.add_argument("--sweep-eval", action="store_true", help="also eval via WindowSweep")
    p.add_argument("--window", type=int, default=40, help="window size for the sweep eval")
    p.add_argument("--inner-steps", type=int, default=48, dest="inner_steps")
    return p


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    nca_config = dict(
        channels=args.channels,
        hidden_channels=list(args.hidden_channels),
        fire_rate=args.fire_rate,
        alive_threshold=args.alive_threshold,
        kernel_size=args.kernel_size,
    )

    # ── Run directory ──────────────────────────────────────────────────────────
    if args.resume:
        if args.resume == "last":
            runs = [
                d
                for d in Path("runs").iterdir()
                if d.name.startswith("gecko_") and (d / "checkpoints" / "nca_last.pt").exists()
            ]
            if not runs:
                raise RuntimeError("No resumable gecko_* runs found")
            args.resume = runs[-1].name
        run_dir = Path("runs") / args.resume
        ckpt_dir = run_dir / "checkpoints"
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        saved = json.loads((run_dir / "config.json").read_text())
        nca_config = saved["nca"]
        for key, value in saved["training"].items():
            if key not in ["steps", "device", "resume", "eval_every", "save_rollout_npz"]:
                setattr(args, key, value)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"gecko_s{args.seed}_{stamp}"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                dict(nca=nca_config, training=vars(args), env=git_info()), indent=2
            )
        )

    # ── Targets (train size + eval canvas) ─────────────────────────────────────
    train_rgb, train_alpha = normalize_target(load_target(args.image, args.size))
    canvas_rgb, canvas_alpha = normalize_target(load_target(args.image, args.canvas))
    train_rgb_b = train_rgb.unsqueeze(0).to(args.device)
    train_alpha_b = train_alpha.unsqueeze(0).to(args.device)
    canvas_rgb_b = canvas_rgb.unsqueeze(0).to(args.device)
    canvas_alpha_b = canvas_alpha.unsqueeze(0).to(args.device)

    seed_fn = lambda: make_seed(train_rgb, train_alpha, random, args.channels)

    nca = GrowingNCA(**nca_config).to(args.device)
    pool = GrowingPool(args.pool_size, seed_fn)
    optim = torch.optim.Adam(nca.parameters(), lr=args.lr)

    start_step = 0
    history, evals = [], []
    if args.resume:
        ckpt = torch.load(
            ckpt_dir / "nca_last.pt", map_location=args.device, weights_only=False
        )
        nca.load_state_dict(ckpt["nca"])
        optim.load_state_dict(ckpt["optim"])
        start_step = ckpt["step"]
        history, evals = ckpt.get("history", []), ckpt.get("evals", [])
        pool.pool = ckpt.get("pool", [None] * pool.size)

    # ── Eval helpers ───────────────────────────────────────────────────────────
    def grow_eval():
        seed = make_seed(canvas_rgb, canvas_alpha, random, args.channels)
        x = nca.forward(seed.unsqueeze(0).to(args.device), args.eval_steps)[:, -1]
        return img_loss(x, canvas_rgb_b, canvas_alpha_b, args.overflow_weight).item()

    def sweep_eval():
        window = min(args.window, args.canvas)
        plan = SweepPlan(
            H=args.canvas, W=args.canvas,
            wh=window, ww=window, hstride=window, wstride=window, n_passes=1,
        )
        board = torch.zeros(1, args.channels, args.canvas, args.canvas)
        alive = canvas_alpha[0] > 0.5
        for cy in range(window // 2, args.canvas, window):
            for cx in range(window // 2, args.canvas, window):
                if alive[cy, cx]:
                    board[0, ALPHA_CH, cy, cx] = 1.0
                    board[0, RGB_CH, cy, cx] = canvas_rgb[:, cy, cx]
        out = WindowSweep(nca)(board.to(args.device), plan, inner_steps=args.inner_steps)
        return img_loss(out, canvas_rgb_b, canvas_alpha_b, args.overflow_weight).item()

    def save_checkpoint(step):
        torch.save(
            dict(
                nca=nca.state_dict(),
                optim=optim.state_dict(),
                step=step,
                history=history,
                evals=evals,
                pool=pool.pool,
            ),
            ckpt_dir / "nca_last.pt",
        )

    # ── Training loop ──────────────────────────────────────────────────────────
    pbar = tqdm(range(start_step, args.steps), ncols=100, desc=f"gecko s{args.seed}")
    log_path = run_dir / "log.jsonl"

    for step in pbar:
        states, indices = pool.sample(args.batch_size, args.device)
        n_updates = random.randint(args.min_steps, args.max_steps)
        x = nca.forward(states, n_updates)[:, -1]
        loss = img_loss(x, train_rgb_b, train_alpha_b, args.overflow_weight)

        optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(nca.parameters(), 1.0)
        optim.step()

        pool.update(indices, x.detach())
        history.append(loss.item())
        pbar.set_description(f"gecko s{args.seed}  loss={loss.item():.4f}")

        with open(log_path, "a") as f:
            f.write(
                json.dumps(
                    dict(
                        step=step,
                        loss=loss.item(),
                        updates=n_updates,
                        grad_norm=grad_norm.item(),
                        ts=datetime.now().isoformat(),
                    )
                )
                + "\n"
            )

        if step % args.eval_every != 0 and step != args.steps - 1:
            continue

        with torch.no_grad():
            nca.eval()
            metrics = {"eval_loss": grow_eval()}
            if args.sweep_eval:
                metrics["sweep_loss"] = sweep_eval()
            nca.train()
        evals.append(dict(step=step, **metrics))
        print(f"\n  EVAL step {step}: {metrics}")
        save_checkpoint(step)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.scatter(range(len(history)), history, s=0.5, alpha=0.4, color="steelblue")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(f"growing NCA s{args.seed} — loss")
        fig.tight_layout()
        fig.savefig(run_dir / "loss_curve.png", dpi=120)
        plt.close(fig)

        if args.save_rollout_npz:
            np.savez_compressed(
                run_dir / "rollout_latest.npz",
                state=states[0].cpu().numpy(),
                grown=x[0].cpu().numpy(),
                target_rgb=train_rgb.numpy(),
                target_alpha=train_alpha.numpy(),
            )

    save_checkpoint(args.steps)
    print(f"\nDone. Run: {run_dir}")


if __name__ == "__main__":
    main()
