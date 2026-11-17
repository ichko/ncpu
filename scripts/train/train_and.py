#!/usr/bin/env python3
"""Two-stage AND curriculum: stationary gate, then stateful displaced windows."""

import json
import random
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ncpu.nca import NeuralCA
from ncpu.window import build_state, make_gate_tile, INPUT_CHANNEL


class ANDTrainer:
    def __init__(self, model, args):
        self.model, self.args = model, args
        self.rng = random.Random(args.seed)

    def batch(self, count):
        # Balance the truth table, including the otherwise rare positive AND.
        samples = []
        blank_count = count // 4 if getattr(self.args, "blank_examples", False) else 0
        for i in range(count + blank_count):
            a, b = divmod(i % 4, 2)
            # Blank inputs have no gate selector or dots; zero is midpoint gray.
            inp, sel, target, mask = make_gate_tile("AND" if i < count else None, a, b)
            samples.append(
                (
                    build_state(
                        inp, sel, position_gradient=self.args.position_gradient
                    ),
                    torch.from_numpy(target),
                    torch.from_numpy(mask),
                )
            )
        return tuple(
            torch.stack([s[i] for s in samples]).to(self.args.device) for i in range(3)
        )

    def displacement_origins(self, rng):
        d = self.args.displacement
        origin = [d, d]
        origins = [tuple(origin)]
        minimum = getattr(self.args, "hop_min", None)
        if minimum is None:
            # Preserve the exact sampling scheme for legacy checkpoint resumes.
            for _ in range(self.args.hops - 1):
                axis = rng.randrange(2)
                origin[axis] = max(
                    0, min(2 * d, origin[axis] + rng.choice([-1, 1]) * self.args.hop)
                )
                origins.append(tuple(origin))
            return origins
        axes = [i % 2 for i in range(self.args.hops - 1)]
        rng.shuffle(axes)  # Every rollout includes horizontal AND vertical motion.
        for axis in axes:
            distance = rng.randint(minimum, self.args.hop_max)
            directions = [
                sign for sign in [-1, 1] if 0 <= origin[axis] + sign * distance <= 2 * d
            ]
            origin[axis] += rng.choice(directions) * distance
            origins.append(tuple(origin))
        return origins

    def advance_window(self, state, y, x, steps):
        halo = getattr(self.args, "context_padding", 0)
        update_mask = None
        if getattr(self.args, "update_head", "uniform") == "gaussian":
            extent = 32 + 2 * halo
            coordinates = (
                torch.arange(extent, device=state.device, dtype=state.dtype)
                - (extent - 1) / 2
            )
            radius2 = coordinates[:, None].square() + coordinates[None, :].square()
            update_mask = torch.exp(
                -(radius2 - radius2.min()) / (2 * self.args.mask_sigma**2)
            )[None, None]
        padding = (x, state.shape[-1] - x - 32, y, state.shape[-2] - y - 32)
        region = F.pad(torch.ones_like(state[:, :1, :32, :32]), padding)
        for _ in range(steps):
            # Read neighbouring board state; update only the central window.
            context = F.pad(state, (halo, halo, halo, halo)) if halo else state
            crop = context[:, :, y : y + 32 + 2 * halo, x : x + 32 + 2 * halo]
            crop = (
                self.model.step(crop, update_mask=update_mask)
                if update_mask is not None
                else self.model.step(crop)
            )
            center = crop[:, :, halo : halo + 32, halo : halo + 32]
            state = state * (1 - region) + F.pad(center, padding)
        return state

    def rollout(self, batch, displaced, rng):
        state, target, mask = batch
        minimum = getattr(self.args, "rollout_steps_min", None)
        maximum = getattr(self.args, "rollout_steps_max", None)
        budget = rng.randint(minimum, maximum) if minimum is not None else None
        if displaced:
            d = self.args.displacement
            state, target, mask = [F.pad(t, (d, d, d, d)) for t in batch]
            origins = self.displacement_origins(rng)
            counts = (
                [
                    budget // len(origins) + (i < budget % len(origins))
                    for i in range(len(origins))
                ]
                if budget is not None
                else [self.args.updates_per_hop] * len(origins)
            )
        else:
            origins = [(0, 0)]
            counts = [budget if budget is not None else self.args.inner_steps]
        self.last_rollout_steps = sum(counts)
        self.last_origins = origins
        initial = state
        for (y, x), steps in zip(origins, counts):
            state = self.advance_window(state, y, x, steps)
        before = state
        state = self.advance_window(state, y, x, self.args.settle_steps)
        return initial, before, state, target, mask

    def loss(self, result):
        initial, before, state, target, mask = result
        pred = state[:, 0]

        def masked(value, weights):
            return (value * weights).sum() / weights.sum().clamp_min(1)

        has_gate = initial[:, 2].abs().sum((-2, -1)) > 0
        output_mask = mask * has_gate[:, None, None]
        # Equal per-sample weight prevents board size / batch size dilution.
        output = masked((pred - target).square(), output_mask)
        output = (output + masked((before[:, 0] - target).square(), output_mask)) / 2
        inputs = (initial[:, INPUT_CHANNEL].abs() > 0).float()
        preserve = masked((pred - initial[:, INPUT_CHANNEL]).square(), inputs)
        background = (1 - mask) * (1 - inputs)
        quiet = masked(pred.square(), background)
        stable = masked((pred - before[:, 0]).square(), mask)
        hidden = state[:, 4 if self.args.position_gradient else 3 :].square().mean()
        blank_mask = (~has_gate)[:, None, None].expand_as(pred)
        blank = (
            masked(pred.square(), blank_mask)
            + masked(before[:, 0].square(), blank_mask)
        ) / 2
        terms = dict(
            output=output,
            inputs=preserve,
            background=quiet,
            stability=stable,
            hidden=hidden,
            blank=blank,
        )
        weights = [
            self.args.output_weight,
            self.args.input_weight,
            self.args.background_weight,
            self.args.stability_weight,
            self.args.hidden_weight,
            getattr(self.args, "blank_weight", 1.0),
        ]
        total = sum(w * t for w, t in zip(weights, terms.values()))
        # Gate-level metric: classify the mean output circle, not individual pixels.
        values = (pred * mask).sum((-2, -1)) / mask.sum((-2, -1))
        expected = (target * mask).sum((-2, -1)) / mask.sum((-2, -1))
        correct = ((values > 0) == (expected > 0)).float()
        acc = (correct * has_gate).sum() / has_gate.sum().clamp_min(1)
        return total, terms, acc

    @torch.no_grad()
    def evaluate(self, fixed_batch):
        self.model.eval()
        metrics, rollouts = {}, {}
        # Repeatable stochastic masks, without disturbing the training RNG.
        devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=devices):
            for displaced, name in [(False, "stationary"), (True, "displaced")]:
                losses, accs, blank_mses, blank_peaks = [], [], [], []
                for i in range(self.args.n_eval_rollouts):
                    torch.manual_seed(self.args.seed + 10000 + i)
                    result = self.rollout(
                        fixed_batch, displaced, random.Random(10000 + i)
                    )
                    loss, _, acc = self.loss(result)
                    losses.append(loss.item())
                    accs.append(acc.item())
                    is_blank = result[0][:, 2].abs().sum((-2, -1)) == 0
                    if is_blank.any():
                        blank_output = result[2][is_blank, 0]
                        blank_mses.append(blank_output.square().mean().item())
                        blank_peaks.append(blank_output.abs().max().item())
                metrics[name + "_loss"] = float(np.mean(losses))
                metrics[name + "_acc"] = float(np.mean(accs))
                if blank_mses:
                    metrics[name + "_blank_mse"] = float(np.mean(blank_mses))
                    metrics[name + "_blank_peak"] = float(max(blank_peaks))
                rollouts[name] = result
        self.model.train()
        return metrics, rollouts


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--position-gradient", action=BooleanOptionalAction, default=True)
    p.add_argument(
        "--blank-examples",
        action=BooleanOptionalAction,
        default=True,
        help="add one no-gate, uniform-background example per four AND examples",
    )
    p.add_argument("--context-padding", type=int, default=3)
    p.add_argument("--update-head", choices=["uniform", "gaussian"], default="uniform")
    p.add_argument("--mask-sigma", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", help="run directory or 'last'")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-rollout-npz", action="store_true")
    p.add_argument("--stage1-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--inner-steps", type=int, default=48)
    p.add_argument(
        "--rollout-steps-min",
        type=int,
        default=64,
        help="minimum total NCA updates per rollout, excluding settling",
    )
    p.add_argument(
        "--rollout-steps-max",
        type=int,
        default=96,
        help="maximum total NCA updates per rollout, inclusive",
    )
    p.add_argument("--updates-per-hop", type=int, default=8)
    p.add_argument("--hops", type=int, default=6)
    p.add_argument("--hop", type=int, default=2)
    p.add_argument("--hop-min", type=int, default=2)
    p.add_argument("--hop-max", type=int, default=24)
    p.add_argument("--displacement", type=int, default=24)
    p.add_argument("--settle-steps", type=int, default=4)
    p.add_argument("--n-eval-rollouts", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-weight", type=float, default=1)
    p.add_argument("--input-weight", type=float, default=0.2)
    p.add_argument("--background-weight", type=float, default=0.05)
    p.add_argument("--stability-weight", type=float, default=0.1)
    p.add_argument("--hidden-weight", type=float, default=1e-4)
    p.add_argument("--blank-weight", type=float, default=1.0)
    args = p.parse_args()
    for key in [
        "steps",
        "eval_every",
        "inner_steps",
        "updates_per_hop",
        "hops",
        "hop",
        "settle_steps",
        "n_eval_rollouts",
    ]:
        if getattr(args, key) < 1:
            p.error(f"{key} must be positive")
    if args.batch_size < 4 or args.batch_size % 4:
        p.error("--batch-size must be a positive multiple of 4")
    if args.displacement < 1:
        p.error("--displacement must be positive")
    if not args.resume and not 0 <= args.stage1_steps <= args.steps:
        p.error("--stage1-steps must be between 0 and --steps")
    if args.resume:
        if args.resume == "last":
            runs = sorted(Path("runs").glob("and_*"))
            runs = [r for r in runs if (r / "checkpoints/nca_last.pt").exists()]
            if not runs:
                p.error("No resumable AND run")
            run = runs[-1]
        else:
            run = Path(args.resume)
        saved = json.loads((run / "config.json").read_text())
        # Keep the curriculum, loss, architecture and RNG consistent on resume.
        for key, value in saved["training"].items():
            if key not in [
                "steps",
                "device",
                "resume",
                "eval_every",
                "save_rollout_npz",
            ]:
                setattr(args, key, value)
        config = saved["nca"]
        args.position_gradient = 3 in config.get("read_only_dims", [])
        # Old checkpoints keep their original fixed-length curriculum.
        args.rollout_steps_min = saved["training"].get("rollout_steps_min")
        args.rollout_steps_max = saved["training"].get("rollout_steps_max")
        args.hop_min = saved["training"].get("hop_min")
        args.hop_max = saved["training"].get("hop_max")
        args.blank_examples = saved["training"].get("blank_examples", False)
        args.context_padding = saved["training"].get("context_padding", 0)
        args.update_head = saved["training"].get("update_head", "uniform")
        args.mask_sigma = saved["training"].get("mask_sigma", 8.0)
    else:
        if args.mask_sigma <= 0:
            p.error("--mask-sigma must be positive")
        if not 0 <= args.context_padding <= 16:
            p.error("--context-padding must be between 0 and 16")
        if not 1 <= args.hop_min <= args.hop_max <= min(31, args.displacement):
            p.error("hop bounds must satisfy 1 <= min <= max <= min(31, displacement)")
        if args.hops < 3:
            p.error("--hops must be at least 3 to include both movement axes")
        if not 1 <= args.rollout_steps_min <= args.rollout_steps_max:
            p.error("rollout step bounds must satisfy 1 <= min <= max")
        if args.rollout_steps_min < args.hops:
            p.error("--rollout-steps-min must allow at least one update per hop")
        run = Path("runs") / f"and_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
        config = dict(
            channels=16,
            hidden_channels=[128],
            fire_rate=0.5,
            alive_threshold=0,
            zero_initialization=True,
            kernel_size=7,
            num_perception_kernels=3,
            read_only_dims=[1, 2, 3] if args.position_gradient else [1, 2],  # Gradients and noise are writable
            padding_type="circular",  # Critical for infinite canvas
        )
        (run / "checkpoints").mkdir(parents=True)
        (run / "config.json").write_text(
            json.dumps(
                dict(
                    nca=config,
                    training=vars(args),
                    board=dict(inner_steps=args.inner_steps, wstride=args.hop),
                    gates=["AND"],
                ),
                indent=2,
            )
        )
    if not 0 <= args.stage1_steps <= args.steps:
        p.error("--steps must cover the saved stage-one curriculum")
    torch.manual_seed(args.seed)
    model = NeuralCA(**config).to(args.device)
    trainer = ANDTrainer(model, args)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, evals, start = [], [], 0
    if args.resume:
        ckpt = torch.load(
            run / "checkpoints/nca_last.pt",
            map_location=args.device,
            weights_only=False,
        )
        model.load_state_dict(ckpt["nca"])
        optim.load_state_dict(ckpt["optim"])
        start, history, evals = ckpt["step"], ckpt["history"], ckpt["evals"]
        if args.steps < start:
            p.error("--steps cannot be less than the checkpoint step")
        saved["training"]["steps"] = args.steps
        (run / "config.json").write_text(json.dumps(saved, indent=2))
        trainer.rng.setstate(ckpt["python_rng"])
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        if str(args.device).startswith("cuda") and ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng"]])
    fixed = trainer.batch(4)
    print(f"Run: {run} · stationary until {args.stage1_steps}, then displaced windows")
    for step in range(start, args.steps):
        result = trainer.rollout(
            trainer.batch(args.batch_size), step >= args.stage1_steps, trainer.rng
        )
        loss, terms, acc = trainer.loss(result)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()
        entry = dict(
            step=step + 1,
            stage=1 if step < args.stage1_steps else 2,
            loss=loss.item(),
            acc=acc.item(),
            rollout_steps=trainer.last_rollout_steps,
            window_origins=trainer.last_origins,
            **{k: v.item() for k, v in terms.items()},
        )
        history.append(entry)
        with (run / "log.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        if (step + 1) % args.eval_every == 0 or step + 1 in [
            args.stage1_steps,
            args.steps,
        ]:
            metrics, rollouts = trainer.evaluate(fixed)
            evals.append(dict(step=step + 1, **metrics))
            print(f"step {step+1}: {metrics}")
            torch.save(
                dict(
                    nca=model.state_dict(),
                    optim=optim.state_dict(),
                    step=step + 1,
                    history=history,
                    evals=evals,
                    python_rng=trainer.rng.getstate(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=(
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                ),
                run / "checkpoints/nca_last.pt",
            )
            torch.save(model.state_dict(), run / f"checkpoints/nca_{step+1:06d}.pt")
            if args.save_rollout_npz:
                np.savez_compressed(
                    run / "rollout_latest.npz",
                    **{
                        f"{name}_{key}": t.cpu().numpy()
                        for name, result in rollouts.items()
                        for key, t in zip(
                            ["initial", "before", "state", "target", "mask"], result
                        )
                    },
                )
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(
                [e["step"] for e in history],
                [e["loss"] for e in history],
                label="train",
            )
            for name in ["stationary", "displaced"]:
                axes[0].plot(
                    [e["step"] for e in evals],
                    [e[name + "_loss"] for e in evals],
                    label=name,
                )
                axes[1].plot(
                    [e["step"] for e in evals],
                    [e[name + "_acc"] for e in evals],
                    label=name,
                )
            axes[0].set_yscale("log")
            axes[0].set_title("Compound loss")
            axes[1].set_title("AND truth-table accuracy")
            axes[1].set_ylim(0, 1.05)
            for ax in axes:
                ax.legend()
                ax.set_xlabel("Training step")
                ax.axvline(args.stage1_steps, color="gray", linestyle="--")
            fig.tight_layout()
            fig.savefig(run / "curves.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
