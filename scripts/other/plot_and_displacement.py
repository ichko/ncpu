#!/usr/bin/env python3
"""Replay a fixed displaced AND evaluation and visualize each window hop."""

import argparse
import importlib.util
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("demo/training_displacement.png"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "train_and", root / "scripts/train/train_and.py"
    )
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    config = json.loads((args.run / "config.json").read_text())
    options = argparse.Namespace(**config["training"])
    options.device = "cpu"
    model = training.NeuralCA(**config["nca"])
    checkpoint = max(
        (args.run / "checkpoints").glob("nca_[0-9]*.pt"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    step = int(checkpoint.stem.split("_")[-1])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    trainer = training.ANDTrainer(model, options)
    d = options.displacement
    state = F.pad(trainer.batch(4)[0], (d, d, d, d))
    initial_gradient = state[:, 3].clone()
    rng = random.Random(10003)
    minimum = getattr(options, "rollout_steps_min", None)
    budget = (
        rng.randint(minimum, options.rollout_steps_max) if minimum is not None else None
    )
    origins = trainer.displacement_origins(rng)
    counts = (
        [
            budget // len(origins) + (i < budget % len(origins))
            for i in range(len(origins))
        ]
        if budget is not None
        else [options.updates_per_hop] * len(origins)
    )
    torch.manual_seed(options.seed + 10003)
    fig, axes = plt.subplots(
        3,
        len(origins),
        figsize=(3.2 * len(origins), 10),
        layout="constrained",
        squeeze=False,
    )
    with torch.no_grad():
        for i, ((y, x), updates) in enumerate(zip(origins, counts)):
            state = trainer.advance_window(state, y, x, updates)
            crop = state[:, :, y : y + 32, x : x + 32]
            panels = [state[3, 0], crop[3, 0], crop[3, 3]]
            for row, panel in enumerate(panels):
                ax = axes[row, i]
                ax.imshow(
                    panel.numpy(),
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            axes[0, i].add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    32,
                    32,
                    fill=False,
                    edgecolor="#e6ae00",
                    linewidth=2,
                )
            )
            axes[0, i].set_title(
                f"Hop {i+1}: origin (x={x}, y={y})\n{sum(counts[:i+1])} cumulative updates"
            )
            if i:
                previous_y, previous_x = origins[i - 1]
                axes[2, i].set_xlabel(f"Movement: Δx={x-previous_x}, Δy={y-previous_y}")
            else:
                axes[2, i].set_xlabel("Start at the tile origin")
    for row, text in enumerate(
        [
            f"Persistent {32+2*d}×{32+2*d} board\nafter window update",
            "32×32 working crop\nseen by the NCA",
            "32×32 read-only gradient\nseen by the NCA",
        ]
    ):
        axes[row, 0].set_ylabel(text)
    stage = 1 if step <= options.stage1_steps else 2
    fig.suptitle(
        f"Displaced AND evaluation replay · checkpoint {step} (training stage {stage}) · inputs 1, 1\n"
        "Yellow box = active window. Only its crop is updated; state outside it is preserved.\n"
        "The board-anchored gradient is cropped, never re-centered. Red = +1; blue = −1; gray = 0.",
        fontsize=14,
    )
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    assert torch.equal(initial_gradient, state[:, 3])
    print(
        f"Saved {args.out}; checkpoint={step}; origins (y,x)={origins}; gradient unchanged"
    )


if __name__ == "__main__":
    main()
