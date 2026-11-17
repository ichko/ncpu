#!/usr/bin/env python3
"""Plot all state channels from a saved AND training rollout."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("demo/training_layers.png"))
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    with np.load(args.run / "rollout_latest.npz") as saved:
        initial = saved["displaced_initial"][3].copy()
        final = saved["displaced_state"][3].copy()
        target = saved["displaced_target"][3].copy()
    # The numbered checkpoint identifies the most recent periodic snapshot.
    checkpoints = list((args.run / "checkpoints").glob("nca_[0-9]*.pt"))
    step = max(int(p.stem.split("_")[-1]) for p in checkpoints)
    stage = 1 if step <= config["training"]["stage1_steps"] else 2
    hidden_scale = max(float(np.abs(final[4:]).max()), 1e-6)
    names = [
        "Working / output",
        "Read-only input",
        "Read-only AND selector",
        "Read-only position gradient",
    ]
    fig, axes = plt.subplots(8, 4, figsize=(15, 25), layout="constrained")
    for phase, state in enumerate([initial, final]):
        for ch in range(16):
            ax = axes[phase * 4 + ch // 4, ch % 4]
            scale = 1 if ch < 4 else hidden_scale
            image = ax.imshow(
                state[ch],
                cmap="coolwarm",
                vmin=-scale,
                vmax=scale,
                interpolation="nearest",
            )
            role = names[ch] if ch < 4 else f"Hidden state {ch}"
            ax.set_title(
                f"{'Initial' if phase == 0 else 'Final'} · ch {ch}: {role}\n"
                f"range [{state[ch].min():.3f}, {state[ch].max():.3f}]",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"AND training · saved step {step} / stage {stage} · input 1 AND 1 = 1\n"
        "Displaced-window rollout · top 16: initial state · bottom 16: final state\n"
        "Channels 0–3 use [−1, +1]; hidden channels share a symmetric scale",
        fontsize=15,
    )
    fig.savefig(args.out, dpi=140)
    plt.close(fig)

    detail = args.out.with_name("training_gradient.png")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), layout="constrained")
    delta = final[3] - initial[3]
    arrays = [initial[3], final[3], delta, target]
    titles = [
        "Position channel 3: initial",
        "Position channel 3: final",
        f"Final − initial: max |Δ| = {np.abs(delta).max():.2g}",
        "Expected AND output (1 AND 1)",
    ]
    for ax, array, title in zip(axes, arrays, titles):
        image = ax.imshow(
            array,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
            origin="upper",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Board x")
        ax.set_ylabel("Board y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"Read-only, board-anchored diagonal ramp · actual training snapshot at step {step}\n"
        "g(x,y) = (x + y)/31 − 1 within the 32×32 tile; surrounding padding is zero",
        fontsize=13,
    )
    fig.savefig(detail, dpi=160)
    plt.close(fig)
    print(f"Step {step}, stage {stage}: {args.out} and {detail}")
    print(f"Position-channel maximum change: {np.abs(delta).max()}")


if __name__ == "__main__":
    main()
