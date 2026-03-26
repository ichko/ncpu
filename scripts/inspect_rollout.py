import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ncpu.config import TINY_AND_FARAWAY_TRAINING_CONFIG
from ncpu.normalizers import normalize_neg1_to_1
from ncpu.dataset import MultiGateDataset
from ncpu.nca import NeuralCA
from ncpu.utils import save_rollout_png, save_grid_image, add_gaussian_noise, save_rollout_gif, save_cross_section_y, save_cross_section_x


def load_config(run_dir):
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}


def find_checkpoint(run_dir):
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    pts = sorted(ckpt_dir.glob("nca_*.pt"))
    return pts[-1] if pts else None


def implant_input(inp, input_implant_type : str, nca_channels: int, H : int , W : int):
    bs = inp.shape[0]
    if input_implant_type == "all":
        first_state = (
            inp.unsqueeze(1)
            .expand(bs, nca_channels, H, W)
            .clone()
        )
        first_state = first_state.to(inp.device)
    elif input_implant_type == "disabled":
        return inp.to(inp.device)
    else:
        first_state = torch.zeros(bs, nca_channels, H, W)
        first_state[:, 0] = inp
    return first_state


def plot_log(run_dir, out_dir):
    log_path = run_dir / "log.jsonl"
    if not log_path.exists():
        return
    steps, losses, bits = [], [], []
    with open(log_path, "r") as f:
        for l in f:
            try:
                obj = json.loads(l)
            except json.JSONDecodeError:
                continue
            steps.append(obj.get("step"))
            losses.append(obj.get("loss"))
            bits.append(obj.get("num_valid_bits"))
    if steps:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(steps, losses, marker=".", markersize=2, label="loss")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "log_loss.png", dpi=120)
        plt.close(fig)

    if steps and any(v is not None for v in bits):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(steps, bits, marker=".", markersize=2, color="orange", label="valid bits")
        ax.set_xlabel("step")
        ax.set_ylabel("num_valid_bits")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "log_bits.png", dpi=120)
        plt.close(fig)


def analyze(run_dir, batch_size=8, steps=64):
    assert run_dir.exists(), f"{run_dir} not found"

    out_dir = run_dir / "best_rollout_analysis_single"
    out_dir.mkdir(exist_ok=True)

    ckpt = find_checkpoint(run_dir)

    checkpoint = torch.load(run_dir / "best_rollout.pt")
    rollout = checkpoint["rollout"][:,:steps,...] # (B, T, C, H, W)
    target = checkpoint["out"]

    # Save rollback plot
    target = target.to(rollout.device)
    expanded_target = target.unsqueeze(1).unsqueeze(2).expand(rollout.shape[0], rollout.shape[1], 1, rollout.shape[3], rollout.shape[4]).contiguous()
    to_save = torch.cat([expanded_target, rollout], dim=2) 

    # for g in [0,1,2,3,4,5,6,7]:
    #     save_rollout_png(
    #         out_dir / f"rollout_{ckpt.stem}_gate_{g}.png",
    #         to_save[g].cpu(),  # first sample: (T, C, H, W)
    #         n_snapshots=2,
    #         max_channels=16,
    #         channels=[0,1]
    #     )

    save_cross_section_y(rollout,
        path= out_dir / f"y_section_{ckpt.stem}.png",
        cross_section=-16)
    save_cross_section_x(rollout,
        path= out_dir / f"x_section_{ckpt.stem}.png",
        cross_section=32)

    # save_rollout_gif(
    #     rollout,
    #     target,
    #     batch_size,
    #     16,
    #     out_dir / f"rollout_{ckpt.stem}.gif",
    # )

    # copy latest for convenience
    (out_dir / "rollout_latest.png").write_bytes((out_dir / f"rollout_{ckpt.stem}_gate_{0}.png").read_bytes())

    # log plots
    plot_log(run_dir, out_dir)

    print(f"inspected run {run_dir} -> {out_dir}")


def analyze_multiple(base_dir=Path.home() / "ncpu" / "runs", batch_size=8, steps=128):
    pattern = f"*_coded_gates_noise*"
    run_dirs = sorted(Path(base_dir).glob(pattern))
    if not run_dirs:
        raise FileNotFoundError(f"No runs found for pattern {pattern} in {base_dir}")

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        print(f"analyzing {run_dir}")
        analyze(run_dir, batch_size=batch_size, steps=steps)

    # analyze(run_dir, batch_size=batch_size, steps=steps)

def main():
    p = argparse.ArgumentParser()
    # p.add_argument("run_dir", type=Path, help="path to experiment folder under runs/")
    p.add_argument("--steps", type=int, default=64, help="rollout steps")
    p.add_argument("--batch", type=int, default=8, help="batch size for evaluation")
    args = p.parse_args()

    # run_dir = args.run_dir.resolve()
    batch_size = args.batch
    steps = args.steps

    analyze_multiple(batch_size=batch_size, steps=steps)
    # analyze(run_dir, batch_size=batch_size, steps=steps)

if __name__ == "__main__":
    main()
