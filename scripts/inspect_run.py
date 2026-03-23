import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ncpu.config import TINY_AND_FARAWAY_TRAINING_CONFIG
from ncpu.normalizers import normalize_neg1_to_1
from ncpu.dataset import MultiGateDataset
from ncpu.nca import NeuralCA
from ncpu.utils import save_rollout_png, save_grid_image, add_gaussian_noise


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
    print(pts)
    return pts[-4] if pts else None


def build_nca(cfg):
    print(
        "channels: " , cfg.get("NCA_CHANNELS", 16),
        "hidden_channels: " , cfg.get("HIDDEN_CHANNELS", [128]),
        "fire_rate: " , cfg.get("fire_rate", 0.99),
        "alive_threshold: " , cfg.get("alive_threshold", 0.1),
        "zero_initialization: " , cfg.get("zero_initialization", False),
        "kernel_size: " , cfg.get("KERNEL_SIZE", 7),
        "padding_type: " , cfg.get("padding_type", "constant"),
        "read_only_dims: " , cfg.get("read_only_dims", [-4, -3, -2, -1]),
        "gaussian_noise: " , cfg.get("GAUSSIAN_NOISE", 0.2),
        "gaussian_noise_fire_rate: " , cfg.get("gaussian_noise_fire_rate", cfg.get("fire_rate", 0.2)),

    )

    return NeuralCA(
        channels=cfg.get("NCA_CHANNELS", 16),
        hidden_channels=cfg.get("HIDDEN_CHANNELS", [128]),
        fire_rate=cfg.get("fire_rate", 0.99),
        alive_threshold=cfg.get("alive_threshold", 0.1),
        zero_initialization=cfg.get("zero_initialization", False),
        kernel_size=cfg.get("KERNEL_SIZE", 7),
        padding_type=cfg.get("padding_type", "constant"),
        read_only_dims=cfg.get("read_only_dims", [-4, -3, -2, -1]),
        gaussian_noise=cfg.get("GAUSSIAN_NOISE", 0.2),
        gaussian_noise_fire_rate=cfg.get("gaussian_noise_fire_rate", cfg.get("fire_rate", 0.2)),
    )

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path, help="path to experiment folder under runs/")
    p.add_argument("--steps", type=int, default=64, help="rollout steps")
    p.add_argument("--batch", type=int, default=8, help="batch size for evaluation")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    print(run_dir)
    assert run_dir.exists(), f"{run_dir} not found"

    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    cfg = load_config(run_dir)
    nca = build_nca(cfg)
    ckpt = find_checkpoint(run_dir)
    if ckpt is None:
        raise RuntimeError(f"No checkpoint in {run_dir}/checkpoints")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        nca.load_state_dict(state["model_state_dict"])
    else:
        nca.load_state_dict(state)

    nca.eval()

    dataset = MultiGateDataset(TINY_AND_FARAWAY_TRAINING_CONFIG, nca_channels=nca.channels)
    dl = dataset.get_dataloader(batch_size=args.batch)
    inp, target = next(iter(dl))
    inp = inp

    first_state = normalize_neg1_to_1(inp)
    target = normalize_neg1_to_1(target)
    first_state = add_gaussian_noise(first_state, 0, 0.2)

    print(inp.shape)

    # first_state = implant_input(inp, input_implant_type="first", nca_channels=nca.channels, H=inp.shape[-2], W=inp.shape[-1])
    with torch.no_grad():
        rollout = nca.forward(first_state, steps=args.steps)

    # Save rollback plot
    target = target.to(rollout.device)
    expanded_target = target.unsqueeze(1).unsqueeze(2).expand(rollout.shape[0], rollout.shape[1], 1, rollout.shape[3], rollout.shape[4]).contiguous()
    to_save = torch.cat([expanded_target, rollout], dim=2) 

    for g in [0,1,2,3,4,5,6,7]:
        save_rollout_png(
            out_dir / f"rollout_{ckpt.stem}_gate_{g}.png",
            to_save[g].cpu(),  # first sample: (T, C, H, W)
            n_snapshots=12,
            max_channels=min(nca.channels, 16),
            # channels=[0,1,2,4,5,-4]
        )

    # copy latest for convenience
    (out_dir / "rollout_latest.png").write_bytes((out_dir / f"rollout_{ckpt.stem}_gate_{0}.png").read_bytes())

    # log plots
    plot_log(run_dir, out_dir)

    print(f"inspected run {run_dir} -> {out_dir}")


if __name__ == "__main__":
    main()
