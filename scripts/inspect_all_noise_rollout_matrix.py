import re
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ncpu.utils import save_rollout_png, save_grid_image, add_gaussian_noise, save_rollout_gif, save_matrix, bit_accuracy
from ncpu.config import TINY_AND_FARAWAY_TRAINING_CONFIG
from ncpu.normalizers import normalize_neg1_to_1
from ncpu.dataset import MultiGateDataset
from ncpu.nca import NeuralCA
from pathlib import Path


def load_config(run_dir):
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}

def load_log(run_dir):
    cfg_path = run_dir / "log.jsonl"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return [json.loads(line) for line in f]
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

def get_rollout(run_dir,steps):
    ckpt = find_checkpoint(run_dir)
    cfg = load_config(run_dir)
    noise=cfg.get("GAUSSIAN_NOISE", 0.2)
    noise_fr=cfg.get("fire_rate", 0.2)

    if ckpt is None:
        return None
    checkpoint = torch.load(run_dir / "best_rollout.pt")
    rollout = checkpoint["rollout"][:,:steps,...] # (B, T, C, H, W)
    target = checkpoint["out"]
    return rollout, target, noise_fr, noise

def analyze(run_dir, batch_size=8, steps=64):
    assert run_dir.exists(), f"{run_dir} not found"

    out_dir = run_dir / "all_noise_inspection"
    out_dir.mkdir(exist_ok=True)

    ckpt = find_checkpoint(run_dir)

    checkpoint = torch.load(run_dir / "best_rollout.pt")
    rollout = checkpoint["rollout"][:,:steps,...] # (B, T, C, H, W)
    target = checkpoint["out"]

    # Save rollback plot
    target = target.to(rollout.device)
    expanded_target = target.unsqueeze(1).unsqueeze(2).expand(rollout.shape[0], rollout.shape[1], 1, rollout.shape[3], rollout.shape[4]).contiguous()
    to_save = torch.cat([expanded_target, rollout], dim=2) 

    for g in [0,1,2,3,4,5,6,7]:
        save_rollout_png(
            out_dir / f"rollout_{ckpt.stem}_gate_{g}.png",
            to_save[g].cpu(),  # first sample: (T, C, H, W)
            n_snapshots=2,
            max_channels=16,
            channels=[0,1]
        )

    save_rollout_gif(
        rollout,
        target,
        batch_size,
        16,
        out_dir / f"rollout_{ckpt.stem}.gif",
    )

    # copy latest for convenience
    (out_dir / "rollout_latest.png").write_bytes((out_dir / f"rollout_{ckpt.stem}_gate_{0}.png").read_bytes())

    # log plots
    plot_log(run_dir, out_dir)

    print(f"inspected run {run_dir} -> {out_dir}")


def analyze_multiple(base_dir=Path.home() / "ncpu" / "runs", batch_size=8, steps=128):
    pattern = f"*_coded_gates_noise*"

    out_dir = base_dir / "all_noise_insepction"
    out_dir.mkdir(exist_ok=True)
    assert out_dir.exists(), f"{out_dir} not found"
    out_dir = out_dir / "all_gate_noises_combined"
    out_dir.mkdir(exist_ok=True)

    run_dirs_raw = sorted(Path(base_dir).glob(pattern))

    # Keep only dirs matching: _coded_gates_noise{number}_fr{number}
    noise_fr_re = re.compile(r"_coded_gates_noise(\d+)_fr(\d+)")
    run_dirs = [p for p in run_dirs_raw if noise_fr_re.search(p.name)]

    if not run_dirs:
        raise FileNotFoundError(f"No runs found for pattern {pattern} in {base_dir}")

    labels_y = ["fr=0.2", "fr=0.4", "fr=0.6", "fr=0.8", "fr=1.0"]
    labels_x = ["std=0.2", "std=0.4", "std=0.6", "std=0.8", "std=1.0"]

    target_frames = torch.empty((5, 5, 64, 64))
    for gate in [0,1,2,3]:
        for step in [0, 12, 24, 31]:
            final_frames = torch.empty((5, 5, 64, 64))
            convergence_grid = torch.empty((5, 5))
            validity_grid = torch.empty((5, 5))
            bit_accuracy_grid = torch.empty((5, 5))
            for n,run_dir in enumerate(run_dirs):
                if not run_dir.is_dir():
                    continue
                log = load_log(run_dir)
                rollout, target, noise_fr, noise = get_rollout(run_dir, steps)
                x = int(noise_fr*10/2 - 1) 
                y = int(noise*10/2 - 1) 

                convergence_step = 0.0
                chunk_size = 2500
                step_size = int(chunk_size/100)
                for n in range(0, len(log), step_size):
                    chunk = [l["num_valid_bits"] for l in log[n:n+chunk_size]] 
                    print("Checking convergence for", run_dir.name, "mean valid bits:", np.mean(chunk))
                    if np.mean(chunk) > 0.95:
                        convergence_step = np.abs(1 - n/len(log))
                        break
    
                final_frames[x,y] = rollout[gate,step,0,:,:]
                validity_grid[x,y] = np.mean([l["num_valid_bits"] for l in log[-2500:]])
                convergence_grid[x,y] = convergence_step
                target_frames[x,y] = target[gate,:,:]
                bit_accuracy_grid[x,y] = bit_accuracy(rollout[gate,step,0,:,:], target[gate,:,:])

            save_matrix(
                out_dir / f"total_bit_validity.png",
                validity_grid,
                labels_x = labels_x,
                labels_y = labels_y,
                title=f"Bit Validity Across 2000 rollouts"
            )

            save_matrix(
                out_dir / f"convergence_grid.png",
                convergence_grid,
                labels_x = labels_x,
                labels_y = labels_y,
                title=f"Convergence step grid (> 0.95)"
            )

            save_matrix(
                out_dir / f"bit_accuracy_step{step}_gate{gate}_.png",
                bit_accuracy_grid,
                labels_x = labels_x,
                labels_y = labels_y,
                title=f"Bit Accuracy — Step {step}, Gate {gate}"
            )

            save_rollout_png(
                out_dir / f"frame_matrix_noise_fr_step_{step}_gate_{gate}.png",
                final_frames.cpu(),  # first sample: (T, C, H, W)
                n_snapshots=4,
                max_channels=4,
                labels_x = labels_x,
                labels_y = labels_y,
            )

        save_rollout_png(
            out_dir / f"frame_matrix_noise_fr_target_gate_{gate}.png",
            target_frames.cpu(),  # first sample: (T, C, H, W)
            n_snapshots=4,
            max_channels=4,
            labels_x = labels_x,
            labels_y = labels_y,
        )

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
