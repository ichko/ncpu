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

from pandas.plotting import parallel_coordinates

def load_log(run_dir):
    print(run_dir)
    cfg_path = run_dir / "log.jsonl"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return [json.loads(line) for line in f]
    return {}

def analyze_multiple(base_dir=Path.home() / "ncpu" / "runs", batch_size=8, steps=128):
    pattern = f"*param_grid_search"
    out_dir = base_dir / "param_grid_search_analysis"
    out_dir.mkdir(exist_ok=True)
    assert out_dir.exists(), f"{out_dir} not found"
    out_dir.mkdir(exist_ok=True)

    run_dirs_raw = sorted(Path(base_dir).glob(pattern))

    for run_dir in run_dirs_raw:
        if run_dir.is_dir():
            analyze_single(run_dir, out_dir)

def analyze_single(run_dir, out_dir):
    range_fire_rate = [0.99, 0.75, 0.5, 0.25, 0.1]
    range_alive_threshold = [0.0, 0.1, 0.3, 0.4, 0.5]
    range_zero_initialization = [True, False]
    range_kernel_size = [3, 5, 7, 9]

    n_rows = len(range_fire_rate) * len(range_alive_threshold) * len(range_zero_initialization) * len(range_kernel_size) 
    
    convergence_grid = torch.zeros((5,n_rows))
    n_row = 0
    for fr in range_fire_rate:
        for am in range_alive_threshold:
            for zi in range_zero_initialization:
                for ks in range_kernel_size:
                    target_dir = run_dir / f"param_{fr}_{am}_{zi}_{ks}"

                    log = load_log(target_dir)
                    convergence_step = 0.0
                    chunk_size = 2500
                    step_size = int(chunk_size/100)
                    for n in range(0, len(log), step_size):
                        chunk = [l["num_valid_bits"] for l in log[n:n+chunk_size]] 
                        print("Checking convergence for", target_dir.name, "mean valid bits:", np.mean(chunk))
                        if np.mean(chunk) > 0.95:
                            convergence_step = np.abs(1 - n/len(log))
                            break

                    convergence_grid[0][n_row] = fr
                    convergence_grid[1][n_row] = am
                    convergence_grid[2][n_row] = zi
                    convergence_grid[3][n_row] = ks
                    convergence_grid[4][n_row] = convergence_step
                    n_row+=1

    data = convergence_grid.T.numpy()  # shape (n_rows, 5)
    print(data.shape)
    fig, ax = plt.subplots(figsize=(8, len(data) * 0.4))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=7)

    ax.set_xticks(range(5))
    ax.set_xticklabels(["fr", "am", "zi", "ks", "convergence"])
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([f"{i}" for i in range(len(data))], fontsize=7)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / "convergence_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

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
