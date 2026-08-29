import re
import json
import csv
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ncpu.utils import save_rollout_png, save_grid_image, add_gaussian_noise, save_rollout_gif, save_matrix, bit_accuracy
from ncpu.config import TINY_AND_FARAWAY_TRAINING_CONFIG
from ncpu.normalizers import normalize_neg1_to_1
from ncpu.dataset import MultiGateDataset
from ncpu.nca import NeuralCA
from pathlib import Path
import matplotlib.cm as cm


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

                    convergence_grid[0][n_row] = am
                    convergence_grid[1][n_row] = fr
                    convergence_grid[2][n_row] = ks
                    convergence_grid[3][n_row] = zi
                    convergence_grid[4][n_row] = convergence_step
                    n_row+=1
    data = convergence_grid.T.numpy()  # shape (n_rows, 5)
    data_sorted = data[np.argsort(data[:, 4])]

    csv_path = out_dir / f"{run_dir.name}_convergence_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_name",
            "alive_threshold",
            "fire_rate",
            "kernel_size",
            "zero_initialization",
            "convergence_step",
            "rank_by_convergence_asc",
        ])
        for rank, row in enumerate(data_sorted, start=1):
            writer.writerow([
                run_dir.name,
                float(row[0]),
                float(row[1]),
                int(row[2]),
                int(row[3]),
                float(row[4]),
                rank,
            ])
    print(f"Saved CSV: {csv_path}")

    # ── half-radar chart ─────────────────────────────────────────
    axes_labels = [
        "alive masking",
        "fire rate",
        "kernel size",
        "weights initialization",
    ]
    N = len(axes_labels)
    angles_360 = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_180 = np.linspace(0, np.pi, N).tolist()

    param_ranges_strong = [
        (min(range_alive_threshold), max(range_alive_threshold)),
        (min(range_fire_rate),       max(range_fire_rate)),
        (max(range_kernel_size),     min(range_kernel_size)),
        (0, 1),
    ]

    axis_tick_values_strong = [
        np.arange(min(range_alive_threshold), max(range_alive_threshold) + 1e-9, 0.1),
        np.array(sorted(range_fire_rate)),
        np.array(sorted(range_kernel_size, reverse=True)),
        np.array([0.0, 1.0]),
    ]

    param_ranges_weak = [
        (min(range_alive_threshold), max(range_alive_threshold)),
        (min(range_fire_rate),       max(range_fire_rate)),
        (0, max(range_kernel_size)),
        (0, 1),
    ]

    axis_tick_values_weak = [
        np.arange(min(range_alive_threshold), max(range_alive_threshold) + 1e-9, 0.1),
        np.array(sorted(range_fire_rate)),
        np.array([0, 3, 5, 7, 9]),
        np.array([0.0, 1.0]),
    ]

    def normalize(val, lo, hi):
        return (val - lo) / (hi - lo) if hi != lo else 0.0

    cmap = cm.get_cmap("Blues")
    center_spread_start = 0.06
    center_spread_step = 0.02

    def render_radar(
        plot_data,
        title,
        output_path,
        param_ranges,
        axis_tick_values,
        angles,
        close_loop,
        draw_borders,
    ):
        fig, ax = plt.subplots(figsize=(9, 5), subplot_kw=dict(polar=True))
        is_half_radar = max(angles) <= np.pi + 1e-9
        if is_half_radar:
            # Keep 180-degree radar horizontal (left-right) instead of vertical.
            ax.set_theta_offset(0)
            ax.set_theta_direction(1)
            ax.set_thetamin(0)
            ax.set_thetamax(180)
        else:
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

        all_points = []

        for row in plot_data:
            conv = row[4]
            values = [normalize(row[i], *param_ranges[i]) for i in range(4)]
            path_angles = angles + [angles[0]] if close_loop else angles
            path_values = values + [values[0]] if close_loop else values
            if draw_borders:
                ax.plot(path_angles, path_values, color=cmap(conv), linewidth=conv, alpha=conv)
                ax.fill(path_angles, path_values, color=cmap(conv), alpha=conv)
            else:
                ax.fill(path_angles, path_values, color=cmap(conv), alpha=conv, edgecolor='none', linewidth=0)
            all_points.extend(zip(angles, values))

        if all_points:
            point_angles, point_values = zip(*all_points)
            ax.scatter(point_angles, point_values, color='red', s=12, zorder=100, alpha=0.9)

        for axis_idx, (angle, ticks) in enumerate(zip(angles, axis_tick_values)):
            lo, hi = param_ranges[axis_idx]
            for actual_val in ticks:
                norm_pos = normalize(actual_val, lo, hi)
                if axis_idx == 0:
                    label = f"{actual_val:.1f}"
                elif axis_idx == 3:
                    continue
                else:
                    label = f"{actual_val:.2g}"
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                radial_offset = 0.03 if sin_a >= 0 else -0.03
                text_r = np.clip(norm_pos + radial_offset, 0.03, 0.97)
                if norm_pos < center_spread_start:
                    text_r = center_spread_start + axis_idx * center_spread_step
                ha = 'left' if cos_a > 0.2 else ('right' if cos_a < -0.2 else 'center')
                va = 'bottom' if sin_a > 0.2 else ('top' if sin_a < -0.2 else 'center')
                ax.text(angle, text_r, label,
                        ha=ha, va=va, fontsize=9, color='dimgray', alpha=0.95, clip_on=False)

        ax.set_yticklabels([])
        ax.set_thetagrids(np.degrees(angles), labels=axes_labels, fontsize=12)
        ax.set_ylim(0, 1)
        ax.tick_params(pad=8)
        ax.grid(color="gray", alpha=0.2, linewidth=0.5)
        ax.spines["polar"].set_visible(False)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
        cbar.ax.set_ylabel("convergence step (higher = converged earlier)", fontsize=9)
        plt.title(title, pad=20, fontsize=13)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def render_parallel_coordinates(plot_data, title, output_path, param_ranges, axis_tick_values):
        fig, ax = plt.subplots(figsize=(10, 5))
        x_positions = np.arange(len(axes_labels))

        for x in x_positions:
            ax.axvline(x, color='lightgray', linewidth=0.8, alpha=0.8)

        for row in plot_data:
            conv = row[4]
            values = [normalize(row[i], *param_ranges[i]) for i in range(4)]
            ax.plot(
                x_positions,
                values,
                color=cmap(conv),
                alpha=max(0.08, conv),
                linewidth=max(0.6, 1.6 * conv),
            )

        for axis_idx, (x, ticks) in enumerate(zip(x_positions, axis_tick_values)):
            lo, hi = param_ranges[axis_idx]
            for actual_val in ticks:
                y = normalize(actual_val, lo, hi)
                if axis_idx == 0:
                    label = f"{actual_val:.1f}"
                elif axis_idx == 3:
                    continue
                else:
                    label = f"{actual_val:.2g}"
                ax.text(x + 0.03, y, label, fontsize=8, color='dimgray', va='center', ha='left')

        ax.set_xlim(-0.2, len(axes_labels) - 0.8)
        ax.set_ylim(0, 1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(axes_labels, fontsize=12)
        ax.set_yticks([])
        ax.grid(axis='y', color='gray', alpha=0.2, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.set_ylabel("convergence step (higher = converged earlier)", fontsize=9)
        plt.title(title, fontsize=13)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def render_nested_heatmaps(plot_data, title, output_path):
        max_am_for_heatmap = 0.3
        am_values = np.array(sorted([am for am in range_alive_threshold if am <= max_am_for_heatmap]))
        zi_values = np.array([1, 0])
        fr_values = np.array(sorted(range_fire_rate, reverse=True))
        ks_values = np.array(sorted(range_kernel_size))
        split_base_stem = output_path.stem
        heatmap_cmap = ListedColormap(cm.get_cmap("Blues")(np.linspace(0.2, 1.0, 256)))

        def heatmap_text_color(cell_val):
            return "black" if cell_val < 0.55 else "white"

        def save_split_heatmap(mat, am, zi):
            split_fig, split_ax = plt.subplots(figsize=(3.4, 2.8))
            split_ax.imshow(mat, cmap=heatmap_cmap, vmin=0.0, vmax=1.0, aspect='auto', origin='upper')

            for fr_idx in range(len(fr_values)):
                for ks_idx in range(len(ks_values)):
                    cell_val = mat[fr_idx, ks_idx]
                    if np.isnan(cell_val):
                        continue
                    split_ax.text(
                        ks_idx,
                        fr_idx,
                        f"{cell_val:.2f}",
                        ha='center',
                        va='center',
                        fontsize=7,
                        color=heatmap_text_color(cell_val),
                        alpha=0.95,
                    )

            split_ax.set_xticks(np.arange(len(ks_values)))
            split_ax.set_xticklabels([str(int(v)) for v in ks_values], fontsize=8)
            split_ax.set_xlabel("kernel size", fontsize=9)
            split_ax.set_yticks(np.arange(len(fr_values)))
            split_ax.set_yticklabels([f"{v:.2g}" for v in fr_values], fontsize=8)
            split_ax.set_ylabel("fire rate", fontsize=9)

            am_token = f"{am:.1f}".replace('.', 'p')
            zi_token = "random_weights_initialization" if int(zi) == 0 else "zero_weights_initialization"
            split_path = output_path.parent / f"{split_base_stem}_am_{am_token}_{zi_token}"
            split_fig.savefig(split_path.with_suffix(".svg"), bbox_inches="tight")
            split_fig.savefig(split_path.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(split_fig)

        fig, axes = plt.subplots(
            len(zi_values),
            len(am_values),
            figsize=(3.6 * len(am_values) + 2.2, 2.2 * len(zi_values) + 1.8),
            sharex=False,
            sharey=False,
        )

        if len(zi_values) == 1 and len(am_values) == 1:
            axes = np.array([[axes]])
        elif len(zi_values) == 1:
            axes = np.array([axes])
        elif len(am_values) == 1:
            axes = np.array([[ax] for ax in axes])

        vmin, vmax = 0.0, 1.0
        im = None
        for i, zi in enumerate(zi_values):
            for j, am in enumerate(am_values):
                ax = axes[i, j]
                mat = np.full((len(fr_values), len(ks_values)), np.nan, dtype=float)

                for row in plot_data:
                    row_am, row_fr, row_ks, row_zi, row_conv = row
                    if np.isclose(row_am, am) and int(row_zi) == int(zi):
                        fr_idx = np.where(np.isclose(fr_values, row_fr))[0]
                        ks_idx = np.where(np.isclose(ks_values, row_ks))[0]
                        if len(fr_idx) and len(ks_idx):
                            mat[fr_idx[0], ks_idx[0]] = row_conv

                save_split_heatmap(mat, am, zi)

                im = ax.imshow(mat, cmap=heatmap_cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='upper')

                for fr_idx in range(len(fr_values)):
                    for ks_idx in range(len(ks_values)):
                        cell_val = mat[fr_idx, ks_idx]
                        if np.isnan(cell_val):
                            continue
                        ax.text(
                            ks_idx,
                            fr_idx,
                            f"{cell_val:.2f}",
                            ha='center',
                            va='center',
                            fontsize=7,
                            color=heatmap_text_color(cell_val),
                            alpha=0.95,
                        )

                if i == len(zi_values) - 1:
                    ax.set_xticks(np.arange(len(ks_values)))
                    ax.set_xticklabels([str(int(v)) for v in ks_values], fontsize=12)
                    ax.set_xlabel("kernel size", fontsize=12)
                else:
                    ax.set_xticks([])
                    ax.set_xlabel("")

                if j == 0:
                    ax.set_yticks(np.arange(len(fr_values)))
                    ax.set_yticklabels([f"{v:.2g}" for v in fr_values], fontsize=12)
                    ax.set_ylabel("fire rate", fontsize=12)
                else:
                    ax.set_yticks([])
                    ax.set_ylabel("")

        fig.subplots_adjust(left=0.15, right=0.87, top=0.90, bottom=0.08, wspace=0.0, hspace=0.0)

        for j, am in enumerate(am_values):
            col_header = f"alive masking {am}"
            axes[0, j].set_title(col_header, fontsize=12, pad=14)

        for i, zi in enumerate(zi_values):
            row_header = "zero weights" if int(zi) == 1 else "random weights"
            row_box = axes[i, 0].get_position()
            fig.text(
                row_box.x0 - 0.075,
                row_box.y0 + row_box.height / 2,
                row_header,
                ha='right',
                va='center',
                fontsize=12,
                rotation=90,
            )

        cbar_ax = fig.add_axes([0.89, 0.12, 0.02, 0.76])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel("convergence step (higher = converged earlier)", fontsize=12)
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    # Strongest-on-top: weakest are drawn first, strongest last.
    render_radar(
        data_sorted,
        "Convergence radar (strongest on top)",
        out_dir / "convergence_radar.png",
        param_ranges_strong,
        axis_tick_values_strong,
        angles_360,
        True,
        True,
    )

    # Weakest-on-top: strongest are drawn first, weakest last.
    render_radar(
        data_sorted[::-1],
        "Convergence radar (weakest on top, 180)",
        out_dir / "convergence_radar_weakest_top.png",
        param_ranges_weak,
        axis_tick_values_weak,
        angles_180,
        False,
        False,
    )

    render_parallel_coordinates(
        data_sorted,
        "Convergence coordinate plot",
        out_dir / "convergence_parallel_coordinates.png",
        param_ranges_weak,
        axis_tick_values_weak,
    )

    render_nested_heatmaps(
        data_sorted,
        "",
        out_dir / "convergence_nested_heatmaps.png",
    )

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
