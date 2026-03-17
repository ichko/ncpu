import torch
import json
import shutil
import numpy as np
import mediapy as media
from datetime import datetime
from ncpu.nca import NeuralCA

from pathlib import Path
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from ncpu.loss import output_masked_rollout_loss, combined_loss
from ncpu.config import TINY_AND_FARAWAY_TRAINING_CONFIG
from ncpu.dataset import MultiGateDataset
from ncpu.trainer import NCPUTrainer
from ncpu.utils import freeze_frame, save_grid_image

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
torch.set_default_device('cuda')

LEARNING_RATE = 0.001
BATCH_SIZE = 8
STEPS = 20_000
PLOT_EVERY = 1_000
NCA_CHANNELS = 16
N_OUTPUT_BITS = 1

GAUSSIAN_NOISE = 0.8
FIRE_RATES = [0.2, 0.4, 0.6, 0.8, 1.0]


def run_experiment(gaussian_noise, fire_rate):
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"coded_gates_noise{int(gaussian_noise*100)}_fr{int(fire_rate*100)}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rollouts").mkdir()
    (run_dir / "snapshots").mkdir()
    log_path = run_dir / "log.jsonl"
    print(f"\n{'='*60}")
    print(f"  Experiment: noise={gaussian_noise}, fire_rate={fire_rate}")
    print(f"  Logging to: {log_path}")
    print(f"{'='*60}\n")

    dataset = MultiGateDataset(TINY_AND_FARAWAY_TRAINING_CONFIG, nca_channels=NCA_CHANNELS)

    nca = NeuralCA(
        channels=NCA_CHANNELS,
        hidden_channels=[128],
        fire_rate=0.99,
        alive_threshold=0.1,
        zero_initialization=False,
        kernel_size=5,
        read_only_dims=[-4, -3, -2, -1],
        gaussian_noise=gaussian_noise,
        gaussian_noise_fire_rate=fire_rate,
        padding_type = "constant",
    )

    trainer = NCPUTrainer(
        nca,
        dataset.get_dataloader(batch_size=BATCH_SIZE),
        lr=LEARNING_RATE,
        gaussian_noise=gaussian_noise,
        loss_fn=combined_loss,
        input_implant_type="disabled",
    )
    trainer.sanity_check()

    pbar = tqdm(range(STEPS), desc=f"noise={gaussian_noise} fr={fire_rate}")
    for step in pbar:
        info = trainer.optim_step(steps=(30, 80), return_rollout=(step % PLOT_EVERY == 0))
        loss = info["loss"]
        num_valid_bits = info["num_valid_bits"]

        grad_norm = trainer.metrics[-1].get("grad_norm") if trainer.metrics else None
        pbar.set_description(
            f"[n={gaussian_noise} fr={fire_rate}] loss={loss:.4f}  bits={num_valid_bits:.2f}/{N_OUTPUT_BITS}"
            + (f"  gnorm={grad_norm:.3f}" if grad_norm else "")
        )

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "step": step, "loss": loss,
                "num_valid_bits": num_valid_bits,
                "grad_norm": grad_norm,
                "gaussian_noise": gaussian_noise,
                "fire_rate": fire_rate,
            }) + "\n")

        if step % PLOT_EVERY == 0:
            rollout = info["rollout"]
            nca_out = info["nca_out"]
            out = info["out"]
            inp = info["inp"]

            if inp.dim() == 4 and inp.shape[1] > 1:
                inp_display = inp[:, 0:1, :, :].squeeze(1)
            else:
                inp_display = inp

            B = nca_out.shape[0]
            if rollout is not None:
                _, T, C, H, W = rollout.shape
            else:
                H, W = nca_out.shape[-2], nca_out.shape[-1]

            print(f"\n{'─'*60}")
            print(f"  step: {step}  loss: {loss:.8f}  bits: {num_valid_bits:.2f}/{N_OUTPUT_BITS}  noise: {gaussian_noise}  fr: {fire_rate}")
            print(f"  nca_out: min={nca_out.min():.3f}  max={nca_out.max():.3f}  mean={nca_out.mean():.4f}")

            trainer.save_checkpoint()

            snap_b = min(B, 4)
            diff = (nca_out[:snap_b] - out[:snap_b]).abs()
            snap_path = run_dir / "snapshots" / f"snapshot_{step:07d}.png"
            save_grid_image(
                snap_path,
                [inp_display[:snap_b].cpu(), out[:snap_b].cpu(),
                 nca_out[:snap_b].detach().cpu(), diff.detach().cpu()],
                row_vmin=[None, None, None, 0],
                row_vmax=[None, None, None, 2],
            )
            shutil.copy(snap_path, run_dir / "snapshot_latest.png")

            losses_hist = [m["loss"] for m in trainer.metrics]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.scatter(range(len(losses_hist)), losses_hist, s=0.5, alpha=0.4, color="steelblue")
            ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("masked MSE loss")
            ax.set_title(f"Multi-Gate NCA (noise={gaussian_noise}, fr={fire_rate}) — step {step}")
            fig.tight_layout(); fig.savefig(run_dir / "loss_curve.png", dpi=120); plt.close(fig)

            bits_vals = [m["num_valid_bits"] for m in trainer.metrics if "num_valid_bits" in m]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.scatter(range(len(bits_vals)), bits_vals, s=0.5, alpha=0.4, color="darkorange")
            ax.set_ylim(0, N_OUTPUT_BITS + 0.5); ax.set_xlabel("step"); ax.set_ylabel("mean valid bits")
            ax.set_title(f"Multi-Gate NCA (noise={gaussian_noise}, fr={fire_rate}) — bits — step {step}")
            fig.tight_layout(); fig.savefig(run_dir / "bits_curve.png", dpi=120); plt.close(fig)

            if rollout is not None:
                gif_b = BATCH_SIZE
                gif_c = min(C, NCA_CHANNELS)
                r = rollout[:gif_b, ::2, :gif_c]
                T_sub = r.shape[1]
                target = out[:gif_b].detach().cpu()

                _, _, _, cH, cW = r.shape
                grid_H = (1 + gif_c) * cH + (gif_c) * 1
                grid_W = gif_b * cW + (gif_b - 1) * 1

                frame_list = []
                for t in range(T_sub):
                    frame = torch.zeros(grid_H, grid_W)

                    for b in range(gif_b):
                        col_start = b * (cW + 1)
                        frame[0:cH, col_start:col_start + cW] = target[b]

                    for ci in range(gif_c):
                        row_start = (ci + 1) * cH + ci * 1
                        for b in range(gif_b):
                            col_start = b * (cW + 1)
                            frame[row_start:row_start + cH, col_start:col_start + cW] = r[b, t, ci].detach().cpu()

                    frame_list.append(frame)

                frames = torch.stack(frame_list)
                frames_np = frames.cpu().numpy()
                frames_rgb = media.to_rgb(frames_np, vmin=-1, vmax=1, cmap="viridis")
                frames_rgb = freeze_frame(torch.from_numpy(frames_rgb), timesteps=[0, -1], repeat=8)
                gif_path = run_dir / "rollouts" / f"rollout_{step:07d}.gif"
                media.write_video(str(gif_path), frames_rgb.numpy(), fps=10, codec="gif")
                shutil.copy(gif_path, run_dir / "rollout_latest.gif")

            print(f"  -> saved artifacts for step {step}")

    trainer.save_checkpoint()


for fr in FIRE_RATES:
    run_experiment(GAUSSIAN_NOISE, fr)