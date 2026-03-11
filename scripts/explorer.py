"""
NCA Explorer — streamlit app for interactively running a trained NCA model.

Usage:
    uv run streamlit run scripts/explorer.py
"""

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.dataset import NCPUDataset, sample_4bit_adder, sample_8bit_adder
from ncpu.nca import NeuralCA
from ncpu.utils import freeze_frame, make_grid, make_io_screen

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
SAMPLER_MAP = {
    "sample_4bit_adder": sample_4bit_adder,
    "sample_8bit_adder": sample_8bit_adder,
}

st.set_page_config(page_title="NCA Explorer", layout="wide")

st.markdown("""
<style>
    /* compact padding — enough top clearance for the fixed Streamlit header */
    .block-container { padding-top: 4rem; padding-bottom: 0.5rem; }
    /* nearest-neighbour scaling for all images */
    img { image-rendering: pixelated; image-rendering: crisp-edges; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_run(run_name: str, checkpoint_name: str):
    run_dir = RUNS_DIR / run_name
    config = json.loads((run_dir / "config.json").read_text())
    ds_cfg, nca_cfg = config["ds"], config["nca"]

    ds_config = Namespace(
        W=ds_cfg["W"], H=ds_cfg["H"], r=ds_cfg["r"],
        spacing=tuple(ds_cfg["spacing"]),
        sampler=SAMPLER_MAP[ds_cfg["sampler"]],
        balanced=False,
    )
    dataset = NCPUDataset(ds_config)

    nca = NeuralCA(
        channels=nca_cfg["channels"],
        hidden_channels=nca_cfg["hidden_channels"],
        fire_rate=nca_cfg["fire_rate"],
        alive_threshold=nca_cfg["alive_threshold"],
        zero_initialization=False,
        kernel_size=nca_cfg["kernel_size"],
        num_perception_kernels=nca_cfg["num_perception_kernels"],
        read_only_dims=nca_cfg.get("read_only_dims", []),
    )
    nca.load_state_dict(torch.load(
        run_dir / "checkpoints" / checkpoint_name, map_location="cpu"
    ))
    nca.eval()
    return nca, dataset, config


# ── Helpers ───────────────────────────────────────────────────────────────────

def int_to_bits(n: int, width: int) -> list[int]:
    return [int(b) for b in f"{n:0{width}b}"]


def make_input_screen(a: int, b: int, dataset: NCPUDataset, bits_per_num: int) -> np.ndarray:
    bits = int_to_bits(a, bits_per_num) + int_to_bits(b, bits_per_num)
    return make_io_screen(
        W=dataset.W, H=dataset.H, r=dataset.r, spacing=dataset.spacing,
        left_input=bits, right_input=[],
    )


def run_nca(nca: NeuralCA, dataset: NCPUDataset, a: int, b: int, steps: int, bits_per_num: int):
    screen = make_input_screen(a, b, dataset, bits_per_num)
    inp = torch.from_numpy(screen).float() / 128.0 - 1.0
    state = torch.zeros(1, nca.channels, dataset.H, dataset.W)
    state[0, 0] = inp
    state[0, 1] = inp
    with torch.no_grad():
        return nca(state, steps=steps)  # (1, T+1, C, H, W)


def decode_output(rollout: torch.Tensor, dataset: NCPUDataset):
    bit_masks = dataset.get_output_bit_masks()   # (n_bits, H, W)
    bm_sum = bit_masks.sum(dim=(-1, -2))
    last = rollout[0, -1, 0]                     # (H, W)
    avg = (last.unsqueeze(0) * bit_masks).sum(dim=(-1, -2)) / bm_sum
    pred_bits = (avg > 0).int().tolist()
    return pred_bits, int("".join(map(str, pred_bits)), 2)


ZOOM = 4


def zoom_nearest(arr: np.ndarray, factor: int = ZOOM) -> np.ndarray:
    """Upscale (H, W, 3) or (T, H, W, 3) by repeating pixels — no interpolation."""
    arr = np.repeat(arr, factor, axis=-3)
    arr = np.repeat(arr, factor, axis=-2)
    return arr


def rollout_to_gif(rollout: torch.Tensor, fps: int = 10) -> str:
    frames = rollout[0, :, 0]   # (T, H, W)
    frames_rgb = torch.from_numpy(
        media.to_rgb(frames.numpy(), vmin=-1, vmax=1, cmap="viridis")
    ).unsqueeze(0)               # (1, T, H, W, 3)
    grid = make_grid(frames_rgb, nrow=1, padding=0)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=15)
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    media.write_video(tmp.name, zoom_nearest(grid.numpy()), fps=fps, codec="gif")
    return tmp.name


def rollout_to_channels_gif(rollout: torch.Tensor, fps: int = 10) -> str:
    C = rollout.shape[2]
    frames = rollout[0].permute(1, 0, 2, 3)  # (C, T, H, W)
    frames_rgb = torch.from_numpy(
        media.to_rgb(frames.numpy(), vmin=-1, vmax=1, cmap="viridis")
    )                            # (C, T, H, W, 3)
    grid = make_grid(frames_rgb, nrow=C // 2, padding=1)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=15)
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    media.write_video(tmp.name, grid.numpy(), fps=fps, codec="gif")
    return tmp.name


# ── UI ────────────────────────────────────────────────────────────────────────

if not RUNS_DIR.exists():
    st.error(f"No runs directory at {RUNS_DIR}")
    st.stop()

runs = sorted(
    [d.name for d in RUNS_DIR.iterdir() if (d / "config.json").exists()],
    reverse=True,
)
if not runs:
    st.error("No completed runs found.")
    st.stop()

# ── Top bar: run + checkpoint ─────────────────────────────────────────────────
top_left, top_right, _ = st.columns([2, 2, 3])
with top_left:
    default_run = "20260310_234709"
    default_idx = runs.index(default_run) if default_run in runs else 0
    run_name = st.selectbox("Run", runs, index=default_idx, label_visibility="collapsed")
with top_right:
    ckpt_dir = RUNS_DIR / run_name / "checkpoints"
    checkpoints = sorted([f.name for f in ckpt_dir.glob("nca_*.pt")], reverse=True)
    if not checkpoints:
        st.error("No checkpoints found.")
        st.stop()
    checkpoint = st.selectbox("Checkpoint", checkpoints, label_visibility="collapsed")

nca, dataset, config = load_run(run_name, checkpoint)
steps = config.get("optim", {}).get("steps_max", 64)
W = dataset.W
_sin, _sout = dataset.sampler()
bits_per_num = len(_sin) // 2
n_out_bits = len(_sout)
max_val = 2 ** bits_per_num - 1

st.divider()

# ── Main layout: controls (left) | output (right) ────────────────────────────
left, right = st.columns([1, 2])

with left:
    col_a, col_b = st.columns(2)
    with col_a:
        a = st.number_input("A", min_value=0, max_value=max_val, value=min(7, max_val), step=1)
    with col_b:
        b = st.number_input("B", min_value=0, max_value=max_val, value=min(5, max_val), step=1)

    c = int(a) + int(b)
    expected_bits = int_to_bits(c, n_out_bits)
    screen = make_input_screen(int(a), int(b), dataset, bits_per_num)
    out_screen = make_io_screen(
        W=dataset.W, H=dataset.H, r=dataset.r, spacing=dataset.spacing,
        left_input=[], right_input=expected_bits,
    )
    img_col, out_col = st.columns(2)
    with img_col:
        st.markdown(f"**A** `{int(a):0{bits_per_num}b}` &nbsp; **B** `{int(b):0{bits_per_num}b}`", unsafe_allow_html=True)
        screen_rgb = media.to_rgb(screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]
        st.image(zoom_nearest(screen_rgb))
    with out_col:
        st.markdown(f"**C** `{c}` &nbsp; `{c:0{n_out_bits}b}`", unsafe_allow_html=True)
        out_rgb = media.to_rgb(out_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]
        st.image(zoom_nearest(out_rgb))

    calculate = st.button("Calculate", type="primary", use_container_width=True)

with right:
    if calculate:
        with st.spinner(f"Running NCA ({steps} steps)…"):
            rollout = run_nca(nca, dataset, int(a), int(b), steps=steps, bits_per_num=bits_per_num)
            gif_path = rollout_to_gif(rollout)
            channels_gif_path = rollout_to_channels_gif(rollout)
            pred_bits, pred_int = decode_output(rollout, dataset)

        expected = int(a) + int(b)
        expected_bits = int_to_bits(expected, len(pred_bits))
        valid = sum(p == e for p, e in zip(pred_bits, expected_bits))

        last_frame = rollout[0, -1, 0].numpy()
        target_frame = (out_screen / 128.0 - 1.0)
        diff_frame = np.abs(last_frame - target_frame)

        bit_masks = dataset.get_output_bit_masks()
        out_mask = bit_masks.sum(0).numpy() > 0  # (H, W)

        last_rgb = media.to_rgb(last_frame[np.newaxis], vmin=-1, vmax=1, cmap="viridis")[0]
        output_only = last_rgb.copy()
        output_only[~out_mask] = 1.0  # white outside output circles

        gif_col, last_col, out_col, diff_col = st.columns(4)
        with gif_col:
            st.caption("rollout — ch 0")
            st.image(gif_path)
        with last_col:
            st.caption("last frame — ch 0")
            last_ann = (last_rgb * 255).astype(np.uint8).copy()
            for mask in bit_masks:
                ys, xs = np.where(mask.numpy() > 0)
                cy, cx = int(ys.mean()), int(xs.mean())
                cv2.circle(last_ann, (cx, cy), dataset.r + 1, (220, 50, 50), 2)
            st.image(zoom_nearest(last_ann))
        with out_col:
            st.caption("output activations")
            st.image(zoom_nearest(output_only))
        with diff_col:
            diff_rgb = media.to_rgb(diff_frame[np.newaxis], vmin=0, vmax=2, cmap="viridis")[0]
            st.caption("|last − target|")
            st.image(zoom_nearest(diff_rgb))

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Predicted", pred_int)
        with r2:
            st.metric("Expected", expected)
        with r3:
            st.metric("Valid bits", f"{valid} / {len(pred_bits)}")

        st.code(
            f"output   {' '.join(map(str, pred_bits))}\n"
            f"expected {' '.join(map(str, expected_bits))}"
        )

        st.caption("all channels")
        st.image(channels_gif_path, use_container_width=True, output_format="GIF")
    else:
        st.caption("← set inputs and click Calculate")
