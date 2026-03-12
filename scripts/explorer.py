"""
NCA Explorer — streamlit app for interactively running a trained NCA model.
Supports adder runs (4-bit, 8-bit) and ALU runs.

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

from ncpu.dataset import ALUDataset, NCPUDataset, sample_4bit_adder, sample_8bit_adder
from ncpu.nca import NeuralCA
from ncpu.utils import freeze_frame, make_alu_screen, make_grid, make_io_screen

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
SAMPLER_MAP = {
    "sample_4bit_adder": sample_4bit_adder,
    "sample_8bit_adder": sample_8bit_adder,
}
ALU_OPCODES = ["ADD", "SUB", "AND", "OR", "XOR", "NOT", "SHL", "SHR"]

st.set_page_config(page_title="NCA Explorer", layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 4rem; padding-bottom: 0.5rem; }
    img { image-rendering: pixelated; image-rendering: crisp-edges; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────

def _build_nca(nca_cfg):
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
    return nca


@st.cache_resource
def load_adder_run(run_name: str, checkpoint_name: str):
    run_dir = RUNS_DIR / run_name
    config  = json.loads((run_dir / "config.json").read_text())
    ds_cfg, nca_cfg = config["ds"], config["nca"]

    dataset = NCPUDataset(Namespace(
        W=ds_cfg["W"], H=ds_cfg["H"], r=ds_cfg["r"],
        spacing=tuple(ds_cfg["spacing"]),
        sampler=SAMPLER_MAP[ds_cfg["sampler"]],
        balanced=False,
    ))
    nca = _build_nca(nca_cfg)
    nca.load_state_dict(torch.load(run_dir / "checkpoints" / checkpoint_name, map_location="cpu"))
    nca.eval()
    return nca, dataset, config


@st.cache_resource
def load_alu_run(run_name: str, checkpoint_name: str):
    run_dir = RUNS_DIR / run_name
    config  = json.loads((run_dir / "config.json").read_text())
    ds_cfg, nca_cfg = config["ds"], config["nca"]

    dataset = ALUDataset(Namespace(
        W=ds_cfg["W"], H=ds_cfg["H"], r=ds_cfg["r"],
        side=ds_cfg["side"], among=ds_cfg["among"],
    ))
    nca = _build_nca(nca_cfg)
    nca.load_state_dict(torch.load(run_dir / "checkpoints" / checkpoint_name, map_location="cpu"))
    nca.eval()
    return nca, dataset, config


def is_alu_run(config):
    return config.get("run", {}).get("task") == "alu_8bit"


# ── Shared helpers ────────────────────────────────────────────────────────────

def int_to_bits(n: int, width: int) -> list[int]:
    return [int(b) for b in f"{n:0{width}b}"]


ZOOM = 4

def zoom_nearest(arr: np.ndarray, factor: int = ZOOM) -> np.ndarray:
    arr = np.repeat(arr, factor, axis=-3)
    arr = np.repeat(arr, factor, axis=-2)
    return arr


def _run_nca(nca, dataset, screen: np.ndarray, steps: int):
    inp = torch.from_numpy(screen).float() / 128.0 - 1.0
    state = torch.zeros(1, nca.channels, dataset.H, dataset.W)
    state[0, 0] = inp
    state[0, 1] = inp
    with torch.no_grad():
        return nca(state, steps=steps)


def decode_bits(rollout: torch.Tensor, dataset):
    bit_masks = dataset.get_output_bit_masks()
    bm_sum    = bit_masks.sum(dim=(-1, -2))
    last      = rollout[0, -1, 0]
    avg       = (last.unsqueeze(0) * bit_masks).sum(dim=(-1, -2)) / bm_sum
    return (avg > 0).int().tolist()


def rollout_to_gif(rollout: torch.Tensor, fps: int = 10) -> str:
    frames     = rollout[0, :, 0]
    frames_rgb = torch.from_numpy(
        media.to_rgb(frames.numpy(), vmin=-1, vmax=1, cmap="viridis")
    ).unsqueeze(0)
    grid = make_grid(frames_rgb, nrow=1, padding=0)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=15)
    tmp  = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    media.write_video(tmp.name, zoom_nearest(grid.numpy()), fps=fps, codec="gif")
    return tmp.name


def rollout_to_channels_gif(rollout: torch.Tensor, fps: int = 10) -> str:
    C          = rollout.shape[2]
    frames     = rollout[0].permute(1, 0, 2, 3)
    frames_rgb = torch.from_numpy(
        media.to_rgb(frames.numpy(), vmin=-1, vmax=1, cmap="viridis")
    )
    grid = make_grid(frames_rgb, nrow=C // 2, padding=1)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=15)
    tmp  = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    media.write_video(tmp.name, grid.numpy(), fps=fps, codec="gif")
    return tmp.name


def show_results(rollout, out_screen, dataset, bit_masks=None):
    """Shared output panel: rollout gif, annotated last frame, activations, diff."""
    if bit_masks is None:
        bit_masks = dataset.get_output_bit_masks()
    out_mask  = bit_masks.sum(0).numpy() > 0

    last_frame   = rollout[0, -1, 0].numpy()
    target_frame = out_screen / 128.0 - 1.0
    diff_frame   = np.abs(last_frame - target_frame)
    last_rgb     = media.to_rgb(last_frame[np.newaxis], vmin=-1, vmax=1, cmap="viridis")[0]

    output_only  = last_rgb.copy()
    output_only[~out_mask] = 1.0

    gif_path = rollout_to_gif(rollout)
    ch_path  = rollout_to_channels_gif(rollout)

    gif_col, last_col, act_col, diff_col = st.columns(4)
    with gif_col:
        st.caption("rollout — ch 0")
        st.image(gif_path)
    with last_col:
        st.caption("last frame")
        ann = (last_rgb * 255).astype(np.uint8).copy()
        for mask in bit_masks:
            ys, xs = np.where(mask.numpy() > 0)
            cy, cx = int(ys.mean()), int(xs.mean())
            cv2.circle(ann, (cx, cy), dataset.r + 1, (220, 50, 50), 2)
        st.image(zoom_nearest(ann))
    with act_col:
        st.caption("output activations")
        st.image(zoom_nearest(output_only))
    with diff_col:
        diff_rgb = media.to_rgb(diff_frame[np.newaxis], vmin=0, vmax=2, cmap="viridis")[0]
        st.caption("|last − target|")
        st.image(zoom_nearest(diff_rgb))

    st.caption("all channels")
    st.image(ch_path, use_container_width=True, output_format="GIF")


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

# ── Run / checkpoint picker ───────────────────────────────────────────────────
top_left, top_right, _ = st.columns([2, 2, 3])
with top_left:
    default_run = "20260310_234709"
    default_idx = runs.index(default_run) if default_run in runs else 0
    run_name = st.selectbox("Run", runs, index=default_idx, label_visibility="collapsed")
with top_right:
    ckpt_dir    = RUNS_DIR / run_name / "checkpoints"
    checkpoints = sorted([f.name for f in ckpt_dir.glob("nca_*.pt")], reverse=True)
    if not checkpoints:
        st.error("No checkpoints found.")
        st.stop()
    checkpoint = st.selectbox("Checkpoint", checkpoints, label_visibility="collapsed")

# detect task type before loading
_peek = json.loads((RUNS_DIR / run_name / "config.json").read_text())
_alu  = is_alu_run(_peek)

if _alu:
    nca, dataset, config = load_alu_run(run_name, checkpoint)
else:
    nca, dataset, config = load_adder_run(run_name, checkpoint)

steps = config.get("optim", {}).get("steps_max", 64)

st.divider()
left, right = st.columns([1, 2])

# ═══════════════════════════════════════════════════════════════════════════════
# ALU UI
# ═══════════════════════════════════════════════════════════════════════════════
if _alu:
    with left:
        op_col, cin_col = st.columns([3, 1])
        with op_col:
            opcode_name = st.selectbox("Opcode", ALU_OPCODES)
        with cin_col:
            carry_in = st.number_input("Cin", min_value=0, max_value=1, value=0, step=1)

        col_a, col_b = st.columns(2)
        with col_a:
            a = st.number_input("A", min_value=0, max_value=255, value=42, step=1)
            st.code(f"{int(a):08b}")
        with col_b:
            b = st.number_input("B", min_value=0, max_value=255, value=27, step=1)
            st.code(f"{int(b):08b}")

        opcode_idx  = ALU_OPCODES.index(opcode_name)
        inp_screen  = dataset._make_screen(
            a=int_to_bits(int(a), 8), b=int_to_bits(int(b), 8),
            carry_in=[int(carry_in)], opcode=int_to_bits(opcode_idx, 3),
        )
        # compute expected output for preview
        from ncpu.dataset import _compute_alu
        opcode_idx_preview = opcode_idx
        exp_result_pre, exp_flags_pre = _compute_alu(int(a), int(b), int(carry_in), opcode_idx_preview)
        exp_out_screen = dataset._make_screen(
            result=int_to_bits(exp_result_pre, 8), flags=exp_flags_pre,
        )

        img_col, out_col = st.columns(2)
        with img_col:
            st.markdown(f"`{int(a):08b}` **{opcode_name}** `{int(b):08b}`", unsafe_allow_html=True)
            inp_rgb = media.to_rgb(inp_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]
            st.image(zoom_nearest(inp_rgb))
        with out_col:
            carry_out_pre, overflow_pre, zero_pre, neg_pre = exp_flags_pre
            st.markdown(f"`{exp_result_pre:08b}` = **{exp_result_pre}**", unsafe_allow_html=True)
            exp_rgb = media.to_rgb(exp_out_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]
            st.image(zoom_nearest(exp_rgb))
            st.caption(
                f"C={carry_out_pre} V={overflow_pre} Z={zero_pre} N={neg_pre}"
            )

        calculate = st.button("Calculate", type="primary", use_container_width=True)

    with right:
        if calculate:
            with st.spinner(f"Running NCA ({steps} steps)…"):
                rollout = _run_nca(nca, dataset, inp_screen, steps)
                bits    = decode_bits(rollout, dataset)

            result_bits = bits[:8]
            flag_bits   = bits[8:]
            result_int  = int("".join(map(str, result_bits)), 2)
            carry_out, overflow, zero, negative = flag_bits

            # expected (software reference)
            exp_result, exp_flags = _compute_alu(int(a), int(b), int(carry_in), opcode_idx)
            exp_bits = int_to_bits(exp_result, 8) + exp_flags
            valid    = sum(p == e for p, e in zip(bits, exp_bits))

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Result", result_int)
            with r2:
                st.metric("Expected", exp_result)
            with r3:
                st.metric("Valid bits", f"{valid} / 12")

            f1, f2, f3, f4 = st.columns(4)
            for col, name, val, exp in zip(
                [f1, f2, f3, f4],
                ["carry", "overflow", "zero", "negative"],
                flag_bits, exp_flags,
            ):
                col.metric(name, val, delta=None,
                           help=f"expected {exp}" + (" ✓" if val == exp else " ✗"))

            st.code(
                f"result   {' '.join(map(str, result_bits))}  ({result_int})\n"
                f"expected {' '.join(map(str, int_to_bits(exp_result, 8)))}  ({exp_result})"
            )

            out_screen = dataset._make_screen(
                result=int_to_bits(exp_result, 8), flags=exp_flags
            )
            show_results(rollout, out_screen, dataset)
        else:
            st.caption("← set inputs and click Calculate")

# ═══════════════════════════════════════════════════════════════════════════════
# Adder UI
# ═══════════════════════════════════════════════════════════════════════════════
else:
    _sin, _sout  = dataset.sampler()
    bits_per_num = len(_sin) // 2
    n_out_bits   = len(_sout)
    max_val      = 2 ** bits_per_num - 1

    with left:
        col_a, col_b = st.columns(2)
        with col_a:
            a = st.number_input("A", min_value=0, max_value=max_val, value=min(7, max_val), step=1)
        with col_b:
            b = st.number_input("B", min_value=0, max_value=max_val, value=min(5, max_val), step=1)

        c            = int(a) + int(b)
        expected_bits = int_to_bits(c, n_out_bits)
        inp_screen = make_io_screen(
            W=dataset.W, H=dataset.H, r=dataset.r, spacing=dataset.spacing,
            left_input=int_to_bits(int(a), bits_per_num) + int_to_bits(int(b), bits_per_num),
            right_input=[],
        )
        out_screen = make_io_screen(
            W=dataset.W, H=dataset.H, r=dataset.r, spacing=dataset.spacing,
            left_input=[], right_input=expected_bits,
        )
        img_col, out_col = st.columns(2)
        with img_col:
            st.markdown(f"**A** `{int(a):0{bits_per_num}b}` &nbsp; **B** `{int(b):0{bits_per_num}b}`",
                        unsafe_allow_html=True)
            st.image(zoom_nearest(media.to_rgb(inp_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]))
        with out_col:
            st.markdown(f"**C** `{c}` &nbsp; `{c:0{n_out_bits}b}`", unsafe_allow_html=True)
            st.image(zoom_nearest(media.to_rgb(out_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]))

        calculate = st.button("Calculate", type="primary", use_container_width=True)

    with right:
        if calculate:
            with st.spinner(f"Running NCA ({steps} steps)…"):
                rollout   = _run_nca(nca, dataset, inp_screen, steps)
                pred_bits = decode_bits(rollout, dataset)

            pred_int = int("".join(map(str, pred_bits)), 2)
            valid    = sum(p == e for p, e in zip(pred_bits, expected_bits))

            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted", pred_int)
            r2.metric("Expected",  c)
            r3.metric("Valid bits", f"{valid} / {n_out_bits}")

            st.code(
                f"output   {' '.join(map(str, pred_bits))}\n"
                f"expected {' '.join(map(str, expected_bits))}"
            )
            show_results(rollout, out_screen, dataset)
        else:
            st.caption("← set inputs and click Calculate")
