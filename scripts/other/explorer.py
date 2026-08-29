"""
NCA Explorer — streamlit app for interactively running a trained NCA model.
Supports adder runs (4-bit, 8-bit) and ALU runs.

Usage:
    uv run streamlit run scripts/other/explorer.py
"""

import io
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.dataset import (
    ALU2Dataset, ALU2_COND_NAMES, ALU2_OP_NAMES, _compute_alu2, _int_to_bits_msb,
    ALUDataset, ExtrapolationDataset, NCPUDataset,
    sample_4bit_adder, sample_4bit_multiplier, sample_8bit_adder,
    sample_AND_gate, sample_OR_gate, sample_NOR_gate, sample_NAND_gate, sample_XOR_gate,
)
from ncpu.nca import NeuralCA
from ncpu.utils import freeze_frame, make_alu2_screen, make_alu_screen, make_grid, make_io_screen, make_io_screen_bottom_aligned, make_io_screen_cols1

RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "runs"
SAMPLER_MAP = {
    "sample_4bit_adder":      sample_4bit_adder,
    "sample_8bit_adder":      sample_8bit_adder,
    "sample_4bit_multiplier": sample_4bit_multiplier,
    "sample_AND_gate":        sample_AND_gate,
    "sample_OR_gate":         sample_OR_gate,
    "sample_NOR_gate":        sample_NOR_gate,
    "sample_NAND_gate":       sample_NAND_gate,
    "sample_XOR_gate":        sample_XOR_gate,
}
SCREEN_FN_MAP = {
    "make_io_screen":               make_io_screen,
    "make_io_screen_cols1":         make_io_screen_cols1,
    "make_io_screen_bottom_aligned": make_io_screen_bottom_aligned,
}
ALU_OPCODES = ["ADD", "SUB", "AND", "OR", "XOR", "NOT", "SHL", "SHR"]

st.set_page_config(page_title="NCA Explorer", layout="wide")
st.markdown("""
<style>
    .block-container {
        padding-top: 4rem;
        padding-bottom: 0.5rem;
        max-width: 1200px;
    }
    .pixelated img { image-rendering: pixelated; image-rendering: crisp-edges; }
    img { border-radius: 0 !important; }
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
        padding_type=nca_cfg.get("padding_type", "circular"),
    )
    return nca


@st.cache_resource
def load_adder_run(run_name: str, checkpoint_name: str):
    run_dir = RUNS_DIR / run_name
    config  = json.loads((run_dir / "config.json").read_text())
    ds_cfg, nca_cfg = config["ds"], config["nca"]

    screen_fn = SCREEN_FN_MAP.get(ds_cfg.get("screen_fn", "make_io_screen"), make_io_screen)
    dataset = NCPUDataset(Namespace(
        W=ds_cfg["W"], H=ds_cfg["H"], r=ds_cfg["r"],
        spacing=tuple(ds_cfg["spacing"]),
        sampler=SAMPLER_MAP[ds_cfg["sampler"]],
        balanced=False,
        screen_fn=screen_fn,
    ))
    nca = _build_nca(nca_cfg)
    nca.load_state_dict(torch.load(run_dir / "checkpoints" / checkpoint_name, map_location="cpu"))
    nca.eval()
    return nca, dataset, config


@st.cache_resource
def load_extrapolation_run(run_name: str, checkpoint_name: str):
    run_dir = RUNS_DIR / run_name
    config  = json.loads((run_dir / "config.json").read_text())
    ds_cfg, nca_cfg = config["ds"], config["nca"]

    dataset = ExtrapolationDataset(
        W=ds_cfg["W"], H=ds_cfg["H"], r=ds_cfg["r"],
        spacing=tuple(ds_cfg["spacing"]),
        min_bits=ds_cfg["min_bits"],
        max_bits=8,  # always load with full 8-bit capacity for inference
        batch_size=ds_cfg["batch_size"],
    )
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
    ))
    nca = _build_nca(nca_cfg)
    nca.load_state_dict(torch.load(run_dir / "checkpoints" / checkpoint_name, map_location="cpu"))
    nca.eval()
    return nca, dataset, config


def is_alu_run(config):
    return config.get("run", {}).get("task") == "alu_8bit"

def is_alu2_run(config):
    return config.get("run", {}).get("experiment") == "E_alu2"

@st.cache_resource
def load_alu2_run(run_name: str, checkpoint_name: str):
    run_dir = RUNS_DIR / run_name
    config  = json.loads((run_dir / "config.json").read_text())
    ds_cfg, nca_cfg = config["ds"], config["nca"]
    dataset = ALU2Dataset(Namespace(
        W=ds_cfg["W"], H=ds_cfg["H"], r=ds_cfg["r"], among_sp=ds_cfg["among_sp"],
        x_a=ds_cfg["x_a"], x_b=ds_cfg["x_b"], x_ctrl=ds_cfg["x_ctrl"], x_out=ds_cfg["x_out"],
    ))
    nca = _build_nca(nca_cfg)
    ckpt_path = run_dir / "checkpoints" / checkpoint_name
    nca.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True), strict=False)
    nca.eval()
    return nca, dataset, config

def is_extrapolation_run(config):
    return "min_bits" in config.get("ds", {})


# ── Shared helpers ────────────────────────────────────────────────────────────

def int_to_bits(n: int, width: int) -> list[int]:
    return [int(b) for b in f"{n:0{width}b}"]


ZOOM = 4

def zoom_nearest(arr: np.ndarray, factor: int = ZOOM) -> np.ndarray:
    arr = np.repeat(arr, factor, axis=-3)
    arr = np.repeat(arr, factor, axis=-2)
    return arr


def st_image_pixelated(img, **kwargs):
    """st.image wrapper that applies pixelated (nearest-neighbour) rendering."""
    st.markdown('<div class="pixelated">', unsafe_allow_html=True)
    st.image(img, **kwargs)
    st.markdown('</div>', unsafe_allow_html=True)


def _run_nca(nca, dataset, screen: np.ndarray, steps: int):
    inp = torch.from_numpy(screen).float() / 128.0 - 1.0
    state = torch.zeros(1, nca.channels, dataset.H, dataset.W)
    state[0, 0] = inp  # channel 0: mutable, starts with input
    state[0, 1] = inp  # channel 1: read-only anchor
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
        st_image_pixelated(zoom_nearest(ann))
    with act_col:
        st.caption("output activations")
        st_image_pixelated(zoom_nearest(output_only))
    with diff_col:
        diff_rgb = media.to_rgb(diff_frame[np.newaxis], vmin=0, vmax=2, cmap="viridis")[0]
        st.caption("|last − target|")
        st_image_pixelated(zoom_nearest(diff_rgb))

    st.caption("all channels")
    st.image(ch_path, use_container_width=True, output_format="GIF")


# ── Training monitor helpers ───────────────────────────────────────────────────


MAX_PLOT_POINTS = 50_000


def load_log(run_dir: Path):
    log_path = run_dir / "log.jsonl"
    if not log_path.exists():
        return [], [], []
    with open(log_path) as f:
        lines = f.readlines()
    n = len(lines)
    step = max(1, n // MAX_PLOT_POINTS)
    sampled = lines[::step]
    if lines and lines[-1] not in sampled:
        sampled.append(lines[-1])
    steps, losses, bits, k_vals = [], [], [], []
    val_records = []  # list of {"step", "k", "val_loss", "val_bits"}
    for line in sampled:
        try:
            d = json.loads(line)
            if d.get("phase") == "val":
                val_records.append({"step": d["step"], "k": d["k"],
                                     "val_loss": d["val_loss"], "val_bits": d["val_bits"]})
            else:
                steps.append(d["step"])
                losses.append(d["loss"])
                bits.append(d.get("num_valid_bits"))
                k_vals.append(d.get("current_k"))
        except Exception:
            pass
    return steps, losses, bits, k_vals, val_records


def _loss_curve_fig(steps, losses, val_by_k=None):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(steps, losses, s=0.5, alpha=0.3, color="steelblue", label="train")
    if val_by_k:
        colors = plt.cm.autumn([i / max(len(val_by_k), 1) for i in range(len(val_by_k))])
        for (k, vd), col in zip(sorted(val_by_k.items()), colors):
            if vd["steps"]:
                ax.plot(vd["steps"], vd["losses"], color=col, linewidth=1.2, label=f"val k={k}")
        ax.legend(fontsize=7, markerscale=4)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"Loss  (step {steps[-1]})")
    fig.tight_layout()
    return fig


def _bits_curve_fig(steps, bits, n_bits, k_vals=None, normalize=False, val_by_k=None):
    has_k = k_vals and any(k is not None for k in k_vals)

    if normalize:
        denoms = [k + 1 if k is not None else n_bits for k in (k_vals or [None] * len(bits))]
        y = [b / d * 100 if d else 0 for b, d in zip(bits, denoms)]
        ylabel, ylim, title_suffix = "valid bits (%)", (0, 105), " (normalised)"
    else:
        y = bits
        ylabel, ylim, title_suffix = "valid bits", (0, n_bits), ""

    fig, ax = plt.subplots(figsize=(6, 3))
    if has_k:
        k_clean = [k if k is not None else 1 for k in k_vals]
        sc = ax.scatter(steps, y, c=k_clean, s=0.5, cmap="plasma", vmin=1, vmax=8, alpha=0.4)
        fig.colorbar(sc, ax=ax, label="train k")
    else:
        ax.scatter(steps, y, s=0.5, alpha=0.4, color="darkorange")

    if val_by_k:
        colors = plt.cm.autumn([i / max(len(val_by_k), 1) for i in range(len(val_by_k))])
        for (k, vd), col in zip(sorted(val_by_k.items()), colors):
            if vd["steps"]:
                vb = vd["bits"]
                if normalize:
                    vb = [b / (k + 1) * 100 for b in vb]
                ax.plot(vd["steps"], vb, color=col, linewidth=1.2, label=f"val k={k}")
        ax.legend(fontsize=7, markerscale=4)

    ax.set_ylim(*ylim)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Valid bits{title_suffix}  (step {steps[-1]})")
    fig.tight_layout()
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────

if not RUNS_DIR.exists():
    st.error(f"No runs directory at {RUNS_DIR}")
    st.stop()

def _run_mtime(name):
    log = RUNS_DIR / name / "log.jsonl"
    try:
        return log.stat().st_mtime
    except FileNotFoundError:
        return 0.0

runs = sorted(
    [d.name for d in RUNS_DIR.iterdir() if (d / "config.json").exists()],
    key=_run_mtime,
    reverse=True,
)
if not runs:
    st.error("No completed runs found.")
    st.stop()

# ── Shared run selector ───────────────────────────────────────────────────────
top_run_col, _ = st.columns([2, 5])
with top_run_col:
    run_name = st.selectbox("Run", runs, index=0, label_visibility="collapsed")

run_dir = RUNS_DIR / run_name

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_mon, tab_inf = st.tabs(["📊 Training Monitor", "🔬 Inference"])

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING MONITOR TAB
# ═══════════════════════════════════════════════════════════════════════════════

@st.fragment(run_every="15s")
def _monitor_fragment(run_dir, run_name):
    steps, losses, bits, k_vals, val_records = load_log(run_dir)

    if not losses:
        st.info("No log.jsonl found yet — training may not have started.")
        return

    # infer n_bits from config
    try:
        _cfg = json.loads((run_dir / "config.json").read_text())
        _ds_cfg = _cfg["ds"]
        if is_alu2_run(_cfg):
            _ds_tmp = ALU2Dataset(Namespace(
                W=_ds_cfg["W"], H=_ds_cfg["H"], r=_ds_cfg["r"], among_sp=_ds_cfg["among_sp"],
                x_a=_ds_cfg["x_a"], x_b=_ds_cfg["x_b"], x_ctrl=_ds_cfg["x_ctrl"], x_out=_ds_cfg["x_out"],
            ))
        elif is_alu_run(_cfg):
            _ds_tmp = ALUDataset(Namespace(W=_ds_cfg["W"], H=_ds_cfg["H"], r=_ds_cfg["r"]))
        elif is_extrapolation_run(_cfg):
            _ds_tmp = ExtrapolationDataset(
                W=_ds_cfg["W"], H=_ds_cfg["H"], r=_ds_cfg["r"],
                spacing=tuple(_ds_cfg["spacing"]),
                min_bits=_ds_cfg["min_bits"], max_bits=8,
                batch_size=_ds_cfg["batch_size"],
            )
        else:
            screen_fn = SCREEN_FN_MAP.get(_ds_cfg.get("screen_fn", "make_io_screen"), make_io_screen)
            _ds_tmp = NCPUDataset(Namespace(
                W=_ds_cfg["W"], H=_ds_cfg["H"], r=_ds_cfg["r"],
                spacing=tuple(_ds_cfg["spacing"]),
                sampler=SAMPLER_MAP[_ds_cfg["sampler"]],
                balanced=False,
                screen_fn=screen_fn,
            ))
        n_bits = len(_ds_tmp.get_output_bit_masks())
    except Exception:
        n_bits = max(b for b in bits if b is not None) if bits else 1

    bits_clean = [b for b in bits if b is not None]
    k_clean    = [k for k, b in zip(k_vals, bits) if b is not None]

    m1, m2, m3 = st.columns(3)
    m1.metric("Steps logged", steps[-1] if steps else 0)
    m2.metric("Last loss", f"{losses[-1]:.6f}")
    if bits_clean:
        m3.metric("Last valid bits", f"{bits_clean[-1]:.2f} / {n_bits}")

    bits_steps = [s for s, b in zip(steps, bits) if b is not None]
    normalize  = st.checkbox("Normalise bits curve (%)", value=False)

    # group val records by k
    val_by_k = {}
    for r in val_records:
        val_by_k.setdefault(r["k"], {"steps": [], "losses": [], "bits": []})
        val_by_k[r["k"]]["steps"].append(r["step"])
        val_by_k[r["k"]]["losses"].append(r["val_loss"])
        val_by_k[r["k"]]["bits"].append(r["val_bits"])

    curve_col, bits_col = st.columns(2)
    with curve_col:
        fig = _loss_curve_fig(steps, losses, val_by_k=val_by_k)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with bits_col:
        if bits_clean:
            fig = _bits_curve_fig(bits_steps, bits_clean, n_bits, k_vals=k_clean,
                                   normalize=normalize, val_by_k=val_by_k)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No valid-bits data yet.")

    gif_path  = run_dir / "rollout_latest.gif"
    snap_path = run_dir / "snapshot_latest.png"

    media_col1, media_col2 = st.columns(2)
    with media_col1:
        st.caption("rollout_latest.gif")
        if gif_path.exists():
            try:
                st.image(str(gif_path), use_container_width=True)
            except Exception:
                st.caption("_(updating…)_")
        else:
            st.info("Not saved yet.")
    with media_col2:
        st.caption("snapshot_latest.png")
        if snap_path.exists():
            try:
                st.image(str(snap_path), use_container_width=True)
            except Exception:
                st.caption("_(updating…)_")
        else:
            st.info("Not saved yet.")


with tab_mon:
    st.divider()
    _monitor_fragment(run_dir, run_name)

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_inf:
    ckpt_dir    = run_dir / "checkpoints"
    checkpoints = sorted([f.name for f in ckpt_dir.glob("nca_*.pt")], reverse=True)
    if not checkpoints:
        st.error("No checkpoints found.")
        st.stop()

    ckpt_col, _ = st.columns([2, 5])
    with ckpt_col:
        checkpoint = st.selectbox("Checkpoint", checkpoints, label_visibility="collapsed")

    # detect task type before loading
    _peek  = json.loads((run_dir / "config.json").read_text())
    _alu   = is_alu_run(_peek)
    _alu2  = is_alu2_run(_peek)
    _extrap = is_extrapolation_run(_peek)

    if _alu2:
        nca, dataset, config = load_alu2_run(run_name, checkpoint)
    elif _alu:
        nca, dataset, config = load_alu_run(run_name, checkpoint)
    elif _extrap:
        nca, dataset, config = load_extrapolation_run(run_name, checkpoint)
    else:
        nca, dataset, config = load_adder_run(run_name, checkpoint)

    steps = config.get("optim", {}).get("steps_max", 64)

    st.divider()
    left, right = st.columns([1, 2])

    # ═══════════════════════════════════════════════════════════════════════════
    # ALU v2 UI
    # ═══════════════════════════════════════════════════════════════════════════
    if _alu2:
        ds_cfg = config["ds"]
        with left:
            op_col, cin_col = st.columns([3, 1])
            with op_col:
                opcode_name = st.selectbox("Opcode", ALU2_OP_NAMES)
            with cin_col:
                carry_in = st.number_input("Cin", min_value=0, max_value=1, value=0, step=1)

            cond_name = st.selectbox("Condition", ALU2_COND_NAMES)

            col_a, col_b = st.columns(2)
            with col_a:
                a = st.number_input("A", min_value=0, max_value=255, value=42, step=1)
                st.code(f"{int(a):08b}")
            with col_b:
                b = st.number_input("B", min_value=0, max_value=255, value=27, step=1)
                st.code(f"{int(b):08b}")

            op_idx   = ALU2_OP_NAMES.index(opcode_name)
            cond_idx = ALU2_COND_NAMES.index(cond_name)
            uses_b   = op_idx not in (5, 6, 7)

            ctrl_bits = _int_to_bits_msb(op_idx, 3) + [int(carry_in)] + _int_to_bits_msb(cond_idx, 3)
            exp_result, exp_cout, exp_branch = _compute_alu2(int(a), int(b), int(carry_in), op_idx, cond_idx)
            out_bits = _int_to_bits_msb(exp_result, 8) + [exp_cout, exp_branch]

            inp_screen = make_alu2_screen(
                ds_cfg["H"], ds_cfg["W"], ds_cfg["r"], ds_cfg["among_sp"],
                ds_cfg["x_a"], ds_cfg["x_b"], ds_cfg["x_ctrl"], ds_cfg["x_out"],
                a_bits=_int_to_bits_msb(int(a), 8),
                b_bits=_int_to_bits_msb(int(b), 8) if uses_b else None,
                ctrl_bits=ctrl_bits,
            )
            exp_out_screen = make_alu2_screen(
                ds_cfg["H"], ds_cfg["W"], ds_cfg["r"], ds_cfg["among_sp"],
                ds_cfg["x_a"], ds_cfg["x_b"], ds_cfg["x_ctrl"], ds_cfg["x_out"],
                out_bits=out_bits,
            )

            img_col, out_col = st.columns(2)
            with img_col:
                st.markdown(f"`{int(a):08b}` **{opcode_name}** `{int(b):08b}`")
                st_image_pixelated(zoom_nearest(media.to_rgb(inp_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]))
            with out_col:
                st.markdown(f"= `{exp_result:08b}` ({exp_result})")
                st_image_pixelated(zoom_nearest(media.to_rgb(exp_out_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]))
                st.caption(f"cout={exp_cout}  branch={exp_branch} ({cond_name})")

            calculate = st.button("Calculate", type="primary", use_container_width=True)

        with right:
            if calculate:
                with st.spinner(f"Running NCA ({steps} steps)…"):
                    rollout   = _run_nca(nca, dataset, inp_screen, steps)
                    pred_bits = decode_bits(rollout, dataset)

                pred_result = int("".join(map(str, pred_bits[:8])), 2)
                pred_cout   = pred_bits[8]
                pred_branch = pred_bits[9]
                valid = sum(p == e for p, e in zip(pred_bits, out_bits))

                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("Result",   pred_result)
                r2.metric("Expected", exp_result)
                r3.metric("Valid bits", f"{valid} / 10")
                r4.metric("cout", pred_cout,
                          help=f"expected {exp_cout}" + (" ✓" if pred_cout == exp_cout else " ✗"))
                r5.metric("branch", pred_branch,
                          help=f"expected {exp_branch}" + (" ✓" if pred_branch == exp_branch else " ✗"))

                st.code(
                    f"result   {' '.join(map(str, pred_bits[:8]))}  ({pred_result})\n"
                    f"expected {' '.join(map(str, _int_to_bits_msb(exp_result, 8)))}  ({exp_result})"
                )
                show_results(rollout, exp_out_screen, dataset)
            else:
                st.caption("← set inputs and click Calculate")

    # ═══════════════════════════════════════════════════════════════════════════
    # ALU UI
    # ═══════════════════════════════════════════════════════════════════════════
    elif _alu:
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
            from ncpu.dataset import _compute_alu
            exp_result_pre, exp_carry_pre = _compute_alu(int(a), int(b), int(carry_in), opcode_idx)
            exp_out_screen = dataset._make_screen(
                result=int_to_bits(exp_result_pre, 8), carry_out=[exp_carry_pre],
            )

            img_col, out_col = st.columns(2)
            with img_col:
                st.markdown(f"`{int(a):08b}` **{opcode_name}** `{int(b):08b}`", unsafe_allow_html=True)
                inp_rgb = media.to_rgb(inp_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]
                st_image_pixelated(zoom_nearest(inp_rgb))
            with out_col:
                st.markdown(f"`{exp_result_pre:08b}` = **{exp_result_pre}**", unsafe_allow_html=True)
                exp_rgb = media.to_rgb(exp_out_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]
                st_image_pixelated(zoom_nearest(exp_rgb))
                st.caption(f"carry_out={exp_carry_pre}")

            calculate = st.button("Calculate", type="primary", use_container_width=True)

        with right:
            if calculate:
                with st.spinner(f"Running NCA ({steps} steps)…"):
                    rollout = _run_nca(nca, dataset, inp_screen, steps)
                    bits    = decode_bits(rollout, dataset)

                result_bits = bits[:8]
                pred_carry  = bits[8]
                result_int  = int("".join(map(str, result_bits)), 2)

                exp_result, exp_carry = _compute_alu(int(a), int(b), int(carry_in), opcode_idx)
                exp_bits = int_to_bits(exp_result, 8) + [exp_carry]
                valid    = sum(p == e for p, e in zip(bits, exp_bits))

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Result", result_int)
                r2.metric("Expected", exp_result)
                r3.metric("Valid bits", f"{valid} / 9")
                r4.metric("carry_out", pred_carry,
                          help=f"expected {exp_carry}" + (" ✓" if pred_carry == exp_carry else " ✗"))

                st.code(
                    f"result   {' '.join(map(str, result_bits))}  ({result_int})\n"
                    f"expected {' '.join(map(str, int_to_bits(exp_result, 8)))}  ({exp_result})"
                )

                out_screen = dataset._make_screen(
                    result=int_to_bits(exp_result, 8), carry_out=[exp_carry]
                )
                show_results(rollout, out_screen, dataset)
            else:
                st.caption("← set inputs and click Calculate")

    # ═══════════════════════════════════════════════════════════════════════════
    # Extrapolation UI
    # ═══════════════════════════════════════════════════════════════════════════
    elif _extrap:
        train_max = config["run"].get("max_train_bits", 5)
        with left:
            k = st.slider("Bit width", min_value=1, max_value=8, value=train_max + 1,
                          help=f"Trained on 1–{train_max} bits. Values above {train_max} are extrapolation.")
            if k > train_max:
                st.warning(f"⚠ Extrapolation: trained up to {train_max} bits")

            max_val = 2 ** k - 1
            col_a, col_b = st.columns(2)
            with col_a:
                a = st.number_input("A", min_value=0, max_value=max_val, value=min(7, max_val), step=1)
            with col_b:
                b = st.number_input("B", min_value=0, max_value=max_val, value=min(5, max_val), step=1)

            c             = int(a) + int(b)
            n_out_bits    = k + 1
            expected_bits = int_to_bits(c, n_out_bits)
            ds_cfg        = config["ds"]

            inp_screen = make_io_screen_bottom_aligned(
                H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"],
                spacing=tuple(ds_cfg["spacing"]),
                a_bits=int_to_bits(int(a), k), b_bits=int_to_bits(int(b), k), output_bits=[],
            )
            out_screen = make_io_screen_bottom_aligned(
                H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"],
                spacing=tuple(ds_cfg["spacing"]),
                a_bits=[], b_bits=[], output_bits=expected_bits,
            )

            img_col, out_col = st.columns(2)
            with img_col:
                st.markdown(f"**A** `{int(a):0{k}b}` &nbsp; **B** `{int(b):0{k}b}`", unsafe_allow_html=True)
                st_image_pixelated(zoom_nearest(media.to_rgb(inp_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]))
            with out_col:
                st.markdown(f"**C** `{c}` &nbsp; `{c:0{n_out_bits}b}`", unsafe_allow_html=True)
                st_image_pixelated(zoom_nearest(media.to_rgb(out_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]))

            calculate = st.button("Calculate", type="primary", use_container_width=True)

        with right:
            if calculate:
                with st.spinner(f"Running NCA ({steps} steps)…"):
                    rollout   = _run_nca(nca, dataset, inp_screen, steps)
                    pred_bits = decode_bits(rollout, dataset)

                pred_bits_k = pred_bits[-(n_out_bits):]  # bottom-aligned: last n_out_bits
                pred_int    = int("".join(map(str, pred_bits_k)), 2)
                valid       = sum(p == e for p, e in zip(pred_bits_k, expected_bits))

                r1, r2, r3 = st.columns(3)
                r1.metric("Predicted", pred_int)
                r2.metric("Expected",  c)
                r3.metric("Valid bits", f"{valid} / {n_out_bits}")

                st.code(
                    f"output   {' '.join(map(str, pred_bits_k))}\n"
                    f"expected {' '.join(map(str, expected_bits))}"
                )
                show_results(rollout, out_screen, dataset)
            else:
                st.caption("← set inputs and click Calculate")

    # ═══════════════════════════════════════════════════════════════════════════
    # Adder UI
    # ═══════════════════════════════════════════════════════════════════════════
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
                st_image_pixelated(zoom_nearest(media.to_rgb(inp_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]))
            with out_col:
                st.markdown(f"**C** `{c}` &nbsp; `{c:0{n_out_bits}b}`", unsafe_allow_html=True)
                st_image_pixelated(zoom_nearest(media.to_rgb(out_screen[np.newaxis], vmin=0, vmax=255, cmap="viridis")[0]))

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
