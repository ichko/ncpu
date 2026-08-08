import os
import re
from typing import Tuple

import panel as pn
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.utils import make_grid

import torch.nn.functional as F
from torch import nn

os.environ["CXX_RNG_USE_RDRND"] = "0"

import cv2
import mediapy as media
import numpy as np
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tempfile
from pathlib import Path

EPS = 1e-8


def git_info():
    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "dirty": None}


def mass_conserving_update(beta, q, affinity, padding_type, pad):
    weights = torch.exp(beta * affinity)
    kernel_size = pad * 2 + 1

    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size),
        device=q.device,
        dtype=q.dtype,
    )

    # Map padding type
    if padding_type == "zeros":
        pad_mode = "constant"
        pad_kwargs = {"value": 0.0}
    else:
        pad_mode = padding_type
        pad_kwargs = {}

    weights_padded = F.pad(
        weights,
        (pad, pad, pad, pad),
        mode=pad_mode,
        **pad_kwargs,
    )
    Z = F.conv2d(weights_padded, kernel)

    q_over_Z = q / (Z + EPS)

    q_over_Z_padded = F.pad(
        q_over_Z,
        (pad, pad, pad, pad),
        mode=pad_mode,
        **pad_kwargs,
    )
    incoming = F.conv2d(q_over_Z_padded, kernel)

    q_next = weights * incoming
    return q_next


def make_sobel_kernels(size: int):
    assert size % 2 == 1 and size >= 3, "sobel_size must be odd and >= 3"

    # Binomial coefficients for smoothing
    def binomial(n):
        row = [1]
        for _ in range(n):
            row = [1] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [1]
        return torch.tensor(row)

    smooth_1d = binomial(size - 1)
    deriv_1d = torch.zeros(size)
    deriv_1d[0] = -1
    deriv_1d[-1] = 1

    smooth_1d = smooth_1d / smooth_1d.sum()

    sobel_x = torch.outer(smooth_1d, deriv_1d)
    sobel_y = torch.outer(deriv_1d, smooth_1d)

    identity = torch.zeros(size, size)
    identity[size // 2, size // 2] = 1.0

    return identity, sobel_x, sobel_y


EPS = 1e-8


def mass_conserving_update(beta, q, affinity, padding_type, pad):
    weights = torch.exp(beta * affinity)
    kernel_size = pad * 2 + 1

    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size),
        device=q.device,
        dtype=q.dtype,
    )

    # Map padding type
    if padding_type == "zeros":
        pad_mode = "constant"
        pad_kwargs = {"value": 0.0}
    else:
        pad_mode = padding_type
        pad_kwargs = {}

    weights_padded = F.pad(
        weights,
        (pad, pad, pad, pad),
        mode=pad_mode,
        **pad_kwargs,
    )
    Z = F.conv2d(weights_padded, kernel)

    q_over_Z = q / (Z + EPS)

    q_over_Z_padded = F.pad(
        q_over_Z,
        (pad, pad, pad, pad),
        mode=pad_mode,
        **pad_kwargs,
    )
    incoming = F.conv2d(q_over_Z_padded, kernel)

    q_next = weights * incoming
    return q_next


def make_sobel_kernels(size: int):
    assert size % 2 == 1 and size >= 3, "sobel_size must be odd and >= 3"

    # Binomial coefficients for smoothing
    def binomial(n):
        row = [1]
        for _ in range(n):
            row = [1] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [1]
        return torch.tensor(row)

    smooth_1d = binomial(size - 1)
    deriv_1d = torch.zeros(size)
    deriv_1d[0] = -1
    deriv_1d[-1] = 1

    smooth_1d = smooth_1d / smooth_1d.sum()

    sobel_x = torch.outer(smooth_1d, deriv_1d)
    sobel_y = torch.outer(deriv_1d, smooth_1d)

    identity = torch.zeros(size, size)
    identity[size // 2, size // 2] = 1.0

    return identity, sobel_x, sobel_y


def print_tensor(title, t):
    shape = tuple(t.shape)
    print(
        f"{title}: {shape} {t.dtype}, min={t.min():.2f}, max={t.max():.2f}, mean={t.mean():.2f}, std={t.std():.2f}"
    )


import torch


def add_progress_bar(
    x,
    pad_px: int,
    bar_height: int,
    progress_color,  # (3,) tuple
    padding_color,  # (3,) tuple
    passed_color,  # (3,) tuple
):
    """
    x: (B, T, H, W, 3)
    returns: (B, T, H + bar_height, W, 3)
    """

    B, T, H, W, C = x.shape
    device = x.device
    dtype = x.dtype

    # Convert colors to tensors once
    progress_color = torch.tensor(progress_color, device=device, dtype=dtype)
    padding_color = torch.tensor(padding_color, device=device, dtype=dtype)
    passed_color = torch.tensor(passed_color, device=device, dtype=dtype)

    # --- Base bar filled with progress color ---
    bar = torch.empty((B, T, bar_height, W, 3), device=device, dtype=dtype)
    bar[:] = progress_color

    # --- Padding (top + bottom inside bar) ---
    if pad_px > 0:
        bar[:, :, :pad_px] = padding_color
        bar[:, :, -pad_px:] = padding_color

    # --- Compute passed width per frame ---
    # progress fraction per time step
    progress = torch.linspace(0, 1, T, device=device, dtype=dtype).view(1, T, 1, 1)

    passed_width = (progress * W).long()  # (1, T, 1, 1)

    # Create width index grid once
    w_idx = torch.arange(W, device=device).view(1, 1, 1, W)

    # Mask where progress has passed
    passed_mask = w_idx < passed_width  # (1, T, 1, W)

    # Apply only to non-padding vertical region
    if pad_px > 0:
        bar[:, :, pad_px:-pad_px][
            passed_mask.expand(B, T, bar_height - 2 * pad_px, W)
        ] = passed_color
    else:
        bar[passed_mask.expand(B, T, bar_height, W)] = passed_color

    # --- Concatenate to bottom ---
    out = torch.cat([x, bar], dim=2)

    return out


def sequence_batch_to_html_gifs(
    tensor, width, height, return_html=False, columns=8, fps=20
):
    tensor = tensor[:, :, 0].detach().cpu().numpy()
    tensor = media.to_rgb(tensor, cmap="viridis", vmin=-1, vmax=1)

    return media.show_videos(
        tensor,
        titles=[f"#{i}" for i in range(tensor.shape[0])],
        fps=fps,
        codec="gif",
        columns=columns,
        width=width,
        height=height,
        return_html=return_html,
    )


def add_gaussian_noise(img, mean=1.0, std=1.0, min_image=0, max_image=255):
    if isinstance(img, torch.Tensor):
        noise = torch.randn_like(img) * std + mean
        noisy = img + noise
        return torch.clamp(noisy, min_image, max_image)
    elif isinstance(img, np.ndarray):
        noise = np.random.normal(mean, std, img.shape).astype(img.dtype)
        noisy = img + noise
        return np.clip(noisy, min_image, max_image).astype(img.dtype)
    else:
        raise TypeError("Input must be torch.Tensor or np.ndarray")


def make_io_screen(H, W, r, spacing, left_input, right_input):
    screen = np.full((H, W), fill_value=128, dtype=np.uint8)
    among_spacing, side_spacing = spacing
    among_spacing = int(among_spacing)
    side_spacing = int(side_spacing)
    r = int(r)

    n_left = len(left_input)
    n_rows = int(np.ceil(n_left / 2))

    v_size = n_rows * r * 2 + among_spacing * (n_rows - 1)
    top_margin = (H - v_size) // 2

    for i, bit in enumerate(left_input):
        col = i // n_rows  # 0 or 1
        row = i % n_rows

        x = side_spacing + col * (2 * r + among_spacing)
        y = top_margin + r + row * (2 * r + among_spacing)

        cv2.circle(screen, (x, y), r, 255 if bit else 0, -1)

    # among_spacing = r + r // 4
    for i, bit in enumerate(right_input):
        x = W - side_spacing
        v_size = len(right_input) * r * 2 + among_spacing * (len(right_input) - 1)
        top_margin = (H - v_size) // 2
        y = top_margin + r + i * (among_spacing + r * 2)
        cv2.circle(screen, (x, y), r, 255 if bit else 0, -1)

    return screen


def make_io_screen_bottom_aligned(H, W, r, spacing, a_bits, b_bits, output_bits):
    """Fixed-size grid with bottom-aligned circles for variable-width adder inputs.

    LSB (last element, MSB-first convention) is anchored to a fixed y position
    regardless of how many bits are provided — so the NCA sees spatially consistent
    encodings across different bit widths.

    Layout:
        Col 0 (x = side_sp):        A operand
        Col 1 (x = side_sp + step): B operand
        Right (x = W - side_sp):    output/result
    """
    screen = np.full((H, W), fill_value=128, dtype=np.uint8)
    among_sp, side_sp = int(spacing[0]), int(spacing[1])
    r = int(r)
    step = 2 * r + among_sp
    bottom_y = H - side_sp  # fixed LSB anchor regardless of bit count

    def draw_col(bits, cx):
        for i, bit in enumerate(reversed(bits)):  # i=0 → LSB at bottom
            y = bottom_y - i * step
            if r <= y < H - r:
                cv2.circle(screen, (cx, y), r, 255 if bit else 0, -1)

    if a_bits:
        draw_col(a_bits, side_sp)
    if b_bits:
        draw_col(b_bits, side_sp + step)
    if output_bits:
        draw_col(output_bits, W - side_sp)

    return screen


def make_io_screen_cols1(H, W, r, spacing, left_input, right_input):
    """Like make_io_screen but inputs and outputs are each a single column."""
    screen = np.full((H, W), fill_value=128, dtype=np.uint8)
    among_spacing, side_spacing = spacing
    among_spacing = int(among_spacing)
    side_spacing = int(side_spacing)
    r = int(r)

    for bits, x in [
        (left_input,  side_spacing),
        (right_input, W - side_spacing),
    ]:
        n = len(bits)
        v_size = n * r * 2 + among_spacing * (n - 1)
        top_margin = (H - v_size) // 2
        for i, bit in enumerate(bits):
            y = top_margin + r + i * (2 * r + among_spacing)
            cv2.circle(screen, (x, y), r, 255 if bit else 0, -1)
    return screen


def make_alu_screen(
    H,
    W,
    r,
    spacing,
    a_bits=None,
    b_bits=None,
    opcode_bits=None,
    result_bits=None,
):
    """Draw an ALU input/output screen with three-column layout.

    Layout (left → middle → right):
      Left   : A (sub-col 0) + B (sub-col 1), 8 rows × 2 columns
      Middle : opcode[0..2], single column at W//2
      Right  : result[0..7], single column at W − side_sp

    Args:
        H, W         : grid size
        r            : circle radius
        spacing      : (among_sp, side_sp)
        a_bits/b_bits: list of 8 ints (MSB-first), or None
        opcode_bits  : list of 3 ints, or None
        result_bits  : list of 8 ints, or None
    """
    among_sp = int(spacing[0])
    side_sp  = int(spacing[1])
    r        = int(r)
    screen   = np.full((H, W), fill_value=128, dtype=np.uint8)

    def _draw_col(bits, cx):
        n  = len(bits)
        v  = n * 2*r + among_sp * (n - 1)
        tm = (H - v) // 2
        for i, bit in enumerate(bits):
            cy = tm + r + i * (2*r + among_sp)
            cv2.circle(screen, (cx, cy), r, 255 if bit else 0, -1)

    # Left: A sub-col 0, B sub-col 1 — both span 8 rows with shared vertical origin
    if a_bits is not None or b_bits is not None:
        n  = 8
        v  = n * 2*r + among_sp * (n - 1)
        tm = (H - v) // 2
        for col_i, bits in enumerate([a_bits or [0]*8, b_bits or [0]*8]):
            cx = side_sp + col_i * (2*r + among_sp)
            for i, bit in enumerate(bits):
                cy = tm + r + i * (2*r + among_sp)
                cv2.circle(screen, (cx, cy), r, 255 if bit else 0, -1)

    # Middle: opcode bits, centered at W//2
    if opcode_bits:
        _draw_col(opcode_bits, W // 2)

    # Right: result bits
    if result_bits:
        _draw_col(result_bits, W - side_sp)

    return screen


def make_alu2_screen(H, W, r, among_sp, x_a, x_b, x_ctrl, x_out,
                     a_bits=None, b_bits=None, ctrl_bits=None, out_bits=None):
    """ALU v2 screen layout with four explicit columns.

    Columns
    -------
    A     : 8 bits (MSB-top, vertically centred)   at x_a
    B     : 8 bits (MSB-top, vertically centred)   at x_b
    CTRL  : 7 bits (MSB-top, vertically centred)   at x_ctrl
            layout: [op2, op1, op0, carry_in, cond2, cond1, cond0]
    OUT   : 13 bits (MSB-top, vertically centred)  at x_out
            layout: [res7..res0, carry_out, zero, neg, overflow, branch_taken]

    Pass None to leave a column blank (circles drawn as mid-grey, i.e. value=128).
    """
    screen = np.full((H, W), fill_value=128, dtype=np.uint8)
    step = 2 * r + among_sp

    def _col(bits, cx):
        if bits is None:
            return
        n = len(bits)
        v = n * 2 * r + among_sp * (n - 1)
        tm = (H - v) // 2
        for i, bit in enumerate(bits):
            cy = tm + r + i * step
            cv2.circle(screen, (cx, cy), r, 255 if bit else 0, -1)

    _col(a_bits,    x_a)
    _col(b_bits,    x_b)
    _col(ctrl_bits, x_ctrl)
    _col(out_bits,  x_out)
    return screen


def conv_stack(layer_sizes, activation, **kwargs):
    layers = []

    for i in range(len(layer_sizes) - 1):
        si, so = layer_sizes[i], layer_sizes[i + 1]
        conv = nn.Conv2d(si, so, **kwargs)

        layers.append(conv)
        layers.append(activation())
    layers.pop()  # remove last activation

    return layers



def meshgrid_xy(H: int, W: int, device=None, dtype=torch.float32):
    x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H, W)
    return xx, yy


def make_grid(
    tensor: Tensor,  # (B, T, H, W, 3)
    nrow: int = 8,
    padding: int = 2,
    pad_value: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tensor:  # (T, H_grid, W_grid, 3)
    B, T, H, W, C = tensor.shape

    xmaps = min(nrow, B)
    ymaps = -(-B // xmaps)
    grid_h = H * ymaps + padding * (ymaps + 1)
    grid_w = W * xmaps + padding * (xmaps + 1)

    grid = (
        tensor.new_tensor(pad_value)
        .view(1, 1, 1, 3)
        .expand(T, grid_h, grid_w, 3)
        .clone()
    )

    for k in range(B):
        y, x = divmod(k, xmaps)
        y0 = padding + y * (H + padding)
        x0 = padding + x * (W + padding)
        grid[:, y0 : y0 + H, x0 : x0 + W, :] = tensor[k]

    return grid


def rolling_temp_path(prefix: str, suffix: str, deque_cap: int) -> Path:
    TMPDIR = Path.home() / ".cache"
    TMPDIR.mkdir(exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}_(?P<idx>\d{{5}}){re.escape(suffix)}$")
    indices = [
        int(m.group("idx")) for f in TMPDIR.iterdir() if (m := pattern.match(f.name))
    ]
    next_idx = (max(indices) + 1) % deque_cap if indices else 0
    return TMPDIR / f"{prefix}_{next_idx:05d}{suffix}"


def tensor_to_video_pane(
    tensor: torch.Tensor,
    fps: int = 10,
    nrow: int = 8,
    padding: int = 4,
    bg_color=(0, 0, 0),
    channel: int = 0,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    deque_cap=5,
    zoom=1,
    format: str = "mp4",
):
    """Convert (B, T, C, H, W) video tensor to a gridded Panel Video/GIF pane."""
    tensor = tensor[:, :, channel]  # (B, T, H, W)
    tensor = media.to_rgb(tensor, vmin=vmin, vmax=vmax, cmap=cmap)  # (B, T, H, W, 3)
    tensor = make_grid(
        torch.from_numpy(tensor), nrow=nrow, padding=padding, pad_value=bg_color
    )
    T, H, W, _3 = tensor.shape
    W = int(zoom * W)
    H = int(zoom * H)

    if format == "gif":
        path = rolling_temp_path("video", ".gif", deque_cap=deque_cap)
        media.write_video(str(path), tensor.cpu().numpy(), fps=fps, codec="gif")
        return pn.pane.GIF(
            str(path), width=W, height=H, styles={"image-rendering": "pixelated"}
        )
    else:
        path = rolling_temp_path("video", ".mp4", deque_cap=deque_cap)
        media.write_video(str(path), tensor.cpu().numpy(), fps=fps, codec="h264")
        return pn.pane.Video(
            str(path),
            autoplay=True,
            loop=True,
            muted=True,
            width=W,
            height=H,
            styles={"image-rendering": "pixelated"},
        )


def freeze_frame(frames, timesteps, repeat):
    """Repeat specific frames to create a pause effect in a video tensor.

    Args:
        frames:    (T, ...) tensor — time on axis 0
        timesteps: list of frame indices to freeze (negative indices supported)
        repeat:    how many times to repeat each specified frame
    """
    T = frames.shape[0]
    frozen = {i % T for i in timesteps}

    chunks = []
    for i in range(T):
        f = frames[i : i + 1]
        n = repeat if i in frozen else 1
        chunks.append(f.expand(n, *frames.shape[1:]))

    return torch.cat(chunks, dim=0)


def save_grid_image(
    path,
    rows,
    nrow=8,
    padding=2,
    vmin=-1,
    vmax=1,
    cmap="viridis",
    row_vmin=None,
    row_vmax=None,
):
    """Save a multi-row grid of images as a PNG without any interpolation.

    Args:
        path: output file path
        rows: list of (B, H, W) float tensors, one per row (e.g. [inputs, outputs])
        nrow: max items per row
        padding: pixels between items
        vmin/vmax/cmap: colour mapping passed to mediapy.to_rgb
        row_vmin/row_vmax: optional per-row overrides (lists, None entries fall back to vmin/vmax)
    """
    row_grids = []
    for i, batch in enumerate(rows):
        rv_min = row_vmin[i] if row_vmin and row_vmin[i] is not None else vmin
        rv_max = row_vmax[i] if row_vmax and row_vmax[i] is not None else vmax
        n = min(nrow, batch.shape[0])
        rgb = torch.from_numpy(
            media.to_rgb(batch[:n].numpy(), vmin=rv_min, vmax=rv_max, cmap=cmap)
        ).unsqueeze(
            1
        )  # (n, 1, H, W, 3)
        row_grids.append(
            make_grid(rgb, nrow=n, padding=padding).squeeze(0)
        )  # (H, W, 3)

    # Each row has `padding` on all sides; strip top padding from rows after the first
    # to avoid double-thickness gaps between rows.
    full = torch.cat(
        [row_grids[0]] + [r[padding:] for r in row_grids[1:]],
        dim=0,
    )
    media.write_image(str(path), full.numpy())


def bit_accuracy(frame, target):
    mask = torch.abs(target) > 0.5
    bits_number = mask.sum().item()

    frame_bit = (frame[mask].sum()).int()/bits_number
    target_bit = (target[mask].sum()).int()/bits_number
    return np.abs(1.0 - np.abs(frame_bit - target_bit))


def save_cross_section_x(rollout, path, cross_section=-4, channel=0):
    path = Path(path)
    B, T, C, H, W = rollout.shape

    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(5, 3.2))

    for batch in range(B):
        line = torch.abs(rollout[batch, -1, channel, cross_section, :]).numpy()
        ax.plot(line, color="#1f77b4", linewidth=0.5, linestyle="--")


    ax.set_xlabel("H index", fontsize=9)
    ax.set_ylabel("Magnitude", fontsize=9)
    ax.set_title(f"Cross-section  idx={cross_section}  C={channel}", fontsize=10)

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linewidth=0.5, linestyle="--", color="#cccccc")
    ax.grid(which="minor", linewidth=0.25, linestyle=":", color="#dddddd")
    ax.tick_params(axis="both", labelsize=8, direction="in",
                   which="both", top=True, right=True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    if B > 1:
        ax.legend(fontsize=7, frameon=True, framealpha=0.9,
                  edgecolor="#cccccc", loc="best")

    fig.tight_layout(pad=0.5)
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_cross_section_y(rollout, path, cross_section=-4, channel=0):
    path = Path(path)
    B, T, C, H, W = rollout.shape

    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(5, 3.2))

    for batch in range(B):
        line = torch.abs(rollout[batch, -1, channel, :, cross_section]).numpy()
        ax.plot(line, color="#1f77b4", linewidth=0.5, linestyle="--")


    ax.set_xlabel("H index", fontsize=9)
    ax.set_ylabel("Magnitude", fontsize=9)
    ax.set_title(f"Cross-section  idx={cross_section}  C={channel}", fontsize=10)

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linewidth=0.5, linestyle="--", color="#cccccc")
    ax.grid(which="minor", linewidth=0.25, linestyle=":", color="#dddddd")
    ax.tick_params(axis="both", labelsize=8, direction="in",
                   which="both", top=True, right=True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    if B > 1:
        ax.legend(fontsize=7, frameon=True, framealpha=0.9,
                  edgecolor="#cccccc", loc="best")

    fig.tight_layout(pad=0.5)
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_matrix(out_dir, matrix, labels_x, labels_y, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.T.numpy(), cmap="coolwarm", vmin=0, vmax=1)

    # Annotate each cell with the value
    for i in range(matrix.T.shape[0]):
        for j in range(matrix.T.shape[1]):
            val = matrix.T[i, j].item()
            text_color = "black" if 0.3 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    # Axis labels
    ax.set_xticks(range(len(labels_x)))
    ax.set_yticks(range(len(labels_y)))
    ax.set_xticklabels(labels_x, rotation=45, ha="right")
    ax.set_yticklabels(labels_y)

    ax.set_xlabel("Noise Std", labelpad=10)
    ax.set_ylabel("Frame Rate", labelpad=10)
    ax.set_title(title, pad=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Bit Accuracy", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(out_dir, dpi=150, bbox_inches="tight")
    plt.close()


def save_rollout_gif(rollout, out, batch_size, nca_channels, gif_path):
    gif_b = batch_size
    gif_c = nca_channels
    r = rollout[:gif_b, ::2, :gif_c]
    T_sub = r.shape[1]
    target = out[:gif_b].detach().cpu()

    # Build grid with 1px black borders between cells
    # Row: target + gif_c channel rows = (1 + gif_c) rows
    # Col: gif_b batch samples
    _, _, _, cH, cW = r.shape  # cell height/width
    grid_H = (1 + gif_c) * cH + (gif_c) * 1      # rows + separators between them
    grid_W = gif_b * cW + (gif_b - 1) * 1         # cols + separators between them

    frame_list = []
    for t in range(T_sub):
        frame = torch.zeros(grid_H, grid_W)

        # Row 0: target
        for b in range(gif_b):
            col_start = b * (cW + 1)
            frame[0:cH, col_start:col_start + cW] = target[b]

        # Rows 1..gif_c: NCA channels
        for ci in range(gif_c):
            row_start = (ci + 1) * cH + ci * 1  # skip target row + separators
            for b in range(gif_b):
                col_start = b * (cW + 1)
                frame[row_start:row_start + cH, col_start:col_start + cW] = r[b, t, ci].detach().cpu()

        frame_list.append(frame)

    frames = torch.stack(frame_list)
    frames_np = frames.cpu().numpy()
    frames_rgb = media.to_rgb(frames_np, vmin=-1, vmax=1, cmap="viridis")
    frames_rgb = freeze_frame(torch.from_numpy(frames_rgb), timesteps=[0, -1], repeat=8)
    # gif_path = run_dir / "rollouts" / f"rollout_{step:07d}.gif"
    media.write_video(str(gif_path), frames_rgb.numpy(), fps=10, codec="gif")
    # shutil.copy(gif_path, run_dir / "rollout_latest.gif")


def save_rollout_png(
    path,
    rollout,          # (T, C, H, W)
    n_snapshots=6,
    snapshot_indices=None,
    max_channels=None,
    vmin=-1,
    vmax=1,
    cmap="viridis",
    dpi=150,
    channels = [], # if non-empty, only plot these channels (0-indexed); otherwise plot all up to max_channels 
    labels_x = [],
    labels_y = [],
    mark_right_output_circle=False,
    output_circle_radius_px=5,
    output_circle_color="gray",
):
    """Save a labelled rollout grid as PNG via matplotlib.

    Rows  = channels (+ one 'target' row at the top if *target* is given).
    Cols  = evenly-spaced timestep snapshots from the rollout.
    One figure per batch element in *batch_indices*.
    """

    path = Path(path)
    T, C, H, W = rollout.shape
    if max_channels is not None:
        C = min(C, max_channels)

    if snapshot_indices is not None and len(snapshot_indices) > 0:
        ts = torch.as_tensor(snapshot_indices, dtype=torch.long)
    else:
        ts = torch.linspace(0, T - 1, n_snapshots).long()

    if channels:
        n_rows = len(channels)
    else:
        n_rows = C + 1
    n_cols = len(ts)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.4 + 0.8, n_rows * 1.4 + 0.6),
        squeeze=False,
    )

    for col_idx, t_idx in enumerate(ts):
        # Target row
        row_offset = 0
        
        # Channel rows
        for ci in range(n_rows):
            ax = axes[ci + row_offset, col_idx]
            if channels:
                tile = rollout[t_idx, channels[ci]]
            else:
                tile = rollout[t_idx, ci]
            ax.imshow(tile, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal", interpolation="nearest")
            if mark_right_output_circle:
                cx = W - output_circle_radius_px - 12
                cy = H // 2
                circ = plt.Circle(
                    (cx, cy),
                    radius=output_circle_radius_px,
                    fill=False,
                    edgecolor=output_circle_color,
                    linewidth=4.0,
                )
                ax.add_patch(circ)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                if labels_y:
                    ax.set_ylabel(labels_y[ci], fontsize=20)
                else:
                    ax.set_ylabel(f"ch {ci}", fontsize=20)
            if ci == 0:
                if labels_x:
                    ax.set_title(labels_x[col_idx], fontsize=20)
                else:
                    ax.set_title(f"t={t_idx.item()}", fontsize=20)

    fig.tight_layout(pad=0.3)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_rollout_svg(path, rollout, target, n_snapshots=6, batch_indices=None, max_channels=None, vmin=-1, vmax=1, cmap="viridis", display_size=32):
    """Save a rollout as a vector SVG with a matplotlib grid.

    Parameters
    ----------
    path : str or Path
        Output file path (should end in .svg).
    rollout : Tensor (B, T, C, H, W)
        Full rollout tensor.
    target : Tensor (B, H, W)
        Target output for the first row.
    n_snapshots : int
        Number of timesteps to sample (evenly spaced).
    batch_indices : list[int] or None
        Which batch samples to show. Default: first 4.
    max_channels : int or None
        Max NCA channels to display. Default: all.
    display_size : int
        Downsample spatial dims to this size before plotting.
    """
    from matplotlib import pyplot as plt
    import numpy as np
    import torch.nn.functional as F

    B_total, T, C, H, W = rollout.shape
    if batch_indices is None:
        batch_indices = list(range(min(B_total, 4)))
    if max_channels is not None:
        C = min(C, max_channels)

    # Downsample rollout: (B, T, C, H, W) → (B, T, C, ds, ds)
    ds = display_size
    if H != ds or W != ds:
        r_flat = rollout[:, :, :C].reshape(-1, 1, H, W).float()
        r_flat = F.interpolate(r_flat, size=(ds, ds), mode='nearest')
        rollout_ds = r_flat.reshape(B_total, T, C, ds, ds)
    else:
        rollout_ds = rollout[:, :, :C]

    # Downsample target: (B, H, W) → (B, ds, ds)
    if target.shape[-2] != ds or target.shape[-1] != ds:
        target_ds = F.interpolate(target.unsqueeze(1).float(), size=(ds, ds), mode='nearest').squeeze(1)
    else:
        target_ds = target

    # pick evenly-spaced timestep indices, always including first and last
    ts = list(dict.fromkeys([0] + list(np.linspace(0, T - 1, n_snapshots, dtype=int)) + [T - 1]))

    n_b = len(batch_indices)
    n_rows = n_b * (1 + C)
    n_cols = len(ts)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.2, n_rows * 1.2),
        squeeze=False,
    )

    for bi, b in enumerate(batch_indices):
        row_offset = bi * (1 + C)
        for col, t in enumerate(ts):
            # target row
            ax = axes[row_offset, col]
            data = target_ds[b].detach().cpu().numpy()
            ax.pcolormesh(np.flipud(data), vmin=vmin, vmax=vmax, cmap=cmap, rasterized=False)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"b{b} tgt", fontsize=6)
            if bi == 0:
                ax.set_title(f"t={t}", fontsize=7)

            # channel rows
            for ci in range(C):
                ax = axes[row_offset + 1 + ci, col]
                data = rollout_ds[b, t, ci].detach().cpu().numpy()
                ax.pcolormesh(np.flipud(data), vmin=vmin, vmax=vmax, cmap=cmap, rasterized=False)
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"ch{ci}", fontsize=6)

    fig.tight_layout(pad=0.3)
    fig.savefig(str(path), format="svg", bbox_inches="tight")
    plt.close(fig)
