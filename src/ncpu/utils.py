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
import tempfile
from pathlib import Path

EPS = 1e-8


def git_info():
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
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
    tensor, width, height, return_html=False, columns=8, fps=20, channels=[0],
):
    for chn in channels:
        np_tensor = tensor[:, :, chn].detach().cpu().numpy()
        np_tensor = media.to_rgb(np_tensor, cmap="viridis", vmin=-1, vmax=1)
        media.show_videos(
            np_tensor,
            titles=[f"#{i}:{chn}" for i in range(np_tensor.shape[0])],
            fps=fps,
            codec="gif",
            columns=columns,
            width=width,
            height=height,
            return_html=return_html,
        )
    print("------------------------------------------------------")

def add_gaussian_noise(img, mean=0.0, std=1.0):
    img_max_int = torch.round(img.max()).int()
    img_min_int = torch.round(img.min()).int()
    if isinstance(img, torch.Tensor):
        noise = torch.randn_like(img) * std + mean
        noisy = img + noise
        noisy = torch.clip(noisy, img_min_int, img_max_int)
        return noisy
    elif isinstance(img, np.ndarray):
        noise = np.random.normal(mean, std, img.shape).astype(img.dtype)
        noisy = img + noise
        return np.clip(noisy, img_min_int, img_max_int).astype(img.dtype)
    else:
        raise TypeError("Input must be torch.Tensor or np.ndarray")

def make_io_screen(H, W, r, spacing, left_input, right_input):
    screen = np.ones((H, W), dtype=np.uint8) * 128
    among_spacing, side_spacing = spacing

    for i, bit in enumerate(left_input):
        x = side_spacing
        v_size = len(left_input) * r * 2 + among_spacing * (len(left_input) - 1)
        top_margin = (H - v_size) // 2
        y = top_margin + r + i * (among_spacing + r * 2)
        # cv2.circle(screen, (x, y), r, 256, -1 if bit else 1)
        cv2.circle(screen, (x, y), r, 256 if bit else 0, -1)

    for i, bit in enumerate(right_input):
        x = W - side_spacing
        v_size = len(right_input) * r * 2 + among_spacing * (len(right_input) - 1)
        top_margin = (H - v_size) // 2
        y = top_margin + r + i * (among_spacing + r * 2)
        cv2.circle(screen, (x, y), r, 256 if bit else 0, -1)

    return screen

# def make_io_screen(H, W, r, spacing, left_input, right_input):
    # screen = np.full((H, W), fill_value=128, dtype=np.uint8)
    # among_spacing, side_spacing = spacing
    # among_spacing = int(among_spacing)
    # side_spacing = int(side_spacing)
    # r = int(r)

    # n_left = len(left_input)
    # n_rows = int(np.ceil(n_left / 2))

    # v_size = n_rows * r * 2 + among_spacing * (n_rows - 1)
    # top_margin = (H - v_size) // 2

    # for i, bit in enumerate(left_input):
    #     col = i // n_rows  # 0 or 1
    #     row = i % n_rows

    #     x = side_spacing + col * (2 * r + among_spacing)
    #     y = top_margin + r + row * (2 * r + among_spacing)

    #     cv2.circle(screen, (x, y), r, 255 if bit else 0, -1)

    # # among_spacing = r + r // 4
    # for i, bit in enumerate(right_input):
    #     x = W - side_spacing
    #     v_size = len(right_input) * r * 2 + among_spacing * (len(right_input) - 1)
    #     top_margin = (H - v_size) // 2
    #     y = top_margin + r + i * (among_spacing + r * 2)
    #     cv2.circle(screen, (x, y), r, 255 if bit else 0, -1)

    # return screen


def make_alu_screen(H, W, r, side, among,
                    a=None, b=None, carry_in=None, opcode=None,
                    result=None, flags=None):
    """Draw an ALU input/output screen.

    Layout (left→right):
        col 0  opcode           (3 bits, vertically centered)
        col 1  A operand        (8 bits, full height)
        col 2  B operand        (8 bits, full height)
        col 3  carry-in         (1 bit,  top-aligned with A/B)
        [4r gap]
        col 4  result           (8 bits, full height)
        col 5  flags            (4 bits, vertically centered)
            flags order: carry-out, overflow, zero, negative

    Args:
        H, W   : grid size
        r      : circle radius
        side   : left/right margin (px)
        among  : spacing between circle edges within a column
        a, b   : list of 8 ints (bits), or None
        carry_in: list of 1 int, or None
        opcode : list of 3 ints, or None
        result : list of 8 ints, or None
        flags  : list of 4 ints [carry_out, overflow, zero, negative], or None
    """
    screen = np.full((H, W), 128, dtype=np.uint8)
    step = 2 * r + among  # center-to-center distance between consecutive bits

    # vertical origin for 8-bit columns
    col8_h = 8 * 2 * r + 7 * among
    top8 = (H - col8_h) // 2

    def col_x(col_idx):
        return side + col_idx * step

    def draw_col(bits, cx, top_y):
        for i, bit in enumerate(bits):
            cy = top_y + r + i * step
            cv2.circle(screen, (cx, cy), r, 255 if bit else 0, -1)

    # col 0: opcode (3 bits, vertically centered)
    if opcode is not None:
        op_h = 3 * 2 * r + 2 * among
        op_top = (H - op_h) // 2
        draw_col(opcode, col_x(0), op_top)
    # col 1: A
    if a is not None:
        draw_col(a, col_x(1), top8)
    # col 2: B
    if b is not None:
        draw_col(b, col_x(2), top8)
    # col 3: carry-in (top-aligned with A/B)
    if carry_in is not None:
        cy = top8 + r
        cv2.circle(screen, (col_x(3), cy), r, 255 if carry_in[0] else 0, -1)

    # gap = 4r edge-to-edge between col 3 and result
    result_cx = col_x(3) + r + 4 * r + r  # = col_x(3) + 6r
    flags_cx  = result_cx + step

    # result: 8 bits, full height
    if result is not None:
        draw_col(result, result_cx, top8)
    # flags: 4 bits, vertically centered
    if flags is not None:
        flags_h = 4 * 2 * r + 3 * among
        flags_top = (H - flags_h) // 2
        draw_col(flags, flags_cx, flags_top)

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
    x = torch.linspace(1, -1, W, device=device, dtype=dtype)
    y = torch.linspace(1, -1, H, device=device, dtype=dtype)
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


def save_grid_image(path, rows, nrow=8, padding=2, vmin=-1, vmax=1, cmap="viridis",
                    row_vmin=None, row_vmax=None):
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
        rv_min = (row_vmin[i] if row_vmin and row_vmin[i] is not None else vmin)
        rv_max = (row_vmax[i] if row_vmax and row_vmax[i] is not None else vmax)
        n = min(nrow, batch.shape[0])
        rgb = torch.from_numpy(
            media.to_rgb(batch[:n].numpy(), vmin=rv_min, vmax=rv_max, cmap=cmap)
        ).unsqueeze(1)  # (n, 1, H, W, 3)
        row_grids.append(make_grid(rgb, nrow=n, padding=padding).squeeze(0))  # (H, W, 3)

    # Each row has `padding` on all sides; strip top padding from rows after the first
    # to avoid double-thickness gaps between rows.
    full = torch.cat(
        [row_grids[0]] + [r[padding:] for r in row_grids[1:]],
        dim=0,
    )
    media.write_image(str(path), full.numpy())

def save_rollout_pdf(path, rollout, target, n_snapshots=6, batch_indices=None, max_channels=None, vmin=-1, vmax=1, cmap="viridis"):
    """Save a rollout as a vector PDF with a matplotlib grid.

    Parameters
    ----------
    path : str or Path
        Output file path (should end in .pdf).
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
    """
    from matplotlib import pyplot as plt
    import numpy as np

    B_total, T, C, H, W = rollout.shape
    if batch_indices is None:
        batch_indices = list(range(min(B_total, 4)))
    if max_channels is not None:
        C = min(C, max_channels)

    # pick evenly-spaced timestep indices
    ts = np.linspace(0, T - 1, n_snapshots, dtype=int)

    n_b = len(batch_indices)
    # rows: for each batch sample → (1 target row + C channel rows)
    # columns: n_snapshots timesteps
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
            # target row (same across timesteps)
            ax = axes[row_offset, col]
            ax.imshow(target[b].detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"b{b} tgt", fontsize=6)
            if bi == 0:
                ax.set_title(f"t={t}", fontsize=7)

            # channel rows
            for ci in range(C):
                ax = axes[row_offset + 1 + ci, col]
                ax.imshow(rollout[b, t, ci].detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"ch{ci}", fontsize=6)

    fig.tight_layout(pad=0.3)
    fig.savefig(str(path), format="pdf", bbox_inches="tight")
    plt.close(fig)
