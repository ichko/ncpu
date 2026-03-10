import os

import torch.nn.functional as F
from torch import nn

os.environ["CXX_RNG_USE_RDRND"] = "0"

import cv2
import mediapy as media
import numpy as np
import torch

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


def add_gaussian_noise(img, mean=1.0, std=1.0, min_image = 0, max_image = 255):
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
