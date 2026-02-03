import os

os.environ["CXX_RNG_USE_RDRND"] = "0"

import cv2
import mediapy as media
import numpy as np
import torch


def print_tensor(title, t):
    shape = tuple(t.shape)
    print(
        f"{title}: {shape} {t.dtype}, min={t.min():.2f}, max={t.max():.2f}, mean={t.mean():.2f}, std={t.std():.2f}"
    )


def sequence_batch_to_html_gifs(
    tensor, width, height, return_html=False, columns=8, fps=20
):
    tensor = tensor[:, :, 0].detach().cpu().numpy()
    tensor = media.to_rgb(tensor, cmap="viridis", vmin=0, vmax=1)

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


def add_gaussian_noise(img, mean=1.0, std=1.0):
    if isinstance(img, torch.Tensor):
        noise = torch.randn_like(img) * std + mean
        noisy = img + noise
        return torch.clamp(noisy, 0, 255)
    elif isinstance(img, np.ndarray):
        noise = np.random.normal(mean, std, img.shape).astype(img.dtype)
        noisy = img + noise
        return np.clip(noisy, 0, 255).astype(img.dtype)
    else:
        raise TypeError("Input must be torch.Tensor or np.ndarray")


def make_io_screen(H, W, r, spacing, left_input, right_input):
    screen = np.zeros((H, W), dtype=np.uint8)
    among_spacing, side_spacing = spacing

    for i, bit in enumerate(left_input):
        x = side_spacing
        v_size = len(left_input) * r * 2 + among_spacing * (len(left_input) - 1)
        top_margin = (H - v_size) // 2
        y = top_margin + r + i * (among_spacing + r * 2)
        cv2.circle(screen, (x, y), r, 255, -1 if bit else 1)

    for i, bit in enumerate(right_input):
        x = W - side_spacing
        v_size = len(right_input) * r * 2 + among_spacing * (len(right_input) - 1)
        top_margin = (H - v_size) // 2
        y = top_margin + r + i * (among_spacing + r * 2)
        cv2.circle(screen, (x, y), r, 255, -1 if bit else 1)

    return screen


# do not remove, it can be useful later
def show_pool_seeds_cv2(pool_seeds, n_cols=16, scale_each=1.0):
    """
    Display a grid of images from pool_seeds using OpenCV.

    Args:
        pool_seeds: Tensor of shape (batch_size, channels, height, width)
        n_cols: Number of columns in the grid
        scale_each: Scale factor for each image
    """
    pool_seeds = pool_seeds.detach().cpu()

    # Make sure it's numpy
    imgs = []
    for img in pool_seeds:
        img = img.permute(1, 2, 0)  # (H, W, C)
        img_np = img.numpy()

        # Rescale if needed
        if scale_each != 1.0:
            img_np = cv2.resize(
                img_np,
                (0, 0),
                fx=scale_each,
                fy=scale_each,
                interpolation=cv2.INTER_NEAREST,
            )
        imgs.append(img_np)

    # Determine grid shape
    batch_size = len(imgs)
    n_rows = (batch_size + n_cols - 1) // n_cols

    # Get individual image size
    h, w = imgs[0].shape[:2]
    c = imgs[0].shape[2] if imgs[0].ndim == 3 else 1

    # Create blank canvas
    grid_img = np.zeros((n_rows * h, n_cols * w, c))  # White background

    # Paste images
    for idx, img in enumerate(imgs):
        row = idx // n_cols
        col = idx % n_cols
        grid_img[row * h : (row + 1) * h, col * w : (col + 1) * w] = img

    # If grayscale, remove last dim
    if c == 1:
        grid_img = grid_img.squeeze(-1)

    # Show
    cv2.imshow("Pool Seeds Grid", grid_img[:, :, :4])
    cv2.waitKey(10)


def make_circle_masks(width, height):
    """
    Make circle masks of size (size, size) with random center and radius.

    Args:
        size (int): size of the image

    Returns:
        mask (torch.Tensor): circle masks of shape (1, size, size)
    """

    # create grid
    x = torch.linspace(-1.0, 1.0, width).unsqueeze(0).unsqueeze(0)
    y = torch.linspace(-1.0, 1.0, height).unsqueeze(1).unsqueeze(0)

    # intialize random center and radius
    center = torch.rand(2, 1, 1, 1).uniform_(-0.5, 0.5)
    r = torch.rand(1, 1, 1).uniform_(0.1, 0.4)

    # calculate mask
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = (x * x + y * y < 1.0).float()

    return mask


def L1(target, cs):
    """
    Calculate the L1 loss between target image and cell state.

    Args:
        target (torch.Tensor): target image of shape (batch_size, 4, size, size)
        cs (torch.Tensor): cell state

    Returns:
        loss_batch (torch.Tensor): L1 loss for each image in batch
        loss (torch.Tensor): L1 loss
    """

    # calculate loss for each image in batch but only take first 4 rgba channels
    loss_batch = (torch.abs(target - rgba_to_grayscale(cs[:, :4, ...]))).mean(
        dim=[1, 2, 3]
    )

    # take mean over loss_batch
    loss = loss_batch.mean()

    return loss_batch, loss


def L2(target, cs):
    """
    Calculate the L2 loss between target image and cell state.

    Args:
        target (torch.Tensor): target image of shape (batch_size, 4, size, size)
        cs (torch.Tensor): cell state

    Returns:
        loss_batch (torch.Tensor): L2 loss for each image in batch
        loss (torch.Tensor): L2 loss
    """

    # calculate loss for each image in batch but only take first 4 rgba channels
    # print(rgba_to_grayscale(cs[:, :4, ...]).shape,target.shape)
    loss_batch = ((target - cs) ** 2).mean(dim=[1, 2, 3])

    # take mean over loss_batch
    loss = loss_batch.mean()

    return loss_batch, loss


def Manhattan(target, cs):
    """
    Calculate the Manhattan loss between target image and cell state.

    Args:
        target (torch.Tensor): target image of shape (batch_size, 4, size, size)
        cs (torch.Tensor): cell state

    Returns:
        loss_batch (torch.Tensor): Manhattan loss for each image in batch
        loss (torch.Tensor): Manhatten loss
    """

    # calculate loss for each image in batch but only take first 4 rgba channels
    loss_batch = (torch.abs(target - cs[:, :4, ...])).sum(dim=[1, 2, 3])

    # take mean over loss_batch
    loss = loss_batch.mean()

    return loss_batch, loss


def Hinge(target, cs):
    """
    Calculate the Hinge loss between target image and cell state.

    Args:
        target (torch.Tensor): target image of shape (batch_size, 4, size, size)
        cs (torch.Tensor): cell state

    Returns:
        loss_batch (torch.Tensor): Hinge loss for each image in batch
        loss (torch.Tensor): Hinge loss
    """

    # calculate loss for each image in batch but only take first 4 rgba channels
    loss_batch = torch.max(
        torch.abs(target - cs[:, :4, ...]) - 0.5, torch.zeros_like(target)
    ).mean(dim=[1, 2, 3])

    # take mean over loss_batch
    loss = loss_batch.mean()

    return loss_batch, loss
