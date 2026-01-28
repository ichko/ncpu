import os

os.environ["CXX_RNG_USE_RDRND"] = "0"

import cv2
import mediapy as media
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


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


def add_gaussian_noise(img, mean=0, std=1.0):
    noise = torch.randn_like(img) * std + mean
    noisy = img + noise
    return torch.clamp(noisy, 0, 255)


def make_io_screen(H, W, r, spacing, left_input, right_input):
    screen = np.zeros((H, W), dtype=np.uint8)
    among_spacing, side_spacing = spacing

    for i, bit in enumerate(left_input):
        x = side_spacing
        v_size = len(left_input) * r * 2 + among_spacing * (len(left_input) - 1)
        top_margin = (H - v_size) // 2
        y = top_margin + r + i * (among_spacing + r * 2)
        cv2.circle(screen, (x, y), r if bit else r // 2, 255, -1)

    for i, bit in enumerate(right_input):
        x = W - side_spacing
        v_size = len(right_input) * r * 2 + among_spacing * (len(right_input) - 1)
        top_margin = (H - v_size) // 2
        y = top_margin + r + i * (among_spacing + r * 2)
        cv2.circle(screen, (x, y), r if bit else r // 2, 255, -1)

    return screen


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


def test_show(model, image, device, steps=5000, channels=16, name="image"):
    board = make_image_seed(image, channels).to(device)

    cv2.imshow(
        name + "first",
        cv2.resize(image[0].detach().permute(1, 2, 0).numpy(), (256, 256)),
    )
    to_show = board[0, :3, ...].detach().cpu().permute(1, 2, 0).numpy().squeeze()
    cv2.imshow(name, cv2.resize(to_show, (256, 256)))

    for _ in range(steps):
        board = model.step(board)

        for n in range(3):
            to_show = (
                board[0, n : n + 1, ...]
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .squeeze()
            )
            cv2.imshow(f"gray_board_{n}", cv2.resize(to_show, (256, 256)))
        rgb_to_show = (
            board[0, :3, ...].detach().cpu().permute(1, 2, 0).numpy().squeeze()
        )
        cv2.imshow(f"rgb", cv2.resize(rgb_to_show, (256, 256)))
        cv2.waitKey(1)


def setup_device(device=None):
    """
    Set up device for training. If no device is specified, use cuda if available,
        otherwise use mps or cpu.
    Args:
        device (str): device to use for training (defaults to None)
    Returns:
        device (torch.device): device to use for training
    """
    if device is not None:
        device = torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def no_grad(func):
    def wrap(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)
            return result

    return wrap


def create_board(width, height, channels, device):
    seed = make_seed(width, height, channels).to(device)
    return seed


def render_cs_image_opencv(
    cs_tensor, render=True, title="cs_image", new_size=(256, 256)
):
    if render:
        cs_numpy = cs_tensor.detach().cpu().numpy()
        cs_scaled = (cs_numpy * 255).astype(np.uint8)

        cs_resized = cv2.resize(
            cs_scaled, new_size, interpolation=cv2.INTER_NEAREST
        )  # or INTER_LINEAR, INTER_CUBIC, etc.
        cs_resized = cv2.applyColorMap(cs_resized, cv2.COLORMAP_JET)

        # Display the image
        cv2.imshow(title, cs_resized)
        cv2.waitKey(1)  # Small delay for display (adjust as needed)


def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor.")


def flatten(x):
    flat_x_i = []
    shapes = []
    for x_i in x:
        shapes.append(np.array(x_i).shape)
        flat_x_i.append(np.array(x_i).flatten())

    flat_x = np.concatenate(flat_x_i)
    return flat_x, shapes


def unflatten(flat_x, shapes):
    x = []
    offset = 0
    for shape in shapes:
        lenght = 1
        for n in shape:
            lenght *= n
        x.append(flat_x[offset : offset + lenght].reshape(shape))
        offset += lenght
    return x


def load_medmnsit_data(data_flag="chestmnist"):
    """
    Load the medmnist data for the given data flag.
        Source: https://github.com/MedMNIST/MedMNIST

    Args:
        data_flag (str): data flag for the medmnist dataset (defaults to "chestmnist")

    Returns:
        train_dataset (MedMNIST): training dataset
        data_flag (str): data flag for the medmnist dataset
    """

    download = True

    NUM_EPOCHS = 3
    BATCH_SIZE = 128
    lr = 0.001

    info = INFO[data_flag]
    task = info["task"]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])

    DataClass = getattr(medmnist_loader, info["python_class"])

    train_dataset = DataClass(split="train", download=download)

    train_loader = get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)

    return train_dataset, data_flag


def save_medmnist_image(train_dataset, data_flag):
    """
    Save the medmnist image for the given data flag.

    Args:
        train_dataset (MedMNIST): training dataset
        data_flag (str): data flag for the medmnist dataset

    Returns:
        None
    """

    # px = 1/plt.rcParams["figure.dpi"]  # pixel in inches
    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(28*px, 28*px)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)

    # ax.imshow(train_dataset.montage(length=1), cmap="viridis")
    # fig.savefig("../data/" + data_flag + ".png")


def rgb_to_grayscale(img):
    """
    Convert an RGB image to a grayscale image.

    Args:
        img (torch.Tensor): image of shape (1, 3, H, W) where the first three channels are RGB.

    Returns:
        torch.Tensor: grayscale image of shape (1, 1, H, W)
    """

    # Extract RGB channels
    rgb = img[:, :3, ...]

    # Grayscale conversion using luminosity method
    grayscale = (
        0.33 * rgb[:, 0:1, ...] + 0.33 * rgb[:, 1:2, ...] + 0.33 * rgb[:, 2:3, ...]
    )
    return grayscale


def rgba_to_grayscale(img):
    """
    Convert an RGBA image to a grayscale image.

    Args:
        img (torch.Tensor): image of shape (1, 4, H, W) where the first three
            channels are RGB and the last channel is the alpha channel.

    Returns:
        torch.Tensor: grayscale image of shape (1, 1, H, W)
    """

    # Extract RGB channels
    rgb = img[:, :4, ...]

    # Grayscale conversion using luminosity method
    grayscale = (
        0.33 * rgb[:, 0:1, ...]
        + 0.33 * rgb[:, 1:2, ...]
        + 0.33 * rgb[:, 2:3, ...]
        + 0.1 * rgb[:, 3:4, ...]
    )
    return grayscale


def load_image(path, size=28):
    """
    Load an image and convert it to a tensor.

    Args:
        path (str): path to image
        size (int, optional): max size of image (defaults to 28)

    Returns:
        img (torch.Tensor): image of shape (1, 4, size, size) where the first three
            channels are RGB and the last channel is the alpha channel
    """

    # load image and resize
    img = Image.open(path).resize((size, size))

    # convert to float and normalize
    img = np.float32(img) / 255.0

    # premultiply RGB channels by alpha channel
    # img[..., :3] *= img[..., 3:]

    # print("img: ", img.shape)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    # print("img gray: ", img.shape)
    # img = rgba_to_grayscale(img)
    # print("img gray: ", img.shape)
    # convert to tensor and permute dimensions
    return img


def rgba_to_rgb(img):
    """
    Convert an RGBA image to an RGB image.

    Args:
        img (torch.Tensor): image of shape (1, 4, size, size) where the first three
            channels are RGB and the last channel is the alpha channel

    Returns:
        img (torch.Tensor): image of shape (1, 3, size, size) where the first three
            channels are RGB
    """

    # separate RGB and alpha channels
    rgb = img[:, :3, ...]
    alpha = torch.clamp(img[:, 3:4, ...], 0.0, 1.0)

    # convert to RGB
    img = torch.clamp(1.0 - alpha + rgb, 0, 1)

    return img


def pad_image(img, p=0):
    """
    Pad an image with zeros.

    Args:
        img (torch.Tensor): image of shape (1, n_channels, size, size)
        p (int, optional): number of pixels to pad image (defaults to 0)

    Returns:
        img (torch.Tensor): padded image of shape (1, n_channels, size + 2p, size + 2p)
    """

    img = nn.functional.pad(img, (p, p, p, p), mode="constant", value=0)

    return img


def make_ma_seed(size, length, agents, n_channels=16):

    asqrt = int(agents**0.5)
    if agents != int(asqrt**2):
        raise ValueError("square root of agents number has to be integer")

    x = torch.zeros((1, n_channels, size, length), dtype=torch.float32)
    width = int(size / asqrt)
    height = int(size / length)
    for w_step in range(asqrt):
        for h_step in range(asqrt):
            x[
                :,
                3:,
                w_step * (width) + (width) // 2,
                h_step * (height) + (height) // 2,
            ] = 1.0

    return x


def make_seed(width, length, n_channels=8):
    """
    Initialize the grid with zeros, except a single seed cell in the center,
        which will have all channels except RGB set to one.

    Args:
        size (int): size of the image
        n_channels (int): number of channels. Defaults to 16 and must be greater
            than 4, because the first 3 channels are RGB and the 4th channel is
            the alpha channel

    Returns:
        x (torch.Tensor): initialization grid of shape (1, n_channels, size, size)
    """

    # if n_channels < 4:
    #     raise ValueError("n_channels must be greater than 4")

    x = torch.zeros((1, n_channels, width, length), dtype=torch.float32)
    x[:, 1:, width // 2, length // 2] = 1.0
    return x


def make_image_seed(image, n_channels=16, size=4):
    """
    Initialize the grid with zeros, except a single seed cell in the center,
        which will have all channels except RGB set to one.

    Args:
        size (int): size of the image
        n_channels (int): number of channels. Defaults to 16 and must be greater
            than 4, because the first 3 channels are RGB and the 4th channel is
            the alpha channel

    Returns:
        x (torch.Tensor): initialization grid of shape (1, n_channels, size, size)
    """

    # if n_channels < 4:
    #     raise ValueError("n_channels must be greater than 4")
    if len(image.shape) != 4:
        raise ValueError("wrong image size. Expected (B,C,W,H)")

    x = torch.zeros(
        (image.shape[0], n_channels, image.shape[2], image.shape[3]),
        dtype=torch.float32,
    )

    l_width = image.shape[2] // 2 - size
    h_width = image.shape[2] // 2 + size
    l_height = image.shape[3] // 2 - size
    h_height = image.shape[3] // 2 + size
    x[:, 1:, l_width:h_width, l_height:h_height] = image[
        :, :1, l_width:h_width, l_height:h_height
    ]

    return x


def make_seed_xy(size_x, size_y, n_channels=16):
    """
    Initialize the grid with zeros, except a single seed cell in the center,
        which will have all channels except RGB set to one.

    Args:
        size (int): size of the image
        n_channels (int): number of channels. Defaults to 16 and must be greater
            than 4, because the first 3 channels are RGB and the 4th channel is
            the alpha channel

    Returns:
        x (torch.Tensor): initialization grid of shape (1, n_channels, size, size)
    """

    # if n_channels < 4:
    #     raise ValueError("n_channels must be greater than 4")

    x = torch.zeros((1, n_channels, size_x, size_y), dtype=torch.float32)
    x[:, 1:, size_x // 2, size_y // 2] = 1.0

    return x


def make_rng_seed(size_x, size_y, n_channels=16):
    """
    Initialize the grid with zeros, except a single seed cell in the center,
    which will have all channels except RGB set to random values.

    Args:
        size (int): Size of the image.
        n_channels (int): Number of channels. Defaults to 16 and must be greater
            than 4, because the first 3 channels are RGB and the 4th channel is
            the alpha channel.

    Returns:
        x (torch.Tensor): Initialization grid of shape (1, n_channels, size, size).
    """
    x = torch.zeros((1, n_channels, size_x, size_y), dtype=torch.float32)
    x[:, 1:, :, :] = torch.rand(1, n_channels - 1, size_x, size_y)

    return x


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
