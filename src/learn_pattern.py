import torch
import numpy as np

from datetime import datetime

from PIL import Image

from src.piotrs_model.utils import L2, make_seed, load_image
from src.piotrs_model.model import NCADish

class NoModelAdaptedException(Exception):
    def __init__(self, message="No Model Adapted!"):
        super().__init__(message)


def adapt_env(model, cs, target):
    length = np.random.randint(64, 96)

    for n in range(length):
        cs = model(cs)

    target_channel = target.shape[1]
    batch_loss, total_loss = L2(cs[:, :target_channel, ...], target)
    return batch_loss, total_loss, cs


if __name__ == "__main__":
    model = NCADish(
        width = 28,
        height = 28,
        channels = 8,
        device="cuda"
    )

    
    target = load_image("test_data/lenna.png", size=28)

    board = make_seed(28,28)
    board = model.seed(board)
    model.target(target)
    testable_model, avg_loss = model.grow(
        env=adapt_env,
        epochs=1000,
        learnign_rate=0.005,
        pool_size=1024,
    )

    frames = []
    for n in range(96*3):
        # Example: list of frames (NumPy arrays)
        frame = testable_model(board)
        frame_np = frame[0,...].permute(1, 2, 0).cpu().detach().numpy()  # (H, W, C)
        frame_np = (frame_np[..., :3] * 255).astype(np.uint8)
        frames.append(frame_np)

    # Convert NumPy arrays to PIL images
    pil_frames = [
        Image.fromarray(frame.resize(128, 128).astype(np.uint8))
        for frame in frames
    ]

    # Save as GIF
    pil_frames[0].save(
        "output.gif",
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,  # milliseconds per frame
        loop=0         # 0 = infinite loop
    )
