"""Animated-GIF writer with no ffmpeg and no cv2 dependency.

``mediapy.write_video(..., codec="gif")`` shells out to ffmpeg, which is absent
on the unite cluster, and ``ncpu.utils`` pulls in cv2, which needs libGL — also
absent there.  This module needs only matplotlib (for the colormap) and Pillow,
so it works on a bare compute node.
"""

import numpy as np
from matplotlib import colormaps
from PIL import Image


def to_rgb(frames, vmin=-1.0, vmax=1.0, cmap="viridis"):
    """(T, H, W) float -> (T, H, W, 3) uint8, colour-mapped."""
    a = np.asarray(frames, dtype=np.float32)
    a = np.clip((a - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    rgba = colormaps[cmap](a)                      # (..., 4) float in [0, 1]
    return (rgba[..., :3] * 255).astype(np.uint8)


def write_gif(path, frames, fps=20, vmin=-1.0, vmax=1.0, cmap="viridis", zoom=1, loop=0):
    """Write (T, H, W) float frames as an animated GIF.

    Args:
        zoom: integer nearest-neighbour upscale (GIFs of a 44x44 grid are tiny)
    """
    rgb = to_rgb(frames, vmin=vmin, vmax=vmax, cmap=cmap)
    imgs = []
    for f in rgb:
        im = Image.fromarray(f)
        if zoom != 1:
            im = im.resize((im.width * zoom, im.height * zoom), Image.NEAREST)
        imgs.append(im)
    duration = max(20, int(round(1000.0 / max(fps, 1))))   # ms per frame
    imgs[0].save(
        str(path), save_all=True, append_images=imgs[1:],
        duration=duration, loop=loop, optimize=False,
    )
    return path
