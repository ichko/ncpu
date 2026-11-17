#!/usr/bin/env python3
"""
Export a window-gate checkpoint to the JS model JSON used by the sliding-window
demo (demo/sweep.html + demo/js/sweep.js).

The format matches the existing hand-exported JSONs in demo/data/ (W1/b1/W2 are
base64 float32 arrays in the exact layout the WebGL2 fragment shader samples).

Usage:
    uv run python scripts/other/export_window_model.py \
        --checkpoint runs/window_gates_4x1_s0_<ts>/checkpoints/nca_last.pt \
        --out demo/data/window_gates.json
"""

import base64
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.utils import make_sobel_kernels
from ncpu.window import (
    GATE_CODES,
    INPUT_CHANNEL,
    INPUT_POS,
    NCA_CHANNELS,
    OUTPUT_POS,
    RADIUS,
    READ_ONLY_DIMS,
    SELECTOR_CHANNEL,
    WINDOW_SIZE,
)


def _b64(t):
    arr = np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint .pt, nca_last.pt, or a run directory",
    )
    parser.add_argument("--out", default="demo/data/window_gates.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    p = Path(args.checkpoint)
    config = {}
    if p.is_dir():
        ckpts = sorted(p.glob("checkpoints/nca_*.pt"))
        if not ckpts:
            sys.exit(f"no checkpoints under {p}")
        ckpt_path = ckpts[-1]
        if (p / "config.json").exists():
            config = json.loads((p / "config.json").read_text())
    else:
        ckpt_path = p
        if (p.parent.parent / "config.json").exists():
            config = json.loads((p.parent.parent / "config.json").read_text())

    nca_cfg = config.get("nca", {})
    channels = int(nca_cfg.get("channels", NCA_CHANNELS))
    hidden = int((nca_cfg.get("hidden_channels", [128]) or [128])[0])
    kernel = int(nca_cfg.get("kernel_size", 7))
    fire_rate = float(nca_cfg.get("fire_rate", 0.5))
    padding = nca_cfg.get("padding_type", "zeros")
    read_only = list(nca_cfg.get("read_only_dims", READ_ONLY_DIMS))
    if (
        channels != 16
        or padding != "zeros"
        or len(nca_cfg.get("hidden_channels", [128])) != 1
    ):
        sys.exit("WebGL export requires 16 channels, one hidden layer and zero padding")

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    weights = ckpt["nca"] if isinstance(ckpt, dict) and "nca" in ckpt else ckpt

    model = NeuralCA(
        channels=channels,
        hidden_channels=[hidden],
        fire_rate=fire_rate,
        alive_threshold=0,
        zero_initialization=False,
        kernel_size=kernel,
        num_perception_kernels=3,
        read_only_dims=read_only,
        padding_type=padding,
    )
    model.load_state_dict(weights)

    sd = model.state_dict()
    W1 = sd["rule.rule.0.weight"].reshape(hidden, 3 * channels)  # [128, 48]
    b1 = sd["rule.rule.0.bias"]  # [128]
    W2 = sd["rule.rule.2.weight"].reshape(channels, hidden)  # [16, 128]

    identity, sobel_x, sobel_y = make_sobel_kernels(kernel)
    sobel = np.concatenate(
        [
            identity.numpy().ravel(),
            sobel_x.numpy().ravel(),
            sobel_y.numpy().ravel(),
        ]
    )

    in3 = 3 * channels
    out = {
        "key": "window_gates",
        "kind": "window_gates",
        "run": str(ckpt_path),
        "channels": channels,
        "hidden": hidden,
        "kernel": kernel,
        "fire_rate": fire_rate,
        "padding": padding,
        "read_only": read_only,
        "input_dims": [0, 1],
        "window": WINDOW_SIZE,
        "radius": RADIUS,
        "inputs": [list(xy) for xy in INPUT_POS],
        "output": list(OUTPUT_POS),
        "gate_codes": GATE_CODES,
        "gates": config.get("gates", ["AND", "OR", "XOR"]),
        "training": config.get("training", {}),
        "selector_channel": SELECTOR_CHANNEL,
        "position_channel": 3 if 3 in read_only else None,
        "position_encoding": "tile_diagonal" if 3 in read_only else None,
        "input_channel": INPUT_CHANNEL,
        "in3": in3,
        "vec4_per_row": in3 // 4,
        "cvec4": channels // 4,
        "sobel": {"data": _b64(torch.from_numpy(sobel)), "k": kernel},
        "W1": {"data": _b64(W1), "w": in3 // 4, "h": hidden},
        "b1": {"data": _b64(b1), "n": hidden},
        "W2": {"data": _b64(W2.permute(1, 0)), "w": channels // 4, "h": hidden},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(
        f"wrote {args.out}  (channels={channels} hidden={hidden} kernel={kernel} "
        f"read_only={read_only})"
    )


if __name__ == "__main__":
    main()
