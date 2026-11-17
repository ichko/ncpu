#!/usr/bin/env python3
"""
Minimal demo script for a trained NCA model.

Loads a trained model checkpoint, generates a simple input, and visualizes the output.

Usage:
    uv run python demo_trained_model.py --checkpoint runs/extrap_train5_s0_20261013_044806/checkpoints/nca_009501.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from argparse import Namespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ncpu.nca import NeuralCA


def load_model(checkpoint_path, device="cuda"):
    """Load a trained NCA model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use the same nca_config as the training script
    nca_config = Namespace(
        channels=16,
        hidden_channels=[128],
        fire_rate=0.5,
        alive_threshold=0,
        zero_initialization=False,
        kernel_size=7,
        perception="static_sobel",
        num_perception_kernels=3,
        learnable_initial_state=False,
        padding_type="zeros",
        read_only_dims=[1],
    )
    
    model = NeuralCA(**vars(nca_config)).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def generate_simple_input(batch_size=1, W=32, H=32, device="cuda"):
    """Generate a simple random input for testing."""
    # Random input with 16 channels (matching the model's expectations)
    inp = torch.rand((batch_size, 16, H, W), device=device) * 2 - 1  # Range: [-1, 1]
    print(f"Generated input shape: {inp.shape}")  # Debug print
    return inp


def visualize_output(inp, pred):
    """Visualize input and prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(inp[0, 0].cpu().numpy(), cmap="viridis")
    axes[0].set_title("Input")
    axes[1].imshow(pred[0, 0].cpu().numpy(), cmap="viridis")
    axes[1].set_title("Prediction")
    plt.tight_layout()
    plt.savefig("demo_visualization.png")
    plt.close()
    print("Saved visualization to demo_visualization.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to run the model on")
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.checkpoint, device=args.device)
    print(f"Loaded model from {args.checkpoint}")
    
    # Generate input
    inp = generate_simple_input(device=args.device)
    print(f"Input shape: {inp.shape}")
    
    # Run the model
    with torch.no_grad():
        rollout = model(inp, steps=32)  # [B, T, C, H, W]
        pred = rollout[:, -1]  # Last timestep
    print(f"Prediction shape: {pred.shape}")
    
    # Visualize
    visualize_output(inp, pred)


if __name__ == "__main__":
    main()