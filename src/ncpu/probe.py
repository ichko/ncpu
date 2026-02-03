from ncpu.dataset import NCPUDataset
from ncpu.config import TINY_NAND_TRAINING_CONFIG
import matplotlib.pyplot as plt
import torch

def probe(image, config, left_n = 0, right_n = 0): #H, W, r, spacing, right, left):
    H, W = image.shape
    r = config.r
    spacing = config.spacing
    among_spacing, side_spacing = spacing

    probed_results = [] 
    
    for i in range(left_n):
        x = side_spacing
        v_size = left_n * r * 2 + among_spacing * (left_n - 1)
        top_margin = (H - v_size) // 2
        y = top_margin + r + i * (among_spacing + r * 2)
        probed_results.append(image[y,x])

    for i in range(right_n):
        x = W - side_spacing
        v_size = right_n * r * 2 + among_spacing * (right_n - 1)
        top_margin = (H - v_size) // 2
        y = top_margin + r + i * (among_spacing + r * 2)
        probed_results.append(image[y,x])

    return torch.stack(probed_results)
