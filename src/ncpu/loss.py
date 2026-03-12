import torch
from torch.nn import functional as F


def loss_mse_rollout(rollout, out, **kwargs):
    N = min(5, rollout.shape[1])
    nca_outs = rollout[:, -N:, 0]
    out_rep = torch.unsqueeze(out, dim=1).repeat(1, N, 1, 1)
    loss = F.mse_loss(nca_outs, out_rep)
    return loss


def loss_white_black(rollout, out, **kwargs):
    nca_out = rollout[:, -1, 0]

    # print(out.shape, out.dtype, out.min(), out.max())
    white_mask = (out > 0.0).float()
    black_mask = (out <= 0.0).float()

    batch_loss = F.mse_loss(nca_out, out, reduction="none")
    white_loss = batch_loss * white_mask
    black_loss = batch_loss * black_mask

    masks_sum = white_mask.sum() + black_mask.sum()
    white_weight = black_mask.sum() / masks_sum
    black_weight = white_mask.sum() / masks_sum

    loss = (white_loss * white_weight + black_loss * black_weight).mean()
    return loss


def output_masked_rollout_loss(rollout, out, mask, **kwargs):
    """MSE over all rollout steps, restricted to output-circle pixels."""
    B = rollout.shape[0]
    T = rollout.shape[1] - 1
    nca_outs = rollout[:, 1:, 0]
    out_rep = out.unsqueeze(1).expand(B, T, *out.shape[1:])
    return ((nca_outs - out_rep) ** 2 * mask).sum() / (mask.sum() * B * T)


def fullscreen_rollout_loss(rollout, out, inp, **kwargs):
    """MSE over all rollout steps across the entire screen, target includes input."""
    B = rollout.shape[0]
    T = rollout.shape[1] - 1
    nca_outs = rollout[:, 1:, 0]
    out_rep = (out + inp).unsqueeze(1).expand(B, T, *out.shape[1:])
    return F.mse_loss(nca_outs, out_rep)
