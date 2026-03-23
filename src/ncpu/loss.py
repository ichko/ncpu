import torch
import torch.nn.functional as F


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


def _dilate_mask(mask, radius):
    """Binary dilation of a (H, W) mask via max_pool2d."""
    k = 2 * radius + 1
    return (
        F.max_pool2d(
            mask.unsqueeze(0).unsqueeze(0), kernel_size=k, stride=1, padding=radius
        )
        .squeeze(0)
        .squeeze(0)
        .clamp(0, 1)
    )


def io_rollout_background_last_loss(
    rollout, out, mask, inp_mask, inp, dilation_radius=2, **kwargs
):
    """
    - Input + output circles: supervised on every rollout step.
    - Dilated halo around io circles: supervised on last step only.

    Args:
        mask:             (H, W) — output circle pixels
        inp_mask:         (H, W) — input circle pixels
        inp:              (B, H, W) — normalized input screen
        dilation_radius:  pixels to dilate the io mask for the background halo
    """
    B = rollout.shape[0]
    T = rollout.shape[1] - 1
    nca_outs = rollout[:, 1:, 0]  # (B, T, H, W)
    last = rollout[:, -1, 0]  # (B, H, W)

    io_mask = (mask + inp_mask).clamp(0, 1)
    halo_mask = (
        _dilate_mask(io_mask, dilation_radius) - io_mask
    )  # ring around io, not io itself

    # target: input pixels from inp, output pixels from out
    target = out * mask + inp * inp_mask  # (B, H, W)

    # io loss over all steps
    target_rep = target.unsqueeze(1).expand(B, T, *out.shape[1:])
    io_loss = ((nca_outs - target_rep) ** 2 * io_mask).sum() / (io_mask.sum() * B * T)

    # halo loss on last frame only
    halo_loss = ((last - target) ** 2 * halo_mask).sum() / (halo_mask.sum() * B)

    return io_loss


def output_rollout_input_last_loss(rollout, out, mask, inp_mask, inp, **kwargs):
    """
    - Output circle pixels: supervised on every rollout step.
    - Input circle pixels: supervised on last frame only.
    """
    B = rollout.shape[0]
    T = rollout.shape[1] - 1
    nca_outs = rollout[:, 1:, 0]  # (B, T, H, W)
    last = rollout[:, -1, 0]  # (B, H, W)

    out_rep = out.unsqueeze(1).expand(B, T, *out.shape[1:])
    out_loss = ((nca_outs - out_rep) ** 2 * mask).sum() / (mask.sum() * B * T)
    inp_loss = ((last - inp) ** 2 * inp_mask).sum() / (inp_mask.sum() * B)

    return out_loss + inp_loss


def output_halo_rollout_loss(rollout, out, mask, dilation_radius=2, **kwargs):
    """MSE over all rollout steps at output circles + a small dilated halo."""
    B = rollout.shape[0]
    T = rollout.shape[1] - 1
    nca_outs = rollout[:, 1:, 0]
    out_rep = out.unsqueeze(1).expand(B, T, *out.shape[1:])
    halo_mask = _dilate_mask(mask, dilation_radius)
    return ((nca_outs - out_rep) ** 2 * halo_mask).sum() / (halo_mask.sum() * B * T)


def output_masked_last_loss(rollout, out, mask, **kwargs):
    """MSE on the last rollout frame, restricted to output-circle pixels."""
    B = rollout.shape[0]
    last = rollout[:, -1, 0]
    return ((last - out) ** 2 * mask).sum() / (mask.sum() * B)


def fullscreen_rollout_loss(rollout, out, **kwargs):
    """MSE over all rollout steps across the entire screen."""
    B = rollout.shape[0]
    T = rollout.shape[1] - 1
    nca_outs = rollout[:, 1:, 0]
    out_rep = out.unsqueeze(1).expand(B, T, *out.shape[1:])
    return F.mse_loss(nca_outs, out_rep)
