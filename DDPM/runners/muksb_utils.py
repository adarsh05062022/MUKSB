"""
runners/muksb_utils.py
======================
Kalai-Smorodinsky (KS) bargaining helpers used by ``Diffusion.muksb_unlearn``.

This is a direct port of the KS core from
``MUKSB/SD/MUKSB_cls_magnitude.py`` and
``MUKSB/Classification/unlearn/MUKSB.py``, kept identical so behaviour and
diagnostics match across SD / classification / DDPM experiments.

Improvements over the vanilla KS bisector
-----------------------------------------
  Fix 2 — Harmonic-scale step size (always active)
      s_KS = 2 * ||g_r|| * ||g_f|| / (||g_r|| + ||g_f||)
      Curvature-aware scale; replaces the unjustified ``lambda_ks * g_star``.

  Fix 3 — Asymmetric priority (gamma)
      gamma = 0.5  -> symmetric KS (default, equal proportional gain)
      gamma > 0.5  -> retain-favoured bisector
      gamma < 0.5  -> forget-favoured bisector

  Fix 4 — Asymmetric loss scales handled via gamma (see Fix 3).

Edge case (anti-parallel gradients): ``g_star`` is zero; the caller should
skip the optimiser step.
"""

import torch


def ks_step(
    gr_flat: torch.Tensor,
    gf_flat: torch.Tensor,
    gamma: float = 0.5,
    eps: float = 1e-8,
):
    """
    Kalai-Smorodinsky bargaining solution for gradient merging.

    Parameters
    ----------
    gr_flat : Tensor, shape (D,)   - flattened retain gradient
    gf_flat : Tensor, shape (D,)   - flattened forget gradient
    gamma   : float                - retain directional priority in [0, 1]
    eps     : float                - numerical stability floor

    Returns
    -------
    lambda_ks       : Tensor (scalar) - diagnostic cos(phi/2)
    cos_phi         : Tensor (scalar) - cosine between the two gradients
    g_star          : Tensor (D,)     - KS-merged unit direction (zero if anti-parallel)
    effective_scale : Tensor (scalar) - harmonic mean of gradient norms
    """
    norm_gr = torch.clamp(torch.norm(gr_flat), min=1e-6)
    norm_gf = torch.clamp(torch.norm(gf_flat), min=1e-6)

    cos_phi = torch.clamp(
        torch.dot(gr_flat, gf_flat) / (norm_gr * norm_gf),
        -1.0 + eps,
        1.0 - eps,
    )

    g_hat_r = gr_flat / norm_gr
    g_hat_f = gf_flat / norm_gf

    g_sum = gamma * g_hat_r + (1.0 - gamma) * g_hat_f
    norm_sum = torch.norm(g_sum)

    if norm_sum < 1e-6:
        zero = torch.zeros_like(gr_flat)
        return (
            torch.tensor(0.0, device=gr_flat.device),
            cos_phi,
            zero,
            torch.tensor(0.0, device=gr_flat.device),
        )

    g_star = g_sum / norm_sum
    effective_scale = 2.0 * norm_gr * norm_gf / (norm_gr + norm_gf)
    lambda_ks = torch.dot(g_hat_r, g_star)

    return lambda_ks, cos_phi, g_star, effective_scale


def flatten_grads(parameters, grads):
    parts = []
    for p, g in zip(parameters, grads):
        parts.append(
            g.detach().reshape(-1)
            if g is not None
            else torch.zeros(p.numel(), device=p.device)
        )
    return torch.cat(parts)


def unpack_to_grads(parameters, flat_vec):
    offset = 0
    for p in parameters:
        n = p.numel()
        chunk = flat_vec[offset:offset + n].view_as(p)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += n


def build_flat_mask(named_parameters, mask_dict, device):
    """
    Build a flat boolean mask aligned with ``flatten_grads(parameters, ...)``.

    Param names in ``mask_dict`` must match the names returned by
    ``model.named_parameters()``. Any parameter missing from ``mask_dict``
    is treated as fully active (ones), matching the behaviour expected by
    SalUn-style sparse masks where only a subset of params get masks.
    """
    parts = []
    for name, p in named_parameters:
        if name in mask_dict:
            parts.append(mask_dict[name].to(device).reshape(-1).bool())
        else:
            parts.append(torch.ones(p.numel(), dtype=torch.bool, device=device))
    return torch.cat(parts)
