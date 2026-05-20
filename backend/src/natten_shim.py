"""Compatibility shim to provide legacy natten function names expected by allin1.

This provides naive (unoptimized) 1D/2D neighborhood attention helpers:
- natten1dqkrpb(query, key, rpb, kernel_size, dilation)
- natten1dav(attention_probs, value, kernel_size, dilation)
- natten2dqkrpb(...)
- natten2dav(...)

These are simple reference implementations using Python loops and torch operations
and are intended only to enable running allin1 in environments where the
installed `natten` package lacks these legacy symbol names.
"""
from __future__ import annotations

import torch


def _safe_dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum().item())


def natten1dqkrpb(query: torch.Tensor, key: torch.Tensor, rpb: torch.Tensor, kernel_size: int, dilation: int):
    """Compute local attention *scores* for 1D neighborhood attention (naive).

    Args:
      query: (B, H, Tq, C)
      key:   (B, H, Tk, C)
      rpb:   (H, 2*kernel_size-1)
      kernel_size: int
      dilation: int

    Returns:
      scores: (B, H, Tq, 2*kernel_size-1)
    """
    B, H, Tq, C = query.shape
    _, _, Tk, _ = key.shape
    win = 2 * kernel_size - 1

    device = query.device
    dtype = query.dtype

    scores = torch.full((B, H, Tq, win), float("-inf"), device=device, dtype=dtype)

    for b in range(B):
        for h in range(H):
            for t in range(Tq):
                qv = query[b, h, t]
                for i in range(win):
                    offset = (i - (kernel_size - 1)) * dilation
                    kpos = t + offset
                    if 0 <= kpos < Tk:
                        kv = key[b, h, kpos]
                        # dot product
                        scores[b, h, t, i] = (qv * kv).sum()
                    else:
                        # very negative so after softmax it's near zero
                        scores[b, h, t, i] = -1e9
    # add relative position bias (rpb shape [H, win])
    # Broadcast over b and t
    if rpb is not None:
        scores = scores + rpb[None, :, None, :]
    return scores


def natten1dav(attention_probs: torch.Tensor, value: torch.Tensor, kernel_size: int, dilation: int):
    """Apply local attention probabilities to value vectors (naive).

    Args:
      attention_probs: (B, H, Tq, win)
      value: (B, H, Tv, C)

    Returns:
      context: (B, H, Tq, C)
    """
    B, H, Tq, win = attention_probs.shape
    _, _, Tv, C = value.shape

    kernel_size = (win + 1) // 2
    context = torch.zeros((B, H, Tq, C), device=attention_probs.device, dtype=attention_probs.dtype)

    for b in range(B):
        for h in range(H):
            for t in range(Tq):
                acc = torch.zeros((C,), device=value.device, dtype=value.dtype)
                for i in range(win):
                    offset = (i - (kernel_size - 1)) * dilation
                    kpos = t + offset
                    if 0 <= kpos < Tv:
                        acc += attention_probs[b, h, t, i] * value[b, h, kpos]
                context[b, h, t] = acc
    return context


# 2D variants: flatten spatial dims and reuse 1D implementations


def natten2dqkrpb(query: torch.Tensor, key: torch.Tensor, rpb: torch.Tensor, kernel_size: int, dilation: int):
    # query/key shapes might be (B, H, X, Y, C) or similar; flatten spatial dims
    if query.ndim == 5:
        B, H, X, Y, C = query.shape
        Tq = X * Y
        q_flat = query.reshape(B, H, Tq, C)
        k_flat = key.reshape(B, H, Tq, C)
        # rpb for 2d expected to be 2D; approximate by summing along one axis
        if rpb is not None:
            rpb_flat = rpb.reshape(H, -1).mean(dim=1, keepdim=True)  # fallback
        else:
            rpb_flat = torch.zeros((H, 1), device=query.device, dtype=query.dtype)
        return natten1dqkrpb(q_flat, k_flat, rpb_flat, kernel_size, dilation).reshape(B, H, X, Y, -1)
    # fallback
    return natten1dqkrpb(query, key, rpb, kernel_size, dilation)


def natten2dav(attention_probs: torch.Tensor, value: torch.Tensor, kernel_size: int, dilation: int):
    if attention_probs.ndim == 5:
        B, H, X, Y, win = attention_probs.shape
        Tq = X * Y
        ap_flat = attention_probs.reshape(B, H, Tq, win)
        val_flat = value.reshape(B, H, Tq, value.shape[-1])
        ctx = natten1dav(ap_flat, val_flat, kernel_size, dilation)
        return ctx.reshape(B, H, X, Y, -1)
    else:
        return natten1dav(attention_probs, value, kernel_size, dilation)
