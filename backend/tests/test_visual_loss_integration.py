"""Integration smoke test to ensure visual losses integrate and provide gradients."""

import torch

from src.visual_proxy import ProxyRenderer
from src.visual_losses import MultiscaleDeltaVLoss, SpeedBoundLoss


def test_visual_loss_integration_smoke():
    # Small synthetic batch T=4, N=4
    N = 4
    renderer = ProxyRenderer(resolution=32, max_iter=8, device="cpu")
    c_prev = torch.randn(N, 2, dtype=torch.float32, requires_grad=True)
    # Make small motions
    c_curr = c_prev + 0.02 * torch.randn_like(c_prev)

    # Render frames
    f_prev = renderer.render(c_prev[:, 0], c_prev[:, 1])
    f_curr = renderer.render(c_curr[:, 0], c_curr[:, 1])

    # Compute losses
    deltav = MultiscaleDeltaVLoss(weight=1.0)
    speed = SpeedBoundLoss(weight=1.0, base_step=0.02)

    h_t = torch.rand(N)
    L_v = deltav(f_prev, f_curr, h_t=h_t)
    L_s = speed(c_prev, c_curr, df=None)

    L = L_v + L_s
    assert torch.isfinite(L)
    L.backward()
    # gradient should reach c_prev or c_curr
    assert (c_prev.grad is not None) or (c_curr.grad is not None)
