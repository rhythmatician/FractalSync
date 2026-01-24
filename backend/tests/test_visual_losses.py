import torch

from src.visual_proxy import ProxyRenderer
from src.visual_losses import MultiscaleDeltaVLoss, SpeedBoundLoss


def test_multiscale_deltav_basic():
    renderer = ProxyRenderer(resolution=16, max_iter=6, device="cpu")
    c_real = torch.tensor([0.0, 0.1], dtype=torch.float32, requires_grad=True)
    c_imag = torch.tensor([0.0, -0.1], dtype=torch.float32, requires_grad=True)
    f1 = renderer.render(c_real, c_imag)
    f2 = renderer.render(c_real + 0.01, c_imag + 0.02)
    loss_fn = MultiscaleDeltaVLoss(weight=1.0, scales=[1, 2])
    h_t = torch.tensor([0.0, 0.5], dtype=torch.float32)
    loss = loss_fn(f1, f2, h_t=h_t)
    assert torch.isfinite(loss)
    loss.backward()


def test_speedbound_loss_grad():
    c_prev = torch.tensor(
        [[0.0, 0.0], [0.5, 0.5]], dtype=torch.float32, requires_grad=True
    )
    c_curr = torch.tensor(
        [[0.1, 0.0], [0.6, 0.5]], dtype=torch.float32, requires_grad=True
    )
    loss_fn = SpeedBoundLoss(weight=1.0, base_step=0.05)
    loss = loss_fn(c_prev, c_curr, df=None)
    assert torch.isfinite(loss)
    loss.backward()
    assert (c_prev.grad is not None) or (c_curr.grad is not None)


def test_hit_alignment_basic():
    from src.visual_losses import HitAlignmentLoss

    be = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=torch.float32)
    gp_good = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.45, 0.45]], dtype=torch.float32)
    gp_bad = torch.tensor([[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]], dtype=torch.float32)

    loss_fn = HitAlignmentLoss(weight=1.0)
    L_good = loss_fn(be, gp_good)
    L_bad = loss_fn(be, gp_bad)
    assert L_good < L_bad


def test_coverage_loss_basic():
    from src.visual_losses import CoverageLoss

    # N=1, T=4 concentrated at angle 0
    c_seq_bad = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [1.1, 0.0], [0.9, 0.0]]], dtype=torch.float32
    )
    # spread across quadrants
    c_seq_good = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]], dtype=torch.float32
    )
    loss_fn = CoverageLoss(weight=1.0, bins=8)
    L_bad = loss_fn(c_seq_bad)
    L_good = loss_fn(c_seq_good)
    assert L_good < L_bad
