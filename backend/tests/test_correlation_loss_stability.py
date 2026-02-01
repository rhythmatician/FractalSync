import torch
from backend.src.control_trainer import CorrelationLoss


def test_correlation_loss_constant_inputs_returns_zero_and_grad_finite():
    loss = CorrelationLoss()
    x = torch.ones(16, dtype=torch.float32, requires_grad=True)
    y = torch.ones(16, dtype=torch.float32)

    out = loss(x, y)
    assert float(out) == 0.0

    # Should be safe to backprop (no NaNs in gradient) — gradient should be zero
    x_grad = torch.autograd.grad(out, x, allow_unused=True)[0]
    assert x_grad is not None
    assert torch.all(torch.isfinite(x_grad)).item()


def test_correlation_loss_near_constant_has_finite_gradients():
    loss = CorrelationLoss()
    x = torch.ones(16, dtype=torch.float32, requires_grad=True) * 1e-9
    y = torch.ones(16, dtype=torch.float32) * 2.0

    out = loss(x, y)
    # either zero or tiny; should not be NaN or inf
    assert torch.isfinite(out).item()
    x_grad = torch.autograd.grad(out, x, allow_unused=True)[0]
    assert x_grad is not None
    assert torch.all(torch.isfinite(x_grad)).item()
