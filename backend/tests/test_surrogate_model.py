import torch
from src.visual_surrogate import SurrogateDeltaV


def test_surrogate_forward_and_grad():
    model = SurrogateDeltaV()
    c_prev = torch.tensor(
        [[0.0, 0.0], [0.5, -0.2]], dtype=torch.float32, requires_grad=True
    )
    c_next = c_prev + 0.01 * torch.randn_like(c_prev)
    d_prev = torch.tensor([0.1, 0.2], dtype=torch.float32)
    grad_prev = torch.zeros((2, 2), dtype=torch.float32)

    pred = model(c_prev, c_next, d_prev, grad_prev)
    assert pred.shape == (2,)
    loss = pred.mean()
    loss.backward()
    assert c_prev.grad is not None
