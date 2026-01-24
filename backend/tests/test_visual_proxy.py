import torch

from src.visual_proxy import ProxyRenderer, frame_diff


def test_proxy_render_shape_and_range():
    renderer = ProxyRenderer(resolution=32, max_iter=8, device="cpu")
    c_real = torch.tensor([0.0, -0.7], dtype=torch.float32)
    c_imag = torch.tensor([0.0, 0.2], dtype=torch.float32)
    frames = renderer.render(c_real, c_imag)
    assert frames.shape == (2, 32, 32)
    assert torch.isfinite(frames).all()
    assert frames.min() >= -1e-6 and frames.max() <= 1.0 + 1e-6


def test_frame_diff_and_autograd():
    renderer = ProxyRenderer(resolution=16, max_iter=6, device="cpu")
    c1 = torch.tensor([0.0, -0.3], dtype=torch.float32, requires_grad=True)
    c2 = torch.tensor([0.01, -0.29], dtype=torch.float32, requires_grad=True)
    f1 = renderer.render(
        c1, c2 * 0 + 0.0
    )  # intentionally misuse shapes to get variation
    f2 = renderer.render(c1 + 0.02, c2 + 0.0)
    diff = frame_diff(f1, f2)
    loss = diff.mean()
    loss.backward()
    assert c1.grad is not None or c2.grad is not None
