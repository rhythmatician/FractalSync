import torch

from src.view_training import view_training_loss


def test_audio_and_julia_visibility_supervise_all_view_outputs() -> None:
    actions = torch.zeros(32, 7, requires_grad=True)
    seeds = torch.full((32,), complex(-0.74, 0.13), dtype=torch.complex64)
    segments = torch.arange(32) // 16
    energy = torch.linspace(-1, 1, 32)
    loss = view_training_loss(actions, seeds, segments, energy, energy.flip(0), energy)
    loss.backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert (actions.grad.abs().sum(0) > 0).all()
