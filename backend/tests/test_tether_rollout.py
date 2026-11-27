import runtime_core
import torch

from src.tether_rollout import tether_rollout


def test_tether_forward_and_seed_gradient_with_segment_reset() -> None:
    phase = torch.arange(100, dtype=torch.float64) * 0.12
    actions = torch.stack(
        (0.7 * phase.cos(), 0.7 * phase.sin(),
         phase.new_full((100,), 0.3), phase.new_full((100,), 0.4)),
        dim=1,
    ).requires_grad_()
    rider = torch.complex(-0.744 + phase * 0.0002, 0.132 + phase * 0.0001)
    segments = torch.arange(100) // 50
    dt = 1024 / 48000
    epsilon = 1e-4

    actual = tether_rollout(rider, actions, segments, dt, epsilon)
    rope = runtime_core.CockpitRope()
    expected = []
    for tick, (c, action) in enumerate(zip(rider.tolist(), actions.tolist(), strict=True)):
        if tick == 50:
            rope.reset()
        rho = runtime_core.tether_rho(c, epsilon)
        expected.append(complex(*rope.step_analog(c.real, c.imag, rho, *action, dt)))
    torch.testing.assert_close(actual, rider.new_tensor(expected), rtol=0, atol=0)

    actual.real.square().mean().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert (actions.grad.abs().sum(0) > 0).all()
