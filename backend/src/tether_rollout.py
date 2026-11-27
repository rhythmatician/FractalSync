"""Exact Rust tether rollout with a local finite-difference training gradient."""

import runtime_core
import torch


def tether_rollout(
    rider: torch.Tensor, actions: torch.Tensor, segments: torch.Tensor,
    dt: float, epsilon: float,
) -> torch.Tensor:
    rope = runtime_core.CockpitRope()
    rider_values = rider.detach().cpu().tolist()
    action_values = actions.detach().cpu().tolist()
    segment_values = segments.reshape(-1).detach().cpu().tolist()
    seeds = []
    derivatives = []
    previous = None
    for c, action, segment in zip(rider_values, action_values, segment_values, strict=True):
        if segment != previous:
            rope.reset()
            previous = segment
        rho = runtime_core.tether_rho(c, epsilon)
        if torch.is_grad_enabled() and actions.requires_grad:
            local = []
            for channel in range(4):
                plus, minus = action.copy(), action.copy()
                plus[channel] += 1e-3
                minus[channel] -= 1e-3
                p = complex(*rope.fork().step_analog(c.real, c.imag, rho, *plus, dt))
                m = complex(*rope.fork().step_analog(c.real, c.imag, rho, *minus, dt))
                local.append((p - m) / 2e-3)
            derivatives.append(local)
        seeds.append(complex(*rope.step_analog(c.real, c.imag, rho, *action, dt)))
    exact = rider.new_tensor(seeds)
    if derivatives:
        # Hold the incoming tether state fixed for this local Jacobian.
        # The rider gradient treats the current tether offset as fixed.
        exact = exact + ((actions - actions.detach()) * rider.new_tensor(derivatives)).sum(-1)
    return exact + (rider - rider.detach())
