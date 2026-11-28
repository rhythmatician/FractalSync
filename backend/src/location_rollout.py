"""Causal observations for training and replay of location-aware policies."""

import torch
import runtime_core

from .control_model import AudioToControlModel


def location_conditioned_rollout(
    model: AudioToControlModel,
    features: torch.Tensor,
    segment_ids: torch.Tensor,
    initial_c: torch.Tensor,
) -> tuple[torch.Tensor, list[complex]]:
    """Read each pre-action state, then advance the canonical Rust integrator."""
    segments = segment_ids.reshape(-1).tolist()
    starts = initial_c.detach().tolist()
    config = runtime_core.ManifoldConfig()
    dt = runtime_core.HOP_LENGTH / runtime_core.SAMPLE_RATE
    outputs: list[torch.Tensor] = []
    positions: list[complex] = []
    re = im = vx = vy = 0.0
    for index, segment in enumerate(segments):
        if index == 0 or segment != segments[index - 1]:
            re, im = starts[index].real, starts[index].imag
            vx = vy = 0.0
        observation = runtime_core.player_location_observation(re, im, vx, vy)
        output = model(features[index:index + 1], features.new_tensor([observation]))
        outputs.append(output)
        parsed = model.parse_output(output.detach())
        motion = runtime_core.MotionControls(
            direction_x=parsed["directionX"].item(),
            direction_y=parsed["directionY"].item(),
            throttle=parsed["throttle"].item(),
            brake=parsed["brake"].item(),
            grip=parsed["grip"].item(),
            impulse=parsed["impulse"].item(),
        )
        re, im, vx, vy, _ = runtime_core.controls_integrate_step(re, im, vx, vy, motion, dt, config)
        positions.append(complex(re, im))
    return torch.cat(outputs, dim=0), positions
