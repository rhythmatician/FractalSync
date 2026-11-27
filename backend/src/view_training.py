"""Audio-conditioned baseline supervision for the Julia presentation."""

import runtime_core
import torch
from torch.nn import functional as F

VIEW_CHANNELS = (
    "zoomDelta", "rotationDelta", "hueDelta", "chromaDelta",
    "lightnessDelta", "accentDelta", "harmonyShift",
)


def view_training_loss(
    actions: torch.Tensor, seeds: torch.Tensor, segments: torch.Tensor,
    energy: torch.Tensor, timbre: torch.Tensor, flux: torch.Tensor,
) -> torch.Tensor:
    # Exact persistent Rust state for the forward pass. Sample the viewport
    # periodically to penalize zooming into a completely dark Julia interior.
    view = runtime_core.JuliaViewState()
    zooms = []
    previous = None
    rows = zip(actions.detach().cpu().tolist(), segments.reshape(-1).cpu().tolist(), strict=True)
    for row, segment in rows:
        if segment != previous:
            view = runtime_core.JuliaViewState()
            previous = segment
        view.apply_controls(runtime_core.JuliaViewControls(*row))
        zooms.append(view.zoom)
    zoom = actions.new_tensor(zooms)
    # Straight-through local derivative of the Rust log-zoom update.
    zoom = zoom * torch.exp(0.05 * (actions[:, 0] - actions[:, 0].detach()))
    axis = actions.new_tensor([-1.1, -0.55, 0.0, 0.55, 1.1])
    points = torch.complex(axis[:, None].expand(5, 5), axis[None, :].expand(5, 5)).reshape(-1)
    z = points[None, :] / zoom[::8, None]
    farthest = z.abs()
    for _ in range(16):
        z = torch.where(z.abs() < 4, z.square() + seeds[::8, None], z)
        farthest = torch.maximum(farthest, z.abs())
    visible = torch.sigmoid((farthest - 2) * 6).mean(dim=1)
    # Treat audible changes as gesture cues. Quiet passages bias toward
    # wider, readable framing and let color accents settle.
    energy = torch.tanh(energy)
    timbre = torch.tanh(timbre)
    flux = torch.tanh(flux)
    pulse = flux > flux.mean() + 0.75 * flux.std(unbiased=False)
    harmony = torch.where(pulse, torch.where(timbre >= 0, 0.85, -0.85), 0.0)
    targets = torch.stack((
        (0.22 * energy - 0.25 * zoom.detach().log()).clamp(-0.7, 0.7),
        0.35 * timbre, 0.3 * timbre, 0.25 * energy, 0.28 * energy,
        0.4 * flux, harmony,
    ), dim=1)
    return F.mse_loss(actions, targets) + 0.3 * F.relu(0.35 - visible).mean()
