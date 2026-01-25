import torch

from src.visual_losses import MultiscaleDeltaVLoss


def test_budgeted_continuity_penalizes_abrupt_jump():
    loss_fn = MultiscaleDeltaVLoss(weight=1.0)

    # Smooth change: small delta
    prev = torch.zeros((1, 8, 8))
    curr_smooth = torch.full((1, 8, 8), 0.05)

    # Abrupt change: large delta
    curr_jump = torch.full((1, 8, 8), 0.5)

    h = torch.zeros(1)  # low transient

    loss_smooth = loss_fn(prev, curr_smooth, h_t=h, budget_idle=0.02, budget_hit=0.08)
    loss_jump = loss_fn(prev, curr_jump, h_t=h, budget_idle=0.02, budget_hit=0.08)

    assert loss_jump > loss_smooth
