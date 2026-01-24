import torch

from src.visual_surrogate import SurrogateDeltaV, SurrogateDataset


def test_surrogate_pipeline_smoke(tmp_path):
    # Build tiny synthetic dataset
    N = 200
    c_prev = torch.randn(N, 2)
    c_next = c_prev + 0.02 * torch.randn_like(c_prev)
    d_prev = torch.rand(N)
    grad_prev = torch.randn(N, 2) * 0.01
    delta_v = torch.rand(N) * 0.1

    dp = tmp_path / "samples.pt"
    torch.save({"c_prev": c_prev, "c_next": c_next, "d_prev": d_prev, "grad_prev": grad_prev, "delta_v": delta_v}, str(dp))

    ds = SurrogateDataset(str(dp))
    model = SurrogateDeltaV()
    # Single training step
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = ds[0]
    pred = model(batch["c_prev"].unsqueeze(0), batch["c_next"].unsqueeze(0), batch["d_prev"].unsqueeze(0), batch["grad_prev"].unsqueeze(0))
    loss = ((pred - batch["delta_v"].unsqueeze(0)) ** 2).mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(pred).all()
