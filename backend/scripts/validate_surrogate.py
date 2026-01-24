"""Validate surrogate model predictions against held-out samples."""

from pathlib import Path
import random

import torch
import numpy as np
from src.visual_surrogate import SurrogateDeltaV


def main():
    data_path = Path("data/surrogate/samples_10k.pt")
    model_path = Path("models/surrogate_initial.pt")

    assert data_path.exists(), f"Data not found: {data_path}"
    assert model_path.exists(), f"Model not found: {model_path}"

    data = torch.load(str(data_path))
    c_prev = data["c_prev"]
    c_next = data["c_next"]
    d_prev = data["d_prev"]
    grad_prev = data["grad_prev"]
    delta_v = data["delta_v"]

    N = len(c_prev)
    idx = list(range(N))
    random.shuffle(idx)
    val_idx = idx[:1000]

    c_prev_v = c_prev[val_idx]
    c_next_v = c_next[val_idx]
    d_prev_v = d_prev[val_idx]
    grad_prev_v = grad_prev[val_idx]
    y_true = delta_v[val_idx]

    model = SurrogateDeltaV.load_checkpoint(str(model_path), device="cpu")
    model.eval()

    with torch.no_grad():
        y_pred = model.predict(c_prev_v, c_next_v, d_prev_v, grad_prev_v)

    y_true = y_true.numpy()
    y_pred = y_pred.cpu().numpy()

    mae = float(np.mean(np.abs(y_pred - y_true)))
    mse = float(np.mean((y_pred - y_true) ** 2))
    corr = float(np.corrcoef(y_pred, y_true)[0, 1])

    print(
        f"Validation - N={len(y_true)}: MAE={mae:.6f}, MSE={mse:.6f}, Corr={corr:.4f}"
    )


if __name__ == "__main__":
    main()
