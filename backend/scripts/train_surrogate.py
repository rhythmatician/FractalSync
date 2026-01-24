"""Train a SurrogateDeltaV model on a generated dataset.

Usage:
  python scripts/train_surrogate.py --data data/surrogate/samples_small.pt --out models/surrogate.pt --epochs 20
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.visual_surrogate import SurrogateDeltaV, SurrogateDataset


def main():
    parser = argparse.ArgumentParser(description="Train surrogate ΔV model")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    ds = SurrogateDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device)
    model = SurrogateDeltaV().to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        t0 = time.time()
        for batch in loader:
            c_prev = batch["c_prev"].to(device)
            c_next = batch["c_next"].to(device)
            d_prev = batch["d_prev"].to(device)
            grad_prev = batch["grad_prev"].to(device)
            y = batch["delta_v"].to(device)

            pred = model(c_prev, c_next, d_prev, grad_prev)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * c_prev.shape[0]
        epoch_loss = epoch_loss / len(ds)
        print(f"Epoch {epoch+1}/{args.epochs}: loss={epoch_loss:.6f} time={time.time()-t0:.2f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    SurrogateDeltaV.save_checkpoint(model, args.out)
    print(f"Saved surrogate model to: {args.out}")


if __name__ == "__main__":
    main()
