#!/usr/bin/env python3
"""Train an LSTM peer-motion predictor on synthetic trajectory data.

Generates random-walk trajectories, slices them into (history → future)
pairs, and trains a small LSTM to forecast future positions.

Usage:
    python3 scripts/train_lstm_predictor.py --epochs 50 --output lstm_predictor.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uav_nav_lab.predictor.lstm import _LSTMForecaster


def generate_trajectories(
    n_trajectories: int = 500,
    traj_len: int = 100,
    ndim: int = 2,
    dt: float = 0.2,
    max_speed: float = 5.0,
    accel_std: float = 1.0,
) -> np.ndarray:
    """Generate random-walk trajectories with bounded acceleration.

    Returns: [n_trajectories, traj_len, ndim]
    """
    trajs = np.zeros((n_trajectories, traj_len, ndim), dtype=np.float32)
    vel = np.random.randn(n_trajectories, ndim).astype(np.float32) * 0.5
    for t in range(1, traj_len):
        accel = np.random.randn(n_trajectories, ndim).astype(np.float32) * accel_std
        vel = vel + accel * dt
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = np.where(speed > max_speed, vel / speed * max_speed, vel)
        trajs[:, t] = trajs[:, t - 1] + vel * dt
    return trajs


def build_dataset(
    trajs: np.ndarray, history_len: int = 10, future_len: int = 30
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice trajectories into (history, future) pairs."""
    X, Y = [], []
    for traj in trajs:
        for i in range(len(traj) - history_len - future_len):
            X.append(traj[i : i + history_len])
            Y.append(traj[i + history_len : i + history_len + future_len])
    return torch.as_tensor(np.array(X), dtype=torch.float32), torch.as_tensor(
        np.array(Y), dtype=torch.float32
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--future-len", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="results/lstm_predictor.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Generate synthetic training data
    print("Generating training trajectories...")
    trajs = generate_trajectories(n_trajectories=500, traj_len=150)
    X, Y = build_dataset(trajs, args.history_len, args.future_len)
    print(f"Dataset: {X.shape[0]} samples, input {X.shape[1]}×{X.shape[2]}, output {Y.shape[1]}×{Y.shape[2]}")

    # Train/val split
    n_train = int(0.8 * len(X))
    train_ds = TensorDataset(X[:n_train], Y[:n_train])
    val_ds = TensorDataset(X[n_train:], Y[n_train:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Model
    ndim = X.shape[2]
    model = _LSTMForecaster(
        input_dim=ndim, hidden_dim=args.hidden_dim, num_layers=2,
        output_dim=ndim, horizon=args.future_len,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"Training {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X[n_train:].to(device))
            val_loss = loss_fn(val_pred, Y[n_train:].to(device)).item()

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")

    # Save
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "input_dim": ndim,
                "hidden_dim": args.hidden_dim,
                "num_layers": 2,
                "output_dim": ndim,
                "horizon": args.future_len,
            },
        },
        str(out_path),
    )
    print(f"Model saved to {out_path}")

    # Final eval: mean position error at different horizons
    model.eval()
    with torch.no_grad():
        pred = model(Y[n_train:].to(device))  # actually predict on val set
        val_x = X[n_train:].to(device)
        val_y = Y[n_train:].to(device)
        pred_val = model(val_x)
        errors = torch.norm(pred_val - val_y, dim=-1).mean(dim=0).cpu().numpy()
        print("\nPosition error by horizon step (val set):")
        for i in [0, 4, 9, 14, 19, 24, 29]:
            if i < len(errors):
                dt_step = (i + 1) * 0.2
                print(f"  horizon {i+1:2d} ({dt_step:.1f}s): {errors[i]:.3f}m")


if __name__ == "__main__":
    main()
