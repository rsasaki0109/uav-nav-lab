"""LSTM peer-motion predictor.

Maintains a sliding window of past positions per tracked obstacle and
feeds them into a small LSTM to forecast future positions at the
requested horizon_dts. Uses greedy nearest-neighbor track association
(matching the kalman_velocity predictor interface).

Train with `scripts/train_lstm_predictor.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from .base import PREDICTOR_REGISTRY, Predictor


class _LSTMForecaster(nn.Module):
    """Small LSTM that maps (seq_len, input_dim) → (horizon, output_dim)."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_dim: int = 2,
        horizon: int = 30,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim * horizon)
        self.horizon = horizon
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        _, (h_n, _) = self.lstm(x)
        # h_n: [num_layers, batch, hidden_dim]
        out = self.fc(h_n[-1])  # [batch, output_dim * horizon]
        return out.view(-1, self.horizon, self.output_dim)


@PREDICTOR_REGISTRY.register("lstm")
class LSTMPredictor(Predictor):
    def __init__(
        self,
        dt: float = 0.2,
        association_threshold: float = 3.0,
        history_len: int = 10,
        model_path: str = "",
        device: str = "cpu",
    ) -> None:
        self.dt = float(dt)
        self.assoc_thr = float(association_threshold)
        self.history_len = int(history_len)
        self._tracks: list[dict[str, Any]] = []
        self._device = torch.device(device)

        self._model: _LSTMForecaster | None = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=False)
        cfg = state.get("config", {})
        self._model_config = cfg
        self._model = _LSTMForecaster(
            input_dim=cfg.get("input_dim", 2),
            hidden_dim=cfg.get("hidden_dim", 32),
            num_layers=cfg.get("num_layers", 2),
            output_dim=cfg.get("output_dim", 2),
            horizon=cfg.get("horizon", 30),
        ).to(self._device)
        self._model.load_state_dict(state["state_dict"])
        self._model.eval()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "LSTMPredictor":
        return cls(
            dt=float(cfg.get("dt", 0.2)),
            association_threshold=float(cfg.get("association_threshold", 3.0)),
            history_len=int(cfg.get("history_len", 10)),
            model_path=str(cfg.get("model_path", "")),
            device=str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        )

    def reset(self, *, seed: int | None = None) -> None:
        self._tracks = []

    def _init_track(self, pos: np.ndarray, ndim: int) -> dict[str, Any]:
        return {"history": [pos.copy()], "ndim": ndim}

    def _associate(self, dyn_obs: list[dict], ndim: int) -> list[int | None]:
        used: set[int] = set()
        assignments: list[int | None] = []
        for obs in dyn_obs:
            p = np.asarray(obs["position"], dtype=float)[:ndim]
            best_i: int | None = None
            best_d2 = np.inf
            for i, t in enumerate(self._tracks):
                if i in used or t["ndim"] != ndim:
                    continue
                last_pos = t["history"][-1][:ndim]
                d2 = float(np.sum((last_pos - p) ** 2))
                if d2 < best_d2 and d2 <= self.assoc_thr * self.assoc_thr:
                    best_i = i
                    best_d2 = d2
            assignments.append(best_i)
            if best_i is not None:
                used.add(best_i)
        return assignments

    def predict(
        self,
        dynamic_obstacles: list[dict],
        horizon_dts: np.ndarray,
    ) -> np.ndarray:
        if not dynamic_obstacles:
            return np.zeros((0, len(horizon_dts), 0), dtype=float)
        ndim = len(dynamic_obstacles[0]["position"])
        H = len(horizon_dts)

        # Associate observations to existing tracks
        assignments = self._associate(dynamic_obstacles, ndim)

        # Update tracks: add new observation, keep sliding window
        new_tracks: list[dict[str, Any]] = []
        for obs, track_idx in zip(dynamic_obstacles, assignments):
            p = np.asarray(obs["position"], dtype=float)[:ndim]
            if track_idx is not None:
                t = self._tracks[track_idx]
                t["history"].append(p)
                if len(t["history"]) > self.history_len:
                    t["history"] = t["history"][-self.history_len:]
                new_tracks.append(t)
            else:
                new_tracks.append(self._init_track(p, ndim))
        self._tracks = new_tracks

        # Predict using LSTM model, fall back to CV if no model loaded
        out = np.empty((len(self._tracks), H, ndim), dtype=float)
        if self._model is not None:
            cfg = getattr(self, "_model_config", {})
            relative = cfg.get("relative", False)
            for k, t in enumerate(self._tracks):
                hist = t["history"]
                if len(hist) < self.history_len:
                    pad = [hist[0]] * (self.history_len - len(hist))
                    hist = pad + hist
                inp = np.array(hist[-self.history_len:], dtype=np.float32)
                inp_t = torch.as_tensor(inp, dtype=torch.float32, device=self._device).unsqueeze(0)
                with torch.no_grad():
                    pred = self._model(inp_t).squeeze(0).cpu().numpy()  # [horizon, ndim]
                if relative:
                    # pred[j] is displacement from last history position
                    last_pos = hist[-1][:ndim]
                    p0 = last_pos[None, :] + pred  # [horizon, ndim]
                    model_dts = np.arange(1, self._model.horizon + 1) * self.dt
                    for d in range(ndim):
                        out[k, :, d] = np.interp(horizon_dts, model_dts, p0[:, d])
                else:
                    model_dts = np.arange(1, self._model.horizon + 1) * self.dt
                    for d in range(min(ndim, pred.shape[1])):
                        out[k, :, d] = np.interp(
                            horizon_dts, model_dts, pred[: self._model.horizon, d]
                        )
        else:
            # Fallback: constant velocity
            dts = np.asarray(horizon_dts, dtype=float)
            for k, t in enumerate(self._tracks):
                if len(t["history"]) < 2:
                    out[k] = t["history"][-1][None, :]
                else:
                    p0 = t["history"][-1]
                    v = (t["history"][-1] - t["history"][-2]) / self.dt
                    out[k] = p0[None, :] + dts[:, None] * v[None, :]
        return out
