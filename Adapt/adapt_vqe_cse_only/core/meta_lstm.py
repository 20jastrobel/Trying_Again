"""Torch-based coordinate-wise LSTM meta-optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:  # optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - handled by caller
    torch = None
    nn = None


@dataclass
class MetaLSTMConfig:
    hidden_size: int = 20
    input_size: int = 2
    r: float = 10.0


if nn is None:  # pragma: no cover - optional dependency
    class CoordinateWiseLSTM:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to use CoordinateWiseLSTM")
else:
    class CoordinateWiseLSTM(nn.Module):
        """Coordinate-wise LSTM cell with shared weights across coordinates."""

        def __init__(self, *, hidden_size: int = 20, input_size: int = 2) -> None:
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.input_size = int(input_size)
            self.lstm = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
            self.head = nn.Linear(self.hidden_size, 1)

        def forward(
            self,
            x,
            state: Tuple[torch.Tensor, torch.Tensor] | None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            if state is None:
                h = torch.zeros((x.shape[0], self.hidden_size), device=x.device)
                c = torch.zeros((x.shape[0], self.hidden_size), device=x.device)
            else:
                h, c = state
            h, c = self.lstm(x, (h, c))
            delta = self.head(h).squeeze(-1)
            return delta, (h, c)


def preprocess_gradients_torch(
    grad: np.ndarray | list[float],
    *,
    r: float = 10.0,
    device=None,
) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("Torch is required to preprocess gradients for the meta-optimizer")

    grad_arr = np.asarray(grad, dtype=np.float32)
    if grad_arr.size == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=device)

    abs_grad = np.abs(grad_arr)
    threshold = float(np.exp(-r))
    x = np.zeros((grad_arr.size, 2), dtype=np.float32)

    mask = abs_grad >= threshold
    if np.any(mask):
        x[mask, 0] = np.log(abs_grad[mask]) / r
        x[mask, 1] = np.sign(grad_arr[mask])
    if np.any(~mask):
        x[~mask, 0] = -1.0
        x[~mask, 1] = np.exp(r) * grad_arr[~mask]

    return torch.tensor(x, dtype=torch.float32, device=device)


def load_meta_lstm(path: str, *, device=None) -> tuple[CoordinateWiseLSTM, dict]:
    if torch is None:
        raise RuntimeError("Torch is required to load meta-optimizer weights")
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    hidden_size = int(config.get("hidden_size", 20))
    input_size = int(config.get("input_size", 2))
    model = CoordinateWiseLSTM(hidden_size=hidden_size, input_size=input_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device=device)
    return model, config


def save_meta_lstm(path: str, model: CoordinateWiseLSTM, config: dict) -> None:
    if torch is None:
        raise RuntimeError("Torch is required to save meta-optimizer weights")
    payload = {
        "state_dict": model.state_dict(),
        "config": config,
    }
    torch.save(payload, path)
