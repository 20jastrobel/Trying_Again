"""Coordinate-wise LSTM meta-optimizer utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


Array = np.ndarray


def preprocess_gradients(grad: Iterable[float], *, r: float = 10.0) -> Array:
    """Preprocess gradients for the LSTM optimizer.

    Returns an array of shape (n, 2) with the per-coordinate features.
    """
    grad_arr = np.asarray(list(grad), dtype=float)
    if grad_arr.size == 0:
        return np.zeros((0, 2), dtype=float)

    abs_grad = np.abs(grad_arr)
    threshold = np.exp(-r)
    x = np.zeros((grad_arr.size, 2), dtype=float)

    mask = abs_grad >= threshold
    if np.any(mask):
        x[mask, 0] = np.log(abs_grad[mask]) / r
        x[mask, 1] = np.sign(grad_arr[mask])
    if np.any(~mask):
        x[~mask, 0] = -1.0
        x[~mask, 1] = np.exp(r) * grad_arr[~mask]

    return x


@dataclass
class LSTMState:
    h: Array
    c: Array


class CoordinateWiseLSTMOptimizer:
    """Coordinate-wise LSTM meta-optimizer with shared weights.

    The LSTM weights are shared across coordinates, but each coordinate
    maintains its own hidden and cell state.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 20,
        input_size: int = 2,
        seed: int | None = None,
        weights: dict[str, Array] | None = None,
        weight_scale: float = 0.1,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)

        if weights is not None:
            self._load_weights(weights)
        else:
            rng = np.random.default_rng(seed)
            self.W_x = rng.normal(scale=weight_scale, size=(4 * self.hidden_size, self.input_size))
            self.W_h = rng.normal(scale=weight_scale, size=(4 * self.hidden_size, self.hidden_size))
            self.b = np.zeros((4 * self.hidden_size,), dtype=float)
            self.W_out = rng.normal(scale=weight_scale, size=(self.hidden_size,))
            self.b_out = 0.0

    def _load_weights(self, weights: dict[str, Array]) -> None:
        self.W_x = np.asarray(weights["W_x"], dtype=float)
        self.W_h = np.asarray(weights["W_h"], dtype=float)
        self.b = np.asarray(weights["b"], dtype=float)
        self.W_out = np.asarray(weights["W_out"], dtype=float)
        self.b_out = float(weights["b_out"])

        self.hidden_size = int(self.W_h.shape[1])
        self.input_size = int(self.W_x.shape[1])

    def to_weights(self) -> dict[str, Array]:
        return {
            "W_x": self.W_x.copy(),
            "W_h": self.W_h.copy(),
            "b": self.b.copy(),
            "W_out": self.W_out.copy(),
            "b_out": np.array(self.b_out, dtype=float),
        }

    def init_state(self, n_coords: int) -> LSTMState:
        return LSTMState(
            h=np.zeros((n_coords, self.hidden_size), dtype=float),
            c=np.zeros((n_coords, self.hidden_size), dtype=float),
        )

    def step(self, grad: Iterable[float], state: LSTMState, *, r: float = 10.0) -> tuple[Array, LSTMState]:
        grad_arr = np.asarray(list(grad), dtype=float)
        if grad_arr.size == 0:
            return np.zeros((0,), dtype=float), state

        x = preprocess_gradients(grad_arr, r=r)
        h = state.h
        c = state.c

        if h.shape[0] != grad_arr.size:
            raise ValueError("State size does not match gradient size")

        z = x @ self.W_x.T + h @ self.W_h.T + self.b
        hs = self.hidden_size
        i = _sigmoid(z[:, 0:hs])
        f = _sigmoid(z[:, hs : 2 * hs])
        o = _sigmoid(z[:, 2 * hs : 3 * hs])
        g = np.tanh(z[:, 3 * hs : 4 * hs])

        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)

        delta = h_new @ self.W_out + self.b_out
        return delta.astype(float), LSTMState(h=h_new, c=c_new)



def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))
