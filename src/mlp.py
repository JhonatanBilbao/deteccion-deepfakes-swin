"""Clasificador MLP de la etapa 3."""
from __future__ import annotations

from typing import Any

from layers import Dropout, Linear, Module, ReLU, Sequential

Array = Any


class MLPClassifier(Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2,
                 dropout: float = 0.3) -> None:
        self.in_dim, self.num_classes = in_dim, num_classes
        self.net = Sequential(
            Linear(in_dim, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Array) -> Array:
        assert x.ndim == 2 and x.shape[1] == self.in_dim, f"MLP esperaba (B, {self.in_dim}), llegó {x.shape}"
        return self.net(x)  # (B, num_classes)

    def backward(self, dlogits: Array) -> Array:
        return self.net.backward(dlogits)  # (B, in_dim)
