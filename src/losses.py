"""Focal Loss con su gradiente."""
from __future__ import annotations

from typing import Any, Optional, Tuple

from backend import xp

Array = Any


class FocalLoss:
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.75) -> None:
        self.gamma, self.alpha = gamma, alpha

    def __call__(self, logits: Array, targets: Array) -> Tuple[float, Array]:
        """logits (B, K) y targets (B,) enteros -> (pérdida media, dL/dlogits de forma (B, K))."""
        B, K = logits.shape
        assert targets.shape == (B,), f"targets debe ser (B,), llegó {targets.shape}"
        z = logits - logits.max(axis=1, keepdims=True)
        logp = z - xp.log(xp.exp(z).sum(axis=1, keepdims=True))  # log-softmax estable
        p = xp.exp(logp)
        rows = xp.arange(B)
        u = logp[rows, targets]  # log p_t <= 0
        pt = xp.exp(u)
        one_m = xp.maximum(-xp.expm1(u), 1e-12)  # 1 - p_t sin cancelación
        if self.alpha is None:
            at = xp.ones_like(u)
        else:
            at = xp.where(targets == 1, self.alpha, 1.0 - self.alpha).astype(logits.dtype)
        loss = (-at * one_m ** self.gamma * u).mean()
        coef = at * (self.gamma * pt * one_m ** (self.gamma - 1.0) * u - one_m ** self.gamma)
        onehot = xp.zeros_like(logits)
        onehot[rows, targets] = 1.0
        dlogits = coef[:, None] * (onehot - p) / B
        return float(loss), dlogits
