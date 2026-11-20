"""Optimizadores (SGD, AdamW), recorte de gradiente y tasa de aprendizaje coseno."""
from __future__ import annotations

import math
from typing import Any, List, Tuple

from backend import xp
from layers import Parameter


def clip_grad_norm(params: List[Parameter], max_norm: float = 5.0) -> float:
    """Recorte por norma global (como torch.nn.utils.clip_grad_norm_)."""
    total = math.sqrt(sum(float((p.grad ** 2).sum()) for p in params))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            p.grad *= scale
    return total


class SGD:
    """SGD con momento: v <- μ*v + g ; θ <- θ - lr*v."""

    def __init__(self, params: List[Parameter], lr: float = 1e-2, momentum: float = 0.9,
                 weight_decay: float = 0.0) -> None:
        self.params, self.lr, self.mu, self.wd = params, lr, momentum, weight_decay
        self.v = [xp.zeros_like(p.data) for p in params]

    def step(self) -> None:
        for p, v in zip(self.params, self.v):
            g = p.grad + self.wd * p.data if (self.wd and p.decay) else p.grad
            v *= self.mu
            v += g
            p.data -= (self.lr * p.lr_mult) * v


class AdamW:
    """AdamW (Loshchilov & Hutter): Adam con corrección de sesgo y weight decay desacoplado."""

    def __init__(self, params: List[Parameter], lr: float = 1e-4,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-4) -> None:
        self.params, self.lr, self.eps, self.wd = params, lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.m = [xp.zeros_like(p.data) for p in params]
        self.v = [xp.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            upd = (m / c1) / (xp.sqrt(v / c2) + self.eps)
            if self.wd and p.decay:
                upd = upd + self.wd * p.data
            p.data -= (self.lr * p.lr_mult) * upd


class CosineLR:
    """lr(e) = η_min + ½*(lr₀ - η_min)*(1 + cos(π*e/T_max))."""

    def __init__(self, optimizer: Any, t_max: int, eta_min: float = 1e-7) -> None:
        self.opt, self.t_max, self.eta_min = optimizer, t_max, eta_min
        self.base_lr = optimizer.lr
        self.epoch = 0

    def step(self) -> float:
        self.epoch += 1
        e = min(self.epoch, self.t_max)
        self.opt.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * e / self.t_max))
        return self.opt.lr
