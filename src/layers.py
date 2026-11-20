"""Capas básicas con forward y backward implementados a mano."""
from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import backend as bk
from backend import xp

Array = Any


class Parameter:
    """Tensor entrenable con su gradiente."""

    def __init__(self, data: Array, decay: bool = True) -> None:
        self.data: Array = data.astype(bk.DTYPE)
        self.grad: Array = xp.zeros_like(self.data)
        self.decay: bool = decay
        self.lr_mult: float = 1.0  # multiplicador de lr (p. ej. backbone preentrenado)


class Module:
    training: bool = True

    def forward(self, *args: Array) -> Array:  # pragma: no cover
        raise NotImplementedError

    def backward(self, dy: Array) -> Array:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, *args: Array) -> Array:
        return self.forward(*args)

    def children(self) -> Iterable["Module"]:
        for v in vars(self).values():
            if isinstance(v, Module):
                yield v
            elif isinstance(v, (list, tuple)):
                for m in v:
                    if isinstance(m, Module):
                        yield m

    def parameters(self) -> List[Parameter]:
        ps = [v for v in vars(self).values() if isinstance(v, Parameter)]
        for c in self.children():
            ps += c.parameters()
        return ps

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for c in self.children():
            c.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad[...] = 0

    def n_params(self) -> int:
        return int(sum(p.data.size for p in self.parameters()))

    # Arreglos que son estado del modelo (no activaciones) y no deben liberarse.
    _PERSISTENT = frozenset({"run_mean", "run_var", "_idx", "_P", "mask"})

    def release(self) -> None:
        """Libera las activaciones guardadas para el backward (de este módulo y sus hijos)."""
        for k, v in list(vars(self).items()):
            if k not in self._PERSISTENT and isinstance(v, (np.ndarray, xp.ndarray)):
                delattr(self, k)
        for c in self.children():
            c.release()


class Sequential(Module):
    def __init__(self, *mods: Module) -> None:
        self.mods: List[Module] = list(mods)

    def forward(self, x: Array) -> Array:
        for m in self.mods:
            x = m(x)
        return x

    def backward(self, dy: Array) -> Array:
        for m in reversed(self.mods):
            dy = m.backward(dy)
        return dy


# Capas densas

class Linear(Module):
    """y = x W + b, con x de forma (..., in) y W de forma (in, out)."""

    def __init__(self, in_f: int, out_f: int, bias: bool = True, init: str = "torch") -> None:
        if init == "trunc":
            w = xp.clip(bk.randn(in_f, out_f) * 0.02, -0.04, 0.04)
            b = xp.zeros(out_f)
        else:
            k = 1.0 / math.sqrt(in_f)
            w = (bk.rand(in_f, out_f) * 2 - 1) * k
            b = (bk.rand(out_f) * 2 - 1) * k
        self.W = Parameter(w)
        self.b: Optional[Parameter] = Parameter(b, decay=False) if bias else None
        self.in_f, self.out_f = in_f, out_f

    def forward(self, x: Array) -> Array:
        assert x.shape[-1] == self.in_f, f"Linear esperaba (..., {self.in_f}), llegó {x.shape}"
        self.x = x
        # Un solo GEMM 2D (mucho más rápido que el matmul por lotes en NumPy/CuPy)
        y = (x.reshape(-1, self.in_f) @ self.W.data).reshape(x.shape[:-1] + (self.out_f,))
        return y + self.b.data if self.b is not None else y

    def backward(self, dy: Array) -> Array:
        x2 = self.x.reshape(-1, self.in_f)
        d2 = dy.reshape(-1, self.out_f)
        self.W.grad = x2.T @ d2
        if self.b is not None:
            self.b.grad = d2.sum(axis=0)
        return (d2 @ self.W.data.T).reshape(self.x.shape)


class ReLU(Module):
    def forward(self, x: Array) -> Array:
        self.mask = x > 0
        return x * self.mask

    def backward(self, dy: Array) -> Array:
        return dy * self.mask


class GELU(Module):
    """GELU (aprox."""

    _C = math.sqrt(2.0 / math.pi)

    def forward(self, x: Array) -> Array:
        self.x = x
        x2 = x * x
        self.t = xp.tanh(self._C * x * (1.0 + 0.044715 * x2))
        return 0.5 * x * (1.0 + self.t)

    def backward(self, dy: Array) -> Array:
        x, t = self.x, self.t
        du = self._C * (1.0 + 3 * 0.044715 * (x * x))
        return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du)


class Dropout(Module):
    """Dropout invertido: en train escala por 1/(1-p); en eval es la identidad."""

    def __init__(self, p: float = 0.3) -> None:
        assert 0.0 <= p < 1.0
        self.p = p
        self.mask: Optional[Array] = None

    def forward(self, x: Array) -> Array:
        if not self.training or self.p == 0.0:
            self.mask = None
            return x
        self.mask = (bk.rand(*x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dy: Array) -> Array:
        return dy if self.mask is None else dy * self.mask


# Normalizaciones

class LayerNorm(Module):
    """Normaliza sobre el último eje."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.gamma = Parameter(xp.ones(dim), decay=False)
        self.beta = Parameter(xp.zeros(dim), decay=False)
        self.eps, self.dim = eps, dim

    def forward(self, x: Array) -> Array:
        assert x.shape[-1] == self.dim
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        self.std = xp.sqrt((xc ** 2).mean(axis=-1, keepdims=True) + self.eps)
        self.xhat = xc / self.std
        return self.gamma.data * self.xhat + self.beta.data

    def backward(self, dy: Array) -> Array:
        ax = tuple(range(dy.ndim - 1))
        self.gamma.grad = (dy * self.xhat).sum(axis=ax)
        self.beta.grad = dy.sum(axis=ax)
        g = dy * self.gamma.data
        return (g - g.mean(axis=-1, keepdims=True)
                - self.xhat * (g * self.xhat).mean(axis=-1, keepdims=True)) / self.std


class BatchNorm2d(Module):
    """BatchNorm sobre (B, H, W) para tensores (B, C, H, W)."""

    def __init__(self, ch: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        self.gamma = Parameter(xp.ones(ch), decay=False)
        self.beta = Parameter(xp.zeros(ch), decay=False)
        self.run_mean = xp.zeros(ch, dtype=bk.DTYPE)
        self.run_var = xp.ones(ch, dtype=bk.DTYPE)
        self.eps, self.momentum, self.ch = eps, momentum, ch

    def forward(self, x: Array) -> Array:
        assert x.ndim == 4 and x.shape[1] == self.ch
        ax = (0, 2, 3)
        if self.training:
            n = x.shape[0] * x.shape[2] * x.shape[3]
            mu = x.mean(axis=ax)
            var = x.var(axis=ax)
            m = self.momentum
            self.run_mean = (1 - m) * self.run_mean + m * mu
            self.run_var = (1 - m) * self.run_var + m * var * (n / max(n - 1, 1))
        else:
            mu, var = self.run_mean, self.run_var
        self.std = xp.sqrt(var + self.eps)[None, :, None, None]
        self.xhat = (x - mu[None, :, None, None]) / self.std
        self.was_training = self.training
        return self.gamma.data[None, :, None, None] * self.xhat + self.beta.data[None, :, None, None]

    def backward(self, dy: Array) -> Array:
        ax = (0, 2, 3)
        self.gamma.grad = (dy * self.xhat).sum(axis=ax)
        self.beta.grad = dy.sum(axis=ax)
        g = dy * self.gamma.data[None, :, None, None]
        if not self.was_training:
            return g / self.std
        return (g - g.mean(axis=ax, keepdims=True)
                - self.xhat * (g * self.xhat).mean(axis=ax, keepdims=True)) / self.std


# Convolución y pooling

def _im2col(x: Array, kh: int, kw: int, s: int, p: int) -> Tuple[Array, int, int]:
    """(B,C,H,W) -> columnas (B, C*kh*kw, Ho*Wo)."""
    B, C, H, W = x.shape
    if p:
        x = xp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    Ho = (H + 2 * p - kh) // s + 1
    Wo = (W + 2 * p - kw) // s + 1
    cols = xp.empty((B, C, kh, kw, Ho, Wo), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = x[:, :, i:i + s * Ho:s, j:j + s * Wo:s]
    return cols.reshape(B, C * kh * kw, Ho * Wo), Ho, Wo


def _col2im(cols: Array, shape: Sequence[int], kh: int, kw: int, s: int, p: int,
            Ho: int, Wo: int) -> Array:
    """Adjunto exacto de _im2col (acumula las contribuciones solapadas)."""
    B, C, H, W = shape
    cols = cols.reshape(B, C, kh, kw, Ho, Wo)
    x = xp.zeros((B, C, H + 2 * p, W + 2 * p), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            x[:, :, i:i + s * Ho:s, j:j + s * Wo:s] += cols[:, :, i, j]
    return x[:, :, p:H + p, p:W + p]


class Conv2d(Module):
    """Convolución 2D (correlación cruzada, igual que PyTorch) vía im2col."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, bias: bool = False) -> None:
        k = kernel_size
        std = math.sqrt(2.0 / (out_ch * k * k))
        self.W = Parameter(bk.randn(out_ch, in_ch, k, k) * std)
        self.b: Optional[Parameter] = Parameter(xp.zeros(out_ch), decay=False) if bias else None
        self.in_ch, self.out_ch, self.k, self.s, self.p = in_ch, out_ch, k, stride, padding

    def forward(self, x: Array) -> Array:
        assert x.ndim == 4 and x.shape[1] == self.in_ch, f"Conv2d esperaba C={self.in_ch}, llegó {x.shape}"
        self.x_shape = x.shape
        self.cols, self.Ho, self.Wo = _im2col(x, self.k, self.k, self.s, self.p)
        y = self.W.data.reshape(self.out_ch, -1) @ self.cols  # (B, O, Ho*Wo)
        if self.b is not None:
            y = y + self.b.data[None, :, None]
        return y.reshape(x.shape[0], self.out_ch, self.Ho, self.Wo)

    def backward(self, dy: Array) -> Array:
        B = dy.shape[0]
        d2 = dy.reshape(B, self.out_ch, -1)  # (B, O, L)
        self.W.grad = (d2 @ self.cols.transpose(0, 2, 1)).sum(axis=0).reshape(self.W.data.shape)
        if self.b is not None:
            self.b.grad = d2.sum(axis=(0, 2))
        dcols = self.W.data.reshape(self.out_ch, -1).T @ d2  # (B, C*k*k, L)
        return _col2im(dcols, self.x_shape, self.k, self.k, self.s, self.p, self.Ho, self.Wo)


class MaxPool2d(Module):
    """MaxPool kxk con stride k (H y W deben ser divisibles por k)."""

    def __init__(self, k: int = 2) -> None:
        self.k = k

    def forward(self, x: Array) -> Array:
        B, C, H, W = x.shape
        k = self.k
        assert H % k == 0 and W % k == 0, f"MaxPool2d: {H}x{W} no divisible por {k}"
        win = x.reshape(B, C, H // k, k, W // k, k).transpose(0, 1, 2, 4, 3, 5)
        win = win.reshape(B, C, H // k, W // k, k * k)
        self.idx = win.argmax(axis=-1)
        self.x_shape = x.shape
        return win.max(axis=-1)

    def backward(self, dy: Array) -> Array:
        B, C, H, W = self.x_shape
        k = self.k
        onehot = xp.arange(k * k)[None, None, None, None, :] == self.idx[..., None]
        g = onehot * dy[..., None]
        g = g.reshape(B, C, H // k, W // k, k, k).transpose(0, 1, 2, 4, 3, 5)
        return g.reshape(B, C, H, W).astype(dy.dtype)


class GlobalAvgPool2d(Module):
    """(B, C, H, W) -> (B, C)."""

    def forward(self, x: Array) -> Array:
        self.x_shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, dy: Array) -> Array:
        B, C, H, W = self.x_shape
        return xp.broadcast_to(dy[:, :, None, None] / (H * W), self.x_shape).astype(dy.dtype)


def softmax(z: Array, axis: int = -1) -> Array:
    """Softmax estable (resta el máximo antes de exponenciar)."""
    e = xp.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)
