"""Swin Transformer (Swin-T) implementado desde cero: rama global 224x224 -> 768 -> 256."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

import backend as bk
from backend import xp
from layers import GELU, LayerNorm, Linear, Module, Parameter, Sequential, softmax

Array = Any
MASK_VALUE = -100.0


# utilidades de ventanas
def window_partition(x: Array, w: int) -> Array:
    """(B, H, W, C) -> (B*nW, w*w, C) con nW = (H/w)*(W/w)."""
    B, H, W, C = x.shape
    x = x.reshape(B, H // w, w, W // w, w, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, w * w, C)


def window_reverse(win: Array, w: int, B: int, H: int, W: int) -> Array:
    """(B*nW, w*w, C) -> (B, H, W, C)."""
    C = win.shape[-1]
    x = win.reshape(B, H // w, W // w, w, w, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, C)


def relative_position_index(w: int) -> np.ndarray:
    """Índice (w^2, w^2) a la tabla de sesgos ((2w-1)^2, h)."""
    coords = np.stack(np.meshgrid(np.arange(w), np.arange(w), indexing="ij")).reshape(2, -1)  # (2, N)
    rel = coords[:, :, None] - coords[:, None, :] + (w - 1)  # (2, N, N)
    return rel[0] * (2 * w - 1) + rel[1]


def shifted_window_mask(H: int, W: int, w: int, s: int) -> np.ndarray:
    """Máscara (nW, N, N): 0 si dos tokens provienen de la misma región antes del roll, -100 si no."""
    img = np.zeros((1, H, W, 1))
    cnt = 0
    for hs in (slice(0, -w), slice(-w, -s), slice(-s, None)):
        for ws in (slice(0, -w), slice(-w, -s), slice(-s, None)):
            img[:, hs, ws, :] = cnt
            cnt += 1
    mw = window_partition(img, w)[..., 0]  # (nW, N)
    diff = mw[:, None, :] - mw[:, :, None]
    return np.where(diff != 0, MASK_VALUE, 0.0)


# capas Swin
class PatchEmbedSwin(Module):
    """Conv(k=p, s=p) implementada como Linear sobre parches + LayerNorm."""

    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 96) -> None:
        assert img_size % patch_size == 0, "img_size debe ser divisible por patch_size"
        self.img_size, self.p, self.c = img_size, patch_size, in_chans
        self.grid = img_size // patch_size
        self.proj = Linear(in_chans * patch_size ** 2, embed_dim, init="trunc")
        self.norm = LayerNorm(embed_dim)

    def forward(self, x: Array) -> Array:
        B, C, H, W = x.shape
        assert (C, H, W) == (self.c, self.img_size, self.img_size), \
            f"PatchEmbedSwin esperaba (B,{self.c},{self.img_size},{self.img_size}), llegó {x.shape}"
        p, g = self.p, self.grid
        patches = x.reshape(B, C, g, p, g, p).transpose(0, 2, 4, 1, 3, 5).reshape(B, g * g, C * p * p)
        return self.norm(self.proj(patches))

    def backward(self, dy: Array) -> Array:
        d = self.proj.backward(self.norm.backward(dy))
        B, p, g, C = d.shape[0], self.p, self.grid, self.c
        return d.reshape(B, g, g, C, p, p).transpose(0, 3, 1, 4, 2, 5).reshape(B, C, g * p, g * p)


class WindowAttention(Module):
    """MSA dentro de cada ventana wxw con sesgo de posición relativa aprendido."""

    def __init__(self, dim: int, window: int, num_heads: int) -> None:
        assert dim % num_heads == 0, f"dim={dim} debe ser divisible por num_heads={num_heads}"
        self.dim, self.w, self.h, self.dh = dim, window, num_heads, dim // num_heads
        self.N = window * window
        self.scale = self.dh ** -0.5
        self.qkv = Linear(dim, 3 * dim, init="trunc")
        self.proj = Linear(dim, dim, init="trunc")
        T = (2 * window - 1) ** 2
        self.bias_table = Parameter(xp.clip(bk.randn(T, num_heads) * 0.02, -0.04, 0.04), decay=False)
        idx = relative_position_index(window).reshape(-1)  # (N^2,)
        onehot = np.zeros((idx.size, T))
        onehot[np.arange(idx.size), idx] = 1.0
        self._idx = xp.asarray(idx)
        self._P = bk.asarray(onehot)  # (N^2, T) constante

    def forward(self, x: Array, mask: Optional[Array] = None) -> Array:
        Bw, N, C = x.shape
        assert N == self.N and C == self.dim, f"WindowAttention esperaba (*, {self.N}, {self.dim}), llegó {x.shape}"
        qkv = self.qkv(x).reshape(Bw, N, 3, self.h, self.dh).transpose(2, 0, 3, 1, 4)  # (3,Bw,h,N,dh)
        self.q, self.k, self.v = qkv[0], qkv[1], qkv[2]
        s = (self.q @ self.k.transpose(0, 1, 3, 2)) * self.scale  # (Bw,h,N,N)
        rel = self.bias_table.data[self._idx].reshape(N, N, self.h).transpose(2, 0, 1)  # (h,N,N)
        s = s + rel[None]
        if mask is not None:
            nW = mask.shape[0]
            s = (s.reshape(Bw // nW, nW, self.h, N, N) + mask[None, :, None]).reshape(Bw, self.h, N, N)
        self.attn = softmax(s, axis=-1)
        out = (self.attn @ self.v).transpose(0, 2, 1, 3).reshape(Bw, N, C)
        return self.proj(out)

    def backward(self, dy: Array) -> Array:
        Bw, N, C = dy.shape
        do = self.proj.backward(dy).reshape(Bw, N, self.h, self.dh).transpose(0, 2, 1, 3)
        da = do @ self.v.transpose(0, 1, 3, 2)
        dv = self.attn.transpose(0, 1, 3, 2) @ do
        ds = self.attn * (da - (da * self.attn).sum(axis=-1, keepdims=True))  # (Bw,h,N,N)
        drel = ds.sum(axis=0).reshape(self.h, N * N)  # (h, N^2)
        self.bias_table.grad = self._P.T @ drel.T  # (T, h)
        dq = (ds @ self.k) * self.scale
        dk = (ds.transpose(0, 1, 3, 2) @ self.q) * self.scale
        dqkv = xp.stack([dq, dk, dv], axis=0).transpose(1, 3, 0, 2, 4).reshape(Bw, N, 3 * C)
        return self.qkv.backward(dqkv)


class SwinBlock(Module):
    """x <- x + Rev(SW/W-MSA(Part(roll(LN(x))))) ; x <- x + MLP(LN(x))."""

    def __init__(self, dim: int, resolution: Tuple[int, int], num_heads: int, window: int = 7,
                 shift: int = 0, mlp_ratio: float = 4.0) -> None:
        H, W = resolution
        if min(H, W) <= window:  # la ventana cubre toda la imagen: sin desplazamiento
            window, shift = min(H, W), 0
        assert H % window == 0 and W % window == 0, f"resolución {resolution} no divisible por ventana {window}"
        assert 0 <= shift < window
        self.dim, self.H, self.W, self.w, self.s = dim, H, W, window, shift
        self.norm1 = LayerNorm(dim)
        self.attn = WindowAttention(dim, window, num_heads)
        self.norm2 = LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = Sequential(Linear(dim, hidden, init="trunc"), GELU(), Linear(hidden, dim, init="trunc"))
        self.mask = bk.asarray(shifted_window_mask(H, W, window, shift)) if shift > 0 else None

    def forward(self, x: Array) -> Array:
        B, L, C = x.shape
        assert L == self.H * self.W and C == self.dim, f"SwinBlock esperaba (B,{self.H * self.W},{self.dim}), llegó {x.shape}"
        h = self.norm1(x).reshape(B, self.H, self.W, C)
        if self.s:
            h = xp.roll(h, (-self.s, -self.s), axis=(1, 2))
        h = self.attn(window_partition(h, self.w), self.mask)
        h = window_reverse(h, self.w, B, self.H, self.W)
        if self.s:
            h = xp.roll(h, (self.s, self.s), axis=(1, 2))
        x = x + h.reshape(B, L, C)
        return x + self.mlp(self.norm2(x))

    def backward(self, dy: Array) -> Array:
        B, L, C = dy.shape
        d = dy + self.norm2.backward(self.mlp.backward(dy))
        g = d.reshape(B, self.H, self.W, C)
        if self.s:  # adjunta de roll(+s) = roll(-s)
            g = xp.roll(g, (-self.s, -self.s), axis=(1, 2))
        g = self.attn.backward(window_partition(g, self.w))  # adjunta de reverse = partition
        g = window_reverse(g, self.w, B, self.H, self.W)
        if self.s:
            g = xp.roll(g, (self.s, self.s), axis=(1, 2))
        return d + self.norm1.backward(g.reshape(B, L, C))


class PatchMerging(Module):
    """(B, H*W, C) -> concat de vecinos 2x2 (B, H/2*W/2, 4C) -> LayerNorm -> Linear(4C, 2C, sin sesgo)."""

    def __init__(self, resolution: Tuple[int, int], dim: int) -> None:
        H, W = resolution
        assert H % 2 == 0 and W % 2 == 0, f"PatchMerging requiere resolución par, llegó {resolution}"
        self.H, self.W, self.dim = H, W, dim
        self.norm = LayerNorm(4 * dim)
        self.red = Linear(4 * dim, 2 * dim, bias=False, init="trunc")

    def forward(self, x: Array) -> Array:
        B, L, C = x.shape
        assert L == self.H * self.W and C == self.dim
        # orden de concatenación oficial: x0=(0,0), x1=(1,0), x2=(0,1), x3=(1,1)
        x = x.reshape(B, self.H // 2, 2, self.W // 2, 2, C).transpose(0, 1, 3, 4, 2, 5)
        x = x.reshape(B, (self.H // 2) * (self.W // 2), 4 * C)
        return self.red(self.norm(x))

    def backward(self, dy: Array) -> Array:
        d = self.norm.backward(self.red.backward(dy))
        B, C = d.shape[0], self.dim
        d = d.reshape(B, self.H // 2, self.W // 2, 2, 2, C).transpose(0, 1, 4, 2, 3, 5)
        return d.reshape(B, self.H * self.W, C)


class SwinBranch(Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 96, depths: Sequence[int] = (2, 2, 6, 2),
                 num_heads: Sequence[int] = (3, 6, 12, 24), window: int = 7,
                 mlp_ratio: float = 4.0, proj_dim: int = 256) -> None:
        assert len(depths) == len(num_heads)
        self.patch_embed = PatchEmbedSwin(img_size, patch_size, in_chans, embed_dim)
        res = self.patch_embed.grid
        self.layers: List[Module] = []
        dim = embed_dim
        for i, (dep, nh) in enumerate(zip(depths, num_heads)):
            for j in range(dep):
                self.layers.append(SwinBlock(dim, (res, res), nh, window,
                                             shift=0 if j % 2 == 0 else window // 2, mlp_ratio=mlp_ratio))
            if i < len(depths) - 1:
                self.layers.append(PatchMerging((res, res), dim))
                res, dim = res // 2, dim * 2
        self.norm = LayerNorm(dim)
        self.proj = Sequential(Linear(dim, proj_dim), LayerNorm(proj_dim))
        self.embed_dim, self.num_features, self.proj_dim = embed_dim, dim, proj_dim

    def forward_features(self, x: Array) -> Array:
        """(B, 3, H, W) -> vector global (B, num_features) = (B, 768) en Swin-T."""
        x = self.patch_embed(x)
        for lyr in self.layers:
            x = lyr(x)
        x = self.norm(x)
        self.n_tokens = x.shape[1]
        return x.mean(axis=1)

    def forward(self, x: Array) -> Array:
        return self.proj(self.forward_features(x))  # (B, 256)

    def backward(self, dy: Array) -> Array:
        d = self.proj.backward(dy)
        d = xp.broadcast_to(d[:, None, :] / self.n_tokens, (d.shape[0], self.n_tokens, d.shape[1]))
        d = self.norm.backward(xp.ascontiguousarray(d))
        for lyr in reversed(self.layers):
            d = lyr.backward(d)
        return self.patch_embed.backward(d)
