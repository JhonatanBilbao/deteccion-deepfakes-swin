"""CNN de la rama lateral (mapa de asimetría 224x112 -> 256)."""
from __future__ import annotations

from typing import Any, Tuple

from layers import (BatchNorm2d, Conv2d, GlobalAvgPool2d, LayerNorm, Linear, MaxPool2d,
                    Module, ReLU, Sequential)

Array = Any


def conv_block(in_ch: int, out_ch: int) -> Sequential:
    return Sequential(
        Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(out_ch),
        ReLU(),
        MaxPool2d(2),
    )


class CNNBranch(Module):
    def __init__(self, in_channels: int = 3, widths: Tuple[int, int, int] = (32, 64, 128),
                 proj_dim: int = 256) -> None:
        self.in_channels = in_channels
        chans = (in_channels,) + tuple(widths)
        self.blocks = Sequential(*[conv_block(chans[i], chans[i + 1]) for i in range(3)])
        self.pool = GlobalAvgPool2d()
        self.proj = Sequential(Linear(widths[-1], proj_dim), LayerNorm(proj_dim))

    def forward(self, x: Array) -> Array:
        assert x.ndim == 4 and x.shape[1] == self.in_channels, f"CNN esperaba (B,{self.in_channels},H,W), llegó {x.shape}"
        assert x.shape[2] % 8 == 0 and x.shape[3] % 8 == 0, "H y W deben ser divisibles por 8 (3 MaxPool)"
        f = self.blocks(x)  # (B, 128, H/8, W/8)
        v = self.pool(f)  # (B, 128)
        return self.proj(v)  # (B, 256)

    def backward(self, dy: Array) -> Array:
        d = self.proj.backward(dy)  # (B, 128)
        d = self.pool.backward(d)  # (B, 128, H/8, W/8)
        return self.blocks.backward(d)  # (B, 3, H, W)
