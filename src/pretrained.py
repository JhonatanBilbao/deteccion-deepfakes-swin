"""Carga de pesos ImageNet de Swin-T (formato safetensors) sin PyTorch."""
from __future__ import annotations

import json
import os
import re
import struct
import urllib.request
from typing import Dict, Optional

import numpy as np

from layers import LayerNorm, Linear
from swin import PatchMerging, SwinBlock, SwinBranch

SWIN_T_URL = ("https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in1k/"
              "resolve/main/model.safetensors")

_DT = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I64": np.int64, "I32": np.int32}


def read_safetensors(path: str) -> Dict[str, np.ndarray]:
    """Lector mínimo del formato safetensors: u64 little-endian (n) + JSON (n bytes) + datos."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    out = {}
    for k, meta in header.items():
        if k == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        raw = blob[a:b]
        if meta["dtype"] == "BF16":  # bfloat16 = 16 bits altos de un float32
            arr = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=np.dtype(_DT[meta["dtype"]]).newbyteorder("<"))
        out[k] = arr.reshape(meta["shape"]).astype(np.float32) if arr.dtype.kind == "f" else arr.reshape(meta["shape"])
    return out


def download(url: str = SWIN_T_URL, dst: str = "swin_tiny_imagenet.safetensors") -> str:
    if not os.path.exists(dst):
        tmp = dst + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dst)
    return dst


def _normalize_keys(sd: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Unifica el formato antiguo (downsample al FINAL de la etapa i) con el de timm >= 0.9 (downsample al INICIO..."""
    old_style = any(k.startswith("layers.0.downsample") for k in sd)
    out = {}
    for k, v in sd.items():
        if k.startswith("head") or "relative_position_index" in k or "attn_mask" in k:
            continue
        m = re.match(r"layers\.(\d+)\.downsample\.(.*)", k)
        if m and old_style:
            k = f"layers.{int(m.group(1)) + 1}.downsample.{m.group(2)}"
        out[k] = v
    return out


def _set(dst, src: np.ndarray, name: str, xp) -> None:
    assert tuple(dst.data.shape) == tuple(src.shape), f"{name}: {src.shape} ≠ {dst.data.shape}"
    dst.data[...] = xp.asarray(src, dtype=dst.data.dtype)


def load_swin_imagenet(model: SwinBranch, path: Optional[str] = None, verbose: bool = True) -> int:
    """Copia los pesos ImageNet en `model` (Swin-T: embed 96, profundidades 2-2-6-2, ventana 7)."""
    from backend import xp

    sd = _normalize_keys(read_safetensors(path or download()))
    used = set()

    def take(k: str) -> np.ndarray:
        used.add(k)
        return sd[k]

    def ln(dst: LayerNorm, pre: str) -> None:
        _set(dst.gamma, take(pre + ".weight"), pre, xp)
        _set(dst.beta, take(pre + ".bias"), pre, xp)

    def lin(dst: Linear, pre: str) -> None:
        _set(dst.W, take(pre + ".weight").T, pre, xp)
        if dst.b is not None:
            _set(dst.b, take(pre + ".bias"), pre, xp)

    pe = model.patch_embed
    w = take("patch_embed.proj.weight")  # (96, 3, 4, 4) -> (3*4*4, 96)
    _set(pe.proj.W, w.reshape(w.shape[0], -1).T, "patch_embed.proj", xp)
    _set(pe.proj.b, take("patch_embed.proj.bias"), "patch_embed.proj", xp)
    ln(pe.norm, "patch_embed.norm")

    stage, block = 0, 0
    for lyr in model.layers:
        if isinstance(lyr, PatchMerging):
            stage, block = stage + 1, 0
            pre = f"layers.{stage}.downsample"
            ln(lyr.norm, pre + ".norm")
            lin(lyr.red, pre + ".reduction")
            continue
        assert isinstance(lyr, SwinBlock)
        pre = f"layers.{stage}.blocks.{block}"
        ln(lyr.norm1, pre + ".norm1")
        lin(lyr.attn.qkv, pre + ".attn.qkv")
        lin(lyr.attn.proj, pre + ".attn.proj")
        _set(lyr.attn.bias_table, take(pre + ".attn.relative_position_bias_table"), pre, xp)
        ln(lyr.norm2, pre + ".norm2")
        lin(lyr.mlp.mods[0], pre + ".mlp.fc1")
        lin(lyr.mlp.mods[2], pre + ".mlp.fc2")
        block += 1
    ln(model.norm, "norm")

    missing = sorted(set(sd) - used)
    assert not missing, f"Pesos del checkpoint sin usar (arquitectura distinta): {missing[:5]}"
    if verbose:
        print(f"Swin-T ImageNet-1k: {len(used)} tensores copiados "
              f"({sum(int(np.prod(sd[k].shape)) for k in used):,} parámetros)")
    return len(used)


def set_lr_mult(model, mult: float) -> None:
    """Multiplicador de tasa de aprendizaje para todos los parámetros de `model` (p."""
    for p in model.parameters():
        p.lr_mult = mult
