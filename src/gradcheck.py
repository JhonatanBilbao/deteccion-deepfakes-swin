"""Verificación numérica de todos los gradientes (diferencias finitas en float64)."""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backend as bk

bk.set_dtype(np.float64)
bk.set_seed(42)
from backend import xp
from cnn import CNNBranch
from detector import DeepfakeDetectorScratch
from layers import (BatchNorm2d, Conv2d, GELU, GlobalAvgPool2d, LayerNorm, Linear, MaxPool2d,
                    Module, Parameter)
from losses import FocalLoss
from mlp import MLPClassifier
from swin import (PatchEmbedSwin, PatchMerging, SwinBlock, SwinBranch, WindowAttention,
                  shifted_window_mask, window_partition, window_reverse)

EPS, TOL, ABS_TOL, N_SAMPLES = 1e-6, 1e-5, 1e-8, 12
_rng = np.random.RandomState(0)


def _rel(a: float, n: float) -> float:
    if abs(a - n) < ABS_TOL:  # por debajo del redondeo de la derivada numérica
        return 0.0
    return abs(a - n) / (abs(a) + abs(n) + 1e-12)


def _check_array(arr: Any, grad: Any, loss_fn: Callable[[], float]) -> float:
    """Compara `grad` con diferencias centrales en entradas aleatorias de `arr`."""
    worst = 0.0
    flat_idx = _rng.choice(arr.size, size=min(N_SAMPLES, arr.size), replace=False)
    for fi in flat_idx:
        idx = np.unravel_index(int(fi), arr.shape)
        old = float(arr[idx])
        arr[idx] = old + EPS
        lp = loss_fn()
        arr[idx] = old - EPS
        lm = loss_fn()
        arr[idx] = old
        worst = max(worst, _rel(float(grad[idx]), (lp - lm) / (2 * EPS)))
    return worst


def check_module(name: str, mod: Module, x: Any) -> bool:
    """L = sum out*R con R fija; verifica dL/dθ de cada parámetro y dL/dx."""
    out = mod(x)
    R = bk.randn(*out.shape)
    loss_fn = lambda: float((mod(x) * R).sum())
    mod(x)
    dx = mod.backward(R)
    grads = [p.grad.copy() for p in mod.parameters()]
    worst = _check_array(x, dx, loss_fn)
    for p, g in zip(mod.parameters(), grads):
        assert g.shape == p.data.shape, f"{name}: grad {g.shape} ≠ param {p.data.shape}"
        worst = max(worst, _check_array(p.data, g, loss_fn))
    ok = worst < TOL
    print(f"  [{'OK ' if ok else 'FALLA'}] {name:<34} out={tuple(out.shape)}  err_rel_max={worst:.2e}")
    return ok


class _MaskedWA(Module):
    """WindowAttention con la máscara real de un mapa 8x8, ventana 4, desplazamiento 2 (nW=4)."""

    def __init__(self) -> None:
        self.wa = WindowAttention(6, 4, 2)
        self.mask = bk.asarray(shifted_window_mask(8, 8, 4, 2))

    def forward(self, x: Any) -> Any:
        return self.wa(x, self.mask)

    def backward(self, dy: Any) -> Any:
        return self.wa.backward(dy)


def _check_swin_structure() -> bool:
    """Partición/reverso inversos exactos y máscara que bloquea regiones no contiguas tras el roll."""
    x = bk.randn(2, 8, 8, 5)
    inv = bool(xp.array_equal(window_reverse(window_partition(x, 4), 4, 2, 8, 8), x))
    m = shifted_window_mask(8, 8, 4, 2)
    blk = SwinBlock(6, (8, 8), 2, window=4, shift=2)
    blk(bk.randn(1, 64, 6))
    a = bk.to_numpy(blk.attn.attn)  # (nW, h, N, N)
    leak = float(a[np.broadcast_to((m != 0)[:, None], a.shape)].max())
    # ventana 0 sin mezclar regiones (todo 0); la última mezcla 4 regiones
    shape_ok = m.shape == (4, 16, 16) and (m[0] == 0).all() and (m[3] != 0).any()
    ok = inv and shape_ok and leak < 1e-30
    print(f"  [{'OK ' if ok else 'FALLA'}] Swin: partición↔reverso exactos, máscara (4,16,16), "
          f"atención máx. entre regiones bloqueadas={leak:.1e}")
    return ok


def main() -> None:
    results: List[bool] = []
    print(f"Backend: {'CuPy (GPU)' if bk.ON_GPU else 'NumPy (CPU)'} | dtype={bk.DTYPE.__name__}")
    print("1) Gradientes capa por capa")
    results += [
        check_module("Linear", Linear(5, 4), bk.randn(3, 7, 5)),
        check_module("GELU", GELU(), bk.randn(4, 6)),
        check_module("LayerNorm", LayerNorm(6), bk.randn(3, 5, 6)),
        check_module("BatchNorm2d (train)", BatchNorm2d(3), bk.randn(4, 3, 5, 4)),
        check_module("Conv2d k3 s1 p1", Conv2d(3, 4, 3, 1, 1, bias=False), bk.randn(2, 3, 6, 5)),
        check_module("Conv2d k3 s2 p1 +bias", Conv2d(2, 3, 3, 2, 1, bias=True), bk.randn(2, 2, 7, 6)),
        check_module("MaxPool2d", MaxPool2d(2), bk.randn(2, 3, 6, 4)),
        check_module("GlobalAvgPool2d", GlobalAvgPool2d(), bk.randn(2, 3, 4, 5)),
        check_module("PatchEmbedSwin (+LN)", PatchEmbedSwin(8, 2, 3, 6), bk.randn(2, 3, 8, 8)),
        check_module("WindowAttention (sin máscara)", WindowAttention(6, 3, 2), bk.randn(4, 9, 6)),
        check_module("WindowAttention (con máscara)", _MaskedWA(), bk.randn(2 * 4, 16, 6)),
        check_module("SwinBlock W-MSA (shift=0)", SwinBlock(6, (8, 8), 2, window=4, shift=0), bk.randn(2, 64, 6)),
        check_module("SwinBlock SW-MSA (shift=2)", SwinBlock(6, (8, 8), 2, window=4, shift=2), bk.randn(2, 64, 6)),
        check_module("PatchMerging", PatchMerging((4, 6), 5), bk.randn(2, 24, 5)),
    ]
    results.append(_check_swin_structure())
    bn = BatchNorm2d(3); bn(bk.randn(4, 3, 5, 4)); bn.eval()
    results.append(check_module("BatchNorm2d (eval)", bn, bk.randn(4, 3, 5, 4)))

    print("2) Gradientes de los tres modelos (configuración reducida)")
    mlp = MLPClassifier(12, 8, 2, dropout=0.0)
    cnn = CNNBranch(3, (4, 6, 8), proj_dim=6)
    # 16x16 -> parches 2 -> 8x8 (ventana 4, con desplazamiento) -> merging -> 4x4 (ventana = mapa)
    vit = SwinBranch(img_size=16, patch_size=2, embed_dim=4, depths=(2, 2), num_heads=(2, 2),
                     window=4, proj_dim=6)
    results += [
        check_module("MLPClassifier", mlp, bk.randn(4, 12)),
        check_module("CNNBranch", cnn, bk.randn(3, 3, 16, 8)),
        check_module("SwinBranch", vit, bk.randn(2, 3, 16, 16)),
    ]

    print("3) Focal Loss (α=0.75, γ=2) respecto a los logits")
    crit = FocalLoss(2.0, 0.75)
    z = bk.randn(6, 2) * 2
    t = xp.asarray([0, 1, 1, 0, 1, 0])
    _, dz = crit(z, t)
    w = _check_array(z, dz, lambda: crit(z, t)[0])
    results.append(w < TOL)
    print(f"  [{'OK ' if w < TOL else 'FALLA'}] FocalLoss{'':<25} err_rel_max={w:.2e}")
    big = xp.asarray([[800.0, -800.0], [-800.0, 800.0]])
    lb, db = crit(big, xp.asarray([0, 0]))
    fin = bool(np.isfinite(lb) and xp.isfinite(db).all())
    results.append(fin)
    print(f"  [{'OK ' if fin else 'FALLA'}] Estabilidad con logits ±800      loss={lb:.4f} (finita)")

    print("4) Detector completo de extremo a extremo (Swin + CNN + fusión + MLP + Focal Loss)")
    det = DeepfakeDetectorScratch(vit, cnn, MLPClassifier(12, 8, 2, dropout=0.0), drop_branch=0.0)
    face, diff = bk.randn(4, 3, 16, 16), bk.randn(4, 3, 16, 8)
    y = xp.asarray([0, 1, 1, 0])
    loss_fn = lambda: crit(det(face, diff), y)[0]
    _, dl = crit(det(face, diff), y)
    det.backward(dl)
    grads = [p.grad.copy() for p in det.parameters()]
    w = max(_check_array(p.data, g, loss_fn) for p, g in zip(det.parameters(), grads))
    results.append(w < TOL)
    print(f"  [{'OK ' if w < TOL else 'FALLA'}] {len(grads)} tensores de parámetros{'':<10} err_rel_max={w:.2e}")

    lm = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)  # frames 2 y 4 con la rama lateral apagada (giro)

    def loss_mask():
        return crit(det(face, diff, lm), y)[0]
    _, dl = crit(det(face, diff, lm), y)
    det.zero_grad()
    det.backward(dl)
    grads = [p.grad.copy() for p in det.parameters()]
    w = max(_check_array(p.data, g, loss_mask) for p, g in zip(det.parameters(), grads))
    results.append(w < TOL)
    print(f"  [{'OK ' if w < TOL else 'FALLA'}] Máscara lateral por giro{'':<21} err_rel_max={w:.2e}")

    for mode in ("solo_swin", "solo_asimetria"):
        det.branches = mode
        _, dl = crit(det(face, diff), y)
        det.zero_grad()
        det.backward(dl)
        grads = [p.grad.copy() for p in det.parameters()]
        w = max(_check_array(p.data, g, loss_fn) for p, g in zip(det.parameters(), grads))
        results.append(w < TOL)
        print(f"  [{'OK ' if w < TOL else 'FALLA'}] Ablación {mode:<16}{'':<12} err_rel_max={w:.2e}")
    det.branches = "ambas"

    print("5) Dimensiones con los tamaños reales del pipeline (float32, B=2, Swin-T completo 2-2-6-2)")
    bk.set_dtype(np.float32)
    bk.set_seed(42)
    B = 2
    vit_r = SwinBranch()
    cnn_r, mlp_r = CNNBranch(), MLPClassifier()
    det_r = DeepfakeDetectorScratch(vit_r, cnn_r, mlp_r)
    face, diff = bk.randn(B, 3, 224, 224), bk.randn(B, 3, 224, 112)
    g768 = vit_r.forward_features(face)
    fg, fl = vit_r(face), cnn_r(diff)
    logits = det_r(face, diff)
    loss, dl = FocalLoss()(logits, xp.asarray([0, 1]))
    det_r.backward(dl)
    shapes = {
        "Swin vector global": (g768.shape, (B, 768)),
        "Swin proyección": (fg.shape, (B, 256)),
        "CNN proyección": (fl.shape, (B, 256)),
        "logits": (logits.shape, (B, 2)),
        "dL/dlogits": (dl.shape, (B, 2)),
    }
    for k, (got, exp) in shapes.items():
        ok = tuple(got) == exp
        results.append(ok)
        print(f"  [{'OK ' if ok else 'FALLA'}] {k:<20} {tuple(got)}")
    ok = all(p.grad.shape == p.data.shape and p.grad.dtype == np.float32 and bool(xp.isfinite(p.grad).all())
             for p in det_r.parameters())
    results.append(ok)
    print(f"  [{'OK ' if ok else 'FALLA'}] grads finitos, float32 y con la forma de su parámetro | loss={loss:.4f}")
    pv = det_r.predict_video(bk.randn(3, 3, 224, 224), bk.randn(3, 3, 224, 112))
    results.append(0.0 <= pv <= 1.0)
    print(f"  [OK ] predict_video → P(deepfake)={pv:.4f}")
    print(f"  Parámetros: CNN={cnn_r.n_params():,} | MLP={mlp_r.n_params():,} | Swin-T={vit_r.n_params():,}")

    print(f"\nRESULTADO: {sum(results)}/{len(results)} verificaciones correctas")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
