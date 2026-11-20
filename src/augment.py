"""Degradaciones de calidad, aumento de datos, normalización robusta y mapa de asimetría."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np

import backend as bk
from backend import xp

Array = Any


# Operadores básicos

@lru_cache(maxsize=64)
def _gauss_kernel_np(sigma: float) -> np.ndarray:
    r = max(1, int(math.ceil(3.0 * sigma)))
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(t ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x: Array, sigma: float) -> Array:
    """Desenfoque gaussiano separable con borde reflejado."""
    if sigma <= 0:
        return x
    k = xp.asarray(_gauss_kernel_np(round(float(sigma), 3)), dtype=x.dtype)
    r = (k.shape[0] - 1) // 2
    H, W = x.shape[-2], x.shape[-1]
    assert r < min(H, W), "sigma demasiado grande para el tamaño de la imagen"
    pad = [(0, 0)] * (x.ndim - 2)
    xh = xp.pad(x, pad + [(0, 0), (r, r)], mode="reflect")
    y = sum(k[i] * xh[..., :, i:i + W] for i in range(2 * r + 1))
    yv = xp.pad(y, pad + [(r, r), (0, 0)], mode="reflect")
    return sum(k[i] * yv[..., i:i + H, :] for i in range(2 * r + 1))


@lru_cache(maxsize=64)
def _bilinear_matrix_np(n_in: int, n_out: int) -> np.ndarray:
    """Matriz (n_out, n_in) de interpolación lineal con centros de píxel (como cv2.INTER_LINEAR)."""
    src = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
    src = np.clip(src, 0.0, n_in - 1.0)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w1 = src - i0
    M = np.zeros((n_out, n_in))
    rows = np.arange(n_out)
    np.add.at(M, (rows, i0), 1.0 - w1)
    np.add.at(M, (rows, i1), w1)
    return M


@lru_cache(maxsize=64)
def _area_matrix_np(n_in: int, n_out: int) -> np.ndarray:
    """Matriz (n_out, n_in) de submuestreo por promedio de área (antialias, como cv2.INTER_AREA)."""
    s = n_in / n_out
    M = np.zeros((n_out, n_in))
    for o in range(n_out):
        a, b = o * s, (o + 1) * s
        for i in range(int(math.floor(a)), min(int(math.ceil(b)), n_in)):
            M[o, i] = (min(b, i + 1) - max(a, i)) / s
    return M


def resize(x: Array, h_out: int, w_out: int, mode: str = "bilinear") -> Array:
    """Redimensiona (..., H, W) -> (..., h_out, w_out) como y = M_h * x * M_w^T."""
    H, W = x.shape[-2], x.shape[-1]
    f = _area_matrix_np if mode == "area" else _bilinear_matrix_np
    Mh = xp.asarray(f(H, h_out), dtype=x.dtype)
    Mw = xp.asarray(f(W, w_out), dtype=x.dtype)
    return xp.matmul(xp.matmul(Mh, x), Mw.T)


def downscale_upscale(x: Array, factor: float) -> Array:
    """Reduce la resolución por `factor` (0 < factor <= 1) y vuelve al tamaño original."""
    if factor >= 1.0:
        return x
    H, W = x.shape[-2], x.shape[-1]
    h, w = max(8, int(round(H * factor))), max(8, int(round(W * factor)))
    return resize(resize(x, h, w, "area"), H, W, "bilinear")


def add_gaussian_noise(x: Array, std: float, rng: Any = None) -> Array:
    if std <= 0:
        return x
    r = rng if rng is not None else xp.random
    return x + std * r.standard_normal(x.shape).astype(x.dtype)


def quantize_uint8(x: Array) -> Array:
    """Recorta a [0,1] y cuantiza a 256 niveles (como una imagen guardada en 8 bits)."""
    return xp.round(xp.clip(x, 0.0, 1.0) * 255.0) / 255.0


def jpeg_compress(x: Array, quality: int) -> Array:
    """Compresión JPEG real con OpenCV (solo evaluación, degradación fuera de distribución)."""
    import cv2

    arr = bk.to_numpy(xp.clip(x, 0, 1))
    out = np.empty_like(arr)
    for b in range(arr.shape[0]):
        img = (arr[b].transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        assert ok, "Falló la codificación JPEG"
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
        out[b] = dec.transpose(2, 0, 1).astype(np.float32) / 255.0
    return bk.asarray(out)


DIFF_MODES = ("pixel", "residual")


def bilateral_diff_map(face: Array, mode: str = "pixel", sigma: float = 1.0) -> Array:
    """Rama lateral: (B,3,H,W) en [0,1] -> mapa de asimetría (B,3,H,W/2) = |fliplr(izq) - der|."""
    assert mode in DIFF_MODES, f"mode debe ser uno de {DIFF_MODES}"
    W = face.shape[-1]
    assert W % 2 == 0, "El ancho del rostro debe ser par para dividir en hemicaras"
    half = W // 2
    if mode == "residual":
        face = face - gaussian_blur(face, sigma)
    left, right = face[..., :half][..., ::-1], face[..., half:]
    if mode == "residual":
        eps = 1e-6
        left = left / (xp.sqrt((left * left).mean(axis=(-2, -1), keepdims=True)) + eps)
        right = right / (xp.sqrt((right * right).mean(axis=(-2, -1), keepdims=True)) + eps)
    return xp.abs(left - right)


# Degradación determinista (evaluación)

# Niveles de calidad para la evaluación comparativa. 'buena' = imagen original.
QUALITY_LEVELS: Dict[str, Dict[str, float]] = {
    "buena":    {},
    "regular":  {"down": 0.75, "blur": 0.8, "noise": 3 / 255},
    "mala":     {"down": 0.50, "blur": 1.2, "noise": 6 / 255},
    "muy_mala": {"down": 0.30, "blur": 2.0, "noise": 12 / 255},
}

# Degradaciones aisladas (diagnóstico) y una no vista en entrenamiento (JPEG).
ISOLATED_DEGRADATIONS: Dict[str, Dict[str, float]] = {
    "solo_desenfoque": {"blur": 1.5},
    "solo_ruido":      {"noise": 10 / 255},
    "solo_baja_res":   {"down": 0.35},
    "jpeg_q30_no_visto": {"jpeg": 30},
}


def apply_degradation(x: Array, spec: Dict[str, float], seed: int = 0) -> Array:
    """Aplica una degradación fija y reproducible."""
    if not spec:
        return x
    rng = xp.random.RandomState(seed)
    if spec.get("down", 1.0) < 1.0:
        x = downscale_upscale(x, spec["down"])
    if spec.get("blur", 0.0) > 0:
        x = gaussian_blur(x, spec["blur"])
    if spec.get("noise", 0.0) > 0:
        x = add_gaussian_noise(x, spec["noise"], rng)
    x = quantize_uint8(x)
    if spec.get("jpeg"):
        x = jpeg_compress(x, int(spec["jpeg"]))
    return x


# Aumento aleatorio (entrenamiento)

@dataclass
class AugConfig:
    """Probabilidades y rangos del aumento de calidad (muestreados por imagen)."""
    p_down: float = 0.5
    down_range: Tuple[float, float] = (0.25, 0.9)
    p_blur: float = 0.5
    sigma_range: Tuple[float, float] = (0.4, 2.2)
    p_noise: float = 0.5
    noise_range: Tuple[float, float] = (1 / 255, 15 / 255)
    p_color: float = 0.3
    brightness: float = 0.08
    contrast: float = 0.15
    # Sin flip horizontal: invertiría las hemicaras del mapa diferencial.


class QualityAugmenter:
    """Aumento de datos de calidad (desenfoque, ruido, baja resolución, brillo/contraste)."""

    def __init__(self, cfg: Optional[AugConfig] = None) -> None:
        self.cfg = cfg or AugConfig()

    def sample_params(self) -> Dict[str, float]:
        c, r = self.cfg, np.random
        p: Dict[str, float] = {}
        if r.rand() < c.p_down:
            p["down"] = float(r.uniform(*c.down_range))
        if r.rand() < c.p_blur:
            p["blur"] = float(r.uniform(*c.sigma_range))
        if r.rand() < c.p_noise:
            p["noise"] = float(r.uniform(*c.noise_range))
        if r.rand() < c.p_color:
            p["bright"] = float(r.uniform(-c.brightness, c.brightness))
            p["contrast"] = float(r.uniform(1 - c.contrast, 1 + c.contrast))
        return p

    def __call__(self, x: Array) -> Array:
        out = xp.empty_like(x)
        for b in range(x.shape[0]):
            p = self.sample_params()
            img = x[b:b + 1]
            if "down" in p:
                img = downscale_upscale(img, p["down"])
            if "blur" in p:
                img = gaussian_blur(img, p["blur"])
            if "contrast" in p:
                m = img.mean(axis=(-2, -1), keepdims=True)
                img = (img - m) * p["contrast"] + m + p["bright"]
            if "noise" in p:
                img = add_gaussian_noise(img, p["noise"])
            out[b:b + 1] = quantize_uint8(img)
        return out


# Normalización robusta

class RobustNormalizer:
    """Normalización por canal con estadísticos robustos: z = clip((x - mediana) / (IQR/1.349), ±c)."""

    def __init__(self, clip: float = 5.0) -> None:
        self.clip = clip
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, max_pixels: int = 2_000_000, seed: int = 42) -> "RobustNormalizer":
        """x: np.ndarray (N, C, H, W) en [0,1] del conjunto de entrenamiento."""
        x = np.asarray(x)
        C = x.shape[1]
        flat = x.transpose(1, 0, 2, 3).reshape(C, -1)
        if flat.shape[1] > max_pixels:
            idx = np.random.RandomState(seed).choice(flat.shape[1], max_pixels, replace=False)
            flat = flat[:, idx]
        q25, q50, q75 = np.percentile(flat, [25, 50, 75], axis=1)
        self.center = q50.astype(np.float32)
        self.scale = np.maximum((q75 - q25) / 1.349, 1e-3).astype(np.float32)
        return self

    def __call__(self, x: Array) -> Array:
        assert self.center is not None, "RobustNormalizer sin ajustar: llama a fit() con train"
        c = xp.asarray(self.center, dtype=x.dtype)[None, :, None, None]
        s = xp.asarray(self.scale, dtype=x.dtype)[None, :, None, None]
        return xp.clip((x - c) / s, -self.clip, self.clip)

    def to_dict(self) -> Dict[str, Any]:
        return {"center": self.center.tolist(), "scale": self.scale.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RobustNormalizer":
        n = cls(d["clip"])
        n.center = np.asarray(d["center"], dtype=np.float32)
        n.scale = np.asarray(d["scale"], dtype=np.float32)
        return n
