"""Backend numérico: CuPy en GPU o NumPy en CPU (DF_BACKEND=numpy fuerza CPU)."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

_cp = None
if os.environ.get("DF_BACKEND", "").lower() != "numpy":
    try:
        import cupy as _cp  # type: ignore

        if _cp.cuda.runtime.getDeviceCount() < 1:
            _cp = None
    except Exception:  # CuPy no instalado o sin GPU visible
        _cp = None

xp: Any = _cp if _cp is not None else np
ON_GPU: bool = _cp is not None

# float32 para entrenar; float64 solo para la verificación numérica de gradientes.
DTYPE: Any = np.float32
SEED: int = 42


def set_dtype(dtype: Any) -> None:
    """Cambia la precisión global."""
    global DTYPE
    DTYPE = dtype


def set_seed(seed: int = SEED) -> None:
    """Fija todas las semillas (python, hash, numpy y cupy)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if ON_GPU:
        xp.random.seed(seed)


def asarray(a: Any) -> Any:
    """Convierte a arreglo del backend activo con la precisión global."""
    return xp.asarray(a, dtype=DTYPE)


def to_numpy(a: Any) -> np.ndarray:
    """Devuelve siempre un np.ndarray en CPU (para métricas con sklearn / gráficas)."""
    return _cp.asnumpy(a) if ON_GPU else np.asarray(a)


def randn(*shape: int) -> Any:
    return xp.random.standard_normal(shape).astype(DTYPE)


def rand(*shape: int) -> Any:
    return xp.random.random_sample(shape).astype(DTYPE)
