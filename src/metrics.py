"""Métricas (AUC, F1, precisión, recall, EER), intervalos bootstrap y figuras."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)


def save_fig(fig: matplotlib.figure.Figure, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def video_scores(frame_probs: np.ndarray, frame_video: np.ndarray, video_ids: np.ndarray) -> np.ndarray:
    """Agregación temporal (Etapa 3): P_video = promedio de p_i sobre sus frames."""
    out = np.empty(len(video_ids))
    for k, v in enumerate(video_ids):
        out[k] = frame_probs[frame_video == v].mean()
    return out


def youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    return float(np.clip(thr[int(np.argmax(tpr - fpr))], 0.0, 1.0))


def eer(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def binary_metrics(y: np.ndarray, s: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=np.float64)
    pred = (s >= threshold).astype(int)
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(auc), "accuracy": float(accuracy_score(y, pred)), "precision": float(p),
            "recall": float(r), "f1": float(f1), "eer": eer(y, s), "threshold": float(threshold),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(len(y))}


# Figuras

def plot_history(histories: Dict[str, List[Dict]], path: str) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, h in histories.items():
        e = [r["epoch"] for r in h]
        ax[0].plot(e, [r["train_loss"] for r in h], marker="o", ms=3, label=name)
        ax[1].plot(e, [r["val_auc_buena"] for r in h], marker="o", ms=3, label=f"{name}")
        ax[2].plot(e, [r["val_auc_mala"] for r in h], marker="s", ms=3, label=f"{name}")
    for a, t in zip(ax, ("Focal Loss (train)", "AUC val · calidad buena", "AUC val · calidad mala")):
        a.set_title(t)
        a.set_xlabel("Época")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[1].set_ylim(0, 1.02)
    ax[2].set_ylim(0, 1.02)
    fig.tight_layout()
    save_fig(fig, path)
    return fig


def plot_roc(curves: Dict[str, tuple], path: str, title: str) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y, s) in curves.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    save_fig(fig, path)
    return fig


def plot_confusions(cms: Dict[str, np.ndarray], path: str) -> matplotlib.figure.Figure:
    n = len(cms)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for a, (name, cm) in zip(axes.ravel(), cms.items()):
        a.axis("on")
        a.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                a.text(j, i, int(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
        a.set_xticks([0, 1], ["Real", "Fake"])
        a.set_yticks([0, 1], ["Real", "Fake"])
        a.set_xlabel("Predicción")
        a.set_ylabel("Etiqueta")
        a.set_title(name, fontsize=9)
    fig.tight_layout()
    save_fig(fig, path)
    return fig


def plot_quality_curve(table: Dict[str, Dict[str, float]], levels: Sequence[str], metric: str,
                       path: str) -> matplotlib.figure.Figure:
    """table[modelo][nivel] = valor de la métrica."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(levels))
    for name, row in table.items():
        ax.plot(x, [row[l] for l in levels], marker="o", lw=2, label=name)
    ax.set_xticks(x, levels)
    ax.set_xlabel("Calidad de la imagen de prueba")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1.02)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_title(f"{metric.upper()} a nivel de video según la calidad")
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, path)
    return fig


# Intervalos de confianza (bootstrap)
# Remuestreo por GRUPOS de identidad (cluster bootstrap): los videos de una misma persona
# Así el intervalo no es artificialmente estrecho.

def _cluster_resample(groups: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    uniq = np.unique(groups)
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    pos = {g: np.flatnonzero(groups == g) for g in uniq}
    return np.concatenate([pos[g] for g in pick])


def bootstrap_auc(y: np.ndarray, s: np.ndarray, groups: Optional[np.ndarray] = None,
                  n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> Dict[str, float]:
    """AUC con IC (1-alpha) por percentiles."""
    y, s = np.asarray(y), np.asarray(s)
    g = np.arange(len(y)) if groups is None else np.asarray(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        i = _cluster_resample(g, rng)
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    vals = np.array(vals)
    return {"auc": float(roc_auc_score(y, s)), "ic_inf": float(np.percentile(vals, 100 * alpha / 2)),
            "ic_sup": float(np.percentile(vals, 100 * (1 - alpha / 2))), "n_boot_validos": int(len(vals))}


def paired_bootstrap_auc_diff(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
                              groups: Optional[np.ndarray] = None, n_boot: int = 2000, seed: int = 42,
                              alpha: float = 0.05) -> Dict[str, float]:
    """Diferencia AUC(a) - AUC(b) sobre los MISMOS videos, con IC y p-valor bilateral (bootstrap)."""
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    g = np.arange(len(y)) if groups is None else np.asarray(groups)
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n_boot):
        i = _cluster_resample(g, rng)
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], s_a[i]) - roc_auc_score(y[i], s_b[i]))
    d = np.array(d)
    obs = float(roc_auc_score(y, s_a) - roc_auc_score(y, s_b))
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"diferencia": obs, "ic_inf": float(np.percentile(d, 100 * alpha / 2)),
            "ic_sup": float(np.percentile(d, 100 * (1 - alpha / 2))), "p_valor": p}
