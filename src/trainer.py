"""Entrenamiento y evaluación de un experimento."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

import backend as bk
from augment import (QualityAugmenter, RobustNormalizer, apply_degradation, bilateral_diff_map)
from backend import xp
from layers import BatchNorm2d, Module, softmax
from losses import FocalLoss
from metrics import video_scores
from optim import AdamW, CosineLR, clip_grad_norm

Array = Any


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    clip_norm: float = 5.0
    frames_per_video: int = 8
    patience: int = 7
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    select_on: str = "val_auc_buena"
    val_bad_level: str = "mala"
    seed: int = 42
    diff_mode: str = "pixel"
    yaw_mask: float = 0.0  # >0: apaga la rama lateral si |yaw| supera este umbral
    balance_classes: bool = False  # True: igual nº de frames real/fake por época (FF++ es 1:4)


def iter_modules(m: Module) -> Iterator[Module]:
    yield m
    for c in m.children():
        yield from iter_modules(c)


def get_state(model: Module) -> List[np.ndarray]:
    st = [bk.to_numpy(p.data).copy() for p in model.parameters()]
    for m in iter_modules(model):
        if isinstance(m, BatchNorm2d):
            st += [bk.to_numpy(m.run_mean).copy(), bk.to_numpy(m.run_var).copy()]
    return st


def set_state(model: Module, st: List[np.ndarray]) -> None:
    params = model.parameters()
    for p, a in zip(params, st):
        assert p.data.shape == a.shape
        p.data[...] = bk.asarray(a)
    k = len(params)
    for m in iter_modules(model):
        if isinstance(m, BatchNorm2d):
            m.run_mean, m.run_var = bk.asarray(st[k]), bk.asarray(st[k + 1])
            k += 2
    assert k == len(st), "El estado no corresponde a la arquitectura"


class Experiment:
    """Un modelo + sus datos + su régimen de aumento."""

    def __init__(self, name: str, model: Module, cfg: TrainConfig, cache: Dict[str, np.ndarray],
                 split_videos: Dict[str, np.ndarray], face_norm: RobustNormalizer,
                 diff_norm: RobustNormalizer, augmenter: Optional[QualityAugmenter],
                 levels: Dict[str, Dict[str, float]], out_dir: str) -> None:
        self.name, self.model, self.cfg = name, model, cfg
        self.cache = cache
        self.faces, self.fvid = cache["faces"], cache["frame_video"]
        self.labels = cache["labels"]
        self.split_videos = split_videos
        self.face_norm, self.diff_norm, self.aug = face_norm, diff_norm, augmenter
        self.levels, self.out_dir = levels, out_dir
        self.crit = FocalLoss(cfg.focal_gamma, cfg.focal_alpha)
        self.opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = CosineLR(self.opt, cfg.epochs, eta_min=1e-7)
        self.history: List[Dict[str, float]] = []
        self.best_state: Optional[List[np.ndarray]] = None
        self.frames_of = {int(v): np.flatnonzero(self.fvid == v) for v in np.unique(self.fvid)}

    # ── preparación del lote ─────────────────────────────────────
    def prepare(self, idx: np.ndarray, augment: bool, spec: Optional[Dict] = None, seed: int = 0,
                cache: Optional[Dict[str, np.ndarray]] = None):
        """Índices de frame -> (rostro normalizado, mapa de asimetría normalizado, máscara lateral)."""
        c = self.cache if cache is None else cache
        idx = np.sort(idx) if isinstance(c["faces"], np.memmap) else idx  # lectura secuencial del disco
        x = bk.asarray(np.asarray(c["faces"][idx])).transpose(0, 3, 1, 2) / 255.0  # (B,3,224,224) en [0,1]
        if augment and self.aug is not None:
            x = self.aug(x)
        if spec:
            x = apply_degradation(x, spec, seed)
        d = bilateral_diff_map(x, self.cfg.diff_mode)  # (B,3,224,112)
        lmask = None
        if self.cfg.yaw_mask > 0 and "frame_yaw" in c:
            lmask = (np.abs(c["frame_yaw"][idx]) <= self.cfg.yaw_mask).astype(np.float32)
        return self.face_norm(x), self.diff_norm(d), lmask, idx

    # ── entrenamiento ────────────────────────────────────────────
    def train_epoch(self, rng: np.random.RandomState) -> float:
        self.model.train()
        tr = [int(v) for v in self.split_videos["train"] if int(v) in self.frames_of]
        k = {0: self.cfg.frames_per_video, 1: self.cfg.frames_per_video}
        if self.cfg.balance_classes:  # mismo nº esperado de frames por clase en cada época
            n_r = sum(self.labels[v] == 0 for v in tr)
            n_f = sum(self.labels[v] == 1 for v in tr)
            k[1] = max(1, int(round(self.cfg.frames_per_video * n_r / max(n_f, 1))))
        idx = []
        for v in tr:
            fr = self.frames_of[v]
            idx.append(rng.choice(fr, size=min(k[int(self.labels[v])], len(fr)), replace=False))
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        bs, losses, n = self.cfg.batch_size, 0.0, 0
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            if len(b) < 2:  # BatchNorm necesita al menos 2 muestras
                continue
            face, diff, lmask, b = self.prepare(b, augment=True)
            y = xp.asarray(self.labels[self.fvid[b]])
            logits = self.model(face, diff, lmask)
            loss, dl = self.crit(logits, y)
            if not np.isfinite(loss):
                raise FloatingPointError(f"Pérdida no finita en {self.name}")
            self.model.backward(dl)
            clip_grad_norm(self.model.parameters(), self.cfg.clip_norm)
            self.opt.step()
            self.model.release()  # libera activaciones antes del siguiente lote
            losses += loss * len(b)
            n += len(b)
        return losses / max(n, 1)

    # ── inferencia ───────────────────────────────────────────────
    def predict_frames(self, frame_idx: np.ndarray, spec: Optional[Dict] = None, seed: int = 1234,
                       cache: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """P(fake) por frame, en el mismo orden que frame_idx."""
        self.model.eval()
        out = np.empty(len(frame_idx), np.float64)
        bs = self.cfg.eval_batch_size
        for k, s in enumerate(range(0, len(frame_idx), bs)):
            chunk = frame_idx[s:s + bs]
            face, diff, lmask, used = self.prepare(chunk, augment=False, spec=spec, seed=seed + k, cache=cache)
            p = bk.to_numpy(softmax(self.model(face, diff, lmask), axis=1)[:, 1])
            out[s:s + len(chunk)] = p[np.argsort(np.argsort(chunk))] if used is not chunk else p
            self.model.release()
        return out

    def video_predictions(self, split: str, spec: Optional[Dict] = None):
        vids = self.split_videos[split]
        fidx = np.concatenate([self.frames_of[int(v)] for v in vids])
        p = self.predict_frames(fidx, spec)
        return self.labels[vids], video_scores(p, self.fvid[fidx], vids)

    def predict_on(self, cache: Dict[str, np.ndarray], spec: Optional[Dict] = None):
        """Evalúa sobre otra caché (p."""
        fv = cache["frame_video"]
        vids = np.unique(fv)
        p = self.predict_frames(np.arange(len(fv)), spec, cache=cache)
        return vids, cache["labels"][vids], video_scores(p, fv, vids)

    # ── bucle principal ──────────────────────────────────────────
    def fit(self, verbose: bool = True) -> List[Dict[str, float]]:
        from sklearn.metrics import roc_auc_score

        cfg = self.cfg
        rng = np.random.RandomState(cfg.seed)
        best, bad = -np.inf, 0
        for ep in range(1, cfg.epochs + 1):
            t0 = time.time()
            tl = self.train_epoch(rng)
            yb, sb = self.video_predictions("val")
            ym, sm = self.video_predictions("val", self.levels[cfg.val_bad_level])
            auc = lambda y, s: float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
            rec = {"epoch": ep, "train_loss": tl, "val_auc_buena": auc(yb, sb), "val_auc_mala": auc(ym, sm),
                   "lr": self.opt.lr, "seg": time.time() - t0}
            self.history.append(rec)
            self.sched.step()
            score = rec[cfg.select_on]
            improved = score > best + 1e-6
            if improved:
                best, bad = score, 0
                self.best_state = get_state(self.model)
                self.best_epoch = ep
            else:
                bad += 1
            if verbose:
                print(f"[{self.name}] ép {ep:3d} | loss {tl:.4f} | AUC val buena {rec['val_auc_buena']:.3f}"
                      f" | mala {rec['val_auc_mala']:.3f} | {rec['seg']:.0f}s{'  ✓' if improved else ''}")
            if bad >= cfg.patience:
                if verbose:
                    print(f"[{self.name}] early stopping en la época {ep}")
                break
        set_state(self.model, self.best_state)
        self.save()
        self.free_training_state()
        return self.history

    def free_training_state(self) -> None:
        """Tras entrenar solo hace falta inferencia: libera estados de AdamW (m, v), gradientes, la copia del mejor..."""
        self.opt = self.sched = None
        self.best_state = None
        for p in self.model.parameters():
            p.grad = None
        self.model.release()

    def save(self) -> str:
        d = os.path.join(self.out_dir, "checkpoints")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{self.name}_best.npz")
        np.savez(path, *get_state(self.model))
        with open(os.path.join(d, f"{self.name}_meta.json"), "w") as f:
            json.dump({"config": asdict(self.cfg), "best_epoch": getattr(self, "best_epoch", None),
                       "face_norm": self.face_norm.to_dict(), "diff_norm": self.diff_norm.to_dict(),
                       "history": self.history}, f, indent=2)
        return path
