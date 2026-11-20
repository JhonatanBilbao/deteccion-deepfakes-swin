"""Detector completo: rama Swin + rama de asimetría + MLP."""
from __future__ import annotations

import random
from typing import Any, Optional

import backend as bk
from backend import xp
from cnn import CNNBranch
from layers import Module, softmax
from mlp import MLPClassifier
from swin import SwinBranch

Array = Any


class DeepfakeDetectorScratch(Module):
    def __init__(self, swin: Optional[SwinBranch] = None, cnn: Optional[CNNBranch] = None,
                 mlp: Optional[MLPClassifier] = None, drop_branch: float = 0.1,
                 branches: str = "ambas") -> None:
        """branches: "ambas" (pipeline completo) | "solo_swin" | "solo_asimetria" (ablaciones)."""
        assert branches in ("ambas", "solo_swin", "solo_asimetria")
        self.branches = branches
        self.swin = swin if swin is not None else SwinBranch()
        self.cnn = cnn if cnn is not None else CNNBranch()
        self.mlp = mlp if mlp is not None else MLPClassifier()
        self.drop_branch = drop_branch
        self._keep_g, self._keep_l = 1.0, 1.0

    def forward(self, face: Array, diff_map: Array, lateral_mask: Optional[Array] = None) -> Array:
        """lateral_mask (B,) con 0 o 1: en 0 se apaga la rama lateral de ese frame (cabeza muy girada)."""
        face, diff_map = bk.asarray(face), bk.asarray(diff_map)
        assert face.shape[0] == diff_map.shape[0], "face y diff_map deben tener el mismo B"
        B = face.shape[0]
        fg = self.swin(face) if self.branches != "solo_asimetria" else xp.zeros((B, self.mlp_half()), dtype=bk.DTYPE)
        fl = self.cnn(diff_map) if self.branches != "solo_swin" else xp.zeros((B, self.mlp_half()), dtype=bk.DTYPE)
        self._keep_g, self._keep_l = 1.0, 1.0
        if self.training and self.drop_branch > 0 and self.branches == "ambas":
            r = random.random()
            if r < self.drop_branch / 2:
                self._keep_g = 0.0
            elif r < self.drop_branch:
                self._keep_l = 0.0
        self._d = fg.shape[1]
        self._lmask = None if lateral_mask is None else bk.asarray(lateral_mask).reshape(-1, 1).astype(fl.dtype)
        if self._lmask is not None:
            assert self._lmask.shape[0] == B, "lateral_mask debe tener B elementos"
            fl = fl * self._lmask
        fused = xp.concatenate([fg * self._keep_g, fl * self._keep_l], axis=1)  # (B, 512)
        return self.mlp(fused)  # (B, 2)

    def backward(self, dlogits: Array) -> None:
        dfused = self.mlp.backward(dlogits)  # (B, 512)
        if self.branches != "solo_asimetria":
            self.swin.backward(dfused[:, :self._d] * self._keep_g)
        if self.branches != "solo_swin":
            dl = dfused[:, self._d:] * self._keep_l
            self.cnn.backward(dl if self._lmask is None else dl * self._lmask)

    def mlp_half(self) -> int:
        return self.swin.proj_dim

    def predict_video(self, faces: Array, diff_maps: Array, chunk: int = 16,
                      lateral_mask: Optional[Array] = None) -> float:
        """faces (N,3,224,224), diff_maps (N,3,224,112) -> P(deepfake) promedio del video."""
        was_training = self.training
        self.eval()
        probs = []
        for i in range(0, faces.shape[0], chunk):
            lm = None if lateral_mask is None else lateral_mask[i:i + chunk]
            logits = self.forward(faces[i:i + chunk], diff_maps[i:i + chunk], lm)
            probs.append(softmax(logits, axis=1)[:, 1])
            self.release()
        self.train(was_training)
        return float(xp.concatenate(probs).mean())
