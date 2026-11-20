"""Diagnóstico del mapa de asimetría (correlaciones y sonda lineal).

Uso: python diagnostico_asimetria.py --dataset RUTA --cache rostros.npz --out salida
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("DF_BACKEND", "numpy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import backend as bk  # noqa: E402
from augment import DIFF_MODES, bilateral_diff_map  # noqa: E402
from data import load_videos, split_official  # noqa: E402

random.seed(SEED)
np.random.seed(SEED)
bk.set_seed(SEED)
FRONTAL = 0.20  # |yaw| por debajo de esto se considera rostro frontal


def frame_features(faces: np.ndarray, mode: str, bs: int = 64):
    """Energía media del mapa, mapa promediado en bloques de 16x16 (3x14x7) y asimetría de brillo."""
    E, P, B = [], [], []
    for s in range(0, len(faces), bs):
        x = bk.asarray(faces[s:s + bs]).transpose(0, 3, 1, 2) / 255.0
        d = bk.to_numpy(bilateral_diff_map(x, mode))  # (b,3,224,112)
        E.append(d.mean(axis=(1, 2, 3)))
        P.append(d.reshape(d.shape[0], 3, 14, 16, 7, 16).mean(axis=(3, 5)).reshape(d.shape[0], -1))
        xn = bk.to_numpy(x)
        B.append(np.abs(xn[..., :112].mean(axis=(1, 2, 3)) - xn[..., 112:].mean(axis=(1, 2, 3))))
    return np.concatenate(E), np.concatenate(P), np.concatenate(B)


def video_auc(labels: np.ndarray, frame_score: np.ndarray, fvid: np.ndarray, vids: np.ndarray,
              mask: np.ndarray | None = None) -> float:
    keep = np.ones(len(fvid), bool) if mask is None else mask
    sc, ys = [], []
    for v in vids:
        m = (fvid == v) & keep
        if m.any():
            sc.append(frame_score[m].mean())
            ys.append(labels[v])
    ys = np.array(ys)
    return float(roc_auc_score(ys, sc)) if len(np.unique(ys)) == 2 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="salida_diag_asimetria")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    videos = load_videos(a.dataset)
    splits = split_official(videos)
    c = dict(np.load(a.cache, allow_pickle=False))
    assert list(c["paths"]) == [v.path for v in videos], "La caché no corresponde a este dataset"
    assert "frame_yaw" in c, "Caché sin índice de giro: regenerarla con ALIGN_VERSION ≥ 3"
    fvid, yaw, labels = c["frame_video"], np.abs(c["frame_yaw"]), c["labels"]
    vid_of = {v.path: i for i, v in enumerate(videos)}
    ids = {k: np.array([vid_of[v.path] for v in vs]) for k, vs in splits.items()}
    in_split = {k: np.isin(fvid, ids[k]) for k in ids}
    frontal = yaw < FRONTAL
    rep = {"n_frames": int(len(fvid)), "frac_frontal": float(frontal.mean()),
           "videos": {k: {"real": int((labels[v] == 0).sum()), "fake": int((labels[v] == 1).sum())}
                      for k, v in ids.items()}, "modos": {}}
    print(f"{len(fvid)} frames | frontales (|yaw|<{FRONTAL}): {frontal.mean():.0%}")

    feats = {}
    for mode in DIFF_MODES:
        E, P, B = frame_features(c["faces"], mode)
        feats[mode] = (E, P)
        r = {"spearman_E_vs_yaw": float(spearmanr(E, yaw).correlation),
             "spearman_E_vs_brillo": float(spearmanr(E, B).correlation)}
        r["auc_energia"] = {k: video_auc(labels, E, fvid, ids[k]) for k in ids}
        r["auc_energia_frontal"] = {k: video_auc(labels, E, fvid, ids[k], frontal) for k in ids}
        held = np.concatenate([ids["val"], ids["test"]])
        for name, mask in (("todos", np.ones_like(frontal)), ("frontales", frontal)):
            tr = in_split["train"] & mask
            mu, sd = P[tr].mean(0), P[tr].std(0) + 1e-8  # estadísticos SOLO de train
            clf = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced", random_state=SEED)
            clf.fit((P[tr] - mu) / sd, labels[fvid[tr]])
            prob = clf.predict_proba((P - mu) / sd)[:, 1]
            r[f"sonda_lineal_{name}"] = {"train": video_auc(labels, prob, fvid, ids["train"], mask),
                                         "val": video_auc(labels, prob, fvid, ids["val"], mask),
                                         "test": video_auc(labels, prob, fvid, ids["test"], mask),
                                         "val+test": video_auc(labels, prob, fvid, held, mask)}
        rep["modos"][mode] = r
        print(f"\n[{mode}] ρ(E,|yaw|)={r['spearman_E_vs_yaw']:+.2f}  ρ(E,brillo)={r['spearman_E_vs_brillo']:+.2f}")
        print("  AUC energía por video      ", {k: round(v, 3) for k, v in r["auc_energia"].items()})
        print("  AUC energía (frontales)    ", {k: round(v, 3) for k, v in r["auc_energia_frontal"].items()})
        print("  Sonda lineal (todos)       ", {k: round(v, 3) for k, v in r["sonda_lineal_todos"].items()})
        print("  Sonda lineal (frontales)   ", {k: round(v, 3) for k, v in r["sonda_lineal_frontales"].items()})

    # Figuras
    fig, ax = plt.subplots(1, len(DIFF_MODES), figsize=(5.2 * len(DIFF_MODES), 4))
    for a_, mode in zip(np.atleast_1d(ax), DIFF_MODES):
        E = feats[mode][0]
        a_.boxplot([E[labels[fvid] == 0], E[labels[fvid] == 1]], showfliers=False)
        a_.set_xticks([1, 2], ["Real", "Fake"])
        a_.set_title(f"Energía del mapa · {mode}")
        a_.set_ylabel("media de |izq. espejo − der.|")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_asim_01_energia_por_clase.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(1, len(DIFF_MODES), figsize=(5.2 * len(DIFF_MODES), 4))
    for a_, mode in zip(np.atleast_1d(ax), DIFF_MODES):
        E = feats[mode][0]
        for lab, col, nm in ((0, "tab:blue", "Real"), (1, "tab:red", "Fake")):
            m = labels[fvid] == lab
            a_.scatter(yaw[m], E[m], s=4, alpha=0.35, c=col, label=nm)
        a_.axvline(FRONTAL, ls="--", c="gray", lw=1)
        a_.set_xlabel("|índice de giro (yaw)|")
        a_.set_ylabel("energía del mapa")
        a_.set_title(f"{mode}: ρ={rep['modos'][mode]['spearman_E_vs_yaw']:+.2f}")
        a_.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_asim_02_energia_vs_giro.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    json.dump(rep, open(os.path.join(a.out, "diagnostico_asimetria.json"), "w"), indent=2, ensure_ascii=False)
    print("\nGuardado en", a.out)


if __name__ == "__main__":
    main()
