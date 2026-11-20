"""Auditoría del dataset (formato, frames, duplicados, partición, detección de rostros).

Uso: python verificar_dataset.py RUTA_DATASET
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (FaceAligner, assign_identity_groups, check_identity_leakage, list_videos,  # noqa: E402
                  read_frames, sample_frame_indices, split_by_identity)

N_FRAMES = 16


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _video_info(path: str) -> Dict:
    cap = cv2.VideoCapture(path)
    ok, _ = cap.read()
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    info = dict(ok=bool(ok), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=round(cap.get(cv2.CAP_PROP_FPS), 2),
                w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                codec="".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)))
    cap.release()
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--frames-rostro", type=int, default=4,
                    help="frames (de los 16 uniformes) por video para medir la detección facial; 0 = omitir")
    ap.add_argument("--salida", default="reporte_dataset.json")
    a = ap.parse_args()

    videos = list_videos(a.root)
    rep: Dict = {"raiz": os.path.abspath(a.root), "problemas": [], "avisos": []}
    lab = collections.Counter("real" if v.label == 0 else "fake" for v in videos)
    met = collections.Counter(v.method for v in videos)
    rep["conteo"] = dict(lab)
    rep["metodos"] = dict(met)
    print(f"Videos: {dict(lab)} | métodos: {dict(met)}")

    # 1) nombres
    bad_names = [v.video_id for v in videos if (v.label == 0 and len(v.identities) != 1)
                 or (v.label == 1 and len(v.identities) != 2)]
    if bad_names:
        rep["problemas"].append(f"Nombres fuera del formato FF++: {bad_names[:10]}")

    # 2) metadatos de video
    infos = {v.video_id: _video_info(v.path) for v in videos}
    unreadable = [k for k, i in infos.items() if not i["ok"]]
    short = [k for k, i in infos.items() if i["frames"] < N_FRAMES]
    if unreadable:
        rep["problemas"].append(f"Videos ilegibles: {unreadable}")
    if short:
        rep["problemas"].append(f"Videos con < {N_FRAMES} frames: {short}")
    for name, lbl in (("real", 0), ("fake", 1)):
        I = [infos[v.video_id] for v in videos if v.label == lbl]
        f = np.array([i["frames"] for i in I])
        rep[f"stats_{name}"] = {
            "frames_min_mediana_max": [int(f.min()), int(np.median(f)), int(f.max())],
            "fps": dict(collections.Counter(i["fps"] for i in I)),
            "resoluciones": dict(collections.Counter(f"{i['w']}x{i['h']}" for i in I).most_common(8)),
            "codec": dict(collections.Counter(i["codec"] for i in I)),
        }
        print(f"{name}: frames {rep[f'stats_{name}']['frames_min_mediana_max']} | fps {rep[f'stats_{name}']['fps']}"
              f" | códec {rep[f'stats_{name}']['codec']} | resol. {rep[f'stats_{name}']['resoluciones']}")

    # 3) duplicados exactos
    by_hash = collections.defaultdict(list)
    for v in videos:
        by_hash[_md5(v.path)].append(v.video_id)
    dups = [g for g in by_hash.values() if len(g) > 1]
    rep["duplicados_exactos"] = dups
    if dups:
        rep["problemas"].append(f"Duplicados exactos (mismo archivo): {dups}")

    # 4) emparejamiento fake -> real destino
    reals = {v.identities[0] for v in videos if v.label == 0}
    orphan = [v.video_id for v in videos if v.label == 1 and v.identities[0] not in reals]
    targets = sorted({v.identities[0] for v in videos if v.label == 1})
    reals_sin_fake = sorted(reals - set(targets))
    rep["fakes_sin_real_destino"] = orphan
    rep["reales_con_fakes"] = len(set(targets) & reals)
    rep["reales_sin_fakes"] = len(reals_sin_fake)
    if orphan:
        rep["avisos"].append(f"{len(orphan)} fakes cuyo video real destino no está en real/")
    if reals_sin_fake:
        rep["avisos"].append(
            f"{len(reals_sin_fake)} de {len(reals)} reales no tienen ningún fake de su identidad "
            f"(los fakes solo cubren los destinos {targets[0]}–{targets[-1]}).")

    # 5) grupos de identidad y partición
    n_groups = assign_identity_groups(videos)
    sizes = collections.Counter(v.group for v in videos)
    rep["grupos_identidad"] = {"n": n_groups, "tamaño_max": max(sizes.values())}
    splits = split_by_identity(videos, (0.70, 0.15, 0.15), seed=42)
    rep["particion"] = {k: dict(collections.Counter("real" if v.label == 0 else "fake" for v in vs))
                        for k, vs in splits.items()}
    rep["fugas"] = check_identity_leakage(splits)
    print("Grupos de identidad:", rep["grupos_identidad"])
    print("Partición por identidad (seed=42):", rep["particion"])
    print("Fugas:", rep["fugas"])
    if any(rep["fugas"].values()):
        rep["problemas"].append("Hay fugas de identidad entre conjuntos")
    for k, c in rep["particion"].items():
        if len(c) < 2:
            rep["problemas"].append(f"El conjunto {k} no tiene ambas clases: {c}")

    # 6) detección facial MediaPipe (misma alineación del pipeline)
    if a.frames_rostro > 0:
        aligner = FaceAligner(224)
        det = {}
        for v in videos:
            idx = sample_frame_indices(infos[v.video_id]["frames"], N_FRAMES)
            idx = [idx[i] for i in np.linspace(0, N_FRAMES - 1, a.frames_rostro).round().astype(int)]
            frames = read_frames(v.path, idx)
            det[v.video_id] = sum(aligner.align(fr)[0] is not None for _, fr in frames) / max(len(idx), 1)
        aligner.close()
        rates = np.array(list(det.values()))
        low = sorted(k for k, r in det.items() if r < 0.5)
        rep["deteccion_rostro"] = {"tasa_media": round(float(rates.mean()), 4),
                                   "videos_<50%": low, "frames_por_video_probados": a.frames_rostro}
        print(f"Detección facial MediaPipe: {rates.mean():.1%} de los frames probados | videos < 50%: {low}")
        if low:
            rep["avisos"].append(f"Videos con poca detección facial: {low}")

    json.dump(rep, open(a.salida, "w"), indent=2, ensure_ascii=False)
    print("\nPROBLEMAS:", rep["problemas"] or "ninguno")
    print("AVISOS:", rep["avisos"] or "ninguno")
    print("Reporte:", a.salida)
    sys.exit(1 if rep["problemas"] else 0)


if __name__ == "__main__":
    main()
