"""Protocolo FF++ en tres fases: preprocesar, entrenar y evaluar, analizar."""
from __future__ import annotations

import glob
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import backend as bk
from augment import (DIFF_MODES, ISOLATED_DEGRADATIONS, QUALITY_LEVELS, AugConfig, QualityAugmenter,
                     RobustNormalizer, bilateral_diff_map)
from data import (VideoItem, assign_identity_groups, build_face_cache_dir, check_identity_leakage,
                  detect_layout, list_videos_celebdf, load_face_cache_dir, load_videos, split_by_identity,
                  split_official)

EXPERIMENTS: Dict[str, Tuple[str, bool]] = {  # nombre -> (ramas, ¿aumento de calidad?)
    "robusto_con_aumento": ("ambas", True),  # modelo propuesto
    "base_sin_aumento": ("ambas", False),
    "ablacion_solo_swin": ("solo_swin", True),
    "ablacion_solo_asimetria": ("solo_asimetria", True),
}
SYNTH_TEST = {**QUALITY_LEVELS, "jpeg_q30_no_visto": ISOLATED_DEGRADATIONS["jpeg_q30_no_visto"]}


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    bk.set_seed(seed)


@dataclass
class PrepConfig:
    dataset_root: str
    out_dir: str
    celebdf_root: Optional[str] = None
    n_grid: int = 16
    budget_train_real: int = 16
    budget_train_fake: int = 4
    budget_val: int = 8
    budget_test: int = 16
    workers: int = 4
    norm_fit_frames: int = 2048
    max_videos_per_split: Optional[int] = None  # solo para pruebas rápidas
    seed: int = 42


def _limit(vs: List[VideoItem], n: Optional[int], seed: int) -> List[VideoItem]:
    if n is None or len(vs) <= n:
        return vs
    rng = np.random.RandomState(seed)
    reals = [v for v in vs if v.label == 0]
    fakes = [v for v in vs if v.label == 1]
    k = n // 2
    pick = [reals[i] for i in sorted(rng.choice(len(reals), min(k, len(reals)), replace=False))] + \
           [fakes[i] for i in sorted(rng.choice(len(fakes), min(n - k, len(fakes)), replace=False))]
    return pick


def preprocesar(cfg: PrepConfig) -> Dict:
    """Fase 1."""
    set_seeds(cfg.seed)
    t0 = time.time()
    os.makedirs(cfg.out_dir, exist_ok=True)
    layout = detect_layout(cfg.dataset_root)
    videos = load_videos(cfg.dataset_root, "c23")
    try:
        splits = split_official(videos)
        protocolo = "oficial FF++ (720/140/140)"
    except ValueError:
        splits = split_by_identity(videos, (0.70, 0.15, 0.15), cfg.seed)
        protocolo = "por identidad (videos fuera de la lista oficial)"
    splits = {k: _limit(vs, cfg.max_videos_per_split, cfg.seed) for k, vs in splits.items()}
    leak = check_identity_leakage(splits)
    assert not any(leak.values()), f"Fuga de identidad: {leak}"
    print(f"Estructura: {layout} | protocolo: {protocolo}")
    for k, vs in splits.items():
        print(f"  {k:5s}: {sum(v.label == 0 for v in vs):4d} reales | {sum(v.label == 1 for v in vs):5d} fakes")

    # Caché principal C23: train + val + test (un solo índice de videos)
    allv = splits["train"] + splits["val"] + splits["test"]
    budget = {"val": cfg.budget_val, "test": cfg.budget_test}
    budgets = [(cfg.budget_train_real if v.label == 0 else cfg.budget_train_fake) if v.split == "train"
               else budget[v.split] for v in allv]
    print("Extrayendo rostros C23…")
    c23 = build_face_cache_dir(allv, os.path.join(cfg.out_dir, "cache_c23"), cfg.n_grid, budgets, cfg.workers)
    caches = {"c23": "cache_c23"}

    # C40 real: mismos videos de test en compresión alta (solo con la estructura oficial)
    extra = {}
    if layout == "ffpp":
        try:
            v40 = {v.video_id: v for v in load_videos(cfg.dataset_root, "c40")}
            t40 = [v40[v.video_id] for v in splits["test"] if v.video_id in v40]
            if len(t40) < len(splits["test"]):
                print(f"AVISO: C40 tiene {len(t40)}/{len(splits['test'])} videos de test")
            if t40:
                for v in t40:
                    v.split = "test"
                assign_identity_groups(t40)
                print("Extrayendo rostros C40 (test)…")
                build_face_cache_dir(t40, os.path.join(cfg.out_dir, "cache_c40_test"), cfg.n_grid,
                                     [cfg.budget_test] * len(t40), cfg.workers)
                caches["c40"] = "cache_c40_test"
                extra["c40"] = t40
        except FileNotFoundError:
            print("Sin versión C40 en el dataset: se omite la prueba de mala calidad real")
    if cfg.celebdf_root:
        cdf = list_videos_celebdf(cfg.celebdf_root)
        assign_identity_groups(cdf)
        print(f"Extrayendo rostros Celeb-DF v2 ({len(cdf)} videos de test)…")
        build_face_cache_dir(cdf, os.path.join(cfg.out_dir, "cache_celebdf_test"), cfg.n_grid,
                             [cfg.budget_test] * len(cdf), cfg.workers)
        caches["celebdf"] = "cache_celebdf_test"
        extra["celebdf"] = cdf

    # Normalización robusta: estadísticos solo de frames de train (rostro y ambos modos del mapa)
    rng = np.random.RandomState(cfg.seed)
    tr_videos = np.array([i for i, v in enumerate(allv) if v.split == "train"])
    tr_frames = np.flatnonzero(np.isin(c23["frame_video"], tr_videos))
    fit_idx = np.sort(rng.choice(tr_frames, size=min(cfg.norm_fit_frames, len(tr_frames)), replace=False))
    faces, maps = [], {m: [] for m in DIFF_MODES}
    for s in range(0, len(fit_idx), 64):
        x = bk.asarray(np.asarray(c23["faces"][fit_idx[s:s + 64]])).transpose(0, 3, 1, 2) / 255.0
        faces.append(bk.to_numpy(x))
        for m in DIFF_MODES:
            maps[m].append(bk.to_numpy(bilateral_diff_map(x, m)))
    norm = {"rostro": RobustNormalizer(clip=5.0).fit(np.concatenate(faces)).to_dict(),
            **{f"mapa_{m}": RobustNormalizer(clip=5.0).fit(np.concatenate(maps[m])).to_dict() for m in DIFF_MODES},
            "n_frames_train": int(len(fit_idx))}

    def vrec(v: VideoItem) -> Dict:
        return {"video_id": v.video_id, "label": int(v.label), "method": v.method, "split": v.split,
                "group": int(v.group), "path": v.path}

    manifest = {"protocolo": protocolo, "layout": layout, "caches": caches, "normalizacion": norm,
                "videos_c23": [vrec(v) for v in allv],
                **{f"videos_{k}": [vrec(v) for v in vs] for k, vs in extra.items()},
                "fugas": leak, "config": asdict(cfg), "minutos": (time.time() - t0) / 60}
    json.dump(manifest, open(os.path.join(cfg.out_dir, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    print(f"Fase 1 completa en {manifest['minutos']:.1f} min → {cfg.out_dir}/manifest.json")
    return manifest


# Fase 2
@dataclass
class TrainSetup:
    prep_dir: str
    out_dir: str
    experiment: str = "robusto_con_aumento"
    seed: int = 42
    diff_mode: str = "residual"
    yaw_mask: float = 0.0
    epochs: int = 15
    patience: int = 4
    frames_per_video: int = 8  # frames de cada real por época; los fakes se equilibran
    lr: float = 1e-4
    swin_lr_mult: float = 0.1
    batch_size: int = 16
    eval_batch_size: int = 32
    swin_pretrained: bool = True
    weights_path: Optional[str] = None
    extra: Dict = field(default_factory=dict)


def make_model(branches: str, pretrained: bool, weights_path: Optional[str], lr_mult: float, seed: int):
    from cnn import CNNBranch
    from detector import DeepfakeDetectorScratch
    from mlp import MLPClassifier
    from pretrained import download, load_swin_imagenet, set_lr_mult
    from swin import SwinBranch

    bk.set_seed(seed)
    swin = SwinBranch()
    if pretrained:
        load_swin_imagenet(swin, weights_path or download(), verbose=False)
        for m in [swin.patch_embed, *swin.layers, swin.norm]:
            set_lr_mult(m, lr_mult)
    return DeepfakeDetectorScratch(swin, CNNBranch(3, (32, 64, 128), 256), MLPClassifier(512, 256, 2, 0.3),
                                   drop_branch=0.1, branches=branches)


def entrenar_y_evaluar(st: TrainSetup) -> str:
    """Fase 2."""
    from trainer import Experiment, TrainConfig

    assert st.experiment in EXPERIMENTS, f"experimento desconocido: {st.experiment}"
    set_seeds(st.seed)
    t0 = time.time()
    os.makedirs(st.out_dir, exist_ok=True)
    man = json.load(open(os.path.join(st.prep_dir, "manifest.json")))
    c23 = load_face_cache_dir(os.path.join(st.prep_dir, man["caches"]["c23"]))
    vids = man["videos_c23"]
    has_face = c23["n_detect"] > 0  # videos sin ningún rostro detectado no se pueden puntuar
    split_videos = {k: np.array([i for i, v in enumerate(vids) if v["split"] == k and has_face[i]])
                    for k in ("train", "val", "test")}
    n_sin = int((~has_face).sum())
    if n_sin:
        print(f"AVISO: {n_sin} videos sin rostros detectados quedan fuera (se reporta en meta)")
    face_norm = RobustNormalizer.from_dict(man["normalizacion"]["rostro"])
    diff_norm = RobustNormalizer.from_dict(man["normalizacion"][f"mapa_{st.diff_mode}"])
    branches, use_aug = EXPERIMENTS[st.experiment]
    model = make_model(branches, st.swin_pretrained, st.weights_path, st.swin_lr_mult, st.seed)
    tcfg = TrainConfig(epochs=st.epochs, batch_size=st.batch_size, eval_batch_size=st.eval_batch_size, lr=st.lr,
                       frames_per_video=st.frames_per_video, patience=st.patience, focal_alpha=0.5,
                       seed=st.seed, diff_mode=st.diff_mode, yaw_mask=st.yaw_mask, balance_classes=True)
    name = f"{st.experiment}_s{st.seed}"
    bk.set_seed(st.seed)
    exp = Experiment(name, model, tcfg, c23, split_videos, face_norm, diff_norm,
                     QualityAugmenter(AugConfig()) if use_aug else None, QUALITY_LEVELS, st.out_dir)
    exp.fit()

    rows = []

    def add(cond: str, vlist: List[Dict], idx: np.ndarray, scores: np.ndarray) -> None:
        for i, s in zip(idx, scores):
            v = vlist[int(i)]
            rows.append({"experimento": st.experiment, "semilla": st.seed, "condicion": cond,
                         "video_id": v["video_id"], "metodo": v["method"], "grupo": v["group"],
                         "etiqueta": v["label"], "score": float(s)})

    yv, sv = exp.video_predictions("val")
    add("val_c23", vids, split_videos["val"], sv)
    for cond, spec in SYNTH_TEST.items():
        _, s = exp.video_predictions("test", spec or None)
        add("c23" if cond == "buena" else f"c23_{cond}", vids, split_videos["test"], s)
    for key in ("c40", "celebdf"):
        if key in man["caches"]:
            cache = load_face_cache_dir(os.path.join(st.prep_dir, man["caches"][key]))
            vv, _, s = exp.predict_on(cache)
            add(key, man[f"videos_{key}"], vv, s)
    import pandas as pd

    df = pd.DataFrame(rows)
    path = os.path.join(st.out_dir, f"scores_{name}.csv")
    df.to_csv(path, index=False)
    json.dump({"setup": asdict(st), "historial": exp.history, "mejor_epoca": exp.best_epoch,
               "minutos": (time.time() - t0) / 60, "backend": "gpu" if bk.ON_GPU else "cpu",
               "videos_sin_rostro": n_sin},
              open(os.path.join(st.out_dir, f"meta_{name}.json"), "w"), indent=1)
    from sklearn.metrics import roc_auc_score
    for cond, g in df.groupby("condicion"):
        if g["etiqueta"].nunique() == 2:
            print(f"  {cond:24s} AUC = {roc_auc_score(g['etiqueta'], g['score']):.3f}  ({len(g)} videos)")
    print(f"Fase 2 ({name}) completa en {(time.time() - t0) / 60:.1f} min → {path}")
    return path


# Fase 3
COND_ORDER = ["c23", "c23_regular", "c23_mala", "c23_muy_mala", "c23_jpeg_q30_no_visto", "c40", "celebdf"]
COMPARISONS = [("robusto_con_aumento", "base_sin_aumento", "efecto del aumento de calidad"),
               ("robusto_con_aumento", "ablacion_solo_swin", "aporte de la rama de asimetría"),
               ("robusto_con_aumento", "ablacion_solo_asimetria", "aporte de la rama Swin")]


def analizar(score_globs: Sequence[str], out_dir: str, n_boot: int = 2000) -> Dict:
    """Fase 3."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    from metrics import binary_metrics, bootstrap_auc, paired_bootstrap_auc_diff, youden_threshold

    files = sorted({f for g in score_globs for f in glob.glob(g, recursive=True)})
    assert files, f"No hay CSV de puntuaciones en {score_globs}"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["experimento", "semilla", "condicion", "video_id"])
    os.makedirs(os.path.join(out_dir, "figuras"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tablas"), exist_ok=True)
    conds = [c for c in COND_ORDER if c in set(df["condicion"])]
    exps = [e for e in EXPERIMENTS if e in set(df["experimento"])]
    print(f"{len(files)} archivos | experimentos: {exps} | semillas: {sorted(df['semilla'].unique())}")

    # 1) AUC por experimento x condición: por semilla (IC bootstrap) y media ± d.e. entre semillas
    rows = []
    for (e, s, c), g in df[df["condicion"].isin(conds)].groupby(["experimento", "semilla", "condicion"]):
        b = bootstrap_auc(g["etiqueta"].values, g["score"].values, g["grupo"].values, n_boot, seed=42)
        rows.append({"experimento": e, "semilla": s, "condicion": c, **b, "n_videos": len(g)})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(out_dir, "tablas", "auc_por_semilla.csv"), index=False)
    summ = per_seed.groupby(["experimento", "condicion"]).agg(
        auc_media=("auc", "mean"), auc_de=("auc", "std"), n_semillas=("auc", "size"),
        ic_inf_medio=("ic_inf", "mean"), ic_sup_medio=("ic_sup", "mean")).reset_index()
    summ.to_csv(os.path.join(out_dir, "tablas", "auc_resumen.csv"), index=False)
    pivot = summ.pivot(index="experimento", columns="condicion", values="auc_media").reindex(index=exps, columns=conds)
    pivot.round(4).to_csv(os.path.join(out_dir, "tablas", "auc_tabla_principal.csv"))
    print("\nAUC medio (entre semillas) por condición:\n", pivot.round(3).to_string())

    # 2) Comparaciones pareadas (mismos videos, misma semilla)
    comp = []
    for a, b_, what in COMPARISONS:
        for c in conds:
            for s in sorted(df["semilla"].unique()):
                ga = df[(df.experimento == a) & (df.semilla == s) & (df.condicion == c)].set_index("video_id")
                gb = df[(df.experimento == b_) & (df.semilla == s) & (df.condicion == c)].set_index("video_id")
                common = ga.index.intersection(gb.index)
                if len(common) == 0 or ga.loc[common, "etiqueta"].nunique() < 2:
                    continue
                r = paired_bootstrap_auc_diff(ga.loc[common, "etiqueta"].values, ga.loc[common, "score"].values,
                                              gb.loc[common, "score"].values, ga.loc[common, "grupo"].values,
                                              n_boot, seed=42)
                comp.append({"comparacion": f"{a} − {b_}", "pregunta": what, "condicion": c, "semilla": s, **r})
    comp = pd.DataFrame(comp, columns=["comparacion", "pregunta", "condicion", "semilla", "diferencia",
                                       "ic_inf", "ic_sup", "p_valor"])
    comp.to_csv(os.path.join(out_dir, "tablas", "comparaciones_pareadas.csv"), index=False)

    # 3) Métricas con umbral (fijado en validación C23 de cada modelo): C23 y C40
    thr_rows = []
    for (e, s), g in df.groupby(["experimento", "semilla"]):
        v = g[g.condicion == "val_c23"]
        if v["etiqueta"].nunique() < 2:
            continue
        thr = youden_threshold(v["etiqueta"].values, v["score"].values)
        for c in [c for c in ("c23", "c40", "celebdf") if c in conds]:
            t = g[g.condicion == c]
            m = binary_metrics(t["etiqueta"].values, t["score"].values, thr)
            thr_rows.append({"experimento": e, "semilla": s, "condicion": c, **m})
    pd.DataFrame(thr_rows if thr_rows else [{"experimento": None}]).dropna(how="all").to_csv(os.path.join(out_dir, "tablas", "metricas_con_umbral.csv"), index=False)

    # 4) Figuras (300 dpi)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(conds))
    for k, e in enumerate(exps):
        sub = summ[summ.experimento == e].set_index("condicion").reindex(conds)
        ax.errorbar(x + (k - len(exps) / 2) * 0.06, sub["auc_media"], yerr=sub["auc_de"].fillna(0),
                    marker="o", capsize=3, lw=1.6, label=e)
    ax.set_xticks(x, conds, rotation=25, ha="right")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_ylabel("AUC (media ± d.e. entre semillas)")
    ax.set_title("AUC por condición de test")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "figuras", "fig_ffpp_01_auc_por_condicion.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    seed0 = int(sorted(df["semilla"].unique())[0])
    roc_conds = [c for c in ("c23", "c40", "celebdf") if c in conds]
    fig, axs = plt.subplots(1, len(roc_conds), figsize=(4.6 * len(roc_conds), 4.2), squeeze=False)
    for a_, c in zip(axs[0], roc_conds):
        for e in exps:
            g = df[(df.experimento == e) & (df.semilla == seed0) & (df.condicion == c)]
            if g["etiqueta"].nunique() == 2:
                f, t, _ = roc_curve(g["etiqueta"], g["score"])
                a_.plot(f, t, lw=1.5, label=f"{e} ({roc_auc_score(g['etiqueta'], g['score']):.3f})")
        a_.plot([0, 1], [0, 1], "--", c="gray", lw=1)
        a_.set_title(f"ROC: {c}: semilla {seed0}")
        a_.set_xlabel("FPR")
        a_.set_ylabel("TPR")
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figuras", "fig_ffpp_02_roc.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    if len(comp):
        cm = comp.groupby(["comparacion", "condicion"]).agg(dif=("diferencia", "mean"), lo=("ic_inf", "mean"),
                                                            hi=("ic_sup", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(8, 4.2))
        comps = list(dict.fromkeys(cm["comparacion"]))
        for k, cname in enumerate(comps):
            sub = cm[cm.comparacion == cname].set_index("condicion").reindex(conds).dropna()
            xs = np.array([conds.index(c) for c in sub.index]) + (k - len(comps) / 2) * 0.12
            ax.errorbar(xs, sub["dif"], yerr=[sub["dif"] - sub["lo"], sub["hi"] - sub["dif"]], fmt="o",
                        capsize=3, label=cname)
        ax.axhline(0, c="gray", lw=1)
        ax.set_xticks(np.arange(len(conds)), conds, rotation=25, ha="right")
        ax.set_ylabel("Δ AUC (IC 95 % bootstrap)")
        ax.set_title("Comparaciones pareadas")
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(out_dir, "figuras", "fig_ffpp_03_comparaciones.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    res = {"archivos": files, "condiciones": conds, "experimentos": exps,
           "auc_tabla_principal": pivot.round(4).to_dict()}
    json.dump(res, open(os.path.join(out_dir, "resumen_analisis.json"), "w"), indent=1, ensure_ascii=False)
    return res
