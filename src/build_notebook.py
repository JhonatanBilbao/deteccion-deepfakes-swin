"""Genera el notebook principal a partir de los módulos."""
import os, sys
import nbformat as nbf

MOD_DIR, OUT = sys.argv[1], sys.argv[2]
DEFAULT_MODE = sys.argv[3] if len(sys.argv) > 3 else "auto"  # "auto" | "smoke"
assert DEFAULT_MODE in ("auto", "smoke", "quick", "full")
MODULES = ["backend", "layers", "losses", "optim", "mlp", "cnn", "swin", "pretrained", "detector",
           "augment", "data", "ff_splits", "metrics", "trainer", "gradcheck"]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Detección de Deepfakes en Video: Pipeline desde cero (NumPy / CuPy)
### Rama Global (Swin Transformer) + Rama Lateral (CNN de Asimetría Diferencial) + MLP, robusto a imágenes de mala calidad

Este cuaderno sigue el pipeline de la tesis:

| Etapa | Entrada → Salida |
|---|---|
| 1. Preprocesamiento por frame | Video → muestreo uniforme N=16 → landmarks MediaPipe → alineación (eje central) → rostro 224×224 |
| 2. Dos ramas + fusión | Rostro (224×224) → Swin Transformer → vector global 768d → proyección 256d: Rostro dividido → residuos de textura por hemicara → \|flip(izq) − der\| (224×112) → CNN 128d → proyección 256d: Concatenación → descriptor híbrido 512d |
| 3. Clasificación + agregación | 512d → MLP (Focal Loss) → p_i por frame → promedio de N=16 → decisión del video |

Todo el modelo está implementado desde cero (forward, backpropagation y optimizador) con NumPy/CuPy, sin PyTorch ni TensorFlow. Los módulos están en la carpeta `code kaggle` y este cuaderno los escribe en `code_kaggle/` para ser autocontenido.

Robustez a mala calidad. Se comparan dos modelos idénticos (misma inicialización, mismos datos y mismo orden de lotes):
- `base_sin_aumento`: entrenado solo con imágenes originales.
- `robusto_con_aumento`: entrenado con aumento de calidad desde cero (desenfoque gaussiano, ruido gaussiano, reducción de resolución, brillo/contraste) y normalización robusta (mediana / IQR de train, con recorte).

Ambos se evalúan en test con calidad buena, regular, mala y muy mala, con degradaciones aisladas y con compresión JPEG (no vista en entrenamiento).

Modos de ejecución (celda de configuración):
- `quick`: prueba rápida con un subconjunto pequeño; sirve para verificar que todo el pipeline funciona. Sus métricas no son resultados de la tesis.
- `full`: entrenamiento completo (se activa por defecto en Kaggle).

Para correr en Kaggle: Settings → Accelerator GPU T4/P100, Internet ON (instala MediaPipe/CuPy y descarga el modelo de landmarks), y añadir el dataset con la estructura `real/*.mp4` y `fake/*.mp4`. Luego *Run All*.
""")

code(r"""
import os, sys, json, time, subprocess, platform, warnings
warnings.filterwarnings("ignore", category=UserWarning)
T_START = time.time()

ON_KAGGLE = os.path.exists("/kaggle/input")
MODE = "__DEFAULT_MODE__"   # "smoke" | "quick" | "full" | "auto" (full en Kaggle, quick en local)
if MODE == "auto":
    MODE = os.environ.get("DF_MODE", "full" if ON_KAGGLE else "quick")
assert MODE in ("smoke", "quick", "full")
SMOKE = MODE == "smoke"

WORK_DIR = "/kaggle/working" if ON_KAGGLE else os.path.abspath(os.environ.get("DF_WORK_DIR", "outputs_avance"))
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
for d in ("code_kaggle", "figuras", "resultados", "checkpoints", "cache"):
    os.makedirs(d, exist_ok=True)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ.setdefault("GLOG_minloglevel", "3")      # silencia los logs C++ de MediaPipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

CFG = {
    "mode": MODE,
    "seed": SEED,
    "n_frames": 16,                         # muestreo uniforme por video (pipeline)
    "img_size": 224,
    "split_ratios": (0.70, 0.15, 0.15),     # train / val / test por grupos de identidad
    "quick_n_real": 25,                     # solo modo quick: 25 reales + 1 fake por video destino
    "norm_fit_frames": 512,                 # frames de train para la normalización robusta
    # Swin-T (Liu et al., 2021): parches 4×4, ventana 7, 96→192→384→768, profundidades 2-2-6-2.
    "swin": dict(img_size=224, patch_size=4, embed_dim=96, window=7, mlp_ratio=4.0, proj_dim=256,
                 num_heads=(3, 6, 12, 24), depths=(2, 2, 6, 2)),
    # Inicialización del backbone con pesos ImageNet-1k (timm/swin_tiny_patch4_window7_224.ms_in1k).
    # Solo cambia el punto de partida: forward, backward y optimizador siguen siendo propios.
    "swin_pretrained": True,
    "swin_lr_mult": 0.1,                    # lr del backbone = 0.1 × lr de las cabezas
    # Modelos a comparar: (nombre, ramas, ¿aumento de calidad?)
    "experiments": [("base_sin_aumento", "ambas", False), ("robusto_con_aumento", "ambas", True),
                    ("ablacion_solo_asimetria", "solo_asimetria", True),
                    ("ablacion_solo_swin", "solo_swin", True)],
    "min_val_auc": 0.55,                    # por debajo: el modelo se marca como "no aprendió"
    "cnn": dict(in_channels=3, widths=(32, 64, 128), proj_dim=256),
    "mlp": dict(in_dim=512, hidden_dim=256, num_classes=2, dropout=0.3),
    "drop_branch": 0.1,
    # Mapa de asimetría: "residual" (residuos de textura normalizados por hemicara; estable entre
    # particiones e insensible a la iluminación, ver diagnostico_asimetria.py) o "pixel" (original).
    "diff_mode": "residual",
    # Partición: "oficial" (lista de FF++ 720/140/140) si todos los videos están en ella; si no, por identidad.
    "protocol": "oficial",
    "train": dict(epochs={"smoke": 1, "quick": 5}.get(MODE, 50),
                  batch_size=16, eval_batch_size=16 if MODE != "full" else 32,
                  lr=3e-4 if MODE == "quick" else 1e-4, weight_decay=1e-4, clip_norm=5.0,
                  frames_per_video=4 if MODE == "quick" else 8,
                  patience=7, focal_gamma=2.0, focal_alpha=0.75,
                  select_on="val_auc_buena", val_bad_level="mala", seed=SEED,
                  diff_mode="residual"),
}
print(f"Entorno: {'Kaggle' if ON_KAGGLE else 'local'} | modo: {MODE} | Python {platform.python_version()}")
print(json.dumps({k: v for k, v in CFG.items() if k not in ("swin", "cnn", "mlp", "train")}, indent=1))
print("Swin :", CFG["swin"]); print("Train:", CFG["train"])
""")

code(r"""
# Dependencias: en Kaggle se instala MediaPipe y, si hay GPU, CuPy (CUDA 12).
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

HAS_NVIDIA = subprocess.run("nvidia-smi", shell=True, capture_output=True).returncode == 0
if ON_KAGGLE:
    try:
        import mediapipe  # noqa
    except ImportError:
        _pip("mediapipe")
    if HAS_NVIDIA:
        try:
            import cupy  # noqa
        except ImportError:
            _pip("cupy-cuda12x")

import numpy as np, cv2, mediapipe as mp, matplotlib, sklearn, pandas as pd
import matplotlib.pyplot as plt
print("numpy", np.__version__, "| opencv", cv2.__version__, "| mediapipe", mp.__version__,
      "| sklearn", sklearn.__version__, "| GPU NVIDIA:", HAS_NVIDIA)
""")

md(r"""
## 0. Módulos desde cero (`code kaggle`)

| Módulo | Contenido |
|---|---|
| `backend.py` | CuPy (GPU) o NumPy (CPU), semillas |
| `layers.py` | Linear, ReLU, GELU, Dropout, LayerNorm, BatchNorm2d, Conv2d (im2col), MaxPool2d, GlobalAvgPool2d: forward y backward manuales |
| `mlp.py`: `cnn.py`: `swin.py` | Clasificador MLP, rama lateral CNN, rama global Swin Transformer (patch embedding 4×4, atención por ventanas W-MSA y desplazadas SW-MSA con máscara, sesgo de posición relativa, patch merging, LayerNorm) |
| `losses.py`: `optim.py` | Focal Loss con gradiente analítico: AdamW, SGD, recorte de gradiente, coseno |
| `detector.py` | Modelo de dos ramas, branch dropout, fusión 512d, agregación temporal |
| `augment.py` | Desenfoque, ruido, baja resolución, JPEG (solo test), mapa diferencial, normalización robusta |
| `data.py` | Listado de videos, partición por identidad, MediaPipe, muestreo N=16, caché |
| `metrics.py`: `trainer.py` | AUC, F1, precisión, recall, matriz de confusión, EER, figuras a 300 dpi: bucle de entrenamiento |
| `gradcheck.py` | Verificación numérica de todas las derivadas |
""")

for m in MODULES:
    src = open(os.path.join(MOD_DIR, f"{m}.py"), encoding="utf-8").read()
    code(f"%%writefile code_kaggle/{m}.py\n" + src)

md(r"""
### Verificación de derivadas y dimensiones
Compara cada gradiente analítico con diferencias centrales en float64 y verifica las formas con los tamaños reales del pipeline. Se ejecuta en un proceso aparte con el mismo backend (CuPy en Kaggle).
""")

code(r"""
r = subprocess.run([sys.executable, "gradcheck.py"], cwd="code_kaggle", capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-2000:])
assert r.returncode == 0, "La verificación de gradientes falló"
""")

code(r"""
sys.path.insert(0, os.path.abspath("code_kaggle"))
import backend as bk
bk.set_dtype(np.float32)
bk.set_seed(SEED)
from backend import xp
from layers import softmax
from swin import SwinBranch
from pretrained import load_swin_imagenet, set_lr_mult
from cnn import CNNBranch
from mlp import MLPClassifier
from detector import DeepfakeDetectorScratch
from augment import (QUALITY_LEVELS, ISOLATED_DEGRADATIONS, AugConfig, QualityAugmenter,
                     RobustNormalizer, apply_degradation, bilateral_diff_map)
from data import (FaceAligner, build_face_cache, check_identity_leakage, find_dataset_dir,
                  list_videos, quick_subset, read_frames, smoke_subset, sample_frame_indices, split_by_identity,
                  split_official)
from metrics import (binary_metrics, plot_confusions, plot_history, plot_quality_curve, plot_roc,
                     save_fig, youden_threshold)
from trainer import Experiment, TrainConfig
from sklearn.metrics import roc_auc_score
print("Backend:", "CuPy (GPU)" if bk.ON_GPU else "NumPy (CPU)", "| dtype:", bk.DTYPE.__name__)
""")

md(r"""
## Etapa 1: Preprocesamiento y extracción por frame
### 1.1 Dataset y partición sin fuga por identidad
En FaceForensics++, `fake/<Método>_TTT_SSS.mp4` se construye a partir del video real `TTT` y la cara de `SSS`. Todos los videos que comparten una identidad forman un grupo, y cada grupo va completo a un solo conjunto. Así ninguna persona, escena o fondo aparece en train y en test a la vez.
""")

code(r"""
search = ["/kaggle/input"] if ON_KAGGLE else [p for p in (os.environ.get("DF_DATA_DIR"), "../code/dataset",
                                                          "../../code/dataset", "dataset") if p]
DATA_DIR = find_dataset_dir(search)
assert DATA_DIR, f"No se encontró una carpeta con real/ y fake/ en {search}"
all_videos = list_videos(DATA_DIR)
if MODE == "smoke":      # 3 reales + 3 fakes de identidades disjuntas → 1 real + 1 fake por conjunto
    videos = smoke_subset(all_videos, 3)
elif MODE == "quick":
    videos = quick_subset(all_videos, CFG["quick_n_real"])
else:
    videos = all_videos
print(f"Dataset: {DATA_DIR}\nVideos disponibles: {len(all_videos)} | usados en modo {MODE}: {len(videos)}")

PROTOCOL = "identidad"
if CFG["protocol"] == "oficial" and not SMOKE:     # smoke usa 6 videos elegidos para la partición por identidad
    try:
        splits = split_official(videos)
        PROTOCOL = "oficial FF++"
    except ValueError as e:
        print("No se puede usar la partición oficial:", e)
if PROTOCOL == "identidad":
    splits = split_by_identity(videos, CFG["split_ratios"], SEED)
print("Partición:", PROTOCOL)
leak = check_identity_leakage(splits)
assert all(v == 0 for v in leak.values()), f"fuga de datos entre conjuntos: {leak}"
print("Verificación de fuga (todo debe ser 0):", leak)

rows = []
for k, vs in splits.items():
    rows.append({"conjunto": k, "videos": len(vs), "reales": sum(v.label == 0 for v in vs),
                 "fakes": sum(v.label == 1 for v in vs), "grupos_identidad": len({v.group for v in vs}),
                 "métodos_fake": ", ".join(sorted({v.method for v in vs if v.label == 1}))})
split_df = pd.DataFrame(rows)
display(split_df)
pd.DataFrame([{"video": v.video_id, "etiqueta": v.label, "metodo": v.method, "identidades": "+".join(v.identities),
               "grupo": v.group, "conjunto": v.split} for v in videos]).to_csv("resultados/particion_videos.csv", index=False)
""")

md(r"""
### 1.2 Muestreo uniforme, landmarks (MediaPipe), alineación al eje central y rostro 224×224
Misma lógica que `preprocess.py`: se nivelan los ojos rotando sobre el centro ocular, el recorte cuadrado se centra en el centro ocular (horizontal) y en el punto medio frente–mentón (vertical), con margen 1.35 y relleno reflejado.
""")

code(r"""
aligner = FaceAligner(output_size=CFG["img_size"], task_model_path=os.path.join(WORK_DIR, "face_landmarker.task"))
print("API de MediaPipe:", aligner.kind)

demo = videos[0]
cap = cv2.VideoCapture(demo.path); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
idxs = sample_frame_indices(total, CFG["n_frames"])
frames = read_frames(demo.path, idxs)
fi0, frame0 = frames[0]
face0, info0 = aligner.align(frame0)
print(f"Video demo: {demo.video_id} | {total} frames | índices muestreados: {idxs}")
print(f"Ángulo corregido: {info0['angle']:.2f}° | landmarks: {info0['landmarks'].shape} | rostro: {face0.shape}")

f01 = face0.astype(np.float32)[None].transpose(0, 3, 1, 2) / 255.0
dm0 = bk.to_numpy(bilateral_diff_map(bk.asarray(f01), CFG["diff_mode"]))[0].transpose(1, 2, 0)
fig, ax = plt.subplots(1, 5, figsize=(17, 3.8))
vis = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
ax[0].imshow(vis); ax[0].scatter(info0["landmarks"][:, 0], info0["landmarks"][:, 1], s=0.4, c="lime")
ax[0].set_title(f"Frame {fi0} + {len(info0['landmarks'])} landmarks")
ax[1].imshow(face0); ax[1].axvline(112, color="lime", ls="--"); ax[1].set_title("Rostro alineado 224×224 (eje x=112)")
ax[2].imshow(np.fliplr(face0[:, :112])); ax[2].set_title("Mitad izquierda espejada (224×112)")
ax[3].imshow(face0[:, 112:]); ax[3].set_title("Mitad derecha (224×112)")
im = ax[4].imshow(dm0.mean(axis=2), cmap="inferno"); ax[4].set_title(f"Mapa de diferencias ({CFG['diff_mode']})")
plt.colorbar(im, ax=ax[4], fraction=0.046)
for a in ax: a.axis("off")
save_fig(fig, "figuras/fig01_preprocesamiento_y_mapa_diferencial.png"); plt.show()
""")

code(r"""
cache_file = f"cache/rostros_N{CFG['n_frames']}_{MODE}.npz"
t0 = time.time()
cache = build_face_cache(videos, CFG["n_frames"], aligner, cache_file)
print(f"Rostros extraídos: {cache['faces'].shape} uint8 | {time.time() - t0:.0f}s")
nd = cache["n_detect"]
print(f"Frames con rostro por video: media {nd.mean():.1f} / {CFG['n_frames']} | videos sin rostro (excluidos): {(nd == 0).sum()}")

vid_of = {v.path: i for i, v in enumerate(videos)}
split_videos = {k: np.array([vid_of[v.path] for v in vs if nd[vid_of[v.path]] > 0]) for k, vs in splits.items()}
for k, ids in split_videos.items():
    print(f"  {k:5s}: {len(ids)} videos | {int(np.isin(cache['frame_video'], ids).sum())} frames")

sel = np.flatnonzero(cache["frame_video"] == 0)[:16]
fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))
for a, i in zip(ax.ravel(), sel):
    a.imshow(cache["faces"][i]); a.set_title(f"frame {cache['frame_index'][i]}", fontsize=8)
for a in ax.ravel(): a.axis("off")
fig.suptitle(f"Muestreo uniforme N={CFG['n_frames']} y rostros normalizados: {videos[0].video_id}")
save_fig(fig, "figuras/fig02_muestreo_uniforme_16_frames.png"); plt.show()
""")

md(r"""
### 1.3 Normalización robusta (estadísticos solo de train)
$z = \mathrm{clip}\big((x - \mathrm{mediana}) / (\mathrm{IQR}/1.349),\ \pm 5\big)$ por canal, para el rostro y para el mapa diferencial. La mediana y el IQR no se desplazan con píxeles saturados o ruido fuerte, y el recorte acota las entradas cuando la imagen está degradada. Se calcula solo con frames de train.
""")

code(r"""
rng_fit = np.random.RandomState(SEED)
train_frames = np.flatnonzero(np.isin(cache["frame_video"], split_videos["train"]))
fit_idx = np.sort(rng_fit.choice(train_frames, size=min(CFG["norm_fit_frames"], len(train_frames)), replace=False))
faces_fit, diffs_fit = [], []
for s in range(0, len(fit_idx), 64):
    x = bk.asarray(cache["faces"][fit_idx[s:s + 64]]).transpose(0, 3, 1, 2) / 255.0
    faces_fit.append(bk.to_numpy(x)); diffs_fit.append(bk.to_numpy(bilateral_diff_map(x, CFG["diff_mode"])))
face_norm = RobustNormalizer(clip=5.0).fit(np.concatenate(faces_fit))
diff_norm = RobustNormalizer(clip=5.0).fit(np.concatenate(diffs_fit))
del faces_fit, diffs_fit
norm_df = pd.DataFrame({"rostro_mediana": face_norm.center, "rostro_escala": face_norm.scale,
                        "mapa_mediana": diff_norm.center, "mapa_escala": diff_norm.scale}, index=["R", "G", "B"])
print(f"Ajustada con {len(fit_idx)} frames de train (ningún frame de val/test).")
display(norm_df.round(4))
json.dump({"rostro": face_norm.to_dict(), "mapa_diferencial": diff_norm.to_dict(), "n_frames_train": int(len(fit_idx))},
          open("resultados/normalizacion_robusta.json", "w"), indent=2)
""")

md(r"""
### 1.4 Aumento de calidad (entrenamiento) y niveles de calidad (evaluación)
- Entrenamiento (`robusto_con_aumento`), por imagen y al azar: reducción de resolución ×0.25–0.9 (p=0.5), desenfoque σ=0.4–2.2 (p=0.5), ruido σ=1–15/255 (p=0.5), brillo/contraste (p=0.3). No se usa flip horizontal porque invertiría las hemicaras.
- Evaluación, degradación fija y reproducible: `regular`, `mala` y `muy_mala` combinan las tres degradaciones; además hay degradaciones aisladas y JPEG calidad 30, que el modelo nunca vio al entrenar.

El mapa diferencial se recalcula después de degradar el rostro.
""")

code(r"""
conds = {**QUALITY_LEVELS, **ISOLATED_DEGRADATIONS}
print(pd.DataFrame(conds).T.fillna("-").to_string())
x0 = bk.asarray(face0[None].astype(np.float32)).transpose(0, 3, 1, 2) / 255.0
fig, ax = plt.subplots(2, len(conds), figsize=(2.3 * len(conds), 5.2))
stats = []
for j, (name, spec) in enumerate(conds.items()):
    xd = apply_degradation(x0, spec, seed=0)
    dmap = bilateral_diff_map(xd, CFG["diff_mode"])
    ax[0, j].imshow(bk.to_numpy(xd[0]).transpose(1, 2, 0)); ax[0, j].set_title(name, fontsize=9)
    dm = bk.to_numpy(dmap[0]).mean(0)
    ax[1, j].imshow(dm, cmap="inferno", vmin=0, vmax=float(np.percentile(dm, 99)) + 1e-8)   # escala por imagen
    stats.append({"condición": name, "media mapa diferencial": float(dmap.mean()),
                  "media mapa normalizado": float(diff_norm(dmap).mean())})
for a in ax.ravel(): a.axis("off")
ax[0, 0].set_ylabel("rostro"); ax[1, 0].set_ylabel("mapa dif.")
fig.suptitle("Niveles de calidad de evaluación: rostro (arriba) y mapa diferencial (abajo)")
save_fig(fig, "figuras/fig03_niveles_de_calidad.png"); plt.show()
display(pd.DataFrame(stats).round(4))

bk.set_seed(SEED)
aug = QualityAugmenter(AugConfig())
xa = aug(xp.repeat(x0, 8, axis=0))
fig, ax = plt.subplots(1, 8, figsize=(16, 2.4))
for a, im in zip(ax, bk.to_numpy(xa)): a.imshow(im.transpose(1, 2, 0)); a.axis("off")
fig.suptitle("Ejemplos aleatorios del aumento de calidad usado en entrenamiento (mismo rostro)")
save_fig(fig, "figuras/fig04_ejemplos_aumento.png"); plt.show()
""")

md(r"""
## Etapa 2: Extracción en dos ramas y fusión balanceada
Verificación de dimensiones con un lote real: rostro → Swin → 768d → 256d; mapa diferencial → CNN → 128d → 256d; concatenación → 512d → MLP → 2 logits.
""")

code(r"""
WEIGHTS_OK = False
if CFG["swin_pretrained"]:
    try:
        from pretrained import download
        SWIN_WEIGHTS = download(dst="checkpoints/swin_tiny_imagenet.safetensors")
        WEIGHTS_OK = True
    except Exception as e:
        if ON_KAGGLE:
            raise RuntimeError("No se pudieron descargar los pesos ImageNet del Swin: activa Internet en Kaggle") from e
        print(f"AVISO: sin pesos ImageNet ({type(e).__name__}); en local se continúa con Swin aleatorio")


def make_model(branches: str = "ambas") -> DeepfakeDetectorScratch:
    bk.set_seed(SEED)   # misma inicialización para todos los modelos comparados
    swin = SwinBranch(**CFG["swin"])
    if WEIGHTS_OK:
        load_swin_imagenet(swin, SWIN_WEIGHTS, verbose=False)
        set_lr_mult(swin.patch_embed, CFG["swin_lr_mult"])
        for lyr in swin.layers:
            set_lr_mult(lyr, CFG["swin_lr_mult"])
        set_lr_mult(swin.norm, CFG["swin_lr_mult"])          # la proyección 768→256 es nueva: lr completo
    return DeepfakeDetectorScratch(swin, CNNBranch(**CFG["cnn"]), MLPClassifier(**CFG["mlp"]),
                                   drop_branch=CFG["drop_branch"], branches=branches)

print("Pesos ImageNet del Swin:", "cargados" if WEIGHTS_OK else "no (inicialización aleatoria)")

probe = make_model().eval()
idx = np.flatnonzero(np.isin(cache["frame_video"], split_videos["train"]))[:4]
x = bk.asarray(cache["faces"][idx]).transpose(0, 3, 1, 2) / 255.0
face_in, diff_in = face_norm(x), diff_norm(bilateral_diff_map(x, CFG["diff_mode"]))
g768 = probe.swin.forward_features(face_in); g256 = probe.swin(face_in)
lat256 = probe.cnn(diff_in)
fused = xp.concatenate([g256, lat256], axis=1); logits = probe.mlp(fused)
shapes = [("Rostro de entrada", face_in.shape), ("Mapa diferencial", diff_in.shape),
          ("Swin: vector global", g768.shape), ("Swin: proyección", g256.shape),
          ("CNN: proyección", lat256.shape), ("Descriptor híbrido", fused.shape), ("Logits MLP", logits.shape)]
for n, s in shapes: print(f"  {n:22s} {tuple(s)}")
assert g768.shape[1] == 768 and g256.shape[1] == 256 and lat256.shape[1] == 256 and fused.shape[1] == 512 and logits.shape[1] == 2
print(f"Parámetros → Swin: {probe.swin.n_params():,} | CNN: {probe.cnn.n_params():,} | MLP: {probe.mlp.n_params():,} | total: {probe.n_params():,}")
probe.release(); del probe
""")

md(r"""
## Entrenamiento comparativo: línea base vs. robusto, más ablaciones de ramas
Todos los modelos parten de la misma inicialización (seed=42; Swin con pesos ImageNet-1k) y ven los mismos frames en el mismo orden. `base_sin_aumento` y `robusto_con_aumento` difieren solo en el aumento de calidad; las ablaciones `solo_asimetria` y `solo_swin` (con aumento) miden el aporte de cada rama frente a la fusión. El backbone preentrenado usa 0.1× la tasa de aprendizaje. AdamW, coseno, Focal Loss (α=0.75, γ=2), recorte de gradiente 5.0 y early stopping por AUC de validación con calidad buena. Se registra también el AUC de validación con calidad mala.
""")

code(r"""
tcfg = TrainConfig(**CFG["train"])
experiments = {}
for name, branches, use_aug in CFG["experiments"]:
    augmenter = QualityAugmenter(AugConfig()) if use_aug else None
    model = make_model(branches)
    bk.set_seed(SEED)
    exp = Experiment(name, model, tcfg, cache, split_videos, face_norm, diff_norm, augmenter,
                     QUALITY_LEVELS, WORK_DIR)
    t0 = time.time()
    exp.fit()
    exp.train_time = time.time() - t0
    print(f"→ {name}: mejor época {exp.best_epoch} | {exp.train_time / 60:.1f} min\n")
    experiments[name] = exp

plot_history({k: e.history for k, e in experiments.items()}, "figuras/fig05_curvas_entrenamiento.png"); plt.show()
""")

md(r"""
## Etapa 3: Clasificación, agregación temporal y evaluación por calidad
Cada video se puntúa con el promedio de p(deepfake) de sus N frames. El umbral se elige en validación con calidad buena (índice de Youden) y se aplica sin cambios a todas las condiciones de test.
""")

code(r"""
rows, preds = [], {}
for name, exp in experiments.items():
    yv, sv = exp.video_predictions("val")
    thr = youden_threshold(yv, sv)
    val_auc = float(roc_auc_score(yv, sv)) if len(np.unique(yv)) == 2 else float("nan")
    frac_fake_val = float((sv >= thr).mean())
    aprendio = bool(val_auc >= CFG["min_val_auc"]) and 0.0 < frac_fake_val < 1.0
    if not aprendio:
        print(f"Aviso: {name}: AUC de validación = {val_auc:.3f} (umbral mínimo {CFG['min_val_auc']}) "
              f"y {frac_fake_val:.0%} de val clasificado como fake -> no aprendió. "
              f"Sus F1/precisión/recall de test no son interpretables; usar solo el AUC.")
    for cond, spec in conds.items():
        y, s = exp.video_predictions("test", spec)
        preds[(name, cond)] = (y, s)
        m = binary_metrics(y, s, thr)
        rows.append({"modelo": name, "calidad": cond, **m, "val_auc": val_auc, "aprendio": aprendio})
res = pd.DataFrame(rows)
res.to_csv("resultados/metricas_test_por_calidad.csv", index=False)
cols = ["modelo", "calidad", "auc", "f1", "precision", "recall", "accuracy", "eer", "threshold", "tn", "fp", "fn", "tp", "n",
        "val_auc", "aprendio"]
display(res[cols].round(3))
""")

code(r"""
# Caída respecto a calidad buena (negativo = pérdida de rendimiento)
drop = []
for name in experiments:
    base = res[(res.modelo == name) & (res.calidad == "buena")].iloc[0]
    for cond in conds:
        if cond == "buena":
            continue
        r = res[(res.modelo == name) & (res.calidad == cond)].iloc[0]
        drop.append({"modelo": name, "calidad": cond, "ΔAUC": r.auc - base.auc, "ΔF1": r.f1 - base.f1,
                     "ΔRecall": r.recall - base.recall, "ΔPrecisión": r.precision - base.precision})
drop_df = pd.DataFrame(drop)
drop_df.to_csv("resultados/caida_por_calidad.csv", index=False)
display(drop_df.round(3))

levels = list(QUALITY_LEVELS)
for met in ("auc", "f1"):
    tab = {n: {c: float(res[(res.modelo == n) & (res.calidad == c)][met].iloc[0]) for c in levels} for n in experiments}
    plot_quality_curve(tab, levels, met, f"figuras/fig06_{met}_vs_calidad.png"); plt.show()
""")

code(r"""
for name in experiments:
    plot_roc({c: preds[(name, c)] for c in ("buena", "regular", "mala", "muy_mala", "jpeg_q30_no_visto")},
             f"figuras/fig07_roc_{name}.png", f"ROC test (video): {name}"); plt.show()

cms = {}
for name in experiments:
    for cond in ("buena", "mala"):
        r = res[(res.modelo == name) & (res.calidad == cond)].iloc[0]
        cms[f"{name}\ncalidad {cond}"] = np.array([[r.tn, r.fp], [r.fn, r.tp]])
plot_confusions(cms, "figuras/fig08_matrices_confusion_buena_vs_mala.png"); plt.show()
""")

md(r"""
### Lista de verificación del smoke test y estimación de tiempos en GPU
Solo se ejecuta en modo `smoke`. Comprueba que no hubo fugas, que las pérdidas y métricas son finitas, que se evaluaron todas las condiciones de calidad, y mide un paso real de entrenamiento (B=16, Swin-T completo) para estimar cuánto tardará el entrenamiento completo con este dataset.
""")

code(r"""
if SMOKE:
    import copy, math
    checks = {
        "sin fugas de identidad entre conjuntos": not any(leak.values()),
        "ambas clases en train/val/test": all(len(set(v.label for v in splits[k])) == 2 for k in ("train", "val", "test")),
        "rostros 224×224×3 uint8": cache["faces"].shape[1:] == (224, 224, 3) and cache["faces"].dtype == np.uint8,
        "pérdidas de entrenamiento finitas": all(math.isfinite(h["train_loss"]) for e in experiments.values() for h in e.history),
        "AUC de validación (buena y mala) finitos": all(math.isfinite(h["val_auc_buena"]) and math.isfinite(h["val_auc_mala"])
                                                       for e in experiments.values() for h in e.history),
        "todas las condiciones de calidad evaluadas": len(res) == len(experiments) * len(conds),
        "métricas de test finitas": bool(np.isfinite(res[["auc", "f1", "precision", "recall"]].to_numpy(float)).all()),
        "checkpoints guardados": all(os.path.exists(f"checkpoints/{n}_best.npz") for n in experiments),
        "pesos ImageNet del Swin cargados (obligatorio en Kaggle)": WEIGHTS_OK or not ON_KAGGLE,
        "ablaciones entrenadas": {"ablacion_solo_asimetria", "ablacion_solo_swin"} <= set(experiments),
    }
    for k, ok in checks.items():
        print(f"  [{'OK ' if ok else 'FALLA'}] {k}")

    # Tiempo real de un paso de entrenamiento (forward + backward + AdamW) con B=16
    from losses import FocalLoss
    from optim import AdamW
    sync = (lambda: xp.cuda.Stream.null.synchronize()) if bk.ON_GPU else (lambda: None)
    m = make_model().train(); opt = AdamW(m.parameters(), lr=1e-4); crit = FocalLoss()
    fb, db = bk.randn(16, 3, 224, 224), bk.randn(16, 3, 224, 112)
    yb = xp.asarray([0, 1] * 8)
    times = []
    for i in range(3):
        sync(); t0 = time.time()
        _, dl = crit(m(fb, db), yb); m.backward(dl); opt.step(); m.release(); sync()
        times.append(time.time() - t0)
    m.eval(); sync(); t0 = time.time(); m(fb, db); m.release(); sync(); t_eval = time.time() - t0
    t_step = float(np.median(times[1:]))           # el primer paso incluye compilación de kernels
    del m, opt, fb, db; import gc; gc.collect()
    try:
        full = split_official(copy.deepcopy(all_videos))
    except ValueError:
        full = split_by_identity(copy.deepcopy(all_videos), CFG["split_ratios"], SEED)
    n_tr, n_va, n_te = (len(full[k]) for k in ("train", "val", "test"))
    ep_train = n_tr * 8 / 16 * t_step                                   # modo full: 8 frames/video
    ep_val = n_va * CFG["n_frames"] * 2 / 16 * t_eval                   # val en calidad buena y mala
    test_all = len(CFG["experiments"]) * n_te * CFG["n_frames"] * len(conds) / 16 * t_eval    # 2 modelos × condiciones
    per_epoch = ep_train + ep_val
    print(f"\nBackend: {'CuPy (GPU)' if bk.ON_GPU else 'NumPy (CPU)'} | paso de entrenamiento B=16: {t_step:.2f}s "
          f"| inferencia B=16: {t_eval:.2f}s")
    print(f"Estimación modo full con este dataset ({n_tr}/{n_va}/{n_te} videos): "
          f"{per_epoch / 60:.1f} min/época por modelo → 50 épocas × {len(CFG['experiments'])} modelos ≈ "
          f"{len(CFG['experiments']) * 50 * per_epoch / 3600:.1f} h "
          f"(+ test {test_all / 60:.0f} min; el early stopping suele cortar antes)")
    smoke_report = {"checks": checks, "t_step_s": t_step, "t_eval_s": t_eval, "min_por_epoca": per_epoch / 60,
                    "horas_full_50ep_todos": len(CFG["experiments"]) * 50 * per_epoch / 3600, "backend": "gpu" if bk.ON_GPU else "cpu"}
    json.dump(smoke_report, open("resultados/smoke_test.json", "w"), indent=2)
    assert all(checks.values()), "El smoke test encontró fallas (ver arriba)"
    print("\nSmoke test superado")
""")

code(r"""
summary = {
    "modo": MODE, "backend": "CuPy (GPU)" if bk.ON_GPU else "NumPy (CPU)", "config": CFG,
    "particion": split_df.to_dict(orient="records"), "fuga_identidad": leak,
    "entrenamiento": {n: {"mejor_epoca": e.best_epoch, "minutos": round(e.train_time / 60, 2),
                          "historial": e.history} for n, e in experiments.items()},
    "test": res[cols].to_dict(orient="records"),
    "tiempo_total_min": round((time.time() - T_START) / 60, 2),
}
json.dump(summary, open("resultados/resumen_avance.json", "w"), indent=2, default=str)
print(f"Tiempo total: {summary['tiempo_total_min']} min")
for d in ("figuras", "resultados", "checkpoints"):
    print(f"{d}/:", ", ".join(sorted(os.listdir(d))))
""")

md(r"""
## Lectura de resultados y siguiente paso

- En modos `smoke`/`quick` el test tiene muy pocos videos; las métricas solo prueban que el pipeline funciona de punta a punta y no deben reportarse como resultados.
- Primero revisar la columna `aprendio`: un modelo con AUC de validación < 0.55 no aprendió y sus F1/precisión/recall de test no son interpretables (el umbral de Youden se vuelve degenerado).
- Hipótesis de robustez: se sostiene si `robusto_con_aumento` pierde claramente menos AUC/F1 que `base_sin_aumento` de *buena* a *mala*/*muy mala* sin perder rendimiento en *buena* (`caida_por_calidad.csv`, `fig06`). JPEG mide si la robustez generaliza a una degradación no vista.
- Aporte de la asimetría: comparar `robusto_con_aumento` (fusión) con `ablacion_solo_swin` y `ablacion_solo_asimetria`. La rama lateral aporta si la fusión supera a `solo_swin`, sobre todo en calidad mala.
- Variabilidad: con ~37 videos de test el AUC tiene un error estándar ≈ 0.08–0.10; las diferencias pequeñas entre modelos no son concluyentes sin repetir con varias semillas o más videos.
- Limitación a declarar: las degradaciones de test son de la misma familia que el aumento de entrenamiento, salvo JPEG; conviene complementar con videos FF++ c40 reales si están disponibles.
""")

for c in cells:
    c.source = c.source.replace("__DEFAULT_MODE__", DEFAULT_MODE)
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"},
                  "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}
nbf.write(nb, OUT)
print("cells:", len(cells))
