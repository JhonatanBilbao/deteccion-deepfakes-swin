"""Genera los notebooks de las tres fases del protocolo FF++."""
import os
import sys

import nbformat as nbf

MOD_DIR, OUT_DIR = sys.argv[1], sys.argv[2]
SMOKE = "--smoke" in sys.argv
MODULES = ["backend", "layers", "losses", "optim", "mlp", "cnn", "swin", "pretrained", "detector",
           "augment", "data", "ff_splits", "metrics", "trainer", "gradcheck", "experimento"]
os.makedirs(OUT_DIR, exist_ok=True)


def notebook(title: str, intro: str, body: list, gradcheck: bool = False, mediapipe: bool = False,
             cupy: bool = False) -> nbf.NotebookNode:
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro.strip()}")]
    cells.append(nbf.v4.new_code_cell(f'''
import os, sys, json, time, glob, subprocess, platform, warnings, random
warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ON_KAGGLE = os.path.exists("/kaggle/input")
SMOKE = {SMOKE}
WORK_DIR = "/kaggle/working" if ON_KAGGLE else os.path.abspath(os.environ.get("DF_WORK_DIR", "salida_fases"))
os.makedirs(WORK_DIR, exist_ok=True); os.chdir(WORK_DIR); os.makedirs("code_kaggle", exist_ok=True)
INPUT_ROOT = "/kaggle/input" if ON_KAGGLE else os.environ.get("DF_INPUT_ROOT", ".")

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
HAS_NVIDIA = subprocess.run("nvidia-smi", shell=True, capture_output=True).returncode == 0
if ON_KAGGLE and {mediapipe}:
    try:
        import mediapipe  # noqa
    except ImportError:
        _pip("mediapipe")
if ON_KAGGLE and {cupy} and HAS_NVIDIA:
    try:
        import cupy  # noqa
    except ImportError:
        _pip("cupy-cuda12x")
if not {cupy}:
    os.environ["DF_BACKEND"] = "numpy"      # fases sin entrenamiento: CPU
import numpy as np
random.seed(SEED); np.random.seed(SEED)
print(f"Entorno: {{'Kaggle' if ON_KAGGLE else 'local'}} | GPU NVIDIA: {{HAS_NVIDIA}} | smoke: {{SMOKE}} | Python {{platform.python_version()}}")
'''.strip()))
    for m in MODULES:
        src = open(os.path.join(MOD_DIR, f"{m}.py"), encoding="utf-8").read()
        cells.append(nbf.v4.new_code_cell(f"%%writefile code_kaggle/{m}.py\n" + src))
    cells.append(nbf.v4.new_code_cell('''
sys.path.insert(0, os.path.abspath("code_kaggle"))
import backend as bk
bk.set_dtype(np.float32); bk.set_seed(SEED)
print("Backend:", "CuPy (GPU)" if bk.ON_GPU else "NumPy (CPU)")
'''.strip()))
    if gradcheck:
        cells.append(nbf.v4.new_code_cell('''
r = subprocess.run([sys.executable, "gradcheck.py"], cwd="code_kaggle", capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
assert r.returncode == 0, "La verificación de gradientes falló"
'''.strip()))
    for kind, src in body:
        cells.append(nbf.v4.new_markdown_cell(src.strip()) if kind == "md" else nbf.v4.new_code_cell(src.strip()))
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


SMOKE_NOTE = ("\n\n> Modo smoke: 10 videos por conjunto y 1 época; solo verifica que el flujo funciona. "
              "Sus métricas no son resultados.") if SMOKE else ""

fase1 = notebook(
    "Fase 1: Preprocesamiento FF++ (protocolo oficial)",
    """
Etapa 1 del pipeline para todos los videos: muestreo uniforme N=16 → landmarks MediaPipe → alineación al eje
central → rostro 224×224, más el índice de giro de la cabeza por frame. Partición oficial de FaceForensics++
(720/140/140, sin fugas de identidad); la normalización robusta se ajusta solo con train.

- Train: 16 frames por real y 4 por fake (equilibra las clases 1:4 de FF++); val: 8; test: 16 (decisión por video con N=16).
- Si el dataset incluye la versión C40, se extraen también sus videos de test (mala calidad real).
- Si se añade Celeb-DF v2, se extrae su lista oficial de test (518 videos) para evaluar generalización.

Esta fase no usa GPU. Sus salidas (cachés + `manifest.json`) son la entrada de la fase 2.""" + SMOKE_NOTE,
    [("code", '''
from data import find_dataset_dir
from experimento import PrepConfig, preprocesar
ff_root = find_dataset_dir([INPUT_ROOT])
assert ff_root, "No se encontró FF++ (original_sequences/ + manipulated_sequences/, o real/ + fake/) en el input"
cdf = [os.path.dirname(p) for p in glob.glob(os.path.join(INPUT_ROOT, "**", "List_of_testing_videos.txt"), recursive=True)]
print("FF++:", ff_root, "| Celeb-DF:", cdf[0] if cdf else "no añadido")
cfg = PrepConfig(dataset_root=ff_root, out_dir="prep", celebdf_root=cdf[0] if cdf else None,
                 workers=max(1, min(4, os.cpu_count() or 1)),
                 max_videos_per_split=10 if SMOKE else None)
manifest = preprocesar(cfg)
'''), ("code", '''
import collections
for k in [k for k in manifest if k.startswith("videos_")]:
    c = collections.Counter((v["split"], "real" if v["label"] == 0 else "fake") for v in manifest[k])
    print(k, dict(sorted(c.items())))
print("Fugas de identidad:", manifest["fugas"])
for c in manifest["caches"].values():
    m = np.load(os.path.join("prep", c, "meta.npz"))
    print(f"{c}: {len(m['frame_video'])} rostros | videos sin rostro: {int((m['n_detect'] == 0).sum())}")
''')], mediapipe=True)

fase2 = notebook(
    "Fase 2: entrenamiento y evaluación de __EXPERIMENTO__ (semilla __SEMILLA__)",
    """
Un modelo del pipeline híbrido (Swin-T con pesos ImageNet + rama de asimetría con residuos de textura + MLP con
Focal Loss), entrenado con FF++ C23 (partición oficial) y evaluado en test C23, C23 con degradaciones sintéticas,
C40 real y Celeb-DF v2 (si se preprocesaron). Guarda las puntuaciones por video para la fase 3.""" + SMOKE_NOTE,
    [("code", '''
from experimento import TrainSetup, entrenar_y_evaluar
man = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", "manifest.json"), recursive=True))
assert man, "Añade como input la salida de la fase 1 (manifest.json)"
prep_dir = os.path.dirname(man[0])
print("Preprocesado:", prep_dir)
st = TrainSetup(prep_dir=prep_dir, out_dir="salida", experiment="__EXPERIMENTO__", seed=__SEMILLA__,
                epochs=1 if SMOKE else 15, patience=1 if SMOKE else 4, frames_per_video=4 if SMOKE else 8,
                swin_pretrained=ON_KAGGLE)
path = entrenar_y_evaluar(st)
''')], gradcheck=True, cupy=True)

fase3 = notebook(
    "Fase 3: Análisis: AUC con IC 95 %, comparaciones pareadas y figuras",
    """
Reúne las puntuaciones por video de todas las sesiones de la fase 2. Reporta para cada experimento y condición el
AUC por semilla con IC 95 % por bootstrap de grupos de identidad, la media ± d.e. entre semillas, las
comparaciones pareadas (mismos videos) y métricas con umbral fijado en validación. Figuras a 300 dpi.""" + SMOKE_NOTE,
    [("code", '''
from experimento import analizar
import pandas as pd
res = analizar([os.path.join(INPUT_ROOT, "**", "scores_*.csv")], "analisis", n_boot=300 if SMOKE else 2000)
'''), ("code", '''
from IPython.display import display, Image
display(pd.read_csv("analisis/tablas/auc_tabla_principal.csv"))
comp = pd.read_csv("analisis/tablas/comparaciones_pareadas.csv")
if len(comp):
    display(comp.round(4))
for f in sorted(glob.glob("analisis/figuras/*.png")):
    display(Image(f, width=900))
''')])

for name, nb in (("fase1_preproceso", fase1), ("fase2_entrenar", fase2), ("fase3_analisis", fase3)):
    nbf.write(nb, os.path.join(OUT_DIR, f"{name}.ipynb"))
    print("escrito", name, len(nb.cells), "celdas")
