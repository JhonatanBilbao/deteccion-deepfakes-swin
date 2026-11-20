#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ejecutar_ffpp.sh — Protocolo oficial FaceForensics++ en Kaggle, por fases.
#
#   Fase 1 (CPU, sin cuota de GPU): rostros alineados + partición oficial + normalización (train).
#   Fase 2 (GPU T4, una sesión por modelo): entrenamiento + puntuaciones por video en
#           C23, degradaciones sintéticas, C40 y Celeb-DF.
#   Fase 3 (CPU): tablas con IC 95 %, comparaciones pareadas y figuras → se descargan aquí.
#
# Uso (desde la raíz del repositorio):
#   export KAGGLE_USERNAME=jhonatanbilbaouscca
#   bash kaggle/ejecutar_ffpp.sh --ffpp-dataset USUARIO/SLUG [--celebdf-dataset USUARIO/SLUG] [--smoke]
#   bash kaggle/ejecutar_ffpp.sh --subir-ffpp /ruta/a/FaceForensics++     # crea el dataset privado y sigue
#
# Opciones:
#   --semillas "42 7 123"   semillas del modelo propuesto (por defecto 3)
#   --ablaciones-semillas "42"   semillas para base y ablaciones (por defecto solo 42: ahorra GPU)
#   --desde-fase N          retoma desde la fase 2 o 3 (las salidas previas ya están en Kaggle)
#   --smoke                 10 videos por conjunto y 1 época (verifica el flujo; ~30 min en total)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
trap 'echo -e "\033[1;31m✖ El script se detuvo en la línea $LINENO (código $?)\033[0m"' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$HERE")"
FF_DS=""; CDF_DS=""; UPLOAD_DIR=""; SMOKE=0; FROM=1
SEEDS="42 7 123"; ABL_SEEDS="42"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ffpp-dataset) FF_DS="$2"; shift 2 ;;
    --celebdf-dataset) CDF_DS="$2"; shift 2 ;;
    --subir-ffpp) UPLOAD_DIR="$2"; shift 2 ;;
    --semillas) SEEDS="$2"; shift 2 ;;
    --ablaciones-semillas) ABL_SEEDS="$2"; shift 2 ;;
    --desde-fase) FROM="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "Opción desconocida: $1"; exit 2 ;;
  esac
done

paso() { printf "\n\033[1;34m▶ %s\033[0m\n" "$*"; }
falla() { printf "\033[1;31m✖ %s\033[0m\n" "$*"; exit 1; }

if [[ -x "$HOME/.venvs/kaggle/bin/kaggle" ]]; then export PATH="$HOME/.venvs/kaggle/bin:$PATH"; fi
command -v kaggle >/dev/null || falla "Falta la CLI: python3 -m venv ~/.venvs/kaggle && ~/.venvs/kaggle/bin/pip install kaggle six"
[[ -n "${KAGGLE_USERNAME:-}" ]] || falla "Define tu usuario: export KAGGLE_USERNAME=tu_usuario"
kaggle kernels list --mine --page-size 1 >/dev/null || falla "Kaggle no te reconoce: ejecuta  kaggle auth login"
U="$KAGGLE_USERNAME"
SUF=""; TAG=""
if [[ $SMOKE -eq 1 ]]; then SUF=" smoke"; TAG="_smoke"; fi
NB_DIR="$HERE/notebooks_ffpp$TAG"
[[ -f "$NB_DIR/fase1_preproceso.ipynb" ]] || falla "No están los notebooks en $NB_DIR"

slug() { echo "$1" | tr 'A-Z' 'a-z' | tr ' ' '-'; }
esperar() {   # $1 = usuario/slug
  local st
  while true; do
    st="$(kaggle kernels status "$1" 2>&1 || true)"
    echo "  $(date +%H:%M:%S)  $st"
    shopt -s nocasematch
    if [[ "$st" == *complete* ]]; then shopt -u nocasematch; return 0; fi
    if [[ "$st" == *error* || "$st" == *cancel* ]]; then shopt -u nocasematch
      kaggle kernels output "$1" -p "$CODE_DIR/salida_ffpp_error/$(basename "$1")" >/dev/null 2>&1 || true
      falla "$1 terminó con error. Log en salida_ffpp_error/ y en https://www.kaggle.com/code/$1"; fi
    shopt -u nocasematch
    sleep 60
  done
}
push() {   # $1 notebook, $2 título, $3 gpu(true/false), $4 dataset_sources JSON, $5 kernel_sources JSON, $6 timeout s
  local s d; s="$(slug "$2")"; d="$HERE/kernel_tmp/$s"; rm -rf "$d"; mkdir -p "$d"
  cp "$1" "$d/notebook.ipynb"
  cat > "$d/kernel-metadata.json" <<EOF
{"id": "$U/$s", "title": "$2", "code_file": "notebook.ipynb", "language": "python",
 "kernel_type": "notebook", "is_private": true, "enable_gpu": $3, "enable_tpu": false,
 "enable_internet": true, "dataset_sources": $4, "competition_sources": [], "kernel_sources": $5}
EOF
  if [[ "$3" == "true" ]]; then kaggle kernels push -p "$d" --accelerator NvidiaTeslaT4 -t "$6"
  else kaggle kernels push -p "$d" -t "$6"; fi
  echo "  https://www.kaggle.com/code/$U/$s"
  esperar "$U/$s"
}

# ── Dataset FF++ ─────────────────────────────────────────────────────────────
if [[ -n "$UPLOAD_DIR" ]]; then
  paso "Subiendo FF++ como dataset PRIVADO (puede tardar horas según tu conexión)"
  [[ -d "$UPLOAD_DIR/original_sequences" || -d "$UPLOAD_DIR/real" ]] || falla "$UPLOAD_DIR no tiene la estructura de FF++"
  FF_DS="$U/faceforensics-pp-tesis"
  cat > "$UPLOAD_DIR/dataset-metadata.json" <<EOF
{"title": "FaceForensics++ tesis", "id": "$FF_DS", "licenses": [{"name": "other"}]}
EOF
  kaggle datasets status "$FF_DS" >/dev/null 2>&1 || kaggle datasets create -p "$UPLOAD_DIR" -r zip
  until [[ "$(kaggle datasets status "$FF_DS" 2>/dev/null || true)" == *ready* ]]; do echo "  procesando dataset…"; sleep 60; done
fi
[[ -n "$FF_DS" || $FROM -gt 1 ]] || falla "Indica --ffpp-dataset USUARIO/SLUG o --subir-ffpp CARPETA"
echo "Usuario: $U | dataset FF++: ${FF_DS:-(de una corrida anterior)} | Celeb-DF: ${CDF_DS:-no} | smoke: $SMOKE"
if [[ -n "$CDF_DS" ]]; then DS_JSON="[\"$FF_DS\", \"$CDF_DS\"]"; else DS_JSON="[\"$FF_DS\"]"; fi

F1="deepfake ffpp fase1 preproceso$SUF"
F3="deepfake ffpp fase3 analisis$SUF"
# ── Fase 1 ───────────────────────────────────────────────────────────────────
if [[ $FROM -le 1 ]]; then
  paso "Fase 1 · preprocesamiento (CPU) con $DS_JSON"
  push "$NB_DIR/fase1_preproceso.ipynb" "$F1" false "$DS_JSON" "[]" 43200
fi

# ── Fase 2 ───────────────────────────────────────────────────────────────────
RUNS=()
for s in $SEEDS; do RUNS+=("robusto_con_aumento:$s"); done
for s in $ABL_SEEDS; do for e in base_sin_aumento ablacion_solo_swin ablacion_solo_asimetria; do RUNS+=("$e:$s"); done; done
if [[ $SMOKE -eq 1 ]]; then RUNS=("robusto_con_aumento:42" "ablacion_solo_swin:42"); fi
K2=()
for r in "${RUNS[@]}"; do
  e="${r%%:*}"; s="${r##*:}"
  short="$(echo "$e" | sed 's/_con_aumento//; s/_sin_aumento//; s/ablacion_//; s/_/ /g')"
  K2+=("$U/$(slug "deepfake ffpp f2 $short s$s$SUF")")
done
if [[ $FROM -le 2 ]]; then
  for i in "${!RUNS[@]}"; do
    e="${RUNS[$i]%%:*}"; s="${RUNS[$i]##*:}"; title="$(basename "${K2[$i]}" | tr '-' ' ')"
    paso "Fase 2 · $e semilla $s (GPU T4)"
    nb="$HERE/kernel_tmp/f2_${e}_s$s.ipynb"; mkdir -p "$HERE/kernel_tmp"
    sed "s/__EXPERIMENTO__/$e/g; s/__SEMILLA__/$s/g" "$NB_DIR/fase2_entrenar.ipynb" > "$nb"
    push "$nb" "$title" true "[]" "[\"$U/$(slug "$F1")\"]" 43200
  done
fi

# ── Fase 3 ───────────────────────────────────────────────────────────────────
paso "Fase 3 · análisis (CPU)"
KS="$(printf '"%s", ' "${K2[@]}")"; KS="[${KS%, }]"
push "$NB_DIR/fase3_analisis.ipynb" "$F3" false "[]" "$KS" 7200
OUT="$CODE_DIR/salida_ffpp$TAG"
rm -rf "$OUT"; mkdir -p "$OUT"
kaggle kernels output "$U/$(slug "$F3")" -p "$OUT" >/dev/null
echo; ls -R "$OUT/analisis" 2>/dev/null | head -30
if [[ -f "$OUT/analisis/tablas/auc_tabla_principal.csv" ]]; then echo; echo "AUC (media entre semillas):"; cat "$OUT/analisis/tablas/auc_tabla_principal.csv"; fi
echo -e "\n✔ Protocolo FF++ completado. Resultados en $OUT"
