#!/usr/bin/env bash
# subir_a_kaggle.sh: Sube el dataset (privado) y ejecuta el notebook en la GPU de Kaggle.
#
# Uso (desde la raíz del repositorio):
#   export KAGGLE_USERNAME=tu_usuario_kaggle
#   bash kaggle/subir_a_kaggle.sh            # smoke test (6 videos, 1 época)
#   bash kaggle/subir_a_kaggle.sh --full     # entrenamiento completo
#   bash kaggle/subir_a_kaggle.sh --actualizar-dataset   # sube una nueva versión del dataset
#
# Requisitos: CLI de Kaggle (pip install --user kaggle) y token en ~/.kaggle/access_token
# o en la variable KAGGLE_API_TOKEN. El token no se guarda en el repositorio.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$HERE")"
DATASET_DIR="${DATASET_DIR:-$CODE_DIR/dataset}"
DATASET_SLUG="${DATASET_SLUG:-deepfake-ffpp-tesis}"
MODE="smoke"; UPDATE_DS=0
for a in "$@"; do
  case "$a" in
    --full) MODE="full" ;;
    --actualizar-dataset) UPDATE_DS=1 ;;
    *) echo "Opción desconocida: $a"; exit 2 ;;
  esac
done

if [[ "$MODE" == "smoke" ]]; then
  NOTEBOOK="$CODE_DIR/notebooks/smoke_test_kaggle.ipynb"; KTITLE="deepfake smoke test swin"; TIMEOUT=7200
else
  NOTEBOOK="$CODE_DIR/notebooks/deepfake_kaggle.ipynb"; KTITLE="deepfake swin asimetria full"; TIMEOUT=43200
fi
KSLUG="$(echo "$KTITLE" | tr ' ' '-')"
OUT_DIR="$CODE_DIR/salida_kaggle_${MODE}"

paso() { printf "\n== %s\n" "$*"; }
falla() { echo "ERROR: $*" >&2; exit 1; }

# 1) CLI y credenciales
paso "1/6 Verificando la CLI de Kaggle"
[[ -x "$HOME/.venvs/kaggle/bin/kaggle" ]] && export PATH="$HOME/.venvs/kaggle/bin:$PATH"
command -v kaggle >/dev/null || falla "No está instalada. Ejecuta:
  python3 -m venv ~/.venvs/kaggle && ~/.venvs/kaggle/bin/pip install kaggle
  (si falla por 'ensurepip': sudo apt install python3-venv  y repite)"
kaggle --version
if [[ -s "$HOME/.kaggle/access_token" ]] && grep -q "TU_TOKEN" "$HOME/.kaggle/access_token"; then
  falla "~/.kaggle/access_token contiene el texto de ejemplo 'TU_TOKEN': guarda ahí tu token real (empieza con KGAT_)"
fi
[[ "${KAGGLE_USERNAME:-}" != "tu_usuario" && "${KAGGLE_USERNAME:-}" != "tu_usuario_kaggle" ]] || falla "Cambia 'tu_usuario' por tu nombre de usuario de Kaggle"
[[ -n "${KAGGLE_USERNAME:-}" ]] || falla "Define tu usuario: export KAGGLE_USERNAME=tu_usuario_kaggle"
kaggle kernels list --mine --page-size 1 >/dev/null || falla "Kaggle no te reconoce: ejecuta  kaggle auth login  (o guarda un token en ~/.kaggle/access_token)"
echo "Autenticado como $KAGGLE_USERNAME"

# 2) Dataset
paso "2/6 Dataset $KAGGLE_USERNAME/$DATASET_SLUG"
[[ -d "$DATASET_DIR/real" && -d "$DATASET_DIR/fake" ]] || falla "No encuentro real/ y fake/ en $DATASET_DIR (define DATASET_DIR)"
echo "real: $(ls "$DATASET_DIR/real" | wc -l) videos | fake: $(ls "$DATASET_DIR/fake" | wc -l) videos"
cat > "$DATASET_DIR/dataset-metadata.json" <<EOF
{
  "title": "Deepfake FF++ tesis",
  "id": "$KAGGLE_USERNAME/$DATASET_SLUG",
  "licenses": [{"name": "other"}],
  "subtitle": "Subconjunto FaceForensics++ (real/ y fake/) para la tesis; uso privado"
}
EOF
if kaggle datasets status "$KAGGLE_USERNAME/$DATASET_SLUG" >/dev/null 2>&1; then
  if [[ $UPDATE_DS -eq 1 ]]; then
    kaggle datasets version -p "$DATASET_DIR" -m "actualización $(date +%F)" -r zip
  else
    echo "Ya existe; se reutiliza (usa --actualizar-dataset para subir una versión nueva)."
  fi
else
  echo "Creando dataset PRIVADO (FF++ no permite redistribución pública)..."
  kaggle datasets create -p "$DATASET_DIR" -r zip
fi
for i in $(seq 1 60); do
  st="$(kaggle datasets status "$KAGGLE_USERNAME/$DATASET_SLUG" 2>/dev/null || true)"
  echo "  estado del dataset: ${st:-desconocido}"
  [[ "$st" == *ready* ]] && break
  sleep 20
done

# 3) Carpeta del kernel
paso "3/6 Preparando el notebook ($MODE)"
[[ -f "$NOTEBOOK" ]] || falla "No existe $NOTEBOOK"
KDIR="$HERE/kernel_$MODE"; rm -rf "$KDIR"; mkdir -p "$KDIR"
cp "$NOTEBOOK" "$KDIR/"
cat > "$KDIR/kernel-metadata.json" <<EOF
{
  "id": "$KAGGLE_USERNAME/$KSLUG",
  "title": "$KTITLE",
  "code_file": "$(basename "$NOTEBOOK")",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["$KAGGLE_USERNAME/$DATASET_SLUG"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
cat "$KDIR/kernel-metadata.json"

# 4) Push + ejecución en GPU
paso "4/6 kaggle kernels push (GPU T4)"
kaggle kernels push -p "$KDIR" --accelerator NvidiaTeslaT4 -t "$TIMEOUT"
echo "Sigue la ejecución en: https://www.kaggle.com/code/$KAGGLE_USERNAME/$KSLUG"

# 5) Esperar a que termine
paso "5/6 Esperando el resultado (Ctrl+C para dejar de esperar; el kernel sigue corriendo)"
while true; do
  st="$(kaggle kernels status "$KAGGLE_USERNAME/$KSLUG" 2>&1 || true)"
  echo "  $(date +%H:%M:%S)  $st"
  shopt -s nocasematch
  if [[ "$st" == *complete* || "$st" == *error* || "$st" == *cancel* ]]; then shopt -u nocasematch; break; fi
  shopt -u nocasematch
  sleep 60
done

# 6) Descargar salidas
paso "6/6 Descargando salidas en $OUT_DIR"
rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"
kaggle kernels output "$KAGGLE_USERNAME/$KSLUG" -p "$OUT_DIR" || true
ls -R "$OUT_DIR" | head -50
if [[ -f "$OUT_DIR/resultados/smoke_test.json" ]]; then
  echo; echo "Resumen del smoke test:"; cat "$OUT_DIR/resultados/smoke_test.json"
fi
shopt -s nocasematch
[[ "$st" == *complete* ]] && echo "Ejecución completada en Kaggle" || falla "La ejecución terminó con error: revisa el log en $OUT_DIR (*.log) o en la web"
