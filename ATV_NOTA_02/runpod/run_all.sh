#!/usr/bin/env bash
# Pipeline completo no RunPod (A40 48 GB). Retomavel: etapas ja concluidas sao puladas.
# Uso (dentro do pod, na pasta do projeto):  bash runpod/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1
# usa o venv criado pelo setup.sh (Python do sistema e "externally managed" no RunPod)
if [ -x .venv/bin/python ]; then export PATH="$PWD/.venv/bin:$PATH"; fi
echo "python: $(command -v python)"
export HF_HUB_DISABLE_PROGRESS_BARS=1
# 5 processos de treino em paralelo: limita threads de CPU por processo para evitar disputa.
# No RunPod, nproc mostra as CPUs do hospedeiro; a cota real do container esta em cpu.max.
CPUS=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ] && read -r Q P < /sys/fs/cgroup/cpu.max && [ "$Q" != "max" ]; then
  CPUS=$(( Q / P )); [ "$CPUS" -lt 1 ] && CPUS=1
fi
echo "CPUs disponiveis (cota do container): $CPUS"
NT=$(( CPUS / 5 )); [ "$NT" -lt 1 ] && NT=1
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT

PRES="P0 P1 P2 P3 P4 P5 P6 P7 P8 P9 P10"

echo "== 1/5 download do subconjunto ($(date))"
python scripts/01_download_subset.py --workers 32 2>&1 | tee outputs/logs/01_download.log

echo "== 2/5 ground truth + pre-processamentos ($(date))"
python scripts/02_build_dataset.py 2>&1 | tee outputs/logs/02_build.log

train_arch () {
  local arch=$1; shift
  local pres="${*:-$PRES}"
  for p in $pres; do
    if [ -f "outputs/models/${arch}_${p}/meta.json" ]; then
      echo "[pula] ${arch}_${p}"; continue
    fi
    python scripts/03_train_predict.py --arch "$arch" --pre "$p" >> "outputs/logs/03_train_${arch}_${p}.log" 2>&1
    tail -n 1 "outputs/logs/03_train_${arch}_${p}.log"
  done
}

echo "== 3/5 treinamento: 20 epocas, entrada 128x128 nativa, 5 filas em paralelo ($(date))"
# Mask2Former e o mais lento e limitado por CPU (casamento hungaro por amostra), entao sua
# fila e dividida em tres. VRAM estimada (lote 32): ~1 + 2.5 + 3x5 GB < 48 GB (A40).
# Acompanhe com: tail -f outputs/logs/03_train_*.log
train_arch unet &                                   PID1=$!
train_arch segformer &                              PID2=$!
train_arch mask2former P0 P1 P2 P3 &                PID3=$!
train_arch mask2former P4 P5 P6 P7 &                PID4=$!
train_arch mask2former P8 P9 P10 &                  PID5=$!
wait $PID1 $PID2 $PID3 $PID4 $PID5
unset OMP_NUM_THREADS MKL_NUM_THREADS

echo "== 3b/5 latencia e memoria medidas com a GPU dedicada ($(date))"
for arch in unet segformer mask2former; do
  for p in $PRES; do
    python scripts/03_train_predict.py --arch "$arch" --pre "$p" --benchmark-only 2>&1 | grep "benchmark isolado"
  done
done

echo "== 4/5 pos-processamento + metricas ($(date))"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/04_postprocess_eval.py --workers 6 2>&1 | tee outputs/logs/04_eval.log

echo "== 5/5 tabelas e figuras ($(date))"
python scripts/05_report.py 2>&1 | tee outputs/logs/05_report.log

echo "== fim ($(date)). Baixe: outputs/results, outputs/models/*/meta.json e history.csv, paper/figures, paper/tables"
