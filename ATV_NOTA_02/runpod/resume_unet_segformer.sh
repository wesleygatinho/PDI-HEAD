#!/usr/bin/env bash
# Filas da U-Net e da SegFormer (pula configuracoes ja concluidas).
# Uso: tmux new-session -d -s us "bash runpod/resume_unet_segformer.sh > outputs/resume_us.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PRES="P0 P1 P2 P3 P4 P5 P6 P7 P8 P9 P10"
queue () {
  local arch=$1
  for p in $PRES; do
    [ -f "outputs/models/${arch}_${p}/meta.json" ] && { echo "[pula] ${arch}_${p}"; continue; }
    python scripts/03_train_predict.py --arch "$arch" --pre "$p" > "outputs/logs/03_train_${arch}_${p}.log" 2>&1
    tail -n 1 "outputs/logs/03_train_${arch}_${p}.log"
  done
}
echo "== U-Net e SegFormer ($(date))"
queue unet & queue segformer & wait
echo "== U-Net e SegFormer concluidos ($(date))"
