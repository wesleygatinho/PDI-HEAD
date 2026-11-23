#!/usr/bin/env bash
# Retoma o Mask2Former (sem loss auxiliar) com fila dinamica e, ao fim de TODOS os treinos,
# executa as etapas finais (benchmark isolado, pos-processamento + metricas, tabelas/figuras).
# Uso: tmux new-session -d -s m2f "bash runpod/resume_m2f.sh > outputs/resume_m2f.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PRES="P0 P1 P2 P3 P4 P5 P6 P7 P8 P9 P10"
mkdir -p outputs/locks outputs/logs
n_done () { ls outputs/models/${1}_P*/meta.json 2>/dev/null | wc -l; }

# fila dinamica: cada worker pega a proxima configuracao livre (lock atomico via mkdir)
worker () {
  for p in $PRES; do
    [ -f "outputs/models/mask2former_${p}/meta.json" ] && continue
    mkdir "outputs/locks/mask2former_${p}" 2>/dev/null || continue
    : > "outputs/logs/03_train_mask2former_${p}.log"
    python scripts/03_train_predict.py --arch mask2former --pre "$p" >> "outputs/logs/03_train_mask2former_${p}.log" 2>&1
    echo "[worker $1] $(tail -n 1 outputs/logs/03_train_mask2former_${p}.log)"
  done
}

echo "== Mask2Former sem loss auxiliar: 3 workers ($(date))"
rm -rf outputs/locks/mask2former_*
worker 1 & W1=$!
sleep 60; worker 2 & W2=$!
sleep 60; worker 3 & W3=$!
# 4o worker quando U-Net e SegFormer terminarem (VRAM: 4 x ~5 GB cabe em 24 GB)
( until [ "$(n_done unet)" -ge 11 ] && [ "$(n_done segformer)" -ge 11 ]; do sleep 120; done
  echo "== U-Net e SegFormer concluidos: abrindo worker 4 ($(date))"; worker 4 ) & W4=$!
wait $W1 $W2 $W3 $W4

until [ "$(n_done unet)" -ge 11 ] && [ "$(n_done segformer)" -ge 11 ] && [ "$(n_done mask2former)" -ge 11 ]; do
  sleep 120
done
echo "== todos os 33 treinos concluidos ($(date))"

echo "== 3b/5 latencia e memoria medidas com a GPU dedicada ($(date))"
for arch in unet segformer mask2former; do
  for p in $PRES; do
    python scripts/03_train_predict.py --arch "$arch" --pre "$p" --benchmark-only 2>&1 | grep "benchmark isolado"
  done
done

echo "== 4/5 pos-processamento + metricas ($(date))"
python scripts/04_postprocess_eval.py --workers 6 2>&1 | tee outputs/logs/04_eval.log

echo "== 5/5 tabelas e figuras ($(date))"
OMP_NUM_THREADS=4 python scripts/05_report.py 2>&1 | tee outputs/logs/05_report.log

echo "== fim ($(date))"
