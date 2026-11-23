#!/usr/bin/env bash
# Prepara o pod (template "RunPod PyTorch 2.x"). Rode uma vez.
# O Python do sistema e "externally managed" (PEP 668): cria-se um venv que herda o PyTorch do
# sistema (--system-site-packages) dentro do projeto, no Network Volume (persiste entre reinicios).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv
fi
PY=.venv/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -r requirements.txt
# pre-baixa os pesos pre-treinados (ImageNet / ADE20K) para nao competir durante o treino paralelo
$PY - <<'EOF'
import sys; sys.path.insert(0, "src")
from gsseg.models import build_model
for a in ("unet", "segformer", "mask2former"):
    build_model(a); print("pesos ok:", a)
EOF
nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null || true; free -g | head -2; df -h . | tail -1
