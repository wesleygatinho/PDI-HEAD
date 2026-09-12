# -*- coding: utf-8 -*-
"""
executar_tudo.py
================
Executa o trabalho inteiro, do zero, na ordem correta:

    1. gerar_midia.py   -> imagens de teste + video de 20 s (com gabarito)
    2. exp01_imagem.py  -> limiarizacao em imagens estaticas
    3. exp02_video.py   -> limiarizacao quadro a quadro no video
    4. exp03_analise.py -> analise comparativa (mapas de erro, varreduras)

Uso:
    python executar_tudo.py                 executa tudo
    python executar_tudo.py --pular-midia   reaproveita a midia ja gerada
    python executar_tudo.py --video meu.mp4 usa um video proprio no passo 3

Tempo total tipico: 4 a 6 minutos.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SRC = RAIZ / "src"


def executar(script: str, args: list[str] | None = None) -> float:
    """Roda um script de src/ e devolve o tempo gasto em segundos."""
    cmd = [sys.executable, "-u", str(SRC / script)] + (args or [])
    print(f"\n{'=' * 74}\n>>> {script} {' '.join(args or [])}\n{'=' * 74}")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(RAIZ))
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"\n!!! {script} terminou com erro (codigo {r.returncode}).")
        sys.exit(r.returncode)
    print(f"--- {script} concluido em {dt:.1f} s")
    return dt


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline completo do T1/P1")
    ap.add_argument("--pular-midia", action="store_true",
                    help="nao regerar as imagens e o video de teste")
    ap.add_argument("--video", default=None,
                    help="usar um video proprio no experimento 2")
    args = ap.parse_args()

    print("#" * 74)
    print("#  T1 / P1 - LIMIARIZACAO - Processamento de Imagens - UFMA")
    print("#  Otsu  |  Multi-Otsu  |  Adaptativa")
    print("#" * 74)

    tempos = {}
    if not args.pular_midia:
        tempos["gerar_midia"] = executar("gerar_midia.py")
    else:
        print("\n>>> gerar_midia.py PULADO (--pular-midia)")

    tempos["exp01_imagem"] = executar("exp01_imagem.py")
    tempos["exp02_video"] = executar(
        "exp02_video.py", ["--video", args.video] if args.video else [])
    tempos["exp03_analise"] = executar("exp03_analise.py")

    print(f"\n{'=' * 74}")
    print("RESUMO DA EXECUCAO")
    print("=" * 74)
    for nome, dt in tempos.items():
        print(f"  {nome:20s} {dt:7.1f} s")
    print(f"  {'TOTAL':20s} {sum(tempos.values()):7.1f} s")

    print(f"""
Resultados gerados em: {RAIZ / 'resultados'}

  imagem/    binarizacoes e paineis comparativos das imagens estaticas
  video/     videos binarizados + mosaico_comparativo.mp4 (o principal)
  frames/    quadros de destaque, um por fase do video
  graficos/  todas as figuras (A..G das imagens, V1..V6 do video, X1..X4 da analise)
  metricas/  CSVs e os relatorios RESUMO_imagem.md e RESUMO_video.md

Leitura recomendada, nesta ordem:
  docs/00_fundamentacao_teorica.md
  docs/01_alunoA_limiarizacao_global_otsu.md
  docs/02_alunoB_multi_otsu.md
  docs/03_alunoC_limiarizacao_adaptativa.md
  docs/04_analise_comparativa.md
  docs/05_roteiro_apresentacao.md
""")


if __name__ == "__main__":
    main()
