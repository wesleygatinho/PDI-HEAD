"""Numeracao sequencial e padronizada dos experimentos (igual para todos os datasets da turma).

Matriz do protocolo por dataset: 10 pre-processamentos x 3 arquiteturas x 5 pos-processamentos = 150
  E001-E050  P1..P10 x U-Net       x Q1..Q5
  E051-E100  P1..P10 x SegFormer   x Q1..Q5
  E101-E150  P1..P10 x Mask2Former x Q1..Q5
(ordem: arquitetura -> pre-processamento -> pos-processamento)

Linhas de baseline (sem pre e/ou sem pos), 16 por arquitetura, numeradas depois:
  E151-E166 U-Net, E167-E182 SegFormer, E183-E198 Mask2Former, cada bloco na ordem
  P0+Q0 (arquitetura sozinha), P0+Q1..Q5 (so pos), P1..P10+Q0 (so pre).
Total por dataset: 11 x 3 x 6 = 198 linhas.
"""
from __future__ import annotations

ARCH_ORDER = ["unet", "segformer", "mask2former"]
N_PRE, N_POST = 10, 5


def exp_number(arch: str, pre: str, post: str) -> int:
    a = ARCH_ORDER.index(arch)
    p, q = int(pre[1:]), int(post[1:])
    if p > 0 and q > 0:
        return a * N_PRE * N_POST + (p - 1) * N_POST + q
    off = 0 if (p == 0 and q == 0) else (q if p == 0 else N_POST + p)
    return len(ARCH_ORDER) * N_PRE * N_POST + 1 + a * (1 + N_POST + N_PRE) + off


def exp_id(arch: str, pre: str, post: str) -> str:
    return f"E{exp_number(arch, pre, post):03d}"


def condition(pre: str, post: str) -> str:
    p, q = pre != "P0", post != "Q0"
    return {(False, False): "No pre/post", (True, False): "Pre", (False, True): "Post",
            (True, True): "Pre+Post"}[(p, q)]


CONDITION_PT = {"No pre/post": "Arquitetura (sem pré e pós)", "Pre": "Com pré", "Post": "Com pós",
                "Pre+Post": "Pré + arquitetura + pós"}


if __name__ == "__main__":  # verificacao: 198 ids unicos e contiguos
    ids = sorted(exp_number(a, f"P{i}", f"Q{j}") for a in ARCH_ORDER for i in range(N_PRE + 1)
                 for j in range(N_POST + 1))
    assert ids == list(range(1, 199)), "numeracao com lacunas ou repeticoes"
    print("ok:", exp_id("unet", "P1", "Q1"), exp_id("mask2former", "P10", "Q5"), exp_id("unet", "P0", "Q0"),
          exp_id("mask2former", "P10", "Q0"))
