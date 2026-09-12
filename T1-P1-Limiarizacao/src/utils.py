# -*- coding: utf-8 -*-
"""
utils.py
========
Funcoes auxiliares: E/S de imagens, metricas de avaliacao e montagem de figuras.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")          # backend sem janela (roda em qualquer terminal)
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402


# ---------------------------------------------------------------------------
# E/S
# ---------------------------------------------------------------------------

def ler_cinza(caminho: str | Path) -> np.ndarray:
    """Le uma imagem do disco em tons de cinza uint8 (aceita caminhos com acento)."""
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nao foi possivel ler a imagem: {caminho}")
    return img


def salvar(caminho: str | Path, img: np.ndarray) -> None:
    """Salva imagem lidando com caminhos que contenham acentos/espacos no Windows."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(caminho.suffix, img)
    if not ok:
        raise IOError(f"Falha ao codificar {caminho}")
    buf.tofile(str(caminho))


def abrir_escritor(caminho: str | Path, fps: float,
                   tamanho: tuple[int, int], colorido: bool = False):
    """
    Abre um cv2.VideoWriter preferindo H.264 (avc1), que gera arquivos cerca de
    3x menores que o MPEG-4 antigo (mp4v) para conteudo binario ruidoso.
    Se o codec nao estiver disponivel na instalacao do OpenCV, cai para mp4v.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    for tag in ("avc1", "mp4v"):
        esc = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*tag),
                              fps, tamanho, isColor=colorido)
        if esc.isOpened():
            return esc
        esc.release()
    raise RuntimeError(f"Nenhum codec disponivel para escrever {caminho}")


# ---------------------------------------------------------------------------
# Metricas de segmentacao (comparacao com gabarito / ground truth)
# ---------------------------------------------------------------------------

def matriz_confusao(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    """
    Conta VP, VN, FP, FN tratando valor > 0 como "objeto" (primeiro plano).
    """
    p = pred > 0
    g = gt > 0
    vp = int(np.count_nonzero(p & g))
    vn = int(np.count_nonzero(~p & ~g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    return vp, vn, fp, fn


def metricas(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Metricas classicas de segmentacao binaria.

        Acuracia  = (VP + VN) / total
        Precisao  = VP / (VP + FP)
        Revocacao = VP / (VP + FN)
        F1 (Dice) = 2*VP / (2*VP + FP + FN)
        IoU       = VP / (VP + FP + FN)            (indice de Jaccard)
        MCC       = (VP*VN - FP*FN) / sqrt((VP+FP)(VP+FN)(VN+FP)(VN+FN))

    O MCC (coeficiente de correlacao de Matthews) e robusto a classes
    desbalanceadas - importante aqui, pois o "objeto" costuma ocupar bem
    menos area que o fundo.
    """
    vp, vn, fp, fn = matriz_confusao(pred, gt)
    total = vp + vn + fp + fn

    def div(a, b):
        return float(a) / float(b) if b else 0.0

    denom_mcc = np.sqrt(float(vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))
    return {
        "VP": vp, "VN": vn, "FP": fp, "FN": fn,
        "acuracia": div(vp + vn, total),
        "precisao": div(vp, vp + fp),
        "revocacao": div(vp, vp + fn),
        "f1_dice": div(2 * vp, 2 * vp + fp + fn),
        "iou": div(vp, vp + fp + fn),
        "mcc": float((vp * vn - fp * fn) / denom_mcc) if denom_mcc > 0 else 0.0,
    }


def fracao_primeiro_plano(binaria: np.ndarray) -> float:
    """Proporcao de pixels marcados como objeto (util quando nao ha gabarito)."""
    return float(np.count_nonzero(binaria > 0)) / binaria.size


def instabilidade_temporal(atual: np.ndarray, anterior: np.ndarray) -> float:
    """
    Fracao de pixels que TROCARAM de classe entre dois quadros consecutivos.

    Em um video cujo conteudo muda pouco entre quadros vizinhos, um valor alto
    indica "cintilacao" (flicker) da segmentacao, ou seja, instabilidade do
    metodo, e nao movimento real da cena.
    """
    if anterior is None:
        return float("nan")
    return float(np.count_nonzero((atual > 0) != (anterior > 0))) / atual.size


def ruido_sal_pimenta(binaria: np.ndarray) -> float:
    """
    Estimativa simples de ruido: fracao de pixels isolados, isto e, pixels cujo
    valor difere de todos os 8 vizinhos (medido via abertura/fechamento 3x3).
    """
    b = (binaria > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    limpo = cv2.morphologyEx(cv2.morphologyEx(b, cv2.MORPH_OPEN, k),
                             cv2.MORPH_CLOSE, k)
    return float(np.count_nonzero(b != limpo)) / b.size


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def painel(imagens: list[tuple[str, np.ndarray]], titulo: str,
           caminho: str | Path, colunas: int = 3,
           cmaps: list[str] | None = None) -> None:
    """Monta uma grade de imagens com titulos e salva em disco."""
    n = len(imagens)
    linhas = int(np.ceil(n / colunas))
    fig, eixos = plt.subplots(linhas, colunas,
                              figsize=(4.2 * colunas, 4.0 * linhas))
    eixos = np.atleast_1d(eixos).ravel()

    for i, (nome, img) in enumerate(imagens):
        cmap = "gray" if cmaps is None else cmaps[i]
        eixos[i].imshow(img, cmap=cmap, vmin=0, vmax=255 if img.dtype == np.uint8 else None)
        eixos[i].set_title(nome, fontsize=10)
        eixos[i].axis("off")
    for j in range(n, len(eixos)):
        eixos[j].axis("off")

    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)


def salvar_figura(fig, caminho: str | Path) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)


def escrever_csv(caminho: str | Path, cabecalho: list[str],
                 linhas: list[list]) -> None:
    """CSV simples (sem pandas) com separador ';' e decimal ',' -> abre no Excel-BR."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8-sig", newline="") as f:
        f.write(";".join(cabecalho) + "\n")
        for linha in linhas:
            campos = []
            for v in linha:
                if isinstance(v, float):
                    campos.append(f"{v:.6f}".replace(".", ","))
                else:
                    campos.append(str(v))
            f.write(";".join(campos) + "\n")


def escrever_markdown_tabela(f, cabecalho: list[str], linhas: list[list]) -> None:
    """Escreve uma tabela Markdown em um arquivo ja aberto."""
    f.write("| " + " | ".join(cabecalho) + " |\n")
    f.write("|" + "|".join(["---"] * len(cabecalho)) + "|\n")
    for linha in linhas:
        campos = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in linha]
        f.write("| " + " | ".join(campos) + " |\n")
    f.write("\n")
