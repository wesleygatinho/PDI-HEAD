"""Desenho do esqueleto e do painel de informacoes, so com OpenCV."""
from __future__ import annotations

import cv2
import numpy as np

# As 35 conexoes oficiais entre os 33 landmarks do MediaPipe Pose.
CONEXOES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),      # rosto
    (11, 12),                                                                      # ombros
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),                    # braco esquerdo
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),                    # braco direito
    (11, 23), (12, 24), (23, 24),                                                  # tronco
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),                              # perna esquerda
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),                              # perna direita
]

# No MediaPipe: do 11 em diante, impar = lado esquerdo, par = lado direito.
_ESQUERDA = {1, 2, 3, 7, 9} | set(range(11, 33, 2))
_DIREITA = {4, 5, 6, 8, 10} | set(range(12, 33, 2))

# Cores em BGR
COR_ESQ = (0, 165, 255)      # laranja
COR_DIR = (255, 200, 0)      # azul claro
COR_CENTRO = (235, 235, 235)
COR_FUNDO = (18, 18, 18)
BRANCO = (255, 255, 255)
VERDE = (120, 230, 120)
VERMELHO = (90, 90, 255)
AMARELO = (80, 220, 255)

_FONTE = cv2.FONT_HERSHEY_SIMPLEX

MODOS = {1: "sobreposto", 2: "avatar", 3: "lado a lado", 4: "rastro"}


def _cor_ponto(i: int):
    return COR_ESQ if i in _ESQUERDA else COR_DIR if i in _DIREITA else COR_CENTRO


def _cor_conexao(a: int, b: int):
    if a in _ESQUERDA and b in _ESQUERDA:
        return COR_ESQ
    if a in _DIREITA and b in _DIREITA:
        return COR_DIR
    return COR_CENTRO


def desenhar_esqueleto(img, pts_px, vis, limiar=0.5, numeros=False, cabeca=False, escala=1.0) -> None:
    """Desenha ossos e articulacoes em `img` (modifica a propria imagem).

    pts_px: matriz 33x2 em pixels. vis: visibilidade de cada landmark;
    pontos abaixo de `limiar` nao sao desenhados (estao fora do quadro ou
    encobertos, e a posicao estimada pelo modelo nao e confiavel).
    """
    h = img.shape[0]
    esp = max(2, int(round(h / 240 * escala)))
    P = np.round(pts_px).astype(int).tolist()
    ok = vis >= limiar

    for a, b in CONEXOES:
        if ok[a] and ok[b]:
            cv2.line(img, P[a], P[b], _cor_conexao(a, b), esp, cv2.LINE_AA)

    if cabeca and ok[0]:
        if ok[7] and ok[8]:
            raio = 0.8 * float(np.linalg.norm(pts_px[7] - pts_px[8]))
        elif ok[11] and ok[12]:
            raio = 0.3 * float(np.linalg.norm(pts_px[11] - pts_px[12]))
        else:
            raio = 0.0
        if raio >= 4:
            cv2.circle(img, P[0], int(raio), COR_CENTRO, esp, cv2.LINE_AA)

    for i in range(len(P)):
        if ok[i]:
            cv2.circle(img, P[i], esp + 1, _cor_ponto(i), -1, cv2.LINE_AA)
            cv2.circle(img, P[i], esp + 1, (40, 40, 40), 1, cv2.LINE_AA)
            if numeros:
                cv2.putText(img, str(i), (P[i][0] + esp + 3, P[i][1] - esp - 1), _FONTE,
                            h / 1800, BRANCO, 1, cv2.LINE_AA)


class Compositor:
    """Monta a imagem final de acordo com o modo de visualizacao."""

    def __init__(self) -> None:
        self._rastro = None

    def compor(self, modo: int, quadro, pts_px, vis, numeros=False):
        h, w = quadro.shape[:2]
        tem_pose = pts_px is not None

        if modo == 1:        # camera + esqueleto por cima
            if tem_pose:
                desenhar_esqueleto(quadro, pts_px, vis, numeros=numeros)
            return quadro

        if modo == 4:        # esqueleto deixa um rastro que vai apagando
            if self._rastro is None or self._rastro.shape != quadro.shape:
                self._rastro = np.zeros_like(quadro)
            self._rastro = cv2.convertScaleAbs(self._rastro, alpha=0.86)
            if tem_pose:
                desenhar_esqueleto(self._rastro, pts_px, vis, cabeca=True, escala=1.3)
            return np.maximum(self._rastro, COR_FUNDO[0])

        avatar = np.full((h, w, 3), COR_FUNDO, dtype=np.uint8)
        if tem_pose:
            desenhar_esqueleto(avatar, pts_px, vis, numeros=numeros, cabeca=True, escala=1.5)
        if modo == 2:        # so o esqueleto
            return avatar
        return np.hstack([quadro, avatar])     # modo 3: camera | esqueleto


def desenhar_painel(img, linhas, canto=(12, 12)) -> None:
    """Caixa semitransparente com linhas de texto. linhas = [(texto, cor), ...]."""
    h = img.shape[0]
    fs = max(0.4, h / 720 * 0.55)
    esp = 1 if fs < 0.9 else 2
    tamanhos = [cv2.getTextSize(t, _FONTE, fs, esp)[0] for t, _ in linhas]
    alt_linha = int(max(a for _, a in tamanhos) * 1.7)
    margem = int(alt_linha * 0.5)
    larg = max(l for l, _ in tamanhos) + 2 * margem
    alt = alt_linha * len(linhas) + margem
    x0, y0 = canto
    x1, y1 = min(x0 + larg, img.shape[1]), min(y0 + alt, img.shape[0])

    img[y0:y1, x0:x1] = cv2.convertScaleAbs(img[y0:y1, x0:x1], alpha=0.35)
    for i, (texto, cor) in enumerate(linhas):
        y = y0 + alt_linha * (i + 1) - int(alt_linha * 0.15)
        cv2.putText(img, texto, (x0 + margem, y), _FONTE, fs, cor, esp, cv2.LINE_AA)


def desenhar_faixa(img, texto, cor=BRANCO) -> None:
    """Faixa grande e centralizada na parte de baixo, legivel de longe."""
    h, w = img.shape[:2]
    fs = h / 720 * 1.25
    esp = max(2, int(round(h / 360)))
    (larg, alt), _ = cv2.getTextSize(texto, _FONTE, fs, esp)
    y1 = h - int(h * 0.07)
    y0 = y1 - alt - int(h * 0.05)
    img[y0:y1] = cv2.convertScaleAbs(img[y0:y1], alpha=0.3)
    cv2.putText(img, texto, ((w - larg) // 2, y1 - int(h * 0.025)), _FONTE, fs, cor, esp, cv2.LINE_AA)


def desenhar_rodape(img, texto) -> None:
    h, w = img.shape[:2]
    fs = max(0.35, h / 720 * 0.45)
    (_, alt), _ = cv2.getTextSize(texto, _FONTE, fs, 1)
    img[h - alt - 14:h] = cv2.convertScaleAbs(img[h - alt - 14:h], alpha=0.35)
    cv2.putText(img, texto, (12, h - 8), _FONTE, fs, (210, 210, 210), 1, cv2.LINE_AA)
