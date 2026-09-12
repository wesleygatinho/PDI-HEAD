# -*- coding: utf-8 -*-
"""
gerar_midia.py
==============
Produz o material de teste do trabalho:

A) IMAGENS  (data/imagens/)
   - tres imagens REAIS do acervo do scikit-image (camera, moedas, pagina);
   - uma imagem SINTETICA com GABARITO (ground truth) conhecido, degradada por
     iluminacao nao uniforme e ruido.

B) VIDEO  (data/videos/)
   - 20 s, 30 fps, 640x480 (600 quadros), com gabarito quadro a quadro.
     O video e dividido em 4 fases de 5 s, cada uma projetada para favorecer
     ou penalizar uma das tecnicas:

       Fase 1 (0-5 s)   Iluminacao uniforme ............ cenario ideal
       Fase 2 (5-10 s)  Holofote movel (gradiente espacial) ... quebra o global
       Fase 3 (10-15 s) Escurecimento global progressivo ..... quebra o limiar fixo
       Fase 4 (15-20 s) Ruido forte + perda de contraste ..... penaliza o local

   O gabarito permite medir IoU/F1 por quadro, e nao apenas comparar
   visualmente.

Uso:
    python src/gerar_midia.py
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from skimage import data

import config
import utils

# Niveis de cinza da cena "limpa" (3 populacoes -> justifica o Multi-Otsu)
FUNDO = 225
OBJ_MEDIO = 130
OBJ_ESCURO = 35


# ---------------------------------------------------------------------------
# Construcao da cena limpa (sem degradacao)
# ---------------------------------------------------------------------------

def cena_limpa(t: float, largura: int, altura: int) -> np.ndarray:
    """
    Desenha a cena ideal no instante normalizado t em [0, 1].

    A cena imita um documento sobre uma mesa clara com objetos se movendo:
      - bloco de texto estatico (objetos escuros, finos -> exigem limiar local);
      - barras de calibracao em cinza medio (a terceira populacao do histograma);
      - tres discos e um retangulo em movimento (conteudo dinamico).
    """
    img = np.full((altura, largura), FUNDO, dtype=np.uint8)

    # --- Bloco de texto estatico (parte superior) --------------------------
    linhas = [
        "PROCESSAMENTO DE IMAGENS - UFMA",
        "T1 / P1  -  LIMIARIZACAO",
        "Otsu  .  Multi-Otsu  .  Adaptativa",
        "segmentacao por limiar de intensidade",
    ]
    for i, txt in enumerate(linhas):
        cv2.putText(img, txt, (28, 52 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, int(OBJ_ESCURO), 2, cv2.LINE_AA)

    # --- Barras de cinza medio (populacao intermediaria) -------------------
    for i in range(4):
        x0 = 30 + i * 68
        cv2.rectangle(img, (x0, 212), (x0 + 46, 268), int(OBJ_MEDIO), -1)

    # --- Objetos em movimento ---------------------------------------------
    fase = 2.0 * np.pi * t
    for i, raio in enumerate((26, 20, 33)):
        cx = int(largura * (0.55 + 0.32 * np.cos(fase + i * 2.1)))
        cy = int(altura * (0.55 + 0.28 * np.sin(1.7 * fase + i * 1.3)))
        nivel = OBJ_ESCURO if i % 2 == 0 else OBJ_MEDIO
        cv2.circle(img, (cx, cy), raio, int(nivel), -1)

    # Retangulo deslizante na base
    rx = int((largura - 150) * ((t * 1.5) % 1.0))
    cv2.rectangle(img, (rx, altura - 78), (rx + 130, altura - 30),
                  int(OBJ_ESCURO), -1)

    # --- Detalhes finos: linhas de 1 px (teste de preservacao de detalhe) --
    for i in range(6):
        y = 300 + i * 9
        cv2.line(img, (330, y), (610, y), int(OBJ_ESCURO), 1)

    return img


def gabarito_binario(limpa: np.ndarray) -> np.ndarray:
    """Objeto = tudo que nao e fundo. Retorna mascara 0/255."""
    return np.where(limpa < (FUNDO - 20), 255, 0).astype(np.uint8)


def gabarito_3_classes(limpa: np.ndarray) -> np.ndarray:
    """Rotulos 0 (escuro), 1 (medio), 2 (fundo) para avaliar o Multi-Otsu."""
    lim = [(OBJ_ESCURO + OBJ_MEDIO) // 2, (OBJ_MEDIO + FUNDO) // 2]
    return np.digitize(limpa, bins=lim).astype(np.uint8)


# ---------------------------------------------------------------------------
# Degradacao radiometrica
# ---------------------------------------------------------------------------

def campo_iluminacao(largura: int, altura: int, modo: str,
                     t: float = 0.0) -> np.ndarray:
    """
    Campo multiplicativo L(x, y) que modela a iluminacao da cena.

    O modelo de formacao de imagem usado e o classico:

        I(x, y) = R(x, y) * L(x, y) + n(x, y)

    onde R e a REFLETANCIA (a cena "limpa", o que queremos segmentar) e L a
    ILUMINACAO. Um limiar global so e valido quando L e aproximadamente
    constante no espaco; e exatamente essa hipotese que o holofote quebra.
    """
    ys, xs = np.mgrid[0:altura, 0:largura]
    xn = xs / (largura - 1)
    yn = ys / (altura - 1)

    if modo == "uniforme":
        return np.ones((altura, largura), np.float32)

    if modo == "diagonal":
        # Gradiente diagonal suave: canto superior-esquerdo claro, inferior escuro
        return (1.0 - 0.62 * (0.55 * xn + 0.45 * yn)).astype(np.float32)

    if modo == "holofote":
        # Holofote que varre a cena da esquerda para a direita
        cx = 0.12 + 0.76 * t
        d2 = (xn - cx) ** 2 + (yn - 0.5) ** 2
        return (0.34 + 0.78 * np.exp(-d2 / 0.055)).astype(np.float32)

    raise ValueError(f"modo de iluminacao desconhecido: {modo}")


def degradar(limpa: np.ndarray, L: np.ndarray, ganho: float = 1.0,
             sigma_ruido: float = 0.0, contraste: float = 1.0,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Aplica  I = clip( ganho * L * R' + ruido ),  onde R' e a cena com contraste
    reduzido em torno da media (contraste < 1 aproxima as classes entre si).
    """
    r = limpa.astype(np.float32)
    if contraste != 1.0:
        media = float(r.mean())
        r = media + contraste * (r - media)

    obs = ganho * L * r
    if sigma_ruido > 0:
        rng = rng or np.random.default_rng(config.SEMENTE)
        obs = obs + rng.normal(0.0, sigma_ruido, obs.shape)
    return np.clip(obs, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# A) Imagens
# ---------------------------------------------------------------------------

def gerar_imagens() -> None:
    config.garantir_pastas()
    rng = np.random.default_rng(config.SEMENTE)

    # --- Imagens reais do acervo scikit-image ------------------------------
    reais = {
        "01_camera.png": data.camera(),      # histograma nitidamente bimodal
        "02_moedas.png": data.coins(),       # fundo com iluminacao desigual
        "03_pagina.png": data.page(),        # texto digitalizado, luz desigual
    }
    for nome, img in reais.items():
        utils.salvar(config.DIR_IMAGENS / nome, img)
        print(f"  [imagem real]      {nome:28s} {img.shape[1]}x{img.shape[0]}")

    # --- Imagem sintetica com gabarito -------------------------------------
    larg, alt = 640, 480
    limpa = cena_limpa(0.25, larg, alt)
    L = campo_iluminacao(larg, alt, "diagonal")
    observada = degradar(limpa, L, sigma_ruido=3.0, rng=rng)

    utils.salvar(config.DIR_IMAGENS / "04_sintetica.png", observada)
    utils.salvar(config.DIR_IMAGENS / "04_sintetica_limpa.png", limpa)
    utils.salvar(config.DIR_IMAGENS / "04_sintetica_gabarito.png",
                 gabarito_binario(limpa))
    utils.salvar(config.DIR_IMAGENS / "04_sintetica_gabarito3.png",
                 (gabarito_3_classes(limpa) * 127).astype(np.uint8))
    utils.salvar(config.DIR_IMAGENS / "04_sintetica_iluminacao.png",
                 (L / L.max() * 255).astype(np.uint8))
    print(f"  [imagem sintetica] 04_sintetica.png            {larg}x{alt}"
          f"  (+ limpa, gabarito, gabarito3, iluminacao)")


# ---------------------------------------------------------------------------
# B) Video
# ---------------------------------------------------------------------------

def parametros_da_fase(k: int, n: int) -> tuple[str, dict]:
    """
    Define a degradacao de cada quadro k (de n) conforme a fase do video.
    Retorna (nome_da_fase, parametros).
    """
    prog = k / n                      # progresso global em [0, 1)
    quarto = int(prog * 4)            # 0, 1, 2 ou 3
    u = (prog * 4) - quarto           # progresso dentro da fase, em [0, 1)

    if quarto == 0:
        return "1-Uniforme", dict(modo="uniforme", t=0.0, ganho=1.0,
                                  sigma=2.0, contraste=1.0)
    if quarto == 1:
        return "2-Holofote", dict(modo="holofote", t=u, ganho=1.0,
                                  sigma=2.0, contraste=1.0)
    if quarto == 2:
        # Escurecimento global progressivo (1.00 -> 0.32)
        return "3-Escurecimento", dict(modo="uniforme", t=0.0,
                                       ganho=1.0 - 0.68 * u,
                                       sigma=2.0, contraste=1.0)
    # Fase 4: ruido crescente e contraste decrescente
    return "4-Ruido", dict(modo="uniforme", t=0.0, ganho=0.95,
                           sigma=4.0 + 26.0 * u, contraste=1.0 - 0.55 * u)


def gerar_video(forcar: bool = False) -> None:
    """
    Gera o video de teste.

    IMPORTANTE - reprodutibilidade
    ------------------------------
    A cena e o ruido sao bit-exatos (semente fixa em config.SEMENTE): gerar os
    quadros em memoria duas vezes produz exatamente os mesmos arrays.

    O CODIFICADOR H.264, porem, NAO e bit-reproduzivel entre execucoes: o
    libx264 escolhe o numero de threads conforme a carga da maquina, e isso muda
    o particionamento em fatias e, portanto, os bytes do arquivo. Como o codec
    tem perdas (erro medio de ~3 niveis de cinza), regerar o video desloca as
    metricas medidas em ate ~0,03 de IoU - o que nao muda nenhuma conclusao, mas
    impediria reproduzir as tabelas do relatorio numero a numero.

    Por isso, se o video ja existe ele e REAPROVEITADO. O arquivo em
    data/videos/ faz parte da entrega: qualquer pessoa que rode os experimentos
    sobre ele obtem exatamente as tabelas documentadas. Use --forcar para
    regerar do zero (e entao reexecute todos os experimentos).
    """
    config.garantir_pastas()
    larg, alt = config.VIDEO_LARGURA, config.VIDEO_ALTURA
    n = config.VIDEO_N_FRAMES
    rng = np.random.default_rng(config.SEMENTE)

    cam_obs = config.DIR_VIDEOS / config.NOME_VIDEO
    cam_gt = config.DIR_VIDEOS / config.NOME_VIDEO_GT

    if cam_obs.exists() and cam_gt.exists() and not forcar:
        print(f"  [video]  {cam_obs.name} ja existe - REAPROVEITADO "
              f"({cam_obs.stat().st_size / 1e6:.1f} MB).")
        print("           Use 'python src/gerar_midia.py --forcar' para regerar.")
        return

    escritor = utils.abrir_escritor(cam_obs, config.VIDEO_FPS, (larg, alt))
    escritor_gt = utils.abrir_escritor(cam_gt, config.VIDEO_FPS, (larg, alt))

    for k in range(n):
        t = k / n
        limpa = cena_limpa(t, larg, alt)
        nome_fase, par = parametros_da_fase(k, n)
        L = campo_iluminacao(larg, alt, par["modo"], par["t"])
        obs = degradar(limpa, L, ganho=par["ganho"], sigma_ruido=par["sigma"],
                       contraste=par["contraste"], rng=rng)

        escritor.write(obs)
        escritor_gt.write(gabarito_binario(limpa))

        if (k + 1) % 150 == 0:
            print(f"    quadro {k + 1:4d}/{n}  (fase {nome_fase})")

    escritor.release()
    escritor_gt.release()
    print(f"  [video]  {cam_obs.name}  "
          f"{config.VIDEO_DURACAO_S}s @ {config.VIDEO_FPS}fps  {larg}x{alt}")
    print(f"  [video]  {cam_gt.name}  (gabarito quadro a quadro)")


def main() -> None:
    print("=" * 74)
    print("GERACAO DO MATERIAL DE TESTE")
    print("=" * 74)
    print("\n[A] Imagens")
    gerar_imagens()
    print("\n[B] Video de 20 s")
    gerar_video()
    print("\nConcluido.\n")


if __name__ == "__main__":
    main()
