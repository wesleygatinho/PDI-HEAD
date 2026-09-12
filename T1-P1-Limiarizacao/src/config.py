# -*- coding: utf-8 -*-
"""
config.py
=========
Parametros centrais do experimento. Alterar aqui muda todos os scripts.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent

DIR_DADOS = RAIZ / "data"
DIR_IMAGENS = DIR_DADOS / "imagens"
DIR_VIDEOS = DIR_DADOS / "videos"

DIR_RESULTADOS = RAIZ / "resultados"
DIR_RES_IMAGEM = DIR_RESULTADOS / "imagem"
DIR_RES_VIDEO = DIR_RESULTADOS / "video"
DIR_RES_GRAFICOS = DIR_RESULTADOS / "graficos"
DIR_RES_METRICAS = DIR_RESULTADOS / "metricas"
DIR_RES_FRAMES = DIR_RESULTADOS / "frames"

TODAS_AS_PASTAS = [
    DIR_IMAGENS, DIR_VIDEOS, DIR_RES_IMAGEM, DIR_RES_VIDEO,
    DIR_RES_GRAFICOS, DIR_RES_METRICAS, DIR_RES_FRAMES,
]


def garantir_pastas() -> None:
    for p in TODAS_AS_PASTAS:
        p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Parametros dos metodos
# --------------------------------------------------------------------------

# Multi-Otsu
N_CLASSES = 3                 # numero de classes (M) -> M-1 = 2 limiares

# Adaptativa
BLOCO = 35                    # tamanho da janela local (impar)
CONSTANTE_C = 10              # constante C subtraida do limiar local
METODO_ADAPTATIVO = "gaussiana"   # "media" ou "gaussiana"

# Bloco "ajustado": escolhido a partir da varredura do exp03. A regra pratica e
# que o bloco deve ser MAIOR que o maior objeto (aqui o retangulo de 130 px) e
# MENOR que a escala em que a iluminacao varia (o holofote, ~300 px).
BLOCO_AJUSTADO = 151

# Varredura de sensibilidade (estudo de hiperparametros)
BLOCOS_TESTE = [11, 21, 35, 51, 75, 101]
CS_TESTE = [0, 2, 5, 10, 15, 20]

# --------------------------------------------------------------------------
# Video sintetico
# --------------------------------------------------------------------------
VIDEO_LARGURA = 640
VIDEO_ALTURA = 480
VIDEO_FPS = 30
VIDEO_DURACAO_S = 20          # requisito do enunciado: video curto de 20 s
VIDEO_N_FRAMES = VIDEO_FPS * VIDEO_DURACAO_S      # 600 frames
SEMENTE = 42                  # reprodutibilidade

NOME_VIDEO = "video_teste_20s.mp4"
NOME_VIDEO_GT = "video_teste_20s_gabarito.mp4"

# Frames salvos individualmente para ilustrar o relatorio (um por fase)
FRAMES_DESTAQUE = [60, 225, 390, 555]

# --------------------------------------------------------------------------
# Estilo dos graficos
# --------------------------------------------------------------------------
DPI = 140
CORES = {
    "otsu": "#1f77b4",
    "multiotsu": "#d62728",
    "adaptativa": "#2ca02c",
    "sauvola": "#9467bd",
    "fixo": "#7f7f7f",
}
ROTULOS = {
    "otsu": "Otsu (global)",
    "multiotsu": "Multi-Otsu (3 classes)",
    "adaptativa": "Adaptativa (gaussiana)",
    "sauvola": "Sauvola (local)",
    "fixo": "Limiar fixo T=127",
}
