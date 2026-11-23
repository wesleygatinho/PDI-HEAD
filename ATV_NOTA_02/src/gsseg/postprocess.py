"""Cinco metodos de pos-processamento do protocolo da turma (Q1-Q5) + baseline sem pos (Q0).

Entrada: probabilidades P[N, T, H, W] (classe "estrutura"), em sequencias temporais por
trajetoria. Saida: mascaras binarias uint8 [N, T, H, W]. Todos os parametros livres sao
escolhidos SOMENTE no conjunto de validacao (Dice medio) e congelados antes do teste
(ver TUNABLE). As operacoes espaciais usam preenchimento periodico ("wrap"), coerente com o
contorno periodico das simulacoes.

  Q0  baseline: P > 0,5
  Q1  threshold optimization            limiar
  Q2  morphological opening             raio do elemento estruturante (disco)
  Q3  morphological closing             raio do elemento estruturante (disco)
  Q4  connected-component filtering     area minima dos componentes
  Q5  temporal consistency filtering    janela da mediana temporal (instantes amostrados)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology as morph

POST_NAMES = {
    "Q0": "Baseline (p > 0.5)",
    "Q1": "Threshold optimization",
    "Q2": "Morphological opening",
    "Q3": "Morphological closing",
    "Q4": "Connected-component filtering",
    "Q5": "Temporal consistency filtering",
}

# grades de busca na validacao (maximiza o Dice medio)
TUNABLE = {
    "Q1": {"thr": [float(round(x, 2)) for x in np.arange(0.05, 0.96, 0.05)]},
    "Q2": {"radius": [1, 2]},
    "Q3": {"radius": [1, 2]},
    "Q4": {"min_size": [2, 4, 8, 16, 32]},
    "Q5": {"window": [3, 5]},
}

POST_PARAMS = {
    "Q0": "threshold 0.5",
    "Q1": "threshold in {0.05, ..., 0.95}, selected on validation",
    "Q2": "disk structuring element, radius in {1, 2}, selected on validation",
    "Q3": "disk structuring element, radius in {1, 2}, selected on validation",
    "Q4": "remove 4-connected components with area < a_min, a_min in {2, ..., 32} px, selected on validation",
    "Q5": "median of the probabilities over w consecutive sampled frames, w in {3, 5}, selected on validation",
}

PAD = 16


def _wrap_frames(fn, masks: np.ndarray, pad: int = PAD) -> np.ndarray:
    """Aplica fn(frame) a cada quadro com preenchimento periodico."""
    flat = masks.reshape(-1, *masks.shape[-2:])
    out = np.empty(flat.shape, dtype=np.uint8)
    for i in range(flat.shape[0]):
        out[i] = fn(np.pad(flat[i], pad, mode="wrap"))[pad:-pad, pad:-pad]
    return out.reshape(masks.shape)


def _remove_small(m, min_size):
    lab, n = ndi.label(m)
    if n == 0:
        return m
    area = np.bincount(lab.ravel())
    keep = area >= min_size
    keep[0] = False
    return keep[lab]


def apply_post(method: str, P: np.ndarray, image: np.ndarray | None = None, **kw) -> np.ndarray:
    """P: [N, T, H, W] probabilidades. `image` e aceito por compatibilidade e nao e usado."""
    P = P.astype(np.float32)
    base = (P > 0.5).astype(np.uint8)
    if method == "Q0":
        return base
    if method == "Q1":
        return (P > kw["thr"]).astype(np.uint8)
    if method == "Q2":
        se = morph.disk(kw.get("radius", 1))
        return _wrap_frames(lambda m: ndi.binary_opening(m, structure=se), base, 4)
    if method == "Q3":
        se = morph.disk(kw.get("radius", 1))
        return _wrap_frames(lambda m: ndi.binary_closing(m, structure=se), base, 4)
    if method == "Q4":
        return _wrap_frames(lambda m: _remove_small(m, kw.get("min_size", 4)), base)
    if method == "Q5":
        w = kw.get("window", 3)
        Pm = ndi.median_filter(P, size=(1, w, 1, 1), mode="nearest")
        return (Pm > 0.5).astype(np.uint8)
    raise ValueError(method)
