"""Construcao do ground truth.

Criterio (identico para todas as arquiteturas e condicoes):
  1. Para cada trajetoria, toma-se o campo B (especie autocatalitica) no frame
     de referencia t_ref = 500 (meio da simulacao), quando o padrao do regime
     ja esta caracterizado.
  2. Calcula-se o limiar de Otsu tau sobre esse frame.
  3. A mascara de cada instante t e M(x, y, t) = 1[B(x, y, t) > tau].

Caso degenerado: em alguns regimes (p.ex. gliders) o padrao pode se extinguir
(B -> 0 em todo o dominio) antes de t_ref. Nesse caso o frame de referencia nao
tem contraste e o Otsu nao e definido; usa-se a mediana dos limiares validos do
mesmo regime no conjunto de TREINO. Assim a mascara continua significativa nos
instantes em que a estrutura ainda existia e fica vazia depois da extincao.

O GT representa "onde a reacao autocatalitica esta ativa" segundo um criterio
fisico fixo por trajetoria; a evolucao temporal das mascaras mostra como essas
estruturas surgem, crescem, se dividem, migram ou desaparecem.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu

MIN_CONTRAST = 0.05  # max(B) - min(B) minimo no frame de referencia para o Otsu ser valido


def ref_index(steps: np.ndarray, ref_step: int) -> int:
    return int(np.argmin(np.abs(steps - ref_step)))


def otsu_taus(B: np.ndarray, steps: np.ndarray, ref_step: int) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (tau [n], valido [n]) usando o frame de referencia de cada trajetoria."""
    j = ref_index(steps, ref_step)
    taus, ok = [], []
    for i in range(B.shape[0]):
        f = B[i, j].astype(np.float64)
        valid = (f.max() - f.min()) > MIN_CONTRAST
        taus.append(threshold_otsu(f, nbins=256) if valid else np.nan)
        ok.append(valid)
    return np.array(taus, dtype=np.float32), np.array(ok)


def build_gt(B: np.ndarray, steps: np.ndarray, ref_step: int, fallback: np.ndarray) -> tuple:
    """B: [n, T, H, W]; fallback: tau por trajetoria usado quando o Otsu e invalido.

    Retorna (mascaras uint8 [n, T, H, W], tau [n], valido [n]).
    """
    taus, ok = otsu_taus(B, steps, ref_step)
    taus = np.where(ok, taus, fallback).astype(np.float32)
    masks = (B > taus[:, None, None, None]).astype(np.uint8)
    return masks, taus, ok
