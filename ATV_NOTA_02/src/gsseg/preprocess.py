"""Dez metodos de pre-processamento do protocolo da turma (P1-P10) + baseline (P0).

Baseline (P0): campo bruto, sem filtro espacial. As concentracoes do Gray-Scott ja sao
adimensionais (concentracao de alimentacao de A igual a 1), e essa e a normalizacao padrao
escolhida como referencia; P1-P5 comparam outras normalizacoes contra ela, e P6-P10 aplicam
cada filtro espacial sobre a mesma entrada de referencia, separando o efeito da normalizacao
do efeito do filtro.

P1-P5 transformam VALORES (estatisticas ajustadas apenas no conjunto de treino, reutilizadas
em validacao e teste). P6-P10 sao filtros ESPACIAIS com contorno periodico (o dominio do
Gray-Scott no The Well e periodico em x e y). Um unico canal fisico (A).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.restoration import denoise_wavelet

PRE_NAMES = {
    "P0": "Baseline (dimensionless raw field, no spatial filter)",
    "P1": "Z-score normalization",
    "P2": "RMS normalization",
    "P3": "Robust scaling (median/IQR)",
    "P4": "Min-max normalization",
    "P5": "Percentile clipping (1-99%)",
    "P6": "Gaussian smoothing",
    "P7": "Median filtering",
    "P8": "Bilateral filtering",
    "P9": "Anisotropic diffusion (Perona-Malik)",
    "P10": "Wavelet denoising",
}

PRE_PARAMS = {
    "P0": "dimensionless concentration A, native 128x128",
    "P1": "mean/std of training set",
    "P2": "x / RMS, RMS = sqrt(mean(x^2)) of training set",
    "P3": "median/IQR of training set",
    "P4": "min/max of training set",
    "P5": "clip to [p1, p99] of training set (no rescaling)",
    "P6": "sigma = 1.0 px, periodic",
    "P7": "3x3 window, periodic",
    "P8": "d = 5, sigma_color = 0.1, sigma_space = 2.0, periodic",
    "P9": "15 iterations, kappa = 0.05, gamma = 0.2, exponential conductance, periodic",
    "P10": "Daubechies-4, BayesShrink soft thresholding, 3 levels",
}


def fit_stats(train: np.ndarray) -> dict:
    """Estatisticas globais do conjunto de TREINO (sem vazamento para val/teste)."""
    x = train.astype(np.float64).ravel()
    sub = x[:: max(1, x.size // 5_000_000)]  # amostra para quantis (rapido e estavel)
    q1, med, q3 = np.percentile(sub, [25, 50, 75])
    p1, p99 = np.percentile(sub, [1, 99])
    return dict(
        mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max()),
        rms=float(np.sqrt((x ** 2).mean())), median=float(med), iqr=float(q3 - q1),
        p1=float(p1), p99=float(p99), frac_below_p1=float((x < p1).mean()), frac_above_p99=float((x > p99).mean()),
    )


def _per_frame(fn, x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1, *x.shape[-2:])
    out = np.empty_like(flat, dtype=np.float32)
    for i in range(flat.shape[0]):
        out[i] = fn(flat[i])
    return out.reshape(x.shape)


def _wrap(fn, pad: int):
    def g(img):
        p = np.pad(img, pad, mode="wrap")
        return fn(p)[pad:-pad, pad:-pad]
    return g


def perona_malik(x: np.ndarray, n_iter=15, kappa=0.05, gamma=0.2) -> np.ndarray:
    """Difusao anisotropica de Perona-Malik vetorizada; np.roll => contorno periodico."""
    u = x.astype(np.float32).copy()
    ax_h, ax_w = u.ndim - 2, u.ndim - 1
    for _ in range(n_iter):
        dn = np.roll(u, 1, ax_h) - u
        ds = np.roll(u, -1, ax_h) - u
        de = np.roll(u, -1, ax_w) - u
        dw = np.roll(u, 1, ax_w) - u
        c = lambda d: np.exp(-(d / kappa) ** 2)  # noqa: E731
        u += gamma * (c(dn) * dn + c(ds) * ds + c(de) * de + c(dw) * dw)
    return u


def apply_pre(method: str, x: np.ndarray, st: dict) -> np.ndarray:
    """x: [..., H, W] campo bruto. Retorna float32 com o mesmo shape."""
    x = x.astype(np.float32)
    if method == "P0":
        return x
    if method == "P1":
        return (x - st["mean"]) / (st["std"] + 1e-8)
    if method == "P2":
        return x / (st["rms"] + 1e-8)
    if method == "P3":
        return (x - st["median"]) / (st["iqr"] + 1e-8)
    if method == "P4":
        return (x - st["min"]) / (st["max"] - st["min"] + 1e-8)
    if method == "P5":
        return np.clip(x, st["p1"], st["p99"]).astype(np.float32)
    if method == "P6":
        return ndi.gaussian_filter(x, sigma=[0] * (x.ndim - 2) + [1.0, 1.0], mode="wrap").astype(np.float32)
    if method == "P7":
        return ndi.median_filter(x, size=[1] * (x.ndim - 2) + [3, 3], mode="wrap").astype(np.float32)
    if method == "P8":
        f = _wrap(lambda im: cv2.bilateralFilter(im, d=5, sigmaColor=0.1, sigmaSpace=2.0), 4)
        return _per_frame(f, x)
    if method == "P9":
        return perona_malik(x)
    if method == "P10":
        f = _wrap(lambda im: denoise_wavelet(im, wavelet="db4", mode="soft", method="BayesShrink",
                                             wavelet_levels=3, rescale_sigma=True).astype(np.float32), 16)
        return _per_frame(f, x)
    raise ValueError(method)
