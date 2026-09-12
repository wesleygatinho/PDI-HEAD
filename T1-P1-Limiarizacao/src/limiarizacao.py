# -*- coding: utf-8 -*-
"""
limiarizacao.py
===============
Nucleo matematico do trabalho T1/P1 - Processamento de Imagens (UFMA).

Este modulo implementa "na mao" (apenas NumPy) os tres metodos de limiarizacao
estudados, para que o comportamento de cada um possa ser inspecionado passo a
passo, e tambem expoe os equivalentes das bibliotecas (scikit-image / OpenCV)
para validacao cruzada.

Metodos
-------
1. Global / Otsu .......... otsu_threshold()          <-> skimage.filters.threshold_otsu
2. Multi-Otsu ............. multiotsu_threshold()     <-> skimage.filters.threshold_multiotsu
3. Adaptativa (local) ..... adaptive_threshold()      <-> cv2.adaptiveThreshold

Convencao: todas as funcoes recebem imagens em tons de cinza uint8 (0..255).
"""

from __future__ import annotations



import cv2
import numpy as np
from skimage.filters import threshold_multiotsu, threshold_otsu

# ---------------------------------------------------------------------------
# Utilidades de histograma
# ---------------------------------------------------------------------------

NBINS = 256


def histograma(img: np.ndarray, nbins: int = NBINS) -> np.ndarray:
    """Histograma absoluto (contagens) de uma imagem uint8."""
    return np.bincount(img.ravel(), minlength=nbins).astype(np.float64)


def histograma_normalizado(img: np.ndarray, nbins: int = NBINS) -> np.ndarray:
    """Histograma de probabilidade p(i) = n_i / N, com soma(p) = 1."""
    h = histograma(img, nbins)
    return h / h.sum()


# ---------------------------------------------------------------------------
# 1) LIMIARIZACAO GLOBAL - METODO DE OTSU (1979)
# ---------------------------------------------------------------------------

def otsu_variancias(img: np.ndarray, nbins: int = NBINS) -> dict:
    """
    Calcula, para TODOS os limiares t = 0..L-2, as grandezas do metodo de Otsu.

    Formulacao
    ----------
    p(i)    = n_i / N                     (histograma normalizado)
    w0(t)   = soma_{i<=t} p(i)            (peso/probabilidade da classe C0)
    w1(t)   = 1 - w0(t)                   (peso da classe C1)
    mu(t)   = soma_{i<=t} i*p(i)          (momento de 1a ordem acumulado)
    muT     = soma_i i*p(i)               (media global)
    mu0(t)  = mu(t) / w0(t)               (media da classe C0)
    mu1(t)  = (muT - mu(t)) / w1(t)       (media da classe C1)

    Variancia ENTRE classes (between):
        sigma_B^2(t) = w0*w1*(mu0 - mu1)^2
                     = (muT*w0(t) - mu(t))^2 / (w0(t)*(1 - w0(t)))

    Variancia DENTRO das classes (within):
        sigma_W^2(t) = w0*sigma0^2 + w1*sigma1^2

    Identidade fundamental (decomposicao da variancia):
        sigma_T^2 = sigma_W^2(t) + sigma_B^2(t)   -> constante para todo t

    Como sigma_T^2 nao depende de t, MAXIMIZAR sigma_B^2 equivale a MINIMIZAR
    sigma_W^2. Otsu escolhe  t* = argmax_t sigma_B^2(t).

    Retorna
    -------
    dict com vetores indexados por t (tamanho nbins-1) e o limiar otimo.
    """
    p = histograma_normalizado(img, nbins)
    niveis = np.arange(nbins, dtype=np.float64)

    w0 = np.cumsum(p)                         # w0(t)
    mu = np.cumsum(niveis * p)                # mu(t)
    muT = mu[-1]                              # media global
    varT = float(np.sum(((niveis - muT) ** 2) * p))   # sigma_T^2

    # Descarta t = L-1 (a classe C1 ficaria vazia).
    w0_t = w0[:-1]
    mu_t = mu[:-1]
    w1_t = 1.0 - w0_t

    valido = (w0_t > 0) & (w1_t > 0)

    mu0 = np.zeros_like(w0_t)
    mu1 = np.zeros_like(w0_t)
    np.divide(mu_t, w0_t, out=mu0, where=valido)
    np.divide(muT - mu_t, w1_t, out=mu1, where=valido)

    sigma_B = np.zeros_like(w0_t)
    np.divide((muT * w0_t - mu_t) ** 2, w0_t * w1_t, out=sigma_B, where=valido)

    sigma_W = varT - sigma_B                  # identidade da decomposicao

    t_otimo = int(np.argmax(sigma_B))

    return {
        "p": p,
        "t": np.arange(nbins - 1),
        "w0": w0_t,
        "w1": w1_t,
        "mu0": mu0,
        "mu1": mu1,
        "muT": muT,
        "sigma_B": sigma_B,
        "sigma_W": sigma_W,
        "sigma_T": varT,
        # separabilidade eta(t) = sigma_B^2(t) / sigma_T^2
        "eta": sigma_B / varT if varT > 0 else np.zeros_like(sigma_B),
        "limiar": t_otimo,
    }


def otsu_threshold(img: np.ndarray, nbins: int = NBINS) -> int:
    """Limiar global de Otsu implementado do zero. Retorna t* inteiro."""
    return otsu_variancias(img, nbins)["limiar"]


def separabilidade(img: np.ndarray) -> float:
    """
    Medida de separabilidade eta = sigma_B^2(t*) / sigma_T^2, no intervalo [0, 1].

    Proposta pelo proprio Otsu como indicador da qualidade do limiar:
    eta -> 1 indica duas classes bem separadas (histograma bimodal);
    eta baixo indica histograma unimodal, onde o limiar global perde sentido.
    """
    r = otsu_variancias(img)
    return float(r["eta"][r["limiar"]])


# ---------------------------------------------------------------------------
# 2) MULTI-OTSU (Otsu multinivel, Liao-Chen-Chung 2001)
# ---------------------------------------------------------------------------

def _tabelas_momentos(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Tabelas de consulta (lookup tables) de momentos acumulados por intervalo,
    conforme Liao, Chen & Chung (2001).

        P(u, v) = soma_{i=u}^{v} p(i)          (momento de ordem zero)
        S(u, v) = soma_{i=u}^{v} i * p(i)      (momento de 1a ordem)

    Construidas em O(L^2) a partir das somas acumuladas; depois cada intervalo
    e avaliado em O(1).
    """
    L = p.size
    niveis = np.arange(L, dtype=np.float64)

    cum_p = np.concatenate(([0.0], np.cumsum(p)))
    cum_s = np.concatenate(([0.0], np.cumsum(niveis * p)))

    # P[u, v] = cum_p[v+1] - cum_p[u]
    P = cum_p[None, 1:] - cum_p[:-1, None]
    S = cum_s[None, 1:] - cum_s[:-1, None]
    return P, S


def multiotsu_threshold(img: np.ndarray, classes: int = 3,
                        nbins: int = NBINS) -> list[int]:
    """
    Multi-Otsu: generaliza Otsu para M classes, buscando M-1 limiares.

    Criterio
    --------
    Para limiares 0 <= t_1 < t_2 < ... < t_{M-1} < L-1 que definem as classes
    C_k = [t_k + 1, t_{k+1}], maximiza-se a variancia entre classes:

        sigma_B^2 = soma_{k=0}^{M-1} w_k * (mu_k - muT)^2

    Usando as tabelas P e S, define-se a "variancia entre classes modificada"

        H(u, v) = S(u, v)^2 / P(u, v)

    e vale  sigma_B^2 = soma_k H(t_k + 1, t_{k+1}) - muT^2. Como muT^2 e
    constante, basta maximizar  J = soma_k H(t_k + 1, t_{k+1}).

    Custo
    -----
    A busca exaustiva ingenua e O(L^{M-1}) com O(L) por avaliacao - proibitiva
    ja para M = 5 (C(255,4) ~ 1.7e8 combinacoes). Aqui a maximizacao e feita por
    PROGRAMACAO DINAMICA sobre a LUT, em O(M*L^2):

        D[1][v]   = H(0, v)
        D[m][v]   = max_{u < v} ( D[m-1][u] + H(u+1, v) )

    onde D[m][v] e o melhor valor de J usando m intervalos para cobrir [0, v].
    A resposta e D[M][L-1] e os limiares saem por retro-rastreamento.
    """
    if classes < 2:
        raise ValueError("classes deve ser >= 2")
    if classes == 2:
        return [otsu_threshold(img, nbins)]

    p = histograma_normalizado(img, nbins)
    P, S = _tabelas_momentos(p)

    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.where(P > 0, (S ** 2) / np.where(P > 0, P, 1.0), 0.0)

    L = nbins
    M = classes

    # D[m, v] e arg[m, v]: programacao dinamica
    D = np.full((M + 1, L), -np.inf)
    arg = np.zeros((M + 1, L), dtype=np.int64)
    D[1, :] = H[0, :]

    for m in range(2, M + 1):
        # candidatos[u, v] = D[m-1, u] + H[u+1, v], apenas para u < v
        cand = np.full((L, L), -np.inf)
        cand[:-1, :] = D[m - 1, :-1, None] + H[1:, :]
        iu = np.arange(L)[:, None]
        iv = np.arange(L)[None, :]
        cand[iu >= iv] = -np.inf
        arg[m, :] = np.argmax(cand, axis=0)
        D[m, :] = cand[arg[m, :], np.arange(L)]

    # Retro-rastreamento a partir de v = L-1
    limiares = []
    v = L - 1
    for m in range(M, 1, -1):
        u = int(arg[m, v])
        limiares.append(u)
        v = u
    limiares.reverse()

    return [int(t) for t in limiares]


def aplicar_multiclasse(img: np.ndarray, limiares: list[int]) -> np.ndarray:
    """Rotula cada pixel na classe 0..M-1 conforme os limiares (np.digitize)."""
    return np.digitize(img, bins=np.asarray(limiares))


def colorir_regioes(rotulos: np.ndarray, n_classes: int) -> np.ndarray:
    """Converte o mapa de rotulos em imagem cinza com niveis equiespacados."""
    niveis = np.linspace(0, 255, n_classes).astype(np.uint8)
    return niveis[rotulos]


# ---------------------------------------------------------------------------
# 3) LIMIARIZACAO ADAPTATIVA (LOCAL)
# ---------------------------------------------------------------------------

def _soma_janela(f: np.ndarray, bloco: int) -> np.ndarray:
    """Soma de f em janelas bloco x bloco via imagem integral (O(1)/pixel)."""
    r = bloco // 2
    pad = cv2.copyMakeBorder(f, r, r, r, r, cv2.BORDER_REPLICATE)
    ii = cv2.integral(pad)                     # (H+1, W+1), com borda de zeros
    h, w = f.shape
    return (ii[bloco:bloco + h, bloco:bloco + w]
            - ii[0:h, bloco:bloco + w]
            - ii[bloco:bloco + h, 0:w]
            + ii[0:h, 0:w])


def media_local_integral(img: np.ndarray, bloco: int) -> np.ndarray:
    """
    Media local m(x, y) em janela bloco x bloco, em tempo CONSTANTE por pixel
    via imagem integral / summed-area table (Crow 1984; Bradley & Roth 2007):

        II(x, y) = soma_{x'<=x, y'<=y} I(x', y')

        soma(janela) = II(x2,y2) - II(x1-1,y2) - II(x2,y1-1) + II(x1-1,y1-1)

    Custo total O(N) independentemente do tamanho do bloco - por isso a
    limiarizacao adaptativa e viavel em video em tempo real.
    """
    f = img.astype(np.float64)
    return _soma_janela(f, bloco) / float(bloco * bloco)


def adaptive_threshold(img: np.ndarray, bloco: int = 35, C: float = 10.0,
                       metodo: str = "gaussiana",
                       inverso: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Limiarizacao adaptativa implementada do zero.

    Ideia
    -----
    Em vez de um unico T para a imagem toda, calcula-se um LIMIAR POR PIXEL a
    partir da vizinhanca B(x, y) de tamanho bloco x bloco:

        MEDIA:     T(x, y) = m(x, y) - C     m = media aritmetica da vizinhanca
        GAUSSIANA: T(x, y) = g(x, y) - C     g = media ponderada por gaussiana

    e entao   dst(x, y) = 255 se I(x, y) > T(x, y), senao 0.

    A constante C desloca o limiar e controla a sensibilidade ao ruido em
    regioes homogeneas (onde I ~ m; sem C o resultado seria ruido puro).

    O sigma da gaussiana segue a regra do OpenCV quando nao informado:
        sigma = 0.3 * ((bloco - 1) * 0.5 - 1) + 0.8

    Retorna
    -------
    (binaria uint8 0/255, mapa de limiares T(x, y) em float)
    """
    if bloco % 2 == 0 or bloco < 3:
        raise ValueError("bloco deve ser impar e >= 3")

    if metodo == "media":
        T = media_local_integral(img, bloco)
    elif metodo == "gaussiana":
        sigma = 0.3 * ((bloco - 1) * 0.5 - 1) + 0.8
        T = cv2.GaussianBlur(img.astype(np.float64), (bloco, bloco), sigma,
                             borderType=cv2.BORDER_REPLICATE)
    else:
        raise ValueError("metodo deve ser 'media' ou 'gaussiana'")

    T = T - C
    if inverso:
        binaria = np.where(img > T, 0, 255).astype(np.uint8)
    else:
        binaria = np.where(img > T, 255, 0).astype(np.uint8)
    return binaria, T


def sauvola_threshold(img: np.ndarray, bloco: int = 35, k: float = 0.2,
                      R: float = 128.0) -> np.ndarray:
    """
    Variante de Sauvola (2000), classica em binarizacao de documentos:

        T(x, y) = m(x, y) * [ 1 + k * ( s(x, y)/R - 1 ) ]

    onde s e o desvio-padrao local e R = 128 (dinamica maxima do desvio).
    Em regioes homogeneas (s -> 0) o limiar cai para m*(1 - k), evitando o
    ruido "sal e pimenta" que o metodo da media produz no fundo liso.
    Incluida para contraste com o metodo adaptativo do OpenCV.

    O desvio local usa a identidade  Var(X) = E[X^2] - (E[X])^2,  com ambos os
    momentos obtidos por imagem integral.
    """
    f = img.astype(np.float64)
    m = _soma_janela(f, bloco) / float(bloco * bloco)
    m2 = _soma_janela(f ** 2, bloco) / float(bloco * bloco)
    s = np.sqrt(np.maximum(m2 - m ** 2, 0.0))
    T = m * (1.0 + k * (s / R - 1.0))
    return np.where(f > T, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Envelopes das bibliotecas (para validacao cruzada)
# ---------------------------------------------------------------------------

def otsu_skimage(img: np.ndarray) -> int:
    return int(threshold_otsu(img))


def multiotsu_skimage(img: np.ndarray, classes: int = 3) -> list[int]:
    return [int(t) for t in threshold_multiotsu(img, classes=classes)]


def otsu_opencv(img: np.ndarray) -> tuple[int, np.ndarray]:
    t, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t), b


def adaptativa_opencv(img: np.ndarray, bloco: int = 35, C: int = 10,
                      metodo: str = "gaussiana") -> np.ndarray:
    m = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if metodo == "gaussiana"
         else cv2.ADAPTIVE_THRESH_MEAN_C)
    return cv2.adaptiveThreshold(img, 255, m, cv2.THRESH_BINARY, bloco, C)


def binarizar(img: np.ndarray, t: float, inverso: bool = False) -> np.ndarray:
    """Aplica um limiar global escalar."""
    if inverso:
        return np.where(img > t, 0, 255).astype(np.uint8)
    return np.where(img > t, 255, 0).astype(np.uint8)
