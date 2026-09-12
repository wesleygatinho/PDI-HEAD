# -*- coding: utf-8 -*-
"""
exp01_imagem.py
===============
Experimento 1 - Limiarizacao aplicada a IMAGENS ESTATICAS.

Produz, em resultados/imagem e resultados/graficos:

  A) try_all_threshold  -> os 7 metodos globais do tutorial basico do skimage
  B) analise de Otsu    -> histograma, curvas sigma_B^2(t) e sigma_W^2(t)
  C) Multi-Otsu         -> limiares, mapa de regioes e estudo de M = 2..5 classes
  D) Adaptativa         -> global x media x gaussiana x Sauvola, e o mapa T(x,y)
  E) sensibilidade      -> varredura de bloco e C na adaptativa
  F) tipos do OpenCV    -> BINARY, BINARY_INV, TRUNC, TOZERO, TOZERO_INV
  G) comparativo geral  -> as tres tecnicas lado a lado, por imagem
  H) metricas           -> IoU/F1/MCC contra o gabarito + tempos de execucao

Uso:
    python src/exp01_imagem.py
"""

from __future__ import annotations

import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from skimage.filters import try_all_threshold  # noqa: E402

import config  # noqa: E402
import limiarizacao as L  # noqa: E402
import utils  # noqa: E402

IMAGENS = [
    ("01_camera", "Camera (histograma bimodal)"),
    ("02_moedas", "Moedas (fundo desigual)"),
    ("03_pagina", "Pagina digitalizada (luz desigual)"),
    ("04_sintetica", "Sintetica com gabarito"),
]


def carregar(nome: str) -> np.ndarray:
    return utils.ler_cinza(config.DIR_IMAGENS / f"{nome}.png")


# ---------------------------------------------------------------------------
# A) Os sete metodos globais do tutorial basico
# ---------------------------------------------------------------------------

def fig_try_all(img: np.ndarray, nome: str, titulo: str) -> None:
    """
    Reproduz skimage.filters.try_all_threshold: compara de uma vez
    Isodata, Li, Mean, Minimum, Otsu, Triangle e Yen.
    """
    fig, _ = try_all_threshold(img, figsize=(9, 12), verbose=False)
    fig.suptitle(f"A) Sete limiares GLOBAIS automaticos - {titulo}",
                 fontsize=13, fontweight="bold", y=1.001)
    utils.salvar_figura(fig, config.DIR_RES_GRAFICOS / f"A_try_all_{nome}.png")


# ---------------------------------------------------------------------------
# B) Anatomia do metodo de Otsu
# ---------------------------------------------------------------------------

def fig_otsu_analise(img: np.ndarray, nome: str, titulo: str) -> None:
    """
    Mostra visualmente POR QUE Otsu escolhe aquele limiar:
    o pico de sigma_B^2(t) coincide com o vale de sigma_W^2(t), e a soma das
    duas curvas e constante e igual a sigma_T^2.
    """
    r = L.otsu_variancias(img)
    t_est = r["limiar"]
    eta = r["eta"][t_est]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

    ax[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Imagem original")
    ax[0, 0].axis("off")

    ax[0, 1].bar(np.arange(256), r["p"], width=1.0, color="#666666")
    ax[0, 1].axvline(t_est, color=config.CORES["otsu"], lw=2,
                     label=f"t* = {t_est}")
    ax[0, 1].axvline(r["muT"], color="#ff7f0e", lw=1.2, ls=":",
                     label=f"media global = {r['muT']:.1f}")
    ax[0, 1].set_title("Histograma normalizado p(i)")
    ax[0, 1].set_xlabel("nivel de cinza i")
    ax[0, 1].set_ylabel("p(i)")
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].plot(r["t"], r["sigma_B"], color=config.CORES["otsu"],
                  label=r"$\sigma_B^2(t)$  (entre classes)")
    ax[1, 0].plot(r["t"], r["sigma_W"], color="#d62728",
                  label=r"$\sigma_W^2(t)$  (dentro das classes)")
    ax[1, 0].axhline(r["sigma_T"], color="k", ls="--", lw=1,
                     label=r"$\sigma_T^2$ (constante)")
    ax[1, 0].axvline(t_est, color="gray", ls=":", lw=1.5)
    ax[1, 0].set_title(r"Criterio de Otsu: $\sigma_T^2=\sigma_W^2(t)+\sigma_B^2(t)$"
                       "\n" + rf"$t^*=\arg\max \sigma_B^2(t) = {t_est}$,"
                       rf"  $\eta={eta:.3f}$")
    ax[1, 0].set_xlabel("limiar t")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.3)

    ax[1, 1].imshow(L.binarizar(img, t_est), cmap="gray", vmin=0, vmax=255)
    ax[1, 1].set_title(f"Binarizacao com t* = {t_est}")
    ax[1, 1].axis("off")

    fig.suptitle(f"B) Anatomia do metodo de Otsu - {titulo}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    utils.salvar_figura(fig, config.DIR_RES_GRAFICOS / f"B_otsu_{nome}.png")


# ---------------------------------------------------------------------------
# C) Multi-Otsu
# ---------------------------------------------------------------------------

def fig_multiotsu(img: np.ndarray, nome: str, titulo: str) -> None:
    lim = L.multiotsu_threshold(img, config.N_CLASSES)
    rot = L.aplicar_multiclasse(img, lim)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))

    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].hist(img.ravel(), bins=256, range=(0, 256), color="#666666")
    for t in lim:
        ax[1].axvline(t, color=config.CORES["multiotsu"], lw=2)
    ax[1].set_title(f"Histograma e limiares {lim}")
    ax[1].set_xlabel("nivel de cinza")

    ax[2].imshow(rot, cmap="viridis")
    ax[2].set_title(f"{config.N_CLASSES} regioes (np.digitize)")
    ax[2].axis("off")

    fig.suptitle(f"C) Multi-Otsu - {titulo}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    utils.salvar_figura(fig, config.DIR_RES_GRAFICOS / f"C_multiotsu_{nome}.png")


def fig_multiotsu_n_classes(img: np.ndarray, nome: str) -> None:
    """Efeito do numero de classes M sobre a particao do histograma."""
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
    for j, M in enumerate([2, 3, 4, 5]):
        lim = L.multiotsu_threshold(img, M)
        rot = L.aplicar_multiclasse(img, lim)

        ax[0, j].hist(img.ravel(), bins=256, range=(0, 256), color="#999999")
        for t in lim:
            ax[0, j].axvline(t, color=config.CORES["multiotsu"], lw=1.8)
        ax[0, j].set_title(f"M = {M} classes\nlimiares = {lim}", fontsize=10)
        ax[0, j].set_yticks([])

        ax[1, j].imshow(rot, cmap="viridis")
        ax[1, j].axis("off")

    fig.suptitle("C2) Multi-Otsu: efeito do numero de classes M",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    utils.salvar_figura(fig,
                        config.DIR_RES_GRAFICOS / f"C2_multiotsu_M_{nome}.png")


# ---------------------------------------------------------------------------
# D) Adaptativa
# ---------------------------------------------------------------------------

def fig_adaptativa(img: np.ndarray, nome: str, titulo: str) -> None:
    t_otsu = L.otsu_threshold(img)
    glob = L.binarizar(img, t_otsu)
    med, T_med = L.adaptive_threshold(img, config.BLOCO, config.CONSTANTE_C, "media")
    gau, T_gau = L.adaptive_threshold(img, config.BLOCO, config.CONSTANTE_C, "gaussiana")
    sau = L.sauvola_threshold(img, config.BLOCO, k=0.2)

    fig, ax = plt.subplots(2, 3, figsize=(14, 9.6))
    fig.subplots_adjust(hspace=0.28)

    ax[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Original")

    ax[0, 1].imshow(glob, cmap="gray", vmin=0, vmax=255)
    ax[0, 1].set_title(f"Global (Otsu), T = {t_otsu} para toda a imagem")

    ax[0, 2].imshow(T_gau, cmap="inferno")
    ax[0, 2].set_title(f"Mapa de limiares T(x,y)\nadaptativa gaussiana, "
                       f"bloco={config.BLOCO}, C={config.CONSTANTE_C}")

    ax[1, 0].imshow(med, cmap="gray", vmin=0, vmax=255)
    ax[1, 0].set_title("Adaptativa - MEDIA")

    ax[1, 1].imshow(gau, cmap="gray", vmin=0, vmax=255)
    ax[1, 1].set_title("Adaptativa - GAUSSIANA")

    ax[1, 2].imshow(sau, cmap="gray", vmin=0, vmax=255)
    ax[1, 2].set_title("Sauvola (usa media e desvio locais)")

    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"D) Limiarizacao adaptativa (local) - {titulo}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    utils.salvar_figura(fig, config.DIR_RES_GRAFICOS / f"D_adaptativa_{nome}.png")


def fig_sensibilidade(img: np.ndarray, nome: str) -> None:
    """Varredura bloco x C: mostra que a adaptativa e sensivel aos dois."""
    fig, ax = plt.subplots(len(config.CS_TESTE), len(config.BLOCOS_TESTE),
                           figsize=(2.3 * len(config.BLOCOS_TESTE),
                                    2.0 * len(config.CS_TESTE)))
    for i, C in enumerate(config.CS_TESTE):
        for j, b in enumerate(config.BLOCOS_TESTE):
            bina, _ = L.adaptive_threshold(img, b, C, "gaussiana")
            ax[i, j].imshow(bina, cmap="gray", vmin=0, vmax=255)
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(f"bloco={b}", fontsize=9)
            if j == 0:
                ax[i, j].text(-0.12, 0.5, f"C={C}", transform=ax[i, j].transAxes,
                              rotation=90, va="center", ha="center", fontsize=9)
    fig.suptitle("E) Adaptativa: sensibilidade ao tamanho do bloco e a constante C",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    utils.salvar_figura(fig,
                        config.DIR_RES_GRAFICOS / f"E_sensibilidade_{nome}.png")


# ---------------------------------------------------------------------------
# F) Os cinco tipos simples de limiarizacao do OpenCV
# ---------------------------------------------------------------------------

def fig_tipos_opencv(img: np.ndarray, nome: str) -> None:
    T = 127
    tipos = [
        ("THRESH_BINARY", cv2.THRESH_BINARY,
         r"dst = 255 se src > T, senao 0"),
        ("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV,
         r"dst = 0 se src > T, senao 255"),
        ("THRESH_TRUNC", cv2.THRESH_TRUNC,
         r"dst = T se src > T, senao src"),
        ("THRESH_TOZERO", cv2.THRESH_TOZERO,
         r"dst = src se src > T, senao 0"),
        ("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV,
         r"dst = 0 se src > T, senao src"),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    ax = ax.ravel()
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Original")
    ax[0].axis("off")
    for i, (rotulo, flag, formula) in enumerate(tipos, start=1):
        _, out = cv2.threshold(img, T, 255, flag)
        ax[i].imshow(out, cmap="gray", vmin=0, vmax=255)
        ax[i].set_title(f"{rotulo}  (T={T})\n{formula}", fontsize=9)
        ax[i].axis("off")
    fig.suptitle("F) Os cinco tipos de limiarizacao simples do OpenCV",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    utils.salvar_figura(fig, config.DIR_RES_GRAFICOS / f"F_tipos_opencv_{nome}.png")


# ---------------------------------------------------------------------------
# G) Comparativo geral das tres tecnicas
# ---------------------------------------------------------------------------

def fig_comparativo(img: np.ndarray, nome: str, titulo: str) -> None:
    t_otsu = L.otsu_threshold(img)
    lim_mo = L.multiotsu_threshold(img, config.N_CLASSES)

    glob = L.binarizar(img, t_otsu)
    mo_rot = L.aplicar_multiclasse(img, lim_mo)
    mo_bin = L.binarizar(L.colorir_regioes(mo_rot, config.N_CLASSES),
                         lim_mo[-1] if lim_mo else 127)
    ada, _ = L.adaptive_threshold(img, config.BLOCO, config.CONSTANTE_C,
                                  config.METODO_ADAPTATIVO)

    imagens = [
        ("Original", img),
        (f"1. Otsu global (T={t_otsu})", glob),
        (f"2. Multi-Otsu ({config.N_CLASSES} classes) T={lim_mo}",
         L.colorir_regioes(mo_rot, config.N_CLASSES)),
        (f"3. Adaptativa (bloco={config.BLOCO}, C={config.CONSTANTE_C})", ada),
    ]
    utils.painel(imagens, f"G) As tres tecnicas lado a lado - {titulo}",
                 config.DIR_RES_IMAGEM / f"G_comparativo_{nome}.png", colunas=4)

    # Tambem salva as saidas isoladas, uteis para colar no relatorio/slides
    utils.salvar(config.DIR_RES_IMAGEM / f"{nome}_1_otsu.png", glob)
    utils.salvar(config.DIR_RES_IMAGEM / f"{nome}_2_multiotsu.png",
                 L.colorir_regioes(mo_rot, config.N_CLASSES))
    utils.salvar(config.DIR_RES_IMAGEM / f"{nome}_3_adaptativa.png", ada)
    _ = mo_bin


# ---------------------------------------------------------------------------
# H) Metricas quantitativas (imagem sintetica, que possui gabarito)
# ---------------------------------------------------------------------------

def tabela_metricas() -> list[list]:
    img = carregar("04_sintetica")
    gt = utils.ler_cinza(config.DIR_IMAGENS / "04_sintetica_gabarito.png")

    t_otsu = L.otsu_threshold(img)
    lim_mo = L.multiotsu_threshold(img, 3)

    candidatos = {
        "Limiar fixo T=127": L.binarizar(img, 127, inverso=True),
        f"Otsu global (T={t_otsu})": L.binarizar(img, t_otsu, inverso=True),
        f"Multi-Otsu, classe escura (T={lim_mo[0]})":
            L.binarizar(img, lim_mo[0], inverso=True),
        f"Multi-Otsu, escura+media (T={lim_mo[1]})":
            L.binarizar(img, lim_mo[1], inverso=True),
        f"Adaptativa media (b={config.BLOCO}, C={config.CONSTANTE_C})":
            L.adaptive_threshold(img, config.BLOCO, config.CONSTANTE_C,
                                 "media", inverso=True)[0],
        f"Adaptativa gaussiana (b={config.BLOCO}, C={config.CONSTANTE_C})":
            L.adaptive_threshold(img, config.BLOCO, config.CONSTANTE_C,
                                 "gaussiana", inverso=True)[0],
        "Sauvola (b=35, k=0.2)": 255 - L.sauvola_threshold(img, 35, 0.2),
    }

    linhas = []
    for nome, pred in candidatos.items():
        m = utils.metricas(pred, gt)
        linhas.append([nome, m["acuracia"], m["precisao"], m["revocacao"],
                       m["f1_dice"], m["iou"], m["mcc"],
                       utils.ruido_sal_pimenta(pred)])
        utils.salvar(config.DIR_RES_IMAGEM /
                     ("H_" + nome.split("(")[0].strip().replace(" ", "_")
                      .replace(",", "").replace("=", "") + ".png"), pred)
    return linhas


def benchmark() -> list[list]:
    """Tempo medio por chamada, em ms, sobre a imagem sintetica 640x480."""
    img = carregar("04_sintetica")
    testes = {
        "Otsu (implementacao propria)": lambda: L.otsu_threshold(img),
        "Otsu (skimage)": lambda: L.otsu_skimage(img),
        "Otsu (OpenCV)": lambda: L.otsu_opencv(img),
        "Multi-Otsu 3 classes (propria)": lambda: L.multiotsu_threshold(img, 3),
        "Multi-Otsu 3 classes (skimage)": lambda: L.multiotsu_skimage(img, 3),
        "Multi-Otsu 4 classes (skimage)": lambda: L.multiotsu_skimage(img, 4),
        "Adaptativa media (propria, integral)":
            lambda: L.adaptive_threshold(img, 35, 10, "media"),
        "Adaptativa gaussiana (propria)":
            lambda: L.adaptive_threshold(img, 35, 10, "gaussiana"),
        "Adaptativa (OpenCV)": lambda: L.adaptativa_opencv(img, 35, 10),
        "Sauvola (propria)": lambda: L.sauvola_threshold(img, 35),
    }
    linhas = []
    for nome, fn in testes.items():
        fn()                                    # aquecimento
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        dt = (time.perf_counter() - t0) / reps * 1000.0
        linhas.append([nome, dt, 1000.0 / dt if dt > 0 else float("inf")])
    return linhas


# ---------------------------------------------------------------------------
# Validacao cruzada com as bibliotecas
# ---------------------------------------------------------------------------

def tabela_validacao() -> list[list]:
    linhas = []
    for nome, _ in IMAGENS:
        img = carregar(nome)
        o_p, o_sk, o_cv = L.otsu_threshold(img), L.otsu_skimage(img), L.otsu_opencv(img)[0]
        m_p, m_sk = L.multiotsu_threshold(img, 3), L.multiotsu_skimage(img, 3)
        a_p, _ = L.adaptive_threshold(img, 35, 10, "gaussiana")
        a_cv = L.adaptativa_opencv(img, 35, 10, "gaussiana")
        conc = 100.0 * float((a_p == a_cv).mean())
        linhas.append([nome, o_p, o_sk, o_cv, str(m_p), str(m_sk), conc,
                       L.separabilidade(img)])
    return linhas


# ---------------------------------------------------------------------------

def main() -> None:
    config.garantir_pastas()
    print("=" * 74)
    print("EXPERIMENTO 1 - IMAGENS ESTATICAS")
    print("=" * 74)

    for nome, titulo in IMAGENS:
        img = carregar(nome)
        print(f"\n[{nome}] {img.shape[1]}x{img.shape[0]}")
        fig_try_all(img, nome, titulo)
        fig_otsu_analise(img, nome, titulo)
        fig_multiotsu(img, nome, titulo)
        fig_multiotsu_n_classes(img, nome)
        fig_adaptativa(img, nome, titulo)
        fig_tipos_opencv(img, nome)
        fig_comparativo(img, nome, titulo)
        print(f"    Otsu t*={L.otsu_threshold(img)}   "
              f"Multi-Otsu={L.multiotsu_threshold(img, 3)}   "
              f"eta={L.separabilidade(img):.3f}")

    # Varredura de hiperparametros apenas nas duas imagens mais ilustrativas
    for nome in ("03_pagina", "04_sintetica"):
        fig_sensibilidade(carregar(nome), nome)

    # ---- Tabelas -----------------------------------------------------------
    print("\n[tabelas]")
    val = tabela_validacao()
    utils.escrever_csv(config.DIR_RES_METRICAS / "validacao_bibliotecas.csv",
                       ["imagem", "otsu_propria", "otsu_skimage", "otsu_opencv",
                        "multiotsu_propria", "multiotsu_skimage",
                        "concordancia_adaptativa_%", "separabilidade_eta"], val)

    met = tabela_metricas()
    utils.escrever_csv(config.DIR_RES_METRICAS / "metricas_imagem_sintetica.csv",
                       ["metodo", "acuracia", "precisao", "revocacao",
                        "f1_dice", "iou", "mcc", "ruido_isolado"], met)

    ben = benchmark()
    utils.escrever_csv(config.DIR_RES_METRICAS / "tempos_execucao.csv",
                       ["metodo", "tempo_ms", "chamadas_por_segundo"], ben)

    # ---- Relatorio em Markdown --------------------------------------------
    cam = config.DIR_RES_METRICAS / "RESUMO_imagem.md"
    with open(cam, "w", encoding="utf-8") as f:
        f.write("# Experimento 1 - Resultados sobre imagens estaticas\n\n")
        f.write("## 1. Validacao das implementacoes proprias\n\n")
        f.write("Os limiares calculados pelo codigo escrito do zero sao comparados "
                "aos das bibliotecas de referencia.\n\n")
        utils.escrever_markdown_tabela(
            f, ["Imagem", "Otsu propria", "Otsu skimage", "Otsu OpenCV",
                "Multi-Otsu propria", "Multi-Otsu skimage",
                "Concord. adapt. (%)", "eta"], val)
        f.write("## 2. Metricas contra o gabarito (imagem 04_sintetica)\n\n")
        f.write("Imagem com iluminacao diagonal nao uniforme e ruido gaussiano "
                "(sigma = 3).\n\n")
        utils.escrever_markdown_tabela(
            f, ["Metodo", "Acuracia", "Precisao", "Revocacao", "F1/Dice",
                "IoU", "MCC", "Ruido isolado"], met)
        f.write("## 3. Custo computacional (imagem 640x480)\n\n")
        utils.escrever_markdown_tabela(
            f, ["Metodo", "Tempo (ms)", "Chamadas/s"], ben)

    print(f"    -> {cam}")
    print("\nConcluido. Veja resultados/graficos, resultados/imagem e "
          "resultados/metricas.\n")


if __name__ == "__main__":
    main()
