# -*- coding: utf-8 -*-
"""
gerar_figuras_slides.py
=======================
Gera versoes das figuras otimizadas para PROJECAO (poucos elementos, fontes
grandes, alto contraste). As figuras de resultados/graficos sao densas demais
para um slide - servem para o relatorio, nao para a sala.

Produz em apresentacao/figuras/:
    hook_holofote.png      o contraste Otsu x Adaptativa em um unico quadro
    modelo_formacao.png    I = R * L  usando as imagens reais do experimento
    strip_fases.png        as 4 fases do video lado a lado
    esvaziamento.png       por que o bloco precisa ser maior que o objeto
    limiar_global_vs_local.png   um numero T contra uma imagem T(x,y)

Uso:
    python apresentacao/figuras/gerar_figuras_slides.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import config  # noqa: E402
import gerar_midia as gm  # noqa: E402
import limiarizacao as L  # noqa: E402
import utils  # noqa: E402

SAIDA = Path(__file__).resolve().parent
DPI = 200

# Paleta alinhada com o tema da apresentacao
INK = "#16262E"
C_OTSU = "#1F6FB2"
C_MULTI = "#C0392B"
C_ADAPT = "#2E8B57"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": "white",
})


def quadro(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (quadro observado, gabarito) do quadro k do video."""
    cap = cv2.VideoCapture(str(config.DIR_VIDEOS / config.NOME_VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError("Rode antes: python executar_tudo.py")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"nao consegui ler o quadro {k}")
    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gt = gm.gabarito_binario(gm.cena_limpa(k / n, 640, 480))
    return frame, gt


def _sem_eixos(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_edgecolor("#C8D0D4")


# ---------------------------------------------------------------------------

def fig_hook() -> None:
    """O slide de abertura: mesmo quadro, dois resultados opostos."""
    frame, gt = quadro(225)
    otsu = L.binarizar(frame, L.otsu_threshold(frame), inverso=True)
    ada, _ = L.adaptive_threshold(frame, config.BLOCO_AJUSTADO,
                                  config.CONSTANTE_C, "gaussiana", inverso=True)
    iou_o = utils.metricas(otsu, gt)["iou"]
    iou_a = utils.metricas(ada, gt)["iou"]

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    dados = [
        (frame, "O quadro", None, INK),
        (otsu, "Limiar global (Otsu)", iou_o, C_OTSU),
        (ada, "Limiar local (adaptativa)", iou_a, C_ADAPT),
    ]
    for a, (img, titulo, iou, cor) in zip(ax, dados):
        a.imshow(img, cmap="gray", vmin=0, vmax=255)
        rot = titulo if iou is None else f"{titulo}\nIoU = {iou:.3f}"
        a.set_title(rot, fontsize=14, color=cor,
                    fontweight="bold" if iou is not None else "normal")
        _sem_eixos(a)
    fig.tight_layout()
    fig.savefig(SAIDA / "hook_holofote.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  hook_holofote.png          Otsu {iou_o:.3f}  x  adaptativa {iou_a:.3f}")


def fig_modelo_formacao() -> None:
    """I = R * L com as imagens reais: refletancia, iluminacao, observada."""
    R = utils.ler_cinza(config.DIR_IMAGENS / "04_sintetica_limpa.png")
    Lum = utils.ler_cinza(config.DIR_IMAGENS / "04_sintetica_iluminacao.png")
    I = utils.ler_cinza(config.DIR_IMAGENS / "04_sintetica.png")

    fig, ax = plt.subplots(1, 5, figsize=(15, 3.4),
                           gridspec_kw={"width_ratios": [1, .16, 1, .16, 1]})
    for a, img, tit, sub in [
        (ax[0], R, "refletância  $R(x,y)$", "o que queremos segmentar"),
        (ax[2], Lum, "iluminação  $L(x,y)$", "o estorvo"),
        (ax[4], I, "imagem  $I(x,y)$", "o que a câmera entrega"),
    ]:
        a.imshow(img, cmap="gray", vmin=0, vmax=255)
        a.set_title(tit, fontsize=15)
        a.set_xlabel(sub, fontsize=11, color="#5A6B73")
        _sem_eixos(a)

    for a, simbolo in [(ax[1], r"$\times$"), (ax[3], r"$=$")]:
        a.text(.5, .5, simbolo, fontsize=34, ha="center", va="center",
               color=INK, transform=a.transAxes)
        a.axis("off")

    fig.tight_layout()
    fig.savefig(SAIDA / "modelo_formacao.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  modelo_formacao.png        I = R x L")


def fig_strip_fases() -> None:
    """As quatro fases do video, com o que cada uma testa."""
    rotulos = [
        (60, "Fase 1  (0–5 s)", "iluminação uniforme"),
        (225, "Fase 2  (5–10 s)", "holofote: a luz varia no ESPAÇO"),
        (390, "Fase 3  (10–15 s)", "a cena escurece: varia no TEMPO"),
        (555, "Fase 4  (15–20 s)", "ruído forte, baixo contraste"),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for a, (k, titulo, sub) in zip(ax, rotulos):
        frame, _ = quadro(k)
        a.imshow(frame, cmap="gray", vmin=0, vmax=255)
        a.set_title(titulo, fontsize=14, fontweight="bold")
        a.set_xlabel(sub, fontsize=11.5, color="#5A6B73")
        _sem_eixos(a)
    fig.tight_layout()
    fig.savefig(SAIDA / "strip_fases.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  strip_fases.png            4 fases do vídeo")


def fig_esvaziamento() -> None:
    """Recorte mostrando o objeto oco com bloco pequeno e cheio com bloco certo."""
    frame, gt = quadro(60)
    y0, y1, x0, x1 = 190, 300, 10, 320      # faixa das barras de cinza medio

    painel = []
    for b in (35, 151):
        m, _ = L.adaptive_threshold(frame, b, config.CONSTANTE_C,
                                    "gaussiana", inverso=True)
        painel.append((b, m, utils.metricas(m, gt)["iou"]))

    fig, ax = plt.subplots(1, 3, figsize=(14, 2.9))
    ax[0].imshow(frame[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("recorte do quadro", fontsize=13)
    _sem_eixos(ax[0])

    for a, (b, m, iou) in zip(ax[1:], painel):
        a.imshow(m[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
        a.set_title(f"bloco = {b} px", fontsize=13,
                    color=C_MULTI if b == 35 else C_ADAPT, fontweight="bold")
        a.set_xlabel("interior perdido" if b == 35 else "objeto preenchido",
                     fontsize=11.5, color="#5A6B73")
        _sem_eixos(a)

    fig.tight_layout()
    fig.savefig(SAIDA / "esvaziamento.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  esvaziamento.png           b=35 IoU {painel[0][2]:.3f} | "
          f"b=151 IoU {painel[1][2]:.3f}")


def fig_global_vs_local() -> None:
    """Um numero contra uma imagem de limiares."""
    frame, _ = quadro(225)
    t = L.otsu_threshold(frame)
    _, T = L.adaptive_threshold(frame, config.BLOCO_AJUSTADO,
                                config.CONSTANTE_C, "gaussiana")

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.7))
    ax[0].imshow(frame, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("imagem", fontsize=14)
    _sem_eixos(ax[0])

    ax[1].text(.5, .55, f"$T = {t}$", fontsize=44, ha="center", va="center",
               color=C_OTSU, transform=ax[1].transAxes)
    ax[1].text(.5, .24, "um número\npara a imagem inteira", fontsize=12.5,
               ha="center", va="center", color="#5A6B73",
               transform=ax[1].transAxes)
    ax[1].set_title("limiar GLOBAL", fontsize=14, color=C_OTSU,
                    fontweight="bold")
    ax[1].axis("off")

    im = ax[2].imshow(T, cmap="inferno")
    ax[2].set_title("limiar LOCAL  $T(x,y)$", fontsize=14, color=C_ADAPT,
                    fontweight="bold")
    ax[2].set_xlabel("uma imagem inteira de limiares", fontsize=12,
                     color="#5A6B73")
    _sem_eixos(ax[2])
    fig.colorbar(im, ax=ax[2], fraction=0.046)

    fig.tight_layout()
    fig.savefig(SAIDA / "limiar_global_vs_local.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("  limiar_global_vs_local.png um número x uma imagem de limiares")


def fig_otsu_anatomia() -> None:
    """
    Versao para projecao da "anatomia do Otsu": so o histograma e as curvas de
    variancia, lado a lado e com fonte grande. A figura do relatorio tem quatro
    paineis e fica ilegivel no projetor.
    """
    img = utils.ler_cinza(config.DIR_IMAGENS / "04_sintetica.png")
    r = L.otsu_variancias(img)
    t = r["limiar"]

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))

    ax[0].bar(np.arange(256), r["p"], width=1.0, color="#9AA7AD")
    ax[0].axvline(t, color=C_OTSU, lw=2.6, label=f"$t^*$ = {t}")
    ax[0].set_title("histograma  $p(i)$", fontsize=15)
    ax[0].set_xlabel("nível de cinza")
    ax[0].set_yticks([])
    ax[0].legend(fontsize=12, frameon=False)

    # Curvas rotuladas no proprio grafico (sem caixa de legenda, que cobriria
    # justamente a regiao de interesse).
    ax[1].plot(r["t"], r["sigma_B"], color=C_OTSU, lw=2.8)
    ax[1].plot(r["t"], r["sigma_W"], color=C_MULTI, lw=2.8)
    ax[1].axhline(r["sigma_T"], color=INK, ls="--", lw=1.6)
    ax[1].axvline(t, color="#9AA7AD", ls=":", lw=2)
    ax[1].set_ylim(-0.06 * r["sigma_T"], 1.20 * r["sigma_T"])

    ax[1].text(126, r["sigma_T"] * 1.06, r"$\sigma_T^2$  constante",
               fontsize=13, color=INK, ha="center")
    ax[1].text(78, 0.17 * r["sigma_T"],
               r"$\sigma_B^2(t)$  entre classes", fontsize=13, color=C_OTSU,
               ha="center", va="center")
    ax[1].text(196, r["sigma_W"][196] + 0.05 * r["sigma_T"],
               r"$\sigma_W^2(t)$  dentro", fontsize=13, color=C_MULTI,
               ha="center", va="bottom")

    ax[1].annotate("pico", xy=(t, r["sigma_B"][t]),
                   xytext=(t - 52, r["sigma_B"][t] + 0.14 * r["sigma_T"]),
                   fontsize=13, color=C_OTSU, ha="center",
                   arrowprops=dict(arrowstyle="->", color=C_OTSU, lw=1.6))
    ax[1].annotate("vale", xy=(t, r["sigma_W"][t]),
                   xytext=(t - 52, r["sigma_W"][t] - 0.13 * r["sigma_T"]),
                   fontsize=13, color=C_MULTI, ha="center",
                   arrowprops=dict(arrowstyle="->", color=C_MULTI, lw=1.6))

    ax[1].set_title("o critério de Otsu", fontsize=15)
    ax[1].set_xlabel("limiar $t$")
    ax[1].set_yticks([])
    for lado in ("top", "right"):
        ax[1].spines[lado].set_visible(False)
        ax[0].spines[lado].set_visible(False)

    fig.tight_layout()
    fig.savefig(SAIDA / "otsu_anatomia.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  otsu_anatomia.png          t* = {t}")


def fig_try_all_paisagem() -> None:
    """Os sete limiares globais em 2x4 (paisagem), legivel no projetor."""
    from skimage.filters import (threshold_isodata, threshold_li,
                                 threshold_mean, threshold_minimum,
                                 threshold_otsu, threshold_triangle,
                                 threshold_yen)

    img = utils.ler_cinza(config.DIR_IMAGENS / "02_moedas.png")
    metodos = [
        ("Mean", threshold_mean), ("Isodata", threshold_isodata),
        ("Otsu", threshold_otsu), ("Li", threshold_li),
        ("Yen", threshold_yen), ("Minimum", threshold_minimum),
        ("Triangle", threshold_triangle),
    ]

    fig, ax = plt.subplots(2, 4, figsize=(15, 6.2))
    ax = ax.ravel()
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("original", fontsize=14)
    _sem_eixos(ax[0])

    for a, (nome, fn) in zip(ax[1:], metodos):
        t = float(fn(img))
        a.imshow(img > t, cmap="gray")
        destaque = nome == "Otsu"
        a.set_title(f"{nome}   T = {t:.0f}", fontsize=14,
                    color=C_OTSU if destaque else INK,
                    fontweight="bold" if destaque else "normal")
        _sem_eixos(a)

    fig.tight_layout()
    fig.savefig(SAIDA / "try_all_paisagem.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  try_all_paisagem.png       7 métodos globais")


def fig_limiar_no_tempo() -> None:
    """
    Versao para projecao do grafico do limiar ao longo do video: fontes grandes,
    sem titulo interno (o titulo e o do slide) e sem a curva do Sauvola, que nao
    tem limiar escalar.
    """
    import csv
    cam = RAIZ / "resultados" / "metricas" / "video_quadro_a_quadro.csv"
    if not cam.exists():
        print("  [pulado] limiar_no_tempo: rode antes o executar_tudo.py")
        return
    linhas = list(csv.DictReader(open(cam, encoding="utf-8-sig"), delimiter=";"))

    def col(nome):
        return np.array([float(r[nome].replace(",", ".")) for r in linhas])

    t = col("tempo_s")
    fases = [(0, 5, "1 · uniforme", "#EAF3EC"), (5, 10, "2 · holofote", "#FBF0E3"),
             (10, 15, "3 · escurece", "#E6EFF7"), (15, 20, "4 · ruído", "#F9E9EE")]

    fig, ax = plt.subplots(figsize=(14, 5.4))
    for ini, fim, nome, cor in fases:
        ax.axvspan(ini, fim, color=cor, zorder=0)
        ax.text((ini + fim) / 2, 247, nome, ha="center", va="top",
                fontsize=12.5, color="#4A5A62", zorder=5)

    curvas = [
        ("limiar_fixo", "#8C9BA3", "fixo  T = 127"),
        ("limiar_otsu", C_OTSU, "Otsu"),
        ("limiar_multiotsu_t1", "#E08A3C", "Multi-Otsu  $t_1$"),
        ("limiar_multiotsu_t2", C_MULTI, "Multi-Otsu  $t_2$"),
        ("limiar_adaptativa_ajustada", C_ADAPT, "adaptativa (média de $T$)"),
    ]
    for chave, cor, rotulo in curvas:
        ax.plot(t, col(chave), color=cor, lw=2.4, label=rotulo, zorder=3)

    ax.annotate("147", xy=(10.1, 147), fontsize=12.5, color=C_OTSU,
                ha="right", va="bottom", fontweight="bold")
    ax.annotate("47", xy=(14.9, 47), fontsize=12.5, color=C_OTSU,
                ha="right", va="top", fontweight="bold")

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 255)
    ax.set_xlabel("tempo (s)", fontsize=14)
    ax.set_ylabel("limiar (nível de cinza)", fontsize=14)
    ax.legend(fontsize=12, ncol=3, frameon=False, loc="lower left")
    ax.grid(alpha=.25, zorder=1)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)

    fig.tight_layout()
    fig.savefig(SAIDA / "limiar_no_tempo_slide.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  limiar_no_tempo_slide.png  limiar ao longo dos 20 s")


def _mapa_erro(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Branco = fundo certo, preto = objeto certo, vermelho = FP, azul = FN."""
    p, g = pred > 0, gt > 0
    rgb = np.zeros((*pred.shape, 3), np.uint8)
    rgb[~p & ~g] = (255, 255, 255)
    rgb[p & g] = (35, 35, 35)
    rgb[p & ~g] = (214, 45, 45)
    rgb[~p & g] = (45, 95, 214)
    return rgb


def fig_erros_slide() -> None:
    """
    Mapas de erro so nas duas fases decisivas (holofote e ruido) e nos tres
    metodos que contam. A figura do relatorio tem 4x5 paineis e fica ilegivel
    no projetor.
    """
    casos = [(225, "Fase 2 · holofote"), (555, "Fase 4 · ruído")]
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.4))

    for i, (k, nome_fase) in enumerate(casos):
        frame, gt = quadro(k)
        otsu = L.binarizar(frame, L.otsu_threshold(frame), inverso=True)
        t1, _ = L.multiotsu_threshold(frame, 3)
        mo = L.binarizar(frame, t1, inverso=True)
        ada, _ = L.adaptive_threshold(frame, config.BLOCO_AJUSTADO,
                                      config.CONSTANTE_C, "gaussiana",
                                      inverso=True)

        ax[i, 0].imshow(frame, cmap="gray", vmin=0, vmax=255)
        ax[i, 0].set_ylabel(nome_fase, fontsize=14, fontweight="bold")
        ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])
        if i == 0:
            ax[i, 0].set_title("quadro", fontsize=14)

        for j, (m, nome, cor) in enumerate(
                [(otsu, "Otsu global", C_OTSU),
                 (mo, "Multi-Otsu $t_1$", C_MULTI),
                 (ada, "Adaptativa $b{=}151$", C_ADAPT)], start=1):
            met = utils.metricas(m, gt)
            ax[i, j].imshow(_mapa_erro(m, gt))
            ax[i, j].axis("off")
            ax[i, j].set_title(
                (f"{nome}\n" if i == 0 else "") + f"IoU = {met['iou']:.3f}",
                fontsize=13, color=cor if i == 0 else INK,
                fontweight="bold" if i == 0 else "normal")

    fig.suptitle("vermelho = falso positivo      azul = falso negativo",
                 fontsize=13.5, color="#4A5A62", y=0.035)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(SAIDA / "erros_slide.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  erros_slide.png            mapas de erro, 2 fases x 3 métodos")


def fig_multiotsu_M() -> None:
    """
    Efeito do numero de classes M, versao para slide: so M = 2, 3 e 4, sem
    titulo interno (o titulo e o do slide) e com fonte grande.
    """
    img = utils.ler_cinza(config.DIR_IMAGENS / "04_sintetica.png")
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 5.0))

    for j, M in enumerate((2, 3, 4)):
        lim = L.multiotsu_threshold(img, M)
        rot = L.aplicar_multiclasse(img, lim)

        ax[0, j].hist(img.ravel(), bins=256, range=(0, 256), color="#9AA7AD")
        for t in lim:
            ax[0, j].axvline(t, color=C_MULTI, lw=2)
        ax[0, j].set_title(f"$M = {M}$   limiares {lim}", fontsize=14)
        ax[0, j].set_yticks([])
        ax[0, j].set_xlim(0, 255)
        for lado in ("top", "right", "left"):
            ax[0, j].spines[lado].set_visible(False)

        ax[1, j].imshow(rot, cmap="viridis")
        _sem_eixos(ax[1, j])

    fig.tight_layout()
    fig.savefig(SAIDA / "multiotsu_M_slide.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  multiotsu_M_slide.png      efeito de M = 2, 3, 4")


def main() -> None:
    print("=" * 74)
    print("FIGURAS PARA OS SLIDES")
    print("=" * 74)
    fig_multiotsu_M()
    fig_hook()
    fig_modelo_formacao()
    fig_strip_fases()
    fig_esvaziamento()
    fig_global_vs_local()
    fig_otsu_anatomia()
    fig_try_all_paisagem()
    fig_limiar_no_tempo()
    fig_erros_slide()
    print("\nConcluido.\n")


if __name__ == "__main__":
    main()
