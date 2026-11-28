# -*- coding: utf-8 -*-
"""
gerar_figuras_slides.py
=======================
Figuras da apresentacao, otimizadas para PROJECAO: poucos elementos, fonte
grande, alto contraste. As figuras de resultados/graficos sao densas demais
para um projetor - servem para o relatorio, nao para a sala.

A maior parte usa a MIDIA REAL baixada por src/baixar_midia.py. A cena
sintetica aparece so onde e' insubstituivel: no modelo I = R*L e nas metricas
que exigem gabarito.

Uso:
    python apresentacao/figuras/gerar_figuras_slides.py
"""

from __future__ import annotations

import csv
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

INK = "#16262E"
CINZA = "#5A6B73"
C_OTSU = "#1F6FB2"
C_MULTI = "#C0392B"
C_ADAPT = "#2E8B57"
C_FIXO = "#8C9BA3"

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
})

VIDEO_REAL = config.DIR_VIDEOS / "canyon_20s.mp4"

# Parametros da adaptativa ajustados a escala do sudoku (traco fino de caneta):
# bloco menor que no video sintetico, porque os objetos aqui sao finos.
SUDOKU_BLOCO, SUDOKU_C = 31, 7


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def _sem_eixos(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_edgecolor("#C8D0D4")


def img(nome: str) -> np.ndarray:
    for ext in (".png", ".jpg"):
        p = config.DIR_IMAGENS / f"{nome}{ext}"
        if p.exists():
            return utils.ler_cinza(p)
    raise FileNotFoundError(f"{nome}: rode antes 'python src/baixar_midia.py'")


def quadro_real(k: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(VIDEO_REAL))
    if not cap.isOpened():
        raise FileNotFoundError("rode antes 'python src/baixar_midia.py'")
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, f = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f"quadro {k} nao lido")
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def quadro_sintetico(k: int) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(config.DIR_VIDEOS / config.NOME_VIDEO))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, f = cap.read(); cap.release()
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return g, gm.gabarito_binario(gm.cena_limpa(k / n, 640, 480))


# ---------------------------------------------------------------------------
# 1. O sudoku: a demonstracao central da adaptativa
# ---------------------------------------------------------------------------

def fig_sudoku() -> None:
    g = img("10_sudoku")
    otsu = L.binarizar(g, L.otsu_threshold(g), inverso=True)
    ada, _ = L.adaptive_threshold(g, SUDOKU_BLOCO, SUDOKU_C, "gaussiana",
                                  inverso=True)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for a, (m, tit, cor) in zip(ax, [
            (g, "A foto", INK),
            (otsu, f"Otsu global  (T = {L.otsu_threshold(g)})", C_OTSU),
            (ada, f"Adaptativa  (b = {SUDOKU_BLOCO}, C = {SUDOKU_C})", C_ADAPT)]):
        a.imshow(m, cmap="gray", vmin=0, vmax=255)
        a.set_title(tit, fontsize=14, color=cor,
                    fontweight="bold" if cor != INK else "normal")
        _sem_eixos(a)
    ax[1].set_xlabel("a sombra do canto vira objeto", fontsize=11.5, color=CINZA)
    ax[2].set_xlabel("grade e dígitos recuperados", fontsize=11.5, color=CINZA)
    fig.tight_layout()
    fig.savefig(SAIDA / "sudoku.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  sudoku.png                 Otsu x adaptativa na foto do tutorial")


# ---------------------------------------------------------------------------
# 2. Modelo de formacao (cena sintetica: e a unica em que R e L sao conhecidos)
# ---------------------------------------------------------------------------

def fig_modelo_formacao() -> None:
    R = img("04_sintetica_limpa")
    Lum = img("04_sintetica_iluminacao")
    I = img("04_sintetica")

    fig, ax = plt.subplots(1, 5, figsize=(15, 3.4),
                           gridspec_kw={"width_ratios": [1, .16, 1, .16, 1]})
    for a, m, tit, sub in [
            (ax[0], R, "refletância  $R(x,y)$", "o que queremos segmentar"),
            (ax[2], Lum, "iluminação  $L(x,y)$", "o estorvo"),
            (ax[4], I, "imagem  $I(x,y)$", "o que a câmera entrega")]:
        a.imshow(m, cmap="gray", vmin=0, vmax=255)
        a.set_title(tit, fontsize=15)
        a.set_xlabel(sub, fontsize=11, color=CINZA)
        _sem_eixos(a)
    for a, s in [(ax[1], r"$\times$"), (ax[3], r"$=$")]:
        a.text(.5, .5, s, fontsize=34, ha="center", va="center",
               color=INK, transform=a.transAxes)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(SAIDA / "modelo_formacao.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  modelo_formacao.png        I = R x L")


# ---------------------------------------------------------------------------
# 3. A midia usada
# ---------------------------------------------------------------------------

def fig_midia() -> None:
    itens = [
        (img("10_sudoku"), "Sudoku fotografado", "OpenCV · Apache 2.0"),
        (img("11_ressonancia"), "Ressonância T1", "Commons · CC BY-SA 4.0"),
        (quadro_real(0), "Vídeo: pôr do sol", "NPS · domínio público"),
        (img("04_sintetica"), "Cena sintética", "gerada · tem gabarito"),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for a, (m, tit, fonte) in zip(ax, itens):
        a.imshow(m, cmap="gray", vmin=0, vmax=255)
        a.set_title(tit, fontsize=14, fontweight="bold")
        a.set_xlabel(fonte, fontsize=11, color=CINZA)
        _sem_eixos(a)
    fig.tight_layout()
    fig.savefig(SAIDA / "midia.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  midia.png                  as 4 fontes de dados")


# ---------------------------------------------------------------------------
# 4. Anatomia do Otsu, sobre um quadro real do video
# ---------------------------------------------------------------------------

def fig_otsu_anatomia() -> None:
    g = quadro_real(0)
    r = L.otsu_variancias(g)
    t = r["limiar"]

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))

    ax[0].bar(np.arange(256), r["p"], width=1.0, color="#9AA7AD")
    ax[0].axvline(t, color=C_OTSU, lw=2.6, label=f"$t^*$ = {t}")
    ax[0].set_title("histograma do quadro", fontsize=15)
    ax[0].set_xlabel("nível de cinza"); ax[0].set_yticks([])
    ax[0].legend(fontsize=12, frameon=False)

    ax[1].plot(r["t"], r["sigma_B"], color=C_OTSU, lw=2.8)
    ax[1].plot(r["t"], r["sigma_W"], color=C_MULTI, lw=2.8)
    ax[1].axhline(r["sigma_T"], color=INK, ls="--", lw=1.6)
    ax[1].axvline(t, color="#9AA7AD", ls=":", lw=2)
    ax[1].set_ylim(-0.10 * r["sigma_T"], 1.30 * r["sigma_T"])
    ax[1].text(40, r["sigma_T"] * 1.14, r"$\sigma_T^2$  constante",
               fontsize=13, color=INK, ha="center")
    # Rotulos das curvas no extremo direito, onde elas estao bem separadas:
    # ali sigma_W ja saturou perto de sigma_T e sigma_B ja caiu para perto de 0.
    ax[1].text(248, r["sigma_T"] * 1.14, r"$\sigma_W^2(t)$  dentro",
               fontsize=13, color=C_MULTI, ha="right")
    ax[1].text(248, -0.045 * r["sigma_T"], r"$\sigma_B^2(t)$  entre classes",
               fontsize=13, color=C_OTSU, ha="right", va="center")
    ax[1].annotate("pico", xy=(t, r["sigma_B"][t]),
                   xytext=(t + 70, r["sigma_B"][t] + 0.20 * r["sigma_T"]),
                   fontsize=13, color=C_OTSU, ha="center",
                   arrowprops=dict(arrowstyle="->", color=C_OTSU, lw=1.6))
    ax[1].annotate("vale", xy=(t, r["sigma_W"][t]),
                   xytext=(max(t - 55, 18), r["sigma_W"][t] - 0.24 * r["sigma_T"]),
                   fontsize=13, color=C_MULTI, ha="center",
                   arrowprops=dict(arrowstyle="->", color=C_MULTI, lw=1.6))
    ax[1].set_title("o critério de Otsu", fontsize=15)
    ax[1].set_xlabel("limiar $t$"); ax[1].set_yticks([])
    for a in ax:
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(SAIDA / "otsu_anatomia.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  otsu_anatomia.png          sobre quadro real, t* = {t}")


# ---------------------------------------------------------------------------
# 5. Multi-Otsu na ressonancia
# ---------------------------------------------------------------------------

def fig_multiotsu_mri() -> None:
    g = img("11_ressonancia")
    t_otsu = L.otsu_threshold(g)
    lim = L.multiotsu_threshold(g, 3)
    rot = L.aplicar_multiclasse(g, lim)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(g, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Ressonância T1", fontsize=14)
    _sem_eixos(ax[0])

    ax[1].hist(g.ravel(), bins=256, range=(0, 256), color="#9AA7AD")
    ax[1].axvline(t_otsu, color=C_OTSU, lw=2.4, label=f"Otsu  {t_otsu}")
    for i, t in enumerate(lim):
        ax[1].axvline(t, color=C_MULTI, lw=2.2, ls="--",
                      label=f"$t_{i+1}$ = {t}")
    ax[1].set_yscale("log")
    ax[1].set_title("histograma", fontsize=14)
    ax[1].set_xlabel("nível de cinza"); ax[1].set_yticks([])
    ax[1].legend(fontsize=11, frameon=False)
    for lado in ("top", "right", "left"):
        ax[1].spines[lado].set_visible(False)

    ax[2].imshow(L.binarizar(g, t_otsu), cmap="gray", vmin=0, vmax=255)
    ax[2].set_title("Otsu: 2 classes", fontsize=14, color=C_OTSU,
                    fontweight="bold")
    ax[2].set_xlabel("cérebro ou não-cérebro", fontsize=11.5, color=CINZA)
    _sem_eixos(ax[2])

    ax[3].imshow(rot, cmap="viridis")
    ax[3].set_title("Multi-Otsu: 3 classes", fontsize=14, color=C_MULTI,
                    fontweight="bold")
    ax[3].set_xlabel("fundo · tecido · gordura e osso", fontsize=11.5,
                     color=CINZA)
    _sem_eixos(ax[3])

    fig.tight_layout()
    fig.savefig(SAIDA / "multiotsu_mri.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  multiotsu_mri.png          Otsu {t_otsu} x Multi-Otsu {lim}")


# ---------------------------------------------------------------------------
# 6. Um numero contra uma imagem de limiares
# ---------------------------------------------------------------------------

def fig_global_vs_local() -> None:
    g = img("10_sudoku")
    t = L.otsu_threshold(g)
    _, T = L.adaptive_threshold(g, SUDOKU_BLOCO, SUDOKU_C, "gaussiana")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].imshow(g, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("a foto", fontsize=14)
    _sem_eixos(ax[0])

    ax[1].text(.5, .58, f"$T = {t}$", fontsize=46, ha="center", va="center",
               color=C_OTSU, transform=ax[1].transAxes)
    ax[1].text(.5, .26, "um número\npara a imagem inteira", fontsize=12.5,
               ha="center", va="center", color=CINZA, transform=ax[1].transAxes)
    ax[1].set_title("limiar GLOBAL", fontsize=14, color=C_OTSU,
                    fontweight="bold")
    ax[1].axis("off")

    im = ax[2].imshow(T, cmap="inferno")
    ax[2].set_title("limiar LOCAL  $T(x,y)$", fontsize=14, color=C_ADAPT,
                    fontweight="bold")
    ax[2].set_xlabel("uma imagem inteira de limiares", fontsize=12, color=CINZA)
    _sem_eixos(ax[2])
    fig.colorbar(im, ax=ax[2], fraction=0.046)

    fig.tight_layout()
    fig.savefig(SAIDA / "limiar_global_vs_local.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("  limiar_global_vs_local.png um número x uma imagem de limiares")


# ---------------------------------------------------------------------------
# 7. Os dois parametros da adaptativa, sobre o sudoku
# ---------------------------------------------------------------------------

def fig_parametros() -> None:
    g = img("10_sudoku")
    casos = [(11, SUDOKU_C, "bloco pequeno"), (SUDOKU_BLOCO, SUDOKU_C, "ajustado"),
             (SUDOKU_BLOCO, 0, "sem a constante C")]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(g, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("original", fontsize=14)
    _sem_eixos(ax[0])
    for a, (b, c, nota) in zip(ax[1:], casos):
        m, _ = L.adaptive_threshold(g, b, c, "gaussiana", inverso=True)
        a.imshow(m, cmap="gray", vmin=0, vmax=255)
        bom = nota == "ajustado"
        a.set_title(f"b = {b},  C = {c}", fontsize=14,
                    color=C_ADAPT if bom else C_MULTI, fontweight="bold")
        a.set_xlabel(nota, fontsize=11.5, color=CINZA)
        _sem_eixos(a)
    fig.tight_layout()
    fig.savefig(SAIDA / "parametros.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  parametros.png             efeito do bloco e do C")


# ---------------------------------------------------------------------------
# 8. O video real: quadros e limiar ao longo do tempo
# ---------------------------------------------------------------------------

def fig_video_real() -> None:
    cap = cv2.VideoCapture(str(VIDEO_REAL))
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    marcas = [0, 160, 320, 478]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.2))
    for a, k in zip(ax, marcas):
        g = quadro_real(k)
        a.imshow(g, cmap="gray", vmin=0, vmax=255)
        a.set_title(f"t = {k/fps:.0f} s", fontsize=14, fontweight="bold")
        a.set_xlabel(f"brilho médio {g.mean():.0f} · Otsu {L.otsu_threshold(g)}",
                     fontsize=11, color=CINZA)
        _sem_eixos(a)
    fig.tight_layout()
    fig.savefig(SAIDA / "video_real_strip.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  video_real_strip.png       4 quadros do pôr do sol")


def fig_limiar_canyon() -> None:
    cam = config.DIR_RES_METRICAS / "video_quadro_a_quadro_canyon.csv"
    if not cam.exists():
        print("  [pulado] limiar_canyon: rode o exp02 com --tag _canyon")
        return
    linhas = list(csv.DictReader(open(cam, encoding="utf-8-sig"), delimiter=";"))
    col = lambda n: np.array([float(r[n].replace(",", ".")) for r in linhas])
    t = col("tempo_s")

    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    for chave, cor, rot in [
            ("limiar_fixo", C_FIXO, "fixo  T = 127"),
            ("limiar_otsu", C_OTSU, "Otsu"),
            ("limiar_multiotsu_t1", "#E08A3C", "Multi-Otsu  $t_1$"),
            ("limiar_multiotsu_t2", C_MULTI, "Multi-Otsu  $t_2$")]:
        ax.plot(t, col(chave), color=cor, lw=2.6, label=rot)

    o = col("limiar_otsu")
    ax.annotate(f"{o[0]:.0f}", xy=(t[0], o[0]), xytext=(t[0] + .3, o[0] + 12),
                fontsize=13, color=C_OTSU, fontweight="bold")
    ax.annotate(f"{o[-1]:.0f}", xy=(t[-1], o[-1]), xytext=(t[-1] - 1.4, o[-1] - 20),
                fontsize=13, color=C_OTSU, fontweight="bold")

    ax.set_xlim(0, t.max()); ax.set_ylim(0, 200)
    ax.set_xlabel("tempo (s)", fontsize=14)
    ax.set_ylabel("limiar (nível de cinza)", fontsize=14)
    ax.legend(fontsize=12, ncol=4, frameon=False, loc="upper right")
    ax.grid(alpha=.25)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(SAIDA / "limiar_canyon.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  limiar_canyon.png          Otsu {o[0]:.0f} -> {o[-1]:.0f} no vídeo real")


# ---------------------------------------------------------------------------
# 9. Mapas de erro (cena sintetica: e a unica com gabarito)
# ---------------------------------------------------------------------------

def _mapa_erro(pred, gt):
    p, g = pred > 0, gt > 0
    rgb = np.zeros((*pred.shape, 3), np.uint8)
    rgb[~p & ~g] = (255, 255, 255)
    rgb[p & g] = (35, 35, 35)
    rgb[p & ~g] = (214, 45, 45)
    rgb[~p & g] = (45, 95, 214)
    return rgb


def fig_erros() -> None:
    casos = [(225, "Fase 2 · holofote"), (555, "Fase 4 · ruído")]
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.4))
    for i, (k, fase) in enumerate(casos):
        g, gt = quadro_sintetico(k)
        otsu = L.binarizar(g, L.otsu_threshold(g), inverso=True)
        t1, _ = L.multiotsu_threshold(g, 3)
        mo = L.binarizar(g, t1, inverso=True)
        ada, _ = L.adaptive_threshold(g, config.BLOCO_AJUSTADO,
                                      config.CONSTANTE_C, "gaussiana",
                                      inverso=True)
        ax[i, 0].imshow(g, cmap="gray", vmin=0, vmax=255)
        ax[i, 0].set_ylabel(fase, fontsize=14, fontweight="bold")
        ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])
        if i == 0:
            ax[i, 0].set_title("quadro", fontsize=14)
        for j, (m, nome, cor) in enumerate(
                [(otsu, "Otsu global", C_OTSU),
                 (mo, "Multi-Otsu $t_1$", C_MULTI),
                 (ada, "Adaptativa", C_ADAPT)], start=1):
            met = utils.metricas(m, gt)
            ax[i, j].imshow(_mapa_erro(m, gt)); ax[i, j].axis("off")
            ax[i, j].set_title((f"{nome}\n" if i == 0 else "") +
                               f"IoU = {met['iou']:.3f}", fontsize=13,
                               color=cor if i == 0 else INK,
                               fontweight="bold" if i == 0 else "normal")
    fig.suptitle("vermelho = falso positivo      azul = falso negativo",
                 fontsize=13.5, color=CINZA, y=.035)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(SAIDA / "erros_slide.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  erros_slide.png            mapas de erro na cena com gabarito")


def main() -> None:
    print("=" * 74)
    print("FIGURAS DOS SLIDES  (midia real + sintetica onde e necessario)")
    print("=" * 74)
    fig_sudoku()
    fig_modelo_formacao()
    fig_midia()
    fig_otsu_anatomia()
    fig_multiotsu_mri()
    fig_global_vs_local()
    fig_parametros()
    fig_video_real()
    fig_limiar_canyon()
    fig_erros()
    print("\nConcluido.\n")


if __name__ == "__main__":
    main()
