# -*- coding: utf-8 -*-
"""
exp03_analise.py
================
Experimento 3 - Analise comparativa: POR QUE cada metodo falha.

Enquanto os experimentos 1 e 2 medem o desempenho, este investiga as causas:

  1) Mapas de erro (FP/FN) - onde exatamente cada metodo erra;
  2) Evolucao do histograma ao longo das 4 fases do video - explica a dinamica
     do limiar de Otsu observada no grafico V1;
  3) Efeito do tamanho do bloco na adaptativa - explica o "esvaziamento" dos
     objetos grandes;
  4) Efeito da constante C;
  5) Painel-sintese com o guia de escolha do metodo.

Uso:
    python src/exp03_analise.py
"""

from __future__ import annotations

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
import gerar_midia as gm  # noqa: E402
import limiarizacao as L  # noqa: E402
import utils  # noqa: E402


# ---------------------------------------------------------------------------
# Leitura de quadros de destaque do video
# ---------------------------------------------------------------------------

def quadros_de_destaque() -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    """Retorna [(indice, nome_da_fase, quadro_observado, gabarito), ...]."""
    cap = cv2.VideoCapture(str(config.DIR_VIDEOS / config.NOME_VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError("Rode antes: python src/gerar_midia.py")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    larg = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    saida = []
    for k in config.FRAMES_DESTAQUE:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, bgr = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gt = gm.gabarito_binario(gm.cena_limpa(k / n, larg, alt))
        saida.append((k, gm.parametros_da_fase(k, n)[0], frame, gt))
    cap.release()
    return saida


def mascaras_de(frame: np.ndarray) -> dict[str, np.ndarray]:
    t_otsu = L.otsu_threshold(frame)
    t1, t2 = L.multiotsu_threshold(frame, 3)
    ada, _ = L.adaptive_threshold(frame, config.BLOCO, config.CONSTANTE_C,
                                  "gaussiana", inverso=True)
    ada2, _ = L.adaptive_threshold(frame, config.BLOCO_AJUSTADO,
                                   config.CONSTANTE_C, "gaussiana", inverso=True)
    return {
        "Otsu global": L.binarizar(frame, t_otsu, inverso=True),
        "Multi-Otsu (t2)": L.binarizar(frame, t2, inverso=True),
        "Adaptativa": ada,
        "Adaptativa ajustada": ada2,
        "Sauvola": 255 - L.sauvola_threshold(frame, config.BLOCO, k=0.2),
    }


# ---------------------------------------------------------------------------
# 1) Mapas de erro
# ---------------------------------------------------------------------------

def mapa_de_erro(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Imagem RGB que separa os dois tipos de erro:
      branco   = acerto de fundo (VN)
      preto    = acerto de objeto (VP)
      VERMELHO = falso positivo (fundo classificado como objeto)
      AZUL     = falso negativo (objeto perdido)
    """
    p, g = pred > 0, gt > 0
    rgb = np.zeros((*pred.shape, 3), np.uint8)
    rgb[~p & ~g] = (255, 255, 255)      # VN
    rgb[p & g] = (30, 30, 30)           # VP
    rgb[p & ~g] = (220, 40, 40)         # FP
    rgb[~p & g] = (40, 90, 220)         # FN
    return rgb


def fig_mapas_de_erro(destaques) -> None:
    metodos = ["Otsu global", "Multi-Otsu (t2)", "Adaptativa", "Sauvola"]
    fig, ax = plt.subplots(len(destaques), len(metodos) + 1,
                           figsize=(4.0 * (len(metodos) + 1),
                                    2.9 * len(destaques)))
    for i, (k, fase, frame, gt) in enumerate(destaques):
        ax[i, 0].imshow(frame, cmap="gray", vmin=0, vmax=255)
        ax[i, 0].set_ylabel(f"fase {fase}\nquadro {k}", fontsize=8)
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        if i == 0:
            ax[i, 0].set_title("Quadro observado", fontsize=10)

        masc = mascaras_de(frame)
        for j, m in enumerate(metodos, start=1):
            met = utils.metricas(masc[m], gt)
            ax[i, j].imshow(mapa_de_erro(masc[m], gt))
            ax[i, j].axis("off")
            ax[i, j].set_title(
                (f"{m}\n" if i == 0 else "") +
                f"IoU={met['iou']:.3f}  FP={met['FP'] / gt.size * 100:.1f}%  "
                f"FN={met['FN'] / gt.size * 100:.1f}%", fontsize=8)

    fig.suptitle("Mapas de erro - vermelho = falso positivo, "
                 "azul = falso negativo", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    utils.salvar_figura(fig, config.DIR_RES_GRAFICOS / "X1_mapas_de_erro.png")


# ---------------------------------------------------------------------------
# 2) Evolucao do histograma ao longo das fases
# ---------------------------------------------------------------------------

def fig_histogramas_das_fases(destaques) -> None:
    fig, ax = plt.subplots(2, len(destaques),
                           figsize=(4.0 * len(destaques), 7.0))
    for i, (k, fase, frame, _) in enumerate(destaques):
        ax[0, i].imshow(frame, cmap="gray", vmin=0, vmax=255)
        ax[0, i].set_title(f"fase {fase}\nquadro {k} "
                           f"(t = {k / config.VIDEO_FPS:.1f} s)", fontsize=10)
        ax[0, i].axis("off")

        r = L.otsu_variancias(frame)
        t1, t2 = L.multiotsu_threshold(frame, 3)
        ax[1, i].bar(np.arange(256), r["p"], width=1.0, color="#888888")
        ax[1, i].axvline(r["limiar"], color="#1f77b4", lw=1.8,
                         label=f"Otsu t*={r['limiar']}")
        ax[1, i].axvline(t1, color="#ff7f0e", lw=1.4, ls="--",
                         label=f"Multi t1={t1}")
        ax[1, i].axvline(t2, color="#d62728", lw=1.4, ls="--",
                         label=f"Multi t2={t2}")
        ax[1, i].set_xlim(0, 255)
        ax[1, i].set_xlabel("nivel de cinza")
        ax[1, i].legend(fontsize=7)
        ax[1, i].set_title(rf"$\eta$ = {r['eta'][r['limiar']]:.3f}", fontsize=9)

    fig.suptitle("Evolucao do histograma nas 4 fases do video\n"
                 "(a bimodalidade se perde -> o limiar global perde sentido)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    utils.salvar_figura(fig,
                        config.DIR_RES_GRAFICOS / "X2_histogramas_das_fases.png")


# ---------------------------------------------------------------------------
# 3) e 4) Efeito do bloco e da constante C
# ---------------------------------------------------------------------------

def fig_efeito_bloco(destaques) -> None:
    blocos = [7, 11, 15, 21, 31, 45, 61, 81, 101, 131, 161, 201]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    for (k, fase, frame, gt) in destaques:
        ious = []
        for b in blocos:
            m, _ = L.adaptive_threshold(frame, b, config.CONSTANTE_C,
                                        "gaussiana", inverso=True)
            ious.append(utils.metricas(m, gt)["iou"])
        ax[0].plot(blocos, ious, "o-", lw=1.6, ms=4, label=f"fase {fase}")

    ax[0].axvline(config.BLOCO, color="k", ls=":", lw=1.5,
                  label=f"bloco usado = {config.BLOCO}")
    ax[0].set_xlabel("tamanho do bloco (pixels)")
    ax[0].set_ylabel("IoU")
    ax[0].set_title("Efeito do tamanho do bloco\n"
                    "(bloco pequeno esvazia objetos grandes;\n"
                    "bloco grande reintroduz o problema do limiar global)",
                    fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    cs = [0, 2, 4, 6, 8, 10, 14, 18, 24, 30, 40]
    for (k, fase, frame, gt) in destaques:
        ious = []
        for c in cs:
            m, _ = L.adaptive_threshold(frame, config.BLOCO, c,
                                        "gaussiana", inverso=True)
            ious.append(utils.metricas(m, gt)["iou"])
        ax[1].plot(cs, ious, "s-", lw=1.6, ms=4, label=f"fase {fase}")

    ax[1].axvline(config.CONSTANTE_C, color="k", ls=":", lw=1.5,
                  label=f"C usado = {config.CONSTANTE_C}")
    ax[1].set_xlabel("constante C")
    ax[1].set_ylabel("IoU")
    ax[1].set_title("Efeito da constante C\n"
                    "(C = 0 gera ruido no fundo liso;\n"
                    "C grande apaga estruturas de baixo contraste)",
                    fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.suptitle("Por que a adaptativa depende tanto dos seus dois parametros",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    utils.salvar_figura(fig,
                        config.DIR_RES_GRAFICOS / "X3_efeito_bloco_e_C.png")

    # Tabela com o melhor bloco por fase
    linhas = []
    for (k, fase, frame, gt) in destaques:
        melhor_b, melhor_iou = None, -1.0
        for b in blocos:
            m, _ = L.adaptive_threshold(frame, b, config.CONSTANTE_C,
                                        "gaussiana", inverso=True)
            v = utils.metricas(m, gt)["iou"]
            if v > melhor_iou:
                melhor_b, melhor_iou = b, v
        padrao, _ = L.adaptive_threshold(frame, config.BLOCO,
                                         config.CONSTANTE_C, "gaussiana",
                                         inverso=True)
        linhas.append([fase, k, config.BLOCO,
                       utils.metricas(padrao, gt)["iou"], melhor_b, melhor_iou])
    utils.escrever_csv(config.DIR_RES_METRICAS / "adaptativa_melhor_bloco.csv",
                       ["fase", "quadro", "bloco_padrao", "iou_padrao",
                        "melhor_bloco", "melhor_iou"], linhas)
    return linhas


# ---------------------------------------------------------------------------
# 5) Painel-sintese
# ---------------------------------------------------------------------------

def fig_sintese(destaques) -> None:
    """Uma figura unica, pronta para o slide de conclusao."""
    k, fase, frame, gt = destaques[1]        # quadro da fase do holofote
    masc = mascaras_de(frame)

    fig = plt.figure(figsize=(16.5, 8.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.3, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Quadro com iluminacao\nespacialmente nao uniforme", fontsize=9.5)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(gt, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Gabarito\n(resposta correta)", fontsize=9.5)
    ax.axis("off")

    paineis = [
        ("Otsu global", f"Otsu GLOBAL\n(colapsa: um unico T)"),
        ("Adaptativa", f"Adaptativa, bloco={config.BLOCO}\n(bloco < objeto: esvazia)"),
        ("Adaptativa ajustada",
         f"Adaptativa, bloco={config.BLOCO_AJUSTADO}\n(bloco ajustado a escala)"),
    ]
    for j, (chave, rotulo) in enumerate(paineis, start=2):
        ax = fig.add_subplot(gs[0, j])
        met = utils.metricas(masc[chave], gt)
        ax.imshow(masc[chave], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{rotulo}\nIoU = {met['iou']:.3f}", fontsize=9.5)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    texto = (
        "QUANDO USAR CADA TECNICA\n\n"
        "  OTSU (global)       .  histograma bimodal e iluminacao espacialmente uniforme.\n"
        "                         E invariante a mudancas GLOBAIS de brilho (reajusta o T a cada quadro),\n"
        "                         mas colapsa quando a iluminacao varia DENTRO da cena.\n"
        "                         Custo baixissimo: 1 limiar para a imagem toda.\n\n"
        "  MULTI-OTSU          .  a cena tem 3 ou mais populacoes de intensidade (fundo, sombra, objeto).\n"
        "                         Mesma fragilidade espacial do Otsu, mas permite isolar so a classe mais\n"
        "                         escura, que costuma sobreviver melhor a variacoes de iluminacao.\n"
        "                         O numero de classes M e uma escolha do projetista.\n\n"
        "  ADAPTATIVA (local)  .  iluminacao desigual, sombras, documentos digitalizados, fotos de quadro.\n"
        "                         Unico metodo que segue o gradiente de luz. Em compensacao: exige que o\n"
        "                         bloco seja MAIOR que o objeto (senao esvazia o interior), amplifica ruido\n"
        "                         em regioes homogeneas e depende de dois parametros (bloco e C).\n\n"
        "  REGRA DO BLOCO      .  maior objeto  <  bloco  <  escala de variacao da iluminacao.\n"
        "                         Nesta cena:  130 px  <  bloco  <  300 px   ->   bloco = 151."
    )
    ax.text(0.01, 0.98, texto, va="top", ha="left", family="monospace",
            fontsize=9.5, linespacing=1.45)

    fig.suptitle("Sintese: a escolha do metodo depende da natureza da iluminacao",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    utils.salvar_figura(fig, config.DIR_RES_GRAFICOS / "X4_sintese.png")


# ---------------------------------------------------------------------------

def main() -> None:
    config.garantir_pastas()
    print("=" * 74)
    print("EXPERIMENTO 3 - ANALISE COMPARATIVA")
    print("=" * 74)

    destaques = quadros_de_destaque()
    print(f"  quadros analisados: {[d[0] for d in destaques]}")

    print("  [1] mapas de erro FP/FN")
    fig_mapas_de_erro(destaques)
    print("  [2] evolucao dos histogramas")
    fig_histogramas_das_fases(destaques)
    print("  [3/4] varredura de bloco e C")
    linhas = fig_efeito_bloco(destaques)
    for f, k, bp, ip, mb, mi in linhas:
        print(f"       fase {f:18s} bloco padrao {bp:3d} -> IoU {ip:.3f} | "
              f"melhor bloco {mb:3d} -> IoU {mi:.3f}")
    print("  [5] painel-sintese")
    fig_sintese(destaques)

    print("\nConcluido. Figuras X1..X4 em resultados/graficos.\n")


if __name__ == "__main__":
    main()
