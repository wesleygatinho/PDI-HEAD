# -*- coding: utf-8 -*-
"""
exp02_video.py
==============
Experimento 2 - Limiarizacao aplicada a VIDEO (20 s, 600 quadros).

E aqui que as diferencas entre as tecnicas ficam evidentes, porque o video
introduz a dimensao TEMPORAL: o limiar precisa ser recalculado a cada quadro e
passa a ter uma dinamica propria.

O que e medido, quadro a quadro:
  - o(s) limiar(es) escolhido(s) por cada metodo;
  - IoU e F1 contra o gabarito analitico (a cena ideal que gerou o quadro);
  - fracao de pixels classificados como objeto;
  - INSTABILIDADE TEMPORAL: fracao de pixels que trocaram de classe em relacao
    ao quadro anterior (mede "cintilacao"/flicker);
  - tempo de processamento por quadro.

Saidas:
  resultados/video/     videos binarizados + mosaico comparativo comentado
  resultados/frames/    quadros de destaque, um por fase
  resultados/graficos/  series temporais
  resultados/metricas/  CSV quadro a quadro e resumo por fase

Uso:
    python src/exp02_video.py                      (usa o video sintetico)
    python src/exp02_video.py --video meu.mp4      (usa um video proprio)
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
import gerar_midia as gm  # noqa: E402
import limiarizacao as L  # noqa: E402
import utils  # noqa: E402

# Ordem em que os metodos aparecem em tabelas e graficos
METODOS = ["fixo", "otsu", "multiotsu_t1", "multiotsu_t2",
           "adaptativa", "adaptativa_ajustada", "sauvola"]

ROTULOS = {
    "fixo": "Limiar fixo T=127",
    "otsu": "Otsu global (por quadro)",
    "multiotsu_t1": "Multi-Otsu (so a classe escura, t1)",
    "multiotsu_t2": "Multi-Otsu (escura + media, t2)",
    "adaptativa": f"Adaptativa gaussiana (b={config.BLOCO}, C={config.CONSTANTE_C})",
    "adaptativa_ajustada":
        f"Adaptativa AJUSTADA (b={config.BLOCO_AJUSTADO}, C={config.CONSTANTE_C})",
    "sauvola": "Sauvola (local, media + desvio)",
}

CORES = {
    "fixo": "#7f7f7f",
    "otsu": "#1f77b4",
    "multiotsu_t1": "#ff7f0e",
    "multiotsu_t2": "#d62728",
    "adaptativa": "#2ca02c",
    "adaptativa_ajustada": "#17becf",
    "sauvola": "#9467bd",
}

FASES = [
    (0.0, 5.0, "1 - Iluminacao uniforme", "#e8f5e9"),
    (5.0, 10.0, "2 - Holofote movel", "#fff3e0"),
    (10.0, 15.0, "3 - Escurecimento global", "#e3f2fd"),
    (15.0, 20.0, "4 - Ruido + baixo contraste", "#fce4ec"),
]


# ---------------------------------------------------------------------------
# Segmentacao de um quadro por todos os metodos
# ---------------------------------------------------------------------------

def segmentar(frame: np.ndarray) -> tuple[dict, dict, dict]:
    """
    Aplica todos os metodos a um quadro em tons de cinza.

    Convencao: o OBJETO e escuro sobre fundo claro, entao a mascara de objeto e
    obtida com o teste  I(x, y) <= T  (parametro inverso=True).

    Retorna (mascaras, limiares, tempos_ms).
    """
    mascaras, limiares, tempos = {}, {}, {}

    # --- 1) Limiar fixo (referencia ingenua) -------------------------------
    t0 = time.perf_counter()
    mascaras["fixo"] = L.binarizar(frame, 127, inverso=True)
    tempos["fixo"] = (time.perf_counter() - t0) * 1000
    limiares["fixo"] = 127.0

    # --- 2) Otsu global, recalculado a cada quadro -------------------------
    t0 = time.perf_counter()
    t_otsu = L.otsu_threshold(frame)
    mascaras["otsu"] = L.binarizar(frame, t_otsu, inverso=True)
    tempos["otsu"] = (time.perf_counter() - t0) * 1000
    limiares["otsu"] = float(t_otsu)

    # --- 3) Multi-Otsu, 3 classes ------------------------------------------
    t0 = time.perf_counter()
    t1, t2 = L.multiotsu_threshold(frame, config.N_CLASSES)
    dt_mo = (time.perf_counter() - t0) * 1000
    mascaras["multiotsu_t1"] = L.binarizar(frame, t1, inverso=True)
    mascaras["multiotsu_t2"] = L.binarizar(frame, t2, inverso=True)
    tempos["multiotsu_t1"] = dt_mo
    tempos["multiotsu_t2"] = dt_mo
    limiares["multiotsu_t1"] = float(t1)
    limiares["multiotsu_t2"] = float(t2)

    # --- 4) Adaptativa gaussiana -------------------------------------------
    t0 = time.perf_counter()
    ada, T = L.adaptive_threshold(frame, config.BLOCO, config.CONSTANTE_C,
                                  "gaussiana", inverso=True)
    tempos["adaptativa"] = (time.perf_counter() - t0) * 1000
    mascaras["adaptativa"] = ada
    limiares["adaptativa"] = float(T.mean())      # limiar MEDIO do mapa T(x,y)

    # --- 5) Adaptativa com bloco ajustado ----------------------------------
    # A varredura do exp03 mostrou que b=35 e menor que os objetos da cena e
    # por isso "esvazia" o interior deles. Este bloco maior corrige o problema.
    t0 = time.perf_counter()
    ada2, T2 = L.adaptive_threshold(frame, config.BLOCO_AJUSTADO,
                                    config.CONSTANTE_C, "gaussiana",
                                    inverso=True)
    tempos["adaptativa_ajustada"] = (time.perf_counter() - t0) * 1000
    mascaras["adaptativa_ajustada"] = ada2
    limiares["adaptativa_ajustada"] = float(T2.mean())

    # --- 6) Sauvola ---------------------------------------------------------
    t0 = time.perf_counter()
    mascaras["sauvola"] = 255 - L.sauvola_threshold(frame, config.BLOCO, k=0.2)
    tempos["sauvola"] = (time.perf_counter() - t0) * 1000
    limiares["sauvola"] = float("nan")

    return mascaras, limiares, tempos


# ---------------------------------------------------------------------------
# Montagem do mosaico comentado
# ---------------------------------------------------------------------------

ALTURA_ROTULO = 36


def _rotular(img: np.ndarray, texto: str, sub: str = "") -> np.ndarray:
    """
    Acrescenta uma FAIXA acima da imagem com o rotulo, em vez de escrever sobre
    ela - assim nenhum pixel do resultado fica escondido pelo texto.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    faixa = np.zeros((ALTURA_ROTULO, w, 3), np.uint8)
    cv2.putText(faixa, texto, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(faixa, sub, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (120, 220, 255), 1, cv2.LINE_AA)
    saida = np.vstack([faixa, img])
    cv2.rectangle(saida, (0, 0), (w - 1, saida.shape[0] - 1), (70, 70, 70), 1)
    return saida


def montar_mosaico(frame, gt, mascaras, limiares, ious, k, fase,
                   larg=400, alt=300) -> np.ndarray:
    """Grade 2x4 com o original, o gabarito e os cinco metodos principais."""
    def prep(img, titulo, sub=""):
        return _rotular(cv2.resize(img, (larg, alt),
                                   interpolation=cv2.INTER_AREA), titulo, sub)

    t_s = k / config.VIDEO_FPS
    paineis = [
        prep(frame, f"ORIGINAL  t={t_s:5.2f}s  quadro {k}", f"fase {fase}"),
        prep(gt, "GABARITO (cena ideal)", "referencia para IoU"),
        prep(mascaras["fixo"], "1. LIMIAR FIXO  T=127",
             f"IoU={ious['fixo']:.3f}"),
        prep(mascaras["otsu"], f"2. OTSU GLOBAL  T={limiares['otsu']:.0f}",
             f"IoU={ious['otsu']:.3f}"),
        prep(mascaras["multiotsu_t1"],
             f"3. MULTI-OTSU  so classe escura  t1={limiares['multiotsu_t1']:.0f}",
             f"IoU={ious['multiotsu_t1']:.3f}"),
        prep(mascaras["multiotsu_t2"],
             f"4. MULTI-OTSU  escura+media  t2={limiares['multiotsu_t2']:.0f}",
             f"IoU={ious['multiotsu_t2']:.3f}"),
        prep(mascaras["adaptativa"],
             f"5. ADAPTATIVA  b={config.BLOCO} C={config.CONSTANTE_C}",
             f"IoU={ious['adaptativa']:.3f}"),
        prep(mascaras["adaptativa_ajustada"],
             f"6. ADAPTATIVA AJUSTADA  b={config.BLOCO_AJUSTADO}",
             f"IoU={ious['adaptativa_ajustada']:.3f}"),
    ]
    linha1 = np.hstack(paineis[:4])
    linha2 = np.hstack(paineis[4:])
    return np.vstack([linha1, linha2])


# ---------------------------------------------------------------------------
# Laco principal sobre o video
# ---------------------------------------------------------------------------

def processar(caminho_video: str | None) -> dict:
    config.garantir_pastas()

    usa_sintetico = caminho_video is None
    cam = (config.DIR_VIDEOS / config.NOME_VIDEO) if usa_sintetico else caminho_video

    cap = cv2.VideoCapture(str(cam))
    if not cap.isOpened():
        raise FileNotFoundError(f"Nao foi possivel abrir o video: {cam}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS
    larg = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  video: {cam}")
    print(f"  {larg}x{alt}, {n_total} quadros, {fps:.1f} fps "
          f"({n_total / fps:.1f} s)")
    if not usa_sintetico:
        print("  [aviso] video externo: sem gabarito, as metricas de IoU/F1 "
              "nao serao calculadas.")

    # --- Escritores de video ------------------------------------------------
    esc_mosaico = utils.abrir_escritor(
        config.DIR_RES_VIDEO / "mosaico_comparativo.mp4", fps,
        (400 * 4, (300 + ALTURA_ROTULO) * 2), colorido=True)
    escritores = {
        m: utils.abrir_escritor(config.DIR_RES_VIDEO / f"saida_{m}.mp4",
                                fps, (larg, alt))
        for m in ("otsu", "multiotsu_t2", "adaptativa", "adaptativa_ajustada")
    }

    registro = []
    anteriores: dict[str, np.ndarray | None] = {m: None for m in METODOS}
    k = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Gabarito ANALITICO: a cena ideal que originou este quadro, livre de
        # iluminacao, ruido e artefatos de compressao do codec.
        if usa_sintetico:
            limpa = gm.cena_limpa(k / n_total, larg, alt)
            gt = gm.gabarito_binario(limpa)
            nome_fase = gm.parametros_da_fase(k, n_total)[0]
        else:
            gt, nome_fase = None, "-"

        mascaras, limiares, tempos = segmentar(frame)

        ious, f1s = {}, {}
        for m in METODOS:
            if gt is not None:
                mm = utils.metricas(mascaras[m], gt)
                ious[m], f1s[m] = mm["iou"], mm["f1_dice"]
            else:
                ious[m] = f1s[m] = float("nan")

        linha = {"quadro": k, "tempo_s": k / fps, "fase": nome_fase,
                 "fg_ideal": (utils.fracao_primeiro_plano(gt)
                              if gt is not None else float("nan"))}
        for m in METODOS:
            linha[f"limiar_{m}"] = limiares[m]
            linha[f"iou_{m}"] = ious[m]
            linha[f"f1_{m}"] = f1s[m]
            linha[f"fg_{m}"] = utils.fracao_primeiro_plano(mascaras[m])
            linha[f"inst_{m}"] = utils.instabilidade_temporal(mascaras[m],
                                                              anteriores[m])
            linha[f"ms_{m}"] = tempos[m]
            anteriores[m] = mascaras[m]
        registro.append(linha)

        # --- Saidas em video ------------------------------------------------
        gt_vis = gt if gt is not None else np.zeros_like(frame)
        esc_mosaico.write(montar_mosaico(frame, gt_vis, mascaras, limiares,
                                         ious, k, nome_fase))
        for m, esc in escritores.items():
            esc.write(mascaras[m])

        # --- Quadros de destaque para o relatorio ---------------------------
        if k in config.FRAMES_DESTAQUE:
            paineis = [("Original", frame)]
            if gt is not None:
                paineis.append(("Gabarito", gt))
            paineis += [(ROTULOS[m] + f"\nIoU = {ious[m]:.3f}", mascaras[m])
                        for m in METODOS]
            utils.painel(paineis,
                         f"Quadro {k} (t = {k / fps:.2f} s) - fase {nome_fase}",
                         config.DIR_RES_FRAMES / f"quadro_{k:04d}.png",
                         colunas=4)

        k += 1
        if k % 100 == 0:
            print(f"    {k}/{n_total} quadros")

    cap.release()
    esc_mosaico.release()
    for esc in escritores.values():
        esc.release()

    print(f"  processados {k} quadros")
    return {"registro": registro, "fps": fps, "sintetico": usa_sintetico}


# ---------------------------------------------------------------------------
# Graficos das series temporais
# ---------------------------------------------------------------------------

def _faixas_de_fase(ax, t_max: float) -> None:
    for ini, fim, nome, cor in FASES:
        if ini >= t_max:
            break
        ax.axvspan(ini, min(fim, t_max), color=cor, alpha=0.75, zorder=0)
        ax.text((ini + min(fim, t_max)) / 2, ax.get_ylim()[1] * 0.97, nome,
                ha="center", va="top", fontsize=7.5, color="#444444", zorder=5)


def graficos(res: dict) -> None:
    reg = res["registro"]
    t = np.array([r["tempo_s"] for r in reg])
    t_max = float(t.max())

    def serie(chave, m):
        return np.array([r[f"{chave}_{m}"] for r in reg], dtype=float)

    # ---- 1) Limiar escolhido ao longo do tempo ----------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in ["fixo", "otsu", "multiotsu_t1", "multiotsu_t2",
              "adaptativa", "adaptativa_ajustada"]:
        ax.plot(t, serie("limiar", m), color=CORES[m], lw=1.4, label=ROTULOS[m])
    ax.set_ylim(0, 260)
    _faixas_de_fase(ax, t_max)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("limiar (nivel de cinza)")
    ax.set_title("Como cada metodo move o limiar ao longo do video\n"
                 "(para a adaptativa e mostrada a MEDIA do mapa T(x,y))",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    utils.salvar_figura(fig, config.DIR_RES_GRAFICOS / "V1_limiar_no_tempo.png")

    if not res["sintetico"]:
        return

    # ---- 2) IoU ao longo do tempo -----------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in METODOS:
        ax.plot(t, serie("iou", m), color=CORES[m], lw=1.3, label=ROTULOS[m])
    ax.set_ylim(0, 1.02)
    _faixas_de_fase(ax, t_max)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("IoU contra o gabarito")
    ax.set_title("Qualidade da segmentacao quadro a quadro (IoU)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    utils.salvar_figura(fig, config.DIR_RES_GRAFICOS / "V2_iou_no_tempo.png")

    # ---- 3) Instabilidade temporal ----------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in METODOS:
        ax.plot(t, 100 * serie("inst", m), color=CORES[m], lw=1.2,
                label=ROTULOS[m])
    _faixas_de_fase(ax, t_max)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("pixels que trocaram de classe (%)")
    ax.set_title("Instabilidade temporal (cintilacao) entre quadros consecutivos",
                 fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    utils.salvar_figura(fig,
                        config.DIR_RES_GRAFICOS / "V3_instabilidade_no_tempo.png")

    # ---- 4) Fracao de primeiro plano --------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in METODOS:
        ax.plot(t, 100 * serie("fg", m), color=CORES[m], lw=1.2, label=ROTULOS[m])
    gt_fg = 100 * np.array([r["fg_ideal"] for r in reg]) if "fg_ideal" in reg[0] \
        else None
    if gt_fg is not None:
        ax.plot(t, gt_fg, "k--", lw=1.6, label="Gabarito")
    _faixas_de_fase(ax, t_max)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("area classificada como objeto (%)")
    ax.set_title("Fracao de primeiro plano: quanto cada metodo 'engorda' ou "
                 "'perde' o objeto", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    utils.salvar_figura(fig,
                        config.DIR_RES_GRAFICOS / "V4_area_no_tempo.png")

    # ---- 5) Resumo por fase (barras agrupadas) ----------------------------
    fases = [f[2] for f in FASES]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    larg = 0.115
    x = np.arange(len(fases))
    for i, m in enumerate(METODOS):
        medias_iou, medias_inst = [], []
        for j, (ini, fim, _, _) in enumerate(FASES):
            sel = (t >= ini) & (t < fim)
            medias_iou.append(_media(serie("iou", m)[sel]))
            medias_inst.append(100 * _media(serie("inst", m)[sel]))
        ax[0].bar(x + i * larg, medias_iou, larg, color=CORES[m],
                  label=ROTULOS[m])
        ax[1].bar(x + i * larg, medias_inst, larg, color=CORES[m])

    for a, titulo, ylab in ((ax[0], "IoU medio por fase", "IoU"),
                            (ax[1], "Instabilidade media por fase", "%")):
        a.set_xticks(x + larg * 3)
        a.set_xticklabels([f.split(" - ")[0] + "\n" + f.split(" - ")[1]
                           for f in fases], fontsize=8)
        a.set_title(titulo, fontweight="bold")
        a.set_ylabel(ylab)
        a.grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    utils.salvar_figura(fig, config.DIR_RES_GRAFICOS / "V5_resumo_por_fase.png")

    # ---- 6) Custo computacional -------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    medias = [_media(serie("ms", m)) for m in METODOS]
    ax.barh([ROTULOS[m] for m in METODOS], medias,
            color=[CORES[m] for m in METODOS])
    for i, v in enumerate(medias):
        ax.text(v, i, f"  {v:.2f} ms  ({1000 / v:.0f} fps)", va="center",
                fontsize=8)
    ax.set_xlabel("tempo medio por quadro (ms)")
    ax.set_title("Custo por quadro (640x480)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    utils.salvar_figura(fig, config.DIR_RES_GRAFICOS / "V6_custo_por_quadro.png")


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

def _media(v: np.ndarray) -> float:
    """Media ignorando NaN; devolve NaN se a serie for inteiramente NaN."""
    v = np.asarray(v, dtype=float)
    return float(np.nanmean(v)) if np.any(np.isfinite(v)) else float("nan")


def _desvio(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.nanstd(v)) if np.any(np.isfinite(v)) else float("nan")


def tabelas(res: dict) -> None:
    reg = res["registro"]

    # CSV completo, quadro a quadro
    chaves = list(reg[0].keys())
    utils.escrever_csv(config.DIR_RES_METRICAS / "video_quadro_a_quadro.csv",
                       chaves, [[r[c] for c in chaves] for r in reg])

    if not res["sintetico"]:
        return

    t = np.array([r["tempo_s"] for r in reg])

    def serie(chave, m):
        return np.array([r[f"{chave}_{m}"] for r in reg], dtype=float)

    # Resumo global e por fase
    resumo_global, resumo_fases = [], []
    for m in METODOS:
        iou, inst, ms = serie("iou", m), serie("inst", m), serie("ms", m)
        lim = serie("limiar", m)
        resumo_global.append([
            ROTULOS[m], _media(iou), _desvio(iou),
            float(np.nanmin(iou)), _media(serie("f1", m)),
            100 * _media(inst), _media(ms),
            _desvio(lim),
        ])
        for ini, fim, nome, _ in FASES:
            sel = (t >= ini) & (t < fim)
            resumo_fases.append([nome, ROTULOS[m],
                                 _media(iou[sel]), _media(serie("f1", m)[sel]),
                                 100 * _media(inst[sel]), _media(lim[sel])])

    utils.escrever_csv(config.DIR_RES_METRICAS / "video_resumo_global.csv",
                       ["metodo", "iou_medio", "iou_desvio", "iou_minimo",
                        "f1_medio", "instabilidade_%", "tempo_ms",
                        "desvio_do_limiar"], resumo_global)
    utils.escrever_csv(config.DIR_RES_METRICAS / "video_resumo_por_fase.csv",
                       ["fase", "metodo", "iou_medio", "f1_medio",
                        "instabilidade_%", "limiar_medio"], resumo_fases)

    # Relatorio Markdown
    cam = config.DIR_RES_METRICAS / "RESUMO_video.md"
    with open(cam, "w", encoding="utf-8") as f:
        f.write("# Experimento 2 - Resultados sobre o video de 20 s\n\n")
        f.write(f"Video sintetico: {config.VIDEO_LARGURA}x{config.VIDEO_ALTURA}, "
                f"{config.VIDEO_FPS} fps, {config.VIDEO_DURACAO_S} s "
                f"({config.VIDEO_N_FRAMES} quadros), com gabarito analitico.\n\n")
        f.write("## 1. Desempenho global (media sobre os 600 quadros)\n\n")
        utils.escrever_markdown_tabela(
            f, ["Metodo", "IoU medio", "Desvio do IoU", "IoU minimo",
                "F1 medio", "Instabilidade (%)", "Tempo (ms)",
                "Desvio do limiar"], resumo_global)
        f.write("## 2. Desempenho por fase\n\n")
        f.write("Cada fase dura 5 s e estressa um aspecto diferente.\n\n")
        por_fase = defaultdict(list)
        for linha in resumo_fases:
            por_fase[linha[0]].append(linha[1:])
        for nome_fase, linhas in por_fase.items():
            f.write(f"### Fase {nome_fase}\n\n")
            utils.escrever_markdown_tabela(
                f, ["Metodo", "IoU medio", "F1 medio", "Instabilidade (%)",
                    "Limiar medio"], linhas)
    print(f"    -> {cam}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Limiarizacao em video")
    ap.add_argument("--video", default=None,
                    help="caminho de um video proprio (opcional). "
                         "Sem este argumento usa o video sintetico de 20 s.")
    args = ap.parse_args()

    print("=" * 74)
    print("EXPERIMENTO 2 - VIDEO")
    print("=" * 74)
    res = processar(args.video)
    print("\n[graficos]")
    graficos(res)
    print("[tabelas]")
    tabelas(res)
    print("\nConcluido.\n")


if __name__ == "__main__":
    main()
