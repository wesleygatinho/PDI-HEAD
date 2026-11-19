"""Gera as figuras dos slides a partir dos resultados ja calculados e do video gravado.

    python apresentacao/gerar_figuras.py                     # usa o video mais recente
    python apresentacao/gerar_figuras.py --video caminho.mp4

Antes, grave o video de poses (de corpo inteiro, cabeca e pes no quadro):
    python espelho.py --gravar-video
Ele ja roda o acuracia.py no final, que gera o cache de landmarks usado aqui.

Figuras:
    tempo_por_quadro.png, estabilidade.png  copiados de resultados/graficos (python analise.py)
    modos.png         os modos do espelho, feitos a partir do video
    lite_vs_full.png  um quadro "em uma perna" em que o Lite erra e o Full acerta
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import cv2
import numpy as np

from acuracia import COR_MODELO, _esqueleto, _hex_bgr, comparar, extrair, pck_por_quadro, video_mais_recente
from analise import plt
from src.desenho import Compositor
from src.detector import DetectorPose
from src.suavizacao import FiltroOneEuro

FIG = Path(__file__).resolve().parent / "figuras"
VIDEO: Path = None      # definido em main()

GRAFICOS = {
    "tempo_por_quadro.png": RAIZ / "resultados" / "graficos" / "tempo_por_quadro.png",
    "estabilidade.png": RAIZ / "resultados" / "graficos" / "estabilidade.png",
}


def ler_fases() -> np.ndarray:
    arq = VIDEO.with_name(VIDEO.stem + "_fases.csv")
    if not arq.exists():
        raise SystemExit(f"Falta {arq.name}: use um video gravado com  python espelho.py --gravar-video")
    with open(arq, encoding="utf-8") as f:
        return np.array([linha["fase"] for linha in csv.DictReader(f)])


def ler_quadro(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, quadro = cap.read()
    if not ok:
        raise RuntimeError(f"quadro {i} nao lido")
    return quadro


def recorte_quadrado(img, centro_x):
    h, w = img.shape[:2]
    x0 = int(np.clip(centro_x - h / 2, 0, w - h))
    return img[:, x0:x0 + h]


def espelho_em_sequencia(cap, inicio, fim, compositor, modo):
    """Roda o pipeline do espelho (Full + One Euro + espelhamento) de `inicio` a `fim`.

    Devolve a tela do ultimo quadro no `modo` pedido e o x (px) do quadril nele.
    """
    detector, filtro = DetectorPose("full"), FiltroOneEuro()
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(inicio))
    tela, cx = None, 640
    for i in range(inicio, fim + 1):
        ok, quadro = cap.read()
        if not ok:
            break
        pontos, _ = detector.detectar(quadro, i * 1000 / 30)
        quadro = cv2.flip(quadro, 1)
        h, w = quadro.shape[:2]
        if pontos is None:
            filtro.reiniciar()
            continue
        pontos[:, 0] = 1 - pontos[:, 0]
        px = filtro(pontos[:, :2], i / 30) * (w, h)
        tela = compositor.compor(modo, quadro.copy(), px, pontos[:, 3])
        cx = float(px[[23, 24], 0].mean())
    detector.fechar()
    return tela, cx


def figura_modos(cap, fases):
    """Os modos do espelho lado a lado: sobreposto, avatar e rastro."""
    abertos = np.where(fases == "bracos_abertos")[0]
    alvo = int(abertos[len(abertos) // 2])
    sobre, cx1 = espelho_em_sequencia(cap, alvo - 45, alvo, Compositor(), 1)
    avatar, _ = espelho_em_sequencia(cap, alvo - 45, alvo, Compositor(), 2)
    livre = np.where(fases == "livre")[0]
    meio = int(livre[len(livre) // 2])
    rastro, cx3 = espelho_em_sequencia(cap, meio - 40, meio, Compositor(), 4)

    paineis = [("1 · sobreposto", recorte_quadrado(sobre, cx1)),
               ("2 · avatar", recorte_quadrado(avatar, cx1)),
               ("4 · rastro", recorte_quadrado(rastro, cx3))]
    fig, eixos = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, (titulo, img) in zip(eixos, paineis):
        ax.imshow(img[..., ::-1])
        ax.set_axis_off()
        ax.set_title(titulo, fontsize=13, loc="center")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "modos.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("  modos.png")


def figura_lite_vs_full(cap, fases):
    """Quadro em que o Lite erra e o Full acerta, com o Heavy (branco) por baixo."""
    pontos = {m: extrair(VIDEO, m, VIDEO.parent / "landmarks") for m in ("lite", "full", "heavy")}
    n = min(len(fases), *(len(v) for v in pontos.values()))
    tamanho = np.array([1280.0, 720.0])
    ref = pontos["heavy"][:n]
    pq = {m: pck_por_quadro(comparar(ref, pontos[m][:n], tamanho), 0.2) for m in ("lite", "full")}
    # Quadro "em uma perna" com o Full certo, boa parte do corpo visivel e o esqueleto do
    # Lite dentro da imagem (para o erro aparecer na figura), onde o Lite mais erra.
    lite = pontos["lite"][:n]
    dentro = ((lite[..., 0] > 0.02) & (lite[..., 0] < 0.98) & (lite[..., 1] > 0.02) & (lite[..., 1] < 0.98)).mean(1)
    n_vis = (np.nan_to_num(ref[..., 3]) >= 0.5).sum(1)
    candidatos = np.where((fases[:n] == "uma_perna") & (pq["full"] >= 0.95) & (n_vis >= 14) & (dentro >= 0.6))[0]
    if len(candidatos) == 0:      # sem quadro bom "em uma perna": tenta qualquer pose
        candidatos = np.where((pq["full"] >= 0.9) & (n_vis >= 14) & (dentro >= 0.6))[0]
    if len(candidatos) == 0:
        print("  lite_vs_full.png NAO gerada: nenhum quadro com o corpo bem visivel e o Full correto.")
        return
    i = int(candidatos[np.nanargmin(pq["lite"][candidatos])])

    quadro = ler_quadro(cap, i)
    vis_ref = np.nan_to_num(ref[i, :, 3]) >= 0.5
    ref_px = ref[i, :, :2] * tamanho
    centro = float(np.nanmean(ref_px[vis_ref, 0]))
    paineis = []
    for m in ("lite", "full"):
        img = quadro.copy()
        _esqueleto(img, ref_px, vis_ref, (30, 30, 30), 8)
        _esqueleto(img, ref_px, vis_ref, (255, 255, 255), 5)
        teste = pontos[m][i]
        if not np.isnan(teste[0, 0]):
            # mesmos pontos que entram na conta do PCK: os que o Heavy considera visiveis
            _esqueleto(img, teste[:, :2] * tamanho, vis_ref, _hex_bgr(COR_MODELO[m]), 4)
        paineis.append((m, recorte_quadrado(img, centro)))

    fig, eixos = plt.subplots(1, 2, figsize=(9, 4.9))
    for ax, (m, img) in zip(eixos, paineis):
        ax.imshow(img[..., ::-1])
        ax.set_axis_off()
        ax.set_title(f"{m.capitalize()}: {pq[m][i] * 100:.0f} % dos pontos corretos".replace(".", ","),
                     fontsize=13, loc="center")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "lite_vs_full.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  lite_vs_full.png (quadro {i}: Lite {pq['lite'][i]:.0%}, Full {pq['full'][i]:.0%})")


def main() -> None:
    global VIDEO
    p = argparse.ArgumentParser(description="Gera as figuras dos slides")
    p.add_argument("--video", type=Path, help="video de poses (padrao: o mais recente de resultados/videos)")
    args = p.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    for nome, origem in GRAFICOS.items():
        if origem.exists():
            shutil.copyfile(origem, FIG / nome)
            print(f"  {nome} (copiado)")
        else:
            print(f"  {nome} nao encontrado: rode antes  python analise.py")
    VIDEO = args.video or video_mais_recente()
    print(f"Video: {VIDEO}")
    fases = ler_fases()
    cap = cv2.VideoCapture(str(VIDEO))
    figura_modos(cap, fases)
    figura_lite_vs_full(cap, fases)
    cap.release()


if __name__ == "__main__":
    main()
