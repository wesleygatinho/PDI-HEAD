"""Taxa de acerto do espelho: compara os landmarks do Lite e do Full com os do Heavy.

Nao ha gabarito anotado a mao, entao o modelo Heavy (o maior e mais preciso
dos tres) serve de REFERENCIA: o mesmo video gravado e processado pelos tres
modelos e mede-se o quanto os pontos do Lite e do Full concordam com os dele.

Metrica principal - PCK (Percentage of Correct Keypoints):
    um landmark esta "correto" se cair a menos de alfa x (comprimento do tronco)
    do landmark de referencia. Comprimento do tronco = distancia entre o ponto
    medio dos ombros e o ponto medio dos quadris, no Heavy, em cada quadro.
    PCK@0,2 (alfa = 0,2) e a "taxa de acerto"; PCK@0,1 e a versao rigorosa.
    So entram landmarks que o Heavy considera visiveis (visibilidade >= 0,5),
    em quadros em que ombros e quadris estao visiveis.

    python acuracia.py                         # usa o video mais recente de resultados/videos
    python acuracia.py --video caminho.mp4

Saidas em resultados/acuracia/<nome do video>/ (tabela e graficos).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from analise import (AZUL, EIXO, LARANJA, MUDO, SUPERFICIE, TINTA, TINTA2, br, plt, salvar)
from src.desenho import CONEXOES
from src.detector import DetectorPose
from src.metricas import LIMIAR_VISIBILIDADE
from src.roteiro import NOME_POSE, POSES

RAIZ = Path(__file__).resolve().parent
REFERENCIA = "heavy"
TESTADOS = ("lite", "full")
COR_MODELO = {"lite": AZUL, "full": LARANJA}
LIMIARES = (0.1, 0.2)
PRINCIPAL = 0.2
QUADRO_CORRETO = 0.9      # quadro "correto" = pelo menos 90 % dos landmarks corretos a PCK@0,2
TRONCO = (11, 12, 23, 24)
GRUPOS = {
    "Rosto": tuple(range(0, 11)),
    "Tronco": (11, 12, 23, 24),
    "Braços": (13, 14, 15, 16),
    "Mãos": tuple(range(17, 23)),
    "Pernas": (25, 26, 27, 28),
    "Pés": tuple(range(29, 33)),
}


# ---------------------------------------------------------------- extracao

def extrair(video: Path, modelo: str, pasta_cache: Path) -> np.ndarray:
    """Landmarks de todos os quadros (N x 33 x 4, coordenadas normalizadas; NaN = sem pose).

    Guarda o resultado em cache para nao reprocessar o video a cada execucao.
    """
    cache = pasta_cache / f"{video.stem}_{modelo}.npy"
    if cache.exists() and cache.stat().st_mtime >= video.stat().st_mtime:
        print(f"  {modelo:5}: usando cache ({cache.name})")
        return np.load(cache)

    detector = DetectorPose(modelo)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saida, t0 = [], perf_counter()
    while True:
        ok, quadro = cap.read()
        if not ok:
            break
        pontos, _ = detector.detectar(quadro, len(saida) * 1000.0 / fps)
        saida.append(pontos if pontos is not None else np.full((33, 4), np.nan, np.float32))
        if len(saida) % 300 == 0:
            print(f"  {modelo:5}: {len(saida)}/{total} quadros")
    cap.release()
    detector.fechar()
    pontos = np.stack(saida)
    pasta_cache.mkdir(parents=True, exist_ok=True)
    np.save(cache, pontos)
    print(f"  {modelo:5}: {len(pontos)} quadros em {perf_counter() - t0:.0f} s")
    return pontos


def ler_fases(video: Path, n: int) -> np.ndarray:
    arq = video.with_name(video.stem + "_fases.csv")
    fases = np.full(n, "", dtype=object)
    if arq.exists():
        with open(arq, encoding="utf-8", newline="") as f:
            for linha in csv.DictReader(f):
                i = int(linha["quadro"])
                if i < n:
                    fases[i] = linha["fase"]
    return fases


# ---------------------------------------------------------------- metricas

def comparar(ref: np.ndarray, teste: np.ndarray, tamanho: np.ndarray) -> dict:
    """Erros de `teste` em relacao a `ref`, por quadro e por landmark."""
    ref_px, teste_px = ref[..., :2] * tamanho, teste[..., :2] * tamanho
    ref_det = ~np.isnan(ref[:, 0, 0])
    ref_vis = np.nan_to_num(ref[..., 3]) >= LIMIAR_VISIBILIDADE

    ombros = ref_px[:, [11, 12]].mean(axis=1)
    quadris = ref_px[:, [23, 24]].mean(axis=1)
    tronco = np.linalg.norm(ombros - quadris, axis=1)
    quadro_valido = ref_det & ref_vis[:, TRONCO].all(axis=1) & (tronco > 1)
    avaliar = ref_vis & quadro_valido[:, None]          # (N, 33): pontos que entram na conta

    teste_det = ~np.isnan(teste[:, 0, 0])
    erro_px = np.linalg.norm(teste_px - ref_px, axis=2)
    erro_px[~teste_det] = np.inf                        # sem pose = erro infinito (nunca correto)
    with np.errstate(invalid="ignore", divide="ignore"):
        erro_rel = erro_px / tronco[:, None]
    return {
        "avaliar": avaliar, "quadro_valido": quadro_valido, "erro_px": erro_px, "erro_rel": erro_rel,
        "teste_det": teste_det, "teste_vis": np.nan_to_num(teste[..., 3]) >= LIMIAR_VISIBILIDADE,
        "tronco": tronco,
    }


def pck(c: dict, alfa: float, linhas=None, colunas=None, minimo: int = 1) -> float:
    """PCK em %. Devolve NaN se houver menos de `minimo` landmarks avaliados (pouco dado para concluir)."""
    av, ok = c["avaliar"], c["erro_rel"] <= alfa
    if linhas is not None:
        av, ok = av[linhas], ok[linhas]
    if colunas is not None:
        av, ok = av[:, list(colunas)], ok[:, list(colunas)]
    return 100.0 * float(ok[av].mean()) if av.sum() >= minimo else float("nan")


def contagem(c: dict, linhas=None, colunas=None) -> int:
    av = c["avaliar"]
    if linhas is not None:
        av = av[linhas]
    if colunas is not None:
        av = av[:, list(colunas)]
    return int(av.sum())


def pck_por_quadro(c: dict, alfa: float) -> np.ndarray:
    av, ok = c["avaliar"], c["erro_rel"] <= alfa
    n = av.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (ok & av).sum(axis=1) / n, np.nan)


def resumir(modelo: str, c: dict, fases: np.ndarray) -> dict:
    av = c["avaliar"]
    finito = av & np.isfinite(c["erro_px"])
    pq = pck_por_quadro(c, PRINCIPAL)
    validos = c["quadro_valido"]
    r = {
        "modelo": modelo,
        "quadros_avaliados": int(validos.sum()),
        "landmarks_avaliados": int(av.sum()),
        **{f"pck_{alfa}": pck(c, alfa) for alfa in LIMIARES},
        "quadros_corretos_pct": 100.0 * float((pq[validos] >= QUADRO_CORRETO).mean()) if validos.any() else float("nan"),
        "erro_medio_px": float(c["erro_px"][finito].mean()) if finito.any() else float("nan"),
        "erro_mediano_px": float(np.median(c["erro_px"][finito])) if finito.any() else float("nan"),
        "erro_medio_tronco_pct": 100.0 * float(c["erro_rel"][finito].mean()) if finito.any() else float("nan"),
        "deteccao_pct": 100.0 * float(c["teste_det"][validos].mean()) if validos.any() else float("nan"),
        "visibilidade_concorda_pct": 100.0 * float(c["teste_vis"][av].mean()) if av.any() else float("nan"),
    }
    # Grupo/pose com menos de 1 s de dados (30 quadros x pontos do grupo) fica sem valor.
    r["grupos"] = {g: pck(c, PRINCIPAL, colunas=idx, minimo=30 * len(idx)) for g, idx in GRUPOS.items()}
    r["n_grupos"] = {g: contagem(c, colunas=idx) for g, idx in GRUPOS.items()}
    r["poses"] = {p: pck(c, PRINCIPAL, linhas=fases == p) for p in POSES if (fases == p).any()}
    return r


# ---------------------------------------------------------------- graficos

def barras_agrupadas(ax, categorias, series, horizontal=False):
    """series = [(rotulo, cor, valores)]; valores em %."""
    n = len(series)
    largura = 0.8 / max(n, 2) * 0.9
    pos = np.arange(len(categorias))
    for k, (rotulo, cor, valores) in enumerate(series):
        desloc = (k - (n - 1) / 2) * (largura + 0.03)
        valores = np.asarray(valores, dtype=float)
        if horizontal:
            ax.barh(pos[::-1] + desloc * -1, valores, height=largura, color=cor, label=rotulo)
            for p, v in zip(pos[::-1] - desloc, valores):
                if not np.isnan(v):
                    ax.text(v + 1, p, br(v) + " %", va="center", fontsize=9.5, color=TINTA)
        else:
            ax.bar(pos + desloc, valores, width=largura, color=cor, label=rotulo)
            for p, v in zip(pos + desloc, valores):
                if not np.isnan(v):
                    ax.annotate(br(v), (p, v), xytext=(0, 4), textcoords="offset points", ha="center",
                                fontsize=9.5, fontweight="bold", color=TINTA)
    if horizontal:
        ax.set_yticks(pos[::-1], categorias)
        ax.set_xlim(0, 112)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlabel("taxa de acerto, PCK@0,2 (%)")
    else:
        ax.set_xticks(pos, categorias)
        ax.set_ylim(0, 112)
        ax.spines["bottom"].set_color(EIXO)
    ax.tick_params(length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=n, handlelength=1, handleheight=1, borderaxespad=0.2)


def grafico_pck(resumos, pasta):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cats = [f"PCK@{br(a)}\n({'rigoroso' if a < PRINCIPAL else 'taxa de acerto'})" for a in LIMIARES] + \
           ["Quadros\ncorretos"]
    series = [(r["modelo"].capitalize(), COR_MODELO[r["modelo"]],
               [r[f"pck_{a}"] for a in LIMIARES] + [r["quadros_corretos_pct"]]) for r in resumos]
    barras_agrupadas(ax, cats, series)
    ax.set_ylabel("% de acerto em relação ao Heavy")
    ax.set_title("Taxa de acerto em relação ao modelo Heavy", pad=30)
    fig.tight_layout()
    salvar(fig, pasta, "acuracia_pck.png")


def grafico_por_parte(resumos, pasta):
    grupos = [g for g in GRUPOS if any(not np.isnan(r["grupos"][g]) for r in resumos)]
    fora = [g.lower() for g in GRUPOS if g not in grupos]
    fig, ax = plt.subplots(figsize=(1.8 + 1.3 * len(grupos), 4.2))
    series = [(r["modelo"].capitalize(), COR_MODELO[r["modelo"]], [r["grupos"][g] for g in grupos]) for r in resumos]
    barras_agrupadas(ax, grupos, series)
    ax.set_ylabel("PCK@0,2 (%)")
    ax.set_title("Taxa de acerto por parte do corpo", pad=30)
    if fora:
        ax.text(0, -0.16, f"Não avaliados (fora do quadro): {', '.join(fora)}.", transform=ax.transAxes,
                fontsize=9, color=TINTA2)
    fig.tight_layout()
    salvar(fig, pasta, "acuracia_por_parte.png")


def grafico_por_pose(resumos, pasta):
    poses = [p for p in POSES if any(p in r["poses"] for r in resumos)]
    if not poses:
        print("  (video sem fases de pose: grafico por pose nao gerado)")
        return
    fig, ax = plt.subplots(figsize=(9.5, 1.4 + 0.62 * len(poses)))
    series = [(r["modelo"].capitalize(), COR_MODELO[r["modelo"]], [r["poses"].get(p, np.nan) for p in poses])
              for r in resumos]
    barras_agrupadas(ax, [NOME_POSE[p] for p in poses], series, horizontal=True)
    ax.set_title("Taxa de acerto por pose", pad=30)
    fig.tight_layout()
    salvar(fig, pasta, "acuracia_por_pose.png")


def _hex_bgr(cor: str):
    return tuple(int(cor[i:i + 2], 16) for i in (5, 3, 1))


def _esqueleto(img, px, vis, cor, esp):
    P = np.round(px).astype(int).tolist()
    for a, b in CONEXOES:
        if vis[a] and vis[b]:
            cv2.line(img, P[a], P[b], cor, esp, cv2.LINE_AA)
    for i in range(33):
        if vis[i]:
            cv2.circle(img, P[i], esp + 1, cor, -1, cv2.LINE_AA)


def grafico_exemplo(video, ref, testes, comps, fases, tamanho, pasta):
    """Pior quadro de cada modelo testado: esqueleto do Heavy (branco) x do modelo (cor)."""
    paineis = []
    for modelo, teste in testes.items():
        pq = pck_por_quadro(comps[modelo], PRINCIPAL)
        if np.isnan(pq).all():
            continue
        i = int(np.nanargmin(pq))
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, quadro = cap.read()
        cap.release()
        if not ok:
            continue
        ref_vis = np.nan_to_num(ref[i, :, 3]) >= LIMIAR_VISIBILIDADE
        _esqueleto(quadro, ref[i, :, :2] * tamanho, ref_vis, (30, 30, 30), 7)
        _esqueleto(quadro, ref[i, :, :2] * tamanho, ref_vis, (255, 255, 255), 4)
        if not np.isnan(teste[i, 0, 0]):
            _esqueleto(quadro, teste[i, :, :2] * tamanho, ref_vis, _hex_bgr(COR_MODELO[modelo]), 3)
        paineis.append((modelo, i, pq[i], quadro[..., ::-1]))
    if not paineis:
        return
    fig, eixos = plt.subplots(1, len(paineis), figsize=(6.2 * len(paineis), 4.4), squeeze=False)
    for ax, (modelo, i, valor, img) in zip(eixos[0], paineis):
        ax.imshow(img)
        ax.set_axis_off()
        pose = NOME_POSE.get(fases[i], "")
        ax.set_title(f"{modelo.capitalize()}: pior quadro (nº {i}{', ' + pose.lower() if pose else ''})\n"
                     f"{br(100 * valor, 0)} % dos landmarks corretos", fontsize=10.5)
        ax.text(0.01, 0.02, f"branco = Heavy (referência)   ·   cor = {modelo.capitalize()}",
                transform=ax.transAxes, fontsize=9, color="white",
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3})
    fig.tight_layout()
    salvar(fig, pasta, "acuracia_exemplo.png")


# ---------------------------------------------------------------- tabelas

def gravar_tabelas(resumos, video, total_quadros, pasta: Path):
    pasta.mkdir(parents=True, exist_ok=True)
    planos = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
               if k not in ("grupos", "poses", "n_grupos")}
              | {f"pck_{g}": round(v, 2) for g, v in r["grupos"].items()}
              | {f"pck_{p}": round(v, 2) for p, v in r["poses"].items()} for r in resumos]
    with open(pasta / "acuracia_por_modelo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(planos[0]))
        w.writeheader()
        w.writerows(planos)

    cab = "| Métrica | " + " | ".join(r["modelo"].capitalize() for r in resumos) + " |"
    sep = "|---|" + "---:|" * len(resumos)
    linha = lambda nome, fn: f"| {nome} | " + " | ".join(fn(r) for r in resumos) + " |"
    pct = lambda v: "–" if np.isnan(v) else br(v) + " %"
    n_quadros = max(resumos[0]["quadros_avaliados"], 1)
    md = [
        "# Taxa de acerto em relação ao modelo Heavy", "",
        f"Vídeo: `{video.name}` · {total_quadros} quadros · {resumos[0]['quadros_avaliados']} quadros avaliados "
        f"(ombros e quadris visíveis no Heavy) · {resumos[0]['landmarks_avaliados']} landmarks avaliados.", "",
        cab, sep,
        linha("**Taxa de acerto (PCK@0,2)**", lambda r: f"**{br(r['pck_0.2'])} %**"),
        linha("PCK@0,1 (rigoroso)", lambda r: br(r["pck_0.1"]) + " %"),
        linha(f"Quadros corretos (≥ {QUADRO_CORRETO:.0%} dos pontos)".replace(".", ","),
              lambda r: br(r["quadros_corretos_pct"]) + " %"),
        linha("Erro médio (px)", lambda r: br(r["erro_medio_px"])),
        linha("Erro mediano (px)", lambda r: br(r["erro_mediano_px"])),
        linha("Erro médio (% do tronco)", lambda r: br(r["erro_medio_tronco_pct"])),
        linha("Detecção nos quadros avaliados", lambda r: br(r["deteccao_pct"]) + " %"),
        linha("Visibilidade concorda com o Heavy", lambda r: br(r["visibilidade_concorda_pct"]) + " %"),
        "", "## Por parte do corpo (PCK@0,2)", "",
        "| Parte | Pontos visíveis por quadro | " + " | ".join(r["modelo"].capitalize() for r in resumos) + " |",
        "|---|---:|" + "---:|" * len(resumos),
        *[f"| {g} | {br(resumos[0]['n_grupos'][g] / n_quadros)} de {len(idx)} | "
          + " | ".join(pct(r["grupos"][g]) for r in resumos) + " |" for g, idx in GRUPOS.items()],
        "", "\"–\" = parte do corpo fora do quadro (menos de 1 s de dados): não avaliada.",
    ]
    poses = [p for p in POSES if any(p in r["poses"] for r in resumos)]
    if poses:
        md += ["", "## Por pose (PCK@0,2)", "", cab, sep,
               *[linha(NOME_POSE[p], lambda r, p=p: pct(r["poses"].get(p, float("nan")))) for p in poses]]
    md += ["", "## Como ler",
           "",
           "- **PCK@α**: % de landmarks a menos de α × comprimento do tronco do ponto dado pelo Heavy. "
           "Com o tronco medindo ~200 px, α = 0,2 equivale a ~40 px e α = 0,1 a ~20 px.",
           "- Só entram landmarks que o Heavy considera visíveis, em quadros com ombros e quadris visíveis.",
           "- **Limitação**: o Heavy é a referência, não um gabarito anotado à mão. Os três modelos têm a mesma "
           "arquitetura e treino, então erros que os três cometem juntos não aparecem aqui; o número mede "
           "**concordância com o modelo mais preciso**.", ""]
    (pasta / "RESUMO_acuracia.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  tabelas: {pasta / 'RESUMO_acuracia.md'}")


# ---------------------------------------------------------------- main

def video_mais_recente() -> Path:
    videos = sorted((RAIZ / "resultados" / "videos").glob("video_*.mp4"), key=lambda p: p.stat().st_mtime)
    if not videos:
        raise SystemExit("Nenhum video em resultados/videos. Grave um com:  python espelho.py --gravar-video")
    return videos[-1]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Taxa de acerto: Lite e Full comparados ao Heavy num video gravado")
    p.add_argument("--video", type=Path, help="video a analisar (padrao: o mais recente de resultados/videos)")
    p.add_argument("--saida", type=Path, default=RAIZ / "resultados")
    args = p.parse_args(argv)

    video = args.video or video_mais_recente()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Nao foi possivel abrir o video: {video}")
    tamanho = np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)])
    cap.release()

    print(f"Video: {video}")
    pasta_cache = video.parent / "landmarks"
    pontos = {m: extrair(video, m, pasta_cache) for m in (*TESTADOS, REFERENCIA)}
    n = min(len(v) for v in pontos.values())
    pontos = {m: v[:n] for m, v in pontos.items()}
    fases = ler_fases(video, n)

    ref = pontos[REFERENCIA]
    comps = {m: comparar(ref, pontos[m], tamanho) for m in TESTADOS}
    resumos = [resumir(m, comps[m], fases) for m in TESTADOS]

    validos = resumos[0]["quadros_avaliados"]
    print(f"\n{n} quadros, {validos} avaliados (ombros e quadris visiveis no Heavy).")
    if validos < 0.5 * n:
        print("ATENCAO: menos da metade dos quadros pode ser avaliada. Fique de corpo inteiro no quadro,\n"
              "         com ombros e quadris sempre visiveis, e grave de novo.")
    fora = [g for g in GRUPOS if np.isnan(resumos[0]["grupos"][g])]
    if fora:
        print(f"ATENCAO: {', '.join(fora).lower()} quase nunca apareceram no video e nao foram avaliados.\n"
              "         Para medir o corpo inteiro, grave mais longe da camera (pes visiveis).")
    print(f"\n  {'':34}" + "".join(f"{r['modelo'].capitalize():>10}" for r in resumos))
    for nome, chave, suf in (("Taxa de acerto, PCK@0,2 (%)", "pck_0.2", ""), ("PCK@0,1 (%)", "pck_0.1", ""),
                             ("Quadros corretos (%)", "quadros_corretos_pct", ""), ("Erro medio (px)", "erro_medio_px", ""),
                             ("Erro mediano (px)", "erro_mediano_px", "")):
        print(f"  {nome:34}" + "".join(f"{br(r[chave]):>10}" for r in resumos))
    for g in GRUPOS:
        print(f"  {'  ' + g + ' (PCK@0,2)':34}" + "".join(f"{br(r['grupos'][g]):>10}" for r in resumos))
    print()

    # Uma pasta por video: gravar de novo nao apaga o resultado das gravacoes anteriores.
    graficos = args.saida / "acuracia" / video.stem
    graficos.mkdir(parents=True, exist_ok=True)
    gravar_tabelas(resumos, video, n, graficos)
    grafico_pck(resumos, graficos)
    grafico_por_parte(resumos, graficos)
    grafico_por_pose(resumos, graficos)
    grafico_exemplo(video, ref, {m: pontos[m] for m in TESTADOS}, comps, fases, tamanho, graficos)


if __name__ == "__main__":
    main()
