"""Analise das sessoes gravadas pelo espelho.py (tecla R ou --registrar).

Le os CSVs de resultados/sessoes/, agrupa por modelo (Lite/Full/Heavy) e gera:
    resultados/metricas/resumo_por_modelo.csv
    resultados/metricas/RESUMO.md
    resultados/graficos/*.png

    python analise.py
    python analise.py --pasta outra/pasta --limiar-parado 4
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent
ORDEM_MODELOS = ("lite", "full", "heavy")
NAN = float("nan")

# Cores (superficie clara, tinta para texto, tres cores de serie)
SUPERFICIE, TINTA, TINTA2, MUDO, GRADE, EIXO = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "text.color": TINTA, "axes.labelcolor": TINTA2, "axes.edgecolor": EIXO,
    "xtick.color": MUDO, "ytick.color": MUDO, "xtick.labelcolor": TINTA2, "ytick.labelcolor": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRADE, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def br(valor: float, casas: int = 1) -> str:
    """Numero com virgula decimal."""
    return "–" if np.isnan(valor) else f"{valor:.{casas}f}".replace(".", ",")


# ---------------------------------------------------------------- leitura

def mediana_movel(v: np.ndarray, janela: int = 9) -> np.ndarray:
    saida = np.full(len(v), NAN)
    meio = janela // 2
    for i in range(len(v)):
        trecho = v[max(0, i - meio):i + meio + 1]
        trecho = trecho[~np.isnan(trecho)]
        if len(trecho):
            saida[i] = np.median(trecho)
    return saida


def carregar(arquivos: list[Path], min_linhas: int, limiar_parado: float) -> tuple[list[dict], int]:
    """Le cada CSV como um dicionario de colunas. Ignora gravacoes muito curtas."""
    sessoes, ignorados = [], 0
    for arq in sorted(arquivos):
        with open(arq, encoding="utf-8", newline="") as f:
            linhas = list(csv.DictReader(f))
        if len(linhas) < min_linhas:
            ignorados += 1
            continue

        def col(nome):
            return np.array([float(l[nome]) if l.get(nome) not in (None, "") else NAN for l in linhas])

        s = {c: col(c) for c in ("fps_inst", "captura_ms", "latencia_ms", "inferencia_ms", "detectado",
                                 "n_visiveis", "vis_media", "jitter_bruto_px", "jitter_exibido_px")}
        s["arquivo"] = arq.name
        s["modelo"] = np.array([l["modelo"] for l in linhas])
        s["suavizado"] = col("suavizacao") == 1
        s["periodo_ms"] = 1000.0 / s["fps_inst"]
        if np.isnan(s["captura_ms"]).all():
            # Gravacoes antigas nao tinham a coluna: o que sobra do periodo e a espera da camera.
            s["captura_ms"] = np.clip(s["periodo_ms"] - s["latencia_ms"], 0, None)
        # "Parado": no roteiro guiado vem marcado na coluna fase. Na gravacao livre e
        # estimado: o deslocamento tipico dos landmarks na vizinhanca do quadro e pequeno.
        fase = np.array([l.get("fase") or "" for l in linhas])
        s["guiado"] = bool((fase != "").any())
        s["parado"] = np.where(fase != "", fase == "parado", mediana_movel(s["jitter_bruto_px"]) < limiar_parado)
        sessoes.append(s)
    return sessoes, ignorados


def resumir(sessoes: list[dict], modelo: str) -> dict:
    partes = [(s, s["modelo"] == modelo) for s in sessoes]
    partes = [(s, m) for s, m in partes if m.any()]

    def junta(nome, filtro=None):
        return np.concatenate([s[nome][m if filtro is None else m & filtro(s)] for s, m in partes])

    media = lambda v: float(np.nanmean(v)) if np.isfinite(v).any() else NAN
    lat, det = junta("latencia_ms"), junta("detectado")
    periodo = junta("periodo_ms")
    parado_bruto = junta("jitter_bruto_px", lambda s: s["parado"])
    parado_suave = junta("jitter_exibido_px", lambda s: s["parado"] & s["suavizado"])
    return {
        "modelo": modelo,
        "quadros": len(lat),
        "duracao_s": float(np.nansum(periodo)) / 1000.0,
        "fps": 1000.0 / media(periodo),
        "latencia_ms": media(lat),
        "latencia_p95_ms": float(np.percentile(lat, 95)),
        "inferencia_ms": media(junta("inferencia_ms")),
        "captura_ms": media(junta("captura_ms")),
        "deteccao_pct": 100.0 * float(det.mean()),
        "trocas_deteccao": int(sum(np.abs(np.diff(s["detectado"][m])).sum() for s, m in partes)),
        "landmarks_visiveis": media(junta("n_visiveis", lambda s: s["detectado"] == 1)),
        "visibilidade": media(junta("vis_media")),
        "quadros_parados": int(np.isfinite(parado_bruto).sum()),
        "tremor_bruto_px": media(parado_bruto),
        "tremor_suavizado_px": media(parado_suave),
    }


# ---------------------------------------------------------------- graficos

def barras(ax, nomes, valores, titulo, casas=1, cor=AZUL, sufixo=""):
    x = np.arange(len(nomes))
    ax.bar(x, valores, width=0.42, color=cor)
    for xi, v in zip(x, valores):
        if not np.isnan(v):
            ax.annotate(br(v, casas) + sufixo, (xi, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold", color=TINTA)
    ax.set_xticks(x, nomes)
    folga = (max(3, len(nomes)) - len(nomes)) / 2      # com 1 ou 2 modelos a barra nao engorda
    ax.set_xlim(-0.6 - folga, len(nomes) - 0.4 + folga)
    ax.set_title(titulo)
    ax.tick_params(length=0)
    ax.spines["bottom"].set_color(EIXO)
    topo = np.nanmax(valores) if np.isfinite(valores).any() else 1.0
    ax.set_ylim(0, topo * 1.18)


def salvar(fig, pasta: Path, nome: str) -> None:
    fig.savefig(pasta / nome, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  grafico: {pasta / nome}")


def grafico_desempenho(resumos, pasta):
    nomes = [r["modelo"].capitalize() for r in resumos]
    fig, eixos = plt.subplots(1, 3, figsize=(10.5, 3.8))
    barras(eixos[0], nomes, np.array([r["fps"] for r in resumos]), "FPS médio")
    barras(eixos[1], nomes, np.array([r["latencia_ms"] for r in resumos]), "Latência média (ms)")
    barras(eixos[2], nomes, np.array([r["inferencia_ms"] for r in resumos]), "Inferência do modelo (ms)")
    fig.suptitle("Desempenho por modelo", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    salvar(fig, pasta, "desempenho_por_modelo.png")


def grafico_tempo_por_quadro(resumos, pasta):
    """Para onde vai o tempo de cada quadro: espera da camera, inferencia, desenho+exibicao."""
    partes = [("Espera da câmera", AZUL, [r["captura_ms"] for r in resumos]),
              ("Inferência do modelo", LARANJA, [r["inferencia_ms"] for r in resumos]),
              ("Desenho e exibição", VERDE, [r["latencia_ms"] - r["inferencia_ms"] for r in resumos])]
    nomes = [r["modelo"].capitalize() for r in resumos]
    y = np.arange(len(resumos))[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 1.5 + 0.75 * len(resumos)))
    esquerda = np.zeros(len(resumos))
    total = sum(np.array(v) for _, _, v in partes)
    for rotulo, cor, valores in partes:
        valores = np.array(valores)
        ax.barh(y, valores, left=esquerda, height=0.42, color=cor, edgecolor=SUPERFICIE, linewidth=2, label=rotulo)
        for yi, v, e, t in zip(y, valores, esquerda, total):
            if v / total.max() > 0.065:     # so rotula o segmento se o texto couber
                ax.text(e + v / 2, yi, br(v, 0) + " ms", ha="center", va="center", color="white",
                        fontsize=9.5, fontweight="bold")
        esquerda += valores
    for yi, t, r in zip(y, total, resumos):
        ax.text(t + total.max() * 0.012, yi, f"{br(t, 0)} ms por quadro = {br(r['fps'])} FPS",
                va="center", fontsize=10, color=TINTA)
    ax.set_yticks(y, nomes)
    ax.set_xlim(0, total.max() * 1.32)
    ax.set_ylim(-0.6, len(resumos) - 0.4)
    ax.set_xlabel("tempo médio por quadro (ms)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(length=0)
    ax.set_title("Para onde vai o tempo de cada quadro", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3, handlelength=1, handleheight=1, columnspacing=1.6,
              borderaxespad=0.2)
    fig.tight_layout()
    salvar(fig, pasta, "tempo_por_quadro.png")


def grafico_latencia_no_tempo(sessoes, resumos, pasta):
    fig, eixos = plt.subplots(len(resumos), 1, figsize=(10.5, 1.2 + 1.9 * len(resumos)), squeeze=False)
    topo = 1.08 * max(np.nanmax(s["latencia_ms"]) for s in sessoes)     # mesma escala em todos os paineis
    for ax, r in zip(eixos[:, 0], resumos):
        t0 = 0.0
        for s in sessoes:
            m = s["modelo"] == r["modelo"]
            if not m.any():
                continue
            t = t0 + np.nancumsum(s["periodo_ms"][m]) / 1000.0
            ax.plot(t, s["latencia_ms"][m], color=AZUL, linewidth=1.4, solid_joinstyle="round")
            t0 = t[-1] + 1.0        # pequeno intervalo entre gravacoes diferentes
        ax.axhline(r["latencia_ms"], color=MUDO, linewidth=0.8)
        ax.text(1.0, 1.03, f"média {br(r['latencia_ms'])} ms (linha cinza) · p95 {br(r['latencia_p95_ms'])} ms",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=TINTA2)
        ax.set_title(r["modelo"].capitalize())
        ax.set_ylabel("latência (ms)")
        ax.set_ylim(0, topo)
        ax.tick_params(length=0)
        ax.spines["bottom"].set_color(EIXO)
    eixos[-1, 0].set_xlabel("tempo de gravação (s)")
    fig.suptitle("Latência quadro a quadro", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    salvar(fig, pasta, "latencia_no_tempo.png")


def grafico_estabilidade(resumos, pasta):
    com_dado = [r for r in resumos if not np.isnan(r["tremor_bruto_px"])]
    if not com_dado:
        print("  (sem trecho parado nas gravacoes: grafico de estabilidade nao gerado)")
        return
    nomes = [r["modelo"].capitalize() for r in com_dado]
    x = np.arange(len(com_dado))
    fig, ax = plt.subplots(figsize=(1.8 + 2.4 * len(com_dado), 3.9))
    for desloc, chave, cor, rotulo in ((-0.17, "tremor_bruto_px", AZUL, "Sem suavização"),
                                       (0.17, "tremor_suavizado_px", LARANJA, "Com filtro One Euro")):
        valores = np.array([r[chave] for r in com_dado])
        ax.bar(x + desloc, valores, width=0.3, color=cor, label=rotulo)
        for xi, v in zip(x + desloc, valores):
            if not np.isnan(v):
                ax.annotate(br(v, 2), (xi, v), xytext=(0, 4), textcoords="offset points", ha="center",
                            fontsize=10, fontweight="bold", color=TINTA)
    ax.set_xticks(x, nomes)
    ax.set_xlim(-0.7, len(com_dado) - 0.3)
    ax.set_ylim(0, max(r["tremor_bruto_px"] for r in com_dado) * 1.25)
    ax.set_ylabel("deslocamento médio (px por quadro)")
    ax.tick_params(length=0)
    ax.spines["bottom"].set_color(EIXO)
    ax.set_title("Tremor dos landmarks com a pessoa parada", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, handlelength=1, handleheight=1, borderaxespad=0.2)
    fig.tight_layout()
    salvar(fig, pasta, "estabilidade.png")


def grafico_deteccao(resumos, pasta):
    nomes = [r["modelo"].capitalize() for r in resumos]
    fig, eixos = plt.subplots(1, 2, figsize=(8, 3.8))
    barras(eixos[0], nomes, np.array([r["deteccao_pct"] for r in resumos]), "Quadros com pose detectada (%)", sufixo=" %")
    barras(eixos[1], nomes, np.array([r["landmarks_visiveis"] for r in resumos]), "Landmarks visíveis (de 33)")
    eixos[1].set_ylim(0, 37)
    fig.suptitle("Detecção por modelo", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    salvar(fig, pasta, "deteccao_por_modelo.png")


# ---------------------------------------------------------------- tabelas

COLUNAS_TABELA = [
    ("Modelo", lambda r: r["modelo"].capitalize()),
    ("Quadros", lambda r: str(r["quadros"])),
    ("Duração (s)", lambda r: br(r["duracao_s"])),
    ("FPS", lambda r: br(r["fps"])),
    ("Latência média (ms)", lambda r: br(r["latencia_ms"])),
    ("Latência p95 (ms)", lambda r: br(r["latencia_p95_ms"])),
    ("Inferência (ms)", lambda r: br(r["inferencia_ms"])),
    ("Espera da câmera (ms)", lambda r: br(r["captura_ms"])),
    ("Detecção (%)", lambda r: br(r["deteccao_pct"])),
    ("Trocas de detecção", lambda r: str(r["trocas_deteccao"])),
    ("Landmarks visíveis", lambda r: br(r["landmarks_visiveis"])),
    ("Tremor parado, bruto (px)", lambda r: br(r["tremor_bruto_px"], 2)),
    ("Tremor parado, suavizado (px)", lambda r: br(r["tremor_suavizado_px"], 2)),
]


def gravar_tabelas(resumos, sessoes, ignorados, pasta: Path, limiar_parado: float) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    with open(pasta / "resumo_por_modelo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumos[0]))
        w.writeheader()
        w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in resumos)

    linhas = ["# Resumo das sessões gravadas", "",
              f"{len(sessoes)} gravações analisadas" + (f" ({ignorados} muito curtas foram ignoradas)." if ignorados else "."),
              "", "| Métrica | " + " | ".join(r["modelo"].capitalize() for r in resumos) + " |",
              "|---|" + "---:|" * len(resumos)]
    for titulo, fn in COLUNAS_TABELA[1:]:
        linhas.append(f"| {titulo} | " + " | ".join(fn(r) for r in resumos) + " |")
    linhas += ["", "- **FPS**: quadros apresentados por segundo.",
               "- **Latência**: do quadro chegar da câmera até estar visível na janela.",
               "- **Espera da câmera**: tempo parado em `cap.read()` aguardando o próximo quadro.",
               "- **Trocas de detecção**: vezes em que a pose foi perdida ou reencontrada.",
               "- **Tremor parado**: deslocamento médio dos landmarks entre quadros, só com a pessoa parada. "
               + ("Os trechos parados são os marcados pelo roteiro guiado (`--roteiro`)."
                  if all(s["guiado"] for s in sessoes) else
                  f"Nas gravações livres, \"parado\" é estimado pelos próprios dados (mediana móvel do "
                  f"deslocamento abaixo de {br(limiar_parado, 0)} px)."), ""]
    (pasta / "RESUMO.md").write_text("\n".join(linhas), encoding="utf-8")
    print(f"  tabelas: {pasta / 'RESUMO.md'}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Gera tabelas e graficos a partir das sessoes gravadas")
    p.add_argument("--pasta", type=Path, default=RAIZ / "resultados" / "sessoes")
    p.add_argument("--arquivos", type=Path, nargs="+", help="analisa so estes CSVs, em vez da pasta inteira")
    p.add_argument("--saida", type=Path, default=RAIZ / "resultados")
    p.add_argument("--min-linhas", type=int, default=30, help="ignora gravacoes com menos quadros que isso")
    p.add_argument("--limiar-parado", type=float, default=4.0, help="deslocamento (px) abaixo do qual a pessoa conta como parada")
    args = p.parse_args(argv)

    arquivos = args.arquivos or list(args.pasta.glob("*.csv"))
    sessoes, ignorados = carregar(arquivos, args.min_linhas, args.limiar_parado)
    if not sessoes:
        raise SystemExit(f"Nenhuma gravacao com pelo menos {args.min_linhas} quadros em {args.arquivos or args.pasta}")
    if any(s["guiado"] for s in sessoes) and not all(s["guiado"] for s in sessoes):
        print("AVISO: ha gravacoes livres misturadas com gravacoes do roteiro guiado. Para comparar os modelos\n"
              "       com justica, analise so o roteiro:  python analise.py --arquivos <roteiro_....csv>\n")
    presentes = {m for s in sessoes for m in np.unique(s["modelo"])}
    resumos = [resumir(sessoes, m) for m in ORDEM_MODELOS if m in presentes]

    print(f"{len(sessoes)} gravacoes analisadas, {ignorados} ignoradas por serem muito curtas.\n")
    for titulo, fn in COLUNAS_TABELA:
        print(f"  {titulo:30}" + "".join(f"{fn(r):>10}" for r in resumos))
    faltam = [m for m in ORDEM_MODELOS if m not in presentes]
    if faltam:
        print(f"\n  ATENCAO: nao ha gravacao do(s) modelo(s): {', '.join(faltam)}.")
    print()

    graficos = args.saida / "graficos"
    graficos.mkdir(parents=True, exist_ok=True)
    gravar_tabelas(resumos, sessoes, ignorados, args.saida / "metricas", args.limiar_parado)
    grafico_desempenho(resumos, graficos)
    grafico_tempo_por_quadro(resumos, graficos)
    grafico_latencia_no_tempo(sessoes, resumos, graficos)
    grafico_estabilidade(resumos, graficos)
    grafico_deteccao(resumos, graficos)


if __name__ == "__main__":
    main()
