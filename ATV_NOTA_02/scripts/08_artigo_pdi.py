"""Tabelas do artigo no template da disciplina (elsarticle 5p, duas colunas), em ingles e em portugues.

Ingles:    artigo_pdi/tables/     (figuras: artigo_pdi/figures/,    copiadas de paper/figures)
Portugues: artigo_pdi/tables_pt/  (figuras: artigo_pdi/figures_pt/, copiadas de artigo_sbc/figuras)

Le apenas os CSVs ja calculados em outputs/results (nao carrega os dados nem as predicoes), entao roda em
segundos. Os rotulos das tabelas (\\label) sao os mesmos nas duas linguas.

Uso: python scripts/08_artigo_pdi.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gsseg.metrics import bootstrap_ci  # noqa: E402

RES = ROOT / "outputs" / "results"
OUT = ROOT / "artigo_pdi"
ARCHS = ["unet", "segformer", "mask2former"]
REGS = ["gliders", "bubbles", "maze", "worms", "spirals", "spots"]
FIGS = ["visual_table_spirals", "visual_table_gliders", "heatmap_dice", "temporal_dice_area"]
DATASET = r"\texttt{gray\_scott\_reaction\_}\newline\texttt{diffusion}"

LANG = "en"
TAB = OUT / "tables"

NAMES = {
    "en": dict(
        ARCH={"unet": "U-Net", "segformer": "SegFormer", "mask2former": "Mask2Former", "otsu_A": "Otsu on $A$"},
        COND={"No pre/post": "Architecture only", "Pre": "Pre + arch.", "Post": "Arch. + post",
              "Pre+Post": "Pre + arch. + post", "Reference": "Reference"},
        PRE={"P0": "Baseline", "P1": "Z-score", "P2": "RMS", "P3": "Robust", "P4": "Min-max", "P5": "Percentile clip.",
             "P6": "Gaussian", "P7": "Median", "P8": "Bilateral", "P9": "Anisotropic diff.", "P10": "Wavelet"},
        POST={"Q0": "Baseline", "Q1": "Threshold optim.", "Q2": "Opening", "Q3": "Closing",
              "Q4": "Component filter", "Q5": "Temporal filter"},
        REG=["Glid.", "Bubb.", "Maze", "Worms", "Spir.", "Spots"], NA="n/a", PRE_TAG="(pre)", POST_TAG="(post)"),
    "pt": dict(
        ARCH={"unet": "U-Net", "segformer": "SegFormer", "mask2former": "Mask2Former", "otsu_A": "Otsu em $A$"},
        COND={"No pre/post": "Arquitetura sozinha", "Pre": "Pré + arq.", "Post": "Arq. + pós",
              "Pre+Post": "Pré + arq. + pós", "Reference": "Referência"},
        PRE={"P0": "\\textit{Baseline}", "P1": "Z-score", "P2": "RMS", "P3": "Robusta", "P4": "Mín-máx",
             "P5": "Recorte por percentis", "P6": "Gaussiano", "P7": "Mediana", "P8": "Bilateral",
             "P9": "Difusão anisotrópica", "P10": "\\textit{Wavelet}"},
        POST={"Q0": "\\textit{Baseline}", "Q1": "Otimização do limiar", "Q2": "Abertura", "Q3": "Fechamento",
              "Q4": "Filtro de componentes", "Q5": "Filtro temporal"},
        REG=["Glid.", "Bubb.", "Maze", "Worms", "Spir.", "Spots"], NA="n/d", PRE_TAG="(pré)", POST_TAG="(pós)"),
}


def N(key):
    return NAMES[LANG][key]


def T(en, pt):
    return pt if LANG == "pt" else en


def nf(x, d=3, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    s = s.replace("-", "$-$")
    return s.replace(".", ",") if LANG == "pt" else s


def pm(r, k, d=3):
    half = (r[f"{k}_hi"] - r[f"{k}_lo"]) / 2
    return f"{nf(r[k], d)}{{\\scriptsize$\\pm${nf(half, d)}}}"


def save(lines, name):
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected(g):
    std = pd.read_csv(RES / "results_standard.csv")
    return [g[(g.exp_id == s.Experiment) & (g.split == "test")].iloc[0] for _, s in std.iterrows()]


def sep(L, prev, r):
    if prev is not None and prev != r.arch:
        L.append(r"\addlinespace")
    return r.arch


def tab_full_results(rows):
    keys = ["IoU", "Dice", "Precision", "Recall"]
    best = {k: max(r[k] for r in rows) for k in keys}
    cap = T(r"\caption{Experimental results on the test set (120 trajectories, 2,520 frames) for the four conditions of "
            r"each architecture and for the learning-free reference: architecture alone (P0+Q0), best pre-processing, "
            r"best post-processing and best pair, each selected on the validation split by mean Dice. Mean over test "
            r"trajectories $\pm$ half-width of the 95\% stratified bootstrap CI. Latency: network + post-processing per "
            r"$128\times128$ frame, batch 1, NVIDIA RTX A5000. Best value per column in bold. The 198 combinations are "
            r"in the supplementary file \texttt{results\_sequential.csv}.}",
            r"\caption{Resultados no teste (120 trajetórias, 2.520 quadros) nas quatro condições de cada arquitetura e "
            r"para a referência sem aprendizado: arquitetura sozinha (P0+Q0), melhor pré-processamento, melhor "
            r"pós-processamento e melhor par, cada um escolhido na validação pelo Dice médio. Média entre trajetórias de "
            r"teste $\pm$ meia-amplitude do IC de 95\% por \textit{bootstrap} estratificado. Latência: rede + "
            r"pós-processamento por quadro de $128\times128$, lote 1, NVIDIA RTX A5000. Em negrito, o melhor valor de "
            r"cada coluna. As 198 combinações estão no arquivo suplementar \texttt{results\_sequential.csv}.}")
    head = T(r"\textbf{Dataset} & \textbf{ID} & \textbf{Pre-processing} & \textbf{Architecture / post-processing} & "
             r"\textbf{IoU} & \textbf{Dice} & \textbf{Prec.} & \textbf{Rec.} & \textbf{Latency (ms)} \\",
             r"\textbf{Conjunto} & \textbf{ID} & \textbf{Pré-processamento} & \textbf{Arquitetura / pós-processamento} & "
             r"\textbf{IoU} & \textbf{Dice} & \textbf{Prec.} & \textbf{Rev.} & \textbf{Latência (ms)} \\")
    L = [r"\begin{table*}[t]", r"\centering", cap, r"\label{tab:full_results}", r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}", r"\begin{tabularx}{\textwidth}{L{2.05cm} L{0.6cm} L{2.3cm} Y c c c c c}",
         r"\toprule", head, r"\midrule"]
    prev = None
    for i, r in enumerate(rows):
        prev = sep(L, prev, r)
        ds = r"\multirow{13}{2.05cm}{\texttt{gray\_scott\_}\newline\texttt{reaction\_}\newline\texttt{diffusion}}" if i == 0 else ""
        cells = [f"\\textbf{{{pm(r, k)}}}" if abs(r[k] - best[k]) < 5e-7 else pm(r, k) for k in keys]
        lat = r.latency_ms_b1 + r.post_ms_per_frame
        L.append(f"{ds} & {r.exp_id} & {r.pre} {N('PRE')[r.pre]} & {N('ARCH')[r.arch]} / {r.post} {N('POST')[r.post]} & "
                 + " & ".join(cells) + f" & {nf(lat, 2)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    save(L, "tab_full_results")


def tab_architecture_comparison(rows):
    by = {(r.arch, r.condition): r for r in rows}
    cap = T(r"\caption{Architecture comparison (test IoU / Dice). Aggregation rule: first row, architecture alone "
            r"(P0+Q0); second row, the pre- + post-processing pair selected on the validation split for each architecture.}",
            r"\caption{Comparação entre arquiteturas (IoU / Dice no teste). Regra de agregação: primeira linha, "
            r"arquitetura sozinha (P0+Q0); segunda linha, o par de pré e pós-processamento escolhido na validação para "
            r"cada arquitetura.}")
    L = [r"\begin{table}[t]", r"\centering", cap, r"\label{tab:architecture_comparison}", r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}", r"\begin{tabularx}{\columnwidth}{L{2.3cm} *{3}{>{\centering\arraybackslash}X}}",
         r"\toprule", T(r"\textbf{Dataset}", r"\textbf{Conjunto}") + r" & \textbf{U-Net} & \textbf{SegFormer} & "
         r"\textbf{Mask2Former} \\", r"\midrule"]
    for cond, lab in [("No pre/post", "P0+Q0"), ("Pre+Post", T("selected P+Q", "P+Q escolhido"))]:
        cells = [f"{nf(by[(a, cond)].IoU)} / {nf(by[(a, cond)].Dice)}" for a in ARCHS]
        L.append(f"\\texttt{{gray\\_scott}} ({lab}) & " + " & ".join(cells) + r" \\")
    sel = [f"{by[(a, 'Pre+Post')].pre}+{by[(a, 'Pre+Post')].post}" for a in ARCHS]
    L.append(T(r"\quad selected pair", r"\quad par escolhido") + " & " + " & ".join(sel) + r" \\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    save(L, "tab_architecture_comparison")


def tab_prepost_effect(boot):
    pt = pd.read_csv(RES / "per_traj_sem_n0.csv")
    pdn = pd.read_csv(RES / "paired_deltas_sem_n0.csv")
    items = [(p, "Q0") for p in list(N("PRE"))[1:]] + [("P0", q) for q in list(N("POST"))[1:]]
    recs = []

    def cell(m, lo, hi):
        s = r"$\pm$" + nf(0, 2) if abs(100 * m) < 0.005 else nf(100 * m, 2, sign=True)
        return f"\\textbf{{{s}}}" if (lo > 0 or hi < 0) else s

    cap = T(r"\caption{Ablation analysis of pre-processing and post-processing effects: paired change of test IoU and "
            r"Dice (percentage points) of each technique with respect to the same architecture without pre- and "
            r"post-processing (P0+Q0), computed on the same 120 test trajectories. $\Delta$Dice$_{n\geq50}$ repeats the "
            r"Dice change without the random initial snapshot $n=0$. Bold: 95\% stratified bootstrap CI excludes zero. "
            r"Values below 0.005 points in magnitude appear as $\pm$0.00.}",
            r"\caption{Análise de ablação dos efeitos de pré e pós-processamento: variação pareada do IoU e do Dice no "
            r"teste (pontos percentuais) de cada técnica em relação à mesma arquitetura sem pré e pós-processamento "
            r"(P0+Q0), calculada nas mesmas 120 trajetórias de teste. $\Delta$Dice$_{n\geq50}$ repete a variação do Dice "
            r"sem o instante inicial aleatório $n=0$. Em negrito: IC de 95\% por \textit{bootstrap} estratificado não "
            r"contém zero. Valores abaixo de 0,005 ponto em módulo aparecem como $\pm$0,00.}")
    L = [r"\begin{table*}[t]", r"\centering", cap, r"\label{tab:prepost_effect}", r"\scriptsize",
         r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabularx}{\textwidth}{L{0.6cm} Y *{9}{c}}", r"\toprule",
         r" & & \multicolumn{3}{c}{\textbf{U-Net}} & \multicolumn{3}{c}{\textbf{SegFormer}} & "
         r"\multicolumn{3}{c}{\textbf{Mask2Former}} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"\textbf{ID} & " + T(r"\textbf{Stage / method}", r"\textbf{Etapa / método}")
         + r" & $\Delta$IoU & $\Delta$Dice & $\Delta$Dice$_{n\geq50}$ & $\Delta$IoU & $\Delta$Dice & "
         r"$\Delta$Dice$_{n\geq50}$ & $\Delta$IoU & $\Delta$Dice & $\Delta$Dice$_{n\geq50}$ \\", r"\midrule"]
    for k, (p, q) in enumerate(items):
        if k == 10:
            L.append(r"\midrule")
        code = p if q == "Q0" else q
        name = f"{N('PRE')[p]} {N('PRE_TAG')}" if q == "Q0" else f"{N('POST')[q]} {N('POST_TAG')}"
        cells = []
        for a in ARCHS:
            base = pt[(pt.arch == a) & (pt.pre == "P0") & (pt.post == "Q0")].set_index("traj")
            cur = pt[(pt.arch == a) & (pt.pre == p) & (pt.post == q)].set_index("traj")
            d = (cur.IoU_all - base.IoU_all).loc[base.index].values
            m, lo, hi = bootstrap_ci(d, boot["n"], seed=boot["seed"], groups=base.regime.values)
            cells.append(cell(m, lo, hi))
            recs.append(dict(arch=a, technique=code, metric="IoU_all", delta=m, ci_low=lo, ci_high=hi))
            for met in ("Dice_all", "Dice_sem_n0"):
                x = pdn[(pdn.arch == a) & (pdn.technique == code) & (pdn.metric == met)].iloc[0]
                cells.append(cell(x.delta, x.ci_low, x.ci_high))
        L.append(f"{code} & {name} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    save(L, "tab_prepost_effect")
    if LANG == "en":
        pd.DataFrame(recs).to_csv(RES / "paired_deltas_iou_dice_artigo.csv", index=False)


def tab_best_and_cross(rows, g):
    val = g[(g.split == "valid") & (g.arch != "otsu_A")]
    vb = val.loc[val.Dice.idxmax()]
    best = g[(g.split == "test") & (g.exp_id == vb.exp_id)].iloc[0]
    base = next(r for r in rows if r.arch == best.arch and r.condition == "No pre/post")
    pt = pd.read_csv(RES / "per_traj_sem_n0.csv")
    sem = {r.exp_id: pt[(pt.arch == r.arch) & (pt.pre == r.pre) & (pt.post == r.post)].Dice_sem_n0.mean()
           for r in (best, base)}
    A, PRE, POST = N("ARCH"), N("PRE"), N("POST")
    L = [r"\begin{table*}[t]", r"\centering",
         T(r"\caption{Best validation-selected configuration and held-out test performance. Selection criterion fixed "
           r"before testing: highest mean validation Dice among all 198 combinations.}",
           r"\caption{Melhor configuração escolhida na validação e desempenho no teste. Critério de seleção fixado antes "
           r"do teste: maior Dice médio de validação entre as 198 combinações.}"),
         r"\label{tab:best_configuration}", r"\small",
         r"\begin{tabularx}{\textwidth}{L{4.3cm} L{1.6cm} L{2.4cm} L{2.8cm} c c c}", r"\toprule",
         T(r"\textbf{Dataset} & \textbf{Architecture} & \textbf{Pre-processing} & \textbf{Post-processing} & "
           r"\textbf{IoU} & \textbf{Dice} & \textbf{95\% CI (Dice)} \\",
           r"\textbf{Conjunto} & \textbf{Arquitetura} & \textbf{Pré-processamento} & \textbf{Pós-processamento} & "
           r"\textbf{IoU} & \textbf{Dice} & \textbf{IC 95\% (Dice)} \\"), r"\midrule",
         f"{DATASET} & {A[best.arch]} & {best.pre} {PRE[best.pre]} & {best.post} {POST[best.post]} & "
         f"{nf(best.IoU)} & {nf(best.Dice)} & [{nf(best.Dice_lo)}; {nf(best.Dice_hi)}]" + r" \\",
         r"\bottomrule", r"\end{tabularx}", r"\vspace{6pt}"]
    gain = 100 * (best.Dice / base.Dice - 1)
    gain_sem = 100 * (sem[best.exp_id] / sem[base.exp_id] - 1)
    note = T(f"Gain over {A[base.arch]} alone ({nf(base.Dice)}) comes from the random initial snapshot; for "
             f"$n\\geq50$, Dice {nf(sem[best.exp_id])} vs.\\ {nf(sem[base.exp_id])} ({nf(gain_sem, 1, True)}\\%).",
             f"O ganho sobre a {A[base.arch]} sozinha ({nf(base.Dice)}) vem do instante inicial aleatório; em "
             f"$n\\geq50$, Dice de {nf(sem[best.exp_id])} contra {nf(sem[base.exp_id])} ({nf(gain_sem, 1, True)}\\%).")
    L += [T(r"\caption{Cross-dataset summary of the selected segmentation approach. Relative gain: test Dice of the "
            r"selected configuration relative to the same architecture without pre- and post-processing.}",
            r"\caption{Resumo entre conjuntos da abordagem de segmentação escolhida. Ganho relativo: Dice no teste da "
            r"configuração escolhida em relação à mesma arquitetura sem pré e pós-processamento.}"),
          r"\label{tab:cross_dataset}", r"\small", r"\begin{tabularx}{\textwidth}{L{4.3cm} L{1.7cm} L{1.4cm} c c Y}",
          r"\toprule",
          T(r"\textbf{Dataset} & \textbf{Selected architecture} & \textbf{Selected P / Q} & \textbf{Test Dice} & "
            r"\textbf{Relative gain (\%)} & \textbf{Generalization notes} \\",
            r"\textbf{Conjunto} & \textbf{Arquitetura escolhida} & \textbf{P / Q escolhidos} & \textbf{Dice no teste} & "
            r"\textbf{Ganho relativo (\%)} & \textbf{Observações de generalização} \\"), r"\midrule",
          f"{DATASET} & {A[best.arch]} & {best.pre} / {best.post} & {nf(best.Dice)} & {nf(gain, 2, True)} & {note}" + r" \\",
          r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    save(L, "tab_best_cross")
    return dict(best=best.exp_id, gain=gain, gain_sem=gain_sem)


def tab_classwise_temporal(rows):
    cap = T(r"\caption{Class-wise and temporal metrics on the test set for the configurations of "
            r"Table~\ref{tab:full_results}. IoU$_{bg}$, Dice$_{bg}$: background class; mIoU: mean IoU of the two classes; "
            r"AVE: area-variation error (\% of the domain per sampled step); CDE: structure-centroid displacement error "
            r"(pixels, periodic distance); TC: temporal consistency of the masks; TCE $=|\mathrm{TC}_{pred}-"
            r"\mathrm{TC}_{GT}|$; memory: peak GPU memory at inference (batch 1). Mean $\pm$ half-width of the 95\% CI.}",
            r"\caption{Métricas por classe e temporais no teste para as configurações da Tabela~\ref{tab:full_results}. "
            r"IoU$_{f}$, Dice$_{f}$: classe fundo; mIoU: média do IoU das duas classes; AVE: erro de variação de área "
            r"(\% do domínio por instante amostrado); CDE: erro de deslocamento dos centroides das estruturas (pixels, "
            r"distância periódica); TC: consistência temporal das máscaras; TCE $=|\mathrm{TC}_{pred}-\mathrm{TC}_{GT}|$; "
            r"memória: pico de memória da GPU na inferência (lote 1). Média $\pm$ meia-amplitude do IC de 95\%.}")
    bg = T("bg", "f")
    L = [r"\begin{table*}[t]", r"\centering", cap, r"\label{tab:classwise_temporal}", r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}", r"\begin{tabularx}{\textwidth}{L{0.7cm} L{1.6cm} Y c c c c c c c c}", r"\toprule",
         r"\textbf{ID} & " + T(r"\textbf{Architecture} & \textbf{Condition}", r"\textbf{Arquitetura} & \textbf{Condição}")
         + f" & \\textbf{{IoU$_{{{bg}}}$}} & \\textbf{{Dice$_{{{bg}}}$}} & "
         + r"\textbf{mIoU} & \textbf{AVE (\%)} & \textbf{CDE (px)} & \textbf{TC} & \textbf{TCE} & "
         + T(r"\textbf{Mem. (MiB)} \\", r"\textbf{Mem. (MiB)} \\"), r"\midrule"]
    prev = None
    for r in rows:
        prev = sep(L, prev, r)
        mem = N("NA") if r.arch == "otsu_A" else f"{r.peak_infer_mem_mib_b1:.0f}"
        L.append(f"{r.exp_id} & {N('ARCH')[r.arch]} & {N('COND')[r.condition]} ({r.pre}+{r.post}) & {pm(r, 'IoU_bg')} & "
                 f"{pm(r, 'Dice_bg')} & {pm(r, 'mIoU')} & {pm(r, 'AVE', 2)} & {pm(r, 'CDE', 2)} & {nf(r.TC)} & "
                 f"{pm(r, 'TCE')} & {mem}" + r" \\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    save(L, "tab_classwise_temporal")


def tab_sensitivity(rows):
    pt = pd.read_csv(RES / "per_traj_sem_n0.csv")
    L = [r"\begin{table}[t]", r"\centering",
         T(r"\caption{Sensitivity to the initial snapshot: mean test Dice over all 21 snapshots, at the random initial "
           r"snapshot $n=0$ only and over the 20 snapshots $n\geq50$.}",
           r"\caption{Sensibilidade ao instante inicial: Dice médio no teste nos 21 instantes, apenas no instante inicial "
           r"aleatório $n=0$ e nos 20 instantes $n\geq50$.}"),
         r"\label{tab:sensitivity}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabularx}{\columnwidth}{L{0.7cm} Y c c c}", r"\toprule",
         r"\textbf{ID} & " + T(r"\textbf{Configuration} & \textbf{All}", r"\textbf{Configuração} & \textbf{Todos}")
         + r" & \textbf{$n=0$} & \textbf{$n\geq50$} \\", r"\midrule"]
    prev = None
    for r in rows:
        prev = sep(L, prev, r)
        x = pt[(pt.arch == r.arch) & (pt.pre == r.pre) & (pt.post == r.post)]
        L.append(f"{r.exp_id} & {N('ARCH')[r.arch]} ({r.pre}+{r.post}) & {nf(x.Dice_all.mean())} & "
                 f"{nf(x.Dice_n0.mean())} & {nf(x.Dice_sem_n0.mean())}" + r" \\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    save(L, "tab_sensitivity")


def tab_regime(rows):
    pr = pd.read_csv(RES / "per_regime.csv")
    L = [r"\begin{table}[t]", r"\centering",
         T(r"\caption{Test Dice per pattern regime (20 test trajectories each) for each architecture alone, for its "
           r"validation-selected pre- + post-processing pair and for the learning-free reference.}",
           r"\caption{Dice no teste por regime de padrão (20 trajetórias de teste cada) para cada arquitetura sozinha, "
           r"para seu par de pré e pós-processamento escolhido na validação e para a referência sem aprendizado.}"),
         r"\label{tab:regime}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabularx}{\columnwidth}{Y *{6}{c}}", r"\toprule",
         T(r"\textbf{Configuration}", r"\textbf{Configuração}") + " & "
         + " & ".join(f"\\textbf{{{x}}}" for x in N("REG")) + r" \\", r"\midrule"]
    for r in rows:
        if r.condition not in ("No pre/post", "Pre+Post", "Reference"):
            continue
        x = pr[(pr.arch == r.arch) & (pr.pre == r.pre) & (pr.post == r.post)].set_index("regime")
        L.append(f"{N('ARCH')[r.arch]} ({r.pre}+{r.post}) & " + " & ".join(nf(x.loc[g, 'Dice']) for g in REGS) + r" \\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    save(L, "tab_regime")


def tab_cost(g):
    L = [r"\begin{table}[t]", r"\centering",
         T(r"\caption{Model size and cost (20 epochs, batch 32, native $128\times128$ input, reference input P0). Best "
           r"epoch: epoch of the checkpoint with the highest validation Dice, range over the 11 inputs P0 to P10. "
           r"Latency and throughput measured with a dedicated GPU after all training runs.}",
           r"\caption{Tamanho e custo dos modelos (20 épocas, lote 32, entrada nativa de $128\times128$, entrada de "
           r"referência P0). Melhor época: época do ponto de verificação com maior Dice de validação, intervalo entre as "
           r"11 entradas P0 a P10. Latência e vazão medidas com a GPU dedicada após todos os treinos.}"),
         r"\label{tab:cost}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabularx}{\columnwidth}{Y c c c c c}", r"\toprule",
         T(r"\textbf{Architecture} & \textbf{Params (M)} & \textbf{Train (min)} & \textbf{Best epoch} & "
           r"\textbf{Latency (ms)} & \textbf{fps} \\",
           r"\textbf{Arquitetura} & \textbf{Parâm. (M)} & \textbf{Treino (min)} & \textbf{Melhor época} & "
           r"\textbf{Latência (ms)} & \textbf{qps} \\"), r"\midrule"]
    t = g[(g.split == "test") & (g.post == "Q0")]
    for a in ARCHS:
        r = t[(t.arch == a) & (t.pre == "P0")].iloc[0]
        be = t[t.arch == a].best_epoch
        sepw = T(" to ", " a ")
        L.append(f"{N('ARCH')[a]} & {nf(r.params_M, 1)} & {r.train_time_min:.0f} & {be.min():.0f}{sepw}{be.max():.0f} & "
                 f"{nf(r.latency_ms_b1, 2)} & {r.throughput_fps:.0f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    save(L, "tab_cost")


def build(lang):
    global LANG, TAB
    LANG = lang
    TAB = OUT / ("tables" if lang == "en" else "tables_pt")
    fig = OUT / ("figures" if lang == "en" else "figures_pt")
    src = ROOT / ("paper/figures" if lang == "en" else "artigo_sbc/figuras")
    TAB.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    for f in FIGS:
        shutil.copy2(src / f"{f}.pdf", fig / f"{f}.pdf")
    g = pd.read_csv(RES / "grid.csv")
    rows = selected(g)
    tab_full_results(rows)
    tab_architecture_comparison(rows)
    tab_prepost_effect({"n": 2000, "seed": 42})
    info = tab_best_and_cross(rows, g)
    tab_classwise_temporal(rows)
    tab_sensitivity(rows)
    tab_regime(rows)
    tab_cost(g)
    return info


def main():
    shutil.copy2(ROOT / "paper" / "references.bib", OUT / "references.bib")
    for lang in ("en", "pt"):
        print(lang, build(lang))


if __name__ == "__main__":
    main()
