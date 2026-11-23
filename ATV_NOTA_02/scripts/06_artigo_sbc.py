"""Tabelas e figuras do artigo SBC (portugues, virgula decimal) a partir de outputs/results/.

Reutiliza as funcoes de figura de 05_report.py, trocando apenas os textos (TXT) e o separador
decimal (DEC). As tabelas sao escritas aqui com legendas em portugues e seguem o formato
sequencial padronizado da turma (Exp. | Pre | Arquitetura | Pos | IoU | Dice | Precisao | Revocacao | Latencia).

Uso:  python scripts/06_artigo_sbc.py          -> artigo_sbc/tabelas e artigo_sbc/figuras
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gsseg.metrics import bootstrap_ci, frame_scores  # noqa: E402
from gsseg.utils import data_dir, load_config, load_json  # noqa: E402

spec = importlib.util.spec_from_file_location("rep", ROOT / "scripts" / "05_report.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

ARCHS = ["unet", "segformer", "mask2former"]
ARCH = {"unet": "U-Net", "segformer": "SegFormer", "mask2former": "Mask2Former", "otsu_A": "Otsu em $A$"}
COND = {"No pre/post": "Arquitetura", "Pre": "Pré + arquitetura", "Post": "Arquitetura + pós",
        "Pre+Post": "Pré + arquitetura + pós"}
PRE_PT = {
    "P0": ("Baseline (campo bruto)", "concentração adimensional $A$, $128\times128$, sem filtro"),
    "P1": ("Normalização \textit{z-score}", "média e desvio padrão do treino"),
    "P2": ("Normalização RMS", "$x/\mathrm{RMS}$, com $\mathrm{RMS}=\sqrt{\overline{x^2}}$ do treino"),
    "P3": ("Escalonamento robusto", "mediana e intervalo interquartil do treino"),
    "P4": ("Normalização mín--máx", "mínimo e máximo do treino"),
    "P5": ("Recorte por percentis", "recorte em $[p_1,p_{99}]$ do treino, sem reescala"),
    "P6": ("Suavização gaussiana", "$\sigma=1$ px, contorno periódico"),
    "P7": ("Filtro de mediana", "janela $3\times3$, contorno periódico"),
    "P8": ("Filtro bilateral", "$d=5$, $\sigma_r=0{,}1$, $\sigma_s=2$, contorno periódico"),
    "P9": ("Difusão anisotrópica", "Perona--Malik, 15 iterações, $\kappa=0{,}05$, $\gamma=0{,}2$"),
    "P10": ("Remoção de ruído por \textit{wavelets}", "Daubechies-4, BayesShrink, 3 níveis"),
}
POST_PT = {
    "Q0": ("Baseline (sem pós)", "decisão $P>0{,}5$"),
    "Q1": ("Otimização do limiar", "limiar em $\{0{,}05,\dots,0{,}95\}$, na validação"),
    "Q2": ("Abertura morfológica", "disco de raio $r\in\{1,2\}$, na validação"),
    "Q3": ("Fechamento morfológico", "disco de raio $r\in\{1,2\}$, na validação"),
    "Q4": ("Filtragem de componentes conexos", "área mínima em $\{2,\dots,32\}$ px, na validação"),
    "Q5": ("Filtragem de consistência temporal", "mediana de $w\in\{3,5\}$ instantes amostrados, na validação"),
}
TXT_PT = {
    "n": "n", "concB": "concentração de $B$", "inputA": "Entrada $A$", "preinput": "Entrada\npré-processada",
    "gt": "Referência\n(GT)", "c_none": "Arquitetura", "c_pre": "Pré +\narquitetura", "c_post": "Arquitetura\n+ pós",
    "c_both": "Pré + arq.\n+ pós", "TP": "VP", "FP": "FP", "FN": "FN",
    "post_axis": "pós-processamento", "pre_axis": "pré-processamento", "test_dice": "Dice no teste",
    "otsu": "Otsu em $A$", "gt_line": "Referência (GT)", "snapshot": "instante n", "mean_dice": "Dice médio (teste)",
    "area": "área segmentada (% do domínio)", "leg_title": "contínua: config. final; pontilhada: só arquitetura",
    "eff_pre": "Pré-processamento + arquitetura (Q0)", "eff_post": "Arquitetura + pós-processamento (P0)",
    "epoch": "época", "val_dice": "Dice na validação (P0)",
}


def f(x, nd=3):
    return f"{x:.{nd}f}".replace(".", "{,}")


def ci(r, k, nd=3):
    hw = (r[f"{k}_hi"] - r[f"{k}_lo"]) / 2
    return f"{f(r[k], nd)}\\,{{\\scriptsize$\\pm${f(hw, nd)}}}"


def save(lines, path: Path):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_open():
    return [r"\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{sbcfitwidth}"]


def tab_dados(cfg, out):
    proc = data_dir(cfg) / "processed"
    Z = {s: dict(np.load(proc / f"{s}.npz")) for s in ("train", "valid", "test")}  # carrega uma vez
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Subconjunto do \texttt{gray\_scott\_reaction\_diffusion} utilizado: parâmetros de cada regime, "
         r"trajetórias por partição (todas as das partições oficiais), limiar médio $\bar\tau$ do GT, fração média de "
         r"pixels de estrutura (fg), trajetórias extintas em $n=500$, comprimento de onda dominante $\lambda$ de $B$ e "
         r"diâmetro equivalente mediano $d_{eq}$ das estruturas do GT em $n=500$ (treino).}",
         r"\label{tab:dados}"] + fit_open() + [r"\begin{tabular}{lcccccccc}", r"\toprule",
         r"Regime & $F$ & $k$ & Treino/val./teste & $\bar\tau$ & fg (\%) & Extintas & $\lambda$ (px) & $d_{eq}$ (px) \\",
         r"\midrule"]
    for i, r in enumerate(list(Z["train"]["regimes"])):
        n = [int((Z[s]["regime"] == i).sum()) for s in Z]
        sel = Z["train"]["regime"] == i
        F, k = Z["train"]["F"][sel][0], Z["train"]["k"][sel][0]
        tau = np.concatenate([Z[s]["tau"][Z[s]["regime"] == i] for s in Z]).mean()
        fg = np.concatenate([Z[s]["gt"][Z[s]["regime"] == i].mean((1, 2, 3)) for s in Z]).mean() * 100
        ext = sum(int((~Z[s]["tau_valid"][Z[s]["regime"] == i]).sum()) for s in Z)
        lam, deq = rep._scales(Z["train"], i)
        L.append(f"{r.capitalize()} & {f(F)} & {f(k)} & {n[0]}/{n[1]}/{n[2]} & {f(tau)} & {f(fg, 1)} & {ext} & "
                 f"{f(lam, 1)} & {f(deq, 1)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{sbcfitwidth}", r"\end{table}"]
    save(L, out / "tab_dados.tex")


def tab_metodos(out):
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Códigos dos métodos de pré-processamento (P) e de pós-processamento (Q) usados nas tabelas. "
         r"As estatísticas de P1--P5 vêm do treino; os parâmetros indicados como escolhidos na validação "
         r"maximizam o Dice médio nesse conjunto.}",
         r"\label{tab:metodos}"] + fit_open() + [r"\begin{tabular}{lll}", r"\toprule",
         r"Código & Método & Parâmetros \\", r"\midrule"]
    for k, (nome, par) in PRE_PT.items():
        L.append(f"{k} & {nome} & {par}" + r" \\")
    L.append(r"\midrule")
    for k, (nome, par) in POST_PT.items():
        L.append(f"{k} & {nome} & {par}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{sbcfitwidth}", r"\end{table}"]
    save(L, out / "tab_metodos.tex")


def tab_principal(rows, g, out):
    best = {k: max(r[4][k] for r in rows) for k in ["IoU", "Dice", "Precision", "Recall"]}
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Resultados no teste (120 trajetórias, 2.520 quadros) nas quatro condições do experimento. Valores "
         r"médios por trajetória, com a meia-amplitude do intervalo de confiança de 95\% por \textit{bootstrap} "
         r"estratificado por regime. O pré e o pós-processamento de cada condição foram escolhidos pelo Dice médio "
         r"na validação. Latência: rede + pós-processamento por quadro de $128\times128$, lote 1, RTX A5000. "
         r"Em negrito, o maior valor de cada coluna.}",
         r"\label{tab:principal}"] + fit_open() + [r"\begin{tabular}{lllllccccc}", r"\toprule",
         r"Exp. & Condição & Pré & Arquitetura & Pós & IoU & Dice & Precisão & Revocação & Latência (ms) \\",
         r"\midrule"]
    prev = None
    for a, c, p, q, r in rows:
        if prev and prev != a:
            L.append(r"\midrule")
        prev = a
        cells = []
        for k in ["IoU", "Dice", "Precision", "Recall"]:
            s = ci(r, k)
            cells.append(f"\\textbf{{{s}}}" if np.isclose(r[k], best[k]) else s)
        lat = r.latency_ms_b1 + r.post_ms_per_frame
        L.append(f"{r.exp_id} & {COND[c]} & {p} & {ARCH[a]} & {q} & " + " & ".join(cells) + f" & {f(lat, 2)}" + r" \\")
    b = g[(g.arch == "otsu_A") & (g.split == "test")].iloc[0]
    L += [r"\midrule", "REF & Referência clássica & -- & Otsu em $A$ & -- & " +
          " & ".join(ci(b, k) for k in ["IoU", "Dice", "Precision", "Recall"]) + f" & {f(b.latency_ms_b1, 2)}" + r" \\",
          r"\bottomrule", r"\end{tabular}", r"\end{sbcfitwidth}", r"\end{table}"]
    save(L, out / "tab_principal.tex")


def tab_temporal(rows, g, tc_gt, out):
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Métricas temporais no teste para as configurações da Tabela~\ref{tab:principal}. AVE: erro de "
         r"variação de área (\% do domínio por instante amostrado); CDE: erro de deslocamento dos centroides das "
         r"estruturas (px, distância periódica); TC: consistência temporal das máscaras, com o valor do GT na última "
         r"linha; TCE $=|TC_{pred}-TC_{GT}|$. Média $\pm$ meia-amplitude do IC de 95\%.}",
         r"\label{tab:temporal}"] + fit_open() + [r"\begin{tabular}{lllcccc}", r"\toprule",
         r"Exp. & Arquitetura & Condição & AVE (\%) & CDE (px) & TC & TCE \\", r"\midrule"]
    prev = None
    for a, c, p, q, r in rows:
        if prev and prev != a:
            L.append(r"\midrule")
        prev = a
        L.append(f"{r.exp_id} & {ARCH[a]} & {COND[c]} & {ci(r, 'AVE')} & {ci(r, 'CDE', 2)} & {f(r.TC)} & "
                 f"{ci(r, 'TCE')}" + r" \\")
    b = g[(g.arch == "otsu_A") & (g.split == "test")].iloc[0]
    L += [r"\midrule", f"REF & Otsu em $A$ & -- & {ci(b, 'AVE')} & {ci(b, 'CDE', 2)} & {f(b.TC)} & {ci(b, 'TCE')}" + r" \\",
          f"GT & Referência & -- & -- & -- & {f(tc_gt)} & --" + r" \\",
          r"\bottomrule", r"\end{tabular}", r"\end{sbcfitwidth}", r"\end{table}"]
    save(L, out / "tab_temporal.tex")


def tab_efeitos(pt, cfg, out):
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Efeito de cada técnica isolada: diferença pareada do Dice no teste (em pontos percentuais) em relação "
         r"à arquitetura sem pré e pós-processamento (P0+Q0), calculada nas mesmas 120 trajetórias, com IC de 95\% por "
         r"\textit{bootstrap} estratificado por regime. Em negrito, intervalos que não contêm zero; "
         r"$\blacktriangle$ indica aumento e $\blacktriangledown$, redução do Dice. Valores com duas casas; efeitos com "
         r"magnitude inferior a 0,005 ponto aparecem como $\pm$0,00, ainda que seu intervalo exclua zero.}",
         r"\label{tab:efeitos}"] + fit_open() + [r"\begin{tabular}{llccc}", r"\toprule",
         r"Código & Técnica & U-Net & SegFormer & Mask2Former \\", r"\midrule"]
    items = [(p, "Q0") for p in list(PRE_PT)[1:]] + [("P0", q) for q in list(POST_PT)[1:]]
    for idx, (p, q) in enumerate(items):
        if idx == len(PRE_PT) - 1:
            L.append(r"\midrule")
        code = p if q == "Q0" else q
        nome = PRE_PT[p][0] if q == "Q0" else POST_PT[q][0]
        cells = []
        for a in ARCHS:
            m, lo, hi = (100 * x for x in rep.paired_delta(pt, cfg, a, p, q))
            sig = lo > 0 or hi < 0
            seta = (r"$\blacktriangle$" if m > 0 else r"$\blacktriangledown$") if sig else ""
            sf = lambda x: f"{x:+.2f}".replace(".", "{,}").replace("-", "$-$")  # noqa: E731
            s = f"{sf(m)} [{sf(lo)}; {sf(hi)}]"
            cells.append(f"\\textbf{{{s}}}{seta}" if sig else s)
        L.append(f"{code} & {nome} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{sbcfitwidth}", r"\end{table}"]
    save(L, out / "tab_efeitos.tex")


def tab_regime(reg, rows, out):
    regimes = list(dict.fromkeys(reg.regime))
    fin = rep.final_cfg(rows)
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Dice no teste por regime para cada arquitetura sozinha (P0+Q0), para a configuração final de cada "
         r"arquitetura (pré + arquitetura + pós) e para a referência clássica. Vinte trajetórias por regime.}",
         r"\label{tab:regime}"] + fit_open() + [r"\begin{tabular}{l" + "c" * len(regimes) + "}", r"\toprule",
         "Configuração & " + " & ".join(x.capitalize() for x in regimes) + r" \\", r"\midrule"]
    for a in ARCHS:
        for p, q in dict.fromkeys([("P0", "Q0"), fin[a]]):
            d = reg[(reg.arch == a) & (reg.pre == p) & (reg.post == q)].set_index("regime")
            L.append(f"{rep.exp_id(a, p, q)} {ARCH[a]} ({p}+{q}) & " + " & ".join(f(d.loc[x, 'Dice']) for x in regimes) + r" \\")
    d = reg[reg.arch == "otsu_A"].set_index("regime")
    L += [r"\midrule", "REF Otsu em $A$ & " + " & ".join(f(d.loc[x, 'Dice']) for x in regimes) + r" \\",
          r"\bottomrule", r"\end{tabular}", r"\end{sbcfitwidth}", r"\end{table}"]
    save(L, out / "tab_regime.tex")


def tab_custo(meta, out):
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Tamanho e custo das arquiteturas na entrada de referência (P0): parâmetros, tempo de treino de 20 "
         r"épocas com até cinco treinos simultâneos na mesma GPU, época selecionada na validação, memória máxima de GPU "
         r"no treino (lote 32) e na inferência (lote 1), latência (lote 1) e vazão (lote 32), estas medidas com a GPU "
         r"dedicada. NVIDIA RTX A5000.}",
         r"\label{tab:custo}"] + fit_open() + [r"\begin{tabular}{lccccccc}", r"\toprule",
         r"Arquitetura & Parâm. (M) & Treino (min) & Época & Mem. treino (MiB) & Mem. inf. (MiB) & Latência (ms) & "
         r"Vazão (quadros/s) \\", r"\midrule"]
    for a in ARCHS:
        m = meta[f"{a}_P0"]
        L.append(f"{ARCH[a]} & {f(m['params'] / 1e6, 1)} & {f(m['train_time_s'] / 60, 1)} & {m['best_epoch']}/20 & "
                 f"{m['peak_train_mem_mib']:.0f} & {m['peak_infer_mem_mib_b1']:.0f} & {f(m['latency_ms_b1'], 2)} & "
                 f"{m['throughput_fps']:.0f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{sbcfitwidth}", r"\end{table}"]
    save(L, out / "tab_custo.tex")


def tab_classes(rows, g, out):
    """Metricas por classe (protocolo, Secao 10): estrutura, fundo e media (mIoU)."""
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Métricas por classe no teste para as configurações da Tabela~\ref{tab:principal}: IoU da classe "
         r"estrutura, IoU e Dice da classe fundo e mIoU, média do IoU das duas classes. Média $\pm$ meia-amplitude "
         r"do IC de 95\%.}",
         r"\label{tab:classes}"] + fit_open() + [r"\begin{tabular}{lllcccc}", r"\toprule",
         r"Exp. & Arquitetura & Condição & IoU estrutura & IoU fundo & Dice fundo & mIoU \\", r"\midrule"]
    prev = None
    for a, c, p, q, r in rows:
        if prev and prev != a:
            L.append(r"\midrule")
        prev = a
        L.append(f"{r.exp_id} & {ARCH[a]} & {COND[c]} & {ci(r, 'IoU')} & {ci(r, 'IoU_bg')} & {ci(r, 'Dice_bg')} & "
                 f"{ci(r, 'mIoU')}" + r" \\")
    b = g[(g.arch == "otsu_A") & (g.split == "test")].iloc[0]
    L += [r"\midrule", f"REF & Otsu em $A$ & -- & {ci(b, 'IoU')} & {ci(b, 'IoU_bg')} & {ci(b, 'Dice_bg')} & "
          f"{ci(b, 'mIoU')}" + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{sbcfitwidth}", r"\end{table}"]
    save(L, out / "tab_classes.tex")


def tab_sensibilidade(rows, cfg, out, rdir):
    """Dice com e sem o instante inicial n = 0 (sensibilidade), mesmas configuracoes da tabela principal."""
    t = pd.read_csv(rdir / "per_traj_sem_n0.csv")
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Sensibilidade ao instante inicial: Dice médio no teste em todos os 21 instantes, apenas em $n=0$ "
         r"e nos 20 instantes $n\geq50$, para as configurações selecionadas na validação. Média por trajetória, com "
         r"IC de 95\% por \textit{bootstrap} estratificado por regime para $n\geq50$.}",
         r"\label{tab:sensibilidade}"] + fit_open() + [r"\begin{tabular}{lllccc}", r"\toprule",
         r"Exp. & Arquitetura & Condição & Dice (todos) & Dice ($n=0$) & Dice ($n\geq50$) \\", r"\midrule"]
    prev = None
    for a, c, p, q, r in rows + [("otsu_A", "Reference", "P0", "Q0", None)]:
        if prev and prev != a:
            L.append(r"\midrule")
        prev = a
        x = t[(t.arch == a) & (t.pre == p) & (t.post == q)]
        m, lo, hi = bootstrap_ci(x.Dice_sem_n0.values, cfg["eval"]["bootstrap_samples"], seed=cfg["seed"],
                                 groups=x.regime.values)
        eid = r.exp_id if r is not None else "REF"
        cond = COND.get(c, "Referência clássica")
        L.append(f"{eid} & {ARCH[a]} & {cond} & {f(x.Dice_all.mean())} & {f(x.Dice_n0.mean())} & "
                 f"{f(m)}\\,{{\\scriptsize$\\pm${f((hi - lo) / 2)}}}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{sbcfitwidth}", r"\end{table}"]
    save(L, out / "tab_sensibilidade.tex")


def tab_sequencial(g, out, rdir):
    t = g[g.split == "test"].sort_values("exp_id")
    t = pd.concat([t[t.exp_id != "REF"], t[t.exp_id == "REF"]])
    L = [r"\begingroup\footnotesize", r"\setlength{\tabcolsep}{3.2pt}",
         r"\begin{longtable}{lllllccccccc}",
         r"\caption{Resultados sequenciais e padronizados de todas as combinações no teste. E001--E150: P1--P10 "
         r"$\times$ arquitetura $\times$ Q1--Q5, matriz do protocolo da turma; E151--E198: linhas de \textit{baseline} "
         r"sem pré e/ou sem pós-processamento; REF: Otsu em $A$. AVE em \% do domínio, CDE em px e latência em ms por quadro.}"
         r"\label{tab:sequencial}\\",
         r"\toprule", r"Exp. & Pré & Arquitetura & Pós & IoU & Dice & Prec. & Rev. & AVE & CDE & TCE & Lat. \\",
         r"\midrule", r"\endfirsthead", r"\toprule",
         r"Exp. & Pré & Arquitetura & Pós & IoU & Dice & Prec. & Rev. & AVE & CDE & TCE & Lat. \\", r"\midrule",
         r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, r in t.iterrows():
        L.append(f"{r.exp_id} & {r.pre} & {ARCH.get(r.arch)} & {r.post} & {f(r.IoU, 4)} & {f(r.Dice, 4)} & "
                 f"{f(r.Precision, 4)} & {f(r.Recall, 4)} & {f(r.AVE)} & {f(r.CDE, 2)} & {f(r.TCE, 4)} & "
                 f"{f(r.latency_ms_b1 + r.post_ms_per_frame, 2)}" + r" \\")
    L += [r"\end{longtable}", r"\endgroup"]
    save(L, out / "tab_sequencial.tex")


def main():
    cfg = load_config()
    rdir = ROOT / "outputs" / "results"
    out = ROOT / "artigo_sbc"
    out_tab, out_fig = out / "tabelas", out / "figuras"
    out_tab.mkdir(parents=True, exist_ok=True); out_fig.mkdir(parents=True, exist_ok=True)
    g = pd.read_csv(rdir / "grid.csv")
    reg = pd.read_csv(rdir / "per_regime.csv")
    ptime = pd.read_csv(rdir / "per_time.csv")
    ptraj = pd.read_csv(rdir / "per_traj.csv")
    params = load_json(rdir / "post_params.json")
    meta = {}
    for d in (ROOT / "outputs" / "models").iterdir():
        if (d / "meta.json").exists() and "_smoke" not in d.name:
            m = load_json(d / "meta.json")
            meta[f"{m['arch']}_{m['pre']}"] = m
    zt = np.load(data_dir(cfg) / "processed" / "test.npz")
    tc_gt = float(frame_scores(zt["gt"][:, 1:], zt["gt"][:, :-1])["IoU"].mean())
    rows = rep.selected(g)

    tab_dados(cfg, out_tab); tab_metodos(out_tab); tab_principal(rows, g, out_tab)
    tab_temporal(rows, g, tc_gt, out_tab); tab_efeitos(ptraj, cfg, out_tab); tab_regime(reg, rows, out_tab)
    tab_custo(meta, out_tab); tab_sequencial(g, out_tab, rdir)
    tab_classes(rows, g, out_tab); tab_sensibilidade(rows, cfg, out_tab, rdir)

    rep.TXT, rep.DEC = TXT_PT, ","
    rep.fig_overview(cfg, out_fig)
    rep.fig_visual_tables(cfg, rows, params, "", out_fig)
    rep.fig_heatmaps(g, out_fig)
    rep.fig_temporal(ptime, rows, out_fig)
    rep.fig_effects(g, out_fig)
    rep.fig_training("", out_fig)
    print("tabelas e figuras do artigo SBC em", out)


if __name__ == "__main__":
    main()
