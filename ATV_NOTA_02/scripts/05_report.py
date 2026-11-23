"""Gera tabelas (CSV + LaTeX) e figuras do artigo a partir de outputs/results*/.

Condicoes do experimento (selecionadas na VALIDACAO, reportadas no TESTE), por arquitetura:
  No pre/post : P0 + arquitetura + Q0          (arquitetura implementada, sem pre e sem pos)
  Pre         : melhor P1..P10 + arquitetura + Q0
  Post        : P0 + arquitetura + melhor Q1..Q5
  Pre+Post    : melhor par (P1..P10, Q1..Q5)  (pre + arquitetura + pos)

Tabelas de metricas em formato sequencial e padronizado (prompt da turma):
  Experimento | Pre-processamento | Arquitetura | Pos-processamento | IoU | Dice | Precision | Recall | Latencia
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gsseg.experiments import CONDITION_PT, exp_id  # noqa: E402
from gsseg.metrics import bootstrap_ci, frame_scores  # noqa: E402
from gsseg.models import ARCH_NAMES  # noqa: E402
from gsseg.postprocess import POST_NAMES, POST_PARAMS, apply_post  # noqa: E402
from gsseg.preprocess import PRE_NAMES, PRE_PARAMS  # noqa: E402
from gsseg.utils import ROOT, data_dir, load_config, load_json  # noqa: E402

FLUID = "gray_scott_reaction_diffusion"
ARCHS = ["unet", "segformer", "mask2former"]
COLORS = {"unet": "#2a78d6", "segformer": "#eb6834", "mask2former": "#1baf7a", "otsu_A": "#52514e"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_B = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
CMAP_DICE = LinearSegmentedColormap.from_list("blue_seq_dice", BLUE_RAMP[1:])
CMAP_GT = ListedColormap(["#ffffff", "#0d366b"])
TP_C, FP_C, FN_C = (0.30, 0.30, 0.29), "#eb6834", "#2a78d6"
CONDS = ["No pre/post", "Pre", "Post", "Pre+Post"]
COND_TEX = {"No pre/post": "Architecture only", "Pre": "Pre + arch.", "Post": "Arch. + post",
            "Pre+Post": "Pre + arch. + post"}
PRIORITY = ["Pre+Post", "Post", "Pre", "No pre/post"]
METRICS = ["IoU", "Dice", "Precision", "Recall", "IoU_bg", "Dice_bg", "mIoU", "AVE", "CDE", "TC", "TCE"]
PRES = list(PRE_NAMES)    # P0 (baseline) .. P10
POSTS = list(POST_NAMES)  # Q0 (baseline) .. Q5

# Textos das figuras (ingles por padrao). scripts/06_artigo_sbc.py troca TXT e DEC para portugues.
TXT = {
    "n": "n", "concB": "concentration $B$", "inputA": "Input $A$", "preinput": "Pre-processed\ninput",
    "gt": "Ground\ntruth", "c_none": "Architecture\nonly", "c_pre": "Pre +\narchitecture",
    "c_post": "Architecture\n+ post", "c_both": "Pre + arch.\n+ post", "TP": "TP", "FP": "FP", "FN": "FN",
    "post_axis": "post-processing", "pre_axis": "pre-processing", "test_dice": "test Dice",
    "otsu": "Otsu on $A$", "gt_line": "Ground truth", "snapshot": "snapshot n", "mean_dice": "mean Dice (test)",
    "area": "segmented area (% of domain)", "leg_title": "solid: final config.; dotted: arch. only",
    "eff_pre": "Pre-processing + architecture (Q0)", "eff_post": "Architecture + post-processing (P0)",
    "epoch": "epoch", "val_dice": "validation Dice (P0)",
}
DEC = "."


def fnum(x, nd=3):
    return f"{x:.{nd}f}".replace(".", DEC)


def _dec_axes(*axes):
    """Aplica o separador decimal DEC aos rotulos numericos dos eixos."""
    if DEC == ".":
        return
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", DEC))
    for ax in axes:
        ax.yaxis.set_major_formatter(f)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


# ----------------------------------------------------------------------------- selecao
def pick(g: pd.DataFrame, arch: str) -> dict:
    v = g[(g.arch == arch) & (g.split == "valid")]
    sel = {
        "No pre/post": v[(v.pre == "P0") & (v.post == "Q0")],
        "Pre": v[(v.pre != "P0") & (v.post == "Q0")],
        "Post": v[(v.pre == "P0") & (v.post != "Q0")],
        "Pre+Post": v[(v.pre != "P0") & (v.post != "Q0")],
    }
    out = {}
    for c in CONDS:
        d = sel[c]
        if len(d):
            r = d.loc[d.Dice.idxmax()]
            out[c] = (r.pre, r.post)
    return out


def test_row(g, arch, pre, post):
    return g[(g.arch == arch) & (g.pre == pre) & (g.post == post) & (g.split == "test")].iloc[0]


def selected(g) -> list:
    """[(arch, cond, pre, post, test_row)] na ordem arquitetura -> condicao."""
    rows = []
    for a in ARCHS:
        if (g.arch == a).any():
            for c, (p, q) in pick(g, a).items():
                rows.append((a, c, p, q, test_row(g, a, p, q)))
    return rows


def final_cfg(rows) -> dict:
    """Configuracao final por arquitetura (Pre+Post ou a condicao mais completa disponivel)."""
    out = {}
    for a in dict.fromkeys(r[0] for r in rows):
        conds = {c: (p, q) for (aa, c, p, q, _r) in rows if aa == a}
        out[a] = conds[next(c for c in PRIORITY if c in conds)]
    return out


# ----------------------------------------------------------------------------- utilidades LaTeX
def ci(r, k, nd=3):
    hw = (r[f"{k}_hi"] - r[f"{k}_lo"]) / 2
    return f"{r[k]:.{nd}f}{{\\scriptsize$\\pm${hw:.{nd}f}}}"


def esc(x: str) -> str:
    return str(x).replace("{", r"\{").replace("}", r"\}").replace("^2", r"$^2$").replace(" < ", r" $<$ ").replace("%", r"\%").replace("_", r"\_").replace(">=", r"$\geq$").replace("> ", r"$>$ ")


def save_tex(lines, path: Path):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- tabelas
def tab_methods(out_tab):
    L = [r"\begin{table}[h!]", r"\centering",
         r"\caption{Codes of the pre-processing (P) and post-processing (Q) methods used in all tables. Statistics of "
         r"P1--P5 come from the training split; parameters marked ``validation'' are selected on the validation split.}",
         r"\label{tab:methods}", r"\scriptsize",
         r"\begin{tabular}{@{}lp{0.34\linewidth}p{0.56\linewidth}@{}}", r"\toprule",
         r"Code & Method & Parameters \\", r"\midrule"]
    for k, v in PRE_NAMES.items():
        par = esc(PRE_PARAMS[k]) if k in PRE_PARAMS else r"raw substrate field $A$, native $128\times128$"
        L.append(f"{k} & {esc(v)} & {par}" + r" \\")
    L.append(r"\midrule")
    for k, v in POST_NAMES.items():
        L.append(f"{k} & {esc(v)} & {esc(POST_PARAMS.get(k, ''))}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    save_tex(L, out_tab / "tab_methods.tex")


def tab_main(rows, g, out_tab, out_csv):
    """Tabela principal: 4 condicoes x 3 arquiteturas, formato sequencial padronizado."""
    std = []
    best = {k: max(r[4][k] for r in rows) for k in ["IoU", "Dice", "Precision", "Recall"]}
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Main results on the test set (\texttt{gray\_scott\_reaction\_diffusion}): the three implemented "
         r"architectures alone, with pre-processing, with post-processing, and with pre-processing + post-processing. "
         r"Mean over test trajectories with the half-width of the 95\% bootstrap CI (stratified by regime). Pre- and "
         r"post-processing of each condition were selected on the validation split by mean Dice. Latency: network + "
         r"post-processing per $128\times128$ frame (batch 1). Best value per column in bold. Codes in Table~\ref{tab:methods}.}",
         r"\label{tab:main}", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}lllllccccc@{}}",
         r"\toprule",
         r"Exp. & Condition & Pre-proc. & Architecture & Post-proc. & IoU $\uparrow$ & Dice $\uparrow$ & "
         r"Precision $\uparrow$ & Recall $\uparrow$ & Latency (ms) $\downarrow$ \\", r"\midrule"]
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
        L.append(f"{r.exp_id} & {COND_TEX[c]} & {p} & {ARCH_NAMES[a]} & {q} & " + " & ".join(cells) + f" & {lat:.2f}" + r" \\")
        std.append(_std_row(r, a, c, p, q))
    b = g[(g.arch == "otsu_A") & (g.split == "test")]
    if len(b):
        r = b.iloc[0]
        L += [r"\midrule", "REF & Classical reference & -- & Otsu on $A$ & -- & " +
              " & ".join(ci(r, k) for k in ["IoU", "Dice", "Precision", "Recall"]) + f" & {r.latency_ms_b1:.2f}" + r" \\"]
        std.append(_std_row(r, "otsu_A", "Reference", "P0", "Q0"))
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    save_tex(L, out_tab / "tab_main.tex")
    pd.DataFrame(std).to_csv(out_csv / "results_standard.csv", index=False)


def _std_row(r, a, c, p, q):
    return {"Experiment": r.exp_id, "Fluid": FLUID, "Condition": c, "Condicao": CONDITION_PT.get(c, "Referência clássica"),
            "Pre": p, "Pre_name": PRE_NAMES[p], "Architecture": ARCH_NAMES.get(a, "Otsu on A (no learning)"),
            "Post": q, "Post_name": POST_NAMES[q],
            **{k: r[k] for k in METRICS},
            **{f"{k}_CI95_low": r[f"{k}_lo"] for k in METRICS}, **{f"{k}_CI95_high": r[f"{k}_hi"] for k in METRICS},
            "Latency_net_ms": r.latency_ms_b1, "Latency_post_ms": r.post_ms_per_frame,
            "Latency_total_ms": r.latency_ms_b1 + r.post_ms_per_frame, "Peak_infer_mem_MiB": r.peak_infer_mem_mib_b1,
            "Peak_train_mem_MiB": r.peak_train_mem_mib, "Params_M": r.params_M, "Train_time_min": r.train_time_min}


def tab_temporal(rows, g, tc_gt, out_tab):
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Temporal and memory metrics on the test set for the configurations of Table~\ref{tab:main}. "
         r"AVE: area-variation error (\% of the domain per sampled step); CDE: structure-centroid displacement error "
         r"(pixels, periodic distance); TC: temporal consistency of the masks (ground-truth value in the last row); "
         r"TCE $=|TC_{pred}-TC_{GT}|$; memory: peak GPU memory at inference (batch 1).}",
         r"\label{tab:temporal}", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}lllcccccc@{}}", r"\toprule",
         r"Exp. & Architecture & Condition & AVE (\%) $\downarrow$ & CDE (px) $\downarrow$ & TC & TCE $\downarrow$ & "
         r"Memory (MiB) $\downarrow$ \\", r"\midrule"]
    prev = None
    for a, c, p, q, r in rows:
        if prev and prev != a:
            L.append(r"\midrule")
        prev = a
        L.append(f"{r.exp_id} & {ARCH_NAMES[a]} & {COND_TEX[c]} & {ci(r, 'AVE')} & {ci(r, 'CDE', 2)} & {r.TC:.3f} & "
                 f"{ci(r, 'TCE')} & {r.peak_infer_mem_mib_b1:.0f}" + r" \\")
    b = g[(g.arch == "otsu_A") & (g.split == "test")]
    if len(b):
        r = b.iloc[0]
        L += [r"\midrule", f"REF & Otsu on $A$ & -- & {ci(r, 'AVE')} & {ci(r, 'CDE', 2)} & {r.TC:.3f} & {ci(r, 'TCE')} & --" + r" \\"]
    L += [f"GT & Ground truth & -- & -- & -- & {tc_gt:.3f} & -- & --" + r" \\",
          r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    save_tex(L, out_tab / "tab_temporal.tex")


def paired_delta(pt: pd.DataFrame, cfg, arch, pre, post, metric="Dice"):
    base = pt[(pt.arch == arch) & (pt.pre == "P0") & (pt.post == "Q0")].set_index("traj")
    cur = pt[(pt.arch == arch) & (pt.pre == pre) & (pt.post == post)].set_index("traj")
    if not len(base) or not len(cur):
        return None
    d = (cur[metric] - base[metric]).loc[base.index]
    return bootstrap_ci(d.values, cfg["eval"]["bootstrap_samples"], seed=cfg["seed"], groups=base.regime.values)


def tab_delta(pt, cfg, out_tab, out_csv):
    """Diferenca PAREADA (mesmas trajetorias) de cada tecnica isolada vs. arquitetura sozinha."""
    recs = []
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Effect of each technique in isolation: paired difference of test Dice (in Dice points $\times100$) "
         r"with respect to the architecture alone (P0+Q0), computed on the same test trajectories, with the 95\% "
         r"stratified bootstrap CI. Bold: CI does not contain zero (the technique significantly improves, $\blacktriangle$, "
         r"or worsens, $\blacktriangledown$, the segmentation).}",
         r"\label{tab:delta}", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Code & Technique & U-Net & SegFormer & Mask2Former \\", r"\midrule"]
    items = [(p, "Q0") for p in PRES[1:]] + [("P0", q) for q in POSTS[1:]]
    for k, (p, q) in enumerate(items):
        if k == len(PRES) - 1:
            L.append(r"\midrule")
        code = p if q == "Q0" else q
        name = PRE_NAMES[p] if q == "Q0" else POST_NAMES[q]
        cells = []
        for a in ARCHS:
            res = paired_delta(pt, cfg, a, p, q)
            if res is None:
                cells.append("--"); continue
            m, lo, hi = (100 * x for x in res)
            sig = lo > 0 or hi < 0
            arrow = (r"$\blacktriangle$" if m > 0 else r"$\blacktriangledown$") if sig else ""
            s = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            cells.append(f"\\textbf{{{s}}}{arrow}" if sig else s)
            recs.append(dict(arch=a, technique=code, name=name, delta_dice=m / 100, ci_low=lo / 100,
                             ci_high=hi / 100, significant=sig))
        L.append(f"{code} & {esc(name)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    save_tex(L, out_tab / "tab_delta.tex")
    pd.DataFrame(recs).to_csv(out_csv / "paired_deltas.csv", index=False)


def protocol_csv(t, out_csv):
    """Uma linha por configuracao, com as colunas da Secao 12 do protocolo da turma."""
    from gsseg.postprocess import POST_PARAMS
    man_path = ROOT / "data" / "subset_manifest.json"
    man = load_json(man_path) if man_path.exists() else {}
    params = load_json(out_csv / "post_params.json") if (out_csv / "post_params.json").exists() else {}
    rows = []
    for _, r in t.iterrows():
        tuned = params.get(f"{r.arch}_{r.pre}", {}).get(r.post, {}) if r.arch != "otsu_A" else {}
        rows.append({
            "experiment": r.exp_id, "dataset": FLUID,
            "dataset_variant": man.get("dataset_variant", ""), "trajectory_split": man.get("trajectory_split", ""),
            "resolution_original": "128x128", "resolution_used": "128x128",
            "sample_budget_gb": man.get("size_gb", ""), "preprocessing_id": r.pre,
            "preprocessing_params": PRE_PARAMS.get(r.pre, ""), "architecture": ARCH_NAMES.get(r.arch, "Otsu on A"),
            "epochs": r.get("epochs", ""), "best_epoch": r.get("best_epoch", ""), "seed": r.get("seed", ""),
            "postprocessing_id": r.post,
            "postprocessing_params": (POST_PARAMS.get(r.post, "") + (f" -> {tuned}" if tuned else "")),
            "IoU": r.IoU, "Dice": r.Dice, "Precision": r.Precision, "Recall": r.Recall,
            "IoU_background": r.IoU_bg, "Dice_background": r.Dice_bg, "mIoU": r.mIoU,
            "IoU_CI95": f"[{r.IoU_lo:.4f}, {r.IoU_hi:.4f}]", "Dice_CI95": f"[{r.Dice_lo:.4f}, {r.Dice_hi:.4f}]",
            "AVE": r.AVE, "CDE": r.CDE, "TC": r.TC, "TCE": r.TCE,
            "latency_ms": r.latency_ms_b1 + r.post_ms_per_frame, "peak_memory_mib": r.peak_infer_mem_mib_b1,
            "notes": ("classical reference, no learning" if r.arch == "otsu_A" else
                      f"test split; latency = network + post on {r.get('gpu', '')}; resampling: none (native)"),
        })
    pd.DataFrame(rows).to_csv(out_csv / "results_protocolo.csv", index=False)


def tab_sequential(g, out_tab, out_csv):
    """Todas as combinacoes (E001..E198 + REF), numeracao do protocolo da turma."""
    t = g[g.split == "test"].sort_values("exp_id")
    t = pd.concat([t[t.exp_id != "REF"], t[t.exp_id == "REF"]])
    cols = {"exp_id": "Experimento", "pre": "Pre-processamento", "arch": "Arquitetura", "post": "Pos-processamento",
            "IoU": "IoU", "Dice": "Dice", "Precision": "Precision", "Recall": "Recall"}
    seq = t[list(cols)].rename(columns=cols)
    seq["Arquitetura"] = seq["Arquitetura"].map(lambda a: ARCH_NAMES.get(a, "Otsu on A"))
    seq["Latencia_ms"] = t.latency_ms_b1 + t.post_ms_per_frame
    for k in ["AVE", "CDE", "TC", "TCE"]:
        seq[k] = t[k].values
    seq.insert(1, "Fluido", FLUID)
    seq.to_csv(out_csv / "results_sequential.csv", index=False)
    protocol_csv(t, out_csv)

    L = [r"\begingroup\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{longtable}{@{}lllllccccccc@{}}",
         r"\caption{Sequential and standardized results of every evaluated combination on the test set "
         r"(E001--E150: P1--P10 $\times$ architecture $\times$ Q1--Q5, the protocol matrix; "
         r"E151--E198: baseline rows without pre- and/or post-processing). Latency in ms per frame.}\label{tab:sequential}\\",
         r"\toprule",
         r"Exp. & Pre & Architecture & Post & IoU & Dice & Prec. & Rec. & AVE & CDE & TCE & Lat. \\", r"\midrule",
         r"\endfirsthead", r"\toprule",
         r"Exp. & Pre & Architecture & Post & IoU & Dice & Prec. & Rec. & AVE & CDE & TCE & Lat. \\", r"\midrule",
         r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, r in t.iterrows():
        arch = ARCH_NAMES.get(r.arch, "Otsu on $A$")
        L.append(f"{r.exp_id} & {r.pre} & {arch} & {r.post} & {r.IoU:.4f} & {r.Dice:.4f} & {r.Precision:.4f} & "
                 f"{r.Recall:.4f} & {r.AVE:.3f} & {r.CDE:.2f} & {r.TCE:.4f} & {r.latency_ms_b1 + r.post_ms_per_frame:.2f}" + r" \\")
    L += [r"\end{longtable}", r"\endgroup"]
    save_tex(L, out_tab / "tab_sequential.tex")


def tab_classes(rows, g, out_tab):
    """Per-class metrics (protocol, Section 10): structure, background and mIoU."""
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Per-class metrics on the test set for the configurations of Table~\ref{tab:main}: IoU of the "
         r"structure class, IoU and Dice of the background class, and mIoU (mean IoU of the two classes). Mean "
         r"$\pm$ half-width of the 95\% CI.}",
         r"\label{tab:classes}", r"\small", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{@{}lllcccc@{}}",
         r"\toprule", r"Exp. & Architecture & Condition & IoU structure & IoU background & Dice background & mIoU \\",
         r"\midrule"]
    prev = None
    for a, c, p, q, r in rows:
        if prev and prev != a:
            L.append(r"\midrule")
        prev = a
        L.append(f"{r.exp_id} & {ARCH_NAMES[a]} & {COND_TEX[c]} & {ci(r, 'IoU')} & {ci(r, 'IoU_bg')} & "
                 f"{ci(r, 'Dice_bg')} & {ci(r, 'mIoU')}" + r" \\")
    b = g[(g.arch == "otsu_A") & (g.split == "test")]
    if len(b):
        r = b.iloc[0]
        L += [r"\midrule", f"REF & Otsu on $A$ & -- & {ci(r, 'IoU')} & {ci(r, 'IoU_bg')} & {ci(r, 'Dice_bg')} & "
              f"{ci(r, 'mIoU')}" + r" \\"]
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    save_tex(L, out_tab / "tab_classes.tex")


def tab_sensitivity(rows, cfg, out_tab, rdir):
    """Dice with and without the initial snapshot n = 0 (requires per_traj_sem_n0.csv from 07)."""
    f = rdir / "per_traj_sem_n0.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Sensitivity to the initial snapshot: mean test Dice over all 21 snapshots, at $n=0$ only and over "
         r"the 20 snapshots $n\geq50$, for the configurations selected on validation. Mean over trajectories; 95\% "
         r"stratified bootstrap CI for $n\geq50$.}",
         r"\label{tab:sensitivity}", r"\small", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{@{}lllccc@{}}",
         r"\toprule", r"Exp. & Architecture & Condition & Dice (all) & Dice ($n=0$) & Dice ($n\geq50$) \\",
         r"\midrule"]
    prev = None
    for a, c, p, q, r in rows + [("otsu_A", "Reference", "P0", "Q0", None)]:
        if prev and prev != a:
            L.append(r"\midrule")
        prev = a
        x = t[(t.arch == a) & (t.pre == p) & (t.post == q)]
        m, lo, hi = bootstrap_ci(x.Dice_sem_n0.values, cfg["eval"]["bootstrap_samples"], seed=cfg["seed"],
                                 groups=x.regime.values)
        eid = r.exp_id if r is not None else "REF"
        cond = COND_TEX.get(c, "Classical reference")
        name = ARCH_NAMES.get(a, "Otsu on $A$")
        L.append(f"{eid} & {name} & {cond} & {x.Dice_all.mean():.3f} & {x.Dice_n0.mean():.3f} & "
                 f"{m:.3f}{{\\scriptsize$\\pm${(hi - lo) / 2:.3f}}}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    save_tex(L, out_tab / "tab_sensitivity.tex")


def tab_regime(reg, rows, out_tab):
    regimes = list(dict.fromkeys(reg.regime))
    fin = final_cfg(rows)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Test Dice per pattern regime for the architecture alone and for the final configuration of each "
         r"architecture, and for the classical reference.}", r"\label{tab:regime}", r"\scriptsize",
         r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}l" + "c" * len(regimes) + "@{}}", r"\toprule",
         "Configuration & " + " & ".join(r.capitalize() for r in regimes) + r" \\", r"\midrule"]
    for a in ARCHS:
        if a not in fin:
            continue
        for p, q in dict.fromkeys([("P0", "Q0"), fin[a]]):
            d = reg[(reg.arch == a) & (reg.pre == p) & (reg.post == q)].set_index("regime")
            if len(d):
                L.append(f"{exp_id(a, p, q)} {ARCH_NAMES[a]} ({p}+{q}) & " +
                         " & ".join(f"{d.loc[x, 'Dice']:.3f}" for x in regimes) + r" \\")
    d = reg[reg.arch == "otsu_A"].set_index("regime")
    if len(d):
        L.append("REF Otsu on $A$ & " + " & ".join(f"{d.loc[x, 'Dice']:.3f}" for x in regimes) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    save_tex(L, out_tab / "tab_regime.tex")


def tab_cost(meta, out_tab):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Model size and training cost (20 epochs, batch 32, native $128\times128$ input; reference input P0). "
         r"Latency and throughput measured with a dedicated GPU.}",
         r"\label{tab:cost}", r"\small", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"Architecture & Params (M) & Train (min) & Best epoch & Train mem. (MiB) & Latency (ms) & Throughput (fps) \\",
         r"\midrule"]
    for a in ARCHS:
        m = meta.get(f"{a}_P0")
        if m:
            L.append(f"{ARCH_NAMES[a]} & {m['params']/1e6:.1f} & {m['train_time_s']/60:.1f} & "
                     f"{m.get('best_epoch', '--')}/{m.get('epochs_run', '--')} & {m['peak_train_mem_mib']:.0f} & "
                     f"{m['latency_ms_b1']:.2f} & {m.get('throughput_fps', m.get('throughput_fps_b64', 0)):.0f}" + r" \\")
    gpu = next((m["gpu"] for m in meta.values()), "")
    L += [r"\bottomrule", r"\end{tabular}}", f"\\\\[2pt]\\footnotesize GPU: {gpu.replace('NVIDIA ', '')}", r"\end{table}"]
    save_tex(L, out_tab / "tab_cost.tex")


def _scales(z, r_i, ref=500):
    """Comprimento de onda dominante de B e diametro equivalente mediano das estruturas do GT.

    Le de z apenas o instante de referencia (cada z[...] de um .npz rele o arranjo inteiro do disco).
    """
    from scipy import ndimage as ndi
    steps, regime = np.asarray(z["steps"]), np.asarray(z["regime"])
    j = int(np.argmin(np.abs(steps - ref)))
    Bj, Gj = np.asarray(z["B"])[:, j], np.asarray(z["gt"])[:, j]
    idx = np.where((regime == r_i) & (Gj.mean((1, 2)) > 0.01))[0]
    H = Bj.shape[-1]
    k = np.sqrt(np.add.outer(np.fft.fftfreq(H) ** 2, np.fft.fftfreq(H) ** 2))
    kb = np.round(k * H).astype(int)
    lams, deqs = [], []
    for i in idx:
        f = Bj[i] - Bj[i].mean()
        prof = np.bincount(kb.ravel(), (np.abs(np.fft.fft2(f)) ** 2).ravel())
        prof[0] = 0
        lams.append(H / np.argmax(prof[: H // 2]))
        lab, _ = ndi.label(Gj[i])
        deqs += list(np.sqrt(4 * np.bincount(lab.ravel())[1:] / np.pi))
    return float(np.median(lams)), float(np.median(deqs))


def tab_data(cfg, out_tab):
    proc = data_dir(cfg) / "processed"
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Subset of \texttt{gray\_scott\_reaction\_diffusion} used in this study (all trajectories of the "
         r"official The Well splits; native $128\times128$ grid; 21 snapshots per trajectory, "
         r"$\Delta n = 50$). $\bar\tau$: mean Otsu threshold on $B$ at $n=500$; fg: mean foreground fraction of the "
         r"ground truth; $\lambda$: dominant wavelength of $B$ at $n=500$ (peak of the radial power spectrum); "
         r"$d_{eq}$: median equivalent diameter $\sqrt{4a/\pi}$ of the ground-truth structures at $n=500$ (training split).}",
         r"\label{tab:data}", r"\small", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lcccccccc@{}}",
         r"\toprule",
         r"Regime & $F$ & $k$ & train/val/test & $\bar\tau$ & fg (\%) & extinct at $n{=}500$ & $\lambda$ (px) & $d_{eq}$ (px) \\",
         r"\midrule"]
    Z = {s: dict(np.load(proc / f"{s}.npz")) for s in ("train", "valid", "test")}  # carrega uma vez
    for i, r in enumerate(list(Z["train"]["regimes"])):
        n = [int((Z[s]["regime"] == i).sum()) for s in Z]
        sel = Z["train"]["regime"] == i
        F, k = Z["train"]["F"][sel][0], Z["train"]["k"][sel][0]
        tau = np.concatenate([Z[s]["tau"][Z[s]["regime"] == i] for s in Z]).mean()
        fg = np.concatenate([Z[s]["gt"][Z[s]["regime"] == i].mean((1, 2, 3)) for s in Z]).mean() * 100
        ext = sum(int((~Z[s]["tau_valid"][Z[s]["regime"] == i]).sum()) for s in Z)
        lam, deq = _scales(Z["train"], i)
        L.append(f"{r.capitalize()} & {F:.3f} & {k:.3f} & {n[0]}/{n[1]}/{n[2]} & {tau:.3f} & {fg:.1f} & {ext} & "
                 f"{lam:.1f} & {deq:.1f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    save_tex(L, out_tab / "tab_data.tex")


# ----------------------------------------------------------------------------- figuras
def _choose_traj(z, r_i):
    idx = np.where(z["regime"] == r_i)[0]
    j = int(np.argmin(np.abs(z["steps"] - 500)))
    return int(idx[np.argmax(z["gt"][idx, j].mean((-2, -1)))])  # trajetoria com padrao em n=500


def _frame_axes(a):
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for s in a.spines.values():
        s.set_visible(True); s.set_linewidth(0.3); s.set_color(INK2)


def _err_rgb(pred, gt):
    rgb = np.ones((*gt.shape, 3))
    hexrgb = lambda h: tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))  # noqa: E731
    rgb[(pred == 1) & (gt == 1)] = TP_C
    rgb[(pred == 1) & (gt == 0)] = hexrgb(FP_C)
    rgb[(pred == 0) & (gt == 1)] = hexrgb(FN_C)
    return rgb


def fig_overview(cfg, out_fig):
    """Visao geral: campo B e GT de cada regime em 11 instantes."""
    z = dict(np.load(data_dir(cfg) / "processed" / "test.npz"))
    vs = cfg["eval"]["visual_steps"]
    cols = [int(np.argmin(np.abs(z["steps"] - s))) for s in vs]
    regs = list(z["regimes"])
    for key, cmap, fname in (("B", CMAP_B, "fluid_by_time"), ("gt", CMAP_GT, "gt_by_time")):
        fig, ax = plt.subplots(len(regs), len(vs), figsize=(len(vs) * 0.68, len(regs) * 0.72))
        for i, r in enumerate(regs):
            t = _choose_traj(z, i)
            for c, j in enumerate(cols):
                a = ax[i, c]
                im = a.imshow(z[key][t, j], cmap=cmap, vmin=0, vmax=1 if key == "gt" else 0.5, interpolation="nearest")
                _frame_axes(a)
                if i == 0:
                    a.set_title(f"{TXT['n']} = {vs[c]}", fontsize=6.5, color=INK)
                if c == 0:
                    a.set_ylabel(r.capitalize(), fontsize=6.5, color=INK)
        fig.subplots_adjust(wspace=0.04, hspace=0.04)
        if key == "B":
            cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
            cb.set_label(TXT["concB"], fontsize=6.5); cb.ax.tick_params(labelsize=5.5); _dec_axes(cb.ax)
        fig.savefig(out_fig / f"{fname}.pdf"); fig.savefig(out_fig / f"{fname}.png")
        plt.close(fig)


def fig_visual_tables(cfg, rows, params, tag, out_fig):
    """Tabela visual pedida: frame 0, dois intermediarios e frame final; para cada arquitetura:
    entrada, entrada pre-processada, GT e mascaras sem pre/pos, com pre, com pos e com pre+pos."""
    proc = data_dir(cfg) / "processed"
    z = dict(np.load(proc / "test.npz"))
    vs = cfg["eval"]["visual_table_steps"]
    cols = [int(np.argmin(np.abs(z["steps"] - s))) for s in vs]
    conds = {a: {c: (p, q) for (aa, c, p, q, _r) in rows if aa == a} for a in ARCHS}
    archs = [a for a in ARCHS if conds[a]]
    Xraw = np.load(proc / "pre" / "P0_test.npy", mmap_mode="r")
    pre_cache, prob_cache = {}, {}

    def prob(a, p):
        if (a, p) not in prob_cache:
            prob_cache[(a, p)] = np.load(ROOT / "outputs" / "preds" / f"{a}_{p}{tag}_test.npy", mmap_mode="r")
        return prob_cache[(a, p)]

    def pre_img(p, t, j):
        if p not in pre_cache:
            pre_cache[p] = np.load(proc / "pre" / f"{p}_test.npy", mmap_mode="r")
        return np.asarray(pre_cache[p][t, j])

    row_labels = [TXT[k] for k in ("inputA", "preinput", "gt", "c_none", "c_pre", "c_post", "c_both")]
    nT = len(vs)
    for r_i, regime in enumerate(z["regimes"]):
        t = _choose_traj(z, r_i)
        gt = z["gt"][t]
        ncol = len(archs) * nT + (len(archs) - 1)
        widths = []
        for k in range(len(archs)):
            widths += [1] * nT + ([0.18] if k < len(archs) - 1 else [])
        fig = plt.figure(figsize=(0.6 * sum(widths) + 0.9, 0.62 * len(row_labels) + 0.75))
        gs = GridSpec(len(row_labels), ncol, figure=fig, width_ratios=widths, wspace=0.05, hspace=0.07)
        for k, a in enumerate(archs):
            c0 = k * (nT + 1)
            cd = conds[a]
            p_pre = cd.get("Pre", ("P0", "Q0"))[0]
            masks = {}
            for c, (p, q) in cd.items():
                P = np.asarray(prob(a, p)[t:t + 1], dtype=np.float32)
                kw = params.get(f"{a}_{p}{tag}", {}).get(q, {})
                masks[c] = apply_post(q, P, np.asarray(Xraw[t:t + 1]), **kw)[0]
            for ci_, j in enumerate(cols):
                for ri, lab in enumerate(row_labels):
                    ax = fig.add_subplot(gs[ri, c0 + ci_])
                    _frame_axes(ax)
                    note = None
                    if ri == 0:
                        ax.imshow(Xraw[t, j], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    elif ri == 1:
                        im = pre_img(p_pre, t, j)
                        lo, hi = np.percentile(im, [0.5, 99.5])
                        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
                        note = p_pre
                    elif ri == 2:
                        ax.imshow(gt[j], cmap=CMAP_GT, vmin=0, vmax=1, interpolation="nearest")
                    else:
                        c = CONDS[ri - 3]
                        if c in masks:
                            ax.imshow(_err_rgb(masks[c][j], gt[j]), interpolation="nearest")
                            p, q = cd[c]
                            note = {"No pre/post": "P0+Q0", "Pre": f"{p}+Q0", "Post": f"P0+{q}"}.get(c, f"{p}+{q}")
                        else:
                            ax.axis("off")
                    if note and ci_ == 0:
                        ax.text(2, 4, note, fontsize=4.2, va="top", ha="left", color=INK,
                                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
                    if ri == 0:
                        ax.set_title(f"{TXT['n']} = {vs[ci_]}", fontsize=6, color=INK, pad=2)
                    if ci_ == 0 and k == 0:
                        ax.set_ylabel(lab, fontsize=5.8, color=INK, rotation=0, ha="right", va="center", labelpad=3)
            # titulo do bloco da arquitetura
            x0 = fig.add_subplot(gs[0, c0:c0 + nT], frameon=False)
            x0.set_xticks([]); x0.set_yticks([]); x0.grid(False)
            x0.set_title(ARCH_NAMES[a], fontsize=8, color=INK, pad=13, fontweight="bold")
        fig.legend(handles=[Patch(color=TP_C, label=TXT["TP"]), Patch(color=FP_C, label=TXT["FP"]),
                            Patch(color=FN_C, label=TXT["FN"])], loc="lower center", ncol=3, fontsize=6.5,
                   bbox_to_anchor=(0.5, 0.0))
        fig.savefig(out_fig / f"visual_table_{regime}.pdf"); fig.savefig(out_fig / f"visual_table_{regime}.png")
        plt.close(fig)


def fig_heatmaps(g, out_fig):
    t = g[g.split == "test"]
    pres, posts = PRES, POSTS
    archs = [a for a in ARCHS if (t.arch == a).any()]
    vals = t[t.arch.isin(archs)].Dice
    vmin, vmax = float(vals.min()), float(vals.max())
    fig, axs = plt.subplots(1, len(archs), figsize=(2.6 * len(archs), 2.7), constrained_layout=True)
    axs = np.atleast_1d(axs)
    for ax, a in zip(axs, archs):
        M = np.full((len(pres), len(posts)), np.nan)
        for i, p in enumerate(pres):
            for j, q in enumerate(posts):
                d = t[(t.arch == a) & (t.pre == p) & (t.post == q)]
                if len(d):
                    M[i, j] = d.Dice.iloc[0]
        im = ax.imshow(M, cmap=CMAP_DICE, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(posts)), posts, fontsize=5.5, rotation=90); ax.set_yticks(range(len(pres)), pres, fontsize=5.5)
        ax.grid(False); ax.set_title(ARCH_NAMES[a], fontsize=8, color=INK)
        ax.set_xlabel(TXT["post_axis"], fontsize=6.5); ax.set_ylabel(TXT["pre_axis"], fontsize=6.5)
        for i in range(len(pres)):
            for j in range(len(posts)):
                if not np.isnan(M[i, j]):
                    v = (M[i, j] - vmin) / (vmax - vmin + 1e-12)
                    ax.text(j, i, fnum(M[i, j])[1:], ha="center", va="center", fontsize=3.6,
                            color="#ffffff" if v > 0.55 else INK)
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01); cb.set_label(TXT["test_dice"], fontsize=6.5); _dec_axes(cb.ax)
    cb.ax.tick_params(labelsize=5.5)
    fig.savefig(out_fig / "heatmap_dice.pdf"); fig.savefig(out_fig / "heatmap_dice.png")
    plt.close(fig)


def fig_temporal(ptime, rows, out_fig):
    fin = final_cfg(rows)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), constrained_layout=True)
    for a, (p, q) in fin.items():
        d = ptime[(ptime.arch == a) & (ptime.pre == p) & (ptime.post == q)].sort_values("step")
        axs[0].plot(d.step, d.Dice, color=COLORS[a], lw=1.4, label=f"{ARCH_NAMES[a]} ({p}+{q})")
        axs[1].plot(d.step, d.area_pred, color=COLORS[a], lw=1.4, label=ARCH_NAMES[a])
        d0 = ptime[(ptime.arch == a) & (ptime.pre == "P0") & (ptime.post == "Q0")].sort_values("step")
        if (p, q) != ("P0", "Q0"):
            axs[0].plot(d0.step, d0.Dice, color=COLORS[a], lw=0.8, ls=":", alpha=0.9)
    d = ptime[ptime.arch == "otsu_A"].sort_values("step")
    if len(d):
        axs[0].plot(d.step, d.Dice, color=COLORS["otsu_A"], lw=1.0, ls="--", label=TXT["otsu"])
    d0 = ptime[ptime.arch == ARCHS[0]].groupby("step").area_gt.first()
    axs[1].plot(d0.index, d0.values, color=INK, lw=1.0, ls=":", label=TXT["gt_line"])
    axs[0].set_xlabel(TXT["snapshot"]); axs[0].set_ylabel(TXT["mean_dice"])
    axs[1].set_xlabel(TXT["snapshot"]); axs[1].set_ylabel(TXT["area"]); _dec_axes(axs[0], axs[1])
    axs[0].legend(fontsize=6, loc="lower right", title=TXT["leg_title"], title_fontsize=5.5)
    axs[1].legend(fontsize=6.5)
    fig.savefig(out_fig / "temporal_dice_area.pdf"); fig.savefig(out_fig / "temporal_dice_area.png")
    plt.close(fig)


def fig_effects(g, out_fig):
    t = g[g.split == "test"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), constrained_layout=True)
    for a in ARCHS:
        d = t[(t.arch == a) & (t.post == "Q0")].copy()
        if not len(d):
            continue
        d["i"] = d.pre.str[1:].astype(int); d = d.sort_values("i")
        axs[0].plot(d.i, d.Dice, "-o", color=COLORS[a], lw=1.2, ms=3.5, label=ARCH_NAMES[a])
        d = t[(t.arch == a) & (t.pre == "P0")].copy()
        d["i"] = d.post.str[1:].astype(int); d = d.sort_values("i")
        axs[1].plot(d.i, d.Dice, "-o", color=COLORS[a], lw=1.2, ms=3.5, label=ARCH_NAMES[a])
    axs[0].set_xticks(range(len(PRES)), PRES); axs[0].set_ylabel(TXT["test_dice"]); _dec_axes(axs[0], axs[1])
    axs[0].set_title(TXT["eff_pre"], fontsize=8)
    axs[1].set_xticks(range(len(POSTS)), POSTS)
    axs[1].set_title(TXT["eff_post"], fontsize=8)
    axs[0].legend(fontsize=6.5)
    fig.savefig(out_fig / "effects_pre_post.pdf"); fig.savefig(out_fig / "effects_pre_post.png")
    plt.close(fig)


def fig_training(tag, out_fig):
    fig, ax = plt.subplots(figsize=(3.4, 2.3), constrained_layout=True)
    for a in ARCHS:
        f = ROOT / "outputs" / "models" / f"{a}_P0{tag}" / "history.csv"
        if f.exists():
            h = pd.read_csv(f)
            x = h["epoch"] if "epoch" in h else h["iteration"]
            ax.plot(x, h.val_dice, "-o", color=COLORS[a], lw=1.4, ms=3, label=ARCH_NAMES[a])
    ax.set_xlabel(TXT["epoch"]); ax.set_ylabel(TXT["val_dice"]); ax.legend(fontsize=6.5); _dec_axes(ax)
    fig.savefig(out_fig / "training_curves.pdf"); fig.savefig(out_fig / "training_curves.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None, help="padrao: paper/ (tag vazia) ou outputs/report{tag}/")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rdir = ROOT / "outputs" / f"results{args.tag}"
    out = Path(args.out) if args.out else (ROOT / "paper" if not args.tag else ROOT / "outputs" / f"report{args.tag}")
    out_fig, out_tab = out / "figures", out / "tables"
    out_fig.mkdir(parents=True, exist_ok=True); out_tab.mkdir(parents=True, exist_ok=True)

    g = pd.read_csv(rdir / "grid.csv")
    reg = pd.read_csv(rdir / "per_regime.csv")
    ptime = pd.read_csv(rdir / "per_time.csv")
    ptraj = pd.read_csv(rdir / "per_traj.csv")
    params = load_json(rdir / "post_params.json")
    meta = {}
    for d in (ROOT / "outputs" / "models").glob(f"*{args.tag}"):
        if (d / "meta.json").exists() and d.name.endswith(args.tag) and (args.tag or "_smoke" not in d.name):
            m = load_json(d / "meta.json")
            meta[f"{m['arch']}_{m['pre']}"] = m
    zt = np.load(data_dir(cfg) / "processed" / "test.npz")
    tc_gt = float(frame_scores(zt["gt"][:, 1:], zt["gt"][:, :-1])["IoU"].mean())

    rows = selected(g)
    tab_methods(out_tab)
    tab_main(rows, g, out_tab, rdir)
    tab_temporal(rows, g, tc_gt, out_tab)
    tab_delta(ptraj, cfg, out_tab, rdir)
    tab_sequential(g, out_tab, rdir)
    tab_regime(reg, rows, out_tab)
    tab_classes(rows, g, out_tab)
    tab_sensitivity(rows, cfg, out_tab, rdir)
    tab_cost(meta, out_tab)
    tab_data(cfg, out_tab)
    fig_overview(cfg, out_fig)
    fig_visual_tables(cfg, rows, params, args.tag, out_fig)
    fig_heatmaps(g, out_fig)
    fig_temporal(ptime, rows, out_fig)
    fig_effects(g, out_fig)
    fig_training(args.tag, out_fig)
    print(pd.read_csv(rdir / "results_standard.csv")[["Experiment", "Architecture", "Condition", "Pre", "Post", "IoU",
                                                       "Dice", "Precision", "Recall", "Latency_total_ms"]].to_string(index=False))
    print(f"tabelas/figuras em {out}")


if __name__ == "__main__":
    main()
