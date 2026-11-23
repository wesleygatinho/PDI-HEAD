"""Sensibilidade ao instante inicial: metricas por trajetoria EXCLUINDO n = 0.

Em n = 0 a condicao inicial e aleatoria e o campo A ainda nao reflete a estrutura de B; esse quadro
(1 de 21) pode dominar diferencas pequenas de Dice medio. Este script recalcula, no teste, o Dice e o
IoU por trajetoria com e sem n = 0 para todas as combinacoes (mesmos parametros de pos-processamento
escolhidos na validacao, lidos de outputs/results/post_params.json) e as diferencas pareadas de cada
tecnica em relacao a arquitetura sozinha.

Saidas: outputs/results/per_traj_sem_n0.csv e outputs/results/paired_deltas_sem_n0.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gsseg.metrics import bootstrap_ci, frame_scores  # noqa: E402
from gsseg.postprocess import apply_post  # noqa: E402
from gsseg.utils import data_dir, load_config, load_json  # noqa: E402

ARCHS = ["unet", "segformer", "mask2former"]


def otsu_on_input(A):
    from skimage.filters import threshold_otsu
    flat = A.reshape(-1, *A.shape[-2:])
    out = np.zeros(flat.shape, dtype=np.uint8)
    for i, f in enumerate(flat):
        if f.max() - f.min() > 0.05:
            out[i] = f < threshold_otsu(f.astype(np.float64), nbins=256)
    return out.reshape(A.shape)


def main():
    cfg = load_config()
    rdir = ROOT / "outputs" / "results"
    params = load_json(rdir / "post_params.json")
    z = np.load(data_dir(cfg) / "processed" / "test.npz")
    G, steps, reg, regimes = z["gt"], z["steps"], z["regime"], list(z["regimes"])
    later = steps > 0
    rows = []

    def add(arch, pre, post, M):
        fs = frame_scores(M, G)
        for i in range(G.shape[0]):
            rows.append(dict(arch=arch, pre=pre, post=post, traj=i, regime=regimes[reg[i]],
                             Dice_all=float(fs["Dice"][i].mean()), Dice_sem_n0=float(fs["Dice"][i, later].mean()),
                             IoU_all=float(fs["IoU"][i].mean()), IoU_sem_n0=float(fs["IoU"][i, later].mean()),
                             Dice_n0=float(fs["Dice"][i, ~later].mean())))

    for a in ARCHS:
        for p in cfg["preprocessing"]:
            P = np.load(ROOT / "outputs" / "preds" / f"{a}_{p}_test.npy").astype(np.float32)
            for q in cfg["postprocessing"]:
                add(a, p, q, apply_post(q, P, None, **params[f"{a}_{p}"].get(q, {})))
            print(f"[ok] {a}_{p}", flush=True)
    add("otsu_A", "P0", "Q0", otsu_on_input(z["A"]))
    df = pd.DataFrame(rows)
    df.to_csv(rdir / "per_traj_sem_n0.csv", index=False)

    recs = []
    for a in ARCHS:
        base = df[(df.arch == a) & (df.pre == "P0") & (df.post == "Q0")].set_index("traj")
        items = [(p, "Q0") for p in cfg["preprocessing"][1:]] + [("P0", q) for q in cfg["postprocessing"][1:]]
        for p, q in items:
            cur = df[(df.arch == a) & (df.pre == p) & (df.post == q)].set_index("traj")
            for col in ("Dice_all", "Dice_sem_n0"):
                d = (cur[col] - base[col]).loc[base.index].values
                m, lo, hi = bootstrap_ci(d, cfg["eval"]["bootstrap_samples"], seed=cfg["seed"], groups=base.regime.values)
                recs.append(dict(arch=a, technique=p if q == "Q0" else q, metric=col, delta=m, ci_low=lo, ci_high=hi,
                                 significant=bool(lo > 0 or hi < 0)))
    pd.DataFrame(recs).to_csv(rdir / "paired_deltas_sem_n0.csv", index=False)
    print("ok")


if __name__ == "__main__":
    main()
