"""Aplica Q0-Q10 as probabilidades salvas de cada modelo e calcula todas as metricas.

Para cada modelo (arquitetura x pre-processamento):
  1. Parametros livres de Q1/Q5/Q6/Q7 sao escolhidos no conjunto de VALIDACAO (Dice medio).
  2. Cada pos-processamento e aplicado em validacao e teste; metricas sao calculadas
     em ambos (a validacao e usada depois para escolher as melhores combinacoes,
     o teste so e reportado).
Saidas em outputs/results/:
  grid.csv          uma linha por (arch, pre, post, split) com medias e IC 95%
  per_regime.csv    Dice/IoU por regime (teste)
  per_time.csv      Dice/IoU/area por instante (teste)
  per_traj.csv      todas as metricas por trajetoria (teste) -> diferencas pareadas no 05
  post_params.json  parametros escolhidos na validacao
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gsseg.experiments import condition, exp_id  # noqa: E402
from gsseg.metrics import bootstrap_ci, evaluate  # noqa: E402
from gsseg.postprocess import TUNABLE, apply_post  # noqa: E402
from gsseg.utils import ROOT, data_dir, load_config, load_json, out_dir, save_json  # noqa: E402

METRICS = ["IoU", "Dice", "Precision", "Recall", "IoU_bg", "Dice_bg", "mIoU", "AVE", "CDE", "TC", "TCE"]


def tune(q, Pv, Gv, imv):
    grid = TUNABLE[q]
    key = next(iter(grid))
    best, best_v = None, -1
    for v in grid[key]:
        m = apply_post(q, Pv, imv, **{key: v})
        d = evaluate(m, Gv, temporal=False)["per_traj"]["Dice"].mean()
        if d > best_v:
            best, best_v = v, d
    return {key: best}


def run_model(args):
    name, cfg_path, tag = args
    t_start = time.time()
    cfg = load_config(cfg_path)
    proc = data_dir(cfg) / "processed"
    pdir = ROOT / "outputs" / "preds"
    meta = load_json(ROOT / "outputs" / "models" / name / "meta.json")
    arch, pre = meta["arch"], meta["pre"]
    regimes = None
    data = {}
    for s in ("valid", "test"):
        z = np.load(proc / f"{s}.npz")
        regimes = list(z["regimes"])
        data[s] = dict(P=np.load(pdir / f"{name}_{s}.npy").astype(np.float32), G=z["gt"],
                       im=np.load(proc / "pre" / f"P0_{s}.npy"), reg=z["regime"], steps=z["steps"])
    rows, reg_rows, time_rows, traj_rows, params = [], [], [], [], {}
    for q in cfg["postprocessing"]:
        kw = tune(q, data["valid"]["P"], data["valid"]["G"], data["valid"]["im"]) if q in TUNABLE else {}
        params[q] = kw
        for s in ("valid", "test"):
            d = data[s]
            t0 = time.perf_counter()
            M = apply_post(q, d["P"], d["im"], **kw)
            post_ms = (time.perf_counter() - t0) / (M.shape[0] * M.shape[1]) * 1000
            ev = evaluate(M, d["G"])
            row = dict(exp_id=exp_id(arch, pre, q), condition=condition(pre, q), arch=arch, pre=pre, post=q,
                       split=s, post_params=str(kw), post_ms_per_frame=post_ms)
            for k in METRICS:
                mean, lo, hi = bootstrap_ci(ev["per_traj"][k], cfg["eval"]["bootstrap_samples"],
                                            seed=cfg["seed"], groups=d["reg"])
                row[k], row[f"{k}_lo"], row[f"{k}_hi"] = mean, lo, hi
            row.update(latency_ms_b1=meta["latency_ms_b1"], peak_infer_mem_mib_b1=meta["peak_infer_mem_mib_b1"],
                       peak_train_mem_mib=meta["peak_train_mem_mib"], params_M=meta["params"] / 1e6,
                       train_time_min=meta["train_time_s"] / 60, throughput_fps=meta.get("throughput_fps", meta.get("throughput_fps_b64")),
                       epochs=meta.get("epochs_run"), best_epoch=meta.get("best_epoch"), seed=meta.get("seed"),
                       gpu=meta.get("gpu"))
            rows.append(row)
            if s == "test":
                for i in range(len(d["reg"])):   # por trajetoria: base das diferencas pareadas
                    traj_rows.append(dict(exp_id=exp_id(arch, pre, q), arch=arch, pre=pre, post=q, traj=i,
                                          regime=regimes[d["reg"][i]],
                                          **{k: float(ev["per_traj"][k][i]) for k in METRICS}))
                for r_i, r in enumerate(regimes):
                    sel = d["reg"] == r_i
                    reg_rows.append(dict(arch=arch, pre=pre, post=q, regime=r,
                                         **{k: float(ev["per_traj"][k][sel].mean()) for k in METRICS}))
                fd, fi = ev["per_frame"]["Dice"], ev["per_frame"]["IoU"]
                ap, ag = M.mean((-2, -1)) * 100, d["G"].mean((-2, -1)) * 100
                for j, st in enumerate(d["steps"]):
                    time_rows.append(dict(arch=arch, pre=pre, post=q, step=int(st),
                                          Dice=float(fd[:, j].mean()), IoU=float(fi[:, j].mean()),
                                          area_pred=float(ap[:, j].mean()), area_gt=float(ag[:, j].mean())))
    print(f"[ok] {name} ({time.time() - t_start:.0f}s)", flush=True)
    return rows, reg_rows, time_rows, {name: params}, traj_rows


def otsu_on_input(A: np.ndarray) -> np.ndarray:
    """Referencia classica sem aprendizado: Otsu por quadro no campo de entrada A.
    A e baixo onde B e alto (consumo do substrato), logo a estrutura e A < tau_A."""
    from skimage.filters import threshold_otsu
    flat = A.reshape(-1, *A.shape[-2:])
    out = np.zeros(flat.shape, dtype=np.uint8)
    for i, f in enumerate(flat):
        if f.max() - f.min() > 0.05:
            out[i] = f < threshold_otsu(f.astype(np.float64), nbins=256)
    return out.reshape(A.shape)


def classical_baseline(cfg):
    proc = data_dir(cfg) / "processed"
    rows, reg_rows, time_rows, traj_rows = [], [], [], []
    for s in ("valid", "test"):
        z = np.load(proc / f"{s}.npz")
        t0 = time.perf_counter()
        M = otsu_on_input(z["A"])
        ms = (time.perf_counter() - t0) / (M.shape[0] * M.shape[1]) * 1000
        ev = evaluate(M, z["gt"])
        if s == "test":
            for i in range(len(z["regime"])):
                traj_rows.append(dict(exp_id="REF", arch="otsu_A", pre="P0", post="Q0", traj=i,
                                      regime=str(z["regimes"][z["regime"][i]]),
                                      **{k: float(ev["per_traj"][k][i]) for k in METRICS}))
            for r_i, r in enumerate(z["regimes"]):
                sel = z["regime"] == r_i
                reg_rows.append(dict(arch="otsu_A", pre="P0", post="Q0", regime=str(r),
                                     **{k: float(ev["per_traj"][k][sel].mean()) for k in METRICS}))
            ap, ag = M.mean((-2, -1)) * 100, z["gt"].mean((-2, -1)) * 100
            for j, st in enumerate(z["steps"]):
                time_rows.append(dict(arch="otsu_A", pre="P0", post="Q0", step=int(st),
                                      Dice=float(ev["per_frame"]["Dice"][:, j].mean()),
                                      IoU=float(ev["per_frame"]["IoU"][:, j].mean()),
                                      area_pred=float(ap[:, j].mean()), area_gt=float(ag[:, j].mean())))
        row = dict(exp_id="REF", condition="Reference", arch="otsu_A", pre="P0", post="Q0", split=s, post_params="{}", post_ms_per_frame=0.0,
                   latency_ms_b1=ms, peak_infer_mem_mib_b1=0.0, peak_train_mem_mib=0.0, params_M=0.0,
                   train_time_min=0.0, throughput_fps=1000 / ms)
        for k in METRICS:
            mean, lo, hi = bootstrap_ci(ev["per_traj"][k], cfg["eval"]["bootstrap_samples"], seed=cfg["seed"],
                                        groups=z["regime"])
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = mean, lo, hi
        rows.append(row)
    return rows, reg_rows, time_rows, {}, traj_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    names = [f"{a}_{p}{args.tag}" for a in cfg["architectures"] for p in cfg["preprocessing"]]
    names = [n for n in names if (ROOT / "outputs" / "models" / n / "meta.json").exists()]
    print(f"{len(names)} modelos encontrados", flush=True)
    jobs = [(n, args.config, args.tag) for n in names]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(run_model, jobs))
    else:
        res = [run_model(j) for j in jobs]
    res.append(classical_baseline(cfg))
    rdir = out_dir("results" + args.tag)
    grid = pd.DataFrame([r for x in res for r in x[0]])
    grid = grid.sort_values(["split", "exp_id"]).reset_index(drop=True)   # ordem sequencial E001..E363, REF
    grid.to_csv(rdir / "grid.csv", index=False)
    pd.DataFrame([r for x in res for r in x[1]]).to_csv(rdir / "per_regime.csv", index=False)
    pd.DataFrame([r for x in res for r in x[2]]).to_csv(rdir / "per_time.csv", index=False)
    pd.DataFrame([r for x in res for r in x[4]]).to_csv(rdir / "per_traj.csv", index=False)
    params = {}
    for x in res:
        params.update(x[3])
    save_json(params, rdir / "post_params.json")
    print(f"resultados em {rdir}")


if __name__ == "__main__":
    main()
