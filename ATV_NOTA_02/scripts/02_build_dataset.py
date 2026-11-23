"""Monta os conjuntos train/valid/test, gera o ground truth e aplica P0-P10.

Saidas:
  data/processed/{split}.npz        A, B, gt, regime, traj_idx, tau, steps, F, k
  data/processed/pre/{P}_{split}.npy  entrada da rede apos o pre-processamento P
  data/processed/pre_stats.json      estatisticas do TREINO usadas por P1-P5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gsseg.gt import build_gt, otsu_taus  # noqa: E402
from gsseg.preprocess import apply_pre, fit_stats  # noqa: E402
from gsseg.utils import data_dir, load_config, save_json  # noqa: E402


def dataset_revision(repo: str) -> str:
    """Commit (sha) do dataset no Hugging Face, para registrar a versao usada."""
    try:
        import requests
        return requests.get(f"https://huggingface.co/api/datasets/{repo}", timeout=30).json().get("sha", "")
    except Exception:  # noqa: BLE001
        return ""


def write_manifest(cfg, d, raw, steps, splits, fallback, st):
    """Registro exigido pelo protocolo: versao, split, IDs/frames, metodo, seed, tamanho, GT."""
    dc = cfg["data"]
    size = sum(f.stat().st_size for f in raw.glob("*.npz"))
    man = {
        "dataset": "gray_scott_reaction_diffusion",
        "source": f"https://huggingface.co/datasets/{dc['hf_repo']}",
        "dataset_variant": f"{dc['hf_repo']}@{dataset_revision(dc['hf_repo'])}",
        "trajectory_split": "official The Well split by trajectory (train/valid/test files), no trajectory shared",
        "sampling_method": ("all trajectories of each official split file; frames n = 0, "
                            f"{dc['time_stride']}, ..., {dc['n_time_total'] - 1} (uniform over the whole trajectory)"),
        "seed": cfg["seed"],
        "fields_stored": ["A", "B"], "input_field": cfg["input"]["fields"],
        "resolution_original": "128x128", "resolution_used": "128x128",
        "resampling": "none (native resolution already at the 128 px cap; no upscaling, no padding)",
        "frames_selected": [int(x) for x in steps],
        "size_bytes": int(size), "size_gb": round(size / 1e9, 2), "budget_gb": 25,
        "ground_truth": {"rule": "M = 1[B > tau], tau = Otsu(B at n = %d) per trajectory" % cfg["ground_truth"]["reference_step"],
                         "fallback_tau_by_regime (extinct reference frame)": fallback, "frozen": True},
        "normalization_stats_train": st,
        "splits": splits,
    }
    save_json(man, d / "subset_manifest.json")
    print(f"manifesto: {d / 'subset_manifest.json'} ({man['size_gb']} GB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = data_dir(cfg)
    raw, proc = d / "raw", d / "processed"
    (proc / "pre").mkdir(parents=True, exist_ok=True)
    regimes = list(cfg["data"]["regimes"].keys())
    field_in = cfg["input"]["fields"]
    assert len(field_in) == 1, "esta implementacao usa um unico campo de entrada"
    field_in = field_in[0]

    ref = cfg["ground_truth"]["reference_step"]
    # limiar de reserva por regime: mediana dos Otsu validos no TREINO
    fallback_by_regime = {}
    for r in regimes:
        z = np.load(raw / f"train_{r}.npz")
        t, ok = otsu_taus(z["B"], z["steps"], ref)
        fallback_by_regime[r] = float(np.median(t[ok]))
        print(f"  {r}: {ok.sum()}/{len(ok)} trajetorias com padrao em t={ref}; tau reserva = {fallback_by_regime[r]:.4f}")
    save_json(fallback_by_regime, proc / "tau_fallback.json")

    inputs, manifest_splits = {}, {}
    for split in cfg["data"]["trajectories_per_regime"]:
        A, B, reg, tr, F, k = [], [], [], [], [], []
        for r_i, r in enumerate(regimes):
            z = np.load(raw / f"{split}_{r}.npz")
            A.append(z["A"]); B.append(z["B"])
            reg += [r_i] * len(z["traj_idx"]); tr += list(z["traj_idx"])
            F += [float(z["F"])] * len(z["traj_idx"]); k += [float(z["k"])] * len(z["traj_idx"])
            steps = z["steps"]
        A, B = np.concatenate(A), np.concatenate(B)
        fb = np.array([fallback_by_regime[regimes[i]] for i in reg], dtype=np.float32)
        gt, tau, ok = build_gt(B, steps, ref, fb)
        np.savez(proc / f"{split}.npz", A=A, B=B, gt=gt, regime=np.array(reg), traj_idx=np.array(tr),
                 tau=tau, tau_valid=ok, steps=steps, F=np.array(F), k=np.array(k), regimes=np.array(regimes))
        fg = gt.mean()
        print(f"[{split}] {A.shape[0]} trajetorias x {A.shape[1]} passos | fracao positiva GT = {fg:.3f} | "
              f"Otsu degenerado em {int((~ok).sum())} trajetorias", flush=True)
        inputs[split] = A if field_in == "A" else B
        manifest_splits[split] = {
            r: {"trajectory_ids": [int(t) for t, g in zip(tr, reg) if g == r_i],
                "tau": [float(x) for x, g in zip(tau, reg) if g == r_i],
                "otsu_valid": [bool(x) for x, g in zip(ok, reg) if g == r_i]}
            for r_i, r in enumerate(regimes)}
        manifest_splits[split]["n_trajectories"] = int(A.shape[0])
        manifest_splits[split]["n_frames"] = int(A.shape[0] * A.shape[1])

    st = fit_stats(inputs["train"])
    save_json(st, proc / "pre_stats.json")
    write_manifest(cfg, d, raw, steps, manifest_splits, fallback_by_regime, st)
    for p in cfg["preprocessing"]:
        for split, x in inputs.items():
            out = proc / "pre" / f"{p}_{split}.npy"
            if out.exists():
                continue
            t0 = time.time()
            np.save(out, apply_pre(p, x, st).astype(np.float32))
            print(f"  {p} {split}: {time.time()-t0:.1f}s", flush=True)
    print("ok")


if __name__ == "__main__":
    main()
