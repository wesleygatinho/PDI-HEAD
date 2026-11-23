"""Treina UMA arquitetura com UM pre-processamento e salva as probabilidades de val/teste.

Uso:  python scripts/03_train_predict.py --arch unet --pre P0
Saidas em outputs/models/{arch}_{pre}/:  best.pt, history.csv, meta.json
       outputs/preds/{arch}_{pre}_{valid|test}.npy  (float16 [N, T, 128, 128])

Protocolo da turma: 20 epocas fixas para todas as arquiteturas, entrada 128x128 nativa.
Aumento de dados fisicamente valido para o Gray-Scott (equacoes isotropicas, contorno
periodico): deslocamentos circulares aleatorios, rotacoes de 90 graus e espelhamentos,
aplicados igualmente a entrada e a mascara.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gsseg.models import build_model, count_params  # noqa: E402
from gsseg.utils import data_dir, load_config, load_json, out_dir, save_json, set_seed  # noqa: E402


def augment(x: torch.Tensor, y: torch.Tensor, g: torch.Generator):
    B, _, H, W = x.shape
    sh = torch.randint(0, H, (B, 2), generator=g)
    rot = torch.randint(0, 4, (B,), generator=g)
    flip = torch.randint(0, 2, (B,), generator=g)
    xs, ys = [], []
    for i in range(B):
        xi = torch.roll(x[i], (int(sh[i, 0]), int(sh[i, 1])), (-2, -1))
        yi = torch.roll(y[i], (int(sh[i, 0]), int(sh[i, 1])), (-2, -1))
        xi, yi = torch.rot90(xi, int(rot[i]), (-2, -1)), torch.rot90(yi, int(rot[i]), (-2, -1))
        if flip[i]:
            xi, yi = xi.flip(-1), yi.flip(-1)
        xs.append(xi); ys.append(yi)
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def predict(model, X: np.ndarray, bs: int, dev: str, amp_dtype) -> np.ndarray:
    model.eval()
    flat = X.reshape(-1, 1, *X.shape[-2:])
    out = np.empty((flat.shape[0], *X.shape[-2:]), dtype=np.float16)
    for s in range(0, flat.shape[0], bs):
        xb = torch.from_numpy(np.array(flat[s:s + bs], dtype=np.float32)).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out[s:s + bs] = model.proba(xb).float().cpu().numpy().astype(np.float16)
    return out.reshape(X.shape)


def dice_at_05(prob: np.ndarray, gt: np.ndarray) -> float:
    p, g = prob > 0.5, gt.astype(bool)
    tp = (p & g).sum((-2, -1)); den = p.sum((-2, -1)) + g.sum((-2, -1))
    return float(np.where(den == 0, 1.0, 2 * tp / np.maximum(den, 1)).mean())


@torch.no_grad()
def benchmark(model, dev: str, amp_dtype, bs: int = 16, n_warm=30, n_iter=200):
    model.eval()
    x1 = torch.rand(1, 1, 128, 128, device=dev)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for _ in range(n_warm):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            model.proba(x1)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n_iter):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            model.proba(x1)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t) / n_iter * 1000
    mem1 = torch.cuda.max_memory_allocated() / 2**20
    xb = torch.rand(bs, 1, 128, 128, device=dev)
    for _ in range(5):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            model.proba(xb)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            model.proba(xb)
    torch.cuda.synchronize()
    fps = bs * 20 / (time.perf_counter() - t)
    return lat, mem1, fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["unet", "segformer", "mask2former"])
    ap.add_argument("--pre", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="padrao da turma: 20 (config)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-train-traj", type=int, default=None, help="apenas para teste rapido")
    ap.add_argument("--tag", default="", help="sufixo do nome do experimento (ex.: _smoke)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--benchmark-only", action="store_true",
                    help="so mede latencia/memoria do best.pt ja treinado (GPU sem outros processos)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["training"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    dev = "cuda"
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if tcfg["amp"] else None
    epochs = args.epochs or tcfg["epochs"]
    bs = args.batch_size or tcfg["batch_size"]
    scale = cfg["input"].get("input_scale", 1)
    name = f"{args.arch}_{args.pre}{args.tag}"
    mdir, pdir = out_dir("models", name), out_dir("preds")

    if args.benchmark_only:
        # mede latencia/vazao/memoria com a GPU dedicada (apos todos os treinos paralelos)
        model = build_model(args.arch, pretrained=False, scale=scale).to(dev)
        model.load_state_dict(torch.load(mdir / "best.pt", map_location=dev, weights_only=True))
        lat, mem1, fps = benchmark(model, dev, amp_dtype, bs)
        meta = load_json(mdir / "meta.json")
        meta.update(latency_ms_b1=lat, peak_infer_mem_mib_b1=mem1, throughput_fps=fps,
                    benchmark="isolated", gpu=torch.cuda.get_device_name(0))
        save_json(meta, mdir / "meta.json")
        print(f"[{name}] benchmark isolado: {lat:.2f} ms/quadro | {fps:.0f} fps (bs {bs}) | {mem1:.0f} MiB", flush=True)
        return

    proc = data_dir(cfg) / "processed"
    X = {s: np.load(proc / "pre" / f"{args.pre}_{s}.npy", mmap_mode="r") for s in ("train", "valid", "test")}
    Y = {s: np.load(proc / f"{s}.npz")["gt"] for s in ("train", "valid", "test")}
    if args.max_train_traj:
        sel = np.linspace(0, X["train"].shape[0] - 1, args.max_train_traj).astype(int)
        X["train"], Y["train"] = X["train"][sel], Y["train"][sel]
    Xtr = torch.from_numpy(np.array(X["train"], dtype=np.float32).reshape(-1, 1, 128, 128))
    Ytr = torch.from_numpy(Y["train"].reshape(-1, 128, 128).astype(np.float32))
    vs = tcfg.get("val_time_stride", 1)
    Xva = np.ascontiguousarray(X["valid"][:, ::vs])
    Yva = Y["valid"][:, ::vs]
    n = Xtr.shape[0]
    steps_per_epoch = n // bs                     # descarta o lote incompleto
    total = epochs * steps_per_epoch
    print(f"[{name}] treino: {n} quadros | val (selecao do checkpoint): {Xva.shape[0]*Xva.shape[1]} | bs={bs} | "
          f"epocas={epochs} ({total} iteracoes) | entrada {128*scale}x{128*scale} | amp={amp_dtype}", flush=True)

    model = build_model(args.arch, scale=scale, m2f_aux_loss=tcfg.get("mask2former_aux_loss", False)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"][args.arch], weight_decay=tcfg["weight_decay"])
    warm = max(1, int(tcfg.get("warmup_epochs", 1) * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    g = torch.Generator().manual_seed(seed)

    hist, best, best_ep = [], -1.0, -1
    torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        tl, t0 = 0.0, time.time()
        for it in range(steps_per_epoch):
            idx = perm[it * bs:(it + 1) * bs]
            xb, yb = augment(Xtr[idx], Ytr[idx], g)
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model.loss(xb, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += float(loss)
        vd = dice_at_05(predict(model, Xva, bs, dev, amp_dtype), Yva)
        torch.cuda.empty_cache()
        hist.append({"epoch": ep, "train_loss": tl / steps_per_epoch, "val_dice": vd,
                     "lr": opt.param_groups[0]["lr"], "epoch_time_s": time.time() - t0,
                     "elapsed_s": time.time() - t_train})
        print(f"  ep {ep:02d}/{epochs} loss {tl/steps_per_epoch:.4f} val_dice {vd:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vd > best:
            best, best_ep = vd, ep
            torch.save(model.state_dict(), mdir / "best.pt")
    train_time = time.time() - t_train
    peak_train = torch.cuda.max_memory_allocated() / 2**20
    with open(mdir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys())); w.writeheader(); w.writerows(hist)

    model.load_state_dict(torch.load(mdir / "best.pt", map_location=dev, weights_only=True))
    for s in ("valid", "test"):
        np.save(pdir / f"{name}_{s}.npy", predict(model, np.ascontiguousarray(X[s]), bs, dev, amp_dtype))
    lat, mem1, fps = benchmark(model, dev, amp_dtype, bs)
    meta = dict(arch=args.arch, pre=args.pre, name=name, seed=seed, params=count_params(model),
                epochs_run=len(hist), best_epoch=best_ep, m2f_aux_loss=tcfg.get("mask2former_aux_loss", False), best_val_dice=best, train_time_s=train_time, input_scale=scale,
                peak_train_mem_mib=peak_train, latency_ms_b1=lat, peak_infer_mem_mib_b1=mem1,
                throughput_fps=fps, batch_size=bs, gpu=torch.cuda.get_device_name(0),
                amp=str(amp_dtype), n_train_frames=n)
    save_json(meta, mdir / "meta.json")
    print(f"[{name}] ok | best val dice {best:.4f} (ep {best_ep}) | treino {train_time/60:.1f} min | "
          f"lat {lat:.2f} ms | mem treino {peak_train:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
